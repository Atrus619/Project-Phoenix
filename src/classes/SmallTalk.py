from pytorch_transformers import GPT2DoubleHeadsModel, GPT2Tokenizer, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer, AdamW, WEIGHTS_NAME
import src.models.SmallTalk.utils as stu
import torch
import math
from pprint import pformat
import os
from config import Config as cfg
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler
from apex import amp
import time
import random
from itertools import chain


class SmallTalk:
    def __init__(self, name, model_name, model_type='gpt2', opt_level=None, lr=6.25e-5, lm_coef=1.0, mc_coef=1.0, gradient_accumulation_steps=8, max_norm=1.0, device='cuda:0'):
        self.lr, self.lm_coef, self.mc_coef, self.gradient_accumulation_steps, self.max_norm, self.device = lr, lm_coef, mc_coef, gradient_accumulation_steps, max_norm, device
        self.name, self.model_name, self.model_type, self.opt_level = name, model_name, model_type, opt_level

        self.logger, self.tb_logger, self.checkpoint_handler = stu.setup_training_loggers(self.name)

        self.verbose = False
        self.epoch = 0

        # TODO: Add logger statement here
        model_class, tokenizer_class = (GPT2DoubleHeadsModel, GPT2Tokenizer) if self.model_type == 'gpt2' else (OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer)
        self.model, self.tokenizer = model_class.from_pretrained(self.model_name).to(self.device), tokenizer_class.from_pretrained(self.model_name)

        stu.add_special_tokens_(model=self.model, tokenizer=self.tokenizer)

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, correct_bias=True)

        if self.opt_level:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.opt_level)

        self.trainer = Engine(self.update)
        self.evaluator = Engine(self.inference)

    def update(self, engine, batch):
        self.model.train()
        batch = tuple(input_tensor.to(self.device) for input_tensor in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        (lm_loss), (mc_loss), *_ = self.model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            mc_labels=mc_labels, lm_labels=lm_labels
        )
        loss = (lm_loss * self.lm_coef + mc_loss * self.mc_coef) / self.gradient_accumulation_steps

        if self.opt_level:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

        if engine.state.iteration % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    def inference(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(self.device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch

            if self.verbose:
                self.logger.info(self.tokenizer.decode(input_ids[0, -1, :].tolist()))

            # if we dont send labels to model, it doesnt return losses
            lm_logits, mc_logits, *_ = self.model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            )

            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)

    def train_model(self, n_epochs, train_loader, val_loader, eval_before_start=True):
        # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: self.evaluator.run(val_loader))
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: self.update_epoch())
        if eval_before_start:
            self.trainer.add_event_handler(Events.STARTED, lambda _: self.evaluator.run(val_loader))

        # Linearly decrease the learning rate from lr to zero
        scheduler = PiecewiseLinear(self.optimizer, "lr", [(0, self.lr), (n_epochs * len(train_loader), 0.0)])
        self.trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

        # Prepare metrics
        RunningAverage(output_transform=lambda x: x).attach(self.trainer, "loss")
        metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0])),
                   "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
        metrics["average_ppl"] = MetricsLambda(math.exp, metrics["nll"])
        for name, metric in metrics.items():
            metric.attach(self.evaluator, name)

        # On the main process: add progress bar, tensorboard, checkpoints and save model
        pbar = ProgressBar(persist=True)
        pbar.attach(self.trainer, metric_names=["loss"])

        if not self.verbose:
            pbar_eval = ProgressBar(persist=False)
            pbar_eval.attach(self.evaluator)

        self.evaluator.add_event_handler(Events.STARTED, lambda _: self.logger.info(f'Beginning validation for epoch {self.epoch}...'))
        self.evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(self.evaluator.state.metrics)))

        self.tb_logger.attach(self.trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        self.tb_logger.attach(self.trainer, log_handler=OptimizerParamsHandler(self.optimizer), event_name=Events.ITERATION_STARTED)
        self.tb_logger.attach(self.evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=self.trainer),
                              event_name=Events.EPOCH_COMPLETED)

        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.checkpoint_handler,
                                       {'mymodel': getattr(self.model, 'module', self.model)})  # "getattr" takes care of distributed encapsulation

        # Run the training
        self.trainer.run(train_loader, max_epochs=n_epochs)

        # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
        if n_epochs > 0:
            os.rename(self.checkpoint_handler._saved[-1][1][-1], os.path.join(cfg.checkpoint_log_folder, self.name, WEIGHTS_NAME))
            self.tb_logger.close()

    def save(self, path, inference_only=False):
        """ Saves important components of model to be imported later. """
        save_dict = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'model_type': self.model_type,
            'opt_level': self.opt_level
        }

        if not inference_only:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()

        torch.save(save_dict, path)

    # TODO: May want to revisit here if we want to do evaluation on a cpu. See https://github.com/NVIDIA/apex/issues/242
    def load(self, path):
        """ Loads important components of model back into memory to pick up where we left off. """
        checkpoint = torch.load(path)
        assert self.model_type == checkpoint['model_type'], f"Model types do not match, current model is {self.model_type} and loaded model is {checkpoint['model_type']}"
        assert self.model_name == checkpoint['model_name'], f"Model names do not match, current model is {self.model_name} and loaded model is {checkpoint['model_name']}"
        assert self.opt_level == checkpoint['opt_level'], f"Model opt_levels do not match, current model is {self.opt_level} and loaded model is {checkpoint['opt_level']}"

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.logger.info('Optimizer information saved for continued training. Loading into model.')
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.logger.info('Model previously saved for inference only.')

        self.epoch = checkpoint['epoch']

    def load_checkpoint(self, path):
        """ Loads an entire checkpoint and overwrite model """
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def update_epoch(self):
        self.epoch += 1

    def get_num_params(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())

    def interact(self, personality=None, max_history=2, max_length=20, min_length=1, temperature=0.7, top_k=0, top_p=0.9, no_sample=False, random_pause=None):
        """
        Interact with bot in python setting
        :param personality: Personality to use to condition model on for chat. None will pull a random one from training data set. List of several short sentences describing personality.
        :param max_history: Number of responses per individual to retain for model to generate text with (in addition to the utterance the model is directly responding to).
        :param max_length: Maximum length of output utterances
        :param min_length: Minimum length of output utterances
        :param temperature: Sampling softmax temperature. 1.0 is standard softmax, as it decreases it allows for less diversity in outputs (makes peaks higher in distribution).
        :param top_k: Filter top_k tokens before sampling (<=0 is no filtering)
        :param top_p: Nucleus filtering
        :param no_sample: Whether to simply choose the most likely token at each sample and skip fancy sampling methods above
        :param random_pause: Whether to pause for random amounts of time to seem more human (should be tuple of low and high value to randomly pause between).
        """
        personality = self.get_personality(personality=personality)
        self.logger.info(self.tokenizer.decode(list(chain(*personality))))

        self.model.eval()
        history = []

        self.logger.info('You may now begin talking to the bot. Don\'t be shy, say hello!')

        while True:
            raw_text = input('>>> ')
            while not raw_text:
                print('Please enter in a non-empty value.')
                raw_text = input('>>> ')
            history.append(self.tokenizer.encode(raw_text))

            if random_pause:
                assert len(random_pause) == 2, 'random_pause arg should be a tuple of length 2 if passed'
                time.sleep(random_pause[0] + random.random() * (random_pause[1] - random_pause[0]))

            with torch.no_grad():
                out_ids = stu.sample_sequence(personality=personality, history=history, tokenizer=self.tokenizer, model=self.model, device=self.device,
                                              max_length=max_length, min_length=min_length, temperature=temperature, top_k=top_k, top_p=top_p, no_sample=no_sample)
            history.append(out_ids)
            history = history[-(2 * max_history + 1):]
            out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            print(out_text)

    def get_reply(self, conversation_history, personality, max_history=2, max_length=20, min_length=1, temperature=0.7, top_k=0, top_p=0.9, no_sample=False, random_pause=None):
        """
        Based heavily on self.interact. See above documentation for detail on parameters.
        Alternate version of interact for use with chatbot. Uses ConversationHistory object to put together the history and return one reply at a time, rather than manage
        an entire conversation.
        """
        self.model.eval()

        # Build history object from ConversationHistory class
        history = conversation_history.get_list_of_conversation_latest_n_exchanges(n=max_history)
        history = [self.tokenizer.encode(msg) for msg in history]

        # Get ids from model
        with torch.no_grad():
            out_ids = stu.sample_sequence(personality=personality, history=history, tokenizer=self.tokenizer, model=self.model, device=self.device,
                                          max_length=max_length, min_length=min_length, temperature=temperature, top_k=top_k, top_p=top_p, no_sample=no_sample)

        return self.tokenizer.decode(out_ids, skip_special_tokens=True)

    def get_personality(self, personality=None):
        """
        Retrieves a random personality if personality is None, otherwise converts personality raw text to a format the model understands.
        :param personality: List of 4-5 sentences in raw text string form
        """
        if personality is None:
            return stu.get_random_personality(self)
        else:
            return [self.tokenizer.encode(sentence) for sentence in personality]

    def print_personality(self, personality):
        print(self.tokenizer.decode(chain(*personality)))
