from pytorch_transformers import GPT2DoubleHeadsModel, GPT2Tokenizer, AdamW, WEIGHTS_NAME
import src.models.utils as smu
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


class SmallTalk:
    def __init__(self, name, size, opt_level, lr=6.25e-5, lm_coef=1.0, mc_coef=1.0, gradient_accumulation_steps=8, max_norm=1.0, device='cuda:0'):
        self.lr, self.lm_coef, self.mc_coef, self.gradient_accumulation_steps, self.max_norm, self.device = lr, lm_coef, mc_coef, gradient_accumulation_steps, max_norm, device

        self.verbose = False
        self.epoch = 0
        self.name = name
        self.size = size
        self.opt_level = opt_level
        self.logger, self.tb_logger, self.checkpoint_handler = smu.setup_training_loggers(self.name)

        assert self.size in ('small', 'medium', 'large')
        model_name = 'gpt2'
        if self.size in ('medium', 'large'):
            model_name += '-' + self.size

        self.model = GPT2DoubleHeadsModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        smu.add_special_tokens_(model=self.model, tokenizer=self.tokenizer)

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, correct_bias=True)

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

        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_norm)

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
            pbar_eval = ProgressBar(persist=True)
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
            'size': self.size,
            'opt_level': self.opt_level
        }

        if not inference_only:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()

        torch.save(save_dict, path)

    def load(self, path):
        """ Loads important components of model back into memory to pick up where we left off. """
        checkpoint = torch.load(path)
        assert self.size == checkpoint['size'], f"Model sizes do not match, current model is {self.size} and loaded model is {checkpoint['size']}"
        assert self.opt_level == checkpoint['opt_level'], f"Model opt_levels do not match, current model is {self.opt_level} and loaded model is {checkpoint['opt_level']}"

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.logger.info('Optimizer information saved for continued training. Loading into model.')
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.epoch = checkpoint['epoch']

    def update_epoch(self):
        self.epoch += 1
