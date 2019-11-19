from src.models.SmallTalk.SmallTalk import SmallTalk
import src.models.SmallTalk.utils as smu
import os

model_config = {
    'gradient_accumulation_steps': 4,  # Use to mimic larger batch sizes along with train_batch_size in data_config
    'lm_coef': 2.0  # Balance weighting of each loss
}

# TODO: Write function to generate unique, but useful name (current time is not a bad choice)
SmallTalkModel = SmallTalk(name='Test_SmallTalk_small_train_batch_size_2',  # Name used for logging
                           model_type='gpt2',  # GPT2 if gpt2 otherwise gpt1
                           model_name='gpt2',  # small if gpt2, medium if medium, large if large
                           opt_level='O2',  # FP16: Either O1 or O2 generally (O2 is faster, also it is the letter O not a zero). None if no FP16.
                           **model_config)
# SmallTalkModel.load('logs/checkpoints/Test_SmallTalk_3/test2.tar')

data_config = {
    'dataset_path': '',
    'dataset_cache': './dataset_cache',
    'num_candidates': 4,  # Number of choices of responses to use. Note: The last index choice is the correct one when building the data set. Val set to 20 by default. Increases size of each batch.
    'max_history': 2,  # How many responses (per person) to retain in history before the utterance being replied to (1 + 2 * max_history). Increases size of each batch by increasing length of longest sequence.
    'personality_permutations': 2,  # Number of times to shift order of personality sentences around (Places the last one at the front). Increases size of data set.
    'train_batch_size': 2,  # Adjust for RAM vs. training time considerations
    'valid_batch_size': 2  # Adjust for RAM vs. training time considerations
}
train_loader, val_loader = smu.get_small_talk_data_loaders(config=data_config, tokenizer=SmallTalkModel.tokenizer, logger=SmallTalkModel.logger)

SmallTalkModel.train_model(n_epochs=3, train_loader=train_loader, val_loader=val_loader, eval_before_start=False)

SmallTalkModel.save(os.path.join('logs', 'checkpoints', SmallTalkModel.name, 'final_model.tar'), inference_only=True)
# dir = smu.download_pretrained_small_talk_model()
# PretrainedModel = SmallTalk(name='Test_SmallTalk_Pretrained', model_type='openai-gpt', model_name=dir, opt_level='O1', **model_config)
#
# train_loader, val_loader = smu.get_small_talk_data_loaders(config=data_config, tokenizer=PretrainedModel.tokenizer, logger=PretrainedModel.logger)
# PretrainedModel.train_model(n_epochs=0, train_loader=train_loader, val_loader=val_loader, eval_before_start=True)
