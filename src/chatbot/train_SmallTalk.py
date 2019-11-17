from src.models.SmallTalk import SmallTalk
import src.models.utils as smu

model_config = {
    'gradient_accumulation_steps': 4,
    'lm_coef': 2
}

# TODO: Write function to generate unique, but useful name (current time is not a bad choice)
SmallTalkModel = SmallTalk(name='Test_SmallTalk_3', model_type='gpt2', model_name='gpt2', opt_level='O1', **model_config)
SmallTalkModel.load('logs/checkpoints/Test_SmallTalk_3/test2.tar')

data_config = {
    'dataset_path': '',
    'dataset_cache': './dataset_cache',
    'num_candidates': 1,#4,
    'max_history': 1,#2,
    'personality_permutations': 1,#,2,
    'train_batch_size': 2,
    'valid_batch_size': 2
}
train_loader, val_loader = smu.get_small_talk_data_loaders(config=data_config, tokenizer=SmallTalkModel.tokenizer, logger=SmallTalkModel.logger)

SmallTalkModel.train_model(n_epochs=0, train_loader=train_loader, val_loader=val_loader, eval_before_start=True)

dir = smu.download_pretrained_small_talk_model()
PretrainedModel = SmallTalk(name='Test_SmallTalk_Pretrained', model_type='openai-gpt', model_name=dir, opt_level='O1', **model_config)

train_loader, val_loader = smu.get_small_talk_data_loaders(config=data_config, tokenizer=PretrainedModel.tokenizer, logger=PretrainedModel.logger)
PretrainedModel.train_model(n_epochs=0, train_loader=train_loader, val_loader=val_loader, eval_before_start=True)
