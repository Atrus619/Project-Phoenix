from src.models.SmallTalk import SmallTalk
import src.models.utils as smu

model_config = {
    'gradient_accumulation_steps': 4,
    'lm_coef': 2
}

SmallTalkModel = SmallTalk(name='Test_SmallTalk_3', size='small', opt_level='O1', **model_config)  # TODO: Write function to generate unique, but useful name (current time is not a bad choice)
SmallTalkModel.load('logs/checkpoints/Test_SmallTalk_3/test.tar')

data_config = {
    'dataset_path': '',
    'dataset_cache': './dataset_cache',
    'num_candidates': 1,#4,
    'max_history': 1,#2,
    'personality_permutations': 1,#,2,
    'train_batch_size': 2,
    'valid_batch_size': 2
}
train_loader, val_loader = smu.get_small_talk_data_loaders(config=data_config, tokenizer=SmallTalkModel.tokenizer)

SmallTalkModel.train_model(n_epochs=0, train_loader=train_loader, val_loader=val_loader, eval_before_start=True)
