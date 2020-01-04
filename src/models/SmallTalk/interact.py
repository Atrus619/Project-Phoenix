from src.classes.SmallTalk import SmallTalk
import src.models.SmallTalk.utils as stu

# SmallTalkModel = SmallTalk(name='Test_SmallTalk_small_train_batch_size_2', model_type='gpt2', model_name='gpt2', opt_level="O2")
# SmallTalkModel.load_checkpoint(os.path.join('logs', 'checkpoints', SmallTalkModel.name, 'checkpoint_mymodel_1.pth'))

dir = stu.download_pretrained_small_talk_model()
PretrainedModel = SmallTalk(name='Test_SmallTalk_Pretrained', model_type='openai-gpt', model_name=dir, opt_level='O1')

interact_config = {
    # 'personality': ['I like to eat pizza.', 'My name is Jenbot3000.', 'I am sexy.', 'I work as a consultant.', 'I like to cuddle.'],
    'max_history': 2,
    'max_length': 20,
    'min_length': 1,
    'temperature': 0.7,
    'top_k': 0,
    'top_p': 0.9,
    'no_sample': False,
    'random_pause': None
}

# SmallTalkModel.interact(**interact_config)
PretrainedModel.interact(**interact_config)