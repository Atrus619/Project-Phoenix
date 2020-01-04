from src.classes.ChatBot import ChatBot
from src.classes.Interpreter import Interpreter
from src.classes.Policy import Policy
from src.classes.SmallTalk import SmallTalk
import src.models.SmallTalk.utils as stu
from config import Config as cfg
import pickle as pkl

# Interpreter, pretrained elsewhere
with open(cfg.default_interpreter_output_path, 'rb') as f:
    interpreter = pkl.load(f)
interpreter.init_BaaS()

# SmallTalk
dir = stu.download_pretrained_small_talk_model()
small_talk = SmallTalk(name='Test_SmallTalk_Pretrained', model_type='openai-gpt', model_name=dir, opt_level='O1')

# Policy
policy = Policy(small_talk=small_talk)
small_talk.print_personality(policy.small_talk_personality)

# ChatBot
chat_bot = ChatBot(interpreter=interpreter,
                   policy=policy)

# Interact
chat_bot.interact()
interpreter.kill_BaaS()

# Debugging
print(chat_bot.conversation_history)
x = chat_bot.conversation_history.get_list_of_conversation_latest_n_exchanges(4)
for y in x:
    print(y)