from src.classes.ChatBot import ChatBot
from src.classes.Interpreter import Interpreter
from src.classes.Policy import Policy
from src.classes.SmallTalk import SmallTalk
import src.models.SmallTalk.utils as stu
from config import Config as cfg
from src.pipeline.utils import kill_BaaS_externally
from prefect import task, utilities
import numpy as np
import time


@task(state_handlers=[kill_BaaS_externally])
def test_chatbot(interpreter_dict_path=cfg.default_interpreter_dict_output_path,
                 add_conv_detail=False,
                 response_delay=2):
    # Interpreter, pretrained elsewhere
    interpreter = Interpreter()
    interpreter.load_dict(interpreter_dict_path)
    interpreter.init_BaaS()

    # SmallTalk
    dir = stu.download_pretrained_small_talk_model()
    small_talk = SmallTalk(name='Test_SmallTalk_Pretrained', model_type='openai-gpt', model_name=dir, opt_level='O1')

    # Policy
    def delay_func():
        return time.sleep(np.clip(np.random.normal(response_delay, response_delay / 2), 0, response_delay * 2))

    policy = Policy(small_talk=small_talk,
                    delay_func=delay_func)
    # small_talk.print_personality(policy.small_talk_personality)

    # ChatBot
    chat_bot = ChatBot(interpreter=interpreter,
                       policy=policy)

    # Interact
    try:
        chat_bot.interact()

    # Output detail of conversation if desired
    finally:
        if add_conv_detail:
            logger = utilities.logging.get_logger(cfg.chat_bot_training_log_name)
            logger.info(chat_bot.conversation_history)
