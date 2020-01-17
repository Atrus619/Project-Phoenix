import sys;

sys.path.append('')
from argparse import ArgumentParser
from src.classes.ChatBot import ChatBot
from src.classes.Interpreter import Interpreter
from src.classes.Policy import Policy
from src.classes.SmallTalk import SmallTalk
import src.models.SmallTalk.utils as stu
from src.models.intent.train import get_interpreter_dict_path
from config import Config as cfg
from prefect import task, utilities
import numpy as np
import time
from prefect import Flow
from src.pipeline.utils import clean_up, init_BaaS


@task
def test_chatbot(model_name,
                 add_conv_detail=False,
                 response_delay=2):
    # Interpreter, pretrained elsewhere
    interpreter = Interpreter()
    interpreter_dict_path = get_interpreter_dict_path(model_name=model_name)
    interpreter.load_dict(interpreter_dict_path)

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='model',
                        help='Title of model. Used for storing model as serialized file.')

    parser.add_argument("--add_conv_detail", dest='add_conv_detail', action='store_true',
                        help="Whether to print out the full conversation at the end with annotations by the chatbot's interpreter. Off by default.")

    parser.add_argument("--response_delay", type=int, default=0,
                        help='Number of seconds to add as a stochastic artifical delay for chat bot. Defaults to 0 seconds (no delay).')

    parser.set_defaults(add_conv_detail=False)
    args = parser.parse_args()

    with Flow(f'{args.model_name} Chatbot Test') as flow:
        BaaS_freshly_initialized = init_BaaS()
        final_status = test_chatbot(model_name=args.model_name,
                                    add_conv_detail=args.add_conv_detail,
                                    response_delay=args.response_delay,
                                    upstream_tasks=[BaaS_freshly_initialized])
        clean_up(pkill_BaaS=BaaS_freshly_initialized,
                 upstream_tasks=[final_status])
        flow.set_reference_tasks([final_status])

        flow.run()
