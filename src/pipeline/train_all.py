import os
import sys; sys.path.append('')

from prefect import Flow, utilities
import logging
from argparse import ArgumentParser
from src.models.ner.utterances_to_tokens import utterances_to_tokens
from src.models.ner.allow_user_update_ner import allow_user_update_ner
from src.models.ner.train import train_ner
from src.models.intent.train import train_intent_and_initialize_interpreter
from src.pipeline.test_chatbot import test_chatbot
from src.pipeline.utils import clear_tmps
from config import Config as cfg


# TODO: [Optional] Add a horrific number of optional cl args
def make_flow(title='train_all',
              path=cfg.ner_and_intent_training_data_path,
              reuse_existing=True,
              remove_caps=True,
              spawn_chatbot=False,
              add_conv_detail=False):
    with Flow(title) as flow:
        # Set up logger
        logger = utilities.logging.get_logger(cfg.chat_bot_training_log_name)
        os.makedirs(os.path.join(cfg.log_folder, cfg.chat_bot_training_log_name), exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(cfg.log_folder, cfg.chat_bot_training_log_name, title + '.log'))
        formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        if add_conv_detail and not spawn_chatbot:
            logger.warning('Warning: Chatbot not specified to be ran at end, but conversation detail requested. Detail will not be logged and printed.')

        # 1. Utterances to tokens
        status_utterances_to_tokens = utterances_to_tokens(path=path, reuse_existing=reuse_existing, remove_caps=remove_caps)

        # 2. Pause while user updates step in between here
        status_allow_user_update_ner = allow_user_update_ner(path=path, upstream_tasks=[status_utterances_to_tokens])

        # 3. Train NER
        status_train_ner = train_ner(data_path=path, upstream_tasks=[status_allow_user_update_ner])

        # 4. Train Intent
        status_train_intent_and_initialize_interpreter = train_intent_and_initialize_interpreter(data_path=path, remove_caps=remove_caps,
                                                                                                 upstream_tasks=[status_train_ner])

        # 5. Spawn Chatbot for testing if requested
        if spawn_chatbot:
            status_spawn_chatbot = test_chatbot(interpreter_dict_path=cfg.default_interpreter_dict_output_path,
                                                add_conv_detail=add_conv_detail,
                                                upstream_tasks=[status_train_intent_and_initialize_interpreter])
            clear_tmps(upstream_tasks=[status_spawn_chatbot])
        else:
            clear_tmps(upstream_tasks=[status_train_intent_and_initialize_interpreter])

    return flow


if __name__ == "__main__":
    # Args
    parser = ArgumentParser()
    parser.add_argument('--title', type=str, default='train_all',
                        help='Title of training run. Used primarily for naming log files.')

    parser.add_argument("--path", type=str, default=cfg.ner_and_intent_training_data_path,
                        help="Path to the excel file containing training examples for intent classification.")

    reuse_existing_help = 'Whether to reuse existing entries in the training file. Defaults to True.'
    parser.add_argument("--reuse_existing", dest='reuse_existing', action='store_true', help=reuse_existing_help)
    parser.add_argument("--overwrite_existing", dest='reuse_existing', action='store_false', help=reuse_existing_help)

    remove_caps_help = 'Whether to apply lower case to all characters fed to the training examples and associated interpreter model. Defaults to True.'
    parser.add_argument("--remove_caps", dest='remove_caps', action='store_true', help=remove_caps_help)
    parser.add_argument("--keep_caps", dest='remove_caps', action='store_false', help=remove_caps_help)

    parser.add_argument("--spawn_chatbot", dest='spawn_chatbot', action='store_true',
                        help='Whether to spawn the chatbot immediately after training to begin testing. Defaults to False.')

    parser.add_argument("--add_conv_detail", dest='add_conv_detail', action='store_true',
                        help="Whether to print out the full conversation at the end with annotations by the chatbot's interpreter. Defaults to False.")

    parser.set_defaults(reuse_existing=True, remove_caps=True, spawn_chatbot=False, add_conv_detail=False)
    args = parser.parse_args()

    # Create flow
    arg_dict = vars(args)
    flow = make_flow(**arg_dict)

    # Run flow
    flow.run()
