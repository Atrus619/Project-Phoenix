import os
import sys; sys.path.append('')

from prefect import Flow, utilities
import logging
from argparse import ArgumentParser
from src.models.ner.utterances_to_tokens import utterances_to_tokens
from src.models.ner.allow_user_update_ner import allow_user_update_ner
from src.models.ner.train import train_ner
from src.models.intent.train import train_intent_and_initialize_interpreter
from src.models.intent.train import train_intent_follow_up
from test_chatbot import test_chatbot
from src.pipeline.utils import clean_up, init_BaaS
from config import Config as cfg


# TODO: Add option to show personality
# TODO: Add option to input personality (function for parsing)
# TODO: Split train and test into two completely separate scripts
# TODO: Update issues
# TODO: Turn entities and intents into special class or enum data type
def make_flow(model_name=cfg.default_model_name,
              path=cfg.ner_and_intent_training_data_path,
              reuse_existing=True,
              remove_caps=True,
              spawn_chatbot=False,
              add_conv_detail=False,
              response_delay=False):
    with Flow(model_name) as flow:
        # Set up logger
        logger = utilities.logging.get_logger(cfg.chat_bot_training_log_name)
        os.makedirs(os.path.join(cfg.log_folder, cfg.chat_bot_training_log_name), exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(cfg.log_folder, cfg.chat_bot_training_log_name, model_name + '.log'))
        formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        if add_conv_detail and not spawn_chatbot:
            logger.warning('Warning: Chatbot not specified to be ran at end, but conversation detail requested. Detail will not be logged and printed.')

        # 1. Utterances to tokens
        status_utterances_to_tokens = utterances_to_tokens(path=path, reuse_existing=reuse_existing, remove_caps=remove_caps)

        # 2. Pause while user updates step in between here
        status_allow_user_update_ner = allow_user_update_ner(path=path,
                                                             upstream_tasks=[status_utterances_to_tokens])

        # 3. Train NER Classifier
        status_train_ner = train_ner(data_path=path, model_name=model_name,
                                     upstream_tasks=[status_allow_user_update_ner])

        # 4. Train Intent Classifier
        BaaS_freshly_initialized = init_BaaS()
        status_train_intent_and_initialize_interpreter = train_intent_and_initialize_interpreter(data_path=path, remove_caps=remove_caps, model_name=model_name,
                                                                                                 upstream_tasks=[status_train_ner, BaaS_freshly_initialized])

        # 5. Train Intent Follow Up Classifier
        status_train_intent_follow_up = train_intent_follow_up(data_path=path, model_name=model_name,
                                                               upstream_tasks=[status_train_intent_and_initialize_interpreter])

        # Final task in case we add more
        final_training_task = status_train_intent_follow_up

        # 6. Spawn Chatbot for testing if requested
        if spawn_chatbot:
            final_status = test_chatbot(model_name=model_name,
                                        add_conv_detail=add_conv_detail,
                                        response_delay=response_delay,
                                        upstream_tasks=[final_training_task])
        else:
            final_status = final_training_task

        clean_up(pkill_BaaS=BaaS_freshly_initialized,
                 upstream_tasks=[final_status])
        flow.set_reference_tasks([final_training_task, final_status])

    return flow


if __name__ == "__main__":
    # Args
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='model',
                        help='Title of model. Used for storing model as serialized file and naming log files.')

    parser.add_argument("--path", type=str, default=cfg.ner_and_intent_training_data_path,
                        help="Path to the excel file containing training examples for intent classification.")

    reuse_existing_help = 'Whether to reuse existing entries in the training file.'
    parser.add_argument("--reuse_existing", dest='reuse_existing', action='store_true', help=reuse_existing_help + ' On by default.')
    parser.add_argument("--overwrite_existing", dest='reuse_existing', action='store_false', help=reuse_existing_help + ' Off by default.')

    remove_caps_help = 'Whether to apply lower case to all characters fed to the training examples and associated interpreter model.'
    parser.add_argument("--remove_caps", dest='remove_caps', action='store_true', help=remove_caps_help + ' On by default.')
    parser.add_argument("--keep_caps", dest='remove_caps', action='store_false', help=remove_caps_help + ' Off by default.')

    parser.add_argument("--spawn_chatbot", dest='spawn_chatbot', action='store_true',
                        help='Whether to spawn the chatbot immediately after training to begin testing. Off by default.')

    parser.add_argument("--add_conv_detail", dest='add_conv_detail', action='store_true',
                        help="Whether to print out the full conversation at the end with annotations by the chatbot's interpreter. Off by default.")

    parser.add_argument("--response_delay", type=int, default=0,
                        help='Number of seconds to add as a stochastic artifical delay for chat bot. Defaults to 0 seconds (no delay).')

    parser.set_defaults(reuse_existing=True, remove_caps=True, spawn_chatbot=False, add_conv_detail=False)
    args = parser.parse_args()

    # Create flow
    arg_dict = vars(args)
    flow = make_flow(**arg_dict)

    # Run flow
    flow.run()
