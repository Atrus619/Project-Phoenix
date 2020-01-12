import os
import sys; sys.path.append('')

from prefect import Flow, Parameter, utilities
import logging
from argparse import ArgumentParser
from src.models.ner.utterances_to_tokens import utterances_to_tokens
from src.models.ner.allow_user_update_ner import allow_user_update_ner
from src.models.ner.train import train_ner
from src.models.intent.train import train_intent_and_initialize_interpreter
from config import Config as cfg

# TODO: Preprocess input to ner model
# TODO: Allow option to go straight into testing chatbot
# TODO: [Optional] Add a horrific number of optional cl args


def make_flow():
    with Flow("train_all") as flow:
        # Set up logger
        logger = utilities.logging.get_logger(cfg.chat_bot_training_log_name)
        fileHandler = logging.FileHandler(os.path.join(cfg.log_folder, cfg.chat_bot_training_log_name + '.log'))
        formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        # 1. Utterances to tokens
        path = Parameter('path', default=cfg.ner_and_intent_training_data_path)
        status_utterances_to_tokens = utterances_to_tokens(path=path)

        # 2. Pause while user updates step in between here
        status_allow_user_update_ner = allow_user_update_ner(path=path, upstream_tasks=[status_utterances_to_tokens])

        # 3. Train NER
        status_train_ner = train_ner(data_path=path, upstream_tasks=[status_allow_user_update_ner])

        # 4. Train Intent
        status_train_intent_and_initialize_interpreter = train_intent_and_initialize_interpreter(data_path=path,  upstream_tasks=[status_train_ner])

    return flow


if __name__ == "__main__":
    # 0. Parse args
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default=cfg.ner_and_intent_training_data_path,
                        help="Path to the excel file containing training examples for intent classification.")
    args = parser.parse_args()

    flow = make_flow()

    parameters = {
        'path': args.path
    }

    flow.run(parameters=parameters)
