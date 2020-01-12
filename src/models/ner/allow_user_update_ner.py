from prefect import task, utilities
import os
from config import Config as cfg


@ task
def allow_user_update_ner(path):
    logger = utilities.logging.get_logger(cfg.chat_bot_training_log_name)
    logger.info('----- Waiting for user to update "Tokens" tab and save before continuing... -----')

    os.system(f"lowriter '{path}'")
    logger.info('----- Update complete. -----')
    return
