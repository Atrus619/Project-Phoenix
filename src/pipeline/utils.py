import os
from config import Config as cfg
import re
import shutil
from prefect import task, utilities


def kill_BaaS_externally(obj, old_state, new_state):
    if new_state.is_failed():
        logger = utilities.logging.get_logger(cfg.chat_bot_training_log_name)
        logger.info('Tearing down BaaS.')
        os.system('pkill bert')
    return new_state


@task
def clear_tmps():
    # Clear all tmp folders in working directory
    tmp_pattern = '^tmp'
    tmp_list = [folder for folder in os.listdir(os.getcwd()) if re.match(tmp_pattern, folder)]
    [shutil.rmtree(tmp_folder) for tmp_folder in tmp_list]

    logger = utilities.logging.get_logger(cfg.chat_bot_training_log_name)
    logger.info('Cleared all tmp folders from primary directory.')

    # Clear tmp directory (fills up when running BaaS repeatedly)
    os.system('./clear_tmp_directory.sh')
    logger.info('Cleared tmp directory.')

