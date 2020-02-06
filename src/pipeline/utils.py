from config import Config as cfg
import re
import shutil
from prefect import task, utilities
from prefect.triggers import always_run
import zmq
import os
import subprocess


@task(trigger=always_run)
def clean_up(pkill_BaaS=False):
    clear_tmps()

    if pkill_BaaS:
        kill_BaaS()


def clear_tmps():
    # Clear all tmp folders in working directory
    tmp_pattern = '^tmp'
    tmp_list = [folder for folder in os.listdir(os.getcwd()) if re.match(tmp_pattern, folder)]
    [shutil.rmtree(tmp_folder) for tmp_folder in tmp_list]

    logger = utilities.logging.get_logger(cfg.chatbot_training_log_name)
    logger.info('Cleared all tmp folders from primary directory.')

    # Clear tmp directory (fills up when running BaaS repeatedly)
    os.system('./clear_tmp_directory.sh')
    logger.info('Cleared tmp directory.')


def kill_BaaS():
    logger = utilities.logging.get_logger(cfg.chatbot_training_log_name)
    logger.info('Shutting down BaaS')
    os.system('pkill bert')


@task
def init_BaaS(num_workers=1):
    # Safely starts a BaaS process (even if one is already going)
    # Returns True if a new BaaS was started, False otherwise
    def start_BaaS_subprocess(num_workers):
        subprocess.Popen(['bert-serving-start', '-model_dir', cfg.bert_dir, '-num_worker', str(num_workers)])

    if is_port_in_use(cfg.bert_port):
        return False
    try:
        start_BaaS_subprocess(num_workers=num_workers)
    except zmq.error.ZMQError:
        kill_BaaS()
        start_BaaS_subprocess(num_workers=num_workers)
    finally:
        return True


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def wrap_websocket_msg(msg):
    return f'42["bot message", "{msg}"]'
