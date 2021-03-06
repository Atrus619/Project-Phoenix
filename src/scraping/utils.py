import re
from bs4 import BeautifulSoup
import random
import time
from src.constants import Constants as cs
import logging
import os
from config import Config as cfg
from datetime import date


def get_soup(session, url, user_agent):
    """
    Helper function to construct a BeautifulSoup representation of a url.
    :param session: requests session object
    :param url: url returned from build_url
    :param user_agent: User Agent to be used with the request
    :return: BeautifulSoup object parsed with html.parser
    """
    headers = cs.base_request_headers
    headers['User-Agent'] = user_agent

    page = custom_get(session=session, url=url, headers=headers)

    return BeautifulSoup(page.text, 'html.parser')


def custom_get(session, url, headers):
    with session.get(url, headers=headers) as page:
        return page


def build_ipvanish_server_list(base_links):
    """
    Produces a list of ipvanish servers based on a list of tuples mapping base link urls with the maximum number of servers at that base link
    """
    server_list = []
    pattern = '\d{2}'
    for base_link in base_links:
        for i in range(1, base_link[1]):
            repl = str(i) if i > 9 else '0' + str(i)
            server_list.append(re.sub(pattern=pattern, repl=repl, string=base_link[0]))
    return server_list


def random_pause(min_pause=2, max_pause=10):
    time.sleep(random.uniform(min_pause, max_pause))
    return


def setup_scrape_logger(name, filename, level=logging.INFO):
    log_setup = logging.getLogger(name)

    if len(log_setup.handlers) == 2:  # Logger already set up for current run
        return

    log_dir = os.path.join(cfg.log_folder, 'scraping')
    os.makedirs(log_dir, exist_ok=True)

    date_specific_filename = filename + '_' + date.today().strftime("%Y%m%d") + '.log'

    fileHandler = logging.FileHandler(os.path.join(log_dir, date_specific_filename), mode='a')
    formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    fileHandler.setFormatter(formatter)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)

    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)
    log_setup.addHandler(consoleHandler)


def get_search_params(config):
    search_params = []
    for job in config.jobs:
        for city in config.cities:
            search_params.append((job, city))
    return search_params


def get_pretty_time(duration, num_digits=2):
    # Duration is assumed to be in seconds. Returns a string with the appropriate suffix (s/m/h)
    if duration > 60**2:
        return str(round(duration / 60**2, num_digits)) + 'h'
    if duration > 60:
        return str(round(duration / 60, num_digits)) + 'm'
    else:
        return str(round(duration, num_digits)) + 's'


def ipvanish_connect(address):
    return os.system('echo %s|sudo -S %s' % (cfg.sudo_password, './src/scraping/change_ip.sh ' + address))


def is_ipvanish_up():
    return os.system('nmcli c show --active | grep vpn') == 0
