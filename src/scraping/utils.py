import re
from bs4 import BeautifulSoup
import requests
import os
import random
import time


def get_soup(url):
    """
    Helper function to construct a BeautifulSoup representation of a url.
    :param url: url returned from build_url
    :return: BeautifulSoup object parsed with html.parser
    """
    page = requests.get(url)
    return BeautifulSoup(page.text, "html.parser")


def build_ipvanish_server_list(base_links):
    """
    Produces a list of ipvanish servers based on a list of tuples mapping base links with the maximum number of servers at that base link
    """
    server_list = []
    pattern = '\d{2}'
    for base_link in base_links:
        for i in range(base_link[1]):
            repl = str(i) if i > 9 else '0' + str(i)
            server_list.append(re.sub(pattern=pattern, repl=repl, string=base_link[0]))
    return server_list


def change_ip(server_list):
    """
    Changes IP to the first server in the list and moves the first item in the list to the end
    :param server_list: List of IP Vanish servers
    :return: Updated server list
    """
    os.system('./src/scraping/change_ip.sh ' + server_list[0])
    return server_list.append(server_list.pop(0))


def random_pause(min_pause=2, max_pause=10):
    time.sleep(random.uniform(min_pause, max_pause))
    return
