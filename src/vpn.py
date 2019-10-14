import requests
from bs4 import BeautifulSoup
import re
from src.config import Config as cfg


# TODO: Write function to call change_ip.sh and loop through various VPNs
login_url = 'https://account.ipvanish.com/login'
validate_url = 'https://account.ipvanish.com/login/validate'
test_url = 'https://account.ipvanish.com/index.php?t=Server+List&page=1'
ipvanish_username = 'aj.gray619@gmail.com'

with requests.Session() as session:
    first_access = session.get(login_url)
    soup = BeautifulSoup(first_access.text, 'html.parser')
    clientToken = soup.find(name='input', attrs={'name': 'clientToken'})['value']

    payload = {'username': ipvanish_username,
               'password': cfg.ipvanish_password,
               'clientToken': clientToken}

    login = session.post(validate_url, data=payload, headers=dict(referer=validate_url))
    page = session.get(test_url, data=payload, headers=dict(referer=validate_url))
    soup2 = BeautifulSoup(page.text, 'html.parser')