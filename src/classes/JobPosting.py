from collections import namedtuple
from bs4 import BeautifulSoup
import re
from config import Config as cfg


class JobPosting(namedtuple('JobPosting', 'job_title company location link descr')):
    def parse(self):
        soup = BeautifulSoup(self.descr, "html.parser")
        raw_descr = soup.find_all(name='div', attrs={'id': 'jobDescriptionText'})
        pattern = re.compile('(<.*?>)|(\\n)|[\[\]]')

        if len(raw_descr) > 0:
            return re.sub(pattern, ' ', str(raw_descr))
        else:
            return cfg.job_description_parse_fail_msg
