import os
import logging
from src.classes.ScrapedJobs import ScrapedJobs
from wordcloud import WordCloud, STOPWORDS
from config import Config as cfg
from src.classes.Extractions import Extractions


def create_wordcloud(scraped_jobs, type='descr', path='app/static/imgs/sample_wordcloud.png'):
    """
    Generates a wordcloud based on a list of
    :param scraped_jobs: ScrapedJobs object
    :param type: One of 'descr' or 'title'
    :param path: output path to save file to
    :return: wordcloud object
    """
    assert isinstance(scraped_jobs, ScrapedJobs)
    assert type in ('descr', 'job_title')

    stopwords = set(STOPWORDS)
    if type == 'descr':
        full_str = ' '.join([scraped_job.parse() for scraped_job in scraped_jobs if scraped_job.parse() != cfg.job_description_parse_fail_msg])
        stopwords.update(['work', 'will', 'need', 'including', 'required'])
    else:  # job_title
        full_str = ' '.join([scraped_job.job_title for scraped_job in scraped_jobs])
        stopwords.update([])

    WordCloud(stopwords=stopwords).generate(full_str).to_file(path)


def run_extractions(job, location):
    setup_extractions_logger(job=job, location=location)
    extractions = Extractions(required_years_experience=5, required_degree=5, travel_percentage=5, salary=5)
    extractions.gather(job, location)
    return extractions


def setup_extractions_logger(job, location, log_folder=cfg.log_folder, level=logging.INFO):
    log_setup = logging.getLogger('extractions')

    filename = os.path.join(log_folder, 'extractions', f'{job}_in_{location}.log')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_handler = logging.FileHandler(filename, mode='a')
    formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    log_setup.setLevel(level)
    log_setup.addHandler(file_handler)
    log_setup.addHandler(console_handler)
