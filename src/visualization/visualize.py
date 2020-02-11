from src.classes.ScrapedJobs import ScrapedJobs
from wordcloud import WordCloud, STOPWORDS
from config import Config as cfg


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
