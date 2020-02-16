from src.classes.ScrapedJobs import ScrapedJobs
from wordcloud import WordCloud, STOPWORDS
from config import Config as cfg
from src.classes.Scraper import Scraper
from src.classes.JobPostingExtractor import JobPostingExtractor
from bert_serving.client import BertClient
import src.pipeline.utils as spu


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


def gather_jpes(job, location, gathered_jpes, vpn=True, ngram_size=8, max_iters=10):
    """Gathers jpes based on specified job and location until requirements are satisfied. Intended to be called asynchronously with redis"""
    current_page = 1
    while (not gathered_jpes.is_complete()) and (current_page <= max_iters):
        # Scrape an entire page off of indeed
        scraped_jobs = Scraper().scrape_page_indeed(job_title=job, location=location, page=current_page, vpn=vpn)

        # Batch collect encodings for jpes where parsing did not fail
        current_jpes = [JobPostingExtractor(job_posting) for job_posting in scraped_jobs]
        current_jpes = [jpe for jpe in current_jpes if jpe.successfully_parsed()]
        current_ngrams_list = []
        current_jpes_ngrams_indices = []
        for jpe in current_jpes:
            jpe._ngram_size = ngram_size
            jpe._ngrams_list = jpe._preprocess_job_posting()

            current_len = len(current_ngrams_list)
            current_ngrams_list += jpe._ngrams_list
            post_len = len(current_ngrams_list)

            current_jpes_ngrams_indices += (current_len, post_len)

        assert spu.is_port_in_use(cfg.bert_port), f'Bert As Service port not in use ({cfg.bert_port}).'
        with BertClient(ignore_all_checks=True) as BaaS:
            master_ngrams_encoded = BaaS.encode(current_ngrams_list)

        # Redistribute batched collected encodings to jpes
        for index, jpe in enumerate(current_jpes):
            start, stop = current_jpes_ngrams_indices[index]
            jpe._ngrams_encoded = master_ngrams_encoded[start:stop]

        # Update reqs based on successes
        gathered_jpes.update(current_jpes)

        # Prepare for next loop iteration
        current_page += 1
