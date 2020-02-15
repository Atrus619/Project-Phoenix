from src.classes.ScrapedJobs import ScrapedJobs
from wordcloud import WordCloud, STOPWORDS
from config import Config as cfg
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams
import re
import numpy as np
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


def extract_required_experience(job_posting):
    similarities = get_similarities(job_posting, reference_sentence, ngram_size)


def extract_qualifications(job_posting):
    similarities = get_similarities(job_posting, reference_sentence, ngram_size)


def extract_salary(job_posting, reference_sentence='Salary: $50,000 / year', ngram_size=6, threshold=0.9):
    similarities = get_similarities(job_posting, reference_sentence, ngram_size)
    if similarities is None:
        return None

    if not np.any(similarities[:, 1] > threshold):
        return None

    rate_pattern = r'(?:[\£\$\€]{1}[,\d]+.?\d*)'
    rates = scan_valid_ngrams(similarities, threshold, rate_pattern)

    if rates is None:
        return None

    parsed_rates = [float(rate[1:].replace(',', '')) for rate in rates]
    selected_rate = np.mean(parsed_rates)

    period_pattern = r'(hour|year|annually|daily|day|week|month)'
    period = scan_valid_ngrams(similarities, threshold, period_pattern)

    if period is None:
        return selected_rate
    elif 'hour' in period:
        selected_rate *= 40 * 52
    elif 'da' in period:
        selected_rate *= 5 * 52
    elif 'week' in period:
        selected_rate *= 52
    elif 'month' in period:
        selected_rate *= 12

    return selected_rate


def extract_benefits(job_posting):
    assert_BaaS_running()


def scan_valid_ngrams(similarities, threshold, pattern):
    """Scans through similarities array (sorted) to find the first match on the provided pattern. If no match is found, returns None"""
    num_ngrams_above_threshold = (similarities[:, 1] > threshold).sum()
    for i in range(num_ngrams_above_threshold):
        matched = re.findall(pattern, similarities[i, 0])
        if len(matched) > 0:
            return matched
    return None


def preprocess_job_posting(job_posting, ngram_size):
    text = job_posting.parse()
    if text == 'Failed to parse.':
        return None

    text = text.lower()
    tokens = [token for token in text.split(' ') if token != '']
    output = list(ngrams(tokens, ngram_size))
    return [' '.join(tuples) for tuples in output]


def get_similarities(job_posting, reference_sentence, ngram_size):
    """
    Calculates cosine similarity between BertAsService encodings of every ngram within a job posting
    :param job_posting: JobPosting object (essentially a named tuple, with some minor additional functionality)
    :param reference_sentence: Sentence to compute similarities against
    :param ngram_size: Size of ngram to generate from job posting
    :return: NumPy Array, sorted in reverse order by cosine similarity
    """
    assert_BaaS_running()
    BaaS = BertClient(ignore_all_checks=True)

    reference_sentence_encoded = BaaS.encode([reference_sentence])

    list_of_ngrams = preprocess_job_posting(job_posting=job_posting, ngram_size=ngram_size)
    if list_of_ngrams is None:
        return None

    ngrams_encoded = BaaS.encode(list_of_ngrams)

    similarities = cosine_similarity(ngrams_encoded, reference_sentence_encoded)

    zipped = zip(list_of_ngrams, similarities)
    return np.array(sorted(zipped, key=lambda x: x[1], reverse=True))


def assert_BaaS_running():
    assert spu.is_port_in_use(cfg.bert_port), 'Bert As Service not currently running'
