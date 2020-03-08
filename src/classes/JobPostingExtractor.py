from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams
import re
import numpy as np
import src.pipeline.utils as spu
from config import Config as cfg
from src.pipeline.utils import setup_extractions_logger
import logging

setup_extractions_logger(__name__)
logger = logging.getLogger(__name__)


class JobPostingExtractor:
    def __init__(self, job_posting):
        self._job_posting = job_posting
        self._parsed_job_posting = self._job_posting.parse()
        self._parse_failed = self._parsed_job_posting == 'Failed to parse.'
        self._ngram_size, self._ngram_stride, self._ngrams_list, self._ngrams_encoded = None, None, None, None

    def set_encodings(self, ngram_size, ngram_stride=1):
        if self._parse_failed:
            logger.debug('Job posting failed to parse. No encodings will be set.')
            return
        self._ngram_size = ngram_size
        self._ngram_stride = ngram_stride
        self._ngrams_list = self._preprocess_job_posting()
        self._ngrams_encoded = self._encode_job_posting_ngrams()

    def extract_required_years_experience(self, reference_sentences=('5+ years of experience', '2-4 years relevant experience', 'required experience: 7 years',), threshold=0.9):
        """Returns average of years of experience"""
        similarities = self._get_similarities(reference_sentences=reference_sentences, threshold=threshold)

        if similarities is None:
            logger.debug(f'No similarities found for years experience above threshold of {threshold}. Returning None.')
            return None

        years_exp_pattern = r'\d+'
        years_exp = self._scan_valid_ngrams(similarities=similarities, pattern=years_exp_pattern)

        if years_exp is None or len(years_exp) == 0:
            logger.debug('Could not find a number of years of experience. Returning None.')
            return None

        selected_years_exp = np.mean([int(year) for year in years_exp])

        return selected_years_exp

    def extract_required_degree(self, reference_sentences=('education: bachelors degree, masters degree, phd or higher',), threshold=0.89):
        """Returns list of degrees mentioned"""
        similarities = self._get_similarities(reference_sentences=reference_sentences, threshold=threshold)

        if similarities is None:
            logger.debug(f'No similarities found for required degree above threshold of {threshold}. Returning None.')
            return None

        degree_pattern = r'(\bbs\b|\bms\b|\bphd\b|bachelor|master|doctorate|advanced)'
        degrees = self._scan_valid_ngrams(similarities=similarities, pattern=degree_pattern)

        if degrees is None or len(degrees) == 0:
            logger.debug('Could not find a degree in parsed job posting. Returning None.')
            return None

        selected_degrees = []
        for degree in degrees:
            if 'bs' in degree or 'bachelor' in degree:
                selected_degrees.append('BS')
            if 'ms' in degree or 'master' in degree:
                selected_degrees.append('MS')
            if 'phd' in degree or 'doctorate' in degree:
                selected_degrees.append('PhD')
            if 'advanced' in degree:
                selected_degrees.append('Advanced Degree')
        return selected_degrees

    def extract_travel_percentage(self, reference_sentences=('Travel: 25-50%',), threshold=0.9):
        """Returns an average integer percentage"""
        similarities = self._get_similarities(reference_sentences=reference_sentences, threshold=threshold)

        if similarities is None:
            logger.debug(f'No similarities found for travel percentage above threshold of {threshold}. Returning None.')
            return None

        travel_percentage_pattern = r'\d+'
        travel_percentages = self._scan_valid_ngrams(similarities=similarities, pattern=travel_percentage_pattern)

        if travel_percentages is None or len(travel_percentages) == 0:
            logger.debug('Could not find valid travel percentages. Returning None.')
            return None

        travel_percentages = [x for x in travel_percentages if float(x) <= 100]
        if len(travel_percentages) == 0:
            logger.debug('Could not find travel percentages below 100%.')
            return None

        parsed_travel_percentages = [float(travel_percentage) for travel_percentage in travel_percentages]
        selected_travel_percentage = np.mean(parsed_travel_percentages)

        return selected_travel_percentage

    def extract_salary(self, reference_sentences=('Salary: $50,000 - $60,000 / year with bonus',), threshold=0.9):
        """Returns a tuple containing average annualized salary and whether bonus is mentioned"""
        if self._job_posting.salary:
            similarities = np.array((self._job_posting.salary, 1)).reshape(1, -1)
        else:
            similarities = self._get_similarities(reference_sentences=reference_sentences, threshold=threshold)

            if similarities is None:
                logger.debug(f'No similarities found for salary above threshold of {threshold}. Returning None.')
                return None

        rate_pattern = r'(?:[\£\$\€]{1}[,\d]+.?\d*)'
        rates = self._scan_valid_ngrams(similarities=similarities, pattern=rate_pattern)

        if rates is None or len(rates) == 0:
            logger.debug('No valid rates found for salary calculation.')
            return None

        parsed_rates = [float(rate[1:].replace(',', '')) for rate in rates]
        selected_rate = np.mean(parsed_rates)

        period_pattern = r'(hour|year|annually|daily|day|week|month)'
        period = self._scan_valid_ngrams(similarities=similarities, pattern=period_pattern)

        if period is None:
            logger.debug('No valid periods found for salary calculation. Assuming default of annual period.')
        elif 'hour' in period:
            selected_rate *= 40 * 52
        elif 'da' in period:
            selected_rate *= 5 * 52
        elif 'week' in period:
            selected_rate *= 52
        elif 'month' in period:
            selected_rate *= 12

        bonus_pattern = r'(bonus)'
        bonus = self._scan_valid_ngrams(similarities=similarities, pattern=bonus_pattern)
        bonus_mentioned = bonus is not None and len(bonus) > 0

        return selected_rate, bonus_mentioned

    def extract_benefits(self, reference_sentences=('',), threshold=0.9):
        similarities = self._get_similarities(reference_sentences=reference_sentences, threshold=threshold)

        if similarities is None:
            return None

    def _preprocess_job_posting(self):
        text = self._parsed_job_posting.lower()
        tokens = [token for token in text.split(' ') if token != '']
        output = list(ngrams(tokens, self._ngram_size))
        return [' '.join(tuples) for i, tuples in enumerate(output) if i % self._ngram_stride == 0]

    def _encode_job_posting_ngrams(self):
        """Returns a list of ngrams from job posting as well as the BaaS encoded representations. Returns None if parse fails."""
        if self._parse_failed:
            logger.debug('Parse failed. Returning None.')
            return None

        with self._get_BaaS() as BaaS:
            ngrams_encoded = BaaS.encode(self._ngrams_list)

        return ngrams_encoded

    def _get_similarities(self, reference_sentences, threshold):
        """
        Calculates cosine similarity between BertAsService encodings of every ngram within a job posting
        :param reference_sentences: Iterable of sentences to compute similarities against
        :return: NumPy Array, sorted in reverse order by cosine similarity
        """
        if isinstance(reference_sentences, str):
            reference_sentences = (reference_sentences, )

        assert len(reference_sentences) > 0

        if self._parse_failed:
            logger.debug('Job posting failed to parse. Returning None.')
            return None

        all_similarities, similarities = None, None
        for reference_sentence in reference_sentences:
            with self._get_BaaS() as BaaS:
                reference_sentence_encoded = BaaS.encode([reference_sentence])

            similarities = cosine_similarity(self._ngrams_encoded, reference_sentence_encoded)
            zipped = zip(self._ngrams_list, similarities)
            similarities = np.array(sorted(zipped, key=lambda x: x[1], reverse=True))
            similarities = similarities[similarities[:, 1] > threshold]

        if not all_similarities:
            all_similarities = similarities
        else:
            for similarity in similarities:
                if similarity[0] in all_similarities:
                    index = np.where(all_similarities[:, 0] == similarity[0])
                    all_similarities[index, 1] = max(similarity[1], all_similarities[index, 1])
                else:
                    np.concatenate((all_similarities, similarities), axis=0)

        if len(all_similarities) == 0:
            return None

        return all_similarities

    @staticmethod
    def _scan_valid_ngrams(similarities, pattern, return_first_only=True):
        """Scans through similarities array (sorted) to find the first match on the provided pattern. If no match is found, returns None"""
        matched = None
        for i in range(len(similarities)):
            matched = re.findall(pattern, similarities[i, 0])
            if return_first_only and len(matched) > 0:
                return matched
        return matched

    @staticmethod
    def _get_BaaS():
        assert spu.is_port_in_use(cfg.bert_port), f'Bert As Service port not in use ({cfg.bert_port}).'
        return BertClient(ignore_all_checks=True)

    def __str__(self):
        return self._parsed_job_posting

    def successfully_parsed(self):
        return not self._parse_failed

    def get_job_posting(self):
        return self._job_posting
