from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams
import re
import numpy as np
import src.pipeline.utils as spu
from warnings import warn
from config import Config as cfg


class JobPostingExtractor:
    def __init__(self, job_posting):
        self._job_posting = job_posting
        self._parsed_job_posting = self._job_posting.parse()
        self._parse_failed = self._parsed_job_posting == 'Failed to parse.'
        self._ngram_size, self._ngrams_list, self._ngrams_encoded = None, None, None

    def set_encodings(self, ngram_size):
        if self._parse_failed:
            warn('Job posting failed to parse. No encodings will be set.')
            return
        self._ngram_size = ngram_size
        self._ngrams_list = self._preprocess_job_posting()
        self._parse_failed = self._ngrams_list is None
        self._ngrams_encoded = self._encode_job_posting_ngrams()

    def extract_required_years_experience(self, reference_sentence='5+ years of experience', threshold=0.9):
        similarities = self._get_similarities(reference_sentence=reference_sentence, threshold=threshold)

        if similarities is None:
            return None

        years_exp_pattern = r'\d+'
        years_exp = self._scan_valid_ngrams(similarities=similarities, pattern=years_exp_pattern)

        if years_exp is None:
            warn('Could not find a number of years of experience. Returning None.')
            return None

        selected_years_exp = np.mean([int(year) for year in years_exp])

        return selected_years_exp

    def extract_required_degree(self, reference_sentence='education: bachelors degree, masters degree, phd or higher', threshold=0.9):
        similarities = self._get_similarities(reference_sentence=reference_sentence, threshold=threshold)

        if similarities is None:
            return None

        degree_pattern = r'(\bbs\b|\bms\b|\bphd\b|bachelor|master|doctorate|advanced)'
        degrees = self._scan_valid_ngrams(similarities=similarities, pattern=degree_pattern)

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

    def extract_travel_percentage(self, reference_sentence='', threshold=0.9):
        similarities = self._get_similarities(reference_sentence=reference_sentence, threshold=threshold)

        if similarities is None:
            return None

    def extract_salary(self, reference_sentence='Salary: $50,000 / year', threshold=0.9):
        similarities = self._get_similarities(reference_sentence=reference_sentence, threshold=threshold)

        if similarities is None:
            return None

        rate_pattern = r'(?:[\£\$\€]{1}[,\d]+.?\d*)'
        rates = self._scan_valid_ngrams(similarities=similarities, pattern=rate_pattern)

        if rates is None:
            warn('No valid rates found for salary calculation.')
            return None

        parsed_rates = [float(rate[1:].replace(',', '')) for rate in rates]
        selected_rate = np.mean(parsed_rates)

        period_pattern = r'(hour|year|annually|daily|day|week|month)'
        period = self._scan_valid_ngrams(similarities=similarities, pattern=period_pattern)

        if period is None:
            warn('No valid periods found for salary calculation. Assuming default of annual period.')
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

    def extract_benefits(self, reference_sentence='', threshold=0.9):
        similarities = self._get_similarities(reference_sentence=reference_sentence, threshold=threshold)

        if similarities is None:
            return None

    def _preprocess_job_posting(self):
        text = self._parsed_job_posting.lower()
        tokens = [token for token in text.split(' ') if token != '']
        output = list(ngrams(tokens, self._ngram_size))
        return [' '.join(tuples) for tuples in output]

    def _encode_job_posting_ngrams(self):
        """Returns a list of ngrams from job posting as well as the BaaS encoded representations. Returns None if parse fails."""
        if self._parse_failed:
            return None

        BaaS = self._get_BaaS()

        ngrams_encoded = BaaS.encode(self._ngrams_list)
        return ngrams_encoded

    def _get_similarities(self, reference_sentence, threshold):
        """
        Calculates cosine similarity between BertAsService encodings of every ngram within a job posting
        :param reference_sentence: Sentence to compute similarities against
        :return: NumPy Array, sorted in reverse order by cosine similarity
        """
        if self._parse_failed:
            warn('Job posting failed to parse. Returning None.')
            return None

        BaaS = self._get_BaaS()

        reference_sentence_encoded = BaaS.encode([reference_sentence])

        similarities = cosine_similarity(self._ngrams_encoded, reference_sentence_encoded)

        zipped = zip(self._ngrams_list, similarities)
        similarities = np.array(sorted(zipped, key=lambda x: x[1], reverse=True))
        similarities = similarities[similarities[:, 1] > threshold]

        if len(similarities) == 0:
            warn(f'No similarities found above threshold of {threshold}.')
            return None

        return similarities

    @staticmethod
    def _scan_valid_ngrams(similarities, pattern):
        """Scans through similarities array (sorted) to find the first match on the provided pattern. If no match is found, returns None"""
        for i in range(len(similarities)):
            matched = re.findall(pattern, similarities[i, 0])
            if len(matched) > 0:
                return matched
        return None

    @staticmethod
    def _get_BaaS():
        assert spu.is_port_in_use(cfg.bert_port), f'Bert As Service port not in use ({cfg.bert_port}).'
        return BertClient(ignore_all_checks=True)

    def __str__(self):
        return self._parsed_job_posting
