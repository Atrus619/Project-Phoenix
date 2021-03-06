from prettytable import PrettyTable
from collections import namedtuple
import re
import requests
from src.classes.Scraper import Scraper
from src.classes.ScrapedJobs import ScrapedJobs
from src.classes.JobPostingExtractor import JobPostingExtractor
from bert_serving.client import BertClient
import src.pipeline.utils as spu
from config import Config as cfg
import logging
from prettytable import PrettyTable
from src.pipeline.utils import setup_extractions_logger

setup_extractions_logger(__name__)
logger = logging.getLogger(__name__)


class Extractions:
    def __init__(self, required_years_experience=0, required_degree=0, travel_percentage=0, salary=0):
        self.required_years_experience = required_years_experience
        self.required_degree = required_degree
        self.travel_percentage = travel_percentage
        self.salary = salary

        self.scraped_jobs, self.scraped_jobs_parsed = [], []
        self.extracted_required_years_experience, self.extracted_required_degrees, self.extracted_travel_percentages, self.extracted_salaries = [], [], [], []
        self.extracted_rye_indices, self.extracted_rd_indices, self.extracted_tp_indices, self.extracted_sal_indices = [],  [], [], []

        self._current_page = 1
        self._table = None

    def is_complete(self):
        return max(self.required_years_experience, self.required_degree, self.travel_percentage, self.salary) <= 0

    def _update(self, jpes):
        current_len = len(self.scraped_jobs_parsed)
        for i, jpe in enumerate(jpes):
            extracted_years_experience = jpe.extract_required_years_experience()
            if extracted_years_experience is not None:
                self.required_years_experience -= 1
                self.extracted_required_years_experience.append(extracted_years_experience)
                self.extracted_rye_indices.append(current_len + i)

            extracted_required_degree = jpe.extract_required_degree()
            if extracted_required_degree is not None:
                self.required_degree -= 1
                self.extracted_required_degrees.append(extracted_required_degree)
                self.extracted_rd_indices.append(current_len + i)

            extracted_travel_percentage = jpe.extract_travel_percentage()
            if extracted_travel_percentage is not None:
                self.travel_percentage -= 1
                self.extracted_travel_percentages.append(extracted_travel_percentage)
                self.extracted_tp_indices.append(current_len + i)

            extracted_salary = jpe.extract_salary()
            if extracted_salary is not None:
                self.salary -= 1
                self.extracted_salaries.append(extracted_salary)
                self.extracted_sal_indices.append(current_len + i)

    def _update_salary(self, jpes):
        current_len = len(self.scraped_jobs_parsed)
        for i, jpe in enumerate(jpes):
            extracted_salary = jpe.extract_salary()
            if extracted_salary is not None:
                self.salary -= 1
                self.extracted_salaries.append(extracted_salary)
                self.extracted_sal_indices.append(current_len + i)

    def gather(self, job, location, ngram_size=8, ngram_stride=2, vpn=False, max_iters=3, max_salary_only_iters=5, source='indeed'):
        """Gathers jpes based on specified job and location until requirements are satisfied. Intended to be called asynchronously with redis"""
        logger.info(f'Beginning gathering of extractions.\n{self}')
        while (not self.is_complete()) and (self._current_page <= max_iters):
            # Scrape an entire page off of indeed
            logger.info(f'-----Scraping page {self._current_page} of indeed for {job} in {location} {"" if vpn else "not"} using vpn.-----')
            try:
                if source == 'indeed':  # Only option currently TODO: Add more options
                    scraped_jobs = Scraper().scrape_page_indeed(job_title=job, location=location, page=self._current_page, vpn=vpn)
            except requests.exceptions.ConnectionError as e:
                logger.warn(f'{e}')
                logger.info(f'Due to connection error, skipping current page ({self._current_page}) and moving to the next one.')
                continue

            # Batch collect encodings for jpes where parsing did not fail
            logger.info('Creating job posting extractors for scraped jobs.')
            current_jpes = []
            for job_posting in scraped_jobs:
                jpe = JobPostingExtractor(job_posting)
                if jpe.successfully_parsed():
                    current_jpes.append(jpe)
            scraped_jobs_parsed = ScrapedJobs(job_postings=[jpe.get_job_posting() for jpe in current_jpes], source=source)  # Keep track of the job postings that were successfully parsed

            logger.info(f'{len(current_jpes)} / {len(scraped_jobs)} job postings successfully parsed.')
            if len(current_jpes) == 0:
                logger.info(f'Since zero job postings from this page ({self._current_page}) were successfully parsed, skipping to next page.')
                continue

            current_ngrams_list = []
            current_jpes_ngrams_indices = []
            for jpe in current_jpes:
                jpe._ngram_size = ngram_size
                jpe._ngram_stride = ngram_stride
                jpe._ngrams_list = jpe._preprocess_job_posting()

                current_len = len(current_ngrams_list)
                current_ngrams_list += jpe._ngrams_list
                post_len = len(current_ngrams_list)

                current_jpes_ngrams_indices.append((current_len, post_len))

            # Encode entire batch in one go
            logger.info(f'Encoding all ngrams in a single batch.')
            assert spu.is_port_in_use(cfg.bert_port), f'Bert As Service port not in use ({cfg.bert_port}).'
            with BertClient(ignore_all_checks=True) as BaaS:
                # TODO: Possibly save on time here by doing more preprocessing for BaaS?
                master_ngrams_encoded = BaaS.encode(current_ngrams_list)

            # Wrote this as a response to a weird bug found earlier - "ValueError: Found array with 0 sample(s) (shape=(0, 1024)) for cosine_similarity"
            if not len(master_ngrams_encoded):
                logger.info('master_ngrams_encoded has len 0 - skipping to next page.')
                continue

            # Redistribute batched collected encodings to jpes
            logger.info('Redistributing batched encodings to jpes.')
            for index, jpe in enumerate(current_jpes):
                start, stop = current_jpes_ngrams_indices[index]
                jpe._ngrams_encoded = master_ngrams_encoded[start:stop]

            # Update reqs based on successes
            self._update(current_jpes)
            logger.info(f'Requirements updated.\n{self}')

            # Prepare for next loop iteration
            self._append_scraped_jobs(scraped_jobs, scraped_jobs_parsed)
            self._current_page += 1

            if self.all_except_salary_complete() and (self.salary > 0):
                self.gather_salary_only(job=job, location=location, ngram_size=ngram_size, ngram_stride=ngram_stride, vpn=vpn, max_iters=max_salary_only_iters, source=source)

        self._build_table()
        logger.info(f'Gather completed in {self._current_page - 1} pages. Requirements{"" if self.is_complete() else " not"} successfully met.')

    def gather_salary_only(self, job, location, ngram_size, ngram_stride, vpn, max_iters, source):
        logger.info(f'----------All other requirements met, now searching for salary specifically (need {self.salary} more).----------')

        while (self.salary > 0) and (self._current_page <= max_iters):
            # Scrape an entire page off of indeed
            logger.info(f'-----Scraping page {self._current_page} of indeed for {job} in {location} {"" if vpn else "not"} using vpn for salaries only.-----')
            try:
                scraped_jobs = Scraper().scrape_page_indeed(job_title=job, location=location, page=self._current_page, vpn=vpn)
            except requests.exceptions.ConnectionError as e:
                logger.warn(f'{e}')
                logger.info(f'Due to connection error, skipping current page ({self._current_page}) and moving to the next one.')
                continue

            # Batch collect encodings for jpes where parsing did not fail
            logger.info('Creating job posting extractors for scraped jobs for salaries only.')
            current_jpes = []
            for job_posting in scraped_jobs:
                jpe = JobPostingExtractor(job_posting)
                if jpe.successfully_parsed():
                    current_jpes.append(jpe)

            logger.info(f'{len(current_jpes)} / {len(scraped_jobs)} job postings successfully parsed.')
            if len(current_jpes) == 0:
                logger.info(f'Since zero job postings from this page ({self._current_page}) were successfully parsed, skipping to next page.')
                continue

            logger.info(f'Skimming for salary information in {len(current_jpes)} postings.')
            current_found_salary_jpes = []
            current_found_salary_parseable_job_postings = []
            for i, jpe in enumerate(current_jpes):
                if jpe.get_job_posting().salary or self._salary_present(jpe):
                    current_found_salary_jpes.append(jpe)
                    current_found_salary_parseable_job_postings.append(jpe.get_job_posting())

            if len(current_found_salary_jpes) == 0:
                logger.info(f'No salary information found on current page {self._current_page}. Skipping to next page.')
                continue

            logger.info(f'Salary information potentially found in {len(current_found_salary_jpes)} postings. Parsing for salary amount and adding now.')
            for jpe in current_found_salary_jpes:
                if not jpe.get_job_posting().salary:  # If salary found in job posting span, don't need to consult BERT
                    jpe.set_encodings(ngram_size=ngram_size, ngram_stride=ngram_stride)

            self._update_salary(current_found_salary_jpes)

            scraped_jobs_parsed = ScrapedJobs(job_postings=current_found_salary_parseable_job_postings, source=source)
            self._append_scraped_jobs(scraped_jobs, scraped_jobs_parsed)

            logger.info(f'Requirements updated (searching for salary primarily).\n{self}')
            self._current_page += 1

    @staticmethod
    def _salary_present(jpe):
        salary_pattern = r'(salary|compensation)'
        match = re.findall(salary_pattern, str(jpe))
        return len(match) > 0

    def all_except_salary_complete(self):
        return max(self.required_years_experience, self.required_degree, self.travel_percentage) <= 0

    def __str__(self):
        table = PrettyTable()
        table.field_names = ['Years Experience', 'Required Degree', 'Travel Percentage', 'Salary']
        table.add_row([self.required_years_experience, self.required_degree, self.travel_percentage, self.salary])
        return f'Remaining requirements:\n{table}'

    def _append_scraped_jobs(self, scraped_jobs, scraped_jobs_parsed):
        """
        :param scraped_jobs: list of job posting extractors
        :param scraped_jobs_parsed: single scraped_jobs object
        :return:
        """
        if len(self.scraped_jobs) == 0:
            self.scraped_jobs = scraped_jobs
        else:
            self.scraped_jobs.append(scraped_jobs)
        if len(self.scraped_jobs_parsed) == 0:
            self.scraped_jobs_parsed = scraped_jobs_parsed
        else:
            self.scraped_jobs_parsed.append(scraped_jobs_parsed)

    def get_scraped_jobs(self, attribute=None):
        assert attribute in (None, 'rye', 'sal', 'tp', 'rd')
        if attribute == 'rye':
            return ScrapedJobs(self.scraped_jobs_parsed[self.extracted_rye_indices])
        elif attribute == 'sal':
            return ScrapedJobs(self.scraped_jobs_parsed[self.extracted_sal_indices])
        elif attribute == 'tp':
            return ScrapedJobs(self.scraped_jobs_parsed[self.extracted_tp_indices])
        elif attribute == 'rd':
            return ScrapedJobs(self.scraped_jobs_parsed[self.extracted_rd_indices])
        else:  # None
            return self.scraped_jobs_parsed

    def __len__(self, parsed=True):
        if parsed:
            return len(self.scraped_jobs_parsed)
        else:
            return len(self.scraped_jobs)

    def __getitem__(self, index, parsed=True):
        ExtractionsOutput = namedtuple('ExtractionsOutput', 'job_posting required_years_experience required_degree travel_percentage salary')
        if parsed:
            rye = self.extracted_required_years_experience[self.extracted_rye_indices.index(index)] if index in self.extracted_rye_indices else None
            rd = self.extracted_required_degrees[self.extracted_rd_indices.index(index)] if index in self.extracted_rd_indices else None
            tp = self.extracted_travel_percentages[self.extracted_tp_indices.index(index)] if index in self.extracted_tp_indices else None
            sal = self.extracted_salaries[self.extracted_sal_indices.index(index)] if index in self.extracted_sal_indices else None

            return ExtractionsOutput(self.scraped_jobs_parsed[index], rye, rd, tp, sal)
        else:
            adjusted_index = self.scraped_jobs_parsed.index(self.scraped_jobs[index]) if self.scraped_jobs[index] in self.scraped_jobs_parsed else None
            if adjusted_index:
                return self.__getitem__(adjusted_index, parsed=True)
            else:
                return ExtractionsOutput(self.scraped_jobs[index], None, None, None, None)

    def _build_table(self):
        # TODO: This probably needs to get fleshed out in a more visual-friendly format
        table = PrettyTable()
        table.field_names = ['Index', 'Job Title', 'Company', 'Location', 'Years Experience', 'Degree', 'Travel Percentage', 'Salary', 'Link']
        for index in range(len(self)):
            job_posting = self[index].job_posting
            table.add_row([index, job_posting.job_title, job_posting.company, job_posting.location,
                           self[index].required_years_experience, self[index].required_degree, self[index].travel_percentage, self[index].salary,
                           job_posting.link])
        self._table = table
        return

    def get_table(self):
        return str(self._table)
