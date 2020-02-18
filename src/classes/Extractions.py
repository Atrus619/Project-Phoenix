import requests
from src.classes.Scraper import Scraper
from src.classes.JobPostingExtractor import JobPostingExtractor
from bert_serving.client import BertClient
import src.pipeline.utils as spu
from config import Config as cfg
import logging
from prettytable import PrettyTable


class Extractions:
    def __init__(self, required_years_experience=0, required_degree=0, travel_percentage=0, salary=0):
        self.required_years_experience = required_years_experience
        self.required_degree = required_degree
        self.travel_percentage = travel_percentage
        self.salary = salary

        self.scraped_jobs = None
        self.extracted_required_years_experience, self.extracted_required_degrees, self.extracted_travel_percentages, self.extracted_salaries = [], [], [], []

        self._current_page = 1

    def is_complete(self):
        return self.required_years_experience == self.required_degree == self.travel_percentage == self.salary == 0

    def _update(self, jpes):
        for jpe in jpes:
            extracted_years_experience = jpe.extract_required_years_experience()
            if extracted_years_experience is not None:
                self.required_years_experience -= 1
                self.extracted_required_years_experience.append(extracted_years_experience)

            extracted_required_degree = jpe.extract_required_degree()
            if extracted_required_degree is not None:
                self.required_degree -= 1
                self.extracted_required_degrees.append(extracted_required_degree)

            extracted_travel_percentage = jpe.extract_travel_percentage()
            if extracted_travel_percentage is not None:
                self.travel_percentage -= 1
                self.extracted_travel_percentages.append(extracted_travel_percentage)

            extracted_salary = jpe.extract_salary()
            if extracted_salary is not None:
                self.salary -= 1
                self.extracted_salaries.append(extracted_salary)

    def gather(self, job, location, ngram_size=8, vpn=True, max_iters=10):
        """Gathers jpes based on specified job and location until requirements are satisfied. Intended to be called asynchronously with redis"""
        logger = logging.getLogger('extractions')
        logger.info(f'Beginning gathering of extractions. {self}')
        while (not self.is_complete()) and (self._current_page <= max_iters):
            # Scrape an entire page off of indeed
            logger.info(f'-----Scraping page {self._current_page} of indeed for {job} in {location} {"" if vpn else "not"} using vpn.-----')
            try:
                scraped_jobs = Scraper().scrape_page_indeed(job_title=job, location=location, page=self._current_page, vpn=vpn)
            except requests.exceptions.ConnectionError as e:
                logger.warn(f'{e}')
                logger.info(f'Due to connection error, skipping current page ({self._current_page}) and moving to the next one.')
                continue

            # Batch collect encodings for jpes where parsing did not fail
            logger.info('Creating job posting extractors for scraped jobs.')
            current_jpes = [JobPostingExtractor(job_posting) for job_posting in scraped_jobs]
            current_jpes = [jpe for jpe in current_jpes if jpe.successfully_parsed()]
            logger.info(f'{len(current_jpes)} / {len(scraped_jobs)} job postings successfully parsed.')
            if len(current_jpes) == 0:
                logger.info(f'Since zero job postings from this page ({self._current_page}) were successfully parsed, skipping to next page.')
                continue

            current_ngrams_list = []
            current_jpes_ngrams_indices = []
            for jpe in current_jpes:
                jpe._ngram_size = ngram_size
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

            # Redistribute batched collected encodings to jpes
            logger.info('Redistributing batched encodings to jpes.')
            for index, jpe in enumerate(current_jpes):
                start, stop = current_jpes_ngrams_indices[index]
                jpe._ngrams_encoded = master_ngrams_encoded[start:stop]

            # Update reqs based on successes
            self._update(current_jpes)
            logger.info(f'Requirements updated. {self}')

            # Prepare for next loop iteration
            if self.scraped_jobs is None:
                self.scraped_jobs = scraped_jobs
            else:
                self.scraped_jobs.append(scraped_jobs)
            self._current_page += 1

        logger.info(f'Gather completed in {self._current_page - 1} pages. Requirements {"" if self.is_complete() else "not"} successfully met.')

    def __str__(self):
        table = PrettyTable()
        table.field_names = ['Years Experience', 'Required Degree', 'Travel Percentage', 'Salary']
        table.add_row([self.required_years_experience, self.required_degree, self.travel_percentage, self.salary])
        return f'Remaining requirements\n{table}'
