from src.classes.Scraper import Scraper
from src.classes.JobPostingExtractor import JobPostingExtractor
from bert_serving.client import BertClient
import src.pipeline.utils as spu
from config import Config as cfg


class Extractions:
    __slots__ = 'required_years_experience', 'required_degree', 'travel_percentage', 'salary', \
                'extracted_required_years_experience', 'extracted_required_degrees', 'extracted_travel_percentages', 'extracted_salaries'

    def __init__(self, required_years_experience=0, required_degree=0, travel_percentage=0, salary=0):
        self.required_years_experience = required_years_experience
        self.required_degree = required_degree
        self.travel_percentage = travel_percentage
        self.salary = salary

        self.extracted_required_years_experience, self.extracted_required_degrees, self.extracted_travel_percentages, self.extracted_salaries = [], [], [], []

    def is_complete(self):
        return self.required_years_experience == self.required_degree == self.travel_percentage == self.salary == 0

    def update(self, jpes):
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
                self.required_degree -= 1
                self.extracted_travel_percentages.append(extracted_travel_percentage)

            extracted_salary = jpe.extract_salary()
            if extracted_salary is not None:
                self.salary -= 1
                self.extracted_salaries.append(extracted_salary)

    def gather(self, job, location, vpn=True, ngram_size=8, max_iters=10):
        """Gathers jpes based on specified job and location until requirements are satisfied. Intended to be called asynchronously with redis"""
        current_page = 1
        while (not self.is_complete()) and (current_page <= max_iters):
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

            # Encode entire batch in one go
            assert spu.is_port_in_use(cfg.bert_port), f'Bert As Service port not in use ({cfg.bert_port}).'
            with BertClient(ignore_all_checks=True) as BaaS:
                master_ngrams_encoded = BaaS.encode(current_ngrams_list)

            # Redistribute batched collected encodings to jpes
            for index, jpe in enumerate(current_jpes):
                start, stop = current_jpes_ngrams_indices[index]
                jpe._ngrams_encoded = master_ngrams_encoded[start:stop]

            # Update reqs based on successes
            self.update(current_jpes)

            # Prepare for next loop iteration
            current_page += 1
