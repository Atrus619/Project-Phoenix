from prettytable import PrettyTable


class ScrapedJobs:
    def __init__(self, source, job_postings):
        self.source = source
        self._job_postings = job_postings

        self._table = None
        self._build_table()

    def __len__(self):
        return len(self._job_postings)

    def __str__(self):
        return str(self._table)

    def __getitem__(self, item):
        return self._job_postings[item]

    def __iter__(self):
        yield from self._job_postings

    def _build_table(self):
        """Run on construction so that there is a pretty printable format for this class"""
        table = PrettyTable()
        table.field_names = ['Index', 'Job Title', 'Company', 'Location']
        for index, posting in enumerate(self._job_postings):
            table.add_row([index, posting.job_title, posting.company, posting.location])
        self._table = table

    def append(self, scraped_jobs):
        """Joins scraped_jobs objects together"""
        assert type(scraped_jobs) is ScrapedJobs
        self._job_postings += scraped_jobs.job_postings()
        self._build_table()

    def job_postings(self):
        return self._job_postings
