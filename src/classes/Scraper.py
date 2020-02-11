import src.scraping.scraping_indeed as indeed
import src.scraping.utils as su
from src.constants import Constants as cs
from src.classes.ScrapedJobs import ScrapedJobs
from config import Config as cfg
import random
import os
import requests
from src.classes.JobPosting import JobPosting


class Scraper:
    def __init__(self):
        # Build list of IPVanish Servers
        self._server_list = su.build_ipvanish_server_list(cs.ipvanish_base_links)
        random.shuffle(self._server_list)

        # Import list of User Agents
        self._user_agents = cs.user_agents
        random.shuffle(self._user_agents)

        self._valid_sources = ('indeed', 'monster')

    def rotate_user_agent(self):
        """Moves the first user agent to the end of the list, so that the first item in the list can be used."""
        self._user_agents.append(self._user_agents.pop(0))

    def rotate_ip(self):
        """
        Changes IP to the first server in the list and moves the first item in the list to the end
        :requires: server_list: List of IP Vanish servers (modifies external object in place)
        :return: Updated server list
        """
        while su.ipvanish_connect(self._server_list[0]) != 0:
            self._server_list.append(self._server_list.pop(0))
        self._server_list.append(self._server_list.pop(0))

    def scrape_page_indeed(self, job_title, location, page=1, vpn=False):
        base_url = indeed.build_url(job_title=job_title, location=location)
        url = indeed.build_url_page_n(url=base_url, n=page)

        if vpn and not su.is_ipvanish_up():
            self.rotate_ip()

        self.rotate_user_agent()

        with requests.Session() as session:
            try:
                soup = su.get_soup(session=session, url=url, user_agent=self._user_agents[0])
            except requests.exceptions.ConnectionError as e:
                raise e

        jobs = indeed.extract_job_title_from_result(soup)
        companies = indeed.extract_company_from_result(soup)
        locations = indeed.extract_location_from_result(soup)
        links = indeed.extract_job_link_from_result(soup)
        descrs = []

        for j, link in enumerate(links):
            try:
                descrs.append(indeed.extract_description_html_from_link(session=session, link=link, user_agent=self._user_agents[0], og_page_url=url))
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                descrs.append('Description Unavailable')

        job_postings = [JobPosting(*x) for x in zip(jobs, companies, locations, links, descrs)]
        scraped_jobs = ScrapedJobs(source='indeed', job_postings=job_postings)

        return scraped_jobs
