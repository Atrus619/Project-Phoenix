import src.scraping.scraping_indeed as indeed
import src.scraping.scraping_monster as monster
import src.db as db
import src.scraping.utils as su
from config import Config as cfg
from src.constants import Constants as cs
import random
import os
import logging
from datetime import date
import pprint
import requests
import time


def scrape_dataset(search_params, num_pages, source):
    """
    Accepts arguments describing the search parameters (probably a list of tuples of arguments for build_url)
    Pulls information from indeed, parses it into meaningful components, and inserts this information to mongodb
    :param search_params: LIST OF TUPLES (job title, location)
    :param num_pages: NUM
    :param source: STRING name of website to pull links from (either indeed or monster as of now)
    :return: Nothing, adds extracted information to mongodb
    """
    # Initialize logger
    su.setup_scrape_logger(name=cfg.scrape_log_name, filename=cfg.scrape_log_name)
    logger = logging.getLogger(cfg.scrape_log_name)

    su.setup_scrape_logger(name=cfg.scrape_error_log_name, filename=cfg.scrape_error_log_name)
    error_logger = logging.getLogger(cfg.scrape_error_log_name)

    logger.info(f'Successfully initialized logger. Beginning scraping on {date.today()} for {num_pages}'
                f' pages each of {len(search_params)} jobs from {source}.')
    logger.info('Search Parameters:\n' + pprint.pformat(search_params))
    start_time = time.time()

    # Define helper functions
    def rotate_user_agent():
        # Moves the first user agent to the end of the list, so that the first item in the list can be used.
        user_agents.append(user_agents.pop(0))

    def rotate_ip():
        """
        Changes IP to the first server in the list and moves the first item in the list to the end
        :requires: server_list: List of IP Vanish servers (modifies external object in place)
        :return: Updated server list
        """
        while os.system('echo %s|sudo -S %s' % (cfg.sudo_password, './src/scraping/change_ip.sh ' + server_list[0])) != 0:
            error_logger.exception(f'VPN address {server_list[0]} invalid. Trying next VPN address.')
            server_list.append(server_list.pop(0))
        return server_list.append(server_list.pop(0))

    # Set correct functions
    if source == 'indeed':
        build_url = indeed.build_url
        build_url_page_n = indeed.build_url_page_n
        extract_job_title_from_result = indeed.extract_job_title_from_result
        extract_company_from_result = indeed.extract_company_from_result
        extract_location_from_result = indeed.extract_location_from_result
        extract_job_link_from_result = indeed.extract_job_link_from_result
        extract_description_html_from_link = indeed.extract_description_html_from_link
    else:  # monster
        build_url = monster.build_url
        build_url_page_n = monster.build_url_page_n
        extract_job_title_from_result = monster.extract_job_title_from_result
        extract_company_from_result = monster.extract_company_from_result
        extract_location_from_result = monster.extract_location_from_result
        extract_job_link_from_result = monster.extract_job_link_from_result
        extract_description_html_from_link = monster.extract_description_html_from_link

    # Build list of IPVanish Servers
    server_list = su.build_ipvanish_server_list(cs.ipvanish_base_links)
    random.shuffle(server_list)

    # Import list of User Agents
    user_agents = cs.user_agents
    random.shuffle(user_agents)

    for job_title, location in search_params:
        base_url = build_url(job_title=job_title, location=location)
        pages = list(range(num_pages))

        loop_time = time.time()
        logger.info(f'Beginning scrape for a {job_title} in {location}. Total time elapsed: {su.get_pretty_time(loop_time - start_time)}.')

        for i in range(num_pages):
            skip_page = False
            page = pages.pop(pages.index(random.choice(pages))) + 1  # Select pages in random order

            rotate_ip()  # Connect to VPN and change IP address
            rotate_user_agent()  # Cycle through user agents
            su.random_pause(min_pause=cfg.min_pause, max_pause=cfg.max_pause)  # Random wait time between requests

            # Log request
            logger.info(f'Using VPN {server_list[-1]} for page {page} for a/an {job_title} in {location} from {source}...')
            page_time = time.time()

            url = build_url_page_n(url=base_url, n=page)  # Build url

            with requests.Session() as session:
                k = 1
                while 1:
                    try:
                        soup = su.get_soup(session=session, url=url, user_agent=user_agents[0], logger=logger)  # Retrieve page from website
                        break
                    except requests.exceptions.ConnectionError as e:
                        if k > cfg.max_retry_attempts:
                            error_logger.exception('Maximum number of retry attempts exceeded. Moving on to next page...')
                            skip_page = True
                            db.insert_error(job_title=job_title, location=location, page=page, error=e, source=source)
                            break
                        else:
                            k += 1
                            error_logger.exception(
                                f'Error accessing page {page} for a/an {job_title} in {location} from {source}. Retrying after '
                                f'waiting for {cfg.fail_wait_time}. Attempt {k} / {cfg.max_retry_attempts}.')
                            time.sleep(cfg.fail_wait_time)

                if skip_page:
                    continue
                else:
                    pass

                jobs = extract_job_title_from_result(soup)
                companies = extract_company_from_result(soup)
                locations = extract_location_from_result(soup)
                links = extract_job_link_from_result(soup)
                descrs = []

                logger.info(f'Extracting job links and descriptions from {len(links)} jobs...')
                for j, link in enumerate(links):
                    su.random_pause(min_pause=cfg.min_pause, max_pause=cfg.max_pause)

                    try:
                        descrs.append(extract_description_html_from_link(session=session, link=link, user_agent=user_agents[0], og_page_url=url, logger=logger))
                    except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                        error_logger.exception(f"Error extracting job description link from {companies[j]}'s posting for a/an {jobs[j]} in {locations[j]}"
                                               f'. Link: {link}')
                        descrs.append(cfg.job_description_link_fail_msg)
                        db.insert_error(job_title=jobs[j], company=companies[j], location=locations[j], page=page, source=source, error=e)

            logger.info(f'Scraping for current page complete. Elapsed time for page: {su.get_pretty_time(time.time() - page_time)}.')
            logger.info('Inserting data into MongoDB...')

            db.insert_data(jobs=jobs, companies=companies, locations=locations, descrs=descrs, source=source)

        logger.info(f'Scraping for all pages for job/loc combo complete. Elapsed time for job/loc: {su.get_pretty_time(time.time() - loop_time)}.')
        logger.info(cs.page_break)

    logger.info(f'Scraping complete. Total elapsed time: {su.get_pretty_time(time.time() - start_time)}.')
