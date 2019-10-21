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


# TODO: Add logging!
# TODO: Error with:
#  requests.exceptions.ConnectionError: HTTPSConnectionPool(host='careers.norc.org', port=443):
#  Max retries exceeded with url: /cw/en-us/job/495395/data-scientist
#  (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x7f6594a5fe10>:
#  Failed to establish a new connection: [Errno 101] Network is unreachable'))
# I THINK I CRASHED MY ROUTER?!?!?!


def scrape_dataset(search_params=cfg.search_params, num_pages=cfg.num_pages, source=cfg.source):
    """
    Accepts arguments describing the search parameters (probably a list of tuples of arguments for build_url)
    Pulls information from indeed, parses it into meaningful components, and inserts this information to mongodb
    :param search_params: LIST OF TUPLES (job title, location)
    :param num_pages: NUM
    :param source: STRING name of website to pull links from (either indeed or monster as of now)
    :return: Nothing, adds extracted information to mongodb
    """
    # Initialize logger
    su.setup_scrape_logger(name=cfg.scrape_log_name)
    logger = logging.getLogger(cfg.scrape_log_name)
    logger.info('Successfully initialized logger. Beginning scraping on ' + str(date.today()) + ' for ' + str(num_pages) + \
                ' pages each of ' + str(len(search_params)) + ' jobs from ' + source + '.')
    logger.info('Search Parameters:\n' + pprint.pformat(search_params))

    # Define helper functions
    def rotate_user_agent():
        # Moves the first user agent to the end of the list, so that the first item in the list can be used.
        user_agents.append(user_agents.pop(0))

    def rotate_ip():
        """
        Changes IP to the first server in the list and moves the first item in the list to the end
        :param server_list: List of IP Vanish servers
        :return: Updated server list
        """
        while os.system('echo %s|sudo -S %s' % (cfg.sudo_password, './src/scraping/change_ip.sh ' + server_list[0])) != 0:
            logger.exception('VPN address ' + server_list[0] + ' invalid. Trying next VPN address.')
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
        extract_description_from_link = indeed.extract_description_from_link
    else:  # monster
        build_url = monster.build_url
        build_url_page_n = monster.build_url_page_n
        extract_job_title_from_result = monster.extract_job_title_from_result
        extract_company_from_result = monster.extract_company_from_result
        extract_location_from_result = monster.extract_location_from_result
        extract_job_link_from_result = monster.extract_job_link_from_result
        extract_description_from_link = monster.extract_description_from_link

    # Build list of IPVanish Servers
    server_list = su.build_ipvanish_server_list(cs.ipvanish_base_links)
    random.shuffle(server_list)

    # Import list of User Agents
    user_agents = cs.user_agents
    random.shuffle(user_agents)

    for job_title, location in search_params:
        base_url = build_url(job_title=job_title, location=location)
        pages = list(range(num_pages))
        for i in range(num_pages):
            page = pages.pop(pages.index(random.choice(pages))) + 1  # Select pages in random order

            rotate_ip()  # Connect to VPN and change IP address
            rotate_user_agent()  # Cycle through user agents
            su.random_pause(min_pause=cfg.min_pause, max_pause=cfg.max_pause)  # Random wait time between requests

            # Log request
            logger.info('VPN: ' + server_list[-1] + '\tUser Agent: ' + user_agents[-1] + '\tJob: ' + job_title + '\tLocation: ' + location + '\tPage: ' + str(page))

            url = build_url_page_n(url=base_url, n=page)  # Build url
            soup = su.get_soup(url=url, user_agent=user_agents[0])  # Retrieve page from website

            jobs = extract_job_title_from_result(soup)
            companies = extract_company_from_result(soup)
            locations = extract_location_from_result(soup)
            links = extract_job_link_from_result(soup)
            descrs = []

            logger.info('Extracting job links from ' + str(len(links)) + ' jobs.')
            for j, link in enumerate(links):
                su.random_pause(min_pause=cfg.min_pause, max_pause=cfg.max_pause)
                rotate_user_agent()

                # Hits the website, so pause/user_agent change is necessary in between each
                # Can fail if website link has issues
                try:
                    descrs.append(extract_description_from_link(link=link, user_agent=user_agents[0], og_page_url=url))
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
                    # TODO: CHECK THIS ERROR
                    import pdb;
                    pdb.set_trace()
                    logger.exception('Error extracting job description link from ' + companies[j] + "'s posting for a " + jobs[j] + ' in ' + location[j] + \
                                     '. Link: ' + link)
                    descrs.append(cfg.job_description_link_fail_msg)

            logger.info('Scraping complete. Inserting data into MongoDB.')
            db.insert_data(jobs=jobs, companies=companies, locations=locations, descrs=descrs, source=source)