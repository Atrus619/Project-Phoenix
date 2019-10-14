import src.scraping.scraping_indeed as indeed
import src.scraping.scraping_monster as monster
import src.db as db
import src.scraping.utils as su
from src.config import Config as cfg
from src.constants import Constants as cs
import random

# https://www.scraperapi.com/blog/5-tips-for-web-scraping
# TODO: Set other request headers  https://httpbin.org/anything
# TODO: Set referer


def make_dataset(search_params, num_pages, source):
    """
    Accepts arguments describing the search parameters (probably a list of tuples of arguments for build_url)
    Pulls information from indeed, parses it into meaningful components, and inserts this information to mongodb
    :param search_params: LIST OF TUPLES (job title, location)
    :param num_pages: NUM
    :param source: STRING name of website to pull links from (either indeed or monster as of now)
    :return: Nothing, adds extracted information to mongodb
    """
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
            page = pages.pop(pages.index(random.choice(pages)))  # Select pages in random order

            server_list = su.change_ip(server_list)  # Connect to VPN and change IP address
            user_agents = su.change_user_agent(user_agents)  # Cycle through user agents
            su.random_pause(min_pause=cfg.min_pause, max_pause=cfg.max_pause)  # Random wait time between requests

            url = build_url_page_n(url=base_url, n=page)  # Build url
            soup = su.get_soup(url=url, user_agent=user_agents[0])  # Retrieve page from website

            jobs = extract_job_title_from_result(soup)
            companies = extract_company_from_result(soup)
            locations = extract_location_from_result(soup)
            links = extract_job_link_from_result(soup)
            descrs = []
            for link in links:
                su.random_pause(min_pause=cfg.min_pause, max_pause=cfg.max_pause)
                user_agents = su.change_user_agent(user_agents)
                descrs.append(extract_description_from_link(link=link, user_agent=user_agents[0]))  # Hits the website, so pause/user_agent change is necessary in between each

            db.insert_data(jobs=jobs, companies=companies, locations=locations, descrs=descrs, source=source)
