import requests
from bs4 import BeautifulSoup
import re
from src.constants import Constants as cs
# TODO: add extract job title from result function!


def monster_str_converter(yarn):
    """
    Within a string converts spaces to - and commas to __2C according to URL logic for monster
    :param yarn: STRING
    :return: STRING
    """
    return yarn.replace(" ", "-").replace(",", "__2C")


def build_url(job_title, location=None):
    """
    Generates a link to search indeed for a specific job title/location.
    Could add additional args.
    :param job_title: STRING
    :param location: STRING (in the format of city, state; i.e. Boston, MA)
    :return: URL to use with requests.get
    """
    base_url = "http://www.monster.com/jobs/search/?q="
    job_url = monster_str_converter(job_title)
    end_url = "&intcid=skr_navigation_nhpso_searchMain"
    location_url = "&where=" + monster_str_converter(location) if location is not None else ""
    return base_url + job_url + location_url + end_url


def build_url_page_n(url, n):
    """
    Generates a link to subsequent n number of pages for a specific job title/location.
    :param url: STRING (comes from build_url)
    :param n: NUM (n is the number of additional pages; n=1 brings you to the second page)
    :return: URL to use with requests.get
    """
    if n > 0:
        return url + "&stpage=1&page=" + str(n)
    else:
        return url


def get_soup(url):
    """
    Helper function to construct a BeautifulSoup representation of a url.
    :param url: url returned from build_url
    :return: BeautifulSoup object parsed with html.parser
    """
    page = requests.get(url)
    return BeautifulSoup(page.text, "html.parser")


def extract_company_from_result(soup):
    companies = []
    for div in soup.find_all(name="div", attrs={"class": "company"}):
        company = div.find_all(name="span", attrs={"class": "name"})
        if len(company) > 0:
            for b in company:
                companies.append(b.text.strip())
        else:
            companies.append(None)
    return companies


def extract_location_from_result(soup):
    locations = []
    for div in soup.find_all(name="div", attrs={"class": "location"}):
        location = div.find_all(name="span", attrs={"class": "name"})
        if len(location) > 0:
            for b in location:
                locations.append(b.text.strip())
    return locations


def extract_job_link_from_result(soup):
    """
    pull links to job posting from job search return on indeed.com
    :param soup: beautiful soup object from a search on indeed.com
    :return: list of links (strings)
    """
    links = []
    for a in soup.find_all(name="h2", attrs={"class": "title"}):
        pattern = 'href=".*?"'
        temp_link = re.search(pattern, str(a))
        pattern2 = '".*?"'
        links.append(re.search(pattern2, temp_link.group()).group().replace('"', ""))
    return links


def extract_description_from_link(link, user_agent, og_page_url):
    """
    Retrieves the full job description from an indeed job posting link
    :param link: indeed job posting link (excludes the indeed.com part)
    :param user_agent: User Agent to be used with the request
    :param og_page_url: Original page URL from which we are grabbing this job link. Replaces google.com as referer.
    :return: text of full job description
    """
    headers = cs.base_request_headers
    headers['User-Agent'] = user_agent
    headers['Referer'] = og_page_url
    url = link  # lol.
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.text, "html.parser")
    raw_descr = soup.find_all(name='div', attrs={'id': 'JobDescription'})

    pattern = re.compile('(<.*?>)|(\\n)|[\[\]]|(\\r)')
    return re.sub(pattern, '', str(raw_descr))
