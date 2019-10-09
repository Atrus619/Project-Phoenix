import requests
from bs4 import BeautifulSoup
import re


def monster_str_converter(yarn):
    """
    Within a string converts spaces to - and commas to __2C according to URL logic for monster
    :param yarn: STRING
    :return: STRING
    """
    return yarn.replace(" ", "-").replace(",", "__2C")


def build_monster_url(job_title, location=None):
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


def build_monster_url_page_n(url, n):
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


def extract_company_from_result_monster(soup):
    companies = []
    for div in soup.find_all(name="div", attrs={"class": "company"}):
        company = div.find_all(name="span", attrs={"class": "name"})
        if len(company) > 0:
            for b in company:
                companies.append(b.text.strip())
        else:
            companies.append(None)
    return companies
