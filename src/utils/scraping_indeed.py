import requests
from bs4 import BeautifulSoup
import re


def indeed_str_converter(yarn):
    """
    Within a string converts spaces to + and commas to %2C according to URL logic for indeed
    :param yarn: STRING
    :return: STRING
    """
    return yarn.replace(" ", "+").replace(",", "%2C")


def build_url(job_title, location=None):
    """
    Generates a link to search indeed for a specific job title/location.
    Could add additional args.
    :param job_title: STRING
    :param location: STRING (in the format of city, state; i.e. Boston, MA)
    :return: URL to use with requests.get
    """
    base_url = "http://indeed.com/jobs?q="
    job_url = indeed_str_converter(job_title)
    location_url = "&l=" + indeed_str_converter(location) if location is not None else ""
    return base_url + job_url + location_url


def build_url_page_n(url, n):
    """
    Generates a link to subsequent n number of pages for a specific job title/location.
    :param url: STRING (comes from build_url)
    :param n: NUM (n is the number of additional pages; n=1 brings you to the second page)
    :return: URL to use with requests.get
    """
    additional_url = "&start=" + str(n*10)
    return url + additional_url


def get_soup(url):
    """
    Helper function to construct a BeautifulSoup representation of a url.
    :param url: url returned from build_url
    :return: BeautifulSoup object parsed with html.parser
    """
    page = requests.get(url)
    return BeautifulSoup(page.text, "html.parser")


def extract_job_title_from_result(soup):
    jobs = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        for a in div.find_all(name="a", attrs={"data-tn-element": "jobTitle"}):
            jobs.append(a["title"])
    return jobs


def extract_company_from_result(soup):
    companies = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        company = div.find_all(name="span", attrs={"class": "company"})
        if len(company) > 0:
            for b in company:
                companies.append(b.text.strip())
        else:
            sec_try = div.find_all(name="span", attrs={"class": "result - link - source"})
            for span in sec_try:
                companies.append(span.text.strip())
    return companies


def extract_location_from_result(soup):
    locations = []
    spans = soup.findAll("span", attrs={'class': 'location'})
    for span in spans:
        locations.append(span.text)
    return locations


def indeed_get_job_links(soup):
    """
    pull links to job posting from job search return on indeed.com
    :param soup: beautiful soup object from a search on indeed.com
    :return: list of links (strings)
    """
    links = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        for a in div.find_all(name="a", attrs={"data-tn-element": "jobTitle"}):
            links.append(a["href"])
    return links


def indeed_get_description(link):
    """
    Retrieves the full job description from an indeed job posting link
    :param link: indeed job posting link (excludes the indeed.com part)
    :return: text of full job description
    """
    url = "https://www.indeed.com" + link
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    raw_descr = soup.find_all(name='div', attrs={'id': 'jobDescriptionText'})

    pattern = re.compile('(<.*?>)|(\\n)|[\[\]]')
    return re.sub(pattern, '', str(raw_descr))
