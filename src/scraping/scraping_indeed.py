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
    if n > 0:
        return url + "&start=" + str(n*10)
    else:
        return url


def extract_job_title_from_result(soup):
    jobs = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        temp_jobs = []
        for a in div.find_all(name="a", attrs={"data-tn-element": "jobTitle"}):
            temp_jobs.append(a["title"])
        if len(temp_jobs) == 0:
            jobs.append(None)
        else:
            jobs.append("|".join(temp_jobs))
    return jobs


def extract_company_from_result(soup):
    companies = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        company = div.find_all(name="span", attrs={"class": "company"})
        temp_company = []
        if len(company) > 0:
            for b in company:
                temp_company.append(b.text.strip())
        else:
            sec_try = div.find_all(name="span", attrs={"class": "result - link - source"})
            for span in sec_try:
                temp_company.append(span.text.strip())
        if len(temp_company) == 0:
            companies.append(None)
        else:
            companies.append("|".join(temp_company))
    return companies


def extract_location_from_result(soup):
    locations = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        location = div.find_all(name="span", attrs={"class": "location"})
        if len(location) > 0:
            for b in location:
                locations.append(b.text.strip())
        else:
            sec_try = div.find_all(name="div", attrs={"class": "location"})
            for c in sec_try:
                locations.append(c.text.strip())
    return locations


def extract_job_link_from_result(soup):
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


def extract_description_from_link(link):
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