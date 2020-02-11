import requests
from bs4 import BeautifulSoup
import re
from src.constants import Constants as cs
import src.scraping.utils as su


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


def extract_description_from_link(link, user_agent, og_page_url):
    """
    Retrieves the full job description from an indeed job posting link
    :param link: indeed job posting link (excludes the indeed.com part)
    :param user_agent: User Agent to be used with the request
    :param og_page_url: Original page URL from which we are grabbing this job link. Replaces google.com as referer.
    :return: text of full job description
    """
    # TODO: There are cases when our original strategy does not work. Come back here for parsing the text once we have it stored in mongodb
    headers = cs.base_request_headers
    headers['User-Agent'] = user_agent
    headers['Referer'] = og_page_url
    url = "https://www.indeed.com" + link
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.text, "html.parser")
    raw_descr = soup.find_all(name='div', attrs={'id': 'jobDescriptionText'})
    pattern = re.compile('(<.*?>)|(\\n)|[\[\]]')

    if len(raw_descr) > 0:
        return re.sub(pattern, '', str(raw_descr))
    else:
        return 'SECOND TRY, RETURNING RAW HTML: ' + page.text


def extract_description_html_from_link(session, link, user_agent, og_page_url):
    """
    Retrieves full html from an indeed job posting link. Extracting the job description specifically varies too much from company website to website.
    :param session: requests.session object
    :param link: indeed job posting link (excludes the indeed.com part)
    :param user_agent: User Agent to be used with the request
    :param og_page_url: Original page URL from which we are grabbing this job link. Replaces google.com as referer.
    :return: html of entire page
    """
    headers = cs.base_request_headers
    headers['User-Agent'] = user_agent
    headers['Referer'] = og_page_url
    url = "https://www.indeed.com" + link
    page = su.custom_get(session=session, url=url, headers=headers)
    return page.text
