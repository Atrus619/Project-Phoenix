import requests
from bs4 import BeautifulSoup
import re

URL = "https://www.indeed.com/jobs?q=data+scientist+%2420%2C000&l=New+York&start=10"
page = requests.get(URL)
soup = BeautifulSoup(page.text, "html.parser")
print(soup.prettify())


def extract_job_title_from_result(soup):
    jobs = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        for a in div.find_all(name="a", attrs={"data-tn-element": "jobTitle"}):
            jobs.append(a["title"])
    return jobs


jobs = extract_job_title_from_result(soup)
print(jobs)


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


companies = extract_company_from_result(soup)
print(companies)


def extract_location_from_result(soup):
    locations = []
    spans = soup.findAll("span", attrs={'class': 'location'})
    for span in spans:
        locations.append(span.text)
    return locations


locations = extract_location_from_result(soup)
print(locations)


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


links = indeed_get_job_links(soup)
print(links)


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


descrs = []
for link in links:
    descrs.append(indeed_get_description(link))
