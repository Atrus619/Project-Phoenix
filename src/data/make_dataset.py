from src.utils.scraping_indeed import *


def make_dataset(search_params, num_pages):
    """
    Accepts arguments describing the search parameters (probably a list of tuples of arguments for build_url)
    Pulls information from indeed, parses it into meaningful components, and inserts this information to mongodb
    :param search_params: LIST OF TUPLES (job title, location)
    :param num_pages: NUM
    :return: Nothing, adds extracted information to mongodb
    """
    # TODO: finish this function
    for job_title, location in search_params:
        base_url = build_url(job_title=job_title, location=location)
        for i in range(num_pages):
            url = build_url_page_n(url=base_url, n=i)



    pass
