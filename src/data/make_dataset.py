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
    for search_param in search_params:
        pass
    pass
