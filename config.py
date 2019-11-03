import os

basedir = os.path.abspath(os.path.dirname(__file__))

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(basedir, '.env'))

except ModuleNotFoundError:
    pass


class Config:
    db = 'phoenixdb'
    collection = 'post'

    # Secrets
    ipvanish_password = os.environ.get('ipvanish_password') or 'you-will-never-guess'
    sudo_password = os.environ.get('sudo_password') or 'good-luck'

    min_pause = 2  # in seconds
    max_pause = 10

    jobs = [
        "machine learning engineer",
        "product development",
        "data scientist",
        "strategy and operations",
        "AI scientist"
    ]
    cities = [
        "Chicago, IL",
        "New York, NY",
        "San Francisco, CA"
    ]


    num_pages = 2

    source = 'indeed'

    log_folder = 'logs'
    scrape_log_name = 'scrape_log'

    job_description_link_fail_msg = 'Job description unavailable'
