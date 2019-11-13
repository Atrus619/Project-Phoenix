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
    ip = os.environ.get('ip') or '127.0.0.1'

    min_pause = 0.5  # in seconds
    max_pause = 2

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
        "San Francisco, CA",
        "Boston, MA",
        "Denver, CO",
        "San Diego, CA"
    ]

    sources = {
        'indeed': 5,
        'monster': 2
    }

    log_folder = 'logs'
    scrape_log_name = 'scrape_log'
    scrape_error_log_name = 'scrape_error_log'

    job_description_link_fail_msg = 'Job description unavailable'

    fail_wait_time = 60  # seconds
    max_retry_attempts = 3  # attempts
