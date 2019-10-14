import os
basedir = os.path.abspath(os.path.dirname(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(basedir, '.env'))

except ModuleNotFoundError:
    pass


class Config:
    db = 'phoenixdb'
    collection = 'scraped_data'

    # IP Vanish Parameters
    ipvanish_password = os.environ.get('ipvanish_password') or 'you-will-never-guess'

    min_pause = 2  # in seconds
    max_pause = 10

    search_params = [
        ('Data Scientist', 'Chicago, IL'),
        ('Data Scientist', 'New York, NY'),
        ('Data Scientist', 'San Francisco, CA'),
        ('Actuary', 'Chicago, IL'),
        ('Actuary', 'New York, NY'),
        ('Actuary', 'San Francisco, CA')
    ]

    num_pages = 10

    source = 'indeed'
