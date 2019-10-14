import os
basedir = os.path.abspath(os.path.dirname(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(basedir, '.env'))

except ModuleNotFoundError:
    pass


class Config:
    db = "pheonixdb"
    collection = "scraped_data"

    # IP Vanish Parameters
    ipvanish_password = os.environ.get('ipvanish_password') or 'you-will-never-guess'
    ipvanish_base_links = [('iad-a01.ipvanish.com', 70),
                           ('jnb-c01.ipvanish.com', 7)]

    min_pause = 2  # in seconds
    max_pause = 10
