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

    min_pause = 2  # in seconds
    max_pause = 10
