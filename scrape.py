from src.scraping.scrape_dataset import scrape_dataset
from config import Config as cfg
import os
import src.scraping.utils as su

# Start MongoDB in background
os.system('echo %s|sudo -S %s' % (cfg.sudo_password, 'mongod --fork --logpath /var/log/mongodb.log'))

# Scrape Indeed
scrape_dataset(search_params=su.get_search_params(config=cfg), num_pages=cfg.num_pages, source='indeed')

# TODO: Scrape Monster

# TODO: Scrape Others??
