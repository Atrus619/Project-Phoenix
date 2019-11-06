from src.scraping.scrape_dataset import scrape_dataset
from config import Config as cfg
import src.scraping.utils as su

for source in cfg.sources:
    scrape_dataset(search_params=su.get_search_params(config=cfg), num_pages=cfg.sources[source], source=source)
