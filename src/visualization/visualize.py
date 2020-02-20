import numpy as np
from src.classes.ScrapedJobs import ScrapedJobs
from wordcloud import WordCloud, STOPWORDS
from config import Config as cfg
from src.classes.Extractions import Extractions
import src.scraping.scraping_google_maps as scraping_google_maps
import gmplot
import pandas as pd


def create_wordcloud(scraped_jobs, type='descr', path='app/static/imgs/sample_wordcloud.png'):
    """
    Generates a wordcloud based on a list of
    :param scraped_jobs: ScrapedJobs object
    :param type: One of 'descr' or 'title'
    :param path: output path to save file to
    :return: wordcloud object
    """
    assert isinstance(scraped_jobs, ScrapedJobs)
    assert type in ('descr', 'job_title')

    stopwords = set(STOPWORDS)
    if type == 'descr':
        full_str = ' '.join([scraped_job.parse() for scraped_job in scraped_jobs if scraped_job.parse() != cfg.job_description_parse_fail_msg])
        stopwords.update(['work', 'will', 'need', 'including', 'required'])
    else:  # job_title
        full_str = ' '.join([scraped_job.job_title for scraped_job in scraped_jobs])
        stopwords.update([])

    WordCloud(stopwords=stopwords).generate(full_str).to_file(path)


def run_extractions(job, location):
    extractions = Extractions(required_years_experience=5, required_degree=5, travel_percentage=5, salary=5)
    extractions.gather(job, location)
    return extractions


def build_heatmap(originally_searched_location, scraped_jobs, output_path=None):
    if not output_path:
        output_path = 'my_heatmap.html'

    company_locations = [(job_posting.company, job_posting.location) for job_posting in scraped_jobs]
    company_location_counts = pd.DataFrame(company_locations, columns=['companies', 'locations']).reset_index().groupby(['companies', 'locations']).count()

    latitudes, longitudes = [], []
    for (company, location), count in company_location_counts.iterrows():
        latitude, longitude = scraping_google_maps.get_latitude_and_longitude(location=location, company=company)
        for i in range(count['index']):
            latitudes.append(latitude)
            longitudes.append(longitude)

    gmap_lat, gmap_long = scraping_google_maps.get_latitude_and_longitude(location=originally_searched_location)
    gmap = gmplot.GoogleMapPlotter(center_lat=gmap_lat, center_lng=gmap_long, zoom=13, apikey=cfg.GCP_API_KEY)
    gmap.heatmap(latitudes, longitudes, radius=50)
    gmap.draw(output_path)

    return
