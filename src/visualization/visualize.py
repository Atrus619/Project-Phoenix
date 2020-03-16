import numpy as np
from src.classes.ScrapedJobs import ScrapedJobs
from wordcloud import WordCloud, STOPWORDS
from config import Config as cfg
from src.classes.Extractions import Extractions
import src.scraping.scraping_google_maps as scraping_google_maps
import gmplot
import pandas as pd
from src.pipeline.utils import setup_extractions_logger
import logging
import threading
import pickle as pkl


setup_extractions_logger(__name__)
logger = logging.getLogger(__name__)


def create_wordcloud(scraped_jobs, attribute='descr', path='app/static/imgs/sample_wordcloud.png'):
    """
    Generates a wordcloud based on a list of
    :param scraped_jobs: ScrapedJobs object
    :param attribute: One of 'descr' or 'title'
    :param path: output path to save file to
    :return: wordcloud object
    """
    assert isinstance(scraped_jobs, ScrapedJobs)
    assert attribute in ('descr', 'job_title')

    logger.info('Creating wordcloud...')
    stopwords = set(STOPWORDS)
    if attribute == 'descr':
        full_str = ' '.join([scraped_job.parse() for scraped_job in scraped_jobs if scraped_job.parse() != cfg.job_description_parse_fail_msg])
        stopwords.update(['work', 'will', 'need', 'including', 'required'])
    else:  # job_title
        full_str = ' '.join([scraped_job.job_title for scraped_job in scraped_jobs])
        stopwords.update([])

    WordCloud(stopwords=stopwords).generate(full_str).to_file(path)
    logger.info(f'Wordcloud successfully created at {path}.')
    return


def run_extractions(job, location):
    logger.info('Running extractions...')
    extractions = Extractions(required_years_experience=5, required_degree=5, travel_percentage=1, salary=5)
    extractions.gather(job, location)
    logger.info('Extractions complete.')

    # TODO: Add specific locations rather than one generic default location
    threads = []
    threads.append(threading.Thread(target=create_wordcloud, args=(extractions.scraped_jobs_parsed, 'descr', 'app/static/imgs/sample_wordcloud.png')))
    threads.append(threading.Thread(target=build_heatmap, args=(extractions.scraped_jobs_parsed, location, 'app/static/imgs/heatmap.html')))
    threads.append(threading.Thread(target=describe_extractions, args=(extractions, 'app/static/imgs/description.pkl')))
    # TODO: Add master table output

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return


def describe_extractions(extractions, path='app/static/imgs/description.pkl'):
    """Returns a string describing the extractions for output to chatbot"""
    logger.info('Describing extractions...')

    total_scraped_jobs = extractions.__len__(False)
    total_parsed_jobs = len(extractions)
    avg_years_exp = np.mean(extractions.extracted_required_years_experience)
    avg_travel_percentage = np.mean(extractions.extracted_travel_percentages)
    avg_salary = np.mean([x[0] for x in extractions.extracted_salaries])
    prop_bs = len([degree_list for degree_list in extractions.extracted_required_degrees if 'BS' in degree_list]) / total_parsed_jobs
    prop_ms = len([degree_list for degree_list in extractions.extracted_required_degrees if 'MS' in degree_list]) / total_parsed_jobs
    prop_phd = len([degree_list for degree_list in extractions.extracted_required_degrees if 'PhD' in degree_list]) / total_parsed_jobs
    # TODO: Could do interactions as well

    description = f'I scraped a total of {total_scraped_jobs} job postings from indeed.\n' \
                  f'Of these, I was able to successfully parse {total_parsed_jobs} of them.\n' \
                  f'Of the job postings where I was able to find each relevant piece of information, I found the following:\n' \
                  f'Years of experience: {avg_years_exp:.1f}\n' \
                  f'Salary: ${avg_salary:,.0f} /  year\n' \
                  f'Travel Percentage (Note: Only includes those that reported this): {avg_travel_percentage:.1f}%\n' \
                  f'Proportion that listed the following degrees as a requirement (a single posting can include multiple):\n' \
                  f'Bachelors degree (BS): {prop_bs*100:.1f}%\n' \
                  f'Masters degree (MS): {prop_ms*100:.1f}%\n' \
                  f'PhD: {prop_phd*100:.1f}%'

    with open(path, 'wb') as f:
        pkl.dump(description, f)

    logger.info(f'Extractions successfully created at {path}.')
    return


def build_heatmap(scraped_jobs, originally_searched_location, output_path='app/static/imgs/heatmap.html'):
    logger.info('Building heatmap...')
    assert isinstance(scraped_jobs, ScrapedJobs)

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
    logger.info(f'Heatmap successfully created at {output_path}.')
    return
