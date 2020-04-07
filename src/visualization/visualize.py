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
import os
from jinja2 import Environment, FileSystemLoader
import string


setup_extractions_logger(__name__)
logger = logging.getLogger(__name__)


def create_wordcloud(scraped_jobs, attribute, path, job, location):
    """
    Generates a wordcloud based on a list of
    :param scraped_jobs: ScrapedJobs object
    :param attribute: One of 'descr' or 'title'
    :param path: output path to save file to
    :param job: job requested by user
    :param location: location requested by user
    :return: None, but saves a wordcloud.png and a wordcloud.html to the static/user_id folder
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

    # Create and export wordcloud png
    WordCloud(stopwords=stopwords).generate(full_str).to_file(path)
    logger.info(f'Wordcloud successfully created at {path}')

    # Create and export wordcloud html
    env = Environment(loader=FileSystemLoader(cfg.templates_folder))
    template = env.get_template('wordcloud.html')
    processed_template = template.render(wordcloud_title=f'Wordcloud for {string.capwords(job)} in {string.capwords(location)}',
                                         wordcloud_path=os.path.basename(path))

    # Change file extension from .png to .html
    with open(os.path.splitext(path)[0] + '.html', 'w') as f:
        f.write(processed_template)
    logger.info('Wordcloud html template successfully created')

    return


def run_extractions(job, location, user_id):
    logger.info('Running extractions...')
    extractions = Extractions(required_years_experience=5, required_degree=5, travel_percentage=1, salary=5)
    extractions.gather(job, location)
    logger.info('Extractions complete.')

    output_dir = os.path.join(cfg.user_output_folder, user_id)
    os.makedirs(output_dir, exist_ok=True)

    threads = list()
    threads.append(threading.Thread(target=create_wordcloud, args=(extractions.scraped_jobs_parsed, 'descr', os.path.join(output_dir, 'wordcloud.png'), job, location)))
    threads.append(threading.Thread(target=build_heatmap, args=(extractions.scraped_jobs_parsed, location, os.path.join(output_dir, 'heatmap.html'))))
    threads.append(threading.Thread(target=describe_extractions, args=(extractions, os.path.join(output_dir, 'description.txt'))))
    # TODO: Add master table output

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return


def describe_extractions(extractions, path='app/static/imgs/description.txt'):
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
                  f'Travel Percentage (Note: only includes those that reported this): {avg_travel_percentage:.1f}%\n' \
                  f'Proportion that listed the following degrees as a requirement (a single posting can include multiple):\n' \
                  f'Bachelors degree (BS): {prop_bs*100:.1f}%\n' \
                  f'Masters degree (MS): {prop_ms*100:.1f}%\n' \
                  f'PhD: {prop_phd*100:.1f}%\n'

    with open(path, 'w') as f:
        f.write(description)

    logger.info(f'Extractions successfully created at {path}')
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
    logger.info(f'Heatmap successfully created at {output_path}')
    return
