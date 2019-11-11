from mongoengine import *
from config import Config as cfg
import datetime
import pymongo
import pandas as pd


class Post(Document):
    job_title = StringField(required=False)
    company = StringField(required=False)
    location = StringField(required=False)
    description = StringField(required=True)
    source = StringField(required=True)
    date_accessed = DateTimeField(default=datetime.datetime.now)


class Error(Document):
    # Can either be an error pulling an entire page, or a specific job posting within a page
    job_title = StringField(required=True)
    page = StringField(required=False)
    company = StringField(required=False)
    location = StringField(required=False)
    error = StringField(required=True)
    source = StringField(required=True)
    date_accessed = DateTimeField(default=datetime.datetime.now)


def insert_data(jobs, companies, locations, descrs, source):
    # Inserts data from a job posting into mongodb
    num_posts = len(jobs)
    assert all(len(arg) == num_posts for arg in [jobs, companies, locations, descrs]), 'all inputs same length'
    connect(cfg.db)
    for i in range(num_posts):
        Post(
            job_title=jobs[i],
            company=companies[i],
            location=locations[i],
            description=descrs[i],
            source=source
            # Date is automatically inserted based on current timestamp
        ).save()


def get_data(query={}, no_id=True):
    """ Read from Mongo and Store into DataFrame """
    # Connect to MongoDB
    client = pymongo.MongoClient(cfg.ip)
    db = client[cfg.db]

    # Make a query to the specific DB and Collection
    cursor = db[cfg.collection].find(query)

    # Expand the cursor and construct the DataFrame
    df = pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df


def insert_error(job_title, error, source, location=None, page=None, company=None):
    # Inserts information about an error into mongodb
    connect(cfg.db)
    Error(
        job_title=job_title,
        company=company,
        page=str(page),
        location=location,
        error=str(error),
        source=source
    ).save()
