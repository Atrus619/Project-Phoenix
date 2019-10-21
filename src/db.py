from mongoengine import *
from config import Config as cfg
import datetime
import pymongo
import pandas as pd


class Post(Document):
    job_title = StringField(required=True)
    company = StringField(required=True)
    location = StringField(required=False)
    description = StringField(required=True)
    source = StringField(required=True)
    date_accessed = DateTimeField(default=datetime.datetime.now)


def insert_data(jobs, companies, locations, descrs, source):
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
        ).save()


def get_data():
    # Returns a pandas dataframe of the entire posts data set
    client = pymongo.MongoClient()
    db = client[cfg.db]
    posts = db[cfg.collection]
    df = pd.DataFrame(columns=posts.find_one().keys())
    for i, post in enumerate(posts.find()):
        df = df.append(pd.DataFrame(post, index=[i]))
    return df
