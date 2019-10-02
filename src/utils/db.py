from pymongo import MongoClient
from mongoengine import *
from src.config import Config as cfg
import datetime


class Post(Document):
    job_title = StringField(required=True)
    company = StringField(required=True)
    location = StringField(required=False)
    description = StringField(required=True)
    date_accessed = DateTimeField(default=datetime.datetime.now)

# client = MongoClient()

def insert_data(jobs, companies, locations, descrs):
    num_posts = len(jobs)
    assert all(len(arg) == num_posts for arg in [jobs, companies, locations, descrs]), "all inputs same length"
    connect(cfg.db)
    for i in range(num_posts):
        Post(
            job_title = jobs[i],
            company = companies[i],
            location = locations[i],
            description = descrs[i]
        ).save()
