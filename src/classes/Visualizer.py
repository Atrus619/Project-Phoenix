import src.visualization.visualize as viz
from src.classes.Extractions import Extractions
from redis import Redis
import rq
from config import Config as cfg
import pickle as pkl


class Visualizer:
    def __init__(self):
        self.redis = Redis.from_url(cfg.REDIS_URL)
        self.task_queue = rq.Queue('extractor', connection=self.redis)

    def process_job_in_location(self, job, location):
        task = self.task_queue.enqueue('src.visualization.visualize.run_extractions',
                                       args=(job, location),
                                       job_timeout=-1)
        return task
