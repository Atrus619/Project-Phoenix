import pickle as pkl
from redis import Redis
import rq
from config import Config as cfg


class Visualizer:
    def __init__(self):
        self._redis = Redis.from_url(cfg.REDIS_URL)
        self._task_queue = rq.Queue('extractor', connection=self._redis)
        self.task = None

    def process_job_in_location(self, job, location):
        self.task = self._task_queue.enqueue('src.visualization.visualize.run_extractions',
                                             args=(job, location))
                                             # job_timeout=-1)

        return

    def get_reply(self):
        assert self.is_task_complete
        with open('app/static/imgs/description.pkl', 'rb') as f:
            reply = pkl.load(f)
        return reply

    def is_task_complete(self):
        if self.task:
            return self.task.is_finished
        else:
            return False

    def is_task_started(self):
        if self.task:
            return self.task.is_started
        else:
            return False

    def is_task_in_progress(self):
        return self.is_task_started() and not self.is_task_complete()
