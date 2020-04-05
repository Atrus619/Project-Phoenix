import os
import pickle as pkl
from redis import Redis
import rq
from config import Config as cfg
import time
from src.classes.Enums import StateBase, IntentBase, IntentFollowUp, RecognizedEntities, EntityRequirements


class Visualizer:
    def __init__(self):
        self._redis = Redis.from_url(cfg.REDIS_URL)
        self._task_queue = rq.Queue('extractor', connection=self._redis)
        self.task = None
        self.user_id = None

    def process_job_in_location(self, job, location):
        self.task = self._task_queue.enqueue('src.visualization.visualize.run_extractions',
                                             args=(job, location, self.user_id))
        # job_timeout=-1)

        return

    def get_reply(self, intent):
        assert self.is_task_complete
        if intent == IntentBase.JOB_in_LOCATION:
            reply = f'I have finished processing the results for your inquiry.\n' \
                    f'You can view a heatmap of the discovered jobs and their locations {self.get_iframe_text(displayed_text="here", file_name="heatmap.html")}, ' \
                    f'you can download a wordcloud {self.get_iframe_text(displayed_text="here", file_name="wordcloud.png")}, ' \
                    f'or you can download a flat file containing detailed information from my discoveries {self.get_iframe_text(displayed_text="here", file_name="description.pkl")}.'
        else:
            raise NotImplementedError
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

    def get_iframe_text(self, displayed_text, file_name):
        return f'<a onclick=loadContent("./users/{os.path.join(self.user_id, file_name)}") href="javascript:void(0);">{displayed_text}</a>'
