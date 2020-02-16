import src.visualization.visualize as viz
from src.classes.Extractions import Extractions


class Visualizer:
    def __init__(self):
        pass

    def process_job_in_location(self, job, location):
        extractions = Extractions(required_years_experience=5, required_degree=5, travel_percentage=5, salary=5)
        extractions.gather(job=job, location=location)
