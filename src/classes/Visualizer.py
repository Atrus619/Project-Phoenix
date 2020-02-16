import src.visualization.visualize as viz
from src.classes.GatheredJPEs import GatheredJPEs


class Visualizer:
    def __init__(self):
        pass

    def process_job_in_location(self, job, location):
        gathered_jpes = GatheredJPEs(required_years_experience=5, required_degree=5, travel_percentage=5, salary=5)
        jpes = viz.gather_jpes(job=job, location=location, gathered_jpes=gathered_jpes)
