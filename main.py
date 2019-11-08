import src.db as db
import src.features.build_features as bf
from src.constants import Constants as cs

data = db.get_data()
descr = data.description[0]
print(bf.get_skills(descr))