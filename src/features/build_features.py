from src.constants import Constants as cs
import re
import os
import pandas as pd


def update_skills():
    """ Appends all .csv files in skills directory to the constants skills dict and returns said dict """
    files = os.listdir("skills")
    final_skill_dict = cs.skills
    for file in files:
        new_skills_df = pd.read_csv(os.path.join("skills", file), index_col=False)
        for i, row in new_skills_df.iterrows():
            final_skill_dict[row[0]] = row[1]
    return final_skill_dict


def get_skills(descr):
    matched_skills = []
    for skill in cs.skills:
        if re.search(re.compile(skill.upper()), descr.upper()):
            matched_skills.append(cs.skills[skill])
    return matched_skills


def find_duplicates():
    # find duplicates in the data
    raise NotImplementedError


