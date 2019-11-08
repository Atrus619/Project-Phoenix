from src.constants import Constants as cs
import re

# TODO: write a function that extracts job descriptions

# TODO: revisit to make more complicated
def get_skills(descr):
    matched_skills = []
    for skill in cs.skills:
        if re.search(re.compile(skill.upper()), descr.upper()):
            matched_skills.append(cs.skills[skill])
    return matched_skills


def find_duplicates():
    # find duplicates in the data
    raise NotImplementedError


