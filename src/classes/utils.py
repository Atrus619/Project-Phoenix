import os
from config import Config as cfg
import logging


def init_recognized_entities_dict():
    output_dict = dict()
    for entity in cfg.entities:
        output_dict[entity] = []
    return output_dict
