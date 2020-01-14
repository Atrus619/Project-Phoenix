import time
from config import Config as cfg
import numpy as np


class Policy:
    """
    Defines how the chatbot responds to the user, calling appropriate external functionality as needed
    """

    def __init__(self, small_talk, delay_func=lambda: time.sleep(np.clip(np.random.normal(2, 1), 0, 4)), small_talk_personality=None):
        self.small_talk = small_talk
        self.small_talk_personality = self.small_talk.get_personality(personality=small_talk_personality)

        self.delay_func = delay_func
        self.seeking_additional_info = False

    def get_reply(self, conversation_history):
        intent = conversation_history.user_msgs[-1].classified_intent
        recognized_entities = conversation_history.user_msgs[-1].recognized_entities

        assert intent in cfg.valid_intents.values(), "Received an invalid intent. This should not have occurred. Please check the interact object associated with this policy."

        if intent == cfg.valid_intents['end_of_conversation']:
            reply = self.get_final_msg()
        elif intent == cfg.valid_intents['small_talk']:
            reply = self.get_reply_small_talk(conversation_history=conversation_history)
        elif intent == cfg.valid_intents['[job]_in_[location]']:
            reply = self.get_reply_job_in_location(recognized_entities=recognized_entities)
        elif intent == cfg.valid_intents['skills_for_[job]']:
            reply = self.get_reply_skills_for_job(recognized_entities=recognized_entities)
        else:
            raise Exception("Received an invalid intent. This should not have occurred. Please check the interact object associated with this policy.")

        self.delay_func()
        return reply

    @staticmethod
    def get_opening_msg():
        return 'Hi! I am a chatbot designed to help you analyze the job market. My functionality is a bit limited at the moment, but I am able to handle ' \
               'several tasks currently. You are welcome to ask me about the job market for specific jobs in specific locations, or ask me about specific ' \
               'skills required for specific jobs. If you are bored, I am also capable of handling a bit of small talk. Ask away!'

    @staticmethod
    def get_final_msg():
        return "Thank you for your time and I hope I was helpful! If you have any questions about me, suggestions for improvement, or want to get involved, " \
               "don't hesitate to reach out to me at xxx@xxx.com. Enjoy the rest of your day!"

    def get_reply_small_talk(self, conversation_history):
        return self.small_talk.get_reply(conversation_history=conversation_history,
                                         personality=self.small_talk_personality,
                                         **cfg.interact_config)

    def get_reply_job_in_location(self, recognized_entities):
        assert recognized_entities is not None
        if ("J" in recognized_entities) and ("L" in recognized_entities):
            reply = f'You are asking for information about a {recognized_entities["J"][0]} in {recognized_entities["L"][0]}.'
            return reply

        intent_descr = 'a specific job in a specific location'
        if "J" not in recognized_entities:  # Simplistic approach for now, only handle one missing entity at a time.
            return self.seek_additional_info(intent_descr=intent_descr, missing_entity="J", recognized_entities=recognized_entities)
        else:
            return self.seek_additional_info(intent_descr=intent_descr, missing_entity="L", recognized_entities=recognized_entities)

    def get_reply_skills_for_job(self, recognized_entities):
        assert recognized_entities is not None
        if "J" in recognized_entities:
            reply = f'You are asking about skills for a {recognized_entities["J"][0]}.'
            return reply
        else:
            return self.seek_additional_info(intent_descr='skill requirements for a specific job', missing_entity="J", recognized_entities=recognized_entities)

    def seek_additional_info(self, intent_descr: str, missing_entity: str, recognized_entities: dict):
        assert missing_entity in cfg.entities, 'Invalid entity passed to seek_additional_info method'
        missing_entities_str = cfg.entities[missing_entity]

        recognized_entities_list = []
        for recognized_entity_code, recognized_entity in recognized_entities.items():
            recognized_entities_list.append(cfg.entities[recognized_entity_code] + ': ' + recognized_entity)
        recognized_entities_str = ', '.join(recognized_entities_list)

        reply = f'I apologize, I am not a perfect chatbot. I identified you are trying to get information about {intent_descr}.'

        if len(recognized_entities_list) > 0:
            reply += f' I identified that you are looking for the following:\n{recognized_entities_str}.'

        reply += f'\nIt appears I am unable to identify a crucial missing piece of information. Could you tell me specifically which {missing_entities_str} you are interested' \
                 f'in getting information about? If I have grossly misunderstood your request, I apologize again. Feel free to let me know if this is the case and we can try ' \
                 f'again from the beginning.'

        self.currently_seeking_additional_info()
        return reply

    def currently_seeking_additional_info(self):
        self.seeking_additional_info = True

    def not_seeking_additional_info(self):
        self.seeking_additional_info = False

    def is_seeking_additional_info(self):
        return self.seeking_additional_info
