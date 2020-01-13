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

    def get_opening_msg(self):
        return 'Hi! I am a chatbot designed to help you analyze the job market. My functionality is a bit limited at the moment, but I am able to handle ' \
               'several tasks currently. You are welcome to ask me about the job market for specific jobs in specific locations, or ask me about specific ' \
               'skills required for specific jobs. If you are bored, I am also capable of handling a bit of small talk. Ask away!'

    def get_final_msg(self):
        return "Thank you for your time and I hope I was helpful! If you have any questions about me, suggestions for improvement, or want to get involved, " \
               "don't hesitate to reach out to me at xxx@xxx.com. Enjoy the rest of your day!"

    def get_reply_small_talk(self, conversation_history):
        return self.small_talk.get_reply(conversation_history=conversation_history,
                                         personality=self.small_talk_personality,
                                         **cfg.interact_config)

    def get_reply_job_in_location(self, recognized_entities):
        assert recognized_entities is not None
        assert len(recognized_entities["J"]) > 0 and len(recognized_entities["L"]) > 0

        reply = f'You are asking for information about a {recognized_entities["J"][0]} in {recognized_entities["L"][0]}.'

        return reply

    def get_reply_skills_for_job(self, recognized_entities):
        assert len(recognized_entities["J"]) > 0
        assert recognized_entities is not None

        reply = f'You are asking about skills for a {recognized_entities["J"][0]}.'

        return reply
