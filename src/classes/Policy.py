import time
from config import Config as cfg
import numpy as np
from src.classes.Visualizer import Visualizer
from src.classes.Enums import StateBase, IntentBase, IntentFollowUp, RecognizedEntities, EntityRequirements, EntityBase


class Policy:
    """
    Defines how the chatbot responds to the user, calling appropriate external functionality as needed
    """

    def __init__(self, small_talk, delay_func=lambda: time.sleep(np.clip(np.random.normal(2, 1), 0, 4)), small_talk_personality=None):
        self.small_talk = small_talk

        small_talk_personality = self.preprocess_small_talk_personality(small_talk_personality)  # Returns None if None
        self.small_talk_personality = self.small_talk.get_personality(personality=small_talk_personality)

        self.delay_func = delay_func
        self.missing_entity = None

        self.visualizer = Visualizer()

    def get_reply(self, conversation_history):
        """Returns a generator that can be iterated through to get multiple replies"""
        _, _, recognized_entities, intent, missing_entities, state = conversation_history.get_latest_msg()

        if state == StateBase.processing:
            return self.get_reply_processing()
        elif state == StateBase.selecting_results:
            raise NotImplementedError
        elif state == StateBase.seeking_additional_info:
            reply = self.get_reply_seeking_additional_info(intent, recognized_entities, missing_entities)
        elif isinstance(intent, IntentBase):
            reply = self.get_reply_base_intent(intent, conversation_history, recognized_entities)
        elif isinstance(intent, IntentFollowUp):
            reply = self.get_reply_follow_up_intent(intent, conversation_history)
        else:
            raise Exception("Received an invalid type of intent. This should not have occurred. Please check the interact object associated with this policy.")

        self.delay_func()
        return reply

    @staticmethod
    def get_reply_processing():
        yield 'I apologize, I am currently handling your previous request. Please ping me again after I am finished and present you with results.'
        return

    def get_reply_base_intent(self, intent, conversation_history, recognized_entities):
        """Returns a generator that can be iterated through to get multiple replies"""
        if intent == IntentBase.end_of_conversation:
            return self.get_final_msg()
        elif intent == IntentBase.small_talk:
            return self.get_reply_small_talk(conversation_history=conversation_history)
        elif intent == IntentBase.JOB_in_LOCATION:
            return self.get_reply_job_in_location(recognized_entities=recognized_entities)
        elif intent == IntentBase.skills_for_JOB:
            return self.get_reply_skills_for_job(recognized_entities=recognized_entities)
        else:
            raise Exception('Received an unexpected base intent.')

    def get_reply_follow_up_intent(self, intent, conversation_history):
        """Returns a generator that can be iterated through to get multiple replies"""
        if intent == IntentFollowUp.reject:
            return self.get_reply_follow_up_intent_rejected()
        elif intent == IntentFollowUp.accept:
            return self.get_reply_follow_up_intent_accepted(latest_intent=conversation_history.get_latest_base_intent(), conversation_history=conversation_history,
                                                            recognized_entities=conversation_history.get_latest_msg().recognized_entities)
        else:
            raise Exception('Received an unexpected follow up intent')

    @staticmethod
    def get_reply_seeking_additional_info(intent, recognized_entities, missing_entities):
        if intent == IntentBase.JOB_in_LOCATION:
            intent_descr = 'a specific job in a specific location'
        elif intent == IntentBase.skills_for_JOB:
            intent_descr = 'skill requirements for a specific job'
        else:
            raise Exception('Received an unexpected type of intent for seeking additional info.')

        yield 'I apologize. I am imperfect. I was unable to identify a required piece of information for me to help you out.'
        yield f'I believe you are trying to get information about {intent_descr}. '

        if not recognized_entities.is_empty():
            yield f'I was able to identify you are looking for the following: {recognized_entities}. '

        if missing_entities.size > 1:
            yield f'I need more than one more piece of information. I am going to attempt to get the remaining information from you one piece at a time. '

        missing_entity = missing_entities.get_missing_entity()
        yield f'Could you tell me specifically which {missing_entity} you are interested in getting information about?'
        return

    @staticmethod
    def get_opening_msg():
        yield 'Hi! I am a chatbot designed to help you analyze the job market. '
        yield 'My functionality is a bit limited at the moment, but I am able to handle ' \
              'several tasks currently. You are welcome to ask me about the job market for specific jobs in specific locations, or ask me about specific ' \
              'skills required for specific jobs.'
        yield 'If you are bored, I am also capable of handling a bit of small talk. Ask away!'
        return

    @staticmethod
    def get_final_msg():
        yield "Thank you for your time and I hope I was helpful! If you have any questions about me, suggestions for improvement, or want to get involved, " \
              "don't hesitate to reach out to me at xxx@xxx.com. Enjoy the rest of your day!"
        return

    def get_reply_small_talk(self, conversation_history):
        reply = self.small_talk.get_reply(conversation_history=conversation_history,
                                          personality=self.small_talk_personality,
                                          **cfg.interact_config)
        yield reply
        return

    def get_reply_job_in_location(self, recognized_entities):
        assert recognized_entities is not None
        job, location = recognized_entities[EntityBase.J][0], recognized_entities[EntityBase.L][0]  # Only return information about the first of each identified for now
        reply = f'You are asking for information about {job} jobs in {location}. ' \
                f'Please wait a moment while I collect the relevant information for you.'
        self.visualizer.process_job_in_location(job=job, location=location)
        yield reply
        return

    # TODO: Add processor function to this guy
    def get_reply_skills_for_job(self, recognized_entities):
        assert recognized_entities is not None
        job = recognized_entities[EntityBase.J][0]
        reply = f'You are asking about skills for a {job}.'
        yield reply
        return

    @staticmethod
    def get_reply_follow_up_intent_rejected():
        yield "I apologize that I wasn't able to help you out with that inquiry. Is there anything else I can help you with?"
        return

    def get_reply_follow_up_intent_accepted(self, latest_intent, conversation_history, recognized_entities):
        yield self.get_reply_base_intent(latest_intent, conversation_history, recognized_entities)
        return

    @staticmethod
    def preprocess_small_talk_personality(small_talk_personality):
        if small_talk_personality is None:
            return None
        small_talk_personality = small_talk_personality.split('. ')[:5]  # Only first five
        for i, sentence in enumerate(small_talk_personality):
            if sentence[-1] != '.':
                small_talk_personality[i] = sentence + '.'
        return small_talk_personality
