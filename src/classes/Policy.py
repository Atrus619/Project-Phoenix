import time
import random
from config import Config as cfg


class Policy:
    """
    Defines how the chatbot responds to the user, calling appropriate external functionality as needed
    """
    def __init__(self, small_talk, delay_func=time.sleep(random.random() * 2 + 1)):
        self.small_talk = small_talk
        self.delay_func = delay_func

    def get_opening_msg(self):
        opening_msg = 'Hi! I am a chatbot designed to help you. My functionality is a bit limited at the moment, but I am able to handle several tasks currently.' \
                      'You are welcome to ask me about the job market for specific jobs in specific locations, or ask me about specific skills required ' \
                      'for specific jobs. If you are bored, I am also capable of handling a bit of small talk. Ask away!'

        self.delay_func()
        return opening_msg

    def get_final_msg(self):
        final_msg = "Thank you for your time and I hope I was helpful! If you have any questions about me, suggestions for improvement, or want to get involved," \
                    "don't hesitate to reach out to me at xxx@xxx.com. Have a great rest of your day!"

        self.delay_func()
        return final_msg

    def get_reply(self, conversation_history, intent):
        assert intent in cfg.valid_intents, "Received an invalid intent. This should not have occurred. Please check the interact object associated with this policy."

        if intent == cfg.valid_intents['small_talk']:
            reply = ''
        elif intent == cfg.valid_intents['end_of_conversation']:
            reply = ''
        elif intent == cfg.valid_intents['[job]_in_[location]']:
            reply = ''
        elif intent == cfg.valid_intents['[skills]_for_[job]']:
            reply = ''
        else:
            raise Exception("Received an invalid intent. This should not have occurred. Please check the interact object associated with this policy.")

        self.delay_func()
        return reply

    def get_reply_small_talk(self, conversation_history):
        pass

    def get_reply_job_in_location(self, conversation_history):
        pass

    def get_reply_skills_for_job(self, conversation_history):
        pass
