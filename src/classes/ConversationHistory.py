from collections import namedtuple
from copy import deepcopy
from src.classes.Enums import StateBase, IntentBase, IntentFollowUp
from enum import Enum, auto


class ConversationHistory:
    """
    Stores processed conversation history
    """
    class Source(Enum):
        user = auto()
        bot = auto()

    def __init__(self):
        self.user_msgs, self.bot_msgs, self.msg_order = [], [], []
        self.ParsedUserMsg = namedtuple('ParsedUserMsg', 'raw_text latent_vector recognized_entities classified_intent missing_entities state')

    def __len__(self):
        return len(self.msg_order)

    def __str__(self):
        user_msgs = deepcopy(self.user_msgs)
        bot_msgs = deepcopy(self.bot_msgs)

        output_str = '--- Conversation History ---'
        for i in range(len(self)):
            if self.msg_order[i] == ConversationHistory.Source.bot:
                msg = bot_msgs.pop(0)
                output_str += '\nBot:\t' + msg
            else:  # User message
                msg = user_msgs.pop(0)
                output_str += f'\nUser:\t {msg.raw_text} (state - {msg.state}, intent - {msg.classified_intent}, ' \
                              f'entities - {msg.recognized_entities}, missing entities - {msg.missing_entities})'
        output_str += '\n--- End of Conversation ---'
        return output_str

    def add_bot_msg(self, raw_text):
        self.msg_order.append(ConversationHistory.Source.bot)
        self.bot_msgs.append(raw_text)

    def add_parsed_user_msg(self, raw_text, latent_vector, recognized_entities, classified_intent, missing_entities, state):
        self.msg_order.append(ConversationHistory.Source.user)
        self.user_msgs.append(
            self.ParsedUserMsg(
                raw_text,  # str, directly from user
                latent_vector,  # numpy array, see interpreter.BaaS.encode()
                recognized_entities,  # RecognizedEntities from Enums.py
                classified_intent,  # enum from Enums.py
                missing_entities,  # EntityRequirements from Enums.py
                state  # state
            )
        )

    def get_latest_msg(self):
        return self.user_msgs[-1]

    def get_latest_base_intent(self):
        all_intents = [user_msg.classified_intent for user_msg in self.user_msgs]
        for intent in reversed(all_intents):
            if isinstance(intent, IntentBase):
                return intent

    def update_latest_recognized_entities(self, entity_letter, additional_entity_text):
        self.user_msgs[-1].recognized_entities[entity_letter].append(additional_entity_text)

    def get_list_of_conversation_latest_n_exchanges(self, n):
        history = []

        bot_msgs = deepcopy(self.bot_msgs)
        user_msgs = deepcopy(self.user_msgs)
        msg_order = deepcopy(self.msg_order)

        # Remove first message as it is irrelevant to conversation.
        if len(bot_msgs) > 0:
            bot_msgs.pop(0)
            msg_order.pop(0)

        for i in range(n):
            try:
                msg_type = msg_order.pop(-1)
                if msg_type == ConversationHistory.Source.bot:
                    history.append(bot_msgs.pop(-1))
                else:  # User msg
                    history.append(user_msgs.pop(-1).raw_text)
            except IndexError:
                pass

        history.reverse()

        return history
