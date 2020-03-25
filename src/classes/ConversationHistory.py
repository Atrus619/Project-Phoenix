from collections import namedtuple
from copy import deepcopy
from src.classes.Enums import StateBase, IntentBase, IntentFollowUp


class ConversationHistory:
    """
    Stores processed conversation history
    """
    def __init__(self):
        self.user_msgs, self.bot_msgs = [], []
        self.ParsedUserMsg = namedtuple('ParsedUserMsg', 'raw_text latent_vector recognized_entities classified_intent missing_entities state')

    def __len__(self):
        return len(self.user_msgs)

    def __str__(self):
        output_str = '--- Conversation History ---'
        try:
            output_str += '\nBot:\t' + self.bot_msgs[0]
            for i in range(len(self)):
                output_str += f'\nUser:\t {self.user_msgs[i].raw_text} (state - {self.user_msgs[i].state}, intent - {self.user_msgs[i].classified_intent}, ' \
                              f'entities - {self.user_msgs[i].recognized_entities}, missing entities - {self.user_msgs[i].missing_entities})'
                output_str += '\nBot:\t' + self.bot_msgs[i + 1]
        except IndexError:
            pass
        output_str += '\n--- End of Conversation ---'
        return output_str

    def add_bot_msg(self, raw_text):
        self.bot_msgs.append(raw_text)

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

        bot_msgs_copy = deepcopy(self.bot_msgs)
        user_msgs_copy = deepcopy(self.user_msgs)

        # Remove first message as it is irrelevant to conversation.
        if len(bot_msgs_copy) > 0:
            bot_msgs_copy.pop(0)

        # If len of bot_msgs == len of user_msgs, then the bot has already replied and we want to ignore this latest reply. Useful for debugging.
        if len(bot_msgs_copy) == len(self.user_msgs):
            bot_msgs_copy.pop(-1)

        try:
            while True:
                history.append(user_msgs_copy.pop(-1).raw_text)
                history.append(bot_msgs_copy.pop(-1))
        except IndexError:
            pass

        assert len(history) > 0
        history.reverse()
        history = history[-(2 * (n + 1)):]

        return history
