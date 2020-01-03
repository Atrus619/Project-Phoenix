from collections import namedtuple


class ConversationHistory:
    """
    Stores processed conversation history
    """
    def __init__(self):
        self.user_msgs, self.bot_msgs = [], []
        self.ParsedUserMsg = namedtuple('ParsedUserMsg', 'raw_text latent_vector recognized_entities classified_intent')

    def __len__(self):
        return len(self.user_msgs)

    def __str__(self):
        output_str = '--- Conversation History ---'
        output_str += '\nBot:\t' + self.bot_msgs[0]
        for i in range(len(self)):
            output_str += '\nUser:\t' + self.user_msgs[i].raw_text
            output_str += '\nBot:\t' + self.bot_msgs[i + 1]
        output_str += '\n--- End of Conversation ---'

    def add_parsed_user_msg(self, raw_text, latent_vector, recognized_entities, classified_intent):
        self.user_msgs.append(
            self.ParsedUserMsg(
                raw_text,
                latent_vector,
                recognized_entities,
                classified_intent
            )
        )

    def add_bot_msg(self, raw_text):
        self.bot_msgs.append(raw_text)

    def get_latest_msg(self):
        return self.user_msgs[-1]
