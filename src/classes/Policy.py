class Policy:
    """
    Defines how the chatbot responds to the user, calling appropriate external functionality as needed
    """
    def __init__(self, small_talk):
        self.small_talk = small_talk

    def get_opening_msg(self):
        opening_msg = ''

        return opening_msg

    def get_final_msg(self):
        final_msg = ''

        return final_msg

    def get_reply(self, conversation_history):
        reply = ''

        return reply
