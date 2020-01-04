from src.classes.ConversationHistory import ConversationHistory
from config import Config as cfg


class ChatBot:
    """
    Master Class that encapsulates other classes defined in this directory (Interpreter, Policy, ConversationHistory)
    """
    def __init__(self, interpreter, policy):
        self.interpreter = interpreter
        self.policy = policy
        self.conversation_history = ConversationHistory()

    def interact(self):
        opening_msg = self.policy.get_opening_msg()
        self.conversation_history.add_bot_msg(raw_text=opening_msg)
        print(opening_msg)

        while True:
            raw_text = input('>>> ')
            while not raw_text:
                print('Please enter in a non-empty value.')
                raw_text = input('>>> ')

            # Parse and update based on user input
            self.conversation_history.add_parsed_user_msg(*self.interpreter.parse_user_msg(raw_text=raw_text))

            # Reply
            reply = self.policy.get_reply(conversation_history=self.conversation_history)
            self.conversation_history.add_bot_msg(raw_text=reply)
            print(reply)

            # Check if exit condition reached and break
            if self.conversation_history.user_msgs[-1].classified_intent == cfg.valid_intents['end_of_conversation']:
                break
