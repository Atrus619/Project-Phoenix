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

    def console_interact(self):
        opening_msg = self.update_history_and_generate_opening_msg()
        print(opening_msg)

        while True:
            raw_text = self.seek_input_from_user()
            parsed_user_msg = self.interpreter.parse_user_msg(raw_text=raw_text)
            reply = self.update_history_and_generate_reply(parsed_user_msg=parsed_user_msg)
            print(reply)

            if self.exit_conversation():
                break

            # Check if additional information was sought and continue asking until all information determined for request
            if self.policy.is_seeking_additional_info():
                additional_info_conv_history = ConversationHistory()
                while self.policy.is_seeking_additional_info():
                    raw_text = self.seek_input_from_user()
                    reply = self.update_history_and_get_more_information(input_msg=raw_text, original_parsed_user_msg=parsed_user_msg,
                                                                         additional_info_conv_history=additional_info_conv_history)
                    print(reply)

    @staticmethod
    def seek_input_from_user():
        raw_text = input('>>> ')
        while not raw_text:
            print('Please enter in a non-empty value.')
            raw_text = input('>>> ')
        return raw_text

    def update_history_and_generate_opening_msg(self):
        opening_msg = self.policy.get_opening_msg()
        self.conversation_history.add_bot_msg(raw_text=opening_msg)
        return opening_msg

    def update_history_and_generate_reply(self, parsed_user_msg):
        # Update history
        self.conversation_history.add_parsed_user_msg(*parsed_user_msg)

        # Reply
        reply = self.policy.get_reply(conversation_history=self.conversation_history)
        self.conversation_history.add_bot_msg(raw_text=reply)
        return reply

    def exit_conversation(self):
        return self.conversation_history.user_msgs[-1].classified_intent == cfg.valid_intents['end_of_conversation']

    def update_history_and_get_more_information(self, input_msg, original_parsed_user_msg, additional_info_conv_history):
        # Parse and updated based on user input, same as earlier, but with the follow up variant of the routine
        parsed_user_follow_up_msg = self.interpreter.parse_user_msg_follow_up(raw_text=input_msg, missing_entity=self.policy.missing_entity)
        self.conversation_history.add_parsed_user_msg(*parsed_user_follow_up_msg)

        # Reply: Either way, reset missing entity. If additional information is sought after accepting, it will get set to missing again before re-entering while loop.
        missing_entity_letter = self.policy.missing_entity
        self.policy.reset_missing_entity()

        # If accept, then call routine to generate shallow conversation_history object with updated query to get the actual reply if everything is satisfied.
        # Otherwise, re-enter while loop at the top to append to this conversation_history object.
        if self.conversation_history.get_latest_msg().classified_intent == 'Acceptance':
            additional_info_conv_history.add_parsed_user_msg(*original_parsed_user_msg)

            additional_entity_text = self.conversation_history.get_latest_msg().recognized_entities[missing_entity_letter][-1]
            additional_info_conv_history.update_latest_recognized_entities(entity_letter=missing_entity_letter,
                                                                           additional_entity_text=additional_entity_text)
            reply = self.policy.get_reply(conversation_history=additional_info_conv_history)
        else:
            reply = self.policy.get_reply(conversation_history=self.conversation_history)

        # Whether accept or reject, need to update reply and continue. If additional information is needed, policy missing entity will get reset and while loop
        # will be re-entered.
        self.conversation_history.add_bot_msg(raw_text=reply)
        return reply

    def wipe_history(self):
        self.conversation_history = ConversationHistory()
