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
            raw_text = self.seek_input_from_user()

            # Parse and update based on user input
            parsed_user_msg = self.interpreter.parse_user_msg(raw_text=raw_text)
            self.conversation_history.add_parsed_user_msg(*parsed_user_msg)

            # Reply
            reply = self.policy.get_reply(conversation_history=self.conversation_history)
            self.conversation_history.add_bot_msg(raw_text=reply)
            print(reply)

            # Check if exit condition reached and break
            if self.conversation_history.user_msgs[-1].classified_intent == cfg.valid_intents['end_of_conversation']:
                break

            # Check if additional information was sought and continue asking until all information determined for request
            if self.policy.is_seeking_additional_info():
                additional_info_conv_history = ConversationHistory()

            while self.policy.is_seeking_additional_info():
                raw_text = self.seek_input_from_user()

                # Parse and updated based on user input, same as earlier, but with the follow up variant of the routine
                parsed_user_follow_up_msg = self.interpreter.parse_user_msg_follow_up(raw_text=raw_text, missing_entity=self.policy.missing_entity)
                self.conversation_history.add_parsed_user_msg(*parsed_user_follow_up_msg)

                # Reply: Either way, reset missing entity. If additional information is sought after accepting, it will get set to missing again before re-entering
                # while loop.
                missing_entity_letter = self.policy.missing_entity
                self.policy.reset_missing_entity()

                # If accept, then call routine to generate shallow conversation_history object with updated query to get the actual reply if everything is satisfied.
                # Otherwise, re-enter while loop at the top to append to this conversation_history object.
                if self.conversation_history.get_latest_msg().classified_intent == 'Acceptance':
                    additional_info_conv_history.add_parsed_user_msg(*parsed_user_msg)

                    additional_entity_text = self.conversation_history.get_latest_msg().recognized_entities[missing_entity_letter][-1]
                    additional_info_conv_history.update_latest_recognized_entities(entity_letter=missing_entity_letter,
                                                                                   additional_entity_text=additional_entity_text)
                    reply = self.policy.get_reply(conversation_history=additional_info_conv_history)
                else:
                    reply = self.policy.get_reply(conversation_history=self.conversation_history)

                # Whether accept or reject, need to update reply and continue. If additional information is needed, policy missing entity will get reset and while loop
                # will be re-entered.
                self.conversation_history.add_bot_msg(raw_text=reply)
                print(reply)

    @staticmethod
    def seek_input_from_user():
        raw_text = input('>>> ')
        while not raw_text:
            print('Please enter in a non-empty value.')
            raw_text = input('>>> ')
        return raw_text
