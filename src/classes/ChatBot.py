from src.classes.ConversationHistory import ConversationHistory
from config import Config as cfg
from src.classes.Enums import StateBase, IntentBase, IntentFollowUp, RecognizedEntities, EntityRequirements


class ChatBot:
    """
    Master Class that encapsulates other classes defined in this directory (Interpreter, Policy, ConversationHistory)
    """
    def __init__(self, interpreter, policy):
        self.interpreter = interpreter
        self.policy = policy
        self.conversation_history = ConversationHistory()
        self.state = StateBase.base
        self.currently_missing_entities = EntityRequirements()
        self.currently_recognzied_entities = RecognizedEntities()

    def get_reply(self, raw_user_text):
        parsed_user_msg = self._parse_user_msg(raw_text=raw_user_text)  # Interpreter
        self._update_state(latest_intent=parsed_user_msg.classified_intent)
        self._add_parsed_user_msg(*parsed_user_msg)  # ConversationHistory
        # TODO: Continue here, test out chatting and see what is hopelessly broken
        self.policy.get_reply(self.conversation_history)  # Policy

        raise NotImplementedError

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

    def _parse_user_msg(self, raw_text):
        raw_text = self.interpreter.preprocess_user_raw_text(raw_text=raw_text)

        if self.state != StateBase.seeking_additional_info:
            latent_vector = self.interpreter.preprocess_input_single(sentence=raw_text, use_entity_features=True)
            self.currently_recognized_entities = self.interpreter.get_recognized_entities(sentence=raw_text)
            classified_intent = IntentBase.factory(self.interpreter.get_intent(sentence=raw_text))
            self.currently_missing_entities = self.interpreter.get_missing_entities(classified_intent, recognized_entities)
        else:
            latent_vector = self.interpreter.preprocess_input_single(sentence=raw_text, use_entity_features=False)
            recognized_entities = RecognizedEntities()
            classified_intent = IntentFollowUp.factory(self.interpreter.get_intent_follow_up(sentence=raw_text))
            if classified_intent == IntentFollowUp.accept:  # Use entire raw text as missing entity TODO: Clean up to extract just the piece we want?
                self.currently_recognized_entities.add(self.currently_missing_entities.get_missing_entity(), raw_text)
                self.currently_missing_entities.subtract(self.currently_missing_entities.get_missing_entity())

        return raw_text, latent_vector, self.currently_recognized_entities, classified_intent, self.currently_missing_entities

    def _add_parsed_user_msg(self, raw_text, latent_vector, recognized_entities, classified_intent, missing_entities):
        self.conversation_history.user_msgs.append(
            self.conversation_history.ParsedUserMsg(
                raw_text,  # str, directly from user
                latent_vector,  # numpy array, see interpreter.BaaS.encode()
                recognized_entities,  # RecognizedEntities from Enums.py
                classified_intent,  # enum from Enums.py
                missing_entities,  # EntityRequirements from Enums.py
                self.state  # state
            )
        )
        return

    def _update_state(self, latest_intent, begin_processing=False, finished_processing=False):
        if begin_processing:
            self.state = StateBase.processing
            return
        elif finished_processing:
            self.state = StateBase.selecting_results
            return
        elif self.state == StateBase.processing:  # Do not update state regardless of user request if currently processing
            return
        else:  # Update based on latest intent / state
            if self.state == StateBase.base:
                if latest_intent == IntentBase.end_of_conversation:
                    self.state = StateBase.conversation_complete
                    return
                elif not self.missing_entities.is_satisfied():
                    self.state = StateBase.seeking_additional_info
                    return
                elif latest_intent.will_process:
                    self.state = StateBase.ready_to_process
                    return
                else:
                    return
            elif self.state == StateBase.seeking_additional_info:
                if latest_intent == IntentFollowUp.accept:
                    self.state = StateBase.ready_to_process
                    return
                else:  # User rejected the follow up response and will still be seeking additional info
                    return

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
        self._add_parsed_user_msg(*parsed_user_msg)

        # Reply
        reply = self.policy.get_reply(conversation_history=self.conversation_history)
        self.conversation_history.add_bot_msg(raw_text=reply)
        return reply

    def exit_conversation(self):
        return self.conversation_history.user_msgs[-1].classified_intent == cfg.valid_intents['end_of_conversation']

    # TODO: Can likely delete once tested
    def update_history_and_get_more_information(self, input_msg, original_parsed_user_msg, additional_info_conv_history):
        # Parse and update based on user input, same as earlier, but with the follow up variant of the routine
        parsed_user_follow_up_msg = self.interpreter.parse_user_msg_follow_up(raw_text=input_msg, missing_entity=self.policy.missing_entity)
        self._add_parsed_user_msg(*parsed_user_follow_up_msg)

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
