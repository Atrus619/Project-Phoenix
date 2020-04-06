import os
import string
import random
from src.classes.ConversationHistory import ConversationHistory
from config import Config as cfg
from src.classes.Enums import StateBase, IntentBase, IntentFollowUp, RecognizedEntities, EntityRequirements
import time
import threading


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
        self.currently_recognized_entities = RecognizedEntities()

    def get_reply(self, raw_user_text):
        parsed_user_msg = self._parse_user_msg(raw_text=raw_user_text)  # Interpreter
        self.state = self._get_next_state(latest_intent=parsed_user_msg[3])
        self._add_parsed_user_msg(*parsed_user_msg)  # ConversationHistory
        reply_generator = self.policy.get_reply(self.conversation_history)  # Policy
        for reply in reply_generator:
            self._add_bot_msg(reply)
            yield reply

    def console_interact(self):
        opening_msgs = self.update_history_and_generate_opening_msg()
        self.set_new_user_id()
        for msg in opening_msgs:
            print(msg)

        while True:
            raw_text = self.seek_input_from_user()
            for reply in self.get_reply(raw_text):
                print(reply)
            self._update_state_if_processing()
            if self.exit_conversation():
                return

    def set_new_user_id(self):
        current_ids = os.listdir(os.path.join('app', 'static'))
        while True:
            new_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            if new_id not in current_ids:
                break
        self.policy.visualizer.user_id = new_id
        return

    def _parse_user_msg(self, raw_text):
        raw_text = self.interpreter.preprocess_user_raw_text(raw_text=raw_text)

        if self.state != StateBase.seeking_additional_info:
            latent_vector = self.interpreter.preprocess_input_single(sentence=raw_text, use_entity_features=True)
            self.currently_recognized_entities = self.interpreter.get_recognized_entities(sentence=raw_text)
            classified_intent = IntentBase.factory(self.interpreter.get_intent(sentence=raw_text))
            self.currently_missing_entities = self.interpreter.get_missing_entities(classified_intent, self.currently_recognized_entities)
        else:
            latent_vector = self.interpreter.preprocess_input_single(sentence=raw_text, use_entity_features=False)
            self.currently_recognized_entities = RecognizedEntities()
            classified_intent = IntentFollowUp.factory(self.interpreter.get_intent_follow_up(sentence=raw_text))
            if classified_intent == IntentFollowUp.accept:  # Use entire raw text as missing entity TODO: Clean up to extract just the piece we want?
                self.currently_recognized_entities.add(self.currently_missing_entities.get_missing_entity(), raw_text)
                self.currently_missing_entities.subtract(self.currently_missing_entities.get_missing_entity())

        return raw_text, latent_vector, self.currently_recognized_entities, classified_intent, self.currently_missing_entities

    def _add_parsed_user_msg(self, raw_text, latent_vector, recognized_entities, classified_intent, missing_entities, state=None):
        if state is None:
            state = self.state
        self.conversation_history.add_parsed_user_msg(raw_text, latent_vector, recognized_entities, classified_intent, missing_entities, state)
        return

    def _add_bot_msg(self, txt):
        self.conversation_history.add_bot_msg(raw_text=txt)

    def _get_next_state(self, latest_intent):
        if self.state == StateBase.processing:  # Do not update state regardless of user request if currently processing
            return self.state
        else:  # Update based on latest intent / state
            if self.state == StateBase.base:
                if latest_intent == IntentBase.end_of_conversation:
                    return StateBase.conversation_complete
                elif not self.currently_missing_entities.is_satisfied():
                    return StateBase.seeking_additional_info
                elif latest_intent.will_process:
                    return StateBase.ready_to_process
                else:
                    return self.state
            elif self.state == StateBase.seeking_additional_info:
                if latest_intent == IntentFollowUp.accept:
                    return StateBase.ready_to_process
                else:  # User rejected the follow up response and will still be seeking additional info
                    return StateBase.seeking_additional_info
            else:
                raise NotImplementedError

    def _update_state_if_processing(self):
        if self.policy.visualizer.task:
            self.state = StateBase.processing
            thread = threading.Thread(target=self._wait_for_task_completion)
            thread.start()
            return

    def _wait_for_task_completion(self, interval=5):
        # Ensure this is kept parallel with the similarly named method in SIO.py
        while True:
            if self.policy.visualizer.is_task_complete():
                break
            time.sleep(interval)

        reply = self.policy.visualizer.get_reply(intent=self.conversation_history.get_latest_base_intent())
        self.conversation_history.add_bot_msg(reply)
        print(reply)
        self.policy.visualizer.task = None
        self.state = StateBase.selecting_results
        return

    @staticmethod
    def seek_input_from_user():
        raw_text = input('>>> ')
        while not raw_text:
            print('Please enter in a non-empty value.')
            raw_text = input('>>> ')
        return raw_text

    def update_history_and_generate_opening_msg(self):
        """Returns a generator"""
        opening_msgs = self.policy.get_opening_msg()
        for msg in opening_msgs:
            self._add_bot_msg(txt=msg)
            yield msg
        return

    def wipe_history(self):
        self.conversation_history = ConversationHistory()
        self.state = StateBase.base
        return

    def exit_conversation(self):
        return self.state == StateBase.conversation_complete
