from bert_serving.client import BertClient
import subprocess
from collections import Counter
import nltk
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.svm import SVC
import os
import pickle as pkl
import zmq
from config import Config as cfg
from src.classes.utils import init_recognized_entities_dict


class Interpreter:
    """
    Processes messages from the user using specific NLP models.
    Contains two models, intent_classifier and intent_follow_up_classifier.
    Intent Classifier is for classifying general text.
    Intent Follow-Up Classifier is for following up and collecting entities that were not collected by the NER component of the intent classifier.
    """
    def __init__(self, ner_tagger=None, target_entities=None, remove_caps=True):
        self.ner_tagger = ner_tagger
        self.target_entities = target_entities
        self.remove_caps = remove_caps

        self.BaaS = None
        self.intent_classifier = None
        self.intent_follow_up_classifier = None

    def parse_user_msg(self, raw_text):
        if self.remove_caps:
            raw_text = raw_text.lower()

        latent_vector = self.preprocess_input_single(sentence=raw_text, use_entity_features=True)
        recognized_entities = self.get_recognized_entities(sentence=raw_text)
        classified_intent = self.get_intent(sentence=raw_text)

        return raw_text, latent_vector, recognized_entities, classified_intent

    def parse_user_msg_follow_up(self, raw_text, missing_entity):
        if self.remove_caps:
            raw_text = raw_text.lower()

        latent_vector = self.preprocess_input_single(sentence=raw_text, use_entity_features=False)
        classified_intent = self.get_intent_follow_up(sentence=raw_text)

        recognized_entities = init_recognized_entities_dict()
        if classified_intent == 'Acceptance':  # Use entire raw text as missing entity TODO: Clean up to extract just the piece we want?
            recognized_entities[missing_entity].append(raw_text)
        else:
            # Just pass to return the empty, initialized recognized_entities dict
            pass

        return raw_text, latent_vector, recognized_entities, classified_intent

    def init_BaaS(self):
        if self.BaaS is not None:
            self.kill_BaaS()

        try:
            subprocess.Popen(['bert-serving-start', '-model_dir', cfg.bert_dir, '-num_worker', '1'])
        except zmq.error.ZMQError:
            self.kill_BaaS()
            subprocess.Popen(['bert-serving-start', '-model_dir', cfg.bert_dir, '-num_worker', '1'])

        self.BaaS = BertClient()

    def kill_BaaS(self):
        os.system('pkill bert')
        self.BaaS = None

    def tag_ner(self, sentence):
        return self.ner_tagger.tag(nltk.word_tokenize(sentence))

    @staticmethod
    def count_ner(tagged_sentence):
        return Counter(tag[1] for tag in tagged_sentence)

    def get_recognized_entities(self, sentence):
        # Output recognized entities is a dictionary.
        # Key is entity letter ('J' for job, 'L' for location)
        # Value is a list of recognized entities as strings, one string for each full recognized entity
        tagged_sentence = self.tag_ner(sentence)
        entity_features = self.get_features_single(sentence)

        output_dict = init_recognized_entities_dict()
        for entity_feature in entity_features:
            current_index = 0

            # Once IndexError is hit, it's time to move to next entity_feature
            try:
                while True:
                    # Find occurrence of entity_feature
                    while tagged_sentence[current_index][1] != entity_feature:
                        current_index += 1

                    # Concatenate stream of same entity
                    output_dict[entity_feature].append(tagged_sentence[current_index][0])
                    current_index += 1
                    while tagged_sentence[current_index][1] == entity_feature:
                        output_dict[entity_feature][-1] += ' ' + tagged_sentence[current_index][0]
                        current_index += 1

            except IndexError:
                continue

        return output_dict

    def get_intent(self, sentence):
        assert self.intent_classifier is not None, "Intent classifier not yet trained"

        latent_vector = self.preprocess_input_single(sentence=sentence, use_entity_features=True)
        return self.intent_classifier.predict(latent_vector).item()

    def get_intent_follow_up(self, sentence):
        assert self.intent_follow_up_classifier is not None, "Intent classifier follow up not yet trained"

        latent_vector = self.preprocess_input_single(sentence=sentence, use_entity_features=False)
        return self.intent_follow_up_classifier.predict(latent_vector).item()

    def get_features_single(self, sentence):
        counter = self.count_ner(self.tag_ner(sentence))
        output_dict = dict()

        for target_entity in self.target_entities:
            output_dict[target_entity] = [1] if target_entity in counter else [0]

        return pd.DataFrame(output_dict)

    def get_features_batch(self, sentences):
        counters_series = sentences.apply(lambda x: self.count_ner(self.tag_ner(x)))
        output_dict = dict()

        for target_entity in self.target_entities:
            output_dict[target_entity] = counters_series.apply(lambda x: 1 if target_entity in x else 0)

        return pd.DataFrame(output_dict)

    def preprocess_input_single(self, sentence, use_entity_features=True):
        if self.BaaS is None:
            self.init_BaaS()

        dense_vector = pd.DataFrame(self.BaaS.encode([sentence]))

        if use_entity_features:  # Calculate entity_features and return both, concatenated together
            entity_vector = self.get_features_single(sentence)
            return pd.concat([dense_vector, entity_vector], axis=1)

        # Else just return the dense features alone
        return dense_vector

    def preprocess_input_batch(self, sentences, use_entity_features=True):
        if self.BaaS is None:
            self.init_BaaS()

        dense_features = pd.DataFrame(self.BaaS.encode(list(sentences)))

        if use_entity_features:  # Calculate entity_features and return both, concatenated together
            entity_features = self.get_features_batch(sentences=sentences)
            return pd.concat([dense_features, entity_features], axis=1)

        # Else just return the dense features alone
        return dense_features

    @staticmethod
    def train_SVM(X, y, num_cv_folds):
        params = {
            "C": [1, 2, 5, 10, 20, 100],
            "kernel": ["linear"]
        }

        model = GridSearchCV(
            SVC(C=1, probability=True, class_weight='balanced'),
            param_grid=params, n_jobs=-1, cv=num_cv_folds, scoring='f1_weighted', verbose=1)

        model.fit(X, y)

        return model.best_estimator_

    @staticmethod
    def eval_trained_model(model, X, y, num_cv_folds):
        assert model, 'Model must already be trained'
        return cross_val_predict(model, X, y, cv=num_cv_folds)

    def save_dict(self, path):
        self.kill_BaaS()
        out_dict = {
            'intent_classifier': self.intent_classifier,
            'intent_follow_up_classifier': self.intent_follow_up_classifier,
            'ner_tagger': self.ner_tagger,
            'target_entities': self.target_entities,
            'remove_caps': self.remove_caps
        }
        with open(path, 'wb') as f:
            pkl.dump(out_dict, f)

    def load_dict(self, path):
        with open(path, 'rb') as f:
            in_dict = pkl.load(f)

        self.intent_classifier = in_dict['intent_classifier']
        self.intent_follow_up_classifier = in_dict['intent_follow_up_classifier']
        self.ner_tagger = in_dict['ner_tagger']
        self.target_entities = in_dict['target_entities']
        self.remove_caps = in_dict['remove_caps']
