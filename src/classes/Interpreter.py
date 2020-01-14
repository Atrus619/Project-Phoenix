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


class Interpreter:
    """
    Processes messages from the user using specific NLP models
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

        latent_vector = self.preprocess_input_single(sentence=raw_text)
        recognized_entities = self.get_recognized_entities(sentence=raw_text)
        classified_intent = self.get_intent(sentence=raw_text)

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

    def count_ner(self, tagged_sentence):
        return Counter(tag[1] for tag in tagged_sentence)

    def get_recognized_entities(self, sentence):
        tagged_sentence = self.tag_ner(sentence)
        entity_features = self.get_features_single(sentence)

        output_dict = dict()
        for entity_feature in entity_features:
            output_dict[entity_feature] = []
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
        assert self.intent_classifier is not None, "Entity classifier not yet trained"

        latent_vector = self.preprocess_input_single(sentence)
        return self.intent_classifier.predict(latent_vector).item()

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
