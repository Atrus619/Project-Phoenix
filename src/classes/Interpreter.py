from bert_serving.client import BertClient
import subprocess
from collections import Counter, defaultdict
import nltk
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import os
import pickle as pkl
import zmq


class Interpreter:
    """
    Processes messages from the user using specific NLP models
    """
    def __init__(self, ner_tagger, target_entities):
        self.ner_tagger = ner_tagger
        self.target_entities = target_entities

        self.BaaS = None
        self.entity_classifier = None

    def init_BaaS(self):
        if self.BaaS is not None:
            self.kill_BaaS()

        try:
            subprocess.Popen(['bert-serving-start', '-model_dir', 'logs/bert/cased_L-24_H-1024_A-16/', '-num_worker', '1'])
        except zmq.error.ZMQError:
            self.kill_BaaS()
            subprocess.Popen(['bert-serving-start', '-model_dir', 'logs/bert/cased_L-24_H-1024_A-16/', '-num_worker', '1'])

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
        entity_features = self.get_entity_features_single(sentence)

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
        assert self.entity_classifier is not None, "Entity classifier not yet trained"

        latent_vector = self.preprocess_input_single(sentence)
        return self.entity_classifier.predict(latent_vector).item()

    def get_entity_features_single(self, sentence):
        counter = self.count_ner(self.tag_ner(sentence))
        output_dict = dict()

        for target_entity in self.target_entities:
            output_dict[target_entity] = [1] if target_entity in counter else [0]

        return pd.DataFrame(output_dict)

    def get_entity_features_batch(self, sentences):
        counters_series = sentences.apply(lambda x: self.count_ner(self.tag_ner(x)))
        output_dict = dict()

        for target_entity in self.target_entities:
            output_dict[target_entity] = counters_series.apply(lambda x: 1 if target_entity in x else 0)

        return pd.DataFrame(output_dict)

    def preprocess_input_single(self, sentence):
        if self.BaaS is None:
            self.init_BaaS()
        dense_vector = pd.DataFrame(self.BaaS.encode([sentence]))
        entity_vector = self.get_entity_features_single(sentence)

        return pd.concat([dense_vector, entity_vector], axis=1)

    def preprocess_input_batch(self, sentences):
        dense_features = pd.DataFrame(self.BaaS.encode(list(sentences)))
        entity_features = self.get_entity_features_batch(sentences=sentences)

        return pd.concat([dense_features, entity_features], axis=1)

    def train_SVC(self, X, y, num_cv_folds):
        params = {
            "C": [1, 2, 5, 10, 20, 100],
            "kernel": ["linear"]
        }

        self.entity_classifier = GridSearchCV(
            SVC(C=1, probability=True, class_weight='balanced'),
            param_grid=params, n_jobs=-1, cv=num_cv_folds, scoring='f1_weighted', verbose=1)
        self.entity_classifier.fit(X, y)

    def save(self, path):
        self.kill_BaaS()
        with open(path, 'wb') as f:
            pkl.dump(self, f)
