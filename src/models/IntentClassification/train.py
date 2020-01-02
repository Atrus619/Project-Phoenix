from argparse import ArgumentParser
import pandas as pd
from bert_serving.client import BertClient
import os
import subprocess
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle as pkl
import nltk
from nltk.tag.stanford import StanfordNERTagger
from collections import Counter, defaultdict


# Define helper routines
# TODO: Replace print statements with logger
def load_data(path):
    df = pd.read_excel(path, sheet_name='TrainingExamples')
    return df


def init_BaaS():
    subprocess.Popen(['bert-serving-start', '-model_dir', 'logs/bert/cased_L-24_H-1024_A-16/', '-num_worker', '1'])
    return BertClient()


def kill_BaaS():
    os.system('pkill bert')


def get_ner_tagger(ner_model_path, jar_path):
    return StanfordNERTagger(ner_model_path, jar_path, encoding='utf8')


def tag_ner(sentence, ner_tagger):
    return ner_tagger.tag(nltk.word_tokenize(sentence))


def count_ner(tagged_sentence):
    return Counter(tag[1] for tag in tagged_sentence)


def get_entity_features(sentences, ner_tagger, target_entities):
    counters_series = sentences.apply(lambda x: count_ner(tag_ner(x, ner_tagger)))
    output_dict = defaultdict()

    for target_entity in target_entities:
        output_dict[target_entity] = counters_series.apply(lambda x: x[target_entity] if target_entity in x else 0)

    return pd.DataFrame(output_dict)


def preprocess_input_data(sentences, BaaS, ner_tagger, target_entities):
    dense_features = pd.DataFrame(BaaS.encode(list(sentences)))
    entity_features = get_entity_features(sentences=sentences, ner_tagger=ner_tagger, target_entities=target_entities)
    X = pd.concat([dense_features, entity_features], axis=1)

    return X


def train_SVC(X, y, num_cv_folds):
    params = {
        "C": [1, 2, 5, 10, 20, 100],
        "kernel": ["linear"]
    }

    clf = GridSearchCV(
        SVC(C=1, probability=True, class_weight='balanced'),
        param_grid=params, n_jobs=-1, cv=num_cv_folds, scoring='f1_weighted', verbose=1)
    clf.fit(X, y)

    return clf


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='src/data/intent_and_ner/Intent Training Examples_JL.xlsx', help='Path to training data file. Should have a .xlsx extension.')
    parser.add_argument('--ner_model_path', type=str, default='src/models/ner/ner-model.ser.gz', help='Path to pre-trained NER model file. Should have a .ser.gz extension.')
    parser.add_argument('--ner_jar_path', type=str, default='logs/ner/stanford-ner-2018-10-16/stanford-ner.jar', help='Path to NER jar file. Should have a .jar extension.')
    parser.add_argument('--target_entities', type=str, nargs='*', help='List of entities to target as additional features (counts the number of occurrences).')
    parser.add_argument('--num_cv_folds', type=int, default=5, help='Number of folds to use for cross-validation.')
    parser.add_argument('--output_path', type=str, default='src/models/IntentClassification/IntentClassifier.pkl', help='Desired output path for model. Should end in [model_name].pkl')
    args = parser.parse_args()

    # Initialize data, BaaS, and ner_tagger
    print('Initializing data and pre-trained models.')
    data = load_data(args.data_path)
    BaaS = init_BaaS()
    ner_tagger = get_ner_tagger(ner_model_path=args.ner_model_path, jar_path=args.ner_jar_path)

    # Preprocess features from NER model by counting occurrence of each non-O entity and adding as features
    print('Preprocessing input data.')
    X = preprocess_input_data(sentences=data.TrainingExample, BaaS=BaaS, ner_tagger=ner_tagger, target_entities=args.target_entities)

    # Train with GridSearchCV and return best model
    print('Training model.')
    model = train_SVC(X=X, y=data.Intent, num_cv_folds=args.num_cv_folds)

    # Export
    print(f'Model training complete. In-sample score is {model.score(X, data.Intent):.2f} and out-of-sample score is {model.best_score_:.2f}.')
    print('Classification Report:')
    print(classification_report(data.Intent, model.predict(X)))

    with open(args.output_path, 'wb') as f:
        pkl.dump(model, f)

    print(f'Model successfully exported to {args.output_path}.')

    # Kill BaaS at end
    kill_BaaS()
