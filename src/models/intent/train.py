import os
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from nltk.tag.stanford import StanfordNERTagger
from src.classes.Interpreter import Interpreter
from config import Config as cfg
from prefect import task, utilities
from src.models.ner.train import get_ner_model_path


# Define helper routines
def load_data(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name)
    return df


def get_ner_tagger(ner_model_path, jar_path):
    return StanfordNERTagger(ner_model_path, jar_path, encoding='utf8')


def get_target_entities(path, sheet_name):
    df = load_data(path=path, sheet_name=sheet_name)
    uniques = df.Label.unique()
    uniques = uniques[~pd.isnull(uniques)]
    uniques = uniques[uniques != 'O']
    return uniques


def get_interpreter_dict_path(model_name):
    return os.path.join(cfg.serialized_model_dir, model_name + '_interpreter_dict.pkl')


@task
def train_intent_and_initialize_interpreter(data_path=cfg.ner_and_intent_training_data_path,  # Path to training data file. Should have a .xlsx extension.
                                            ner_jar_path=cfg.ner_jar_path,  # Path to NER jar file. Should have a .jar extension.
                                            num_cv_folds=cfg.intent_training_num_cv,  # Number of folds to use for cross-validation
                                            model_name=cfg.default_model_name,  # Output file will end in _interpreter_dict.pkl
                                            remove_caps=True):  # Whether to remove caps upon parsing raw text. Should be the same as how the NER classifier was trained

    logger = utilities.logging.get_logger(cfg.chat_bot_training_log_name)
    logger.info('----- Training Intent Classifier and Initializing Interpreter Class -----')

    # Initialize data, BaaS, and ner_tagger
    logger.info('Initializing data and pre-trained models.')
    data = load_data(path=data_path, sheet_name='TrainingExamples')
    target_entities = get_target_entities(path=data_path, sheet_name='Tokens')
    ner_tagger = get_ner_tagger(ner_model_path=get_ner_model_path(model_name=model_name), jar_path=ner_jar_path)
    interpreter = Interpreter(ner_tagger=ner_tagger, target_entities=target_entities, remove_caps=remove_caps)
    # interpreter.init_BaaS()

    # Preprocess features from NER model by counting occurrence of each non-O entity and adding as features
    logger.info('Preprocessing input data.')
    X = interpreter.preprocess_input_batch(sentences=data.TrainingExample, use_entity_features=True)

    # Train with GridSearchCV and return best model
    logger.info('Training model.')
    interpreter.intent_classifier = interpreter.train_SVM(X=X, y=data.Intent, num_cv_folds=num_cv_folds)

    # Export
    in_sample_predictions = interpreter.intent_classifier.predict(X)
    logger.info('In-Sample Classification Report:')
    logger.info('\n' + classification_report(data.Intent, in_sample_predictions))

    out_of_fold_predictions = interpreter.eval_trained_model(model=interpreter.intent_classifier, X=X, y=data.Intent, num_cv_folds=num_cv_folds)
    logger.info('Out-of-Sample Classification Report:')
    logger.info('\n' + classification_report(data.Intent, out_of_fold_predictions))

    logger.info(
        f'Intent Classifier Model training complete. In-sample score is {interpreter.intent_classifier.score(X, data.Intent):.2f} '
        f'and out-of-sample score is {f1_score(data.Intent, out_of_fold_predictions, average="weighted"):.2f}.'
    )

    output_path = get_interpreter_dict_path(model_name=model_name)
    interpreter.save_dict(output_path)

    logger.info(f'Model dict successfully exported to {output_path}.')
    logger.info('----- Intent classifier successfully trained and interpreter successfully initialized and serialized -----')


@task
def train_intent_follow_up(data_path=cfg.ner_and_intent_training_data_path,
                           model_name=cfg.default_model_name,
                           num_cv_folds=cfg.intent_follow_up_training_num_cv):
    logger = utilities.logging.get_logger(cfg.chat_bot_training_log_name)
    logger.info('----- Training Intent Follow Up Model -----')

    # Initialize pretrained interpreter
    logger.info('Initializing pretrained interpreter.')
    interpreter = Interpreter()
    interpreter_dict_path = get_interpreter_dict_path(model_name=model_name)
    interpreter.load_dict(interpreter_dict_path)

    # Load in training data
    logger.info('Loading in data and preprocessing.')
    data = load_data(path=data_path, sheet_name='Intent_Follow_Up_Training_Examples')
    X = interpreter.preprocess_input_batch(sentences=data.TrainingExample, use_entity_features=False)

    # Train model using built-in method
    logger.info('Training model.')
    interpreter.intent_follow_up_classifier = interpreter.train_SVM(X=X, y=data.Intent, num_cv_folds=num_cv_folds)

    # Output classification report
    in_sample_predictions = interpreter.intent_follow_up_classifier.predict(X)
    logger.info('In-Sample Classification Report:')
    logger.info('\n' + classification_report(data.Intent, in_sample_predictions))

    out_of_fold_predictions = interpreter.eval_trained_model(model=interpreter.intent_follow_up_classifier, X=X, y=data.Intent, num_cv_folds=num_cv_folds)
    logger.info('Out-of-Sample Classification Report:')
    logger.info('\n' + classification_report(data.Intent, out_of_fold_predictions))

    logger.info(
        f'Intent Follow-Up Classifier Model training complete. In-sample score is {interpreter.intent_follow_up_classifier.score(X, data.Intent):.2f} '
        f'and out-of-sample score is {f1_score(data.Intent, out_of_fold_predictions, average="weighted"):.2f}.'
    )

    # Re-serialize model
    interpreter.save_dict(interpreter_dict_path)

    logger.info(f'Model dict successfully exported to {interpreter_dict_path}.')
    logger.info('----- Intent follow-up classifier successfully trained and interpreter successfully re-serialized -----')
