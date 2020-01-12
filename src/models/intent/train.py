import pandas as pd
from sklearn.metrics import classification_report
from nltk.tag.stanford import StanfordNERTagger
from src.classes.Interpreter import Interpreter
from config import Config as cfg
from prefect import task, utilities
from src.pipeline.utils import kill_BaaS_externally


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


@task(state_handlers=[kill_BaaS_externally])
def train_intent_and_initialize_interpreter(data_path=cfg.ner_and_intent_training_data_path,  # Path to training data file. Should have a .xlsx extension.
                                            ner_model_path=cfg.ner_model_path,  # Path to pre-trained NER model file. Should have a .ser.gz extension.
                                            ner_jar_path=cfg.ner_jar_path,  # Path to NER jar file. Should have a .jar extension.
                                            num_cv_folds=cfg.intent_training_num_cv,  # Number of folds to use for cross-validation
                                            output_path=cfg.default_interpreter_dict_output_path,  # Desired output path for model. Should end in [model_name].pkl
                                            remove_caps=True):  # Whether to remove caps upon parsing raw text. Should be the same as how the NER classifier was trained

    logger = utilities.logging.get_logger(cfg.chat_bot_training_log_name)
    logger.info('----- Training Intent Classifier and Initializing Interpreter Class -----')

    # Initialize data, BaaS, and ner_tagger
    logger.info('Initializing data and pre-trained models.')
    data = load_data(path=data_path, sheet_name='TrainingExamples')
    target_entities = get_target_entities(path=data_path, sheet_name='Tokens')
    ner_tagger = get_ner_tagger(ner_model_path=ner_model_path, jar_path=ner_jar_path)
    interpreter = Interpreter(ner_tagger=ner_tagger, target_entities=target_entities, remove_caps=remove_caps)
    interpreter.init_BaaS()

    # Preprocess features from NER model by counting occurrence of each non-O entity and adding as features
    logger.info('Preprocessing input data.')
    X = interpreter.preprocess_input_batch(sentences=data.TrainingExample)

    # Train with GridSearchCV and return best model
    logger.info('Training model.')
    interpreter.train_SVC(X=X, y=data.Intent, num_cv_folds=num_cv_folds)

    # Export
    logger.info(
        f'Model training complete. In-sample score is {interpreter.entity_classifier.score(X, data.Intent):.2f} '
        f'and out-of-sample score is {interpreter.entity_classifier.best_score_:.2f}.'
    )

    in_sample_predictions = interpreter.entity_classifier.predict(X)
    logger.info('In-Sample Classification Report:')
    logger.info('\n' + classification_report(data.Intent, in_sample_predictions))

    out_of_fold_predictions = interpreter.eval_trained_model(X=X, y=data.Intent, num_cv_folds=num_cv_folds)
    logger.info('Out-of-Sample Classification Report:')
    logger.info('\n' + classification_report(data.Intent, out_of_fold_predictions))

    interpreter.save_dict(output_path)
    interpreter.kill_BaaS()

    logger.info(f'Model dict successfully exported to {output_path}.')
    logger.info('----- Intent classifier successfully trained and interpreter successfully initialized and serialized -----')
