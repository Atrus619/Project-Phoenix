from argparse import ArgumentParser
import pandas as pd
from sklearn.metrics import classification_report
from nltk.tag.stanford import StanfordNERTagger
import sys; sys.path.append('')
from src.classes.Interpreter import Interpreter
from config import Config as cfg


# Define helper routines
# TODO: Replace print statements with logger
def load_data(path):
    df = pd.read_excel(path, sheet_name='TrainingExamples')
    return df


def get_ner_tagger(ner_model_path, jar_path):
    return StanfordNERTagger(ner_model_path, jar_path, encoding='utf8')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default=cfg.ner_and_intent_training_data_path, help='Path to training data file. Should have a .xlsx extension.')
    parser.add_argument('--ner_model_path', type=str, default=cfg.ner_model_path, help='Path to pre-trained NER model file. Should have a .ser.gz extension.')
    parser.add_argument('--ner_jar_path', type=str, default=cfg.ner_jar_path, help='Path to NER jar file. Should have a .jar extension.')
    parser.add_argument('--target_entities', type=str, nargs='*', default=cfg.target_entities, help='List of entities to target as additional features (counts the number of occurrences).')
    parser.add_argument('--num_cv_folds', type=int, default=cfg.intent_training_num_cv, help='Number of folds to use for cross-validation.')
    parser.add_argument('--output_path', type=str, default=cfg.default_interpreter_output_path, help='Desired output path for model. Should end in [model_name].pkl')
    args = parser.parse_args()

    # Initialize data, BaaS, and ner_tagger
    print('Initializing data and pre-trained models.')
    data = load_data(args.data_path)
    ner_tagger = get_ner_tagger(ner_model_path=args.ner_model_path, jar_path=args.ner_jar_path)
    interpreter = Interpreter(ner_tagger=ner_tagger, target_entities=args.target_entities)
    interpreter.init_BaaS()

    # Preprocess features from NER model by counting occurrence of each non-O entity and adding as features
    print('Preprocessing input data.')
    X = interpreter.preprocess_input_batch(sentences=data.TrainingExample)

    # Train with GridSearchCV and return best model
    print('Training model.')
    interpreter.train_SVC(X=X, y=data.Intent, num_cv_folds=args.num_cv_folds)

    # Export
    print(f'Model training complete. In-sample score is {interpreter.entity_classifier.score(X, data.Intent):.2f} and out-of-sample score is {interpreter.entity_classifier.best_score_:.2f}.')
    print('Classification Report:')
    print(classification_report(data.Intent, interpreter.entity_classifier.predict(X)))

    interpreter.save(args.output_path)

    print(f'Model successfully exported to {args.output_path}.')
