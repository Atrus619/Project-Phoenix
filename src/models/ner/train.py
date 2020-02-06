import pandas as pd
import os
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report
from config import Config as cfg
from prefect import task, utilities


def get_ner_model_path(model_name):
    return os.path.join(cfg.serialized_model_dir, model_name + "_ner.ser.gz")


@task
def train_ner(data_path=cfg.ner_and_intent_training_data_path,
              prop_path=cfg.ner_prop_path,
              ner_jar_path=cfg.ner_jar_path,
              model_name=cfg.default_model_name,
              num_folds=cfg.ner_training_num_cv,
              training_folder=cfg.ner_training_folder,
              full_train_path=cfg.ner_full_train_path):
    """
    Trains NER model
    :param data_path: Path to training data (excel file output from utterances_to_tokens that has been annotated by user)
    :param prop_path: Path to properties file
    :param ner_jar_path: Path to stanford NER jar file
    :param model_name: Name of model. Ultimate file will exist in cfg.serialized_model_dir and extension will be _ner.ser.gz
    :param num_folds: Number of folds to use for cross-validation
    :param training_folder: Location to place cv folds while training
    :param full_train_path: Location to place model trained on full data as .tsv
    """
    logger = utilities.logging.get_logger(cfg.chatbot_training_log_name)

    def create_training_files(data_path, num_folds, training_folder, full_train_path):
        """
        Loads in an excel file containing manually labeled tokens and creates a tab delimited file for use with training stanford CRF NER model
        Blank rows are imported as NaN which are intended to be blanks in training file to seperate "documents"
        :return: Saves a temporary folder in logs/ner_cv that will be used for cross-validation (will be cleaned up by later function TODO)
        """
        df = pd.read_excel(data_path, sheet_name='Tokens')
        group_kfold = GroupKFold(n_splits=num_folds)
        group_kfold.get_n_splits()

        for i, (train_index, test_index) in enumerate(group_kfold.split(df.Token, df.Label, df.OG_Text)):
            cv_folder = os.path.join(training_folder, f'fold_{i+1}')
            os.makedirs(cv_folder, exist_ok=True)
            df.iloc[train_index].to_csv(os.path.join(cv_folder, 'train.tsv'), columns=['Token', 'Label'], sep='\t', index=False, header=False)
            df.iloc[test_index].to_csv(os.path.join(cv_folder, 'test.tsv'), columns=['Token'], sep='\t', index=False, header=False)
            df.iloc[test_index].to_csv(os.path.join(cv_folder, 'labels.tsv'), columns=['Token', 'Label'], sep='\t', index=False, header=False)

        df.to_csv(full_train_path, columns=['Token', 'Label'], sep='\t', index=False, header=False)

    def ner_train_test_cv(num_folds, ner_jar_path, output_path, prop_path, full_train_path):
        oof_metrics = {'predictions': [], 'labels': []}
        for i in range(num_folds):
            train_path = os.path.join(training_folder, f'fold_{i+1}', 'train.tsv')
            test_path = os.path.join(training_folder, f'fold_{i+1}', 'test.tsv')
            label_path = os.path.join(training_folder, f'fold_{i+1}', 'labels.tsv')
            predict_path = os.path.join(training_folder, f'fold_{i+1}', 'predict.tsv')

            # Train model
            os.system(f'java -cp {ner_jar_path} edu.stanford.nlp.ie.crf.CRFClassifier '
                      f'-trainFile {train_path} '
                      f'-serializeTo {output_path} '
                      f'-prop {prop_path}')

            # Eval model
            os.system(f'java -cp {ner_jar_path} edu.stanford.nlp.ie.crf.CRFClassifier '
                      f'-loadClassifier {output_path} '
                      f'-textFile {test_path} '
                      f'> {predict_path}')

            # Compute statistics on fold
            predictions = pd.read_csv(predict_path, sep='/', lineterminator=' ', header=None, names=['Token', 'Prediction'])
            predictions = predictions[~predictions.Token.isin(["''", "\n"])]
            oof_metrics['predictions'] += predictions.Prediction.values.tolist()

            labels = pd.read_csv(label_path, names=['Token', 'Label'], sep='\t')
            labels = labels.dropna()
            oof_metrics['labels'] += labels.Label.values.tolist()

            # Too much information, can uncomment if desired
            # logger.info(f'\nCV Fold {i+1}:\n' + classification_report(labels.Label.values, predictions.Prediction.values))

        logger.info(f'\nOut-of-fold metrics across all folds:\n' + classification_report(oof_metrics["labels"], oof_metrics["predictions"]))

        # Train model on full data
        os.system(f'java -cp {ner_jar_path} edu.stanford.nlp.ie.crf.CRFClassifier '
                  f'-trainFile {full_train_path} '
                  f'-serializeTo {output_path} '
                  f'-prop {prop_path}')

    # Run UDFs
    logger.info('----- Training Named Entity Recognition (NER) Model -----')
    logger.info('Generating training files for NER training.')
    create_training_files(data_path=data_path,
                          num_folds=num_folds,
                          training_folder=training_folder,
                          full_train_path=full_train_path)

    logger.info(f'Implementing {num_folds}-fold cross-validation to train NER model.')
    output_path = get_ner_model_path(model_name=model_name)
    ner_train_test_cv(num_folds=num_folds,
                      ner_jar_path=ner_jar_path,
                      output_path=output_path,
                      prop_path=prop_path,
                      full_train_path=full_train_path)
    logger.info('----- NER Training Complete. -----')
