from argparse import ArgumentParser
import pandas as pd
import os
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report
import pickle as pkl


FOLDER = 'logs/ner/cv'
FULL_TRAIN_PATH = os.path.join(FOLDER, 'full_train.tsv')


def create_training_files(path, num_folds=5):
    """
    Loads in an excel file containing manually labeled tokens and creates a tab delimited file for use with training stanford CRF NER model
    Blank rows are imported as NaN which are intended to be blanks in training file to seperate "documents"
    :param path: Path to excel file
    :param num_folds: Number of splits to use for cross validation
    :return: Saves a temporary folder in logs/ner_cv that will be used for cross-validation (will be cleaned up by later function TODO)
    """
    df = pd.read_excel(path, sheet_name='Tokens')
    group_kfold = GroupKFold(n_splits=num_folds)
    group_kfold.get_n_splits()

    for i, (train_index, test_index) in enumerate(group_kfold.split(df.Token, df.Label, df.OG_Text)):
        cv_folder = os.path.join(FOLDER, f'fold_{i+1}')
        os.makedirs(cv_folder, exist_ok=True)
        df.iloc[train_index].to_csv(os.path.join(cv_folder, 'train.tsv'), columns=['Token', 'Label'], sep='\t', index=False, header=False)
        df.iloc[test_index].to_csv(os.path.join(cv_folder, 'test.tsv'), columns=['Token'], sep='\t', index=False, header=False)
        df.iloc[test_index].to_csv(os.path.join(cv_folder, 'labels.tsv'), columns=['Token', 'Label'], sep='\t', index=False, header=False)

    df.to_csv(FULL_TRAIN_PATH, columns=['Token', 'Label'], sep='\t', index=False, header=False)


def ner_train_test_cv(num_folds, jar_path, output_path, prop_path, oof_metrics_path):
    oof_metrics = []
    for i in range(num_folds):
        train_path = os.path.join(FOLDER, f'fold_{i+1}', 'train.tsv')
        test_path = os.path.join(FOLDER, f'fold_{i+1}', 'test.tsv')
        label_path = os.path.join(FOLDER, f'fold_{i+1}', 'labels.tsv')
        predict_path = os.path.join(FOLDER, f'fold_{i+1}', 'predict.tsv')

        # Train model
        os.system(f'java -cp {jar_path} edu.stanford.nlp.ie.crf.CRFClassifier '
                  f'-trainFile {train_path} '
                  f'-serializeTo {output_path} '
                  f'-prop {prop_path}')

        # Eval model
        os.system(f'java -cp {jar_path} edu.stanford.nlp.ie.crf.CRFClassifier '
                  f'-loadClassifier {output_path} '
                  f'-textFile {test_path} '
                  f'> {predict_path}')

        # Compute statistics on fold
        predictions = pd.read_csv(predict_path, sep='/', lineterminator=' ', header=None, names=['Token', 'Prediction'])
        predictions = predictions[~predictions.Token.isin(["''", "\n"])]

        labels = pd.read_csv(label_path, names=['Token', 'Label'], sep='\t')
        labels = labels.dropna()

        oof_metrics.append(classification_report(labels.Label, predictions.Prediction, output_dict=True))

    # TODO: Come back here when training on actual examples to decide what to do with this object.
    with open(oof_metrics_path, 'wb') as f:
        pkl.dump(oof_metrics, f)

    # Train model on full data
    os.system(f'java -cp {jar_path} edu.stanford.nlp.ie.crf.CRFClassifier '
              f'-trainFile {FULL_TRAIN_PATH} '
              f'-serializeTo {output_path} '
              f'-prop {prop_path}')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--train_path', type=str, default='', help='Path to training file.')
    parser.add_argument('--prop_path', type=str, default='src/models/ner/config.prop', help='Path to properties file.')
    parser.add_argument('--ner_jar_path', type=str, default='logs/ner/stanford-ner-2018-10-16/stanford-ner.jar', help='Path to stanford ner jar file.')
    parser.add_argument('--output_path', type=str, default='src/models/ner/ner-model.ser.gz', help='Desired output path for model. Should end in .ser.gz.')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds to use for cross-validation.')
    parser.add_argument('--oof_metrics_path', type=str, default='logs/ner/cv/oof_metrics.pkl', help='Desired output path for output metrics. Pickle file that should end in .pkl')
    args = parser.parse_args()

    create_training_files(path=args.train_path, num_folds=args.num_folds)

    ner_train_test_cv(num_folds=args.num_folds, jar_path=args.ner_jar_path, output_path=args.output_path, prop_path=args.prop_path, oof_metrics_path=args.oof_metrics_path)
