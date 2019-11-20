import pandas as pd
import nltk
from openpyxl import load_workbook
from argparse import ArgumentParser


def utterances_to_tokens(path):
    """
    Adds a sheet to excel file containing utterances for training intent classifier for the purpose of creating labels for NER
    :param path: Path to excel sheet
    :return: Modifies excel sheet in place. Returns True if successful.
    """
    df = pd.read_excel(path, sheet_name='TrainingExamples')

    new_df = pd.DataFrame(columns=['OG_Text', 'Token', 'Label'])

    for i, example in df.iterrows():
        tokens = nltk.word_tokenize(example[1])
        for token in tokens:
            new_df = new_df.append(pd.DataFrame({
                'OG_Text': [example[1]],
                'Token': [token],
                'Label': ['O']
            }))
        new_df = new_df.append(pd.DataFrame({
            'OG_Text': [example[1]],
            'Token': [''],
            'Label': ['']
        }))

    book = load_workbook(path)

    if 'Tokens' in book.sheetnames:
        del book['Tokens']

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        writer.book = book
        new_df.to_excel(writer, sheet_name='Tokens', index=False)
        writer.save()

    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="", help="Path to the excel file containing training examples for intent classification.")
    args = parser.parse_args()

    utterances_to_tokens(path=args.path)
    print(f'Conversion of utterances to tokens successful. Please refer to updated file at {args.path}.')
