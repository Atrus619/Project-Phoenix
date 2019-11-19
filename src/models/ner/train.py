import pandas as pd
import nltk
from openpyxl import load_workbook


def convert_utterances_to_labelable_tokens(path):
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

    book = load_workbook(path)
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        writer.book = book
        new_df.to_excel(writer, sheet_name='Tokens', index=False)
        writer.save()

    return True


path = 'logs/ner/Intent Training Examples.xlsx'
convert_utterances_to_labelable_tokens(path=path)
