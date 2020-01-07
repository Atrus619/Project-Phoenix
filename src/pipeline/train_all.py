import prefect
from argparse import ArgumentParser


if __name__ == "__main__":
    # 0. Parse args
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="", help="Path to the excel file containing training examples for intent classification.")

    # 1. Utterances to tokens
    # TODO: Begin prefect work with utterances to tokens

    # 2. Pause while user updates step in between here


    # 3. Train NER


    # 4. Train Intent
