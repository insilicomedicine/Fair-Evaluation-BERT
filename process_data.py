from utils import read_dataset, save_dataset
from argparse import ArgumentParser
import pandas as pd
from typing import List, Dict


def process_dataset(test_dataset: List[Dict[str, str]], train_dataset: List[Dict[str, str]]):
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)
    refined_set = test_df[~test_df.entity_text.isin(train_df.entity_text)]
    return refined_set.drop_duplicates().to_dict('records')


def get_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--train_data_folder')
    parser.add_argument('--test_data_folder')
    parser.add_argument('--save_to')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    train_dataset = read_dataset(args.train_data_folder)
    test_dataset = read_dataset(args.test_data_folder)

    refined_test_set = process_dataset(test_dataset, train_dataset)
    save_dataset(refined_test_set, args.save_to)
