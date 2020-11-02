from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm

from utils import read_vocab, read_dataset
from bert_ranker import BERTRanker
from typing import List


def check_label(predicted_cui: str, golden_cui: str) -> int:
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.replace('+', '|').split("|")).intersection(set(golden_cui.replace('+', '|').split("|"))))>0)


def is_correct(meddra_code: str, candidates: List[str], topk: int = 1) -> int:
    for candidate in candidates[:topk]:
        if check_label(candidate, meddra_code): return 1
    return 0


def get_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--model_dir', help='Path to the directory containing BERT checkpoint', type=str)
    parser.add_argument('--data_folder', help='Path to the directory containing BioSyn format dataset', type=str)
    parser.add_argument('--vocab', help='Path to the vocabulary file in BioSyn format', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()

    ################
    entities = read_dataset(args.data_folder)
    ################
    entity_texts = [e['entity_text'].lower() for e in entities]
    labels = [e['label'] for e in entities]
    ##################
    vocab = read_vocab(args.vocab)
    bert_ranker = BERTRanker(args.model_dir, vocab)

    predicted_labels = bert_ranker.predict(entity_texts)
    correct_top1 = []
    correct_top5 = []
    for label, predicted_top_labels in tqdm(zip(labels, predicted_labels), total=len(labels)):
        correct_top1.append(is_correct(label, predicted_top_labels, topk=1))
        correct_top5.append(is_correct(label, predicted_top_labels, topk=5))

    print("Acc@1 is ", np.mean(correct_top1))
    print("Acc@5 is ", np.mean(correct_top5))
