import pandas as pd
import os
from glob import glob
from typing import List, Dict


def read_vocab(path: str) -> pd.DataFrame:
    data = []
    with open(path, encoding='utf-8') as input_stream:
        for line in input_stream:
            data.append({'label': line.split('||')[0], 'concept_name':line.strip().split('||')[1]})
    return pd.DataFrame(data)


def read_annotation_file(ann_file_path: str) -> List[Dict[str, str]]:
    data: List[Dict[str, str]] = []
    with open(ann_file_path, encoding='utf-8') as input_stream:
        for row_id, line in enumerate(input_stream):
            splitted_line = line.strip().split('||')
            mention = splitted_line[-2]
            concept_id = splitted_line[-1]
            data.append({'entity_text': mention, 'label':concept_id})
    return data


def read_dataset(dataset_folder: str) -> List[Dict[str, str]]:
    ann_file_pattern: str = os.path.join(dataset_folder, '*.concept')
    dataset: List[Dict[str, str]] = []
    for ann_file_path in glob(ann_file_pattern):
        dataset += read_annotation_file(ann_file_path)
    return dataset


def save_dataset(dataset: List[Dict[str, str]], path: str) -> None:
    if not os.path.exists(path): os.mkdir(path)
    fpath: str = os.path.join(path, '0.concept')
    with open(fpath, 'w', encoding='utf-8') as output_stream:
        for entity in dataset:
            output_stream.write(f"-1||0|0||Disease||{entity['entity_text']}||{entity['label']}\n")
