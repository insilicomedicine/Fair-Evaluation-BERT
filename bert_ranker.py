from sentence_transformers import SentenceTransformer, models
from sklearn.neighbors import KDTree
import pandas as pd
from typing import List, Optional
import numpy as np


class BERTRanker:

    def __init__(self, model_dir: str, vocab: Optional[pd.DataFrame] = None) -> None:
        word_embedding_model = models.BERT(model_dir)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=False,
                                       pooling_mode_cls_token=True,
                                       pooling_mode_max_tokens=False)
        self.encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        if vocab is not None:
            self.vocab2index(vocab)
        else:
            self.codes = []
            self.concept_names = []
            self.tree_index = None

    def encode(self, texts: List[str]) -> np.array:
        list_of_encodings = self.encoder.encode(texts, batch_size=128, show_progress_bar=True)
        return np.vstack(list_of_encodings)

    def vocab2index(self, vocab: pd.DataFrame) -> None:
        self.codes = vocab.label.values
        self.concept_names = vocab.concept_name.values
        vocab_embeddings = self.encode(vocab.concept_name.str.lower().tolist())
        self.tree_index = KDTree(vocab_embeddings)

    def predict(self, entity_texts: List[str]) -> np.array:
        if self.tree_index is None:
            raise Exception("The index was not initialized")
        entities_embeddings = self.encode(entity_texts)
        prediction_dists, prediction_idx = self.tree_index.query(entities_embeddings, k=20)
        return self.codes[prediction_idx]