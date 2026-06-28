import json
import math
import os
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple


class BM25SparseVectorizer:
    """
    Small corpus-side BM25 encoder for Qdrant sparse vectors.

    Document vectors store BM25 term weights. Query vectors store the matching
    query terms with value 1, so Qdrant sparse dot product equals BM25 scoring.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avgdl = 0.0
        self.doc_count = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        normalized = unicodedata.normalize("NFKC", text).lower()
        return re.findall(r"[\w]+", normalized, flags=re.UNICODE)

    def fit(self, texts: Sequence[str]) -> "BM25SparseVectorizer":
        tokenized_docs = [self.tokenize(text) for text in texts]
        self.doc_count = len(tokenized_docs)
        total_length = sum(len(tokens) for tokens in tokenized_docs)
        self.avgdl = total_length / self.doc_count if self.doc_count else 0.0

        document_frequency: Counter[str] = Counter()
        for tokens in tokenized_docs:
            document_frequency.update(set(tokens))

        terms = sorted(document_frequency)
        self.vocab = {term: idx for idx, term in enumerate(terms)}
        self.idf = {
            term: math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }
        return self

    def encode_document(self, text: str) -> Tuple[List[int], List[float]]:
        tokens = self.tokenize(text)
        if not tokens or not self.vocab:
            return [], []

        counts = Counter(tokens)
        doc_len = len(tokens)
        length_norm = self.k1 * (1.0 - self.b + self.b * doc_len / (self.avgdl or 1.0))

        sparse_items = []
        for term, tf in counts.items():
            index = self.vocab.get(term)
            if index is None:
                continue
            numerator = tf * (self.k1 + 1.0)
            weight = self.idf.get(term, 0.0) * numerator / (tf + length_norm)
            if weight > 0.0:
                sparse_items.append((index, float(weight)))

        sparse_items.sort(key=lambda item: item[0])
        return [idx for idx, _ in sparse_items], [value for _, value in sparse_items]

    def encode_query(self, text: str) -> Tuple[List[int], List[float]]:
        tokens = self.tokenize(text)
        if not tokens or not self.vocab:
            return [], []

        counts = Counter(tokens)
        sparse_items = [
            (self.vocab[term], float(tf))
            for term, tf in counts.items()
            if term in self.vocab
        ]
        sparse_items.sort(key=lambda item: item[0])
        return [idx for idx, _ in sparse_items], [value for _, value in sparse_items]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            "avgdl": self.avgdl,
            "doc_count": self.doc_count,
            "vocab": self.vocab,
            "idf": self.idf,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BM25SparseVectorizer":
        vectorizer = cls(k1=float(data.get("k1", 1.5)), b=float(data.get("b", 0.75)))
        vectorizer.avgdl = float(data.get("avgdl", 0.0))
        vectorizer.doc_count = int(data.get("doc_count", 0))
        vectorizer.vocab = {str(term): int(index) for term, index in data.get("vocab", {}).items()}
        vectorizer.idf = {str(term): float(value) for term, value in data.get("idf", {}).items()}
        return vectorizer

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BM25SparseVectorizer":
        with open(path, "r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))


def bm25_index_path(db_path: str, collection_name: str) -> str:
    return os.path.join(db_path, f"{collection_name}.bm25.json")
