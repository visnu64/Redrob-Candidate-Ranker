"""Custom, dependency-free BM25 implementation for high-speed sparse retrieval."""

from __future__ import annotations

import math
import pickle
import re
from pathlib import Path

# Standard English stopwords to keep the index clean and relevant
STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves"
})


def tokenize(text: str) -> list[str]:
    """Tokenize text by splitting on non-alphanumeric chars and lowercasing."""
    if not text:
        return []
    words = re.findall(r'[a-z0-9]+', text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


class BM25Index:
    """A memory-efficient, fast-retrieving BM25 index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.candidate_ids: list[str] = []
        self.doc_lens: list[int] = []
        self.avg_doc_len: float = 0.0
        # Inverted index: {token: [(candidate_index, term_frequency), ...]}
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        # Token -> IDF map
        self.idf: dict[str, float] = {}

    def build(self, documents: list[tuple[str, str]]) -> None:
        """Build index from a list of (candidate_id, text) tuples."""
        self.candidate_ids = []
        self.doc_lens = []
        self.inverted_index = {}
        self.idf = {}

        temp_inverted: dict[str, dict[int, int]] = {}
        total_tokens = 0
        num_docs = len(documents)

        for doc_idx, (cid, text) in enumerate(documents):
            self.candidate_ids.append(cid)
            tokens = tokenize(text)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_tokens += doc_len

            # Calculate term frequencies within this document
            for token in tokens:
                if token not in temp_inverted:
                    temp_inverted[token] = {}
                temp_inverted[token][doc_idx] = temp_inverted[token].get(doc_idx, 0) + 1

        self.avg_doc_len = total_tokens / max(1, num_docs)

        # Convert temp inverted index to lists and calculate IDFs
        for token, doc_tfs in temp_inverted.items():
            self.inverted_index[token] = sorted(doc_tfs.items())
            df = len(doc_tfs)
            # Standard BM25 IDF with smoothing
            self.idf[token] = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))

    def query(self, query_text: str, top_k: int = 1000) -> list[tuple[str, float]]:
        """Query index. Returns [(candidate_id, BM25_score), ...]."""
        query_tokens = tokenize(query_text)
        if not query_tokens:
            return []

        # Accumulate scores for each document that matches query tokens
        doc_scores: dict[int, float] = {}
        
        # Cache parameter calculations
        k1 = self.k1
        b = self.b
        avg_doc_len = self.avg_doc_len

        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            idf_val = self.idf[token]
            for doc_idx, tf in self.inverted_index[token]:
                doc_len = self.doc_lens[doc_idx]
                denominator = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
                score = idf_val * (tf * (k1 + 1.0)) / denominator
                doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + score

        if not doc_scores:
            return []

        # Sort and take top_k
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.candidate_ids[idx], score) for idx, score in sorted_results]

    def save(self, path: Path | str) -> None:
        """Save index to disk using pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "candidate_ids": self.candidate_ids,
            "doc_lens": self.doc_lens,
            "avg_doc_len": self.avg_doc_len,
            "inverted_index": self.inverted_index,
            "idf": self.idf
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path | str) -> BM25Index:
        """Load index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        index = cls(k1=data["k1"], b=data["b"])
        index.candidate_ids = data["candidate_ids"]
        index.doc_lens = data["doc_lens"]
        index.avg_doc_len = data["avg_doc_len"]
        index.inverted_index = data["inverted_index"]
        index.idf = data["idf"]
        return index
