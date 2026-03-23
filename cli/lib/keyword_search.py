from itertools import islice
import math
import string
import os
import pickle
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
from .search_utils import BM25_B, BM25_K1, CACHE_DIR, DEFAULT_SEARCH_LIST, format_search_result, load_movies, load_stopwords


class InvertedIndex:
    """**index**: tokens -> set(document IDs) 
    ```
    "matrix": [1, 2, 3],
    "bomb": [4, 3, 7]
    ```
    **docmap**: ID -> Doc Object
    ```
    1: {"id": 1, "title": "title", "description": "description"},
    2: {"id": 2, "title": "title2", "description": "description2"}
    ```"""

    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")


    def build(self):
        """iterate over all the movies and add them to both the index and the docmap."""
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            doc_description = f"{movie["title"]} {movie["description"]}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_description)


    def save(self):
        """
        Save the index and docmap attributes to disk
        using the [`pickle`](https://docs.python.org/3/library/pickle.html) module's
        [`dump`](https://docs.python.org/3/library/pickle.html#pickle.dump) function.\n 
        - File path/name `cache/index.pkl` for the index.
        - File path/name `cache/docmap.pkl` for the docmasp.
        - Creates the `cache` directory if it doesn't exist 
        (before trying to write files into it).
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)
        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)
        with open(self.term_frequencies_path, "wb") as file:
            pickle.dump(self.term_frequencies, file)
        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)


    def load(self):
        """Load assigns the index and docmap dicts with their respected files\n
        `@throws ValueError` if index.pkl or docmap.pkl files don't exist"""
        if not os.path.isfile(self.index_path) or not os.path.isfile(self.docmap_path):
            raise ValueError("index.pkl or docmap.pkl does not exist")

        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)
        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)
        with open(self.term_frequencies_path, "rb") as file:
            self.term_frequencies = pickle.load(file)
        with open(self.doc_lengths_path, "rb") as file:
            self.doc_lengths = pickle.load(file)


    def get_documents(self, term: str) -> list:
        '''Gets the set of document IDs for a given token,\n 
        **@returns** list of the document IDs, sorted in ascending order.
        '''
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str):
        text_tokens = tokenize_text(text)
        for token in set(text_tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id] = Counter(text_tokens)
        self.doc_lengths[doc_id] = len(text_tokens)

    def __get_avg_doc_length(self) -> float:
        total_doc_length = 0

        if len(self.doc_lengths) < 1:
            return 0.0

        for value in self.doc_lengths.values():
            total_doc_length += value

        return total_doc_length / len(self.doc_lengths)


    def get_tf(self, doc_id, term):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Can not have more than one token")
        term_count = self.term_frequencies.get(doc_id)
        if not term_count:
            return 0

        token = tokens[0]
        return term_count[token]

    def get_idf(self, term: str):
        """returns the inverse document frequency score for a given term"""
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")

        doc_total = len(self.docmap)
        document_frequency = len(self.index.get(tokens[0], set()))
        return math.log((doc_total + 1) / (document_frequency + 1))

    def get_tf_idf(self, doc_id: int, term: str):
        """returns the product of tf and idf"""
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term cannot be more than one token")

        token = tokens[0]
        doc_count = len(self.docmap)
        df = len(self.index[token])

        # Formula -> log((N - df + 0.5) / (df + 0.5) + 1)
        return math.log((doc_count - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)

        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()

        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1

        # Formula -> (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return (tf * (k1 + 1) / (tf + k1 * length_norm))

    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        tokens = tokenize_text(query)

        scores = {}
        for doc_id in self.docmap:
            total_score = 0.0
            for token in tokens:
                total_score += self.bm25(doc_id, token)
            scores[doc_id] = total_score

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_id, score in sorted_scores[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)

        return results


def build_command() -> None:
    """Build the Inverted Index of movies.json and save to disk:\n
    **Save Location:** (cache/docmap.pkl & cache/index.pkl)
    """
    idx = InvertedIndex()
    idx.build()
    idx.save()
    print("Inverted Index built successfully")


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIST, ) -> list[dict]:
    idx = InvertedIndex()
    try:
        idx.load()
    except ValueError as e:
        print("Failed to load inverted index: ", e)

    seen, results = set(), []
    query_tokens = tokenize_text(query)
    for token in query_tokens:
        matching_doc_ids = idx.get_documents(token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                break

    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)

    stop_words = load_stopwords()
    filtered_tokens = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_tokens.append(word)

    stemmer = PorterStemmer()
    stemmed_tokens = []
    for token in filtered_tokens:
        stemmed_tokens.append(stemmer.stem(token))

    return stemmed_tokens


def tf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    try:
        idx.load()
    except ValueError as e:
        print("Failed to load inverted index: ", e)

    return idx.get_tf(doc_id, term)

def idf_command(term: str):
    idx = InvertedIndex()
    try:
        idx.load()
    except ValueError as e:
        print("Failed to load inverted index: ", e)

    return idx.get_idf(term)

def tf_idf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    try:
        idx.load()
    except ValueError as e:
        print("Failed to load inverted index: ", e)

    return idx.get_tf_idf(doc_id, term)

def bm25_idf_command(term: str):
    idx = InvertedIndex()
    try:
        idx.load()
    except ValueError as e:
        print("Failed to load inverted index: ", e)

    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float, b: float):
    idx = InvertedIndex()
    try:
        idx.load()
    except ValueError as e:
        print("Failed to load inverted index: ", e)

    return idx.get_bm25_tf(doc_id, term, k1, b)


def bm25_search_command(query: str, limit: int):
    idx = InvertedIndex()
    try:
        idx.load()
    except ValueError as e:
        print("Failed to load inverted index: ", e)

    return idx.bm25_search(query, limit)
