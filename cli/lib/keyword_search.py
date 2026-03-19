import string
import os
import pickle
from collections import defaultdict
from nltk.stem import PorterStemmer
from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIST, load_movies, load_stopwords


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
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")


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

    def load(self):
        if not os.path.isfile(self.index_path) and not os.path.isfile(self.docmap_path):
            raise ValueError("index.pkl or docmap.pkl does not exist")
        
        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)

        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)



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


def build_command() -> None:
    """Build the Inverted Index of movies.json and save to disk:\n
    **Save Location:** (cache/docmap.pkl & cache/index.pkl)
    """
    idx = InvertedIndex()
    idx.build()
    idx.save()
    print("Inverted Index created and saved successfully")


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
    tokens = text.split(" ")
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

