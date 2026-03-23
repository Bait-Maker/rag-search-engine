
import json
import os

DEFAULT_SEARCH_LIST = 5
SCORE_PRECISION = 3

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORD_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")


def load_movies() -> list[dict]:
    """Read and load the movies.json file\n
    **@returns** list of movie dictionaries\n
    example:
    ```
    {
        "id": 1,
        "title": "Predator",
        "description": "One of Arnold's best films."
    },
    {
        "id": 2,
        "title": "The Thing",
        "description": "American researchers in Antarctica discover..."
    }
    ```"""
    with open(DATA_PATH, "r",) as file:
        data = json.load(file)

    return data["movies"]

def load_stopwords() -> list[str]:
    with open(STOPWORD_PATH, "r", encoding="utf-8") as file:
        stopwords = file.read().splitlines()
        return stopwords

def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: any
) -> dict[str, any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }