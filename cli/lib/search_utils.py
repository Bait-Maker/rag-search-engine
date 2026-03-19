
import json
import os

DEFAULT_SEARCH_LIST = 5

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
    
