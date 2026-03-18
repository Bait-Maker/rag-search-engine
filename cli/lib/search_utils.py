
import json
import os


DEFAULT_SEARCH_LIST = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORD_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r",) as file:
        data = json.load(file)

    return data["movies"]

def load_stopwords() -> list[str]:
    with open(STOPWORD_PATH, "r", encoding="utf-8") as file:
        stopwords = file.read().splitlines()
        return stopwords
    
