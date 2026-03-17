import string
from .search_utils import DEFAULT_SEARCH_LIST, load_movies


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIST, ) -> list[dict]:
    movies = load_movies()
    result = []
    for movie in movies:
        preprocessed_query = preprocess_text(query)
        preprocess_title = preprocess_text(movie["title"])
        if preprocessed_query in preprocess_title:
            result.append(movie)

        if len(result) >= limit:
            break

    return result


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text
