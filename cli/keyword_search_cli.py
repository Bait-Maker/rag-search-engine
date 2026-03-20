#!/usr/bin/env python3

import argparse

from lib.keyword_search import build_command, idf_command, search_command, tf_command, tfidf_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index of movies.json")

    tf_parser = subparsers.add_parser("tf", help="Find the term frequency of a given movie")
    tf_parser.add_argument("id", type=int, help="Document ID of the movie")
    tf_parser.add_argument("term", type=str, help="Word to get the frequency of")

    idf_parser = subparsers.add_parser(
        "idf", help="Calculate the inverse document frequency of a given term"
    )
    idf_parser.add_argument("term", type=str, help="Term to get the IDF score for")

    tfxIdf_parser = subparsers.add_parser(
        "tfidf",
        help="Calculate the Term Frequency-Inverse Document Frequency for a term in a document"
    )
    tfxIdf_parser.add_argument("doc_id", type=int, help="document id to use for tf score")
    tfxIdf_parser.add_argument("term", type=str, help="Term to get tfIdf for")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            results = search_command(args.query)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
        case "build":
            print("Building inverted index...")
            build_command()
        case "tf":
            print(f"Getting term frequency for: ID: {args.id}, term: {args.term}")
            tf_command(args.id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
