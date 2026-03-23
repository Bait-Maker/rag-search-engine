#!/usr/bin/env python3

import argparse

from lib.keyword_search import bm25_search_command, bm25_tf_command, build_command, idf_command, bm25_idf_command, search_command, tf_command, tf_idf_command
from lib.search_utils import BM25_B, BM25_K1, DEFAULT_SEARCH_LIST

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

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Calculate the mb25idf score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get the mb25idf for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to calculate BM2-TF for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 B parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIST, help="Number of query result that appear"
    )

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
            tf = tf_command(args.id, args.term)
            print(tf)
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = tf_idf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            results = bm25_search_command(args.query, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res["id"]}) {res["title"]} - Score: {res["score"]:.2f}" )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
