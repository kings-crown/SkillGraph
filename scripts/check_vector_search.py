"""Sanity check for the SkillGraph Qdrant vector store."""

from __future__ import annotations

import argparse
from pprint import pprint

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neo4jRag import SkillGraphVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SkillGraph vector store availability and skill lookup.")
    parser.add_argument("skill", nargs="?", default="natural language processing", help="Skill phrase to search for")
    parser.add_argument("--code", default="15-2051.00", help="Occupation code to filter by (default: 15-2051.00)")
    parser.add_argument("--top", type=int, default=5, help="Number of vector matches to return")
    args = parser.parse_args()

    store = SkillGraphVectorStore()
    if not store.available():
        print("Vector store unavailable. Ensure Qdrant + OpenAI credentials are set.")
        return

    store.ensure_collection()
    matches = store.search_skills(args.skill, occupation_code=args.code, top_k=args.top)
    if not matches:
        print("No matches returned from vector store.")
        return

    print(f"Top {len(matches)} matches for '{args.skill}' (occupation {args.code}):")
    for hit in matches:
        print(
            "- {name} (element {id}, score {score:.3f}, importance {importance})".format(
                name=hit.get("name"),
                id=hit.get("element_id"),
                score=hit.get("score", 0.0),
                importance=hit.get("importance"),
            )
        )
        print("  description:", hit.get("description"))


if __name__ == "__main__":
    main()
