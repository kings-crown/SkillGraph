"""Quick sanity check for SkillGraph Neo4j/CSV connectivity."""

from __future__ import annotations

import argparse
import json
from pprint import pprint

from neo4jRag import SkillGraphClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SkillGraph occupation lookup and skill retrieval.")
    parser.add_argument("query", help="Occupation title or O*NET code (e.g. 'Data Scientist' or '15-2051.00')")
    parser.add_argument("--limit", type=int, default=5, help="Number of candidate occupations to display")
    args = parser.parse_args()

    client = SkillGraphClient()
    matches = client.resolve_occupation(args.query, limit=args.limit)
    if not matches:
        print("No occupation candidates found for", args.query)
        return

    print("Candidates:")
    for match in matches:
        print(f"- {match['code']} :: {match['title']} (score {match.get('score', 0):.3f})")

    best = matches[0]["code"]
    skills = client.occupation_skills(best, limit=10)
    print("\nTop skills for", best)
    if not skills:
        print("  [no skills found]")
    for skill in skills:
        print(f"  - {skill.name} (importance {skill.importance}, level {skill.level}, source {skill.source})")

    context = client.occupation_context(best, top_skills=3, top_activities=3, related_limit=5)
    print("\nOccupation context (trimmed):")
    print(json.dumps({
        "occupation": context.get("occupation"),
        "skills": context.get("skills"),
        "activities": context.get("activities"),
        "related_occupations": context.get("related_occupations"),
    }, indent=2))


if __name__ == "__main__":
    main()
