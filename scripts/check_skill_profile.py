"""Quick skill profile check using the SkillGraph RAG pipeline."""

from __future__ import annotations

import argparse
import json

from neo4jRag import build_skill_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a SkillGraph profile for a simple resume sample.")
    parser.add_argument(
        "--role",
        default="Data Scientist",
        help="Target role to evaluate (default: Data Scientist)",
    )
    parser.add_argument(
        "--skills",
        nargs="*",
        default=["Python", "Machine Learning", "SQL", "TensorFlow", "Data Visualization"],
        help="Technical skills to seed the profile",
    )
    parser.add_argument(
        "--soft-skills",
        nargs="*",
        default=["Communication", "Collaboration"],
        help="Soft skills to include",
    )
    args = parser.parse_args()

    parsed_resume = {
        "technical_skills": args.skills,
        "soft_skills": args.soft_skills,
        "experience": [],
        "education": [],
    }

    profile = build_skill_profile(parsed_resume, [args.role], summarise=False)
    data = profile["profiles"][0]
    print(json.dumps({
        "role": data.get("requested_role"),
        "coverage": data.get("coverage_stats"),
        "skills_covered": [
            {
                "name": row.get("name"),
                "importance": row.get("importance"),
                "source": row.get("match_source"),
                "resume": row.get("resume_term"),
            }
            for row in data.get("skills_covered_table", [])
        ],
        "skill_gaps": [row.get("name") for row in data.get("skill_gaps_table", [])[:5]],
    }, indent=2))


if __name__ == "__main__":
    main()
