"""Utility to exercise both skill profile generation flows on a PDF resume.

The script performs the following steps:
1. Extract text from the provided resume PDF.
2. Parse the resume into structured fields using the existing Azure-powered parser.
3. Produce a narrative skill assessment using the legacy `SkillAnalysis` pipeline.
4. Generate a graph-grounded profile using `SkillGraphRAG` backed by the Neo4j O*NET graph.

Environment
-----------
Ensure the following secrets are available via environment variables (either globally or
by placing a populated `.env` next to this script):
- Azure OpenAI credentials required by `AzureAIClient` for resume parsing and the
  `SkillAnalysis` pipeline.
- `NEO4J_*` and `OPENAI_API_KEY` values for the SkillGraph RAG module.

Usage
-----
```bash
python3 test_skill_profiles.py path/to/resume.pdf --role "Data Scientist" --role "Business Analyst"
```
If `--role` is omitted the script will fall back to the next-step positions predicted by
`CareerPath.suggest_path`.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from openai import AsyncOpenAI

try:  # Optional dependency; fall back to a tiny loader if unavailable.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - executed only when python-dotenv missing
    def load_dotenv(path: Path | None = None) -> bool:
        """Minimal .env loader used when python-dotenv is unavailable."""

        if path is None:
            return False

        try:
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                key, sep, value = stripped.partition("=")
                if sep:
                    os.environ.setdefault(key.strip(), value.strip())
            return True
        except FileNotFoundError:
            return False


def _flatten_skill_terms(skills) -> List[str]:
    """Normalise skill payloads to plain text for prompting/logging."""
    flattened: List[str] = []
    seen: set = set()
    if not skills:
        return flattened
    for skill in skills:
        candidate = None
        if isinstance(skill, dict):
            candidate = (
                skill.get("normalised_term")
                or skill.get("raw_term")
                or skill.get("skill")
                or skill.get("name")
            )
        elif isinstance(skill, str):
            candidate = skill
        if not candidate:
            continue
        text = str(candidate).strip()
        key = text.lower()
        if key and key not in seen:
            seen.add(key)
            flattened.append(text)
    return flattened


# Ensure repository modules (backend/career_coach, etc.) are importable when the script
# is executed from within ``SkillGraph``.
REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if BACKEND_DIR.exists():
    sys.path.insert(0, str(BACKEND_DIR))

from career_coach.career_path import CareerPath
from career_coach.resume_parser import ResumeParser, SaveResumeToDisk
from career_coach.skill_analysis import SkillAnalysis
from career_coach.utils.skill_normaliser import SkillNormaliser

try:
    from SkillGraph.neo4jRag import SkillGraphRAG, SkillRecord
except ImportError:
    sys.path.insert(0, str(REPO_ROOT / "SkillGraph"))
    try:
        from neo4jRag import SkillGraphRAG, SkillRecord
    except ImportError as exc:  # pragma: no cover - fail fast with helpful message
        raise SystemExit(
            "SkillGraph.neo4jRag (or local neo4jRag.py) is required. Ensure the module exists and dependencies are installed."
        ) from exc


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _load_env() -> None:
    """Load environment variables from an adjacent .env file if present."""
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)  # falls back to noop if file missing


def extract_pdf_text(pdf_path: Path) -> str:
    """Return the concatenated text of every page in the PDF."""
    logger.info("Reading resume PDF: %s", pdf_path)
    return SaveResumeToDisk.retrieve_resume(str(pdf_path))


_SKILL_NORMALISER = SkillNormaliser()


def _collect_skill_contexts(parsed_resume: Dict) -> List[str]:
    """Collect textual snippets that give provenance for skill mentions."""
    contexts: List[str] = []
    for entry in parsed_resume.get("experience", []):
        if not isinstance(entry, dict):
            continue
        snippets = [
            entry.get("job_title", ""),
            entry.get("company", ""),
            entry.get("details", ""),
        ]
        context = " | ".join(part for part in snippets if part)
        if context:
            contexts.append(context)
    for entry in parsed_resume.get("education", []):
        if not isinstance(entry, dict):
            continue
        snippets = [
            entry.get("degree", ""),
            entry.get("institution", ""),
            entry.get("year", ""),
        ]
        context = " | ".join(part for part in snippets if part)
        if context:
            contexts.append(context)
    return contexts


def _normalise_parsed_skills(parsed_resume: Dict) -> None:
    """Ensure skill lists are tagged and canonicalised against the alias catalog."""
    contexts = _collect_skill_contexts(parsed_resume)

    def _normalise_field(field: str) -> None:
        raw = parsed_resume.get(field) or []
        # Avoid double-normalising if the backend already produced structured objects.
        if raw and isinstance(raw[0], dict) and "normalised_term" in raw[0]:
            return
        parsed_resume[field] = _SKILL_NORMALISER.normalise_skill_list(
            raw,
            source_section=field,
            contexts=contexts,
        )

    for field in ("technical_skills", "soft_skills"):
        _normalise_field(field)

    top_skills = parsed_resume.get("Top_10_Skills") or []
    if top_skills:
        enriched: List[Dict[str, object]] = []
        for entry in top_skills:
            if not isinstance(entry, dict):
                continue
            normalised = _SKILL_NORMALISER.normalise_skill(
                entry.get("skill", ""),
                source_section="top_10_skills",
            )
            updated = dict(entry)
            if normalised:
                updated["normalised_term"] = normalised["normalised_term"]
                updated["confidence"] = normalised["confidence"]
                if normalised.get("element_id"):
                    updated["element_id"] = normalised["element_id"]
                if normalised.get("aliases"):
                    updated["aliases"] = normalised["aliases"]
                if normalised.get("related_tools"):
                    updated["related_tools"] = normalised["related_tools"]
            enriched.append(updated)
        parsed_resume["Top_10_Skills"] = enriched


def _merge_unique_list(
    existing: Optional[List[Any]],
    incoming: Optional[List[Any]],
) -> List[Any]:
    result: List[Any] = []
    seen: set[str] = set()
    for collection in (existing or []), (incoming or []):
        for item in collection:
            if isinstance(item, dict):
                key = item.get("skill") or item.get("normalised_term") or json.dumps(item, sort_keys=True)
            else:
                key = str(item).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(item)
    return result


def _merge_parsed_runs(parsed_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not parsed_runs:
        return {}
    merged = copy.deepcopy(parsed_runs[0])
    for run in parsed_runs[1:]:
        for field in ("technical_skills", "soft_skills"):
            merged[field] = _merge_unique_list(merged.get(field), run.get(field))
        merged["Top_10_Skills"] = _merge_unique_list(merged.get("Top_10_Skills"), run.get("Top_10_Skills"))
    return merged


async def parse_resume(pdf_text: str) -> Dict:
    """Parse resume text into structured fields using the existing parser."""
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        logger.info("Parsing resume via Azure OpenAI helper …")
        parser = ResumeParser()
        runs = max(1, int(os.getenv("RESUME_PARSER_RUNS", "1")))
        parsed_runs: List[Dict[str, Any]] = []
        username = None
        for _ in range(runs):
            username, parsed_resume = await parser.resume_parse(pdf_text)
            parsed_runs.append(parsed_resume)
        merged = _merge_parsed_runs(parsed_runs)
        logger.info("Parsed resume for user: %s", username or merged.get("name") or "(unknown)")
        merged["username"] = username or merged.get("name")
        _normalise_parsed_skills(merged)
        if not merged.get("technical_skills"):
            logger.warning("Parsed resume contains no technical skills; downstream matching may be sparse.")
        if not merged.get("soft_skills"):
            logger.warning(
                "Parsed resume contains no soft skills; consider reviewing the parsing prompt or cached result."
            )
        return merged

    logger.info("Parsing resume via standard OpenAI helper …")
    runs = max(1, int(os.getenv("RESUME_PARSER_RUNS", "1")))
    parsed_runs: List[Dict[str, Any]] = []
    for _ in range(runs):
        parsed_runs.append(await parse_resume_with_openai(pdf_text))
    merged = _merge_parsed_runs(parsed_runs)
    username = merged.get("name")
    merged["username"] = username
    logger.info("Parsed resume for user: %s", username or "(unknown)")
    _normalise_parsed_skills(merged)
    if not merged.get("technical_skills"):
        logger.warning("Parsed resume contains no technical skills; downstream matching may be sparse.")
    if not merged.get("soft_skills"):
        logger.warning(
            "Parsed resume contains no soft skills; consider reviewing the parsing prompt or cached result."
        )
    return merged


async def parse_resume_with_openai(pdf_text: str) -> Dict:
    """Fallback resume parser that uses OpenAI's public API instead of Azure."""
    client = AsyncOpenAI()
    model = os.getenv("OPENAI_RESUME_MODEL", "gpt-5-mini-2025-08-07")

    system_prompt = (
        "You are a resume parser. Extract relevant details from resumes into a comprehensive and "
        "structured JSON format. Ensure no information is missed. If any field is too long, summarise "
        "the information appropriately. Provide detailed output for each section such as name, phone, "
        "email, education, experience, technical skills, soft skills, and LinkedIn. When listing skills, "
        "prefer canonical O*NET-style names (e.g., 'Critical Thinking', 'Complex Problem Solving'). "
        "Return skills as title-cased noun phrases. Do not invent synonyms; if a skill cannot be mapped "
        "confidently to a canonical label, omit it instead of paraphrasing."
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_resume_data",
                "description": "Extracts detailed resume data and summarizes if necessary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "phone": {"type": "string"},
                        "email": {"type": "string"},
                        "education": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "degree": {"type": "string"},
                                    "institution": {"type": "string"},
                                    "year": {"type": "string"},
                                },
                                "additionalProperties": False,
                                "required": ["degree", "institution", "year"],
                            },
                        },
                        "experience": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "job_title": {"type": "string"},
                                    "company": {"type": "string"},
                                    "years": {"type": "string"},
                                    "details": {"type": "string"},
                                },
                                "additionalProperties": False,
                                "required": ["job_title", "company", "years", "details"],
                            },
                        },
                        "technical_skills": {"type": "array", "items": {"type": "string"}},
                        "soft_skills": {"type": "array", "items": {"type": "string"}},
                        "Top_10_Skills": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "skill": {"type": "string"},
                                    "proficiency": {
                                        "type": "string",
                                        "enum": ["beginner", "intermediate", "advanced", "expert"],
                                    },
                                    "years": {"type": "string"},
                                },
                                "additionalProperties": False,
                                "required": ["skill", "proficiency", "years"],
                            },
                            "maxItems": 10,
                        },
                    },
                    "required": [
                        "name",
                        "phone",
                        "email",
                        "education",
                        "experience",
                        "technical_skills",
                        "soft_skills",
                        "Top_10_Skills",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pdf_text},
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "extract_resume_data"}},
    )

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        arguments = tool_calls[0].function.arguments
        parsed = json.loads(arguments)
        # Ensure list fields exist even if omitted
        parsed.setdefault("education", [])
        parsed.setdefault("experience", [])
        parsed.setdefault("technical_skills", [])
        parsed.setdefault("soft_skills", [])
        parsed.setdefault("Top_10_Skills", [])
        return parsed

    raise RuntimeError("OpenAI resume parser did not return structured data.")


async def generate_skill_analysis(parsed_resume: Dict, qna: Sequence[Dict]) -> Dict[str, str]:
    """Produce the legacy skill analysis narrative and supporting metadata."""
    logger.info("Generating SkillAnalysis narrative …")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if azure_endpoint:
        career_path_client = CareerPath(kind="suggested")
        suggested_path = await career_path_client.suggest_path(parsed_resume)

        skill_analysis_client = SkillAnalysis()
        username = parsed_resume.get("username") or "Candidate"
        narrative = await skill_analysis_client.skill_analysis(
            suggested_path,
            username=username,
            qna=qna,
            parsed_resume=parsed_resume,
        )
        return {"narrative": narrative, "career_path": suggested_path}

    logger.info("Azure credentials missing; using OpenAI fallback for career path and skill analysis …")
    suggested_path = await suggest_career_path_with_openai(parsed_resume)
    username = parsed_resume.get("username") or suggested_path.get("current_position", "Candidate")
    narrative = await skill_analysis_with_openai(parsed_resume, suggested_path, username, qna)
    return {"narrative": narrative, "career_path": suggested_path}


async def suggest_career_path_with_openai(parsed_resume: Dict) -> Dict:
    """Fallback generator for the career path suggestions."""
    client = AsyncOpenAI()
    model = os.getenv("OPENAI_CAREER_MODEL", "gpt-5-mini-2025-08-07")

    system_prompt = (
        "You are an expert in career progression analysis. Evaluate the candidate's resume and identify "
        "their current position plus up to three next positions for the first and second jumps."
    )

    user_context = json.dumps(
        {
            "experience": parsed_resume.get("experience"),
            "education": parsed_resume.get("education"),
            "soft_skills": _flatten_skill_terms(parsed_resume.get("soft_skills")),
            "technical_skills": _flatten_skill_terms(parsed_resume.get("technical_skills")),
        },
        indent=2,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "career_path_details",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "current_position": {"type": "string"},
                        "next_positions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 3,
                        },
                        "second_jump_positions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 3,
                        },
                    },
                    "required": [
                        "current_position",
                        "next_positions",
                        "second_jump_positions",
                    ],
                    "additionalProperties": False,
                },
            },
        }
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context},
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "career_path_details"}},
    )

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        return json.loads(tool_calls[0].function.arguments)

    raise RuntimeError("OpenAI career path helper did not return structured data.")


async def skill_analysis_with_openai(
    parsed_resume: Dict,
    career_path: Dict,
    username: str,
    qna: Sequence[Dict],
) -> str:
    """Generate the skill analysis narrative using OpenAI instead of Azure."""
    client = AsyncOpenAI()
    model = os.getenv("OPENAI_SKILL_MODEL", "gpt-5-mini-2025-08-07")

    system_prompt = (
        "You are a career coach specialising in skill assessment. Ground your analysis in the provided "
        "skills and career trajectory. Identify strengths, gaps, and actionable recommendations."
    )

    payload = {
        "career_path": career_path,
        "resume": {
            "experience": parsed_resume.get("experience"),
            "education": parsed_resume.get("education"),
            "soft_skills": _flatten_skill_terms(parsed_resume.get("soft_skills")),
            "technical_skills": _flatten_skill_terms(parsed_resume.get("technical_skills")),
        },
        "qna": list(qna),
        "username": username,
    }

    user_content = (
        "Using the JSON context, write a personalised skill analysis for the candidate. "
        "Address them directly, highlight aligned skills, gaps, recommendations, and end with encouragement.\n\n"
        f"Context:\n{json.dumps(payload, indent=2)}"
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    return response.choices[0].message.content or ""


def _serialise_skill_record(record: SkillRecord) -> Dict[str, Optional[float]]:
    """Convert a SkillRecord dataclass into a plain dictionary."""
    return asdict(record)


def _serialise_rag_profiles(raw_profiles: Sequence[Dict]) -> List[Dict]:
    """Normalise SkillGraphRAG structures for JSON dumps."""
    serialised: List[Dict] = []
    for profile in raw_profiles:
        serialised.append(
            {
                "requested_role": profile.get("requested_role"),
                "occupation_match": profile.get("occupation_match"),
                "alternate_matches": profile.get("alternate_matches", []),
                "coverage_stats": profile.get("coverage_stats"),
                "occupation_context": profile.get("occupation_context"),
                "skills_covered": [
                    {
                        "skill": _serialise_skill_record(item["skill"]),
                        "score": item.get("score"),
                        "matched_user_skill": item.get("matched_user_skill"),
                    }
                    for item in profile.get("skills_covered", [])
                ],
                "skill_gaps": [
                    {
                        "skill": _serialise_skill_record(item["skill"]),
                        "score": item.get("score"),
                        "matched_user_skill": item.get("matched_user_skill"),
                    }
                    for item in profile.get("skill_gaps", [])
                ],
                "all_skills": [
                    _serialise_skill_record(record) for record in profile.get("all_skills", [])
                ],
                "skills_covered_table": profile.get("skills_covered_table", []),
                "skill_gaps_table": profile.get("skill_gaps_table", []),
                "all_skills_table": profile.get("all_skills_table", []),
            }
        )
    return serialised


def _choose_target_roles(user_specified: Optional[Sequence[str]], suggested_path: Dict) -> List[str]:
    """Determine which roles to evaluate in the SkillGraph pipeline."""
    if user_specified:
        roles = [role for role in user_specified if role]
        if roles:
            return roles
    next_positions = suggested_path.get("next_positions") or []
    if next_positions:
        return list(next_positions)
    current = suggested_path.get("current_position")
    return [current] if current else []


async def generate_rag_profile(
    parsed_resume: Dict,
    suggested_path: Dict,
    *,
    target_roles: Optional[Sequence[str]],
    qna: Sequence[Dict],
    max_skills: int,
    include_summary: bool,
) -> Dict:
    """Run the SkillGraph RAG pipeline and return serialised outputs."""
    roles = _choose_target_roles(target_roles, suggested_path)
    if not roles:
        raise ValueError("No target roles available for SkillGraph evaluation.")

    logger.info("Running SkillGraph RAG against roles: %s", ", ".join(roles))
    rag = SkillGraphRAG()
    profile = rag.generate_skill_profile(
        parsed_resume=parsed_resume,
        target_roles=roles,
        qna=qna,
        max_skills_per_role=max_skills,
        summarise=include_summary,
    )
    profile["profiles"] = _serialise_rag_profiles(profile.get("profiles", []))
    return profile


async def orchestrate(
    pdf_path: Path,
    *,
    target_roles: Optional[Sequence[str]] = None,
    max_skills: int = 25,
    include_summary: bool = True,
) -> Dict:
    """Main orchestration routine for running both profile generators."""
    pdf_text = extract_pdf_text(pdf_path)
    parsed_resume = await parse_resume(pdf_text)
    qna: List[Dict] = []  # Extend with interview answers if available

    legacy_result = await generate_skill_analysis(parsed_resume, qna)
    rag_result = await generate_rag_profile(
        parsed_resume,
        legacy_result["career_path"],
        target_roles=target_roles,
        qna=qna,
        max_skills=max_skills,
        include_summary=include_summary,
    )

    return {
        "username": parsed_resume.get("username"),
        "legacy": legacy_result,
        "rag": rag_result,
    }


def _dump_output(result: Dict, *, output_path: Optional[Path]) -> None:
    """Print results to stdout and optionally write them to disk."""
    narrative = result["legacy"]["narrative"]
    print("\n=== SkillAnalysis Narrative ===\n")
    print(narrative)

    print("\n=== SkillGraph RAG Summary ===\n")
    print(result["rag"].get("analysis") or "(summary disabled)")

    print("\n=== SkillGraph RAG Structured Profiles ===\n")
    print(json.dumps(result["rag"]["profiles"], indent=2))

    for profile in result["rag"].get("profiles", []):
        coverage = profile.get("coverage_stats") or {}
        role = profile.get("requested_role") or "(role)"
        if coverage:
            pct = coverage.get("coverage_pct", 0.0) * 100
            covered = coverage.get("covered_count", 0)
            total = coverage.get("total_count", 0)
            print(
                f"- Coverage for {role}: {pct:.1f}% ({covered}/{total} skills; "
                f"importance {coverage.get('covered_importance', 0)}/{coverage.get('total_importance', 0)})"
            )
            split = coverage.get("split") or {}
            official = split.get("official", {})
            vector = split.get("vector", {})
            if official.get("total_count"):
                official_pct = (coverage.get("official_pct") or 0.0) * 100
                print(
                    "  • Official coverage: {pct:.1f}% ({count}/{total})".format(
                        pct=official_pct,
                        count=official.get("count", 0),
                        total=official.get("total_count", 0),
                    )
                )
            if vector.get("count"):
                vector_pct = (coverage.get("vector_pct") or 0.0) * 100
                print(
                    "  • Vector-assisted coverage: {pct:.1f}% ({count}/{total})".format(
                        pct=vector_pct,
                        count=vector.get("count", 0),
                        total=vector.get("total_count", 0),
                    )
                )
                print(
                    "    importance {importance}/{total}".format(
                        importance=vector.get("importance", 0),
                        total=vector.get("total_importance", 0),
                    )
                )
        top_rows = profile.get("skills_covered_table", [])[:5]
        if top_rows:
            print(f"  Top validated skills for {role}:")
            for row in top_rows:
                print(
                    "    - {name} (importance {importance}, similarity {similarity}, source {source}, resume '{resume}')".format(
                        name=row["name"],
                        importance=row["importance"],
                        similarity=row["similarity"],
                        source=row.get("match_source", "graph"),
                        resume=row["resume_term"] or "-",
                    )
                )

    if output_path:
        payload = {
            "username": result.get("username"),
            "legacy": result["legacy"],
            "rag": result["rag"],
        }
        output_path.write_text(json.dumps(payload, indent=2))
        logger.info("Saved combined output to %s", output_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test both skill profile generators on a resume PDF.")
    parser.add_argument("pdf", type=Path, help="Path to the resume PDF")
    parser.add_argument(
        "--role",
        dest="roles",
        action="append",
        help="Target role title to evaluate (repeatable). Defaults to suggested next roles.",
    )
    parser.add_argument(
        "--max-skills",
        type=int,
        default=25,
        help="Maximum number of graph skills to inspect per role (default: 25)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the LLM-generated summary from the SkillGraph pipeline.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path to write a JSON dump of all results.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    pdf_path: Path = args.pdf

    if not pdf_path.exists():
        raise SystemExit(f"Resume PDF not found: {pdf_path}")

    _load_env()

    result = asyncio.run(
        orchestrate(
            pdf_path,
            target_roles=args.roles,
            max_skills=args.max_skills,
            include_summary=not args.no_summary,
        )
    )

    _dump_output(result, output_path=args.out)


if __name__ == "__main__":  # pragma: no cover - manual execution entrypoint
    main()
