"""SkillGraph RAG helpers for grounding skill profiles on the Neo4j O*NET graph.

This module exposes a small retrieval-augmented pipeline that:
1. Connects to the deployed Neo4j knowledge graph built from the O*NET delivery.
2. Retrieves occupation-specific skills from the graph.
3. Uses OpenAI embeddings to align a user's stated skills with the graph descriptors.
4. Summarises the aligned vs. missing skills via an LLM to produce a grounded profile.

It is designed so backend components (for example those under
``careerCoach/backend/career_coach``) can import and reuse the same retrieval logic
without having to know any Neo4j details.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv

try:  # Optional dependency guard for clearer import errors downstream
    from langchain_neo4j import Neo4jGraph
except ImportError as exc:  # pragma: no cover - provides actionable hint if package missing
    raise ImportError(
        "langchain-neo4j is required. Install it with `pip install langchain-neo4j`."
    ) from exc

try:  # Same idea for langchain OpenAI helpers
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError as exc:  # pragma: no cover - provides actionable hint if package missing
    raise ImportError(
        "langchain-openai is required. Install it with `pip install langchain-openai`."
    ) from exc

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _load_environment() -> None:
    """Load Neo4j/OpenAI environment variables from `.env` if present."""
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Fallback to default behaviour (load from current working directory / process env)
        load_dotenv()


@dataclass
class SkillRecord:
    """Lightweight container for skill descriptors returned from the graph."""

    element_id: str
    name: str
    description: str
    importance: float
    level: float

    @property
    def text(self) -> str:
        """Canonical text representation used for embedding comparisons."""
        if self.description:
            return f"{self.name}: {self.description}"
        return self.name


@dataclass
class OccupationSuggestion:
    """Represents an occupation surfaced via graph-based similarity."""

    code: str
    title: str
    overlap_score: float
    shared_skill_count: int


class SkillGraphClient:
    """Wrapper around Neo4j queries used by the RAG pipeline."""

    CODE_PATTERN = re.compile(r"^\d{2}-\d{4}\.\d{2}$")
    _local_loaded: bool = False
    _local_content_map: Dict[str, Dict[str, str]] = {}
    _local_skill_records: Dict[str, List[SkillRecord]] = {}
    _local_skill_importance: Dict[str, Dict[str, float]] = {}
    _local_activity_records: Dict[str, List[Dict[str, Any]]] = {}
    _local_occupation_map: Dict[str, Dict[str, str]] = {}

    def __init__(
        self,
        graph: Optional[Neo4jGraph] = None,
        *,
        database: Optional[str] = None,
    ) -> None:
        _load_environment()

        if graph is not None:
            self.graph = graph
        else:
            uri = os.getenv("NEO4J_URI")
            username = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")
            database = database or os.getenv("NEO4J_DATABASE", "neo4j")

            if not all([uri, username, password]):
                logger.warning("Neo4j credentials missing; SkillGraphClient will use local CSV fallback.")
                self.graph = None
            else:
                try:
                    self.graph = Neo4jGraph(url=uri, username=username, password=password, database=database)
                except Exception as exc:
                    logger.warning("Failed to connect to Neo4j (%s); falling back to local CSV data.", exc)
                    self.graph = None

    # ---------------------------------------------------------------------
    # Occupation lookups
    # ---------------------------------------------------------------------
    def resolve_occupation(self, title_or_code: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Return candidate occupations for a user supplied title or O*NET code."""
        candidate_title = title_or_code.strip()
        params: Dict[str, Any] = {"title": candidate_title, "limit": limit * 5}

        if self.CODE_PATTERN.match(candidate_title):
            logger.debug("Resolving occupation by exact code: %s", candidate_title)
            if self.graph:
                query = (
                    "MATCH (o:Occupation {code: $code})\n"
                    "RETURN o.code AS code, o.title AS title, o.description AS description"
                )
                try:
                    result = self.graph.query(query, {"code": candidate_title})
                    if result:
                        return result
                except Exception as exc:
                    logger.debug("Neo4j resolve by code failed: %s", exc)
            self._ensure_local_cache()
            record = self._local_occupation_map.get(candidate_title)
            if record:
                return [{"code": candidate_title, **record, "score": 1.0}]
            return []

        records: List[Dict[str, Any]] = []
        if self.graph:
            query = (
                "MATCH (o:Occupation)\n"
                "WHERE toLower(o.title) CONTAINS toLower($title)\n"
                "   OR toLower($title) CONTAINS toLower(o.title)\n"
                "RETURN o.code AS code, o.title AS title, o.description AS description\n"
                "LIMIT toInteger($limit)"
            )
            try:
                records = self.graph.query(query, params)
                if not records:
                    logger.debug("No direct substring match for '%s'; broadening search via all occupations.", candidate_title)
                    broad_query = (
                        "MATCH (o:Occupation)\n"
                        "RETURN o.code AS code, o.title AS title, o.description AS description\n"
                        "LIMIT toInteger($limit)"
                    )
                    records = self.graph.query(broad_query, {"limit": limit * 20})
            except Exception as exc:
                logger.debug("Neo4j occupation search failed: %s", exc)
                records = []

        if not records:
            self._ensure_local_cache()
            records = self._local_resolve(candidate_title, limit)

        scored: List[Dict[str, Any]] = []
        for record in records:
            title = record.get("title", "")
            ratio = SequenceMatcher(None, candidate_title.lower(), title.lower()).ratio() if title else 0.0
            scored.append({**record, "score": ratio})

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]

    # ---------------------------------------------------------------------
    # Skill retrieval
    # ---------------------------------------------------------------------
    def occupation_skills(self, occupation_code: str, *, limit: Optional[int] = None) -> List[SkillRecord]:
        """Fetch skill descriptors tied to an occupation ordered by importance."""
        if self.graph:
            params = {"code": occupation_code}
            cypher = (
                "MATCH (o:Occupation {code: $code})-[r:REQUIRES_SKILL]->(skill:ContentElement)\n"
                "WHERE 'Skill' IN labels(skill)\n"
                "RETURN skill.element_id AS element_id,\n"
                "       skill.name AS name,\n"
                "       coalesce(skill.description, '') AS description,\n"
                "       toFloat(coalesce(r.importance, 0)) AS importance,\n"
                "       toFloat(coalesce(r.level, 0)) AS level\n"
                "ORDER BY importance DESC"
            )

            if limit:
                cypher += "\nLIMIT toInteger($limit)"
                params["limit"] = limit

            try:
                records = self.graph.query(cypher, params)
                if records:
                    return [
                        SkillRecord(
                            element_id=record["element_id"],
                            name=record["name"],
                            description=record["description"],
                            importance=float(record["importance"] or 0.0),
                            level=float(record["level"] or 0.0),
                        )
                        for record in records
                    ]
            except Exception as exc:
                logger.debug("Neo4j occupation skills query failed: %s", exc)

        self._ensure_local_cache()
        records = list(self._local_skill_records.get(occupation_code, []))
        if limit:
            records = records[:limit]
        return records

    def related_occupations_by_skill(
        self,
        occupation_code: str,
        *,
        limit: int = 10,
    ) -> List[OccupationSuggestion]:
        """Return occupations that share overlapping skills with the given occupation."""

        if self.graph:
            cypher = (
                "MATCH (current:Occupation {code: $code})-[r:REQUIRES_SKILL]->(skill:ContentElement)\n"
                "MATCH (other:Occupation)-[r2:REQUIRES_SKILL]->(skill)\n"
                "WHERE other <> current\n"
                "WITH other, sum(coalesce(r.importance, 0) * coalesce(r2.importance, 0)) AS weighted_overlap,\n"
                "     count(skill) AS shared_skill_count\n"
                "RETURN other.code AS code,\n"
                "       other.title AS title,\n"
                "       weighted_overlap AS overlap_score,\n"
                "       shared_skill_count AS shared_skill_count\n"
                "ORDER BY overlap_score DESC, shared_skill_count DESC, title ASC\n"
                "LIMIT toInteger($limit)"
            )

            try:
                records = self.graph.query(cypher, {"code": occupation_code, "limit": limit})
                if records:
                    suggestions: List[OccupationSuggestion] = []
                    for record in records:
                        suggestions.append(
                            OccupationSuggestion(
                                code=record["code"],
                                title=record["title"],
                                overlap_score=float(record["overlap_score"] or 0.0),
                                shared_skill_count=int(record["shared_skill_count"] or 0),
                            )
                        )
                    return suggestions
            except Exception as exc:
                logger.debug("Neo4j related occupation query failed: %s", exc)

        self._ensure_local_cache()
        base = self._local_skill_importance.get(occupation_code)
        if not base:
            return []

        suggestions: List[OccupationSuggestion] = []
        for other_code, importance_map in self._local_skill_importance.items():
            if other_code == occupation_code:
                continue
            overlap = 0.0
            shared = 0
            for element_id, importance in base.items():
                other_imp = importance_map.get(element_id)
                if other_imp is not None:
                    overlap += importance * other_imp
                    shared += 1
            if shared:
                occupation = self._local_occupation_map.get(other_code, {})
                suggestions.append(
                    OccupationSuggestion(
                        code=other_code,
                        title=occupation.get("title", other_code),
                        overlap_score=overlap,
                        shared_skill_count=shared,
                    )
                )

        suggestions.sort(key=lambda item: (item.overlap_score, item.shared_skill_count), reverse=True)
        return suggestions[:limit]

    def occupation_context(
        self,
        occupation_code: str,
        *,
        top_skills: int = 15,
        top_activities: int = 8,
        related_limit: int = 8,
    ) -> Dict[str, Any]:
        """Return a structured snapshot of an occupation for grounding LLM responses."""

        context: Dict[str, Any] = {
            "occupation": None,
            "skills": [],
            "activities": [],
            "related_occupations": [],
        }

        details = []
        if self.graph:
            details_query = (
                "MATCH (o:Occupation {code: $code})\n"
                "RETURN o.code AS code, o.title AS title, o.description AS description"
            )
            try:
                details = self.graph.query(details_query, {"code": occupation_code})
            except Exception as exc:
                logger.debug("Neo4j occupation context lookup failed: %s", exc)

        if not details:
            self._ensure_local_cache()
            local_details = self._local_occupation_map.get(occupation_code)
            if local_details:
                context["occupation"] = {"code": occupation_code, **local_details}
        else:
            context["occupation"] = details[0]

        skills = []
        if self.graph:
            skills_query = (
                "MATCH (o:Occupation {code: $code})-[r:REQUIRES_SKILL]->(skill:ContentElement)\n"
                "WHERE 'Skill' IN labels(skill)\n"
                "RETURN skill.element_id AS element_id, skill.name AS name,\n"
                "       coalesce(skill.description, '') AS description,\n"
                "       toFloat(coalesce(r.importance, 0)) AS importance,\n"
                "       toFloat(coalesce(r.level, 0)) AS level\n"
                "ORDER BY importance DESC\n"
                "LIMIT toInteger($limit)"
            )
            try:
                skills = self.graph.query(
                    skills_query,
                    {"code": occupation_code, "limit": top_skills},
                )
            except Exception as exc:
                logger.debug("Neo4j occupation skills context failed: %s", exc)
        if not skills:
            self._ensure_local_cache()
            skills = [
                {
                    "element_id": record.element_id,
                    "name": record.name,
                    "description": record.description,
                    "importance": record.importance,
                    "level": record.level,
                }
                for record in self._local_skill_records.get(occupation_code, [])[:top_skills]
            ]
        context["skills"] = skills

        activities = []
        if self.graph:
            activities_query = (
                "MATCH (o:Occupation {code: $code})-[r:INVOLVES_ACTIVITY]->(activity:ContentElement)\n"
                "RETURN activity.element_id AS element_id, activity.name AS name,\n"
                "       coalesce(activity.description, '') AS description,\n"
                "       toFloat(coalesce(r.importance, 0)) AS importance\n"
                "ORDER BY importance DESC\n"
                "LIMIT toInteger($limit)"
            )
            try:
                activities = self.graph.query(
                    activities_query,
                    {"code": occupation_code, "limit": top_activities},
                )
            except Exception as exc:
                logger.debug("Neo4j activities context failed: %s", exc)
        if not activities:
            self._ensure_local_cache()
            activities = self._local_activity_records.get(occupation_code, [])[:top_activities]
        context["activities"] = activities

        context["related_occupations"] = [
            {
                "code": suggestion.code,
                "title": suggestion.title,
                "overlap_score": suggestion.overlap_score,
                "shared_skill_count": suggestion.shared_skill_count,
            }
            for suggestion in self.related_occupations_by_skill(occupation_code, limit=related_limit)
        ]

        return context

    # ------------------------------------------------------------------
    # Local CSV helpers
    # ------------------------------------------------------------------

    def _ensure_local_cache(self) -> None:
        cls = type(self)
        if cls._local_loaded:
            return

        base_dir = Path(__file__).resolve().parent / "neo4j_csv"
        nodes_dir = base_dir / "nodes"
        rel_dir = base_dir / "relationships"
        occupations_path = nodes_dir / "occupations.csv"
        content_path = nodes_dir / "content_elements.csv"
        occ_skill_path = rel_dir / "occupation_requires_skill.csv"
        occ_activity_path = rel_dir / "occupation_involves_activity.csv"

        if not nodes_dir.exists() or not rel_dir.exists():
            logger.debug("Local Neo4j CSV directory missing at %s", base_dir)
            cls._local_loaded = True
            return

        try:
            if occupations_path.exists():
                with occupations_path.open(encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        code = row.get("code:ID(Occupation)")
                        if not code:
                            continue
                        cls._local_occupation_map[code] = {
                            "title": row.get("title", ""),
                            "description": row.get("description", ""),
                        }

            if content_path.exists():
                with content_path.open(encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        element_id = row.get("element_id:ID(ContentElement)")
                        if not element_id:
                            continue
                        cls._local_content_map[element_id] = {
                            "name": row.get("name", ""),
                            "description": row.get("description", ""),
                            "labels": row.get("labels:LABEL", ""),
                        }

            if occ_skill_path.exists():
                with occ_skill_path.open(encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        occ_code = row.get("start:START_ID(Occupation)")
                        elem_id = row.get("end:END_ID(ContentElement)")
                        if not occ_code or not elem_id:
                            continue
                        content = cls._local_content_map.get(elem_id, {})
                        labels = content.get("labels", "")
                        if "Skill" not in labels:
                            continue
                        try:
                            importance = float(row.get("importance", "") or 0.0)
                        except ValueError:
                            importance = 0.0
                        try:
                            level = float(row.get("level", "") or 0.0)
                        except ValueError:
                            level = 0.0
                        record = SkillRecord(
                            element_id=elem_id,
                            name=content.get("name", elem_id),
                            description=content.get("description", ""),
                            importance=importance,
                            level=level,
                        )
                        cls._local_skill_records.setdefault(occ_code, []).append(record)
                        importance_map = cls._local_skill_importance.setdefault(occ_code, {})
                        importance_map[elem_id] = importance

                for records in cls._local_skill_records.values():
                    records.sort(key=lambda rec: rec.importance, reverse=True)

            if occ_activity_path.exists():
                with occ_activity_path.open(encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        occ_code = row.get("start:START_ID(Occupation)")
                        elem_id = row.get("end:END_ID(ContentElement)")
                        if not occ_code or not elem_id:
                            continue
                        content = cls._local_content_map.get(elem_id, {})
                        try:
                            importance = float(row.get("importance", "") or 0.0)
                        except ValueError:
                            importance = 0.0
                        record = {
                            "element_id": elem_id,
                            "name": content.get("name", elem_id),
                            "description": content.get("description", ""),
                            "importance": importance,
                        }
                        cls._local_activity_records.setdefault(occ_code, []).append(record)

                for records in cls._local_activity_records.values():
                    records.sort(key=lambda rec: rec.get("importance", 0.0), reverse=True)

        except Exception as exc:
            logger.debug("Failed to load local SkillGraph CSV cache: %s", exc)
        finally:
            cls._local_loaded = True

    def _local_resolve(self, candidate_title: str, limit: int) -> List[Dict[str, Any]]:
        self._ensure_local_cache()
        results: List[Dict[str, Any]] = []
        candidate_lower = candidate_title.lower()
        for code, data in cls._local_occupation_map.items():
            title = data.get("title", "")
            if not title:
                continue
            if candidate_lower in title.lower() or title.lower() in candidate_lower:
                results.append({"code": code, **data})
        if results:
            return results[:limit]

        # fallback to fuzzy match by ratio across all occupations
        scored: List[Tuple[float, str, Dict[str, str]]] = []
        for code, data in cls._local_occupation_map.items():
            title = data.get("title", "")
            if not title:
                continue
            ratio = SequenceMatcher(None, candidate_lower, title.lower()).ratio()
            scored.append((ratio, code, data))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {"code": code, **data}
            for ratio, code, data in scored[: min(limit * 5, len(scored))]
        ]


class SkillMatcher:
    """Embeds and aligns free-form user skills against graph descriptors."""

    def __init__(self, embedder: Optional[OpenAIEmbeddings] = None) -> None:
        _load_environment()
        self.embedder = embedder or OpenAIEmbeddings(model="text-embedding-3-small")
        self._skill_embedding_cache: Dict[str, np.ndarray] = {}

    def match(
        self,
        user_skill_texts: Sequence[str],
        graph_skills: Sequence[SkillRecord],
        *,
        similarity_threshold: float = 0.72,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return skills the user already covers and those likely missing."""
        if not graph_skills:
            return {"matched": [], "missing": []}

        if not user_skill_texts:
            missing_payload = [
                {
                    "skill": skill,
                    "score": None,
                    "matched_user_skill": None,
                }
                for skill in graph_skills
            ]
            return {"matched": [], "missing": missing_payload}

        # Deduplicate user skill phrases while keeping order
        seen = set()
        normalised_user_skills = []
        for skill_text in user_skill_texts:
            key = skill_text.strip().lower()
            if key and key not in seen:
                normalised_user_skills.append(skill_text.strip())
                seen.add(key)

        user_vectors = self.embedder.embed_documents(normalised_user_skills)
        user_matrix = np.asarray(user_vectors, dtype=np.float32)

        skill_matrix = []
        for skill in graph_skills:
            vector = self._cached_skill_embedding(skill)
            skill_matrix.append(vector)
        skill_matrix_np = np.asarray(skill_matrix, dtype=np.float32)

        user_norms = np.linalg.norm(user_matrix, axis=1, keepdims=True)
        skill_norms = np.linalg.norm(skill_matrix_np, axis=1, keepdims=True)
        user_norms[user_norms == 0] = 1e-12
        skill_norms[skill_norms == 0] = 1e-12
        similarity_matrix = (user_matrix / user_norms) @ (skill_matrix_np / skill_norms).T

        best_user_indices = np.argmax(similarity_matrix, axis=0)
        best_scores = similarity_matrix[best_user_indices, np.arange(similarity_matrix.shape[1])]

        matched: List[Dict[str, Any]] = []
        missing: List[Dict[str, Any]] = []
        for idx, skill in enumerate(graph_skills):
            score = float(best_scores[idx])
            best_user_skill = normalised_user_skills[int(best_user_indices[idx])]
            payload = {
                "skill": skill,
                "score": round(score, 3),
                "matched_user_skill": best_user_skill,
            }
            if score >= similarity_threshold:
                matched.append(payload)
            else:
                payload["matched_user_skill"] = None
                missing.append(payload)

        matched.sort(key=lambda item: (item["skill"].importance, item["score"]), reverse=True)
        missing.sort(key=lambda item: item["skill"].importance, reverse=True)
        return {"matched": matched, "missing": missing}

    # ------------------------------------------------------------------
    def _cached_skill_embedding(self, skill: SkillRecord) -> np.ndarray:
        cached = self._skill_embedding_cache.get(skill.element_id)
        if cached is not None:
            return cached
        embedding = self.embedder.embed_query(skill.text)
        vector = np.asarray(embedding, dtype=np.float32)
        self._skill_embedding_cache[skill.element_id] = vector
        return vector


class SkillGraphCareerPlanner:
    """Suggests career progressions by leveraging the SkillGraph data."""

    def __init__(self, client: Optional[SkillGraphClient] = None) -> None:
        self.client = client or SkillGraphClient()

    def suggest_from_title(
        self,
        current_title: str,
        *,
        next_limit: int = 5,
        second_limit: int = 5,
        candidate_limit: int = 8,
    ) -> Optional[Dict[str, Any]]:
        """Return a career path suggestion anchored to the nearest occupation."""

        if not current_title:
            return None

        occupation_matches = self.client.resolve_occupation(current_title, limit=candidate_limit)
        if not occupation_matches:
            return None

        primary = occupation_matches[0]
        first_jump = self._suggest_successors(primary["code"], limit=max(next_limit * 2, next_limit))
        next_positions = self._dedupe_titles([item.title for item in first_jump])[:next_limit]

        second_jump: List[str] = []
        for suggestion in first_jump[: min(len(first_jump), candidate_limit)]:
            follow_on = self._suggest_successors(suggestion.code, limit=second_limit * 2)
            second_jump.extend(item.title for item in follow_on)
            if len(second_jump) >= second_limit * 2:
                break
        second_positions = self._dedupe_titles(second_jump)[:second_limit]

        return {
            "current_position": primary["title"],
            "current_code": primary["code"],
            "next_positions": next_positions,
            "second_jump_positions": second_positions,
            "alternate_matches": occupation_matches[1:],
        }

    def _suggest_successors(self, occupation_code: str, *, limit: int) -> List[OccupationSuggestion]:
        try:
            return self.client.related_occupations_by_skill(occupation_code, limit=limit)
        except Exception as exc:  # pragma: no cover - logging without logger dependency
            logger.debug("SkillGraph successor lookup failed: %s", exc)
            return []

    @staticmethod
    def _dedupe_titles(titles: Iterable[str]) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []
        for title in titles:
            clean = title.strip()
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(clean)
        return ordered


class SkillGraphRAG:
    """High-level orchestrator that produces grounded skill profiles."""

    def __init__(
        self,
        *,
        graph_client: Optional[SkillGraphClient] = None,
        matcher: Optional[SkillMatcher] = None,
        llm: Optional[ChatOpenAI] = None,
        similarity_threshold: float = 0.72,
    ) -> None:
        _load_environment()
        self.graph_client = graph_client or SkillGraphClient()
        self.matcher = matcher or SkillMatcher()
        self.similarity_threshold = similarity_threshold

        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    def generate_skill_profile(
        self,
        parsed_resume: Dict[str, Any],
        target_roles: Sequence[str],
        *,
        qna: Optional[Sequence[Dict[str, str]]] = None,
        max_skills_per_role: int = 25,
        summarise: bool = True,
    ) -> Dict[str, Any]:
        """Build a structured, graph-grounded skill profile for the user."""
        technical_skills = parsed_resume.get("technical_skills", [])
        soft_skills = parsed_resume.get("soft_skills", [])
        user_skill_texts = [*technical_skills, *soft_skills]

        role_profiles: List[Dict[str, Any]] = []
        for requested_role in target_roles:
            occupation_matches = self.graph_client.resolve_occupation(requested_role, limit=3)
            if not occupation_matches:
                role_profiles.append(
                    {
                        "requested_role": requested_role,
                        "occupation_match": None,
                        "skills_covered": [],
                        "skill_gaps": [],
                        "all_skills": [],
                    }
                )
                continue

            best_match = occupation_matches[0]
            occ_code = best_match["code"]
            graph_skills = self.graph_client.occupation_skills(occ_code, limit=max_skills_per_role)
            matching = self.matcher.match(user_skill_texts, graph_skills, similarity_threshold=self.similarity_threshold)
            total_importance = sum(skill.importance for skill in graph_skills if skill.importance)
            covered_importance = sum(item["skill"].importance for item in matching["matched"] if item["skill"].importance)
            coverage_pct = covered_importance / total_importance if total_importance else 0.0
            coverage_stats = {
                "coverage_pct": round(coverage_pct, 4),
                "covered_importance": round(covered_importance, 3),
                "total_importance": round(total_importance, 3),
                "covered_count": len(matching["matched"]),
                "total_count": len(graph_skills),
            }

            skills_covered_table = [
                {
                    "element_id": item["skill"].element_id,
                    "name": item["skill"].name,
                    "importance": round(item["skill"].importance, 3),
                    "level": round(item["skill"].level, 3),
                    "similarity": item["score"],
                    "resume_term": item["matched_user_skill"],
                    "description": item["skill"].description,
                }
                for item in matching["matched"]
            ]
            skill_gaps_table = [
                {
                    "element_id": item["skill"].element_id,
                    "name": item["skill"].name,
                    "importance": round(item["skill"].importance, 3),
                    "level": round(item["skill"].level, 3),
                    "description": item["skill"].description,
                }
                for item in matching["missing"]
            ]
            all_skills_table = [
                {
                    "element_id": skill.element_id,
                    "name": skill.name,
                    "importance": round(skill.importance, 3),
                    "level": round(skill.level, 3),
                    "description": skill.description,
                }
                for skill in graph_skills
            ]

            occupation_context = self.graph_client.occupation_context(occ_code)

            role_profiles.append(
                {
                    "requested_role": requested_role,
                    "occupation_match": best_match,
                    "alternate_matches": occupation_matches[1:],
                    "skills_covered": matching["matched"],
                    "skill_gaps": matching["missing"],
                    "all_skills": graph_skills,
                    "coverage_stats": coverage_stats,
                    "skills_covered_table": skills_covered_table,
                    "skill_gaps_table": skill_gaps_table,
                    "all_skills_table": all_skills_table,
                    "occupation_context": occupation_context,
                }
            )

        analysis = None
        if summarise:
            analysis = self._summarise_profile(role_profiles, user_skill_texts, qna=qna, parsed_resume=parsed_resume)

        return {
            "user_skills": user_skill_texts,
            "profiles": role_profiles,
            "analysis": analysis,
        }

    # ------------------------------------------------------------------
    def _summarise_profile(
        self,
        role_profiles: Sequence[Dict[str, Any]],
        user_skills: Sequence[str],
        *,
        qna: Optional[Sequence[Dict[str, str]]] = None,
        parsed_resume: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a narrative summary grounded in the knowledge graph facts."""
        summary_payload = {
            "user_skills": list(user_skills),
            "qna": list(qna) if qna else [],
            "profiles": [
                {
                    "requested_role": profile["requested_role"],
                    "occupation_match": profile["occupation_match"],
                    "coverage": profile.get("coverage_stats"),
                    "top_skills_covered": [
                        {
                            "name": item["skill"].name,
                            "importance": item["skill"].importance,
                            "score": item["score"],
                        }
                        for item in profile["skills_covered"][:8]
                    ],
                    "top_skill_gaps": [
                        {
                            "name": item["skill"].name,
                            "importance": item["skill"].importance,
                        }
                        for item in profile["skill_gaps"][:8]
                    ],
                    "skills_covered_table": profile.get("skills_covered_table", [])[:12],
                    "skill_gaps_table": profile.get("skill_gaps_table", [])[:12],
                    "occupation_context": profile.get("occupation_context"),
                }
                for profile in role_profiles
            ],
        }

        if parsed_resume:
            summary_payload["resume_snapshot"] = {
                "experience": parsed_resume.get("experience"),
                "education": parsed_resume.get("education"),
                "summary": parsed_resume.get("summary"),
            }

        messages = [
            SystemMessage(
                content=(
                    "You are an AI career coach that must ground every assessment in the provided "
                    "knowledge graph facts. Do not hallucinate new skills. When discussing gaps, reference "
                    "the occupation title and highlight why the skill matters."
                )
            ),
            HumanMessage(
                content=(
                    "Using the structured context below, craft a concise skill profile that\n"
                    "1. Highlights the strongest validated skills.\n"
                    "2. Lists the most critical skill gaps per role with actionable guidance.\n"
                    "3. Mentions any notable preferences from the questionnaire if available.\n"
                    "4. Quantify alignment using the provided coverage metrics (percentages and counts).\n"
                    "Context: " + json.dumps(summary_payload, indent=2)
                )
            ),
        ]
        response = self.llm.invoke(messages)
        return response.content


def build_skill_profile(
    parsed_resume: Dict[str, Any],
    target_roles: Sequence[str],
    *,
    qna: Optional[Sequence[Dict[str, str]]] = None,
    summarise: bool = True,
    similarity_threshold: float = 0.72,
    max_skills_per_role: int = 25,
) -> Dict[str, Any]:
    """Convenience helper to run the end-to-end SkillGraph RAG pipeline."""
    rag = SkillGraphRAG(similarity_threshold=similarity_threshold)
    return rag.generate_skill_profile(
        parsed_resume,
        target_roles,
        qna=qna,
        max_skills_per_role=max_skills_per_role,
        summarise=summarise,
    )


_WRITE_KEYWORDS = {"create", "merge", "delete", "detach", "remove", "set", "call db.index", "call db.schema"}


@tool("skillgraph-cypher")
def skillgraph_cypher(query: str) -> List[Dict[str, Any]]:
    """Run a read-only Cypher query against the SkillGraph O*NET database."""

    lowered = query.strip().lower()
    if not lowered:
        raise ValueError("Provide a Cypher query to execute.")

    if any(keyword in lowered for keyword in _WRITE_KEYWORDS):
        raise ValueError("Write operations are not permitted through this tool.")

    client = SkillGraphClient()
    if client.graph is None:
        raise ValueError("Neo4j connection unavailable; Cypher queries cannot be executed in offline mode.")
    return client.graph.query(query)


def _demo() -> None:  # pragma: no cover - utility for manual verification
    """Quick demonstration when executing this module directly."""
    sample_resume = {
        "technical_skills": ["Python", "Data Analysis", "Machine Learning", "SQL"],
        "soft_skills": ["Collaboration", "Communication"],
        "experience": "2 years as a junior data analyst focusing on reporting and dashboarding.",
        "education": ["B.S. in Information Systems"],
    }
    target_roles = ["Data Scientist", "Business Intelligence Analyst"]

    rag = SkillGraphRAG()
    profile = rag.generate_skill_profile(sample_resume, target_roles, summarise=True)

    print(json.dumps({
        "profiles": profile["profiles"],
        "analysis": profile["analysis"],
    }, indent=2))


if __name__ == "__main__":  # pragma: no cover - manual execution guard
    try:
        _demo()
    except Exception as exc:  # Basic diagnostic when running ad hoc from CLI
        logger.error("Demo execution failed: %s", exc)
        raise
