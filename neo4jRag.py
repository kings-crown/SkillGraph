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
from functools import lru_cache
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

try:  # Optional Qdrant dependency for hybrid retrieval
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except ImportError:  # pragma: no cover - keep runtime optional
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore

logger = logging.getLogger(__name__)


def _load_environment() -> None:
    """Load Neo4j/OpenAI environment variables from `.env` if present."""
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Fallback to default behaviour (load from current working directory / process env)
        load_dotenv()


def _coerce_skill_terms(skills: Sequence[Any]) -> List[str]:
    """Normalise mixed skill payloads (strings or dictionaries) into text terms."""
    terms: List[str] = []
    seen: set = set()
    for skill in skills or []:
        candidate: Optional[str] = None
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
        cleaned = candidate.strip()
        key = cleaned.lower()
        if key and key not in seen:
            seen.add(key)
            terms.append(cleaned)
    return terms


@lru_cache(maxsize=1)
def _load_alias_metadata() -> Dict[str, Dict[str, Any]]:
    """Load curated alias metadata for enriching vector payloads."""
    candidates = [
        Path(__file__).resolve().parents[1] / "backend" / "career_coach" / "resources" / "skills" / "skills.json",
        Path(__file__).resolve().parent / "resources" / "skills.json",
    ]

    data: Dict[str, Any] = {}
    for path in candidates:
        if path.exists():
            try:
                with path.open(encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception as exc:
                logger.debug("Failed to load alias metadata from %s: %s", path, exc)
                data = {}
            break

    alias_entries = data.get("aliases", []) if isinstance(data, dict) else []
    by_element: Dict[Optional[str], Dict[str, Any]] = {}
    by_canonical: Dict[str, Dict[str, Any]] = {}

    for entry in alias_entries:
        if not isinstance(entry, dict):
            continue
        element_id = entry.get("element_id")
        canonical = entry.get("canonical_name") or entry.get("name")
        if element_id:
            by_element[element_id] = entry
        if canonical:
            by_canonical[canonical.lower()] = entry

    return {
        "by_element": by_element,
        "by_canonical": by_canonical,
    }


def _apply_alias_metadata(payload: Dict[str, Any], alias_catalog: Dict[str, Dict[str, Any]]) -> None:
    alias_by_element = alias_catalog.get("by_element", {})
    alias_by_canonical = alias_catalog.get("by_canonical", {})

    entry: Optional[Dict[str, Any]] = None
    element_id = payload.get("element_id")
    name = payload.get("name")
    if element_id:
        entry = alias_by_element.get(element_id)
    if entry is None and name:
        entry = alias_by_canonical.get(str(name).lower())
    if not entry:
        return

    canonical = entry.get("canonical_name") or entry.get("name")
    if canonical:
        payload["canonical_name"] = canonical
    aliases = entry.get("aliases") or []
    if aliases:
        payload["aliases"] = list(aliases)
    abbreviations = entry.get("abbreviations") or []
    if abbreviations:
        payload["abbreviations"] = list(abbreviations)
    tooling = entry.get("tooling") or []
    if tooling:
        payload["related_tools"] = list(tooling)
    notes = entry.get("notes")
    if notes:
        payload["alias_notes"] = notes


def _skill_record_from_payload(
    payload: Dict[str, Any],
    alias_catalog: Dict[str, Dict[str, Any]],
    *,
    source: str,
) -> SkillRecord:
    enriched = dict(payload)
    _apply_alias_metadata(enriched, alias_catalog)

    element_id = str(enriched.get("element_id", ""))
    name = enriched.get("name", element_id)
    description = enriched.get("description", "")
    try:
        importance = float(enriched.get("importance") or 0.0)
    except (TypeError, ValueError):
        importance = 0.0
    try:
        level = float(enriched.get("level") or 0.0)
    except (TypeError, ValueError):
        level = 0.0

    return SkillRecord(
        element_id=element_id,
        name=name,
        description=description,
        importance=importance,
        level=level,
        source=source,
        canonical_name=enriched.get("canonical_name"),
        aliases=enriched.get("aliases"),
        abbreviations=enriched.get("abbreviations"),
        related_tools=enriched.get("related_tools"),
        notes=enriched.get("alias_notes"),
    )


@dataclass
class SkillRecord:
    """Lightweight container for skill descriptors returned from the graph or vector store."""

    element_id: str
    name: str
    description: str
    importance: float
    level: float
    source: str = "graph"  # graph | vector
    canonical_name: Optional[str] = None
    aliases: Optional[List[str]] = None
    abbreviations: Optional[List[str]] = None
    related_tools: Optional[List[str]] = None
    notes: Optional[str] = None

    @property
    def text(self) -> str:
        """Canonical text representation used for embedding comparisons."""
        parts = [self.name]
        if self.description:
            parts.append(self.description)
        if self.aliases:
            parts.append("Aliases: " + ", ".join(self.aliases))
        if self.related_tools:
            parts.append("Related tools: " + ", ".join(self.related_tools))
        if self.notes:
            parts.append(self.notes)
        return " | ".join(parts)


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
                    alias_catalog = _load_alias_metadata()
                    return [
                        _skill_record_from_payload(
                            {
                                "element_id": record["element_id"],
                                "name": record["name"],
                                "description": record["description"],
                                "importance": record["importance"],
                                "level": record["level"],
                            },
                            alias_catalog,
                            source="graph",
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


class SkillGraphVectorStore:
    """Vector search helper backed by Qdrant for semantic skill matching."""

    def __init__(
        self,
        *,
        collection_name: str = "skillgraph_skills",
        embedding_model: str = "text-embedding-3-small",
        batch_size: int = 128,
    ) -> None:
        _load_environment()
        self._collection = collection_name
        self._embedding_model = embedding_model
        self._batch_size = batch_size
        self._client: Optional[QdrantClient] = None
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._available = False

        if QdrantClient is None or qmodels is None:
            logger.debug("Qdrant client unavailable; vector store disabled.")
            return

        from os import getenv

        url = getenv("QDRANT_URL")
        api_key = getenv("QDRANT_API_KEY")

        alias_catalog = _load_alias_metadata()

        try:
            if url:
                self._client = QdrantClient(url=url, api_key=api_key)
            else:
                storage_path = Path(__file__).resolve().parent / "qdrant_storage"
                storage_path.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(storage_path))
        except Exception as exc:  # pragma: no cover - environment specific
            logger.warning("Failed to initialise Qdrant client: %s", exc)
            self._client = None
            return

        try:
            self._embeddings = OpenAIEmbeddings(model=self._embedding_model)
        except Exception as exc:  # pragma: no cover - network/model availability
            logger.warning("Failed to initialise OpenAI embeddings for vector store: %s", exc)
            self._client = None
            return

        self._available = True

    # ------------------------------------------------------------------
    def available(self) -> bool:
        return self._available and self._client is not None and self._embeddings is not None

    def ensure_collection(self) -> bool:
        if not self.available():
            return False
        assert self._client is not None
        assert self._embeddings is not None

        try:
            dim = len(self._embeddings.embed_query("ping"))
        except Exception as exc:
            logger.warning("Vector store embedding check failed: %s", exc)
            self._available = False
            return False
        try:
            self._client.get_collection(self._collection)
        except Exception:
            logger.info("Creating Qdrant collection '%s'", self._collection)
            self._client.recreate_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            )

        try:
            count = self._client.count(self._collection)
            if count.count:
                return True
        except Exception:
            pass

        logger.info("Populating Qdrant collection '%s'", self._collection)
        self._bulk_upsert()
        return True

    def _bulk_upsert(self) -> None:
        if not self.available():
            return
        assert self._client is not None
        assert self._embeddings is not None

        client = SkillGraphClient()

        payloads: List[Dict[str, Any]] = []
        alias_catalog = _load_alias_metadata()

        if client.graph is not None:
            try:
                records = client.graph.query(
                    """
                    MATCH (o:Occupation)-[r:REQUIRES_SKILL]->(skill:ContentElement)
                    RETURN o.code AS occupation_code,
                           skill.element_id AS element_id,
                           skill.name AS name,
                           coalesce(skill.description, '') AS description,
                           toFloat(coalesce(r.importance, 0)) AS importance,
                           toFloat(coalesce(r.level, 0)) AS level
                    """
                )
                for row in records:
                    occ_code = row["occupation_code"]
                    payload = {
                        "occupation_code": occ_code,
                        "soc_prefix": occ_code.split(".")[0],
                        "element_id": row["element_id"],
                        "name": row["name"],
                        "description": row["description"],
                        "importance": row["importance"],
                        "level": row["level"],
                        "source": "graph",
                    }
                    _apply_alias_metadata(payload, alias_catalog)
                    payloads.append(payload)
            except Exception as exc:
                logger.warning("Failed to pull skills from Neo4j for vector store: %s", exc)

        if not payloads:
            client._ensure_local_cache()
            for occ_code, skills in client._local_skill_records.items():
                prefix = occ_code.split(".")[0]
                for record in skills:
                    payload = {
                        "occupation_code": occ_code,
                        "soc_prefix": prefix,
                        "element_id": record.element_id,
                        "name": record.name,
                        "description": record.description,
                        "importance": record.importance,
                        "level": record.level,
                        "source": record.source,
                    }
                    _apply_alias_metadata(payload, alias_catalog)
                    payloads.append(payload)

        if not payloads:
            logger.warning("SkillGraphVectorStore: no payloads to ingest.")
            return

        texts: List[str] = []
        for payload in payloads:
            alias_fragment = ""
            aliases = payload.get("aliases") or []
            if aliases:
                alias_fragment = "Aliases: " + ", ".join(aliases)
            abbreviations = payload.get("abbreviations") or []
            abbreviation_fragment = ""
            if abbreviations:
                abbreviation_fragment = "Abbreviations: " + ", ".join(abbreviations)
            tools = payload.get("related_tools") or []
            tools_fragment = ""
            if tools:
                tools_fragment = "Related tools: " + ", ".join(tools)
            notes_fragment = payload.get("alias_notes") or ""
            components = [
                str(payload.get("name", "")),
                payload.get("description", ""),
                alias_fragment,
                abbreviation_fragment,
                tools_fragment,
                notes_fragment,
            ]
            texts.append("\n".join(part for part in components if part))
        vectors = self._embed_documents(texts)

        points: List[qmodels.PointStruct] = []
        for idx, (vector, payload) in enumerate(zip(vectors, payloads)):
            points.append(
                qmodels.PointStruct(
                    id=idx,
                    vector=vector,
                    payload=payload,
                )
            )

        for start in range(0, len(points), self._batch_size):
            batch = points[start : start + self._batch_size]
            self._client.upsert(collection_name=self._collection, points=batch)

    def _embed_documents(self, documents: List[str]) -> List[List[float]]:
        assert self._embeddings is not None
        vectors: List[List[float]] = []
        for start in range(0, len(documents), self._batch_size):
            batch = documents[start : start + self._batch_size]
            vectors.extend(self._embeddings.embed_documents(batch))
        return vectors

    def search_skills(
        self,
        text: str,
        *,
        occupation_code: Optional[str] = None,
        top_k: int = 5,
        score_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        if not self.available():
            return []
        assert self._client is not None
        assert self._embeddings is not None

        self.ensure_collection()
        vector = self._embeddings.embed_query(text)

        must_conditions: List[qmodels.FieldCondition] = []
        if occupation_code:
            prefix = occupation_code.split(".")[0]
            must_conditions.append(
                qmodels.FieldCondition(
                    key="soc_prefix",
                    match=qmodels.MatchValue(value=prefix),
                )
            )

        query_filter = qmodels.Filter(must=must_conditions) if must_conditions else None

        try:
            response = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=top_k,
                query_filter=query_filter,
            )
        except Exception as exc:
            logger.debug("Qdrant search failed: %s", exc)
            return []

        matches: List[Dict[str, Any]] = []
        for hit in response:
            if hit.score is None or hit.payload is None:
                continue
            if score_threshold and hit.score < score_threshold:
                continue
            payload = dict(hit.payload)
            payload["score"] = float(hit.score)
            matches.append(payload)
        return matches

    def fetch_skills_for_prefix(self, prefix: str, *, top_k: int = 50) -> List[SkillRecord]:
        matches = self.search_skills(
            text=prefix,
            occupation_code=f"{prefix}.00",
            top_k=top_k,
            score_threshold=0.0,
        )
        skills: Dict[str, SkillRecord] = {}
        alias_catalog = _load_alias_metadata()
        for payload in matches:
            element_id = payload.get("element_id")
            if not element_id:
                continue
            record = _skill_record_from_payload(
                payload,
                alias_catalog,
                source="vector",
            )
            existing = skills.get(element_id)
            if existing is None or record.importance > existing.importance:
                skills[element_id] = record
        return sorted(skills.values(), key=lambda rec: rec.importance, reverse=True)
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
                        record = _skill_record_from_payload(
                            {
                                "element_id": elem_id,
                                "name": content.get("name", elem_id),
                                "description": content.get("description", ""),
                                "importance": importance,
                                "level": level,
                            },
                            alias_catalog,
                            source="graph",
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

        # Aggregate base occupation records from variant codes where needed
        variant_groups: Dict[str, List[str]] = {}
        for code in cls._local_skill_records.keys():
            if "." in code:
                prefix = code.split(".")[0]
                variant_groups.setdefault(prefix, []).append(code)

        for prefix, codes in variant_groups.items():
            base_code = f"{prefix}.00"
            if base_code in cls._local_skill_records or base_code not in cls._local_occupation_map:
                continue

            aggregated: Dict[str, SkillRecord] = {}
            for variant_code in codes:
                for record in cls._local_skill_records.get(variant_code, []):
                    existing = aggregated.get(record.element_id)
                    if existing is None or record.importance > existing.importance:
                        aggregated[record.element_id] = record

            if not aggregated:
                continue

            ordered = sorted(aggregated.values(), key=lambda rec: rec.importance, reverse=True)
            cls._local_skill_records[base_code] = ordered
            cls._local_skill_importance[base_code] = {
                record.element_id: record.importance for record in ordered
            }

            activity_records: List[Dict[str, Any]] = []
            for variant_code in codes:
                for record in cls._local_activity_records.get(variant_code, []):
                    element_id = record.get("element_id")
                    if not element_id:
                        continue
                    existing = next((item for item in activity_records if item.get("element_id") == element_id), None)
                    if existing is None or record.get("importance", 0.0) > existing.get("importance", 0.0):
                        if existing is not None:
                            activity_records.remove(existing)
                        activity_records.append(record)

            activity_records.sort(key=lambda rec: rec.get("importance", 0.0), reverse=True)
            cls._local_activity_records[base_code] = activity_records

    def _local_resolve(self, candidate_title: str, limit: int) -> List[Dict[str, Any]]:
        self._ensure_local_cache()
        cls = type(self)
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

    SEMANTIC_WEIGHT = 0.6
    LEXICAL_WEIGHT = 0.3
    ALIAS_WEIGHT = 0.1

    def __init__(self, embedder: Optional[OpenAIEmbeddings] = None) -> None:
        _load_environment()
        self.embedder = embedder or OpenAIEmbeddings(model="text-embedding-3-small")
        self._skill_embedding_cache: Dict[str, np.ndarray] = {}

    def match(
        self,
        user_skill_texts: Sequence[Any],
        graph_skills: Sequence[SkillRecord],
        *,
        similarity_threshold: float = 0.72,
        lexical_threshold: float = 0.68,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return skills the user already covers and those likely missing."""
        if not graph_skills:
            return {"matched": [], "missing": []}

        normalised_user_skills = _coerce_skill_terms(user_skill_texts)

        if not normalised_user_skills:
            missing_payload = [
                {
                    "skill": skill,
                    "score": None,
                    "matched_user_skill": None,
                    "score_breakdown": {},
                }
                for skill in graph_skills
            ]
            return {"matched": [], "missing": missing_payload}

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
            semantic_score = float(best_scores[idx])
            best_user_skill = normalised_user_skills[int(best_user_indices[idx])]
            lexical_score = self._lexical_similarity(best_user_skill, skill)
            alias_hit = self._alias_hit(best_user_skill, skill)
            blended_score = self._blend_scores(semantic_score, lexical_score, alias_hit)

            score_breakdown = {
                "semantic": round(semantic_score, 3),
                "lexical": round(lexical_score, 3),
                "alias": bool(alias_hit),
                "blended": round(blended_score, 3),
            }

            payload = {
                "skill": skill,
                "score": score_breakdown["blended"],
                "matched_user_skill": best_user_skill,
                "match_source": "graph",
                "score_breakdown": score_breakdown,
                "semantic_score": score_breakdown["semantic"],
                "lexical_score": score_breakdown["lexical"],
            }

            if alias_hit or blended_score >= similarity_threshold or lexical_score >= lexical_threshold:
                matched.append(payload)
            else:
                payload["matched_user_skill"] = None
                missing.append(payload)

        matched.sort(key=lambda item: (item["skill"].importance, item["score"]), reverse=True)
        missing.sort(key=lambda item: item["skill"].importance, reverse=True)

        if matched or missing:
            metrics = {
                "matched_count": len(matched),
                "missing_count": len(missing),
                "similarity_threshold": similarity_threshold,
                "lexical_threshold": lexical_threshold,
                "semantic_avg": round(
                    float(np.mean([item["score_breakdown"]["semantic"] for item in matched])) if matched else 0.0,
                    3,
                ),
                "blended_avg": round(
                    float(np.mean([item["score_breakdown"]["blended"] for item in matched])) if matched else 0.0,
                    3,
                ),
            }
            logger.info("SkillMatcher metrics: %s", metrics)

        return {"matched": matched, "missing": missing}

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

    def _lexical_similarity(self, user_term: str, skill: SkillRecord) -> float:
        candidates = [skill.name]
        if skill.canonical_name:
            candidates.append(skill.canonical_name)
        if skill.aliases:
            candidates.extend(skill.aliases)
        scores = [
            SequenceMatcher(None, user_term.lower(), candidate.lower()).ratio()
            for candidate in candidates
            if candidate
        ]
        return max(scores) if scores else 0.0

    def _alias_hit(self, user_term: str, skill: SkillRecord) -> bool:
        term = user_term.lower()
        candidates = []
        if skill.aliases:
            candidates.extend(alias.lower() for alias in skill.aliases if alias)
        if skill.abbreviations:
            candidates.extend(abbrev.lower() for abbrev in skill.abbreviations if abbrev)
        if skill.canonical_name:
            candidates.append(skill.canonical_name.lower())
        candidates.append(skill.name.lower())
        return term in candidates

    def _blend_scores(self, semantic: float, lexical: float, alias_hit: bool) -> float:
        weights = self.SEMANTIC_WEIGHT + self.LEXICAL_WEIGHT + self.ALIAS_WEIGHT
        alias_score = 1.0 if alias_hit else 0.0
        blended = (
            semantic * self.SEMANTIC_WEIGHT
            + lexical * self.LEXICAL_WEIGHT
            + alias_score * self.ALIAS_WEIGHT
        )
        return blended / weights if weights else semantic


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
        lexical_threshold: float = 0.68,
        vector_store: Optional['SkillGraphVectorStore'] = None,
        vector_similarity_threshold: float = 0.5,
    ) -> None:
        _load_environment()
        self.graph_client = graph_client or SkillGraphClient()
        self.matcher = matcher or SkillMatcher()
        self.similarity_threshold = similarity_threshold
        self.lexical_threshold = lexical_threshold

        self.vector_store = vector_store or SkillGraphVectorStore()
        if self.vector_store and not self.vector_store.available():
            self.vector_store = None
        self.vector_similarity_threshold = vector_similarity_threshold

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
        technical_skills = _coerce_skill_terms(parsed_resume.get("technical_skills", []))
        soft_skills = _coerce_skill_terms(parsed_resume.get("soft_skills", []))
        user_skill_texts = _coerce_skill_terms(
            list(parsed_resume.get("technical_skills", [])) + list(parsed_resume.get("soft_skills", []))
        )

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
            if not graph_skills and self.vector_store:
                prefix = occ_code.split(".")[0]
                vector_skills = self.vector_store.fetch_skills_for_prefix(prefix, top_k=max_skills_per_role)
                if vector_skills:
                    graph_skills.extend(vector_skills)
            matching = self.matcher.match(
                user_skill_texts,
                graph_skills,
                similarity_threshold=self.similarity_threshold,
                lexical_threshold=self.lexical_threshold,
            )
            matched_terms = {
                item["matched_user_skill"]
                for item in matching["matched"]
                if item.get("matched_user_skill")
            }
            unmatched_terms = [term for term in user_skill_texts if term not in matched_terms]
            if unmatched_terms:
                logger.debug(
                    "SkillGraphRAG unmatched terms for %s: %s",
                    requested_role,
                    unmatched_terms,
                )
            vector_matches = self._vector_backfill(
                occ_code,
                user_skill_texts,
                matching,
                graph_skills,
                max_skills=max_skills_per_role,
            )
            coverage_stats = self._compute_coverage(graph_skills, matching)
            matcher_metrics = {
                "matched": len(matching["matched"]),
                "missing": len(matching["missing"]),
                "similarity_threshold": self.similarity_threshold,
                "lexical_threshold": self.lexical_threshold,
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
                    "match_source": item.get("match_source", "graph"),
                    "score_breakdown": item.get("score_breakdown", {}),
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
                    "score_breakdown": item.get("score_breakdown", {}),
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
                    "vector_matches": vector_matches,
                    "matcher_metrics": matcher_metrics,
                    "unmatched_user_skills": unmatched_terms,
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
    def _compute_coverage(
        self,
        graph_skills: Sequence[SkillRecord],
        matching: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        unique_skills: Dict[str, SkillRecord] = {}
        for skill in graph_skills:
            unique_skills.setdefault(skill.element_id, skill)

        total_importance = sum(skill.importance for skill in unique_skills.values() if skill.importance)
        total_count = len(unique_skills)

        covered_ids = set()
        covered_importance = 0.0
        covered_vector_importance = 0.0
        covered_official_importance = 0.0
        covered_count = 0
        covered_vector_count = 0
        covered_official_count = 0

        for item in matching["matched"]:
            skill: SkillRecord = item["skill"]
            if skill.element_id in covered_ids:
                continue
            covered_ids.add(skill.element_id)
            covered_count += 1
            covered_importance += skill.importance
            if skill.source == "vector" or item.get("match_source") == "vector":
                covered_vector_count += 1
                covered_vector_importance += skill.importance
            else:
                covered_official_count += 1
                covered_official_importance += skill.importance

        official_skills = [skill for skill in unique_skills.values() if skill.source != "vector"]
        vector_skills = [skill for skill in unique_skills.values() if skill.source == "vector"]
        official_total_importance = sum(skill.importance for skill in official_skills if skill.importance)
        vector_total_importance = sum(skill.importance for skill in vector_skills if skill.importance)
        official_total_count = len(official_skills)
        vector_total_count = len(vector_skills)

        def safe_div(numerator: float, denominator: float) -> float:
            return numerator / denominator if denominator else 0.0

        coverage = {
            "coverage_pct": round(safe_div(covered_importance, total_importance), 4) if total_importance else 0.0,
            "covered_importance": round(covered_importance, 3),
            "total_importance": round(total_importance, 3),
            "covered_count": covered_count,
            "total_count": total_count,
            "official_pct": round(safe_div(covered_official_importance, official_total_importance), 4)
            if official_total_importance
            else None,
            "vector_pct": round(safe_div(covered_vector_importance, vector_total_importance or total_importance), 4)
            if (vector_total_importance or total_importance)
            else None,
            "split": {
                "official": {
                    "count": covered_official_count,
                    "total_count": official_total_count,
                    "importance": round(covered_official_importance, 3),
                    "total_importance": round(official_total_importance, 3),
                },
                "vector": {
                    "count": covered_vector_count,
                    "total_count": vector_total_count,
                    "importance": round(covered_vector_importance, 3),
                    "total_importance": round(vector_total_importance, 3),
                },
            },
        }
        return coverage

    def _vector_backfill(
        self,
        occupation_code: str,
        user_skills: Sequence[str],
        matching: Dict[str, List[Dict[str, Any]]],
        graph_skills: List[SkillRecord],
        *,
        max_skills: int,
    ) -> List[Dict[str, Any]]:
        if not self.vector_store or not self.vector_store.available():
            return []

        if not self.vector_store.ensure_collection():
            return []

        element_map: Dict[str, SkillRecord] = {skill.element_id: skill for skill in graph_skills}
        covered_ids = {item["skill"].element_id for item in matching["matched"]}
        additions: List[Dict[str, Any]] = []
        alias_catalog = _load_alias_metadata()

        for resume_skill in user_skills:
            if not resume_skill:
                continue
            hits = self.vector_store.search_skills(
                resume_skill,
                occupation_code=occupation_code,
                top_k=5,
                score_threshold=self.vector_similarity_threshold,
            )
            for payload in hits:
                metadata = dict(payload)
                element_id = metadata.get("element_id")
                if not element_id or element_id in covered_ids:
                    continue
                _apply_alias_metadata(metadata, alias_catalog)

                skill_record = element_map.get(element_id)
                if skill_record is None:
                    skill_record = _skill_record_from_payload(
                        metadata,
                        alias_catalog,
                        source="vector",
                    )
                    graph_skills.append(skill_record)
                    element_map[element_id] = skill_record
                else:
                    if metadata.get("aliases"):
                        skill_record.aliases = metadata.get("aliases")
                    if metadata.get("abbreviations"):
                        skill_record.abbreviations = metadata.get("abbreviations")
                    if metadata.get("related_tools"):
                        skill_record.related_tools = metadata.get("related_tools")
                    if metadata.get("alias_notes"):
                        skill_record.notes = metadata.get("alias_notes")

                additions.append(
                    {
                        "skill": skill_record,
                        "score": round(payload.get("score", 0.0), 3),
                        "matched_user_skill": resume_skill,
                        "match_source": "vector",
                    }
                )
                covered_ids.add(element_id)
                break

        if additions:
            matched_ids = {item["skill"].element_id for item in additions}
            matching["missing"] = [
                item for item in matching["missing"] if item["skill"].element_id not in matched_ids
            ]
            matching["matched"].extend(additions)

        # Cap overall skill list to avoid unbounded growth
        if len(graph_skills) > max_skills:
            graph_skills.sort(key=lambda record: record.importance, reverse=True)
            del graph_skills[max_skills:]

        return additions

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
