#!/usr/bin/env python3
"""Utility helpers for turning the O*NET Excel delivery into graph-friendly CSVs.

Run this script after downloading and unzipping the O*NET Excel bundle (e.g. ``db_30_0_excel``).
It will create node and relationship CSVs that can be bulk imported into Neo4j or used to
prototype the ontology locally.

Example::

    python Onet.py build --source db_30_0_excel --out neo4j_csv

The script uses pandas with the ``openpyxl`` engine. Install the dependency once via
``python3 -m pip install openpyxl`` if it is missing.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass(frozen=True)
class DescriptorSpec:
    filename: str
    label: str
    relationship: str
    scale_map: Dict[str, str]
    keep_columns: Optional[Iterable[str]] = None


@dataclass(frozen=True)
class CrosswalkSpec:
    filename: str
    start_column: str
    end_column: str
    relationship: str


DESCRIPTOR_SPECS: List[DescriptorSpec] = [
    DescriptorSpec(
        filename="Abilities.xlsx",
        label="Ability",
        relationship="REQUIRES_ABILITY",
        scale_map={"IM": "importance", "LV": "level"},
    ),
    DescriptorSpec(
        filename="Knowledge.xlsx",
        label="KnowledgeArea",
        relationship="REQUIRES_KNOWLEDGE",
        scale_map={"IM": "importance", "LV": "level"},
    ),
    DescriptorSpec(
        filename="Skills.xlsx",
        label="Skill",
        relationship="REQUIRES_SKILL",
        scale_map={"IM": "importance", "LV": "level"},
    ),
    DescriptorSpec(
        filename="Work Activities.xlsx",
        label="WorkActivity",
        relationship="INVOLVES_ACTIVITY",
        scale_map={"IM": "importance", "LV": "level"},
    ),
    DescriptorSpec(
        filename="Work Context.xlsx",
        label="WorkContext",
        relationship="OPERATES_IN_CONTEXT",
        scale_map={"CX": "context_score"},
    ),
    DescriptorSpec(
        filename="Work Styles.xlsx",
        label="WorkStyle",
        relationship="VALUES_STYLE",
        scale_map={"IM": "importance"},
    ),
    DescriptorSpec(
        filename="Work Values.xlsx",
        label="WorkValue",
        relationship="SUPPORTS_VALUE",
        scale_map={"EX": "extent"},
    ),
    DescriptorSpec(
        filename="Interests.xlsx",
        label="Interest",
        relationship="EXPRESSES_INTEREST",
        scale_map={"OI": "interest_score"},
    ),
]


CROSSWALK_SPECS: List[CrosswalkSpec] = [
    CrosswalkSpec(
        filename="Abilities to Work Activities.xlsx",
        start_column="Abilities Element ID",
        end_column="Work Activities Element ID",
        relationship="ABILITY_SUPPORTS_ACTIVITY",
    ),
    CrosswalkSpec(
        filename="Abilities to Work Context.xlsx",
        start_column="Abilities Element ID",
        end_column="Work Context Element ID",
        relationship="ABILITY_INFLUENCES_CONTEXT",
    ),
    CrosswalkSpec(
        filename="Skills to Work Activities.xlsx",
        start_column="Skills Element ID",
        end_column="Work Activities Element ID",
        relationship="SKILL_SUPPORTS_ACTIVITY",
    ),
    CrosswalkSpec(
        filename="Skills to Work Context.xlsx",
        start_column="Skills Element ID",
        end_column="Work Context Element ID",
        relationship="SKILL_RELATES_CONTEXT",
    ),
    CrosswalkSpec(
        filename="Basic Interests to RIASEC.xlsx",
        start_column="Basic Interests Element ID",
        end_column="RIASEC Element ID",
        relationship="BASIC_INTEREST_IN_RIASEC",
    ),
]


NODE_SANITIZE_SPECS: Dict[str, Dict[str, str]] = {
    "occupations.csv": {
        "code:ID(Occupation)": "code",
        "labels:LABEL": "labels",
    },
    "content_elements.csv": {
        "element_id:ID(ContentElement)": "element_id",
        "labels:LABEL": "labels",
        "depth:int": "depth",
    },
}


REL_SANITIZE_RENAME_MAP: Dict[str, str] = {
    "start:START_ID(ContentElement)": "start_content_element_id",
    "start:START_ID(Occupation)": "start_occupation_id",
    "end:END_ID(ContentElement)": "end_content_element_id",
    "end:END_ID(Occupation)": "end_occupation_id",
    ":TYPE": "type",
}


def _sanitize_scale_property(scale_id: str, used: set[str]) -> str:
    """Create a stable property name for a scale id, avoiding collisions."""
    tokens = re.sub(r"[^0-9A-Za-z]+", "_", scale_id).strip("_").lower() or "scale"
    name = tokens if tokens.startswith("scale_") else f"scale_{tokens}"
    candidate = name
    counter = 2
    while candidate in used:
        candidate = f"{name}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def load_excel_table(path: Path, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, dtype=str, engine="openpyxl")
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit(
            "openpyxl is required to read Excel files. Install it with `python3 -m pip install openpyxl`."
        ) from exc
    except FileNotFoundError:
        raise SystemExit(f"Missing expected Excel file: {path}")

    if columns is not None:
        columns = list(columns)
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise SystemExit(f"Columns {missing} are missing from {path.name}. Found columns: {list(df.columns)}")
        df = df[columns]
    return df.fillna("")


def compute_parent(element_id: str) -> Optional[str]:
    if not element_id or "." not in element_id:
        return None
    return ".".join(element_id.split(".")[:-1])


def element_depth(element_id: str) -> int:
    if not element_id:
        return 0
    return element_id.count(".")


def base_type_for_depth(depth: int) -> str:
    if depth == 0:
        return "Domain"
    if depth == 1:
        return "Category"
    if depth == 2:
        return "Subcategory"
    if depth == 3:
        return "Facet"
    return "Descriptor"


def build_content_model(source_dir: Path) -> pd.DataFrame:
    content_path = source_dir / "Content Model Reference.xlsx"
    df = load_excel_table(content_path)
    expected_cols = {"Element ID", "Element Name", "Description"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise SystemExit(
            f"Unexpected schema in {content_path.name}. Missing columns: {sorted(missing)}."
        )

    df = df.rename(
        columns={
            "Element ID": "element_id",
            "Element Name": "name",
            "Description": "description",
        }
    ).assign(
        element_id=lambda frame: frame["element_id"].str.strip(),
        name=lambda frame: frame["name"].str.strip(),
        description=lambda frame: frame["description"].str.strip(),
    )
    df["parent_id"] = df["element_id"].apply(compute_parent)
    df["depth"] = df["element_id"].apply(element_depth)
    df["base_type"] = df["depth"].apply(base_type_for_depth)
    return df


def write_nodes(df: pd.DataFrame, path: Path, id_header: str) -> None:
    if id_header not in df.columns:
        raise SystemExit(f"Expected column '{id_header}' in node frame for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} rows -> {path}")


def write_relationship(df: pd.DataFrame, path: Path, relationship_type: str) -> None:
    if df.empty:
        print(f"Skipping {relationship_type} because no rows were produced")
        return
    df = df.copy()
    df[":TYPE"] = relationship_type
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} rows -> {path}")


def build_occupation_nodes(source_dir: Path) -> pd.DataFrame:
    occ_path = source_dir / "Occupation Data.xlsx"
    df = load_excel_table(occ_path, ["O*NET-SOC Code", "Title", "Description"])
    df = df.rename(
        columns={
            "O*NET-SOC Code": "code:ID(Occupation)",
            "Title": "title",
            "Description": "description",
        }
    )
    df["labels:LABEL"] = "Occupation"
    return df


def prepare_descriptor_relationships(
    source_dir: Path,
    spec: DescriptorSpec,
    element_records: Dict[str, Dict[str, str]],
    element_labels: Dict[str, set[str]],
) -> pd.DataFrame:
    path = source_dir / spec.filename
    df = load_excel_table(path)

    required_cols = {"O*NET-SOC Code", "Element ID", "Element Name", "Scale ID", "Data Value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"{spec.filename} is missing columns {sorted(missing)}")

    if "Recommend Suppress" in df.columns:
        df = df[df["Recommend Suppress"] != "Y"]
    if "Not Relevant" in df.columns:
        df = df[df["Not Relevant"] != "Y"]

    if df.empty:
        return pd.DataFrame()

    df["Data Value"] = pd.to_numeric(df["Data Value"], errors="coerce")
    df = df.dropna(subset=["Data Value"])
    if df.empty:
        return pd.DataFrame()

    observed_scales = {scale for scale in df["Scale ID"].dropna().unique()}
    known_scales = set(spec.scale_map.keys())
    missing_scales = observed_scales - known_scales
    if missing_scales:
        # Surface new scale identifiers so analysts know the spec may require updates.
        print(
            f"  > Detected new scale ids in {spec.filename}: {sorted(missing_scales)}. "
            "They will be exported using generic property names."
        )

    # map labels for the descriptor elements
    for element_id, element_name in zip(df["Element ID"], df["Element Name"]):
        record = element_records.get(element_id)
        if record is None:
            element_records[element_id] = {
                "element_id": element_id,
                "name": element_name,
                "description": "",
                "parent_id": compute_parent(element_id),
                "depth": element_depth(element_id),
                "base_type": base_type_for_depth(element_depth(element_id)),
            }
            element_labels[element_id] = {"ContentElement"}
        element_labels.setdefault(element_id, {"ContentElement"}).add(spec.label)

    pivot = (
        df.pivot_table(
            index=["O*NET-SOC Code", "Element ID"],
            columns="Scale ID",
            values="Data Value",
            aggfunc="first",
        )
        .reset_index()
    )

    scale_name_map: Dict[str, str] = {}
    used_names: set[str] = set()
    for scale in sorted(observed_scales):
        if scale in spec.scale_map:
            base_name = spec.scale_map[scale]
            name = base_name
            suffix = 2
            while name in used_names:
                name = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(name)
        else:
            # Fall back to sanitized property names so newly published scales are not lost.
            name = _sanitize_scale_property(scale, used_names)
        scale_name_map[scale] = name

    pivot = pivot.rename(columns=scale_name_map)

    meta_cols = [col for col in ("Date", "Domain Source", "N") if col in df.columns]
    if meta_cols:
        meta = df.groupby(["O*NET-SOC Code", "Element ID"])[meta_cols].first().reset_index()
        pivot = pivot.merge(meta, on=["O*NET-SOC Code", "Element ID"], how="left")

    rename_map = {
        "O*NET-SOC Code": "start:START_ID(Occupation)",
        "Element ID": "end:END_ID(ContentElement)",
    }
    pivot = pivot.rename(columns=rename_map)

    for prop in scale_name_map.values():
        if prop in pivot.columns:
            pivot[prop] = pd.to_numeric(pivot[prop], errors="coerce")
    if "N" in pivot.columns:
        pivot["sample_size"] = pd.to_numeric(pivot.pop("N"), errors="coerce")

    if "Date" in pivot.columns:
        pivot["date"] = pivot.pop("Date")
    if "Domain Source" in pivot.columns:
        pivot["source"] = pivot.pop("Domain Source")

    ordered_cols = [
        "start:START_ID(Occupation)",
        "end:END_ID(ContentElement)",
        *sorted(col for col in pivot.columns if col not in {"start:START_ID(Occupation)", "end:END_ID(ContentElement)"}),
    ]
    pivot = pivot[ordered_cols]
    return pivot


def prepare_crosswalk_relationships(
    source_dir: Path,
    spec: CrosswalkSpec,
    element_records: Dict[str, Dict[str, str]],
    element_labels: Dict[str, set[str]],
) -> pd.DataFrame:
    path = source_dir / spec.filename
    df = load_excel_table(path, [spec.start_column, spec.end_column])
    df = df.rename(
        columns={
            spec.start_column: "start:START_ID(ContentElement)",
            spec.end_column: "end:END_ID(ContentElement)",
        }
    ).drop_duplicates()

    # make sure crosswalk elements are registered
    for column in ("start:START_ID(ContentElement)", "end:END_ID(ContentElement)"):
        for element_id in df[column].unique():
            if element_id not in element_records:
                element_records[element_id] = {
                    "element_id": element_id,
                    "name": element_id,
                    "description": "",
                    "parent_id": compute_parent(element_id),
                    "depth": element_depth(element_id),
                    "base_type": base_type_for_depth(element_depth(element_id)),
                }
                element_labels[element_id] = {"ContentElement"}
    return df


def assemble_element_nodes(
    element_records: Dict[str, Dict[str, str]], element_labels: Dict[str, set[str]]
) -> pd.DataFrame:
    records = []
    for element_id, record in element_records.items():
        labels = element_labels.get(element_id, {"ContentElement"})
        labels.add(record.get("base_type", base_type_for_depth(record.get("depth", 0))))
        records.append(
            {
                "element_id:ID(ContentElement)": element_id,
                "name": record.get("name", element_id),
                "description": record.get("description", ""),
                "depth:int": record.get("depth", element_depth(element_id)),
                "parent_id": record.get("parent_id"),
                "labels:LABEL": ";".join(sorted(labels)),
            }
        )
    df = pd.DataFrame(records)
    return df.sort_values(by="element_id:ID(ContentElement)")


def assemble_hierarchy_relationships(content_df: pd.DataFrame) -> pd.DataFrame:
    hierarchy = content_df.dropna(subset=["parent_id"])
    if hierarchy.empty:
        return pd.DataFrame()
    return hierarchy.rename(
        columns={
            "parent_id": "start:START_ID(ContentElement)",
            "element_id": "end:END_ID(ContentElement)",
        }
    )[["start:START_ID(ContentElement)", "end:END_ID(ContentElement)"]]


def sanitize_csv_file(in_path: Path, out_path: Path, rename_map: Dict[str, str], chunk_size: int = 100_000) -> None:
    if not in_path.exists():
        print(f"Skipping missing source file {in_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    first_chunk = True
    for chunk in pd.read_csv(in_path, dtype=str, chunksize=chunk_size):
        applicable_map = {src: dst for src, dst in rename_map.items() if src in chunk.columns}
        if applicable_map:
            chunk = chunk.rename(columns=applicable_map)

        ordered_columns: List[str] = []
        for dest in rename_map.values():
            if dest in chunk.columns and dest not in ordered_columns:
                ordered_columns.append(dest)

        remaining_columns = [col for col in chunk.columns if col not in ordered_columns]
        chunk = chunk[ordered_columns + remaining_columns] if ordered_columns else chunk

        mode = "w" if first_chunk else "a"
        chunk.to_csv(out_path, index=False, mode=mode, header=first_chunk)
        first_chunk = False

    if first_chunk:
        out_path.write_text("")

    print(f"  Sanitized {in_path.name} -> {out_path.name}")


def collect_relationship_types(csv_path: Path, chunk_size: int = 100_000) -> List[str]:
    """Return all distinct relationship types in a CSV without sampling."""
    type_values: set[str] = set()
    try:
        for chunk in pd.read_csv(csv_path, dtype=str, chunksize=chunk_size):
            if ":TYPE" not in chunk.columns:
                return []
            type_values.update(value for value in chunk[":TYPE"].dropna().unique())
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return []
    return sorted(type_values)


def emit_importer_ready_assets(out_dir: Path) -> None:
    importer_dir = out_dir / "data_importer_ready"
    importer_dir.mkdir(parents=True, exist_ok=True)

    nodes_dir = out_dir / "nodes"
    relationships_dir = out_dir / "relationships"

    print("\nCreating Data Importer–friendly copies under", importer_dir)
    for filename, rename_map in NODE_SANITIZE_SPECS.items():
        source_path = nodes_dir / filename
        if source_path.exists():
            target_name = f"{Path(filename).stem}_sanitized.csv"
            sanitize_csv_file(source_path, importer_dir / target_name, rename_map)
        else:
            print(f"  Skipping node file {filename} (not found)")

    type_samples: Dict[str, List[str]] = {}
    if relationships_dir.exists():
        for rel_path in sorted(relationships_dir.glob("*.csv")):
            target_name = f"{rel_path.stem}_sanitized.csv"
            sanitize_csv_file(rel_path, importer_dir / target_name, REL_SANITIZE_RENAME_MAP)

            types = collect_relationship_types(rel_path)
            type_samples[rel_path.name] = types

    if type_samples:
        print("\nRelationship :TYPE values detected:")
        for name, values in type_samples.items():
            if not values:
                print(f"  {name}: (no :TYPE column)")
            elif isinstance(values, list):
                print(f"  {name}: {', '.join(values)}")
            else:  # should not happen, but keep defensive
                print(f"  {name}: {values}")

    print("Finished preparing Data Importer assets. Files are located at", importer_dir)


def build_graph_assets(source: Path, out_dir: Path, include_crosswalks: bool, emit_importer_ready: bool) -> None:
    out_nodes = out_dir / "nodes"
    out_relationships = out_dir / "relationships"

    content_df = build_content_model(source)
    element_records = {
        row.element_id: {
            "element_id": row.element_id,
            "name": row.name,
            "description": row.description,
            "parent_id": row.parent_id,
            "depth": int(row.depth),
            "base_type": row.base_type,
        }
        for row in content_df.itertuples()
    }
    element_labels: Dict[str, set[str]] = {}
    for row in content_df.itertuples():
        element_labels[row.element_id] = {"ContentElement", row.base_type}

    occ_nodes = build_occupation_nodes(source)
    write_nodes(occ_nodes, out_nodes / "occupations.csv", "code:ID(Occupation)")

    hierarchy = assemble_hierarchy_relationships(content_df)
    write_relationship(hierarchy, out_relationships / "content_hierarchy.csv", "HAS_CHILD")

    relationship_frames = []
    for spec in DESCRIPTOR_SPECS:
        rel_df = prepare_descriptor_relationships(source, spec, element_records, element_labels)
        if rel_df.empty:
            print(f"No relationships generated for {spec.relationship}")
            continue
        relationship_frames.append((spec.relationship, rel_df))

    for rel_type, rel_df in relationship_frames:
        rel_path = out_relationships / f"occupation_{rel_type.lower()}.csv"
        write_relationship(rel_df, rel_path, rel_type)

    if include_crosswalks:
        for spec in CROSSWALK_SPECS:
            rel_df = prepare_crosswalk_relationships(source, spec, element_records, element_labels)
            if rel_df.empty:
                print(f"No rows found in {spec.filename}, skipping")
                continue
            rel_path = out_relationships / f"descriptor_{spec.relationship.lower()}.csv"
            write_relationship(rel_df, rel_path, spec.relationship)

    element_nodes = assemble_element_nodes(element_records, element_labels)
    write_nodes(element_nodes, out_nodes / "content_elements.csv", "element_id:ID(ContentElement)")

    if emit_importer_ready:
        emit_importer_ready_assets(out_dir)

    print("\nNext steps:")
    print("  - Inspect the CSVs under", out_dir)
    print(
        "  - Use neo4j-admin import or LOAD CSV scripts to ingest the node and relationship files"
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="Produce Neo4j-friendly CSVs from the Excel bundle")
    build_parser.add_argument("--source", type=Path, required=True, help="Directory containing the Excel files (e.g. db_30_0_excel)")
    build_parser.add_argument("--out", type=Path, required=True, help="Directory where CSVs should be written")
    build_parser.add_argument(
        "--include-crosswalks",
        action="store_true",
        help="Also emit descriptor-to-descriptor crosswalk relationships",
    )
    build_parser.add_argument(
        "--emit-importer-ready",
        action="store_true",
        help="Create sanitized copies of the CSVs that work well with Neo4j Data Importer",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.command == "build":
        source = args.source.expanduser().resolve()
        out_dir = args.out.expanduser().resolve()
        if not source.is_dir():
            sys.exit(f"Source directory {source} does not exist")
        out_dir.mkdir(parents=True, exist_ok=True)
        build_graph_assets(
            source,
            out_dir,
            include_crosswalks=args.include_crosswalks,
            emit_importer_ready=args.emit_importer_ready,
        )
    else:  # pragma: no cover - defensive fallback
        sys.exit(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
