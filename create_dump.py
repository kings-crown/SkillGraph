#!/usr/bin/env python3
"""Create a Neo4j database dump alongside the O*NET assets.

This helper wraps `neo4j-admin database dump` so you can generate
`<database>.dump` and the zipped archive (`<database>_dump.zip`) right next
to `Onet.py` / the downloaded O*NET bundle. Before dumping it ensures the
Excel bundle has been unpacked (it extracts `db_30_0_excel.zip` when needed)
so downstream scripts have the raw spreadsheets available.

Example usage::

    sudo python3 create_dump.py --database onet

By default it looks for `neo4j-admin` under `/usr/share/neo4j/bin/neo4j-admin`.
Provide `--neo4j-admin` if your installation lives elsewhere.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional
import zipfile
import csv
import os
import shutil


def run_command(cmd: list[str], *, cwd: Optional[Path] = None) -> None:
    """Run a shell command, streaming output. Raise on failure."""
    try:
        subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}"
        ) from exc


def ensure_executable(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"neo4j-admin not found at {path}. Provide --neo4j-admin.")
    if not path.is_file():
        raise SystemExit(f"Expected a file at {path}, but found a directory.")
    if not path.stat().st_mode & 0o111:
        raise SystemExit(f"neo4j-admin at {path} is not executable.")
    return path


def ensure_excel_bundle(zip_path: Path, target_dir: Path) -> Path:
    """Make sure the Excel delivery is unpacked before we generate dumps."""
    if target_dir.exists():
        return target_dir

    if not zip_path.exists():
        raise SystemExit(
            f"Excel directory {target_dir} is missing and archive {zip_path} was not found."
        )

    print(f"Extracting {zip_path} -> {zip_path.parent}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(zip_path.parent)

    if not target_dir.exists():
        raise SystemExit(
            f"Extraction from {zip_path} completed but {target_dir} was not created."
        )
    return target_dir


def verify_python_runtime(python_exec: Path) -> None:
    if not python_exec.exists():
        raise SystemExit(f"Python executable not found: {python_exec}")
    if not python_exec.stat().st_mode & 0o111:
        raise SystemExit(f"Python executable is not runnable: {python_exec}")

    missing_modules = []
    for module in ("pandas", "openpyxl"):
        proc = subprocess.run(
            [str(python_exec), "-c", f"import {module}"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            missing_modules.append(module)
    if missing_modules:
        modules_str = " ".join(missing_modules)
        raise SystemExit(
            f"Python runtime {python_exec} is missing required modules: {modules_str}.\n"
            f"Install them with '{python_exec} -m pip install pandas openpyxl'."
        )


def run_onet_build(
    python_exec: Path, script_path: Path, source_dir: Path, out_dir: Path
) -> None:
    if not script_path.exists():
        raise SystemExit(f"Onet.py not found at {script_path}")
    verify_python_runtime(python_exec)
    print(f"Running Onet.py build with {python_exec} …")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python_exec),
        str(script_path),
        "build",
        "--source",
        str(source_dir),
        "--out",
        str(out_dir),
        "--include-crosswalks",
        "--emit-importer-ready",
    ]
    run_command(cmd, cwd=script_path.parent)


def clean_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)


def copy_csvs(source: Path, target: Path) -> None:
    clean_directory(target)
    for csv_file in sorted(source.glob("*.csv")):
        shutil.copy2(csv_file, target / csv_file.name)


def infer_relationship_type(csv_path: Path) -> Optional[str]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return None
        try:
            type_idx = header.index(":TYPE")
        except ValueError:
            return None
        for row in reader:
            if len(row) > type_idx and row[type_idx]:
                return row[type_idx]
    return None


def build_import_command(
    neo4j_admin: Path,
    neo4j_home: Path,
    import_dir: Path,
    database: str,
) -> list[str]:
    cmd = [
        str(neo4j_admin),
        "database",
        "import",
        "full",
        "--overwrite-destination",
    ]

    node_dir = import_dir / "nodes"
    rel_dir = import_dir / "relationships"

    node_files = list(sorted(node_dir.glob("*.csv")))
    if not node_files:
        raise SystemExit(f"No node CSVs found in {node_dir}")
    for node_file in node_files:
        cmd.append(f"--nodes={node_file.relative_to(neo4j_home).as_posix()}")

    rel_files = list(sorted(rel_dir.glob("*.csv")))
    if not rel_files:
        raise SystemExit(f"No relationship CSVs found in {rel_dir}")
    for rel_file in rel_files:
        rel_type = infer_relationship_type(rel_file)
        rel_arg = (
            f"--relationships={rel_type}={rel_file.relative_to(neo4j_home).as_posix()}"
            if rel_type
            else f"--relationships={rel_file.relative_to(neo4j_home).as_posix()}"
        )
        cmd.append(rel_arg)

    cmd.extend(["--", database])
    return cmd


def create_dump(
    neo4j_admin: Path,
    database: str,
    dump_dir: Path,
    keep_raw: bool,
) -> Path:
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / f"{database}.dump"
    if dump_path.exists():
        dump_path.unlink()

    print(f"Creating dump: {dump_path}")
    cmd = [
        str(neo4j_admin),
        "database",
        "dump",
        database,
        f"--to-path={dump_dir}",
    ]
    run_command(cmd)

    zip_path = dump_dir / f"{database}_dump.zip"
    if zip_path.exists():
        zip_path.unlink()

    print(f"Compressing dump into: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(dump_path, dump_path.name)

    if not keep_raw:
        dump_path.unlink(missing_ok=True)
        print(f"Removed raw dump {dump_path}")

    return zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--neo4j-admin",
        type=Path,
        default=Path("/usr/share/neo4j/bin/neo4j-admin"),
        help="Path to the neo4j-admin executable",
    )
    parser.add_argument(
        "--build-python",
        type=Path,
        default=Path(shutil.which("python3") or sys.executable),
        help="Python interpreter to run Onet.py (default: python3 on PATH)",
    )
    parser.add_argument(
        "--neo4j-home",
        type=Path,
        default=Path("/usr/share/neo4j"),
        help="Neo4j installation directory (default: /usr/share/neo4j)",
    )
    parser.add_argument(
        "--neo4j-data",
        type=Path,
        default=Path("/var/lib/neo4j"),
        help="Neo4j data directory (default: /var/lib/neo4j)",
    )
    parser.add_argument(
        "--service",
        default="neo4j",
        help="Systemd service name to manage (default: neo4j)",
    )
    parser.add_argument(
        "--database",
        default="onet",
        help="Name of the database to dump (default: onet)",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory in which to write the dump and archive",
    )
    parser.add_argument(
        "--excel-zip",
        type=Path,
        default=Path(__file__).resolve().with_name("db_30_0_excel.zip"),
        help="Path to the O*NET Excel archive (default: ./db_30_0_excel.zip)",
    )
    parser.add_argument(
        "--excel-dir",
        type=Path,
        default=Path(__file__).resolve().with_name("db_30_0_excel"),
        help="Directory where the Excel files should exist (default: ./db_30_0_excel)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip the Excel bundle extraction step",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the raw <database>.dump file alongside the zip",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip running Onet.py build",
    )
    parser.add_argument(
        "--skip-import",
        action="store_true",
        help="Skip Neo4j bulk import (assumes database already populated)",
    )
    parser.add_argument(
        "--skip-service",
        action="store_true",
        help="Do not stop/start the Neo4j systemd service",
    )
    parser.add_argument(
        "--restart-service",
        action="store_true",
        help="Restart the Neo4j service after dump creation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if os.geteuid() != 0:
        raise SystemExit("This script must be run as root (sudo).")

    neo4j_admin = ensure_executable(args.neo4j_admin)
    target_dir = args.target_dir.resolve()
    
    base_dir = Path(__file__).resolve().parent
    excel_zip = args.excel_zip.resolve()
    excel_dir = args.excel_dir.resolve()
    csv_dir = base_dir / "neo4j_csv"
    onet_script = base_dir / "Onet.py"

    if not args.skip_extract:
        ensure_excel_bundle(excel_zip, excel_dir)
    elif not excel_dir.exists():
        print(
            f"Warning: Excel directory {excel_dir} missing and extraction was skipped.",
            file=sys.stderr,
        )

    build_python = args.build_python.resolve()

    if not args.skip_build:
        run_onet_build(build_python, onet_script, excel_dir, csv_dir)
    else:
        print("Skipping Onet.py build step")

    nodes_dir = csv_dir / "nodes"
    rel_dir = csv_dir / "relationships"
    if not nodes_dir.exists() or not rel_dir.exists():
        raise SystemExit(
            "Node/relationship CSVs not found. Run Onet.py build or remove --skip-build."
        )

    neo4j_home = args.neo4j_home.resolve()
    neo4j_data = args.neo4j_data.resolve()
    import_dir = neo4j_home / "import"
    import_nodes_dir = import_dir / "nodes"
    import_rels_dir = import_dir / "relationships"

    service_managed = not args.skip_service
    if service_managed:
        print(f"Stopping systemd service {args.service} …")
        run_command(["systemctl", "stop", args.service])

    if not args.skip_import:
        print("Preparing Neo4j import directories …")
        copy_csvs(nodes_dir, import_nodes_dir)
        copy_csvs(rel_dir, import_rels_dir)

        db_path = neo4j_data / "databases" / args.database
        tx_path = neo4j_data / "transactions" / args.database
        for path in (db_path, tx_path):
            if path.exists():
                shutil.rmtree(path)

        print("Running neo4j-admin database import …")
        import_cmd = build_import_command(neo4j_admin, neo4j_home, import_dir, args.database)
        run_command(import_cmd, cwd=neo4j_home)
    else:
        print("Skipping Neo4j import step")

    zip_path = create_dump(neo4j_admin, args.database, target_dir, args.keep_raw)

    if service_managed and args.restart_service:
        print(f"Restarting systemd service {args.service} …")
        run_command(["systemctl", "start", args.service])

    print("\nDump archive ready:", zip_path)
    print("Upload this file via Aura's Backup & Restore to restore the database.")


if __name__ == "__main__":
    main()
