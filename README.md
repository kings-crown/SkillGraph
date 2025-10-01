# O*NET Graph Builder

This project converts the O*NET Excel delivery (v30.x) into Neo4j graph assets and
packages them for local exploration or Neo4j Aura deployment.

## Prerequisites
- Python 3.9+
- `pip install pandas openpyxl`
- Official O*NET Excel bundle (`db_30_0_excel.zip` placed alongside `Onet.py`)
- Local Neo4j 5.x install (needed for bulk import and dump creation)

## Environment Variables
Some downstream tooling (for example Aura access or OpenAI-powered helpers) expects
connection details and API credentials to be exposed as environment variables.

1. Copy `.env.example` to `.env` and replace the placeholder values with your own
   Neo4j URI, password, and OpenAI key:
   ```bash
   cp .env.example .env
   # edit .env with your preferred editor
   ```
2. Make the variables available in your shell before running automation scripts:
   ```bash
   set -a
   source .env
   set +a
   ```
   (You can also export them manually or load them with tools like `direnv`.)
3. Access the values from Python as usual:
   ```python
   import os

   neo4j_uri = os.environ["NEO4J_URI"]
   neo4j_user = os.environ["NEO4J_USERNAME"]
   neo4j_password = os.environ["NEO4J_PASSWORD"]
   openai_key = os.environ["OPENAI_API_KEY"]
   ```

Avoid committing the populated `.env` file; it is already ignored via `.gitignore`.

## 1. Generate Graph Assets
```bash
python3 Onet.py build \
  --source db_30_0_excel \
  --out neo4j_csv \
  --include-crosswalks \
  --emit-importer-ready
```
The script creates:
- `neo4j_csv/nodes/` and `neo4j_csv/relationships/`: CSVs for `neo4j-admin` or `LOAD CSV` imports.
- `neo4j_csv/data_importer_ready/`: sanitized copies with unambiguous headers (`start_occupation_id`, …) for the browser Data Importer.

During extraction it keeps every descriptor scale (including newly introduced ones) and reports
all relationship `:TYPE` values so you can verify coverage.

## 2. Build the Neo4j Database Locally (optional)
1. Copy the generated CSVs into Neo4j’s import directory:
   ```bash
   sudo mkdir -p /usr/share/neo4j/import/{nodes,relationships}
   sudo cp neo4j_csv/nodes/*.csv /usr/share/neo4j/import/nodes/
   sudo cp neo4j_csv/relationships/*.csv /usr/share/neo4j/import/relationships/
   ```
2. Run the bulk importer (target DB must not exist yet):
   ```bash
   cd /usr/share/neo4j
   sudo ./bin/neo4j-admin database import full \
     --overwrite-destination \
     --nodes=Occupation=import/nodes/occupations.csv \
     --nodes=ContentElement=import/nodes/content_elements.csv \
     --relationships=HAS_CHILD=import/relationships/content_hierarchy.csv \
     --relationships=REQUIRES_ABILITY=import/relationships/occupation_requires_ability.csv \
     --relationships=REQUIRES_KNOWLEDGE=import/relationships/occupation_requires_knowledge.csv \
     --relationships=REQUIRES_SKILL=import/relationships/occupation_requires_skill.csv \
     --relationships=INVOLVES_ACTIVITY=import/relationships/occupation_involves_activity.csv \
     --relationships=OPERATES_IN_CONTEXT=import/relationships/occupation_operates_in_context.csv \
     --relationships=VALUES_STYLE=import/relationships/occupation_values_style.csv \
     --relationships=SUPPORTS_VALUE=import/relationships/occupation_supports_value.csv \
     --relationships=EXPRESSES_INTEREST=import/relationships/occupation_expresses_interest.csv \
     --relationships=ABILITY_SUPPORTS_ACTIVITY=import/relationships/descriptor_ability_supports_activity.csv \
     --relationships=ABILITY_INFLUENCES_CONTEXT=import/relationships/descriptor_ability_influences_context.csv \
     --relationships=SKILL_SUPPORTS_ACTIVITY=import/relationships/descriptor_skill_supports_activity.csv \
     --relationships=SKILL_RELATES_CONTEXT=import/relationships/descriptor_skill_relates_context.csv \
     --relationships=BASIC_INTEREST_IN_RIASEC=import/relationships/descriptor_basic_interest_in_riasec.csv \
     -- \
     onet
   ```
3. Start Neo4j and check a few samples:
   ```cypher
   :use onet
   MATCH ()-[r]->() RETURN type(r), count(*) ORDER BY count DESC LIMIT 10;
   MATCH (o:Occupation {code:'11-1011.00'})-[r]->(e) RETURN type(r), e.name, r.importance LIMIT 20;
   ```
4. Extra validation
   - Storage: `sudo ./bin/neo4j-admin database check onet --verbose`
   - Statistics: `CALL db.stats.retrieve('SAMPLE');`
   - Connectivity: ensure every `ContentElement` has at least one relationship.

## 3. Automate Zip Extraction & Dump Creation
Use `create_dump.py` to orchestrate the full pipeline (extract bundle → run `Onet.py build` → bulk import into Neo4j → create dump + zip archive):
```bash
sudo python3 create_dump.py --database onet
```
This performs:
1. Extract `db_30_0_excel.zip` → `db_30_0_excel/` (skipped with `--skip-extract`).
2. Run `Onet.py build --source db_30_0_excel ...` to regenerate the CSVs.
3. Stop the Neo4j service, copy the CSVs into `import/`, wipe the existing `onet` store, and execute the `neo4j-admin database import full …` command.
4. Run `neo4j-admin database dump onet --to-path=.`
5. Create `./onet_dump.zip` (raw `.dump` removed unless `--keep-raw`).

CLI options:
- `--build-python /path/to/python` to select the interpreter that runs `Onet.py` (ensure it has `pandas` and `openpyxl`).
- `--neo4j-admin /path/to/neo4j-admin` if Neo4j is installed elsewhere.
- `--excel-zip` / `--excel-dir` to point to custom bundle locations.
- `--skip-build`, `--skip-import`, or `--skip-service` if you want to run only portions of the pipeline.
- `--restart-service` to bring the Neo4j systemd service back online after dumping.

## 4. Restore the Dump into Neo4j Aura
1. Open [console.neo4j.io](https://console.neo4j.io) and create a **new** Aura instance.
2. In the instance card select **Backup & Restore → Upload Dump**.
3. Upload unzipped `onet_dump.zip`, keep the desired database name, and confirm. Aura restarts with the imported data.
4. Launch Neo4j Browser from the console, run `:use onet`, and execute the validation queries above.

## Alternative: Data Importer Only
If you prefer not to run Neo4j locally:
1. `zip -r data_importer_ready.zip neo4j_csv/data_importer_ready`
2. In Aura → **Data Importer**, upload the archive.
3. Map nodes (`occupations_sanitized.csv`, `content_elements_sanitized.csv`) and every relationship file using the `type` column.
4. Run the import, then validate in the Browser.

## Maintenance Tips
- Rerun `Onet.py build …` whenever O*NET releases an update. The script prints unfamiliar scale IDs so you can extend the descriptor mappings if desired.
- Repeat `create_dump.py` followed by Aura’s Backup & Restore upload to refresh the managed instance.
- Track graph evolution with `CALL db.stats.retrieve('ALL')`, `CALL db.constraints()`, and bespoke Cypher tests.

## Reference Queries
```cypher
// Shared skills between occupations
MATCH (o1:Occupation {code:'11-1011.00'})-[:REQUIRES_SKILL]->(s)<-[:REQUIRES_SKILL]-(o2)
WHERE o2 <> o1
RETURN o2.title, collect(s.name) AS sharedSkills LIMIT 10;

// Descriptor hierarchy walk
MATCH path = (root:ContentElement {element_id:'2.A.1.a'})-[:HAS_CHILD*0..3]->(leaf)
RETURN path LIMIT 5;

// Inspect dynamic scale properties
MATCH (o:Occupation)-[r:OPERATES_IN_CONTEXT]->(ctx)
WHERE r.scale_ct IS NOT NULL
RETURN o.code, ctx.element_id, r.scale_ct, r.scale_ctp, r.scale_cxp LIMIT 10;
```

This workflow lets you generate the graph, validate it locally, and deliver the same dataset in Aura with minimal manual steps.
