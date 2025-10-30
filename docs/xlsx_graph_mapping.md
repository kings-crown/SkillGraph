# O*NET Excel → SkillGraph Mapping Guide

This reference explains how every Excel workbook consumed by `Onet.py` is transformed into
Neo4j-friendly CSV assets. For each source sheet you will find:

* the target CSV file(s) under `neo4j_csv/`
* the node labels or relationship types created in the graph
* the column-by-column mapping, including data type coercions, filters,
  and any derived properties

The mappings apply to the default pipeline invoked with:

```bash
python3 Onet.py build \
  --source db_30_0_excel \
  --out neo4j_csv \
  --include-crosswalks \
  --emit-importer-ready
```

All column names below refer to the headers in the original Excel workbooks published in
the O*NET 30.x release unless noted otherwise. Derived CSV columns appear in `code` font.

---

## Core Structure

### `Content Model Reference.xlsx`

* **Nodes** → `neo4j_csv/nodes/content_elements.csv`
  * `Element ID` → `element_id:ID(ContentElement)`
  * `Element Name` → `name`
  * `Description` → `description`
  * Computed:
    * `parent_id` via truncating the trailing segment of `Element ID`
    * `depth:int` ← dot-count in `Element ID`
    * `base_type` ← depth bucket (`Domain`, `Category`, `Subcategory`, `Facet`, `Descriptor`)
    * `labels:LABEL` ← `ContentElement;{base_type}` plus extra tags from descriptor/task crosswalks
* **Relationships** → `neo4j_csv/relationships/content_hierarchy.csv`
  * Every row with a non-empty `parent_id` yields `(:ContentElement)-[:HAS_CHILD]->(:ContentElement)`

### `Occupation Data.xlsx`

* **Nodes** → `neo4j_csv/nodes/occupations.csv`
  * `O*NET-SOC Code` → `code:ID(Occupation)`
  * `Title` → `title`
  * `Description` → `description`
  * Derived label: `labels:LABEL = "Occupation"`

---

## Occupation ↔ Descriptor Relationships

Each descriptor sheet contributes a relationship CSV under `neo4j_csv/relationships/`. The
pipeline drops rows flagged with `Recommend Suppress = Y` or `Not Relevant = Y`, converts
`Data Value` to numeric, and pivots `Scale ID` into separate property columns.

| Worksheet | Relationship type | Output CSV | Scale mapping |
|-----------|-------------------|------------|---------------|
| `Abilities.xlsx` | `REQUIRES_ABILITY` | `occupation_requires_ability.csv` | `IM → importance`, `LV → level` |
| `Knowledge.xlsx` | `REQUIRES_KNOWLEDGE` | `occupation_requires_knowledge.csv` | `IM → importance`, `LV → level` |
| `Skills.xlsx` | `REQUIRES_SKILL` | `occupation_requires_skill.csv` | `IM → importance`, `LV → level` |
| `Work Activities.xlsx` | `INVOLVES_ACTIVITY` | `occupation_involves_activity.csv` | `IM → importance`, `LV → level` |
| `Work Context.xlsx` | `OPERATES_IN_CONTEXT` | `occupation_operates_in_context.csv` | `CX → context_score`; unfamiliar scales sanitize to `scale_*` columns (e.g. CT, CTP, CXP) |
| `Work Styles.xlsx` | `VALUES_STYLE` | `occupation_values_style.csv` | `IM → importance` |
| `Work Values.xlsx` | `SUPPORTS_VALUE` | `occupation_supports_value.csv` | `EX → extent` |
| `Interests.xlsx` | `EXPRESSES_INTEREST` | `occupation_expresses_interest.csv` | `OI → interest_score`; additional scales (e.g. IH) become sanitized `scale_*` columns |

Common column handling:

* `O*NET-SOC Code` → `start:START_ID(Occupation)`
* `Element ID` → `end:END_ID(ContentElement)`
* `Element Name` ensures every descriptor element is registered as a `ContentElement` node.
* Optional metadata (`Date`, `Domain Source`, `N`) becomes `date`, `source`, and numeric `sample_size`.
* Additional columns (e.g. `Standard Error`) are preserved when supplied by O*NET.

### Descriptor Crosswalks (emitted with `--include-crosswalks`)

| Worksheet | Relationship type | CSV | Columns |
|-----------|-------------------|-----|---------|
| `Abilities to Work Activities.xlsx` | `ABILITY_SUPPORTS_ACTIVITY` | `descriptor_ability_supports_activity.csv` | Ability element ID ↔ Work Activity element ID |
| `Abilities to Work Context.xlsx` | `ABILITY_INFLUENCES_CONTEXT` | `descriptor_ability_influences_context.csv` | Ability element ID ↔ Work Context element ID |
| `Skills to Work Activities.xlsx` | `SKILL_SUPPORTS_ACTIVITY` | `descriptor_skill_supports_activity.csv` | Skill element ID ↔ Work Activity element ID |
| `Skills to Work Context.xlsx` | `SKILL_RELATES_CONTEXT` | `descriptor_skill_relates_context.csv` | Skill element ID ↔ Work Context element ID |
| `Basic Interests to RIASEC.xlsx` | `BASIC_INTEREST_IN_RIASEC` | `descriptor_basic_interest_in_riasec.csv` | Basic Interest element ID ↔ RIASEC element ID |

Every involved element is registered in `content_elements.csv` with a composite label such
as `ContentElement;DetailedWorkActivity`.

---

## Job Zone Assets

### `Job Zone Reference.xlsx`

* **Nodes** → `neo4j_csv/nodes/job_zones.csv`
  * `Job Zone` → numeric bucket, used to build `job_zone_id:ID(JobZone)` (`job_zone_1` … `job_zone_5`)
  * `Name` → `title`
  * `Experience`, `Education`, `Job Training`, `Examples`, `SVP Range` → stored verbatim
  * `labels:LABEL = "JobZone"`
  * `description` is left empty because the workbook does not provide a narrative field.

### `Job Zones.xlsx`

* **Relationships** → `occupation_has_job_zone.csv` (`HAS_JOB_ZONE`)
  * `O*NET-SOC Code` → `start:START_ID(Occupation)`
  * `Job Zone` → mapped to the same `job_zone_id` as above (`end:END_ID(JobZone)`)
  * `Date` → `date`
  * `Domain Source` → `source`
  * Each occupation keeps the latest entry per SOC code.

---

## Technology & Tools

### `Technology Skills.xlsx`

* **Nodes** → `neo4j_csv/nodes/technologies.csv`
  * `Commodity Code` → `technology_id:ID(Technology)`
  * `Commodity Title` → `name`
  * Label: `Technology`
* **Relationships** → `occupation_uses_technology.csv` (`USES_TECHNOLOGY`)
  * `O*NET-SOC Code` → `start:START_ID(Occupation)`
  * `Commodity Code` → `end:END_ID(Technology)`
  * `Example` → `example`
  * `Hot Technology` / `In Demand` → boolean flags `hot_technology`, `in_demand`
    (the helper `_normalize_flag` maps `Y/N` to `True/False` and preserves blanks as `NULL`).

### `Tools Used.xlsx`

* **Nodes** → `neo4j_csv/nodes/tools.csv`
  * `Commodity Code` → `tool_id:ID(Tool)`
  * `Commodity Title` → `name`
  * Label: `Tool`
* **Relationships** → `occupation_uses_tool.csv` (`USES_TOOL`)
  * `O*NET-SOC Code` → `start:START_ID(Occupation)`
  * `Commodity Code` → `end:END_ID(Tool)`
  * `Example` → `example`

Commodity rows lacking a code or title are filtered out to keep identifiers stable. Each CSV
is deduplicated on `(occupation, commodity)` pairs.

---

## Task Ecosystem

### `Task Statements.xlsx`

* **Nodes** → `neo4j_csv/nodes/tasks.csv`
  * `Task ID` → `task_id:ID(Task)` (prefixed with `task_` to avoid collisions)
  * `Task` → `name`
  * `Task Type` → `task_type`
  * `Incumbents Responding` → `incumbents_responding` (numeric)
  * `Date` → `date`
  * `Domain Source` → `source`
  * Label: `Task`
* For each `Task ID`, the most recent row by `Date` is retained; IDs are sanitised to strings.

### `Task Ratings.xlsx`

* **Relationships** → `occupation_performs_task.csv` (`PERFORMS_TASK`)
  * `O*NET-SOC Code` → `start:START_ID(Occupation)`
  * `Task ID` → `end:END_ID(Task)` (`task_{id}`)
  * `Scale ID` values pivot into:
    * `FT` → `frequency`
    * `IM` → `importance`
    * `RT` → `relevance`
    * Any additional scales are sanitised to `scale_*` columns automatically.
  * Metadata averages (first non-null per pair):
    * `Category` → `category`
    * `N` → numeric `sample_size`
    * `Standard Error` → `standard_error`
    * `Lower CI Bound` → `lower_ci`
    * `Upper CI Bound` → `upper_ci`
    * `Date` → `date`
    * `Domain Source` → `source`
  * Rows with `Recommend Suppress = Y` or non-numeric `Data Value` are discarded before pivoting.

### `Tasks to DWAs.xlsx`

* **Relationships** → `task_aligns_dwa.csv` (`ALIGNS_WITH_DWA`)
  * `Task ID` → `start:START_ID(Task)` (`task_{id}`)
  * `DWA ID` → `end:END_ID(ContentElement)`
  * `Date` → `date`
  * `Domain Source` → `source`
* Side-effect: any DWA referenced but missing from `Content Model Reference.xlsx` is added to
  the content element registry with label `DetailedWorkActivity`. The associated `DWA Title`
  becomes the node name.

---

## Maintained CSV Directory Structure

After `Onet.py build`, the `--out` directory contains:

```
neo4j_csv/
  nodes/
    occupations.csv
    content_elements.csv
    job_zones.csv
    technologies.csv
    tools.csv
    tasks.csv
  relationships/
    content_hierarchy.csv
    occupation_requires_*.csv
    occupation_involves_activity.csv
    occupation_operates_in_context.csv
    occupation_values_style.csv
    occupation_supports_value.csv
    occupation_expresses_interest.csv
    occupation_has_job_zone.csv
    occupation_uses_technology.csv
    occupation_uses_tool.csv
    occupation_performs_task.csv
    task_aligns_dwa.csv
    descriptor_*.csv (crosswalks)
  data_importer_ready/
    *_sanitized.csv (optional, produced with --emit-importer-ready)
```

Sanitized variants rename Neo4j-specific headers (`start:START_ID(...)`, `:TYPE`) to
Data-Importer friendly columns (`start_occupation_id`, `type`, etc.) while preserving the same
payload structure described above.

---

## Handling New O*NET Scales & Columns

When future O*NET releases introduce new `Scale ID` values, `Onet.py` will:

1. Log the unfamiliar codes during the build step.
2. Sanitize them automatically into `scale_<identifier>` property columns so the data is not
   lost.
3. Keep the first available value per `(SOC code, element)` combination.

Additional metadata columns can be surfaced by extending the relevant helper in
`Onet.py`—this guide should be updated in tandem to keep the documentation accurate.

---

## Currently Excluded Workbooks

The O*NET Excel delivery ships additional workbooks that the SkillGraph pipeline does not yet
ingest. Reasons include missing downstream consumers, ambiguous semantics, or the need for
additional modeling work. The table below summarises their contents so you can prioritise
future extensions.

| Workbook | Summary |
| - | - |
| `Alternate Titles.xlsx` | Extended occupation naming variants (alternate/short titles with sources). |
| `Sample of Reported Titles.xlsx` | Titles supplied by incumbents and flags for My Next Move visibility. |
| `Related Occupations.xlsx` | SOC-to-SOC relatedness tiers and composite similarity index. |
| `Education, Training, and Experience.xlsx` | Distributions across education level, related work, and training categories. |
| `Education, Training, and Experience Categories.xlsx` | Code tables for the education/training scales referenced above. |
| `Occupation Level Metadata.xlsx` | Survey metadata (response distributions by item) for each occupation. |
| `Emerging Tasks.xlsx` | Proposed or newly observed tasks with provenance and original task IDs. |
| `Task Categories.xlsx` | Lookup for task rating categories and descriptive text. |
| `Interests Illustrative Activities.xlsx` | Example activities linked to interest element IDs. |
| `Interests Illustrative Occupations.xlsx` | Occupations illustrating each interest element. |
| `RIASEC Keywords.xlsx` | Keyword corpus for interests with keyword type tags. |
| `Survey Booklet Locations.xlsx` | Mapping from element IDs/scales to survey instrument item numbers. |
| `Work Context Categories.xlsx` | Category definitions for Work Context descriptors. |
| `DWA Reference.xlsx`, `IWA Reference.xlsx` | Detailed and intermediate work activity hierarchies and their relationships. |
| `Level Scale Anchors.xlsx` | Narrative anchor descriptions for each scale level (min/max). |
| `Scales Reference.xlsx` | Scale metadata (name, min, max) used across descriptor files. |
| `UNSPSC Reference.xlsx` | Commodity taxonomy hierarchy (segment/family/class) supporting tool/technology codes. |

You can extend `Onet.py` to incorporate any of these sources; update both this section and the
relevant mapping tables above to keep the documentation comprehensive once new imports land.

---

For questions, updates, or corrections, edit both `Onet.py` and this guide to ensure the
pipeline and documentation stay in sync.
