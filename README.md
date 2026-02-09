# ERG Heatmap Streamlit app

## What it does

- Upload a CSV (like `activity_CHEMBL202_ERG.csv`).
- Keeps `pchembl_value` (if present) and only feature columns starting with `fr_` or `ERG...`.
- Drops zero-variance feature columns.
- Creates `class` where `pchembl_value >= 6.5` → 1 else 0.
- Sorts compounds descending by `pchembl_value` (fallback: `class`).
- Renders a normalized (0..1) heatmap for the feature matrix.
- Shows the activity column as a separate strip on the left.

## Run

From this folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Then upload a CSV.

---

# ChEMBL36 target-loop model builder (ErG + fragments)

This folder also contains a CLI training pipeline:

- `erg_eda_enhanced_general_CHEMBL36.py`

It loops over ChEMBL targets, fetches activity rows, computes ErG + fragment/topological descriptors from SMILES, and trains per-target models.

## Target (what it does)

For each target in a targets CSV:

1. Fetches activities from either:
	 - a local ChEMBL SQLite database (recommended), or
	 - the ChEMBL Web Services API.
2. Filters rows (defaults):
	 - `pchembl_value > 4.0`
	 - `MW <= 2000`
	 - `standard_type` in `IC50/KI/KD/AC50/EC50/DC50`
3. Splits into assay groups and directions:
	 - assay_type `B` → `binding`, `F` → `functional`
	 - `KI/IC50/DC50` → `inhibition`, `KD/EC50/AC50` → `activation`
4. Deduplicates to one row per `(molecule_chembl_id, canonical_smiles)` using median pChEMBL.
5. Computes descriptors using RDKit + `FixedPharmacophoreAnalyzer` (from `erg_calc_fragments_topo.py`).
6. If the group has enough data (`N > --min-n`, default `50`):
	 - trains regression models to predict pChEMBL
	 - trains classification models to predict active/inactive (`active = pchembl_value > 6.5`)
7. Optionally tunes RF/XGB hyperparameters via Optuna (if installed).
8. Saves models only if quality thresholds are met:
	 - regression: best test R² ≥ 0.7
	 - classification: best test AUC ≥ 0.8

Notes:
- The name contains `eda` but it performs training; it does not generate plots.

## Inputs

### Targets CSV

Provided via `--targets-csv` (default: `Uniprot_ID_with_only_Chembl_teil2.csv`).

Required columns (names can vary; the script searches several candidates):
- one of: `chembl_target_id`, `target_chembl_id`, `target_id`, `chembl_id`
- one of: `uniprot_accession`, `uniprot_id`, `uniprot`, `accession`

Optional:
- one of: `chembl_target_name`, `target_name`, `pref_name`

### ChEMBL source

- SQLite mode (recommended): pass `--chembl-source sqlite --chembl-sqlite path/to/chembl_36.db`
- API mode: pass `--chembl-source api` (uses `--chembl-api-base`, default `https://www.ebi.ac.uk/chembl/api/data`)

## Outputs

All outputs are written under `--out-dir` (default: `output/models`).

- Training report (always written at end):
	- `output/models/model_training_report.csv`

- Per assay group subfolders:
	- `output/models/binding/`
	- `output/models/functional/`

- Feature lists (per target/group):
	- `{CHEMBLID}_{UNIPROT}_{activation|inhibition}_feature_columns.txt`
	- `{CHEMBLID}_{UNIPROT}_{activation|inhibition}_feature_columns.json`

- Saved models (only if thresholds are met):
	- `{CHEMBLID}_{UNIPROT}_{activation|inhibition}_reg_rf.pkl`
	- `{CHEMBLID}_{UNIPROT}_{activation|inhibition}_reg_xgb.pkl` (if XGBoost available)
	- `{CHEMBLID}_{UNIPROT}_{activation|inhibition}_cls_rf.pkl`
	- `{CHEMBLID}_{UNIPROT}_{activation|inhibition}_cls_xgb.pkl` (if XGBoost available)

- Optional dataset exports (if flags enabled):
	- raw activities: `output/datasets/raw/*_activities.csv.gz` (requires `--export-raw-activities`)
	- descriptor datasets: `output/datasets/{binding|functional}/*_descriptors.csv.gz` (requires `--export-descriptor-datasets`)

- Optional API cache (API mode):
	- `output/api_cache/activity_<target>.csv.gz` (controlled by `--api-cache-dir` and `--refresh-api-cache`)

Saved `.pkl` payload includes the fitted model, fitted `StandardScaler`, and ordered `feature_names`.

## Dependencies

The Streamlit heatmap app uses the packages in `requirements.txt`.

The training script (`erg_eda_enhanced_general_CHEMBL36.py`) additionally requires a scientific/cheminformatics stack that is not listed in `requirements.txt`:

- required: `rdkit`, `scikit-learn`
- optional: `requests` (required for API mode), `optuna` (hyperparameter tuning), `xgboost` (XGB models), `imbalanced-learn` (SMOTE)

If you use Conda, installing RDKit via `conda-forge` is usually the easiest.

## Run examples

SQLite mode:

```powershell
python erg_eda_enhanced_general_CHEMBL36.py --chembl-source sqlite --chembl-sqlite C:/path/to/chembl_36.db
```

API mode (no local DB):

```powershell
python erg_eda_enhanced_general_CHEMBL36.py --chembl-source api --targets-csv my_targets.csv
```

Export descriptors only (no training):

```powershell
python erg_eda_enhanced_general_CHEMBL36.py --download-only --export-descriptor-datasets --chembl-source sqlite --chembl-sqlite C:/path/to/chembl_36.db
```

Multiprocessing (CPU):

```powershell
python erg_eda_enhanced_general_CHEMBL36.py --workers 4 --chunk-size 2 --chembl-source sqlite --chembl-sqlite C:/path/to/chembl_36.db
```

GPU note:
- If you run with `--xgb-use-gpu`, the script forces `--workers=1` to avoid GPU contention.

