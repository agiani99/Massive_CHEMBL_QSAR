"""erg_eda_enhanced_with_optuna.py

ChEMBL36 target-loop model builder (ErG + RDKit fragments + Optuna-tuned RF/XGB).

What it does
1) For each target in a targets CSV (default: Example_Target.csv) it extracts activities
   from a *local* ChEMBL36 SQLite database.
2) Splits assays into two assay groups:
   - assay_type = B  -> binding
   - assay_type = F  -> functional
3) Keeps only:
   - MW <= 2000
   - pChEMBL > 4.0
   - standard_type in: IC50/Ki/Kd/AC50/EC50/DC50
   Then splits into:
   - inhibition: Ki/IC50/DC50
   - activation: Kd/EC50/AC50
4) If N > 50 (unique molecules after de-duping), trains Optuna-optimized:
   - regression models predicting pChEMBL (objective: maximize CV R²; accept if test R²>=0.7)
   - classification models (active=1 if pChEMBL>6.5; objective: maximize CV MCC; accept if test AUC>=0.8)
   Uses stratified 80/20 test split for classification. Applies SMOTE on training split
   if minority/majority ratio < 0.4.

No plots / EDA.

Outputs
- Writes models under: output/models/{binding|functional}/
- Filenames match requested pattern (per assay_group folder):
  {CHEMBLID}_{UNIPROT}_{activation|inhibition}_{reg|cls}_{rf|xgb}.pkl
- Also writes: output/models/model_training_report.csv

Saved PKL payload includes: fitted model, fitted StandardScaler, ordered feature_names.

Usage
  python erg_eda_enhanced_with_optuna.py --chembl-sqlite C:/path/to/chembl_36.db

Optional deps
- optuna (for tuning; else defaults)
- xgboost (for xgb models)
- imbalanced-learn (for SMOTE)
"""

from __future__ import annotations

import argparse
import gc
import json
import textwrap
import logging
import os
import pickle
import sqlite3
import sys
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from urllib.parse import urljoin
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple, Iterator

try:
    import tracemalloc

    TRACEMALLOC_AVAILABLE = True
except Exception:
    tracemalloc = None
    TRACEMALLOC_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False

import numpy as np
import pandas as pd
from rdkit import Chem

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import matthews_corrcoef, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore")

try:
    import requests

    REQUESTS_AVAILABLE = True
except Exception:
    requests = None
    REQUESTS_AVAILABLE = False

# Optional dependencies
try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:
    optuna = None
    TPESampler = None
    OPTUNA_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
    try:
        import xgboost as xgboost_module
    except Exception:
        xgboost_module = None
except Exception:
    XGBClassifier = None
    XGBRegressor = None
    XGBOOST_AVAILABLE = False
    xgboost_module = None

try:
    from imblearn.over_sampling import SMOTE

    IMBLEARN_AVAILABLE = True
except Exception:
    SMOTE = None
    IMBLEARN_AVAILABLE = False

from erg_calc_fragments_topo import FixedPharmacophoreAnalyzer


ASSAY_TYPE_MAP = {"B": "binding", "F": "functional"}
INHIBITION_TYPES = {"KI", "IC50", "DC50"}
ACTIVATION_TYPES = {"KD", "EC50", "AC50"}
ALLOWED_STANDARD_TYPES = sorted(INHIBITION_TYPES | ACTIVATION_TYPES)


@dataclass(frozen=True)
class TargetRow:
    chembl_target_id: str
    chembl_target_name: str
    uniprot_accession: str


@dataclass(frozen=True)
class RunConfig:
    chembl_sqlite: Optional[str]
    chembl_source: str
    chembl_api_base: str
    api_cache_dir: Optional[str]
    refresh_api_cache: bool
    api_polite_delay_s: float
    api_max_pages: Optional[int]
    out_dir: str
    export_raw_activities: bool
    export_descriptor_datasets: bool
    download_only: bool
    max_mw: float
    min_pchembl: float
    min_n: int
    n_trials: int
    use_target_name: bool
    cleanup_between_steps: bool
    log_memory: bool
    workers: int
    chunk_size: int
    model_n_jobs: int
    xgb_use_gpu: bool
    xgb_gpu_id: int
    log_level: str


def _process_targets_chunk(targets: List[TargetRow], cfg: RunConfig) -> List[dict]:
    """Process a list of targets sequentially.

    Designed to be run inside a worker process when cfg.workers > 1.
    Returns report rows.
    """

    logger = _setup_logger(cfg.log_level)

    if cfg.log_memory and TRACEMALLOC_AVAILABLE and tracemalloc is not None and not tracemalloc.is_tracing():
        tracemalloc.start(10)

    _ensure_dir(cfg.out_dir)
    if cfg.api_cache_dir:
        _ensure_dir(cfg.api_cache_dir)

    datasets_dir = os.path.join(cfg.out_dir, "..", "datasets")
    if cfg.export_raw_activities or cfg.export_descriptor_datasets:
        _ensure_dir(datasets_dir)

    # Reuse analyzer within process to avoid repeated init/fragment discovery.
    try:
        analyzer = FixedPharmacophoreAnalyzer(verbose=False)
    except TypeError:
        analyzer = FixedPharmacophoreAnalyzer()

    source = cfg.chembl_source.lower().strip()
    conn: Optional[sqlite3.Connection] = None
    if source == "sqlite":
        if not cfg.chembl_sqlite:
            raise ValueError("--chembl-sqlite is required when --chembl-source=sqlite")
        conn = sqlite3.connect(cfg.chembl_sqlite)

    report_rows: List[dict] = []
    try:
        for t in targets:
            target_id = t.chembl_target_id
            if cfg.use_target_name:
                if source == "sqlite":
                    assert conn is not None
                    resolved = resolve_target_id_from_name(conn, t.chembl_target_name)
                else:
                    resolved = resolve_target_id_from_name_api(t.chembl_target_name, cfg.chembl_api_base)
                if resolved is None:
                    logger.warning(f"Target name not found: {t.chembl_target_name}. Skipping.")
                    continue
                target_id = resolved

            logger.info("=" * 80)
            logger.info(f"Target: {target_id} ({t.uniprot_accession})")

            try:
                if source == "sqlite":
                    assert conn is not None
                    df_raw = fetch_target_activities(
                        conn,
                        chembl_target_id=target_id,
                        max_mw=cfg.max_mw,
                        min_pchembl=cfg.min_pchembl,
                        allowed_standard_types=ALLOWED_STANDARD_TYPES,
                        assay_types=("B", "F"),
                    )
                else:
                    df_raw = fetch_target_activities_api(
                        chembl_target_id=target_id,
                        base_url=cfg.chembl_api_base,
                        min_pchembl=cfg.min_pchembl,
                        allowed_standard_types=ALLOWED_STANDARD_TYPES,
                        assay_types=("B", "F"),
                        polite_delay_s=cfg.api_polite_delay_s,
                        cache_dir=cfg.api_cache_dir,
                        refresh_cache=cfg.refresh_api_cache,
                        max_pages=cfg.api_max_pages,
                    )
            except Exception as e:
                logger.warning(f"  Fetch failed for {target_id}: {e}")
                report_rows.append(
                    {
                        "chembl_target_id": target_id,
                        "uniprot_accession": t.uniprot_accession,
                        "assay_group": None,
                        "direction": None,
                        "n": None,
                        "status": "failed_fetch",
                        "error": str(e),
                    }
                )
                continue
            if df_raw.empty:
                logger.info("  No activities after filtering")
                continue

            cleanup_memory(logger, f"{target_id} after fetch", enabled=cfg.cleanup_between_steps, log_memory=cfg.log_memory)

            if cfg.export_raw_activities:
                raw_dir = os.path.join(datasets_dir, "raw")
                _ensure_dir(raw_dir)
                raw_path = os.path.join(
                    raw_dir,
                    f"{_sanitize_token(target_id)}_{_sanitize_token(t.uniprot_accession)}_activities.csv.gz",
                )
                df_raw.to_csv(raw_path, index=False, compression="gzip")

            groups = build_groups(df_raw)
            if not groups:
                logger.info("  No assay_group/direction groups")
                continue

            if cfg.cleanup_between_steps:
                del df_raw
                cleanup_memory(logger, f"{target_id} after build_groups", enabled=True, log_memory=cfg.log_memory)

            for (assay_group, direction), g in groups.items():
                g = dedupe_per_molecule(g)
                n0 = len(g)
                logger.info(f"  Group: {assay_group}/{direction} N={n0}")
                if n0 <= cfg.min_n:
                    report_rows.append(
                        {
                            "chembl_target_id": target_id,
                            "uniprot_accession": t.uniprot_accession,
                            "assay_group": assay_group,
                            "direction": direction,
                            "n": n0,
                            "status": "skipped_N",
                        }
                    )
                    continue

                df_desc = compute_erg_descriptors(g, analyzer=analyzer)
                if df_desc.empty:
                    report_rows.append(
                        {
                            "chembl_target_id": target_id,
                            "uniprot_accession": t.uniprot_accession,
                            "assay_group": assay_group,
                            "direction": direction,
                            "n": n0,
                            "status": "skipped_no_descriptors",
                        }
                    )
                    continue

                cleanup_memory(
                    logger,
                    f"{target_id} {assay_group}/{direction} after descriptors",
                    enabled=cfg.cleanup_between_steps,
                    log_memory=cfg.log_memory,
                )

                if "MW" in df_desc.columns:
                    df_desc = df_desc[pd.to_numeric(df_desc["MW"], errors="coerce") <= float(cfg.max_mw)].copy()
                if len(df_desc) <= cfg.min_n:
                    report_rows.append(
                        {
                            "chembl_target_id": target_id,
                            "uniprot_accession": t.uniprot_accession,
                            "assay_group": assay_group,
                            "direction": direction,
                            "n": int(len(df_desc)),
                            "status": "skipped_N_after_mw_filter",
                        }
                    )
                    continue

                if cfg.export_descriptor_datasets:
                    ds_group_dir = os.path.join(datasets_dir, _sanitize_token(assay_group))
                    _ensure_dir(ds_group_dir)
                    ds_path = os.path.join(
                        ds_group_dir,
                        f"{_sanitize_token(target_id)}_{_sanitize_token(t.uniprot_accession)}_{_sanitize_token(direction)}_descriptors.csv.gz",
                    )
                    df_desc.to_csv(ds_path, index=False, compression="gzip")

                if cfg.download_only:
                    report_rows.append(
                        {
                            "chembl_target_id": target_id,
                            "uniprot_accession": t.uniprot_accession,
                            "assay_group": assay_group,
                            "direction": direction,
                            "n": int(len(df_desc)),
                            "status": "download_only",
                        }
                    )
                    continue

                feat_cols = get_feature_columns(df_desc)
                if len(feat_cols) < 10:
                    report_rows.append(
                        {
                            "chembl_target_id": target_id,
                            "uniprot_accession": t.uniprot_accession,
                            "assay_group": assay_group,
                            "direction": direction,
                            "n": int(len(df_desc)),
                            "status": "skipped_too_few_features",
                        }
                    )
                    continue

                try:
                    reg_out, cls_out = train_for_group(
                        df_desc=df_desc,
                        feature_cols=feat_cols,
                        n_trials=cfg.n_trials,
                        cls_threshold=6.5,
                        model_n_jobs=cfg.model_n_jobs,
                        xgb_use_gpu=cfg.xgb_use_gpu,
                        xgb_gpu_id=cfg.xgb_gpu_id,
                        logger=logger,
                    )
                except Exception as e:
                    logger.exception(f"  Training failed for {assay_group}/{direction}: {e}")
                    report_rows.append(
                        {
                            "chembl_target_id": target_id,
                            "uniprot_accession": t.uniprot_accession,
                            "assay_group": assay_group,
                            "direction": direction,
                            "n": int(len(df_desc)),
                            "status": "failed_training",
                            "error": str(e),
                        }
                    )
                    continue

                cleanup_memory(
                    logger,
                    f"{target_id} {assay_group}/{direction} after training",
                    enabled=cfg.cleanup_between_steps,
                    log_memory=cfg.log_memory,
                )

                reg_r2s = [v for v in [reg_out.get("test_r2_rf"), reg_out.get("test_r2_xgb")] if v is not None]
                best_reg_r2 = max(reg_r2s) if reg_r2s else None

                best_cls_auc = None
                if not cls_out.get("skipped", False):
                    cls_aucs = [v for v in [cls_out.get("test_auc_rf"), cls_out.get("test_auc_xgb")] if v is not None]
                    best_cls_auc = max(cls_aucs) if cls_aucs else None

                group_dir = os.path.join(cfg.out_dir, _sanitize_token(assay_group))
                _ensure_dir(group_dir)

                prefix = f"{_sanitize_token(target_id)}_{_sanitize_token(t.uniprot_accession)}_{_sanitize_token(direction)}"

                feature_cols_path = os.path.join(group_dir, f"{prefix}_feature_columns.txt")
                write_feature_columns_txt(feature_cols_path, list(reg_out["feature_names"]))

                feature_cols_json_path = os.path.join(group_dir, f"{prefix}_feature_columns.json")
                write_feature_columns_json(feature_cols_json_path, list(reg_out["feature_names"]))

                saved_reg = False
                if best_reg_r2 is not None and best_reg_r2 >= 0.7:
                    common = {
                        "chembl_target_id": target_id,
                        "uniprot_accession": t.uniprot_accession,
                        "assay_group": assay_group,
                        "direction": direction,
                        "task": "reg",
                        "feature_names": reg_out["feature_names"],
                        "scaler": reg_out["scaler"],
                        "metrics": {
                            "test_r2_rf": reg_out.get("test_r2_rf"),
                            "test_r2_xgb": reg_out.get("test_r2_xgb"),
                        },
                    }
                    save_payload(
                        os.path.join(group_dir, f"{prefix}_reg_rf.pkl"),
                        {
                            **common,
                            "model_family": "rf",
                            "best_params": reg_out.get("rf_best_params"),
                            "model": reg_out["rf"],
                        },
                    )
                    saved_reg = True
                    if reg_out.get("xgb") is not None:
                        save_payload(
                            os.path.join(group_dir, f"{prefix}_reg_xgb.pkl"),
                            {
                                **common,
                                "model_family": "xgb",
                                "best_params": reg_out.get("xgb_best_params"),
                                "model": reg_out["xgb"],
                            },
                        )
                else:
                    logger.info(f"  Dropping regression models (best test R2={best_reg_r2})")

                saved_cls = False
                if not cls_out.get("skipped", False) and best_cls_auc is not None and best_cls_auc >= 0.8:
                    common = {
                        "chembl_target_id": target_id,
                        "uniprot_accession": t.uniprot_accession,
                        "assay_group": assay_group,
                        "direction": direction,
                        "task": "cls",
                        "feature_names": reg_out["feature_names"],
                        "scaler": reg_out["scaler"],
                        "classification_threshold": 6.5,
                        "smote_applied": cls_out.get("smote_applied"),
                        "metrics": {
                            "test_auc_rf": cls_out.get("test_auc_rf"),
                            "test_auc_xgb": cls_out.get("test_auc_xgb"),
                        },
                    }
                    save_payload(
                        os.path.join(group_dir, f"{prefix}_cls_rf.pkl"),
                        {
                            **common,
                            "model_family": "rf",
                            "best_params": cls_out.get("rf_best_params"),
                            "model": cls_out.get("rf"),
                        },
                    )
                    saved_cls = True
                    if cls_out.get("xgb") is not None:
                        save_payload(
                            os.path.join(group_dir, f"{prefix}_cls_xgb.pkl"),
                            {
                                **common,
                                "model_family": "xgb",
                                "best_params": cls_out.get("xgb_best_params"),
                                "model": cls_out.get("xgb"),
                            },
                        )
                else:
                    if cls_out.get("skipped", False):
                        logger.info(f"  Dropping classification models (skipped: {cls_out.get('reason')})")
                    else:
                        logger.info(f"  Dropping classification models (best test AUC={best_cls_auc})")

                report_rows.append(
                    {
                        "chembl_target_id": target_id,
                        "uniprot_accession": t.uniprot_accession,
                        "assay_group": assay_group,
                        "direction": direction,
                        "n": int(len(df_desc)),
                        "feature_columns_txt": feature_cols_path,
                        "feature_columns_json": feature_cols_json_path,
                        "test_r2_rf": reg_out.get("test_r2_rf"),
                        "test_r2_xgb": reg_out.get("test_r2_xgb"),
                        "best_reg_r2": best_reg_r2,
                        "test_auc_rf": None if cls_out.get("skipped") else cls_out.get("test_auc_rf"),
                        "test_auc_xgb": None if cls_out.get("skipped") else cls_out.get("test_auc_xgb"),
                        "best_cls_auc": best_cls_auc,
                        "cls_skipped": bool(cls_out.get("skipped", False)),
                        "cls_skip_reason": cls_out.get("reason"),
                        "saved_reg": bool(saved_reg),
                        "saved_cls": bool(saved_cls),
                    }
                )

                if cfg.cleanup_between_steps:
                    del g, df_desc, feat_cols, reg_out, cls_out
                    cleanup_memory(
                        logger,
                        f"{target_id} {assay_group}/{direction} end group",
                        enabled=True,
                        log_memory=cfg.log_memory,
                    )

            if cfg.cleanup_between_steps:
                del groups
                cleanup_memory(logger, f"{target_id} end target", enabled=True, log_memory=cfg.log_memory)

    finally:
        if conn is not None:
            conn.close()

    return report_rows


def _setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("chembl_target_loop")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _format_mb(value_mb: Optional[float]) -> str:
    if value_mb is None:
        return "n/a"
    return f"{value_mb:.1f} MB"


def _get_rss_mb() -> Optional[float]:
    """Best-effort RSS (resident set size) in MB.

    Uses psutil if available; otherwise returns None.
    """

    if not PSUTIL_AVAILABLE or psutil is None:
        return None
    try:
        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _get_tracemalloc_mb() -> Tuple[Optional[float], Optional[float]]:
    """(current, peak) python allocation sizes via tracemalloc, in MB."""

    if not TRACEMALLOC_AVAILABLE or tracemalloc is None:
        return None, None
    if not tracemalloc.is_tracing():
        return None, None
    try:
        current, peak = tracemalloc.get_traced_memory()
        return float(current) / (1024.0 * 1024.0), float(peak) / (1024.0 * 1024.0)
    except Exception:
        return None, None


def cleanup_memory(logger: logging.Logger, label: str, enabled: bool, log_memory: bool) -> None:
    """Optional cleanup hook to reduce peak memory between pipeline steps.

    Notes:
    - Python may not return memory to the OS immediately, but this helps drop large objects
      and can reduce swapping/pagefile pressure.
    - `gc.collect()` can cost time; keep it opt-in.
    """

    if not enabled:
        return

    gc.collect()

    if not log_memory:
        return

    rss = _get_rss_mb()
    cur_py, peak_py = _get_tracemalloc_mb()
    logger.info(
        f"[mem] {label}: rss={_format_mb(rss)}; py_cur={_format_mb(cur_py)}; py_peak={_format_mb(peak_py)}"
    )


def validate_chembl_sqlite_schema(conn: sqlite3.Connection) -> None:
    """Fail fast if the SQLite file is not a ChEMBL SQLite distribution."""

    required_tables = {
        "activities",
        "assays",
        "target_dictionary",
        "molecule_dictionary",
        "compound_structures",
        "molecule_properties",
    }
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    present = {str(r[0]).lower() for r in cur.fetchall()}
    missing = sorted(t for t in required_tables if t not in present)
    if not missing:
        return

    msg = textwrap.dedent(
        f"""
        The SQLite file you provided does not look like a ChEMBL SQLite database.
        Missing required table(s): {', '.join(missing)}

        This script expects the official ChEMBL SQLite release (e.g. ChEMBL 36), which contains
        tables like 'activities' and 'assays'. Your file is likely a different SQLite DB.

        Fix:
        - Download the ChEMBL SQLite package for your release (ChEMBL 36) from the ChEMBL/EBI FTP.
        - Extract it; inside you'll find a .db file (often named like 'chembl_36.db').
        - Re-run with: --chembl-sqlite path/to/chembl_36.db

        Windows tip: you can usually extract with:
          tar -xf chembl_36_sqlite.tar.gz
        """
    ).strip()
    raise RuntimeError(msg)


def _http_get_json(session, url: str, params: Optional[dict] = None, timeout_s: int = 60, retries: int = 6) -> dict:
    if session is None:
        raise RuntimeError("requests is required for API mode. Please install it (pip install requests).")

    last_exc: Optional[Exception] = None
    transient_statuses = {429, 500, 502, 503, 504}
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            # Fail fast for non-transient HTTP codes.
            try:
                if requests is not None and isinstance(e, requests.exceptions.HTTPError):
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    if status is not None and int(status) not in transient_statuses:
                        raise
            except Exception:
                # If the "fail fast" branch raised, propagate it.
                raise

            last_exc = e
            base_sleep_s = min(60, 2 ** (attempt - 1))
            # Add a small jitter so parallel workers don't retry in lockstep.
            sleep_s = base_sleep_s + random.uniform(0.0, 1.0)
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to GET {url}: {last_exc}")


def fetch_target_activities_api(
    chembl_target_id: str,
    base_url: str,
    min_pchembl: float,
    allowed_standard_types: Iterable[str],
    assay_types: Iterable[str] = ("B", "F"),
    polite_delay_s: float = 0.1,
    cache_dir: Optional[str] = None,
    refresh_cache: bool = False,
    max_pages: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch activities from ChEMBL Web Services API.

    Uses /activity.json which includes canonical_smiles + assay_type so we can build
    the same dataset as the SQLite path without downloading the full DB.
    """

    cache_path = None
    if cache_dir:
        _ensure_dir(cache_dir)
        cache_path = os.path.join(cache_dir, f"activity_{_sanitize_token(chembl_target_id)}.csv.gz")
        if os.path.exists(cache_path) and not refresh_cache:
            dfc = pd.read_csv(cache_path)
            return dfc

    if not REQUESTS_AVAILABLE:
        raise RuntimeError("API mode requires the 'requests' package.")

    assert requests is not None

    allowed_standard_types = [str(t).upper() for t in allowed_standard_types]
    assay_types = [str(a).upper() for a in assay_types]

    session = requests.Session()

    url = base_url.rstrip("/") + "/activity.json"
    params = {
        "target_chembl_id": chembl_target_id,
        # Try server-side filtering to cut payload size; we also re-filter client-side.
        "pchembl_value__gt": str(float(min_pchembl)),
        "limit": "1000",
    }

    rows: List[dict] = []
    pages = 0
    while True:
        payload = _http_get_json(session, url, params=params)
        activities = payload.get("activities") or []
        for a in activities:
            rows.append(
                {
                    "chembl_target_id": a.get("target_chembl_id") or chembl_target_id,
                    "chembl_target_name": a.get("target_pref_name"),
                    "molecule_chembl_id": a.get("molecule_chembl_id"),
                    "canonical_smiles": a.get("canonical_smiles"),
                    "mw": np.nan,
                    "assay_type": a.get("assay_type"),
                    "standard_type": a.get("standard_type"),
                    "pchembl_value": a.get("pchembl_value"),
                }
            )

        pages += 1
        if max_pages is not None and pages >= max_pages:
            break

        page_meta = payload.get("page_meta") or {}
        next_rel = page_meta.get("next")
        if not next_rel:
            break

        # next is a relative path like: /chembl/api/data/activity.json?...offset=...
        url = urljoin(base_url, next_rel)
        params = None
        time.sleep(polite_delay_s)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["standard_type"] = df["standard_type"].astype(str).str.upper().str.strip()
    df["assay_type"] = df["assay_type"].astype(str).str.upper().str.strip()
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")

    df = df.dropna(subset=["pchembl_value", "canonical_smiles", "molecule_chembl_id"]).copy()
    df = df[df["pchembl_value"] > float(min_pchembl)].copy()
    df = df[df["standard_type"].isin([t.upper() for t in allowed_standard_types])].copy()
    df = df[df["assay_type"].isin([a.upper() for a in assay_types])].copy()

    if cache_path and not df.empty:
        _ensure_dir(os.path.dirname(cache_path))
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        df.to_csv(tmp_path, index=False, compression="gzip")
        os.replace(tmp_path, cache_path)
    return df


def _chunked(items: List[TargetRow], chunk_size: int) -> Iterator[List[TargetRow]]:
    chunk_size = max(1, int(chunk_size))
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _xgb_device_params(use_gpu: bool, gpu_id: int) -> Dict:
    if not use_gpu:
        return {}
    gid = int(gpu_id)
    # XGBoost>=2 prefers device='cuda[:id]' + tree_method='hist'.
    return {
        "device": f"cuda:{gid}",
        "tree_method": "hist",
    }


def _sanitize_token(s: str) -> str:
    keep = "-_.()"
    s = str(s)
    return "".join(c for c in s if c.isalnum() or c in keep).strip() or "NA"


def read_targets_csv(path: str) -> List[TargetRow]:
    df = pd.read_csv(path)

    def _pick_col(candidates: List[str]) -> Optional[str]:
        cols = {str(c): c for c in df.columns}
        for name in candidates:
            if name in cols:
                return cols[name]
        return None

    col_target_id = _pick_col(["chembl_target_id", "target_chembl_id", "target_id", "chembl_id"])
    col_uniprot = _pick_col(["uniprot_accession", "uniprot_id", "uniprot", "accession"])
    col_target_name = _pick_col(
        [
            "chembl_target_name",
            "chembl_target_name (First)",
            "target_name",
            "pref_name",
        ]
    )

    missing_required: List[str] = []
    if col_target_id is None:
        missing_required.append("chembl_target_id")
    if col_uniprot is None:
        missing_required.append("uniprot_accession")
    if missing_required:
        raise ValueError(
            "Targets CSV missing required columns. "
            f"Need at least {missing_required}. Found columns: {list(df.columns)}"
        )

    assert col_target_id is not None
    assert col_uniprot is not None
    col_target_id_s: str = str(col_target_id)
    col_uniprot_s: str = str(col_uniprot)

    rows: List[TargetRow] = []
    for _, r in df.iterrows():
        rows.append(
            TargetRow(
                chembl_target_id=str(r[col_target_id_s]).strip(),
                chembl_target_name="" if col_target_name is None else str(r[col_target_name]).strip(),
                uniprot_accession=str(r[col_uniprot_s]).strip(),
            )
        )

    dedup: Dict[Tuple[str, str], TargetRow] = {}
    for r in rows:
        dedup[(r.chembl_target_id, r.uniprot_accession)] = r
    return list(dedup.values())


def resolve_target_id_from_name(conn: sqlite3.Connection, target_name: str) -> Optional[str]:
    q = """
        SELECT target_chembl_id
        FROM target_dictionary
        WHERE pref_name = ?
        LIMIT 1
    """
    cur = conn.execute(q, (target_name,))
    row = cur.fetchone()
    return row[0] if row else None


def resolve_target_id_from_name_api(target_name: str, base_url: str) -> Optional[str]:
    if not REQUESTS_AVAILABLE:
        return None
    assert requests is not None
    session = requests.Session()
    url = base_url.rstrip("/") + "/target/search.json"
    payload = _http_get_json(session, url, params={"q": target_name, "limit": "1"})
    targets = payload.get("targets") or []
    if not targets:
        return None
    return targets[0].get("target_chembl_id")


def fetch_target_activities(
    conn: sqlite3.Connection,
    chembl_target_id: str,
    max_mw: float,
    min_pchembl: float,
    allowed_standard_types: Iterable[str],
    assay_types: Iterable[str] = ("B", "F"),
) -> pd.DataFrame:
    allowed_standard_types = [str(t).upper() for t in allowed_standard_types]
    assay_types = [str(a).upper() for a in assay_types]

    pt = ",".join(["?"] * len(allowed_standard_types))
    pa = ",".join(["?"] * len(assay_types))

    q = f"""
        SELECT
            td.target_chembl_id AS chembl_target_id,
            td.pref_name        AS chembl_target_name,
            md.molecule_chembl_id,
            cs.canonical_smiles,
            mp.full_mwt         AS mw,
            a.assay_type,
            act.standard_type,
            act.pchembl_value
        FROM activities act
        JOIN assays a
            ON act.assay_id = a.assay_id
        JOIN target_dictionary td
            ON a.tid = td.tid
        JOIN molecule_dictionary md
            ON act.molregno = md.molregno
        LEFT JOIN compound_structures cs
            ON md.molregno = cs.molregno
        LEFT JOIN molecule_properties mp
            ON md.molregno = mp.molregno
        WHERE
            td.target_chembl_id = ?
            AND act.pchembl_value IS NOT NULL
            AND act.pchembl_value > ?
            AND UPPER(act.standard_type) IN ({pt})
            AND UPPER(a.assay_type) IN ({pa})
            AND (mp.full_mwt IS NULL OR mp.full_mwt <= ?)
            AND cs.canonical_smiles IS NOT NULL
            AND TRIM(cs.canonical_smiles) <> ''
    """

    params: Tuple[object, ...] = (
        chembl_target_id,
        float(min_pchembl),
        *allowed_standard_types,
        *assay_types,
        float(max_mw),
    )

    df = pd.read_sql_query(q, conn, params=params)
    df["standard_type"] = df["standard_type"].astype(str).str.upper().str.strip()
    df["assay_type"] = df["assay_type"].astype(str).str.upper().str.strip()
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value", "canonical_smiles", "molecule_chembl_id"]).copy()
    return df


def assign_direction(standard_type: str) -> Optional[str]:
    st = str(standard_type).upper().strip()
    if st in INHIBITION_TYPES:
        return "inhibition"
    if st in ACTIVATION_TYPES:
        return "activation"
    return None


def build_groups(df: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    if df.empty:
        return out
    # Mutate in-place to avoid a full DataFrame copy (df is short-lived in the caller).
    df["assay_group"] = df["assay_type"].map(ASSAY_TYPE_MAP)
    df["direction"] = df["standard_type"].map(assign_direction)
    df = df.dropna(subset=["assay_group", "direction"]).copy()
    for (assay_group, direction), g in df.groupby(["assay_group", "direction"], dropna=True):
        out[(str(assay_group), str(direction))] = g.reset_index(drop=True)
    return out


def dedupe_per_molecule(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(["molecule_chembl_id", "canonical_smiles"], as_index=False)
        .agg(
            pchembl_value=("pchembl_value", "median"),
            mw=("mw", "median"),
            chembl_target_id=("chembl_target_id", "first"),
            chembl_target_name=("chembl_target_name", "first"),
            assay_type=("assay_type", "first"),
            assay_group=("assay_group", "first"),
            direction=("direction", "first"),
        )
        .reset_index(drop=True)
    )


def compute_erg_descriptors(df: pd.DataFrame, analyzer: Optional[FixedPharmacophoreAnalyzer] = None) -> pd.DataFrame:
    if df.empty:
        return df
    analyzer = analyzer or FixedPharmacophoreAnalyzer()

    # Build rows only for valid molecules to avoid holding both a full copy of df and fps.
    keep_idx: List[int] = []
    fps: List[dict] = []
    smiles_values = df["canonical_smiles"].astype(str).to_numpy()

    for i, smi in enumerate(smiles_values):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = analyzer.generate_erg_fingerprint(mol, smi)
        if fp is None:
            continue
        keep_idx.append(i)
        fps.append(fp)

    if not fps:
        return pd.DataFrame()

    fp_df = pd.DataFrame(fps).drop(columns=["SMILES"], errors="ignore")
    base = df.iloc[keep_idx].reset_index(drop=True).copy()
    out = pd.concat([base, fp_df.reset_index(drop=True)], axis=1)

    # Ensure all descriptor columns from erg_calc_fragments_topo.py are numeric.
    # Any missing values (rare for valid RDKit mols) are set to 0.
    descriptor_cols = [c for c in fp_df.columns if c in out.columns]
    for c in descriptor_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[descriptor_cols] = out[descriptor_cols].fillna(0)
    return out


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    # Use the *full* numeric descriptor set generated by FixedPharmacophoreAnalyzer
    # (from erg_calc_fragments_topo.py). This is robust to future descriptor additions.
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    excluded = {"pchembl_value", "mw"}
    feat_cols = [c for c in numeric_cols if c not in excluded]
    return feat_cols


def _remove_zero_variance(X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    mask = np.var(X, axis=0) > 1e-8
    return X[:, mask], [f for f, ok in zip(feature_names, mask) if ok]


def _should_apply_smote(y_train: np.ndarray, ratio_threshold: float = 0.4) -> bool:
    if SMOTE is None:
        return False
    uniq, cnt = np.unique(y_train, return_counts=True)
    if len(uniq) < 2:
        return False
    minority, majority = cnt.min(), cnt.max()
    return majority > 0 and minority >= 2 and (minority / majority) < ratio_threshold


def _maybe_smote(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    if not _should_apply_smote(y_train):
        return X_train, y_train, False
    assert SMOTE is not None

    class_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    minority = min(class_counts.values())
    k_neighbors = min(5, minority - 1)
    if k_neighbors < 1:
        return X_train, y_train, False

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    resampled = smote.fit_resample(X_train, y_train)
    X_res = resampled[0]
    y_res = resampled[1]
    return np.asarray(X_res), np.asarray(y_res), True


def _optuna_sampler():
    return TPESampler(seed=42) if TPESampler is not None else None


def tune_rf_reg(X: np.ndarray, y: np.ndarray, n_trials: int, n_jobs: int) -> Dict:
    if optuna is None:
        return {"n_estimators": 600, "max_depth": 12, "max_features": "sqrt"}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": 42,
            "n_jobs": int(n_jobs),
        }
        model = RandomForestRegressor(**params)
        cv = KFold(n_splits=min(5, max(2, len(y) // 20)), shuffle=True, random_state=42)
        scores: List[float] = []
        for tr, va in cv.split(X):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[va])
            scores.append(r2_score(y[va], pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=_optuna_sampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgb_reg(X: np.ndarray, y: np.ndarray, n_trials: int, n_jobs: int, xgb_device_params: Dict) -> Dict:
    if optuna is None or XGBRegressor is None:
        return {
            "learning_rate": 0.05,
            "max_depth": 6,
            "n_estimators": 600,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        }

    XGBRegressorLocal = XGBRegressor

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": int(n_jobs),
            "verbosity": 0,
            **(xgb_device_params or {}),
        }
        model = XGBRegressorLocal(**params)
        cv = KFold(n_splits=min(5, max(2, len(y) // 20)), shuffle=True, random_state=42)
        scores: List[float] = []
        for tr, va in cv.split(X):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[va])
            scores.append(r2_score(y[va], pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=_optuna_sampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_rf_cls_mcc(X: np.ndarray, y: np.ndarray, n_trials: int, n_jobs: int) -> Dict:
    if optuna is None:
        return {"n_estimators": 600, "max_depth": 12, "max_features": "sqrt"}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": 42,
            "n_jobs": int(n_jobs),
        }
        cv = StratifiedKFold(n_splits=min(5, np.bincount(y).min()), shuffle=True, random_state=42)
        scores: List[float] = []
        for tr, va in cv.split(X, y):
            X_tr, y_tr = X[tr], y[tr]
            X_va, y_va = X[va], y[va]
            X_tr2, y_tr2, _ = _maybe_smote(X_tr, y_tr)
            model = RandomForestClassifier(**params)
            model.fit(X_tr2, y_tr2)
            pred = model.predict(X_va)
            scores.append(matthews_corrcoef(y_va, pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=_optuna_sampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgb_cls_mcc(X: np.ndarray, y: np.ndarray, n_trials: int, n_jobs: int, xgb_device_params: Dict) -> Dict:
    if optuna is None or XGBClassifier is None:
        return {
            "learning_rate": 0.05,
            "max_depth": 6,
            "n_estimators": 600,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        }

    XGBClassifierLocal = XGBClassifier

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": int(n_jobs),
            "eval_metric": "logloss",
            "verbosity": 0,
            **(xgb_device_params or {}),
        }
        cv = StratifiedKFold(n_splits=min(5, np.bincount(y).min()), shuffle=True, random_state=42)
        scores: List[float] = []
        for tr, va in cv.split(X, y):
            X_tr, y_tr = X[tr], y[tr]
            X_va, y_va = X[va], y[va]
            X_tr2, y_tr2, _ = _maybe_smote(X_tr, y_tr)
            model = XGBClassifierLocal(**params)
            model.fit(X_tr2, y_tr2)
            pred = model.predict(X_va)
            scores.append(matthews_corrcoef(y_va, pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=_optuna_sampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_for_group(
    df_desc: pd.DataFrame,
    feature_cols: List[str],
    n_trials: int,
    cls_threshold: float,
    model_n_jobs: int,
    xgb_use_gpu: bool,
    xgb_gpu_id: int,
    logger: Optional[logging.Logger] = None,
) -> Tuple[dict, dict]:
    y_reg = df_desc["pchembl_value"].to_numpy(dtype=float)
    X = df_desc[feature_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs, feature_cols2 = _remove_zero_variance(Xs, feature_cols)
    if Xs.shape[1] < 5:
        raise ValueError("Too few usable features after zero-variance removal")

    Xtr, Xte, ytr, yte = train_test_split(Xs, y_reg, test_size=0.2, random_state=42)

    rf_reg_params = tune_rf_reg(Xtr, ytr, n_trials, n_jobs=model_n_jobs)
    rf_reg = RandomForestRegressor(**rf_reg_params, random_state=42, n_jobs=model_n_jobs)
    rf_reg.fit(Xtr, ytr)
    r2_rf = float(r2_score(yte, rf_reg.predict(Xte)))

    xgb_reg = None
    xgb_reg_params = None
    r2_xgb = None
    if XGBRegressor is not None:
        xgb_device_params = _xgb_device_params(use_gpu=bool(xgb_use_gpu), gpu_id=int(xgb_gpu_id))
        xgb_reg_params = tune_xgb_reg(
            Xtr, ytr, n_trials, n_jobs=model_n_jobs, xgb_device_params=xgb_device_params
        )
        try:
            xgb_reg = XGBRegressor(
                **xgb_reg_params,
                random_state=42,
                n_jobs=model_n_jobs,
                verbosity=0,
                **(xgb_device_params or {}),
            )
            xgb_reg.fit(Xtr, ytr)
            r2_xgb = float(r2_score(yte, xgb_reg.predict(Xte)))
        except Exception as e:
            if xgb_use_gpu and logger is not None:
                logger.warning(f"XGBoost GPU failed; falling back to CPU: {e}")
            xgb_reg = XGBRegressor(**xgb_reg_params, random_state=42, n_jobs=model_n_jobs, verbosity=0)
            xgb_reg.fit(Xtr, ytr)
            r2_xgb = float(r2_score(yte, xgb_reg.predict(Xte)))

    reg_out = {
        "scaler": scaler,
        "feature_names": feature_cols2,
        "rf": rf_reg,
        "rf_best_params": rf_reg_params,
        "test_r2_rf": r2_rf,
        "xgb": xgb_reg,
        "xgb_best_params": xgb_reg_params,
        "test_r2_xgb": r2_xgb,
    }

    y_cls = (y_reg > cls_threshold).astype(int)
    if len(np.unique(y_cls)) < 2:
        return reg_out, {"skipped": True, "reason": "single_class"}

    try:
        Xtrc, Xtec, ytrc, ytec = train_test_split(
            Xs, y_cls, test_size=0.2, random_state=42, stratify=y_cls
        )
    except ValueError:
        return reg_out, {"skipped": True, "reason": "cannot_stratify"}

    Xtrc2, ytrc2, smote_applied = _maybe_smote(Xtrc, ytrc)

    rf_cls_params = tune_rf_cls_mcc(Xtrc, ytrc, n_trials, n_jobs=model_n_jobs)
    rf_cls = RandomForestClassifier(**rf_cls_params, random_state=42, n_jobs=model_n_jobs)
    rf_cls.fit(Xtrc2, ytrc2)
    auc_rf = float(roc_auc_score(ytec, rf_cls.predict_proba(Xtec)[:, 1]))

    xgb_cls = None
    xgb_cls_params = None
    auc_xgb = None
    if XGBClassifier is not None:
        xgb_device_params = _xgb_device_params(use_gpu=bool(xgb_use_gpu), gpu_id=int(xgb_gpu_id))
        xgb_cls_params = tune_xgb_cls_mcc(
            Xtrc, ytrc, n_trials, n_jobs=model_n_jobs, xgb_device_params=xgb_device_params
        )
        try:
            xgb_cls = XGBClassifier(
                **xgb_cls_params,
                random_state=42,
                n_jobs=model_n_jobs,
                eval_metric="logloss",
                verbosity=0,
                **(xgb_device_params or {}),
            )
            xgb_cls.fit(Xtrc2, ytrc2)
            auc_xgb = float(roc_auc_score(ytec, xgb_cls.predict_proba(Xtec)[:, 1]))
        except Exception as e:
            if xgb_use_gpu and logger is not None:
                logger.warning(f"XGBoost GPU failed; falling back to CPU: {e}")
            xgb_cls = XGBClassifier(
                **xgb_cls_params,
                random_state=42,
                n_jobs=model_n_jobs,
                eval_metric="logloss",
                verbosity=0,
            )
            xgb_cls.fit(Xtrc2, ytrc2)
            auc_xgb = float(roc_auc_score(ytec, xgb_cls.predict_proba(Xtec)[:, 1]))

    cls_out = {
        "skipped": False,
        "smote_applied": bool(smote_applied),
        "threshold": cls_threshold,
        "rf": rf_cls,
        "rf_best_params": rf_cls_params,
        "test_auc_rf": auc_rf,
        "xgb": xgb_cls,
        "xgb_best_params": xgb_cls_params,
        "test_auc_xgb": auc_xgb,
    }
    return reg_out, cls_out


def save_payload(path: str, payload: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def write_feature_columns_txt(path: str, feature_names: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for name in feature_names:
            f.write(f"{name}\n")


def write_feature_columns_json(path: str, feature_names: List[str]) -> None:
    payload = {
        "feature_names": list(feature_names),
        "n_features": int(len(feature_names)),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run(
    targets_csv: str,
    chembl_sqlite: Optional[str],
    chembl_source: str,
    chembl_api_base: str,
    api_cache_dir: Optional[str],
    refresh_api_cache: bool,
    api_polite_delay_s: float,
    api_max_pages: Optional[int],
    out_dir: str,
    export_raw_activities: bool,
    export_descriptor_datasets: bool,
    download_only: bool,
    max_mw: float,
    min_pchembl: float,
    min_n: int,
    n_trials: int,
    use_target_name: bool,
    cleanup_between_steps: bool,
    log_memory: bool,
    workers: int,
    chunk_size: int,
    model_n_jobs: int,
    xgb_use_gpu: bool,
    xgb_gpu_id: int,
    log_level: str,
) -> None:
    logger = _setup_logger(log_level)

    # Normalize paths to avoid cwd surprises (common when running from different shells/IDEs).
    targets_csv = os.path.abspath(str(targets_csv))
    out_dir = os.path.abspath(str(out_dir))
    if chembl_sqlite:
        chembl_sqlite = os.path.abspath(str(chembl_sqlite))
    if api_cache_dir:
        api_cache_dir = os.path.abspath(str(api_cache_dir))

    _ensure_dir(out_dir)
    if api_cache_dir:
        _ensure_dir(api_cache_dir)

    if log_memory and TRACEMALLOC_AVAILABLE and tracemalloc is not None and not tracemalloc.is_tracing():
        # Keep small stack depth to reduce overhead; enough to identify top allocators when needed.
        tracemalloc.start(10)

    logger.info(f"Targets CSV: {targets_csv}")
    logger.info(f"ChEMBL source: {chembl_source}")
    if chembl_source in {"sqlite", "auto"}:
        logger.info(f"ChEMBL SQLite: {chembl_sqlite}")
    if chembl_source in {"api", "auto"}:
        logger.info(f"ChEMBL API base: {chembl_api_base}")
    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Filters: MW<={max_mw}, pChEMBL>{min_pchembl}, N>{min_n}")
    logger.info(f"Optuna available={OPTUNA_AVAILABLE} (trials={n_trials})")
    logger.info(f"XGBoost available={XGBOOST_AVAILABLE}; SMOTE available={IMBLEARN_AVAILABLE}")

    targets = read_targets_csv(targets_csv)
    if not targets:
        raise RuntimeError("No targets loaded")

    started = datetime.now()

    # Decide source
    source = chembl_source.lower().strip()
    if source == "auto":
        source = "sqlite" if chembl_sqlite else "api"

    if source not in {"sqlite", "api"}:
        raise ValueError("--chembl-source must be one of: auto, sqlite, api")

    # Normalize execution knobs
    workers_i = max(1, int(workers))
    chunk_size_i = max(1, int(chunk_size))

    xgb_use_gpu_b = bool(xgb_use_gpu)
    if xgb_use_gpu_b and not XGBOOST_AVAILABLE:
        logger.warning("--xgb-use-gpu requested but XGBoost is not available; disabling GPU.")
        xgb_use_gpu_b = False

    if xgb_use_gpu_b and workers_i > 1:
        logger.warning("--xgb-use-gpu with --workers>1 can cause GPU contention; forcing --workers=1")
        workers_i = 1

    model_n_jobs_i = int(model_n_jobs)
    if model_n_jobs_i == 0:
        model_n_jobs_i = 1 if workers_i > 1 else -1

    cfg = RunConfig(
        chembl_sqlite=chembl_sqlite,
        chembl_source=source,
        chembl_api_base=chembl_api_base,
        api_cache_dir=api_cache_dir,
        refresh_api_cache=bool(refresh_api_cache),
        api_polite_delay_s=float(api_polite_delay_s),
        api_max_pages=api_max_pages,
        out_dir=out_dir,
        export_raw_activities=bool(export_raw_activities),
        export_descriptor_datasets=bool(export_descriptor_datasets),
        download_only=bool(download_only),
        max_mw=float(max_mw),
        min_pchembl=float(min_pchembl),
        min_n=int(min_n),
        n_trials=int(n_trials),
        use_target_name=bool(use_target_name),
        cleanup_between_steps=bool(cleanup_between_steps),
        log_memory=bool(log_memory),
        workers=workers_i,
        chunk_size=chunk_size_i,
        model_n_jobs=model_n_jobs_i,
        xgb_use_gpu=bool(xgb_use_gpu_b),
        xgb_gpu_id=int(xgb_gpu_id),
        log_level=str(log_level),
    )

    # Fail fast on SQLite schema in the parent.
    if source == "sqlite":
        if not chembl_sqlite:
            raise ValueError("--chembl-sqlite is required when --chembl-source=sqlite")
        conn = sqlite3.connect(chembl_sqlite)
        try:
            validate_chembl_sqlite_schema(conn)
        finally:
            conn.close()

    report_rows: List[dict] = []
    if cfg.workers <= 1:
        report_rows = _process_targets_chunk(targets, cfg)
    else:
        chunks = list(_chunked(targets, cfg.chunk_size))
        logger.info(
            f"Parallel enabled: workers={cfg.workers}, chunks={len(chunks)}, chunk_size={cfg.chunk_size}, model_n_jobs={cfg.model_n_jobs}"
        )
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futures = {ex.submit(_process_targets_chunk, chunk, cfg): i for i, chunk in enumerate(chunks, start=1)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    rows = fut.result()
                except Exception as e:
                    logger.exception(f"Chunk {idx} failed: {e}")
                    rows = [{"status": "failed_chunk", "chunk": idx, "error": str(e)}]
                report_rows.extend(rows)
                logger.info(f"Completed chunk {idx}/{len(chunks)}")

    report = pd.DataFrame(report_rows)
    report_path = os.path.join(out_dir, "model_training_report.csv")
    report.to_csv(report_path, index=False)
    logger.info("=" * 80)
    logger.info(f"Report written: {report_path}")
    logger.info(f"Elapsed: {datetime.now() - started}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch train per-target RF/XGB models from ChEMBL36 SQLite")
    p.add_argument("--targets-csv", default="Uniprot_ID_with_only_Chembl_teil2.csv")
    p.add_argument(
        "--chembl-source",
        default="auto",
        choices=["auto", "sqlite", "api"],
        help="Data source: auto uses sqlite if --chembl-sqlite is provided, else api",
    )
    p.add_argument("--chembl-sqlite", required=False, help="Path to chembl_36.db (SQLite)")
    p.add_argument(
        "--chembl-api-base",
        default="https://www.ebi.ac.uk/chembl/api/data",
        help="ChEMBL API base URL for API mode",
    )
    p.add_argument(
        "--api-cache-dir",
        default=os.path.join("output", "api_cache"),
        help="Cache directory for API mode (csv.gz per target)",
    )
    p.add_argument("--refresh-api-cache", action="store_true", help="Ignore cached API responses")
    p.add_argument("--api-polite-delay-s", type=float, default=0.1, help="Delay between API pages")
    p.add_argument(
        "--api-max-pages",
        type=int,
        default=None,
        help="Optional cap on API pagination pages (debug / to reduce downloads)",
    )
    p.add_argument("--out-dir", default=os.path.join("output", "models"))
    p.add_argument("--export-raw-activities", action="store_true", help="Write raw activity rows per target to output/datasets/raw")
    p.add_argument(
        "--export-descriptor-datasets",
        action="store_true",
        help="Write per-group descriptor datasets (ErG+fr_*) to output/datasets/{binding|functional}",
    )
    p.add_argument(
        "--download-only",
        action="store_true",
        help="Only download/export datasets (no model training/saving).",
    )
    p.add_argument("--max-mw", type=float, default=2000.0)
    p.add_argument("--min-pchembl", type=float, default=4.0)
    p.add_argument("--min-n", type=int, default=50, help="Train only if N > min_n")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (per-target chunk). Use 1 to disable multiprocessing.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Targets per worker task (reduce overhead if >1; keep 1 for best load balancing).",
    )
    p.add_argument(
        "--model-n-jobs",
        type=int,
        default=0,
        help="Threads per model. 0=auto (1 if workers>1 else -1).",
    )
    p.add_argument(
        "--xgb-use-gpu",
        action="store_true",
        help="Use GPU for XGBoost (requires CUDA build). Forces --workers=1 to avoid GPU contention.",
    )
    p.add_argument("--xgb-gpu-id", type=int, default=0, help="GPU id for XGBoost when --xgb-use-gpu")
    p.add_argument("--use-target-name", action="store_true", help="Resolve target_id from chembl_target_name (pref_name)")
    p.add_argument(
        "--cleanup-memory",
        action="store_true",
        help="Run gc + drop large objects between targets/groups (helps avoid swapping; can cost time)",
    )
    p.add_argument(
        "--log-memory",
        action="store_true",
        help="Log RSS (if psutil installed) and Python allocs (tracemalloc) at cleanup checkpoints",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run(
        targets_csv=args.targets_csv,
        chembl_sqlite=args.chembl_sqlite,
        chembl_source=str(args.chembl_source),
        chembl_api_base=str(args.chembl_api_base),
        api_cache_dir=str(args.api_cache_dir) if args.api_cache_dir else None,
        refresh_api_cache=bool(args.refresh_api_cache),
        api_polite_delay_s=float(args.api_polite_delay_s),
        api_max_pages=int(args.api_max_pages) if args.api_max_pages is not None else None,
        out_dir=args.out_dir,
        export_raw_activities=bool(args.export_raw_activities),
        export_descriptor_datasets=bool(args.export_descriptor_datasets),
        download_only=bool(args.download_only),
        max_mw=args.max_mw,
        min_pchembl=args.min_pchembl,
        min_n=args.min_n,
        n_trials=args.n_trials,
        workers=int(args.workers),
        chunk_size=int(args.chunk_size),
        model_n_jobs=int(args.model_n_jobs),
        xgb_use_gpu=bool(args.xgb_use_gpu),
        xgb_gpu_id=int(args.xgb_gpu_id),
        use_target_name=bool(args.use_target_name),
        cleanup_between_steps=bool(args.cleanup_memory),
        log_memory=bool(args.log_memory),
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()

