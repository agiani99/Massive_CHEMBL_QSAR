"""Predict and evaluate selected models on DELTA biological data.

This script:
- Reads DELTA_biological_data.csv (SMILES + pChEMBL + CLASS)
- Reads delta_vs_paper_both__model_selection.csv (selected PKL models per target)
- Generates ErG/topological + RDKit fragment descriptors from SMILES
- Loads each model bundle PKL and runs predictions
- Writes per-molecule predictions and per-model evaluation metrics
 - Writes per-molecule predictions and per-model evaluation metrics
 - Adds an applicability-domain label per prediction (MACCS keys + Tanimoto)

Notes
- Models are stored under a models root containing (at least) subfolders: binding/ and functional/
- Many PKLs are dict "bundles" with keys like: model, scaler, feature_names, classification_threshold
- Descriptor generation requires RDKit.

Typical usage:
  python predict_and_evaluate_delta.py \
    --models-root ..\\output\\filtered\\models

Outputs (by default) are created next to this script.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _default_out_paths(script_dir: Path, out_prefix: str) -> Tuple[Path, Path]:
    prefix = out_prefix
    if not prefix:
        prefix = "DELTA_prediction_comparison_"
    return (
        (script_dir / f"{prefix}predictions.csv").resolve(),
        (script_dir / f"{prefix}metrics.csv").resolve(),
    )


def _derive_active_label_from_pchembl(pchembl: pd.Series, cutoff: float) -> pd.Series:
    y = pd.to_numeric(pchembl, errors="coerce")
    return (y >= float(cutoff)).astype("Int64")


def _choose_true_label(df: pd.DataFrame, label_source: str, pchembl_cutoff: float) -> pd.Series:
    src = str(label_source).strip().lower()
    if src == "class":
        if "CLASS" not in df.columns:
            raise ValueError("Predictions data is missing CLASS column")
        return pd.to_numeric(df["CLASS"], errors="coerce").astype("Int64")
    if src == "pchembl":
        if "pChEMBL" not in df.columns:
            raise ValueError("Predictions data is missing pChEMBL column")
        return _derive_active_label_from_pchembl(df["pChEMBL"], cutoff=pchembl_cutoff)
    raise ValueError(f"Unknown label_source: {label_source!r} (use 'class' or 'pchembl')")


def _recompute_metrics_from_predictions(
    pred_df: pd.DataFrame,
    label_source: str,
    pchembl_cutoff: float,
    proba_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Recompute per-model metrics using the saved predictions file.

    This is useful when you want to change the evaluation label definition
    (e.g. use pChEMBL>=6.5) without rerunning descriptors/models.
    """

    if pred_df.empty:
        return pd.DataFrame()

    # Ensure consistent types
    pred_df = pred_df.copy()
    if "Model_Type" not in pred_df.columns:
        pred_df["Model_Type"] = ""
    if "Activity_Type" not in pred_df.columns:
        pred_df["Activity_Type"] = ""
    pred_df["Model_Type"] = pred_df["Model_Type"].astype(str)
    pred_df["Activity_Type"] = pred_df["Activity_Type"].astype(str)

    group_cols = [
        "TARGET ID",
        "TARGET NAME",
        "chembl_target_id",
        "uniprot_accession",
        "Activity_Type",
        "Model_Type",
        "Model_Name",
        "PKL_File",
    ]
    for col in group_cols:
        if col not in pred_df.columns:
            pred_df[col] = ""

    rows: List[Dict[str, Any]] = []
    for keys, g in pred_df.groupby(group_cols, dropna=False):
        (
            target_id,
            target_name,
            chembl_target_id,
            uniprot_accession,
            activity_type,
            model_type,
            model_name,
            pkl_file,
        ) = keys

        model_type_norm = str(model_type).strip().lower()
        out: Dict[str, Any] = {
            "target_id": str(target_id),
            "target_name": str(target_name),
            "chembl_target_id": str(chembl_target_id),
            "uniprot_accession": str(uniprot_accession),
            "activity_type": str(activity_type),
            "model_type": str(model_type),
            "model_name": str(model_name),
            "pkl_file": str(pkl_file),
            "n_rows": int(len(g)),
            "n_ok_descriptors": int(len(g)),
            "n_predictions": int(len(g)),
            "spearman": None,
            "auc": None,
            "tn": None,
            "fp": None,
            "fn": None,
            "tp": None,
            "error": "",
            "eval_label_source": str(label_source),
            "eval_pchembl_cutoff": float(pchembl_cutoff),
            "eval_proba_threshold": (None if proba_threshold is None else float(proba_threshold)),
        }

        try:
            if model_type_norm.startswith("reg"):
                if "pChEMBL" not in g.columns or "y_pred" not in g.columns:
                    raise ValueError("missing pChEMBL or y_pred")
                y_true = pd.to_numeric(g["pChEMBL"], errors="coerce").to_numpy(dtype=float)
                y_pred = pd.to_numeric(g["y_pred"], errors="coerce").to_numpy(dtype=float)
                mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
                out["spearman"] = _spearman(y_true[mask], y_pred[mask])
            elif model_type_norm.startswith("class"):
                y_true_s = _choose_true_label(g, label_source=label_source, pchembl_cutoff=pchembl_cutoff)
                # Choose predicted labels
                if proba_threshold is not None and "y_score" in g.columns:
                    y_score = pd.to_numeric(g["y_score"], errors="coerce")
                    y_pred = (y_score >= float(proba_threshold)).astype("Int64")
                else:
                    if "y_pred" not in g.columns:
                        raise ValueError("missing y_pred")
                    y_pred = pd.to_numeric(g["y_pred"], errors="coerce").astype("Int64")

                y_true = y_true_s.dropna().astype(int).to_numpy()
                y_pred = y_pred.loc[y_true_s.notna()].astype(int).to_numpy()

                if len(y_true) > 0:
                    tn, fp, fn, tp = _confusion(y_true, y_pred)
                    out["tn"], out["fp"], out["fn"], out["tp"] = tn, fp, fn, tp

                if "y_score" in g.columns:
                    y_score = pd.to_numeric(g["y_score"], errors="coerce")
                    y_score = y_score.loc[y_true_s.notna()].dropna().to_numpy(dtype=float)
                    y_true_for_auc = y_true_s.loc[y_true_s.notna()].astype(int).to_numpy()
                    if len(y_score) == len(y_true_for_auc) and len(y_score) > 0:
                        out["auc"] = _roc_auc(y_true_for_auc, y_score)
            else:
                out["error"] = f"unknown_model_type: {model_type}"
        except Exception as exc:
            out["error"] = f"recompute_error: {type(exc).__name__}: {str(exc)[:200]}"

        rows.append(out)

    return pd.DataFrame(rows)


def _make_plots(
    pred_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    script_dir: Path,
    out_prefix: str,
    *,
    label_source: str = "class",
    pchembl_cutoff: float = 6.5,
    proba_threshold: Optional[float] = None,
) -> None:
    """Create lightweight plots to visualize fit quality.

    Produces PNGs next to the CSV outputs.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    prefix = out_prefix or "DELTA_prediction_comparison_"

    def save_fig(name: str) -> None:
        out_path = (script_dir / f"{prefix}{name}.png").resolve()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    # Normalize a few columns defensively
    if "Model_Type" in pred_df.columns:
        pred_df["Model_Type"] = pred_df["Model_Type"].astype(str)
    if "Activity_Type" in metrics_df.columns:
        metrics_df["activity_type"] = metrics_df["activity_type"].astype(str)

    # Targets that have at least one regression AND one classification model
    has_reg = metrics_df["model_type"].astype(str).str.lower().str.startswith("reg")
    has_cls = metrics_df["model_type"].astype(str).str.lower().str.startswith("class")
    reg_targets = set(metrics_df.loc[has_reg, "chembl_target_id"].astype(str).dropna().unique())
    cls_targets = set(metrics_df.loc[has_cls, "chembl_target_id"].astype(str).dropna().unique())
    both_targets = reg_targets.intersection(cls_targets)

    # -----------------
    # Regression plots
    # -----------------
    reg = pred_df[pred_df["Model_Type"].str.lower().str.startswith("reg")].copy() if not pred_df.empty else pred_df
    if not reg.empty:
        if "pChEMBL" in reg.columns and "y_pred" in reg.columns:
            reg["pChEMBL"] = pd.to_numeric(reg["pChEMBL"], errors="coerce")
            reg["y_pred"] = pd.to_numeric(reg["y_pred"], errors="coerce")
            reg = reg.dropna(subset=["pChEMBL", "y_pred"]).copy()
        else:
            reg = pd.DataFrame()

    if reg is not None and len(reg) > 10:
        plt.figure(figsize=(7, 6))
        plt.hexbin(reg["pChEMBL"], reg["y_pred"], gridsize=45, cmap="viridis", mincnt=1)
        plt.colorbar(label="count")
        min_v = float(np.nanmin([reg["pChEMBL"].min(), reg["y_pred"].min()]))
        max_v = float(np.nanmax([reg["pChEMBL"].max(), reg["y_pred"].max()]))
        plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="white", linewidth=1)
        plt.xlabel("Real pChEMBL")
        plt.ylabel("Predicted")
        plt.title("Regression: predicted vs real (all models)")
        save_fig("regression_pred_vs_real_hexbin")

        # Spearman heatmap (top 50 targets by n_rows)
        reg_m = metrics_df[metrics_df["model_type"].astype(str).str.lower().str.startswith("reg")].copy()
        if "spearman" in reg_m.columns:
            reg_m["spearman"] = pd.to_numeric(reg_m["spearman"], errors="coerce")
        if "n_rows" in reg_m.columns:
            reg_m["n_rows"] = pd.to_numeric(reg_m["n_rows"], errors="coerce")
        reg_m = reg_m.dropna(subset=["spearman"]).copy()
        if not reg_m.empty:
            # Keep only targets that also have classification models
            reg_m = reg_m[reg_m["chembl_target_id"].astype(str).isin(both_targets)].copy()
            if not reg_m.empty:
                # One value per (chembl_target_id, activity_type): take max Spearman across regression models
                reg_best = (
                    reg_m.groupby(["chembl_target_id", "activity_type"], dropna=False)["spearman"]
                    .max()
                    .reset_index()
                )
                # Select top targets by data size (n_rows from any row)
                top_targets = (
                    reg_m.groupby("chembl_target_id", dropna=False)["n_rows"]
                    .max()
                    .sort_values(ascending=False)
                    .head(50)
                    .index
                )
                reg_best = reg_best[reg_best["chembl_target_id"].isin(top_targets)].copy()
                pivot = reg_best.pivot(index="chembl_target_id", columns="activity_type", values="spearman")
                if pivot.shape[0] >= 2 and pivot.shape[1] >= 1:
                    plt.figure(figsize=(10, max(6, int(0.22 * pivot.shape[0]))))
                    sns.heatmap(pivot, vmin=-1, vmax=1, center=0, cmap="vlag", linewidths=0.2, linecolor="white")
                    plt.title("Regression Spearman (targets with both reg+cls; top 50; best per activity)")
                    plt.xlabel("Activity_Type")
                    plt.ylabel("chembl_target_id")
                    save_fig("spearman_heatmap_top50")

    # -----------------
    # Classification plots
    # -----------------
    cls = pred_df[pred_df["Model_Type"].str.lower().str.startswith("class")].copy() if not pred_df.empty else pred_df
    if not cls.empty:
        if "CLASS" in cls.columns and "y_pred" in cls.columns:
            cls["CLASS"] = pd.to_numeric(cls["CLASS"], errors="coerce")
            cls["y_pred"] = pd.to_numeric(cls["y_pred"], errors="coerce")
            cls = cls.dropna(subset=["CLASS", "y_pred"]).copy()
        else:
            cls = pd.DataFrame()

    if cls is not None and len(cls) > 10:
        # Aggregate confusion matrix across all classification predictions
        try:
            from sklearn.metrics import confusion_matrix

            y_true_s = _choose_true_label(cls, label_source=label_source, pchembl_cutoff=pchembl_cutoff)
            cls_eval = cls.loc[y_true_s.notna()].copy()
            y_true = y_true_s.loc[y_true_s.notna()].astype(int).to_numpy()
            if proba_threshold is not None and "y_score" in cls_eval.columns:
                y_score = pd.to_numeric(cls_eval["y_score"], errors="coerce").to_numpy(dtype=float)
                y_pred = (y_score >= float(proba_threshold)).astype(int)
            else:
                y_pred = pd.to_numeric(cls_eval["y_pred"], errors="coerce").astype(int).to_numpy()
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            plt.figure(figsize=(4.5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xticks([0.5, 1.5], ["Pred 0", "Pred 1"])
            plt.yticks([0.5, 1.5], ["True 0", "True 1"], rotation=0)
            title = f"Classification: confusion matrix (all models; y_true={label_source}"
            if str(label_source).lower() == "pchembl":
                title += f" cutoff={pchembl_cutoff}"
            if proba_threshold is not None:
                title += f"; p>= {float(proba_threshold):.2f}"
            title += ")"
            plt.title(title)
            save_fig("classification_confusion_all")
        except Exception:
            pass

        # AUC heatmap (top 50 targets by n_rows)
        cls_m = metrics_df[metrics_df["model_type"].astype(str).str.lower().str.startswith("class")].copy()
        if "auc" in cls_m.columns:
            cls_m["auc"] = pd.to_numeric(cls_m["auc"], errors="coerce")
        if "n_rows" in cls_m.columns:
            cls_m["n_rows"] = pd.to_numeric(cls_m["n_rows"], errors="coerce")
        cls_m = cls_m.dropna(subset=["auc"]).copy()
        if not cls_m.empty:
            # Keep only targets that also have regression models
            cls_m = cls_m[cls_m["chembl_target_id"].astype(str).isin(both_targets)].copy()
            if not cls_m.empty:
                cls_best = (
                    cls_m.groupby(["chembl_target_id", "activity_type"], dropna=False)["auc"].max().reset_index()
                )
                top_targets = (
                    cls_m.groupby("chembl_target_id", dropna=False)["n_rows"]
                    .max()
                    .sort_values(ascending=False)
                    .head(50)
                    .index
                )
                cls_best = cls_best[cls_best["chembl_target_id"].isin(top_targets)].copy()
                pivot = cls_best.pivot(index="chembl_target_id", columns="activity_type", values="auc")
                if pivot.shape[0] >= 2 and pivot.shape[1] >= 1:
                    plt.figure(figsize=(10, max(6, int(0.22 * pivot.shape[0]))))
                    sns.heatmap(pivot, vmin=0.5, vmax=1.0, cmap="YlGnBu", linewidths=0.2, linecolor="white")
                    plt.title("Classification AUC (targets with both reg+cls; top 50; best per activity)")
                    plt.xlabel("Activity_Type")
                    plt.ylabel("chembl_target_id")
                    save_fig("auc_heatmap_top50")


def _require_rdkit() -> None:
    try:
        import rdkit  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "RDKit is required to generate descriptors but is not available.\n"
            "Install options:\n"
            "- conda: conda install -c conda-forge rdkit\n"
            "- pip (may be limited): pip install rdkit-pypi\n"
        ) from exc


def _resolve_api_cache_training_file(
    api_cache_root: Path,
    *,
    uniprot_accession: str | None = None,
    chembl_target_id: str | None = None,
    activity_type: str | None = None,
) -> Optional[Path]:
    if api_cache_root is None or not api_cache_root.exists():
        return None

    uni = ("" if uniprot_accession is None else str(uniprot_accession).strip())
    chembl = ("" if chembl_target_id is None else str(chembl_target_id).strip())
    act = ("" if activity_type is None else str(activity_type).strip())

    # Common naming conventions observed in this repo:
    # - <uniprot>.csv.gz (as originally expected)
    # - <chembl_target_id>.csv.gz
    # - activity_<chembl_target_id>.csv.gz
    # - activity_<chembl_target_id>_<activity_type>.csv.gz (future-proof)
    base_names: List[str] = []
    if uni:
        base_names.append(uni)
    if chembl:
        base_names.append(chembl)
        base_names.append(f"activity_{chembl}")
        if act:
            base_names.append(f"activity_{chembl}_{act}")

    # First: exact matches.
    for base in base_names:
        for ext in (".csv.gz", ".csv"):
            p = api_cache_root / f"{base}{ext}"
            if p.exists():
                return p

    # Second: prefix matches.
    for base in base_names:
        try:
            globbed = list(api_cache_root.glob(f"{base}*.csv*"))
        except Exception:
            globbed = []
        if globbed:
            return sorted(globbed, key=lambda p: (len(p.name), p.name.lower()))[0]

    # Third: contains matches (slower; last resort).
    for token in [t for t in (uni, chembl) if t]:
        try:
            globbed = list(api_cache_root.glob(f"*{token}*.csv*"))
        except Exception:
            globbed = []
        if globbed:
            return sorted(globbed, key=lambda p: (len(p.name), p.name.lower()))[0]

    return None


def _read_training_smiles_from_api_cache(
    api_cache_root: Path,
    *,
    uniprot_accession: str | None = None,
    chembl_target_id: str | None = None,
    activity_type: str | None = None,
) -> tuple[List[str], Optional[Path]]:
    """Load training-set SMILES for a target/model from api_cache.

    The cache is expected to contain one CSV/CSV.GZ per UniProt accession.
    This function is intentionally defensive about filenames and SMILES column names.
    """

    path = _resolve_api_cache_training_file(
        api_cache_root,
        uniprot_accession=uniprot_accession,
        chembl_target_id=chembl_target_id,
        activity_type=activity_type,
    )

    if path is None or not path.exists():
        return [], None

    try:
        df = pd.read_csv(path, compression="infer", low_memory=False)
    except Exception:
        return [], path

    if df.empty:
        return [], path

    smiles_col: Optional[str] = None
    for col in df.columns:
        c = str(col).strip().lower()
        if c == "smiles" or c.endswith("smiles") or "smiles" in c:
            smiles_col = str(col)
            break
    if smiles_col is None:
        # Fallback: take the first column.
        smiles_col = str(df.columns[0])

    smiles = (
        df[smiles_col]
        .dropna()
        .astype(str)
        .map(lambda s: s.strip())
    )
    smiles = smiles[smiles != ""]
    # Keep stable order but unique
    seen: set[str] = set()
    out: List[str] = []
    for s in smiles.tolist():
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out, path


def _maccs_fps_from_smiles(smiles_list: List[str]) -> List[Any]:
    """Return RDKit MACCS fingerprints (or None for invalid SMILES) for each input."""

    _require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys as _maccskeys

    fps: List[Any] = []
    for smi in smiles_list:
        smi_s = "" if smi is None else str(smi).strip()
        if not smi_s:
            fps.append(None)
            continue
        mol = Chem.MolFromSmiles(smi_s)
        if mol is None:
            fps.append(None)
            continue
        try:
            gen = getattr(_maccskeys, "GenMACCSKeys", None)
            if gen is None:
                raise AttributeError("RDKit MACCSkeys.GenMACCSKeys is not available")
            fps.append(gen(mol))
        except Exception:
            fps.append(None)
    return fps


def _min_pairwise_similarity_within_training_set(train_fps: List[Any]) -> Optional[float]:
    """Compute the global minimum pairwise Tanimoto similarity across the training set.

    This follows the user's spec: take the minimum similarity from the full all-vs-all
    similarity matrix (excluding self-similarity).
    """

    if train_fps is None or len(train_fps) < 2:
        return None

    # Filter out any None fingerprints defensively.
    fps = [fp for fp in train_fps if fp is not None]
    if len(fps) < 2:
        return None

    _require_rdkit()
    from rdkit import DataStructs

    global_min = 1.0
    n = len(fps)
    for i in range(n - 1):
        # Similarities to the remaining fingerprints only.
        try:
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        except Exception:
            continue
        if not sims:
            continue
        try:
            local_min = float(min(sims))
        except Exception:
            continue
        if np.isfinite(local_min) and local_min < global_min:
            global_min = local_min
            if global_min <= 0.0:
                return 0.0

    if not np.isfinite(global_min):
        return None
    return float(global_min)


def _nearest_neighbor_similarities_within_training_set(train_fps: List[Any]) -> List[float]:
    """For each training molecule i, compute s_i = max_{j!=i} Tanimoto(i, j)."""

    if train_fps is None or len(train_fps) < 2:
        return []

    fps = [fp for fp in train_fps if fp is not None]
    if len(fps) < 2:
        return []

    _require_rdkit()
    from rdkit import DataStructs

    n = len(fps)
    nn_sims: List[float] = []
    for i in range(n):
        try:
            sims = list(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
        except Exception:
            continue
        if len(sims) != n:
            continue
        sims[i] = -1.0
        try:
            nn = float(max(sims))
        except Exception:
            continue
        if np.isfinite(nn):
            nn_sims.append(nn)

    return nn_sims


def _applicability_domain_labels(
    *,
    query_fps: List[Any],
    api_cache_root: Path,
    uniprot_accession: str,
    chembl_target_id: str,
    activity_type: str,
    ad_percentile: Optional[float],
    cache: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Label each query molecule as INSIDE/OUTSIDE/UNKNOWN for the model's domain."""

    uni = str(uniprot_accession or "").strip()
    chembl = str(chembl_target_id or "").strip()
    act = str(activity_type or "").strip()

    # Cache per resolved training file (preferred), else by identifiers.
    cache_key = f"{uni}|{chembl}|{act}".strip("|")

    train_smiles: List[str] = []
    used_path: Optional[Path] = None

    cached = cache.get(cache_key)
    if cached is None:
        train_smiles, used_path = _read_training_smiles_from_api_cache(
            api_cache_root,
            uniprot_accession=uni or None,
            chembl_target_id=chembl or None,
            activity_type=act or None,
        )
        if used_path is not None:
            cache_key = str(used_path.resolve())
            cached = cache.get(cache_key)
            if cached is not None:
                # Already computed for this training file.
                train_fps = cached.get("train_fps") or []
                thr = cached.get("threshold")
                if not train_fps or thr is None or not np.isfinite(float(thr)):
                    return ["UNKNOWN"] * len(query_fps)
                # Continue to labeling below.
            else:
                cached = None

    if cached is None:
        train_fps = _maccs_fps_from_smiles(train_smiles) if train_smiles else []
        train_fps = [fp for fp in train_fps if fp is not None]

        # Choose cutoff definition.
        method = "min_pairwise"
        thr: Optional[float]
        min_sim = _min_pairwise_similarity_within_training_set(train_fps)
        nn_sims: List[float] = []
        if ad_percentile is not None:
            try:
                p = float(ad_percentile)
            except Exception:
                p = float("nan")
            if np.isfinite(p) and 0.0 <= p <= 100.0:
                nn_sims = _nearest_neighbor_similarities_within_training_set(train_fps)
                if nn_sims:
                    method = f"nn_percentile_{p:g}"
                    thr = float(np.percentile(np.asarray(nn_sims, dtype=float), p))
                else:
                    thr = None
            else:
                thr = None
        else:
            # Default: strict cutoff based on training-set global minimum pairwise similarity.
            thr = (None if min_sim is None else float(min_sim))

        cached = {
            "train_fps": train_fps,
            "min_similarity": min_sim,
            "nn_sims": nn_sims,
            "threshold": thr,
            "method": method,
        }
        cache[cache_key] = cached

    train_fps = cached.get("train_fps") or []
    thr = cached.get("threshold")

    if not train_fps or thr is None or not np.isfinite(float(thr)):
        return ["UNKNOWN"] * len(query_fps)

    _require_rdkit()
    from rdkit import DataStructs

    threshold = float(thr)
    labels: List[str] = []
    for qfp in query_fps:
        if qfp is None:
            labels.append("UNKNOWN")
            continue
        try:
            sims = DataStructs.BulkTanimotoSimilarity(qfp, train_fps)
            max_sim = float(max(sims)) if sims else float("nan")
        except Exception:
            labels.append("UNKNOWN")
            continue

        if not np.isfinite(max_sim):
            labels.append("UNKNOWN")
        elif max_sim < threshold:
            labels.append("OUTSIDE")
        else:
            labels.append("INSIDE")

    return labels


def _safe_is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _load_model_bundle(pkl_path: Path) -> Dict[str, Any]:
    try:
        import joblib
    except Exception as exc:  # pragma: no cover
        raise SystemExit("joblib is required to load .pkl models. Try: pip install joblib") from exc

    obj = joblib.load(pkl_path)
    if isinstance(obj, dict):
        return obj
    # Wrap plain estimators into a bundle-like dict
    return {"model": obj}


def _extract_bundle_fields(bundle: Dict[str, Any]) -> Tuple[Any, Any, Optional[List[str]], Optional[float]]:
    model = bundle.get("model")
    if model is None:
        # Some bundles might store the estimator under a different key; best effort.
        for alt_key in ("estimator", "clf", "reg"):
            if alt_key in bundle:
                model = bundle[alt_key]
                break
    scaler = bundle.get("scaler")
    feature_names = bundle.get("feature_names")
    if isinstance(feature_names, (np.ndarray, tuple)):
        feature_names = list(feature_names)
    if not isinstance(feature_names, list):
        feature_names = None
    threshold = bundle.get("classification_threshold")
    try:
        threshold = float(threshold) if threshold is not None else None
    except Exception:
        threshold = None
    return model, scaler, feature_names, threshold


def _find_pkl(models_root: Path, pkl_file: str) -> Optional[Path]:
    """Find PKL file under models_root/binding or models_root/functional (or anywhere under models_root)."""
    direct_candidates = [
        models_root / "binding" / pkl_file,
        models_root / "functional" / pkl_file,
        models_root / pkl_file,
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate
    # Fallback: recursive search (slower but robust)
    try:
        for path in models_root.rglob(pkl_file):
            if path.name.lower() == pkl_file.lower():
                return path
    except Exception:
        return None
    return None


def _calc_descriptors_from_smiles(smiles_list: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute ErG+fragment descriptors for each SMILES.

    Returns:
      - descriptors_df: numeric descriptor matrix (rows align to smiles_list)
      - ok_mask: boolean series True where descriptor generation succeeded
    """

    _require_rdkit()
    from rdkit import Chem

    # Local import from this repo
    from erg_calc_fragments_topo import FixedPharmacophoreAnalyzer

    analyzer = FixedPharmacophoreAnalyzer(verbose=False)

    fingerprints: List[Optional[Dict[str, Any]]] = []
    ok: List[bool] = []

    for smiles in smiles_list:
        if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
            fingerprints.append(None)
            ok.append(False)
            continue
        smiles_str = str(smiles).strip()
        if not smiles_str:
            fingerprints.append(None)
            ok.append(False)
            continue
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            fingerprints.append(None)
            ok.append(False)
            continue
        fp = analyzer.generate_erg_fingerprint(mol, smiles_str)
        if fp is None:
            fingerprints.append(None)
            ok.append(False)
            continue
        fp = dict(fp)
        fp.pop("SMILES", None)
        fingerprints.append(fp)
        ok.append(True)

    ok_mask = pd.Series(ok)

    # Build dataframe for all rows; missing rows become NaN then filled with 0
    # Column order is driven by the first successful fingerprint (consistent with training code).
    first_fp = next((fp for fp in fingerprints if fp is not None), None)
    if first_fp is None:
        return pd.DataFrame(index=range(len(smiles_list))), ok_mask

    ordered_cols = list(first_fp.keys())

    # Create df with stable column order; union any new cols defensively.
    df = pd.DataFrame([{k: (fp.get(k) if fp is not None else np.nan) for k in ordered_cols} for fp in fingerprints])

    # Add any extra columns observed later (unlikely if generator is consistent)
    extra_cols: List[str] = []
    for fp in fingerprints:
        if fp is None:
            continue
        for k in fp.keys():
            if k not in df.columns:
                extra_cols.append(k)
    for k in extra_cols:
        df[k] = [fp.get(k) if fp is not None else np.nan for fp in fingerprints]

    # Ensure numeric dtype; fragment descriptors and distance bins should be numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df, ok_mask


def _build_X_for_bundle(
    descriptors: pd.DataFrame,
    model: Any,
    scaler: Any,
    feature_names: Optional[List[str]],
) -> np.ndarray:
    """Align descriptors to model expectations, applying scaler if present."""

    # If a scaler exists and was fit on a DataFrame, use its recorded feature names.
    scaler_feature_names: Optional[List[str]] = None
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        try:
            scaler_feature_names = list(getattr(scaler, "feature_names_in_"))
        except Exception:
            scaler_feature_names = None

    if scaler is not None and scaler_feature_names is not None:
        X_full = descriptors.reindex(columns=scaler_feature_names, fill_value=0.0)
        X_full_scaled = scaler.transform(X_full)

        if feature_names:
            # Select model features (subset of full features) by name
            name_to_idx = {name: idx for idx, name in enumerate(scaler_feature_names)}
            indices = [name_to_idx.get(name, None) for name in feature_names]

            # Build X_model with missing features filled with 0
            X_model = np.zeros((X_full_scaled.shape[0], len(feature_names)), dtype=float)
            for j, idx in enumerate(indices):
                if idx is None:
                    continue
                X_model[:, j] = X_full_scaled[:, idx]
            return X_model

        return np.asarray(X_full_scaled)

    # Scaler exists but doesn't expose feature names. If it was fit on the full
    # descriptor matrix, we can still scale everything and then subset by name.
    if scaler is not None:
        n_in = getattr(scaler, "n_features_in_", None)
        if n_in is not None and int(n_in) == int(descriptors.shape[1]):
            X_full_scaled = np.asarray(scaler.transform(descriptors), dtype=float)
            if feature_names:
                missing = [c for c in feature_names if c not in descriptors.columns]
                if missing:
                    raise ValueError(
                        f"feature_names contains {len(missing)} columns not present in computed descriptors; "
                        f"example: {missing[0]!r}"
                    )
                X_scaled_df = pd.DataFrame(X_full_scaled, columns=descriptors.columns)
                return X_scaled_df.loc[:, feature_names].to_numpy(dtype=float)
            return X_full_scaled

    # Otherwise: scaler without names (or no scaler). Best-effort based on n_features.
    if feature_names:
        X_sub = descriptors.reindex(columns=feature_names, fill_value=0.0)
        if scaler is None:
            return X_sub.to_numpy(dtype=float)

        n_in = getattr(scaler, "n_features_in_", None)
        if n_in is not None and int(n_in) != X_sub.shape[1]:
            raise ValueError(
                f"Scaler expects {n_in} features, but feature_names provides {X_sub.shape[1]}. "
                "Cannot safely align features."
            )
        return np.asarray(scaler.transform(X_sub), dtype=float)

    # No feature_names: try to transform raw descriptor matrix
    if scaler is None:
        return descriptors.to_numpy(dtype=float)

    n_in = getattr(scaler, "n_features_in_", None)
    if n_in is not None and int(n_in) != descriptors.shape[1]:
        raise ValueError(
            f"Scaler expects {n_in} features, but computed {descriptors.shape[1]} descriptors. "
            "Cannot safely align features."
        )
    return np.asarray(scaler.transform(descriptors), dtype=float)


def _predict_for_model(
    model: Any,
    X: np.ndarray,
    model_type: str,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (y_pred, y_score/proba_if_available)."""

    model_type_norm = str(model_type).strip().lower()

    if model_type_norm.startswith("class"):
        y_score = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                y_score = proba[:, 1]
            else:
                y_score = proba.ravel()
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X)

        if y_score is not None:
            thr = 0.5 if threshold is None else float(threshold)
            y_pred = (np.asarray(y_score) >= thr).astype(int)
            return y_pred, np.asarray(y_score)

        y_pred = model.predict(X)
        return np.asarray(y_pred).astype(int), None

    # Regression
    y_pred = model.predict(X)
    return np.asarray(y_pred, dtype=float), None


@dataclass
class MetricsRow:
    target_id: str
    target_name: str
    chembl_target_id: str
    uniprot_accession: str
    activity_type: str
    model_type: str
    model_name: str
    pkl_file: str
    n_rows: int
    n_ok_descriptors: int
    n_predictions: int
    spearman: Optional[float] = None
    auc: Optional[float] = None
    tn: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
    tp: Optional[int] = None
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    try:
        from scipy.stats import spearmanr

        if len(y_true) < 2:
            return None
        res = spearmanr(y_true, y_pred)
        val = getattr(res, "correlation", None)
        if val is None:
            try:
                val = res[0]
            except Exception:
                return None
        try:
            val_any: Any = val
            val_f = float(val_any)
        except Exception:
            return None
        if np.isnan(val_f):
            return None
        return val_f
    except Exception:
        return None


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape != (2, 2):
            return None, None, None, None
        tn, fp, fn, tp = cm.ravel()
        return int(tn), int(fp), int(fn), int(tp)
    except Exception:
        # Fallback manual
        try:
            y_true_i = np.asarray(y_true).astype(int)
            y_pred_i = np.asarray(y_pred).astype(int)
            tn = int(np.sum((y_true_i == 0) & (y_pred_i == 0)))
            fp = int(np.sum((y_true_i == 0) & (y_pred_i == 1)))
            fn = int(np.sum((y_true_i == 1) & (y_pred_i == 0)))
            tp = int(np.sum((y_true_i == 1) & (y_pred_i == 1)))
            return tn, fp, fn, tp
        except Exception:
            return None, None, None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delta-csv",
        default="DELTA_biological_data.csv",
        help="Path to DELTA biological CSV (default: DELTA_biological_data.csv)",
    )
    parser.add_argument(
        "--model-selection-csv",
        default="delta_vs_paper_both__model_selection.csv",
        help="Path to selected model list CSV (default: delta_vs_paper_both__model_selection.csv)",
    )
    parser.add_argument(
        "--models-root",
        default=None,
        help="Root folder containing models subfolders (binding/ and functional/). Default: ../output/filtered/models",
    )
    parser.add_argument(
        "--api-cache-root",
        default=None,
        help=(
            "Folder containing cached training sets (one CSV/CSV.GZ per UniProt accession). "
            "Default: ../output/filtered/api_cache"
        ),
    )
    parser.add_argument(
        "--no-applicability-domain",
        action="store_true",
        help="Disable computing Applicability_Domain labels in the predictions CSV.",
    )
    parser.add_argument(
        "--ad-percentile",
        type=float,
        default=None,
        help=(
            "Optional: define AD cutoff as this percentile (0-100) of training nearest-neighbor "
            "similarities (k=1). Example: --ad-percentile 5. If not set, uses the strict "
            "minimum pairwise similarity in the training set."
        ),
    )
    parser.add_argument(
        "--out-prefix",
        default="DELTA_prediction_comparison_",
        help="Prefix for output files (CSV + PNG plots). Default: DELTA_prediction_comparison_",
    )
    parser.add_argument(
        "--out-predictions",
        default=None,
        help="Optional explicit output CSV for per-molecule predictions (overrides --out-prefix)",
    )
    parser.add_argument(
        "--out-metrics",
        default=None,
        help="Optional explicit output CSV for per-model evaluation metrics (overrides --out-prefix)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable writing PNG summary plots",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Only (re)generate plots from existing output CSVs for the given --out-prefix.",
    )
    parser.add_argument(
        "--recompute-metrics-from-predictions",
        action="store_true",
        help="Recompute metrics from the predictions CSV (no model inference).",
    )
    parser.add_argument(
        "--eval-label-source",
        choices=["class", "pchembl"],
        default="class",
        help="For classification evaluation: use DELTA 'CLASS' or derive label from pChEMBL cutoff.",
    )
    parser.add_argument(
        "--pchembl-cutoff",
        type=float,
        default=6.5,
        help="If --eval-label-source=pchembl, label is 1 when pChEMBL >= cutoff (default: 6.5).",
    )
    parser.add_argument(
        "--proba-threshold",
        type=float,
        default=None,
        help="Optional: for classification evaluation, override predicted labels as (y_score >= threshold).",
    )
    parser.add_argument(
        "--limit-targets",
        type=int,
        default=None,
        help="Optional: limit number of targets processed (for quick smoke runs)",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    delta_csv = (script_dir / args.delta_csv).resolve() if not os.path.isabs(args.delta_csv) else Path(args.delta_csv)
    sel_csv = (
        (script_dir / args.model_selection_csv).resolve()
        if not os.path.isabs(args.model_selection_csv)
        else Path(args.model_selection_csv)
    )

    if args.models_root is None:
        models_root = (script_dir.parent / "output" / "filtered" / "models").resolve()
    else:
        models_root = (Path(args.models_root).resolve() if not os.path.isabs(args.models_root) else Path(args.models_root))

    if args.api_cache_root is None:
        api_cache_root = (script_dir.parent / "output" / "filtered" / "api_cache").resolve()
    else:
        api_cache_root = (
            Path(args.api_cache_root).resolve()
            if not os.path.isabs(args.api_cache_root)
            else Path(args.api_cache_root)
        )

    default_preds, default_metrics = _default_out_paths(script_dir, args.out_prefix)
    out_predictions = (
        (script_dir / args.out_predictions).resolve() if args.out_predictions else default_preds
    )
    out_metrics = (
        (script_dir / args.out_metrics).resolve() if args.out_metrics else default_metrics
    )

    if args.plots_only:
        if args.no_plots:
            print("--plots-only has no effect with --no-plots")
            return 0
        if not out_predictions.exists() or not out_metrics.exists():
            raise SystemExit(
                "--plots-only requires existing output CSVs. Missing one of:\n"
                f"- {out_predictions}\n"
                f"- {out_metrics}"
            )
        pred_df = pd.read_csv(out_predictions)
        metrics_df = pd.read_csv(out_metrics)
        _make_plots(
            pred_df,
            metrics_df,
            script_dir=script_dir,
            out_prefix=args.out_prefix,
            label_source=args.eval_label_source,
            pchembl_cutoff=args.pchembl_cutoff,
            proba_threshold=args.proba_threshold,
        )
        print("Wrote summary plots (PNG).")
        return 0

    if args.recompute_metrics_from_predictions:
        if not out_predictions.exists():
            raise SystemExit(f"Predictions CSV not found: {out_predictions}")
        pred_df = pd.read_csv(out_predictions)
        metrics_df = _recompute_metrics_from_predictions(
            pred_df,
            label_source=args.eval_label_source,
            pchembl_cutoff=args.pchembl_cutoff,
            proba_threshold=args.proba_threshold,
        )
        print(f"Writing metrics: {out_metrics}")
        metrics_df.to_csv(out_metrics, index=False)
        if not args.no_plots:
            _make_plots(
                pred_df,
                metrics_df,
                script_dir=script_dir,
                out_prefix=args.out_prefix,
                label_source=args.eval_label_source,
                pchembl_cutoff=args.pchembl_cutoff,
                proba_threshold=args.proba_threshold,
            )
            print("Wrote summary plots (PNG).")
        print("Done.")
        return 0

    if not delta_csv.exists():
        raise SystemExit(f"DELTA CSV not found: {delta_csv}")
    if not sel_csv.exists():
        raise SystemExit(f"Model selection CSV not found: {sel_csv}")
    if not models_root.exists():
        raise SystemExit(
            f"Models root not found: {models_root}\n"
            "Pass --models-root to point at the folder containing binding/ and functional/"
        )

    print(f"Loading DELTA: {delta_csv}")
    delta = pd.read_csv(delta_csv)
    required_cols = {"TARGET ID", "TARGET NAME", "MOLECULE NAME", "SMILES", "pChEMBL", "CLASS"}
    missing = required_cols - set(delta.columns)
    if missing:
        raise SystemExit(f"DELTA CSV is missing columns: {sorted(missing)}")

    print(f"Loading model selection: {sel_csv}")
    selection = pd.read_csv(sel_csv)

    # Keep only rows that have a PKL_File and Model_Name
    selection = selection[~selection["PKL_File"].isna()].copy()
    selection = selection[selection["PKL_File"].astype(str).str.strip() != ""].copy()

    if selection.empty:
        raise SystemExit("No selectable models found (PKL_File column is empty).")

    # Group by target to compute descriptors once per target
    targets = list(selection["TARGET ID"].dropna().astype(str).unique())
    if args.limit_targets is not None:
        targets = targets[: int(args.limit_targets)]

    print(f"Targets to process: {len(targets)}")

    prediction_rows: List[Dict[str, Any]] = []
    metrics_rows: List[MetricsRow] = []

    # Cache training fingerprints + thresholds per UniProt to avoid repeated work.
    ad_cache: Dict[str, Dict[str, Any]] = {}

    # Pre-index DELTA rows by target for speed
    delta_by_target = {tid: grp for tid, grp in delta.groupby(delta["TARGET ID"].astype(str))}

    for t_i, target_id in enumerate(targets, start=1):
        sel_t = selection[selection["TARGET ID"].astype(str) == str(target_id)].copy()
        delta_t = delta_by_target.get(str(target_id))
        if delta_t is None or delta_t.empty:
            continue

        target_name = str(delta_t["TARGET NAME"].iloc[0])

        smiles_list = delta_t["SMILES"].tolist()
        try:
            descriptors, ok_mask = _calc_descriptors_from_smiles(smiles_list)
        except SystemExit as exc:
            # RDKit missing etc.
            raise
        except Exception as exc:
            # Descriptor calculation failure for the whole target
            for _, row in sel_t.iterrows():
                metrics_rows.append(
                    MetricsRow(
                        target_id=str(target_id),
                        target_name=target_name,
                        chembl_target_id=str(row.get("chembl_target_id", "")),
                        uniprot_accession=str(row.get("uniprot_accession", "")),
                        activity_type=str(row.get("Activity_Type", "")),
                        model_type=str(row.get("Model_Type", "")),
                        model_name=str(row.get("Model_Name", "")),
                        pkl_file=str(row.get("PKL_File", "")),
                        n_rows=int(len(delta_t)),
                        n_ok_descriptors=0,
                        n_predictions=0,
                        error=f"descriptor_error: {type(exc).__name__}: {str(exc)[:200]}",
                    )
                )
            continue

        n_ok = int(ok_mask.sum())

        # Slice to only rows with valid descriptors; keep mapping to original rows
        ok_indices = ok_mask[ok_mask].index.to_numpy()
        if len(ok_indices) == 0:
            for _, row in sel_t.iterrows():
                metrics_rows.append(
                    MetricsRow(
                        target_id=str(target_id),
                        target_name=target_name,
                        chembl_target_id=str(row.get("chembl_target_id", "")),
                        uniprot_accession=str(row.get("uniprot_accession", "")),
                        activity_type=str(row.get("Activity_Type", "")),
                        model_type=str(row.get("Model_Type", "")),
                        model_name=str(row.get("Model_Name", "")),
                        pkl_file=str(row.get("PKL_File", "")),
                        n_rows=int(len(delta_t)),
                        n_ok_descriptors=0,
                        n_predictions=0,
                        error="no_valid_descriptors",
                    )
                )
            continue

        descriptors_ok = descriptors.iloc[ok_indices].reset_index(drop=True)
        delta_ok = delta_t.iloc[ok_indices].reset_index(drop=True)

        # Precompute query fingerprints once per target (reused across all models for the target).
        if args.no_applicability_domain:
            query_fps: List[Any] = []
        else:
            query_fps = _maccs_fps_from_smiles(delta_ok["SMILES"].astype(str).tolist())

        for _, model_row in sel_t.iterrows():
            pkl_file = str(model_row.get("PKL_File", "")).strip()
            model_name = str(model_row.get("Model_Name", "")).strip()
            model_type = str(model_row.get("Model_Type", "")).strip()
            activity_type = str(model_row.get("Activity_Type", "")).strip()
            chembl_target_id = str(model_row.get("chembl_target_id", "")).strip()
            uniprot_accession = str(model_row.get("uniprot_accession", "")).strip()

            metrics = MetricsRow(
                target_id=str(target_id),
                target_name=target_name,
                chembl_target_id=chembl_target_id,
                uniprot_accession=uniprot_accession,
                activity_type=activity_type,
                model_type=model_type,
                model_name=model_name,
                pkl_file=pkl_file,
                n_rows=int(len(delta_t)),
                n_ok_descriptors=n_ok,
                n_predictions=0,
            )

            pkl_path = _find_pkl(models_root, pkl_file)
            if pkl_path is None:
                metrics.error = "pkl_not_found"
                metrics_rows.append(metrics)
                continue

            try:
                bundle = _load_model_bundle(pkl_path)
                model, scaler, feature_names, threshold = _extract_bundle_fields(bundle)
                if model is None:
                    raise ValueError("No estimator found in bundle")

                X = _build_X_for_bundle(descriptors_ok, model=model, scaler=scaler, feature_names=feature_names)
                y_pred, y_score = _predict_for_model(model, X, model_type=model_type, threshold=threshold)
                metrics.n_predictions = int(len(y_pred))

                # Assemble per-row predictions
                base_cols = {
                    "TARGET ID": delta_ok["TARGET ID"].astype(str),
                    "TARGET NAME": delta_ok["TARGET NAME"].astype(str),
                    "MOLECULE NAME": delta_ok["MOLECULE NAME"].astype(str),
                    "SMILES": delta_ok["SMILES"].astype(str),
                    "pChEMBL": pd.to_numeric(delta_ok["pChEMBL"], errors="coerce"),
                    "CLASS": pd.to_numeric(delta_ok["CLASS"], errors="coerce"),
                    "chembl_target_id": chembl_target_id,
                    "uniprot_accession": uniprot_accession,
                    "Activity_Type": activity_type,
                    "Model_Type": model_type,
                    "Model_Name": model_name,
                    "PKL_File": pkl_file,
                    "model_path": str(pkl_path),
                }
                pred_df = pd.DataFrame(base_cols)
                pred_df["y_pred"] = y_pred
                if y_score is not None:
                    pred_df["y_score"] = y_score

                # Applicability domain label (MACCS keys + Tanimoto to training set).
                if args.no_applicability_domain:
                    pred_df["Applicability_Domain"] = ["UNKNOWN"] * len(pred_df)
                else:
                    pred_df["Applicability_Domain"] = _applicability_domain_labels(
                        query_fps=query_fps,
                        api_cache_root=api_cache_root,
                        uniprot_accession=uniprot_accession,
                        chembl_target_id=chembl_target_id,
                        activity_type=activity_type,
                        ad_percentile=args.ad_percentile,
                        cache=ad_cache,
                    )

                pred_records = pred_df.to_dict(orient="records")
                prediction_rows.extend([{str(k): v for k, v in rec.items()} for rec in pred_records])

                # Compute evaluation metrics
                model_type_norm = model_type.strip().lower()
                if model_type_norm.startswith("reg"):
                    y_true = pd.to_numeric(delta_ok["pChEMBL"], errors="coerce").to_numpy(dtype=float)
                    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
                    metrics.spearman = _spearman(y_true[mask], y_pred[mask])

                elif model_type_norm.startswith("class"):
                    y_true = pd.to_numeric(delta_ok["CLASS"], errors="coerce").to_numpy(dtype=int)
                    tn, fp, fn, tp = _confusion(y_true, y_pred)
                    metrics.tn, metrics.fp, metrics.fn, metrics.tp = tn, fp, fn, tp
                    if y_score is not None:
                        metrics.auc = _roc_auc(y_true, np.asarray(y_score))

                metrics_rows.append(metrics)

            except Exception as exc:
                metrics.error = f"predict_error: {type(exc).__name__}: {str(exc)[:200]}"
                metrics_rows.append(metrics)
                continue

        if t_i % 10 == 0:
            print(f"Processed {t_i}/{len(targets)} targets")

    print(f"Writing predictions: {out_predictions}")
    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(out_predictions, index=False)

    print(f"Writing metrics: {out_metrics}")
    metrics_df = pd.DataFrame([m.as_dict() for m in metrics_rows])
    metrics_df.to_csv(out_metrics, index=False)

    if not args.no_plots:
        try:
            _make_plots(
                pred_df,
                metrics_df,
                script_dir=script_dir,
                out_prefix=args.out_prefix,
                label_source=args.eval_label_source,
                pchembl_cutoff=args.pchembl_cutoff,
                proba_threshold=args.proba_threshold,
            )
            print("Wrote summary plots (PNG).")
        except Exception as exc:
            print(f"Plotting skipped due to error: {type(exc).__name__}: {str(exc)[:200]}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
