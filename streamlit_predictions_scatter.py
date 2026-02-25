from __future__ import annotations

import io
import numbers
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


COMMON_CSV_ENCODINGS: Tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin1",
)


def _fig_to_bytes(fig: Figure, *, fmt: str, dpi: int | None = None) -> bytes:
    buf = io.BytesIO()
    if dpi is None:
        fig.savefig(buf, format=fmt, bbox_inches="tight")
    else:
        fig.savefig(buf, format=fmt, dpi=int(dpi), bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _read_csv_with_encoding_fallback(source: str | io.BytesIO, **kwargs) -> pd.DataFrame:
    last_exc: Exception | None = None
    tried: list[str] = []

    for enc in COMMON_CSV_ENCODINGS:
        tried.append(enc)
        try:
            if not isinstance(source, str):
                source.seek(0)
            return pd.read_csv(source, encoding=enc, low_memory=False, **kwargs)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_exc = e

    msg = "Unable to decode CSV. Tried encodings: " + ", ".join(tried)
    if last_exc is not None:
        msg += f". Last error: {last_exc}"
    raise ValueError(msg) from last_exc


@st.cache_data(show_spinner=False)
def load_predictions_csv_from_path(path: str) -> pd.DataFrame:
    return _read_csv_with_encoding_fallback(path)


@st.cache_data(show_spinner=False)
def load_predictions_csv_from_upload(uploaded_bytes: bytes) -> pd.DataFrame:
    buf = io.BytesIO(uploaded_bytes)
    return _read_csv_with_encoding_fallback(buf)


def _as_regression_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "Model_Type" not in df.columns:
        return pd.DataFrame()
    model_type = df["Model_Type"].astype(str).str.strip().str.lower()
    return df[model_type.str.startswith("reg")].copy()


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pearson_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return None, None

    # Pearson
    try:
        pearson = float(np.corrcoef(x, y)[0, 1])
    except Exception:
        pearson = None

    # Spearman (fallback to pandas rank if scipy missing)
    spearman: float | None
    try:
        from scipy.stats import spearmanr

        res = spearmanr(x, y)
        # SciPy has changed the return type over time (tuple vs result object).
        corr = getattr(res, "correlation", None)
        if corr is None:
            corr = getattr(res, "statistic", None)
        if corr is None and isinstance(res, tuple) and len(res) >= 1:
            corr = res[0]
        corr_val = corr
        if isinstance(corr_val, (tuple, list, np.ndarray)):
            corr_val = corr_val[0] if len(corr_val) > 0 else None

        if corr_val is None:
            spearman = None
        elif isinstance(corr_val, (numbers.Real, str, bytes, bytearray, np.number)):
            try:
                spearman = float(corr_val)
            except Exception:
                spearman = None
        else:
            spearman = None
    except Exception:
        try:
            spearman = float(pd.Series(x).rank().corr(pd.Series(y).rank(), method="pearson"))
        except Exception:
            spearman = None

    return pearson, spearman


def _smooth_binned_band(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int | None = None,
    smooth_window: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute a smooth trend line and uncertainty band using x-binned statistics.

    Returns (x_center, y_mean, y_lo, y_hi) where the band is approx 95% CI:
    mean ± 1.96 * (std/sqrt(n)). Curves are smoothed with a centered rolling mean.
    """

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 10:
        return None

    if n_bins is None:
        n_bins = int(np.clip(np.sqrt(x.size), 10, 40))

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return None

    bins = np.linspace(x_min, x_max, int(n_bins) + 1)
    bin_idx = np.digitize(x, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)

    dfb = pd.DataFrame({"bin": bin_idx, "x": x, "y": y})
    stats = (
        dfb.groupby("bin", dropna=False)
        .agg(x_center=("x", "mean"), y_mean=("y", "mean"), y_std=("y", "std"), n=("y", "size"))
        .reset_index()
    )
    stats = stats[stats["n"] >= 2].copy()
    if stats.empty:
        return None

    stats["y_std"] = stats["y_std"].fillna(0.0)
    stats["sem"] = stats["y_std"] / np.sqrt(stats["n"].astype(float))
    stats["y_lo"] = stats["y_mean"] - 1.96 * stats["sem"]
    stats["y_hi"] = stats["y_mean"] + 1.96 * stats["sem"]

    w = int(max(1, smooth_window))
    for col in ["y_mean", "y_lo", "y_hi"]:
        stats[col] = stats[col].rolling(window=w, center=True, min_periods=1).mean()

    return (
        stats["x_center"].to_numpy(dtype=float),
        stats["y_mean"].to_numpy(dtype=float),
        stats["y_lo"].to_numpy(dtype=float),
        stats["y_hi"].to_numpy(dtype=float),
    )


def main() -> None:
    st.set_page_config(page_title="DELTA – Regression Scatter", layout="wide")
    sns.set_theme(style="whitegrid")

    st.title("Regression: pChEMBL vs y_pred")
    st.caption(
        "Select a regression group and target from DELTA_prediction_comparison_predictions.csv, then view the scatter plot and correlation."
    )

    with st.sidebar:
        st.header("Data")
        source = st.radio("Load CSV from", options=["Workspace file", "Upload"], index=0)

        df: pd.DataFrame
        if source == "Workspace file":
            default_path = "DELTA_prediction_comparison_predictions.csv"
            path = st.text_input("CSV path", value=default_path)
            try:
                df = load_predictions_csv_from_path(path)
            except Exception as e:
                st.error(f"Failed to read CSV from path: {path}\n\n{e}")
                st.stop()
        else:
            uploaded = st.file_uploader("Upload predictions CSV", type=["csv"])
            if uploaded is None:
                st.info("Upload a CSV to start.")
                st.stop()
            df = load_predictions_csv_from_upload(uploaded.getvalue())

        st.divider()
        st.header("Applicability domain")
        has_ad = "Applicability_Domain" in df.columns
        if not has_ad:
            st.caption("No 'Applicability_Domain' column found in this CSV.")
            exclude_outside = False
            exclude_unknown = False
        else:
            exclude_outside = st.checkbox(
                "Exclude OUTSIDE compounds",
                value=False,
                help="Filters out rows where Applicability_Domain == OUTSIDE before plotting and computing correlations.",
            )
            exclude_unknown = st.checkbox(
                "Exclude UNKNOWN too",
                value=False,
                help="Also filters out rows where Applicability_Domain == UNKNOWN.",
            )

    if df.empty:
        st.warning("No data loaded.")
        st.stop()

    reg = _as_regression_rows(df)
    if reg.empty:
        counts = df.get("Model_Type", pd.Series(dtype=str)).astype(str).str.lower().value_counts().head(20)
        st.warning("No regression rows found (Model_Type starting with 'reg').")
        if not counts.empty:
            st.markdown("**Model_Type value counts (top 20)**")
            st.dataframe(counts.rename("count").to_frame(), use_container_width=True)
        st.stop()

    required = ["pChEMBL", "y_pred"]
    missing = [c for c in required if c not in reg.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # Regression group = activity type (matches metrics.csv activity_type)
    if "Activity_Type" not in reg.columns:
        st.error("Missing 'Activity_Type' column; cannot provide regression group selector.")
        st.stop()

    # Target selector (prefer chembl_target_id + name)
    if "chembl_target_id" not in reg.columns:
        st.error("Missing 'chembl_target_id' column; cannot provide target selector.")
        st.stop()

    # Pre-filter to only rows with valid numeric values, so dropdowns don't lead to empty plots.
    work = reg.copy()
    work["Activity_Type"] = work["Activity_Type"].astype(str)
    work["chembl_target_id"] = work["chembl_target_id"].astype(str)
    work["__pchembl__"] = _safe_numeric(work["pChEMBL"])
    work["__ypred__"] = _safe_numeric(work["y_pred"])
    work = work.dropna(subset=["__pchembl__", "__ypred__"]).copy()

    # Optional applicability-domain filtering (applies to selectors + metrics + plot)
    if "Applicability_Domain" in work.columns and (exclude_outside or exclude_unknown):
        ad = work["Applicability_Domain"].astype(str).str.strip().str.upper()
        keep = pd.Series(True, index=work.index)
        if exclude_outside:
            keep &= ad != "OUTSIDE"
        if exclude_unknown:
            keep &= ad != "UNKNOWN"
        work = work.loc[keep].copy()

    group_values = sorted([x for x in work["Activity_Type"].dropna().astype(str).unique() if str(x).strip()])
    if not group_values:
        st.error("No Activity_Type values have valid pChEMBL + y_pred for regression rows.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        selected_group = st.selectbox("Regression group (Activity_Type)", options=group_values, index=0, key="activity_type")

    group_df = work[work["Activity_Type"].astype(str) == str(selected_group)].copy()
    if group_df.empty:
        st.info("No rows available for the selected Activity_Type.")
        st.stop()

    target_name_col = "TARGET NAME" if "TARGET NAME" in group_df.columns else None
    target_df = group_df[["chembl_target_id"] + ([target_name_col] if target_name_col else [])].drop_duplicates().copy()
    if target_name_col:
        target_df[target_name_col] = target_df[target_name_col].astype(str)
        target_df["__label__"] = target_df["chembl_target_id"].astype(str) + " | " + target_df[target_name_col]
    else:
        target_df["__label__"] = target_df["chembl_target_id"].astype(str)

    target_labels = target_df.sort_values("chembl_target_id")["__label__"].tolist()
    if not target_labels:
        st.info("No targets available for the selected Activity_Type.")
        st.stop()

    with c2:
        selected_target_label = st.selectbox("Target", options=target_labels, index=0, key="target")

    # Resolve target id
    selected_target_id = str(selected_target_label).split(" | ", 1)[0].strip()

    sub = group_df[group_df["chembl_target_id"].astype(str) == selected_target_id].copy()

    if "Applicability_Domain" in sub.columns and (exclude_outside or exclude_unknown):
        ad = sub["Applicability_Domain"].astype(str).str.strip().str.upper()
        keep = pd.Series(True, index=sub.index)
        if exclude_outside:
            keep &= ad != "OUTSIDE"
        if exclude_unknown:
            keep &= ad != "UNKNOWN"
        sub = sub.loc[keep].copy()

    if sub.empty:
        st.info("No rows available for the selected target.")
        st.stop()

    n_total = int(len(sub))
    if "Applicability_Domain" in group_df.columns:
        ad_all = group_df["Applicability_Domain"].astype(str).str.strip().str.upper()
        n_outside_all = int((ad_all == "OUTSIDE").sum())
        n_unknown_all = int((ad_all == "UNKNOWN").sum())
        n_inside_all = int((ad_all == "INSIDE").sum())
    else:
        n_outside_all = n_unknown_all = n_inside_all = 0
    max_points = 25_000

    x_all = sub["__pchembl__"].to_numpy(dtype=float)
    y_all = sub["__ypred__"].to_numpy(dtype=float)
    pearson, spearman = _pearson_spearman(x_all, y_all)

    x = x_all
    y = y_all
    if n_total > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(np.arange(n_total), size=max_points, replace=False)
        x = x_all[idx]
        y = y_all[idx]
        st.caption(f"Showing a random sample of {max_points} points (out of {n_total}).")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("N (after filter)", f"{n_total}")
    m2.metric("Pearson r", "" if pearson is None else f"{pearson:.3f}")
    m3.metric("Spearman ρ", "" if spearman is None else f"{spearman:.3f}")
    if "Applicability_Domain" in group_df.columns:
        m4.metric("AD counts (target)", f"IN {n_inside_all} | OUT {n_outside_all} | UNK {n_unknown_all}")
    else:
        m4.metric("AD counts", "n/a")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(x, y, s=12, alpha=0.35)

    band = _smooth_binned_band(x_all, y_all, smooth_window=5)
    if band is not None:
        x_line, y_mean, y_lo, y_hi = band
        ax.plot(x_line, y_mean, color="tab:orange", linewidth=2)
        ax.fill_between(x_line, y_lo, y_hi, color="tab:orange", alpha=0.18, linewidth=0)

    min_v = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
    max_v = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="black", linewidth=1)

    ax.set_xlabel("pChEMBL (true)")
    ax.set_ylabel("model predicted pXC50")
    title = f"{selected_group} | {selected_target_label}"
    if pearson is not None:
        title += f" | r={pearson:.3f}"
    if spearman is not None:
        title += f" | ρ={spearman:.3f}"
    ax.set_title(title)
    fig.tight_layout()

    # IMPORTANT: create export bytes BEFORE calling st.pyplot with clear_figure=True,
    # otherwise the figure can be cleared and exports become blank.
    svg_bytes = _fig_to_bytes(fig, fmt="svg")
    png600_bytes = _fig_to_bytes(fig, fmt="png", dpi=600)

    st.pyplot(fig, clear_figure=False)

    base_name = f"reg_scatter_{str(selected_group).strip()}_{str(selected_target_id).strip()}"
    base_name = base_name.replace(" ", "_").replace("/", "-").replace("\\\\", "-")
    svg_name = f"{base_name}.svg"
    png_name = f"{base_name}_600dpi.png"

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "Download SVG",
            data=svg_bytes,
            file_name=svg_name,
            mime="image/svg+xml",
        )
    with d2:
        st.download_button(
            "Download PNG (600 dpi)",
            data=png600_bytes,
            file_name=png_name,
            mime="image/png",
        )
    with d3:
        if st.button("Save SVG to cwd"):
            with open(svg_name, "wb") as f:
                f.write(svg_bytes)
            st.success(f"Saved: {svg_name}")
    with d4:
        if st.button("Save PNG (600 dpi) to cwd"):
            with open(png_name, "wb") as f:
                f.write(png600_bytes)
            st.success(f"Saved: {png_name}")


if __name__ == "__main__":
    main()
