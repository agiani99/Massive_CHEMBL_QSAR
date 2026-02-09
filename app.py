import io
import math
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


PCHEMBL_CANDIDATES = [
    "pchembl_value",
    "pchembl",
    "pchemblvalue",
    "pchembl-value",
]

CLASS_CANDIDATES = [
    "class",
    "activity_class",
    "active",
]

ID_CANDIDATES = [
    "molecule_chembl_id",
    "chembl_id",
    "molecule_id",
    "compound_id",
    "id",
]


@dataclass(frozen=True)
class Processed:
    raw: pd.DataFrame
    meta: pd.DataFrame
    activity: pd.Series
    activity_class: pd.Series
    fragments: pd.DataFrame
    erg: pd.DataFrame
    fragments_scaled: pd.DataFrame
    erg_scaled: pd.DataFrame
    sort_key: str


def _norm_sf_abs_z(abs_z: float) -> float:
    """Two-sided p-value for a normal z statistic, using erfc."""
    # p = 2 * (1 - Phi(|z|)) = erfc(|z| / sqrt(2))
    return float(math.erfc(float(abs_z) / math.sqrt(2.0)))


def mann_whitney_u_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """Approximate two-sided Mann–Whitney U p-value via normal approximation.

    Notes:
    - Handles ties with the standard tie correction.
    - Returns 1.0 when the test is not defined (too few samples or zero variance).
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n1 = int(x.size)
    n2 = int(y.size)
    if n1 < 2 or n2 < 2:
        return 1.0

    combined = np.concatenate([x, y], axis=0)
    ranks = pd.Series(combined).rank(method="average").to_numpy(dtype=float)
    r1 = float(ranks[:n1].sum())

    u1 = r1 - (n1 * (n1 + 1)) / 2.0
    u2 = (n1 * n2) - u1
    u = float(min(u1, u2))

    mean_u = (n1 * n2) / 2.0

    # Tie correction
    _, tie_counts = np.unique(combined, return_counts=True)
    tie_counts = tie_counts[tie_counts > 1]
    if tie_counts.size:
        tie_term = float(np.sum(tie_counts**3 - tie_counts))
        tie_correction = 1.0 - tie_term / ((n1 + n2) * (n1 + n2 - 1.0))
    else:
        tie_correction = 1.0

    var_u = (n1 * n2 / 12.0) * ((n1 + n2 + 1.0) * tie_correction)
    if not np.isfinite(var_u) or var_u <= 0.0:
        return 1.0

    # Continuity correction
    z = (u - mean_u + 0.5) / math.sqrt(var_u)
    return _norm_sf_abs_z(abs(z))


def welch_ttest_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided Welch t-test p-value using a normal approximation.

    Avoids SciPy dependency. For moderate/large samples this is a good approximation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n1 = int(x.size)
    n2 = int(y.size)
    if n1 < 2 or n2 < 2:
        return 1.0

    m1 = float(np.mean(x))
    m2 = float(np.mean(y))
    v1 = float(np.var(x, ddof=1))
    v2 = float(np.var(y, ddof=1))

    se = math.sqrt((v1 / n1) + (v2 / n2))
    if not np.isfinite(se) or se <= 0.0:
        return 1.0

    z = (m1 - m2) / se
    return _norm_sf_abs_z(abs(z))


def select_top_features(
    matrix01: pd.DataFrame,
    activity_class: pd.Series,
    method: str,
    p_threshold: float,
    max_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (filtered_matrix01, stats_df).

    stats_df columns: feature, effect, p_value
    - effect: median difference (class1 - class0) for Mann–Whitney
              mean difference (class1 - class0) for Welch
    """

    if matrix01.empty:
        return matrix01, pd.DataFrame(columns=["feature", "effect", "p_value"])

    y = activity_class.reindex(matrix01.index)
    y = y.fillna(0).astype(int)
    mask1 = y.to_numpy() == 1
    mask0 = ~mask1
    if mask1.sum() < 2 or mask0.sum() < 2:
        return matrix01, pd.DataFrame(columns=["feature", "effect", "p_value"])

    rows = []
    data = matrix01.to_numpy(dtype=float)
    for j, feature in enumerate(matrix01.columns):
        col = data[:, j]
        x0 = col[mask0]
        x1 = col[mask1]

        if method == "mannwhitney":
            effect = float(np.nanmedian(x1) - np.nanmedian(x0))
            p = mann_whitney_u_pvalue(x0, x1)
        else:
            effect = float(np.nanmean(x1) - np.nanmean(x0))
            p = welch_ttest_pvalue(x0, x1)

        rows.append((str(feature), effect, float(p)))

    stats = pd.DataFrame(rows, columns=["feature", "effect", "p_value"])
    stats = stats.replace([np.inf, -np.inf], np.nan).dropna(subset=["p_value"])  # keep NaN effects OK

    passed = stats[stats["p_value"] <= float(p_threshold)].copy()
    if passed.empty:
        # Keep something sensible rather than returning an empty heatmap
        stats_sorted = stats.assign(abs_effect=stats["effect"].abs()).sort_values(
            ["abs_effect", "p_value"], ascending=[False, True], kind="mergesort"
        )
        keep = stats_sorted.head(int(max_features))["feature"].tolist()
        return matrix01[keep], stats_sorted.drop(columns=["abs_effect"]).head(int(max_features))

    passed = passed.assign(abs_effect=passed["effect"].abs()).sort_values(
        ["abs_effect", "p_value"], ascending=[False, True], kind="mergesort"
    )
    keep = passed.head(int(max_features))["feature"].tolist()
    return matrix01[keep], passed.drop(columns=["abs_effect"]).head(int(max_features))


def _normalize_colname(name: str) -> str:
    return re.sub(r"\s+", "_", str(name).strip().lower())


def find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    if df.empty:
        return None

    norm_map = {_normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]

    # fallback: contains match
    for cand in candidates:
        for norm, orig in norm_map.items():
            if norm == cand or norm.startswith(cand) or cand in norm:
                return orig

    return None


def _looks_like_fragment_col(col: str) -> bool:
    return _normalize_colname(col).startswith("fr_")


def _is_activity_col(col: str) -> bool:
    return _normalize_colname(col) in {"pchembl_value", "class"}


def _summarize_erg_distance_bins(erg_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize columns like HA_HA_d1..d15 into a single ERG_HA_HA column (sum)."""
    if erg_df.empty:
        return erg_df

    bin_regex = re.compile(r"^(?P<base>.+)_d(?P<dist>\d+)$", re.IGNORECASE)
    groups: dict[str, list[str]] = {}
    bin_cols: list[str] = []
    for col in erg_df.columns:
        m = bin_regex.match(str(col))
        if not m:
            continue
        base = m.group("base")
        base = str(base)
        groups.setdefault(base, []).append(col)
        bin_cols.append(col)

    if not groups:
        return erg_df

    summarized = {}
    for base, cols in groups.items():
        # prefix ERG_ to make these stand out
        out_col = f"ERG_{base}"
        summarized[out_col] = erg_df[cols].sum(axis=1)

    # Drop bin columns and add summarized
    out = erg_df.drop(columns=bin_cols, errors="ignore").copy()
    for k, v in summarized.items():
        if k in out.columns:
            continue
        out[k] = v

    return out


def split_numeric_features(
    df: pd.DataFrame,
    fill_na_with_zero: bool,
    summarize_erg_bins: bool,
    prefix_erg_cols: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (fragments_df, erg_df) as numeric matrices.

    - fragments_df: columns starting with fr_
    - erg_df: all other numeric columns excluding activity columns
    """

    # Exclude known non-feature columns
    exclude_norm = {
        "chembl_target_id",
        "chembl_target_name",
        "molecule_chembl_id",
        "smiles",
        "mw",  # may be present as metadata, but we still pick numeric columns below if it is numeric
        "assay_type",
        "standard_type",
        "standard_value",
        "standard_units",
    }

    candidate_cols: list[str] = []
    for col in df.columns:
        norm = _normalize_colname(col)
        if _is_activity_col(col):
            continue
        if norm in exclude_norm:
            # We'll still let numeric conversion handle MW etc via other variants (e.g. 'MW')
            # but avoid pulling in obvious strings.
            continue
        candidate_cols.append(col)

    if not candidate_cols:
        return pd.DataFrame(index=df.index), pd.DataFrame(index=df.index)

    numeric = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    # keep only columns that have at least one numeric value
    numeric = numeric.loc[:, numeric.notna().any(axis=0)]

    if fill_na_with_zero:
        numeric = numeric.fillna(0.0)

    fragment_cols = [c for c in numeric.columns if _looks_like_fragment_col(c)]
    fragments = numeric[fragment_cols].copy() if fragment_cols else pd.DataFrame(index=df.index)
    erg = numeric.drop(columns=fragment_cols, errors="ignore").copy()

    if summarize_erg_bins:
        erg = _summarize_erg_distance_bins(erg)

    if prefix_erg_cols and not erg.empty:
        rename = {}
        for c in erg.columns:
            if str(c).startswith("ERG_"):
                continue
            rename[c] = f"ERG_{c}"
        erg = erg.rename(columns=rename)

    return fragments, erg


def ensure_activity_columns(
    df: pd.DataFrame, pchembl_col: Optional[str], class_col: Optional[str], threshold: float
) -> tuple[pd.DataFrame, Optional[str], str]:
    """Returns (df, pchembl_col, activity_label).

    activity_label is either 'pchembl_value' (existing/normalized name) or 'class'.
    """
    out = df.copy()

    # Create/normalize pchembl_value
    if pchembl_col is not None:
        out["pchembl_value"] = pd.to_numeric(out[pchembl_col], errors="coerce")
        pchembl_col = "pchembl_value"

    # Create class based on pchembl if present
    if pchembl_col is not None:
        out["class"] = (out[pchembl_col] >= threshold).astype("int64")
        return out, pchembl_col, "pchembl_value"

    # Otherwise, try to use an existing class-like column
    if class_col is not None:
        out["class"] = pd.to_numeric(out[class_col], errors="coerce")
        # map truthy/1/0-ish values into 0/1
        out["class"] = out["class"].fillna(0)
        out["class"] = (out["class"] >= 0.5).astype("int64")
        return out, None, "class"

    # Nothing available
    out["class"] = 0
    return out, None, "class"


def drop_zero_variance_columns(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features

    # nunique(dropna=False) treats all-NaN as a single value -> dropped
    keep_cols = [c for c in features.columns if features[c].nunique(dropna=False) > 1]
    return features[keep_cols]


def minmax_scale(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    col_min = df.min(axis=0, skipna=True)
    col_max = df.max(axis=0, skipna=True)
    denom = (col_max - col_min).replace(0, np.nan)
    scaled = (df - col_min) / denom
    return scaled.fillna(0.0)


def process_dataframe(
    df: pd.DataFrame,
    threshold: float = 6.5,
    fill_na_with_zero: bool = True,
    summarize_erg_bins: bool = True,
    prefix_erg_cols: bool = True,
) -> Processed:
    if df.empty:
        raise ValueError("CSV appears to be empty.")

    pchembl_col = find_column(df, PCHEMBL_CANDIDATES)
    class_col = find_column(df, CLASS_CANDIDATES)
    id_col = find_column(df, ID_CANDIDATES)

    df2, _pchembl_col2, activity_label = ensure_activity_columns(df, pchembl_col, class_col, threshold)

    fragments, erg = split_numeric_features(
        df2,
        fill_na_with_zero=fill_na_with_zero,
        summarize_erg_bins=summarize_erg_bins,
        prefix_erg_cols=prefix_erg_cols,
    )

    fragments = drop_zero_variance_columns(fragments)
    erg = drop_zero_variance_columns(erg)

    if fragments.empty and erg.empty:
        raise ValueError(
            "No numeric feature columns found. Expected fragment columns (fr_*) and/or numeric ERG descriptor columns."
        )

    fragments_scaled = minmax_scale(fragments)
    erg_scaled = minmax_scale(erg)

    # meta is just for identification / debugging (kept as-is)
    meta_cols = []
    if id_col is not None:
        meta_cols.append(id_col)
    for c in ["molecule_chembl_id", "SMILES", "smiles"]:
        if c in df2.columns and c not in meta_cols:
            meta_cols.append(c)

    if meta_cols:
        meta = df2[meta_cols].copy()
    else:
        meta = pd.DataFrame(index=df2.index)

    if id_col is None:
        # create a stable ID for plotting
        meta["compound"] = [f"row_{i+1}" for i in range(len(df2))]
        id_col = "compound"

    activity = df2["pchembl_value"] if activity_label == "pchembl_value" else df2["class"]
    activity_class = df2["class"].copy()

    sort_key = "pchembl_value" if "pchembl_value" in df2.columns and df2["pchembl_value"].notna().any() else "class"

    # Sort descending by pchembl_value if present, else by class
    sorter = df2[sort_key].fillna(-np.inf)
    order = sorter.sort_values(ascending=False, kind="mergesort").index

    meta = meta.loc[order]
    activity = activity.loc[order]
    activity_class = activity_class.loc[order]
    fragments = fragments.loc[order]
    erg = erg.loc[order]
    fragments_scaled = fragments_scaled.loc[order]
    erg_scaled = erg_scaled.loc[order]

    # index the matrices by compound id for nicer y labels.
    # Plotly categorical axes behave best with unique category values;
    # if IDs are duplicated, disambiguate them while preserving the original text.
    y_labels = meta[id_col].astype(str)
    if y_labels.duplicated().any():
        # Make only the duplicated labels unique with a stable suffix.
        dup_mask = y_labels.duplicated(keep=False)
        dup_rank = y_labels.groupby(y_labels).cumcount().add(1).astype(str)
        y_labels = y_labels.where(~dup_mask, y_labels + " #" + dup_rank)
    fragments.index = y_labels
    erg.index = y_labels
    fragments_scaled.index = y_labels
    erg_scaled.index = y_labels
    activity.index = y_labels
    activity_class.index = y_labels

    return Processed(
        raw=df2,
        meta=meta,
        activity=activity,
        activity_class=activity_class,
        fragments=fragments,
        erg=erg,
        fragments_scaled=fragments_scaled,
        erg_scaled=erg_scaled,
        sort_key=sort_key,
    )


def activity_strip_figure(activity: pd.Series, activity_label: str, height: int) -> go.Figure:
    z = activity.to_numpy().reshape(-1, 1)

    if activity_label == "pchembl_value":
        # scale pchembl for coloring in the strip
        vals = activity.astype(float)
        vmin = float(np.nanmin(vals.to_numpy())) if np.isfinite(vals).any() else 0.0
        vmax = float(np.nanmax(vals.to_numpy())) if np.isfinite(vals).any() else 1.0
        if vmin == vmax:
            vmin, vmax = 0.0, 1.0
        zmin, zmax = vmin, vmax
        colorscale = "RdBu"
        hovertemplate = "compound=%{y}<br>pchembl=%{z:.2f}<extra></extra>"
    else:
        zmin, zmax = 0, 1
        colorscale = [[0.0, "#e6e6e6"], [1.0, "#2b2b2b"]]
        hovertemplate = "compound=%{y}<br>class=%{z}<extra></extra>"

    fig = go.Figure(
        data=
        [
            go.Heatmap(
                z=z,
                x=["Activity"],
                y=activity.index.tolist(),
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                showscale=False,
                hovertemplate=hovertemplate,
            )
        ]
    )

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(side="top", tickfont=dict(size=12)),
        # Force category ordering to match the data order (prevents Plotly from
        # implicitly sorting labels alphabetically).
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=10),
            categoryorder="array",
            categoryarray=activity.index.tolist(),
        ),
    )
    return fig


def combined_activity_heatmap_figure(
    activity: pd.Series,
    activity_label: str,
    matrix01: pd.DataFrame,
    height: int,
    show_x_labels: bool,
    color_scale: str = "Viridis",
) -> go.Figure:
    """Single figure with shared y-axis to guarantee perfect row alignment."""

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        column_widths=[0.12, 0.88],
    )

    # Activity strip trace
    z_act = activity.to_numpy().reshape(-1, 1)
    if activity_label == "pchembl_value":
        vals = activity.astype(float).to_numpy()
        finite = np.isfinite(vals)
        vmin = float(np.nanmin(vals[finite])) if finite.any() else 0.0
        vmax = float(np.nanmax(vals[finite])) if finite.any() else 1.0
        if vmin == vmax:
            vmin, vmax = 0.0, 1.0
        zmin, zmax = vmin, vmax
        colorscale_act = "RdBu"
        hover_act = "compound=%{y}<br>pchembl=%{z:.2f}<extra></extra>"
    else:
        zmin, zmax = 0, 1
        colorscale_act = [[0.0, "#e6e6e6"], [1.0, "#2b2b2b"]]
        hover_act = "compound=%{y}<br>class=%{z}<extra></extra>"

    fig.add_trace(
        go.Heatmap(
            z=z_act,
            x=["Activity"],
            y=activity.index.tolist(),
            colorscale=colorscale_act,
            zmin=zmin,
            zmax=zmax,
            showscale=False,
            hovertemplate=hover_act,
        ),
        row=1,
        col=1,
    )

    # Feature heatmap trace
    if matrix01.empty:
        # Keep an empty placeholder to avoid subplot layout errors
        z = np.zeros((len(activity), 1))
        x = ["(no features)"]
    else:
        z = matrix01.to_numpy()
        x = list(matrix01.columns)

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x,
            y=activity.index.tolist(),
            colorscale=color_scale,
            zmin=0,
            zmax=1,
            colorbar=dict(title="", thickness=12),
            hovertemplate="%{x}<br>compound=%{y}<br>value=%{z:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    # Y-axis shared; show IDs only on the activity strip
    y_order = activity.index.tolist()
    fig.update_yaxes(
        autorange="reversed",
        categoryorder="array",
        categoryarray=y_order,
        tickfont=dict(size=10),
        showticklabels=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        autorange="reversed",
        categoryorder="array",
        categoryarray=y_order,
        showticklabels=False,
        row=1,
        col=2,
    )

    # X-axes on top
    fig.update_xaxes(side="top", tickfont=dict(size=12), row=1, col=1)
    if not show_x_labels:
        fig.update_xaxes(showticklabels=False, row=1, col=2)
    else:
        fig.update_xaxes(side="top", tickangle=90, tickfont=dict(size=9), row=1, col=2)

    return fig


def heatmap_figure(
    matrix01: pd.DataFrame,
    height: int,
    show_x_labels: bool,
    color_scale: str = "Viridis",
) -> go.Figure:
    # plotly express imshow tends to be faster and handles aspect
    fig = px.imshow(
        matrix01,
        color_continuous_scale=color_scale,
        zmin=0,
        zmax=1,
        aspect="auto",
        origin="upper",
    )

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(title="", thickness=12),
    )

    # y labels are shown on the activity strip instead
    fig.update_yaxes(showticklabels=False, categoryorder="array", categoryarray=list(matrix01.index))

    if not show_x_labels:
        fig.update_xaxes(showticklabels=False)
    else:
        fig.update_xaxes(tickangle=90, tickfont=dict(size=9))

    return fig


def compute_plot_height(n_rows: int, cell_px: int, min_h: int = 260, max_h: int = 900) -> int:
    return int(max(min_h, min(max_h, n_rows * cell_px + 60)))


def read_uploaded_csv(uploaded) -> pd.DataFrame:
    # Streamlit gives UploadedFile or BytesIO-like
    raw = uploaded.getvalue()
    return pd.read_csv(io.BytesIO(raw))


def main() -> None:
    st.set_page_config(page_title="ERG heatmap", layout="wide")

    st.title("ERG / fragments heatmap")
    st.caption(
        "Upload a CSV with a `pchembl_value` column and feature columns starting with `fr_` or `HBA/HBD/AR/HY/PLUS/MINUS*`. "
        "The app drops zero-variance features, creates a binary class at pChEMBL ≥ 6.5, sorts compounds by activity, and renders a normalized heatmap."
    )

    with st.sidebar:
        st.header("Settings")
        threshold = st.number_input("Class threshold (pChEMBL)", value=6.5, step=0.1)
        fill_na_with_zero = st.checkbox("Fill missing feature values with 0", value=True)
        summarize_erg_bins = st.checkbox("Summarize ERG distance bins (_d1..d15)", value=False)
        prefix_erg_cols = st.checkbox("Prefix ERG columns with 'ERG_'", value=True)
        cell_px = st.slider("Row height (px)", min_value=3, max_value=18, value=7)
        show_x_labels = st.checkbox("Show feature names (x labels)", value=False)
        max_rows = st.number_input("Max compounds to display (0 = all)", value=0, min_value=0, step=50)

        st.divider()
        st.subheader("Feature selection")
        enable_feature_selection = st.checkbox(
            "Show only most discriminative features",
            value=False,
            help="Filters features by class-separation significance and effect size before plotting.",
        )
        feat_method = st.radio(
            "Test / score",
            options=[
                "Mann–Whitney (median diff, nonparametric)",
                "Welch t-test (mean diff, approx)",
            ],
            index=0,
            horizontal=False,
            disabled=not enable_feature_selection,
        )
        p_threshold = st.selectbox(
            "p-value threshold",
            options=[0.05, 0.01, 0.001],
            index=0,
            disabled=not enable_feature_selection,
        )
        max_features = st.slider(
            "Max features to display",
            min_value=10,
            max_value=300,
            value=80,
            step=10,
            disabled=not enable_feature_selection,
        )

    uploaded = st.file_uploader("Upload ERG CSV", type=["csv"])  # noqa: B008

    if uploaded is None:
        st.info("Upload a CSV to start. You can test with `activity_CHEMBL202_ERG.csv` in this folder.")
        return

    try:
        df = read_uploaded_csv(uploaded)
        processed = process_dataframe(
            df,
            threshold=float(threshold),
            fill_na_with_zero=fill_na_with_zero,
            summarize_erg_bins=summarize_erg_bins,
            prefix_erg_cols=prefix_erg_cols,
        )
    except Exception as e:
        st.error(f"Failed to parse/process CSV: {e}")
        return

    activity_label = "pchembl_value" if processed.sort_key == "pchembl_value" else "class"

    fr01 = processed.fragments_scaled
    erg01 = processed.erg_scaled
    activity = processed.activity
    activity_class = processed.activity_class

    if max_rows and max_rows > 0:
        fr01 = fr01.iloc[: int(max_rows), :]
        erg01 = erg01.iloc[: int(max_rows), :]
        activity = activity.iloc[: int(max_rows)]
        activity_class = activity_class.iloc[: int(max_rows)]

    stats_erg = pd.DataFrame()
    stats_fr = pd.DataFrame()
    if enable_feature_selection:
        method_key = "mannwhitney" if feat_method.startswith("Mann") else "welch"
        if erg01.shape[1] > 0:
            erg01, stats_erg = select_top_features(
                matrix01=erg01,
                activity_class=activity_class,
                method=method_key,
                p_threshold=float(p_threshold),
                max_features=int(max_features),
            )
        if fr01.shape[1] > 0:
            fr01, stats_fr = select_top_features(
                matrix01=fr01,
                activity_class=activity_class,
                method=method_key,
                p_threshold=float(p_threshold),
                max_features=int(max_features),
            )

    st.write(
        {
            "compounds": int(activity.shape[0]),
            "erg_features": int(erg01.shape[1]),
            "fragment_features": int(fr01.shape[1]),
            "sorted_by": processed.sort_key,
        }
    )

    if erg01.shape[1] == 0 and fr01.shape[1] == 0:
        st.warning("After dropping zero-variance columns, no features remain.")
        return

    height = compute_plot_height(n_rows=int(activity.shape[0]), cell_px=int(cell_px))

    tabs = st.tabs(["ERG descriptors", "fr_* fragments"])
    with tabs[0]:
        if erg01.shape[1] == 0:
            st.info("No ERG features available after filtering.")
        else:
            if enable_feature_selection and not stats_erg.empty:
                st.caption(
                    f"Showing {erg01.shape[1]} ERG features (p ≤ {float(p_threshold):g}, max {int(max_features)})."
                )
            st.plotly_chart(
                combined_activity_heatmap_figure(
                    activity=activity,
                    activity_label=activity_label,
                    matrix01=erg01,
                    height=height,
                    show_x_labels=show_x_labels,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            if enable_feature_selection and not stats_erg.empty:
                with st.expander("Selected ERG features (effect & p-value)"):
                    st.dataframe(stats_erg, use_container_width=True)

    with tabs[1]:
        if fr01.shape[1] == 0:
            st.info("No fragment (fr_*) features available after filtering.")
        else:
            if enable_feature_selection and not stats_fr.empty:
                st.caption(
                    f"Showing {fr01.shape[1]} fragment features (p ≤ {float(p_threshold):g}, max {int(max_features)})."
                )
            st.plotly_chart(
                combined_activity_heatmap_figure(
                    activity=activity,
                    activity_label=activity_label,
                    matrix01=fr01,
                    height=height,
                    show_x_labels=show_x_labels,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            if enable_feature_selection and not stats_fr.empty:
                with st.expander("Selected fragment features (effect & p-value)"):
                    st.dataframe(stats_fr, use_container_width=True)

    with st.expander("Preview processed table"):
        preview = pd.DataFrame({"compound": activity.index, "activity": activity.values})
        st.dataframe(preview, use_container_width=True)


if __name__ == "__main__":
    main()
