#!/usr/bin/env python3
"""
Cross-pipeline validation — organized by workflow tier.

Comparison groups
-----------------
  MINIMAL : pipelines that ran normalize → HVG → PCA (unscaled) → ...
  FULL    : pipelines that ran normalize → HVG → regress_out → scale → PCA → ...
  CROSS   : minimal vs full (same platform, different workflow)

Each group writes to its own output directory so results never collide:
  write/validation_{prefix}_minimal/
  write/validation_{prefix}_full/
  write/validation_{prefix}_cross/

Usage
-----
  python validate_cross_pipeline_gpu_v5.py --dataset mouse_brain_1m --gpu
  python validate_cross_pipeline_gpu_v5.py --dataset pbmc3k
  python validate_cross_pipeline_gpu_v5.py --dataset lung65k

Add --groups minimal full cross  to run only specific groups (default: all).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# ============================================================
# Data classes
# ============================================================

@dataclass
class PipelineFiles:
    name: str
    clusters: str
    markers: str | None
    h5ad: str | None
    kind: str          # scanpy_like | seurat | scalesc
    workflow: str       # minimal | full


# ============================================================
# Pipeline registry — add new pipelines here
# ============================================================

def pipeline_registry(prefix: str) -> dict[str, PipelineFiles]:
    """
    Central registry of every pipeline variant.

    Naming convention:
      minimal : write/{prefix}_{pipeline}_harmonized_*
      full    : write/{prefix}_{pipeline}_full_*
    """
    W = "write"
    return {
        # ---- MINIMAL (harmonized) ----
        "scanpy_cpu": PipelineFiles(
            name="scanpy_cpu",
            clusters=f"{W}/{prefix}_scanpy_cpu_harmonized_clusters.csv",
            markers=f"{W}/{prefix}_scanpy_cpu_harmonized_markers.csv",
            h5ad=f"{W}/{prefix}_scanpy_cpu_harmonized.h5ad",
            kind="scanpy_like",
            workflow="minimal",
        ),
        "rsc_gpu_0141": PipelineFiles(
            name="rsc_gpu_0141",
            clusters=f"{W}/{prefix}_rsc_gpu_0141_harmonized_clusters.csv",
            markers=f"{W}/{prefix}_rsc_gpu_0141_harmonized_markers.csv",
            h5ad=f"{W}/{prefix}_rsc_gpu_0141_harmonized.h5ad",
            kind="scanpy_like",
            workflow="minimal",
        ),
        "rsc_gpu_015": PipelineFiles(
            name="rsc_gpu_015",
            clusters=f"{W}/{prefix}_rsc_gpu_015_harmonized_clusters.csv",
            markers=f"{W}/{prefix}_rsc_gpu_015_harmonized_markers.csv",
            h5ad=f"{W}/{prefix}_rsc_gpu_015_harmonized.h5ad",
            kind="scanpy_like",
            workflow="minimal",
        ),
        "seurat_cpu": PipelineFiles(
            name="seurat_cpu",
            clusters=f"{W}/{prefix}_seurat_cpu_harmonized_clusters.csv",
            markers=f"{W}/{prefix}_seurat_cpu_harmonized_markers.csv",
            h5ad=None,
            kind="seurat",
            workflow="minimal",  # Seurat always scales, but lives here for comparison
        ),
        "scalesc_gpu": PipelineFiles(
            name="scalesc_gpu",
            clusters=f"{W}/{prefix}_scalesc_gpu_harmonized_clusters.csv",
            markers=None,
            h5ad=f"{W}/{prefix}_scalesc_gpu_harmonized.h5ad",
            kind="scalesc",
            workflow="minimal",
        ),

        # ---- FULL (regress_out + scale) ----
        "scanpy_cpu_full": PipelineFiles(
            name="scanpy_cpu_full",
            clusters=f"{W}/{prefix}_scanpy_cpu_full_clusters.csv",
            markers=f"{W}/{prefix}_scanpy_cpu_full_markers.csv",
            h5ad=f"{W}/{prefix}_scanpy_cpu_full.h5ad",
            kind="scanpy_like",
            workflow="full",
        ),
        "rsc_gpu_0141_full": PipelineFiles(
            name="rsc_gpu_0141_full",
            clusters=f"{W}/{prefix}_rsc_gpu_0141_full_clusters.csv",
            markers=f"{W}/{prefix}_rsc_gpu_0141_full_markers.csv",
            h5ad=f"{W}/{prefix}_rsc_gpu_0141_full.h5ad",
            kind="scanpy_like",
            workflow="full",
        ),
        "rsc_gpu_015_full": PipelineFiles(
            name="rsc_gpu_015_full",
            clusters=f"{W}/{prefix}_rsc_gpu_015_full_clusters.csv",
            markers=f"{W}/{prefix}_rsc_gpu_015_full_markers.csv",
            h5ad=f"{W}/{prefix}_rsc_gpu_015_full.h5ad",
            kind="scanpy_like",
            workflow="full",
        ),
    }


# ============================================================
# Argument parsing
# ============================================================

VALID_DATASETS = ["pbmc3k", "lung65k", "mouse_brain_1m"]
VALID_GROUPS = ["minimal", "full", "cross"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, choices=VALID_DATASETS)
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__) or ".", "benchmark_config.json"),
    )
    parser.add_argument(
        "--gpu", action="store_true", default=False,
        help="Use GPU-accelerated DE via rapids-singlecell.",
    )
    parser.add_argument(
        "--groups", nargs="+", default=VALID_GROUPS, choices=VALID_GROUPS,
        help="Which comparison groups to run (default: all).",
    )
    return parser.parse_args()


# ============================================================
# Config + IO helpers
# ============================================================

def load_config(path: str, dataset_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["global"], cfg["datasets"][dataset_key]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# Barcode normalization
# ============================================================

def normalize_barcode(x: str) -> str:
    x = str(x).strip().replace('"', "")
    return re.sub(r"-1$", "", x)


def make_unique_normalized_index(index: Iterable[str], label: str) -> pd.Index:
    normalized = pd.Index([normalize_barcode(x) for x in index])
    dup = normalized[normalized.duplicated()].unique().tolist()
    if dup:
        raise ValueError(
            f"Barcode collisions in {label}: {dup[:5]}"
        )
    return normalized


# ============================================================
# Data loaders
# ============================================================

def load_clusters(path: str) -> pd.Series:
    df = pd.read_csv(path)
    bc = "barcode" if "barcode" in df.columns else df.columns[0]
    cl = "leiden" if "leiden" in df.columns else df.columns[-1]
    s = pd.Series(df[cl].astype(str).values, index=df[bc].astype(str).values, name="cluster")
    s.index = make_unique_normalized_index(s.index, path)
    return s


def load_native_markers(path: str, kind: str) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    if kind in {"scanpy_like", "scalesc"}:
        return {str(g): gdf.copy() for g, gdf in df.groupby("group")}
    if kind == "seurat":
        return {str(g): gdf.copy() for g, gdf in df.groupby("cluster")}
    raise ValueError(f"Unknown marker kind: {kind}")


def get_marker_columns(kind: str) -> tuple[str, str, str, str]:
    """Returns (gene_col, fc_col, padj_col, group_col)."""
    if kind in {"scanpy_like", "scalesc"}:
        return ("names", "logfoldchanges", "pvals_adj", "group")
    if kind == "seurat":
        return ("gene", "avg_log2FC", "p_val_adj", "cluster")
    raise ValueError(kind)


# ============================================================
# Core metrics
# ============================================================

def align_on_common_barcodes(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError("No common barcodes.")
    return a.loc[common], b.loc[common]


def compute_matching(labels_a: pd.Series, labels_b: pd.Series) -> dict[str, str]:
    ct = pd.crosstab(labels_a, labels_b)
    ri, ci = linear_sum_assignment(-ct.to_numpy())
    rows, cols = ct.index.astype(str).tolist(), ct.columns.astype(str).tolist()
    return {rows[i]: cols[j] for i, j in zip(ri, ci)}


def compute_dice(labels_a: pd.Series, labels_b: pd.Series, matching: dict[str, str]) -> dict[str, float]:
    out = {}
    for ca, cb in matching.items():
        sa = set(labels_a.index[labels_a == ca])
        sb = set(labels_b.index[labels_b == cb])
        d = len(sa) + len(sb)
        out[f"{ca}<->{cb}"] = (2.0 * len(sa & sb) / d) if d else np.nan
    return out


def filtered_gene_set(df: pd.DataFrame, gene_col: str, fc_col: str, padj_col: str) -> set[str]:
    keep = df[(df[padj_col] < 0.05) & (df[fc_col] > 0.1)]
    return set(keep[gene_col].astype(str).tolist())


def jaccard_and_spearman(
    markers_a: dict[str, pd.DataFrame],
    markers_b: dict[str, pd.DataFrame],
    kind_a: str, kind_b: str,
    matching: dict[str, str],
) -> tuple[dict[str, float], dict[str, float]]:
    gc_a, fc_a, pa_a, _ = get_marker_columns(kind_a)
    gc_b, fc_b, pa_b, _ = get_marker_columns(kind_b)

    jaccard, rho = {}, {}
    for ca, cb in matching.items():
        da = markers_a.get(str(ca), pd.DataFrame())
        db = markers_b.get(str(cb), pd.DataFrame())
        ga = filtered_gene_set(da, gc_a, fc_a, pa_a) if not da.empty else set()
        gb = filtered_gene_set(db, gc_b, fc_b, pa_b) if not db.empty else set()

        pair = f"{ca}<->{cb}"
        union = ga | gb
        jaccard[pair] = (len(ga & gb) / len(union)) if union else np.nan

        if da.empty or db.empty:
            rho[pair] = np.nan
            continue
        merged = pd.merge(
            da[[gc_a, fc_a]].rename(columns={gc_a: "gene", fc_a: "fc_a"}),
            db[[gc_b, fc_b]].rename(columns={gc_b: "gene", fc_b: "fc_b"}),
            on="gene", how="inner",
        )
        rho[pair] = float(spearmanr(merged["fc_a"], merged["fc_b"]).statistic) if len(merged) >= 3 else np.nan

    return jaccard, rho


# ============================================================
# Module scores (biology preservation)
# ============================================================

def compute_module_scores(adata: sc.AnnData, marker_sets: dict[str, list[str]]) -> pd.DataFrame:
    scores = {}
    for name, genes in marker_sets.items():
        present = [g for g in genes if g in adata.var_names]
        if len(present) < 2:
            continue
        key = f"score_{name}"
        sc.tl.score_genes(adata, gene_list=present, score_name=key, use_raw=False)
        scores[name] = np.asarray(adata.obs[key])
    return pd.DataFrame(scores, index=adata.obs_names.astype(str))


def module_profile_correlations(
    score_df: pd.DataFrame,
    labels_a: pd.Series, labels_b: pd.Series,
    matching: dict[str, str],
) -> dict[str, float]:
    common = score_df.index.intersection(labels_a.index).intersection(labels_b.index)
    if len(common) == 0:
        return {}
    mean_a = score_df.loc[common].assign(cluster=labels_a.loc[common].values).groupby("cluster").mean(numeric_only=True)
    mean_b = score_df.loc[common].assign(cluster=labels_b.loc[common].values).groupby("cluster").mean(numeric_only=True)

    out = {}
    for ca, cb in matching.items():
        if ca not in mean_a.index or cb not in mean_b.index:
            out[f"{ca}<->{cb}"] = np.nan
            continue
        va, vb = mean_a.loc[ca].astype(float), mean_b.loc[cb].astype(float)
        shared = va.index.intersection(vb.index)
        out[f"{ca}<->{cb}"] = float(spearmanr(va.loc[shared], vb.loc[shared]).statistic) if len(shared) >= 2 else np.nan
    return out


# ============================================================
# Standardized DE (on canonical matrix, using each pipeline's clusters)
# ============================================================

def standardized_markers(
    canonical_h5ad: str,
    labels: pd.Series,
    target_sum: float,
    method: str,
    corr_method: str,
    out_csv: str,
    use_gpu: bool = False,
) -> dict[str, pd.DataFrame]:
    adata = sc.read_h5ad(canonical_h5ad)
    adata.obs_names = make_unique_normalized_index(adata.obs_names, canonical_h5ad)

    common = adata.obs_names.intersection(labels.index)
    if len(common) == 0:
        raise ValueError("No overlap between canonical AnnData and cluster labels.")
    adata = adata[common].copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()

    adata.obs["cluster_labels"] = pd.Categorical(labels.loc[common].astype(str).values)

    if use_gpu:
        import rapids_singlecell as rsc

        rsc.get.anndata_to_GPU(adata)
        rsc.pp.normalize_total(adata, target_sum=target_sum)
        rsc.pp.log1p(adata)

        rank_sig = inspect.signature(rsc.tl.rank_genes_groups)
        rank_kw: dict[str, Any] = {
            "groupby": "cluster_labels",
            "method": method, "corr_method": corr_method,
            "use_raw": False, "pts": True,
            "key_added": "std_rank_genes_groups",
        }
        for k, v in {"pre_load": True, "tie_correct": False, "use_continuity": False}.items():
            if k in rank_sig.parameters:
                rank_kw[k] = v
        rsc.tl.rank_genes_groups(adata, **rank_kw)
        rsc.get.anndata_to_CPU(adata)
    else:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        sc.tl.rank_genes_groups(
            adata, groupby="cluster_labels",
            method=method, corr_method=corr_method,
            use_raw=False, pts=True,
            key_added="std_rank_genes_groups",
        )

    df = sc.get.rank_genes_groups_df(adata, group=None, key="std_rank_genes_groups")
    df.to_csv(out_csv, index=False)
    return {str(g): gdf.copy() for g, gdf in df.groupby("group")}


# ============================================================
# Pairwise comparison engine
# ============================================================

def safe_mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if pd.notna(v)]
    return None if not vals else float(np.mean(vals))


def compare_pair(
    name_a: str, name_b: str,
    cluster_data: dict[str, pd.Series],
    native_markers: dict[str, dict[str, pd.DataFrame]],
    standardized_marker_data: dict[str, dict[str, pd.DataFrame]],
    score_df: pd.DataFrame,
    pipelines: dict[str, PipelineFiles],
    out_dir: str,
    prefix: str,
) -> dict[str, Any]:
    labels_a, labels_b = align_on_common_barcodes(cluster_data[name_a], cluster_data[name_b])
    matching = compute_matching(labels_a, labels_b)
    dice = compute_dice(labels_a, labels_b, matching)
    module_corr = module_profile_correlations(score_df, labels_a, labels_b, matching)

    result: dict[str, Any] = {
        "n_common_cells": int(len(labels_a)),
        "n_clusters_a": int(labels_a.nunique()),
        "n_clusters_b": int(labels_b.nunique()),
        "ARI": float(adjusted_rand_score(labels_a, labels_b)),
        "NMI": float(normalized_mutual_info_score(labels_a, labels_b)),
        "matching": matching,
        "dice_per_cluster": dice,
        "mean_dice": safe_mean(dice.values()),
        "module_profile_rho_per_cluster": module_corr,
        "mean_module_profile_rho": safe_mean(module_corr.values()),
    }

    # Native DE comparison
    if name_a in native_markers and name_b in native_markers:
        nj, nr = jaccard_and_spearman(
            native_markers[name_a], native_markers[name_b],
            pipelines[name_a].kind, pipelines[name_b].kind, matching,
        )
        result.update({
            "native_deg_jaccard_per_cluster": nj,
            "native_deg_spearman_per_cluster": nr,
            "mean_native_deg_jaccard": safe_mean(nj.values()),
            "mean_native_deg_spearman": safe_mean(nr.values()),
        })
    else:
        result.update({
            "native_deg_jaccard_per_cluster": None,
            "native_deg_spearman_per_cluster": None,
            "mean_native_deg_jaccard": None,
            "mean_native_deg_spearman": None,
        })

    # Standardized DE comparison
    if name_a in standardized_marker_data and name_b in standardized_marker_data:
        sj, sr = jaccard_and_spearman(
            standardized_marker_data[name_a], standardized_marker_data[name_b],
            "scanpy_like", "scanpy_like", matching,
        )
        result.update({
            "standardized_deg_jaccard_per_cluster": sj,
            "standardized_deg_spearman_per_cluster": sr,
            "mean_standardized_deg_jaccard": safe_mean(sj.values()),
            "mean_standardized_deg_spearman": safe_mean(sr.values()),
        })
    else:
        result.update({
            "standardized_deg_jaccard_per_cluster": None,
            "standardized_deg_spearman_per_cluster": None,
            "mean_standardized_deg_jaccard": None,
            "mean_standardized_deg_spearman": None,
        })

    # Contingency table
    contingency = pd.crosstab(labels_a, labels_b)
    contingency.to_csv(os.path.join(out_dir, f"{prefix}_{name_a}__vs__{name_b}_contingency.csv"))

    return result


# ============================================================
# Group logic
# ============================================================

def split_by_workflow(
    pipelines: dict[str, PipelineFiles],
) -> tuple[dict[str, PipelineFiles], dict[str, PipelineFiles]]:
    """Split detected pipelines into minimal and full groups."""
    minimal = {k: v for k, v in pipelines.items() if v.workflow == "minimal"}
    full = {k: v for k, v in pipelines.items() if v.workflow == "full"}
    return minimal, full


def get_cross_pairs(
    minimal: dict[str, PipelineFiles],
    full: dict[str, PipelineFiles],
) -> list[tuple[str, str]]:
    """
    Cross-workflow pairs: same platform, different workflow.
    E.g. scanpy_cpu (minimal) vs scanpy_cpu_full (full).
    """
    pairs = []
    platform_map = {
        "scanpy_cpu": "scanpy_cpu_full",
        "rsc_gpu_0141": "rsc_gpu_0141_full",
        "rsc_gpu_015": "rsc_gpu_015_full",
    }
    for m_name, f_name in platform_map.items():
        if m_name in minimal and f_name in full:
            pairs.append((m_name, f_name))
    return pairs


# ============================================================
# Summary + reporting
# ============================================================

def make_summary_row(
    dataset_name: str, name_a: str, name_b: str,
    group: str, result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "group": group,
        "comparison": f"{name_a} vs {name_b}",
        "common_cells": result["n_common_cells"],
        "clusters_a": result["n_clusters_a"],
        "clusters_b": result["n_clusters_b"],
        "ARI": result["ARI"],
        "NMI": result["NMI"],
        "mean_dice": result["mean_dice"],
        "mean_native_deg_jaccard": result["mean_native_deg_jaccard"],
        "mean_native_deg_spearman": result["mean_native_deg_spearman"],
        "mean_standardized_deg_jaccard": result["mean_standardized_deg_jaccard"],
        "mean_standardized_deg_spearman": result["mean_standardized_deg_spearman"],
        "mean_module_profile_rho": result["mean_module_profile_rho"],
    }


def print_group_table(rows: list[dict], group_name: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = ["comparison", "clusters_a", "clusters_b", "ARI", "NMI",
            "mean_dice", "mean_module_profile_rho",
            "mean_standardized_deg_jaccard", "mean_standardized_deg_spearman"]
    display_cols = [c for c in cols if c in df.columns]
    print(f"\n{'─' * 72}")
    print(f"  {group_name}")
    print(f"{'─' * 72}")
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.3f}".format):
        print(df[display_cols].to_string(index=False))


def df_to_notion_md(df: pd.DataFrame, title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"### {title}\n")
    cols = df.columns.tolist()
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values
        ) + " |")
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    gcfg, dcfg = load_config(args.config, args.dataset)
    prefix = dcfg["pipeline_prefix"]

    # GPU init
    if args.gpu:
        try:
            import cupy as cp
            import rmm
            from rmm.allocators.cupy import rmm_cupy_allocator
            rmm.reinitialize(pool_allocator=True, managed_memory=True)
            cp.cuda.set_allocator(rmm_cupy_allocator)
            print("GPU mode: RMM initialized\n")
        except Exception as e:
            print(f"WARNING: GPU init failed ({e}), falling back to CPU\n")
            args.gpu = False

    # Discover pipelines
    registry = pipeline_registry(prefix)
    detected = {k: v for k, v in registry.items() if os.path.exists(v.clusters)}
    minimal, full = split_by_workflow(detected)

    print("=" * 72)
    print(f"Validation — {dcfg['name']}")
    print("=" * 72)
    print(f"\nMinimal pipelines ({len(minimal)}):")
    for n, f in minimal.items():
        print(f"  {n:25s} {f.clusters}")
    print(f"\nFull pipelines ({len(full)}):")
    for n, f in full.items():
        print(f"  {n:25s} {f.clusters}")
    print()

    # Load all cluster data
    cluster_data = {name: load_clusters(files.clusters) for name, files in detected.items()}

    # Load native markers
    native_markers: dict[str, dict[str, pd.DataFrame]] = {}
    for name, files in detected.items():
        if files.markers and os.path.exists(files.markers):
            native_markers[name] = load_native_markers(files.markers, files.kind)

    # Standardized DE for all detected pipelines
    std_dir = f"write/validation_{prefix}_standardized_de"
    ensure_dir(std_dir)
    standardized_marker_data: dict[str, dict[str, pd.DataFrame]] = {}
    for name in detected:
        out_csv = os.path.join(std_dir, f"{prefix}_{name}_standardized_markers.csv")
        print(f"  Standardized DE: {name:25s} ({'GPU' if args.gpu else 'CPU'}) ...", end=" ", flush=True)
        t0 = time.time()
        standardized_marker_data[name] = standardized_markers(
            canonical_h5ad=dcfg["canonical_h5ad"],
            labels=cluster_data[name],
            target_sum=gcfg["target_sum"],
            method=gcfg["de_method"],
            corr_method=gcfg["de_corr_method"],
            out_csv=out_csv,
            use_gpu=args.gpu,
        )
        print(f"{time.time() - t0:.1f}s")

    # Biology-preservation scores (shared across all groups)
    print("\n  Computing module scores ...", end=" ", flush=True)
    t0 = time.time()
    adata_for_scores = sc.read_h5ad(dcfg["canonical_h5ad"])
    adata_for_scores.obs_names = make_unique_normalized_index(adata_for_scores.obs_names, dcfg["canonical_h5ad"])
    if "counts" in adata_for_scores.layers:
        adata_for_scores.X = adata_for_scores.layers["counts"].copy()
    sc.pp.normalize_total(adata_for_scores, target_sum=gcfg["target_sum"])
    sc.pp.log1p(adata_for_scores)
    score_df = compute_module_scores(adata_for_scores, dcfg["known_marker_sets"])
    del adata_for_scores
    print(f"{time.time() - t0:.1f}s")

    # ---- Run comparison groups ----

    all_results: dict[str, dict[str, Any]] = {}
    all_summary_rows: list[dict[str, Any]] = []

    def run_group(group_name: str, pairs: list[tuple[str, str]], out_dir: str):
        ensure_dir(out_dir)
        group_results = {}
        group_rows = []
        for na, nb in pairs:
            pair_key = f"{na}__vs__{nb}"
            print(f"    {na} vs {nb} ...", end=" ", flush=True)
            t0 = time.time()
            result = compare_pair(
                na, nb, cluster_data, native_markers,
                standardized_marker_data, score_df, detected, out_dir, prefix,
            )
            print(f"{time.time() - t0:.1f}s  ARI={result['ARI']:.3f}  Module ρ={result.get('mean_module_profile_rho', 'N/A')}")
            group_results[pair_key] = result
            group_rows.append(make_summary_row(dcfg["name"], na, nb, group_name, result))

        if group_rows:
            df = pd.DataFrame(group_rows)
            df.to_csv(os.path.join(out_dir, f"{prefix}_{group_name}_summary.csv"), index=False)
            with open(os.path.join(out_dir, f"{prefix}_{group_name}_details.json"), "w") as f:
                json.dump(group_results, f, indent=2)
            print_group_table(group_rows, f"{dcfg['name']} — {group_name.upper()}")

        all_results.update(group_results)
        all_summary_rows.extend(group_rows)

    # Group 1: MINIMAL vs MINIMAL
    if "minimal" in args.groups and len(minimal) >= 2:
        print(f"\n{'=' * 72}")
        print("  GROUP: MINIMAL (within-workflow)")
        print(f"{'=' * 72}")
        out_dir = f"write/validation_{prefix}_minimal"
        pairs = list(combinations(minimal.keys(), 2))
        run_group("minimal", pairs, out_dir)
    elif "minimal" in args.groups:
        print("\n  Skipping MINIMAL group: fewer than 2 pipelines detected.")

    # Group 2: FULL vs FULL
    if "full" in args.groups and len(full) >= 2:
        print(f"\n{'=' * 72}")
        print("  GROUP: FULL (within-workflow)")
        print(f"{'=' * 72}")
        out_dir = f"write/validation_{prefix}_full"
        pairs = list(combinations(full.keys(), 2))
        run_group("full", pairs, out_dir)
    elif "full" in args.groups:
        print("\n  Skipping FULL group: fewer than 2 pipelines detected.")

    # Group 3: CROSS (minimal vs full, same platform)
    if "cross" in args.groups:
        cross_pairs = get_cross_pairs(minimal, full)
        if len(cross_pairs) >= 1:
            print(f"\n{'=' * 72}")
            print("  GROUP: CROSS (minimal vs full, same platform)")
            print(f"{'=' * 72}")
            out_dir = f"write/validation_{prefix}_cross"
            run_group("cross", cross_pairs, out_dir)
        else:
            print("\n  Skipping CROSS group: no matching minimal/full platform pairs.")

    # ---- Combined output ----
    if all_summary_rows:
        combined_df = pd.DataFrame(all_summary_rows)
        combined_csv = f"write/validation_{prefix}_all_groups_summary.csv"
        combined_json = f"write/validation_{prefix}_all_groups_details.json"
        combined_df.to_csv(combined_csv, index=False)
        with open(combined_json, "w") as f:
            json.dump(all_results, f, indent=2)

        # Notion markdown
        notion_md = f"write/validation_{prefix}_notion.md"
        with open(notion_md, "w") as f:
            f.write(f"# Validation: {dcfg['name']}\n\n")
            for group_name in ["minimal", "full", "cross"]:
                gdf = combined_df[combined_df["group"] == group_name]
                if gdf.empty:
                    continue
                display_cols = ["comparison", "clusters_a", "clusters_b", "ARI", "NMI",
                                "mean_dice", "mean_module_profile_rho",
                                "mean_standardized_deg_jaccard"]
                f.write(df_to_notion_md(gdf[[c for c in display_cols if c in gdf.columns]], title=group_name.upper()))
                f.write("\n\n")

        print(f"\n{'=' * 72}")
        print("  OUTPUT FILES")
        print(f"{'=' * 72}")
        print(f"  Combined summary : {combined_csv}")
        print(f"  Combined details : {combined_json}")
        print(f"  Notion markdown  : {notion_md}")
        for group_name in ["minimal", "full", "cross"]:
            d = f"write/validation_{prefix}_{group_name}"
            if os.path.isdir(d):
                print(f"  {group_name:8s} dir     : {d}/")
        print(f"  Standardized DE  : {std_dir}/")
    else:
        print("\nNo comparisons were run.")


if __name__ == "__main__":
    main()
