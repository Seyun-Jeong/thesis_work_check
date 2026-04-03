#!/usr/bin/env python3
"""
Cross-pipeline validation for the harmonized benchmark.

What this script does better than the earlier notebook/script
------------------------------------------------------------
1. It never aligns cells by position.
2. It normalizes barcodes explicitly and errors on duplicate collisions.
3. It uses Hungarian matching for cluster correspondence.
4. It compares both native DE and standardized DE separately.
5. It computes biology-preservation metrics on a canonical normalized matrix.

Why standardized DE matters
---------------------------
If each library runs a different DE engine, then differences in DEG lists can
come from either:
- different cluster assignments, or
- different DE implementations.

To separate those, this script recomputes *standardized* Wilcoxon DE for each
pipeline's cluster labels on the same canonical normalized/log1p matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass
class PipelineFiles:
    name: str
    clusters: str
    markers: str | None
    h5ad: str | None
    kind: str  # scanpy_like | seurat | scalesc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["pbmc3k", "lung65k"])
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "benchmark_config.json"),
    )
    return parser.parse_args()


def load_config(path: str, dataset_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["global"], cfg["datasets"][dataset_key]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_barcode(x: str) -> str:
    x = str(x).strip()
    x = x.replace('"', "")
    x = re.sub(r"-1$", "", x)
    return x


def make_unique_normalized_index(index: Iterable[str], label: str) -> pd.Index:
    normalized = pd.Index([normalize_barcode(x) for x in index])
    dup = normalized[normalized.duplicated()].unique().tolist()
    if dup:
        raise ValueError(
            f"Normalized barcode collisions detected in {label}. "
            f"Examples: {dup[:5]}"
        )
    return normalized


def load_clusters(path: str) -> pd.Series:
    df = pd.read_csv(path)
    barcode_col = "barcode" if "barcode" in df.columns else df.columns[0]
    cluster_col = "leiden" if "leiden" in df.columns else df.columns[-1]
    s = pd.Series(df[cluster_col].astype(str).values, index=df[barcode_col].astype(str).values, name="cluster")
    s.index = make_unique_normalized_index(s.index, path)
    return s


def load_native_markers(path: str, kind: str) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    out: dict[str, pd.DataFrame] = {}

    if kind in {"scanpy_like", "scalesc"}:
        if "group" not in df.columns or "names" not in df.columns:
            raise ValueError(f"Expected Scanpy-like marker columns in {path}")
        for grp, gdf in df.groupby("group"):
            out[str(grp)] = gdf.copy()
        return out

    if kind == "seurat":
        if "cluster" not in df.columns or "gene" not in df.columns:
            raise ValueError(f"Expected Seurat marker columns in {path}")
        for grp, gdf in df.groupby("cluster"):
            out[str(grp)] = gdf.copy()
        return out

    raise ValueError(f"Unknown marker kind: {kind}")


def get_native_marker_columns(kind: str) -> tuple[str, str, str, str]:
    if kind in {"scanpy_like", "scalesc"}:
        return ("names", "logfoldchanges", "pvals_adj", "group")
    if kind == "seurat":
        return ("gene", "avg_log2FC", "p_val_adj", "cluster")
    raise ValueError(kind)


def align_on_common_barcodes(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError("No common normalized barcodes between the two pipelines.")
    return a.loc[common], b.loc[common]


def compute_matching(labels_a: pd.Series, labels_b: pd.Series) -> dict[str, str]:
    contingency = pd.crosstab(labels_a, labels_b)
    row_labels = contingency.index.astype(str).tolist()
    col_labels = contingency.columns.astype(str).tolist()
    cost = -contingency.to_numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return {row_labels[i]: col_labels[j] for i, j in zip(row_ind, col_ind)}


def compute_dice(labels_a: pd.Series, labels_b: pd.Series, matching: dict[str, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for ca, cb in matching.items():
        set_a = set(labels_a.index[labels_a == ca])
        set_b = set(labels_b.index[labels_b == cb])
        denom = len(set_a) + len(set_b)
        out[f"{ca}<->{cb}"] = (2.0 * len(set_a & set_b) / denom) if denom else np.nan
    return out


def filtered_gene_set(df: pd.DataFrame, gene_col: str, fc_col: str, padj_col: str) -> set[str]:
    keep = df[(df[padj_col] < 0.05) & (df[fc_col] > 0.1)]
    return set(keep[gene_col].astype(str).tolist())


def jaccard_and_spearman(
    markers_a: dict[str, pd.DataFrame],
    markers_b: dict[str, pd.DataFrame],
    kind_a: str,
    kind_b: str,
    matching: dict[str, str],
) -> tuple[dict[str, float], dict[str, float]]:
    gene_col_a, fc_col_a, padj_col_a, _ = get_native_marker_columns(kind_a)
    gene_col_b, fc_col_b, padj_col_b, _ = get_native_marker_columns(kind_b)

    jaccard: dict[str, float] = {}
    rho: dict[str, float] = {}

    for ca, cb in matching.items():
        df_a = markers_a.get(str(ca), pd.DataFrame())
        df_b = markers_b.get(str(cb), pd.DataFrame())
        genes_a = filtered_gene_set(df_a, gene_col_a, fc_col_a, padj_col_a) if not df_a.empty else set()
        genes_b = filtered_gene_set(df_b, gene_col_b, fc_col_b, padj_col_b) if not df_b.empty else set()

        pair = f"{ca}<->{cb}"
        union = genes_a | genes_b
        jaccard[pair] = (len(genes_a & genes_b) / len(union)) if union else np.nan

        if df_a.empty or df_b.empty:
            rho[pair] = np.nan
            continue

        merged = pd.merge(
            df_a[[gene_col_a, fc_col_a]].rename(columns={gene_col_a: "gene", fc_col_a: "fc_a"}),
            df_b[[gene_col_b, fc_col_b]].rename(columns={gene_col_b: "gene", fc_col_b: "fc_b"}),
            on="gene",
            how="inner",
        )
        if len(merged) < 3:
            rho[pair] = np.nan
            continue
        rho[pair] = float(spearmanr(merged["fc_a"], merged["fc_b"]).statistic)

    return jaccard, rho


def compute_module_scores(adata: sc.AnnData, marker_sets: dict[str, list[str]]) -> pd.DataFrame:
    scores: dict[str, np.ndarray] = {}
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
    labels_a: pd.Series,
    labels_b: pd.Series,
    matching: dict[str, str],
) -> dict[str, float]:
    common = score_df.index.intersection(labels_a.index).intersection(labels_b.index)
    if len(common) == 0:
        return {}

    mean_a = score_df.loc[common].assign(cluster=labels_a.loc[common].values).groupby("cluster").mean(numeric_only=True)
    mean_b = score_df.loc[common].assign(cluster=labels_b.loc[common].values).groupby("cluster").mean(numeric_only=True)

    out: dict[str, float] = {}
    for ca, cb in matching.items():
        if ca not in mean_a.index or cb not in mean_b.index:
            out[f"{ca}<->{cb}"] = np.nan
            continue
        vec_a = mean_a.loc[ca].astype(float)
        vec_b = mean_b.loc[cb].astype(float)
        shared = vec_a.index.intersection(vec_b.index)
        if len(shared) < 2:
            out[f"{ca}<->{cb}"] = np.nan
            continue
        out[f"{ca}<->{cb}"] = float(spearmanr(vec_a.loc[shared], vec_b.loc[shared]).statistic)
    return out


def standardized_markers(
    canonical_h5ad: str,
    labels: pd.Series,
    target_sum: float,
    method: str,
    corr_method: str,
    out_csv: str,
) -> dict[str, pd.DataFrame]:
    adata = sc.read_h5ad(canonical_h5ad)
    adata.obs_names = make_unique_normalized_index(adata.obs_names, canonical_h5ad)

    common = adata.obs_names.intersection(labels.index)
    if len(common) == 0:
        raise ValueError("No overlap between canonical AnnData and cluster labels for standardized DE.")

    adata = adata[common].copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.obs["cluster_labels"] = pd.Categorical(labels.loc[common].astype(str).values)

    sc.tl.rank_genes_groups(
        adata,
        groupby="cluster_labels",
        method=method,
        corr_method=corr_method,
        use_raw=False,
        pts=True,
        key_added="std_rank_genes_groups",
    )
    df = sc.get.rank_genes_groups_df(adata, group=None, key="std_rank_genes_groups")
    df.to_csv(out_csv, index=False)

    out: dict[str, pd.DataFrame] = {}
    for grp, gdf in df.groupby("group"):
        out[str(grp)] = gdf.copy()
    return out


def pipeline_registry(prefix: str) -> dict[str, PipelineFiles]:
    return {
        "scanpy_cpu": PipelineFiles(
            name="scanpy_cpu",
            clusters=f"write/{prefix}_scanpy_cpu_harmonized_clusters.csv",
            markers=f"write/{prefix}_scanpy_cpu_harmonized_markers.csv",
            h5ad=f"write/{prefix}_scanpy_cpu_harmonized.h5ad",
            kind="scanpy_like",
        ),
        "rsc_gpu_0141": PipelineFiles(
            name="rsc_gpu_0141",
            clusters=f"write/{prefix}_rsc_gpu_0141_harmonized_clusters.csv",
            markers=f"write/{prefix}_rsc_gpu_0141_harmonized_markers.csv",
            h5ad=f"write/{prefix}_rsc_gpu_0141_harmonized.h5ad",
            kind="scanpy_like",
        ),
        "seurat_cpu": PipelineFiles(
            name="seurat_cpu",
            clusters=f"write/{prefix}_seurat_cpu_harmonized_clusters.csv",
            markers=f"write/{prefix}_seurat_cpu_harmonized_markers.csv",
            h5ad=None,
            kind="seurat",
        ),
        "scalesc_gpu": PipelineFiles(
            name="scalesc_gpu",
            clusters=f"write/{prefix}_scalesc_gpu_harmonized_clusters.csv",
            markers=None,
            h5ad=f"write/{prefix}_scalesc_gpu_harmonized.h5ad",
            kind="scalesc",
        ),
        "rsc_gpu_015": PipelineFiles(
            name="rsc_gpu_015",
            clusters=f"write/{prefix}_rsc_gpu_015_harmonized_clusters.csv",
            markers=f"write/{prefix}_rsc_gpu_015_harmonized_markers.csv",
            h5ad=f"write/{prefix}_rsc_gpu_015_harmonized.h5ad",
            kind="scanpy_like",
        ),
    }


def existing_pipelines(registry: dict[str, PipelineFiles]) -> dict[str, PipelineFiles]:
    return {
        name: files
        for name, files in registry.items()
        if os.path.exists(files.clusters)
    }


def safe_mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if pd.notna(v)]
    return None if not vals else float(np.mean(vals))


def main() -> None:
    args = parse_args()
    gcfg, dcfg = load_config(args.config, args.dataset)
    prefix = dcfg["pipeline_prefix"]
    out_dir = f"write/validation_{prefix}_harmonized"
    ensure_dir(out_dir)

    registry = pipeline_registry(prefix)
    pipelines = existing_pipelines(registry)
    if len(pipelines) < 2:
        raise RuntimeError(
            f"Need at least two completed pipelines for validation. Found: {list(pipelines)}"
        )

    print("=" * 72)
    print(f"Validation — {dcfg['name']}")
    print("=" * 72)
    print("Pipelines detected:")
    for name, files in pipelines.items():
        print(f"  {name:12s} {files.clusters}")
    print()

    cluster_data = {name: load_clusters(files.clusters) for name, files in pipelines.items()}

    # Native markers when available.
    native_markers: dict[str, dict[str, pd.DataFrame]] = {}
    for name, files in pipelines.items():
        if files.markers and os.path.exists(files.markers):
            native_markers[name] = load_native_markers(files.markers, files.kind)

    # Standardized DE on a common matrix for every pipeline with clusters.
    standardized_marker_data: dict[str, dict[str, pd.DataFrame]] = {}
    for name in pipelines:
        out_csv = os.path.join(out_dir, f"{prefix}_{name}_standardized_markers.csv")
        standardized_marker_data[name] = standardized_markers(
            canonical_h5ad=dcfg["canonical_h5ad"],
            labels=cluster_data[name],
            target_sum=gcfg["target_sum"],
            method=gcfg["de_method"],
            corr_method=gcfg["de_corr_method"],
            out_csv=out_csv,
        )

    # Biology-preservation scores on a common normalized matrix.
    adata_for_scores = sc.read_h5ad(dcfg["canonical_h5ad"])
    adata_for_scores.obs_names = make_unique_normalized_index(adata_for_scores.obs_names, dcfg["canonical_h5ad"])
    if "counts" in adata_for_scores.layers:
        adata_for_scores.X = adata_for_scores.layers["counts"].copy()
    sc.pp.normalize_total(adata_for_scores, target_sum=gcfg["target_sum"])
    sc.pp.log1p(adata_for_scores)
    score_df = compute_module_scores(adata_for_scores, dcfg["known_marker_sets"])

    all_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for name_a, name_b in combinations(pipelines.keys(), 2):
        labels_a, labels_b = align_on_common_barcodes(cluster_data[name_a], cluster_data[name_b])
        matching = compute_matching(labels_a, labels_b)
        dice = compute_dice(labels_a, labels_b, matching)
        module_corr = module_profile_correlations(score_df, labels_a, labels_b, matching)

        result = {
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

        if name_a in native_markers and name_b in native_markers:
            nat_j, nat_rho = jaccard_and_spearman(
                native_markers[name_a], native_markers[name_b],
                pipelines[name_a].kind, pipelines[name_b].kind,
                matching,
            )
            result["native_deg_jaccard_per_cluster"] = nat_j
            result["native_deg_spearman_per_cluster"] = nat_rho
            result["mean_native_deg_jaccard"] = safe_mean(nat_j.values())
            result["mean_native_deg_spearman"] = safe_mean(nat_rho.values())
        else:
            result["native_deg_jaccard_per_cluster"] = None
            result["native_deg_spearman_per_cluster"] = None
            result["mean_native_deg_jaccard"] = None
            result["mean_native_deg_spearman"] = None

        std_j, std_rho = jaccard_and_spearman(
            standardized_marker_data[name_a], standardized_marker_data[name_b],
            "scanpy_like", "scanpy_like", matching,
        )
        result["standardized_deg_jaccard_per_cluster"] = std_j
        result["standardized_deg_spearman_per_cluster"] = std_rho
        result["mean_standardized_deg_jaccard"] = safe_mean(std_j.values())
        result["mean_standardized_deg_spearman"] = safe_mean(std_rho.values())

        pair_key = f"{name_a}__vs__{name_b}"
        all_results[pair_key] = result

        # Save contingency table for transparency.
        contingency = pd.crosstab(labels_a, labels_b)
        contingency.to_csv(os.path.join(out_dir, f"{prefix}_{pair_key}_contingency.csv"))

        summary_rows.append({
            "dataset": dcfg["name"],
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
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_dir, f"{prefix}_validation_summary.csv")
    detail_json = os.path.join(out_dir, f"{prefix}_validation_details.json")
    summary_df.to_csv(summary_csv, index=False)
    with open(detail_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("Summary:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df.to_string(index=False))
    print("\nWrote:")
    print(f"  {summary_csv}")
    print(f"  {detail_json}")
    print(f"  standardized markers in {out_dir}/")


if __name__ == "__main__":
    main()
