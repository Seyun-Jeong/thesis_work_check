#!/usr/bin/env python3
"""
Harmonized Scanpy CPU pipeline.

Design goal
-----------
This is the CPU reference pipeline for the benchmark. It uses the canonical
filtered input produced by `prepare_canonical_inputs.py`, so all downstream
libraries start from the same cells and genes.

Important choices
-----------------
- HVG: `seurat_v3` on the raw-count layer (`layer='counts'`)
- PCA: Scanpy CPU with `svd_solver='arpack'`
- Neighbors: exact sklearn brute-force transformer (not size-dependent default)
- Leiden: Scanpy igraph flavor, with explicit `n_iterations` only if configured
- UMAP: visualization only; not used as the primary validation target
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd
import scanpy as sc
from sklearn.neighbors import KNeighborsTransformer


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


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def output_paths(prefix: str) -> dict[str, str]:
    base = "write"
    return {
        "h5ad": f"{base}/{prefix}_scanpy_cpu_harmonized.h5ad",
        "clusters": f"{base}/{prefix}_scanpy_cpu_harmonized_clusters.csv",
        "markers": f"{base}/{prefix}_scanpy_cpu_harmonized_markers.csv",
        "markers_filtered": f"{base}/{prefix}_scanpy_cpu_harmonized_markers_filtered.csv",
        "umap": f"{base}/{prefix}_scanpy_cpu_harmonized_umap.csv",
        "hvg": f"{base}/{prefix}_scanpy_cpu_harmonized_hvg.csv",
        "spec": f"{base}/{prefix}_scanpy_cpu_harmonized_spec.json",
    }


def main() -> None:
    args = parse_args()
    gcfg, dcfg = load_config(args.config, args.dataset)
    paths = output_paths(dcfg["pipeline_prefix"])
    for p in paths.values():
        ensure_parent(p)

    adata = sc.read_h5ad(dcfg["canonical_h5ad"])
    if "counts" not in adata.layers:
        raise ValueError(
            f"Canonical input {dcfg['canonical_h5ad']} is missing .layers['counts']. "
            "Re-run prepare_canonical_inputs.py first."
        )

    print("=" * 72)
    print(f"Scanpy CPU harmonized run — {dcfg['name']}")
    print("=" * 72)
    print(f"Input           : {dcfg['canonical_h5ad']}")
    print(f"Cells × genes   : {adata.n_obs:,} × {adata.n_vars:,}")
    print()

    # Normalize + log1p
    sc.pp.normalize_total(adata, target_sum=gcfg["target_sum"])
    sc.pp.log1p(adata)

    # HVG on raw counts for seurat_v3-style selection.
    sc.pp.highly_variable_genes(
        adata,
        layer="counts",
        n_top_genes=dcfg["n_top_genes"],
        flavor="seurat_v3",
        subset=False,
        inplace=True,
    )

    # PCA on HVGs.
    sc.pp.pca(
        adata,
        n_comps=gcfg["pca_n_comps"],
        svd_solver="arpack",
        random_state=gcfg["random_state"],
        mask_var="highly_variable",
    )

    # Exact CPU KNN to avoid Scanpy's size-dependent default behavior.
    transformer = KNeighborsTransformer(
        n_neighbors=dcfg["n_neighbors"],
        mode="distance",
        metric=gcfg["neighbor_metric"],
        algorithm="brute",
    )
    sc.pp.neighbors(
        adata,
        n_neighbors=dcfg["n_neighbors"],
        n_pcs=dcfg["neighbors_n_pcs"],
        use_rep="X_pca",
        method=gcfg["neighbor_method"],
        transformer=transformer,
        metric=gcfg["neighbor_metric"],
        random_state=gcfg["random_state"],
    )

    leiden_kwargs = {
        "resolution": dcfg["leiden_resolution"],
        "flavor": gcfg["scanpy_leiden_flavor"],
        "random_state": gcfg["random_state"],
        "key_added": "leiden",
    }
    if gcfg.get("scanpy_leiden_n_iterations") is not None:
        leiden_kwargs["n_iterations"] = gcfg["scanpy_leiden_n_iterations"]
    sc.tl.leiden(adata, **leiden_kwargs)

    sc.tl.umap(
        adata,
        min_dist=gcfg["umap_min_dist"],
        spread=gcfg["umap_spread"],
        init_pos="spectral",
        random_state=gcfg["random_state"],
    )

    sc.tl.rank_genes_groups(
        adata,
        groupby="leiden",
        method=gcfg["de_method"],
        corr_method=gcfg["de_corr_method"],
        use_raw=False,
        pts=True,
    )

    markers = sc.get.rank_genes_groups_df(adata, group=None)
    markers_filtered = markers[
        (markers["pvals_adj"] < 0.05) & (markers["logfoldchanges"] > 0.1)
    ].copy()

    clusters_df = pd.DataFrame({
        "barcode": adata.obs_names.astype(str),
        "leiden": adata.obs["leiden"].astype(str).values,
    })
    umap_df = pd.DataFrame(
        adata.obsm["X_umap"],
        index=adata.obs_names.astype(str),
        columns=["UMAP_1", "UMAP_2"],
    ).reset_index(names="barcode")
    hvg_df = adata.var.loc[adata.var["highly_variable"].astype(bool), []].copy()
    hvg_df.index.name = "gene"
    hvg_df = hvg_df.reset_index()

    clusters_df.to_csv(paths["clusters"], index=False)
    markers.to_csv(paths["markers"], index=False)
    markers_filtered.to_csv(paths["markers_filtered"], index=False)
    umap_df.to_csv(paths["umap"], index=False)
    hvg_df.to_csv(paths["hvg"], index=False)
    adata.write_h5ad(paths["h5ad"], compression="gzip")

    spec = {
        "pipeline": "scanpy_cpu_harmonized",
        "dataset": dcfg["name"],
        "scanpy_version": sc.__version__,
        "input": dcfg["canonical_h5ad"],
        "parameters": {
            "target_sum": gcfg["target_sum"],
            "hvg_flavor": "seurat_v3",
            "hvg_layer": "counts",
            "n_top_genes": dcfg["n_top_genes"],
            "pca_n_comps": gcfg["pca_n_comps"],
            "pca_svd_solver": "arpack",
            "n_neighbors": dcfg["n_neighbors"],
            "neighbors_n_pcs": dcfg["neighbors_n_pcs"],
            "neighbor_metric": gcfg["neighbor_metric"],
            "neighbor_method": gcfg["neighbor_method"],
            "neighbor_backend": "sklearn.KNeighborsTransformer(brute)",
            "leiden_flavor": gcfg["scanpy_leiden_flavor"],
            "leiden_n_iterations": gcfg.get("scanpy_leiden_n_iterations"),
            "leiden_resolution": dcfg["leiden_resolution"],
            "umap_min_dist": gcfg["umap_min_dist"],
            "umap_spread": gcfg["umap_spread"],
            "umap_init_pos": "spectral",
            "random_state": gcfg["random_state"],
            "de_method": gcfg["de_method"],
            "de_corr_method": gcfg["de_corr_method"],
        },
        "results": {
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_hvg": int(adata.var["highly_variable"].sum()),
            "n_clusters": int(adata.obs["leiden"].nunique()),
            "n_marker_rows": int(len(markers)),
            "n_marker_rows_filtered": int(len(markers_filtered)),
        },
    }
    with open(paths["spec"], "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    print(f"Clusters found  : {spec['results']['n_clusters']}")
    print(f"HVGs selected   : {spec['results']['n_hvg']}")
    print(f"Marker rows     : {spec['results']['n_marker_rows']}")
    print(f"Filtered marker : {spec['results']['n_marker_rows_filtered']}")
    print("\nWrote:")
    for p in paths.values():
        print(f"  {p}")


if __name__ == "__main__":
    main()
