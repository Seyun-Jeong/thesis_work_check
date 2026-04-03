#!/usr/bin/env python3
"""
Harmonized rapids-singlecell GPU pipeline.

Design goal
-----------
This script mirrors the Scanpy CPU reference as closely as the GPU API honestly
allows, while keeping backend-specific semantics explicit.

Important choices
-----------------
- Same canonical filtered input as the CPU reference
- Same nominal normalization / HVG / PCA / kNN / resolution / seed
- Exact GPU KNN via `algorithm='brute'`
- Leiden iteration count is *not* forced unless explicitly configured
- Native GPU DE is run here, but standardized DE is recomputed later in the
  validation script on a common canonical matrix to isolate cluster-label drift
  from DE-engine drift
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from typing import Any

import pandas as pd
import scanpy as sc

import rapids_singlecell as rsc
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import scipy.sparse as sp

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["pbmc3k", "lung65k"])
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "benchmark_config.json"),
    )
    parser.add_argument(
        "--rmm-mode",
        choices=["none", "pool", "managed"],
        default="none",
        help="Optional RAPIDS memory mode. Use managed for memory-constrained GPUs.",
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
        "h5ad": f"{base}/{prefix}_rsc_gpu_harmonized.h5ad",
        "clusters": f"{base}/{prefix}_rsc_gpu_harmonized_clusters.csv",
        "markers": f"{base}/{prefix}_rsc_gpu_harmonized_markers.csv",
        "markers_filtered": f"{base}/{prefix}_rsc_gpu_harmonized_markers_filtered.csv",
        "umap": f"{base}/{prefix}_rsc_gpu_harmonized_umap.csv",
        "hvg": f"{base}/{prefix}_rsc_gpu_harmonized_hvg.csv",
        "spec": f"{base}/{prefix}_rsc_gpu_harmonized_spec.json",
    }


def maybe_init_rmm(mode: str) -> dict[str, Any]:
    details = {"requested": mode, "active": False, "error": None}
    if mode == "none":
        return details
    try:
        import cupy as cp
        import rmm

        rmm.reinitialize(
            pool_allocator=True,
            managed_memory=(mode == "managed"),
        )
        cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
        details["active"] = True
    except Exception as e:  # pragma: no cover - depends on GPU runtime
        details["error"] = repr(e)
    return details


def move_anndata_to_gpu(adata):
    if hasattr(rsc, "get") and hasattr(rsc.get, "anndata_to_GPU"):
        rsc.get.anndata_to_GPU(adata)
        return
    raise RuntimeError("rapids_singlecell.get.anndata_to_GPU is unavailable in this environment.")


def move_anndata_to_cpu(adata):
    if hasattr(rsc, "get") and hasattr(rsc.get, "anndata_to_CPU"):
        rsc.get.anndata_to_CPU(adata)
        return
    raise RuntimeError("rapids_singlecell.get.anndata_to_CPU is unavailable in this environment.")


def main() -> None:
    args = parse_args()
    gcfg, dcfg = load_config(args.config, args.dataset)
    paths = output_paths(dcfg["pipeline_prefix"])
    for p in paths.values():
        ensure_parent(p)

    rmm_info = maybe_init_rmm(args.rmm_mode)

    adata = sc.read_h5ad(dcfg["canonical_h5ad"])
    if "counts" not in adata.layers:
        raise ValueError(
            f"Canonical input {dcfg['canonical_h5ad']} is missing .layers['counts']. "
            "Re-run prepare_canonical_inputs.py first."
        )

    print("=" * 72)
    print(f"rapids-singlecell GPU harmonized run — {dcfg['name']}")
    print("=" * 72)
    print(f"Input           : {dcfg['canonical_h5ad']}")
    print(f"Cells × genes   : {adata.n_obs:,} × {adata.n_vars:,}")
    print(f"RMM mode        : {args.rmm_mode}")
    print()

    move_anndata_to_gpu(adata)



    # Force counts layer onto GPU if your rsc version leaves it on CPU
    counts = adata.layers["counts"]

    if sp.issparse(counts):
        counts = counts.tocsr()
        counts = cpx_sparse.csr_matrix(
            (cp.asarray(counts.data),
            cp.asarray(counts.indices),
            cp.asarray(counts.indptr)),
            shape=counts.shape,
        )   
    else:
        counts = cp.asarray(counts)

    adata.layers["counts"] = counts

    rsc.pp.normalize_total(adata, target_sum=gcfg["target_sum"])
    rsc.pp.log1p(adata)
    
    rsc.pp.highly_variable_genes(
        adata,
        layer="counts",
        n_top_genes=dcfg["n_top_genes"],
        flavor="seurat_v3",
    )
    rsc.pp.pca(
        adata,
        n_comps=gcfg["pca_n_comps"],
        mask_var="highly_variable",
    )
    rsc.pp.neighbors(
        adata,
        n_neighbors=dcfg["n_neighbors"],
        n_pcs=dcfg["neighbors_n_pcs"],
        use_rep="X_pca",
        algorithm="brute",
        metric=gcfg["neighbor_metric"],
        method=gcfg["neighbor_method"],
        random_state=gcfg["random_state"],
    )

    leiden_kwargs = {
        "resolution": dcfg["leiden_resolution"],
        "random_state": gcfg["random_state"],
        "key_added": "leiden",
    }
    if gcfg.get("rsc_leiden_n_iterations") is not None:
        leiden_kwargs["n_iterations"] = gcfg["rsc_leiden_n_iterations"]
    rsc.tl.leiden(adata, **leiden_kwargs)

    rsc.tl.umap(
        adata,
        min_dist=gcfg["umap_min_dist"],
        spread=gcfg["umap_spread"],
        init_pos="spectral",
        random_state=gcfg["random_state"],
    )

    # Be defensive across versions: only pass supported kwargs.
    rank_sig = inspect.signature(rsc.tl.rank_genes_groups)
    rank_kwargs = {
        "groupby": "leiden",
        "method": gcfg["de_method"],
        "corr_method": gcfg["de_corr_method"],
        "use_raw": False,
        "pts": True,
    }
    optional_rank_kwargs = {
        "pre_load": True,
        "tie_correct": False,
        "use_continuity": False,
    }
    for key, value in optional_rank_kwargs.items():
        if key in rank_sig.parameters:
            rank_kwargs[key] = value
    rsc.tl.rank_genes_groups(adata, **rank_kwargs)

    # Move back to CPU before serialization.
    move_anndata_to_cpu(adata)

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
        "pipeline": "rsc_gpu_harmonized",
        "dataset": dcfg["name"],
        "rapids_singlecell_version": getattr(rsc, "__version__", "UNKNOWN"),
        "input": dcfg["canonical_h5ad"],
        "rmm": rmm_info,
        "parameters": {
            "target_sum": gcfg["target_sum"],
            "hvg_flavor": "seurat_v3",
            "hvg_layer": "counts",
            "n_top_genes": dcfg["n_top_genes"],
            "pca_n_comps": gcfg["pca_n_comps"],
            "n_neighbors": dcfg["n_neighbors"],
            "neighbors_n_pcs": dcfg["neighbors_n_pcs"],
            "neighbor_metric": gcfg["neighbor_metric"],
            "neighbor_method": gcfg["neighbor_method"],
            "neighbor_backend": "rapids_singlecell.pp.neighbors(algorithm='brute')",
            "leiden_n_iterations": gcfg.get("rsc_leiden_n_iterations"),
            "leiden_resolution": dcfg["leiden_resolution"],
            "umap_min_dist": gcfg["umap_min_dist"],
            "umap_spread": gcfg["umap_spread"],
            "umap_init_pos": "spectral",
            "random_state": gcfg["random_state"],
            "de_method": gcfg["de_method"],
            "de_corr_method": gcfg["de_corr_method"],
            "de_extra_kwargs_used": {k: v for k, v in rank_kwargs.items() if k not in {"groupby", "method", "corr_method", "use_raw", "pts"}},
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
