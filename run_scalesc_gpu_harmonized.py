#!/usr/bin/env python3
"""
Guarded ScaleSC benchmark runner.

Read this before using the script
--------------------------------
ScaleSC is *not* treated here as a clean like-for-like native DEG comparator.
This script is for the clustering / dimensionality-reduction arm of the study.

Why guarded?
------------
The current public README documents:
- a >24 GB VRAM requirement,
- a `data_dir` folder-style input,
- neighbors defaulting to `algorithm='cagra'`,
- Leiden and UMAP wrappers around rapids-singlecell,
- NSForest-oriented marker identification in the project description.

Therefore this script makes only claims that are documented by the public API.
It does **not** assume undocumented direct compatibility with arbitrary h5ad
inputs and it does **not** pretend ScaleSC exposes a native Wilcoxon-DE API that
is directly comparable to Scanpy / rapids-singlecell / Seurat.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["pbmc3k", "lung65k"])
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "benchmark_config.json"),
    )
    parser.add_argument(
        "--scalesc-data-dir",
        required=True,
        help="Path to a ScaleSC-compatible input folder prepared outside this script.",
    )
    parser.add_argument(
        "--neighbors-algorithm",
        default="cagra",
        help="ScaleSC neighbors() algorithm. Default matches the public README.",
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
        "clusters": f"{base}/{prefix}_scalesc_gpu_harmonized_clusters.csv",
        "umap": f"{base}/{prefix}_scalesc_gpu_harmonized_umap.csv",
        "spec": f"{base}/{prefix}_scalesc_gpu_harmonized_spec.json",
        "saved_h5ad": f"{base}/{prefix}_scalesc_gpu_harmonized.h5ad",
    }


def query_gpu_vram_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
        first = out.strip().splitlines()[0].strip()
        return int(float(first))
    except Exception:
        return None


def load_allowlist(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def main() -> None:
    args = parse_args()
    gcfg, dcfg = load_config(args.config, args.dataset)
    paths = output_paths(dcfg["pipeline_prefix"])
    for p in paths.values():
        ensure_parent(p)

    vram_mib = query_gpu_vram_mib()
    if vram_mib is not None and vram_mib < 24 * 1024:
        raise RuntimeError(
            f"Detected {vram_mib} MiB VRAM. Public ScaleSC docs require >24 GB VRAM. "
            "Run this on Libra L40S / A100 / H100 instead."
        )

    try:
        import scalesc
        from scalesc import ScaleSC
    except Exception as e:
        raise RuntimeError(
            "ScaleSC is not importable in the current environment. Install it in a dedicated "
            "ScaleSC environment on Libra, then rerun."
        ) from e

    print("=" * 72)
    print(f"ScaleSC guarded run — {dcfg['name']}")
    print("=" * 72)
    print(f"ScaleSC input dir : {args.scalesc_data_dir}")
    print(f"Neighbors alg     : {args.neighbors_algorithm}")
    print(f"Target output     : {paths['saved_h5ad']}")
    print()

    ssc = ScaleSC(
        data_dir=args.scalesc_data_dir,
        preload_on_cpu=True,
        preload_on_gpu=True,
        output_dir="write",
    )

    # Optional sanity check against canonical allowlists.
    allow_cells = load_allowlist(dcfg["canonical_cells_txt"])
    allow_genes = load_allowlist(dcfg["canonical_genes_txt"])
    adata_meta = ssc.adata
    cell_overlap = len(set(map(str, adata_meta.obs_names)) & allow_cells)
    gene_overlap = len(set(map(str, adata_meta.var_names)) & allow_genes)

    ssc.normalize_log1p(target_sum=gcfg["target_sum"])
    ssc.highly_variable_genes(n_top_genes=dcfg["n_top_genes"], method="seurat_v3")
    ssc.pca(n_components=gcfg["pca_n_comps"], hvg_var="highly_variable")
    ssc.neighbors(
        n_neighbors=dcfg["n_neighbors"],
        n_pcs=dcfg["neighbors_n_pcs"],
        use_rep="X_pca",
        algorithm=args.neighbors_algorithm,
    )
    ssc.leiden(
        resolution=dcfg["leiden_resolution"],
        random_state=gcfg["random_state"],
    )
    ssc.umap(random_state=gcfg["random_state"])
    ssc.save(data_name=f"{dcfg['pipeline_prefix']}_scalesc_gpu_harmonized")

    adata = ssc.adata
    clusters_df = pd.DataFrame({
        "barcode": adata.obs_names.astype(str),
        "leiden": adata.obs["leiden"].astype(str).values,
    })
    clusters_df.to_csv(paths["clusters"], index=False)

    if "X_umap" in adata.obsm:
        umap_df = pd.DataFrame(
            adata.obsm["X_umap"],
            index=adata.obs_names.astype(str),
            columns=["UMAP_1", "UMAP_2"],
        ).reset_index(names="barcode")
        umap_df.to_csv(paths["umap"], index=False)

    spec = {
        "pipeline": "scalesc_gpu_harmonized_guarded",
        "dataset": dcfg["name"],
        "scalesc_version": getattr(scalesc, "__version__", "UNKNOWN"),
        "scale_sc_input_dir": args.scalesc_data_dir,
        "public_api_constraints": {
            "expects_data_dir_folder": True,
            "documented_neighbors_default": "cagra",
            "native_wilcoxon_de_exposed": False,
            "validation_de_strategy": "standardized external DE in validate_cross_pipeline_harmonized.py",
        },
        "parameters": {
            "target_sum": gcfg["target_sum"],
            "n_top_genes": dcfg["n_top_genes"],
            "pca_n_comps": gcfg["pca_n_comps"],
            "n_neighbors": dcfg["n_neighbors"],
            "neighbors_n_pcs": dcfg["neighbors_n_pcs"],
            "neighbors_algorithm": args.neighbors_algorithm,
            "leiden_resolution": dcfg["leiden_resolution"],
            "random_state": gcfg["random_state"],
        },
        "canonical_overlap_sanity_check": {
            "cell_overlap_count": cell_overlap,
            "expected_cells": len(allow_cells),
            "gene_overlap_count": gene_overlap,
            "expected_genes": len(allow_genes),
        },
        "results": {
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_clusters": int(adata.obs["leiden"].nunique()),
        },
    }
    with open(paths["spec"], "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    print(f"Clusters found  : {spec['results']['n_clusters']}")
    print(f"Cell overlap    : {cell_overlap} / {len(allow_cells)} canonical cells")
    print(f"Gene overlap    : {gene_overlap} / {len(allow_genes)} canonical genes")
    print("\nWrote:")
    print(f"  {paths['clusters']}")
    if os.path.exists(paths["umap"]):
        print(f"  {paths['umap']}")
    print(f"  {paths['spec']}")
    print(f"  {paths['saved_h5ad']}")


if __name__ == "__main__":
    main()
