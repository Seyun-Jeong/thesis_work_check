#!/usr/bin/env python3
"""
Prepare a canonical filtered input for all downstream pipelines.

Why this exists
---------------
The point of this script is to make sure Scanpy, rapids-singlecell, Seurat,
and optionally ScaleSC all start from the *same* cells and *same* genes.

This prevents hidden divergence caused by:
- loader-specific filtering,
- CreateSeuratObject(min.cells/min.features) side effects,
- barcode formatting drift,
- small QC-order differences.

Outputs
-------
For each dataset it writes:
- canonical filtered h5ad
- cell allowlist (.txt)
- gene allowlist (.txt)
- QC summary (.json)

Notes
-----
Filtering order is intentionally explicit and conservative:
1. filter genes by min_cells
2. filter cells by min_genes / max_genes / optional mt%
3. do NOT re-filter genes a second time unless you intentionally change this

That order matches many common scRNA-seq workflows and is easier to reproduce
across libraries than mixed implicit+explicit filtering.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


@dataclass
class DatasetConfig:
    name: str
    input_type: str
    input_path: str
    pipeline_prefix: str
    canonical_h5ad: str
    canonical_cells_txt: str
    canonical_genes_txt: str
    canonical_summary_json: str
    min_cells: int
    min_genes: int
    max_genes: int
    max_pct_mt: float | None
    n_top_genes: int
    neighbors_n_pcs: int
    n_neighbors: int
    leiden_resolution: float
    known_marker_sets: dict[str, list[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["pbmc3k", "lung65k"])
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "benchmark_config.json"),
        help="Path to benchmark_config.json",
    )
    return parser.parse_args()


def load_config(path: str, dataset_key: str) -> tuple[dict[str, Any], DatasetConfig]:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    ds = DatasetConfig(**config["datasets"][dataset_key])
    return config["global"], ds


RIBO_RE = re.compile(r"^RP[SL]", flags=re.IGNORECASE)
HB_RE = re.compile(r"^HB(?!P)", flags=re.IGNORECASE)


def make_var_names_unique(adata):
    adata.var_names = pd.Index(adata.var_names.astype(str))
    adata.var_names_make_unique()
    return adata


def load_input(ds: DatasetConfig):
    if ds.input_type == "10x_mtx":
        adata = sc.read_10x_mtx(ds.input_path, var_names="gene_symbols", cache=False)
    elif ds.input_type == "h5ad":
        adata = sc.read_h5ad(ds.input_path)
    else:
        raise ValueError(f"Unsupported input_type: {ds.input_type}")

    adata = make_var_names_unique(adata)

    # Make sure counts are preserved.
    if sparse.issparse(adata.X):
        adata.X = adata.X.tocsr()
        adata.layers["counts"] = adata.X.copy()
    else:
        adata.X = np.asarray(adata.X)
        adata.layers["counts"] = adata.X.copy()

    return adata


def annotate_qc_vars(adata):
    var_upper = pd.Index(adata.var_names.astype(str)).str.upper()
    adata.var["mt"] = var_upper.str.startswith("MT-")
    adata.var["ribo"] = [bool(RIBO_RE.match(g)) for g in var_upper]
    adata.var["hb"] = [bool(HB_RE.match(g)) for g in var_upper]

    qc_vars = [col for col in ["mt", "ribo", "hb"] if col in adata.var.columns]
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=qc_vars,
        inplace=True,
        log1p=False,
        percent_top=None,
    )
    return adata


def filter_adata(adata, ds: DatasetConfig):
    before_cells, before_genes = adata.n_obs, adata.n_vars

    # 1) Gene filter
    sc.pp.filter_genes(adata, min_cells=ds.min_cells)
    after_gene_filter_cells, after_gene_filter_genes = adata.n_obs, adata.n_vars

    # 2) Cell filter
    cell_mask = (adata.obs["n_genes_by_counts"] >= ds.min_genes) & (
        adata.obs["n_genes_by_counts"] <= ds.max_genes
    )
    if ds.max_pct_mt is not None and "pct_counts_mt" in adata.obs.columns:
        cell_mask &= adata.obs["pct_counts_mt"] < ds.max_pct_mt
    adata = adata[cell_mask].copy()

    summary = {
        "dataset": ds.name,
        "input_path": ds.input_path,
        "before": {
            "n_cells": int(before_cells),
            "n_genes": int(before_genes),
        },
        "after_gene_filter": {
            "n_cells": int(after_gene_filter_cells),
            "n_genes": int(after_gene_filter_genes),
        },
        "after_cell_filter": {
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
        },
        "filters": {
            "min_cells": int(ds.min_cells),
            "min_genes": int(ds.min_genes),
            "max_genes": int(ds.max_genes),
            "max_pct_mt": None if ds.max_pct_mt is None else float(ds.max_pct_mt),
        },
        "qc_columns_present": [col for col in adata.obs.columns if col.startswith(("n_", "pct_"))],
    }
    return adata, summary


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def write_outputs(adata, summary: dict[str, Any], ds: DatasetConfig) -> None:
    for path in [
        ds.canonical_h5ad,
        ds.canonical_cells_txt,
        ds.canonical_genes_txt,
        ds.canonical_summary_json,
    ]:
        ensure_parent(path)

    adata.write_h5ad(ds.canonical_h5ad, compression="gzip")

    with open(ds.canonical_cells_txt, "w", encoding="utf-8") as f:
        for bc in adata.obs_names.astype(str):
            f.write(f"{bc}\n")

    with open(ds.canonical_genes_txt, "w", encoding="utf-8") as f:
        for gene in adata.var_names.astype(str):
            f.write(f"{gene}\n")

    with open(ds.canonical_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    global_cfg, ds = load_config(args.config, args.dataset)

    print("=" * 72)
    print(f"Preparing canonical filtered input: {ds.name}")
    print("=" * 72)
    print(f"Input path        : {ds.input_path}")
    print(f"Canonical h5ad    : {ds.canonical_h5ad}")
    print(f"Cell allowlist    : {ds.canonical_cells_txt}")
    print(f"Gene allowlist    : {ds.canonical_genes_txt}")
    print()

    adata = load_input(ds)
    print(f"Loaded raw matrix : {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    adata = annotate_qc_vars(adata)
    adata, summary = filter_adata(adata, ds)

    print(f"Filtered matrix   : {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    if ds.max_pct_mt is None:
        print("mt% filter        : not applied")
    else:
        print(f"mt% filter        : pct_counts_mt < {ds.max_pct_mt}")

    # Store the global benchmark defaults in the summary for provenance.
    summary["global_defaults"] = global_cfg

    write_outputs(adata, summary, ds)

    print("\nWrote:")
    print(f"  {ds.canonical_h5ad}")
    print(f"  {ds.canonical_cells_txt}")
    print(f"  {ds.canonical_genes_txt}")
    print(f"  {ds.canonical_summary_json}")


if __name__ == "__main__":
    main()
