# Harmonized four-library scRNA-seq benchmark

This bundle is designed for a paper where the goal is **not** to pretend that Scanpy, rapids-singlecell, Seurat, and ScaleSC are mathematically identical. The goal is to make them **as aligned as the official APIs honestly allow**, while documenting the remaining irreducible differences.

## Core principle

Split the study into two layers:

1. **Harmonized preprocessing + clustering benchmark**
   - Same filtered cells
   - Same filtered genes
   - Same normalization target sum
   - Same nominal HVG count
   - Same nominal PCA dimensionality
   - Same nominal neighbor count and PCs used for graph construction
   - Same clustering resolution
   - Same random seed where supported

2. **Validation benchmark**
   - Global cluster agreement: ARI, NMI
   - Matched-cluster overlap: Dice
   - DEG agreement: Jaccard + Spearman
   - Biology preservation: module-score profile concordance
   - Native DE and standardized DE reported separately

## Why the canonical input step matters

The single biggest avoidable source of drift is hidden filtering.

Examples:
- `CreateSeuratObject(min.cells=..., min.features=...)` silently filters before the user’s explicit QC section.
- Different loaders can mangle barcodes or gene names.
- Different pipelines may recompute QC metrics in slightly different orders.

To prevent this, the benchmark starts with `prepare_canonical_inputs.py`, which creates:
- one **canonical filtered h5ad** per dataset,
- one **canonical cell allowlist**,
- one **canonical gene allowlist**,
- one **QC summary json**.

All downstream pipelines should use those exact cell and gene sets.

## Recommended paper framing

### Primary benchmark
Use three pipelines as your core, fully analyzable comparison:
- Scanpy CPU
- rapids-singlecell GPU
- Seurat CPU

### Secondary / scalability benchmark
Use ScaleSC as a separate clustering/scalability arm unless you can verify all of the following on Libra:
- enough VRAM,
- documented input format compatibility,
- successful clustering on both datasets,
- clear handling of DE (native or standardized-only).

### Do not overclaim
You should **not** say the four pipelines are identical.
You should say they are **functionally harmonized up to documented library constraints**.

## Remaining irreducible differences

These remain even after careful harmonization:

- **Seurat requires scaled data for PCA input features**.
- **Seurat default clustering workflow is SNN-based**, but this bundle gives you an NN-graph option to reduce divergence.
- **Leiden iteration semantics differ across backends**.
- **DE implementations differ across libraries**.
- **ScaleSC is not a clean native Wilcoxon-DE comparator**.

## What each script does

### `prepare_canonical_inputs.py`
Builds the shared filtered dataset used by all pipelines.

### `run_scanpy_cpu_harmonized.py`
CPU reference run from the canonical filtered input.

### `run_rsc_gpu_harmonized.py`
GPU run from the canonical filtered input.

### `run_seurat_cpu_harmonized.R`
Seurat CPU run using the canonical allowlists so that it starts from the exact same cells and genes.

### `run_scalesc_gpu_harmonized.py`
A guarded ScaleSC runner. It treats ScaleSC as clustering-first and expects a ScaleSC-compatible input folder. Standardized DE is handled in validation, not here.

### `validate_cross_pipeline_harmonized.py`
Robust cross-pipeline validation with:
- exact barcode normalization,
- no positional fallback,
- Hungarian matching for cluster alignment,
- both native-DE and standardized-DE comparison modes.

### `capture_benchmark_env.sh`
Hardware/software provenance capture for paper appendix and methods.

## Suggested run order

```bash
python prepare_canonical_inputs.py --dataset pbmc3k
python prepare_canonical_inputs.py --dataset lung65k

python run_scanpy_cpu_harmonized.py --dataset pbmc3k
python run_scanpy_cpu_harmonized.py --dataset lung65k

python run_rsc_gpu_harmonized.py --dataset pbmc3k
python run_rsc_gpu_harmonized.py --dataset lung65k

Rscript run_seurat_cpu_harmonized.R --dataset pbmc3k
Rscript run_seurat_cpu_harmonized.R --dataset lung65k

# ScaleSC only on Libra / L40S / H100 with a verified ScaleSC-compatible input folder
python run_scalesc_gpu_harmonized.py --dataset pbmc3k --scalesc-data-dir /path/to/scalesc_ready_pbmc3k
python run_scalesc_gpu_harmonized.py --dataset lung65k --scalesc-data-dir /path/to/scalesc_ready_lung65k

python validate_cross_pipeline_harmonized.py --dataset pbmc3k
python validate_cross_pipeline_harmonized.py --dataset lung65k
```

## Recommended tables for the paper

### Table 1. Harmonized parameter table
Report the nominally matched parameters and explicitly separate:
- truly matched parameters,
- backend-specific approximations,
- unavoidable library-specific differences.

### Table 2. Cluster concordance
- cluster count
- ARI
- NMI
- mean Dice

### Table 3. DEG concordance
Report this twice:
- **native DE**
- **standardized DE on the same canonical matrix**

This prevents “different DE engine” from being confused with “different clustering result”.

### Table 4. Biology preservation
- mean module-score profile correlation
- optional curated marker recovery table

## Hard rule for the write-up

Never write “everything else is identical parameters” unless you have verified the semantics are identical, not just the parameter names.
