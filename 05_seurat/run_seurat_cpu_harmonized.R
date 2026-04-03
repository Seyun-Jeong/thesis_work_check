#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(jsonlite)
})

# -----------------------------------------------------------------------------
# Harmonized Seurat CPU pipeline
# -----------------------------------------------------------------------------
# Goals:
#   - Minimize avoidable divergence from Scanpy / rapids-singlecell
#   - Preserve canonical cells/genes exactly
#   - Use exact CPU nearest neighbors via RANN (nn.eps = 0)
#   - Cluster on the NN graph when requested
#   - Keep Seurat-only requirements explicit (e.g. ScaleData before RunPCA)
#
# Important notes:
#   - Seurat still requires scaled features for RunPCA
#   - Seurat Leiden semantics differ from Scanpy / cuGraph
#   - Native Seurat DE can be slow on large datasets; this script writes core
#     outputs BEFORE DE so long runs do not lose clustering/UMAP results
# -----------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

get_script_dir <- function() {
  x <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", x, value = TRUE)
  if (length(file_arg) == 0) return(getwd())
  dirname(normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = FALSE))
}

get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (!is.na(i) && i < length(args)) return(args[[i + 1]])
  default
}

`%||%` <- function(x, y) if (is.null(x)) y else x

dataset_key <- get_arg("--dataset")
if (is.null(dataset_key)) {
  stop("Usage: Rscript run_seurat_cpu_harmonized.R --dataset <pbmc3k|lung65k> [--config path]")
}

config_path <- get_arg("--config", file.path(get_script_dir(), "benchmark_config.json"))

cfg <- fromJSON(config_path, simplifyVector = TRUE)
if (!(dataset_key %in% names(cfg$datasets))) {
  stop("Unknown dataset: ", dataset_key)
}

gcfg <- cfg$global
dcfg <- cfg$datasets[[dataset_key]]
presto_installed <- requireNamespace("presto", quietly = TRUE)

out_prefix <- file.path("write", paste0(dcfg$pipeline_prefix, "_seurat_cpu_harmonized"))
dir.create("write", showWarnings = FALSE, recursive = TRUE)

out_paths <- list(
  rds = paste0(out_prefix, ".rds"),
  clusters = paste0(out_prefix, "_clusters.csv"),
  markers = paste0(out_prefix, "_markers.csv"),
  markers_filtered = paste0(out_prefix, "_markers_filtered.csv"),
  umap = paste0(out_prefix, "_umap.csv"),
  hvg = paste0(out_prefix, "_hvg.csv"),
  spec = paste0(out_prefix, "_spec.json")
)

read_allowlist <- function(path) {
  x <- readLines(path, warn = FALSE)
  x[nzchar(x)]
}

empty_marker_df <- function() {
  data.frame(
    p_val = numeric(),
    avg_log2FC = numeric(),
    pct.1 = numeric(),
    pct.2 = numeric(),
    p_val_adj = numeric(),
    cluster = character(),
    gene = character(),
    stringsAsFactors = FALSE
  )
}

load_h5ad_counts <- function(h5ad_path) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("reticulate is required to read h5ad input in this script.")
  }

  reticulate::use_python(Sys.which("python"), required = FALSE)
  anndata <- reticulate::import("anndata", convert = FALSE)
  ad <- anndata$read_h5ad(h5ad_path)

  # Prefer counts layer; fall back to X
  x <- tryCatch(ad$layers$get("counts"), error = function(e) NULL)
  if (is.null(x) || inherits(x, "python.builtin.NoneType")) {
    x <- ad$X
  }

  # Convert matrix robustly
  x_r <- tryCatch(reticulate::py_to_r(x), error = function(e) x)

  if (inherits(x_r, "sparseMatrix")) {
    # Already converted to an R sparse matrix: cells x genes -> genes x cells
    counts <- Matrix::t(x_r)
    if (!inherits(counts, "CsparseMatrix")) {
      counts <- as(counts, "CsparseMatrix")
    }
    counts <- as(counts, "dgCMatrix")
  } else if (is.matrix(x_r)) {
    # Dense R matrix fallback
    counts <- as(Matrix::Matrix(t(x_r), sparse = TRUE), "dgCMatrix")
  } else {
    # Still a Python sparse matrix object
    x_csc <- x$transpose()$tocsc()
    p <- as.integer(reticulate::py_to_r(x_csc$indptr))
    i <- as.integer(reticulate::py_to_r(x_csc$indices))
    vals <- as.numeric(reticulate::py_to_r(x_csc$data))
    dims <- as.integer(reticulate::py_to_r(x_csc$shape))
    counts <- new("dgCMatrix", p = p, i = i, x = vals, Dim = dims)
  }

  gene_names <- tryCatch(
    as.character(reticulate::py_to_r(ad$var_names$tolist())),
    error = function(e) as.character(reticulate::py_to_r(ad$var_names$to_list()))
  )

  cell_names <- tryCatch(
    as.character(reticulate::py_to_r(ad$obs_names$tolist())),
    error = function(e) as.character(reticulate::py_to_r(ad$obs_names$to_list()))
  )

  rownames(counts) <- make.unique(gene_names)
  colnames(counts) <- cell_names
  counts
}

load_counts <- function(ds) {
  if (ds$input_type == "10x_mtx") {
    counts <- Read10X(data.dir = ds$input_path)
    rownames(counts) <- make.unique(rownames(counts))
    return(counts)
  }
  if (ds$input_type == "h5ad") {
    return(load_h5ad_counts(ds$input_path))
  }
  stop("Unsupported input_type: ", ds$input_type)
}

subset_to_canonical <- function(counts, allow_genes, allow_cells) {
  common_genes <- intersect(allow_genes, rownames(counts))
  common_cells <- intersect(allow_cells, colnames(counts))

  if (length(common_genes) == 0) stop("No overlapping genes between raw input and canonical gene allowlist.")
  if (length(common_cells) == 0) stop("No overlapping cells between raw input and canonical cell allowlist.")

  counts[common_genes, common_cells, drop = FALSE]
}

cat(strrep("=", 72), "\n", sep = "")
cat("Seurat CPU harmonized run — ", dcfg$name, "\n", sep = "")
cat(strrep("=", 72), "\n", sep = "")
cat("Dataset key      : ", dataset_key, "\n", sep = "")
cat("Raw input        : ", dcfg$input_path, "\n", sep = "")
cat("Canonical cells  : ", dcfg$canonical_cells_txt, "\n", sep = "")
cat("Canonical genes  : ", dcfg$canonical_genes_txt, "\n\n", sep = "")

allow_cells <- read_allowlist(dcfg$canonical_cells_txt)
allow_genes <- read_allowlist(dcfg$canonical_genes_txt)

counts <- load_counts(dcfg)
counts <- subset_to_canonical(counts, allow_genes, allow_cells)

cat("Counts after canonical subset: ", ncol(counts), " cells × ", nrow(counts), " genes\n\n", sep = "")

obj <- CreateSeuratObject(
  counts = counts,
  project = dcfg$name,
  min.cells = 0,
  min.features = 0
)

obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")
obj[["percent.ribo"]] <- PercentageFeatureSet(obj, pattern = "^RP[SL]")
obj[["percent.hb"]] <- PercentageFeatureSet(obj, pattern = "^HB[^(P)]")

# -----------------------------------------------------------------------------
# Normalize
# -----------------------------------------------------------------------------
obj <- NormalizeData(
  obj,
  normalization.method = "LogNormalize",
  scale.factor = gcfg$target_sum,
  verbose = FALSE
)

# -----------------------------------------------------------------------------
# HVG
# -----------------------------------------------------------------------------
obj <- FindVariableFeatures(
  obj,
  selection.method = "vst",
  nfeatures = dcfg$n_top_genes,
  verbose = FALSE
)

# -----------------------------------------------------------------------------
# Scale (required by Seurat for PCA features)
# -----------------------------------------------------------------------------
obj <- ScaleData(
  obj,
  features = VariableFeatures(obj),
  verbose = FALSE
)

# -----------------------------------------------------------------------------
# PCA
# -----------------------------------------------------------------------------
obj <- RunPCA(
  obj,
  features = VariableFeatures(obj),
  npcs = gcfg$pca_n_comps,
  seed.use = gcfg$random_state,
  verbose = FALSE
)

# -----------------------------------------------------------------------------
# Neighbors
# -----------------------------------------------------------------------------
graph_mode <- gcfg$seurat_graph_mode
if (isTRUE(graph_mode == "nn")) {
  obj <- FindNeighbors(
    obj,
    reduction = "pca",
    dims = seq_len(dcfg$neighbors_n_pcs),
    k.param = dcfg$n_neighbors,
    return.neighbor = FALSE,
    compute.SNN = FALSE,
    nn.method = gcfg$seurat_nn_method,
    nn.eps = gcfg$seurat_nn_eps,
    graph.name = "nn",
    verbose = FALSE
  )
  graph_for_clustering <- "nn"
} else {
  obj <- FindNeighbors(
    obj,
    reduction = "pca",
    dims = seq_len(dcfg$neighbors_n_pcs),
    k.param = dcfg$n_neighbors,
    return.neighbor = FALSE,
    compute.SNN = TRUE,
    prune.SNN = gcfg$seurat_prune_snn,
    nn.method = gcfg$seurat_nn_method,
    nn.eps = gcfg$seurat_nn_eps,
    graph.name = c("RNA_nn", "RNA_snn"),
    verbose = FALSE
  )
  graph_for_clustering <- "RNA_snn"
}

# -----------------------------------------------------------------------------
# Leiden
# -----------------------------------------------------------------------------
obj <- FindClusters(
  obj,
  graph.name = graph_for_clustering,
  resolution = dcfg$leiden_resolution,
  algorithm = 4,
  leiden_method = gcfg$seurat_leiden_method,
  n.start = gcfg$seurat_leiden_n_start,
  n.iter = gcfg$seurat_leiden_n_iter,
  random.seed = gcfg$random_state,
  verbose = FALSE
)

# -----------------------------------------------------------------------------
# UMAP
# -----------------------------------------------------------------------------
obj <- RunUMAP(
  obj,
  reduction = "pca",
  dims = seq_len(dcfg$neighbors_n_pcs),
  umap.method = "uwot",
  n.neighbors = dcfg$n_neighbors,
  metric = gcfg$neighbor_metric,
  min.dist = gcfg$umap_min_dist,
  spread = gcfg$umap_spread,
  seed.use = gcfg$random_state,
  verbose = FALSE
)

# -----------------------------------------------------------------------------
# Save core artifacts BEFORE DE
# -----------------------------------------------------------------------------
saveRDS(obj, out_paths$rds)

write.csv(
  data.frame(
    barcode = colnames(obj),
    leiden = as.character(Idents(obj)),
    stringsAsFactors = FALSE
  ),
  out_paths$clusters,
  row.names = FALSE
)

write.csv(
  data.frame(
    barcode = rownames(Embeddings(obj, "umap")),
    Embeddings(obj, "umap"),
    stringsAsFactors = FALSE
  ),
  out_paths$umap,
  row.names = FALSE
)

write.csv(
  data.frame(gene = VariableFeatures(obj), stringsAsFactors = FALSE),
  out_paths$hvg,
  row.names = FALSE
)

spec_core <- list(
  pipeline = "seurat_cpu_harmonized",
  dataset = dcfg$name,
  seurat_version = as.character(packageVersion("Seurat")),
  seurat_object_version = as.character(packageVersion("SeuratObject")),
  matrix_version = as.character(packageVersion("Matrix")),
  input = dcfg$input_path,
  canonical_cells = dcfg$canonical_cells_txt,
  canonical_genes = dcfg$canonical_genes_txt,
  parameters = list(
    target_sum = gcfg$target_sum,
    hvg_method = "vst",
    n_top_genes = dcfg$n_top_genes,
    scale_data = TRUE,
    scale_features = "VariableFeatures(obj)",
    pca_n_comps = gcfg$pca_n_comps,
    n_neighbors = dcfg$n_neighbors,
    neighbors_n_pcs = dcfg$neighbors_n_pcs,
    neighbor_metric = gcfg$neighbor_metric,
    nn_method = gcfg$seurat_nn_method,
    nn_eps = gcfg$seurat_nn_eps,
    graph_mode = graph_mode,
    clustering_graph = graph_for_clustering,
    clustering_algorithm = "Leiden",
    leiden_method = gcfg$seurat_leiden_method,
    leiden_n_start = gcfg$seurat_leiden_n_start,
    leiden_n_iter = gcfg$seurat_leiden_n_iter,
    leiden_resolution = dcfg$leiden_resolution,
    umap_method = "uwot",
    umap_n_neighbors = dcfg$n_neighbors,
    umap_metric = gcfg$neighbor_metric,
    umap_min_dist = gcfg$umap_min_dist,
    umap_spread = gcfg$umap_spread,
    random_state = gcfg$random_state,
    de_test = "wilcox",
    de_min_pct = 0,
    de_logfc_threshold = 0,
    presto_installed = presto_installed
  ),
  results = list(
    n_cells = ncol(obj),
    n_genes = nrow(obj),
    n_hvg = length(VariableFeatures(obj)),
    n_clusters = length(levels(Idents(obj))),
    n_marker_rows = NA,
    n_marker_rows_filtered = NA
  )
)

writeLines(toJSON(spec_core, pretty = TRUE, auto_unbox = TRUE), out_paths$spec)

cat("Core outputs written before DE.\n")
flush.console()
cat("Starting FindAllMarkers...\n")
flush.console()

# -----------------------------------------------------------------------------
# Differential expression
# -----------------------------------------------------------------------------
if (requireNamespace("future", quietly = TRUE)) {
  future::plan("sequential")
}
options(future.globals.maxSize = 8 * 1024^3)

markers <- tryCatch(
  FindAllMarkers(
    obj,
    only.pos = FALSE,
    test.use = "wilcox",
    min.pct = 0,
    logfc.threshold = 0,
    return.thresh = 1,
    verbose = FALSE
  ),
  error = function(e) {
    warning("FindAllMarkers failed: ", conditionMessage(e))
    NULL
  }
)

if (is.null(markers) || nrow(markers) == 0) {
  warning("FindAllMarkers returned no rows; writing empty marker tables.")
  markers <- empty_marker_df()
  markers_filtered <- empty_marker_df()
} else {
  required_cols <- c("cluster", "gene", "avg_log2FC", "p_val", "p_val_adj")
  missing_cols <- setdiff(required_cols, colnames(markers))

  if (length(missing_cols) > 0) {
    warning(
      sprintf(
        "Marker table missing columns: %s. Writing unfiltered markers and empty filtered markers.",
        paste(missing_cols, collapse = ", ")
      )
    )
    markers_filtered <- empty_marker_df()
  } else {
    markers_filtered <- subset(markers, p_val_adj < 0.05 & avg_log2FC > 0.1)
  }
}

cat("Finished FindAllMarkers.\n")
flush.console()

write.csv(markers, out_paths$markers, row.names = FALSE)
write.csv(markers_filtered, out_paths$markers_filtered, row.names = FALSE)

spec_final <- spec_core
spec_final$results$n_marker_rows <- nrow(markers)
spec_final$results$n_marker_rows_filtered <- nrow(markers_filtered)

writeLines(toJSON(spec_final, pretty = TRUE, auto_unbox = TRUE), out_paths$spec)

cat("Clusters found  : ", spec_final$results$n_clusters, "\n", sep = "")
cat("HVGs selected   : ", spec_final$results$n_hvg, "\n", sep = "")
cat("Marker rows     : ", spec_final$results$n_marker_rows, "\n", sep = "")
cat("Filtered marker : ", spec_final$results$n_marker_rows_filtered, "\n\n", sep = "")
cat("Wrote:\n")
for (p in out_paths) cat("  ", p, "\n", sep = "")