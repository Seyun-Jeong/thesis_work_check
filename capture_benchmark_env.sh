#!/usr/bin/env bash
set -euo pipefail

OUT_JSON="${1:-benchmark_env_snapshot.json}"

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

json_escape() {
  python - <<'PY' "$1"
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

capture_python_versions() {
  python - <<'PY'
import json
mods = ["scanpy", "anndata", "rapids_singlecell", "cupy", "numpy", "scipy", "sklearn", "pandas", "scalesc"]
out = {}
for mod in mods:
    try:
        m = __import__(mod)
        out[mod] = getattr(m, "__version__", "UNKNOWN")
    except Exception as e:
        out[mod] = None
print(json.dumps(out))
PY
}

capture_r_versions() {
  if ! command -v R >/dev/null 2>&1; then
    echo "{}"
    return
  fi
  R --vanilla --slave <<'RS'
mods <- c("Seurat", "SeuratObject", "Matrix", "jsonlite", "SeuratDisk", "reticulate")
out <- list()
out[["R"]] <- paste(R.version$major, R.version$minor, sep = ".")
for (m in mods) {
  out[[m]] <- if (requireNamespace(m, quietly = TRUE)) as.character(packageVersion(m)) else NULL
}
cat(jsonlite::toJSON(out, auto_unbox = TRUE))
RS
}

OS_NAME="$(uname -s || true)"
HOSTNAME="$(hostname || true)"
ARCH="$(uname -m || true)"
KERNEL="$(uname -r || true)"
DATE_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ || true)"

if [[ "$OS_NAME" == "Darwin" ]]; then
  OS_PRETTY="$(sw_vers -productName 2>/dev/null || echo Darwin) $(sw_vers -productVersion 2>/dev/null || true)"
  CPU_MODEL="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || sysctl -n hw.model 2>/dev/null || true)"
  CPU_CORES="$(sysctl -n hw.ncpu 2>/dev/null || true)"
  MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || true)"
else
  if [[ -f /etc/os-release ]]; then
    OS_PRETTY="$(grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '"')"
  else
    OS_PRETTY="$OS_NAME"
  fi
  CPU_MODEL="$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^ +/, "", $2); print $2; exit}')"
  CPU_CORES="$(nproc 2>/dev/null || true)"
  MEM_BYTES="$(awk '/MemTotal/ {print $2*1024}' /proc/meminfo 2>/dev/null || true)"
fi

GPU_JSON='[]'
if has_cmd nvidia-smi; then
  GPU_JSON="$(python - <<'PY'
import csv, io, json, subprocess
cmd = [
    'nvidia-smi',
    '--query-gpu=name,driver_version,memory.total,compute_cap,uuid',
    '--format=csv,noheader,nounits'
]
try:
    out = subprocess.check_output(cmd, text=True)
    rows = []
    reader = csv.reader(io.StringIO(out))
    for row in reader:
        if not row:
            continue
        name, driver, mem_mb, cc, uuid = [x.strip() for x in row]
        rows.append({
            'name': name,
            'driver_version': driver,
            'memory_total_mib': int(float(mem_mb)),
            'memory_total_gib': round(float(mem_mb) / 1024.0, 3),
            'compute_capability': cc,
            'uuid': uuid,
        })
    print(json.dumps(rows))
except Exception:
    print('[]')
PY
)"
fi

CUDA_RUNTIME=""
if has_cmd nvcc; then
  CUDA_RUNTIME="$(nvcc --version 2>/dev/null | tail -n 1 | sed 's/^ *//')"
fi

PYTHON_VERSIONS="{}"
if has_cmd python; then
  PYTHON_VERSIONS="$(capture_python_versions 2>/dev/null || echo '{}')"
fi

R_VERSIONS="{}"
if has_cmd R; then
  R_VERSIONS="$(capture_r_versions 2>/dev/null || echo '{}')"
fi

cat > "$OUT_JSON" <<EOF_JSON
{
  "captured_at_utc": $(json_escape "$DATE_UTC"),
  "hostname": $(json_escape "$HOSTNAME"),
  "os": {
    "kernel_family": $(json_escape "$OS_NAME"),
    "pretty_name": $(json_escape "$OS_PRETTY"),
    "kernel_release": $(json_escape "$KERNEL"),
    "architecture": $(json_escape "$ARCH")
  },
  "cpu": {
    "model": $(json_escape "$CPU_MODEL"),
    "logical_cores": $(json_escape "$CPU_CORES"),
    "memory_bytes": $(json_escape "$MEM_BYTES")
  },
  "gpu": $GPU_JSON,
  "cuda_runtime": $(json_escape "$CUDA_RUNTIME"),
  "python_packages": $PYTHON_VERSIONS,
  "r_packages": $R_VERSIONS
}
EOF_JSON

echo "Wrote $OUT_JSON"
cat "$OUT_JSON"
