#!/usr/bin/env bash
set -euo pipefail

DATA="/mnt/Jupiter/dataset/ambi/clic2024_split/train_100"
CFG="ambi/config.yaml"

ITERS="${ITERS:-1}"
STEPS="${STEPS:-4096}"   # total samples per iter across all envs
LR="${LR:-1e-4}"
DEVICE="${DEVICE:-cpu}"

TTHREADS=(1)
BTHREADS=(1)
WORKERS=(6)
ENVS=(6)

export AMBI_NOPROGRESS=1

have_gnu_time=0
if command -v /usr/bin/time >/dev/null 2>&1; then
  have_gnu_time=1
fi

ceil_div() { echo $(( ($1 + $2 - 1) / $2 )); }

run_one() {
  tthreads="$1"
  bthreads="$2"
  workers="$3"
  envs="$4"
  steps_per_env="$(ceil_div "$STEPS" "$envs")"
  name="t${tthreads}_b${bthreads}_w${workers}_e${envs}_s${steps_per_env}"
  export OMP_NUM_THREADS="$bthreads"
  export MKL_NUM_THREADS="$bthreads"
  export OPENBLAS_NUM_THREADS="$bthreads"
  export NUMEXPR_NUM_THREADS="$bthreads"
  out="/tmp/${name}.ts"
  tmp="./.time_${name}"
  secs=""
  if [[ "$have_gnu_time" -eq 1 ]]; then
    secs=$(/usr/bin/time -f "%e" \
      ambi-train "$DATA" --config "$CFG" --out "$out" \
        --iters "$ITERS" --envs "$envs" --workers "$workers" --steps "$steps_per_env" \
        --lr "$LR" --device "$DEVICE" \
        --torch-threads "$tthreads" --blas-threads "$bthreads" \
      1>/dev/null 2>"$tmp" || true; tail -n1 "$tmp")
  else
    { time -p ambi-train "$DATA" --config "$CFG" --out "$out" \
        --iters "$ITERS" --envs "$envs" --workers "$workers" --steps "$steps_per_env" \
        --lr "$LR" --device "$DEVICE" \
        --torch-threads "$tthreads" --blas-threads "$bthreads" \
      1>/dev/null; } 2>"$tmp" || true
    secs=$(awk '/^real /{print $2}' "$tmp" | tail -n1)
  fi
  printf "%-24s %6ss\n" "$name" "${secs:-NA}"
}

printf "DATA=%s  CFG=%s  ITERS=%s  TOTAL_STEPS=%s  LR=%s  DEVICE=%s\n" "$DATA" "$CFG" "$ITERS" "$STEPS" "$LR" "$DEVICE"
printf "%-24s %6s\n" "combo" "seconds"

for t in "${TTHREADS[@]}"; do
  for b in "${BTHREADS[@]}"; do
    for w in "${WORKERS[@]}"; do
      for e in "${ENVS[@]}"; do
        run_one "$t" "$b" "$w" "$e"
      done
    done
  done
done
