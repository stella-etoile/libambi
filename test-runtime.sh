#!/usr/bin/env bash
set -euo pipefail

DATA="/mnt/Jupiter/dataset/ambi/clic2024_split/train_500"
CFG="ambi/config.yaml"

ITERS="${ITERS:-1}"
STEPS="${STEPS:-4096}"
LR="${LR:-1e-4}"
DEVICE="${DEVICE:-cpu}"

TTHREADS=(1 2 4)
BTHREADS=(1 2 4)
WORKERS=(1 2)
ENVS=(1 2)

export AMBI_NOPROGRESS=1

have_gnu_time=0
if command -v /usr/bin/time >/dev/null 2>&1; then
  have_gnu_time=1
fi

run_one() {
  tthreads="$1"
  bthreads="$2"
  workers="$3"
  envs="$4"
  name="t${tthreads}_b${bthreads}_w${workers}_e${envs}"
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
        --iters "$ITERS" --envs "$envs" --workers "$workers" --steps "$STEPS" \
        --lr "$LR" --device "$DEVICE" \
        --torch-threads "$tthreads" --blas-threads "$bthreads" \
      1>/dev/null 2>"$tmp" || true; tail -n1 "$tmp")
  else
    { time -p ambi-train "$DATA" --config "$CFG" --out "$out" \
        --iters "$ITERS" --envs "$envs" --workers "$workers" --steps "$STEPS" \
        --lr "$LR" --device "$DEVICE" \
        --torch-threads "$tthreads" --blas-threads "$bthreads" \
      1>/dev/null; } 2>"$tmp" || true
    secs=$(awk '/^real /{print $2}' "$tmp" | tail -n1)
  fi
  printf "%-18s %6ss\n" "$name" "${secs:-NA}"
}

printf "DATA=%s  CFG=%s  ITERS=%s  STEPS=%s  LR=%s  DEVICE=%s\n" "$DATA" "$CFG" "$ITERS" "$STEPS" "$LR" "$DEVICE"
printf "%-18s %6s\n" "combo" "seconds"

for t in "${TTHREADS[@]}"; do
  for b in "${BTHREADS[@]}"; do
    for w in "${WORKERS[@]}"; do
      for e in "${ENVS[@]}"; do
        run_one "$t" "$b" "$w" "$e"
      done
    done
  done
done
