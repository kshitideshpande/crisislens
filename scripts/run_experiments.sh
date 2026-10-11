#!/usr/bin/env bash
# Run every experiment, in order, and build the results table.
#
#   bash scripts/run_experiments.sh                      # everything
#   SEEDS="42" bash scripts/run_experiments.sh           # one seed (quick first pass)
#   WANDB=1 bash scripts/run_experiments.sh              # also log to Weights & Biases
#   DATA_DIR=/path/to/crisismmd bash scripts/run_experiments.sh
#
# Safe to stop and restart: finished runs (metrics.json exists) are skipped.
# Rough time on an M4 Pro (16 GB), per seed: text ~30-45 min, image ~30-45 min,
# each fusion model ~60-90 min. Run it overnight; keep the laptop plugged in
# and stop it sleeping:  caffeinate -i bash scripts/run_experiments.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/crisismmd}"
SEEDS="${SEEDS:-42 43 44}"
KEEP_SEED="${KEEP_SEED:-42}"          # this seed's gated model is kept for export and the demo
EXTRA=()
[[ "${WANDB:-0}" == "1" ]] && EXTRA+=(--wandb)

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

train() {  # train <config-name> <seed> [--keep-model]
  local name=$1 seed=$2; shift 2
  if [[ -f "outputs/${name}/seed_${seed}/metrics.json" ]]; then
    echo "skip ${name} seed ${seed} (already done)"; return
  fi
  step "train ${name} seed ${seed}"
  python -m crisislens.train --config "configs/${name}.yaml" --seed "${seed}" --data-dir "${DATA_DIR}" \
    "$@" ${EXTRA[@]+"${EXTRA[@]}"}
}

step "check data"
python scripts/check_data.py --data-dir "${DATA_DIR}"

step "baseline: TF-IDF + logistic regression"
python -m crisislens.train_baseline --config configs/baseline_tfidf.yaml --data-dir "${DATA_DIR}" ${EXTRA[@]+"${EXTRA[@]}"}

for seed in ${SEEDS}; do
  keep=(); [[ "${seed}" == "${KEEP_SEED}" ]] && keep=(--keep-model)
  train text_bertweet "${seed}"
  train image_clip "${seed}"
  step "late fusion seed ${seed}"
  python -m crisislens.late_fusion --seed "${seed}"
  train fusion_concat "${seed}"
  train fusion_gated "${seed}" ${keep[@]+"${keep[@]}"}

  step "calibration + flag threshold: gated seed ${seed}"
  if [[ "${seed}" == "${KEEP_SEED}" ]]; then
    python -m crisislens.evaluate --run "outputs/fusion_gated/seed_${seed}" --robustness
  else
    python -m crisislens.evaluate --run "outputs/fusion_gated/seed_${seed}"
  fi
done

train fusion_gated_holdout "${KEEP_SEED}"

step "explanations for the kept gated model (local only, never publish)"
python -m crisislens.explain --run "outputs/fusion_gated/seed_${KEEP_SEED}" --n 12

step "results table"
python -m crisislens.report
echo
echo "Done. Next: python -m crisislens.export_onnx --run outputs/fusion_gated/seed_${KEEP_SEED} --out exported/"
