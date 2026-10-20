#!/usr/bin/env bash
# Follow-up to run_experiments.sh: fix the gated model's weak spot on posts without an image.
#
#   caffeinate -i bash scripts/run_followup.sh            # all seeds (~5 h on an M4 Pro)
#   SEEDS="42" caffeinate -i bash scripts/run_followup.sh  # kept seed only (~2 h)
#
# 1. Gated fusion with modality dropout 0.4 (was 0.15), same seeds as the main table.
# 2. BERTweet seed 42 again, this time keeping model.pt, as the text-only fallback for the demo.
# Then rebuilds the results table. Safe to stop and restart: finished runs are skipped.
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/crisismmd}"
SEEDS="${SEEDS:-42 43 44}"
KEEP_SEED="${KEEP_SEED:-42}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

for seed in ${SEEDS}; do
  run="outputs/fusion_gated_md40/seed_${seed}"
  keep=(); robust=()
  [[ "${seed}" == "${KEEP_SEED}" ]] && keep=(--keep-model) && robust=(--robustness)
  if [[ ! -f "${run}/metrics.json" ]]; then
    step "train fusion_gated_md40 seed ${seed}"
    python -m crisislens.train --config configs/fusion_gated_md40.yaml --seed "${seed}" --data-dir "${DATA_DIR}" \
      ${keep[@]+"${keep[@]}"}
  fi
  if [[ ! -f "${run}/evaluation.json" ]]; then
    step "calibration + flag threshold: fusion_gated_md40 seed ${seed}"
    python -m crisislens.evaluate --run "${run}" ${robust[@]+"${robust[@]}"}
  fi
done

fallback="outputs/text_bertweet_kept/seed_${KEEP_SEED}"
if [[ ! -f "${fallback}/model.pt" ]]; then
  step "train text fallback (BERTweet seed ${KEEP_SEED}, model kept)"
  python -m crisislens.train --config configs/text_bertweet.yaml --seed "${KEEP_SEED}" --data-dir "${DATA_DIR}" \
    --output-dir outputs/text_bertweet_kept --keep-model
fi
[[ -f "${fallback}/evaluation.json" ]] || python -m crisislens.evaluate --run "${fallback}"

step "results table"
python -m crisislens.report
cat <<MSG

Done. Compare "no image" humanitarian F1 for the two gated models above, then export the better one:
  python -m crisislens.export_onnx --run outputs/<fusion_gated or fusion_gated_md40>/seed_${KEEP_SEED} --out exported/
  python -m crisislens.export_onnx --run ${fallback} --out exported/text_fallback/
MSG
