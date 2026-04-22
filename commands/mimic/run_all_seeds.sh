#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../.."

SCRIPTS=(
  commands/mimic/MIMIC_NON_IID_SUD_FEDAVG.sh
  commands/mimic/MIMIC_NON_IID_SUD_MAXCS.sh
  commands/mimic/MIMIC_NON_IID_SUD_UCBCS.sh
  commands/mimic/MIMIC_NON_IID_SUD_UCBCS_DELTA.sh
  commands/mimic/MIMIC_NON_IID_SUD_UCBCS_PARAM.sh
)

SEEDS=($(seq 42 51))
PARALLEL=5

for script in "${SCRIPTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    SEED="$seed" EXP_SUFFIX="_seed${seed}" bash "$script" &
    # throttle: block while PARALLEL jobs are active (sleep-poll for bash 3.2 compat)
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do
      sleep 1
    done
  done
done

wait
echo "All runs completed."
