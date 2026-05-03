#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../.."

SCRIPTS=(
  commands/mimic/MIMIC_NON_IID_SUD_FEDAVG.sh
  commands/mimic/MIMIC_NON_IID_SUD_MAXCS.sh
  commands/mimic/MIMIC_NON_IID_SUD_UCBCS.sh
  commands/mimic/MIMIC_NON_IID_SUD_UCBCS_W5.sh
  commands/mimic/MIMIC_NON_IID_SUD_UCBCS_W8.sh
)

SEEDS=($(seq 42 42))
PARALLEL=5

TOTAL=$(( ${#SCRIPTS[@]} * ${#SEEDS[@]} ))
PROGRESS_FILE=$(mktemp -t run_all_seeds.XXXXXX)
trap 'rm -f "$PROGRESS_FILE"' EXIT

run_one() {
  local label="$1" script="$2" seed="$3"
  local t_start=$SECONDS
  SEED="$seed" EXP_SUFFIX="_seed${seed}" bash "$script" >/dev/null 2>&1
  local rc=$?
  local dur=$((SECONDS - t_start))
  # atomic single-line append (< PIPE_BUF)
  echo "$label $rc" >> "$PROGRESS_FILE"
  local done_count
  done_count=$(awk 'END{print NR}' "$PROGRESS_FILE")
  local tag="ok"
  [ $rc -ne 0 ] && tag="FAIL(rc=$rc)"
  printf '[%d/%d] %-8s %s (%ds)\n' "$done_count" "$TOTAL" "$tag" "$label" "$dur"
}

printf 'Launching %d runs (%d scripts x %d seeds), parallelism %d\n' \
  "$TOTAL" "${#SCRIPTS[@]}" "${#SEEDS[@]}" "$PARALLEL"

for script in "${SCRIPTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    label="$(basename "$script" .sh)_seed${seed}"
    run_one "$label" "$script" "$seed" &
    # throttle: block while PARALLEL jobs are active (sleep-poll for bash 3.2 compat)
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do
      sleep 1
    done
  done
done

wait

fails=$(awk '$NF != 0' "$PROGRESS_FILE" | wc -l | tr -d ' ')
if [ "$fails" -eq 0 ]; then
  echo "All $TOTAL runs completed."
else
  echo "Completed $TOTAL runs; $fails failed."
fi
