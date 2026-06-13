#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-}"
GPU="${2:-0}"

if [[ -z "${PROFILE}" ]]; then
  echo "Usage: bash experiments/run_reviewer_job.sh <profile> [gpu]"
  echo "Profiles: seed_variance, non_iid_vs_malicious, malicious_ratio, poison_strength, attack_success, baseline_methods, smoke"
  exit 1
fi

cd "$(dirname "$0")/.."

CONDA_SH="${CONDA_SH:-/home/sutongtong/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-wwt310}"
MODEL_TYPE="${MODEL_TYPE:-internvl}"
MODEL_PATH="${MODEL_PATH:-/home/sutongtong/wwt/model/InternVL3-2B}"
DATAROOT="${DATAROOT:-/home/sutongtong/LanTu_team3/dataset/nuScenes/train}"
VERSION="${VERSION:-v1.0-trainval}"

USER_NUM_CLIENTS="${NUM_CLIENTS-}"
USER_NUM_ROUNDS="${NUM_ROUNDS-}"
NUM_CLIENTS="${NUM_CLIENTS:-4}"
NUM_ROUNDS="${NUM_ROUNDS:-5}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_SYNTHETIC_SAMPLES="${NUM_SYNTHETIC_SAMPLES:-1000}"
MALICIOUS_RATIO="${MALICIOUS_RATIO:-0.2}"
MALICIOUS_ATTACK_RATIO="${MALICIOUS_ATTACK_RATIO:-0.8}"
LOGIT_POISONING_STRENGTH="${LOGIT_POISONING_STRENGTH:-5.0}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-experiments/reviewer_runs/${PROFILE}_${RUN_ID}}"
CONSOLE_LOG="${RUN_ROOT}/console.log"
PID_FILE="${RUN_ROOT}/pid.txt"
COMMAND_FILE="${RUN_ROOT}/command.txt"
mkdir -p "${RUN_ROOT}"

case "${PROFILE}" in
  smoke)
    SEEDS="${SEEDS:-1}"
    if [[ -z "${USER_NUM_CLIENTS}" ]]; then NUM_CLIENTS=2; fi
    if [[ -z "${USER_NUM_ROUNDS}" ]]; then NUM_ROUNDS=1; fi
    RATIOS="${RATIOS:-0,0.1,0.2,0.3,0.4,0.5}"
    STRENGTHS="${STRENGTHS:-0,2,5,8,12}"
    DIRICHLET_ALPHAS="${DIRICHLET_ALPHAS:-1.0,0.1,0.05}"
    METHODS="${METHODS:-active_inference,fedavg,fedprox}"
    ;;
  seed_variance)
    SEEDS="${SEEDS:-1,2,3,4,5}"
    RATIOS="${RATIOS:-0,0.1,0.2,0.3,0.4,0.5}"
    STRENGTHS="${STRENGTHS:-0,2,5,8,12}"
    DIRICHLET_ALPHAS="${DIRICHLET_ALPHAS:-1.0,0.1,0.05}"
    METHODS="${METHODS:-active_inference,fedavg,fedprox}"
    ;;
  non_iid_vs_malicious)
    SEEDS="${SEEDS:-1,2,3}"
    RATIOS="${RATIOS:-0,0.1,0.2,0.3,0.4,0.5}"
    STRENGTHS="${STRENGTHS:-0,2,5,8,12}"
    DIRICHLET_ALPHAS="${DIRICHLET_ALPHAS:-1.0,0.1,0.05}"
    METHODS="${METHODS:-active_inference,fedavg,fedprox}"
    ;;
  malicious_ratio)
    SEEDS="${SEEDS:-1,2,3}"
    RATIOS="${RATIOS:-0,0.1,0.2,0.3,0.4,0.5}"
    STRENGTHS="${STRENGTHS:-0,2,5,8,12}"
    DIRICHLET_ALPHAS="${DIRICHLET_ALPHAS:-1.0,0.1,0.05}"
    METHODS="${METHODS:-active_inference,fedavg,fedprox}"
    ;;
  poison_strength)
    SEEDS="${SEEDS:-1,2,3}"
    RATIOS="${RATIOS:-0,0.1,0.2,0.3,0.4,0.5}"
    STRENGTHS="${STRENGTHS:-0,2,5,8,12}"
    DIRICHLET_ALPHAS="${DIRICHLET_ALPHAS:-1.0,0.1,0.05}"
    METHODS="${METHODS:-active_inference,fedavg,fedprox}"
    ;;
  attack_success)
    SEEDS="${SEEDS:-1}"
    RATIOS="${RATIOS:-0.1,0.3,0.5}"
    STRENGTHS="${STRENGTHS:-2,5,8}"
    DIRICHLET_ALPHAS="${DIRICHLET_ALPHAS:-1.0,0.1,0.05}"
    METHODS="${METHODS:-active_inference,fedavg,fedprox}"
    ;;
  baseline_methods)
    SEEDS="${SEEDS:-1,2,3}"
    RATIOS="${RATIOS:-0,0.1,0.2,0.3,0.4,0.5}"
    STRENGTHS="${STRENGTHS:-0,2,5,8,12}"
    DIRICHLET_ALPHAS="${DIRICHLET_ALPHAS:-1.0,0.1,0.05}"
    METHODS="${METHODS:-active_inference,fedavg,fedprox}"
    ;;
  *)
    echo "Unknown profile: ${PROFILE}"
    exit 1
    ;;
esac

CMD=(
  python experiments/run_reviewer_experiments.py
  --profile "${PROFILE}"
  --run_root "${RUN_ROOT}"
  --gpu "${GPU}"
  --model_type "${MODEL_TYPE}"
  --model_path "${MODEL_PATH}"
  --dataroot "${DATAROOT}"
  --version "${VERSION}"
  --num_clients "${NUM_CLIENTS}"
  --num_rounds "${NUM_ROUNDS}"
  --local_epochs "${LOCAL_EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --num_synthetic_samples "${NUM_SYNTHETIC_SAMPLES}"
  --seeds "${SEEDS}"
  --ratios "${RATIOS}"
  --strengths "${STRENGTHS}"
  --dirichlet_alphas "${DIRICHLET_ALPHAS}"
  --methods "${METHODS}"
  --malicious_ratio "${MALICIOUS_RATIO}"
  --malicious_attack_ratio "${MALICIOUS_ATTACK_RATIO}"
  --logit_poisoning_strength "${LOGIT_POISONING_STRENGTH}"
)

{
  echo "PROFILE=${PROFILE}"
  echo "GPU=${GPU}"
  echo "RUN_ROOT=${RUN_ROOT}"
  printf '%q ' "${CMD[@]}"
  echo
} > "${COMMAND_FILE}"

nohup bash -lc "
  set -euo pipefail
  source '${CONDA_SH}'
  export CUDA_VISIBLE_DEVICES='${GPU}'
  export TOKENIZERS_PARALLELISM=false
  conda run -n '${CONDA_ENV}' ${CMD[*]}
  conda run -n '${CONDA_ENV}' python experiments/summarize_reviewer_results.py '${RUN_ROOT}'
" > "${CONSOLE_LOG}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_FILE}"
echo "Started ${PROFILE}"
echo "  PID: ${PID}"
echo "  GPU: ${GPU}"
echo "  Run root: ${RUN_ROOT}"
echo "  Console log: ${CONSOLE_LOG}"
echo "  Command: ${COMMAND_FILE}"
