#!/usr/bin/env bash
# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
#
# Baseline-noise probe: run the RankMixer / KuaiRand_Video_Action baseline five times,
# changing nothing but the seed, and record where the logs landed.
#
# What it is for: a single training run's test AUC differs from the next one's even with
# identical code. Any automated loop that calls a +0.002 difference an "improvement" has
# to know how big that run-to-run spread is first, otherwise it is amplifying noise. This
# script produces the five runs; the spread is computed from their logs afterwards.
#
# Derived from run_all.sh on purpose -- same torchrun invocation, same per-run temporary
# directory isolation, same port allocation, same tee'd logs. A baseline measured through
# a different launcher would measure the launcher too.
#
# Three deliberate differences from run_all.sh, each for a reason:
#
#   1. MAX_RETRIES=0. run_all.sh retries once. A retried run here would be worse than a
#      failed one: `tee -a` appends the second attempt to the same log file, so one file
#      would contain two test evaluations, and a consumer reading it gets a silent
#      mixture. A noise measurement also must not quietly drop the seed that failed --
#      that is selection on success. Fail loudly, fix, rerun.
#   2. MR_OBSV / MR_MAX_STEPS are explicitly exported as off. They are the opt-in hooks
#      from unirank/pytorch/observables.py. If either were left set in the shell, all five
#      runs would be short or instrumented, and the numbers would still look plausible.
#   3. A provenance header (commit, tree state, torch/CUDA version, GPU state, expids) is
#      written into the master log before anything starts, and the working tree must be
#      clean. "Which code and which environment produced this ruler" is part of the
#      measurement, not a detail -- and it cannot be recovered afterwards.
#
# Usage (from the unirank environment):
#     bash run_probe.sh                 # ~5 x 45 min, both GPUs
#     bash run_probe.sh --dry-run       # checks + provenance only, no training
#
# =========================================================================

set -euo pipefail

cd "$(dirname "$0")"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ---- same knobs as run_all.sh -------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1

CONFIG_DIR="./config"
NPROC=2

BASE_PORT=29600          # offset from run_all.sh's 29500 so a leftover TIME_WAIT socket
RUN_IDX=0                # from an earlier run_all.sh cannot collide with this one

MAX_RETRIES=0            # see note 1 in the header
RETRY_WAIT_SECONDS=10

LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

# Must stay short: python multiprocessing puts AF_UNIX sockets under TMPDIR and a long
# path triggers "OSError: AF_UNIX path too long".
TEMP_ROOT="$(mktemp -d /tmp/ur.XXXXXX)"

TIMESTAMP=$(date '+%F_%H-%M-%S')
MASTER_LOG="${LOG_DIR}/run_probe_${TIMESTAMP}.log"

# ---- the five runs: identical except for seed ---------------------------------
EXPIDS=(
    "RankMixer_KuaiRand_Video_Action_NoiseProbe_s11"
    "RankMixer_KuaiRand_Video_Action_NoiseProbe_s12"
    "RankMixer_KuaiRand_Video_Action_NoiseProbe_s13"
    "RankMixer_KuaiRand_Video_Action_NoiseProbe_s14"
    "RankMixer_KuaiRand_Video_Action_NoiseProbe_s15"
)

# ---- the hooks must be off (note 2) ------------------------------------------
export MR_OBSV=0
export MR_MAX_STEPS=0

cleanup_temp_path() {
    local path="${1:-}"
    if [[ -n "${path}" && "${path}" == "${TEMP_ROOT}/"* && -d "${path}" ]]; then
        rm -rf "${path}" 2>/dev/null || true
    fi
}

cleanup_temp_dirs() {
    if [[ -d "${TEMP_ROOT}" ]]; then
        find "${TEMP_ROOT}" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
        rmdir "${TEMP_ROOT}" 2>/dev/null || true
    fi
}

prepare_run_temp_dir() {
    local attempt="$1"
    local run_temp_dir="${TEMP_ROOT}/t${RUN_IDX}_a${attempt}"
    mkdir -p "${run_temp_dir}"
    printf '%s\n' "${run_temp_dir}"
}

cleanup() {
    echo "[$(date '+%F %T')] Cleaning temporary files..."
    cleanup_temp_dirs
}
trap cleanup EXIT INT TERM

die() {
    echo "ERROR: $*" >&2
    exit 2
}

# ---- preflight: everything that can be checked before spending 3.75 h of GPU --
preflight() {
    command -v git >/dev/null || die "git not found"
    git rev-parse --git-dir >/dev/null 2>&1 || die "not a git repository"

    local dirty
    dirty="$(git status --porcelain)"
    if [[ -n "${dirty}" ]]; then
        die "working tree is not clean:
${dirty}
  The measurement has to be attributable to a commit. Anything uncommitted is invisible
  to whoever reads the numbers later, and to any worktree created from that commit."
    fi

    local expid
    for expid in "${EXPIDS[@]}"; do
        grep -q "^${expid}:" "${CONFIG_DIR}/model_config.yaml" \
            || die "expid ${expid} is missing from ${CONFIG_DIR}/model_config.yaml"
    done

    command -v torchrun >/dev/null || die "torchrun not on PATH -- activate the unirank environment"
    python -c "import torch" 2>/dev/null || die "cannot import torch -- wrong environment?"

    local busy
    busy="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true)"
    if [[ -n "${busy}" ]]; then
        die "GPUs are already busy:
${busy}
  Sharing the GPUs would fold a neighbour's memory pressure into the measured spread."
    fi
}

provenance() {
    echo "=================================================="
    echo "Baseline-noise probe"
    echo "  date            : $(date '+%F %T %z')"
    echo "  host            : $(hostname)"
    echo "  commit          : $(git rev-parse HEAD)"
    echo "  branch          : $(git rev-parse --abbrev-ref HEAD)"
    echo "  tree            : clean"
    echo "  config dir      : ${CONFIG_DIR}"
    echo "  python          : $(command -v python)"
    python - <<'PY'
import torch, sys
print(f"  torch           : {torch.__version__}  cuda={torch.version.cuda}  "
      f"devices={torch.cuda.device_count()}")
print(f"  sys.executable  : {sys.executable}")
PY
    echo "  CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
    echo "  NPROC           : ${NPROC}"
    echo "  MR_OBSV         : ${MR_OBSV}   (off)"
    echo "  MR_MAX_STEPS    : ${MR_MAX_STEPS}   (off)"
    echo "  MAX_RETRIES     : ${MAX_RETRIES}   (a retry would append a second run to one log)"
    echo "  expids          :"
    local expid
    for expid in "${EXPIDS[@]}"; do
        echo "      ${expid}  seed=$(awk "/^${expid}:/{f=1} f&&/seed:/{print \$2; exit}" \
            "${CONFIG_DIR}/model_config.yaml")"
    done
    echo "  master log      : ${MASTER_LOG}"
    echo "=================================================="
}

run_exp() {
    local expid="$1"
    local exp_log="${LOG_DIR}/${expid}_${TIMESTAMP}.log"
    local attempt=0

    while true; do
        attempt=$((attempt + 1))
        local port=$((BASE_PORT + RUN_IDX))
        RUN_IDX=$((RUN_IDX + 1))

        echo "=================================================="
        echo "[$(date '+%F %T')] Starting: ${expid} | attempt=${attempt} | master_port=${port}"
        echo "  per-run log: ${exp_log}"
        echo "=================================================="

        cleanup_temp_dirs
        local run_temp_dir
        run_temp_dir="$(prepare_run_temp_dir "${attempt}")"
        echo "[$(date '+%F %T')] Temporary dir: ${run_temp_dir}"

        local rc=0
        if TMPDIR="${run_temp_dir}" \
            TMP="${run_temp_dir}" \
            TEMP="${run_temp_dir}" \
            TORCHINDUCTOR_CACHE_DIR="${run_temp_dir}/torchinductor" \
            TRITON_CACHE_DIR="${run_temp_dir}/triton" \
            TORCH_EXTENSIONS_DIR="${run_temp_dir}/torch_extensions" \
            torchrun \
            --standalone \
            --master_port="${port}" \
            --nproc_per_node="${NPROC}" \
            run_expid.py \
            --config "${CONFIG_DIR}" \
            --expid "${expid}" \
            --gpu "${CUDA_VISIBLE_DEVICES}" 2>&1 | tee -a "${exp_log}"; then
            rc=0
        else
            rc=${PIPESTATUS[0]}
        fi

        cleanup_temp_path "${run_temp_dir}"
        cleanup_temp_dirs

        if [[ "${rc}" -eq 0 ]]; then
            echo "[$(date '+%F %T')] Finished: ${expid} (attempt=${attempt})"
            echo
            break
        fi

        echo "[$(date '+%F %T')] ERROR: ${expid} failed (attempt=${attempt}, exit_code=${rc})"
        if [[ "${MAX_RETRIES}" -ge 0 && "${attempt}" -gt "${MAX_RETRIES}" ]]; then
            echo "[$(date '+%F %T')] Aborting the whole probe."
            echo "  Not continuing with the remaining seeds: a spread computed from"
            echo "  whichever runs happened to succeed is conditioned on success, which is"
            echo "  exactly the bias this measurement exists to rule out."
            exit "${rc}"
        fi
        echo "[$(date '+%F %T')] Retrying after ${RETRY_WAIT_SECONDS}s..."
        sleep "${RETRY_WAIT_SECONDS}"
    done
}

preflight

# Everything from here also goes to the master log.
exec > >(tee -a "${MASTER_LOG}") 2>&1

provenance

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo
    echo "--dry-run: checks passed, nothing was trained."
    echo "Rerun without --dry-run to start ${#EXPIDS[@]} runs (~45 min each, sequential)."
    exit 0
fi

START_TS=$(date +%s)
for expid in "${EXPIDS[@]}"; do
    run_exp "${expid}"
done
echo "=================================================="
echo "[$(date '+%F %T')] All ${#EXPIDS[@]} runs finished in $((($(date +%s) - START_TS) / 60)) min"
echo "Per-run logs:"
for expid in "${EXPIDS[@]}"; do
    echo "  ${LOG_DIR}/${expid}_${TIMESTAMP}.log"
done
echo "Master log: ${MASTER_LOG}"
echo "=================================================="
