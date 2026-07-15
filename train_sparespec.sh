#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

mode=""
cuda_devices="${CUDA_VISIBLE_DEVICES:-4}"

base_model="/home/user/project/model/Qwen2.5-VL-3B-Instruct"
configpath="vispec/train/qwen2.5_vl_3B_sparespec_config.json"

sharegpt_data_path="/home/user/project/dataset/ShareGPT_Vicuna_unfiltered"
llava_data_path="/home/user/project/dataset/LLaVA-Pretrain"
legacy_data_path=""
outdir="vispec_data/sparespec_data"

stage1_data=""
stage2_data=""
cpdir=""
loadpath=""

start="0"
end="1000"
gpus_per_model="1"

max_new_tokens="1024"
temperature="0"
save_attentions="true"

lr="3e-5"
bs="1"
grad_accum="1"
num_workers="2"
max_len="3200"
num_epochs="20"
max_train_steps="0"
max_val_batches="0"
kacc_batches="10"

num_hidden_levels="3"
vis_select_tokens="64"
min_vis_select_tokens="16"
vis_entropy_alpha="1.2"
vis_query_window="8"
max_total_vis_select_tokens="0"

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

usage() {
  cat <<EOF
Usage:
  ./train_sparespec.sh --mode MODE [options]

Modes:
  stage1-data     Generate text-only ShareGPT hidden-state data.
  stage1-train    Train the text draft model on generated stage1 data.
  stage2-data     Generate multimodal LLaVA-Pretrain data with visual fields.
  stage2-train    Train/fine-tune SpareSpec on generated stage2 data.

Common:
  --cuda-devices DEVICES       Physical CUDA devices. Default: CUDA_VISIBLE_DEVICES or 4
  --base-model PATH            Base VLM. Default: /home/user/project/model/Qwen2.5-VL-3B-Instruct
  --configpath PATH            Draft config for train modes.

Stage1 data:
  --sharegpt-data-path PATH    ShareGPT directory/json.
  --data-path PATH             Legacy alias for the active data mode.
  --start N --end N            Dataset slice. Default: 0 1000
  --outdir PATH                Generated data root. Default: vispec_data/sparespec_data
  --gpus-per-model N           GPUs per generation worker. Default: 1

Stage1 train:
  --stage1-data PATH           Generated stage1 data directory. Required.
  --cpdir PATH                 Checkpoint output directory. Required.
  --lr FLOAT                   Default: 3e-5
  --bs N                       Default: 1
  --gradient-accumulation-steps N  Default: 1
  --num-workers N              Default: 2
  --max-len N                  Default: 3200
  --num-epochs N               Default: 20
  --max-train-steps N          Debug cap. 0 means full epoch.
  --max-val-batches N          Debug cap. 0 means full validation.
  --kacc-batches N             Validation k-step accuracy batches. 0 disables.

Stage2 data:
  --llava-data-path PATH       LLaVA-Pretrain directory.
  --data-path PATH             Legacy alias for the active data mode.
  --start N --end N            Dataset slice. Default: 0 1000
  --outdir PATH                Generated data root. Default: vispec_data/sparespec_data
  --gpus-per-model N           GPUs per generation worker. Default: 1
  --max-new-tokens N           Generated answer length. Default: 1024
  --temperature FLOAT          Generation temperature. Default: 0
  --save-attentions BOOL       Save compact vis_attn_scores. Default: true
  --vis-query-window N         Query-token window for attention scores. Default: 8

Stage2 train:
  --stage2-data PATH           Generated stage2 data directory. Required.
  --cpdir PATH                 Checkpoint output directory. Required.
  --loadpath PATH              Stage1 checkpoint model.safetensors.
  --vis-select-tokens N        Global selected visual-token cap if max-total is 0. Default: 64
  --min-vis-select-tokens N    Entropy budget floor. Default: 16
  --vis-entropy-alpha FLOAT    Entropy budget multiplier. Default: 1.2
  --vis-query-window N         Recent/query text tokens for selector. Default: 8
  --max-total-vis-select-tokens N  Explicit global visual-token cap. 0 disables.

Examples:
  CUDA_VISIBLE_DEVICES=4 conda run -n vispec bash ./train_sparespec.sh --mode stage1-train --stage1-data vispec_data/sparespec_data/qwen2.5vl_shargpt_sparespec_0_67999_mubf16 --cpdir checkpoints/sparespec_qwen25vl3b_stage1
  CUDA_VISIBLE_DEVICES=4 conda run -n vispec bash ./train_sparespec.sh --mode stage2-data --start 0 --end 1000 --llava-data-path /home/user/project/dataset/LLaVA-Pretrain
EOF
}

die() {
  echo "Error: $*" >&2
  echo >&2
  usage >&2
  exit 1
}

require_value() {
  local opt="$1"
  local value="${2:-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    die "missing value for $opt"
  fi
}

require_arg() {
  local opt="$1"
  local value="$2"
  [[ -n "$value" ]] || die "missing required $opt"
}

bool_enabled() {
  case "${1,,}" in
    true|1|yes|y) return 0 ;;
    false|0|no|n) return 1 ;;
    *) die "invalid boolean: $1" ;;
  esac
}

print_run_header() {
  echo "[SpareSpec] root=$ROOT_DIR"
  echo "[SpareSpec] mode=$mode"
  echo "[SpareSpec] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) require_value "$1" "${2:-}"; mode="$2"; shift 2 ;;
    --cuda-devices) require_value "$1" "${2:-}"; cuda_devices="$2"; shift 2 ;;
    --base-model) require_value "$1" "${2:-}"; base_model="$2"; shift 2 ;;
    --configpath) require_value "$1" "${2:-}"; configpath="$2"; shift 2 ;;

    --sharegpt-data-path) require_value "$1" "${2:-}"; sharegpt_data_path="$2"; shift 2 ;;
    --llava-data-path) require_value "$1" "${2:-}"; llava_data_path="$2"; shift 2 ;;
    --data-path) require_value "$1" "${2:-}"; legacy_data_path="$2"; shift 2 ;;
    --outdir) require_value "$1" "${2:-}"; outdir="$2"; shift 2 ;;
    --start) require_value "$1" "${2:-}"; start="$2"; shift 2 ;;
    --end) require_value "$1" "${2:-}"; end="$2"; shift 2 ;;
    --gpus-per-model) require_value "$1" "${2:-}"; gpus_per_model="$2"; shift 2 ;;

    --stage1-data) require_value "$1" "${2:-}"; stage1_data="$2"; shift 2 ;;
    --stage2-data) require_value "$1" "${2:-}"; stage2_data="$2"; shift 2 ;;
    --cpdir) require_value "$1" "${2:-}"; cpdir="$2"; shift 2 ;;
    --loadpath) require_value "$1" "${2:-}"; loadpath="$2"; shift 2 ;;

    --max-new-tokens) require_value "$1" "${2:-}"; max_new_tokens="$2"; shift 2 ;;
    --temperature) require_value "$1" "${2:-}"; temperature="$2"; shift 2 ;;
    --save-attentions) require_value "$1" "${2:-}"; save_attentions="$2"; shift 2 ;;

    --lr) require_value "$1" "${2:-}"; lr="$2"; shift 2 ;;
    --bs) require_value "$1" "${2:-}"; bs="$2"; shift 2 ;;
    --gradient-accumulation-steps) require_value "$1" "${2:-}"; grad_accum="$2"; shift 2 ;;
    --num-workers) require_value "$1" "${2:-}"; num_workers="$2"; shift 2 ;;
    --max-len) require_value "$1" "${2:-}"; max_len="$2"; shift 2 ;;
    --num-epochs) require_value "$1" "${2:-}"; num_epochs="$2"; shift 2 ;;
    --max-train-steps) require_value "$1" "${2:-}"; max_train_steps="$2"; shift 2 ;;
    --max-val-batches) require_value "$1" "${2:-}"; max_val_batches="$2"; shift 2 ;;
    --kacc-batches) require_value "$1" "${2:-}"; kacc_batches="$2"; shift 2 ;;

    --num-hidden-levels) require_value "$1" "${2:-}"; num_hidden_levels="$2"; shift 2 ;;
    --vis-select-tokens) require_value "$1" "${2:-}"; vis_select_tokens="$2"; shift 2 ;;
    --min-vis-select-tokens) require_value "$1" "${2:-}"; min_vis_select_tokens="$2"; shift 2 ;;
    --vis-entropy-alpha) require_value "$1" "${2:-}"; vis_entropy_alpha="$2"; shift 2 ;;
    --vis-query-window) require_value "$1" "${2:-}"; vis_query_window="$2"; shift 2 ;;
    --max-total-vis-select-tokens) require_value "$1" "${2:-}"; max_total_vis_select_tokens="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

require_arg "--mode" "$mode"

if [[ -n "$legacy_data_path" ]]; then
  case "$mode" in
    stage1-data) sharegpt_data_path="$legacy_data_path" ;;
    stage2-data) llava_data_path="$legacy_data_path" ;;
    *) die "--data-path is only valid for stage1-data or stage2-data; use --stage1-data/--stage2-data for train modes" ;;
  esac
fi

export CUDA_VISIBLE_DEVICES="$cuda_devices"

# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

run_stage1_data() {
  local cmd=(
    python -m vispec.ge_data.allocation_qwen_shargpt_sparespec
    --start "$start"
    --end "$end"
    --outdir "$outdir"
    --model "$base_model"
    --data-path "$sharegpt_data_path"
    --gpus_per_model "$gpus_per_model"
  )
  "${cmd[@]}"
}

run_stage2_data() {
  local cmd=(
    python -m vispec.ge_data.allocation_qwen_pretrain_gen_sparespec
    --start "$start"
    --end "$end"
    --outdir "$outdir"
    --model "$base_model"
    --data-path "$llava_data_path"
    --max_new_tokens "$max_new_tokens"
    --temperature "$temperature"
    --vis-query-window "$vis_query_window"
    --gpus_per_model "$gpus_per_model"
  )

  if bool_enabled "$save_attentions"; then
    cmd+=(--save-attentions)
  fi

  "${cmd[@]}"
}

run_stage1_train() {
  require_arg "--stage1-data" "$stage1_data"
  require_arg "--cpdir" "$cpdir"

  local cmd=(
    python -m vispec.train.main_sparespec
    --stage 1
    --tmpdir "$stage1_data"
    --cpdir "$cpdir"
    --basepath "$base_model"
    --configpath "$configpath"
    --lr "$lr"
    --bs "$bs"
    --gradient-accumulation-steps "$grad_accum"
    --num-workers "$num_workers"
    --max-len "$max_len"
    --num-hidden-levels "$num_hidden_levels"
    --num-epochs "$num_epochs"
    --max-train-steps "$max_train_steps"
    --max-val-batches "$max_val_batches"
    --kacc-batches "$kacc_batches"
  )
  "${cmd[@]}"
}

run_stage2_train() {
  require_arg "--stage2-data" "$stage2_data"
  require_arg "--cpdir" "$cpdir"

  local cmd=(
    python -m vispec.train.main_sparespec
    --stage 2
    --tmpdir "$stage2_data"
    --cpdir "$cpdir"
    --basepath "$base_model"
    --configpath "$configpath"
    --lr "$lr"
    --bs "$bs"
    --gradient-accumulation-steps "$grad_accum"
    --num-workers "$num_workers"
    --max-len "$max_len"
    --num-hidden-levels "$num_hidden_levels"
    --vis-select-tokens "$vis_select_tokens"
    --min-vis-select-tokens "$min_vis_select_tokens"
    --vis-entropy-alpha "$vis_entropy_alpha"
    --vis-query-window "$vis_query_window"
    --max-total-vis-select-tokens "$max_total_vis_select_tokens"
    --num-epochs "$num_epochs"
    --max-train-steps "$max_train_steps"
    --max-val-batches "$max_val_batches"
    --kacc-batches "$kacc_batches"
  )

  if [[ -n "$loadpath" ]]; then
    cmd+=(--loadpath "$loadpath")
  fi

  "${cmd[@]}"
}

print_run_header

case "$mode" in
  stage1-data) run_stage1_data ;;
  stage2-data) run_stage2_data ;;
  stage1-train) run_stage1_train ;;
  stage2-train) run_stage2_train ;;
  *) die "unknown mode: $mode" ;;
esac
