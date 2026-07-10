#!/bin/bash
set -euo pipefail

mode=""
cuda_devices="${CUDA_VISIBLE_DEVICES:-4}"
base_model="/home/user/project/model/Qwen2.5-VL-3B-Instruct"
sharegpt_data_path="/home/user/project/dataset/ShareGPT_Vicuna_unfiltered"
llava_data_path="/home/user/project/dataset/LLaVA-Pretrain"
outdir="vispec_data/sparespec_data"
stage1_data=""
stage2_data=""
cpdir=""
configpath="vispec/train/qwen2.5_vl_3B_sparespec_config.json"
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

usage() {
  cat <<'EOF'
Usage: ./train_sparespec.sh --mode <stage1-data|stage2-data|stage1-train|stage2-train> [options]

Common options:
  --cuda-devices DEVICES       Physical CUDA devices to use. Default: 4
  --base-model PATH            Base VLM path. Default: /home/user/project/model/Qwen2.5-VL-3B-Instruct
  --outdir PATH                Data generation output root. Default: vispec_data/sparespec_data
  --configpath PATH            Draft config json for training modes. Default: vispec/train/qwen2.5_vl_3B_sparespec_config.json
  --cpdir PATH                 Checkpoint output dir for training modes.

Data generation:
  --start N --end N            Dataset slice. Default: 0 1000
  --gpus-per-model N           GPUs per process. Default: 1
  --data-path PATH             Backward-compatible dataset path; applied to both stage1/stage2.
  --sharegpt-data-path PATH    Local ShareGPT path for stage1-data. Default: /home/user/project/dataset/ShareGPT_Vicuna_unfiltered
  --llava-data-path PATH       LLaVA-Pretrain path for stage2-data. Default: /home/user/project/dataset/LLaVA-Pretrain
  --max-new-tokens N           Stage2 generated answer length. Default: 1024
  --temperature FLOAT          Stage2 generation temperature. Default: 0
  --save-attentions BOOL       Save query-token vis_attn_scores for stage2-data. Default: true

Training:
  --stage1-data PATH           Stage1 generated data dir for stage1-train.
  --stage2-data PATH           Stage2 generated data dir for stage2-train.
  --loadpath PATH              Stage1 checkpoint model.safetensors for stage2-train.
  --num-epochs N               Default: 20
  --max-train-steps N          Debug cap. 0 means full epoch.
  --max-val-batches N          Debug cap. 0 means full validation.
  --kacc-batches N             Validation k-acc batches. Use 0 for fast smoke.
  --vis-select-tokens N        Per-image segment max selected visual tokens. Default: 64
  --min-vis-select-tokens N    Entropy budget floor. Default: 16
  --vis-entropy-alpha FLOAT    Entropy budget multiplier. Default: 1.2
  --vis-query-window N         Recent text query window for selector. Default: 8
  --max-total-vis-select-tokens N  Optional global visual token cap. 0 disables.
EOF
}

require_arg() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "Error: missing required $name" >&2
    usage
    exit 1
  fi
}

str_bool_true() {
  case "${1,,}" in
    true|1|yes|y) return 0 ;;
    false|0|no|n) return 1 ;;
    *) echo "Error: invalid bool '$1'" >&2; exit 1 ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) mode="$2"; shift 2 ;;
    --cuda-devices) cuda_devices="$2"; shift 2 ;;
    --base-model) base_model="$2"; shift 2 ;;
    --data-path) sharegpt_data_path="$2"; llava_data_path="$2"; shift 2 ;;
    --sharegpt-data-path) sharegpt_data_path="$2"; shift 2 ;;
    --llava-data-path) llava_data_path="$2"; shift 2 ;;
    --outdir) outdir="$2"; shift 2 ;;
    --stage1-data) stage1_data="$2"; shift 2 ;;
    --stage2-data) stage2_data="$2"; shift 2 ;;
    --cpdir) cpdir="$2"; shift 2 ;;
    --configpath) configpath="$2"; shift 2 ;;
    --loadpath) loadpath="$2"; shift 2 ;;
    --start) start="$2"; shift 2 ;;
    --end) end="$2"; shift 2 ;;
    --gpus-per-model) gpus_per_model="$2"; shift 2 ;;
    --max-new-tokens) max_new_tokens="$2"; shift 2 ;;
    --temperature) temperature="$2"; shift 2 ;;
    --save-attentions) save_attentions="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --bs) bs="$2"; shift 2 ;;
    --gradient-accumulation-steps) grad_accum="$2"; shift 2 ;;
    --num-workers) num_workers="$2"; shift 2 ;;
    --max-len) max_len="$2"; shift 2 ;;
    --num-epochs) num_epochs="$2"; shift 2 ;;
    --max-train-steps) max_train_steps="$2"; shift 2 ;;
    --max-val-batches) max_val_batches="$2"; shift 2 ;;
    --kacc-batches) kacc_batches="$2"; shift 2 ;;
    --num-hidden-levels) num_hidden_levels="$2"; shift 2 ;;
    --vis-select-tokens) vis_select_tokens="$2"; shift 2 ;;
    --min-vis-select-tokens) min_vis_select_tokens="$2"; shift 2 ;;
    --vis-entropy-alpha) vis_entropy_alpha="$2"; shift 2 ;;
    --vis-query-window) vis_query_window="$2"; shift 2 ;;
    --max-total-vis-select-tokens) max_total_vis_select_tokens="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

require_arg "--mode" "$mode"
export CUDA_VISIBLE_DEVICES="$cuda_devices"

echo "[SpareSpec] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

case "$mode" in
  stage1-data)
    python -m vispec.ge_data.allocation_qwen_shargpt_sparespec \
      --start "$start" \
      --end "$end" \
      --outdir "$outdir" \
      --model "$base_model" \
      --data-path "$sharegpt_data_path" \
      --gpus_per_model "$gpus_per_model"
    ;;

  stage2-data)
    cmd=(python -m vispec.ge_data.allocation_qwen_pretrain_gen_sparespec
      --start "$start"
      --end "$end"
      --outdir "$outdir"
      --model "$base_model"
      --data-path "$llava_data_path"
      --max_new_tokens "$max_new_tokens"
      --temperature "$temperature"
      --vis-query-window "$vis_query_window"
      --gpus_per_model "$gpus_per_model")
    if str_bool_true "$save_attentions"; then
      cmd+=(--save-attentions)
    fi
    "${cmd[@]}"
    ;;

  stage1-train)
    require_arg "--stage1-data" "$stage1_data"
    require_arg "--cpdir" "$cpdir"
    require_arg "--configpath" "$configpath"
    python -m vispec.train.main_sparespec \
      --stage 1 \
      --tmpdir "$stage1_data" \
      --cpdir "$cpdir" \
      --basepath "$base_model" \
      --configpath "$configpath" \
      --lr "$lr" \
      --bs "$bs" \
      --gradient-accumulation-steps "$grad_accum" \
      --num-workers "$num_workers" \
      --max-len "$max_len" \
      --num-hidden-levels "$num_hidden_levels" \
      --num-epochs "$num_epochs" \
      --max-train-steps "$max_train_steps" \
      --max-val-batches "$max_val_batches" \
      --kacc-batches "$kacc_batches"
    ;;

  stage2-train)
    require_arg "--stage2-data" "$stage2_data"
    require_arg "--cpdir" "$cpdir"
    require_arg "--configpath" "$configpath"
    python -m vispec.train.main_sparespec \
      --stage 2 \
      --tmpdir "$stage2_data" \
      --cpdir "$cpdir" \
      --basepath "$base_model" \
      --configpath "$configpath" \
      ${loadpath:+--loadpath "$loadpath"} \
      --lr "$lr" \
      --bs "$bs" \
      --gradient-accumulation-steps "$grad_accum" \
      --num-workers "$num_workers" \
      --max-len "$max_len" \
      --num-hidden-levels "$num_hidden_levels" \
      --vis-select-tokens "$vis_select_tokens" \
      --min-vis-select-tokens "$min_vis_select_tokens" \
      --vis-entropy-alpha "$vis_entropy_alpha" \
      --vis-query-window "$vis_query_window" \
      --max-total-vis-select-tokens "$max_total_vis_select_tokens" \
      --num-epochs "$num_epochs" \
      --max-train-steps "$max_train_steps" \
      --max-val-batches "$max_val_batches" \
      --kacc-batches "$kacc_batches"
    ;;

  *)
    echo "Error: unknown mode '$mode'" >&2
    usage
    exit 1
    ;;
esac
