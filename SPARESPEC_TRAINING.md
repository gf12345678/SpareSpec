# SpareSpec Training

本文档记录当前 SpareSpec 的训练流程。所有 GPU smoke/测试默认固定在物理 GPU 4 上运行：

```bash
export CUDA_VISIBLE_DEVICES=4
```

旧 ViSpec 数据不能直接训练新版 SpareSpec：旧数据的 `hidden_state` 是单层 `H`，并且 stage2 数据缺少 `vis_anchor` / `vis_attn_scores`。新版 SpareSpec 需要重新生成数据：

- stage1 text-only: `hidden_state = [low, middle, high]`，形状为 `3H`。
- stage2 multimodal: 额外保存 `image_mask`、`vis_anchor`，建议打开 `--save-attentions` 保存 query-token `vis_attn_scores`。

## 1. Stage1 Text Data


Stage1 默认读取本地 ShareGPT 目录：

```text
/home/user/project/dataset/ShareGPT_Vicuna_unfiltered
```

该目录下的 `ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json` 会被优先使用；如果没有这个文件，会回退到 `ShareGPT_V3_unfiltered_cleaned_split.json`。

生成 ShareGPT text-only 数据：

```bash
./train_sparespec.sh \
  --mode stage1-data \
  --cuda-devices 4 \
  --base-model /home/user/project/model/Qwen2.5-VL-3B-Instruct \
  --sharegpt-data-path /home/user/project/dataset/ShareGPT_Vicuna_unfiltered \
  --outdir vispec_data/sparespec_data \
  --start 0 \
  --end 67999
```

输出目录形如：

```text
vispec_data/sparespec_data/qwen2.5vl_shargpt_sparespec_0_67999_mubf16/0
```

## 2. Stage1 Train

Stage1 只训练 EAGLE3 风格 text token 路径：

```bash
./train_sparespec.sh \
  --mode stage1-train \
  --cuda-devices 4 \
  --base-model /home/user/project/model/Qwen2.5-VL-3B-Instruct \
  --stage1-data vispec_data/sparespec_data/qwen2.5vl_shargpt_sparespec_0_67999_mubf16 \
  --cpdir vispec_data/checkpoints/sparespec_stage1 \
  --configpath vispec/train/qwen2.5_vl_3B_sparespec_config.json \
  --bs 4 \
  --gradient-accumulation-steps 1 \
  --num-epochs 21 \
  --max-len 4096
```

Stage1 产出的 checkpoint 用于初始化 Stage2：

```text
vispec_data/checkpoints/sparespec_stage1/state_<epoch>/model.safetensors
```

## 3. Stage2 Multimodal Data

生成 multimodal 数据，并保存 selector 训练需要的 query-token `vis_attn_scores`：

```bash
./train_sparespec.sh \
  --mode stage2-data \
  --cuda-devices 4 \
  --base-model /home/user/project/model/Qwen2.5-VL-3B-Instruct \
  --llava-data-path /home/user/project/dataset/LLaVA-Pretrain \
  --outdir vispec_data/sparespec_data \
  --start 0 \
  --end 67999 \
  --save-attentions true
```

输出目录形如：

```text
vispec_data/sparespec_data/qwen2.5vl_pretrain_sparespec_0_67999_mubf16/0
```

每条 ckpt 应包含：

```text
inputs_embeds: [seq, H]
hidden_state:  [seq, 3H]
loss_mask:     [seq]
image_mask:    [seq]
vis_anchor:    [1, H]
query_token_mask: [seq]  # 用户问题文本 token；非 query 位置为 0
vis_attn_scores: [seq]  # query_token_mask 最近文本 token 对视觉 token 的 attention mean；非视觉位置为 0
```

## 4. Stage2 Train

Stage2 训练完整 SpareSpec：text token 路径、global `vis_anchor`、selected vision tokens、vision adapter、entropy selector。

```bash
./train_sparespec.sh \
  --mode stage2-train \
  --cuda-devices 4 \
  --base-model /home/user/project/model/Qwen2.5-VL-3B-Instruct \
  --stage2-data vispec_data/sparespec_data/qwen2.5vl_pretrain_sparespec_0_67999_mubf16 \
  --cpdir vispec_data/checkpoints/sparespec_stage2 \
  --configpath vispec/train/qwen2.5_vl_3B_sparespec_config.json \
  --loadpath vispec_data/checkpoints/sparespec_stage1/state_20/model.safetensors \
  --bs 1 \
  --gradient-accumulation-steps 4 \
  --num-epochs 21 \
  --max-len 4096 \
  --vis-select-tokens 64 \
  --min-vis-select-tokens 16 \
  --vis-entropy-alpha 1.2 \
  --vis-query-window 8 \
  --max-total-vis-select-tokens 0
```

说明：

- `--vis-select-tokens`: 每个图像段最多保留多少 selected visual detail tokens。
- `--min-vis-select-tokens`: entropy adaptive budget 的下限。
- `--vis-entropy-alpha`: `ceil(alpha * exp(entropy))` 的放大系数。
- `--vis-query-window`: 使用最近多少个 text tokens 的 text-to-vision attention 做 selector。
- `--max-total-vis-select-tokens`: 多图/视频时的全局 visual token cap；`0` 表示关闭。

当前 stage2 forward 的 vision token 插入逻辑只支持 batch size 1，因此脚本会在 `--stage 2` 且 `--bs != 1` 时自动改为 1。需要更大有效 batch 时用 `--gradient-accumulation-steps`。

## 5. Smoke Commands

生成 1 条真实 stage2 数据：

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n vispec python -m vispec.ge_data.ge_data_all_qwen_pretrain_gen_sparespec \
  --start 0 \
  --end 1 \
  --index 0 \
  --gpu_index 4 \
  --outdir /tmp/sparespec_smoke \
  --max_new_tokens 4 \
  --model /home/user/project/model/Qwen2.5-VL-3B-Instruct \
  --data-path /home/user/project/dataset/LLaVA-Pretrain \
  --temperature 0 \
  --save-attentions
```

用 tiny config 跑 1 step 训练 smoke：

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n vispec python -m vispec.train.main_sparespec \
  --stage 2 \
  --tmpdir /tmp/sparespec_smoke \
  --cpdir /tmp/sparespec_train_smoke_run \
  --basepath /home/user/project/model/Qwen2.5-VL-3B-Instruct \
  --configpath /tmp/sparespec_tiny_config.json \
  --num-epochs 1 \
  --max-train-steps 1 \
  --max-val-batches 1 \
  --kacc-batches 0 \
  --bs 1 \
  --num-workers 0 \
  --max-len 512 \
  --vis-select-tokens 32 \
  --min-vis-select-tokens 4 \
  --vis-entropy-alpha 1.0 \
  --vis-query-window 4 \
  --max-total-vis-select-tokens 16
```
