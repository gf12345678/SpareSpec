# SpareSpec 四模型数据生成与训练命令

本文档覆盖以下基础模型：

- LLaVA-v1.6 Vicuna 7B
- LLaVA-v1.6 Vicuna 13B
- Qwen2.5-VL 3B Instruct
- Qwen2.5-VL 7B Instruct

所有命令都应在 SpareSpec 项目根目录执行：

```bash
cd /data/gaofeng/SpareSpec
conda activate Sparespec
```

本机数据集路径：

```text
/data/gaofeng/dataset/ShareGPT_Vicuna_unfiltered
/data/gaofeng/dataset/LLaVA-Pretrain
```

约定：

- 数据生成使用 8 张卡并行，每张卡运行一个模型实例。
- Stage 1/2 训练示例使用物理 GPU 7，且 `batch_size=1`。
- Stage 1 默认学习率为 `3e-5`，Stage 2 默认学习率为 `3e-6`。
- Stage 1 使用单步训练；Stage 2 使用两步训练，第二步 loss 权重为 `0.5`。
- Stage 2 训练前，将 `STAGE1_CKPT` 设置为实际 Stage 1 checkpoint 中的 `model.safetensors`。
- 下列 `--end 67999` 表示处理索引 `[0, 67999)`；因无效样本过滤，最终 `.ckpt` 数可能少于 67999。

## 1. LLaVA-v1.6 Vicuna 7B

### Stage 1 数据生成

```bash
./train_sparespec.sh \
  --mode stage1-data \
  --cuda-devices 0,1,2,3,4,5,6,7 \
  --base-model /data/gaofeng/model/llava-v1.6-vicuna-7b-hf \
  --sharegpt-data-path /data/gaofeng/dataset/ShareGPT_Vicuna_unfiltered \
  --outdir vispec_data/llava-v1.6-vicuna-7b-hf/sparespec_data \
  --start 0 \
  --end 67999 \
  --gpus-per-model 1
```

Stage 1 数据目录：

```text
vispec_data/llava-v1.6-vicuna-7b-hf/sparespec_data/llava_v1.6_shargpt_sparespec_0_67999_mubf16
```

### Stage 1 训练

```bash
./train_sparespec.sh \
  --mode stage1-train \
  --cuda-devices 7 \
  --base-model /data/gaofeng/model/llava-v1.6-vicuna-7b-hf \
  --stage1-data vispec_data/llava-v1.6-vicuna-7b-hf/sparespec_data/llava_v1.6_shargpt_sparespec_0_67999_mubf16 \
  --cpdir vispec_data/llava-v1.6-vicuna-7b-hf/checkpoints/stage1 \
  --configpath vispec/train/llava_1.6_7B_config.json \
  --bs 1 \
  --gradient-accumulation-steps 1 \
  --num-workers 8 \
  --num-epochs 20 \
  --max-len 2048
```

### Stage 2 数据生成

```bash
./train_sparespec.sh \
  --mode stage2-data \
  --cuda-devices 0,1,2,3,4,5,6,7 \
  --base-model /data/gaofeng/model/llava-v1.6-vicuna-7b-hf \
  --llava-data-path /data/gaofeng/dataset/LLaVA-Pretrain \
  --outdir vispec_data/llava-v1.6-vicuna-7b-hf/sparespec_data \
  --start 0 \
  --end 67999 \
  --gpus-per-model 1 \
  --max-new-tokens 1024 \
  --temperature 0 \
  --save-attentions true \
  --vis-query-window 8
```

### Stage 2 训练

```bash
STAGE1_CKPT=vispec_data/llava-v1.6-vicuna-7b-hf/checkpoints/stage1/state_19/model.safetensors

./train_sparespec.sh \
  --mode stage2-train \
  --cuda-devices 7 \
  --base-model /data/gaofeng/model/llava-v1.6-vicuna-7b-hf \
  --stage2-data vispec_data/llava-v1.6-vicuna-7b-hf/sparespec_data/llava_v1.6_pretrain_sparespec_0_67999_mubf16 \
  --cpdir vispec_data/llava-v1.6-vicuna-7b-hf/checkpoints/stage2 \
  --loadpath "$STAGE1_CKPT" \
  --configpath vispec/train/llava_1.6_7B_config.json \
  --bs 1 \
  --gradient-accumulation-steps 1 \
  --num-workers 8 \
  --num-epochs 20 \
  --max-len 2048
```

## 2. LLaVA-v1.6 Vicuna 13B

### Stage 1 数据生成

```bash
./train_sparespec.sh \
  --mode stage1-data \
  --cuda-devices 0,1,2,3,4,5,6,7 \
  --base-model /data/gaofeng/model/llava-v1.6-vicuna-13b-hf \
  --sharegpt-data-path /data/gaofeng/dataset/ShareGPT_Vicuna_unfiltered \
  --outdir vispec_data/llava-v1.6-vicuna-13b-hf/sparespec_data \
  --start 0 \
  --end 67999 \
  --gpus-per-model 1
```

### Stage 1 训练

```bash
./train_sparespec.sh \
  --mode stage1-train \
  --cuda-devices 7 \
  --base-model /data/gaofeng/model/llava-v1.6-vicuna-13b-hf \
  --stage1-data vispec_data/llava-v1.6-vicuna-13b-hf/sparespec_data/llava_v1.6_shargpt_sparespec_0_67999_mubf16 \
  --cpdir vispec_data/llava-v1.6-vicuna-13b-hf/checkpoints/stage1 \
  --configpath vispec/train/llava_1.6_13B_config.json \
  --bs 1 \
  --gradient-accumulation-steps 1 \
  --num-workers 8 \
  --num-epochs 20 \
  --max-len 2048
```

### Stage 2 数据生成

```bash
./train_sparespec.sh \
  --mode stage2-data \
  --cuda-devices 0,1,2,3,4,5,6,7 \
  --base-model /data/gaofeng/model/llava-v1.6-vicuna-13b-hf \
  --llava-data-path /data/gaofeng/dataset/LLaVA-Pretrain \
  --outdir vispec_data/llava-v1.6-vicuna-13b-hf/sparespec_data \
  --start 0 \
  --end 67999 \
  --gpus-per-model 1 \
  --max-new-tokens 1024 \
  --temperature 0 \
  --save-attentions true \
  --vis-query-window 8
```

### Stage 2 训练

```bash
STAGE1_CKPT=vispec_data/llava-v1.6-vicuna-13b-hf/checkpoints/stage1/state_19/model.safetensors

./train_sparespec.sh \
  --mode stage2-train \
  --cuda-devices 7 \
  --base-model /data/gaofeng/model/llava-v1.6-vicuna-13b-hf \
  --stage2-data vispec_data/llava-v1.6-vicuna-13b-hf/sparespec_data/llava_v1.6_pretrain_sparespec_0_67999_mubf16 \
  --cpdir vispec_data/llava-v1.6-vicuna-13b-hf/checkpoints/stage2 \
  --loadpath "$STAGE1_CKPT" \
  --configpath vispec/train/llava_1.6_13B_config.json \
  --bs 1 \
  --gradient-accumulation-steps 1 \
  --num-workers 8 \
  --num-epochs 20 \
  --max-len 2048
```

## 3. Qwen2.5-VL 3B Instruct

### Stage 1 数据生成

```bash
./train_sparespec.sh \
  --mode stage1-data \
  --cuda-devices 0,1,2,3,4,5,6,7 \
  --base-model /data/gaofeng/model/Qwen2.5-VL-3B-Instruct \
  --sharegpt-data-path /data/gaofeng/dataset/ShareGPT_Vicuna_unfiltered \
  --outdir vispec_data/Qwen2.5-VL-3B-Instruct/sparespec_data \
  --start 0 \
  --end 67999 \
  --gpus-per-model 1
```

### Stage 1 训练

```bash
./train_sparespec.sh \
  --mode stage1-train \
  --cuda-devices 7 \
  --base-model /data/gaofeng/model/Qwen2.5-VL-3B-Instruct \
  --stage1-data vispec_data/Qwen2.5-VL-3B-Instruct/sparespec_data/qwen2.5vl_shargpt_sparespec_0_67999_mubf16 \
  --cpdir vispec_data/Qwen2.5-VL-3B-Instruct/checkpoints/stage1 \
  --configpath vispec/train/qwen2.5_vl_3B_sparespec_config.json \
  --bs 1 \
  --gradient-accumulation-steps 1 \
  --num-workers 8 \
  --num-epochs 20 \
  --max-len 2048
```

### Stage 2 数据生成

```bash
./train_sparespec.sh \
  --mode stage2-data \
  --cuda-devices 0,1,2,3,4,5,6,7 \
  --base-model /data/gaofeng/model/Qwen2.5-VL-3B-Instruct \
  --llava-data-path /data/gaofeng/dataset/LLaVA-Pretrain \
  --outdir vispec_data/Qwen2.5-VL-3B-Instruct/sparespec_data \
  --start 0 \
  --end 67999 \
  --gpus-per-model 1 \
  --max-new-tokens 1024 \
  --temperature 0 \
  --save-attentions true \
  --vis-query-window 8
```

### Stage 2 训练

```bash
STAGE1_CKPT=vispec_data/Qwen2.5-VL-3B-Instruct/checkpoints/stage1/state_19/model.safetensors

./train_sparespec.sh \
  --mode stage2-train \
  --cuda-devices 7 \
  --base-model /data/gaofeng/model/Qwen2.5-VL-3B-Instruct \
  --stage2-data vispec_data/Qwen2.5-VL-3B-Instruct/sparespec_data/qwen2.5vl_pretrain_sparespec_0_67999_mubf16 \
  --cpdir vispec_data/Qwen2.5-VL-3B-Instruct/checkpoints/stage2 \
  --loadpath "$STAGE1_CKPT" \
  --configpath vispec/train/qwen2.5_vl_3B_sparespec_config.json \
  --bs 1 \
  --gradient-accumulation-steps 1 \
  --num-workers 8 \
  --num-epochs 20 \
  --max-len 2048
```

## 4. Qwen2.5-VL 7B Instruct

### Stage 1 数据生成

```bash
./train_sparespec.sh \
  --mode stage1-data \
  --cuda-devices 0,1,2,3,4,5,6,7 \
  --base-model /data/gaofeng/model/Qwen2.5-VL-7B-Instruct \
  --sharegpt-data-path /data/gaofeng/dataset/ShareGPT_Vicuna_unfiltered \
  --outdir vispec_data/Qwen2.5-VL-7B-Instruct/sparespec_data \
  --start 0 \
  --end 67999 \
  --gpus-per-model 1
```

### Stage 1 训练

```bash
./train_sparespec.sh \
  --mode stage1-train \
  --cuda-devices 7 \
  --base-model /data/gaofeng/model/Qwen2.5-VL-7B-Instruct \
  --stage1-data vispec_data/Qwen2.5-VL-7B-Instruct/sparespec_data/qwen2.5vl_shargpt_sparespec_0_67999_mubf16 \
  --cpdir vispec_data/Qwen2.5-VL-7B-Instruct/checkpoints/stage1 \
  --configpath vispec/train/qwen2.5_vl_7B_config.json \
  --bs 1 \
  --gradient-accumulation-steps 1 \
  --num-workers 8 \
  --num-epochs 20 \
  --max-len 2048
```

### Stage 2 数据生成

```bash
./train_sparespec.sh \
  --mode stage2-data \
  --cuda-devices 0,1,2,3,4,5,6,7 \
  --base-model /data/gaofeng/model/Qwen2.5-VL-7B-Instruct \
  --llava-data-path /data/gaofeng/dataset/LLaVA-Pretrain \
  --outdir vispec_data/Qwen2.5-VL-7B-Instruct/sparespec_data \
  --start 0 \
  --end 67999 \
  --gpus-per-model 1 \
  --max-new-tokens 1024 \
  --temperature 0 \
  --save-attentions true \
  --vis-query-window 8
```

### Stage 2 训练

```bash
STAGE1_CKPT=vispec_data/Qwen2.5-VL-7B-Instruct/checkpoints/stage1/state_19/model.safetensors

./train_sparespec.sh \
  --mode stage2-train \
  --cuda-devices 7 \
  --base-model /data/gaofeng/model/Qwen2.5-VL-7B-Instruct \
  --stage2-data vispec_data/Qwen2.5-VL-7B-Instruct/sparespec_data/qwen2.5vl_pretrain_sparespec_0_67999_mubf16 \
  --cpdir vispec_data/Qwen2.5-VL-7B-Instruct/checkpoints/stage2 \
  --loadpath "$STAGE1_CKPT" \
  --configpath vispec/train/qwen2.5_vl_7B_config.json \
  --bs 1 \
  --gradient-accumulation-steps 1 \
  --num-workers 8 \
  --num-epochs 20 \
  --max-len 2048
```

## 5. Checkpoint 与断点续训

每个 epoch 会保存完整 Accelerate 状态：

```text
<cpdir>/state_<epoch>/
```

其中包括模型、优化器、学习率调度器和随机状态。使用相同 `--cpdir` 重新运行同一阶段时，训练代码会自动从最后一个完整状态继续。

Stage 2 初次训练必须通过 `--loadpath` 加载 Stage 1 的 `model.safetensors`。Stage 2 已产生自身 checkpoint 后，再次使用同一个 Stage 2 `--cpdir` 时会优先恢复完整 Stage 2 状态。

## 6. 常见提示

- WandB 或 TensorBoard 导入失败只会关闭对应日志，不影响训练和 checkpoint。
- 数据目录下允许包含 GPU worker 子目录，例如 `0/` 到 `7/`；训练加载器会递归查找 `.ckpt`。
- Stage 2 目前只支持 `batch_size=1`。
- 13B 数据生成会在每张卡各加载一份模型；若单卡显存不足，当前脚本不能仅通过增大 `--gpus-per-model` 实现模型并行。
