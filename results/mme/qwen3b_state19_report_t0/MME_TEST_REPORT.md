# SpareSpec MME 测试报告

- 测试日期：2026-07-23
- 模型：Qwen2.5-VL-3B-Instruct
- Draft checkpoint：`state_19`
- GPU：4
- 数据：MME 固定随机种子 42 的 100 题速度评测子集
- 解码：temperature=0，max_new_tokens=1024
- SpareSpec：total_token=30，depth=3，top_k=8，vis_select_tokens=64，vis_query_window=8

## 总体结果

| 指标 | Baseline | SpareSpec |
|---|---:|---:|
| 样本数 | 100 | 100 |
| Yes/No 正确数 | 78 | 78 |
| Yes/No 准确率 | 78.00% | 78.00% |
| 生成 token 数 | 6702 | 6617 |
| 生成 wall time | 131.34 s | 68.65 s |
| 吞吐 | 51.03 token/s | 96.38 token/s |
| 每 token 延迟 | 19.60 ms | 10.38 ms |
| 平均接受长度 | — | 2.386 |
| 接受长度中位数 | — | 2.0 |

## 对比结论

- 按 ViSpec 的平均每 token 时间口径，加速比为 **1.89×**。
- 按总生成 wall time，加速比为 **1.91×**。
- Baseline 与 SpareSpec 的 Yes/No 决策一致：**100/100**。
- 完整生成文本逐字一致：49/100。
- 生成 token 数一致：57/100。
- 两种模式准确率均为 **78.00%**，该子集上无任务准确率下降。

## 类别明细

| 类别 | 样本数 | Baseline | SpareSpec |
|---|---:|---:|---:|
| OCR | 1 | 1/1 (100.0%) | 1/1 (100.0%) |
| artwork | 13 | 9/13 (69.2%) | 9/13 (69.2%) |
| celebrity | 12 | 10/12 (83.3%) | 10/12 (83.3%) |
| code_reasoning | 1 | 1/1 (100.0%) | 1/1 (100.0%) |
| color | 2 | 1/2 (50.0%) | 1/2 (50.0%) |
| commonsense_reasoning | 4 | 2/4 (50.0%) | 2/4 (50.0%) |
| count | 3 | 1/3 (33.3%) | 1/3 (33.3%) |
| existence | 2 | 2/2 (100.0%) | 2/2 (100.0%) |
| landmark | 17 | 12/17 (70.6%) | 12/17 (70.6%) |
| numerical_calculation | 5 | 4/5 (80.0%) | 4/5 (80.0%) |
| position | 1 | 1/1 (100.0%) | 1/1 (100.0%) |
| posters | 15 | 14/15 (93.3%) | 14/15 (93.3%) |
| scene | 21 | 17/21 (81.0%) | 17/21 (81.0%) |
| text_translation | 3 | 3/3 (100.0%) | 3/3 (100.0%) |

## 说明

- 该脚本按照 ViSpec 仓库的速度测试方式，从 MME 中固定随机选取 100 题；它不是官方完整 2374 题 MME benchmark 总分。
- wall time 是脚本记录的模型生成阶段时间，包含每题 target prefill 和解码，但不包含模型加载、三次预热、图片读取及结果写盘。
- SpareSpec 的 selector attention 辅助 prefill与正式 target KV 已隔离；target 候选验证继续使用 ViSpec 的 logits tree verification。

## 结果文件

- `baseline.jsonl`：标准自回归输出
- `sparespec.jsonl`：SpareSpec 输出
- `metrics.json`：机器可读指标
