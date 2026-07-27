"""
SpareSpec multimodal data generation (Stage 2).
Saves multi-level hidden states, vis_anchor, and image_mask.
"""
import argparse

parser = argparse.ArgumentParser(description="SpareSpec Stage2 data generation")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=100)
parser.add_argument("--index", type=int, default=1)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])
parser.add_argument("--outdir", type=str, default="outdir0")
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument("--model", type=str, default="llava-hf/llava-v1.6-vicuna-7b-hf")
parser.add_argument("--data-path", type=str, default="LLaVA-Pretrain/")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--save-attentions", "--save_attentions", dest="save_attentions", action="store_true")
parser.add_argument("--vis-query-window", "--vis_query_window", dest="vis_query_window", type=int, default=8)
args = parser.parse_args()
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

import json
from typing import Dict

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

bigname = args.model

LLAVA_VICUNA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] != 'system' %}"
    "{{ message['role'].upper() + ': '}}"
    "{% endif %}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>\n' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% if message['role'] != 'assistant' %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ content['text'] + ' '}}"
    "{% endfor %}"
    "{% else %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{% generation %}{{ content['text'] + ' '}}{% endgeneration %}"
    "{% endfor %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
)


def build_dataset_rank(processor, path):
    with open(os.path.join(path, "blip_laion_cc_sbu_558k.json")) as f:
        ds = json.load(f)
    ds = Dataset.from_list(ds)
    ds = ds.shuffle(seed=42)
    ds1: Dataset = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names

    def preprocess_function(examples):
        conversation = [{
            "role": "system",
            "content": [{
                "type": "text",
                "text": "A chat between a curious human and an artificial intelligence assistant. "
                        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            }],
        }]
        query_texts = []
        for conv in examples["conversations"]:
            if conv["from"] == "human":
                assert conv["value"].endswith("\n<image>") or conv["value"].startswith("<image>\n")
                query_text = conv["value"].strip().strip("<image>").strip()
                query_texts.append(query_text)
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query_text},
                        {"type": "image"},
                        {"type": "text", "text": "Please answer with at least 1000 words."},
                    ],
                })
            elif conv["from"] == "gpt":
                pass
            else:
                raise ValueError("Unknown role")
        prompt_input = processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        image_file = os.path.join(path, examples["image"])
        if not os.path.exists(image_file):
            image_file = os.path.join(path, "images", examples["image"])
        outputs = {
            "image_files": [image_file],
            "text": prompt_input,
            "query_text": "\n".join(query_texts),
        }
        return outputs

    ds1 = ds1.map(
        preprocess_function, batched=False, num_proc=2,
        remove_columns=original_columns1, load_from_cache_file=False,
    )
    return ds1


processor = AutoProcessor.from_pretrained(bigname, use_fast=True)
model_config = AutoConfig.from_pretrained(bigname)
if not getattr(processor, "chat_template", None):
    processor.chat_template = LLAVA_VICUNA_CHAT_TEMPLATE
if getattr(processor, "patch_size", None) is None:
    processor.patch_size = model_config.vision_config.patch_size
if getattr(processor, "vision_feature_select_strategy", None) is None:
    processor.vision_feature_select_strategy = (
        model_config.vision_feature_select_strategy
    )
processor.num_additional_image_tokens = 1
ds = build_dataset_rank(processor, args.data_path)
print(ds)

bigmodel = AutoModelForImageTextToText.from_pretrained(
    bigname, device_map="auto", torch_dtype=torch.float16
)
bigmodel.eval()

# Compute layer indices via dummy forward
with torch.no_grad():
    dummy = bigmodel(
        torch.tensor([[0]]).cuda(),
        output_hidden_states=True,
    )
    n_layers = len(dummy.hidden_states) - 1

shallow_idx = max(1, n_layers // 4 + 1)
middle_idx = n_layers // 2 + 1
deep_idx = -1

print(f"[SpareSpec Stage2] Model: {bigname}, n_layers={n_layers}")
print(f"  shallow_idx={shallow_idx}, middle_idx={middle_idx}, deep_idx={deep_idx}")


def _extract_recent_vis_attn_scores(
    generate_outputs, target_len, layer_idx, image_mask, query_window, query_token_mask
):
    attentions = getattr(generate_outputs, "attentions", None)
    if attentions is None:
        return None

    image_mask = image_mask[:target_len].to(torch.bool).cpu()
    vis_ids = torch.where(image_mask)[0]
    query_token_mask = query_token_mask[:target_len].to(torch.bool).cpu()
    text_ids = torch.where(query_token_mask & ~image_mask)[0]
    if vis_ids.numel() == 0 or text_ids.numel() == 0:
        return None

    query_window = min(query_window, text_ids.numel())
    query_ids = text_ids[-query_window:]
    query_set = {int(idx): row for row, idx in enumerate(query_ids.tolist())}
    score_sum = torch.zeros(vis_ids.numel(), dtype=torch.float32)
    score_count = 0

    row_start = 0
    for step_attn in attentions:
        if step_attn is None:
            continue
        if isinstance(step_attn, (tuple, list)):
            if len(step_attn) == 0:
                continue
            attn_idx = layer_idx if layer_idx >= 0 else len(step_attn) + layer_idx
            attn_idx = min(max(attn_idx, 0), len(step_attn) - 1)
            step_attn = step_attn[attn_idx]
        if isinstance(step_attn, (tuple, list)):
            if len(step_attn) == 0:
                continue
            step_attn = step_attn[0]
        if not torch.is_tensor(step_attn) or step_attn.dim() < 4:
            continue

        attn = step_attn[0].float().mean(dim=0).cpu()
        q_len = min(attn.shape[-2], target_len - row_start)
        if q_len <= 0:
            break
        valid_vis = vis_ids[vis_ids < min(attn.shape[-1], target_len)]
        if valid_vis.numel() == 0:
            row_start += q_len
            continue
        valid_vis_order = torch.searchsorted(vis_ids, valid_vis)
        for local_row in range(q_len):
            global_row = row_start + local_row
            if global_row not in query_set:
                continue
            score_sum[valid_vis_order] += attn[local_row, valid_vis]
            score_count += 1
        row_start += q_len

    if score_count == 0:
        return None

    vis_attn_scores = torch.zeros(target_len, dtype=torch.float16)
    vis_attn_scores[vis_ids] = (score_sum / score_count).to(torch.float16)
    return vis_attn_scores


def _find_subsequence(sequence, pattern):
    if len(pattern) == 0 or len(pattern) > len(sequence):
        return None
    last_start = len(sequence) - len(pattern)
    for start in range(last_start + 1):
        if sequence[start : start + len(pattern)] == pattern:
            return start, start + len(pattern)
    return None


def _build_query_token_mask(input_ids, query_text, target_len, prompt_len, image_mask):
    query_token_mask = torch.zeros(target_len, dtype=torch.bool)
    tokenizer = processor.tokenizer
    query_ids = tokenizer(query_text, add_special_tokens=False).input_ids
    if query_ids:
        prompt_ids = input_ids[0, :prompt_len].tolist()
        span = _find_subsequence(prompt_ids, query_ids)
        if span is None:
            for prefix in (" ", "\n"):
                query_ids = tokenizer(prefix + query_text, add_special_tokens=False).input_ids
                span = _find_subsequence(prompt_ids, query_ids)
                if span is not None:
                    break
        if span is not None:
            start, end = span
            query_token_mask[start : min(end, target_len)] = True

    if not query_token_mask.any():
        prompt_len = max(0, min(int(prompt_len), target_len))
        query_token_mask[:prompt_len] = True

    query_token_mask &= ~image_mask[:target_len].to(torch.bool).cpu()
    return query_token_mask


@torch.no_grad()
def extract_vis_anchor_llava(pixel_values):
    """Project the ViT layer -2 CLS token into the LLM hidden space."""
    vision_tower = bigmodel.vision_tower
    projector = bigmodel.multi_modal_projector
    if pixel_values.dim() == 5:
        pixel_values = pixel_values.reshape(-1, *pixel_values.shape[-3:])
    vision_param = next(vision_tower.parameters())
    pixel_values = pixel_values.to(
        device=vision_param.device, dtype=vision_param.dtype
    )
    vision_outputs = vision_tower(
        pixel_values, output_hidden_states=True, return_dict=True
    )
    cls_token = vision_outputs.hidden_states[-2][:, 0]
    projector_param = next(projector.parameters())
    cls_token = cls_token.to(
        device=projector_param.device, dtype=projector_param.dtype
    )
    return projector(cls_token).mean(dim=0, keepdim=True)

@torch.no_grad()
def ge(data: Dict):
    images = [Image.open(f) for f in data["image_files"]]
    inputs = processor(images=images, text=data["text"], return_tensors="pt").to(bigmodel.device)

    # Extract vis_anchor before generation
    vis_anchor = None
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        vis_anchor = extract_vis_anchor_llava(inputs["pixel_values"])

    # Generate with output_hidden_states
    outs_big = bigmodel.generate(
        **inputs,
        output_hidden_states=True,
        output_attentions=args.save_attentions,
        return_dict_in_generate=True,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature != 0,
        temperature=args.temperature if args.temperature != 0 else 1.0,
    )

    # Multi-level hidden states: cat(shallow, middle, deep) per step
    # outs_big.hidden_states: tuple of tuples
    #   outer tuple: [step_0, step_1, ...]  (prefill + each generation step)
    #   inner tuple: [emb, layer_0, ..., layer_N-1]
    hidden_state_list = [
        torch.cat([x[shallow_idx], x[middle_idx], x[deep_idx]], dim=-1)
        for x in outs_big.hidden_states
    ]
    hidden_state_big = torch.cat(hidden_state_list, dim=1)  # [1, total_seq, 3*H]

    # inputs_embeds: embedding from each step
    inputs_embeds_list = [x[0] for x in outs_big.hidden_states]
    inputs_embeds_big = torch.cat(inputs_embeds_list, dim=1)

    # image_mask over the full generated sequence
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    image_mask = (outs_big.sequences == image_token_id)[..., :-1]  # [1, total_seq-1]

    # loss_mask: only on assistant response (generated part)
    input_len = inputs["input_ids"].shape[-1]
    loss_mask = torch.ones(outs_big.sequences[:, :-1].shape, dtype=bool)
    loss_mask[:, : input_len - 1] = 0
    query_token_mask = _build_query_token_mask(
        inputs["input_ids"],
        data["query_text"],
        hidden_state_big.shape[1],
        input_len - 1,
        image_mask[0],
    )

    td = {
        "inputs_embeds": inputs_embeds_big.cpu()[0],         # [total_seq, H]
        "hidden_state": hidden_state_big.cpu()[0],           # [total_seq, 3*H]
        "loss_mask": loss_mask.cpu()[0],                     # [total_seq]
        "image_mask": image_mask.cpu()[0],                   # [total_seq]
        "query_token_mask": query_token_mask,                # [total_seq]
    }
    if vis_anchor is not None:
        td["vis_anchor"] = vis_anchor.cpu()                  # [1, H]
    if args.save_attentions:
        vis_attn_scores = _extract_recent_vis_attn_scores(
            outs_big,
            td["hidden_state"].shape[0],
            middle_idx - 1,
            td["image_mask"],
            args.vis_query_window,
            td["query_token_mask"],
        )
        if vis_attn_scores is not None:
            td["vis_attn_scores"] = vis_attn_scores

    return td


outdir = f"{args.outdir}/{args.index}"
if not os.path.exists(outdir):
    os.makedirs(outdir)


def writedata(name, data_point, idx):
    if not os.path.exists(name):
        os.makedirs(name)
    final_path = os.path.join(name, f"data_{idx}.ckpt")
    tmp_path = f"{final_path}.tmp.{os.getpid()}"
    try:
        torch.save(data_point, tmp_path)
        os.replace(tmp_path, final_path)
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


skipped = 0
progress = tqdm(ds)

for i, data in enumerate(progress):
    final_path = os.path.join(outdir, f"data_{i}.ckpt")

    if os.path.isfile(final_path):
        skipped += 1
        if skipped % 100 == 0:
            progress.set_postfix(skipped=skipped)
        continue

    outdata = ge(data)
    writedata(outdir, outdata, i)
