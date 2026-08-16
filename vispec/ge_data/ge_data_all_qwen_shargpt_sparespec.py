"""
SpareSpec text-only data generation (Stage 1).
Saves multi-level (shallow/middle/deep) hidden states from the target model.
"""
import argparse

parser = argparse.ArgumentParser(description="SpareSpec Stage1 data generation")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=100)
parser.add_argument("--index", type=int, default=1)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])
parser.add_argument("--outdir", type=str, default="outdir0")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--data-path", type=str, default=None)
args = parser.parse_args()
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_index))

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoTokenizer

modelname = args.model


def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))
    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break
    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def _load_sharegpt_dataset(path=None):
    if path is None:
        ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered")
        return ds["train"]

    if os.path.isdir(path):
        candidates = [
            "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
            "ShareGPT_V3_unfiltered_cleaned_split.json",
        ]
        for filename in candidates:
            candidate = os.path.join(path, filename)
            if os.path.exists(candidate):
                path = candidate
                break

    if os.path.isfile(path) and path.endswith(".json"):
        with open(path) as f:
            return Dataset.from_list(json.load(f))

    ds = load_dataset(path)
    if isinstance(ds, dict):
        return ds["train"]
    return ds


def build_dataset_rank(tokenizer, split="train", select=None):
    ds = _load_sharegpt_dataset(args.data_path)
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    ds1 = ds1.add_column("source_index", list(range(args.start, args.end)))
    original_columns1 = ds1.column_names
    num_proc = 2

    def preprocess_function(examples):
        new_examples = {
            "conversation": [], "input_ids": [], "loss_mask": [], "source_index": []
        }
        convroles = ["user", "assistant"]
        roles = {"human": "user", "gpt": "assistant"}
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.unk_token_id

        for row_idx in range(len(examples["conversations"])):
            source = examples["conversations"][row_idx]
            if not source:
                continue

            first_role = roles.get(source[0].get("from"))
            if first_role != "user":
                source = source[1:]
            if not source:
                continue

            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            valid = True
            for turn_idx, sentence in enumerate(source):
                role = roles.get(sentence.get("from"))
                value = sentence.get("value")
                if role is None or value is None or role != convroles[turn_idx % 2]:
                    valid = False
                    break
                messages.append({"role": role, "content": value})
            if not valid or len(messages) <= 1:
                continue

            full_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
            )
            if torch.is_tensor(full_ids):
                full_ids = full_ids.reshape(-1).tolist()
            elif full_ids and isinstance(full_ids[0], list):
                full_ids = full_ids[0]
            input_ids = torch.tensor(full_ids[:2048], dtype=torch.long)
            if input_ids.numel() == 0:
                continue

            # Derive assistant spans from the tokenizer's own chat template.  This
            # remains correct if Qwen special-token lengths or separators change.
            loss_mask = torch.zeros_like(input_ids)
            spans_valid = True
            for message_idx, message in enumerate(messages):
                if message["role"] != "assistant":
                    continue
                before_ids = tokenizer.apply_chat_template(
                    messages[:message_idx], tokenize=True, add_generation_prompt=True,
                )
                after_ids = tokenizer.apply_chat_template(
                    messages[: message_idx + 1], tokenize=True, add_generation_prompt=False,
                )
                if torch.is_tensor(before_ids):
                    before_ids = before_ids.reshape(-1).tolist()
                if torch.is_tensor(after_ids):
                    after_ids = after_ids.reshape(-1).tolist()
                if before_ids and isinstance(before_ids[0], list):
                    before_ids = before_ids[0]
                if after_ids and isinstance(after_ids[0], list):
                    after_ids = after_ids[0]
                if after_ids[: len(before_ids)] != before_ids or full_ids[: len(after_ids)] != after_ids:
                    spans_valid = False
                    break
                start = min(len(before_ids), input_ids.numel())
                end = min(len(after_ids), input_ids.numel())
                loss_mask[start:end] = 1
            if not spans_valid:
                continue
            if loss_mask.sum().item() == 0:
                continue

            conversation = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["source_index"].append(examples["source_index"][row_idx])
        return new_examples

    ds1 = ds1.map(
        preprocess_function, batched=True, remove_columns=original_columns1,
        load_from_cache_file=False,
    )
    ds1.set_format(type="torch")
    return ds1


tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=False)
dataset = build_dataset_rank(tokenizer)

model = AutoModelForImageTextToText.from_pretrained(
    modelname, device_map="auto", torch_dtype=torch.bfloat16
)
model.eval()

# Compute layer indices once
# hidden_states tuple: [emb, layer_0, layer_1, ..., layer_N-1]
# Do a dummy forward to get n_layers
with torch.no_grad():
    dummy = model(torch.tensor([[0]]).cuda(), output_hidden_states=True)
    n_layers = len(dummy.hidden_states) - 1

shallow_idx = max(1, n_layers // 4 + 1)
middle_idx = n_layers // 2 + 1
deep_idx = -1

print(f"[SpareSpec Stage1] Model: {modelname}, n_layers={n_layers}")
print(f"  shallow_idx={shallow_idx}, middle_idx={middle_idx}, deep_idx={deep_idx}")


@torch.no_grad()
def ge(data):
    input_ids = data["input_ids"]
    outs_big = model(input_ids.cuda(), output_hidden_states=True)
    all_hidden = outs_big.hidden_states

    inputs_embeds_big = all_hidden[0]  # embedding output
    hidden_state_big = torch.cat(
        [all_hidden[shallow_idx], all_hidden[middle_idx], all_hidden[deep_idx]],
        dim=-1,
    )  # [Seq, 3*H]

    td = {
        "inputs_embeds": inputs_embeds_big.cpu()[0],
        "input_ids": input_ids.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": data["loss_mask"].cpu()[0],
        "source_index": int(data["source_index"]),
        "data_format_version": 2,
    }
    return td


outdir = args.outdir
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


for data in tqdm(dataset):
    outdata = ge(data)
    writedata(outdir, outdata, outdata["source_index"])
