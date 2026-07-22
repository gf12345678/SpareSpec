import argparse

parser = argparse.ArgumentParser(description="sp")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=100)
parser.add_argument("--index", type=int, default=1)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])
parser.add_argument("--outdir", type=str, default="outdir0")
parser.add_argument("--model", type=str, default="llava-hf/llava-v1.6-vicuna-7b-hf")
parser.add_argument("--data-path", type=str, default=None)
args = parser.parse_args()
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]


import torch
from datasets import Dataset, load_dataset
from fastchat.model.model_adapter import get_conversation_template
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
)

bigname = args.model


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


def build_dataset_rank(
    tokenizer,
    split="train",
    select=None,
):
    if args.data_path:
        data_path = args.data_path
        if os.path.isdir(data_path):
            for filename in (
                "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
                "ShareGPT_V3_unfiltered_cleaned_split.json",
            ):
                candidate = os.path.join(data_path, filename)
                if os.path.exists(candidate):
                    data_path = candidate
                    break
        if os.path.isfile(data_path):
            with open(data_path) as f:
                ds = Dataset.from_list(json.load(f))
        else:
            loaded = load_dataset(data_path)
            ds = loaded["train"] if isinstance(loaded, dict) else loaded
    else:
        ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered")["train"]
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names
    # original_columns2 = ds2.column_names
    num_proc = 1

    def preprocess_function(examples):
        new_examples = {"conversation": [], "input_ids": [], "loss_mask": []}
        for row_idx in range(len(examples["id"])):
            conv = get_conversation_template("vicuna")
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            source = examples["conversations"][row_idx]
            if not source:
                continue

            first_message = source[0]
            first_role = (
                roles.get(first_message.get("from"))
                if isinstance(first_message, dict)
                else None
            )
            if first_role != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            if not source:
                continue

            conv.messages = []
            valid = True
            for j, sentence in enumerate(source):
                if not isinstance(sentence, dict):
                    valid = False
                    break
                role = roles.get(sentence.get("from"))
                value = sentence.get("value")
                if role is None or value is None or role != conv.roles[j % 2]:
                    valid = False
                    break
                conv.append_message(role, value)
            if not valid or not conv.messages:
                continue

            conversation = conv.get_prompt()
            # if i==56:
            #     print(i)
            # if i==57:
            #     print(i)
            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids[0]
            loss_mask = torch.ones_like(input_ids)
            # print(i)

            sep = conv.sep + conv.roles[1] + ": "

            total_len = int(input_ids.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2)
            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    instruction_len -= 1

                # Ignore the user instructions
                loss_mask[cur_len : cur_len + instruction_len] = 0
                cur_len += turn_len

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    cur_len -= 1

            loss_mask[cur_len:] = 0

            if input_ids.numel() == 0 or loss_mask.sum().item() == 0:
                continue

            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        # num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False,
    )

    ds1.set_format(type="torch")
    return ds1


bigtokenizer = AutoTokenizer.from_pretrained(bigname, use_fast=False)
ds = build_dataset_rank(bigtokenizer)
print(ds)
# bigmodel = AutoModelForCausalLM.from_pretrained(bigname,  device_map="auto",torch_dtype=torch.float16)
bigmodel = AutoModelForImageTextToText.from_pretrained(
    bigname, device_map="auto", torch_dtype=torch.float16
)
bigmodel.eval()

with torch.no_grad():
    dummy = bigmodel(torch.tensor([[0]]).cuda(), output_hidden_states=True)
    n_layers = len(dummy.hidden_states) - 1
shallow_idx = max(1, n_layers // 4 + 1)
middle_idx = n_layers // 2 + 1
deep_idx = -1
print(f"[SpareSpec Stage1] Model: {bigname}, n_layers={n_layers}")
print(f"  shallow_idx={shallow_idx}, middle_idx={middle_idx}, deep_idx={deep_idx}")


@torch.no_grad()
def ge(data):
    input_ids = data["input_ids"]
    outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True)
    inputs_embeds_big = outs_big.hidden_states[0]
    hidden_state_big = torch.cat(
        [
            outs_big.hidden_states[shallow_idx],
            outs_big.hidden_states[middle_idx],
            outs_big.hidden_states[deep_idx],
        ],
        dim=-1,
    )
    max_prob_tokens_big = torch.argmax(outs_big.logits, dim=-1)
    probs = torch.softmax(outs_big.logits, dim=-1)
    maxp = probs[0].max(dim=1).values
    td = {
        "inputs_embeds": inputs_embeds_big.cpu()[0],
        "input_ids": input_ids.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": data["loss_mask"].cpu()[0],
    }
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


for i, data in enumerate(tqdm(ds)):
    outdata = ge(data)
    writedata(outdir, outdata, i)
