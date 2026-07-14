import argparse

parser = argparse.ArgumentParser(description="SpareSpec Training")
parser.add_argument("--basepath", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--configpath", type=str, default="config.json")
parser.add_argument("--loadpath", type=str, default=None)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--bs", type=int, default=4)
parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
parser.add_argument("--tmpdir", type=str, default="0")
parser.add_argument("--cpdir", type=str, default="0")
parser.add_argument("--pw", type=float, default=0.1)
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--max-len", type=int, default=3200)
parser.add_argument("--num-hidden-levels", type=int, default=3)
parser.add_argument("--vis-select-tokens", type=int, default=64)
parser.add_argument("--min-vis-select-tokens", type=int, default=16)
parser.add_argument("--vis-entropy-alpha", type=float, default=1.2)
parser.add_argument("--vis-query-window", type=int, default=8)
parser.add_argument("--max-total-vis-select-tokens", type=int, default=0)
parser.add_argument("--stage", type=int, default=1, help="1=text-only, 2=multimodal")
parser.add_argument("--begin-epoch", type=int, default=0)
parser.add_argument("--num-epochs", type=int, default=20)
parser.add_argument("--max-train-steps", type=int, default=0)
parser.add_argument("--max-val-batches", type=int, default=0)
parser.add_argument("--kacc-batches", type=int, default=10)
args = parser.parse_args()
if args.stage == 2 and args.bs != 1:
    print("[SpareSpec] stage 2 currently supports batch size 1; overriding --bs to 1.")
    args.bs = 1

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": args.num_epochs,
    "p_w": args.pw,
    "v_w": 1.0,
    "head_w": 0.1,
    "num_workers": args.num_workers,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": args.max_len,
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 5,
}
import json
import os

try:
    from torch_npu.contrib import transfer_to_npu
except:
    pass

import torch
from safetensors import safe_open

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(
    gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
)
from typing import Any, Dict, List

import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    get_linear_schedule_with_warmup,
)

from ..model.cnets_sparespec import Model
from ..model.configs import EConfig

class _NoOpWandb:
    def init(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None


class _NoOpWriter:
    def add_scalars(self, *args, **kwargs):
        return None

    def close(self):
        return None


if accelerator.is_main_process:
    try:
        import wandb

        wandb.init(
            project="sparespec", entity="yuhui-li", mode="offline", config=train_config
        )
    except ImportError:
        print("[SpareSpec] wandb is not installed; logging to wandb is disabled.")
        wandb = _NoOpWandb()

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=f"{args.cpdir}/run")
    except ImportError:
        print("[SpareSpec] tensorboard is not installed; tensorboard logging is disabled.")
        writer = _NoOpWriter()

try:
    baseconfig = AutoConfig.from_pretrained(args.basepath)
    try:
        hidden_size = baseconfig.hidden_size
        head = torch.nn.Linear(
            baseconfig.hidden_size, baseconfig.vocab_size, bias=False
        )
    except:
        hidden_size = baseconfig.text_config.hidden_size
        head = torch.nn.Linear(
            baseconfig.text_config.hidden_size,
            baseconfig.text_config.vocab_size,
            bias=False,
        )
    train_config["hidden_size"] = hidden_size

    try:
        try:
            with open(
                os.path.join(args.basepath, "model.safetensors.index.json"), "r"
            ) as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"]["lm_head.weight"]
            with safe_open(
                os.path.join(args.basepath, head_path), framework="pt", device="cpu"
            ) as f:
                tensor_slice = f.get_slice("lm_head.weight")
                vocab_size, hidden_dim = tensor_slice.get_shape()
                tensor = tensor_slice[:, :hidden_dim].float()
        except:
            with open(
                os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r"
            ) as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"]["lm_head.weight"]
            weights = torch.load(os.path.join(args.basepath, head_path))
            tensor = weights["lm_head.weight"].float()
    except:
        m = AutoModelForImageTextToText.from_pretrained(
            args.basepath, torch_dtype="auto"
        )
        try:
            tensor = m.language_model.lm_head.weight.float()
        except:
            tensor = m.lm_head.weight.float()
        del m
except:
    tensor = torch.load(args.basepath)["lm_head.weight"].float()
    head = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)

head.weight.data = tensor
head.eval()

for param in head.parameters():
    param.requires_grad = False


def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            if not file.endswith(".ckpt"):
                continue
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return sorted(datapath)


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform
        self.hidden_size = train_config.get("hidden_size", 3584)
        self.num_hidden_levels = args.num_hidden_levels
        self.stage = args.stage

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data["hidden_state"][: train_config["max_len"]][None, :]
        inputs_embeds = data["inputs_embeds"][: train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][: train_config["max_len"]][None, :]

        length = hidden_state.shape[1]
        attention_mask = [1] * length
        raw_loss_mask = loss_mask[0].to(torch.bool)
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        # target: next-step deep-level hidden state [Seq, H]
        # hidden_state is multi-level [Seq, num_levels * H], target is [H] only
        target = hidden_state[:, 1:, -self.hidden_size:]
        zeropadding = torch.zeros(1, 1, self.hidden_size)
        target = torch.cat((target, zeropadding), dim=1)

        inputs_embeds_target = inputs_embeds[:, 1:]
        zeropadding = torch.zeros_like(inputs_embeds_target[:, 0, ...]).unsqueeze(1)
        inputs_embeds_target = torch.cat((inputs_embeds_target, zeropadding), dim=1)
        loss_mask[-1] = 0

        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["inputs_embeds"] = inputs_embeds_target

        # Stage 2: load vis_anchor, image_mask, and compact text-to-vision scores.
        if self.stage == 2:
            if "vis_anchor" in data:
                new_data["vis_anchor"] = data["vis_anchor"]

            img_msk = None
            if "image_mask" in data:
                img_msk = data["image_mask"][: train_config["max_len"]]
                if len(img_msk) < length:
                    pad_len = length - len(img_msk)
                    img_msk = torch.cat(
                        [img_msk, torch.zeros(pad_len, dtype=img_msk.dtype)]
                    )
                img_msk = img_msk[:length].to(torch.bool)
                new_data["image_mask"] = img_msk

            query_msk = None
            if "query_token_mask" in data:
                query_msk = data["query_token_mask"][: train_config["max_len"]]
                if len(query_msk) < length:
                    pad_len = length - len(query_msk)
                    query_msk = torch.cat(
                        [query_msk, torch.zeros(pad_len, dtype=query_msk.dtype)]
                    )
                query_msk = query_msk[:length].to(torch.bool)
                new_data["query_token_mask"] = query_msk

            vis_attn_scores = data.get("vis_attn_scores")
            if vis_attn_scores is not None:
                vis_attn_scores = vis_attn_scores.squeeze()[: train_config["max_len"]]
                if vis_attn_scores.numel() < length:
                    pad = vis_attn_scores.new_zeros(length - vis_attn_scores.numel())
                    vis_attn_scores = torch.cat((vis_attn_scores, pad), dim=0)
                new_data["vis_attn_scores"] = vis_attn_scores[:length][None, :].float()
            else:
                # Backward compatibility for old ckpts that stored full [seq, seq]
                # attention maps. Convert once here to the compact score vector the
                # selector actually uses.
                text_attn_vis = data.get("text_attn_vis")
                if text_attn_vis is None:
                    text_attn_vis = data.get("attentions")
                if text_attn_vis is not None and img_msk is not None:
                    if text_attn_vis.dim() == 2:
                        text_attn_vis = text_attn_vis[None, ...]
                    elif text_attn_vis.dim() == 3:
                        text_attn_vis = text_attn_vis[
                            :, : train_config["max_len"], : train_config["max_len"]
                        ]
                        if text_attn_vis.shape[0] != 1:
                            text_attn_vis = text_attn_vis.mean(dim=0, keepdim=True)
                    elif text_attn_vis.dim() == 4:
                        text_attn_vis = text_attn_vis[
                            :, :, : train_config["max_len"], : train_config["max_len"]
                        ].mean(dim=1)
                        if text_attn_vis.shape[0] != 1:
                            text_attn_vis = text_attn_vis.mean(dim=0, keepdim=True)
                    else:
                        raise ValueError(
                            "text_attn_vis should have shape [seq, seq], [heads, seq, seq], or [batch, heads, seq, seq]"
                        )
                    attn = text_attn_vis[0].float()
                    vis_ids = torch.where(img_msk)[0]
                    if query_msk is not None and query_msk.any():
                        text_ids = torch.where(query_msk & ~img_msk)[0]
                    else:
                        prompt_msk = ~raw_loss_mask[:length]
                        text_ids = torch.where(prompt_msk & ~img_msk)[0]
                    score_vec = torch.zeros(length, dtype=torch.float32)
                    if vis_ids.numel() > 0 and text_ids.numel() > 0:
                        query_window = min(args.vis_query_window, text_ids.numel())
                        query_ids = text_ids[-query_window:]
                        valid_query = query_ids[query_ids < attn.shape[-2]]
                        valid_vis = vis_ids[vis_ids < attn.shape[-1]]
                        if valid_query.numel() > 0 and valid_vis.numel() > 0:
                            score_vec[valid_vis] = attn[valid_query][:, valid_vis].mean(dim=0)
                    new_data["vis_attn_scores"] = score_vec[None, :]

        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingattention(self, intensors, N):
        B, q, k = intensors.shape
        outtensors = torch.zeros(B, N, N, dtype=intensors.dtype)
        outtensors[:, :q, :k] = intensors
        return outtensors

    def paddingvector(self, intensors, N):
        B, n = intensors.shape
        outtensors = torch.zeros(B, N, dtype=intensors.dtype)
        outtensors[:, :n] = intensors
        return outtensors

    def padding1d(self, intensors, N):
        outtensors = torch.zeros(N, dtype=intensors.dtype)
        outtensors[: intensors.shape[0]] = intensors
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["hidden_state_big"].shape[1] for item in features)
        batch_inputs_embeds = torch.cat(
            [self.paddingtensor(item["inputs_embeds"], max_length) for item in features]
        )
        batch_hidden_states = torch.cat(
            [
                self.paddingtensor(item["hidden_state_big"], max_length)
                for item in features
            ]
        )
        batch_target = torch.cat(
            [self.paddingtensor(item["target"], max_length) for item in features]
        )
        batch_loss_mask = torch.tensor(
            [
                item["loss_mask"] + [0] * (max_length - len(item["loss_mask"]))
                for item in features
            ]
        )
        batch_attention_mask = torch.tensor(
            [
                item["attention_mask"]
                + [0] * (max_length - len(item["attention_mask"]))
                for item in features
            ]
        )
        batch = {
            "inputs_embeds": batch_inputs_embeds,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        # Stage 2: pass vis_anchor and image_mask through
        if "vis_anchor" in features[0]:
            batch["vis_anchor"] = torch.stack([f["vis_anchor"] for f in features])
        if "image_mask" in features[0]:
            batch["image_mask"] = torch.stack(
                [self.padding1d(f["image_mask"], max_length) for f in features]
            )
        if "query_token_mask" in features[0]:
            batch["query_token_mask"] = torch.stack(
                [self.padding1d(f["query_token_mask"], max_length) for f in features]
            )
        if "vis_attn_scores" in features[0]:
            batch["vis_attn_scores"] = torch.cat(
                [self.paddingvector(f["vis_attn_scores"], max_length) for f in features]
            )
        if "text_attn_vis" in features[0]:
            batch["text_attn_vis"] = torch.cat(
                [self.paddingattention(f["text_attn_vis"], max_length) for f in features]
            )
        return batch


def top_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def compute_loss(target, target_p, predict, loss_mask):
    loss_mask = loss_mask.to(bool)
    out_head = head(predict)
    out_logp = nn.LogSoftmax(dim=-1)(out_head[loss_mask[..., 0]])
    if out_logp.numel() == 0:
        return out_logp.sum(), out_logp.sum(), out_head
    target_p = target_p[loss_mask[..., 0]]
    plogp = target_p * out_logp
    ploss = -torch.mean(plogp.sum(-1))
    vloss = criterion(predict[loss_mask[..., 0]], target[loss_mask[..., 0]])
    vloss = torch.mean(vloss.mean(-1))

    _, topk_indices = torch.topk(target_p, k=10, dim=-1)
    student_topk_logits = out_head[loss_mask[..., 0]].gather(-1, topk_indices)
    reversed_logits = torch.flip(student_topk_logits, dims=[-1])
    log_cumsum_exp = torch.logcumsumexp(reversed_logits, dim=-1)
    log_denominator = torch.flip(log_cumsum_exp, dims=[-1])
    log_likelihood = student_topk_logits - log_denominator
    rloss = -torch.mean(log_likelihood.sum(-1))

    return vloss, ploss + 0.1 * rloss, out_head


@torch.no_grad()
def getkacc(model, data, head, max_length=5):
    def generate(
        hidden_states,
        inputs_embeds,
        head,
        max_length=4,
        use_cache=True,
        image_mask=None,
        text_attn_vis=None,
        vis_attn_scores=None,
        query_token_mask=None,
        vis_anchor=None,
    ):
        output_ids = []
        if use_cache:
            past_key_values = None
            for i in range(max_length):
                if past_key_values != None:
                    model_kwargs = {
                        "input_ids": token,
                        "past_key_values": past_key_values,
                        "use_cache": True,
                    }
                    if vis_anchor is not None:
                        model_kwargs["vis_anchor"] = vis_anchor
                    out_hidden, past_key_values = model(last_hidden, **model_kwargs)
                else:
                    model_kwargs = {
                        "inputs_embeds": inputs_embeds,
                        "use_cache": True,
                    }
                    if image_mask is not None:
                        model_kwargs["image_mask"] = image_mask
                    if text_attn_vis is not None:
                        model_kwargs["text_attn_vis"] = text_attn_vis
                    if vis_attn_scores is not None:
                        model_kwargs["vis_attn_scores"] = vis_attn_scores
                    if query_token_mask is not None:
                        model_kwargs["query_token_mask"] = query_token_mask
                    if vis_anchor is not None:
                        model_kwargs["vis_anchor"] = vis_anchor
                    out_hidden, past_key_values = model(hidden_states, **model_kwargs)
                last_hidden = out_hidden[:, -1:]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout, dim=-1)
                output_ids.append(token)

        else:
            raise NotImplementedError

        return torch.cat(output_ids, dim=1)

    hidden_states = data["hidden_states"]
    inputs_embeds = data["inputs_embeds"]
    loss_mask = data["loss_mask"]
    target = data["target"]
    total = [0 for _ in range(max_length)]
    correct = [0 for _ in range(max_length)]
    bs, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = head(target)
    target_ids = target_headout.argmax(dim=2)

    for pre_len in range(1, seq_len):
        if loss_mask[:, pre_len].sum() == 0:
            continue
        pre_hidden_states = hidden_states[:, :pre_len]
        pre_inputs_embeds = inputs_embeds[:, :pre_len]
        pre_image_mask = data.get("image_mask")
        if pre_image_mask is not None:
            pre_image_mask = pre_image_mask[:, :pre_len]
        pre_text_attn_vis = data.get("text_attn_vis")
        if pre_text_attn_vis is not None:
            pre_text_attn_vis = pre_text_attn_vis[:, :pre_len, :pre_len]
        pre_vis_attn_scores = data.get("vis_attn_scores")
        if pre_vis_attn_scores is not None:
            pre_vis_attn_scores = pre_vis_attn_scores[:, :pre_len]
        pre_query_token_mask = data.get("query_token_mask")
        if pre_query_token_mask is not None:
            pre_query_token_mask = pre_query_token_mask[:, :pre_len]
        outs = generate(
            pre_hidden_states,
            pre_inputs_embeds,
            head,
            max_length=max_length,
            image_mask=pre_image_mask,
            text_attn_vis=pre_text_attn_vis,
            vis_attn_scores=pre_vis_attn_scores,
            query_token_mask=pre_query_token_mask,
            vis_anchor=data.get("vis_anchor"),
        )
        generate_ids = outs
        for bid in range(bs):
            for k in range(max_length):
                if loss_mask[bid, pre_len + k] == 0:
                    break
                if pre_len + k >= seq_len:
                    break
                total[k] += 1
                if generate_ids[bid, k] == target_ids[bid, pre_len + k - 1]:
                    correct[k] += 1
                else:
                    for kk in range(k + 1, max_length):
                        total[kk] += 1
                    break

    acc = [correct[i] / total[i] if total[i] != 0 else 0 for i in range(len(correct))]
    return acc


if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

datapath = list_files(train_config["datapath"])
if len(datapath) == 0:
    raise ValueError(f"No .ckpt files found in {train_config['datapath']}")

split_idx = int(len(datapath) * 0.95)
if len(datapath) == 1:
    split_idx = 1
else:
    split_idx = min(max(split_idx, 1), len(datapath) - 1)
traindatapath = datapath[:split_idx]
testdatapath = datapath[split_idx:] or datapath[:1]

traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)
train_loader = DataLoader(
    traindataset,
    batch_size=train_config["bs"],
    shuffle=True,
    collate_fn=DataCollatorWithPadding(),
    num_workers=train_config["num_workers"],
    pin_memory=True,
)
test_loader = DataLoader(
    testdataset,
    batch_size=train_config["bs"],
    shuffle=False,
    collate_fn=DataCollatorWithPadding(),
    num_workers=train_config["num_workers"],
    pin_memory=True,
)

if not os.path.exists(args.cpdir):
    if accelerator.is_main_process:
        os.makedirs(args.cpdir)
else:
    ckpts = os.listdir(args.cpdir)
    if ckpts:
        begin_epoch = max(
            int(c.split("_")[1]) + 1 if c.startswith("state") else 0 for c in ckpts
        )
        loadpath = os.path.join(
            args.cpdir, f"state_{begin_epoch - 1}", "model.safetensors"
        )
        if os.path.exists(loadpath):
            print(f"resume from {loadpath}")
            args.loadpath = loadpath
            args.begin_epoch = begin_epoch


config = EConfig.from_pretrained(train_config["config_path"])
model = Model(
    config,
    load_emb=True,
    path=args.basepath,
    num_hidden_levels=args.num_hidden_levels,
    vis_select_tokens=args.vis_select_tokens,
    min_vis_select_tokens=args.min_vis_select_tokens,
    vis_entropy_alpha=args.vis_entropy_alpha,
    vis_query_window=args.vis_query_window,
    max_total_vis_select_tokens=args.max_total_vis_select_tokens,
)

if args.loadpath:
    with open(args.loadpath, "rb") as f:
        from safetensors.torch import load

        state_dict = load(f.read())
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(missing_keys) > 0:
            print(f"missing_keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"unexpected_keys: {unexpected_keys}")


criterion = nn.SmoothL1Loss(reduction="none")
optimizer = optim.AdamW(
    model.parameters(),
    lr=train_config["lr"],
    betas=(train_config["b1"], train_config["b2"]),
)

num_epochs = train_config["num_epochs"]
num_warmup_steps = max(len(train_loader) * 1, 1)
total_steps = max(len(train_loader) * num_epochs, 1)
is_warmup = train_config["is_warmup"]

if is_warmup:
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader, scheduler
    )
else:
    model, head, optimizer, train_loader, test_loader = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader
    )

if is_warmup:
    for i in range(args.begin_epoch * len(train_loader)):
        scheduler.step()

global_train_steps = 0

for epoch in range(args.begin_epoch, num_epochs):
    if args.max_train_steps > 0 and global_train_steps >= args.max_train_steps:
        break
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    epoch_vloss = 0
    epoch_ploss = 0
    num_batches = 0
    model.train()
    for batch_idx, data in enumerate(
        tqdm(train_loader, disable=not accelerator.is_local_main_process)
    ):

        with accelerator.accumulate(model):
            optimizer.zero_grad()
            model_kwargs = {
                "inputs_embeds": data["inputs_embeds"],
                "attention_mask": data["attention_mask"],
            }
            if args.stage == 2:
                model_kwargs["image_mask"] = data.get("image_mask")
                model_kwargs["vis_anchor"] = data.get("vis_anchor")
                model_kwargs["text_attn_vis"] = data.get("text_attn_vis")
                model_kwargs["vis_attn_scores"] = data.get("vis_attn_scores")
                model_kwargs["query_token_mask"] = data.get("query_token_mask")

            predict = model(data["hidden_states"], **model_kwargs)
            with torch.no_grad():
                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
            loss_mask = data["loss_mask"][:, :, None]
            vloss, ploss, out_head = compute_loss(
                data["target"], target_p, predict, loss_mask
            )
            loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_value_(
                    model.parameters(), train_config["grad_clip"]
                )
            optimizer.step()
            if is_warmup:
                scheduler.step()

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if accelerator.is_main_process and ct != 0:
            logdict = {
                "train/lr": optimizer.optimizer.param_groups[0]["lr"],
                "train/vloss": vloss.item(),
                "train/ploss": ploss.item(),
                "train/loss": loss.item(),
                "train/acc": cc / ct,
            }
            for id, i in enumerate(top_3acc):
                logdict[f"train/top_{id + 1}_acc"] = topkacc[id].item() / ct
            wandb.log(logdict)
            writer.add_scalars("train", logdict, epoch * len(train_loader) + batch_idx)

        epoch_loss += loss.item()
        epoch_vloss += vloss.item()
        epoch_ploss += ploss.item()
        num_batches += 1
        global_train_steps += 1

        del ploss, vloss
        if args.max_train_steps > 0 and global_train_steps >= args.max_train_steps:
            break

    if num_batches == 0:
        break

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    epoch_vloss /= num_batches
    epoch_ploss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_main_process:
        for id, i in enumerate(top_3acc):
            wandb.log({f"train/epochtop_{id + 1}_acc": i.sum().item() / total})
            writer.add_scalars(
                "train_epoch",
                {f"train/epochtop_{id + 1}_acc": i.sum().item() / total},
                epoch,
            )
    if accelerator.is_main_process:
        print(
            "Epoch [{}/{}], Loss: {:.4f}, Vloss: {:.4f}, Ploss: {:.4f}".format(
                epoch + 1, num_epochs, epoch_loss, epoch_vloss, epoch_ploss
            )
        )
        print("Train Accuracy: {:.2f}%".format(100 * correct / total))
        wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})
        writer.add_scalars(
            "train_epoch",
            {"train/epochacc": correct / total, "train/epochloss": epoch_loss},
            epoch,
        )

    # Validation
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    epoch_vloss = 0
    epoch_ploss = 0
    num_batches = 0
    model.eval()

    k_acc = [[] for i in range(5)]
    for batch_idx, data in enumerate(
        tqdm(test_loader, disable=not accelerator.is_local_main_process)
    ):
        with torch.no_grad():
            if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                break
            if batch_idx < args.kacc_batches:
                acces = getkacc(model, data, head, max_length=5)
                for i in range(len(acces)):
                    k_acc[i].append(acces[i])
            model_kwargs_val = {
                "inputs_embeds": data["inputs_embeds"],
                "attention_mask": data["attention_mask"],
            }
            if args.stage == 2:
                model_kwargs_val["image_mask"] = data.get("image_mask")
                model_kwargs_val["vis_anchor"] = data.get("vis_anchor")
                model_kwargs_val["text_attn_vis"] = data.get("text_attn_vis")
                model_kwargs_val["vis_attn_scores"] = data.get("vis_attn_scores")
                model_kwargs_val["query_token_mask"] = data.get("query_token_mask")

            predict = model(data["hidden_states"], **model_kwargs_val)
            target_head = head(data["target"])
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()
            loss_mask = data["loss_mask"][:, :, None]
            vloss, ploss, out_head = compute_loss(
                data["target"], target_p, predict, loss_mask
            )
            loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[
                loss_mask.reshape(-1) == 1
            ]
            target = target.reshape(-1)[loss_mask.reshape(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        epoch_loss += loss.item()
        epoch_vloss += vloss.item()
        epoch_ploss += ploss.item()
        num_batches += 1

    if num_batches == 0:
        continue

    mean_acces = []
    for id, i in enumerate(k_acc):
        if len(i) == 0:
            mean_acc = torch.tensor(0.0).cuda()
        else:
            mean_acc = np.array(i).mean()
            mean_acc = torch.tensor(mean_acc).cuda()
        mean_acces.append(mean_acc)

    mean_acces = accelerator.gather_for_metrics(mean_acces)
    if accelerator.is_main_process:
        for id, i in enumerate(mean_acces):
            mean_acc = i.mean().item()
            wandb.log({f"test/{id}_acc": mean_acc})
            writer.add_scalars("test", {f"test/{id}_acc": mean_acc}, epoch)

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_main_process:
        for id, i in enumerate(top_3acc):
            wandb.log({f"test/top_{id + 1}_acc": i.sum().item() / total})
            writer.add_scalars(
                "test", {f"test/top_{id + 1}_acc": i.sum().item() / total}, epoch
            )
    epoch_loss /= num_batches
    epoch_vloss /= num_batches
    epoch_ploss /= num_batches
    if accelerator.is_main_process:
        print(
            "Test Epoch [{}/{}], Loss: {:.4f}, Vloss: {:.4f}, Ploss: {:.4f}".format(
                epoch + 1, num_epochs, epoch_loss, epoch_vloss, epoch_ploss
            )
        )
        print("Test Accuracy: {:.2f}%".format(100 * correct / total))
        wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
        writer.add_scalars(
            "test",
            {"test/epochacc": correct / total, "test/epochloss": epoch_loss},
            epoch,
        )
        accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")
        import shutil

        shutil.copyfile(args.configpath, f"{args.cpdir}/state_{epoch}/config.json")


if accelerator.is_main_process:
    writer.close()
