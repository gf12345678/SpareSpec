import argparse
import math

parser = argparse.ArgumentParser(description="SpareSpec Training")
parser.add_argument("--basepath", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--configpath", type=str, default="config.json")
parser.add_argument("--loadpath", type=str, default=None)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--bs", type=int, default=4)
parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
parser.add_argument("--tmpdir", type=str, default="0")
parser.add_argument("--cpdir", type=str, default="0")
parser.add_argument("--pw", type=float, default=0.1, help="Deprecated alias")
parser.add_argument("--hidden-loss-weight", type=float, default=1.0)
parser.add_argument("--kd-loss-weight", type=float, default=0.1)
parser.add_argument("--ranking-loss-weight", type=float, default=0.01)
parser.add_argument("--ranking-topk", type=int, default=10)
parser.add_argument("--mtp-steps", type=int, default=None)
parser.add_argument("--mtp-loss-weight", type=float, default=0.5)
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--max-len", type=int, default=4096)
parser.add_argument("--num-hidden-levels", type=int, default=3)
parser.add_argument("--vis-select-tokens", type=int, default=64)
parser.add_argument("--min-vis-select-tokens", type=int, default=16)
parser.add_argument("--vis-entropy-alpha", type=float, default=1.2)
parser.add_argument("--vis-query-window", type=int, default=8)
parser.add_argument("--max-total-vis-select-tokens", type=int, default=0)
parser.add_argument("--stage", type=int, default=1, help="1=text-only, 2=multimodal")
parser.add_argument("--begin-epoch", type=int, default=0)
parser.add_argument("--num-epochs", type=int, default=21)
parser.add_argument("--max-train-steps", type=int, default=0)
parser.add_argument("--max-val-batches", type=int, default=0)
parser.add_argument("--kacc-batches", type=int, default=10)
args = parser.parse_args()
if args.mtp_steps is None:
    args.mtp_steps = 1 if args.stage == 2 else 0
if args.stage == 1 and args.mtp_steps != 0:
    raise ValueError("SpareSpec stage 1 requires --mtp-steps 0")
if args.stage == 2 and args.mtp_steps not in (0, 1):
    raise ValueError("SpareSpec stage 2 currently supports --mtp-steps 0 or 1")
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
    "p_w": args.kd_loss_weight,
    "v_w": args.hidden_loss_weight,
    "ranking_w": args.ranking_loss_weight,
    "ranking_topk": args.ranking_topk,
    "mtp_steps": args.mtp_steps,
    "mtp_w": args.mtp_loss_weight,
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
    except Exception as exc:
        print(f"[SpareSpec] wandb is unavailable; logging to wandb is disabled: {exc}")
        wandb = _NoOpWandb()

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=f"{args.cpdir}/run")
    except Exception as exc:
        print(f"[SpareSpec] tensorboard is unavailable; tensorboard logging is disabled: {exc}")
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


def expected_stage2_anchor_type(config):
    architectures = " ".join(getattr(config, "architectures", None) or []).lower()
    model_type = str(getattr(config, "model_type", "")).lower()
    identity = f"{architectures} {model_type} {args.basepath.lower()}"
    if "qwen2_5_vl" in identity or "qwen2.5-vl" in identity:
        return "qwen_penultimate_vit_attention_v1"
    if "llava" in identity:
        return "llava_penultimate_vit_cls_projected_v1"
    raise ValueError(
        "Stage 2 data validation does not know the anchor type for base model "
        f"{args.basepath!r} (architectures={getattr(config, 'architectures', None)!r})"
    )


stage2_anchor_type = (
    expected_stage2_anchor_type(baseconfig) if args.stage == 2 else None
)


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
        self.expected_anchor_type = stage2_anchor_type

    def _validate_stage2_data(self, data, path):
        required = {
            "inputs_embeds", "hidden_state", "loss_mask", "image_mask",
            "query_token_mask", "vis_attn_scores", "vis_anchor",
            "data_format_version", "anchor_type",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(f"Invalid Stage 2 ckpt {path}: missing fields {missing}")
        if data["data_format_version"] != 2:
            raise ValueError(
                f"Invalid Stage 2 ckpt {path}: expected data_format_version=2, "
                f"got {data['data_format_version']!r}"
            )
        if data["anchor_type"] != self.expected_anchor_type:
            raise ValueError(
                f"Invalid Stage 2 ckpt {path}: base model requires anchor_type="
                f"{self.expected_anchor_type!r}, got {data['anchor_type']!r}"
            )

        seq_len = data["hidden_state"].shape[0]
        sequence_fields = (
            "inputs_embeds", "loss_mask", "image_mask",
            "query_token_mask", "vis_attn_scores",
        )
        bad_lengths = {}
        for name in sequence_fields:
            value = data[name]
            if not torch.is_tensor(value) or value.dim() == 0:
                bad_lengths[name] = f"non-sequence value {type(value).__name__}"
            elif value.shape[0] != seq_len:
                bad_lengths[name] = value.shape[0]
        if bad_lengths:
            raise ValueError(
                f"Invalid Stage 2 ckpt {path}: expected sequence length {seq_len}, "
                f"got {bad_lengths}"
            )
        expected_multi_hidden = self.num_hidden_levels * self.hidden_size
        if data["hidden_state"].dim() != 2 or data["hidden_state"].shape[1] != expected_multi_hidden:
            raise ValueError(
                f"Invalid Stage 2 ckpt {path}: hidden_state must be [seq, "
                f"{expected_multi_hidden}], got {tuple(data['hidden_state'].shape)}"
            )
        if data["inputs_embeds"].dim() != 2 or data["inputs_embeds"].shape[1] != self.hidden_size:
            raise ValueError(
                f"Invalid Stage 2 ckpt {path}: inputs_embeds must be [seq, "
                f"{self.hidden_size}], got {tuple(data['inputs_embeds'].shape)}"
            )
        anchor = data["vis_anchor"]
        if not torch.is_tensor(anchor) or tuple(anchor.shape) != (1, self.hidden_size):
            raise ValueError(
                f"Invalid Stage 2 ckpt {path}: vis_anchor must have shape "
                f"(1, {self.hidden_size}), got {getattr(anchor, 'shape', None)}"
            )
        image_mask = data["image_mask"].to(torch.bool)
        query_mask = data["query_token_mask"].to(torch.bool)
        scores = data["vis_attn_scores"].float()
        if not image_mask.any():
            raise ValueError(f"Invalid Stage 2 ckpt {path}: image_mask has no visual tokens")
        if not query_mask.any():
            raise ValueError(f"Invalid Stage 2 ckpt {path}: query_token_mask is empty")
        image_scores = scores[image_mask]
        if not torch.isfinite(image_scores).all() or image_scores.clamp_min(0).sum() <= 0:
            raise ValueError(
                f"Invalid Stage 2 ckpt {path}: visual attention scores are absent, "
                "non-finite, or all zero"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]
        data = torch.load(path)
        if self.stage == 2:
            self._validate_stage2_data(data, path)
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

    @staticmethod
    def has_consistent_field(features, name):
        present = [name in feature and feature[name] is not None for feature in features]
        if any(present) and not all(present):
            raise ValueError(f"Inconsistent optional field '{name}' within one batch")
        return all(present)

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
        if self.has_consistent_field(features, "vis_anchor"):
            batch["vis_anchor"] = torch.stack([f["vis_anchor"] for f in features])
        if self.has_consistent_field(features, "image_mask"):
            batch["image_mask"] = torch.stack(
                [self.padding1d(f["image_mask"], max_length) for f in features]
            )
        if self.has_consistent_field(features, "query_token_mask"):
            batch["query_token_mask"] = torch.stack(
                [self.padding1d(f["query_token_mask"], max_length) for f in features]
            )
        if self.has_consistent_field(features, "vis_attn_scores"):
            batch["vis_attn_scores"] = torch.cat(
                [self.paddingvector(f["vis_attn_scores"], max_length) for f in features]
            )
        if self.has_consistent_field(features, "text_attn_vis"):
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


def compute_loss(target, predict, loss_mask):
    """Compute losses only at supervised positions to avoid full-sequence vocab logits."""
    valid_mask = loss_mask.to(dtype=torch.bool)
    if valid_mask.dim() == 3:
        valid_mask = valid_mask[..., 0]

    valid_target = target[valid_mask]
    valid_predict = predict[valid_mask]
    if valid_predict.numel() == 0:
        zero = predict.sum() * 0.0
        empty = predict.new_empty((0, head.weight.shape[0]))
        return zero, zero, zero, zero, empty, empty

    vloss = criterion(valid_predict, valid_target).mean()
    student_logits = head(valid_predict)
    with torch.no_grad():
        teacher_logits = head(valid_target)
        teacher_p = torch.softmax(teacher_logits, dim=-1)

    student_logp = torch.log_softmax(student_logits, dim=-1)
    ploss = -(teacher_p * student_logp).sum(dim=-1).mean()

    topk = min(train_config["ranking_topk"], teacher_p.shape[-1])
    if train_config["ranking_w"] > 0 and topk > 1:
        topk_indices = torch.topk(teacher_p, k=topk, dim=-1).indices
        ranked_logits = student_logits.gather(-1, topk_indices)
        reversed_logits = torch.flip(ranked_logits, dims=[-1])
        log_denominator = torch.flip(
            torch.logcumsumexp(reversed_logits, dim=-1), dims=[-1]
        )
        rloss = -(ranked_logits - log_denominator).sum(dim=-1).mean()
    else:
        rloss = student_logits.sum() * 0.0

    loss = (
        train_config["v_w"] * vloss
        + train_config["p_w"] * ploss
        + train_config["ranking_w"] * rloss
    )
    return loss, vloss, ploss, rloss, student_logits, teacher_logits


def build_rollout_hidden(hidden_states, predict):
    """Shift draft states by one position for one-step training-time rollout."""
    hidden_size = predict.shape[-1]
    seed_hidden = hidden_states[:, :1, -hidden_size:]
    return torch.cat((seed_hidden, predict[:, :-1]), dim=1)


def forward_with_training_loss(model, data, model_kwargs):
    loss_mask = data["loss_mask"][:, :, None]
    predict0 = model(data["hidden_states"], **model_kwargs)
    step0 = compute_loss(data["target"], predict0, loss_mask)
    total_loss = step0[0]
    step1 = None

    if args.stage == 2 and train_config["mtp_steps"] == 1:
        rollout_hidden = build_rollout_hidden(data["hidden_states"], predict0)
        predict1 = model(rollout_hidden, **model_kwargs)
        step1 = compute_loss(data["target"], predict1, loss_mask)
        total_loss = total_loss + train_config["mtp_w"] * step1[0]

    return total_loss, step0, step1


def module_grad_norm(module):
    norms = [p.grad.detach().float().norm(2) for p in module.parameters() if p.grad is not None]
    if not norms:
        return 0.0
    return torch.stack(norms).norm(2).item()


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

resume_state_dir = None
if not os.path.exists(args.cpdir):
    if accelerator.is_main_process:
        os.makedirs(args.cpdir)
else:
    ckpts = os.listdir(args.cpdir)
    if ckpts:
        begin_epoch = max(
            int(c.split("_")[1]) + 1 if c.startswith("state") else 0 for c in ckpts
        )
        candidate_state_dir = os.path.join(args.cpdir, f"state_{begin_epoch - 1}")
        if os.path.exists(os.path.join(candidate_state_dir, "model.safetensors")):
            resume_state_dir = candidate_state_dir
            args.begin_epoch = begin_epoch
            print(f"resume full training state from {resume_state_dir}")


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

# Stage 1 is text-only, so the visual detail pooler cannot participate in its
# computation graph. Keep it frozen for text pretraining; a Stage 2 process
# constructs a fresh model with this module trainable and loads the Stage 1
# checkpoint before multimodal optimization.
if args.stage == 1:
    for param in model.vis_detail_pooler.parameters():
        param.requires_grad = False

if args.loadpath and resume_state_dir is None:
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
updates_per_epoch = max(
    math.ceil(len(train_loader) / train_config["gradient_accumulation_steps"]), 1
)
total_steps = max(
    args.max_train_steps
    if args.max_train_steps > 0
    else updates_per_epoch * num_epochs,
    1,
)
# Preserve the one-epoch warmup for normal training, but always leave at least
# one non-warmup optimizer update in explicitly step-limited runs.
num_warmup_steps = min(updates_per_epoch, max(total_steps - 1, 0))
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

if resume_state_dir is not None:
    accelerator.load_state(resume_state_dir)
elif is_warmup and args.begin_epoch > 0:
    for _ in range(args.begin_epoch * updates_per_epoch):
        scheduler.step()

global_train_steps = 0  # completed optimizer updates, not micro-batches

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

            loss, step0, step1 = forward_with_training_loss(
                model, data, model_kwargs
            )
            vloss, ploss, rloss = step0[1:4]
            out_head, target_head = step0[4:6]
            accelerator.backward(loss)
            pooler_grad_norm = 0.0
            if args.stage == 2:
                pooler_grad_norm = module_grad_norm(
                    accelerator.unwrap_model(model).vis_detail_pooler
                )
            if accelerator.sync_gradients:
                accelerator.clip_grad_value_(
                    model.parameters(), train_config["grad_clip"]
                )
            optimizer.step()
            did_optimizer_update = (
                accelerator.sync_gradients
                and not accelerator.optimizer_step_was_skipped
            )
            if is_warmup:
                scheduler.step()

        with torch.no_grad():
            target = torch.argmax(target_head, dim=-1)
            predicted = torch.argmax(out_head, dim=-1)
            ct = target.numel()
            cc = (predicted == target).sum().item()
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if accelerator.is_main_process and ct != 0:
            logdict = {
                "train/lr": optimizer.optimizer.param_groups[0]["lr"],
                "train/vloss": vloss.item(),
                "train/ploss_step0": ploss.item(),
                "train/rloss_step0": rloss.item(),
                "train/loss": loss.item(),
                "train/acc_step0": cc / ct,
                "train/vis_detail_pooler_grad_norm": pooler_grad_norm,
            }
            if step1 is not None:
                logdict.update({
                    "train/vloss_step1": step1[1].item(),
                    "train/ploss_step1": step1[2].item(),
                    "train/rloss_step1": step1[3].item(),
                })
                step1_target = torch.argmax(step1[5], dim=-1)
                step1_pred = torch.argmax(step1[4], dim=-1)
                logdict["train/acc_step1"] = (
                    (step1_pred == step1_target).float().mean().item()
                )
            for id, i in enumerate(top_3acc):
                logdict[f"train/top_{id + 1}_acc"] = topkacc[id].item() / ct
            wandb.log(logdict)
            writer.add_scalars("train", logdict, epoch * len(train_loader) + batch_idx)

        epoch_loss += loss.item()
        epoch_vloss += vloss.item()
        epoch_ploss += ploss.item()
        num_batches += 1
        if did_optimizer_update:
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

            loss, step0, step1 = forward_with_training_loss(
                model, data, model_kwargs_val
            )
            vloss, ploss, rloss = step0[1:4]
            out_head, target_head = step0[4:6]
            target = torch.argmax(target_head, dim=-1)
            predicted = torch.argmax(out_head, dim=-1)
            ct = target.numel()
            cc = (predicted == target).sum().item()
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
