import argparse
import sys

parser = argparse.ArgumentParser(description="SpareSpec Stage1 allocation")
parser.add_argument("--outdir", type=str, default="0")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=67999)
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--data-path", type=str, default=None)
parser.add_argument("--gpus_per_model", type=int, default=1)
args = parser.parse_args()

import os
import torch
from concurrent.futures import ThreadPoolExecutor

s = args.start
e = args.end
num_p = torch.cuda.device_count()
visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
visible_device_ids = None
if visible_devices:
    visible_device_ids = [x.strip() for x in visible_devices.split(",") if x.strip()]
    if len(visible_device_ids) != num_p:
        visible_device_ids = None

gpus = [
    [j for j in range(i, i + args.gpus_per_model)]
    for i in range(0, num_p, args.gpus_per_model)
]
if visible_device_ids is not None:
    gpus = [[visible_device_ids[j] for j in group] for group in gpus]
num_p = len(gpus)

outdir = "{}/qwen2.5vl_shargpt_sparespec_{}_{}_mubf16".format(args.outdir, s, e)


def split_range(start, end, n, over=False):
    length = end - start + 1
    base_interval = length // n
    additional = length % n
    intervals = []
    previous = start
    for i in range(n):
        current_interval = base_interval + (1 if i < additional else 0)
        if over:
            intervals.append((previous, previous + current_interval))
        else:
            intervals.append((previous, previous + current_interval - 1))
        previous += current_interval
    return intervals


def run_command(cmd):
    os.system(cmd)


if not os.path.exists(outdir):
    os.makedirs(outdir)

data_a = split_range(s, e, num_p, over=True)
commands = []
for i in range(num_p):
    index = i
    start, end = data_a[i]
    gpu_index = gpus[i]
    gpu_index_str = " ".join(map(str, gpu_index))
    command = (
        "{} -m vispec.ge_data.ge_data_all_qwen_shargpt_sparespec "
        "--start={} --end={} --index={} --gpu_index {} --outdir {} --model {}".format(
            sys.executable, start, end, index, gpu_index_str, outdir, args.model
        )
    )
    if args.data_path:
        command += f" --data-path {args.data_path}"
    commands.append(command)

with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    for command in commands:
        executor.submit(run_command, command)
        print(command)
