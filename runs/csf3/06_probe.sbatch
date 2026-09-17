#!/bin/bash
#SBATCH --job-name=gvm-probe
#SBATCH --partition=gpuH_short,gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Staged probes, cheapest first, each with its own timeout. The 4-hour smoke test
# hung with zero output, which told us nothing about where. These bound each
# stage so a hang names itself.

set -uo pipefail
GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
source "$GVM_ROOT/runs/csf3/env.sh"
source "$GVM_ROOT/runs/csf3/config.sh"
cd "$GVM_ROOT"
gvm_activate

stamp() { date "+%H:%M:%S"; }
probe() {  # probe <name> <timeout-seconds> <command...>
    local name="$1" secs="$2"; shift 2
    echo "[$(stamp)] >>> $name (limit ${secs}s)"
    if timeout --signal=KILL "$secs" "$@"; then
        echo "[$(stamp)] <<< $name OK"
    else
        local rc=$?
        [[ $rc -eq 137 ]] && echo "[$(stamp)] <<< $name TIMED OUT -- this is the hang" \
                          || echo "[$(stamp)] <<< $name FAILED rc=$rc"
        return 1
    fi
}

echo "node=$(hostname) gpus=$(nvidia-smi -L | wc -l) RAY_TMPDIR=$RAY_TMPDIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
ip -o -4 addr show | awk '{print "  iface:", $2, $4}'

probe "torch+NCCL 2-GPU allreduce" 300 python -c "
import os, torch, torch.distributed as dist, datetime
print('torch', torch.__version__, 'devices', torch.cuda.device_count(), flush=True)
os.environ.setdefault('MASTER_ADDR','127.0.0.1'); os.environ.setdefault('MASTER_PORT','29511')
dist.init_process_group('nccl', rank=0, world_size=1, timeout=datetime.timedelta(seconds=120))
t = torch.ones(8, device='cuda:0'); dist.all_reduce(t)
print('allreduce ok', t.sum().item(), flush=True)
dist.destroy_process_group()
"

probe "ray.init" 300 python -c "
import ray, pprint
ray.init(num_cpus=4, include_dashboard=False, log_to_driver=True)
pprint.pprint(ray.cluster_resources()); ray.shutdown(); print('ray ok', flush=True)
"

probe "vLLM load + generate" 900 python -c "
import torch
from vllm import LLM, SamplingParams
llm = LLM('$GVM_MODEL', dtype=torch.bfloat16, gpu_memory_utilization=0.6,
          enforce_eager=True, max_model_len=2048)
o = llm.generate(['What is 2+2? Answer within \\\\boxed{}.'], SamplingParams(max_tokens=64, temperature=0))
print('vllm ok:', repr(o[0].outputs[0].text[:120]), flush=True)
"

echo "[$(stamp)] PROBE_DONE"
