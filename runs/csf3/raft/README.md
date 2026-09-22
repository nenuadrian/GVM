# Replicating RAFT with LMFlow on CSF3

Runs [RAFT](https://arxiv.org/abs/2304.06767) (Reward rAnked FineTuning) through
[LMFlow](https://github.com/OptimalScale/LMFlow)'s own `raft_aligner` — the
implementation the RAFT authors shipped — on HH-RLHF, on 4×H200, logging to
Weights & Biases.

This is the baseline GVM improves on, run in its original codebase rather than
verl's `adv_estimator=raft`. Separate conda env, separate LMFlow checkout,
separate wandb project; it shares nothing with `runs/csf3/` but the H200 quota.

## What "smaller than LLaMA-7B" means here

| | paper | here |
|---|---|---|
| policy | LLaMA-7B → SFT on HH-RLHF preferred responses | OpenLLaMA-3B-v2 → same SFT |
| reward model | HH-RLHF RM | `weqweasdas/hh_rlhf_rm_open_llama_3b` |
| prompts | HH-RLHF, ≤256 tokens | same, 112,052 |
| best-of-K | K=8 (`top_reward_percentage=0.125`) | same |
| iterations | 20 × 1024 prompts | same |

The reward model is the authors' own — its card cites arXiv:2304.06767 — and is
itself an OpenLLaMA-3B, so policy and RM share a lineage the way LLaMA-7B-SFT and
its RM did. It is the one thing that should not be swapped: it defines the reward
scale the whole replication is measured on.

The SFT stage is not optional. RAFT ranks the policy's *own* samples, and the RM
was trained on `###Human:`/`###Assistant:` text. A base model that has never seen
that format produces samples the RM scores as noise, and the run converges to
nothing.

## The version wall, which is the whole story

LMFlow's `raft_trainer.py` is a fork of the transformers 4.28 `Trainer`. It
imports `ShardedDDPOption` from `transformers.trainer_utils` and
`is_fairscale_available` from `transformers.integrations`. Both were **deleted in
transformers 4.35.0**, so the module stops importing. LMFlow knows this — its own
`auto_pipeline.py` reads:

```python
if not is_package_version_at_least("transformers", "4.35.0"):
    from lmflow.pipeline.raft_aligner import RaftAligner
    PIPELINE_MAPPING["raft_aligner"] = RaftAligner
else:
    PIPELINE_NEEDS_EXTRAS.append("raft_aligner")
```

On a modern install the pipeline is silently *absent*, not broken — you get a
`KeyError` on `"raft_aligner"` with no explanation. (v0.0.6, which this pins, has
no such gate: it imports `RaftAligner` unconditionally, so on transformers ≥ 4.35
it fails loudly at import instead. Either way 4.34 is the ceiling.) Everything
else follows from that pin: `tokenizers==0.14.1` is the only range transformers
4.34.1 accepts, and it in turn caps `huggingface_hub<0.18`; `numpy<2` because
torch 2.1 is built against the numpy 1.x C ABI; `setuptools<81` because torch
2.1's `cpp_extension.py` opens with `from pkg_resources import packaging`.
`torch==2.1.2+cu121` is the floor that still ships sm_90 kernels for H200.

**LMFlow is pinned to v0.0.6 (Dec 2023), not HEAD.** On HEAD,
`AutoModel.get_model(arch_type="text_regression")` returns `HFTextRegressionModel`,
whose own `inference()` shadows the reward function `raft_align.py` registers via
`register_inference_function` — so the reward model you pass on the command line
is loaded, ignored, and silently replaced by a gpt2 head. v0.0.6 still returns the
plain `TextRegressionModel`, where registration works.

## Layout

| File | Role |
|---|---|
| `env.sh` | paths, caches, wandb, `raft_activate` |
| `config.sh` | every hyperparameter, all overridable from the environment |
| `patches/apply_patches.py` | wandb logging + reward-scoring fixes, idempotent |
| `configs/ds_zero2_no_offload.json` | ZeRO-2 without CPU offload |
| `00_build_env.sbatch` | conda env + LMFlow v0.0.6 + patches, ~40 min |
| `01_prepare_data.sbatch` | HH-RLHF download, eval slice, HF cache warm |
| `05_smoke.sbatch` | 5-phase shakedown, ~20 min — run this first |
| `02_sft.sbatch` | OpenLLaMA-3B → `*-SFT` on 112K preferred responses |
| `03_raft.sbatch` | the RAFT loop: 20 × (best-of-8 → SFT on the argmax) |
| `04_eval.sbatch` | mean reward on 2K held-out prompts, base vs sft vs raft |
| `reward_eval.py` | the evaluator 04 calls |
| `submit_all.sh` | queues the lot as one dependency chain |

## Running

```bash
sbatch runs/csf3/raft/00_build_env.sbatch     # once; read the output
```

It ends with `BUILD_OK` only if `raft_aligner` is actually in `PIPELINE_MAPPING`.
Then:

```bash
bash runs/csf3/raft/submit_all.sh
```

Useful variants:

```bash
bash runs/csf3/raft/submit_all.sh --from raft      # skip straight to alignment
bash runs/csf3/raft/submit_all.sh --no-smoke
RAFT_DRY_RUN=1 bash runs/csf3/raft/submit_all.sh   # print, don't queue
RAFT_BASE_MODEL=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  bash runs/csf3/raft/submit_all.sh                # a 1.1B run instead
```

Any substitute policy must be an architecture transformers 4.34 knows.
TinyLlama, GPT-Neo and OPT are fine. **Qwen2 / Qwen2.5 are not** — Qwen2 support
landed in transformers 4.37, above the 4.35 ceiling.

## What the smoke test checks, in order

1. `raft_aligner` is in `PIPELINE_MAPPING` — i.e. the transformers pin held.
2. Both patches are present in the checkout.
3. **The reward model separates chosen from rejected** on 64 HH-RLHF pairs, and
   asserts pairwise accuracy > 0.6. The card reports ~69% for the GPT-Neo RM and
   ~79% for LLaMA-7B. Near chance means the format or dtype is wrong, and every
   reward in the run would be noise.
4. Two real RAFT iterations at `raft_batch_size=16`, and a model gets saved.
5. `reward_eval.py` runs and reaches wandb.

## wandb

Project `raft-lmflow-hh` (`RAFT_PROJECT`). Run ids are md5 of project/run-name, so
a requeue appends to the same run.

`raft/mean_reward` is the curve — the reward the policy's own samples earn, per
iteration. `raft/mean_reward_selected` is the best-of-8 subset that actually gets
trained on, so `raft/reward_gain` is the headroom RAFT is exploiting; it should
shrink as the policy improves. `eval/{base,sft,raft}/mean_reward` land in the run
summary.

Upstream logs none of this. `reward_seq` and `train_reawrd` are accumulated in
memory and rendered to `training_reward.png` on rank 0, and that is all — the
numbers never leave the node. `patches/apply_patches.py` adds the wandb call.

The inner trainer runs with `--report_to none` on purpose: `raft_aligner` calls
`trainer.train()` once per RAFT iteration, so `global_step` restarts at 0 each
time and wandb drops every step after the first iteration. The patch logs one row
per iteration on a monotonic axis instead.

## Expected noise in the logs

Every job prints a `CUDA SETUP` banner ending in `libcusparse.so.12: cannot open
shared object file`. bitsandbytes cannot find cusparse — torch bundles cublas and
cudart but not cusparse, and CSF3 has no CUDA toolkit module — and LMFlow has a
bare `import bitsandbytes` in `hf_decoder_model.py`. bitsandbytes 0.43.3 warns and
continues where 0.41.x treated it as fatal, which is the only reason that pin is
not era-matched. Nothing here quantizes, so importable is all it needs to be.

Also expected: `transformers.deepspeed module is deprecated`, from LMFlow's
`from transformers.deepspeed import HfDeepSpeedConfig`.

## Upstream bugs worked around

- `scripts/run_raft_align.sh` passes `--dataset_path data/hh_rlhf/rlhf_prompt`.
  The tarball unpacks to `data/hh_rlhf/rlhf/rlhf_prompt`. The upstream path does
  not exist.
- The default reward model in `examples/raft_align.py` is `weqweasdas/hh_rlhf_rm`,
  which **404s**. `03_raft.sbatch` always passes `--reward_model_or_path`.
- `hh_rlhf.tar.gz` was rolled on macOS and ships AppleDouble siblings, including
  `._hh_rlhf_rlhf_prompt.json`. LMFlow globs `*.json` over the directory and hands
  those 176-byte binary stubs to the json loader. `01_prepare_data.sbatch` deletes
  them.
- The reward pipeline hardcoded `batch_size: 1`, so a best-of-8 group cost 8
  sequential RM forward passes. Patched to score the group in one, in bf16, with
  truncation to the RM's 2048-token window.
- `device=f"cuda:{pipeline_args.local_rank}"` asks for `cuda:-1` without a
  distributed launcher. Patched to `max(local_rank, 0)`.
- `collection_strategy=top` cannot run at all: `align()` does an unconditional
  `print(M, K)` while `K` is bound only inside the `local` branch, so it raises
  `NameError` before the first iteration. Left alone — `local` is the right
  strategy for HH-RLHF anyway, and it is what the paper's own script uses.
- HEAD's `configs/archive/ds_config_zero2.json` (referenced by the archived
  script) does not exist; v0.0.6's does but enables CPU optimizer offload, which
  needs `nvcc` to JIT `DeepSpeedCPUAdam`. Pointless for a 3B model on a 141 GB
  card, so this ships `ds_zero2_no_offload.json`.

## Cost

Roughly, on 4×H200: SFT ~2–4 h (112K samples, 1 epoch), RAFT ~3 h (20 iterations
× 1024 prompts × 8 samples), each eval ~20 min. Call it a day of wall clock.

`gpu-h200-fse` caps **4 H200s via GrpTRES**, which is a QOS *group* limit shared
across every user on that QOS — not a per-account one. `gpuH` can therefore sit at
`QOSGrpGRES` with none of your own jobs running, because other members of the
group hold the four cards. It is also the same pool `runs/csf3/submit_all.sh`
draws on, so the two chains compete with each other as well.

`gpu-free` (gpuL, gpuA) is a separate pool, so `--target l40s` neither waits on
nor consumes the H200 quota, and gpuL has 22 nodes against gpuH's 4. A 3B policy
plus a 3B reward model in bf16 is ~15 GB before optimizer state, so this fits an
L40S's 48 GB comfortably — the H200's 141 GB buys nothing here but queue time.

```bash
bash runs/csf3/raft/submit_all.sh --target l40s
```
