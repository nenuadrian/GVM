# Replicating GVM Table 1 (Qwen2.5-Math-1.5B / MATH500) on CSF3

Reproduces the Qwen2.5-Math-1.5B block of Table 1 in
[arXiv:2505.02391](https://arxiv.org/abs/2505.02391) — GRPO, RAFT++, GVM-GRPO and
GVM-RAFT++ — on 4×H200, logging to Weights & Biases.

## Target numbers

| Method      | Math500 | Minerva | Olympiad | AIME24 | AMC23 | 5 Average |
|-------------|---------|---------|----------|--------|-------|-----------|
| Base        | 56.35   | 17.00   | 25.22    |  3.33  | 37.81 | 27.94     |
| GRPO        | 70.78   | 29.00   | 33.56    | 10.00  | 47.19 | 38.11     |
| RAFT++      | 69.02   | 27.71   | 31.74    |  9.58  | 44.06 | 36.42     |
| GVM-GRPO    | 73.92   | 29.96   | 36.26    | 12.92  | 49.06 | 40.42     |
| GVM-RAFT++  | 72.90   | 29.04   | 36.20    |  9.17  | 51.88 | 39.64     |

Scored as avg@8 at temperature 1.0 with Math-Verify, which is what both the
in-training validation and `eval/` are configured to do here.

## Layout

| File | Role |
|---|---|
| `env.sh` | paths, caches, wandb, `gvm_activate` |
| `config.sh` | every hyperparameter, all overridable from the environment |
| `00_build_env.sbatch` | conda env: python 3.10, torch 2.6, vLLM 0.8.3, flash-attn 2.7.4 |
| `01_prepare_data.sbatch` | MATH500 + 15 Numina shards + combined split → parquet |
| `05_smoke.sbatch` | 256-prompt end-to-end shakedown of both code paths |
| `02_train_baseline.sbatch` | GRPO / RAFT++ |
| `03_train_gvm.sbatch` | the EM loop: 15 × (E-step + 9 M-steps) |
| `04_eval.sbatch` | offline avg@8 over the five benchmarks |
| `submit_all.sh` | queues the lot as one `afterok` chain |

## Running

```bash
sbatch runs/csf3/00_build_env.sbatch     # once, ~40 min
bash   runs/csf3/submit_all.sh           # data → smoke → base eval → 4 runs → 4 evals
```

Single method:

```bash
bash runs/csf3/submit_all.sh gvm-grpo
```

If a job hits the 4-day wall clock, submit the same script again — baselines
resume from the last checkpoint (`save_freq=5`, `resume_mode=auto`) and the GVM
loop skips iterations whose merged HF export already exists. Both land back in
the same wandb run.

## How the runs are configured

Everything below is from the paper unless marked otherwise.

**Shared** (Table 2): batch 1024 prompts, mini-batch 256, max prompt 1024, max
response 3072, constant lr 1e-6 with no warmup, KL loss coefficient 0.001
(`low_var_kl`), α = 1e-3, β = 2.

**Baselines** train one epoch over the concatenated 15 Numina shards (149,882
prompts → 146 steps) with 8 rollouts per prompt.

**GVM** runs 15 iterations, one per Numina shard. Appendix B: "each iteration
consists of nine steps of update as the amount of training data per iteration is
slightly less than 10 × 1024" — each shard is ~9,986 prompts, so 9 steps, 135
total. Each iteration is:

1. **E-step** — `stage_1_collect_data.py` draws N′ = 8 pilot rollouts per prompt
   (4 GPUs, one contiguous shard each); `stage_2_calc_acceptRates_grads.py`
   turns those into accept rates pᵢ and gradient norms Gᵢ on `embed_tokens`;
   `stage_2_calc_sample_size.py` applies Proposition 1 to get per-prompt
   budgets nᵢ summing to N.
2. **M-step** — verl with `use_em=True`, which repeats each prompt nᵢ times
   before rollout.
3. **Merge** — FSDP shards → HF, so the next E-step can load the policy in vLLM.

The parquet row order must match the HF split row order, because `sample_sizes`
is indexed by `extra_info.index`; that is why `01_prepare_data.sbatch` passes no
`--train_end`.

**Per-method settings**

| | adv_estimator | policy_loss | clip low/high | rollout n | budget |
|---|---|---|---|---|---|
| GRPO | `grpo` | `plusplus` | 0.2 / 0.2 | 8 | — |
| RAFT++ | `raft` | `plusplus` | 0.2 / 0.35 | 8 | — |
| GVM-GRPO | `grpo` | `plusplus` | 0.2 / 0.2 | 4 | N/4 copies |
| GVM-RAFT++ | `raft` | `plusplus` | 0.2 / 0.35 | 1 | N copies |

GRPO omits clip-higher on purpose — §4.1 reports it hurt. GVM-GRPO's rollout
number of 4 and the matching budget division are from Appendix B: "we divide the
sample budget calculated in the second stage by a fixed size (for example, 4 in
our experiments), and copy the prompts multiple times by nᵢ/4". Every method
therefore spends the same 8 samples per prompt per step on average.

## Deviations, and why

- **ε_high = 0.35.** §4.1 reads "ε_high = 0.28, 0.4 then 0.35 for RAFT++ and
  GVM-RAFT++ respectively" — three values for two methods. 0.35 reads as the
  value they settled on. Override with `GVM_CLIP_HIGH_RAFT`.
- **`use_dynamic_bsz=True` everywhere.** `run_raft.sh` and `run_em.sh` already
  use it; `run_grpo.sh` uses a fixed micro-batch of 4, which on H200 is mostly
  idle silicon. Same objective, different micro-batch packing.
- **In-training validation is avg@8 at T=1.0**, not verl's greedy default, so
  the wandb curve is on the same scale as Table 1.
- **`max_num_batched_tokens=16384`** (repo uses 8192) and
  `ppo_max_token_len_per_gpu=32768`. Scheduling only.
- **`ppo_mini_batch_size=256` for all four**, exactly as the authors' scripts
  have it. Worth knowing what that implies: verl scales it by `rollout.n`, so at
  8192 sequences per step the baselines take 4 gradient updates per step, while
  GVM-GRPO takes 8 and GVM-RAFT++ takes 32. That is what `run_em.sh` does, and
  the paper's RAFT++ description ("multiple gradient steps per iteration in a
  mini-batch way", with importance sampling and clipping to absorb the
  off-policyness) reads as deliberate. Set `GVM_PPO_MINI_BATCH` to equalise it
  if you want that confound removed.
- **`runs/scripts/run_grpo.sh` points at `./data/numina_math/train.parquet`**,
  which `numina_process.py` never writes (it writes `_1`…`_15` and `_15_all`).
  These scripts use `numina_math_15_all`, matching `run_raft.sh`.

## Patches to the repo

- `em/stage_1_collect_data.py`, `em/stage_2_calc_acceptRates_grads.py`: timing
  files were written to a hardcoded `/home/ubuntu/projects/gvm/GVM/em/` path.
- `verl/utils/tracking.py`: added `VERL_LOG_STEP_OFFSET`, so the 15 separate
  verl invocations of a GVM run land on one continuous wandb step axis.
- `requirements.txt` line 201 is `-e file:///home/ubuntu/projects/gvm/GVM`;
  `00_build_env.sbatch` strips it and installs the local checkout instead.

## wandb

Project `gvm-qwen15-math500` (set `GVM_PROJECT` to change). One run per method,
with a stable run id derived from the name so requeues and the 15 GVM
iterations all append to the same run. `val-core/.../mean@8` is the MATH500
curve. GVM runs also get `estep/*`: accept-rate distribution, gradient norms,
the allocated budget, and a per-iteration scatter of sample size against accept
rate (Figure 2). Final `eval/` numbers land in the run summary as `final/*`.

Credentials come from `~/.netrc` (`machine api.wandb.ai`), already present on
CSF3. Compute nodes have outbound internet, so `WANDB_MODE=online` works.

## Cost

Appendix B: ~90 minutes per GVM iteration on 4×H100 at N′ = 8, N = 8n. So
roughly 20–25 h per GVM run and 13–18 h per baseline on 4×H200, plus ~1 h per
evaluation. Call it 3–4 days of wall clock for all four, before queueing.

The `gpu-h200-fse` QOS caps this account at **4 H200s in total**, so the runs
cannot overlap — and any other gpuH job you have queued competes for the same
four cards.
