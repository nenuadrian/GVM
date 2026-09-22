# RAFT replication settings (arXiv:2304.06767, "RAFT: Reward rAnked FineTuning"),
# run through LMFlow's own raft_aligner. Sourced after env.sh.
# Every value can be overridden from the environment.

# --- what we run instead of LLaMA-7B ---------------------------------------
# The paper aligns LLaMA-7B-SFT against the HH-RLHF reward model. LLaMA-1 weights
# are not redistributable and 7B is more than this needs, so the policy is
# OpenLLaMA-3B-v2, SFT'd here on the same HH-RLHF preferred-response split that
# the paper's SFT model was trained on. The reward model is the authors' own
# (its card cites arXiv:2304.06767) and is itself an OpenLLaMA-3B, so policy and
# reward model share a lineage the way LLaMA-7B-SFT and its RM did.
#
# Anything smaller must still be an architecture transformers 4.34 knows:
# TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T, EleutherAI/gpt-neo-1.3B
# and facebook/opt-1.3b all work. Qwen2/Qwen2.5 do NOT -- Qwen2 support landed
# in transformers 4.37, above the 4.35 ceiling this code imposes.
export RAFT_BASE_MODEL="${RAFT_BASE_MODEL:-openlm-research/open_llama_3b_v2}"
export RAFT_REWARD_MODEL="${RAFT_REWARD_MODEL:-weqweasdas/hh_rlhf_rm_open_llama_3b}"

# --- SFT stage (paper section 4.2 / LMFlow docs section 2.1) ---------------
# "We only use the preferred responses so we get 112K training samples."
export RAFT_SFT_LR="${RAFT_SFT_LR:-2e-5}"
export RAFT_SFT_EPOCHS="${RAFT_SFT_EPOCHS:-1}"
export RAFT_SFT_BS="${RAFT_SFT_BS:-8}"
export RAFT_SFT_BLOCK="${RAFT_SFT_BLOCK:-512}"
# Empty = the full 112,052-sample split, which is what the paper does.
export RAFT_SFT_MAX_SAMPLES="${RAFT_SFT_MAX_SAMPLES:-}"

# --- RAFT stage (LMFlow docs Table 1, scripts/run_raft_align.sh) -----------
export RAFT_ITERS="${RAFT_ITERS:-20}"
export RAFT_BATCH_SIZE="${RAFT_BATCH_SIZE:-1024}"
# top_reward_percentage 0.125 -> K = 1/0.125 = 8 samples per prompt, best-of-8.
export RAFT_TOP_PCT="${RAFT_TOP_PCT:-0.125}"
# "local" ranks the K samples of one prompt against each other -- the right
# choice here, because HH-RLHF rewards are dominated by the prompt. "top" ranks
# globally across prompts, which the paper shows is more reward-efficient but
# degenerates when the prompt drives the reward.
#
# Only "local" actually runs. raft_aligner.align() does an unconditional
# `print(M, K)` while K is bound only inside the `local` branch, so "top" dies
# with NameError before the first iteration. Fixing that is a separate job.
export RAFT_COLLECTION="${RAFT_COLLECTION:-local}"
export RAFT_LR="${RAFT_LR:-2e-5}"
export RAFT_EPOCHS="${RAFT_EPOCHS:-4}"
export RAFT_TRAIN_BS="${RAFT_TRAIN_BS:-1}"
export RAFT_MIN_NEW="${RAFT_MIN_NEW:-96}"
export RAFT_MAX_NEW="${RAFT_MAX_NEW:-128}"
# The selected best-of-K texts are concatenated and re-chunked into blocks of
# this size before the SFT step. Left implicit, raft_aligner derives it from
# tokenizer.model_max_length and lands on 512 for any model with a window above
# 1024 -- so this only makes the existing behaviour explicit and model-independent.
export RAFT_BLOCK="${RAFT_BLOCK:-512}"

# --- evaluation ------------------------------------------------------------
# The paper's headline number for this experiment is mean reward on held-out
# prompts. 2K of the 12,451 eval prompts, matching "We additionally use 2K
# samples from the test set to test the performance of models."
export RAFT_EVAL_N="${RAFT_EVAL_N:-2000}"
export RAFT_EVAL_TEMP="${RAFT_EVAL_TEMP:-0.85}"
export RAFT_EVAL_BS="${RAFT_EVAL_BS:-16}"

# --- cluster shape ---------------------------------------------------------
# QOS gpu-h200-fse caps this account at gres/gpu:h200=4 across all jobs, so this
# competes directly with anything queued from runs/csf3/submit_all.sh.
export RAFT_GPUS="${RAFT_GPUS:-4}"
# deepspeed is what LMFlow's own script uses and is the better-tested path
# through this 2023-era trainer fork. torchrun is the fallback if deepspeed's
# JIT op build fails on a node with no CUDA toolkit.
export RAFT_LAUNCHER="${RAFT_LAUNCHER:-deepspeed}"

export RAFT_PROJECT="${RAFT_PROJECT:-raft-lmflow-hh}"
export RAFT_RUN_NAME="${RAFT_RUN_NAME:-raft-openllama3b}"
