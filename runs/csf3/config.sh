# Replication settings for GVM (arXiv:2505.02391) Table 1, Qwen2.5-Math-1.5B / MATH500.
# Sourced after env.sh. Every value can be overridden from the environment.

# --- paper Table 2 (Appendix B) -------------------------------------------
export GVM_MODEL="${GVM_MODEL:-Qwen/Qwen2.5-Math-1.5B}"
export GVM_ALPHA="${GVM_ALPHA:-0.001}"          # alpha = 1e-3
export GVM_BETA="${GVM_BETA:-2.0}"              # beta  = 2
export GVM_TRAIN_BATCH="${GVM_TRAIN_BATCH:-1024}"
export GVM_PPO_MINI_BATCH="${GVM_PPO_MINI_BATCH:-256}"
export GVM_MAX_PROMPT_LEN="${GVM_MAX_PROMPT_LEN:-1024}"
export GVM_MAX_RESP_LEN="${GVM_MAX_RESP_LEN:-3072}"
export GVM_LR="${GVM_LR:-1e-6}"
export GVM_KL_COEF="${GVM_KL_COEF:-0.001}"

# --- paper section 4.1 ----------------------------------------------------
# "eps_low = 0.2, and eps_high = 0.28, 0.4 then 0.35 for RAFT++ and GVM-RAFT++".
# Three values for two methods; 0.35 reads as the setting they landed on.
# GRPO deliberately omits clip-higher ("it leads to worse performance").
export GVM_CLIP_LOW="${GVM_CLIP_LOW:-0.2}"
export GVM_CLIP_HIGH_RAFT="${GVM_CLIP_HIGH_RAFT:-0.35}"
export GVM_CLIP_HIGH_GRPO="${GVM_CLIP_HIGH_GRPO:-0.2}"

# Sampling budget: N' = 8 pilot samples per prompt, N = 8n total (Figure 3, left).
export GVM_STAGE1_SAMPLES="${GVM_STAGE1_SAMPLES:-8}"
export GVM_ROLLOUT_N="${GVM_ROLLOUT_N:-8}"      # baselines: 8 rollouts/prompt
# Appendix B: "for GVM-GRPO, we divide the sample budget [...] by a fixed size
# (4 in our experiments) [...] then we set the rollout number to 4".
export GVM_GRPO_ROLLOUT_N="${GVM_GRPO_ROLLOUT_N:-4}"
export GVM_RAFT_ROLLOUT_N="${GVM_RAFT_ROLLOUT_N:-1}"

# Evaluation: "performance is measured by Average @ 8 [...] we use a temperature
# of 1.0 in evaluation" for Qwen2.5-Math-1.5B.
export GVM_VAL_N="${GVM_VAL_N:-8}"
export GVM_VAL_TEMP="${GVM_VAL_TEMP:-1.0}"

# --- cluster shape --------------------------------------------------------
# QOS gpu-h200-fse caps this account at gres/gpu:h200=4 across all jobs, so the
# four runs are inherently sequential.
export GVM_GPUS="${GVM_GPUS:-4}"
export GVM_EM_ITERS="${GVM_EM_ITERS:-15}"       # 15 Numina shards x 9 steps = 135 steps
export GVM_SAVE_FREQ="${GVM_SAVE_FREQ:-5}"
export GVM_TEST_FREQ="${GVM_TEST_FREQ:-5}"

# Baseline step budget. Empty = one full epoch over the 149,882-prompt split
# (146 steps), which is what the paper does. Set it to 9 * GVM_EM_ITERS to hold
# the baselines to the same number of steps as a shortened GVM run, otherwise
# the comparison is against a baseline that saw more data.
export GVM_BASELINE_STEPS="${GVM_BASELINE_STEPS:-}"

# Benchmarks for 04_eval.sbatch. All five reproduce Table 1; math500 alone is
# ~3x quicker and is the headline column.
export GVM_EVAL_DATA="${GVM_EVAL_DATA:-math500,minerva_math,olympiad_bench,aime24,amc23}"

# Throughput knobs (scheduling only; they do not change the objective).
export GVM_MAX_BATCHED_TOKENS="${GVM_MAX_BATCHED_TOKENS:-16384}"
export GVM_MAX_TOKEN_LEN_PER_GPU="${GVM_MAX_TOKEN_LEN_PER_GPU:-32768}"
export GVM_GPU_MEM_UTIL="${GVM_GPU_MEM_UTIL:-0.75}"

export GVM_PROJECT="${GVM_PROJECT:-gvm-qwen15-math500}"
