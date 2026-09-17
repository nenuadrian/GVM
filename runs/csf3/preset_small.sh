# Scaled-down replication: enough to show the GVM-vs-baseline separation without
# a week of queueing. Source via GVM_PRESET=small.
#
# 5 EM iterations x 9 steps = 45 steps, and the baselines are held to the same 45
# so the comparison is at equal data and equal sample budget. Figures 3 and 5 put
# the GVM/baseline gap clearly open by ~40 steps, so the ordering should be
# visible here -- the absolute numbers will sit below Table 1, which trains to
# ~135-146 steps.

export GVM_EM_ITERS=5
export GVM_BASELINE_STEPS=$(( GVM_EM_ITERS * 9 ))
export GVM_EVAL_DATA=math500
export GVM_PROJECT=gvm-qwen15-math500-small
