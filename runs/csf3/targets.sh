# Cluster targets. gpu-h200-fse and gpu-free are separate QOS pools, so an H200
# job and an L40S/A100 job can be in flight at the same time without competing.
#
# Estimated starts for a 4-GPU / 24h job as of 2026-09-17:
#   gpuH  (4 nodes,  16 pending)  ~2 weeks
#   gpuL  (22 nodes, 167 pending) ~6 days     <- shortest
#   gpuA  (19 nodes, 209 pending) ~2 weeks    <- deepest backlog on the cluster
#
# gvm_target <name> exports GVM_SB_* for submit_all.sh to pass to sbatch, plus
# any per-device config overrides.

gvm_target() {
    case "${1:?usage: gvm_target h200|l40s|a100}" in
        h200)
            GVM_SB_PARTITION=gpuH
            GVM_SB_ACCOUNT=gpu-h200-fse-pgdr
            GVM_SB_GRES="gpu:h200:${GVM_GPUS:-4}"
            GVM_SB_MEM=200G
            : "${GVM_GPU_MEM_UTIL:=0.75}"
            ;;
        l40s)
            # 48 GB per card rather than 140. vLLM sleeps its cache engine before
            # the update step, so 0.70 still leaves room, but the margin is thinner.
            GVM_SB_PARTITION=gpuL
            GVM_SB_ACCOUNT=gpu-free
            GVM_SB_GRES="gpu:l40s:${GVM_GPUS:-4}"
            GVM_SB_MEM=160G
            : "${GVM_GPU_MEM_UTIL:=0.70}"
            # No NVLink: keep the reference policy off-GPU rather than paying for
            # extra PCIe traffic during the log-prob pass.
            : "${GVM_MAX_BATCHED_TOKENS:=8192}"
            ;;
        a100)
            GVM_SB_PARTITION=gpuA
            GVM_SB_ACCOUNT=gpu-free
            GVM_SB_GRES="gpu:a100_80g:${GVM_GPUS:-4}"
            GVM_SB_MEM=160G
            : "${GVM_GPU_MEM_UTIL:=0.75}"
            ;;
        *) echo "unknown target '$1' (h200|l40s|a100)" >&2; return 2 ;;
    esac
    export GVM_SB_PARTITION GVM_SB_ACCOUNT GVM_SB_GRES GVM_SB_MEM
    export GVM_GPU_MEM_UTIL
    [[ -n "${GVM_MAX_BATCHED_TOKENS:-}" ]] && export GVM_MAX_BATCHED_TOKENS
    return 0
}
