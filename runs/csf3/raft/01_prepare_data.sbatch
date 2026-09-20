#!/bin/bash
#SBATCH --job-name=raft-data
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
source "$GVM_ROOT/runs/csf3/raft/env.sh"
source "$GVM_ROOT/runs/csf3/raft/config.sh"
raft_activate
cd "$LMFLOW_ROOT"

# HH-RLHF, as LMFlow packages it: 112,052 SFT pairs, the same 112,052 as bare
# RAFT prompts, and 12,451 held-out eval prompts. ~457 MB unpacked.
if [[ ! -d data/hh_rlhf ]]; then
    ( cd data && ./download.sh hh_rlhf )
fi

# The tarball was rolled on macOS and carries AppleDouble siblings -- including
# data/hh_rlhf/rlhf/rlhf_prompt/._hh_rlhf_rlhf_prompt.json. LMFlow's Dataset
# globs *.json over the directory, so these 176-byte binary stubs get handed to
# the json loader and it dies. Delete them before anything reads the directory.
find data/hh_rlhf -name '._*' -delete
echo "AppleDouble stubs remaining: $(find data/hh_rlhf -name '._*' | wc -l)"

# A held-out eval slice for 04_eval.sbatch, in LMFlow's text_only schema.
python - <<PY
import json, os, random
from pathlib import Path

src = Path("data/hh_rlhf/rlhf/rlhf_eval/hh_rlhf_rlhf_eval_prompt.json")
out = Path(os.environ["RAFT_OUT"]) / "data" / "eval_prompts"
out.mkdir(parents=True, exist_ok=True)

d = json.load(src.open())
inst = d["instances"]
random.Random(0).shuffle(inst)
n = int(os.environ["RAFT_EVAL_N"])
sub = {"type": "text_only", "instances": inst[:n]}
(out / "eval_prompts.json").write_text(json.dumps(sub))
print(f"eval slice: {len(sub['instances'])} of {len(inst)} -> {out/'eval_prompts.json'}")
PY

# Warm the HF cache on a node with internet so GPU jobs never block on a
# download. The reward model ships a .bin, not safetensors, and no
# tokenizer_config.json -- AutoTokenizer falls back to config.json's
# model_type=llama plus the bundled tokenizer.model, which is fine on 4.34.
python - <<'PY'
import os
from huggingface_hub import snapshot_download
for repo in (os.environ["RAFT_BASE_MODEL"], os.environ["RAFT_REWARD_MODEL"]):
    p = snapshot_download(repo)
    print(f"cached {repo} -> {p}")
PY

echo "=== dataset sizes ==="
python - <<'PY'
import json, glob
for f in sorted(glob.glob("data/hh_rlhf/*/*.json") + glob.glob("data/hh_rlhf/*/*/*.json")):
    d = json.load(open(f))
    print(f"{len(d['instances']):>7d}  {d.get('type')}  {f}")
PY

echo "=== the policy must speak this format, or the reward model is meaningless ==="
python - <<'PY'
import json
d = json.load(open("data/hh_rlhf/rlhf/rlhf_prompt/hh_rlhf_rlhf_prompt.json"))
print(repr(d["instances"][0]["text"][:220]))
PY
echo "DATA_OK"
