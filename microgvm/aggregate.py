"""Summarise sweep results as MSE relative to uniform|per_prompt, ordered by accept rate."""
import argparse, glob, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pattern", default="results/*.json")
args = ap.parse_args()

rows = []
for f in sorted(glob.glob(args.pattern)):
    d = json.load(open(f))
    meta = d.pop("_meta")
    base = d["uniform|per_prompt"]["mse"]
    rows.append(dict(name=f.split("/")[-1].replace(".json", ""),
                     p_mean=meta["p_mean"], p_spread=meta["p_spread"],
                     **{k: v["mse"] / base for k, v in d.items()}))
if not rows:
    raise SystemExit(f"no results matching {args.pattern}")

keys = [k for k in rows[0] if "|" in k]
rows.sort(key=lambda r: r["p_mean"])
print(f"{'run':>24} {'p_mean':>7} {'p_sprd':>7} | " + " ".join(f"{k.split('|')[0][:6]:>7}/{k.split('|')[1][:4]:<4}" for k in keys))
for r in rows:
    print(f"{r['name']:>24} {r['p_mean']:>7.3f} {r['p_spread']:>7.3f} | " +
          " ".join(f"{r[k]:>12.2f}" for k in keys))

print("\n=== mean over runs (MSE vs uniform|per_prompt; <1 better) ===")
for k in sorted(keys, key=lambda k: np.mean([r[k] for r in rows])):
    v = [r[k] for r in rows]
    print(f"  {k:28s} {np.mean(v):5.2f}x   (min {np.min(v):.2f}, max {np.max(v):.2f}, n={len(v)})")

print("\n=== per_prompt vs per_sample, paired by run ===")
for alloc in sorted({k.split("|")[0] for k in keys}):
    pp, ps = f"{alloc}|per_prompt", f"{alloc}|per_sample"
    if pp in keys and ps in keys:
        ratio = [r[ps] / r[pp] for r in rows]
        print(f"  {alloc:16s} per_sample is {np.mean(ratio):5.2f}x the MSE of per_prompt "
              f"(worse in {sum(x > 1 for x in ratio)}/{len(ratio)} runs)")
