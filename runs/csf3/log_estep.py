"""Log one GVM E-step's budget allocation to the run's wandb timeline.

The E-step happens outside verl, so nothing about it would otherwise reach
wandb: accept rates, gradient norms and the resulting per-prompt sample sizes
are exactly the quantities Proposition 1 balances, so they are worth watching.
Reuses WANDB_RUN_ID/WANDB_RESUME from the environment to land on the same run
as the M-steps.
"""

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--em_dir", required=True)
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--world_size", type=int, required=True)
    ap.add_argument("--step", type=int, required=True)
    args = ap.parse_args()

    accept, grads = [], []
    for k in range(args.world_size):
        with open(os.path.join(args.em_dir, f"accept_rates_{k}.json")) as f:
            accept.extend(json.load(f))
        with open(os.path.join(args.em_dir, f"grads_{k}.json")) as f:
            grads.extend(json.load(f))
    with open(args.sizes) as f:
        sizes = json.load(f)

    accept = np.asarray(accept, dtype=float)
    grads = np.asarray(grads, dtype=float)
    sizes = np.asarray(sizes, dtype=float)

    data = {
        "estep/iteration": args.iter,
        "estep/num_prompts": len(accept),
        "estep/accept_rate_mean": float(accept.mean()),
        "estep/accept_rate_zero_frac": float((accept == 0).mean()),
        "estep/accept_rate_one_frac": float((accept == 1).mean()),
        "estep/grad_norm_mean": float(grads.mean()),
        "estep/budget_total": float(sizes.sum()),
        "estep/sample_size_mean": float(sizes.mean()),
        "estep/sample_size_max": float(sizes.max()),
        "estep/sample_size_p90": float(np.percentile(sizes, 90)),
        "estep/sample_size_zero_frac": float((sizes == 0).mean()),
    }

    import wandb

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        name=os.environ.get("WANDB_NAME"),
        id=os.environ.get("WANDB_RUN_ID"),
        resume="allow",
        group=os.environ.get("WANDB_RUN_GROUP"),
    )
    # Reproduces Figure 2: allocated budget against accept rate.
    table = wandb.Table(
        columns=["accept_rate", "sample_size"],
        data=[[float(a), float(s)] for a, s in zip(accept, sizes)],
    )
    data[f"estep/budget_vs_accept_iter{args.iter}"] = wandb.plot.scatter(
        table, "accept_rate", "sample_size", title=f"Sample size vs accept rate (iter {args.iter})"
    )
    run.log(data, step=args.step)
    run.finish()
    print(json.dumps({k: v for k, v in data.items() if isinstance(v, (int, float))}, indent=2))


if __name__ == "__main__":
    main()
