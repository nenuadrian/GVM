"""Push the repo's offline avg@8 evaluation into wandb as a Table 1 row.

eval/aggregate.py leaves result/<name>/results.csv behind; this attaches those
numbers to the same wandb run the training wrote to, as summary fields and as a
row in a shared `table1` artifact-style table.
"""

import argparse
import hashlib
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="model_prefix used by eval/gen.py")
    ap.add_argument("--data", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    csv = args.csv or os.path.join("eval", "result", args.name, "results.csv")
    df = pd.read_csv(csv)

    scores = {str(r["dataset"]): float(r["accuracy"]) for _, r in df.iterrows()}
    summary = {f"final/{k}": 100.0 * v for k, v in scores.items()}
    # aggregate.py averages over whatever rows it has, so a math500-only eval
    # would report "5 average" == math500. Only pass it on when it means what
    # the paper's column means.
    if "5 average" in df.columns and len(scores) == 5:
        summary["final/5_average"] = 100.0 * float(df["5 average"].iloc[0])
    if "3 average" in df.columns and len(scores) >= 3:
        summary["final/3_average"] = 100.0 * float(df["3 average"].iloc[0])

    import wandb

    project = os.environ.get("WANDB_PROJECT", "gvm-qwen15-math500")
    # Reattach to the training run when this eval belongs to one; otherwise
    # (e.g. the base model) start a run keyed on the same naming scheme.
    run_id = os.environ.get("WANDB_RUN_ID") or hashlib.md5(
        f"{project}/{args.name}".encode()
    ).hexdigest()[:16]

    run = wandb.init(project=project, name=args.name, id=run_id, resume="allow")
    run.summary.update(summary)
    print(f"{args.name}: " + "  ".join(f"{k.split('/')[-1]}={v:.2f}" for k, v in summary.items()))
    run.finish()


if __name__ == "__main__":
    main()
