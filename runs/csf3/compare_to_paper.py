"""Print the replicated Table 1 next to the published one.

Reads whatever eval/result/<name>/results.csv files exist, so it is useful
part-way through the sweep as well as at the end.
"""

import argparse
import os

import pandas as pd

# Table 1, Qwen2.5-Math-1.5B block (arXiv:2505.02391).
PAPER = {
    "base":       {"math500": 56.35, "minerva_math": 17.00, "olympiad_bench": 25.22, "aime24": 3.33,  "amc23": 37.81, "5 average": 27.94},
    "grpo":       {"math500": 70.78, "minerva_math": 29.00, "olympiad_bench": 33.56, "aime24": 10.00, "amc23": 47.19, "5 average": 38.11},
    "raftpp":     {"math500": 69.02, "minerva_math": 27.71, "olympiad_bench": 31.74, "aime24": 9.58,  "amc23": 44.06, "5 average": 36.42},
    "gvm-grpo":   {"math500": 73.92, "minerva_math": 29.96, "olympiad_bench": 36.26, "aime24": 12.92, "amc23": 49.06, "5 average": 40.42},
    "gvm-raftpp": {"math500": 72.90, "minerva_math": 29.04, "olympiad_bench": 36.20, "aime24": 9.17,  "amc23": 51.88, "5 average": 39.64},
}
COLS = ["math500", "minerva_math", "olympiad_bench", "aime24", "amc23", "5 average"]


def read_run(result_dir, name):
    """Map an eval/result/<dir> back to the paper's row label."""
    csv = os.path.join(result_dir, name, "results.csv")
    if not os.path.isfile(csv):
        return None
    df = pd.read_csv(csv)
    got = {str(r["dataset"]): 100.0 * float(r["accuracy"]) for _, r in df.iterrows()}
    # Only a full five-benchmark run gives a "5 average" comparable to Table 1;
    # aggregate.py would otherwise average whatever subset was evaluated.
    if "5 average" in df.columns and len(got) == 5:
        got["5 average"] = 100.0 * float(df["5 average"].iloc[0])
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default="eval/result")
    args = ap.parse_args()

    rows = []
    for method in PAPER:
        # 04_eval.sbatch names the base model "base" and the others "<method>-qwen15".
        for candidate in (method, f"{method}-qwen15"):
            got = read_run(args.result_dir, candidate)
            if got:
                break
        for col in COLS:
            rows.append(
                {
                    "method": method,
                    "benchmark": col,
                    "paper": PAPER[method][col],
                    "ours": round(got[col], 2) if got and col in got else None,
                    "delta": round(got[col] - PAPER[method][col], 2) if got and col in got else None,
                }
            )

    df = pd.DataFrame(rows)
    for col in COLS:
        sub = df[df.benchmark == col].set_index("method")[["paper", "ours", "delta"]]
        print(f"\n=== {col} ===")
        print(sub.to_string(na_rep="  --  "))


if __name__ == "__main__":
    main()
