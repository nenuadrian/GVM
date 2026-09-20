#!/usr/bin/env python
"""Mean reward on held-out HH-RLHF prompts.

The headline number of the RAFT paper's HH-RLHF experiment is the reward the
aligned policy earns on prompts it was not trained on, measured by the same
reward model that ranked its samples during training. This reproduces that for
one checkpoint and logs it to wandb.

Sampling and post-processing deliberately mirror RaftAligner._get_batch_dataset_local
(temperature 0.85, top_k/top_p off, the same "###Human" truncation), so the
number is on the same scale as the raft/mean_reward curve from training.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def clean_text(text):
    """Verbatim RaftAligner._clean_text: keep only the first assistant turn."""
    if len(text) == 0:
        return text
    stext = [x for x in text.split("###Human") if x]
    return stext[0].strip().strip("#")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reward_model", required=True)
    ap.add_argument("--prompts", required=True, help="LMFlow text_only json")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_prompt_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True, help="base | sft | raft")
    ap.add_argument("--out", default=None)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda"

    prompts = [i["text"] for i in json.load(open(args.prompts))["instances"]]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"  # decoder-only batch generation

    # The aligner drops prompts over 256 tokens ("a long context window will lead
    # to a heavy burden on the GPU memory"); hold the eval set to the same rule.
    prompts = [p for p in prompts if len(tok.encode(p)) <= args.max_prompt_tokens][: args.n]
    print(f"{len(prompts)} eval prompts (<= {args.max_prompt_tokens} tokens)", flush=True)

    policy = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(dev).eval()

    rm_tok = AutoTokenizer.from_pretrained(args.reward_model, use_fast=False)
    rm_tok.pad_token = rm_tok.eos_token
    rm_tok.pad_token_id = rm_tok.eos_token_id
    rm_tok.padding_side = "left"
    rm = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model, torch_dtype=torch.bfloat16, num_labels=1
    ).to(dev).eval()

    gen_kwargs = dict(
        min_length=1,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tok.eos_token_id,
    )

    rewards, lengths, samples = [], [], []
    t0 = time.time()
    for s in range(0, len(prompts), args.batch_size):
        batch = prompts[s : s + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            out = policy.generate(**enc, **gen_kwargs)
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        responses = [clean_text(d.replace(p, "")) for d, p in zip(decoded, batch)]

        scored = [p + r for p, r in zip(batch, responses)]
        renc = rm_tok(
            scored, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(dev)
        with torch.no_grad():
            scores = rm(**renc).logits.squeeze(-1).float().cpu().numpy()

        rewards.extend(scores.tolist())
        lengths.extend(len(tok.encode(r)) for r in responses)
        if len(samples) < 8:
            samples.append({"prompt": batch[0], "response": responses[0], "reward": float(scores[0])})

        done = s + len(batch)
        if (s // args.batch_size) % 10 == 0:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done}/{len(prompts)}  mean_reward={np.mean(rewards):.4f}  "
                  f"{rate:.1f} prompt/s", flush=True)

    res = {
        "tag": args.tag,
        "model": args.model,
        "reward_model": args.reward_model,
        "n": len(rewards),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "sem_reward": float(np.std(rewards) / np.sqrt(len(rewards))),
        "median_reward": float(np.median(rewards)),
        "mean_response_tokens": float(np.mean(lengths)),
        "temperature": args.temperature,
        "seconds": time.time() - t0,
    }
    print(json.dumps(res, indent=2))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({**res, "rewards": rewards, "samples": samples}, f)
        print("wrote", args.out)

    if args.wandb:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "raft-lmflow"),
            name=os.environ.get("WANDB_NAME"),
            id=os.environ.get("WANDB_RUN_ID"),
            resume=os.environ.get("WANDB_RESUME", "allow"),
        )
        run.summary.update({f"eval/{args.tag}/{k}": v for k, v in res.items()
                            if isinstance(v, (int, float))})
        run.log({f"eval/{args.tag}/reward_hist": wandb.Histogram(rewards)})
        tbl = wandb.Table(columns=["prompt", "response", "reward"])
        for smp in samples:
            tbl.add_data(smp["prompt"], smp["response"], smp["reward"])
        run.log({f"eval/{args.tag}/samples": tbl})
        run.finish()


if __name__ == "__main__":
    main()
