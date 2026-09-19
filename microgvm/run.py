"""Compare rollout allocators against a measurable ground truth.

  --mode estimator   The direct test. Measure the true gradient, then for each
                     allocator draw a fresh budget-C sample and measure how far
                     its estimator lands from truth. Repeated R times gives bias
                     and variance. No training, minutes on CPU.
  --mode train       RL training with one allocator, for the downstream curve.

Everything is compared at EQUAL BUDGET C = m * n_base. GVM's pilot pass is extra
and is reported separately as rollouts_pilot, because comparing on C alone
flatters it.
"""

import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from task import ModChainTask
from model import TinyLM
from measure import (ground_truth, prompt_gradients, prompt_contribution,
                     flat_grad, estimator_error)
from allocators import (uniform_alloc, gvm_alloc, vip_alloc, neyman_alloc,
                        PromptSuccessGP, alloc_stats)


def warmup(model, task, steps, lr, bs, seed=0, log=print):
    """Teacher-forced pretraining, the analogue of nanochat's SFT step.

    Without it every prompt sits at p=0 and no allocator has anything to work
    with -- the exact degeneracy that makes GSM8K useless at small scale. The
    number of steps is the knob that sets the accept-rate spread.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    model.train()
    for s in range(steps):
        idx = rng.integers(0, len(task), size=bs)
        x, y = task.batch(idx)
        xp, yp = torch.tensor(x), torch.tensor(y)
        seq = torch.cat([xp, yp.clamp(min=0)], dim=1)
        logits = model(seq[:, :-1])[:, -yp.shape[1]:, :]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yp.reshape(-1),
                               ignore_index=-1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if s % max(steps // 5, 1) == 0:
            log(f"  warmup {s}/{steps} loss {loss.item():.4f}")
    model.eval()
    return model


def draw(model, task, idx_set, n_alloc, temperature, gen):
    """Draw the allocated rollouts once and return each prompt's summed
    contribution. Both weightings are then formed from the SAME rollouts, which
    makes the comparison paired -- otherwise the difference between weightings
    would be partly sampling noise."""
    sums = []
    for i, n_i in zip(idx_set, n_alloc):
        if n_i <= 0:
            sums.append(None); continue
        g, _ = prompt_contribution(model, task, i, int(n_i), temperature, gen)
        sums.append(g)
    return sums


def weight(sums, n_alloc, weighting):
    m = len(n_alloc)
    total_n = int(sum(n_alloc))
    D = next(v for v in sums if v is not None).numel()
    est = torch.zeros(D)
    for v, n_i in zip(sums, n_alloc):
        if v is None:
            continue
        if weighting == "per_prompt":
            # Each prompt contributes its own mean, prompts averaged equally:
            # GVM Algorithm 1 line 8.
            est += v / (int(n_i) * m)
        else:
            # Flat mean over all rollouts, so a prompt's weight grows with the
            # budget it was given. This is what verl's token-mean does.
            est += v / total_n
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="estimator", choices=["estimator", "train"])
    ap.add_argument("--modulus", type=int, default=7)
    ap.add_argument("--max-k", type=int, default=4)
    ap.add_argument("--n-prompts", type=int, default=256)
    ap.add_argument("--batch-prompts", type=int, default=32, help="m, prompts per step")
    ap.add_argument("--n-base", type=int, default=8, help="budget C = m * n_base")
    ap.add_argument("--pilot", type=int, default=4, help="N', GVM pilot rollouts")
    ap.add_argument("--gt-rollouts", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=16)
    ap.add_argument("--warmup-steps", type=int, default=400)
    ap.add_argument("--warmup-lr", type=float, default=3e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="microgvm")
    ap.add_argument("--run", default="dummy")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")) or 1)

    use_wandb = args.run != "dummy"
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run, config=vars(args))

    def log(*a):
        print(*a, flush=True)

    task = ModChainTask(args.modulus, args.max_k, seed=args.seed, n_prompts=args.n_prompts)
    model = TinyLM(task.vocab_size, d=args.d, n_layer=args.n_layer,
                   max_len=task.prompt_len + args.max_k + 2)
    log(f"task: vocab={task.vocab_size} prompts={len(task)} | "
        f"model params={sum(p.numel() for p in model.parameters())/1e3:.0f}K")

    t0 = time.time()
    warmup(model, task, args.warmup_steps, args.warmup_lr, bs=64, seed=args.seed, log=log)
    log(f"warmup done in {time.time()-t0:.1f}s")

    idx_set = list(range(args.batch_prompts))
    m = len(idx_set)
    C = m * args.n_base

    log(f"measuring ground truth ({args.gt_rollouts} rollouts x {m} prompts)...")
    t0 = time.time()
    gt = ground_truth(model, task, idx_set, args.gt_rollouts, args.temperature, args.seed)
    log(f"  done in {time.time()-t0:.1f}s | p mean {gt['p'].mean():.3f} "
        f"spread {gt['p'].std():.3f} degenerate {(100*((gt['p']<=0)|(gt['p']>=1)).mean()):.0f}%")

    if args.mode != "estimator":
        log("train mode not yet implemented; use --mode estimator")
        return

    # VIP sees only features, never the measured p. Features are the prompt token
    # ids, which is the information a GP could plausibly exploit.
    feats = np.zeros((m, task.prompt_len))
    for r, i in enumerate(idx_set):
        pr = task.prompts[i]; feats[r, task.prompt_len - len(pr):] = pr
    gp = PromptSuccessGP(feats)
    for _ in range(8):                       # let the GP see a few rounds first
        gp.update(range(m), gt["p"])

    allocs = {
        "uniform": uniform_alloc(m, C),
        "gvm":     gvm_alloc(gt["p"], gt["G"], C),
        "vip":     vip_alloc(np.clip(gp.predict(range(m)), 1e-3, 1 - 1e-3), C, lo=3, hi=C),
        "neyman": neyman_alloc(gt["sigma"], C),
    }
    pilot_cost = {"uniform": 0, "gvm": args.pilot * m, "vip": 0, "neyman": args.gt_rollouts * m}

    results = {}
    gen = torch.Generator().manual_seed(args.seed + 999)
    R = args.repeats
    for name, n_alloc in allocs.items():
        ests = {"per_prompt": [], "per_sample": []}
        for _ in range(R):
            sums = draw(model, task, idx_set, n_alloc, args.temperature, gen)
            for w in ests:
                ests[w].append(weight(sums, n_alloc, w))
        for w, lst in ests.items():
            E = torch.stack(lst)
            var = E.var(0).sum().item()
            # ||mean - truth||^2 over R samples overestimates bias^2 by var/R,
            # which at small R can swamp the real bias. Subtract it.
            raw_bias2 = (E.mean(0) - gt["true_grad"]).pow(2).sum().item()
            bias2 = max(raw_bias2 - var / R, 0.0)
            key = f"{name}|{w}"
            results[key] = dict(
                bias2=bias2, variance=var, mse=bias2 + var,
                cos=float(np.mean([estimator_error(e, gt["true_grad"])["cos"] for e in lst])),
                rollouts_train=int(sum(n_alloc)), rollouts_pilot=int(pilot_cost[name]),
            )
            log(f"  {key:28s} mse {results[key]['mse']:.4g}  var {var:.4g}  "
                f"bias^2 {bias2:.4g}  cos {results[key]['cos']:.3f}")
            if use_wandb:
                import wandb
                wandb.log({f"{key}/{k}": v for k, v in results[key].items()})

    base = results["uniform|per_prompt"]["mse"]
    log("\n=== MSE relative to uniform+per_prompt ===")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["mse"]):
        log(f"  {k:28s} {v['mse']/base:6.2f}x   total rollouts "
            f"{v['rollouts_train']+v['rollouts_pilot']:>6d}")
    if use_wandb:
        import wandb
        wandb.log({"summary/" + k.replace("|", "_") + "_mse_rel": v["mse"] / base
                   for k, v in results.items()})
        wandb.finish()
    if args.out:
        payload = {k: v for k, v in results.items()}
        pp = gt["p"]
        payload["_meta"] = dict(p_mean=float(pp.mean()), p_spread=float(pp.std()),
                                # p=0 or p=1 means every advantage in that group is
                                # zero under mean-centring, so the prompt contributes
                                # nothing no matter how it is allocated.
                                p_frac_degenerate=float(((pp <= 0) | (pp >= 1)).mean()),
                                config=vars(args),
                                alloc={k: np.asarray(v).tolist() for k, v in allocs.items()})
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(payload, open(args.out, "w"), indent=2)
        log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
