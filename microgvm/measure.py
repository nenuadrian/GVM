"""Ground-truth gradient, per-prompt variance, and exact estimator diagnostics.

This is what the testbed exists for. At 1.5B parameters you can only validate an
allocator by downstream accuracy, which confounds variance reduction with every
other effect. At 400K you can draw enough rollouts to treat the resulting
gradient as truth and measure each estimator's bias and variance against it
directly.
"""

import numpy as np
import torch


def flat_grad(model, loss, retain=False):
    g = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad],
                            retain_graph=retain, allow_unused=True)
    parts = [(torch.zeros_like(p) if gi is None else gi).reshape(-1)
             for gi, p in zip(g, [p for p in model.parameters() if p.requires_grad])]
    return torch.cat(parts)


def prompt_contribution(model, task, idx, n_rollouts, temperature=1.0, generator=None):
    """Summed gradient contribution for one prompt, in a SINGLE backward pass.

    sum_j a_j * grad log p(y_j) == grad [ sum_j a_j * log p(y_j) ] because the
    advantages a_j are constants. Forming the sum inside the graph turns n_i
    backward passes into one, which is ~8x for a typical allocation and changes
    nothing about the result.

    Only used where the SUM is wanted. Estimating per-prompt noise sigma_i needs
    the individual gradients, so ground_truth() still pays the per-rollout cost.
    """
    x, _ = task.batch([idx])
    k = len(task.targets[idx])
    prompt = torch.tensor(x).repeat(n_rollouts, 1)
    comp = model.sample(prompt, n_new=k, temperature=temperature, generator=generator)
    rewards = task.reward([idx] * n_rollouts, comp.numpy())
    adv = torch.tensor(rewards - rewards.mean(), dtype=torch.float32)
    lp = model.logprobs(prompt, comp).sum(dim=1)          # (n,)
    surrogate = (adv * lp).sum()
    g = flat_grad(model, surrogate).detach()
    model.zero_grad(set_to_none=True)
    return g, rewards


def prompt_gradients(model, task, idx, n_rollouts, temperature=1.0, generator=None):
    """Per-rollout gradient contributions for one prompt.

    Returns (grads [n, D], rewards [n]). The contribution of a single rollout to
    the REINFORCE objective is (r - baseline) * grad log p(y|x); the baseline is
    applied by the caller so the same rollouts can be reused across estimators.
    """
    x, _ = task.batch([idx])
    k = len(task.targets[idx])
    prompt = torch.tensor(x).repeat(n_rollouts, 1)
    comp = model.sample(prompt, n_new=k, temperature=temperature, generator=generator)
    rewards = task.reward([idx] * n_rollouts, comp.numpy())

    grads = []
    for j in range(n_rollouts):
        lp = model.logprobs(prompt[j:j + 1], comp[j:j + 1]).sum()
        grads.append(flat_grad(model, lp).detach())
        model.zero_grad(set_to_none=True)
    return torch.stack(grads), rewards


def ground_truth(model, task, idx_set, n_rollouts=256, temperature=1.0, seed=0):
    """Reference gradient and per-prompt statistics.

    true_grad   mean over prompts of each prompt's mean centred contribution --
                the quantity every estimator is trying to approximate, with each
                prompt weighted equally (GVM Algorithm 1 line 8).
    sigma_i     std of prompt i's single-rollout contribution. Feeds the Neyman
                oracle: minimising sum sigma_i^2/n_i gives n_i proportional to sigma_i.
    """
    gen = torch.Generator().manual_seed(seed)
    per_prompt_mean, sigma, p_hat, g_norm = [], [], [], []
    for i in idx_set:
        g, r = prompt_gradients(model, task, i, n_rollouts, temperature, gen)
        adv = torch.tensor(r - r.mean(), dtype=g.dtype)      # mean-centred, as in the RL loop
        contrib = g * adv[:, None]
        per_prompt_mean.append(contrib.mean(0))
        sigma.append(contrib.std(0).norm().item())           # scalar scale of the noise
        p_hat.append(float(r.mean()))
        # GVM's G_i: mean over ACCEPTED rollouts of ||grad log p||
        acc = g[torch.tensor(r > 0)]
        g_norm.append(float(acc.norm(dim=1).mean().item()) if len(acc) else 0.0)
    true_grad = torch.stack(per_prompt_mean).mean(0)
    return dict(true_grad=true_grad,
                sigma=np.asarray(sigma),
                p=np.asarray(p_hat),
                G=np.asarray(g_norm))


def estimator_error(est_grad, true_grad):
    d = est_grad - true_grad
    return dict(l2=float(d.norm().item()),
                rel=float((d.norm() / true_grad.norm().clamp(min=1e-12)).item()),
                cos=float(torch.nn.functional.cosine_similarity(
                    est_grad[None], true_grad[None]).item()))


def logit_grad_scale(model, task, idx, n_rollouts, temperature=1.0, generator=None,
                     accepted_only=True):
    """G_i from the forward pass alone, via the Fisher identity.

    For a categorical over logits z with probabilities q,

        d log q_a / d z_b = delta_ab - q_b   =>   grad = e_a - q
        E_{a~q} ||grad||^2 = 1 - ||q||^2

    so the expected squared gradient norm at the logit layer is one minus the
    collision probability, available from the forward pass with NO backward pass
    and no sampling of the gradient. Summed over response positions this is a
    per-sequence scale; GVM's G_i is a norm, so the sqrt is taken.

    This is the exact second moment, where GVM estimates a MEAN of norms from N'
    samples -- which is also the wrong moment (Section 3.1 defines G_i^2 as an
    expected square, Algorithm 2 averages norms). That Jensen gap disappears here.

    Two honest differences from GVM's G_i:
      - logit space, not parameter space. GVM already approximates by using only
        embed_tokens, so this is a question of which approximation is better.
      - the expectation is over the model's own distribution at each position,
        whereas GVM averages over ACCEPTED rollouts. accepted_only=True restricts
        the outer average to accepted sequences to stay comparable.

    Returns (G_logit, rewards, saturation) where saturation is the mean collision
    probability: if it is near 0 the quantity pins at 1 for every prompt and
    cannot discriminate, which is the main way this idea could fail.
    """
    import torch.nn.functional as F
    x, _ = task.batch([idx])
    k = len(task.targets[idx])
    prompt = torch.tensor(x).repeat(n_rollouts, 1)
    comp = model.sample(prompt, n_new=k, temperature=temperature, generator=generator)
    rewards = task.reward([idx] * n_rollouts, comp.numpy())

    with torch.no_grad():
        seq = torch.cat([prompt, comp], dim=1)
        logits = model(seq[:, :-1])[:, -k:, :]
        q = F.softmax(logits, dim=-1)
        collision = q.pow(2).sum(-1)                  # (n, k)
        per_seq = (1.0 - collision).sum(-1)           # (n,) = E||grad_z log p||^2

    mask = rewards > 0 if accepted_only else np.ones(n_rollouts, dtype=bool)
    if not mask.any():
        return 0.0, rewards, float(collision.mean().item())
    G = float(per_seq[torch.tensor(mask)].sqrt().mean().item())
    return G, rewards, float(collision.mean().item())
