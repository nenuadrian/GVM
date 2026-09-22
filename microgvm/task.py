"""Modular arithmetic chains: a CoT task small enough to measure exactly.

Prompt:  a1 op a2 op ... op ak =
Target:  the k running partial results, the last of which is the answer.

    3 + 4 * 2 =   (mod 10)  ->  7 4        (3+4=7, then 7*2=14=4)

Two knobs that move INDEPENDENTLY, which is the whole point of choosing this over
something like sequence reversal:

    k          number of operations -> output length
    op mix     multiplication is harder than addition at the same k

In reversal, difficulty and output length are the same variable, so a length
effect and a difficulty effect are indistinguishable. Here you can hold k fixed
and vary difficulty, which is what isolates GVM's G_i-as-length-proxy problem
(its gradient norm sums over response tokens, so it grows with output length).

Output length is exactly k, so there is no EOS handling and no truncation.
"""

import numpy as np

ADD, MUL = 0, 1


class ModChainTask:
    def __init__(self, modulus=7, max_k=4, seed=0, n_prompts=256, mul_prob=0.5):
        self.p = modulus
        self.max_k = max_k
        self.mul_prob = mul_prob
        rng = np.random.default_rng(seed)
        # vocab: digits 0..p-1, then '+', '*', '='
        self.PLUS, self.STAR, self.EQ = self.p, self.p + 1, self.p + 2
        self.vocab_size = self.p + 3

        self.prompts, self.targets, self.ks = [], [], []
        for _ in range(n_prompts):
            k = int(rng.integers(1, max_k + 1))
            operands = rng.integers(0, self.p, size=k + 1)
            ops = (rng.random(k) < mul_prob).astype(int)
            toks, acc, steps = [int(operands[0])], int(operands[0]), []
            for j in range(k):
                toks.append(self.STAR if ops[j] == MUL else self.PLUS)
                toks.append(int(operands[j + 1]))
                acc = (acc * int(operands[j + 1]) if ops[j] == MUL
                       else acc + int(operands[j + 1])) % self.p
                steps.append(acc)
            toks.append(self.EQ)
            self.prompts.append(toks)
            self.targets.append(steps)          # length k
            self.ks.append(k)
        self.prompt_len = max(len(t) for t in self.prompts)

    def __len__(self):
        return len(self.prompts)

    def batch(self, idx):
        """Left-padded prompts + targets for a set of prompt indices."""
        idx = list(idx)
        maxp = max(len(self.prompts[i]) for i in idx)
        maxk = max(len(self.targets[i]) for i in idx)
        # Left pad so every prompt ends at the same position; generation then
        # starts at a single shared offset and no per-row bookkeeping is needed.
        x = np.full((len(idx), maxp), self.EQ, dtype=np.int64)
        for r, i in enumerate(idx):
            pr = self.prompts[i]
            x[r, maxp - len(pr):] = pr
        y = np.full((len(idx), maxk), -1, dtype=np.int64)
        for r, i in enumerate(idx):
            t = self.targets[i]
            y[r, :len(t)] = t
        return x, y

    def reward(self, idx, generated):
        """1 if the FINAL partial result is right. Intermediate steps are free,
        exactly as in a verifier-scored CoT setting."""
        out = np.zeros(len(idx), dtype=np.float64)
        for r, i in enumerate(idx):
            k = len(self.targets[i])
            out[r] = float(generated[r, k - 1] == self.targets[i][k - 1])
        return out
