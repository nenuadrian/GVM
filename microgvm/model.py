"""A transformer small enough that the true gradient is computable.

~500K parameters at the defaults, which is the point: with a model this size and
a fixed prompt set you can draw enough rollouts to treat the resulting gradient
as ground truth, and measure every estimator's bias and variance against it.
That measurement is what the GVM paper never makes and what a 1.5B model on
Numina makes intractable.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab_size, d=128, n_layer=2, n_head=4, max_len=64):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.max_len = max_len
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_len, f"sequence {T} exceeds max_len {self.max_len}"
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def sample(self, prompt, n_new, temperature=1.0, generator=None):
        """Sample n_new tokens. Output length is fixed by the task (= k), so
        there is no EOS logic and every row in the batch stays aligned."""
        seq = prompt
        for _ in range(n_new):
            logits = self(seq)[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
            seq = torch.cat([seq, nxt], dim=1)
        return seq[:, prompt.shape[1]:]

    def logprobs(self, prompt, completion):
        """Per-token log p(completion | prompt). Shape (B, n_new)."""
        seq = torch.cat([prompt, completion], dim=1)
        logits = self(seq[:, :-1])
        lp = F.log_softmax(logits, dim=-1)
        n = completion.shape[1]
        lp = lp[:, -n:, :]
        return lp.gather(-1, completion.unsqueeze(-1)).squeeze(-1)
