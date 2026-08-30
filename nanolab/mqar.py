"""Multi-query associative recall (MQAR), the metric axis of the E8 backlog item.

Held-out CE at 512 tokens barely exercises in-context recall -- the documented
failure mode of recurrent mixers and the stated reason hybrids keep attention
layers at all (Arora et al., "Zoology: Measuring and Improving Recall in
Efficient Language Models", arXiv:2312.04927). PAPER section 4.5's tie between
`attention` and `hybrid_mingru10_attn2` was measured at one metric; this module
supplies a second one.

A sequence is a list of key-value pairs followed by queries::

    k3 v7  k1 v2  k5 v9   |   k1 v2   k5 v9
    <------ pairs ----->      <-- queries -->

Under next-token prediction the model sees ``k1`` at a query position and must
emit ``v2``. Every other position is masked to ``ignore_index`` (-1), which both
loss paths in ``model.py`` already honour -- so the training loss IS recall loss,
with no change to the model or the training loop.

Keys are unique within a sequence, so each query has exactly one right answer and
the task is well posed rather than merely hard. `recall_accuracy` reports exact
match at the query positions, which is the number the Zoology line of work
reports and is not recoverable from CE.

UNTIE THE EMBEDDINGS. With ``tie_embeddings=True`` -- the default, and what every
suite in PAPER section 4 runs -- attention caps at 0.555 on this task and no
amount of training moves it (0.575 at 3k steps, 0.566 at 8k). Untied it reaches
0.990 under an otherwise identical config. Measured 2026-08-27 at P=4/Q=4/K=16/
V=16, d=256, 4 layers; `qk_norm`, `zero_init_proj` and full-width RoPE were
ablated in the same sweep and moved nothing (0.504-0.557).

The mechanism: under tying the readout matrix IS the input embedding, so the
residual stream at a query position -- which carries ``embed(k)`` -- projects
onto token ``k`` itself. Emitting ``v`` means cancelling that first, and at this
depth the model settles for a partial solution instead.

This is a property of the probe worth reporting rather than a knob to set
quietly: with the board's own default, the recall metric cannot separate the
arms, because the reference arm is already at a ceiling that has nothing to do
with recall.
"""
from __future__ import annotations

import torch

IGNORE = -1          # matches model.py's cross-entropy ignore_index


def vocab_for(n_keys: int, n_values: int) -> int:
    """Token ids: 0 unused, then keys, then values."""
    return 1 + n_keys + n_values


class MQARBatcher:
    """Batcher-compatible MQAR sampler.

    Exposes the same ``batch()`` / ``iterator()`` contract as ``data.Batcher``,
    so ``train.train`` and ``train.evaluate`` consume it unchanged. Sequence
    length is ``2 * (n_pairs + n_queries)`` and is a property of the task, not a
    free parameter -- a caller asking for a block size the task does not produce
    is a misconfiguration, not something to pad around silently.
    """

    def __init__(self, cfg, device, split: str = "train", *, n_pairs=None,
                 n_queries=None, n_keys=None, n_values=None):
        self.n_pairs = int(n_pairs if n_pairs is not None else cfg.mqar_n_pairs)
        self.n_queries = int(n_queries if n_queries is not None
                             else cfg.mqar_n_queries)
        self.n_keys = int(n_keys if n_keys is not None else cfg.mqar_n_keys)
        self.n_values = int(n_values if n_values is not None else cfg.mqar_n_values)
        if self.n_queries > self.n_pairs:
            raise ValueError(
                f"n_queries ({self.n_queries}) > n_pairs ({self.n_pairs}): a query "
                "with no stored pair has no right answer")
        if self.n_pairs > self.n_keys:
            raise ValueError(
                f"n_pairs ({self.n_pairs}) > n_keys ({self.n_keys}): keys must be "
                "unique within a sequence or a query has two right answers")
        self.batch_size = cfg.batch_size
        self.device = device
        self.seq_len = 2 * (self.n_pairs + self.n_queries)
        # x is seq[:-1] and y is seq[1:], so a T-token model needs T+1 raw tokens.
        if cfg.block_size != self.seq_len - 1:
            raise ValueError(
                f"block_size {cfg.block_size} does not match the MQAR sequence "
                f"({self.n_pairs} pairs + {self.n_queries} queries needs "
                f"block_size {self.seq_len - 1}). Set it from the task.")
        self.vocab_size = vocab_for(self.n_keys, self.n_values)
        self.gen = torch.Generator(device="cpu").manual_seed(
            cfg.seed + (0 if split == "train" else 1))

    def __len__(self) -> int:
        return self.batch_size * self.seq_len

    def _key(self, i):    # token id of key i
        return 1 + i

    def _value(self, i):  # token id of value i
        return 1 + self.n_keys + i

    def batch(self, block_size=None, frontier=1.0):
        """(x, y) with y masked to IGNORE everywhere but the query positions.

        ``block_size`` and ``frontier`` exist for Batcher compatibility. The
        curricula they drive are properties of a text corpus; refusing a
        non-default value beats silently ignoring it.
        """
        if block_size is not None and block_size != self.seq_len - 1:
            raise ValueError("MQAR sequence length is fixed by the task")
        if frontier != 1.0:
            raise ValueError("MQAR is generated, not sorted; frontier is meaningless")
        b, p, q = self.batch_size, self.n_pairs, self.n_queries

        # distinct keys per row, and an independently drawn value per pair
        keys = torch.stack([torch.randperm(self.n_keys, generator=self.gen)[:p]
                            for _ in range(b)])
        vals = torch.randint(self.n_values, (b, p), generator=self.gen)
        # which pairs get queried, in a random order, without replacement
        pick = torch.stack([torch.randperm(p, generator=self.gen)[:q]
                            for _ in range(b)])

        seq = torch.empty(b, self.seq_len, dtype=torch.long)
        seq[:, 0:2 * p:2] = self._key(keys)
        seq[:, 1:2 * p:2] = self._value(vals)
        qk = torch.gather(keys, 1, pick)
        qv = torch.gather(vals, 1, pick)
        seq[:, 2 * p::2] = self._key(qk)
        seq[:, 2 * p + 1::2] = self._value(qv)

        x = seq[:, :-1].contiguous()
        y = torch.full_like(x, IGNORE)
        # y[t] supervises the token after x[t]; the answers sit at odd offsets
        # into the query block, so their predicting positions are the even ones.
        ans_at = torch.arange(2 * p, self.seq_len - 1, 2)
        y[:, ans_at] = seq[:, ans_at + 1]

        if str(self.device).startswith(("cuda", "mps")):
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def iterator(self):
        while True:
            yield self.batch()


@torch.no_grad()
def recall_accuracy(model, batcher: MQARBatcher, ctx, iters: int = 20) -> float:
    """Exact-match accuracy at the query positions.

    The metric the recall literature reports, and the one CE cannot stand in
    for: a model can lower CE by sharpening the value marginal while never
    retrieving the right value for the queried key.
    """
    was_training = model.training
    model.eval()
    right = total = 0
    for _ in range(iters):
        x, y = batcher.batch()
        with ctx:
            logits, _ = model(x, y)
        if logits is None:
            raise RuntimeError(
                "recall_accuracy needs logits; run with cfg.fused_ce disabled "
                "(the fused path returns loss only and cannot be scored)")
        mask = y != IGNORE
        pred = logits.argmax(-1)
        right += int((pred[mask] == y[mask]).sum())
        total += int(mask.sum())
    if was_training:
        model.train()
    if not total:
        raise RuntimeError("no query positions scored; the batch carried no answers")
    return right / total
