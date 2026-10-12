"""Scaled dot-product & multi-head attention (Vaswani et al., 2017).

Pure-Python realization of the Transformer's attention mechanism: queries,
keys, values → softmax(QKᵀ/√d_k)V, paralleled across heads. Invariant: every
attention row sums to 1 (a proper probability distribution over positions).

Author: 晨星
"""
import math
from ..core import linalg as L


def softmax(M):
    out = []
    for row in M:
        m = max(row)
        e = [math.exp(x - m) for x in row]
        s = sum(e)
        out.append([v / s for v in e])
    return out


def scaled_dot_product_attention(Q, K, V):
    n = len(Q)
    dk = len(K[0])
    scores = L.matmul(Q, L.transpose(K))
    scores = [[s / math.sqrt(dk) for s in row] for row in scores]
    attn = softmax(scores)
    out = L.matmul(attn, V)
    return out, attn


def multi_head_attention(Q, K, V, n_heads):
    n = len(Q)
    d = len(Q[0])
    assert d % n_heads == 0, "d_model must be divisible by n_heads"
    dh = d // n_heads
    per_head = []
    for h in range(n_heads):
        sl = slice(h * dh, (h + 1) * dh)
        Qh = [row[sl] for row in Q]
        Kh = [row[sl] for row in K]
        Vh = [row[sl] for row in V]
        out, _ = scaled_dot_product_attention(Qh, Kh, Vh)
        per_head.append(out)
    # concatenate heads along the feature axis
    O = [[0.0] * d for _ in range(n)]
    for h in range(n_heads):
        for i in range(n):
            O[i][h * dh:(h + 1) * dh] = per_head[h][i]
    return O
