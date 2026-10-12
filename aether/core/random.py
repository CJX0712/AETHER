"""World-class seeded PRNG for fully reproducible AI experiments.

Reuses Vigna's splitmix64 (2013) — a modern standard generator used to seed
stronger families (PCG/xoshiro). Zero third-party dependency, deterministic
across every Python 3 runtime, so every benchmark in AETHER is reproducible.

Author: 晨星
"""
import math

_MASK64 = (1 << 64) - 1


class RNG:
    """splitmix64-based generator with uniform / normal / integer helpers."""

    def __init__(self, seed: int = 0):
        self.s = (seed ^ 0x9E3779B97F4A7C15) & _MASK64

    def next_u64(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & _MASK64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return (z ^ (z >> 31)) & _MASK64

    def rand(self) -> float:
        return self.next_u64() / 2 ** 64

    def randn(self) -> float:
        # Box-Muller transform (G. E. P. Box & M. E. Muller, 1958).
        u1 = max(self.rand(), 1e-12)
        u2 = self.rand()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)

    def randint(self, lo: int, hi: int) -> int:
        return lo + self.next_u64() % (hi - lo)

    def choice(self, n: int) -> int:
        return self.next_u64() % n

    def shuffle(self, arr):
        # Fisher-Yates (R. A. Fisher & F. Yates, 1938).
        for i in range(len(arr) - 1, 0, -1):
            j = self.next_u64() % (i + 1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr


def seeded(seed: int = 0) -> RNG:
    return RNG(seed)
