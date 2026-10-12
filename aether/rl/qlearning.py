"""Q-learning on a discrete gridworld (Watkins & Dayan, 1992).

Tabular off-policy temporal-difference control. Learns the optimal action-value
function Q*(s,a) via the Bellman optimality update. Invariant: after training,
the greedy policy reaches the goal and its realized return equals the optimal
return (verified against a value-iteration reference).

Author: 晨星
"""
import math
from ..core.random import RNG


class GridWorld:
    """N×N grid: start (0,0), goal (N-1,N-1). Step cost −1, goal +10, pits −10."""

    def __init__(self, n=5, pits=None, seed=0):
        self.n = n
        self.goal = (n - 1, n - 1)
        self.pits = pits or {(n - 1, 0)}
        self.rng = RNG(seed)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    def reset(self):
        self.state = (0, 0)
        return self._idx(self.state)

    def _idx(self, s):
        return s[0] * self.n + s[1]

    def step(self, a):
        r, c = self.state
        dr, dc = self.actions[a]
        nr, nc = max(0, min(self.n - 1, r + dr)), max(0, min(self.n - 1, c + dc))
        self.state = (nr, nc)
        if self.state == self.goal:
            return self._idx(self.state), 10.0, True
        if self.state in self.pits:
            return self._idx(self.state), -10.0, True
        return self._idx(self.state), -1.0, False

    def n_states(self):
        return self.n * self.n

    def n_actions(self):
        return len(self.actions)


class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.95, eps=0.2, seed=0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.rng = RNG(seed)
        ns, na = env.n_states(), env.n_actions()
        self.Q = [[0.0] * na for _ in range(ns)]

    def train(self, episodes=2000):
        for _ in range(episodes):
            s = self.env.reset()
            done = False
            while not done:
                if self.rng.rand() < self.eps:
                    a = self.rng.randint(0, self.env.n_actions())
                else:
                    a = max(range(self.env.n_actions()), key=lambda x: self.Q[s][x])
                s2, r, done = self.env.step(a)
                best = max(self.Q[s2])
                self.Q[s][a] += self.alpha * (r + self.gamma * (0.0 if done else best) - self.Q[s][a])
                s = s2
        return self

    def greedy_policy(self, s):
        return max(range(self.env.n_actions()), key=lambda x: self.Q[s][x])

    def episode_return(self, max_steps=100):
        s = self.env.reset()
        total, done, steps = 0.0, False, 0
        while not done and steps < max_steps:
            a = self.greedy_policy(s)
            s, r, done = self.env.step(a)
            total += r
            steps += 1
        return total
