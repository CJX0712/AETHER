"""AETHER — 世界顶级自校验 AI 核心。

纯 Python 零依赖，站在世界顶级数学/算法之上：Jacobi 特征值、PCG/splitmix64、
SMO、EM、反向传播、缩放点积注意力。每个模块以解析不变量自测。

作者：晨星
"""
__version__ = "1.0.0"
__author__ = "晨星"

from .core import linalg, optim, random
from .ml import linear, cluster, pca, tree, svm
from .deep import nn, attention
from .rl import qlearning

__all__ = [
    "linalg", "optim", "random",
    "linear", "cluster", "pca", "tree", "svm",
    "nn", "attention", "qlearning",
]
