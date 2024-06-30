# `D:\src\scipysrc\scipy\scipy\cluster\__init__.py`

```
"""
=========================================
Clustering package (:mod:`scipy.cluster`)
=========================================

.. currentmodule:: scipy.cluster

.. toctree::
   :hidden:

   cluster.vq
   cluster.hierarchy

Clustering algorithms are useful in information theory, target detection,
communications, compression, and other areas. The `vq` module only
supports vector quantization and the k-means algorithms.

The `hierarchy` module provides functions for hierarchical and
agglomerative clustering.  Its features include generating hierarchical
clusters from distance matrices,
calculating statistics on clusters, cutting linkages
to generate flat clusters, and visualizing clusters with dendrograms.

"""

# 将模块中公开的符号限制为 'vq' 和 'hierarchy'
__all__ = ['vq', 'hierarchy']

# 从当前目录导入 'vq' 和 'hierarchy' 模块
from . import vq, hierarchy

# 导入用于测试的 PytestTester 类
from scipy._lib._testutils import PytestTester

# 创建一个测试器对象，用于当前模块的测试
test = PytestTester(__name__)

# 删除不再需要的 PytestTester 类，避免污染命名空间
del PytestTester
```