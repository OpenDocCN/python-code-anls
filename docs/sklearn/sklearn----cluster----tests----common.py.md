# `D:\src\scipysrc\scikit-learn\sklearn\cluster\tests\common.py`

```
"""
Common utilities for testing clustering.

"""

import numpy as np

###############################################################################
# Generate sample data

# 生成聚类测试数据的函数
def generate_clustered_data(
    seed=0, n_clusters=3, n_features=2, n_samples_per_cluster=20, std=0.4
):
    # 使用给定的种子创建随机数生成器对象
    prng = np.random.RandomState(seed)

    # 数据被故意偏移离零点，用来检查聚类算法对于非中心化数据的鲁棒性
    # 设置每个聚类的均值
    means = (
        np.array(
            [
                [1, 1, 1, 0],
                [-1, -1, 0, 1],
                [1, -1, 1, 1],
                [-1, 1, 1, 0],
            ]
        )
        + 10
    )

    # 创建一个空的数组用来存放生成的数据点
    X = np.empty((0, n_features))
    for i in range(n_clusters):
        # 将每个聚类的均值加上随机扰动，生成该聚类的数据点
        X = np.r_[
            X,
            means[i][:n_features] + std * prng.randn(n_samples_per_cluster, n_features),
        ]
    return X
```