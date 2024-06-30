# `D:\src\scipysrc\scikit-learn\benchmarks\bench_plot_ward.py`

```
"""
Benchmark scikit-learn's Ward implement compared to SciPy's
"""

# 导入所需的库
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy  # 导入 SciPy 的层次聚类模块

from sklearn.cluster import AgglomerativeClustering  # 导入 scikit-learn 的凝聚聚类模块

# 创建 AgglomerativeClustering 对象，使用 Ward 方法，指定聚类数为 3
ward = AgglomerativeClustering(n_clusters=3, linkage="ward")

# 定义生成数据的样本数和特征数的范围
n_samples = np.logspace(0.5, 3, 9)
n_features = np.logspace(1, 3.5, 7)
N_samples, N_features = np.meshgrid(n_samples, n_features)

# 初始化用于存储 scikit-learn 和 scipy 的执行时间的数组
scikits_time = np.zeros(N_samples.shape)
scipy_time = np.zeros(N_samples.shape)

# 循环遍历不同的样本数和特征数组合
for i, n in enumerate(n_samples):
    for j, p in enumerate(n_features):
        # 生成服从正态分布的随机数据矩阵 X
        X = np.random.normal(size=(n, p))
        
        # 使用 scikit-learn 的 Ward 聚类方法并记录执行时间
        t0 = time.time()
        ward.fit(X)
        scikits_time[j, i] = time.time() - t0
        
        # 使用 SciPy 的 Ward 方法并记录执行时间
        t0 = time.time()
        hierarchy.ward(X)
        scipy_time[j, i] = time.time() - t0

# 计算 scikit-learn 执行时间与 scipy 执行时间的比率
ratio = scikits_time / scipy_time

# 绘制比率的对数图像
plt.figure("scikit-learn Ward's method benchmark results")
plt.imshow(np.log(ratio), aspect="auto", origin="lower")
plt.colorbar()
plt.contour(
    ratio,
    levels=[
        1,
    ],
    colors="k",
)
plt.yticks(range(len(n_features)), n_features.astype(int))
plt.ylabel("N features")
plt.xticks(range(len(n_samples)), n_samples.astype(int))
plt.xlabel("N samples")
plt.title("Scikit's time, in units of scipy time (log)")
plt.show()
```