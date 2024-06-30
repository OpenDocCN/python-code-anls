# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_elastic_net_precomputed_gram_matrix_with_weighted_samples.py`

```
# %%
# Let's start by loading the dataset and creating some sample weights.
import numpy as np

from sklearn.datasets import make_regression

rng = np.random.RandomState(0)

n_samples = int(1e5)
# 生成具有线性关系的数据集，包括 X（特征）和 y（目标值）
X, y = make_regression(n_samples=n_samples, noise=0.5, random_state=rng)

# 使用对数正态分布生成样本权重
sample_weight = rng.lognormal(size=n_samples)
# 标准化样本权重，使其总和为 n_samples
normalized_weights = sample_weight * (n_samples / (sample_weight.sum()))

# %%
# To fit the elastic net using the `precompute` option together with the sample
# weights, we must first center the design matrix,  and rescale it by the
# normalized weights prior to computing the gram matrix.

# 计算特征矩阵 X 的加权平均值，以便进行中心化
X_offset = np.average(X, axis=0, weights=normalized_weights)
# 中心化设计矩阵 X
X_centered = X - np.average(X, axis=0, weights=normalized_weights)
# 对中心化后的设计矩阵 X 进行按照样本权重标准化处理
X_scaled = X_centered * np.sqrt(normalized_weights)[:, np.newaxis]
# 计算标准化后的设计矩阵 X 的 Gram 矩阵
gram = np.dot(X_scaled.T, X_scaled)

# %%
# We can now proceed with fitting. We must passed the centered design matrix to
# `fit` otherwise the elastic net estimator will detect that it is uncentered
# and discard the gram matrix we passed. However, if we pass the scaled design
# matrix, the preprocessing code will incorrectly rescale it a second time.

# 导入 ElasticNet 模型
from sklearn.linear_model import ElasticNet

# 创建 ElasticNet 模型的实例，使用预先计算的 Gram 矩阵
lm = ElasticNet(alpha=0.01, precompute=gram)
# 使用中心化后的设计矩阵 X_centered 和样本权重 normalized_weights 进行模型拟合
lm.fit(X_centered, y, sample_weight=normalized_weights)
```