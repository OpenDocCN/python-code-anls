# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_omp.py`

```
"""
===========================
Orthogonal Matching Pursuit
===========================

Using orthogonal matching pursuit for recovering a sparse signal from a noisy
measurement encoded with a dictionary

"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 用于绘图
import numpy as np  # 导入 numpy 进行数值计算

from sklearn.datasets import make_sparse_coded_signal  # 导入数据生成函数
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV  # 导入正交匹配追踪算法

n_components, n_features = 512, 100  # 设置数据维度和特征数量
n_nonzero_coefs = 17  # 稀疏信号中非零系数的数量

# 生成数据

# y = Xw
# |x|_0 = n_nonzero_coefs

y, X, w = make_sparse_coded_signal(
    n_samples=1,
    n_components=n_components,
    n_features=n_features,
    n_nonzero_coefs=n_nonzero_coefs,
    random_state=0,
)
X = X.T

(idx,) = w.nonzero()  # 找到稀疏向量 w 中非零元素的索引

# 扭曲干净信号
y_noisy = y + 0.05 * np.random.randn(len(y))  # 添加噪声到信号中

# 绘制稀疏信号
plt.figure(figsize=(7, 7))
plt.subplot(4, 1, 1)
plt.xlim(0, 512)
plt.title("Sparse signal")
plt.stem(idx, w[idx])

# 绘制无噪声重建信号
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
omp.fit(X, y)
coef = omp.coef_
(idx_r,) = coef.nonzero()
plt.subplot(4, 1, 2)
plt.xlim(0, 512)
plt.title("Recovered signal from noise-free measurements")
plt.stem(idx_r, coef[idx_r])

# 绘制带噪声重建信号
omp.fit(X, y_noisy)
coef = omp.coef_
(idx_r,) = coef.nonzero()
plt.subplot(4, 1, 3)
plt.xlim(0, 512)
plt.title("Recovered signal from noisy measurements")
plt.stem(idx_r, coef[idx_r])

# 使用交叉验证设置非零系数数量的带噪声重建信号
omp_cv = OrthogonalMatchingPursuitCV()
omp_cv.fit(X, y_noisy)
coef = omp_cv.coef_
(idx_r,) = coef.nonzero()
plt.subplot(4, 1, 4)
plt.xlim(0, 512)
plt.title("Recovered signal from noisy measurements with CV")
plt.stem(idx_r, coef[idx_r])

# 调整子图布局和全局标题
plt.subplots_adjust(0.06, 0.04, 0.94, 0.90, 0.20, 0.38)
plt.suptitle("Sparse signal recovery with Orthogonal Matching Pursuit", fontsize=16)
plt.show()
```