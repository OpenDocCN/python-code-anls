# `D:\src\scipysrc\scikit-learn\examples\covariance\plot_sparse_cov.py`

```
# %%
# 生成数据
# -----------------
# 导入所需的库
import numpy as np  # 导入NumPy库，用于数值计算
from scipy import linalg  # 导入SciPy库中的linalg模块，用于线性代数运算

from sklearn.datasets import make_sparse_spd_matrix  # 导入make_sparse_spd_matrix函数，用于生成稀疏正定矩阵

# 定义数据维度和样本数
n_samples = 60  # 样本数
n_features = 20  # 特征数

prng = np.random.RandomState(1)  # 创建一个随机数生成器对象，用于重现随机结果
# 生成一个稀疏正定矩阵，参数包括特征数、稀疏度alpha、最小系数、最大系数等
prec = make_sparse_spd_matrix(
    n_features, alpha=0.98, smallest_coef=0.4, largest_coef=0.7, random_state=prng
)
cov = linalg.inv(prec)  # 计算该稀疏正定矩阵的逆矩阵（协方差矩阵）
# %%
# Estimate the covariance
# -----------------------
# 导入所需的类和函数
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf

# 计算经验协方差矩阵
emp_cov = np.dot(X.T, X) / n_samples

# 创建 GraphicalLassoCV 模型并拟合数据
model = GraphicalLassoCV()
model.fit(X)
cov_ = model.covariance_
prec_ = model.precision_

# 使用 Ledoit-Wolf 方法估计协方差矩阵
lw_cov_, _ = ledoit_wolf(X)
lw_prec_ = linalg.inv(lw_cov_)

# %%
# Plot the results
# ----------------
# 导入 matplotlib.pyplot 库
import matplotlib.pyplot as plt

# 创建一个 10x6 大小的图形
plt.figure(figsize=(10, 6))
# 调整子图布局，左右边距分别为 0.02 和 0.98
plt.subplots_adjust(left=0.02, right=0.98)

# 绘制协方差矩阵
covs = [
    ("Empirical", emp_cov),
    ("Ledoit-Wolf", lw_cov_),
    ("GraphicalLassoCV", cov_),
    ("True", cov),
]
vmax = cov_.max()
for i, (name, this_cov) in enumerate(covs):
    plt.subplot(2, 4, i + 1)
    # 显示协方差矩阵的热图
    plt.imshow(
        this_cov, interpolation="nearest", vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r
    )
    plt.xticks(())
    plt.yticks(())
    plt.title("%s covariance" % name)

# 绘制精度矩阵
precs = [
    ("Empirical", linalg.inv(emp_cov)),
    ("Ledoit-Wolf", lw_prec_),
    ("GraphicalLasso", prec_),
    ("True", prec),
]
vmax = 0.9 * prec_.max()
for i, (name, this_prec) in enumerate(precs):
    ax = plt.subplot(2, 4, i + 5)
    # 显示精度矩阵的热图，用 masked array 屏蔽为 0 的元素
    plt.imshow(
        np.ma.masked_equal(this_prec, 0),
        interpolation="nearest",
        vmin=-vmax,
        vmax=vmax,
        cmap=plt.cm.RdBu_r,
    )
    plt.xticks(())
    plt.yticks(())
    plt.title("%s precision" % name)
    if hasattr(ax, "set_facecolor"):
        ax.set_facecolor(".7")
    else:
        ax.set_axis_bgcolor(".7")

# %%

# 绘制模型选择指标
plt.figure(figsize=(4, 3))
# 创建子图
plt.axes([0.2, 0.15, 0.75, 0.7])
# 绘制 alpha 参数与交叉验证分数的关系图
plt.plot(model.cv_results_["alphas"], model.cv_results_["mean_test_score"], "o-")
plt.axvline(model.alpha_, color=".5")
plt.title("Model selection")
plt.ylabel("Cross-validation score")
plt.xlabel("alpha")

# 显示图形
plt.show()
```