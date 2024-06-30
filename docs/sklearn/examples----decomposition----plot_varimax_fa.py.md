# `D:\src\scipysrc\scikit-learn\examples\decomposition\plot_varimax_fa.py`

```
"""
===============================================================
Factor Analysis (with rotation) to visualize patterns
===============================================================

Investigating the Iris dataset, we see that sepal length, petal
length and petal width are highly correlated. Sepal width is
less redundant. Matrix decomposition techniques can uncover
these latent patterns. Applying rotations to the resulting
components does not inherently improve the predictive value
of the derived latent space, but can help visualise their
structure; here, for example, the varimax rotation, which
is found by maximizing the squared variances of the weights,
finds a structure where the second component only loads
positively on sepal width.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图
import numpy as np  # 导入 numpy 库用于数值计算

from sklearn.datasets import load_iris  # 导入 load_iris 函数用于加载数据集
from sklearn.decomposition import PCA, FactorAnalysis  # 导入 PCA 和 FactorAnalysis 类
from sklearn.preprocessing import StandardScaler  # 导入 StandardScaler 类用于数据标准化

# %%
# Load Iris data
data = load_iris()  # 加载鸢尾花数据集
X = StandardScaler().fit_transform(data["data"])  # 对数据集进行标准化处理
feature_names = data["feature_names"]  # 获取数据集特征名列表

# %%
# Plot covariance of Iris features
ax = plt.axes()  # 创建绘图坐标轴对象

im = ax.imshow(np.corrcoef(X.T), cmap="RdBu_r", vmin=-1, vmax=1)  # 绘制特征之间的协方差矩阵

ax.set_xticks([0, 1, 2, 3])  # 设置 X 轴刻度位置
ax.set_xticklabels(list(feature_names), rotation=90)  # 设置 X 轴刻度标签并旋转 90 度
ax.set_yticks([0, 1, 2, 3])  # 设置 Y 轴刻度位置
ax.set_yticklabels(list(feature_names))  # 设置 Y 轴刻度标签

plt.colorbar(im).ax.set_ylabel("$r$", rotation=0)  # 添加颜色条并设置标签
ax.set_title("Iris feature correlation matrix")  # 设置图表标题
plt.tight_layout()  # 调整布局使图表紧凑显示

# %%
# Run factor analysis with Varimax rotation
n_comps = 2  # 指定因子分析的成分数为 2

methods = [
    ("PCA", PCA()),  # 使用 PCA 方法进行主成分分析
    ("Unrotated FA", FactorAnalysis()),  # 使用因子分析但不进行旋转
    ("Varimax FA", FactorAnalysis(rotation="varimax")),  # 使用 Varimax 旋转的因子分析
]
fig, axes = plt.subplots(ncols=len(methods), figsize=(10, 8), sharey=True)  # 创建子图及其布局

for ax, (method, fa) in zip(axes, methods):
    fa.set_params(n_components=n_comps)  # 设置因子分析的成分数
    fa.fit(X)  # 在标准化后的数据上拟合因子分析模型

    components = fa.components_.T  # 获取因子分析后的成分并转置
    print("\n\n %s :\n" % method)
    print(components)  # 打印每种方法的成分

    vmax = np.abs(components).max()  # 计算成分的最大值
    ax.imshow(components, cmap="RdBu_r", vmax=vmax, vmin=-vmax)  # 绘制成分热图
    ax.set_yticks(np.arange(len(feature_names)))  # 设置 Y 轴刻度位置
    ax.set_yticklabels(feature_names)  # 设置 Y 轴刻度标签
    ax.set_title(str(method))  # 设置子图标题
    ax.set_xticks([0, 1])  # 设置 X 轴刻度位置
    ax.set_xticklabels(["Comp. 1", "Comp. 2"])  # 设置 X 轴刻度标签
fig.suptitle("Factors")  # 设置整体图表标题
plt.tight_layout()  # 调整布局使图表紧凑显示
plt.show()  # 显示图表
```