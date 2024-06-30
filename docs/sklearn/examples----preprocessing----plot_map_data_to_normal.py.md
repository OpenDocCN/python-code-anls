# `D:\src\scipysrc\scikit-learn\examples\preprocessing\plot_map_data_to_normal.py`

```
"""
=================================
Map data to a normal distribution
=================================

.. currentmodule:: sklearn.preprocessing

This example demonstrates the use of the Box-Cox and Yeo-Johnson transforms
through :class:`~PowerTransformer` to map data from various
distributions to a normal distribution.

The power transform is useful as a transformation in modeling problems where
homoscedasticity and normality are desired. Below are examples of Box-Cox and
Yeo-Johnson applied to six different probability distributions: Lognormal,
Chi-squared, Weibull, Gaussian, Uniform, and Bimodal.

Note that the transformations successfully map the data to a normal
distribution when applied to certain datasets, but are ineffective with others.
This highlights the importance of visualizing the data before and after
transformation.

Also note that even though Box-Cox seems to perform better than Yeo-Johnson for
lognormal and chi-squared distributions, keep in mind that Box-Cox does not
support inputs with negative values.

For comparison, we also add the output from
:class:`~QuantileTransformer`. It can force any arbitrary
distribution into a gaussian, provided that there are enough training samples
(thousands). Because it is a non-parametric method, it is harder to interpret
than the parametric ones (Box-Cox and Yeo-Johnson).

On "small" datasets (less than a few hundred points), the quantile transformer
is prone to overfitting. The use of the power transform is then recommended.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

N_SAMPLES = 1000
FONT_SIZE = 6
BINS = 30


rng = np.random.RandomState(304)  # 使用种子 304 创建随机数生成器对象

bc = PowerTransformer(method="box-cox")  # 创建 Box-Cox 变换对象
yj = PowerTransformer(method="yeo-johnson")  # 创建 Yeo-Johnson 变换对象

# n_quantiles 设置为训练集大小，而不是默认值，以避免本示例引发警告
qt = QuantileTransformer(
    n_quantiles=500, output_distribution="normal", random_state=rng
)  # 创建 Quantile 变换对象

size = (N_SAMPLES, 1)


# lognormal distribution
X_lognormal = rng.lognormal(size=size)  # 生成 lognormal 分布的随机数据

# chi-squared distribution
df = 3
X_chisq = rng.chisquare(df=df, size=size)  # 生成 chi-squared 分布的随机数据

# weibull distribution
a = 50
X_weibull = rng.weibull(a=a, size=size)  # 生成 Weibull 分布的随机数据

# gaussian distribution
loc = 100
X_gaussian = rng.normal(loc=loc, size=size)  # 生成 Gaussian 分布的随机数据

# uniform distribution
X_uniform = rng.uniform(low=0, high=1, size=size)  # 生成 Uniform 分布的随机数据

# bimodal distribution
loc_a, loc_b = 100, 105
X_a, X_b = rng.normal(loc=loc_a, size=size), rng.normal(loc=loc_b, size=size)
X_bimodal = np.concatenate([X_a, X_b], axis=0)  # 生成 Bimodal 分布的随机数据


# create plots
distributions = [
    ("Lognormal", X_lognormal),
    ("Chi-squared", X_chisq),
    ("Weibull", X_weibull),
    ("Gaussian", X_gaussian),
    ("Uniform", X_uniform),
    ("Bimodal", X_bimodal),
]
# 定义颜色列表，用于不同图形的颜色填充
colors = ["#D81B60", "#0188FF", "#FFC107", "#B7A2FF", "#000000", "#2EC5AC"]

# 创建包含多个子图的图形对象，设置为8行3列的网格布局，图形比例为2:1
fig, axes = plt.subplots(nrows=8, ncols=3, figsize=plt.figaspect(2))
# 将二维数组展平为一维数组，便于后续操作
axes = axes.flatten()

# 定义子图索引列表，每个子列表包含4个子图的索引
axes_idxs = [
    (0, 3, 6, 9),
    (1, 4, 7, 10),
    (2, 5, 8, 11),
    (12, 15, 18, 21),
    (13, 16, 19, 22),
    (14, 17, 20, 23),
]
# 根据索引列表生成子图对象的列表
axes_list = [(axes[i], axes[j], axes[k], axes[l]) for (i, j, k, l) in axes_idxs]

# 遍历分布列表和颜色列表，并将每种分布对应的数据分成训练集和测试集
for distribution, color, axes in zip(distributions, colors, axes_list):
    name, X = distribution
    X_train, X_test = train_test_split(X, test_size=0.5)

    # 对训练集进行幂转换和分位数转换
    X_trans_bc = bc.fit(X_train).transform(X_test)
    lmbda_bc = round(bc.lambdas_[0], 2)
    X_trans_yj = yj.fit(X_train).transform(X_test)
    lmbda_yj = round(yj.lambdas_[0], 2)
    X_trans_qt = qt.fit(X_train).transform(X_test)

    # 分别将四个子图对象从axes_list中解包赋值给对应变量
    ax_original, ax_bc, ax_yj, ax_qt = axes

    # 在原始数据的子图上绘制直方图，设置颜色和分箱数
    ax_original.hist(X_train, color=color, bins=BINS)
    ax_original.set_title(name, fontsize=FONT_SIZE)  # 设置子图标题和字体大小
    ax_original.tick_params(axis="both", which="major", labelsize=FONT_SIZE)  # 设置刻度标签大小

    # 遍历三种转换方法及其对应的子图对象，绘制转换后数据的直方图
    for ax, X_trans, meth_name, lmbda in zip(
        (ax_bc, ax_yj, ax_qt),
        (X_trans_bc, X_trans_yj, X_trans_qt),
        ("Box-Cox", "Yeo-Johnson", "Quantile transform"),
        (lmbda_bc, lmbda_yj, None),
    ):
        ax.hist(X_trans, color=color, bins=BINS)
        title = "After {}".format(meth_name)
        if lmbda is not None:
            title += "\n$\\lambda$ = {}".format(lmbda)  # 如果有λ值，添加到子图标题中
        ax.set_title(title, fontsize=FONT_SIZE)  # 设置子图标题和字体大小
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)  # 设置刻度标签大小
        ax.set_xlim([-3.5, 3.5])  # 设置x轴限制

# 调整子图布局，使其紧凑显示
plt.tight_layout()
# 显示图形
plt.show()
```