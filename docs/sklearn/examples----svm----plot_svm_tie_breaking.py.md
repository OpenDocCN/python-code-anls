# `D:\src\scipysrc\scikit-learn\examples\svm\plot_svm_tie_breaking.py`

```
"""
=========================================================
SVM Tie Breaking Example
=========================================================
Tie breaking is costly if ``decision_function_shape='ovr'``, and therefore it
is not enabled by default. This example illustrates the effect of the
``break_ties`` parameter for a multiclass classification problem and
``decision_function_shape='ovr'``.

The two plots differ only in the area in the middle where the classes are
tied. If ``break_ties=False``, all input in that area would be classified as
one class, whereas if ``break_ties=True``, the tie-breaking mechanism will
create a non-convex decision boundary in that area.

"""

# Code source: Andreas Mueller, Adrin Jalali
# SPDX-License-Identifier: BSD-3-Clause

# 导入所需的库
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# 生成随机数据集
X, y = make_blobs(random_state=27)

# 创建绘图窗口和子图
fig, sub = plt.subplots(2, 1, figsize=(5, 8))
titles = ("break_ties = False", "break_ties = True")

# 遍历不同的 break_ties 参数值
for break_ties, title, ax in zip((False, True), titles, sub.flatten()):
    # 使用线性核的支持向量机进行训练
    svm = SVC(
        kernel="linear", C=1, break_ties=break_ties, decision_function_shape="ovr"
    ).fit(X, y)

    # 设置 x 和 y 轴的范围
    xlim = [X[:, 0].min(), X[:, 0].max()]
    ylim = [X[:, 1].min(), X[:, 1].max()]

    # 生成用于绘制决策边界的网格点
    xs = np.linspace(xlim[0], xlim[1], 1000)
    ys = np.linspace(ylim[0], ylim[1], 1000)
    xx, yy = np.meshgrid(xs, ys)

    # 对网格点进行预测
    pred = svm.predict(np.c_[xx.ravel(), yy.ravel()])

    # 设置颜色
    colors = [plt.cm.Accent(i) for i in [0, 4, 7]]

    # 绘制散点图
    points = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="Accent")
    classes = [(0, 1), (0, 2), (1, 2)]
    line = np.linspace(X[:, 1].min() - 5, X[:, 1].max() + 5)
    ax.imshow(
        -pred.reshape(xx.shape),
        cmap="Accent",
        alpha=0.2,
        extent=(xlim[0], xlim[1], ylim[1], ylim[0]),
    )

    # 绘制决策边界
    for coef, intercept, col in zip(svm.coef_, svm.intercept_, classes):
        line2 = -(line * coef[1] + intercept) / coef[0]
        ax.plot(line2, line, "-", c=colors[col[0]])
        ax.plot(line2, line, "--", c=colors[col[1]])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_aspect("equal")

# 显示绘图结果
plt.show()
```