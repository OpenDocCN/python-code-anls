# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_successive_halving_heatmap.py`

```
"""
Comparison between grid search and successive halving
=====================================================

This example compares the parameter search performed by
:class:`~sklearn.model_selection.HalvingGridSearchCV` and
:class:`~sklearn.model_selection.GridSearchCV`.

"""

from time import time  # 导入时间模块中的时间函数

import matplotlib.pyplot as plt  # 导入 matplotlib 中的 pyplot 模块并重命名为 plt
import numpy as np  # 导入 numpy 模块并重命名为 np
import pandas as pd  # 导入 pandas 模块并重命名为 pd

from sklearn import datasets  # 导入 sklearn 中的 datasets 模块
from sklearn.experimental import enable_halving_search_cv  # 导入启用半数减少搜索交叉验证的实验特性
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV  # 从 sklearn 中导入 GridSearchCV 和 HalvingGridSearchCV 类
from sklearn.svm import SVC  # 从 sklearn 中导入支持向量机（SVM）的 SVC 类

# %%
# We first define the parameter space for an :class:`~sklearn.svm.SVC`
# estimator, and compute the time required to train a
# :class:`~sklearn.model_selection.HalvingGridSearchCV` instance, as well as a
# :class:`~sklearn.model_selection.GridSearchCV` instance.

rng = np.random.RandomState(0)  # 创建一个随机数生成器对象 rng，并设置种子为 0
X, y = datasets.make_classification(n_samples=1000, random_state=rng)  # 使用 datasets 模块生成一个样本集 X 和标签 y

gammas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]  # 设定 gamma 参数的候选列表
Cs = [1, 10, 100, 1e3, 1e4, 1e5]  # 设定 C 参数的候选列表
param_grid = {"gamma": gammas, "C": Cs}  # 构建参数网格 param_grid，包含 gamma 和 C 参数的组合

clf = SVC(random_state=rng)  # 创建一个 SVM 分类器对象 clf，使用随机数生成器 rng 初始化

tic = time()  # 记录当前时间，作为开始时间
gsh = HalvingGridSearchCV(  # 创建一个 HalvingGridSearchCV 对象 gsh，使用参数：估计器 clf、参数网格 param_grid、因子 2、随机数生成器 rng
    estimator=clf, param_grid=param_grid, factor=2, random_state=rng
)
gsh.fit(X, y)  # 在样本集 X 和标签 y 上拟合 gsh 对象
gsh_time = time() - tic  # 计算拟合过程耗费的时间

tic = time()  # 记录当前时间，作为开始时间
gs = GridSearchCV(estimator=clf, param_grid=param_grid)  # 创建一个 GridSearchCV 对象 gs，使用参数：估计器 clf、参数网格 param_grid
gs.fit(X, y)  # 在样本集 X 和标签 y 上拟合 gs 对象
gs_time = time() - tic  # 计算拟合过程耗费的时间

# %%
# We now plot heatmaps for both search estimators.


def make_heatmap(ax, gs, is_sh=False, make_cbar=False):
    """Helper to make a heatmap."""
    results = pd.DataFrame(gs.cv_results_)  # 将交叉验证结果 gs.cv_results_ 转换为 pandas DataFrame 格式的 results
    results[["param_C", "param_gamma"]] = results[["param_C", "param_gamma"]].astype(
        np.float64
    )  # 将 param_C 和 param_gamma 列转换为 float64 类型

    if is_sh:
        # SH dataframe: get mean_test_score values for the highest iter
        scores_matrix = results.sort_values("iter").pivot_table(
            index="param_gamma",
            columns="param_C",
            values="mean_test_score",
            aggfunc="last",
        )  # 对于半数减少搜索的情况，从结果中选择最高迭代的 mean_test_score 值，构建得分矩阵 scores_matrix
    else:
        scores_matrix = results.pivot(
            index="param_gamma", columns="param_C", values="mean_test_score"
        )  # 对于普通的网格搜索，直接构建 mean_test_score 的得分矩阵 scores_matrix

    im = ax.imshow(scores_matrix)  # 在坐标轴 ax 上绘制热图，使用 scores_matrix 作为数据源

    ax.set_xticks(np.arange(len(Cs)))  # 设置 x 轴刻度位置
    ax.set_xticklabels(["{:.0E}".format(x) for x in Cs])  # 设置 x 轴刻度标签，使用科学计数法显示 C 参数的值
    ax.set_xlabel("C", fontsize=15)  # 设置 x 轴标签和字体大小

    ax.set_yticks(np.arange(len(gammas)))  # 设置 y 轴刻度位置
    ax.set_yticklabels(["{:.0E}".format(x) for x in gammas])  # 设置 y 轴刻度标签，使用科学计数法显示 gamma 参数的值
    ax.set_ylabel("gamma", fontsize=15)  # 设置 y 轴标签和字体大小

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")  # 旋转 x 轴刻度标签，设置对齐方式为右对齐
    # 如果参数 is_sh 为真，则执行以下代码块
    if is_sh:
        # 从结果数据框中生成透视表，索引为 param_gamma，列为 param_C，值为 iter 的最大值
        iterations = results.pivot_table(
            index="param_gamma", columns="param_C", values="iter", aggfunc="max"
        ).values
        # 遍历 gammas 列表的长度
        for i in range(len(gammas)):
            # 遍历 Cs 列表的长度
            for j in range(len(Cs)):
                # 在图形 ax 上添加文本，位置为 (j, i)，文本内容为 iterations[i, j]
                ax.text(
                    j,
                    i,
                    iterations[i, j],
                    ha="center",   # 水平对齐方式为居中
                    va="center",   # 垂直对齐方式为居中
                    color="w",     # 文本颜色为白色
                    fontsize=20,   # 字体大小为20
                )

    # 如果参数 make_cbar 为真，则执行以下代码块
    if make_cbar:
        # 调整子图布局，使 colorbar 适当显示
        fig.subplots_adjust(right=0.8)
        # 在图形上添加颜色条轴
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # 创建颜色条并添加到颜色条轴上
        fig.colorbar(im, cax=cbar_ax)
        # 设置颜色条轴的标签为 "mean_test_score"，旋转角度为 -90 度，垂直对齐方式为底部，字体大小为15
        cbar_ax.set_ylabel("mean_test_score", rotation=-90, va="bottom", fontsize=15)
# 创建包含两个子图的图形，共享 y 轴
fig, axes = plt.subplots(ncols=2, sharey=True)
# 将子图对象分别赋值给 ax1 和 ax2
ax1, ax2 = axes

# 在 ax1 上创建一个热力图，显示 gsh 数据，标记为半透明
make_heatmap(ax1, gsh, is_sh=True)
# 在 ax2 上创建一个热力图，显示 gs 数据，并添加颜色条
make_heatmap(ax2, gs, make_cbar=True)

# 设置 ax1 的标题，显示 "Successive Halving" 和计算时间
ax1.set_title("Successive Halving\ntime = {:.3f}s".format(gsh_time), fontsize=15)
# 设置 ax2 的标题，显示 "GridSearch" 和计算时间
ax2.set_title("GridSearch\ntime = {:.3f}s".format(gs_time), fontsize=15)

# 显示整个图形
plt.show()
```