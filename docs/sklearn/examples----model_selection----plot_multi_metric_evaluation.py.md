# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_multi_metric_evaluation.py`

```
"""
============================================================================
演示交叉验证评估中的多指标评估在 cross_val_score 和 GridSearchCV 中的应用
============================================================================

可以通过将 ``scoring`` 参数设置为指标评分器名称列表或将评分器名称映射到评分器可调用对象的字典来进行多指标参数搜索。

所有评分器的分数都可以在 ``cv_results_`` 字典中的以 ``'_<scorer_name>'`` 结尾的键中找到
（例如 ``'mean_test_precision'``, ``'rank_test_precision'`` 等）。

``best_estimator_``, ``best_index_``, ``best_score_`` 和 ``best_params_``
对应于设置为 ``refit`` 属性的评分器（键）。

"""

# 作者：scikit-learn 开发人员
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import make_hastie_10_2
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# %%
# 使用多个评估指标运行 ``GridSearchCV``
# ----------------------------------------------------------
#

X, y = make_hastie_10_2(n_samples=8000, random_state=42)

# 评分器可以是预定义的度量字符串之一，也可以是评分器可调用对象，例如由 make_scorer 返回的对象
scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}

# 设置 refit='AUC'，在整个数据集上用具有最佳交叉验证 AUC 分数的参数设置重新拟合估计器。
# 最佳估计器可通过 ``gs.best_estimator_`` 获得，同时还可以获取 ``gs.best_score_``,
# ``gs.best_params_`` 和 ``gs.best_index_`` 等参数。
gs = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={"min_samples_split": range(2, 403, 20)},
    scoring=scoring,
    refit="AUC",
    n_jobs=2,
    return_train_score=True,
)
gs.fit(X, y)
results = gs.cv_results_

# %%
# 绘制结果
# -------------------

plt.figure(figsize=(13, 13))
plt.title("使用多个评分器同时评估的 GridSearchCV", fontsize=16)

plt.xlabel("min_samples_split")
plt.ylabel("Score")

ax = plt.gca()
ax.set_xlim(0, 402)
ax.set_ylim(0.73, 1)

# 从 MaskedArray 获取常规的 numpy 数组
X_axis = np.array(results["param_min_samples_split"].data, dtype=float)

for scorer, color in zip(sorted(scoring), ["g", "k"]):
    for sample, style in (("train", "--"), ("test", "-")):
        # 获取样本的平均分数和标准差
        sample_score_mean = results["mean_%s_%s" % (sample, scorer)]
        sample_score_std = results["std_%s_%s" % (sample, scorer)]
        # 在图表中填充区域，表示平均分数加减一个标准差的范围
        ax.fill_between(
            X_axis,
            sample_score_mean - sample_score_std,
            sample_score_mean + sample_score_std,
            alpha=0.1 if sample == "test" else 0,
            color=color,
        )
        # 绘制平均分数的折线图
        ax.plot(
            X_axis,
            sample_score_mean,
            style,
            color=color,
            alpha=1 if sample == "test" else 0.7,
            label="%s (%s)" % (scorer, sample),
        )

    # 找出在测试集中得分排名第一的索引
    best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
    # 获取在测试集中得分最好的分数
    best_score = results["mean_test_%s" % scorer][best_index]

    # 绘制一个虚线垂直线，标记在该得分下的最佳分数
    ax.plot(
        [
            X_axis[best_index],
        ]
        * 2,
        [0, best_score],
        linestyle="-.",
        color=color,
        marker="x",
        markeredgewidth=3,
        ms=8,
    )

    # 在图表中标注出最佳分数
    ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))
# 添加图例，位置自动选择最佳位置
plt.legend(loc="best")
# 关闭网格线显示
plt.grid(False)
# 显示绘图结果
plt.show()
```