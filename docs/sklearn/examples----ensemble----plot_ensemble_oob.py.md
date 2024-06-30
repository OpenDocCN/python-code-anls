# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_ensemble_oob.py`

```
"""
=============================
OOB Errors for Random Forests
=============================

The ``RandomForestClassifier`` is trained using *bootstrap aggregation*, where
each new tree is fit from a bootstrap sample of the training observations
:math:`z_i = (x_i, y_i)`. The *out-of-bag* (OOB) error is the average error for
each :math:`z_i` calculated using predictions from the trees that do not
contain :math:`z_i` in their respective bootstrap sample. This allows the
``RandomForestClassifier`` to be fit and validated whilst being trained [1]_.

The example below demonstrates how the OOB error can be measured at the
addition of each new tree during training. The resulting plot allows a
practitioner to approximate a suitable value of ``n_estimators`` at which the
error stabilizes.

.. [1] T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical
       Learning Ed. 2", p592-593, Springer, 2009.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from collections import OrderedDict  # 导入有序字典模块

import matplotlib.pyplot as plt  # 导入matplotlib用于绘图

from sklearn.datasets import make_classification  # 导入用于生成分类数据的函数
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器

RANDOM_STATE = 123  # 设置随机种子

# Generate a binary classification dataset.
X, y = make_classification(
    n_samples=500,
    n_features=25,
    n_clusters_per_class=1,
    n_informative=15,
    random_state=RANDOM_STATE,
)

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
# 设置 `warm_start` 构造参数为 `True`，禁用并行集成支持，但在训练期间跟踪OOB误差轨迹是必要的。
ensemble_clfs = [
    (
        "RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(
            warm_start=True,
            oob_score=True,
            max_features="sqrt",
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(
            warm_start=True,
            max_features="log2",
            oob_score=True,
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "RandomForestClassifier, max_features=None",
        RandomForestClassifier(
            warm_start=True,
            max_features=None,
            oob_score=True,
            random_state=RANDOM_STATE,
        ),
    ),
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
# 将分类器名称映射到 (<n_estimators>, <error rate>) 列表的字典中。
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
# 探索 `n_estimators` 取值的范围。
min_estimators = 15
max_estimators = 150

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1, 5):
        clf.set_params(n_estimators=i)  # 设置随机森林分类器的 n_estimators 参数为当前值 i
        clf.fit(X, y)  # 使用训练数据 X, y 进行训练

        # Record the OOB error for each `n_estimators=i` setting.
        # 记录每个 `n_estimators=i` 设置下的 OOB 错误率。
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
# 生成 "OOB error rate" vs. "n_estimators" 的图表。
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    # 绘制一个简单的线图，使用给定的 xs 和 ys 数据，并添加一个标签
    plt.plot(xs, ys, label=label)
# 设置图表的 x 轴范围，从 min_estimators 到 max_estimators
plt.xlim(min_estimators, max_estimators)
# 设置 x 轴的标签文本为 "n_estimators"
plt.xlabel("n_estimators")
# 设置 y 轴的标签文本为 "OOB error rate"
plt.ylabel("OOB error rate")
# 在图表中添加图例，位置在右上角
plt.legend(loc="upper right")
# 显示图表
plt.show()
```