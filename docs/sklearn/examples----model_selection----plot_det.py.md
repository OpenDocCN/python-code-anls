# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_det.py`

```
"""
====================================
Detection error tradeoff (DET) curve
====================================

In this example, we compare two binary classification multi-threshold metrics:
the Receiver Operating Characteristic (ROC) and the Detection Error Tradeoff
(DET). For such purpose, we evaluate two different classifiers for the same
classification task.

ROC curves feature true positive rate (TPR) on the Y axis, and false positive
rate (FPR) on the X axis. This means that the top left corner of the plot is the
"ideal" point - a FPR of zero, and a TPR of one.

DET curves are a variation of ROC curves where False Negative Rate (FNR) is
plotted on the y-axis instead of the TPR. In this case the origin (bottom left
corner) is the "ideal" point.

.. note::

    - See :func:`sklearn.metrics.roc_curve` for further information about ROC
      curves.

    - See :func:`sklearn.metrics.det_curve` for further information about
      DET curves.

    - This example is loosely based on
      :ref:`sphx_glr_auto_examples_classification_plot_classifier_comparison.py`
      example.

    - See :ref:`sphx_glr_auto_examples_model_selection_plot_roc_crossval.py` for
      an example estimating the variance of the ROC curves and ROC-AUC.

"""

# %%
# Generate synthetic data
# -----------------------

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 生成具有指定特征的合成数据集，包括分类标签
X, y = make_classification(
    n_samples=1_000,          # 样本数为1000
    n_features=2,             # 特征数为2
    n_redundant=0,            # 无冗余特征
    n_informative=2,          # 有信息的特征为2
    random_state=1,           # 随机数种子，确保可重复性
    n_clusters_per_class=1,   # 每个类别内的簇数为1
)

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# %%
# Define the classifiers
# ----------------------
#
# Here we define two different classifiers. The goal is to visually compare their
# statistical performance across thresholds using the ROC and DET curves. There
# is no particular reason why these classifiers are chosen other classifiers
# available in scikit-learn.

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

# 定义两种不同的分类器
classifiers = {
    "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(C=0.025)),
    "Random Forest": RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1
    ),
}

# %%
# Plot ROC and DET curves
# -----------------------
#
# DET curves are commonly plotted in normal deviate scale. To achieve this the
# DET display transforms the error rates as returned by the
# :func:`~sklearn.metrics.det_curve` and the axis scale using
# `scipy.stats.norm`.

import matplotlib.pyplot as plt

from sklearn.metrics import DetCurveDisplay, RocCurveDisplay

# 创建包含两个子图的图形对象
fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))

# 遍历每个分类器，并在 ROC 和 DET 曲线上显示其性能
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)

    # 在 ROC 曲线上显示分类器性能
    RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_roc, name=name)
    # 使用给定的分类器（clf）、测试数据集（X_test, y_test）和坐标轴（ax_det），生成检测曲线显示对象
    DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_det, name=name)
ax_roc.set_title("Receiver Operating Characteristic (ROC) curves")
ax_det.set_title("Detection Error Tradeoff (DET) curves")

ax_roc.grid(linestyle="--")  # 设置 ROC 曲线图的网格线样式为虚线
ax_det.grid(linestyle="--")  # 设置 DET 曲线图的网格线样式为虚线

plt.legend()  # 添加图例，显示每条曲线对应的分类算法名称
plt.show()  # 显示图形界面，展示绘制的 ROC 和 DET 曲线

# %%
# 注意，通过 DET 曲线比 ROC 曲线更容易直观评估不同分类算法的整体性能。
# ROC 曲线在线性刻度上绘制，不同分类器在大部分图表中通常表现相似，主要差异出现在图表的左上角。
# 相比之下，DET 曲线在正态分布刻度上呈现直线，更易于区分整体情况，感兴趣的区域覆盖了图表的大部分。
#
# DET 曲线直接反映了检测错误权衡，有助于操作点分析。用户可以决定他们愿意接受的假阴率（FNR），以牺牲假阳率（FPR）或反之。
```