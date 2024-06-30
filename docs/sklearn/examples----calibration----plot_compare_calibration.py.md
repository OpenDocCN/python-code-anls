# `D:\src\scipysrc\scikit-learn\examples\calibration\plot_compare_calibration.py`

```
"""
========================================
Comparison of Calibration of Classifiers
========================================

Well calibrated classifiers are probabilistic classifiers for which the output
of :term:`predict_proba` can be directly interpreted as a confidence level.
For instance, a well calibrated (binary) classifier should classify the samples
such that for the samples to which it gave a :term:`predict_proba` value close
to 0.8, approximately 80% actually belong to the positive class.

In this example we will compare the calibration of four different
models: :ref:`Logistic_regression`, :ref:`gaussian_naive_bayes`,
:ref:`Random Forest Classifier <forest>` and :ref:`Linear SVM
<svm_classification>`.

"""

# %%
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Dataset
# -------
#
# We will use a synthetic binary classification dataset with 100,000 samples
# and 20 features. Of the 20 features, only 2 are informative, 2 are
# redundant (random combinations of the informative features) and the
# remaining 16 are uninformative (random numbers).
#
# Of the 100,000 samples, 100 will be used for model fitting and the remaining
# for testing. Note that this split is quite unusual: the goal is to obtain
# stable calibration curve estimates for models that are potentially prone to
# overfitting. In practice, one should rather use cross-validation with more
# balanced splits but this would make the code of this example more complicated
# to follow.

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic dataset for binary classification
X, y = make_classification(
    n_samples=100_000, n_features=20, n_informative=2, n_redundant=2, random_state=42
)

train_samples = 100  # Samples used for training the models

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    shuffle=False,  # Shuffle is disabled to maintain consistency for calibration
    test_size=100_000 - train_samples,  # Use 100 samples for training
)

# %%
# Calibration curves
# ------------------
#
# Below, we train each of the four models with the small training dataset, then
# plot calibration curves (also known as reliability diagrams) using
# predicted probabilities of the test dataset. Calibration curves are created
# by binning predicted probabilities, then plotting the mean predicted
# probability in each bin against the observed frequency ('fraction of
# positives'). Below the calibration curve, we plot a histogram showing
# the distribution of the predicted probabilities or more specifically,
# the number of samples in each predicted probability bin.

import numpy as np
from sklearn.svm import LinearSVC


class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output."""

    def fit(self, X, y):
        super().fit(X, y)
        # Compute decision function output for scaling
        df = self.decision_function(X)
        # Store minimum and maximum decision function values
        self.df_min_ = df.min()
        self.df_max_ = df.max()
    # 定义方法 `predict_proba`，用于预测样本 X 的类别概率
    def predict_proba(self, X):
        """Min-max scale output of `decision_function` to [0,1]."""
        # 调用 `decision_function` 方法计算样本 X 的决策函数值
        df = self.decision_function(X)
        # 对决策函数值进行 min-max 标定，将其缩放到区间 [0,1]
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        # 将标定后的值限制在 [0,1] 区间内，得到正类的概率
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        # 计算负类的概率，由于概率和为 1，所以负类概率为 1 减去正类概率
        proba_neg_class = 1 - proba_pos_class
        # 将正类和负类的概率合并为一个二维数组，作为最终的预测概率输出
        proba = np.c_[proba_neg_class, proba_pos_class]
        # 返回预测的概率数组
        return proba
# %%

# 导入需要使用的库和模块
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB

# 定义在研究中要比较的分类器。
#
# 注意我们使用了能够自动调整正则化参数的逻辑回归模型的变体。
#
# 为了公平比较，我们应该对所有分类器进行超参数搜索，
# 但为了保持示例代码的简洁和执行速度，我们在这里不做这些。
lr = LogisticRegressionCV(
    Cs=np.logspace(-6, 6, 101), cv=10, scoring="neg_log_loss", max_iter=1_000
)
gnb = GaussianNB()
svc = NaivelyCalibratedLinearSVC(C=1.0)  # NaivelyCalibratedLinearSVC是未定义的类，这里可能存在错误
rfc = RandomForestClassifier(random_state=42)

# 将分类器和它们的名称组成列表
clf_list = [
    (lr, "Logistic Regression"),
    (gnb, "Naive Bayes"),
    (svc, "SVC"),
    (rfc, "Random forest"),
]

# %%

# 导入绘图所需的库和模块
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 创建一个大小为10x10的新图形
fig = plt.figure(figsize=(10, 10))
# 使用GridSpec定义4行2列的子图网格
gs = GridSpec(4, 2)
# 获取一个颜色映射
colors = plt.get_cmap("Dark2")

# 在网格的前两行前两列添加子图用于显示校准曲线
ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
markers = ["^", "v", "s", "o"]
# 对每个分类器进行迭代
for i, (clf, name) in enumerate(clf_list):
    # 使用训练集拟合分类器
    clf.fit(X_train, y_train)
    # 创建校准显示对象并将其添加到校准曲线的子图上
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
        marker=markers[i],
    )
    calibration_displays[name] = display

# 在校准曲线子图上添加网格和标题
ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots")

# 添加直方图
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
# 对每个分类器及其名称进行迭代
for i, (_, name) in enumerate(clf_list):
    # 获取当前子图的行和列位置
    row, col = grid_positions[i]
    # 在指定位置添加一个子图
    ax = fig.add_subplot(gs[row, col])

    # 绘制直方图，显示每个分类器的预测概率分布
    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    # 设置子图标题、x轴标签和y轴标签
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

# 调整子图的布局以防止重叠
plt.tight_layout()
# 显示图形
plt.show()

# %%

#
# 结果分析
# -----------------------
#
# :class:`~sklearn.linear_model.LogisticRegressionCV` 尽管训练集规模较小，但返回了合理校准的预测结果：
# 其可靠性曲线在四个模型中最接近对角线。
#
# 逻辑回归通过最小化对数损失进行训练，对数损失是严格的适当性评分规则：
# 在无限的训练数据极限下，严格的适当性评分规则将由预测真实条件概率的模型最小化。
# 然而，单独使用适当的评分规则作为训练目标并不能保证一个良好校准的模型：
# 即使有非常大的训练集，如果逻辑回归过于正则化或者输入特征选择和预处理不当，
# 它仍然可能是校准不良的。
# features made this model mis-specified (e.g. if the true decision boundary of
# the dataset is a highly non-linear function of the input features).
#
# In this example the training set was intentionally kept very small. In this
# setting, optimizing the log-loss can still lead to poorly calibrated models
# because of overfitting. To mitigate this, the
# :class:`~sklearn.linear_model.LogisticRegressionCV` class was configured to
# tune the `C` regularization parameter to also minimize the log-loss via inner
# cross-validation so as to find the best compromise for this model in the
# small training set setting.
#
# Because of the finite training set size and the lack of guarantee for
# well-specification, we observe that the calibration curve of the logistic
# regression model is close but not perfectly on the diagonal. The shape of the
# calibration curve of this model can be interpreted as slightly
# under-confident: the predicted probabilities are a bit too close to 0.5
# compared to the true fraction of positive samples.
#
# The other methods all output less well calibrated probabilities:
#
# * :class:`~sklearn.naive_bayes.GaussianNB` tends to push probabilities to 0
#   or 1 (see histogram) on this particular dataset (over-confidence). This is
#   mainly because the naive Bayes equation only provides correct estimate of
#   probabilities when the assumption that features are conditionally
#   independent holds [2]_. However, features can be correlated and this is the case
#   with this dataset, which contains 2 features generated as random linear
#   combinations of the informative features. These correlated features are
#   effectively being 'counted twice', resulting in pushing the predicted
#   probabilities towards 0 and 1 [3]_. Note, however, that changing the seed
#   used to generate the dataset can lead to widely varying results for the
#   naive Bayes estimator.
#
# * :class:`~sklearn.svm.LinearSVC` is not a natural probabilistic classifier.
#   In order to interpret its prediction as such, we naively scaled the output
#   of the :term:`decision_function` into [0, 1] by applying min-max scaling in
#   the `NaivelyCalibratedLinearSVC` wrapper class defined above. This
#   estimator shows a typical sigmoid-shaped calibration curve on this data:
#   predictions larger than 0.5 correspond to samples with an even larger
#   effective positive class fraction (above the diagonal), while predictions
#   below 0.5 corresponds to even lower positive class fractions (below the
#   diagonal). This under-confident predictions are typical for maximum-margin
#   methods [1]_.
#
# * :class:`~sklearn.ensemble.RandomForestClassifier`'s prediction histogram
#   shows peaks at approx. 0.2 and 0.9 probability, while probabilities close to
#   0 or 1 are very rare. An explanation for this is given by [1]_:
#   "Methods such as bagging and random forests that average
#   predictions from a base set of models can have difficulty making
#   predictions near 0 and 1 because variance in the underlying base models
#   will bias predictions that should be near zero or one away from these
#   values. Because predictions are restricted to the interval [0, 1], errors
#   caused by variance tend to be one-sided near zero and one. For example, if
#   a model should predict p = 0 for a case, the only way bagging can achieve
#   this is if all bagged trees predict zero. If we add noise to the trees that
#   bagging is averaging over, this noise will cause some trees to predict
#   values larger than 0 for this case, thus moving the average prediction of
#   the bagged ensemble away from 0. We observe this effect most strongly with
#   random forests because the base-level trees trained with random forests
#   have relatively high variance due to feature subsetting." This effect can
#   make random forests under-confident. Despite this possible bias, note that
#   the trees themselves are fit by minimizing either the Gini or Entropy
#   criterion, both of which lead to splits that minimize proper scoring rules:
#   the Brier score or the log-loss respectively. See :ref:`the user guide
#   <tree_mathematical_formulation>` for more details. This can explain why
#   this model shows a good enough calibration curve on this particular example
#   dataset. Indeed the Random Forest model is not significantly more
#   under-confident than the Logistic Regression model.
#
# Feel free to re-run this example with different random seeds and other
# dataset generation parameters to see how different the calibration plots can
# look. In general, Logistic Regression and Random Forest will tend to be the
# best calibrated classifiers, while SVC will often display the typical
# under-confident miscalibration. The naive Bayes model is also often poorly
# calibrated but the general shape of its calibration curve can vary widely
# depending on the dataset.
#
# Finally, note that for some dataset seeds, all models are poorly calibrated,
# even when tuning the regularization parameter as above. This is bound to
# happen when the training size is too small or when the model is severely
# misspecified.
#
# References
# ----------
#
# .. [1] `Predicting Good Probabilities with Supervised Learning
#        <https://dl.acm.org/doi/pdf/10.1145/1102351.1102430>`_, A.
#        Niculescu-Mizil & R. Caruana, ICML 2005
#
# .. [2] `Beyond independence: Conditions for the optimality of the simple
#        bayesian classifier
#        <https://www.ics.uci.edu/~pazzani/Publications/mlc96-pedro.pdf>`_
#        Domingos, P., & Pazzani, M., Proc. 13th Intl. Conf. Machine Learning.
#        1996.
#
# .. [3] `Obtaining calibrated probability estimates from decision trees and
#        naive Bayesian classifiers
#        <https://citeseerx.ist.psu.edu/doc_view/pid/4f67a122ec3723f08ad5cbefecad119b432b3304>`_
#        Zadrozny, Bianca, and Charles Elkan. Icml. Vol. 1. 2001.
```