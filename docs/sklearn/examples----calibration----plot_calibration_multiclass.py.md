# `D:\src\scipysrc\scikit-learn\examples\calibration\plot_calibration_multiclass.py`

```
# %%
# Data
# ----
# Below, we generate a classification dataset with 2000 samples, 2 features
# and 3 target classes. We then split the data as follows:
#
# * train: 600 samples (for training the classifier)
# * valid: 400 samples (for calibrating predicted probabilities)
# * test: 1000 samples
#
# Note that we also create `X_train_valid` and `y_train_valid`, which consists
# of both the train and valid subsets. This is used when we only want to train
# the classifier but not calibrate the predicted probabilities.
#
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from sklearn.datasets import make_blobs

np.random.seed(0)

X, y = make_blobs(
    n_samples=2000, n_features=2, centers=3, random_state=42, cluster_std=5.0
)
X_train, y_train = X[:600], y[:600]
X_valid, y_valid = X[600:1000], y[600:1000]
X_train_valid, y_train_valid = X[:1000], y[:1000]
X_test, y_test = X[1000:], y[1000:]

# %%
# Fitting and calibration
# -----------------------
#
# First, we will train a :class:`~sklearn.ensemble.RandomForestClassifier`
# with 25 base estimators (trees) on the concatenated train and validation
# data (1000 samples). This is the uncalibrated classifier.

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train_valid, y_train_valid)

# %%
# To train the calibrated classifier, we start with the same
# :class:`~sklearn.ensemble.RandomForestClassifier` but train it using only
# the train data subset (600 samples) then calibrate, with `method='sigmoid'`,
# using the valid data subset (400 samples) in a 2-stage process.

from sklearn.calibration import CalibratedClassifierCV

clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train, y_train)
cal_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
cal_clf.fit(X_valid, y_valid)

# %%
# Compare probabilities
# ---------------------
# Below we plot a 2-simplex with arrows showing the change in predicted
# probabilities of the test samples.

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
colors = ["r", "g", "b"]

clf_probs = clf.predict_proba(X_test)
cal_clf_probs = cal_clf.predict_proba(X_test)
# Plot arrows
for i in range(clf_probs.shape[0]):
    # 在图中绘制箭头，起始点的 x 坐标为 clf_probs[i, 0]，y 坐标为 clf_probs[i, 1]
    # 箭头的方向由终点的坐标 cal_clf_probs[i, 0], cal_clf_probs[i, 1] 减去起始点的坐标确定
    # 箭头的颜色根据 y_test[i] 确定，对应的颜色由 colors[y_test[i]] 决定
    # 箭头头部的宽度设定为 1e-2
    plt.arrow(
        clf_probs[i, 0],
        clf_probs[i, 1],
        cal_clf_probs[i, 0] - clf_probs[i, 0],
        cal_clf_probs[i, 1] - clf_probs[i, 1],
        color=colors[y_test[i]],
        head_width=1e-2,
    )
# 绘制完美预测的点，每个顶点一个类别
plt.plot([1.0], [0.0], "ro", ms=20, label="Class 1")  # 在图中绘制红色圆点表示 Class 1
plt.plot([0.0], [1.0], "go", ms=20, label="Class 2")  # 在图中绘制绿色圆点表示 Class 2
plt.plot([0.0], [0.0], "bo", ms=20, label="Class 3")  # 在图中绘制蓝色圆点表示 Class 3

# 绘制单位单纯形的边界
plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], "k", label="Simplex")  # 在图中绘制黑色线条表示单纯形边界

# 标注单纯形周围的6个点和单纯形内部的中点
plt.annotate(
    r"($\frac{1}{3}$, $\frac{1}{3}$, $\frac{1}{3}$)",
    xy=(1.0 / 3, 1.0 / 3),
    xytext=(1.0 / 3, 0.23),
    xycoords="data",
    arrowprops=dict(facecolor="black", shrink=0.05),
    horizontalalignment="center",
    verticalalignment="center",
)
plt.plot([1.0 / 3], [1.0 / 3], "ko", ms=5)  # 在图中绘制黑色圆点表示 ($\frac{1}{3}$, $\frac{1}{3}$, $\frac{1}{3}$)

plt.annotate(
    r"($\frac{1}{2}$, $0$, $\frac{1}{2}$)",
    xy=(0.5, 0.0),
    xytext=(0.5, 0.1),
    xycoords="data",
    arrowprops=dict(facecolor="black", shrink=0.05),
    horizontalalignment="center",
    verticalalignment="center",
)
plt.annotate(
    r"($0$, $\frac{1}{2}$, $\frac{1}{2}$)",
    xy=(0.0, 0.5),
    xytext=(0.1, 0.5),
    xycoords="data",
    arrowprops=dict(facecolor="black", shrink=0.05),
    horizontalalignment="center",
    verticalalignment="center",
)
plt.annotate(
    r"($\frac{1}{2}$, $\frac{1}{2}$, $0$)",
    xy=(0.5, 0.5),
    xytext=(0.6, 0.6),
    xycoords="data",
    arrowprops=dict(facecolor="black", shrink=0.05),
    horizontalalignment="center",
    verticalalignment="center",
)
plt.annotate(
    r"($0$, $0$, $1$)",
    xy=(0, 0),
    xytext=(0.1, 0.1),
    xycoords="data",
    arrowprops=dict(facecolor="black", shrink=0.05),
    horizontalalignment="center",
    verticalalignment="center",
)
plt.annotate(
    r"($1$, $0$, $0$)",
    xy=(1, 0),
    xytext=(1, 0.1),
    xycoords="data",
    arrowprops=dict(facecolor="black", shrink=0.05),
    horizontalalignment="center",
    verticalalignment="center",
)
plt.annotate(
    r"($0$, $1$, $0$)",
    xy=(0, 1),
    xytext=(0.1, 1),
    xycoords="data",
    arrowprops=dict(facecolor="black", shrink=0.05),
    horizontalalignment="center",
    verticalalignment="center",
)

# 添加网格线
plt.grid(False)
for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    plt.plot([0, x], [x, 0], "k", alpha=0.2)  # 绘制倾斜线段1
    plt.plot([0, 0 + (1 - x) / 2], [x, x + (1 - x) / 2], "k", alpha=0.2)  # 绘制倾斜线段2
    plt.plot([x, x + (1 - x) / 2], [0, 0 + (1 - x) / 2], "k", alpha=0.2)  # 绘制倾斜线段3

# 设置图的标题及坐标轴标签
plt.title("Change of predicted probabilities on test samples after sigmoid calibration")
plt.xlabel("Probability class 1")
plt.ylabel("Probability class 2")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
_ = plt.legend(loc="best")

# %%
# 在上述图中，单纯形的每个顶点代表一个完美预测的类别（例如，1, 0, 0）。单纯形内部的中点代表三个类别预测概率相等（即，1/3, 1/3, 1/3）。
# 每个箭头从未校准的概率开始，箭头头部指向校准后的概率。箭头的颜色代表该测试样本的真实类别。
# sample.
#
# The uncalibrated classifier is overly confident in its predictions and
# incurs a large :ref:`log loss <log_loss>`. The calibrated classifier incurs
# a lower :ref:`log loss <log_loss>` due to two factors. First, notice in the
# figure above that the arrows generally point away from the edges of the
# simplex, where the probability of one class is 0. Second, a large proportion
# of the arrows point towards the true class, e.g., green arrows (samples where
# the true class is 'green') generally point towards the green vertex. This
# results in fewer over-confident, 0 predicted probabilities and at the same
# time an increase in the predicted probabilities of the correct class.
# Thus, the calibrated classifier produces more accurate predicted probabilities
# that incur a lower :ref:`log loss <log_loss>`
#
# We can show this objectively by comparing the :ref:`log loss <log_loss>` of
# the uncalibrated and calibrated classifiers on the predictions of the 1000
# test samples. Note that an alternative would have been to increase the number
# of base estimators (trees) of the
# :class:`~sklearn.ensemble.RandomForestClassifier` which would have resulted
# in a similar decrease in :ref:`log loss <log_loss>`.

# Importing the log_loss function from sklearn.metrics
from sklearn.metrics import log_loss

# Calculating the log loss of the uncalibrated classifier
score = log_loss(y_test, clf_probs)
# Calculating the log loss of the calibrated classifier
cal_score = log_loss(y_test, cal_clf_probs)

# Printing the results
print("Log-loss of")
print(f" * uncalibrated classifier: {score:.3f}")
print(f" * calibrated classifier: {cal_score:.3f}")

# %%
# Finally we generate a grid of possible uncalibrated probabilities over
# the 2-simplex, compute the corresponding calibrated probabilities and
# plot arrows for each. The arrows are colored according the highest
# uncalibrated probability. This illustrates the learned calibration map:

# Importing matplotlib.pyplot for plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

# Generate grid of probability values
p1d = np.linspace(0, 1, 20)
p0, p1 = np.meshgrid(p1d, p1d)
p2 = 1 - p0 - p1
p = np.c_[p0.ravel(), p1.ravel(), p2.ravel()]
p = p[p[:, 2] >= 0]

# Use the three class-wise calibrators to compute calibrated probabilities
calibrated_classifier = cal_clf.calibrated_classifiers_[0]
prediction = np.vstack(
    [
        calibrator.predict(this_p)
        for calibrator, this_p in zip(calibrated_classifier.calibrators, p.T)
    ]
).T

# Re-normalize the calibrated predictions to make sure they stay inside the
# simplex. This same renormalization step is performed internally by the
# predict method of CalibratedClassifierCV on multiclass problems.
prediction /= prediction.sum(axis=1)[:, None]

# Plot changes in predicted probabilities induced by the calibrators
for i in range(prediction.shape[0]):
    plt.arrow(
        p[i, 0],
        p[i, 1],
        prediction[i, 0] - p[i, 0],
        prediction[i, 1] - p[i, 1],
        head_width=1e-2,
        color=colors[np.argmax(p[i])],
    )

# Plot the boundaries of the unit simplex
plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], "k", label="Simplex")

# Turn off the grid lines
plt.grid(False)
# 循环遍历给定的一系列浮点数，用于绘制多个线条到当前图形中
for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    # 绘制从 (0, x) 到 (x, 0) 的线条，颜色为黑色，透明度为0.2
    plt.plot([0, x], [x, 0], "k", alpha=0.2)
    # 绘制从 (0, x) 到 (0 + (1 - x) / 2, x + (1 - x) / 2) 的线条，颜色为黑色，透明度为0.2
    plt.plot([0, 0 + (1 - x) / 2], [x, x + (1 - x) / 2], "k", alpha=0.2)
    # 绘制从 (x, 0) 到 (x + (1 - x) / 2, 0 + (1 - x) / 2) 的线条，颜色为黑色，透明度为0.2
    plt.plot([x, x + (1 - x) / 2], [0, 0 + (1 - x) / 2], "k", alpha=0.2)

# 设置图形标题
plt.title("Learned sigmoid calibration map")
# 设置 x 轴标签
plt.xlabel("Probability class 1")
# 设置 y 轴标签
plt.ylabel("Probability class 2")
# 设置 x 轴的显示范围
plt.xlim(-0.05, 1.05)
# 设置 y 轴的显示范围
plt.ylim(-0.05, 1.05)

# 显示绘制的图形
plt.show()
```