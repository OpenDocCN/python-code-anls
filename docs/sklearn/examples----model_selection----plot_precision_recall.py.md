# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_precision_recall.py`

```
"""
================
Precision-Recall
================

Example of Precision-Recall metric to evaluate classifier output quality.

Precision-Recall is a useful measure of success of prediction when the
classes are very imbalanced. In information retrieval, precision is a
measure of result relevancy, while recall is a measure of how many truly
relevant results are returned.

The precision-recall curve shows the tradeoff between precision and
recall for different threshold. A high area under the curve represents
both high recall and high precision, where high precision relates to a
low false positive rate, and high recall relates to a low false negative
rate. High scores for both show that the classifier is returning accurate
results (high precision), as well as returning a majority of all positive
results (high recall).

A system with high recall but low precision returns many results, but most of
its predicted labels are incorrect when compared to the training labels. A
system with high precision but low recall is just the opposite, returning very
few results, but most of its predicted labels are correct when compared to the
training labels. An ideal system with high precision and high recall will
return many results, with all results labeled correctly.

Precision (:math:`P`) is defined as the number of true positives (:math:`T_p`)
over the number of true positives plus the number of false positives
(:math:`F_p`).

:math:`P = \\frac{T_p}{T_p+F_p}`

Recall (:math:`R`) is defined as the number of true positives (:math:`T_p`)
over the number of true positives plus the number of false negatives
(:math:`F_n`).

:math:`R = \\frac{T_p}{T_p + F_n}`

These quantities are also related to the :math:`F_1` score, which is the
harmonic mean of precision and recall. Thus, we can compute the :math:`F_1`
using the following formula:

:math:`F_1 = \\frac{2T_p}{2T_p + F_p + F_n}`

Note that the precision may not decrease with recall. The
definition of precision (:math:`\\frac{T_p}{T_p + F_p}`) shows that lowering
the threshold of a classifier may increase the denominator, by increasing the
number of results returned. If the threshold was previously set too high, the
new results may all be true positives, which will increase precision. If the
previous threshold was about right or too low, further lowering the threshold
will introduce false positives, decreasing precision.

Recall is defined as :math:`\\frac{T_p}{T_p+F_n}`, where :math:`T_p+F_n` does
not depend on the classifier threshold. This means that lowering the classifier
threshold may increase recall, by increasing the number of true positive
results. It is also possible that lowering the threshold may leave recall
unchanged, while the precision fluctuates.

The relationship between recall and precision can be observed in the
stairstep area of the plot - at the edges of these steps a small change
in the threshold considerably reduces precision, with only a minor gain in
recall.
"""
# %%
# Linear SVC classifier is imported to perform binary classification on iris dataset.
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris dataset, returning both features (X) and labels (y).
X, y = load_iris(return_X_y=True)

# %%
# Add noisy features to the dataset to increase dimensionality.
# This is done using a random state to ensure reproducibility.
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)

# %%
# Restrict the dataset to the first two classes (0 and 1) and split it into
# training and testing sets.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X[y < 2], y[y < 2], test_size=0.5, random_state=random_state
)

# %%
# Data scaling is performed using StandardScaler to ensure each feature
# has a similar range of values, which is expected by Linear SVC.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Create a pipeline that scales the data and applies Linear SVC classifier.
classifier = make_pipeline(StandardScaler(), LinearSVC(random_state=random_state))
classifier.fit(X_train, y_train)

# %%
# Plotting the Precision-Recall curve using PrecisionRecallDisplay.
# This displays the precision-recall curve, leveraging predictions
# made by the classifier. The curve illustrates the trade-off between
# precision and recall at different thresholds.
# 导入 PrecisionRecallDisplay 类，用于绘制精确率-召回率曲线
from sklearn.metrics import PrecisionRecallDisplay

# 使用 PrecisionRecallDisplay 类的 from_estimator 方法创建一个显示对象，从分类器、测试数据和标签创建
# 该对象并设置标题为 "LinearSVC"，同时绘制机会水平线
display = PrecisionRecallDisplay.from_estimator(
    classifier, X_test, y_test, name="LinearSVC", plot_chance_level=True
)

# 设置绘图的标题为 "2-class Precision-Recall curve"
_ = display.ax_.set_title("2-class Precision-Recall curve")

# %%
# 如果已经获取了模型的估计概率或分数，可以使用 from_predictions 函数来创建 PrecisionRecallDisplay 对象
# :func:`~sklearn.metrics.PrecisionRecallDisplay.from_predictions` 用于这种情况
y_score = classifier.decision_function(X_test)

# 使用 from_predictions 方法创建另一个 PrecisionRecallDisplay 对象，从真实标签和预测分数创建
# 设置标题为 "LinearSVC"，同时绘制机会水平线
display = PrecisionRecallDisplay.from_predictions(
    y_test, y_score, name="LinearSVC", plot_chance_level=True
)

# 设置绘图的标题为 "2-class Precision-Recall curve"
_ = display.ax_.set_title("2-class Precision-Recall curve")

# %%
# 在多标签设置中
# -----------------------
#
# 精确率-召回率曲线不支持多标签设置。然而，可以决定如何处理这种情况。我们在下面展示一个示例。
#
# 创建多标签数据，拟合并预测
# .........................................
#
# 创建一个多标签数据集，以说明多标签设置中的精确率-召回率。

# 导入 label_binarize 函数，将标签二值化以创建多标签样式的设置
from sklearn.preprocessing import label_binarize

# 使用 label_binarize 函数进行多标签样式的设置
Y = label_binarize(y, classes=[0, 1, 2])
n_classes = Y.shape[1]

# 将数据集分割为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.5, random_state=random_state
)

# %%
# 我们使用 :class:`~sklearn.multiclass.OneVsRestClassifier` 进行多标签预测。
from sklearn.multiclass import OneVsRestClassifier

# 使用 OneVsRestClassifier 将线性 SVM 与标准化管道组合，创建多标签分类器
classifier = OneVsRestClassifier(
    make_pipeline(StandardScaler(), LinearSVC(random_state=random_state))
)

# 拟合分类器使用训练数据
classifier.fit(X_train, Y_train)

# 使用 decision_function 方法获取预测的分数
y_score = classifier.decision_function(X_test)

# %%
# 在多标签设置中计算平均精确度分数
# ...................................................
from sklearn.metrics import average_precision_score, precision_recall_curve

# 初始化精确率、召回率和平均精确度的字典
precision = dict()
recall = dict()
average_precision = dict()

# 对每个类别计算精确率-召回率曲线和平均精确度
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# 计算 "micro-average"：计算所有类别的联合得分
precision["micro"], recall["micro"], _ = precision_recall_curve(
    Y_test.ravel(), y_score.ravel()
)
average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")

# %%
# 绘制 "micro-averaged" 精确率-召回率曲线
# ..............................................
from collections import Counter

# 使用 PrecisionRecallDisplay 类创建显示对象，传入 "micro" 的召回率、精确率和平均精确度
# 同时计算正例的概率
display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
    prevalence_pos_label=Counter(Y_test.ravel())[1] / Y_test.size,
)

# 绘制精确率-召回率曲线，同时显示机会水平线
display.plot(plot_chance_level=True)

# 设置绘图的标题为 "Micro-averaged over all classes"
_ = display.ax_.set_title("Micro-averaged over all classes")

# %%
# 绘制每个类别的精确率-召回率曲线和 iso-f1 曲线
# ............................................................
from itertools import cycle
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图

# 设置绘图的颜色循环
colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

# 创建一个图形和一个坐标轴对象，设置图形大小为 (7, 8) inches
_, ax = plt.subplots(figsize=(7, 8))

# 在 0.2 到 0.8 之间生成四个均匀分布的数值作为 f 分数
f_scores = np.linspace(0.2, 0.8, num=4)

# 初始化线条和标签列表
lines, labels = [], []

# 遍历每个 f 分数
for f_score in f_scores:
    # 在 x 轴上生成从 0.01 到 1 的均匀分布的数值
    x = np.linspace(0.01, 1)
    # 计算对应的 y 值，这是一个 F1 分数的函数
    y = f_score * x / (2 * x - f_score)
    # 绘制灰色的 iso-f1 曲线，透明度设为 0.2
    (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
    # 在图中标注 F1 分数的数值
    plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

# 创建 PrecisionRecallDisplay 对象，并绘制微平均精确度-召回率曲线，颜色为金色
display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

# 遍历每个类别和颜色，分别绘制精确度-召回率曲线
for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

# 获取 iso-f1 曲线的图例句柄和标签
handles, labels = display.ax_.get_legend_handles_labels()
# 将灰色的 iso-f1 曲线的句柄添加到图例句柄列表中
handles.extend([l])
# 将 "iso-f1 curves" 添加到标签列表中
labels.extend(["iso-f1 curves"])
# 设置图例和坐标轴
ax.legend(handles=handles, labels=labels, loc="best")
# 设置图的标题
ax.set_title("Extension of Precision-Recall curve to multi-class")

# 显示图形
plt.show()
```