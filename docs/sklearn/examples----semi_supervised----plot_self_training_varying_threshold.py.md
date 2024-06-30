# `D:\src\scipysrc\scikit-learn\examples\semi_supervised\plot_self_training_varying_threshold.py`

```
"""
=============================================
Effect of varying threshold for self-training
=============================================

This example illustrates the effect of a varying threshold on self-training.
The `breast_cancer` dataset is loaded, and labels are deleted such that only 50
out of 569 samples have labels. A `SelfTrainingClassifier` is fitted on this
dataset, with varying thresholds.

The upper graph shows the amount of labeled samples that the classifier has
available by the end of fit, and the accuracy of the classifier. The lower
graph shows the last iteration in which a sample was labeled. All values are
cross validated with 3 folds.

At low thresholds (in [0.4, 0.5]), the classifier learns from samples that were
labeled with a low confidence. These low-confidence samples are likely have
incorrect predicted labels, and as a result, fitting on these incorrect labels
produces a poor accuracy. Note that the classifier labels almost all of the
samples, and only takes one iteration.

For very high thresholds (in [0.9, 1)) we observe that the classifier does not
augment its dataset (the amount of self-labeled samples is 0). As a result, the
accuracy achieved with a threshold of 0.9999 is the same as a normal supervised
classifier would achieve.

The optimal accuracy lies in between both of these extremes at a threshold of
around 0.7.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 NumPy 数学计算库

from sklearn import datasets  # 导入 scikit-learn 中的数据集模块
from sklearn.metrics import accuracy_score  # 导入 scikit-learn 中的准确率评估函数
from sklearn.model_selection import StratifiedKFold  # 导入 scikit-learn 中的分层 K 折交叉验证
from sklearn.semi_supervised import SelfTrainingClassifier  # 导入 scikit-learn 中的半监督学习的自训练分类器
from sklearn.svm import SVC  # 导入 scikit-learn 中的支持向量机分类器
from sklearn.utils import shuffle  # 导入 scikit-learn 中的数据洗牌函数

n_splits = 3  # 设置交叉验证的折数为 3

X, y = datasets.load_breast_cancer(return_X_y=True)  # 加载乳腺癌数据集并获取特征 X 和标签 y
X, y = shuffle(X, y, random_state=42)  # 将数据集随机打乱，确保数据的随机性
y_true = y.copy()  # 复制一份原始标签 y 作为真实标签备份
y[50:] = -1  # 删除数据集中除前 50 个样本外的所有标签，用 -1 表示未标记样本
total_samples = y.shape[0]  # 获取数据集总样本数

base_classifier = SVC(probability=True, gamma=0.001, random_state=42)  # 初始化支持向量机分类器作为基分类器，设置参数

x_values = np.arange(0.4, 1.05, 0.05)  # 创建一个数组，包含从 0.4 到 1.0 步长为 0.05 的数值
x_values = np.append(x_values, 0.99999)  # 添加额外的值 0.99999 到 x_values 数组中
scores = np.empty((x_values.shape[0], n_splits))  # 创建一个空的二维数组，用于存储每个阈值对应的交叉验证分数
amount_labeled = np.empty((x_values.shape[0], n_splits))  # 创建一个空的二维数组，用于存储每个阈值下的标记样本数量
amount_iterations = np.empty((x_values.shape[0], n_splits))  # 创建一个空的二维数组，用于存储每个阈值下的迭代次数

for i, threshold in enumerate(x_values):  # 迭代 x_values 数组中的阈值
    self_training_clf = SelfTrainingClassifier(base_classifier, threshold=threshold)  # 初始化自训练分类器对象

    # We need manual cross validation so that we don't treat -1 as a separate
    # class when computing accuracy
    skfolds = StratifiedKFold(n_splits=n_splits)  # 使用分层 K 折交叉验证，将数据集划分为 n_splits 折
    # 使用 StratifiedKFold 进行交叉验证划分数据集，返回每折的训练集和测试集索引
    for fold, (train_index, test_index) in enumerate(skfolds.split(X, y)):
        # 根据交叉验证划分得到的索引，从原始数据集 X 和 y 中分别取出训练集和测试集数据
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        # 取出真实的测试集标签，用于评估预测准确率
        y_test_true = y_true[test_index]

        # 使用自我训练分类器拟合训练集数据
        self_training_clf.fit(X_train, y_train)

        # 计算在当前折结束训练后，被标记的样本数量
        amount_labeled[i, fold] = (
            total_samples
            - np.unique(self_training_clf.labeled_iter_, return_counts=True)[1][0]
        )
        # 记录分类器最后一次迭代中标记样本的迭代次数
        amount_iterations[i, fold] = np.max(self_training_clf.labeled_iter_)

        # 使用训练好的分类器进行测试集的预测
        y_pred = self_training_clf.predict(X_test)
        # 计算预测准确率并存储到 scores 数组中
        scores[i, fold] = accuracy_score(y_test_true, y_pred)
# 创建一个子图 ax1，在第一个位置（2行1列中的第1行），用于绘制误差条图
ax1 = plt.subplot(211)

# 在 ax1 上绘制误差条图，显示平均值和标准差，x轴为 x_values，y轴为 scores 的均值，误差为 scores 的标准差，误差条帽长为2，颜色为蓝色
ax1.errorbar(
    x_values, scores.mean(axis=1), yerr=scores.std(axis=1), capsize=2, color="b"
)

# 设置 ax1 的 y轴标签为 "Accuracy"，颜色为蓝色
ax1.set_ylabel("Accuracy", color="b")

# 设置 ax1 的 y轴刻度参数，使其颜色与标签相同（蓝色）
ax1.tick_params("y", colors="b")

# 创建一个与 ax1 共享 x轴的双轴图 ax2
ax2 = ax1.twinx()

# 在 ax2 上绘制误差条图，显示平均值和标准差，x轴为 x_values，y轴为 amount_labeled 的均值，误差为 amount_labeled 的标准差，误差条帽长为2，颜色为绿色
ax2.errorbar(
    x_values,
    amount_labeled.mean(axis=1),
    yerr=amount_labeled.std(axis=1),
    capsize=2,
    color="g",
)

# 设置 ax2 的 y轴下限为0
ax2.set_ylim(bottom=0)

# 设置 ax2 的 y轴标签为 "Amount of labeled samples"，颜色为绿色
ax2.set_ylabel("Amount of labeled samples", color="g")

# 设置 ax2 的 y轴刻度参数，使其颜色与标签相同（绿色）
ax2.tick_params("y", colors="g")

# 创建一个子图 ax3，在第二个位置（2行1列中的第2行），与 ax1 共享 x轴
ax3 = plt.subplot(212, sharex=ax1)

# 在 ax3 上绘制误差条图，显示平均值和标准差，x轴为 x_values，y轴为 amount_iterations 的均值，误差为 amount_iterations 的标准差，误差条帽长为2，颜色为蓝色
ax3.errorbar(
    x_values,
    amount_iterations.mean(axis=1),
    yerr=amount_iterations.std(axis=1),
    capsize=2,
    color="b",
)

# 设置 ax3 的 y轴下限为0
ax3.set_ylim(bottom=0)

# 设置 ax3 的 y轴标签为 "Amount of iterations"
ax3.set_ylabel("Amount of iterations")

# 设置 ax3 的 x轴标签为 "Threshold"
ax3.set_xlabel("Threshold")

# 显示绘制的所有子图
plt.show()
```