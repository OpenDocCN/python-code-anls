# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_gradient_boosting_oob.py`

```
"""
======================================
Gradient Boosting Out-of-Bag estimates
======================================
Out-of-bag (OOB) estimates can be a useful heuristic to estimate
the "optimal" number of boosting iterations.
OOB estimates are almost identical to cross-validation estimates but
they can be computed on-the-fly without the need for repeated model
fitting.
OOB estimates are only available for Stochastic Gradient Boosting
(i.e. ``subsample < 1.0``), the estimates are derived from the improvement
in loss based on the examples not included in the bootstrap sample
(the so-called out-of-bag examples).
The OOB estimator is a pessimistic estimator of the true
test loss, but remains a fairly good approximation for a small number of trees.
The figure shows the cumulative sum of the negative OOB improvements
as a function of the boosting iteration. As you can see, it tracks the test
loss for the first hundred iterations but then diverges in a
pessimistic way.
The figure also shows the performance of 3-fold cross validation which
usually gives a better estimate of the test loss
but is computationally more demanding.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，用于绘图
import numpy as np  # 导入numpy模块，用于数值计算
from scipy.special import expit  # 导入scipy.special模块中的expit函数，用于逻辑斯蒂回归的激活函数

from sklearn import ensemble  # 导入sklearn中的ensemble模块，用于集成学习方法
from sklearn.metrics import log_loss  # 导入sklearn中的log_loss函数，用于计算逻辑回归的损失
from sklearn.model_selection import KFold, train_test_split  # 导入sklearn中的KFold和train_test_split函数，用于交叉验证和数据集划分

# Generate data (adapted from G. Ridgeway's gbm example)
n_samples = 1000  # 设置样本数量为1000
random_state = np.random.RandomState(13)  # 使用随机种子13创建随机状态对象
x1 = random_state.uniform(size=n_samples)  # 生成均匀分布的随机数作为特征x1
x2 = random_state.uniform(size=n_samples)  # 生成均匀分布的随机数作为特征x2
x3 = random_state.randint(0, 4, size=n_samples)  # 生成0到3之间的随机整数作为特征x3

p = expit(np.sin(3 * x1) - 4 * x2 + x3)  # 计算概率p，使用expit函数对逻辑斯蒂回归进行建模
y = random_state.binomial(1, p, size=n_samples)  # 生成二项分布的随机数作为目标变量y

X = np.c_[x1, x2, x3]  # 将特征x1, x2, x3按列合并成特征矩阵X

X = X.astype(np.float32)  # 将特征矩阵X的数据类型转换为np.float32
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=9)  # 将数据集划分为训练集和测试集

# Fit classifier with out-of-bag estimates
params = {  # 定义梯度提升分类器的参数
    "n_estimators": 1200,  # 弱学习器的数量
    "max_depth": 3,  # 每个弱学习器的最大深度
    "subsample": 0.5,  # 子样本的比例
    "learning_rate": 0.01,  # 学习率
    "min_samples_leaf": 1,  # 叶节点最少样本数
    "random_state": 3,  # 随机种子
}
clf = ensemble.GradientBoostingClassifier(**params)  # 创建梯度提升分类器对象，使用指定参数

clf.fit(X_train, y_train)  # 在训练集上拟合分类器
acc = clf.score(X_test, y_test)  # 在测试集上评估分类器的准确率
print("Accuracy: {:.4f}".format(acc))  # 打印分类器在测试集上的准确率

n_estimators = params["n_estimators"]  # 获取弱学习器的数量
x = np.arange(n_estimators) + 1  # 创建包含弱学习器数量的数组，从1开始计数


def heldout_score(clf, X_test, y_test):
    """compute deviance scores on ``X_test`` and ``y_test``."""
    score = np.zeros((n_estimators,), dtype=np.float64)  # 创建一个全零数组，用于存储每个迭代步骤的评分
    for i, y_proba in enumerate(clf.staged_predict_proba(X_test)):  # 枚举每个迭代步骤的预测概率
        score[i] = 2 * log_loss(y_test, y_proba[:, 1])  # 计算对数损失并存储到score数组中
    return score  # 返回评分数组


def cv_estimate(n_splits=None):
    cv = KFold(n_splits=n_splits)  # 创建指定折数的交叉验证对象
    cv_clf = ensemble.GradientBoostingClassifier(**params)  # 创建梯度提升分类器对象，使用指定参数
    val_scores = np.zeros((n_estimators,), dtype=np.float64)  # 创建一个全零数组，用于存储每个迭代步骤的验证分数
    for train, test in cv.split(X_train, y_train):  # 遍历交叉验证的每一折
        cv_clf.fit(X_train[train], y_train[train])  # 在训练集的当前折上拟合分类器
        val_scores += heldout_score(cv_clf, X_train[test], y_train[test])  # 计算当前折的验证分数并累加到val_scores中
    # 将交叉验证每次的验证分数取平均值，得到最终的验证分数
    val_scores /= n_splits
    # 返回最终的验证分数作为函数的输出结果
    return val_scores
# 通过交叉验证估算最佳的 n_estimator
cv_score = cv_estimate(3)

# 计算测试数据的最佳 n_estimator
test_score = heldout_score(clf, X_test, y_test)

# 计算负的累积OOB改进值
cumsum = -np.cumsum(clf.oob_improvement_)

# 根据OOB计算最小损失
oob_best_iter = x[np.argmin(cumsum)]

# 根据测试数据计算最小损失（使第一个损失为0进行归一化）
test_score -= test_score[0]
test_best_iter = x[np.argmin(test_score)]

# 根据交叉验证数据计算最小损失（使第一个损失为0进行归一化）
cv_score -= cv_score[0]
cv_best_iter = x[np.argmin(cv_score)]

# 设定三条曲线的颜色
oob_color = list(map(lambda x: x / 256.0, (190, 174, 212)))
test_color = list(map(lambda x: x / 256.0, (127, 201, 127)))
cv_color = list(map(lambda x: x / 256.0, (253, 192, 134)))

# 设定三条曲线的线型
oob_line = "dashed"
test_line = "solid"
cv_line = "dashdot"

# 绘制曲线并在最佳迭代次数处绘制垂直线
plt.figure(figsize=(8, 4.8))
plt.plot(x, cumsum, label="OOB loss", color=oob_color, linestyle=oob_line)
plt.plot(x, test_score, label="Test loss", color=test_color, linestyle=test_line)
plt.plot(x, cv_score, label="CV loss", color=cv_color, linestyle=cv_line)
plt.axvline(x=oob_best_iter, color=oob_color, linestyle=oob_line)
plt.axvline(x=test_best_iter, color=test_color, linestyle=test_line)
plt.axvline(x=cv_best_iter, color=cv_color, linestyle=cv_line)

# 将三条垂直线添加到x轴刻度
xticks = plt.xticks()
xticks_pos = np.array(
    xticks[0].tolist() + [oob_best_iter, cv_best_iter, test_best_iter]
)
xticks_label = np.array(list(map(lambda t: int(t), xticks[0])) + ["OOB", "CV", "Test"])
ind = np.argsort(xticks_pos)
xticks_pos = xticks_pos[ind]
xticks_label = xticks_label[ind]
plt.xticks(xticks_pos, xticks_label, rotation=90)

plt.legend(loc="upper center")
plt.ylabel("normalized loss")
plt.xlabel("number of iterations")

plt.show()
```