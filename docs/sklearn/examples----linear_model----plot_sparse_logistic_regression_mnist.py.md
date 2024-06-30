# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_sparse_logistic_regression_mnist.py`

```
"""
=====================================================
MNIST classification using multinomial logistic + L1
=====================================================

Here we fit a multinomial logistic regression with L1 penalty on a subset of
the MNIST digits classification task. We use the SAGA algorithm for this
purpose: this a solver that is fast when the number of samples is significantly
larger than the number of features and is able to finely optimize non-smooth
objective functions which is the case with the l1-penalty. Test accuracy
reaches > 0.8, while weight vectors remains *sparse* and therefore more easily
*interpretable*.

Note that this accuracy of this l1-penalized linear model is significantly
below what can be reached by an l2-penalized linear model or a non-linear
multi-layer perceptron model on this dataset.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml              # 导入用于获取数据集的函数
from sklearn.linear_model import LogisticRegression    # 导入逻辑回归模型
from sklearn.model_selection import train_test_split   # 导入数据集分割函数
from sklearn.preprocessing import StandardScaler       # 导入数据标准化函数
from sklearn.utils import check_random_state           # 导入用于检查随机状态的函数

# Turn down for faster convergence
t0 = time.time()                                         # 记录开始时间
train_samples = 5000                                     # 训练样本数目

# Load data from https://www.openml.org/d/554
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)  # 获取MNIST数据集

random_state = check_random_state(0)                      # 设置随机数生成器的种子
permutation = random_state.permutation(X.shape[0])        # 对数据进行随机排列
X = X[permutation]                                        # 对特征进行重排
y = y[permutation]                                        # 对标签进行重排
X = X.reshape((X.shape[0], -1))                           

X_train, X_test, y_train, y_test = train_test_split(      # 将数据集划分为训练集和测试集
    X, y, train_size=train_samples, test_size=10000
)

scaler = StandardScaler()                                 # 创建数据标准化器
X_train = scaler.fit_transform(X_train)                   # 对训练集进行标准化
X_test = scaler.transform(X_test)                         # 对测试集进行标准化

# Turn up tolerance for faster convergence
clf = LogisticRegression(                                 # 创建逻辑回归分类器
    C=50.0 / train_samples, penalty="l1", solver="saga", tol=0.1
)
clf.fit(X_train, y_train)                                 # 在训练集上拟合模型
sparsity = np.mean(clf.coef_ == 0) * 100                  # 计算稀疏性
score = clf.score(X_test, y_test)                         # 计算模型在测试集上的得分
# print('Best C % .4f' % clf.C_)
print("Sparsity with L1 penalty: %.2f%%" % sparsity)      # 打印稀疏性
print("Test score with L1 penalty: %.4f" % score)         # 打印测试集得分

coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))                               # 创建画布
scale = np.abs(coef).max()                                # 计算权重系数的最大绝对值
for i in range(10):                                       # 遍历10个数字类别
    l1_plot = plt.subplot(2, 5, i + 1)                    # 创建子图
    l1_plot.imshow(                                        # 在子图上绘制权重向量
        coef[i].reshape(28, 28),
        interpolation="nearest",
        cmap=plt.cm.RdBu,
        vmin=-scale,
        vmax=scale,
    )
    l1_plot.set_xticks(())                                # 设置坐标轴刻度
    l1_plot.set_yticks(())
    l1_plot.set_xlabel("Class %i" % i)                    # 设置x轴标签
plt.suptitle("Classification vector for...")               # 设置总标题

run_time = time.time() - t0                               # 计算运行时间
print("Example run in %.3f s" % run_time)                  # 打印运行时间
plt.show()                                                # 显示图形
```