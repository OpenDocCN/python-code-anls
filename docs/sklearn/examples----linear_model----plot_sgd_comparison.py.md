# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_sgd_comparison.py`

```
"""
==================================
Comparing various online solvers
==================================
An example showing how different online solvers perform
on the hand-written digits dataset.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图
import numpy as np  # 导入 numpy 库用于数值计算

from sklearn import datasets  # 导入 sklearn 中的数据集模块
from sklearn.linear_model import (  # 导入 sklearn 中的线性模型
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    SGDClassifier,
)
from sklearn.model_selection import train_test_split  # 导入 sklearn 中的数据集划分模块

heldout = [0.95, 0.90, 0.75, 0.50, 0.01]  # 定义测试集大小的列表
# Number of rounds to fit and evaluate an estimator.
rounds = 10  # 定义每个测试集大小下进行训练和评估的轮数
X, y = datasets.load_digits(return_X_y=True)  # 加载手写数字数据集

classifiers = [  # 定义不同分类器及其参数的列表
    ("SGD", SGDClassifier(max_iter=110)),
    ("ASGD", SGDClassifier(max_iter=110, average=True)),
    ("Perceptron", Perceptron(max_iter=110)),
    (
        "Passive-Aggressive I",
        PassiveAggressiveClassifier(max_iter=110, loss="hinge", C=1.0, tol=1e-4),
    ),
    (
        "Passive-Aggressive II",
        PassiveAggressiveClassifier(
            max_iter=110, loss="squared_hinge", C=1.0, tol=1e-4
        ),
    ),
    (
        "SAG",
        LogisticRegression(max_iter=110, solver="sag", tol=1e-1, C=1.0e4 / X.shape[0]),
    ),
]

xx = 1.0 - np.array(heldout)  # 计算训练集比例

for name, clf in classifiers:  # 遍历分类器列表
    print("training %s" % name)  # 打印当前分类器的训练信息
    rng = np.random.RandomState(42)  # 创建随机数生成器
    yy = []
    for i in heldout:  # 遍历测试集大小列表
        yy_ = []
        for r in range(rounds):  # 进行指定轮数的训练和评估
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=i, random_state=rng
            )  # 划分训练集和测试集
            clf.fit(X_train, y_train)  # 训练分类器
            y_pred = clf.predict(X_test)  # 预测测试集
            yy_.append(1 - np.mean(y_pred == y_test))  # 计算测试错误率
        yy.append(np.mean(yy_))  # 计算平均测试错误率
    plt.plot(xx, yy, label=name)  # 绘制学习曲线

plt.legend(loc="upper right")  # 设置图例位置
plt.xlabel("Proportion train")  # 设置 x 轴标签
plt.ylabel("Test Error Rate")  # 设置 y 轴标签
plt.show()  # 显示图形
```