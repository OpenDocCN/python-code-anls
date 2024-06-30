# `D:\src\scipysrc\scikit-learn\examples\exercises\plot_digits_classification_exercise.py`

```
"""
================================
Digits Classification Exercise
================================

A tutorial exercise regarding the use of classification techniques on
the Digits dataset.

This exercise is used in the :ref:`clf_tut` part of the
:ref:`supervised_learning_tut` section of the
:ref:`stat_learn_tut_index`.

"""

# 导入需要的模块和函数
from sklearn import datasets, linear_model, neighbors

# 加载手写数字数据集，返回特征矩阵 X_digits 和目标向量 y_digits
X_digits, y_digits = datasets.load_digits(return_X_y=True)

# 将特征矩阵 X_digits 归一化，除以其最大值
X_digits = X_digits / X_digits.max()

# 获取样本数
n_samples = len(X_digits)

# 划分数据集为训练集和测试集，按照 9:1 的比例
X_train = X_digits[: int(0.9 * n_samples)]  # 训练集特征
y_train = y_digits[: int(0.9 * n_samples)]  # 训练集目标
X_test = X_digits[int(0.9 * n_samples) :]   # 测试集特征
y_test = y_digits[int(0.9 * n_samples) :]   # 测试集目标

# 创建 KNN 分类器对象
knn = neighbors.KNeighborsClassifier()

# 创建 Logistic 回归分类器对象，设定最大迭代次数为1000
logistic = linear_model.LogisticRegression(max_iter=1000)

# 训练并打印 KNN 分类器在测试集上的得分
print("KNN score: %f" % knn.fit(X_train, y_train).score(X_test, y_test))

# 训练并打印 Logistic 回归分类器在测试集上的得分
print(
    "LogisticRegression score: %f"
    % logistic.fit(X_train, y_train).score(X_test, y_test)
)
```