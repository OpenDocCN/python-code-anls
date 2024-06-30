# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_confusion_matrix.py`

```
"""
================
混淆矩阵
================

使用混淆矩阵来评估分类器在鸢尾花数据集上输出质量的示例。对角线元素表示预测标签等于真实标签的数据点数量，而非对角线元素则表示分类器误分类的数据点数量。混淆矩阵的对角线值越高越好，表示有许多正确的预测。

图表展示了带有和不带有类支持大小（每个类中元素数量）归一化的混淆矩阵。这种归一化在存在类别不平衡时很有意义，可以更直观地解释哪些类别被误分类。

在这里，结果并不如预期那样好，因为我们选择的正则化参数 C 不是最佳的。在实际应用中，通常使用 :ref:`grid_search` 来选择这个参数。

"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 模块，用于数值计算

from sklearn import datasets, svm  # 导入 sklearn 中的 datasets 和 svm 模块
from sklearn.metrics import ConfusionMatrixDisplay  # 从 sklearn.metrics 中导入 ConfusionMatrixDisplay 类
from sklearn.model_selection import train_test_split  # 从 sklearn.model_selection 中导入 train_test_split 函数

# 导入一些数据用于后续处理
iris = datasets.load_iris()  # 加载鸢尾花数据集
X = iris.data  # 特征数据
y = iris.target  # 目标数据（标签）
class_names = iris.target_names  # 类别名称

# 将数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 运行分类器，使用一个过于正则化的模型（C 值太低），以便观察结果的影响
classifier = svm.SVC(kernel="linear", C=0.01).fit(X_train, y_train)

np.set_printoptions(precision=2)  # 设置打印选项，精确度为两位小数

# 绘制非归一化的混淆矩阵
titles_options = [
    ("混淆矩阵，不进行归一化", None),
    ("归一化的混淆矩阵", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,  # 使用蓝色色谱绘制图表
        normalize=normalize,
    )
    disp.ax_.set_title(title)  # 设置图表标题

    print(title)  # 打印标题
    print(disp.confusion_matrix)  # 打印混淆矩阵数据

plt.show()  # 显示绘制的图表
```