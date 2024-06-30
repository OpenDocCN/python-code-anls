# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_iris_logistic.py`

```
"""
=========================================================
Logistic Regression 3-class Classifier
=========================================================

Show below is a logistic-regression classifiers decision boundaries on the
first two dimensions (sepal length and width) of the `iris
<https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ dataset. The datapoints
are colored according to their labels.

"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# 导入 sklearn 库中的相关模块和类
from sklearn import datasets  # 导入数据集模块
from sklearn.inspection import DecisionBoundaryDisplay  # 导入绘制决策边界的显示类
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型

# 从 sklearn 中导入 iris 数据集
iris = datasets.load_iris()
X = iris.data[:, :2]  # 取出数据集的前两个特征作为输入特征
Y = iris.target  # 取出数据集的标签

# 创建一个逻辑回归分类器的实例，并拟合数据
logreg = LogisticRegression(C=1e5)
logreg.fit(X, Y)

# 创建一个大小为 (4, 3) 的子图
_, ax = plt.subplots(figsize=(4, 3))

# 使用 DecisionBoundaryDisplay 类从分类器创建决策边界的显示
DecisionBoundaryDisplay.from_estimator(
    logreg,
    X,
    cmap=plt.cm.Paired,  # 设置颜色映射
    ax=ax,  # 指定绘图的轴对象
    response_method="predict",  # 设置响应方法为预测方法
    plot_method="pcolormesh",  # 使用 pcolormesh 方法绘制图形
    shading="auto",  # 根据数据自动填充颜色
    xlabel="Sepal length",  # x 轴标签
    ylabel="Sepal width",  # y 轴标签
    eps=0.5,  # 控制图形的边界扩展
)

# 绘制训练数据点，根据标签颜色进行着色，边缘用黑色表示
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors="k", cmap=plt.cm.Paired)

# 设置 x 轴和 y 轴的刻度为空
plt.xticks(())
plt.yticks(())

# 显示绘制的图形
plt.show()
```