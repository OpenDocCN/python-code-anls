# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_sgd_iris.py`

```
"""
========================================
Plot multi-class SGD on the iris dataset
========================================

Plot decision surface of multi-class SGD on iris dataset.
The hyperplanes corresponding to the three one-versus-all (OVA) classifiers
are represented by the dashed lines.

"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

from sklearn import datasets  # 导入sklearn中的datasets模块，用于加载数据集
from sklearn.inspection import DecisionBoundaryDisplay  # 导入sklearn中的DecisionBoundaryDisplay模块，用于显示决策边界
from sklearn.linear_model import SGDClassifier  # 导入sklearn中的SGDClassifier模块，用于SGD分类器

# import some data to play with
# 加载鸢尾花数据集
iris = datasets.load_iris()

# we only take the first two features. We could
# avoid this ugly slicing by using a two-dim dataset
# 只取前两个特征，避免使用切片可以通过使用二维数据集来避免
X = iris.data[:, :2]  # 取出数据集中的前两列特征作为X
y = iris.target  # 将数据集的目标标签赋值给y
colors = "bry"  # 设置颜色标记

# shuffle
# 打乱数据顺序
idx = np.arange(X.shape[0])  # 创建一个与样本数相同的索引数组
np.random.seed(13)  # 设定随机种子，保证每次运行结果一致
np.random.shuffle(idx)  # 将索引数组随机打乱
X = X[idx]  # 根据打乱后的索引重新排序特征数据
y = y[idx]  # 根据打乱后的索引重新排序标签数据

# standardize
# 数据标准化
mean = X.mean(axis=0)  # 计算特征的均值
std = X.std(axis=0)  # 计算特征的标准差
X = (X - mean) / std  # 标准化特征数据

# 使用SGDClassifier拟合数据
clf = SGDClassifier(alpha=0.001, max_iter=100).fit(X, y)  # 使用SGDClassifier拟合数据集
ax = plt.gca()  # 获取当前的Axes对象

# 绘制决策边界
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    cmap=plt.cm.Paired,  # 设置颜色映射为Paired
    ax=ax,  # 指定绘图的Axes对象
    response_method="predict",  # 设置响应方法为predict
    xlabel=iris.feature_names[0],  # x轴标签为第一个特征的名称
    ylabel=iris.feature_names[1],  # y轴标签为第二个特征的名称
)
plt.axis("tight")  # 自动调整坐标轴范围以适应数据点

# Plot also the training points
# 绘制训练数据点
for i, color in zip(clf.classes_, colors):
    idx = np.where(y == i)  # 找到标签为i的数据索引
    plt.scatter(
        X[idx, 0],
        X[idx, 1],
        c=color,
        label=iris.target_names[i],
        edgecolor="black",
        s=20,
    )

plt.title("Decision surface of multi-class SGD")  # 设置图表标题
plt.axis("tight")  # 自动调整坐标轴范围以适应数据点

# Plot the three one-against-all classifiers
# 绘制三个一对多分类器的超平面
xmin, xmax = plt.xlim()  # 获取x轴坐标范围
ymin, ymax = plt.ylim()  # 获取y轴坐标范围
coef = clf.coef_  # 获取分类器的系数
intercept = clf.intercept_  # 获取分类器的截距


def plot_hyperplane(c, color):
    # 定义绘制超平面的函数
    def line(x0):
        return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

    plt.plot([xmin, xmax], [line(xmin), line(xmax)], ls="--", color=color)  # 绘制超平面线段


for i, color in zip(clf.classes_, colors):
    plot_hyperplane(i, color)  # 调用函数绘制每个类别的超平面

plt.legend()  # 显示图例
plt.show()  # 展示图表
```