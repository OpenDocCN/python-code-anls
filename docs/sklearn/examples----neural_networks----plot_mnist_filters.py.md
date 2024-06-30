# `D:\src\scipysrc\scikit-learn\examples\neural_networks\plot_mnist_filters.py`

```
"""
=====================================
Visualization of MLP weights on MNIST
=====================================

Sometimes looking at the learned coefficients of a neural network can provide
insight into the learning behavior. For example if weights look unstructured,
maybe some were not used at all, or if very large coefficients exist, maybe
regularization was too low or the learning rate too high.

This example shows how to plot some of the first layer weights in a
MLPClassifier trained on the MNIST dataset.

The input data consists of 28x28 pixel handwritten digits, leading to 784
features in the dataset. Therefore the first layer weight matrix has the shape
(784, hidden_layer_sizes[0]).  We can therefore visualize a single column of
the weight matrix as a 28x28 pixel image.

To make the example run faster, we use very few hidden units, and train only
for a very short time. Training longer would result in weights with a much
smoother spatial appearance. The example will throw a warning because it
doesn't converge, in this case this is what we want because of resource
usage constraints on our Continuous Integration infrastructure that is used
to build this documentation on a regular basis.
"""

import warnings

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Load data from https://www.openml.org/d/554
# 从OpenML下载MNIST数据集，版本1，返回特征矩阵X和标签向量y
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
# 特征归一化到0到1之间
X = X / 255.0

# Split data into train partition and test partition
# 将数据分割成训练集和测试集，测试集占70%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.7)

# 创建MLP分类器对象
mlp = MLPClassifier(
    hidden_layer_sizes=(40,),   # 设置隐藏层神经元数量为40
    max_iter=8,                 # 最大迭代次数为8
    alpha=1e-4,                 # L2正则化参数
    solver="sgd",               # 使用随机梯度下降优化器
    verbose=10,                 # 输出训练过程信息每10次迭代
    random_state=1,             # 随机种子设置为1，保证结果可复现性
    learning_rate_init=0.2,     # 初始学习率为0.2
)

# this example won't converge because of resource usage constraints on
# our Continuous Integration infrastructure, so we catch the warning and
# ignore it here
# 捕获MLPClassifier可能抛出的收敛警告，忽略该警告信息
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    # 使用训练集训练MLP模型
    mlp.fit(X_train, y_train)

# 输出训练集和测试集的准确率评分
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

# 创建子图，每个子图显示一个隐藏层的权重
fig, axes = plt.subplots(4, 4)
# 使用全局最小值和最大值确保所有权重在同一比例尺上显示
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
# 遍历第一个隐藏层的权重，绘制成28x28像素的灰度图像
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=0.5 * vmin, vmax=0.5 * vmax)
    ax.set_xticks(())   # 不显示X轴刻度
    ax.set_yticks(())   # 不显示Y轴刻度

# 显示绘制的子图
plt.show()
```