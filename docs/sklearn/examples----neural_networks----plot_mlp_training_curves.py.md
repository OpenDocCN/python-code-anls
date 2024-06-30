# `D:\src\scipysrc\scikit-learn\examples\neural_networks\plot_mlp_training_curves.py`

```
"""
========================================================
Compare Stochastic learning strategies for MLPClassifier
========================================================

This example visualizes some training loss curves for different stochastic
learning strategies, including SGD and Adam. Because of time-constraints, we
use several small datasets, for which L-BFGS might be more suitable. The
general trend shown in these examples seems to carry over to larger datasets,
however.

Note that those results can be highly dependent on the value of
``learning_rate_init``.

"""

# 导入警告模块
import warnings

# 导入绘图模块
import matplotlib.pyplot as plt

# 导入数据集和异常处理模块
from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning

# 导入神经网络模型和数据预处理模块
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

# 不同的学习率调度和动量参数设定
params = [
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "momentum": 0,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "momentum": 0.9,
        "nesterovs_momentum": False,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "momentum": 0.9,
        "nesterovs_momentum": True,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
        "momentum": 0,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
        "momentum": 0.9,
        "nesterovs_momentum": False,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
        "momentum": 0.9,
        "nesterovs_momentum": True,
        "learning_rate_init": 0.2,
    },
    {"solver": "adam", "learning_rate_init": 0.01},
]

# 不同策略的标签
labels = [
    "constant learning-rate",
    "constant with momentum",
    "constant with Nesterov's momentum",
    "inv-scaling learning-rate",
    "inv-scaling with momentum",
    "inv-scaling with Nesterov's momentum",
    "adam",
]

# 绘图参数设定
plot_args = [
    {"c": "red", "linestyle": "-"},
    {"c": "green", "linestyle": "-"},
    {"c": "blue", "linestyle": "-"},
    {"c": "red", "linestyle": "--"},
    {"c": "green", "linestyle": "--"},
    {"c": "blue", "linestyle": "--"},
    {"c": "black", "linestyle": "-"},
]

# 定义绘图函数，用于不同数据集上的绘图
def plot_on_dataset(X, y, ax, name):
    # 打印当前数据集名称
    print("\nlearning on dataset %s" % name)
    # 设置子图标题为数据集名称
    ax.set_title(name)

    # 数据归一化处理
    X = MinMaxScaler().fit_transform(X)
    
    # 根据数据集不同选择最大迭代次数
    mlps = []
    if name == "digits":
        # digits 数据集较大但收敛较快
        max_iter = 15
    else:
        max_iter = 400
    # 遍历标签和参数的组合，用于训练多层感知机分类器
    for label, param in zip(labels, params):
        # 输出当前正在训练的标签信息
        print("training: %s" % label)
        # 使用给定的参数创建多层感知机分类器对象
        mlp = MLPClassifier(random_state=0, max_iter=max_iter, **param)

        # 由于某些参数组合会导致模型不收敛，因此这些情况会被忽略
        with warnings.catch_warnings():
            # 忽略 sklearn 中的 ConvergenceWarning 警告
            warnings.filterwarnings(
                "ignore", category=ConvergenceWarning, module="sklearn"
            )
            # 使用训练数据 X 和标签 y 训练多层感知机模型
            mlp.fit(X, y)

        # 将训练好的模型加入到 mlps 列表中
        mlps.append(mlp)
        # 输出训练集的得分
        print("Training set score: %f" % mlp.score(X, y))
        # 输出训练集的损失值
        print("Training set loss: %f" % mlp.loss_)

    # 遍历已训练好的多层感知机模型，为每个模型绘制损失曲线
    for mlp, label, args in zip(mlps, labels, plot_args):
        # 在图表上绘制模型的损失曲线，使用对应的标签和绘图参数
        ax.plot(mlp.loss_curve_, label=label, **args)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# 创建一个包含 2x2 子图的图表，并设置图表大小为 15x10

# 加载或生成一些示例数据集
iris = datasets.load_iris()
X_digits, y_digits = datasets.load_digits(return_X_y=True)
data_sets = [
    (iris.data, iris.target),  # 将鸢尾花数据集添加到数据集列表中
    (X_digits, y_digits),  # 将手写数字数据集添加到数据集列表中
    datasets.make_circles(noise=0.2, factor=0.5, random_state=1),  # 生成圆形数据集并添加到数据集列表中
    datasets.make_moons(noise=0.3, random_state=0),  # 生成月牙形数据集并添加到数据集列表中
]

# 遍历子图和数据集，对每个子图调用 plot_on_dataset 函数绘制数据
for ax, data, name in zip(
    axes.ravel(), data_sets, ["iris", "digits", "circles", "moons"]
):
    plot_on_dataset(*data, ax=ax, name=name)

# 在图表中添加图例，使用 ax.get_lines() 获取所有线条对象，labels 是之前定义的标签列表，设置列数为 3，位置为上中
fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
# 显示图表
plt.show()
```