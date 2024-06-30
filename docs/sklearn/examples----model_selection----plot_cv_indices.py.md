# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_cv_indices.py`

```
# 导入所需的库和模块
import matplotlib.pyplot as plt  # 导入 matplotlib 绘图库
import numpy as np  # 导入 numpy 数值计算库
from matplotlib.patches import Patch  # 导入用于绘图的 Patch 类

from sklearn.model_selection import (  # 导入用于模型选择的各种交叉验证类
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)

rng = np.random.RandomState(1338)  # 设定随机种子，确保结果可重复

cmap_data = plt.cm.Paired  # 数据颜色映射
cmap_cv = plt.cm.coolwarm  # 交叉验证结果颜色映射
n_splits = 4  # 分割数据的次数

# %%
# 可视化我们的数据
# ------------------
#
# 首先，我们需要了解我们数据的结构。数据集包含100个随机生成的输入数据点，
# 3个不均匀分布的类别，以及10个“组”，这些组在数据点上均匀分布。
#
# 正如我们将看到的，一些交叉验证对象对标记数据执行特定操作，而其他对象对分组数据执行不同操作，
# 还有一些对象不使用这些信息。
#
# 首先，我们将可视化我们的数据。

# 生成类别和组数据
n_points = 100
X = rng.randn(100, 10)  # 生成100x10的随机数据矩阵 X

percentiles_classes = [0.1, 0.3, 0.6]
y = np.hstack([[ii] * int(100 * perc) for ii, perc in enumerate(percentiles_classes)])  # 生成标签数据 y

# 生成不均匀的组
group_prior = rng.dirichlet([2] * 10)  # 根据狄利克雷分布生成组先验权重
groups = np.repeat(np.arange(10), rng.multinomial(100, group_prior))  # 根据先验权重生成组数据

def visualize_groups(classes, groups, name):
    # 可视化数据集中的组
    fig, ax = plt.subplots()  # 创建图形和子图对象
    ax.scatter(
        range(len(groups)),
        [0.5] * len(groups),
        c=groups,
        marker="_",
        lw=50,
        cmap=cmap_data,
    )  # 绘制组数据的散点图
    ax.scatter(
        range(len(groups)),
        [3.5] * len(groups),
        c=classes,
        marker="_",
        lw=50,
        cmap=cmap_data,
    )  # 绘制类别数据的散点图
    ax.set(
        ylim=[-1, 5],
        yticks=[0.5, 3.5],
        yticklabels=["数据组", "数据类别"],
        xlabel="样本索引",
    )  # 设置图形的坐标轴标签和范围

visualize_groups(y, groups, "no groups")

# %%
# 定义一个函数来可视化交叉验证的行为
# --------------------------------------------------------
#
# 我们将定义一个函数，用于可视化每个交叉验证对象的行为。我们将对数据进行4次拆分。
# 在每次拆分中，我们将可视化用于训练集（蓝色）和测试集（红色）的索引。

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """创建一个用于展示交叉验证对象索引的示例图。"""
    use_groups = "Group" in type(cv).__name__  # 检查交叉验证对象是否使用了分组信息
    groups = group if use_groups else None  # 如果使用分组信息，则传入组数据，否则传入 None
    # 为每个交叉验证拆分生成训练/测试集的可视化
    # 使用交叉验证对象 cv 对数据集 X 和标签 y 进行划分，同时考虑分组信息 groups
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups)):
        # 将训练集和测试集的索引标记为NaN
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1  # 将测试集的索引标记为1
        indices[tr] = 0  # 将训练集的索引标记为0

        # 可视化结果
        ax.scatter(
            range(len(indices)),  # x轴使用样本索引
            [ii + 0.5] * len(indices),  # y轴使用当前交叉验证迭代次数加上偏移量
            c=indices,  # 根据索引值确定颜色
            marker="_",  # 使用下划线作为标记点
            lw=lw,  # 设置标记线宽度
            cmap=cmap_cv,  # 指定使用的颜色映射
            vmin=-0.2,  # 最小值映射
            vmax=1.2,  # 最大值映射
        )

    # 在图中绘制数据的类别和分组信息
    ax.scatter(
        range(len(X)),  # x轴使用样本索引
        [ii + 1.5] * len(X),  # y轴位置略高于测试集数据的位置
        c=y,  # 根据数据的类别确定颜色
        marker="_",  # 使用下划线作为标记点
        lw=lw,  # 设置标记线宽度
        cmap=cmap_data,  # 指定使用的颜色映射
    )

    ax.scatter(
        range(len(X)),  # x轴使用样本索引
        [ii + 2.5] * len(X),  # y轴位置略高于分组信息数据的位置
        c=group,  # 根据分组信息确定颜色
        marker="_",  # 使用下划线作为标记点
        lw=lw,  # 设置标记线宽度
        cmap=cmap_data,  # 指定使用的颜色映射
    )

    # 设置图表格式
    yticklabels = list(range(n_splits)) + ["class", "group"]  # y轴刻度标签包括交叉验证迭代次数和类别、分组信息
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,  # 设置y轴刻度位置
        yticklabels=yticklabels,  # 设置y轴刻度标签
        xlabel="Sample index",  # 设置x轴标签
        ylabel="CV iteration",  # 设置y轴标签
        ylim=[n_splits + 2.2, -0.2],  # 设置y轴范围
        xlim=[0, 100],  # 设置x轴范围
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)  # 设置图表标题为交叉验证对象的名称
    return ax  # 返回绘制完成的图表对象
# %%
# 使用 :class:`~sklearn.model_selection.KFold` 交叉验证对象来展示其效果
# 创建一个包含图和轴的 subplot
fig, ax = plt.subplots()
# 创建一个 KFold 交叉验证对象 cv
cv = KFold(n_splits)
# 绘制交叉验证的指示图，显示数据集 X, y 和可能的组 groups
plot_cv_indices(cv, X, y, groups, ax, n_splits)

# %%
# 可以看到，默认情况下，KFold 交叉验证迭代器不考虑数据点类别或组信息。
# 我们可以通过以下方式进行修改：
#
# - 使用 ``StratifiedKFold`` 以保留每个类别的样本百分比。
# - 使用 ``GroupKFold`` 确保同一组不会出现在两个不同的折叠中。
# - 使用 ``StratifiedGroupKFold`` 在保留 ``GroupKFold`` 约束的同时尝试返回分层折叠。
cvs = [StratifiedKFold, GroupKFold, StratifiedGroupKFold]

for cv in cvs:
    # 创建一个新的图和轴，设置图的大小为 (6, 3)
    fig, ax = plt.subplots(figsize=(6, 3))
    # 绘制当前交叉验证对象 cv 的指示图
    plot_cv_indices(cv(n_splits), X, y, groups, ax, n_splits)
    # 添加图例，表示测试集和训练集
    ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Testing set", "Training set"],
        loc=(1.02, 0.8),
    )
    # 调整布局使图例适合
    plt.tight_layout()
    # 调整子图，使右边距为 0.7
    fig.subplots_adjust(right=0.7)

# %%
# 接下来，我们将为多个交叉验证迭代器可视化这种行为。
#
# 可视化多个 CV 对象的交叉验证指示图
# ------------------------------------------------------
#
# 让我们通过循环遍历几个常见的 scikit-learn 交叉验证对象，
# 可视化每个对象的行为。
#
# 注意某些对象使用组/类别信息，而其他对象则不使用。

cvs = [
    KFold,
    GroupKFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedGroupKFold,
    GroupShuffleSplit,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
]

for cv in cvs:
    # 创建当前 cv 对象的实例 this_cv，并设置折叠数为 n_splits
    this_cv = cv(n_splits=n_splits)
    # 创建一个新的图和轴，设置图的大小为 (6, 3)
    fig, ax = plt.subplots(figsize=(6, 3))
    # 绘制当前交叉验证对象的指示图
    plot_cv_indices(this_cv, X, y, groups, ax, n_splits)

    # 添加图例，表示测试集和训练集
    ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Testing set", "Training set"],
        loc=(1.02, 0.8),
    )
    # 调整布局使图例适合
    plt.tight_layout()
    # 调整子图，使右边距为 0.7
    fig.subplots_adjust(right=0.7)

plt.show()
```