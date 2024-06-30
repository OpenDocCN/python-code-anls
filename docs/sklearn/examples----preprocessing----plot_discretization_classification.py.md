# `D:\src\scipysrc\scikit-learn\examples\preprocessing\plot_discretization_classification.py`

```
"""
======================
Feature discretization
======================

A demonstration of feature discretization on synthetic classification datasets.
Feature discretization decomposes each feature into a set of bins, here equally
distributed in width. The discrete values are then one-hot encoded, and given
to a linear classifier. This preprocessing enables a non-linear behavior even
though the classifier is linear.

On this example, the first two rows represent linearly non-separable datasets
(moons and concentric circles) while the third is approximately linearly
separable. On the two linearly non-separable datasets, feature discretization
largely increases the performance of linear classifiers. On the linearly
separable dataset, feature discretization decreases the performance of linear
classifiers. Two non-linear classifiers are also shown for comparison.

This example should be taken with a grain of salt, as the intuition conveyed
does not necessarily carry over to real datasets. Particularly in
high-dimensional spaces, data can more easily be separated linearly. Moreover,
using feature discretization and one-hot encoding increases the number of
features, which easily lead to overfitting when the number of samples is small.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""

# Code source: Tom Dupré la Tour
# Adapted from plot_classifier_comparison by Gaël Varoquaux and Andreas Müller
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 NumPy 数学计算库，并使用别名 np
from matplotlib.colors import ListedColormap  # 导入 ListedColormap 类，用于绘图中颜色映射

from sklearn.datasets import make_circles, make_classification, make_moons  # 导入生成数据集的函数
from sklearn.ensemble import GradientBoostingClassifier  # 导入梯度提升分类器
from sklearn.exceptions import ConvergenceWarning  # 导入收敛警告异常类
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归分类器
from sklearn.model_selection import GridSearchCV, train_test_split  # 导入网格搜索交叉验证和数据集划分函数
from sklearn.pipeline import make_pipeline  # 导入构建管道的函数
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler  # 导入特征离散化和标准化函数
from sklearn.svm import SVC, LinearSVC  # 导入支持向量分类器和线性支持向量分类器
from sklearn.utils._testing import ignore_warnings  # 导入用于忽略警告的函数

h = 0.02  # 在网格中的步长大小


def get_name(estimator):
    """
    返回估计器的类名或管道中估计器的名字

    Args:
    estimator: sklearn 估计器对象

    Returns:
    估计器的类名或管道中估计器的名字
    """
    name = estimator.__class__.__name__
    if name == "Pipeline":
        name = [get_name(est[1]) for est in estimator.steps]
        name = " + ".join(name)
    return name


# list of (estimator, param_grid), where param_grid is used in GridSearchCV
# The parameter spaces in this example are limited to a narrow band to reduce
# its runtime. In a real use case, a broader search space for the algorithms
# should be used.
classifiers = [
    (
        make_pipeline(StandardScaler(), LogisticRegression(random_state=0)),
        {"logisticregression__C": np.logspace(-1, 1, 3)},
    ),
    (
        make_pipeline(StandardScaler(), LinearSVC(random_state=0)),
        {"linearsvc__C": np.logspace(-1, 1, 3)},
    ),
    # 创建多个机器学习模型及其参数组合，用于后续的网格搜索和交叉验证
    
    (
        # 创建一个包含标准化、KBins离散化和逻辑回归的机器学习管道
        make_pipeline(
            StandardScaler(); 
            KBinsDiscretizer(encode="onehot", random_state=0); 
            LogisticRegression(random_state=0),
        ),
        # 定义适用于该管道的参数字典，包括KBins离散器的箱数和逻辑回归的正则化参数C
        {
            "kbinsdiscretizer__n_bins": np.arange(5, 8), 
            "logisticregression__C": np.logspace(-1, 1, 3),
        },
    ),
    
    (
        # 创建一个包含标准化、KBins离散化和线性支持向量机的机器学习管道
        make_pipeline(
            StandardScaler(); 
            KBinsDiscretizer(encode="onehot", random_state=0); 
            LinearSVC(random_state=0),
        ),
        # 定义适用于该管道的参数字典，包括KBins离散器的箱数和线性SVC的正则化参数C
        {
            "kbinsdiscretizer__n_bins": np.arange(5, 8), 
            "linearsvc__C": np.logspace(-1, 1, 3),
        },
    ),
    
    (
        # 创建一个包含标准化和梯度提升分类器的机器学习管道
        make_pipeline(
            StandardScaler(); 
            GradientBoostingClassifier(n_estimators=5, random_state=0),
        ),
        # 定义适用于该管道的参数字典，包括梯度提升分类器的学习率参数
        {
            "gradientboostingclassifier__learning_rate": np.logspace(-2, 0, 5),
        },
    ),
    
    (
        # 创建一个包含标准化和支持向量机的机器学习管道
        make_pipeline(
            StandardScaler(); 
            SVC(random_state=0),
        ),
        # 定义适用于该管道的参数字典，包括支持向量机的正则化参数C
        {
            "svc__C": np.logspace(-1, 1, 3),
        },
    ),
# 定义一个列表 names，包含了经过处理的分类器名称，去掉了前缀 "StandardScaler + "
names = [get_name(e).replace("StandardScaler + ", "") for e, _ in classifiers]

# 设定数据集的样本数目
n_samples = 100
# 创建三个数据集并存储在 datasets 列表中
datasets = [
    make_moons(n_samples=n_samples, noise=0.2, random_state=0),  # 生成月亮形状的数据集
    make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1),  # 生成环形数据集
    make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=2,
        n_clusters_per_class=1,
    ),  # 生成分类数据集
]

# 创建一个图表对象，包含多行和 (len(classifiers) + 1) 列，设置图表尺寸为 (21, 9)
fig, axes = plt.subplots(
    nrows=len(datasets), ncols=len(classifiers) + 1, figsize=(21, 9)
)

# 设定两种颜色映射
cm_piyg = plt.cm.PiYG
cm_bright = ListedColormap(["#b30065", "#178000"])

# 遍历每个数据集
for ds_cnt, (X, y) in enumerate(datasets):
    print(f"\ndataset {ds_cnt}\n---------")

    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # 设定背景颜色网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 绘制数据集
    ax = axes[ds_cnt, 0]
    if ds_cnt == 0:
        ax.set_title("Input data")
    # 绘制训练点
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # 绘制测试点
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    # 遍历每个分类器
    # 对于每个分类器和其参数网格，使用 enumerate 函数获取索引和元组(name, (estimator, param_grid))
    for est_idx, (name, (estimator, param_grid)) in enumerate(zip(names, classifiers)):
        # 在子图中选择第 ds_cnt 行，第 est_idx + 1 列的轴
        ax = axes[ds_cnt, est_idx + 1]

        # 使用 GridSearchCV 对象 clf，根据给定的分类器和参数网格进行网格搜索交叉验证
        clf = GridSearchCV(estimator=estimator, param_grid=param_grid)
        # 忽略收敛警告进行拟合
        with ignore_warnings(category=ConvergenceWarning):
            clf.fit(X_train, y_train)
        # 计算分类器在测试集上的准确率得分
        score = clf.score(X_test, y_test)
        # 打印分类器名称及其得分
        print(f"{name}: {score:.2f}")

        # 绘制决策边界。为此，将为网格中的每个点分配一种颜色，范围为 [x_min, x_max]*[y_min, y_max]
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
        else:
            Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]

        # 将结果绘制成颜色填充图
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm_piyg, alpha=0.8)

        # 绘制训练数据点
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # 绘制测试数据点
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )
        # 设置坐标轴的范围
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        # 隐藏坐标轴刻度
        ax.set_xticks(())
        ax.set_yticks(())

        # 如果是第一个数据集，则设置子图标题，替换 " + " 为换行符
        if ds_cnt == 0:
            ax.set_title(name.replace(" + ", "\n"))
        # 在子图中添加文本，显示分类器得分（去除左侧多余的零），放置在右下角
        ax.text(
            0.95,
            0.06,
            (f"{score:.2f}").lstrip("0"),
            size=15,
            bbox=dict(boxstyle="round", alpha=0.8, facecolor="white"),
            transform=ax.transAxes,
            horizontalalignment="right",
        )
# 调整图像布局，使其更紧凑
plt.tight_layout()

# 在图像上方添加总标题
plt.subplots_adjust(top=0.90)

# 定义三个总标题，分别对应不同的图像子区域
suptitles = [
    "Linear classifiers",
    "Feature discretization and linear classifiers",
    "Non-linear classifiers",
]

# 遍历三个总标题，分别添加到对应的子图上
for i, suptitle in zip([1, 3, 5], suptitles):
    # 获取指定位置的子图对象
    ax = axes[0, i]
    
    # 在子图指定位置添加文本作为总标题
    ax.text(
        1.05,  # x 坐标位置，稍微超出子图右侧
        1.25,  # y 坐标位置，略高于子图上方
        suptitle,  # 添加的文本内容，即总标题
        transform=ax.transAxes,  # 坐标变换方式，使得坐标是相对于子图的
        horizontalalignment="center",  # 水平对齐方式为居中
        size="x-large",  # 文本大小设为较大
    )

# 显示图形
plt.show()
```