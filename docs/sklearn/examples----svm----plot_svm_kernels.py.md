# `D:\src\scipysrc\scikit-learn\examples\svm\plot_svm_kernels.py`

```
# 创建一个包含 16 个样本和两个类别的二维分类数据集
# 数据集 X 是一个包含所有样本特征的 NumPy 数组
import matplotlib.pyplot as plt
import numpy as np

X = np.array(
    # 定义一个包含多个二元列表的列表，每个二元列表包含两个浮点数
    [
        [0.4, -0.7],     # 第一个二元列表，包含两个浮点数 0.4 和 -0.7
        [-1.5, -1.0],    # 第二个二元列表，包含两个浮点数 -1.5 和 -1.0
        [-1.4, -0.9],    # 第三个二元列表，包含两个浮点数 -1.4 和 -0.9
        [-1.3, -1.2],    # 第四个二元列表，包含两个浮点数 -1.3 和 -1.2
        [-1.1, -0.2],    # 第五个二元列表，包含两个浮点数 -1.1 和 -0.2
        [-1.2, -0.4],    # 第六个二元列表，包含两个浮点数 -1.2 和 -0.4
        [-0.5, 1.2],     # 第七个二元列表，包含两个浮点数 -0.5 和 1.2
        [-1.5, 2.1],     # 第八个二元列表，包含两个浮点数 -1.5 和 2.1
        [1.0, 1.0],      # 第九个二元列表，包含两个浮点数 1.0 和 1.0
        [1.3, 0.8],      # 第十个二元列表，包含两个浮点数 1.3 和 0.8
        [1.2, 0.5],      # 第十一个二元列表，包含两个浮点数 1.2 和 0.5
        [0.2, -2.0],     # 第十二个二元列表，包含两个浮点数 0.2 和 -2.0
        [0.5, -2.4],     # 第十三个二元列表，包含两个浮点数 0.5 和 -2.4
        [0.2, -2.3],     # 第十四个二元列表，包含两个浮点数 0.2 和 -2.3
        [0.0, -2.7],     # 第十五个二元列表，包含两个浮点数 0.0 和 -2.7
        [1.3, 2.1],      # 第十六个二元列表，包含两个浮点数 1.3 和 2.1
    ]
# Import necessary libraries and modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay

# Sample data: X contains 2D points, y contains corresponding class labels
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1], [-1, 1], [0, -1], [-2, 2], [1, -2], [2, 2], [-1, -2], [0, 1], [-2, -1], [1, 2], [2, -2], [-1, 2], [0, -2]])

# Class labels for the samples
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

# Plotting settings
# Create a figure and axis with a specific size
fig, ax = plt.subplots(figsize=(4, 3))

# Define the limits for the x and y axes of the plot
x_min, x_max, y_min, y_max = -3, 3, -3, 3
ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

# Plot samples using a scatter plot, coloring points based on class labels
scatter = ax.scatter(X[:, 0], X[:, 1], s=150, c=y, label=y, edgecolors="k")

# Add a legend to the plot using elements from the scatter plot
ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")

# Set the title of the plot
ax.set_title("Samples in two-dimensional feature space")

# Display the plot
_ = plt.show()

# %%
# We can see that the samples are not clearly separable by a straight line.
#
# Training SVC model and plotting decision boundaries
# ---------------------------------------------------
# Define a function that fits an SVC classifier with a specified kernel,
# then plots the decision boundaries learned by the model using
# DecisionBoundaryDisplay.
#
# Notice that for simplicity, the C parameter is set to its default (C=1),
# and gamma is set to gamma=2 across all kernels (though ignored for linear).
# In real tasks, parameter tuning using techniques like GridSearchCV is
# recommended for better performance.
#
# Setting response_method="predict" in DecisionBoundaryDisplay colors areas
# based on predicted class. Using response_method="decision_function" allows
# plotting decision boundary and margins. Support vectors used during training
# are identified by the support_vectors_ attribute of the trained SVCs.
def plot_training_data_with_decision_boundary(
    kernel, ax=None, long_title=True, support_vectors=True
):
    # Train an SVC classifier with specified kernel and gamma
    clf = svm.SVC(kernel=kernel, gamma=2).fit(X, y)

    # Settings for plotting
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    x_min, x_max, y_min, y_max = -3, 3, -3, 3
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundaries and margins using DecisionBoundaryDisplay
    common_params = {"estimator": clf, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )
    if support_vectors:
        # 如果存在支持向量，则在支持向量周围画出大圆圈
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,  # 设置散点大小为150
            facecolors="none",  # 散点内部不填充颜色
            edgecolors="k",  # 散点边缘颜色为黑色
        )

    # 根据分类标签y的不同取值，用不同颜色绘制样本点，并添加图例
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")  # 添加图例，显示不同类别的标签

    if long_title:
        ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")  # 设置标题，显示SVC中使用的核函数类型
    else:
        ax.set_title(kernel)  # 设置简略标题，显示核函数类型

    if ax is None:
        plt.show()  # 如果ax为空，则显示绘图
# %%
# Linear kernel
# *************
# Linear kernel is the dot product of the input samples:
#
# .. math:: K(\mathbf{x}_1, \mathbf{x}_2) = \mathbf{x}_1^\top \mathbf{x}_2
#
# It is then applied to any combination of two data points (samples) in the
# dataset. The dot product of the two points determines the
# :func:`~sklearn.metrics.pairwise.cosine_similarity` between both points. The
# higher the value, the more similar the points are.
plot_training_data_with_decision_boundary("linear")

# %%
# Training a :class:`~sklearn.svm.SVC` on a linear kernel results in an
# untransformed feature space, where the hyperplane and the margins are
# straight lines. Due to the lack of expressivity of the linear kernel, the
# trained classes do not perfectly capture the training data.
#
# Polynomial kernel
# *****************
# The polynomial kernel changes the notion of similarity. The kernel function
# is defined as:
#
# .. math::
#   K(\mathbf{x}_1, \mathbf{x}_2) = (\gamma \cdot \
#       \mathbf{x}_1^\top\mathbf{x}_2 + r)^d
#
# where :math:`{d}` is the degree (`degree`) of the polynomial, :math:`{\gamma}`
# (`gamma`) controls the influence of each individual training sample on the
# decision boundary and :math:`{r}` is the bias term (`coef0`) that shifts the
# data up or down. Here, we use the default value for the degree of the
# polynomial in the kernel function (`degree=3`). When `coef0=0` (the default),
# the data is only transformed, but no additional dimension is added. Using a
# polynomial kernel is equivalent to creating
# :class:`~sklearn.preprocessing.PolynomialFeatures` and then fitting a
# :class:`~sklearn.svm.SVC` with a linear kernel on the transformed data,
# although this alternative approach would be computationally expensive for most
# datasets.
plot_training_data_with_decision_boundary("poly")

# %%
# The polynomial kernel with `gamma=2`` adapts well to the training data,
# causing the margins on both sides of the hyperplane to bend accordingly.
#
# RBF kernel
# **********
# The radial basis function (RBF) kernel, also known as the Gaussian kernel, is
# the default kernel for Support Vector Machines in scikit-learn. It measures
# similarity between two data points in infinite dimensions and then approaches
# classification by majority vote. The kernel function is defined as:
#
# .. math::
#   K(\mathbf{x}_1, \mathbf{x}_2) = \exp\left(-\gamma \cdot
#       {\|\mathbf{x}_1 - \mathbf{x}_2\|^2}\right)
#
# where :math:`{\gamma}` (`gamma`) controls the influence of each individual
# training sample on the decision boundary.
#
# The larger the euclidean distance between two points
# :math:`\|\mathbf{x}_1 - \mathbf{x}_2\|^2`
# the closer the kernel function is to zero. This means that two points far away
# are more likely to be dissimilar.
plot_training_data_with_decision_boundary("rbf")

# %%
# In the plot we can see how the decision boundaries tend to contract around
# data points that are close to each other.
#
# Sigmoid kernel
# %%
# 根据指定的核函数类型绘制训练数据和决策边界
plot_training_data_with_decision_boundary("sigmoid")

# %%
# 使用sigmoid核函数得到的决策边界呈现出曲线和不规则的特征。决策边界尝试通过拟合一个sigmoid形状的曲线来分隔类别，
# 这导致了一个复杂的边界，可能不会很好地推广到未见过的数据。从这个例子可以看出，sigmoid核函数在处理呈sigmoid形状的数据时有特定的用途。
# 在本例中，通过仔细调整可能会找到更具泛化能力的决策边界。由于其特定性，sigmoid核函数在实践中比其他核函数使用更少。
#
# 结论
# ----
# 在本例中，我们可视化了使用提供的数据集训练的决策边界。这些图表作为直观演示，展示了不同核函数如何利用训练数据确定分类边界。
#
# 超平面和间隔虽然间接计算，但可以想象为转换后特征空间中的平面。然而，在图表中，它们相对于原始特征空间表示，导致了多项式、RBF和sigmoid核函数的曲线决策边界。
#
# 请注意，这些图表不评估各个核函数的准确性或质量。它们旨在提供对不同核函数如何利用训练数据的视觉理解。
#
# 对于全面的评估，建议使用诸如:class:`~sklearn.model_selection.GridSearchCV`等技术对:class:`~sklearn.svm.SVC`参数进行精细调整，以捕获数据内部结构。

# %%
# XOR数据集
# -----------
# XOR模式是一个经典的非线性可分数据集示例。在这里，我们演示了不同核函数在这种数据集上的工作效果。

# 生成一个网格来绘制决策边界
xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
# 设置随机种子
np.random.seed(0)
# 生成随机数据集
X = np.random.randn(300, 2)
# 生成标签，符合XOR逻辑
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# 创建一个包含4个子图的图表
_, ax = plt.subplots(2, 2, figsize=(8, 8))
# 参数设置
args = dict(long_title=False, support_vectors=False)
# 在第一个子图中绘制线性核函数的决策边界
plot_training_data_with_decision_boundary("linear", ax[0, 0], **args)
# 在第二个子图中绘制多项式核函数的决策边界
plot_training_data_with_decision_boundary("poly", ax[0, 1], **args)
# 使用"rbf"核绘制训练数据和决策边界到指定的子图(ax[1, 0])上，传入args参数
plot_training_data_with_decision_boundary("rbf", ax[1, 0], **args)
# 使用"sigmoid"核绘制训练数据和决策边界到指定的子图(ax[1, 1])上，传入args参数
plot_training_data_with_decision_boundary("sigmoid", ax[1, 1], **args)
# 展示绘制的所有图形
plt.show()

# %%
# 从上述图中可以看出，仅有"rbf"核能够为以上数据集找到合理的决策边界。
```