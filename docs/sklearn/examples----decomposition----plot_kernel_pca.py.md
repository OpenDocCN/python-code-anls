# `D:\src\scipysrc\scikit-learn\examples\decomposition\plot_kernel_pca.py`

```
"""
==========
Kernel PCA
==========

This example shows the difference between the Principal Components Analysis
(:class:`~sklearn.decomposition.PCA`) and its kernelized version
(:class:`~sklearn.decomposition.KernelPCA`).

On the one hand, we show that :class:`~sklearn.decomposition.KernelPCA` is able
to find a projection of the data which linearly separates them while it is not the case
with :class:`~sklearn.decomposition.PCA`.

Finally, we show that inverting this projection is an approximation with
:class:`~sklearn.decomposition.KernelPCA`, while it is exact with
:class:`~sklearn.decomposition.PCA`.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Projecting data: `PCA` vs. `KernelPCA`
# --------------------------------------
#
# In this section, we show the advantages of using a kernel when
# projecting data using a Principal Component Analysis (PCA). We create a
# dataset made of two nested circles.
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# Generate a dataset of nested circles with noise
X, y = make_circles(n_samples=1_000, factor=0.3, noise=0.05, random_state=0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# %%
# Let's have a quick first look at the generated dataset.
import matplotlib.pyplot as plt

# Create subplots for training and testing data visualization
_, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))

# Plot training data
train_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
train_ax.set_ylabel("Feature #1")
train_ax.set_xlabel("Feature #0")
train_ax.set_title("Training data")

# Plot testing data
test_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
test_ax.set_xlabel("Feature #0")
_ = test_ax.set_title("Testing data")

# %%
# The samples from each class cannot be linearly separated: there is no
# straight line that can split the samples of the inner set from the outer
# set.
#
# Now, we will use PCA with and without a kernel to see what is the effect of
# using such a kernel. The kernel used here is a radial basis function (RBF)
# kernel.
from sklearn.decomposition import PCA, KernelPCA

# Initialize PCA with 2 components
pca = PCA(n_components=2)

# Initialize KernelPCA with RBF kernel
kernel_pca = KernelPCA(
    n_components=None, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1
)

# Project the testing data using PCA and KernelPCA
X_test_pca = pca.fit(X_train).transform(X_test)
X_test_kernel_pca = kernel_pca.fit(X_train).transform(X_test)

# %%
fig, (orig_data_ax, pca_proj_ax, kernel_pca_proj_ax) = plt.subplots(
    ncols=3, figsize=(14, 4)
)

# Plot original testing data
orig_data_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
orig_data_ax.set_ylabel("Feature #1")
orig_data_ax.set_xlabel("Feature #0")
orig_data_ax.set_title("Testing data")

# Plot PCA projection of testing data
pca_proj_ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test)
pca_proj_ax.set_ylabel("Principal component #1")
pca_proj_ax.set_xlabel("Principal component #0")
pca_proj_ax.set_title("Projection of testing data\n using PCA")

# Plot KernelPCA projection of testing data
kernel_pca_proj_ax.scatter(X_test_kernel_pca[:, 0], X_test_kernel_pca[:, 1], c=y_test)
kernel_pca_proj_ax.set_ylabel("Principal component #1")
# 创建新的 x 轴标签为 "Principal component #0"
kernel_pca_proj_ax.set_xlabel("Principal component #0")

# 设置图表标题为 "Projection of testing data\n using KernelPCA"
_ = kernel_pca_proj_ax.set_title("Projection of testing data\n using KernelPCA")

# %%
# 我们回顾一下PCA将数据线性变换。直觉上，这意味着坐标系将被居中，每个主成分相对于其方差重新缩放，并最终被旋转。
# 从这个变换中得到的数据是各向同性的，现在可以投影到其*主成分*上。
#
# 因此，观察使用PCA进行的投影（即中间的图），我们可以看到在缩放方面没有变化；实际上，数据是两个以零为中心的同心圆，
# 原始数据已经是各向同性的。然而，我们可以看到数据已经被旋转。总之，我们看到这样的投影不能帮助我们定义一个线性分类器来区分两类样本。
#
# 使用核函数可以进行非线性投影。在这里，通过使用RBF核函数，我们期望投影将展开数据集，同时大致保持原始空间中彼此接近的数据点对的相对距离。
#
# 我们在右侧的图中观察到这种行为：同一类别的样本彼此更接近，而来自相对类别的样本则更分散。现在，我们可以使用线性分类器来分离这两类样本。
#
# 投影回原始特征空间
# ------------------------------------------
#
# 在使用:class:`~sklearn.decomposition.KernelPCA`时要记住的一个特点与重构有关（即在原始特征空间中的反投影）。使用
# :class:`~sklearn.decomposition.PCA`时，如果`n_components`与原始特征的数量相同，则重构将是精确的。这在本例中是成立的。
#
# 我们可以调查一下使用:class:`~sklearn.decomposition.KernelPCA`反投影时是否得到原始数据集。
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstructed_kernel_pca = kernel_pca.inverse_transform(kernel_pca.transform(X_test))

# %%
# 创建一个包含三个子图的图表，宽度共享，高度共享，尺寸为13x4
fig, (orig_data_ax, pca_back_proj_ax, kernel_pca_back_proj_ax) = plt.subplots(
    ncols=3, sharex=True, sharey=True, figsize=(13, 4)
)

# 在原始数据的轴上绘制散点图，以 Feature #0 为 x 轴，Feature #1 为 y 轴，颜色由 y_test 决定
orig_data_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
orig_data_ax.set_ylabel("Feature #1")
orig_data_ax.set_xlabel("Feature #0")
orig_data_ax.set_title("Original test data")

# 在PCA反投影的轴上绘制散点图，以 Feature #0 为 x 轴，Feature #1 为 y 轴，颜色由 y_test 决定
pca_back_proj_ax.scatter(X_reconstructed_pca[:, 0], X_reconstructed_pca[:, 1], c=y_test)
pca_back_proj_ax.set_xlabel("Feature #0")
pca_back_proj_ax.set_title("Reconstruction via PCA")

# 在KernelPCA反投影的轴上绘制散点图，以 Feature #0 为 x 轴，Feature #1 为 y 轴，颜色由 y_test 决定
kernel_pca_back_proj_ax.scatter(
    X_reconstructed_kernel_pca[:, 0], X_reconstructed_kernel_pca[:, 1], c=y_test
)
kernel_pca_back_proj_ax.set_xlabel("Feature #0")
_ = kernel_pca_back_proj_ax.set_title("Reconstruction via KernelPCA")

# %%
# 尽管我们看到了一个完美的重构
# :class:`~sklearn.decomposition.PCA` we observe a different result for
# :class:`~sklearn.decomposition.KernelPCA`.
#
# Indeed, :meth:`~sklearn.decomposition.KernelPCA.inverse_transform` cannot
# rely on an analytical back-projection and thus an exact reconstruction.
# Instead, a :class:`~sklearn.kernel_ridge.KernelRidge` is internally trained
# to learn a mapping from the kernalized PCA basis to the original feature
# space. This method therefore comes with an approximation introducing small
# differences when back projecting in the original feature space.
#
# To improve the reconstruction using
# :meth:`~sklearn.decomposition.KernelPCA.inverse_transform`, one can tune
# `alpha` in :class:`~sklearn.decomposition.KernelPCA`, the regularization term
# which controls the reliance on the training data during the training of
# the mapping.
```