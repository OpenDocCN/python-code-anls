# `D:\src\scipysrc\scikit-learn\examples\manifold\plot_lle_digits.py`

```
"""
=============================================================================
Manifold learning on handwritten digits: Locally Linear Embedding, Isomap...
=============================================================================

We illustrate various embedding techniques on the digits dataset.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


# %%
# Load digits dataset
# -------------------
# We will load the digits dataset and only use six first of the ten available classes.
from sklearn.datasets import load_digits

# 加载手写数字数据集，并只使用前六个类别
digits = load_digits(n_class=6)
X, y = digits.data, digits.target
n_samples, n_features = X.shape
n_neighbors = 30

# %%
# We can plot the first hundred digits from this data set.
import matplotlib.pyplot as plt

# 创建包含100个图像的子图
fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(6, 6))
for idx, ax in enumerate(axs.ravel()):
    # 展示每个图像，并使用二进制灰度图像显示
    ax.imshow(X[idx].reshape((8, 8)), cmap=plt.cm.binary)
    ax.axis("off")
_ = fig.suptitle("A selection from the 64-dimensional digits dataset", fontsize=16)

# %%
# Helper function to plot embedding
# ---------------------------------
# Below, we will use different techniques to embed the digits dataset. We will plot
# the projection of the original data onto each embedding. It will allow us to
# check whether or digits are grouped together in the embedding space, or
# scattered across it.
import numpy as np
from matplotlib import offsetbox

from sklearn.preprocessing import MinMaxScaler

# 定义绘制嵌入图的辅助函数
def plot_embedding(X, title):
    _, ax = plt.subplots()
    # 对数据进行归一化处理
    X = MinMaxScaler().fit_transform(X)

    for digit in digits.target_names:
        # 根据目标值不同，使用不同的标记和颜色绘制散点图
        ax.scatter(
            *X[y == digit].T,
            marker=f"${digit}$",
            s=60,
            color=plt.cm.Dark2(digit),
            alpha=0.425,
            zorder=2,
        )
    shown_images = np.array([[1.0, 1.0]])  # just something big
    for i in range(X.shape[0]):
        # 在嵌入空间中绘制每个数字的图像
        # 显示数字的组的注释框
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # 不显示距离太近的点
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
        )
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)

    ax.set_title(title)
    ax.axis("off")


# %%
# Embedding techniques comparison
# -------------------------------
#
# Below, we compare different techniques. However, there are a couple of things
# to note:
#
# * the :class:`~sklearn.ensemble.RandomTreesEmbedding` is not
#   technically a manifold embedding method, as it learn a high-dimensional
#   representation on which we apply a dimensionality reduction method.
#   However, it is often useful to cast a dataset into a representation in
#   which the classes are linearly-separable.
# %%
# 所有感兴趣的方法已经声明完成，我们可以运行并执行数据的投影。
# 我们将存储投影后的数据以及执行每个投影所需的计算时间。
from time import time

# 预先声明空字典用于存储各个方法的投影结果和计时信息
projections, timing = {}, {}

# 遍历嵌入方法字典中的每个项目
for name, transformer in embeddings.items():
    # 如果方法名以"Linear Discriminant Analysis"开头，则对数据进行复制并使其可逆
    if name.startswith("Linear Discriminant Analysis"):
        data = X.copy()
        data.flat[:: X.shape[1] + 1] += 0.01  # 使 X 可逆
    else:
        data = X

    # 打印当前正在计算的方法名称
    print(f"Computing {name}...")
    # 记录开始时间
    start_time = time()
    # 使用指定的数据集(data)和标签(y)，对转换器(transformer)进行拟合和转换，将结果存储在 projections 字典中的 name 键下
    projections[name] = transformer.fit_transform(data, y)
    # 计算从开始时间(start_time)到当前时间的经过时间，并将结果存储在 timing 字典中的 name 键下
    timing[name] = time() - start_time
# %%
# 最终，我们可以绘制每种方法得到的投影结果。

# 遍历存储在 `timing` 字典中的每个方法名称
for name in timing:
    # 构建每幅图的标题，包括方法名称和所用时间（格式化为三位小数的秒数）
    title = f"{name} (time {timing[name]:.3f}s)"
    # 调用 plot_embedding 函数绘制投影结果，传入投影数据和标题
    plot_embedding(projections[name], title)

# 显示所有绘制的图形
plt.show()
```