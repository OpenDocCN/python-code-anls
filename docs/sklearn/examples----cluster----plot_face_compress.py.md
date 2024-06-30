# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_face_compress.py`

```
"""
===========================
Vector Quantization Example
===========================

This example shows how one can use :class:`~sklearn.preprocessing.KBinsDiscretizer`
to perform vector quantization on a set of toy image, the raccoon face.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Original image
# --------------
#
# We start by loading the raccoon face image from SciPy. We will additionally check
# a couple of information regarding the image, such as the shape and data type used
# to store the image.
#
# Note that depending of the SciPy version, we have to adapt the import since the
# function returning the image is not located in the same module. Also, SciPy >= 1.10
# requires the package `pooch` to be installed.
try:  # Scipy >= 1.10
    from scipy.datasets import face
except ImportError:
    from scipy.misc import face

# Load the raccoon face image in grayscale
raccoon_face = face(gray=True)

# Print the dimensions of the image
print(f"The dimension of the image is {raccoon_face.shape}")
# Print the data type used to encode the image
print(f"The data used to encode the image is of type {raccoon_face.dtype}")
# Print the number of bytes taken by the image in RAM
print(f"The number of bytes taken in RAM is {raccoon_face.nbytes}")

# %%
# Thus the image is a 2D array of 768 pixels in height and 1024 pixels in width. Each
# value is a 8-bit unsigned integer, which means that the image is encoded using 8
# bits per pixel. The total memory usage of the image is 786 kilobytes (1 byte equals
# 8 bits).
#
# Using 8-bit unsigned integer means that the image is encoded using 256 different
# shades of gray, at most. We can check the distribution of these values.
import matplotlib.pyplot as plt

# Create subplots for rendering the image and plotting the histogram
fig, ax = plt.subplots(ncols=2, figsize=(12, 4))

# Show the raccoon face image in grayscale
ax[0].imshow(raccoon_face, cmap=plt.cm.gray)
ax[0].axis("off")
ax[0].set_title("Rendering of the image")

# Plot the histogram of pixel values
ax[1].hist(raccoon_face.ravel(), bins=256)
ax[1].set_xlabel("Pixel value")
ax[1].set_ylabel("Count of pixels")
ax[1].set_title("Distribution of the pixel values")

# Set the title for the entire figure
_ = fig.suptitle("Original image of a raccoon face")

# %%
# Compression via vector quantization
# -----------------------------------
#
# The idea behind compression via vector quantization is to reduce the number of
# gray levels to represent an image. For instance, we can use 8 values instead
# of 256 values. Therefore, it means that we could efficiently use 3 bits instead
# of 8 bits to encode a single pixel and therefore reduce the memory usage by a
# factor of approximately 2.5. We will later discuss about this memory usage.
#
# Encoding strategy
# """""""""""""""""
#
# The compression can be done using a
# :class:`~sklearn.preprocessing.KBinsDiscretizer`. We need to choose a strategy
# to define the 8 gray values to sub-sample. The simplest strategy is to define
# them equally spaced, which correspond to setting `strategy="uniform"`. From
# the previous histogram, we know that this strategy is certainly not optimal.

from sklearn.preprocessing import KBinsDiscretizer

# Define the number of bins for quantization
n_bins = 8

# Initialize the KBinsDiscretizer with specified parameters
encoder = KBinsDiscretizer(
    n_bins=n_bins,
    encode="ordinal",
    strategy="uniform",
    random_state=0,


# 设置聚类算法的初始化策略为"uniform"，表示初始化簇中心时使用均匀分布
strategy="uniform",
# 设置随机数生成器的种子为0，确保每次运行结果一致性
random_state=0,
# 使用编码器对象对狸猫的面部图像进行压缩，并将其重新整形为与原始图像相同的形状
compressed_raccoon_uniform = encoder.fit_transform(raccoon_face.reshape(-1, 1)).reshape(
    raccoon_face.shape
)

# 创建一个包含两个子图的图像窗口，设置每个子图的尺寸为12x4英寸
fig, ax = plt.subplots(ncols=2, figsize=(12, 4))

# 在第一个子图中显示经过均匀策略压缩后的狸猫面部图像，使用灰度颜色映射
ax[0].imshow(compressed_raccoon_uniform, cmap=plt.cm.gray)
ax[0].axis("off")  # 关闭坐标轴显示
ax[0].set_title("Rendering of the image")  # 设置子图标题

# 在第二个子图中绘制经过压缩后的像素值分布直方图，使用256个bins
ax[1].hist(compressed_raccoon_uniform.ravel(), bins=256)
ax[1].set_xlabel("Pixel value")  # 设置X轴标签
ax[1].set_ylabel("Count of pixels")  # 设置Y轴标签
ax[1].set_title("Sub-sampled distribution of the pixel values")  # 设置子图标题

# 设置整体图像的标题
_ = fig.suptitle("Raccoon face compressed using 3 bits and a uniform strategy")

# %%
# 在质量上，我们可以看到一些小区域，例如右下角的树叶，显示出了压缩的效果。
# 但总体而言，压缩后的图像仍然看起来很好。
#
# 我们观察到像素值的分布已经映射到了8个不同的值。我们可以检查这些值与原始像素值的对应关系。
bin_edges = encoder.bin_edges_[0]
bin_center = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
bin_center

# %%
# 创建一个新的图像窗口
_, ax = plt.subplots()

# 绘制原始狸猫面部图像的像素值分布直方图，使用256个bins
ax.hist(raccoon_face.ravel(), bins=256)
color = "tab:orange"

# 在直方图上用橙色线条和文本标出每个bin的中心值
for center in bin_center:
    ax.axvline(center, color=color)
    ax.text(center - 10, ax.get_ybound()[1] + 100, f"{center:.1f}", color=color)

# %%
# 正如之前所述，均匀采样策略并非最佳选择。例如，映射到值为7的像素将编码相对较少的信息，
# 而映射到值为3的像素将代表较大数量的像素。
# 我们可以使用聚类策略如k-means来找到一个更优的映射。
encoder = KBinsDiscretizer(
    n_bins=n_bins,
    encode="ordinal",
    strategy="kmeans",
    random_state=0,
)

# 使用k-means策略对狸猫面部图像进行压缩，并将其重新整形为与原始图像相同的形状
compressed_raccoon_kmeans = encoder.fit_transform(raccoon_face.reshape(-1, 1)).reshape(
    raccoon_face.shape
)

# 创建一个包含两个子图的图像窗口，设置每个子图的尺寸为12x4英寸
fig, ax = plt.subplots(ncols=2, figsize=(12, 4))

# 在第一个子图中显示经过k-means策略压缩后的狸猫面部图像，使用灰度颜色映射
ax[0].imshow(compressed_raccoon_kmeans, cmap=plt.cm.gray)
ax[0].axis("off")  # 关闭坐标轴显示
ax[0].set_title("Rendering of the image")  # 设置子图标题

# 在第二个子图中绘制经过压缩后的像素值分布直方图，使用256个bins
ax[1].hist(compressed_raccoon_kmeans.ravel(), bins=256)
ax[1].set_xlabel("Pixel value")  # 设置X轴标签
ax[1].set_ylabel("Number of pixels")  # 设置Y轴标签
ax[1].set_title("Distribution of the pixel values")  # 设置子图标题

# 设置整体图像的标题
_ = fig.suptitle("Raccoon face compressed using 3 bits and a K-means strategy")

# %%
bin_edges = encoder.bin_edges_[0]
bin_center = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
bin_center

# %%
# 现在bins中的计数更加平衡，并且它们的中心值不再等间距分布。
# 注意，我们可以通过使用`strategy="quantile"`而不是`strategy="kmeans"`来强制每个bin中有相同数量的像素。
#
# Memory footprint
# """"""""""""""""
#
# 我们之前提到应该节省大约8倍的内存。让我们验证一下。
# 打印压缩后的图像在内存中占用的字节数
print(f"The number of bytes taken in RAM is {compressed_raccoon_kmeans.nbytes}")

# %%
# 令人惊讶的是，我们的压缩图像占用的内存比原始图像多了8倍。这与我们的预期完全相反。
# 主要原因是所用于编码图像的数据类型不同。
# 
# 打印压缩图像的数据类型
print(f"Compression ratio: {compressed_raccoon_kmeans.nbytes / raccoon_face.nbytes}")

# %%
# 实际上，`sklearn.preprocessing.KBinsDiscretizer` 的输出是一个64位浮点数数组。
# 这意味着它占用了8倍的内存。然而，我们使用这种64位浮点数表示来编码8个值。
# 如果将压缩图像转换为3位整数数组，将能节省内存。但是，3位整数表示并不存在，
# 而要编码8个值，我们只能使用8位无符号整数表示。
#
# 实际上，要观察到内存节省，原始图像必须以64位浮点数表示。
print(f"Type of the compressed image: {compressed_raccoon_kmeans.dtype}")

# %%
# 实际上，`sklearn.preprocessing.KBinsDiscretizer` 的输出是一个64位浮点数数组。
# 这意味着它占用了8倍的内存。然而，我们使用这种64位浮点数表示来编码8个值。
# 如果将压缩图像转换为3位整数数组，将能节省内存。但是，3位整数表示并不存在，
# 而要编码8个值，我们只能使用8位无符号整数表示。
#
# 实际上，要观察到内存节省，原始图像必须以64位浮点数表示。
```