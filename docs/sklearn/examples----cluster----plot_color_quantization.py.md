# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_color_quantization.py`

```
"""
==================================
Color Quantization using K-Means
==================================

Performs a pixel-wise Vector Quantization (VQ) of an image of the summer palace
(China), reducing the number of colors required to show the image from 96,615
unique colors to 64, while preserving the overall appearance quality.

In this example, pixels are represented in a 3D-space and K-means is used to
find 64 color clusters. In the image processing literature, the codebook
obtained from K-means (the cluster centers) is called the color palette. Using
a single byte, up to 256 colors can be addressed, whereas an RGB encoding
requires 3 bytes per pixel. The GIF file format, for example, uses such a
palette.

For comparison, a quantized image using a random codebook (colors picked up
randomly) is also shown.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from time import time  # 导入时间计算功能

import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入数值计算库

from sklearn.cluster import KMeans  # 导入K均值聚类算法
from sklearn.datasets import load_sample_image  # 导入加载示例图像的函数
from sklearn.metrics import pairwise_distances_argmin  # 导入计算最近距离函数
from sklearn.utils import shuffle  # 导入数据洗牌工具函数

n_colors = 64  # 指定颜色数目为64

# Load the Summer Palace photo
china = load_sample_image("china.jpg")  # 加载示例图像“china.jpg”

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow works well on float data (need to
# be in the range [0-1])
china = np.array(china, dtype=np.float64) / 255  # 转换图像数据类型为浮点数，并进行归一化处理

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(china.shape)  # 获取图像的宽度、高度和深度
assert d == 3  # 确保图像为RGB格式
image_array = np.reshape(china, (w * h, d))  # 将图像转换为2D数组形式

print("Fitting model on a small sub-sample of the data")
t0 = time()  # 记录当前时间
image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)  # 对图像数据进行洗牌并取样1,000个数据点
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)  # 使用K均值算法对取样后的数据进行聚类
print(f"done in {time() - t0:0.3f}s.")  # 输出训练模型所花费的时间

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()  # 记录当前时间
labels = kmeans.predict(image_array)  # 对整个图像数据进行颜色索引预测
print(f"done in {time() - t0:0.3f}s.")  # 输出预测所花费的时间


codebook_random = shuffle(image_array, random_state=0, n_samples=n_colors)  # 随机选择颜色来构建随机代码簿
print("Predicting color indices on the full image (random)")
t0 = time()  # 记录当前时间
labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)  # 使用最近距离法预测随机颜色索引
print(f"done in {time() - t0:0.3f}s.")


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1)  # 根据代码簿和标签重建（压缩）图像


# Display all results, alongside original image
plt.figure(1)
plt.clf()
plt.axis("off")
plt.title("Original image (96,615 colors)")
plt.imshow(china)  # 显示原始图像

plt.figure(2)
plt.clf()
plt.axis("off")
plt.title(f"Quantized image ({n_colors} colors, K-Means)")
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))  # 显示使用K均值量化后的图像

plt.figure(3)
plt.clf()
plt.axis("off")
plt.title(f"Quantized image ({n_colors} colors, Random)")
plt.imshow(recreate_image(codebook_random, labels_random, w, h))  # 显示使用随机代码簿量化后的图像
plt.show()
```