# `D:\src\scipysrc\scikit-learn\examples\datasets\plot_digits_last_image.py`

```
"""
=========================================================
The Digit Dataset
=========================================================

This dataset is made up of 1797 8x8 images. Each image,
like the one shown below, is of a hand-written digit.
In order to utilize an 8x8 figure like this, we'd have to
first transform it into a feature vector with length 64.

See `here
<https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits>`_
for more information about this dataset.

"""

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# SPDX-License-Identifier: BSD-3-Clause

# 导入 matplotlib 的 pyplot 模块作为 plt
import matplotlib.pyplot as plt

# 导入 sklearn 的 datasets 模块
from sklearn import datasets

# Load the digits dataset
# 载入手写数字数据集
digits = datasets.load_digits()

# Display the last digit
# 创建一个大小为 3x3 英寸的图形窗口，并显示数据集中的最后一个手写数字图像
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
```