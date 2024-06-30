# `D:\src\scipysrc\scikit-learn\examples\neighbors\plot_digits_kde_sampling.py`

```
"""
=========================
Kernel Density Estimation
=========================

This example shows how kernel density estimation (KDE), a powerful
non-parametric density estimation technique, can be used to learn
a generative model for a dataset.  With this generative model in place,
new samples can be drawn.  These new samples reflect the underlying model
of the data.

"""

# 导入 matplotlib 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于数值计算
import numpy as np

# 导入 load_digits 数据集和 PCA、GridSearchCV、KernelDensity 模块
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

# 加载手写数字数据集
digits = load_digits()

# 使用 PCA 将 64 维数据投影到 15 维，不进行白化
pca = PCA(n_components=15, whiten=False)
data = pca.fit_transform(digits.data)

# 使用网格搜索交叉验证来优化带宽参数
params = {"bandwidth": np.logspace(-1, 1, 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(data)

# 打印出最佳带宽参数值
print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

# 使用最佳估算器计算核密度估计
kde = grid.best_estimator_

# 从数据中抽样生成 44 个新数据点
new_data = kde.sample(44, random_state=0)
# 将生成的数据点逆转换回原始空间
new_data = pca.inverse_transform(new_data)

# 将数据转换为 4x11 的网格
new_data = new_data.reshape((4, 11, -1))
real_data = digits.data[:44].reshape((4, 11, -1))

# 绘制真实数字和重新采样的数字
fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
for j in range(11):
    # 隐藏多余的子图区域
    ax[4, j].set_visible(False)
    # 绘制真实数据
    for i in range(4):
        im = ax[i, j].imshow(
            real_data[i, j].reshape((8, 8)), cmap=plt.cm.binary, interpolation="nearest"
        )
        im.set_clim(0, 16)
        # 绘制重新采样的数据
        im = ax[i + 5, j].imshow(
            new_data[i, j].reshape((8, 8)), cmap=plt.cm.binary, interpolation="nearest"
        )
        im.set_clim(0, 16)

# 设置子图标题
ax[0, 5].set_title("Selection from the input data")
ax[5, 5].set_title('"New" digits drawn from the kernel density model')

# 展示图像
plt.show()
```