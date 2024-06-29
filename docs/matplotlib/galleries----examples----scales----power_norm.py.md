# `D:\src\scipysrc\matplotlib\galleries\examples\scales\power_norm.py`

```py
"""
========================
Exploring normalizations
========================

Various normalization on a multivariate normal distribution.

"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
import numpy as np  # 导入numpy库用于数值计算
from numpy.random import multivariate_normal  # 导入multivariate_normal函数用于生成多元正态分布数据

import matplotlib.colors as mcolors  # 导入matplotlib.colors模块，用于颜色映射

# 设置随机种子以便结果可重现
np.random.seed(19680801)

# 生成两组多元正态分布数据
data = np.vstack([
    multivariate_normal([10, 10], [[3, 2], [2, 3]], size=100000),
    multivariate_normal([30, 20], [[3, 1], [1, 3]], size=1000)
])

# 不同的 gamma 值列表
gammas = [0.8, 0.5, 0.3]

# 创建一个2x2的子图
fig, axs = plt.subplots(nrows=2, ncols=2)

# 第一个子图：线性归一化
axs[0, 0].set_title('Linear normalization')
axs[0, 0].hist2d(data[:, 0], data[:, 1], bins=100)

# 其他三个子图：使用不同的 Power law gamma 值
for ax, gamma in zip(axs.flat[1:], gammas):
    ax.set_title(r'Power law $(\gamma=%1.1f)$' % gamma)
    ax.hist2d(data[:, 0], data[:, 1], bins=100, norm=mcolors.PowerNorm(gamma))

# 调整子图布局
fig.tight_layout()

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.colors`
#    - `matplotlib.colors.PowerNorm`
#    - `matplotlib.axes.Axes.hist2d`
#    - `matplotlib.pyplot.hist2d`
```