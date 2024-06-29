# `D:\src\scipysrc\matplotlib\galleries\examples\style_sheets\bmh.py`

```py
"""
========================================
Bayesian Methods for Hackers style sheet
========================================

This example demonstrates the style used in the Bayesian Methods for Hackers
[1]_ online book.

.. [1] http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/

"""

# 导入 matplotlib 的 pyplot 模块，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 模块，并重命名为 np
import numpy as np

# 设置随机种子以便结果可复现
np.random.seed(19680801)

# 使用 'bmh' 风格样式
plt.style.use('bmh')

# 定义一个绘制 Beta 分布直方图的函数
def plot_beta_hist(ax, a, b):
    # 生成 Beta 分布的随机样本并绘制直方图
    ax.hist(np.random.beta(a, b, size=10000),
            histtype="stepfilled", bins=25, alpha=0.8, density=True)

# 创建一个图形和坐标轴对象
fig, ax = plt.subplots()
# 调用 plot_beta_hist 函数绘制四种 Beta 分布的直方图
plot_beta_hist(ax, 10, 10)
plot_beta_hist(ax, 4, 12)
plot_beta_hist(ax, 50, 12)
plot_beta_hist(ax, 6, 55)
# 设置图形标题
ax.set_title("'bmh' style sheet")

# 显示图形
plt.show()
```