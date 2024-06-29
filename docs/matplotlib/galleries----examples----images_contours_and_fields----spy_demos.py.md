# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\spy_demos.py`

```py
"""
=========
Spy Demos
=========

Plot the sparsity pattern of arrays.
"""

# 导入必要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 库并使用 np 别名

# 设置随机种子以便结果可复现
np.random.seed(19680801)

# 创建一个包含4个子图的图形对象
fig, axs = plt.subplots(2, 2)
ax1 = axs[0, 0]  # 左上角子图对象
ax2 = axs[0, 1]  # 右上角子图对象
ax3 = axs[1, 0]  # 左下角子图对象
ax4 = axs[1, 1]  # 右下角子图对象

# 生成一个大小为 (20, 20) 的随机数组
x = np.random.randn(20, 20)
# 将第5行所有元素置为0
x[5, :] = 0.
# 将第12列所有元素置为0
x[:, 12] = 0.

# 在 ax1 子图上绘制 x 的稀疏模式，使用默认精度和标记大小为5
ax1.spy(x, markersize=5)
# 在 ax2 子图上绘制 x 的稀疏模式，精度设置为0.1，标记大小为5
ax2.spy(x, precision=0.1, markersize=5)

# 在 ax3 子图上绘制 x 的稀疏模式，默认精度
ax3.spy(x)
# 在 ax4 子图上绘制 x 的稀疏模式，精度设置为0.1
ax4.spy(x, precision=0.1)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.spy` / `matplotlib.pyplot.spy`
```