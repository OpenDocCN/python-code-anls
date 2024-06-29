# `D:\src\scipysrc\matplotlib\galleries\examples\shapes_and_collections\scatter.py`

```py
"""
============
Scatter plot
============

This example showcases a simple scatter plot.
"""
# 导入 matplotlib 的 pyplot 模块，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并重命名为 np
import numpy as np

# 设置随机数种子，以便结果可复现
np.random.seed(19680801)

# 定义数据点数量 N
N = 50
# 生成随机数组 x，包含 N 个在 [0, 1) 范围内的随机数
x = np.random.rand(N)
# 生成随机数组 y，包含 N 个在 [0, 1) 范围内的随机数
y = np.random.rand(N)
# 生成随机数组 colors，包含 N 个在 [0, 1) 范围内的随机数
colors = np.random.rand(N)
# 生成随机数组 area，元素为 (30 * [0, 1) 范围内的随机数)^2，即 0 到 900 的随机数
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

# 绘制散点图，设置数据点的位置、大小、颜色和透明度
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.scatter` / `matplotlib.pyplot.scatter`
```