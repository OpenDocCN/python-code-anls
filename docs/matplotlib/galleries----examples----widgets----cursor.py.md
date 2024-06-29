# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\cursor.py`

```py
"""
======
Cursor
======

"""
# 导入 matplotlib.pyplot 作为 plt，导入 numpy 作为 np
import matplotlib.pyplot as plt
import numpy as np

# 从 matplotlib.widgets 模块中导入 Cursor 类
from matplotlib.widgets import Cursor

# 设定随机数生成的种子，以便结果可复现性
np.random.seed(19680801)

# 创建一个大小为 (8, 6) 的图形对象和坐标轴对象
fig, ax = plt.subplots(figsize=(8, 6))

# 生成两组包含 100 个元素的随机数，范围在 -2 到 2 之间
x, y = 4*(np.random.rand(2, 100) - .5)
# 在坐标轴上绘制散点图
ax.plot(x, y, 'o')
# 设置 x 轴和 y 轴的显示范围为 -2 到 2
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# 创建一个 Cursor 对象，关联到当前的坐标轴 ax，使用红色线条，线宽为 2，启用 useblit 提高性能
cursor = Cursor(ax, useblit=True, color='red', linewidth=2)

# 显示绘图结果
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.Cursor`
```