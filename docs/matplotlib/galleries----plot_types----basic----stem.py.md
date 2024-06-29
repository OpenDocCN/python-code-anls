# `D:\src\scipysrc\matplotlib\galleries\plot_types\basic\stem.py`

```py
"""
==========
stem(x, y)
==========
Create a stem plot.

See `~matplotlib.axes.Axes.stem`.
"""

# 导入 matplotlib.pyplot 库，并命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并命名为 np
import numpy as np

# 使用指定的样式表 '_mpl-gallery'
plt.style.use('_mpl-gallery')

# 创建数据
x = 0.5 + np.arange(8)  # 生成从0.5开始的连续8个数作为 x 轴数据
y = [4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6, 3.0]  # 定义与 x 对应的 y 轴数据

# 创建图形和轴对象
fig, ax = plt.subplots()

# 绘制 stem plot (干茎图)
ax.stem(x, y)

# 设置 x 轴和 y 轴的范围和刻度
ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

# 显示图形
plt.show()
```