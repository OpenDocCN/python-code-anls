# `D:\src\scipysrc\matplotlib\galleries\examples\style_sheets\dark_background.py`

```py
"""
===========================
Dark background style sheet
===========================

This example demonstrates the "dark_background" style, which uses white for
elements that are typically black (text, borders, etc). Note that not all plot
elements default to colors defined by an rc parameter.

"""
# 导入 matplotlib 的 pyplot 模块，并将其简称为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简称为 np
import numpy as np

# 使用 'dark_background' 风格样式
plt.style.use('dark_background')

# 创建一个包含图形和轴对象的图形窗口
fig, ax = plt.subplots()

# 定义一个长度为 6 的数组
L = 6
# 在 [0, L] 区间内生成均匀间隔的数字序列
x = np.linspace(0, L)
# 获取当前图形属性循环中定义的颜色数量
ncolors = len(plt.rcParams['axes.prop_cycle'])
# 在 [0, L] 区间内生成均匀间隔的数字序列，与颜色数量相等
shift = np.linspace(0, L, ncolors, endpoint=False)
# 遍历 shift 数组中的每个值
for s in shift:
    # 绘制曲线，x 轴为 x，y 轴为 np.sin(x + s)，使用圆圈标记
    ax.plot(x, np.sin(x + s), 'o-')

# 设置 x 轴标签
ax.set_xlabel('x-axis')
# 设置 y 轴标签
ax.set_ylabel('y-axis')
# 设置图形标题
ax.set_title("'dark_background' style sheet")

# 显示图形
plt.show()
```