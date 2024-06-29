# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\barh.py`

```
"""
====================
Horizontal bar chart
====================

This example showcases a simple horizontal bar chart.
"""
# 导入 matplotlib 的 pyplot 模块，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 模块，并重命名为 np
import numpy as np

# 设定随机种子以保证结果可复现性
np.random.seed(19680801)

# 创建图形和坐标轴对象
fig, ax = plt.subplots()

# 示例数据
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
# 生成一个和 people 长度相同的序列，用于设置水平条形图的位置
y_pos = np.arange(len(people))
# 生成一个和 people 长度相同的随机性能数据
performance = 3 + 10 * np.random.rand(len(people))
# 生成一个和 people 长度相同的随机误差数据
error = np.random.rand(len(people))

# 绘制水平条形图，设置误差条和对齐方式
ax.barh(y_pos, performance, xerr=error, align='center')
# 设置 y 轴刻度的位置和标签
ax.set_yticks(y_pos, labels=people)
# 反转 y 轴，使标签从上到下显示
ax.invert_yaxis()  # labels read top-to-bottom
# 设置 x 轴标签
ax.set_xlabel('Performance')
# 设置图表标题
ax.set_title('How fast do you want to go today?')

# 展示图表
plt.show()
```