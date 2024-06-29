# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\scatter_demo2.py`

```
"""
=============
Scatter Demo2
=============

Demo of scatter plot with varying marker colors and sizes.
"""

# 导入matplotlib.pyplot库并命名为plt，用于绘图
import matplotlib.pyplot as plt
# 导入numpy库并命名为np，用于数值计算
import numpy as np
# 导入matplotlib.cbook库，提供一些额外的工具函数
import matplotlib.cbook as cbook

# 从mpl-data/sample_data目录中读取yahoo csv数据，并使用字段date, open, high,
# low, close, volume, adj_close创建一个numpy记录数组。date列以np.datetime64类型存储，
# 时间单位为天（'D'）。
price_data = cbook.get_sample_data('goog.npz')['price_data']
# 只保留最近的250个交易日的数据
price_data = price_data[-250:]

# 计算每日调整收盘价的百分比变化
delta1 = np.diff(price_data["adj_close"]) / price_data["adj_close"][:-1]

# 计算每日交易量的标记大小，单位为点的平方
volume = (15 * price_data["volume"][:-2] / price_data["volume"][0])**2
# 计算每日开盘价与收盘价的比例，并乘以一个常数，用作标记颜色
close = 0.003 * price_data["close"][:-2] / 0.003 * price_data["open"][:-2]

# 创建一个图形和一个坐标系对象
fig, ax = plt.subplots()
# 绘制散点图，x轴为delta1[:-1]，y轴为delta1[1:]，颜色为close，大小为volume，透明度为0.5
ax.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.5)

# 设置x轴标签为'$\Delta_i$'，字体大小为15
ax.set_xlabel(r'$\Delta_i$', fontsize=15)
# 设置y轴标签为'$\Delta_{i+1}$'，字体大小为15
ax.set_ylabel(r'$\Delta_{i+1}$', fontsize=15)
# 设置图表标题为'Volume and percent change'
ax.set_title('Volume and percent change')

# 打开坐标系的网格线
ax.grid(True)
# 调整图形布局以适应所有子图和标签
fig.tight_layout()

# 显示图形
plt.show()
```