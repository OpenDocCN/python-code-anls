# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\figure_title.py`

```py
"""
=============================================
Figure labels: suptitle, supxlabel, supylabel
=============================================

Each Axes can have a title (or actually three - one each with *loc* "left",
"center", and "right"), but is sometimes desirable to give a whole figure
(or `.SubFigure`) an overall title, using `.Figure.suptitle`.

We can also add figure-level x- and y-labels using `.Figure.supxlabel` and
`.Figure.supylabel`.
"""

# 导入 matplotlib 库并取别名 plt
import matplotlib.pyplot as plt
# 导入 numpy 库并取别名 np
import numpy as np
# 从 matplotlib.cbook 中导入 get_sample_data 函数
from matplotlib.cbook import get_sample_data

# 生成一个包含均匀间隔数据的数组 x
x = np.linspace(0.0, 5.0, 501)

# 创建一个包含两个子图的图形对象 fig，分别存储在 ax1 和 ax2 中
fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained', sharey=True)
# 在第一个子图 ax1 上绘制曲线，设置标题和坐标轴标签
ax1.plot(x, np.cos(6*x) * np.exp(-x))
ax1.set_title('damped')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('amplitude')

# 在第二个子图 ax2 上绘制曲线，设置标题和坐标轴标签
ax2.plot(x, np.cos(6*x))
ax2.set_xlabel('time (s)')
ax2.set_title('undamped')

# 设置整个图形的标题
fig.suptitle('Different types of oscillations', fontsize=16)

# %%
# 可以使用 `.Figure.supxlabel` 和 `.Figure.supylabel` 方法设置全局的 x 或 y 标签。

# 使用 get_sample_data 函数获取示例数据集 'Stocks.csv'
with get_sample_data('Stocks.csv') as file:
    # 从 CSV 文件中读取数据到结构化数组 stocks
    stocks = np.genfromtxt(
        file, delimiter=',', names=True, dtype=None,
        converters={0: lambda x: np.datetime64(x, 'D')}, skip_header=1)

# 创建包含 4x2 子图的图形对象 fig，并存储在 axs 中
fig, axs = plt.subplots(4, 2, figsize=(9, 5), layout='constrained',
                        sharex=True, sharey=True)
# 遍历所有子图，每个子图绘制一条曲线，并设置标题
for nn, ax in enumerate(axs.flat):
    column_name = stocks.dtype.names[1+nn]
    y = stocks[column_name]
    line, = ax.plot(stocks['Date'], y / np.nanmax(y), lw=2.5)
    ax.set_title(column_name, fontsize='small', loc='left')

# 设置整个图形的全局 x 标签和 y 标签
fig.supxlabel('Year')
fig.supylabel('Stock price relative to max')

# 显示图形
plt.show()
```