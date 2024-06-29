# `D:\src\scipysrc\matplotlib\galleries\examples\showcase\stock_prices.py`

```py
"""
==========================
Stock prices over 32 years
==========================

.. redirect-from:: /gallery/showcase/bachelors_degrees_by_gender

A graph of multiple time series that demonstrates custom styling of plot frame,
tick lines, tick labels, and line graph properties. It also uses custom
placement of text labels along the right edge as an alternative to a
conventional legend.

Note: The third-party mpl style dufte_ produces similar-looking plots with less
code.

.. _dufte: https://github.com/nschloe/dufte
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库用于绘图
import numpy as np  # 导入 numpy 库用于数值计算

from matplotlib.cbook import get_sample_data  # 从 matplotlib.cbook 中导入获取示例数据的函数
import matplotlib.transforms as mtransforms  # 导入 matplotlib.transforms 库用于坐标转换

# 使用 get_sample_data 函数获取示例数据文件 'Stocks.csv' 并读取为 numpy 数组
with get_sample_data('Stocks.csv') as file:
    stock_data = np.genfromtxt(
        file, delimiter=',', names=True, dtype=None,
        converters={0: lambda x: np.datetime64(x, 'D')}, skip_header=1)

# 创建图和坐标轴
fig, ax = plt.subplots(1, 1, figsize=(6, 8), layout='constrained')

# 设置绘图时使用的颜色循环
ax.set_prop_cycle(color=[
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
    '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
    '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
    '#17becf', '#9edae5'])

# 设置股票名称和股票代码
stocks_name = ['IBM', 'Apple', 'Microsoft', 'Xerox', 'Amazon', 'Dell',
               'Alphabet', 'Adobe', 'S&P 500', 'NASDAQ']
stocks_ticker = ['IBM', 'AAPL', 'MSFT', 'XRX', 'AMZN', 'DELL', 'GOOGL',
                 'ADBE', 'GSPC', 'IXIC']

# 手动调整每支股票标签在垂直方向上的位置（单位为 points，1/72 英寸）
y_offsets = {k: 0 for k in stocks_ticker}
y_offsets['IBM'] = 5
y_offsets['AAPL'] = -5
y_offsets['AMZN'] = -6

# 遍历每支股票
for nn, column in enumerate(stocks_ticker):
    # 绘制每条线，每条线使用自己的颜色
    # 只包含非 NaN 数据
    good = np.nonzero(np.isfinite(stock_data[column]))
    line, = ax.plot(stock_data['Date'][good], stock_data[column][good], lw=2.5)

    # 在每条线的右端添加文本标签。以下代码主要是为了调整某些标签的垂直位置以防重叠。
    y_pos = stock_data[column][-1]

    # 使用偏移变换来调整需要向上或向下微调的文本的位置
    offset = y_offsets[column] / 72
    trans = mtransforms.ScaledTranslation(0, offset, fig.dpi_scale_trans)
    trans = ax.transData + trans

    # 确保所有标签足够大，以便查看者能够轻松阅读
    ax.text(np.datetime64('2022-10-01'), y_pos, stocks_name[nn],
            color=line.get_color(), transform=trans)

# 设置 x 轴的显示范围
ax.set_xlim(np.datetime64('1989-06-01'), np.datetime64('2023-01-01'))

# 设置图标题
fig.suptitle("Technology company stocks prices dollars (1990-2022)",
             ha="center")

# 移除图的边框线，这里不需要它们
ax.spines[:].set_visible(False)

# 确保坐标轴的刻度只显示在图的底部和左侧
# Ticks on the right and top of the plot are generally unnecessary.
# 移动 x 轴刻度至图的底部
ax.xaxis.tick_bottom()
# 移动 y 轴刻度至图的左侧
ax.yaxis.tick_left()
# 设置 y 轴为对数坐标轴
ax.set_yscale('log')

# Provide tick lines across the plot to help your viewers trace along
# the axis ticks. Make sure that the lines are light and small so they
# don't obscure the primary data lines.
# 在图上绘制网格线，帮助观众跟踪坐标轴刻度，确保网格线轻且细，不遮挡主要数据线条。
ax.grid(True, 'major', 'both', ls='--', lw=.5, c='k', alpha=.3)

# Remove the tick marks; they are unnecessary with the tick lines we just
# plotted. Make sure your axis ticks are large enough to be easily read.
# You don't want your viewers squinting to read your plot.
# 移除坐标轴上的刻度标记，因为刚刚绘制了网格线。确保坐标轴刻度足够大，易于阅读。
ax.tick_params(axis='both', which='both', labelsize='large',
               bottom=False, top=False, labelbottom=True,
               left=False, right=False, labelleft=True)

# Finally, save the figure as a PNG.
# You can also save it as a PDF, JPEG, etc.
# Just change the file extension in this call.
# 将图保存为 PNG 格式。也可以保存为 PDF、JPEG 等格式。在此调用中只需更改文件扩展名。
fig.savefig('stock-prices.png', bbox_inches='tight')
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.pyplot.subplots`
#    - `matplotlib.axes.Axes.text`
#    - `matplotlib.axis.XAxis.tick_bottom`
#    - `matplotlib.axis.YAxis.tick_left`
#    - `matplotlib.artist.Artist.set_visible`
```