# `D:\src\scipysrc\matplotlib\galleries\users_explain\axes\axes_ticks.py`

```py
#
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Create a figure with subplots arranged in a constrained layout
fig, axs = plt.subplots(2, 1, figsize=(5.4, 5.4), layout='constrained')

# Generate data for x-axis
x = np.arange(100)

# Iterate through each subplot
for nn, ax in enumerate(axs):
    # Plot a simple line graph on each subplot
    ax.plot(x, x)
    # Conditionally customize each subplot based on index
    if nn == 1:
        # Set title for the subplot
        ax.set_title('Manual ticks')
        # Set specific y-axis tick locations
        ax.set_yticks(np.arange(0, 100.1, 100/3))
        # Define custom x-axis tick locations and labels
        xticks = np.arange(0.50, 101, 20)
        xlabels = [f'\\${x:1.2f}' for x in xticks]
        ax.set_xticks(xticks, labels=xlabels)
    else:
        # Set title for the subplot
        ax.set_title('Automatic ticks')

# %%
#
# Note that the length of the ``labels`` argument must have the same length as
# the array used to specify the ticks.
#
# By default `~.axes.Axes.set_xticks` and `~.axes.Axes.set_yticks` act on the
# major ticks of an Axis, however it is possible to add minor ticks:

# Create another figure with subplots arranged in a constrained layout
fig, axs = plt.subplots(2, 1, figsize=(5.4, 5.4), layout='constrained')

# Generate data for x-axis
x = np.arange(100)

# Iterate through each subplot
for nn, ax in enumerate(axs):
    # Plot a simple line graph on each subplot
    ax.plot(x, x)
    # Conditionally customize each subplot based on index
    if nn == 1:
        # Set title for the subplot
        ax.set_title('Manual ticks')
        # Set specific y-axis tick locations for both major and minor ticks
        ax.set_yticks(np.arange(0, 100.1, 100/3))
        ax.set_yticks(np.arange(0, 100.1, 100/30), minor=True)
    else:
        # Set title for the subplot
        ax.set_title('Automatic ticks')


# %%
#
# Locators and Formatters
# =======================
#
# Manually setting the ticks as above works well for specific final plots, but
# does not adapt as the user interacts with the Axes.   At a lower level,
# Matplotlib has ``Locators`` that are meant to automatically choose ticks
# depending on the current view limits of the axis, and ``Formatters`` that are
# meant to format the tick labels automatically.
#
# The full list of locators provided by Matplotlib are listed at
# :ref:`locators`, and the formatters at :ref:`formatters`.

# %%

# Function to set up common parameters for Axes in the example
def setup(ax, title):
    """Set up common parameters for the Axes in the example."""
    # Hide all spines except the bottom one
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines[['left', 'right', 'top']].set_visible(False)

    # Set ticks position to bottom
    ax.xaxis.set_ticks_position('bottom')
    
    # Set parameters for major and minor ticks
    ax.tick_params(which='major', width=1.00, length=5)
    ax.tick_params(which='minor', width=0.75, length=2.5)
    
    # Set limits for x and y axes
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    
    # Add text to the Axes
    ax.text(0.0, 0.2, title, transform=ax.transAxes,
            fontsize=14, fontname='Monospace', color='tab:blue')


# Create a figure with 8 subplots arranged in a constrained layout
fig, axs = plt.subplots(8, 1, layout='constrained')

# Call setup function for the first subplot
setup(axs[0], title="NullLocator()")
# 设置第一个子图的 x 轴主刻度定位器为空定位器，即不显示主刻度
axs[0].xaxis.set_major_locator(ticker.NullLocator())
# 设置第一个子图的 x 轴次刻度定位器为空定位器，即不显示次刻度
axs[0].xaxis.set_minor_locator(ticker.NullLocator())

# 设置第二个子图的标题为 "MultipleLocator(0.5)"
setup(axs[1], title="MultipleLocator(0.5)")
# 设置第二个子图的 x 轴主刻度定位器为多重刻度器，每隔0.5显示一个主刻度
axs[1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
# 设置第二个子图的 x 轴次刻度定位器为多重刻度器，每隔0.1显示一个次刻度
axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# 设置第三个子图的标题为 "FixedLocator([0, 1, 5])"
setup(axs[2], title="FixedLocator([0, 1, 5])")
# 设置第三个子图的 x 轴主刻度定位器为固定刻度器，显示主刻度在 [0, 1, 5] 这些位置
axs[2].xaxis.set_major_locator(ticker.FixedLocator([0, 1, 5]))
# 设置第三个子图的 x 轴次刻度定位器为固定刻度器，均匀分布在 0.2 到 0.8 之间
axs[2].xaxis.set_minor_locator(ticker.FixedLocator(np.linspace(0.2, 0.8, 4)))

# 设置第四个子图的标题为 "LinearLocator(numticks=3)"
setup(axs[3], title="LinearLocator(numticks=3)")
# 设置第四个子图的 x 轴主刻度定位器为线性刻度器，分布3个主刻度
axs[3].xaxis.set_major_locator(ticker.LinearLocator(3))
# 设置第四个子图的 x 轴次刻度定位器为线性刻度器，分布31个次刻度
axs[3].xaxis.set_minor_locator(ticker.LinearLocator(31))

# 设置第五个子图的标题为 "IndexLocator(base=0.5, offset=0.25)"
setup(axs[4], title="IndexLocator(base=0.5, offset=0.25)")
# 在第五个子图上绘制一条白色线，用于显示效果，无实际数据
axs[4].plot(range(0, 5), [0]*5, color='white')
# 设置第五个子图的 x 轴主刻度定位器为索引刻度器，基础刻度为0.5，偏移0.25
axs[4].xaxis.set_major_locator(ticker.IndexLocator(base=0.5, offset=0.25))

# 设置第六个子图的标题为 "AutoLocator()"
setup(axs[5], title="AutoLocator()")
# 设置第六个子图的 x 轴主刻度定位器为自动刻度器，自动决定主刻度位置
axs[5].xaxis.set_major_locator(ticker.AutoLocator())
# 设置第六个子图的 x 轴次刻度定位器为自动次刻度器，自动决定次刻度位置
axs[5].xaxis.set_minor_locator(ticker.AutoMinorLocator())

# 设置第七个子图的标题为 "MaxNLocator(n=4)"
setup(axs[6], title="MaxNLocator(n=4)")
# 设置第七个子图的 x 轴主刻度定位器为最大 N 刻度器，显示最多4个主刻度
axs[6].xaxis.set_major_locator(ticker.MaxNLocator(4))
# 设置第七个子图的 x 轴次刻度定位器为最大 N 刻度器，显示最多40个次刻度
axs[6].xaxis.set_minor_locator(ticker.MaxNLocator(40))

# 设置第八个子图的标题为 "LogLocator(base=10, numticks=15)"
setup(axs[7], title="LogLocator(base=10, numticks=15)")
# 设置第八个子图 x 轴的数据范围为 10^3 到 10^10
axs[7].set_xlim(10**3, 10**10)
# 设置第八个子图 x 轴的刻度为对数刻度
axs[7].set_xscale('log')
# 设置第八个子图的 x 轴主刻度定位器为对数刻度器，基数为10，显示15个主刻度
axs[7].xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
plt.show()

# 定义函数 setup，用于设置示例中所有子图的公共参数
def setup(ax, title):
    """Set up common parameters for the Axes in the example."""
    # 仅显示底部轴线
    ax.yaxis.set_major_locator(ticker.NullLocator())
    # 隐藏左、右、顶部边框线
    ax.spines[['left', 'right', 'top']].set_visible(False)

    # 设置 x 轴主刻度定位器为多重刻度器，每隔1显示一个主刻度
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.00))
    # 设置 x 轴次刻度定位器为多重刻度器，每隔0.25显示一个次刻度
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    # 将 x 轴刻度显示在底部
    ax.xaxis.set_ticks_position('bottom')
    # 设置主刻度的参数：宽度为1.00，长度为5
    ax.tick_params(which='major', width=1.00, length=5)
    # 设置次刻度的参数：宽度为0.75，长度为2.5，标签大小为10
    ax.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
    # 设置 x 轴数据范围为 0 到 5
    ax.set_xlim(0, 5)
    # 设置 y 轴数据范围为 0 到 1
    ax.set_ylim(0, 1)
    # 在绘图中添加文本注释，放置在坐标轴的相对位置 (0.0, 0.2)，文本内容为变量 title
    # 使用 ax.transAxes 指定相对坐标系进行定位，确保文本随图形缩放而移动
    # 设置文本字体大小为 14 像素，字体为 'Monospace'，颜色为 'tab:blue'
    ax.text(0.0, 0.2, title, transform=ax.transAxes,
            fontsize=14, fontname='Monospace', color='tab:blue')
# 创建一个大小为8x8的图形对象，布局为'constrained'
fig = plt.figure(figsize=(8, 8), layout='constrained')
# 将图形对象分为3个子图，高度比例分别为1.5:1.5:7.5
fig0, fig1, fig2 = fig.subfigures(3, height_ratios=[1.5, 1.5, 7.5])

# 在第一个子图上设置标题为'String Formatting'，字体大小为16，水平对齐方式为左对齐
fig0.suptitle('String Formatting', fontsize=16, x=0, ha='left')
# 在第一个子图上创建一个坐标轴
ax0 = fig0.subplots()

# 调用setup函数设置坐标轴属性，标题为'{x} km'
setup(ax0, title="'{x} km'")
# 设置x轴的主要刻度格式为'{x} km'
ax0.xaxis.set_major_formatter('{x} km')

# 在第二个子图上设置标题为'Function Formatting'，字体大小为16，水平对齐方式为左对齐
fig1.suptitle('Function Formatting', fontsize=16, x=0, ha='left')
# 在第二个子图上创建一个坐标轴
ax1 = fig1.subplots()

# 调用setup函数设置坐标轴属性，标题为"def(x, pos): return str(x-5)"
setup(ax1, title="def(x, pos): return str(x-5)")
# 设置x轴的主要刻度格式为lambda函数，格式为str(x-5)
ax1.xaxis.set_major_formatter(lambda x, pos: str(x-5))

# 在第三个子图上设置标题为'Formatter Object Formatting'，字体大小为16，水平对齐方式为左对齐
fig2.suptitle('Formatter Object Formatting', fontsize=16, x=0, ha='left')
# 在第三个子图上创建7个坐标轴
axs2 = fig2.subplots(7, 1)

# 调用setup函数设置第一个坐标轴属性，标题为'NullFormatter()'
setup(axs2[0], title="NullFormatter()")
# 设置第一个坐标轴x轴的主要刻度格式为NullFormatter()
axs2[0].xaxis.set_major_formatter(ticker.NullFormatter())

# 调用setup函数设置第二个坐标轴属性，标题为"StrMethodFormatter('{x:.3f}')"
setup(axs2[1], title="StrMethodFormatter('{x:.3f}')")
# 设置第二个坐标轴x轴的主要刻度格式为StrMethodFormatter，格式为"{x:.3f}"
axs2[1].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))

# 调用setup函数设置第三个坐标轴属性，标题为"FormatStrFormatter('#%d')"
setup(axs2[2], title="FormatStrFormatter('#%d')")
# 设置第三个坐标轴x轴的主要刻度格式为FormatStrFormatter，格式为"#%d"
axs2[2].xaxis.set_major_formatter(ticker.FormatStrFormatter("#%d"))

# 定义一个函数fmt_two_digits，用于格式化数字为两位小数
def fmt_two_digits(x, pos):
    return f'[{x:.2f}]'

# 调用setup函数设置第四个坐标轴属性，标题为'FuncFormatter("[{:.2f}]".format)'
setup(axs2[3], title='FuncFormatter("[{:.2f}]".format)')
# 设置第四个坐标轴x轴的主要刻度格式为FuncFormatter，使用fmt_two_digits函数进行格式化
axs2[3].xaxis.set_major_formatter(ticker.FuncFormatter(fmt_two_digits))

# 调用setup函数设置第五个坐标轴属性，标题为"FixedFormatter(['A', 'B', 'C', 'D', 'E', 'F'])"
setup(axs2[4], title="FixedFormatter(['A', 'B', 'C', 'D', 'E', 'F'])")
# FixedFormatter应该与FixedLocator一起使用，否则无法确定标签的位置
# 创建位置和标签列表
positions = [0, 1, 2, 3, 4, 5]
labels = ['A', 'B', 'C', 'D', 'E', 'F']
# 设置第五个坐标轴x轴的主要刻度定位器为FixedLocator，设置标签格式为FixedFormatter
axs2[4].xaxis.set_major_locator(ticker.FixedLocator(positions))
axs2[4].xaxis.set_major_formatter(ticker.FixedFormatter(labels))

# 调用setup函数设置第六个坐标轴属性，标题为"ScalarFormatter()"
setup(axs2[5], title="ScalarFormatter()")
# 设置第六个坐标轴x轴的主要刻度格式为ScalarFormatter
axs2[5].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

# 调用setup函数设置第七个坐标轴属性，标题为"PercentFormatter(xmax=5)"
setup(axs2[6], title="PercentFormatter(xmax=5)")
# 设置第七个坐标轴x轴的主要刻度格式为PercentFormatter，最大值为5
axs2[6].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=5))

# 创建一个包含2个子图的图形对象，大小为6.4x3.2，布局为'constrained'
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.2), layout='constrained')

# 遍历每个子图，绘制一个包含100个点的折线图
for nn, ax in enumerate(axs):
    ax.plot(np.arange(100))
    # 如果 nn 等于 1，则执行以下操作
    if nn == 1:
        # 打开坐标轴网格线显示
        ax.grid('on')
        # 设置 y 轴的刻度参数：右侧显示、左侧隐藏、颜色为红色、刻度长度为16、无网格线
        ax.tick_params(right=True, left=False, axis='y', color='r', length=16,
                       grid_color='none')
        # 设置 x 轴的刻度参数：颜色为品红、刻度长度为4、刻度方向向内、刻度线宽度为4、标签颜色为绿色、无网格线
        ax.tick_params(axis='x', color='m', length=4, direction='in', width=4,
                       labelcolor='g', grid_color='b')
```