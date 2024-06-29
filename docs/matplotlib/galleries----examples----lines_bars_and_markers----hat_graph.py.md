# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\hat_graph.py`

```py
"""
=========
Hat graph
=========
This example shows how to create a `hat graph`_ and how to annotate it with
labels.

.. _hat graph: https://doi.org/10.1186/s41235-019-0182-3
"""
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import numpy as np  # 导入NumPy数值计算库


def hat_graph(ax, xlabels, values, group_labels):
    """
    Create a hat graph.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes to plot into.
    xlabels : list of str
        The category names to be displayed on the x-axis.
    values : (M, N) array-like
        The data values.
        Rows are the groups (len(group_labels) == M).
        Columns are the categories (len(xlabels) == N).
    group_labels : list of str
        The group labels displayed in the legend.
    """

    def label_bars(heights, rects):
        """Attach a text label on top of each bar."""
        for height, rect in zip(heights, rects):
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset.
                        textcoords='offset points',
                        ha='center', va='bottom')

    values = np.asarray(values)  # 将数据转换为NumPy数组
    x = np.arange(values.shape[1])  # 在x轴上创建等间隔的标签位置
    ax.set_xticks(x, labels=xlabels)  # 设置x轴的刻度和标签
    spacing = 0.3  # 定义帽子图组之间的间距
    width = (1 - spacing) / values.shape[0]  # 计算每个组的宽度
    heights0 = values[0]  # 获取第一个组的高度作为基准
    for i, (heights, group_label) in enumerate(zip(values, group_labels)):
        style = {'fill': False} if i == 0 else {'edgecolor': 'black'}
        rects = ax.bar(x - spacing/2 + i * width, heights - heights0,
                       width, bottom=heights0, label=group_label, **style)
        label_bars(heights, rects)


# initialise labels and a numpy array make sure you have
# N labels of N number of values in the array
xlabels = ['I', 'II', 'III', 'IV', 'V']  # x轴上的标签
playerA = np.array([5, 15, 22, 20, 25])  # 第一个玩家的分数数组
playerB = np.array([25, 32, 34, 30, 27])  # 第二个玩家的分数数组

fig, ax = plt.subplots()  # 创建一个新的图形和轴

# 调用hat_graph函数绘制帽子图
hat_graph(ax, xlabels, [playerA, playerB], ['Player A', 'Player B'])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Games')  # 设置x轴标签
ax.set_ylabel('Score')  # 设置y轴标签
ax.set_ylim(0, 60)  # 设置y轴范围
ax.set_title('Scores by number of game and players')  # 设置图表标题
ax.legend()  # 显示图例

fig.tight_layout()  # 调整布局使得所有内容适应图形区域
plt.show()  # 显示图形
# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.bar` / `matplotlib.pyplot.bar`
#    - `matplotlib.axes.Axes.annotate` / `matplotlib.pyplot.annotate`
```