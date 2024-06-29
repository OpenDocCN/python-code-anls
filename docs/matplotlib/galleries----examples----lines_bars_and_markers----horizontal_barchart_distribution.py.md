# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\horizontal_barchart_distribution.py`

```
"""
=============================================
Discrete distribution as horizontal bar chart
=============================================

Stacked bar charts can be used to visualize discrete distributions.

This example visualizes the result of a survey in which people could rate
their agreement to questions on a five-element scale.

The horizontal stacking is achieved by calling `~.Axes.barh()` for each
category and passing the starting point as the cumulative sum of the
already drawn bars via the parameter ``left``.
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于处理数据
import numpy as np

# 定义类别名称列表
category_names = ['Strongly disagree', 'Disagree',
                  'Neither agree nor disagree', 'Agree', 'Strongly agree']
# 定义包含调查结果的字典
results = {
    'Question 1': [10, 15, 17, 32, 26],
    'Question 2': [26, 22, 29, 10, 13],
    'Question 3': [35, 37, 7, 2, 19],
    'Question 4': [32, 11, 9, 15, 33],
    'Question 5': [21, 29, 5, 5, 40],
    'Question 6': [8, 19, 5, 30, 38]
}

# 定义调查函数，用于绘制水平条形图
def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    # 获取问题标签列表
    labels = list(results.keys())
    # 将调查结果转换为 numpy 数组
    data = np.array(list(results.values()))
    # 计算累积数据
    data_cum = data.cumsum(axis=1)
    # 根据数据的列数生成颜色映射
    category_colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, data.shape[1]))

    # 创建图形和坐标轴对象
    fig, ax = plt.subplots(figsize=(9.2, 5))
    # 反转 y 轴以确保顶部是最先显示的数据
    ax.invert_yaxis()
    # 隐藏 x 轴
    ax.xaxis.set_visible(False)
    # 设置 x 轴的范围
    ax.set_xlim(0, np.sum(data, axis=1).max())

    # 遍历类别名称和对应的颜色，绘制水平条形图
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        # 根据条形的颜色判断文本颜色，确保可读性
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # 添加条形图上的文本标签
        ax.bar_label(rects, label_type='center', color=text_color)
    
    # 添加图例，并根据类别数量调整布局
    ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    # 返回图形和坐标轴对象
    return fig, ax


# 调用 survey 函数，生成并展示水平条形图
survey(results, category_names)
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.barh` / `matplotlib.pyplot.barh`
#    - `matplotlib.axes.Axes.bar_label` / `matplotlib.pyplot.bar_label`
#    - `matplotlib.axes.Axes.legend` / `matplotlib.pyplot.legend`
```