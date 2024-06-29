# `D:\src\scipysrc\matplotlib\galleries\examples\specialty_plots\ishikawa_diagram.py`

```py
"""
================
Ishikawa Diagram
================

Ishikawa Diagrams, fishbone diagrams, herringbone diagrams, or cause-and-effect
diagrams are used to identify problems in a system by showing how causes and
effects are linked.
Source: https://en.wikipedia.org/wiki/Ishikawa_diagram

"""
import math  # 导入数学库

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块

from matplotlib.patches import Polygon, Wedge  # 从 matplotlib.patches 模块导入 Polygon 和 Wedge 类

fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')  # 创建一个大小为 10x6 的子图，并进行约束布局
ax.set_xlim(-5, 5)  # 设置 X 轴的显示范围为 -5 到 5
ax.set_ylim(-5, 5)  # 设置 Y 轴的显示范围为 -5 到 5
ax.axis('off')  # 关闭坐标轴显示


def problems(data: str,
             problem_x: float, problem_y: float,
             angle_x: float, angle_y: float):
    """
    Draw each problem section of the Ishikawa plot.

    Parameters
    ----------
    data : str
        The name of the problem category.
    problem_x, problem_y : float, optional
        The `X` and `Y` positions of the problem arrows (`Y` defaults to zero).
    angle_x, angle_y : float, optional
        The angle of the problem annotations. They are always angled towards
        the tail of the plot.

    Returns
    -------
    None.

    """
    ax.annotate(str.upper(data), xy=(problem_x, problem_y),
                xytext=(angle_x, angle_y),
                fontsize=10,
                color='white',
                weight='bold',
                xycoords='data',
                verticalalignment='center',
                horizontalalignment='center',
                textcoords='offset fontsize',
                arrowprops=dict(arrowstyle="->", facecolor='black'),
                bbox=dict(boxstyle='square',
                          facecolor='tab:blue',
                          pad=0.8))


def causes(data: list,
           cause_x: float, cause_y: float,
           cause_xytext=(-9, -0.3), top: bool = True):
    """
    Place each cause to a position relative to the problems
    annotations.

    Parameters
    ----------
    data : indexable object
        The input data. IndexError is
        raised if more than six arguments are passed.
    cause_x, cause_y : float
        The `X` and `Y` position of the cause annotations.
    cause_xytext : tuple, optional
        Adjust to set the distance of the cause text from the problem
        arrow in fontsize units.
    top : bool, default: True
        Determines whether the next cause annotation will be
        plotted above or below the previous one.

    Returns
    -------
    None.

    """
    for index, cause in enumerate(data):
        # 枚举数据列表中的索引和原因内容

        # [<x pos>, <y pos>]
        # 定义坐标偏移列表，用于控制注释位置
        coords = [[0.02, 0],
                  [0.23, 0.5],
                  [-0.46, -1],
                  [0.69, 1.5],
                  [-0.92, -2],
                  [1.15, 2.5]]

        # 第一个原因的注释位于“问题”箭头的中间，
        # 每个后续原因按顺序在其上方或下方绘制
        cause_x -= coords[index][0]
        # 如果top为真，则在cause_y上加上coords[index][1]，否则减去
        cause_y += coords[index][1] if top else -coords[index][1]

        # 在图形上进行注释，标注原因文本
        ax.annotate(cause, xy=(cause_x, cause_y),
                    horizontalalignment='center',
                    xytext=cause_xytext,
                    fontsize=9,
                    xycoords='data',
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    facecolor='black'))
# 定义绘制主体函数，用于根据数据绘制图表中的主体部分
def draw_body(data: dict):
    """
    Place each problem section in its correct place by changing
    the coordinates on each loop.

    Parameters
    ----------
    data : dict
        The input data (can be a dict of lists or tuples). ValueError
        is raised if more than six arguments are passed.

    Returns
    -------
    None.

    """
    # 根据 'problem' 分类的数量设置脊柱的长度
    length = (math.ceil(len(data) / 2)) - 1
    # 调用绘制脊柱的函数，确定其起始和结束位置
    draw_spine(-2 - length, 2 + length)

    # 在每个问题部分渲染后，根据需要更改 'problem' 注释的坐标
    offset = 0  # 初始化偏移量
    prob_section = [1.55, 0.8]  # 初始 'problem' 注释的坐标位置
    for index, problem in enumerate(data.values()):
        plot_above = index % 2 == 0  # 判断是否位于偶数位置，用于决定 'cause' 箭头的位置
        cause_arrow_y = 1.7 if plot_above else -1.7  # 'cause' 箭头的纵坐标
        y_prob_angle = 16 if plot_above else -16  # 'problem' 注释的倾斜角度

        # 将每个 'problem' 部分成对绘制在主脊柱上
        prob_arrow_x = prob_section[0] + length + offset  # 计算 'problem' 注释的横坐标
        cause_arrow_x = prob_section[1] + length + offset  # 计算 'cause' 箭头的横坐标
        if not plot_above:
            offset -= 2.5  # 更新偏移量以适应下一个 'problem' 注释的位置
        if index > 5:
            raise ValueError(f'Maximum number of problems is 6, you have entered '
                             f'{len(data)}')

        # 调用问题绘制函数，传入问题名称、横坐标、固定纵坐标、旋转角度和倾斜角度
        problems(list(data.keys())[index], prob_arrow_x, 0, -12, y_prob_angle)
        # 调用原因绘制函数，传入原因名称、横坐标、纵坐标、是否位于上方的标记
        causes(problem, cause_arrow_x, cause_arrow_y, top=plot_above)


def draw_spine(xmin: int, xmax: int):
    """
    Draw main spine, head and tail.

    Parameters
    ----------
    xmin : int
        The default position of the head of the spine's
        x-coordinate.
    xmax : int
        The default position of the tail of the spine's
        x-coordinate.

    Returns
    -------
    None.

    """
    # 绘制主脊柱
    ax.plot([xmin - 0.1, xmax], [0, 0], color='tab:blue', linewidth=2)
    # 绘制鱼的头部
    ax.text(xmax + 0.1, - 0.05, 'PROBLEM', fontsize=10,
            weight='bold', color='white')
    semicircle = Wedge((xmax, 0), 1, 270, 90, fc='tab:blue')
    ax.add_patch(semicircle)
    # 绘制鱼的尾部
    tail_pos = [[xmin - 0.8, 0.8], [xmin - 0.8, -0.8], [xmin, -0.01]]
    triangle = Polygon(tail_pos, fc='tab:blue')
    ax.add_patch(triangle)
```