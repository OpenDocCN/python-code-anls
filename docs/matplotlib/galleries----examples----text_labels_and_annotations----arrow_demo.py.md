# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\arrow_demo.py`

```
"""
==========
Arrow Demo
==========

Three ways of drawing arrows to encode arrow "strength" (e.g., transition
probabilities in a Markov model) using arrow length, width, or alpha (opacity).
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np


def make_arrow_graph(ax, data, size=4, display='length', shape='right',
                     max_arrow_width=0.03, arrow_sep=0.02, alpha=0.5,
                     normalize_data=False, ec=None, labelcolor=None,
                     **kwargs):
    """
    Makes an arrow plot.

    Parameters
    ----------
    ax
        The Axes where the graph is drawn.
    data
        Dict with probabilities for the bases and pair transitions.
    size
        Size of the plot, in inches.
    display : {'length', 'width', 'alpha'}
        The arrow property to change.
    shape : {'full', 'left', 'right'}
        For full or half arrows.
    max_arrow_width : float
        Maximum width of an arrow, in data coordinates.
    arrow_sep : float
        Separation between arrows in a pair, in data coordinates.
    alpha : float
        Maximum opacity of arrows.
    **kwargs
        `.FancyArrow` properties, e.g. *linewidth* or *edgecolor*.
    """

    # 设置图形的坐标轴范围和标题
    ax.set(xlim=(-0.25, 1.25), ylim=(-0.25, 1.25), xticks=[], yticks=[],
           title=f'flux encoded as arrow {display}')

    # 设置文本显示的最大和最小尺寸以及标签文本的尺寸
    max_text_size = size * 12
    min_text_size = size
    label_text_size = size * 4

    # 定义碱基和它们在图中的坐标位置
    bases = 'ATGC'
    coords = {
        'A': np.array([0, 1]),
        'T': np.array([1, 1]),
        'G': np.array([0, 0]),
        'C': np.array([1, 0]),
    }
    # 碱基对应的颜色
    colors = {'A': 'r', 'T': 'k', 'G': 'g', 'C': 'b'}

    # 在图上绘制每个碱基的标签
    for base in bases:
        fontsize = np.clip(max_text_size * data[base]**(1/2),
                           min_text_size, max_text_size)
        ax.text(*coords[base], f'${base}_3$',
                color=colors[base], size=fontsize,
                horizontalalignment='center', verticalalignment='center',
                weight='bold')

    # 箭头在水平方向的偏移量（经验确定的数据坐标）
    arrow_h_offset = 0.25
    # 最大箭头长度，箭头头部的最大宽度和长度
    max_arrow_length = 1 - 2 * arrow_h_offset
    max_head_width = 2.5 * max_arrow_width
    max_head_length = 2 * max_arrow_width
    sf = 0.6  # 最大箭头大小在数据坐标中的比例因子

    # 如果需要归一化数据
    if normalize_data:
        # 找到所有长度为2的键中的最大值
        max_val = max((v for k, v in data.items() if len(k) == 2), default=0)
        # 将所有值除以最大值，并乘以箭头的比例因子
        for k, v in data.items():
            data[k] = v / max_val * sf

    # 迭代处理字符串 'AT', 'TA', 'AG', 'GA' 等
    for pair in map(''.join, itertools.permutations(bases, 2)):
        # 遍历由基元素bases的所有长度为2的排列组合组成的迭代器，每个pair是一个由两个字符组成的字符串
        
        # 设置箭头的长度
        if display == 'length':
            length = (max_head_length
                      + data[pair] / sf * (max_arrow_length - max_head_length))
        else:
            length = max_arrow_length
        # 根据数据的比例设置箭头的透明度
        if display == 'alpha':
            alpha = min(data[pair] / sf, alpha)
        # 根据数据的比例设置箭头的宽度和头部宽度与长度
        elif display == 'width':
            scale = data[pair] / sf
            width = max_arrow_width * scale
            head_width = max_head_width * scale
            head_length = max_head_length * scale
        else:
            width = max_arrow_width
            head_width = max_head_width
            head_length = max_head_length

        # 获取箭头的颜色
        fc = colors[pair[0]]

        # 获取箭头起点和终点的坐标
        cp0 = coords[pair[0]]
        cp1 = coords[pair[1]]
        
        # 计算箭头方向的单位向量
        delta = cos, sin = (cp1 - cp0) / np.hypot(*(cp1 - cp0))
        
        # 计算箭头的位置
        x_pos, y_pos = (
            (cp0 + cp1) / 2  # 箭头中点位置
            - delta * length / 2  # 箭头长度的一半
            + np.array([-sin, cos]) * arrow_sep  # 按箭头间隔向外移动
        )
        
        # 绘制箭头
        ax.arrow(
            x_pos, y_pos, cos * length, sin * length,
            fc=fc, ec=ec or fc, alpha=alpha, width=width,
            head_width=head_width, head_length=head_length, shape=shape,
            length_includes_head=True,
            **kwargs
        )

        # 计算文本标签的坐标位置
        orig_positions = {
            'base': [3 * max_arrow_width, 3 * max_arrow_width],
            'center': [length / 2, 3 * max_arrow_width],
            'tip': [length - 3 * max_arrow_width, 3 * max_arrow_width],
        }
        
        # 根据箭头的方向确定文本标签的位置
        where = 'base' if (cp0 != cp1).all() else 'center'
        
        # 根据箭头的方向（cos, sin）进行旋转
        M = [[cos, -sin], [sin, cos]]
        x, y = np.dot(M, orig_positions[where]) + [x_pos, y_pos]
        
        # 创建文本标签
        label = r'$r_{_{\mathrm{%s}}}$' % (pair,)
        ax.text(x, y, label, size=label_text_size, ha='center', va='center',
                color=labelcolor or fc)
if __name__ == '__main__':
    data = {  # test data
        'A': 0.4, 'T': 0.3, 'G': 0.6, 'C': 0.2,
        'AT': 0.4, 'AC': 0.3, 'AG': 0.2,
        'TA': 0.2, 'TC': 0.3, 'TG': 0.4,
        'CT': 0.2, 'CG': 0.3, 'CA': 0.2,
        'GA': 0.1, 'GT': 0.4, 'GC': 0.1,
    }  # 定义一个包含核酸碱基及其对应概率的测试数据字典

    size = 4  # 设置图形的大小因子
    fig = plt.figure(figsize=(3 * size, size), layout="constrained")  # 创建一个指定大小和布局约束的图形对象
    axs = fig.subplot_mosaic([["length", "width", "alpha"]])  # 在图形中创建一个包含三个子图的布局，分别命名为'length'、'width'和'alpha'

    for display, ax in axs.items():
        make_arrow_graph(
            ax, data, display=display, linewidth=0.001, edgecolor=None,
            normalize_data=True, size=size)
        # 在每个子图中调用 make_arrow_graph 函数，传入相应的参数进行图形绘制

    plt.show()  # 显示生成的图形
```