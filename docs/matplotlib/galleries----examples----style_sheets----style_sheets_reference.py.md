# `D:\src\scipysrc\matplotlib\galleries\examples\style_sheets\style_sheets_reference.py`

```py
"""
======================
Style sheets reference
======================

This script demonstrates the different available style sheets on a
common set of example plots: scatter plot, image, bar graph, patches,
line plot and histogram.

Any of these style sheets can be imported (i.e. activated) by its name.
For example for the ggplot style:

>>> plt.style.use('ggplot')

The names of the available style sheets can be found
in the list `matplotlib.style.available`
(they are also printed in the corner of each plot below).

See more details in :ref:`Customizing Matplotlib
using style sheets<customizing-with-style-sheets>`.
"""

import matplotlib.pyplot as plt  # 导入 Matplotlib 的 pyplot 模块，并使用 plt 别名
import numpy as np  # 导入 NumPy 库，并使用 np 别名

import matplotlib.colors as mcolors  # 导入 Matplotlib 的 colors 模块，使用 mcolors 别名
from matplotlib.patches import Rectangle  # 从 Matplotlib 的 patches 模块导入 Rectangle 类

# Fixing random state for reproducibility
np.random.seed(19680801)  # 设定随机数种子以确保结果可复现


def plot_scatter(ax, prng, nb_samples=100):
    """Scatter plot."""
    for mu, sigma, marker in [(-.5, 0.75, 'o'), (0.75, 1., 's')]:
        x, y = prng.normal(loc=mu, scale=sigma, size=(2, nb_samples))
        ax.plot(x, y, ls='none', marker=marker)  # 绘制散点图
    ax.set_xlabel('X-label')  # 设置 X 轴标签
    ax.set_title('Axes title')  # 设置图表标题
    return ax


def plot_colored_lines(ax):
    """Plot lines with colors following the style color cycle."""
    t = np.linspace(-10, 10, 100)

    def sigmoid(t, t0):
        return 1 / (1 + np.exp(-(t - t0)))

    nb_colors = len(plt.rcParams['axes.prop_cycle'])
    shifts = np.linspace(-5, 5, nb_colors)
    amplitudes = np.linspace(1, 1.5, nb_colors)
    for t0, a in zip(shifts, amplitudes):
        ax.plot(t, a * sigmoid(t, t0), '-')  # 绘制带有颜色循环的线条
    ax.set_xlim(-10, 10)
    return ax


def plot_bar_graphs(ax, prng, min_value=5, max_value=25, nb_samples=5):
    """Plot two bar graphs side by side, with letters as x-tick labels."""
    x = np.arange(nb_samples)
    ya, yb = prng.randint(min_value, max_value, size=(2, nb_samples))
    width = 0.25
    ax.bar(x, ya, width)  # 绘制第一个柱状图
    ax.bar(x + width, yb, width, color='C2')  # 绘制第二个柱状图，并使用颜色 'C2'
    ax.set_xticks(x + width, labels=['a', 'b', 'c', 'd', 'e'])  # 设置 X 轴刻度及其标签
    return ax


def plot_colored_circles(ax, prng, nb_samples=15):
    """
    Plot circle patches.

    NB: draws a fixed amount of samples, rather than using the length of
    the color cycle, because different styles may have different numbers
    of colors.
    """
    for sty_dict, j in zip(plt.rcParams['axes.prop_cycle'](), range(nb_samples)):
        ax.add_patch(plt.Circle(prng.normal(scale=3, size=2), radius=1.0, color=sty_dict['color']))
    ax.grid(visible=True)  # 显示网格

    # Add title for enabling grid
    plt.title('ax.grid(True)', family='monospace', fontsize='small')  # 添加标题以启用网格显示

    ax.set_xlim([-4, 8])  # 设置 X 轴范围
    ax.set_ylim([-5, 6])  # 设置 Y 轴范围
    ax.set_aspect('equal', adjustable='box')  # 将绘图比例设置为等距，以确保圆形显示为圆形
    return ax


def plot_image_and_patch(ax, prng, size=(20, 20)):
    """Plot an image with random values and superimpose a circular patch."""
    values = prng.random_sample(size=size)
    # 在ax对象上显示二维数组values，禁用插值
    ax.imshow(values, interpolation='none')
    
    # 创建一个半径为5的圆形对象，并添加到ax对象中作为一个图形补丁，标签为'patch'
    c = plt.Circle((5, 5), radius=5, label='patch')
    ax.add_patch(c)
    
    # 移除x轴和y轴的刻度
    ax.set_xticks([])
    ax.set_yticks([])
def plot_histograms(ax, prng, nb_samples=10000):
    """Plot 4 histograms and a text annotation."""
    # 定义参数列表，每个参数是一个元组，包含两个数值
    params = ((10, 10), (4, 12), (50, 12), (6, 55))
    # 遍历参数列表，每次取出两个数值作为参数a和b
    for a, b in params:
        # 使用指定的参数a和b生成beta分布的随机数值
        values = prng.beta(a, b, size=nb_samples)
        # 绘制直方图，并使用stepfilled风格填充，设置bins为30
        # alpha为0.8表示部分透明，density=True表示绘制概率密度直方图
        ax.hist(values, histtype="stepfilled", bins=30,
                alpha=0.8, density=True)

    # 添加一个小注释
    ax.annotate('Annotation', xy=(0.25, 4.25),
                xytext=(0.9, 0.9), textcoords=ax.transAxes,
                va="top", ha="right",
                bbox=dict(boxstyle="round", alpha=0.2),
                arrowprops=dict(
                          arrowstyle="->",
                          connectionstyle="angle,angleA=-95,angleB=35,rad=10"),
                )
    # 返回绘制直方图后的轴对象
    return ax


def plot_figure(style_label=""):
    """Setup and plot the demonstration figure with a given style."""
    # 使用特定的RandomState实例来确保在不同的图形中绘制相同的“随机”值
    prng = np.random.RandomState(96917002)

    # 创建一个包含6个子图的图形对象，并指定大小和布局
    fig, axs = plt.subplots(ncols=6, nrows=1, num=style_label,
                            figsize=(14.8, 2.8), layout='constrained')

    # 根据背景颜色设置标题颜色
    background_color = mcolors.rgb_to_hsv(
        mcolors.to_rgb(plt.rcParams['figure.facecolor']))[2]
    if background_color < 0.5:
        title_color = [0.8, 0.8, 1]
    else:
        title_color = np.array([19, 6, 84]) / 256
    # 添加总标题，设置其位置、对齐方式、颜色和字体等属性
    fig.suptitle(style_label, x=0.01, ha='left', color=title_color,
                 fontsize=14, fontfamily='DejaVu Sans', fontweight='normal')

    # 分别调用各个子图的绘图函数
    plot_scatter(axs[0], prng)
    plot_image_and_patch(axs[1], prng)
    plot_bar_graphs(axs[2], prng)
    plot_colored_lines(axs[3])
    plot_histograms(axs[4], prng)  # 调用绘制直方图的函数
    plot_colored_circles(axs[5], prng)

    # 添加分隔线矩形对象到第4个子图上
    rec = Rectangle((1 + 0.025, -2), 0.05, 16,
                    clip_on=False, color='gray')
    axs[4].add_artist(rec)

if __name__ == "__main__":

    # 设置一个包含所有可用样式的列表，按字母顺序排列，
    # 但是'default'和'classic'样式会分别排在首位
    style_list = ['default', 'classic'] + sorted(
        style for style in plt.style.available
        if style != 'classic' and not style.startswith('_'))

    # 为每个可用的样式绘制一个演示图形
    for style_label in style_list:
        with plt.rc_context({"figure.max_open_warning": len(style_list)}):
            with plt.style.context(style_label):
                plot_figure(style_label=style_label)

    # 显示所有绘制的图形
    plt.show()
```