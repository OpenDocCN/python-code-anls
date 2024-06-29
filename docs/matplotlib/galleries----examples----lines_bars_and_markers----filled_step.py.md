# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\filled_step.py`

```
"""
=========================
Hatch-filled histograms
=========================

Hatching capabilities for plotting histograms.
"""

# 导入必要的库
from functools import partial  # 导入 partial 函数，用于创建带有预设参数的新函数
import itertools  # 导入 itertools 库，用于创建迭代器的函数

from cycler import cycler  # 从 cycler 库导入 cycler 类，用于指定颜色和样式的循环

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

import matplotlib.ticker as mticker  # 导入 matplotlib 的 ticker 模块，用于刻度设置


def filled_hist(ax, edges, values, bottoms=None, orientation='v',
                **kwargs):
    """
    Draw a histogram as a stepped patch.

    Parameters
    ----------
    ax : Axes
        The Axes to plot to.

    edges : array
        A length n+1 array giving the left edges of each bin and the
        right edge of the last bin.

    values : array
        A length n array of bin counts or values

    bottoms : float or array, optional
        A length n array of the bottom of the bars.  If None, zero is used.

    orientation : {'v', 'h'}
       Orientation of the histogram.  'v' (default) has
       the bars increasing in the positive y-direction.

    **kwargs
        Extra keyword arguments are passed through to `.fill_between`.

    Returns
    -------
    ret : PolyCollection
        Artist added to the Axes
    """
    print(orientation)  # 打印直方图的方向（垂直或水平）
    if orientation not in 'hv':
        raise ValueError(f"orientation must be in {{'h', 'v'}} "
                         f"not {orientation}")  # 如果方向不是 'h' 或 'v'，则抛出错误

    kwargs.setdefault('step', 'post')  # 设置关键字参数 'step' 的默认值为 'post'
    kwargs.setdefault('alpha', 0.7)  # 设置关键字参数 'alpha' 的默认值为 0.7
    edges = np.asarray(edges)  # 将边界转换为 NumPy 数组
    values = np.asarray(values)  # 将值转换为 NumPy 数组
    if len(edges) - 1 != len(values):
        raise ValueError(f'Must provide one more bin edge than value not: '
                         f'{len(edges)=} {len(values)=}')  # 如果边界的数量比值的数量少1，则抛出错误

    if bottoms is None:
        bottoms = 0  # 如果底部值为 None，则设置为 0
    bottoms = np.broadcast_to(bottoms, values.shape)  # 将底部值广播为与值数组相同的形状

    values = np.append(values, values[-1])  # 在值数组末尾添加最后一个值，用于绘制闭合图形
    bottoms = np.append(bottoms, bottoms[-1])  # 在底部数组末尾添加最后一个底部值，用于绘制闭合图形
    if orientation == 'h':
        return ax.fill_betweenx(edges, values, bottoms,
                                **kwargs)  # 如果方向为水平，则使用 fill_betweenx 方法绘制直方图
    elif orientation == 'v':
        return ax.fill_between(edges, values, bottoms,
                               **kwargs)  # 如果方向为垂直，则使用 fill_between 方法绘制直方图
    else:
        raise AssertionError("you should never be here")  # 如果出现不可能的情况，则引发断言错误


def stack_hist(ax, stacked_data, sty_cycle, bottoms=None,
               hist_func=None, labels=None,
               plot_func=None, plot_kwargs=None):
    """
    Parameters
    ----------
    ax : axes.Axes
        The Axes to add artists to.

    stacked_data : array or Mapping
        A (M, N) shaped array.  The first dimension will be iterated over to
        compute histograms row-wise

    sty_cycle : Cycler or operable of dict
        Style to apply to each set

    bottoms : array, default: 0
        The initial positions of the bottoms.

    hist_func : callable, optional
        Must have signature `bin_vals, bin_edges = f(data)`.
        `bin_edges` expected to be one longer than `bin_vals`

    """
    # 这里是 stack_hist 函数的参数说明和文档字符串，详细描述了每个参数的作用
    # deal with default binning function
    if hist_func is None:
        hist_func = np.histogram

    # deal with default plotting function
    if plot_func is None:
        plot_func = filled_hist

    # deal with default plot keyword arguments
    if plot_kwargs is None:
        plot_kwargs = {}
    # 打印当前的绘图关键字参数
    print(plot_kwargs)
    try:
        # 尝试获取 stacked_data 的键集合
        l_keys = stacked_data.keys()
        label_data = True
        # 如果 labels 未指定，则默认使用 l_keys 中的键作为标签
        if labels is None:
            labels = l_keys

    except AttributeError:
        # 如果 stacked_data 不是映射类型，则无法获取其键集合
        label_data = False
        # 如果 labels 未指定，使用 itertools.repeat(None) 生成一个迭代器
        if labels is None:
            labels = itertools.repeat(None)

    if label_data:
        # 如果 stacked_data 是映射类型，则使用 labels 和 sty_cycle 迭代
        loop_iter = enumerate((stacked_data[lab], lab, s)
                              for lab, s in zip(labels, sty_cycle))
    else:
        # 如果 stacked_data 不是映射类型，则直接使用 stacked_data 和 labels 迭代
        loop_iter = enumerate(zip(stacked_data, labels, sty_cycle))

    # 初始化艺术品字典
    arts = {}
    for j, (data, label, sty) in loop_iter:
        if label is None:
            # 如果标签未指定，则使用默认标签格式
            label = f'dflt set {j}'
        # 从样式中弹出 'label' 键，并将其作为标签
        label = sty.pop('label', label)
        # 计算数据的直方图值和边界
        vals, edges = hist_func(data)
        if bottoms is None:
            # 如果底部未指定，则初始化为与 vals 相同形状的零数组
            bottoms = np.zeros_like(vals)
        # 计算柱状图的顶部位置
        top = bottoms + vals
        # 打印当前的样式参数
        print(sty)
        # 更新样式参数，传递给绘图函数
        sty.update(plot_kwargs)
        # 打印更新后的样式参数
        print(sty)
        # 调用绘图函数绘制直方图，并返回绘制的艺术品对象
        ret = plot_func(ax, edges, top, bottoms=bottoms,
                        label=label, **sty)
        # 更新底部位置为当前柱状图的顶部位置
        bottoms = top
        # 将绘制的艺术品对象存入艺术品字典中，以标签为键
        arts[label] = ret
    # 在图形上添加图例，设置字体大小为10
    ax.legend(fontsize=10)
    # 返回艺术品字典
    return arts
# 设置直方图函数，使用固定的边界
edges = np.linspace(-3, 3, 20, endpoint=True)
hist_func = partial(np.histogram, bins=edges)

# 设置样式循环
color_cycle = cycler(facecolor=plt.rcParams['axes.prop_cycle'][:4])
label_cycle = cycler(label=[f'set {n}' for n in range(4)])
hatch_cycle = cycler(hatch=['/', '*', '+', '|'])

# 为了可重现性设置随机种子
np.random.seed(19680801)

# 生成随机数据矩阵
stack_data = np.random.randn(4, 12250)
# 创建标签数据字典
dict_data = dict(zip((c['label'] for c in label_cycle), stack_data))

# %%
# 使用普通数组进行处理

# 创建包含两个子图的图形对象
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
# 在第一个子图上绘制堆叠直方图
arts = stack_hist(ax1, stack_data, color_cycle + label_cycle + hatch_cycle,
                  hist_func=hist_func)
# 在第二个子图上绘制堆叠直方图，并设定额外的绘图参数
arts = stack_hist(ax2, stack_data, color_cycle,
                  hist_func=hist_func,
                  plot_kwargs=dict(edgecolor='w', orientation='h'))
# 设置第一个子图的标签和坐标轴标签
ax1.set_ylabel('counts')
ax1.set_xlabel('x')
# 设置第二个子图的坐标轴标签
ax2.set_xlabel('counts')
ax2.set_ylabel('x')

# %%
# 使用带标签的数据进行处理

# 创建包含两个子图的图形对象，共享y轴
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5),
                               tight_layout=True, sharey=True)
# 在第一个子图上绘制堆叠直方图，使用颜色和填充样式循环
arts = stack_hist(ax1, dict_data, color_cycle + hatch_cycle,
                  hist_func=hist_func)
# 在第二个子图上绘制堆叠直方图，使用颜色和填充样式循环，并设定标签
arts = stack_hist(ax2, dict_data, color_cycle + hatch_cycle,
                  hist_func=hist_func, labels=['set 0', 'set 3'])
# 设置第一个子图的x轴主刻度定位器和坐标轴标签
ax1.xaxis.set_major_locator(mticker.MaxNLocator(5))
ax1.set_xlabel('counts')
ax1.set_ylabel('x')
# 设置第二个子图的y轴标签
ax2.set_ylabel('x')

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.fill_betweenx` / `matplotlib.pyplot.fill_betweenx`
#    - `matplotlib.axes.Axes.fill_between` / `matplotlib.pyplot.fill_between`
#    - `matplotlib.axis.Axis.set_major_locator`
```