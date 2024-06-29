# `D:\src\scipysrc\matplotlib\lib\matplotlib\tri\_triplot.py`

```py
import numpy as np  # 导入 NumPy 库，用于数值计算
from matplotlib.tri._triangulation import Triangulation  # 导入 Triangulation 类，用于处理三角网格
import matplotlib.cbook as cbook  # 导入 matplotlib 的 cbook 模块，用于杂项功能
import matplotlib.lines as mlines  # 导入 matplotlib 的 lines 模块，用于绘制线条

def triplot(ax, *args, **kwargs):
    """
    Draw an unstructured triangular grid as lines and/or markers.

    Call signatures::

      triplot(triangulation, ...)
      triplot(x, y, [triangles], *, [mask=mask], ...)

    The triangular grid can be specified either by passing a `.Triangulation`
    object as the first parameter, or by passing the points *x*, *y* and
    optionally the *triangles* and a *mask*. If neither of *triangulation* or
    *triangles* are given, the triangulation is calculated on the fly.

    Parameters
    ----------
    triangulation : `.Triangulation`
        An already created triangular grid.
    x, y, triangles, mask
        Parameters defining the triangular grid. See `.Triangulation`.
        This is mutually exclusive with specifying *triangulation*.
    other_parameters
        All other args and kwargs are forwarded to `~.Axes.plot`.

    Returns
    -------
    lines : `~matplotlib.lines.Line2D`
        The drawn triangles edges.
    markers : `~matplotlib.lines.Line2D`
        The drawn marker nodes.
    """
    import matplotlib.axes  # 导入 matplotlib 的 axes 模块

    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)  # 解析参数得到 Triangulation 对象及其余的参数
    x, y, edges = (tri.x, tri.y, tri.edges)  # 从 Triangulation 对象获取 x 坐标、y 坐标和边界信息

    # Decode plot format string, e.g., 'ro-'
    fmt = args[0] if args else ""  # 获取绘图格式字符串
    linestyle, marker, color = matplotlib.axes._base._process_plot_format(fmt)  # 解析绘图格式字符串

    # Insert plot format string into a copy of kwargs (kwargs values prevail).
    kw = cbook.normalize_kwargs(kwargs, mlines.Line2D)  # 根据 Line2D 的参数规范化 kwargs
    for key, val in zip(('linestyle', 'marker', 'color'),
                        (linestyle, marker, color)):
        if val is not None:
            kw.setdefault(key, val)  # 将解析得到的线型、标记和颜色设置到 kw 中

    # Draw lines without markers.
    # Note 1: If we drew markers here, most markers would be drawn more than
    #         once as they belong to several edges.
    # Note 2: We insert nan values in the flattened edges arrays rather than
    #         plotting directly (triang.x[edges].T, triang.y[edges].T)
    #         as it considerably speeds-up code execution.
    linestyle = kw['linestyle']  # 获取线型风格
    kw_lines = {
        **kw,
        'marker': 'None',  # 不绘制标记
        'zorder': kw.get('zorder', 1),  # 使用路径默认的 zorder
    }
    if linestyle not in [None, 'None', '', ' ']:
        tri_lines_x = np.insert(x[edges], 2, np.nan, axis=1)  # 在边界数组中插入 NaN 值，加速绘图
        tri_lines_y = np.insert(y[edges], 2, np.nan, axis=1)  # 在边界数组中插入 NaN 值，加速绘图
        tri_lines = ax.plot(tri_lines_x.ravel(), tri_lines_y.ravel(),
                            **kw_lines)  # 绘制三角形的边界线条
    else:
        tri_lines = ax.plot([], [], **kw_lines)  # 如果没有指定线型，则不绘制任何内容

    # Draw markers separately.
    marker = kw['marker']  # 获取标记风格
    kw_markers = {
        **kw,
        'linestyle': 'None',  # 不绘制线条
    }
    kw_markers.pop('label', None)  # 移除标签信息
    # 如果标记不在指定的几种可能的值中（None, 'None', '', ' '），则绘制包含数据点的图形对象
    if marker not in [None, 'None', '', ' ']:
        # 使用给定的 x 和 y 值以及附加的标记样式参数绘制数据点图形
        tri_markers = ax.plot(x, y, **kw_markers)
    else:
        # 如果标记是空或未定义，则创建一个空的图形对象
        tri_markers = ax.plot([], [], **kw_markers)
    
    # 返回包含线条图形对象和数据点图形对象的列表
    return tri_lines + tri_markers
```