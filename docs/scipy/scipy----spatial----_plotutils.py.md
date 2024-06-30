# `D:\src\scipysrc\scipy\scipy\spatial\_plotutils.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from scipy._lib.decorator import decorator as _decorator  # 导入装饰器

__all__ = ['delaunay_plot_2d', 'convex_hull_plot_2d', 'voronoi_plot_2d']  # 公开的函数列表

@_decorator
def _held_figure(func, obj, ax=None, **kw):
    import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图

    if ax is None:
        fig = plt.figure()  # 如果未提供轴对象，则创建新的图形
        ax = fig.gca()  # 获取当前图形的坐标轴
        return func(obj, ax=ax, **kw)  # 调用被装饰的函数并传递坐标轴参数

    # 从 Matplotlib 2.0 开始，"hold" 机制已被弃用。
    # 当不再支持 Matplotlib 1.x 时，可以移除此检查。
    was_held = getattr(ax, 'ishold', lambda: True)()  # 检查轴是否被 hold
    if was_held:
        return func(obj, ax=ax, **kw)  # 如果轴之前被 hold，则直接调用函数
    try:
        ax.hold(True)  # 设置轴为 hold 状态
        return func(obj, ax=ax, **kw)  # 调用函数
    finally:
        ax.hold(was_held)  # 恢复轴的 hold 状态

def _adjust_bounds(ax, points):
    margin = 0.1 * np.ptp(points, axis=0)  # 计算点集的范围，并计算边界调整的边距
    xy_min = points.min(axis=0) - margin  # 计算坐标的最小边界
    xy_max = points.max(axis=0) + margin  # 计算坐标的最大边界
    ax.set_xlim(xy_min[0], xy_max[0])  # 设置 x 轴的范围
    ax.set_ylim(xy_min[1], xy_max[1])  # 设置 y 轴的范围

@_held_figure
def delaunay_plot_2d(tri, ax=None):
    """
    在二维中绘制给定的 Delaunay 三角剖分图

    Parameters
    ----------
    tri : scipy.spatial.Delaunay instance
        要绘制的三角剖分
    ax : matplotlib.axes.Axes instance, optional
        要绘制在其上的坐标轴

    Returns
    -------
    fig : matplotlib.figure.Figure instance
        绘图的图形对象

    See Also
    --------
    Delaunay
    matplotlib.pyplot.triplot

    Notes
    -----
    需要 Matplotlib 库。

    Examples
    --------

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.spatial import Delaunay, delaunay_plot_2d

    一组随机点的 Delaunay 三角剖分：

    >>> rng = np.random.default_rng()
    >>> points = rng.random((30, 2))
    >>> tri = Delaunay(points)

    绘制它：

    >>> _ = delaunay_plot_2d(tri)
    >>> plt.show()

    """
    if tri.points.shape[1] != 2:
        raise ValueError("Delaunay 三角剖分不是二维的")

    x, y = tri.points.T  # 提取点的 x 和 y 坐标
    ax.plot(x, y, 'o')  # 在坐标轴上绘制点
    ax.triplot(x, y, tri.simplices.copy())  # 绘制三角形

    _adjust_bounds(ax, tri.points)  # 调整坐标轴范围

    return ax.figure  # 返回图形对象

@_held_figure
def convex_hull_plot_2d(hull, ax=None):
    """
    在二维中绘制给定的凸包图

    Parameters
    ----------
    hull : scipy.spatial.ConvexHull instance
        要绘制的凸包
    ax : matplotlib.axes.Axes instance, optional
        要绘制在其上的坐标轴

    Returns
    -------
    fig : matplotlib.figure.Figure instance
        绘图的图形对象

    See Also
    --------
    ConvexHull

    Notes
    -----
    需要 Matplotlib 库。

    Examples
    --------

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.spatial import ConvexHull, convex_hull_plot_2d

    随机点集的凸包：

    >>> rng = np.random.default_rng()
    >>> points = rng.random((30, 2))
    >>> hull = ConvexHull(points)

    绘制它：

    >>> _ = convex_hull_plot_2d(hull)
    >>> plt.show()

    """
    # 导入 LineCollection 类，用于绘制线段集合
    from matplotlib.collections import LineCollection
    
    # 检查凸包点集的维度是否为二维，如果不是则引发 ValueError 异常
    if hull.points.shape[1] != 2:
        raise ValueError("Convex hull is not 2-D")
    
    # 在图形中绘制凸包的所有点，用圆圈表示
    ax.plot(hull.points[:, 0], hull.points[:, 1], 'o')
    
    # 创建线段集合，每个线段由凸包的一个面（simplex）的顶点组成
    line_segments = [hull.points[simplex] for simplex in hull.simplices]
    
    # 将线段集合添加到图形中，使用黑色实线表示
    ax.add_collection(LineCollection(line_segments,
                                     colors='k',
                                     linestyle='solid'))
    
    # 调整图形的边界，确保凸包的所有点都可见
    _adjust_bounds(ax, hull.points)
    
    # 返回图形对象所在的图形 Figure 对象
    return ax.figure
# 使用装饰器 `_held_figure` 来修饰下面的函数，可能用于管理图形显示相关的功能
@_held_figure
def voronoi_plot_2d(vor, ax=None, **kw):
    """
    Plot the given Voronoi diagram in 2-D

    Parameters
    ----------
    vor : scipy.spatial.Voronoi instance
        Diagram to plot
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on
    show_points : bool, optional
        Add the Voronoi points to the plot.
    show_vertices : bool, optional
        Add the Voronoi vertices to the plot.
    line_colors : string, optional
        Specifies the line color for polygon boundaries
    line_width : float, optional
        Specifies the line width for polygon boundaries
    line_alpha : float, optional
        Specifies the line alpha for polygon boundaries
    point_size : float, optional
        Specifies the size of points

    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot

    See Also
    --------
    Voronoi

    Notes
    -----
    Requires Matplotlib.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.spatial import Voronoi, voronoi_plot_2d

    Create a set of points for the example:

    >>> rng = np.random.default_rng()
    >>> points = rng.random((10,2))

    Generate the Voronoi diagram for the points:

    >>> vor = Voronoi(points)

    Use `voronoi_plot_2d` to plot the diagram:

    >>> fig = voronoi_plot_2d(vor)

    Use `voronoi_plot_2d` to plot the diagram again, with some settings
    customized:

    >>> fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
    ...                       line_width=2, line_alpha=0.6, point_size=2)
    >>> plt.show()

    """
    # 导入必要的库 - matplotlib 中的 LineCollection 类
    from matplotlib.collections import LineCollection

    # 如果 Voronoi 图不是二维的，则抛出异常
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    # 如果设置中包括显示点（默认显示），则根据设置决定点的大小并绘制
    if kw.get('show_points', True):
        point_size = kw.get('point_size', None)
        ax.plot(vor.points[:, 0], vor.points[:, 1], '.', markersize=point_size)
    
    # 如果设置中包括显示顶点（默认显示），则绘制 Voronoi 图的顶点
    if kw.get('show_vertices', True):
        ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'o')

    # 从设置中获取线的颜色，默认为黑色
    line_colors = kw.get('line_colors', 'k')
    # 从设置中获取线的宽度，默认为 1.0
    line_width = kw.get('line_width', 1.0)
    # 从设置中获取线的透明度，默认为 1.0
    line_alpha = kw.get('line_alpha', 1.0)

    # 计算 Voronoi 图的中心点坐标
    center = vor.points.mean(axis=0)
    # 计算 Voronoi 图点集的极差
    ptp_bound = np.ptp(vor.points, axis=0)

    # 初始化有限线段和无限线段列表
    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        # 遍历 Voronoi 图的 ridge points 和 ridge vertices
        simplex = np.asarray(simplex)
        # 将简单形状转换为 NumPy 数组

        if np.all(simplex >= 0):
            # 如果所有顶点都是有限的（即没有 -1 的情况）
            finite_segments.append(vor.vertices[simplex])
            # 将有限的线段添加到 finite_segments 中
        else:
            # 如果存在 -1，表示至少一个顶点是无限的 Voronoi 顶点
            i = simplex[simplex >= 0][0]  # 找到第一个有限端的 Voronoi 顶点索引

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # 计算切线向量
            t /= np.linalg.norm(t)  # 将切线向量单位化
            n = np.array([-t[1], t[0]])  # 计算法向量

            midpoint = vor.points[pointidx].mean(axis=0)  # 计算简单形状的中点
            direction = np.sign(np.dot(midpoint - center, n)) * n  # 计算方向向量
            if (vor.furthest_site):
                direction = -direction  # 如果是最远点，反转方向向量
            aspect_factor = abs(ptp_bound.max() / ptp_bound.min())  # 计算宽高比因子
            far_point = vor.vertices[i] + direction * ptp_bound.max() * aspect_factor
            # 计算无限线段的远端点位置

            infinite_segments.append([vor.vertices[i], far_point])
            # 将无限线段添加到 infinite_segments 中

    ax.add_collection(LineCollection(finite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='solid'))
    # 将有限线段集合添加到图形的 axes 中，用于绘制实线

    ax.add_collection(LineCollection(infinite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='dashed'))
    # 将无限线段集合添加到图形的 axes 中，用于绘制虚线

    _adjust_bounds(ax, vor.points)
    # 调整图形的边界，根据 Voronoi 图的点集

    return ax.figure
    # 返回包含 Voronoi 图绘制结果的图形对象
```