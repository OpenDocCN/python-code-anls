# `D:\src\scipysrc\matplotlib\lib\matplotlib\tri\_tripcolor.py`

```
    # 导入NumPy库，通常用于数值计算
    import numpy as np

    # 导入Matplotlib的私有API模块和必要的绘图类
    from matplotlib import _api
    from matplotlib.collections import PolyCollection, TriMesh
    from matplotlib.tri._triangulation import Triangulation

    # 定义函数tripcolor，用于绘制无结构三角网格的伪彩色图
    def tripcolor(ax, *args, alpha=1.0, norm=None, cmap=None, vmin=None,
                  vmax=None, shading='flat', facecolors=None, **kwargs):
        """
        Create a pseudocolor plot of an unstructured triangular grid.

        Call signatures::

          tripcolor(triangulation, c, *, ...)
          tripcolor(x, y, c, *, [triangles=triangles], [mask=mask], ...)

        The triangular grid can be specified either by passing a `.Triangulation`
        object as the first parameter, or by passing the points *x*, *y* and
        optionally the *triangles* and a *mask*. See `.Triangulation` for an
        explanation of these parameters.

        It is possible to pass the triangles positionally, i.e.
        ``tripcolor(x, y, triangles, c, ...)``. However, this is discouraged.
        For more clarity, pass *triangles* via keyword argument.

        If neither of *triangulation* or *triangles* are given, the triangulation
        is calculated on the fly. In this case, it does not make sense to provide
        colors at the triangle faces via *c* or *facecolors* because there are
        multiple possible triangulations for a group of points and you don't know
        which triangles will be constructed.

        Parameters
        ----------
        triangulation : `.Triangulation`
            An already created triangular grid.
        x, y, triangles, mask
            Parameters defining the triangular grid. See `.Triangulation`.
            This is mutually exclusive with specifying *triangulation*.
        c : array-like
            The color values, either for the points or for the triangles. Which one
            is automatically inferred from the length of *c*, i.e. does it match
            the number of points or the number of triangles. If there are the same
            number of points and triangles in the triangulation it is assumed that
            color values are defined at points; to force the use of color values at
            triangles use the keyword argument ``facecolors=c`` instead of just
            ``c``.
            This parameter is position-only.
        facecolors : array-like, optional
            Can be used alternatively to *c* to specify colors at the triangle
            faces. This parameter takes precedence over *c*.
        shading : {'flat', 'gouraud'}, default: 'flat'
            If  'flat' and the color values *c* are defined at points, the color
            values used for each triangle are from the mean c of the triangle's
            three points. If *shading* is 'gouraud' then color values must be
            defined at points.
        other_parameters
            All other parameters are the same as for `~.Axes.pcolor`.
        """
        # 检查参数shading是否在合法列表['flat', 'gouraud']中，若不在则会引发异常
        _api.check_in_list(['flat', 'gouraud'], shading=shading)

        # 从传入的参数和关键字参数中获取三角化对象Triangulation以及其他参数
        tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)

        # 解析颜色，使其符合以下一种（另一变量将为None）：
        # - facecolors：如果在三角形面上指定了颜色
    # - point_colors: if specified at the points
    # 检查是否指定了 facecolors 参数
    if facecolors is not None:
        # 如果同时使用了位置参数 args，给出警告信息，因为 facecolors 参数会覆盖位置参数 c
        if args:
            _api.warn_external(
                "Positional parameter c has no effect when the keyword "
                "facecolors is given")
        # 初始化 point_colors 为 None
        point_colors = None
        # 检查 facecolors 的长度是否与三角形数目相同，如果不同则引发 ValueError
        if len(facecolors) != len(tri.triangles):
            raise ValueError("The length of facecolors must match the number "
                             "of triangles")
    else:
        # 如果没有指定 facecolors，则需检查位置参数 c 的情况
        # 如果没有传入位置参数，则引发 TypeError
        if not args:
            raise TypeError(
                "tripcolor() missing 1 required positional argument: 'c'; or "
                "1 required keyword-only argument: 'facecolors'")
        # 如果传入了多余的位置参数，则引发 TypeError，并指明多余的参数
        elif len(args) > 1:
            raise TypeError(f"Unexpected positional parameters: {args[1:]!r}")
        # 将位置参数 c 转换为 numpy 数组
        c = np.asarray(args[0])
        # 根据 c 的长度确定是给节点着色还是给三角形着色
        if len(c) == len(tri.x):
            # 如果 c 的长度与节点数目相同，则将其视为节点颜色
            point_colors = c
            facecolors = None
        elif len(c) == len(tri.triangles):
            # 如果 c 的长度与三角形数目相同，则将其视为三角形颜色
            point_colors = None
            facecolors = c
        else:
            # 否则引发 ValueError，因为 c 的长度既不匹配节点数目也不匹配三角形数目
            raise ValueError('The length of c must match either the number '
                             'of points or the number of triangles')

    # Handling of linewidths, shading, edgecolors and antialiased as
    # in Axes.pcolor
    # 处理 linewidths 参数，默认为 (0.25,)
    linewidths = (0.25,)
    # 如果 kwargs 中包含 'linewidth'，则将其改为 'linewidths'
    if 'linewidth' in kwargs:
        kwargs['linewidths'] = kwargs.pop('linewidth')
    # 设置默认的 'linewidths' 参数为 linewidths 变量中的值
    kwargs.setdefault('linewidths', linewidths)

    # 处理 edgecolors 参数，默认为 'none'
    edgecolors = 'none'
    # 如果 kwargs 中包含 'edgecolor'，则将其改为 'edgecolors'
    if 'edgecolor' in kwargs:
        kwargs['edgecolors'] = kwargs.pop('edgecolor')
    # 设置默认的 'edgecolors' 参数为 edgecolors 变量中的值
    ec = kwargs.setdefault('edgecolors', edgecolors)

    # 处理 antialiased 参数，默认情况下与 'edgecolors' 参数为 'none' 时为 False
    if 'antialiased' in kwargs:
        kwargs['antialiaseds'] = kwargs.pop('antialiased')
    # 如果 'antialiaseds' 不在 kwargs 中且 ec 的小写形式为 "none"，则设置为 False
    if 'antialiaseds' not in kwargs and ec.lower() == "none":
        kwargs['antialiaseds'] = False

    # 根据 shading 参数的值创建对应的集合对象
    if shading == 'gouraud':
        # 当 shading='gouraud' 时，如果指定了 facecolors 参数则引发 ValueError
        if facecolors is not None:
            raise ValueError(
                "shading='gouraud' can only be used when the colors "
                "are specified at the points, not at the faces.")
        # 创建 TriMesh 对象，使用节点颜色进行着色
        collection = TriMesh(tri, alpha=alpha, array=point_colors,
                             cmap=cmap, norm=norm, **kwargs)
    else:  # shading == 'flat'
        # 获取被掩码的三角形
        maskedTris = tri.get_masked_triangles()
        # 获取三角形的顶点坐标
        verts = np.stack((tri.x[maskedTris], tri.y[maskedTris]), axis=-1)

        # 获取颜色值
        if facecolors is None:
            # 每个三角形一个颜色，为三个顶点颜色值的平均值
            colors = point_colors[maskedTris].mean(axis=1)
        elif tri.mask is not None:
            # 去除掩码三角形的颜色值
            colors = facecolors[~tri.mask]
        else:
            colors = facecolors
        # 创建 PolyCollection 对象，使用 colors 进行着色
        collection = PolyCollection(verts, alpha=alpha, array=colors,
                                    cmap=cmap, norm=norm, **kwargs)
    # 调用 collection 对象的 _scale_norm 方法，设置其标准化范围为 norm，最小值为 vmin，最大值为 vmax
    collection._scale_norm(norm, vmin, vmax)
    # 在图形 ax 上关闭网格线
    ax.grid(False)

    # 计算三角形集合 tri 的 x 和 y 坐标的最小值和最大值
    minx = tri.x.min()
    maxx = tri.x.max()
    miny = tri.y.min()
    maxy = tri.y.max()
    # 构建表示图形边界的坐标元组
    corners = (minx, miny), (maxx, maxy)
    # 更新 ax 的数据限制，以便显示 corners 定义的边界
    ax.update_datalim(corners)
    # 自动调整 ax 的视图范围
    ax.autoscale_view()
    # 向 ax 中添加三角形集合的图形对象 collection
    ax.add_collection(collection)
    # 返回添加的图形对象 collection
    return collection
```