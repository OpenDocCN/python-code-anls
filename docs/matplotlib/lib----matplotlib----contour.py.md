# `D:\src\scipysrc\matplotlib\lib\matplotlib\contour.py`

```py
"""
Classes to support contour plotting and labelling for the Axes class.
"""

from contextlib import ExitStack
import functools
import math
from numbers import Integral

import numpy as np
from numpy import ma

import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.font_manager as font_manager
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms


def _contour_labeler_event_handler(cs, inline, inline_spacing, event):
    """
    Event handler for contour labels.

    Parameters:
    - cs: ContourSet object
    - inline: Boolean indicating whether labels are inline
    - inline_spacing: Spacing for inline labels
    - event: Event triggering the handler

    Handles different events:
    - Middle mouse button or escape/enter key: Stops the event loop.
    - Right mouse button or backspace/delete key: Removes the last label.
    - Left mouse button or any other key press: Adds a new label near the click.
    """
    canvas = cs.axes.figure.canvas
    is_button = event.name == "button_press_event"
    is_key = event.name == "key_press_event"

    if (is_button and event.button == MouseButton.MIDDLE
            or is_key and event.key in ["escape", "enter"]):
        canvas.stop_event_loop()
    elif (is_button and event.button == MouseButton.RIGHT
          or is_key and event.key in ["backspace", "delete"]):
        if not inline:
            cs.pop_label()
            canvas.draw()
    elif (is_button and event.button == MouseButton.LEFT
          or is_key and event.key is not None):
        if cs.axes.contains(event)[0]:
            cs.add_label_near(event.x, event.y, transform=False,
                              inline=inline, inline_spacing=inline_spacing)
            canvas.draw()


class ContourLabeler:
    """
    Mixin class to provide labeling capability to ContourSet objects.
    """

    def print_label(self, linecontour, labelwidth):
        """
        Determines if a contour is long enough to accommodate a label.

        Parameters:
        - linecontour: Contour line coordinates
        - labelwidth: Width threshold for label eligibility

        Returns:
        - Boolean indicating whether the contour can hold a label.
        """
        return (len(linecontour) > 10 * labelwidth
                or (len(linecontour)
                    and (np.ptp(linecontour, axis=0) > 1.2 * labelwidth).any()))

    def too_close(self, x, y, lw):
        """
        Checks if a label is already too close to a specified location.

        Parameters:
        - x: X-coordinate of the location
        - y: Y-coordinate of the location
        - lw: Label width threshold

        Returns:
        - Boolean indicating if a label is too close to the location.
        """
        thresh = (1.2 * lw) ** 2
        return any((x - loc[0]) ** 2 + (y - loc[1]) ** 2 < thresh
                   for loc in self.labelXYs)
    # 返回第 nth 个标签的宽度，单位为像素
    def _get_nth_label_width(self, nth):
        # 获取当前轴所属的图形对象
        fig = self.axes.figure
        # 获取图形对象的渲染器
        renderer = fig._get_renderer()
        # 创建一个临时的文本对象，用于计算标签的宽度
        return (Text(0, 0,
                     self.get_text(self.labelLevelList[nth], self.labelFmt),
                     figure=fig, fontproperties=self._label_font_props)
                .get_window_extent(renderer).width)

    # 获取标签的文本内容
    def get_text(self, lev, fmt):
        # 如果 lev 是字符串，直接返回
        if isinstance(lev, str):
            return lev
        # 如果 fmt 是字典，根据 lev 获取对应的格式化字符串，否则使用默认格式 '%1.3f'
        elif isinstance(fmt, dict):
            return fmt.get(lev, '%1.3f')
        # 如果 fmt 具有 format_ticks 方法，调用其格式化函数
        elif callable(getattr(fmt, "format_ticks", None)):
            return fmt.format_ticks([*self.labelLevelList, lev])[-1]
        # 如果 fmt 是可调用的，则将 lev 传递给 fmt 进行格式化
        elif callable(fmt):
            return fmt(lev)
        # 否则，使用 fmt 格式化 lev
        else:
            return fmt % lev

    # 定位标签的位置，以便在轮廓的相对平坦部分绘制标签
    def locate_label(self, linecontour, labelwidth):
        """
        Find good place to draw a label (relatively flat part of the contour).
        """
        # 获取轮廓的长度
        ctr_size = len(linecontour)
        # 计算将轮廓分成几个块，确保每个块的长度接近 labelwidth
        n_blocks = int(np.ceil(ctr_size / labelwidth)) if labelwidth > 1 else 1
        block_size = ctr_size if n_blocks == 1 else int(labelwidth)
        # 将轮廓分割成长度为 block_size 的块，最后一个块通过循环轮廓的起点来填充
        xx = np.resize(linecontour[:, 0], (n_blocks, block_size))
        yy = np.resize(linecontour[:, 1], (n_blocks, block_size))
        yfirst = yy[:, :1]
        ylast = yy[:, -1:]
        xfirst = xx[:, :1]
        xlast = xx[:, -1:]
        s = (yfirst - yy) * (xlast - xfirst) - (xfirst - xx) * (ylast - yfirst)
        l = np.hypot(xlast - xfirst, ylast - yfirst)
        # 忽略除零错误的警告，因为这是一个有效的选项
        with np.errstate(divide='ignore', invalid='ignore'):
            # 计算每个块中轮廓的偏离程度，用于确定最适合绘制标签的位置
            distances = (abs(s) / l).sum(axis=-1)
        # 标签绘制在块的中间位置（hbsize），这个位置轮廓距离直线最近，但不会与现有标签太近
        hbsize = block_size // 2
        adist = np.argsort(distances)
        # 如果所有候选位置都与现有标签太近，则返回距离直线最近的部分（adist[0]）
        for idx in np.append(adist, adist[0]):
            x, y = xx[idx, hbsize], yy[idx, hbsize]
            if not self.too_close(x, y, labelwidth):
                break
        # 返回标签的位置坐标及其在轮廓中的索引位置
        return x, y, (idx * block_size + hbsize) % ctr_size

    # 标记为已过时的函数，在版本 3.8 中将不再使用
    @_api.deprecated("3.8")
    # 定义一个方法，用于向图表添加等高线标签，根据设置决定是否使用*use_clabeltext*。
    def add_label(self, x, y, rotation, lev, cvalue):
        """Add a contour label, respecting whether *use_clabeltext* was set."""
        # 将数据坐标(x, y)转换为图表坐标系中的坐标
        data_x, data_y = self.axes.transData.inverted().transform((x, y))
        # 创建一个文本对象，用于显示等高线标签
        t = Text(
            data_x, data_y,
            text=self.get_text(lev, self.labelFmt),  # 获取要显示的标签文本
            rotation=rotation,  # 设定文本的旋转角度
            horizontalalignment='center', verticalalignment='center',  # 设置文本水平和垂直对齐方式
            zorder=self._clabel_zorder,  # 设定文本的绘制顺序
            color=self.labelMappable.to_rgba(cvalue, alpha=self.get_alpha()),  # 设定文本的颜色
            fontproperties=self._label_font_props,  # 设定文本的字体属性
            clip_box=self.axes.bbox)  # 设定文本的裁剪框

        # 如果设置了_use_clabeltext标志，则调整文本旋转角度
        if self._use_clabeltext:
            data_rotation, = self.axes.transData.inverted().transform_angles(
                [rotation], [[x, y]])
            t.set(rotation=data_rotation, transform_rotates_text=True)

        # 将文本对象添加到标签文本列表中
        self.labelTexts.append(t)
        # 将文本对应的值添加到标签值列表中
        self.labelCValues.append(cvalue)
        # 将文本的位置坐标添加到标签坐标列表中
        self.labelXYs.append((x, y))
        # 将文本对象添加到图表中
        self.axes.add_artist(t)

    # 定义一个废弃的方法，以add_label为替代，支持旧版本的兼容性
    @_api.deprecated("3.8", alternative="add_label")
    def add_label_clabeltext(self, x, y, rotation, lev, cvalue):
        """Add contour label with `.Text.set_transform_rotates_text`."""
        # 使用上下文管理器设置_use_clabeltext标志为True，并调用add_label方法
        with cbook._setattr_cm(self, _use_clabeltext=True):
            self.add_label(x, y, rotation, lev, cvalue)
    def add_label_near(self, x, y, inline=True, inline_spacing=5,
                       transform=None):
        """
        在点 ``(x, y)`` 附近添加一个标签。

        Parameters
        ----------
        x, y : float
            标签的大致位置坐标。
        inline : bool, default: True
            如果为 *True*，则在标签下方移除轮廓的部分。
        inline_spacing : int, default: 5
            当进行内联放置时，在标签的两侧留下的像素空间。
            对于直线轮廓上的标签，这种间距是精确的，对于曲线轮廓上的标签则不太精确。
        transform : `.Transform` 或 `False`, default: ``self.axes.transData``
            应用于 ``(x, y)`` 的变换，用于标记之前的位置。默认情况下，
            ``(x, y)`` 被解释为数据坐标。`False` 是 `.IdentityTransform` 的同义词；
            即 ``(x, y)`` 应该被解释为显示坐标。

        如果 transform 为 None，则使用 self.axes.transData 进行变换
        """
        # 如果 transform 不为 None，则将 (x, y) 进行坐标变换
        if transform is None:
            transform = self.axes.transData
        if transform:
            x, y = transform.transform((x, y))

        # 在 labelIndiceList 中找到最近轮廓的索引和最近顶点的索引及其投影
        idx_level_min, idx_vtx_min, proj = self._find_nearest_contour(
            (x, y), self.labelIndiceList)
        # 获取与 idx_level_min 对应的路径
        path = self._paths[idx_level_min]
        # 获取 labelIndiceList 中 idx_level_min 的索引
        level = self.labelIndiceList.index(idx_level_min)
        # 获取第 level 级别标签的宽度
        label_width = self._get_nth_label_width(level)
        # 获取路径的旋转角度和分割后的路径
        rotation, path = self._split_path_and_get_label_rotation(
            path, idx_vtx_min, proj, label_width, inline_spacing)
        # 添加标签到 proj 位置，带有旋转角度和对应的标签级别及值
        self.add_label(*proj, rotation, self.labelLevelList[idx_level_min],
                       self.labelCValueList[idx_level_min])

        # 如果 inline 为 True，则更新路径
        if inline:
            self._paths[idx_level_min] = path

    def pop_label(self, index=-1):
        """默认移除最后一个标签，但可以提供任意索引"""
        # 从 labelCValues 中移除指定索引的值
        self.labelCValues.pop(index)
        # 从 labelTexts 中移除指定索引的文本对象
        t = self.labelTexts.pop(index)
        # 移除文本对象 t
        t.remove()
    # 定义一个方法用于处理标签的添加，接收两个参数：inline（布尔值，表示是否内联添加），inline_spacing（整数，表示内联添加的间距）
    def labels(self, inline, inline_spacing):
        # 使用 enumerate 函数遍历三个列表的元素：self.labelIndiceList, self.labelLevelList, self.labelCValueList
        for idx, (icon, lev, cvalue) in enumerate(zip(
                self.labelIndiceList,
                self.labelLevelList,
                self.labelCValueList,
        )):
            # 获取当前对象的仿射变换
            trans = self.get_transform()
            # 获取当前标签的宽度
            label_width = self._get_nth_label_width(idx)
            # 用于存储将要添加到路径中的新路径
            additions = []
            # 遍历当前图标对象的路径的所有连接组件
            for subpath in self._paths[icon]._iter_connected_components():
                # 将路径的顶点坐标变换到屏幕坐标系中
                screen_xys = trans.transform(subpath.vertices)
                # 检查是否路径足够长以添加标签
                if self.print_label(screen_xys, label_width):
                    # 定位标签的位置和索引
                    x, y, idx = self.locate_label(screen_xys, label_width)
                    # 分离路径并获取标签的旋转角度和新路径
                    rotation, path = self._split_path_and_get_label_rotation(
                        subpath, idx, (x, y),
                        label_width, inline_spacing)
                    # 添加标签到当前对象中
                    self.add_label(x, y, rotation, lev, cvalue)  # 真正添加标签的操作
                    # 如果是内联模式，添加新路径到 additions 中
                    if inline:
                        additions.append(path)
                else:
                    # 如果不添加标签，保留原来的路径
                    additions.append(subpath)
            # 如果是内联模式，用 additions 中的新路径替换当前图标对象的旧路径
            if inline:
                self._paths[icon] = Path.make_compound_path(*additions)

    # 覆盖父类的 remove 方法，并移除所有与标签相关的文本
    def remove(self):
        super().remove()  # 调用父类的 remove 方法
        # 遍历所有标签文本对象，并逐个移除
        for text in self.labelTexts:
            text.remove()
def _is_closed_polygon(X):
    """
    Return whether first and last object in a sequence are the same. These are
    presumably coordinates on a polygonal curve, in which case this function
    tests if that curve is closed.
    """
    # 检查序列的第一个和最后一个对象是否相同，用于判断多边形曲线是否闭合
    return np.allclose(X[0], X[-1], rtol=1e-10, atol=1e-13)


def _find_closest_point_on_path(xys, p):
    """
    Parameters
    ----------
    xys : (N, 2) array-like
        Coordinates of vertices.
    p : (float, float)
        Coordinates of point.

    Returns
    -------
    d2min : float
        Minimum square distance of *p* to *xys*.
    proj : (float, float)
        Projection of *p* onto *xys*.
    imin : (int, int)
        Consecutive indices of vertices of segment in *xys* where *proj* is.
        Segments are considered as including their end-points; i.e. if the
        closest point on the path is a node in *xys* with index *i*, this
        returns ``(i-1, i)``.  For the special case where *xys* is a single
        point, this returns ``(0, 0)``.
    """
    # 如果只有一个点，直接计算到该点的距离并返回
    if len(xys) == 1:
        return (((p - xys[0]) ** 2).sum(), xys[0], (0, 0))
    
    # 计算相邻点之间的向量
    dxys = xys[1:] - xys[:-1]  # Individual segment vectors.
    
    # 计算每个向量的平方范数
    norms = (dxys ** 2).sum(axis=1)
    norms[norms == 0] = 1  # 处理长度为零的向量，避免出现除以零错误
    
    # 计算点到每条线段的投影在0-1之间的相对坐标
    rel_projs = np.clip(
        ((p - xys[:-1]) * dxys).sum(axis=1) / norms,
        0, 1)[:, None]
    
    # 计算点到每条线段的投影点坐标
    projs = xys[:-1] + rel_projs * dxys
    
    # 计算点到每个投影点的平方距离
    d2s = ((projs - p) ** 2).sum(axis=1)
    
    # 找到距离最小的投影点的索引
    imin = np.argmin(d2s)
    
    # 返回最小距离的平方、最近的投影点、以及包含该点的线段的索引对
    return (d2s[imin], projs[imin], (imin, imin+1))


_docstring.interpd.update(contour_set_attributes=r"""
Attributes
----------
ax : `~matplotlib.axes.Axes`
    The Axes object in which the contours are drawn.

collections : `.silent_list` of `.PathCollection`\s
    The `.Artist`\s representing the contour. This is a list of
    `.PathCollection`\s for both line and filled contours.

levels : array
    The values of the contour levels.

layers : array
    Same as levels for line contours; half-way between
    levels for filled contours.  See ``ContourSet._process_colors``.
""")


@_docstring.dedent_interpd
class ContourSet(ContourLabeler, mcoll.Collection):
    """
    Store a set of contour lines or filled regions.

    User-callable method: `~.Axes.clabel`

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`

    levels : [level0, level1, ..., leveln]
        A list of floating point numbers indicating the contour levels.

    allsegs : [level0segs, level1segs, ...]
        List of all the polygon segments for all the *levels*.
        For contour lines ``len(allsegs) == len(levels)``, and for
        filled contour regions ``len(allsegs) = len(levels)-1``. The lists
        should look like ::

            level0segs = [polygon0, polygon1, ...]
            polygon0 = [[x0, y0], [x1, y1], ...]


    """
    # 存储一组等高线或填充区域

    def __init__(self, ax, levels, allsegs):
        super().__init__(ax)

        # 初始化等高线集合的属性
        self.levels = levels
        self.collections = self._init_collections(allsegs)
        self.layers = self._init_layers()

    def _init_collections(self, allsegs):
        """
        Initialize collections representing contour lines or filled regions.

        Parameters
        ----------
        allsegs : list
            List of polygon segments for all the *levels*.

        Returns
        -------
        collections : list
            List of `.PathCollection`\s representing the contour.
        """
        # 初始化表示等高线或填充区域的集合
        collections = []
        for level_segs in allsegs:
            paths = [mpath.Path(seg) for seg in level_segs]
            collections.append(mcoll.PathCollection(paths))
        return collections

    def _init_layers(self):
        """
        Initialize layers for contour lines or filled regions.

        Returns
        -------
        layers : array
            Array representing levels or half-way between levels for filled
            contours.
        """
        # 初始化等高线或填充区域的层级
        layers = np.array(self.levels[:-1]) + np.diff(self.levels) / 2
        return layers
    allkinds : ``None`` or [level0kinds, level1kinds, ...]
        Optional list of all the polygon vertex kinds (code types), as
        described and used in Path. This is used to allow multiply-
        connected paths such as holes within filled polygons.
        If not ``None``, ``len(allkinds) == len(allsegs)``. The lists
        should look like ::

            level0kinds = [polygon0kinds, ...]
            polygon0kinds = [vertexcode0, vertexcode1, ...]

        If *allkinds* is not ``None``, usually all polygons for a
        particular contour level are grouped together so that
        ``level0segs = [polygon0]`` and ``level0kinds = [polygon0kinds]``.

    **kwargs
        Keyword arguments are as described in the docstring of
        `~.Axes.contour`.

    %(contour_set_attributes)s
    """



    allsegs = property(lambda self: [
        [subp.vertices for subp in p._iter_connected_components()]
        for p in self.get_paths()])

lambda 函数定义了一个属性 `allsegs`，它返回一个列表，包含所有路径对象的顶点列表。每个路径对象通过 `_iter_connected_components()` 方法连接其子路径，使用 `get_paths()` 获取所有路径对象。


    allkinds = property(lambda self: [
        [subp.codes for subp in p._iter_connected_components()]
        for p in self.get_paths()])

lambda 函数定义了一个属性 `allkinds`，它返回一个列表，包含所有路径对象的顶点类型列表。每个路径对象通过 `_iter_connected_components()` 方法连接其子路径，使用 `get_paths()` 获取所有路径对象。


    tcolors = _api.deprecated("3.8")(property(lambda self: [
        (tuple(rgba),) for rgba in self.to_rgba(self.cvalues, self.alpha)]))

`_api.deprecated` 标记了这个属性 `tcolors` 已在版本 3.8 中废弃。它返回一个元组列表，包含每个颜色值 `(rgba)` 组成的元组，由 `to_rgba` 方法生成，接受 `cvalues` 和 `alpha` 作为参数。


    tlinewidths = _api.deprecated("3.8")(property(lambda self: [
        (w,) for w in self.get_linewidths()]))

`_api.deprecated` 标记了这个属性 `tlinewidths` 已在版本 3.8 中废弃。它返回一个包含每条线宽度 `w` 的元组列表，由 `get_linewidths()` 方法生成。


    alpha = property(lambda self: self.get_alpha())

lambda 函数定义了一个属性 `alpha`，它返回当前对象的 alpha 值，由 `get_alpha()` 方法获取。


    linestyles = property(lambda self: self._orig_linestyles)

lambda 函数定义了一个属性 `linestyles`，它返回当前对象的原始线条样式 `_orig_linestyles`。


    @_api.deprecated("3.8", alternative="set_antialiased or get_antialiased",
                     addendum="Note that get_antialiased returns an array.")
    @property
    def antialiased(self):
        return all(self.get_antialiased())

`@property` 装饰器定义了一个属性 `antialiased`，用于获取当前对象的抗锯齿属性。它在版本 3.8 中已被废弃，推荐使用 `set_antialiased` 或 `get_antialiased` 替代。注：`get_antialiased` 返回一个数组。


    @antialiased.setter
    def antialiased(self, aa):
        self.set_antialiased(aa)

`@antialiased.setter` 装饰器定义了 `antialiased` 属性的设置方法，用于设置抗锯齿属性为 `aa`。


    @_api.deprecated("3.8")
    @property

`@property` 装饰器定义了一个已废弃的属性。
    def collections(self):
        # 当访问此方法时，将对象设为不可见，并替换为旧样式的集合对象
        # （每个级别一个 PathCollection）。我们不尝试进一步将轮廓分割为连接的组件，
        # 因为我们已经无法追踪哪些轮廓对需要被视为单个单位来绘制带孔填充区域。

        # 如果对象还没有属性 "_old_style_split_collections"
        if not hasattr(self, "_old_style_split_collections"):
            # 设置当前对象为不可见
            self.set_visible(False)
            # 获取当前对象的填充颜色、边缘颜色、线宽和线型
            fcs = self.get_facecolor()
            ecs = self.get_edgecolor()
            lws = self.get_linewidth()
            lss = self.get_linestyle()
            # 初始化一个空的列表用于存放旧样式的集合对象
            self._old_style_split_collections = []
            # 遍历所有路径，并创建 PathCollection 对象
            for idx, path in enumerate(self._paths):
                pc = mcoll.PathCollection(
                    [path] if len(path.vertices) else [],  # 使用当前路径创建 PathCollection
                    alpha=self.get_alpha(),  # 设置透明度
                    antialiaseds=self._antialiaseds[idx % len(self._antialiaseds)],  # 设置是否抗锯齿
                    transform=self.get_transform(),  # 设置变换方式
                    zorder=self.get_zorder(),  # 设置绘制顺序
                    label="_nolegend_",  # 设置标签
                    facecolor=fcs[idx] if len(fcs) else "none",  # 设置填充颜色
                    edgecolor=ecs[idx] if len(ecs) else "none",  # 设置边缘颜色
                    linewidths=[lws[idx % len(lws)]],  # 设置线宽
                    linestyles=[lss[idx % len(lss)]],  # 设置线型
                )
                # 如果设置了填充，设置图案填充方式
                if self.filled:
                    pc.set(hatch=self.hatches[idx % len(self.hatches)])
                # 将创建的 PathCollection 对象添加到列表中
                self._old_style_split_collections.append(pc)
            # 将所有创建的旧样式集合对象添加到当前对象所属的坐标系中
            for col in self._old_style_split_collections:
                self.axes.add_collection(col)
        
        # 返回所有旧样式集合对象的列表
        return self._old_style_split_collections

    def get_transform(self):
        """返回此 ContourSet 使用的 `.Transform` 实例。"""
        # 如果当前对象的变换属性 _transform 是空的
        if self._transform is None:
            # 设置为数据坐标系
            self._transform = self.axes.transData
        # 如果 _transform 不是 matplotlib 的 Transform 实例，并且具有 _as_mpl_transform 方法
        elif (not isinstance(self._transform, mtransforms.Transform)
              and hasattr(self._transform, '_as_mpl_transform')):
            # 调用 _as_mpl_transform 方法转换为 matplotlib 的 Transform 实例
            self._transform = self._transform._as_mpl_transform(self.axes)
        # 返回当前对象使用的变换方式
        return self._transform

    def __getstate__(self):
        state = self.__dict__.copy()
        # C 对象 _contour_generator 目前无法被 pickle 化。
        # 这不是大问题，因为一旦计算出轮廓，它就不再被使用。
        state['_contour_generator'] = None
        # 返回对象的状态信息
        return state
    def legend_elements(self, variable_name='x', str_format=str):
        """
        Return a list of artists and labels suitable for passing through
        to `~.Axes.legend` which represent this ContourSet.

        The labels have the form "0 < x <= 1" stating the data ranges which
        the artists represent.

        Parameters
        ----------
        variable_name : str
            The string used inside the inequality used on the labels.
        str_format : function: float -> str
            Function used to format the numbers in the labels.

        Returns
        -------
        artists : list[`.Artist`]
            A list of the artists.
        labels : list[str]
            A list of the labels.
        """
        # Initialize empty lists for artists and labels
        artists = []
        labels = []

        # Check if ContourSet is filled
        if self.filled:
            # Obtain lower and upper bounds for levels
            lowers, uppers = self._get_lowers_and_uppers()
            n_levels = len(self._paths)
            
            # Iterate over levels to create rectangle artists
            for idx in range(n_levels):
                artists.append(mpatches.Rectangle(
                    (0, 0), 1, 1,
                    facecolor=self.get_facecolor()[idx],
                    hatch=self.hatches[idx % len(self.hatches)],
                ))
                # Format lower and upper bounds as strings
                lower = str_format(lowers[idx])
                upper = str_format(uppers[idx])
                
                # Determine label format based on extend option
                if idx == 0 and self.extend in ('min', 'both'):
                    labels.append(fr'${variable_name} \leq {lower}$')
                elif idx == n_levels - 1 and self.extend in ('max', 'both'):
                    labels.append(fr'${variable_name} > {upper}$')
                else:
                    labels.append(fr'${lower} < {variable_name} \leq {upper}$')
        else:
            # Create Line2D artists and labels for each level
            for idx, level in enumerate(self.levels):
                artists.append(Line2D(
                    [], [],
                    color=self.get_edgecolor()[idx],
                    linewidth=self.get_linewidths()[idx],
                    linestyle=self.get_linestyles()[idx],
                ))
                labels.append(fr'${variable_name} = {str_format(level)}$')

        # Return the lists of artists and labels
        return artists, labels
    def _process_args(self, *args, **kwargs):
        """
        Process *args* and *kwargs*; override in derived classes.

        Must set self.levels, self.zmin and self.zmax, and update Axes limits.
        """
        # 设置 self.levels 为参数 args 的第一个元素
        self.levels = args[0]
        # 设置 allsegs 为参数 args 的第二个元素
        allsegs = args[1]
        # 如果 args 的长度大于2，则设置 allkinds 为参数 args 的第三个元素；否则设为 None
        allkinds = args[2] if len(args) > 2 else None
        # 设置 self.zmax 为 self.levels 中的最大值
        self.zmax = np.max(self.levels)
        # 设置 self.zmin 为 self.levels 中的最小值
        self.zmin = np.min(self.levels)

        # 如果 allkinds 为 None，则设置其为一个二维列表，每个内部列表长度与对应的 allsegs 中的列表长度相同
        if allkinds is None:
            allkinds = [[None] * len(segs) for segs in allsegs]

        # 检查 filled 标志，验证 allsegs 的长度与 levels 的数量相符
        if self.filled:
            if len(allsegs) != len(self.levels) - 1:
                raise ValueError('must be one less number of segments as '
                                 'levels')
        else:
            # 如果 filled 为 False，则验证 allsegs 的长度与 levels 的数量相同
            if len(allsegs) != len(self.levels):
                raise ValueError('must be same number of segments as levels')

        # 检查 allkinds 的长度与 allsegs 的长度是否一致
        if len(allkinds) != len(allsegs):
            raise ValueError('allkinds has different length to allsegs')

        # 计算所有线段的最小和最大值，更新 axes 数据限制
        flatseglist = [s for seg in allsegs for s in seg]
        points = np.concatenate(flatseglist, axis=0)
        self._mins = points.min(axis=0)
        self._maxs = points.max(axis=0)

        # 遍历 allsegs 和 allkinds，构建路径并连接起来，形成 _paths 列表
        self._paths = [Path.make_compound_path(*map(Path, segs, kinds))
                       for segs, kinds in zip(allsegs, allkinds)]

        # 返回 kwargs 参数
        return kwargs

    def _make_paths_from_contour_generator(self):
        """Compute ``paths`` using C extension."""
        # 如果 _paths 不为 None，则直接返回 _paths
        if self._paths is not None:
            return self._paths
        # 否则，利用 _contour_generator 计算顶点和路径码，并生成 Path 对象列表
        cg = self._contour_generator
        empty_path = Path(np.empty((0, 2)))
        vertices_and_codes = (
            map(cg.create_filled_contour, *self._get_lowers_and_uppers())
            if self.filled else
            map(cg.create_contour, self.levels))
        return [Path(np.concatenate(vs), np.concatenate(cs)) if len(vs) else empty_path
                for vs, cs in vertices_and_codes]

    def _get_lowers_and_uppers(self):
        """
        Return ``(lowers, uppers)`` for filled contours.
        """
        # 返回用于填充等高线的下限和上限数组
        lowers = self._levels[:-1]
        if self.zmin == lowers[0]:
            # 如果 zmin 等于 lowers 的第一个元素，则调整 lowers 数组，确保最小值包含在最低区间内
            lowers = lowers.copy()  # 复制一份，避免改变 self._levels
            if self.logscale:
                lowers[0] = 0.99 * self.zmin
            else:
                lowers[0] -= 1
        # 返回上限数组
        uppers = self._levels[1:]
        return (lowers, uppers)
    def changed(self):
        # 如果对象没有属性 "cvalues"，则调用 _process_colors() 方法设置它
        if not hasattr(self, "cvalues"):
            self._process_colors()  # 设置 cvalues.
        
        # 立即强制进行自动缩放，因为 self.to_rgba() 方法内部会使用传递给它的数据调用 autoscale_None()，
        # 所以如果 vmin/vmax 还未设置，则会使用 *cvalues* 的内容覆盖它们，而不是我们想要的 levels
        self.norm.autoscale_None(self.levels)
        
        # 设置数组的值为 self.cvalues
        self.set_array(self.cvalues)
        
        # 更新标量映射对象
        self.update_scalarmappable()
        
        # 使用广播方式获取 alpha 值，使其与 self.cvalues 的长度相匹配
        alphas = np.broadcast_to(self.get_alpha(), len(self.cvalues))
        
        # 遍历标签文本、颜色值和透明度，分别设置每个标签的透明度和颜色
        for label, cv, alpha in zip(self.labelTexts, self.labelCValues, alphas):
            label.set_alpha(alpha)
            label.set_color(self.labelMappable.to_rgba(cv))
        
        # 调用父类的 changed() 方法
        super().changed()

    def _autolev(self, N):
        """
        Select contour levels to span the data.

        The target number of levels, *N*, is used only when the
        scale is not log and default locator is used.

        We need two more levels for filled contours than for
        line contours, because for the latter we need to specify
        the lower and upper boundary of each range. For example,
        a single contour boundary, say at z = 0, requires only
        one contour line, but two filled regions, and therefore
        three levels to provide boundaries for both regions.
        """
        # 如果未指定 locator，则根据 logscale 来选择合适的 MaxNLocator 或 LogLocator
        if self.locator is None:
            if self.logscale:
                self.locator = ticker.LogLocator()
            else:
                self.locator = ticker.MaxNLocator(N + 1, min_n_ticks=1)

        # 获取轮廓线的数值范围
        lev = self.locator.tick_values(self.zmin, self.zmax)

        try:
            # 如果 locator 具有 _symmetric 属性，则返回 lev
            if self.locator._symmetric:
                return lev
        except AttributeError:
            pass

        # 剔除 locator 提供的多余的级别
        under = np.nonzero(lev < self.zmin)[0]
        i0 = under[-1] if len(under) else 0
        over = np.nonzero(lev > self.zmax)[0]
        i1 = over[0] + 1 if len(over) else len(lev)
        if self.extend in ('min', 'both'):
            i0 += 1
        if self.extend in ('max', 'both'):
            i1 -= 1

        # 如果级别数少于 3，则返回整个 lev；否则返回裁剪后的 lev
        if i1 - i0 < 3:
            i0, i1 = 0, len(lev)

        return lev[i0:i1]
    def _process_contour_level_args(self, args, z_dtype):
        """
        确定等高线的级别并存储在 self.levels 中。
        """
        # 如果未指定 levels 属性，则根据参数 args 确定
        if self.levels is None:
            if args:
                levels_arg = args[0]
            # 如果 z_dtype 是布尔类型且 filled 属性为 True，则设置默认的 levels
            elif np.issubdtype(z_dtype, bool):
                if self.filled:
                    levels_arg = [0, .5, 1]
                else:
                    levels_arg = [.5]
            else:
                levels_arg = 7  # 默认值，硬编码
        else:
            levels_arg = self.levels
        
        # 根据 levels_arg 的类型设置 self.levels
        if isinstance(levels_arg, Integral):
            self.levels = self._autolev(levels_arg)
        else:
            self.levels = np.asarray(levels_arg, np.float64)
        
        # 如果是 filled 等高线并且 levels 的数量小于 2，则抛出异常
        if self.filled and len(self.levels) < 2:
            raise ValueError("Filled contours require at least 2 levels.")
        
        # 检查 levels 是否严格递增
        if len(self.levels) > 1 and np.min(np.diff(self.levels)) <= 0.0:
            raise ValueError("Contour levels must be increasing")

    def _process_levels(self):
        """
        根据 levels 属性分配值给 layers 属性，
        如果填充等高线则根据需要添加扩展层。

        对于线条等高线，层简单地与 levels 相符；
        线是一个薄层。线条等高线不需要扩展级别。
        """
        # 创建私有属性 _levels 来包含扩展区域；我们希望保持原始 levels 属性不变。
        # （即使对于线条等高线，Colorbar 也需要这个属性。）
        self._levels = list(self.levels)

        # 根据 logscale 属性设置下限和上限值
        if self.logscale:
            lower, upper = 1e-250, 1e250
        else:
            lower, upper = -1e250, 1e250

        # 根据 extend 属性在 _levels 中插入扩展级别
        if self.extend in ('both', 'min'):
            self._levels.insert(0, lower)
        if self.extend in ('both', 'max'):
            self._levels.append(upper)
        self._levels = np.asarray(self._levels)

        # 如果不是填充等高线，则 layers 直接等于 levels
        if not self.filled:
            self.layers = self.levels
            return

        # 填充等高线时，层的值位于屏幕空间中级别的中间位置
        if self.logscale:
            # 通过在相乘之前取平方根来避免溢出
            self.layers = (np.sqrt(self._levels[:-1])
                           * np.sqrt(self._levels[1:]))
        else:
            self.layers = 0.5 * (self._levels[:-1] + self._levels[1:])
    def _process_colors(self):
        """
        Color argument processing for contouring.

        Note that we base the colormapping on the contour levels
        and layers, not on the actual range of the Z values.  This
        means we don't have to worry about bad values in Z, and we
        always have the full dynamic range available for the selected
        levels.

        The color is based on the midpoint of the layer, except for
        extended end layers.  By default, the norm vmin and vmax
        are the extreme values of the non-extended levels.  Hence,
        the layer color extremes are not the extreme values of
        the colormap itself, but approach those values as the number
        of levels increases.  An advantage of this scheme is that
        line contours, when added to filled contours, take on
        colors that are consistent with those of the filled regions;
        for example, a contour line on the boundary between two
        regions will have a color intermediate between those
        of the regions.

        """
        # 设置单色模式为 cmap 的单色属性
        self.monochrome = self.cmap.monochrome
        if self.colors is not None:
            # 生成整数列表以直接进行索引
            i0, i1 = 0, len(self.levels)
            if self.filled:
                i1 -= 1
                # 对超出范围的索引进行处理
                if self.extend in ('both', 'min'):
                    i0 -= 1
                if self.extend in ('both', 'max'):
                    i1 += 1
            # 使用列表推导生成颜色值列表
            self.cvalues = list(range(i0, i1))
            # 设置归一化为无归一化效果
            self.set_norm(mcolors.NoNorm())
        else:
            # 如果未提供颜色，直接使用 layers 作为颜色值列表
            self.cvalues = self.layers
        # 自动调整归一化范围
        self.norm.autoscale_None(self.levels)
        # 设置颜色数组
        self.set_array(self.cvalues)
        # 更新标量映射对象
        self.update_scalarmappable()
        # 根据 extend 属性设置是否裁剪归一化范围
        if self.extend in ('both', 'max', 'min'):
            self.norm.clip = False

    def _process_linewidths(self, linewidths):
        # 获取层数
        Nlev = len(self.levels)
        if linewidths is None:
            # 获取默认线宽度参数
            default_linewidth = mpl.rcParams['contour.linewidth']
            if default_linewidth is None:
                default_linewidth = mpl.rcParams['lines.linewidth']
            # 返回一个默认线宽度列表
            return [default_linewidth] * Nlev
        elif not np.iterable(linewidths):
            # 如果 linewidths 不可迭代，返回一个相同线宽度列表
            return [linewidths] * Nlev
        else:
            linewidths = list(linewidths)
            # 返回根据层数调整后的线宽度列表
            return (linewidths * math.ceil(Nlev / len(linewidths)))[:Nlev]
    # 处理线条样式的内部方法，根据给定的线条样式参数生成最终的线条样式列表
    def _process_linestyles(self, linestyles):
        # 获取等高线级别的数量
        Nlev = len(self.levels)
        
        # 如果未提供线条样式，则默认使用'solid'样式
        if linestyles is None:
            tlinestyles = ['solid'] * Nlev
            # 如果是单色模式，根据阈值调整为负值线条样式
            if self.monochrome:
                eps = - (self.zmax - self.zmin) * 1e-15
                for i, lev in enumerate(self.levels):
                    if lev < eps:
                        tlinestyles[i] = self.negative_linestyles
        else:
            # 如果线条样式是字符串，则所有等高线级别使用同一样式
            if isinstance(linestyles, str):
                tlinestyles = [linestyles] * Nlev
            # 如果线条样式是可迭代对象，则转换为列表
            elif np.iterable(linestyles):
                tlinestyles = list(linestyles)
                # 如果线条样式数量少于等高线级别数量，进行重复填充
                if len(tlinestyles) < Nlev:
                    nreps = int(np.ceil(Nlev / len(linestyles)))
                    tlinestyles = tlinestyles * nreps
                # 如果线条样式数量多于等高线级别数量，截取到等高线级别数量
                if len(tlinestyles) > Nlev:
                    tlinestyles = tlinestyles[:Nlev]
            else:
                # 抛出异常，表示未识别的线条样式类型
                raise ValueError("Unrecognized type for linestyles kwarg")
        
        # 返回处理后的线条样式列表
        return tlinestyles

    # 在未填充的等高线图中，查找离参考点最近的等高线段
    def _find_nearest_contour(self, xy, indices=None):
        """
        Find the point in the unfilled contour plot that is closest (in screen
        space) to point *xy*.

        Parameters
        ----------
        xy : tuple[float, float]
            The reference point (in screen space).
        indices : list of int or None, default: None
            Indices of contour levels to consider.  If None (the default), all levels
            are considered.

        Returns
        -------
        idx_level_min : int
            The index of the contour level closest to *xy*.
        idx_vtx_min : int
            The index of the `.Path` segment closest to *xy* (at that level).
        proj : (float, float)
            The point in the contour plot closest to *xy*.
        """

        # 将每个等高线段转换为像素坐标，然后比较给定点与每个等高线段的距离，以找到最近的点
        # 在正常情况下，这种方法足够快速，但可能存在优化空间。

        # 如果是填充的等高线图，则抛出异常
        if self.filled:
            raise ValueError("Method does not support filled contours")

        # 如果未指定要考虑的等高线级别的索引，则考虑所有级别
        if indices is None:
            indices = range(len(self._paths))

        # 初始化最小距离及其对应的索引和投影点
        d2min = np.inf
        idx_level_min = idx_vtx_min = proj_min = None

        # 遍历指定的等高线级别索引
        for idx_level in indices:
            path = self._paths[idx_level]
            idx_vtx_start = 0
            # 对路径中的每个连接子路径进行处理
            for subpath in path._iter_connected_components():
                # 如果子路径中没有顶点，则跳过
                if not len(subpath.vertices):
                    continue
                # 将路径顶点转换为当前坐标系的像素坐标
                lc = self.get_transform().transform(subpath.vertices)
                # 在路径上找到距离给定点最近的点及其投影
                d2, proj, leg = _find_closest_point_on_path(lc, xy)
                # 如果找到更近的点，则更新最小距离及其对应的索引和投影点
                if d2 < d2min:
                    d2min = d2
                    idx_level_min = idx_level
                    idx_vtx_min = leg[1] + idx_vtx_start
                    proj_min = proj
                idx_vtx_start += len(subpath)

        # 返回最接近给定点的等高线级别索引、路径段索引及其投影点
        return idx_level_min, idx_vtx_min, proj_min
    def find_nearest_contour(self, x, y, indices=None, pixel=True):
        """
        Find the point in the contour plot that is closest to ``(x, y)``.

        This method does not support filled contours.

        Parameters
        ----------
        x, y : float
            The reference point.
        indices : list of int or None, default: None
            Indices of contour levels to consider.  If None (the default), all
            levels are considered.
        pixel : bool, default: True
            If *True*, measure distance in pixel (screen) space, which is
            useful for manual contour labeling; else, measure distance in axes
            space.

        Returns
        -------
        path : int
            The index of the path that is closest to ``(x, y)``.  Each path corresponds
            to one contour level.
        subpath : int
            The index within that closest path of the subpath that is closest to
            ``(x, y)``.  Each subpath corresponds to one unbroken contour line.
        index : int
            The index of the vertices within that subpath that are closest to
            ``(x, y)``.
        xmin, ymin : float
            The point in the contour plot that is closest to ``(x, y)``.
        d2 : float
            The squared distance from ``(xmin, ymin)`` to ``(x, y)``.
        """
        # 初始化变量
        segment = index = d2 = None

        # 创建 ExitStack 对象，用于动态管理上下文
        with ExitStack() as stack:
            if not pixel:
                # 如果不是在像素空间中测量距离，将转换设置为单位矩阵，以禁用变换
                stack.enter_context(self._cm_set(
                    transform=mtransforms.IdentityTransform()))

            # 调用内部方法 _find_nearest_contour，获取最接近点的信息
            i_level, i_vtx, (xmin, ymin) = self._find_nearest_contour((x, y), indices)

        # 如果找到最接近点的信息
        if i_level is not None:
            # 计算连接组件的累积长度
            cc_cumlens = np.cumsum(
                [*map(len, self._paths[i_level]._iter_connected_components())])
            # 在累积长度数组中搜索 i_vtx 的位置
            segment = cc_cumlens.searchsorted(i_vtx, "right")
            # 计算在该段内的索引
            index = i_vtx if segment == 0 else i_vtx - cc_cumlens[segment - 1]
            # 计算最近点的平方距离
            d2 = (xmin-x)**2 + (ymin-y)**2

        # 返回结果元组
        return (i_level, segment, index, xmin, ymin, d2)

    def draw(self, renderer):
        # 获取路径列表和路径数目
        paths = self._paths
        n_paths = len(paths)
        # 如果没有填充或所有的填充图案都为 None，则调用父类的 draw 方法并返回
        if not self.filled or all(hatch is None for hatch in self.hatches):
            super().draw(renderer)
            return

        # 当存在填充图案时，逐个绘制轮廓
        for idx in range(n_paths):
            # 进入上下文管理，设置当前路径为 paths[idx]，同时设置画布属性
            with cbook._setattr_cm(self, _paths=[paths[idx]]), self._cm_set(
                hatch=self.hatches[idx % len(self.hatches)],
                array=[self.get_array()[idx]],
                linewidths=[self.get_linewidths()[idx % len(self.get_linewidths())]],
                linestyles=[self.get_linestyles()[idx % len(self.get_linestyles())]],
            ):
                # 调用父类的 draw 方法进行绘制
                super().draw(renderer)
# 定义 QuadContourSet 类，继承自 ContourSet 类，用于存储轮廓线或填充区域集合
@_docstring.dedent_interpd
class QuadContourSet(ContourSet):
    """
    Create and store a set of contour lines or filled regions.

    This class is typically not instantiated directly by the user but by
    `~.Axes.contour` and `~.Axes.contourf`.

    %(contour_set_attributes)s
    """

    # 处理参数和关键字参数的方法
    def _process_args(self, *args, corner_mask=None, algorithm=None, **kwargs):
        """
        Process args and kwargs.
        """
        # 如果参数 args 存在且第一个参数是 QuadContourSet 类的实例
        if args and isinstance(args[0], QuadContourSet):
            # 如果当前对象的 levels 属性为 None，则使用 args[0] 的 levels 属性
            if self.levels is None:
                self.levels = args[0].levels
            # 设置当前对象的 zmin 和 zmax 属性为 args[0] 的相应属性
            self.zmin = args[0].zmin
            self.zmax = args[0].zmax
            # 设置当前对象的 _corner_mask 属性为 args[0] 的相应属性
            self._corner_mask = args[0]._corner_mask
            # 设置当前对象的 _contour_generator 属性为 args[0] 的 _contour_generator 属性
            contour_generator = args[0]._contour_generator
            # 设置当前对象的 _mins 和 _maxs 属性为 args[0] 的相应属性
            self._mins = args[0]._mins
            self._maxs = args[0]._maxs
            # 设置当前对象的 _algorithm 属性为 args[0] 的 _algorithm 属性
            self._algorithm = args[0]._algorithm
        else:
            import contourpy
            
            # 导入 contourpy 库，用于生成轮廓线
            # 如果未指定 algorithm，则从 mpl.rcParams 中获取 contour.algorithm 设置
            if algorithm is None:
                algorithm = mpl.rcParams['contour.algorithm']
            # 验证并更新当前对象的 _algorithm 属性为指定的算法
            mpl.rcParams.validate["contour.algorithm"](algorithm)
            self._algorithm = algorithm

            # 如果未指定 corner_mask，则根据当前算法设置默认值
            if corner_mask is None:
                if self._algorithm == "mpl2005":
                    # 对于算法 "mpl2005"，不支持 corner_mask=True，因此如果未显式请求，则禁用它
                    corner_mask = False
                else:
                    corner_mask = mpl.rcParams['contour.corner_mask']
            # 设置当前对象的 _corner_mask 属性为指定的角点遮罩值
            self._corner_mask = corner_mask

            # 从参数 args 和 kwargs 中提取 x、y、z 数据
            x, y, z = self._contour_args(args, kwargs)

            # 使用 contourpy.contour_generator 函数生成轮廓生成器
            contour_generator = contourpy.contour_generator(
                x, y, z, name=self._algorithm, corner_mask=self._corner_mask,
                line_type=contourpy.LineType.SeparateCode,
                fill_type=contourpy.FillType.OuterCode,
                chunk_size=self.nchunk)

            # 获取当前对象的坐标变换
            t = self.get_transform()

            # 如果变换不是转换到数据坐标系（transData），且某些部分包含 transData，则转换 xs 和 ys 到数据坐标
            if (t != self.axes.transData and
                    any(t.contains_branch_seperately(self.axes.transData))):
                # 计算变换到数据坐标系后的点集
                trans_to_data = t - self.axes.transData
                pts = np.vstack([x.flat, y.flat]).T
                transformed_pts = trans_to_data.transform(pts)
                x = transformed_pts[..., 0]
                y = transformed_pts[..., 1]

            # 计算并设置当前对象的 _mins 和 _maxs 属性为 x 和 y 的最小值和最大值
            self._mins = [ma.min(x), ma.min(y)]
            self._maxs = [ma.max(x), ma.max(y)]

        # 设置当前对象的 _contour_generator 属性为轮廓生成器
        self._contour_generator = contour_generator

        # 返回剩余的关键字参数
        return kwargs
    # 根据传入的参数列表 args 和 kwargs 决定要使用的轮廓绘制函数名称
    if self.filled:
        fn = 'contourf'
    else:
        fn = 'contour'
    
    # 获取传入参数的数量
    nargs = len(args)
    
    # 根据参数数量的不同情况进行处理
    if 0 < nargs <= 2:
        # 如果 nargs 为 1 或 2，则假定参数为 z 和其他可能的参数
        z, *args = args
        # 将 z 转换为 MaskedArray 类型
        z = ma.asarray(z)
        # 初始化 x 和 y 坐标
        x, y = self._initialize_x_y(z)
        
    elif 2 < nargs <= 4:
        # 如果 nargs 为 3 或 4，则假定参数为 x, y, z 和其他可能的参数
        x, y, z_orig, *args = args
        # 检查并调整 x, y, z 的形状和类型
        x, y, z = self._check_xyz(x, y, z_orig, kwargs)
        
    else:
        # 如果 nargs 超出了预期范围，抛出参数数量错误
        raise _api.nargs_error(fn, takes="from 1 to 4", given=nargs)
    
    # 将 z 中的无效数据（如 NaN 或 Inf）标记为无效值
    z = ma.masked_invalid(z, copy=False)
    
    # 计算 z 的最大值和最小值，并将其转换为浮点数
    self.zmax = z.max().astype(float)
    self.zmin = z.min().astype(float)
    
    # 如果启用了对数刻度且 zmin 小于等于 0，则将 z 中小于等于 0 的值标记为无效
    if self.logscale and self.zmin <= 0:
        z = ma.masked_where(z <= 0, z)
        # 发出警告，说明在对数刻度下，已经将 z 中小于等于 0 的值屏蔽掉
        _api.warn_external('Log scale: values of z <= 0 have been masked')
        # 更新 zmin 的值为去除无效值后的最小值
        self.zmin = z.min().astype(float)
    
    # 处理轮廓绘制函数的其他参数，并根据 z 的数据类型进行处理
    self._process_contour_level_args(args, z.dtype)
    
    # 返回处理后的 x, y, z 数据元组
    return (x, y, z)
    
def _check_xyz(self, x, y, z, kwargs):
    """
    Check that the shapes of the input arrays match; if x and y are 1D,
    convert them to 2D using meshgrid.
    """
    # 使用 axes._process_unit_info 方法处理 x 和 y，确保它们的单位信息正确
    x, y = self.axes._process_unit_info([("x", x), ("y", y)], kwargs)
    
    # 将 x 和 y 转换为 float64 类型的 NumPy 数组
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # 将 z 转换为 MaskedArray 类型
    z = ma.asarray(z)
    
    # 检查 z 的维度是否为 2
    if z.ndim != 2:
        raise TypeError(f"Input z must be 2D, not {z.ndim}D")
    
    # 检查 z 的形状是否至少为 (2, 2)
    if z.shape[0] < 2 or z.shape[1] < 2:
        raise TypeError(f"Input z must be at least a (2, 2) shaped array, "
                        f"but has shape {z.shape}")
    
    # 获取 z 的行数 Ny 和列数 Nx
    Ny, Nx = z.shape
    
    # 检查 x 和 y 的维度是否相同
    if x.ndim != y.ndim:
        raise TypeError(f"Number of dimensions of x ({x.ndim}) and y "
                        f"({y.ndim}) do not match")
    
    # 如果 x 是 1D 数组，则获取其长度 nx 和 y 的长度 ny，并使用 meshgrid 函数生成对应的 2D 网格
    if x.ndim == 1:
        nx, = x.shape
        ny, = y.shape
        if nx != Nx:
            raise TypeError(f"Length of x ({nx}) must match number of "
                            f"columns in z ({Nx})")
        if ny != Ny:
            raise TypeError(f"Length of y ({ny}) must match number of "
                            f"rows in z ({Ny})")
        x, y = np.meshgrid(x, y)
    
    # 如果 x 是 2D 数组，则检查其形状是否与 z 相同
    elif x.ndim == 2:
        if x.shape != z.shape:
            raise TypeError(
                f"Shapes of x {x.shape} and z {z.shape} do not match")
        if y.shape != z.shape:
            raise TypeError(
                f"Shapes of y {y.shape} and z {z.shape} do not match")
    
    # 如果 x 的维度既不是 1D 也不是 2D，则抛出类型错误
    else:
        raise TypeError(f"Inputs x and y must be 1D or 2D, not {x.ndim}D")
    
    # 返回经过检查和可能调整后的 x, y, z 数据
    return x, y, z
    def _initialize_x_y(self, z):
        """
        Return X, Y arrays such that contour(Z) will match imshow(Z)
        if origin is not None.
        The center of pixel Z[i, j] depends on origin:
        if origin is None, x = j, y = i;
        if origin is 'lower', x = j + 0.5, y = i + 0.5;
        if origin is 'upper', x = j + 0.5, y = Nrows - i - 0.5
        If extent is not None, x and y will be scaled to match,
        as in imshow.
        If origin is None and extent is not None, then extent
        will give the minimum and maximum values of x and y.
        """
        # 检查 z 的维度是否为二维
        if z.ndim != 2:
            raise TypeError(f"Input z must be 2D, not {z.ndim}D")
        elif z.shape[0] < 2 or z.shape[1] < 2:
            # 检查 z 的形状是否至少为 (2, 2)
            raise TypeError(f"Input z must be at least a (2, 2) shaped array, "
                            f"but has shape {z.shape}")
        else:
            Ny, Nx = z.shape  # 获取 z 的行数和列数

        # 如果 self.origin 为 None，则返回未匹配图像的 X, Y 网格
        if self.origin is None:  # Not for image-matching.
            if self.extent is None:
                # 如果 extent 为 None，则返回基本网格
                return np.meshgrid(np.arange(Nx), np.arange(Ny))
            else:
                # 如果 extent 不为 None，则按照 extent 缩放 x 和 y
                x0, x1, y0, y1 = self.extent
                x = np.linspace(x0, x1, Nx)
                y = np.linspace(y0, y1, Ny)
                return np.meshgrid(x, y)

        # 匹配图像行为的情况
        if self.extent is None:
            x0, x1, y0, y1 = (0, Nx, 0, Ny)
        else:
            x0, x1, y0, y1 = self.extent

        # 计算网格的步长
        dx = (x1 - x0) / Nx
        dy = (y1 - y0) / Ny

        # 计算 x 和 y 的中心坐标
        x = x0 + (np.arange(Nx) + 0.5) * dx
        y = y0 + (np.arange(Ny) + 0.5) * dy

        # 如果 origin 为 'upper'，反转 y 坐标
        if self.origin == 'upper':
            y = y[::-1]

        return np.meshgrid(x, y)
# 定义模块 _docstring.interpd 中的 contour_doc 文档字符串，描述了 .contour 和 .contourf 函数的作用，
# 分别用于绘制等高线线条和填充的等高线。
_docstring.interpd.update(contour_doc="""
`.contour` and `.contourf` draw contour lines and filled contours,
respectively.  Except as noted, function signatures and return values
are the same for both versions.

Parameters
----------
X, Y : array-like, optional
    The coordinates of the values in *Z*.

    *X* and *Y* must both be 2D with the same shape as *Z* (e.g.
    created via `numpy.meshgrid`), or they must both be 1-D such
    that ``len(X) == N`` is the number of columns in *Z* and
    ``len(Y) == M`` is the number of rows in *Z*.

    *X* and *Y* must both be ordered monotonically.

    If not given, they are assumed to be integer indices, i.e.
    ``X = range(N)``, ``Y = range(M)``.

Z : (M, N) array-like
    The height values over which the contour is drawn.  Color-mapping is
    controlled by *cmap*, *norm*, *vmin*, and *vmax*.

levels : int or array-like, optional
    Determines the number and positions of the contour lines / regions.

    If an int *n*, use `~matplotlib.ticker.MaxNLocator`, which tries
    to automatically choose no more than *n+1* "nice" contour levels
    between minimum and maximum numeric values of *Z*.

    If array-like, draw contour lines at the specified levels.
    The values must be in increasing order.

Returns
-------
`~.contour.QuadContourSet`

Other Parameters
----------------
corner_mask : bool, default: :rc:`contour.corner_mask`
    Enable/disable corner masking, which only has an effect if *Z* is
    a masked array.  If ``False``, any quad touching a masked point is
    masked out.  If ``True``, only the triangular corners of quads
    nearest those points are always masked out, other triangular
    corners comprising three unmasked points are contoured as usual.

colors : :mpltype:`color` or list of :mpltype:`color`, optional
    The colors of the levels, i.e. the lines for `.contour` and the
    areas for `.contourf`.

    The sequence is cycled for the levels in ascending order. If the
    sequence is shorter than the number of levels, it's repeated.

    As a shortcut, single color strings may be used in place of
    one-element lists, i.e. ``'red'`` instead of ``['red']`` to color
    all levels with the same color. This shortcut does only work for
    color strings, not for other ways of specifying colors.

    By default (value *None*), the colormap specified by *cmap*
    will be used.

alpha : float, default: 1
    The alpha blending value, between 0 (transparent) and 1 (opaque).

%(cmap_doc)s

    This parameter is ignored if *colors* is set.

%(norm_doc)s

    This parameter is ignored if *colors* is set.

%(vmin_vmax_doc)s

    If *vmin* or *vmax* are not given, the default color scaling is based on
    *levels*.

    This parameter is ignored if *colors* is set.

origin : {*None*, 'upper', 'lower', 'image'}, default: None
    Determines the orientation and exact position of *Z* by specifying
    the position of ``Z[0, 0]``.  This is only relevant, if *X*, *Y*
    are not given.
""")
    - *None*: ``Z[0, 0]`` is at X=0, Y=0 in the lower left corner.
    # 当origin为None时，数组Z中的元素Z[0, 0]在图像坐标系中位于左下角的位置，即X=0，Y=0。

    - 'lower': ``Z[0, 0]`` is at X=0.5, Y=0.5 in the lower left corner.
    # 当origin为'lower'时，数组Z中的元素Z[0, 0]在图像坐标系中位于左下角的位置，即X=0.5，Y=0.5。

    - 'upper': ``Z[0, 0]`` is at X=N+0.5, Y=0.5 in the upper left corner.
    # 当origin为'upper'时，数组Z中的元素Z[0, 0]在图像坐标系中位于左上角的位置，即X=N+0.5，Y=0.5，其中N是数组Z的行数。

    - 'image': Use the value from :rc:`image.origin`.
    # 当origin为'image'时，使用rc参数image.origin中指定的值来确定数组Z中元素的位置。
extent : (x0, x1, y0, y1), optional
    # 如果 origin 参数不为 None，则 extent 被解释为像 imshow 函数一样：它表示外部像素边界。
    # 在这种情况下，Z[0, 0] 的位置是像素的中心，而不是角落。
    # 如果 origin 参数为 None，则 (x0, y0) 是 Z[0, 0] 的位置，(x1, y1) 是 Z[-1, -1] 的位置。

locator : ticker.Locator subclass, optional
    # 如果未显式指定 levels 参数，则 locator 用于确定等高线的级别。
    # 默认为 ~.ticker.MaxNLocator。

extend : {'neither', 'both', 'min', 'max'}, default: 'neither'
    # 决定超出 levels 范围的值在 contourf 中的着色方式。

    # 如果为 'neither'，超出 levels 范围的值不会被着色。
    # 如果为 'min'、'max' 或 'both'，则会着色低于、高于或低于和高于 levels 范围的值。

    # 小于 min(levels) 或大于 max(levels) 的值将映射到 colormap 的 under/over 值。
    # 大多数 colormap 默认没有专门的颜色用于这些值，因此 over 和 under 值是 colormap 的边界值。
    # 可以通过 .Colormap.set_under 和 .Colormap.set_over 显式设置这些值。

    # 注意：
    # 如果修改了 colormap 的属性，则现有的 .QuadContourSet 不会收到通知。
    # 因此，在修改 colormap 后需要显式调用 .QuadContourSet.changed()。
    # 如果给 .QuadContourSet 分配了 colorbar，则不需要显式调用，因为它会内部调用 .QuadContourSet.changed()。

    # 示例：
    # x = np.arange(1, 10)
    # y = x.reshape(-1, 1)
    # h = x * y

    # cs = plt.contourf(h, levels=[10, 30, 50],
    #     colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
    # cs.cmap.set_over('red')
    # cs.cmap.set_under('blue')
    # cs.changed()

xunits, yunits : registered units, optional
    # 通过指定 matplotlib.units.ConversionInterface 的实例，可以覆盖轴的单位。

antialiased : bool, optional
    # 启用抗锯齿，覆盖默认设置。
    # 对于填充的等高线，默认为 False。
    # 对于线条等高线，取决于 :rc:`lines.antialiased` 的设置。

nchunk : int >= 0, optional
    # 如果为 0，则不对域进行分割。
    # 指定正整数将域划分为 nchunk x nchunk 个四边形的子域。
    # 分块减少了等高线算法生成的多边形的最大长度，从而减少传递给后端的渲染工作量，也需要稍少的 RAM。
    # 然而，它可能会引入渲染工件，具体取决于后端、antialiased 标志和 alpha 的值。

linewidths : float or array-like, default: :rc:`contour.linewidth`
    # 等高线的线宽度，默认为 :rc:`contour.linewidth`。
    *Only applies to* `.contour`.

    The line width of the contour lines.

    If a number, all levels will be plotted with this linewidth.

    If a sequence, the levels in ascending order will be plotted with
    the linewidths in the order specified.

    If None, this falls back to :rc:`lines.linewidth`.
# 定义了可选参数 linestyles，用于指定轮廓线的样式，仅适用于 .contour 方法
linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, optional
    *Only applies to* `.contour`.

    如果 linestyles 是 None，则默认为 'solid'，除非轮廓线是单色的。
    在这种情况下，负的轮廓线样式将使用 negative_linestyles 参数指定。

    linestyles 也可以是上述字符串的可迭代对象，指定要使用的一组轮廓线样式。
    如果此可迭代对象比轮廓线级别的数量短，将会重复使用直至满足轮廓线级别的数量。

# 可选参数 negative_linestyles，仅适用于 .contour 方法
negative_linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, \
                       optional
    *Only applies to* `.contour`.

    如果 linestyles 是 None 并且轮廓线是单色的，此参数指定负轮廓线的线型样式。

    如果 negative_linestyles 是 None，则默认从 :rc:`contour.negative_linestyles` 中获取。

    negative_linestyles 也可以是上述字符串的可迭代对象，指定要使用的一组轮廓线样式。
    如果此可迭代对象比轮廓线级别的数量短，将会重复使用直至满足轮廓线级别的数量。

# 可选参数 hatches，仅适用于 .contourf 方法
hatches : list[str], optional
    *Only applies to* `.contourf`.

    用于填充区域的交叉填充图案列表。
    如果为 None，则轮廓填充区域不会添加任何填充图案。

# 可选参数 algorithm，指定计算轮廓线和多边形的轮廓算法
algorithm : {'mpl2005', 'mpl2014', 'serial', 'threaded'}, optional
    Which contouring algorithm to use to calculate the contour lines and
    polygons. The algorithms are implemented in
    `ContourPy <https://github.com/contourpy/contourpy>`_, consult the
    `ContourPy documentation <https://contourpy.readthedocs.io>`_ for
    further information.

    默认值从 :rc:`contour.algorithm` 中获取。

# 可选参数 clip_path，用于设置剪切路径
clip_path : `~matplotlib.patches.Patch` or `.Path` or `.TransformedPath`
    Set the clip path.  See `~matplotlib.artist.Artist.set_clip_path`.

    .. versionadded:: 3.8

# 可选参数 data，索引对象，占位符
data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER

Notes
-----
1. `.contourf` 与 MATLAB 版本不同，不会绘制多边形边缘。要绘制边缘，请使用 `.contour` 调用添加线轮廓。
   
2. `.contourf` 填充了顶部封闭的区间；即对于边界 *z1* 和 *z2*，填充的区域为::

      z1 < Z <= z2

   最低的区间除外，它在两侧都封闭（即包括最低值）。

3. `.contour` 和 `.contourf` 使用 `Marching Squares <https://en.wikipedia.org/wiki/Marching_squares>`_ 算法计算轮廓位置。
   更多信息请参考 `ContourPy documentation <https://contourpy.readthedocs.io>`_。
```