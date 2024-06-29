# `D:\src\scipysrc\matplotlib\lib\matplotlib\legend_handler.py`

```
"""
Default legend handlers.

.. important::

    This is a low-level legend API, which most end users do not need.

    We recommend that you are familiar with the :ref:`legend guide
    <legend_guide>` before reading this documentation.

Legend handlers are expected to be a callable object with a following
signature::

    legend_handler(legend, orig_handle, fontsize, handlebox)

Where *legend* is the legend itself, *orig_handle* is the original
plot, *fontsize* is the fontsize in pixels, and *handlebox* is an
`.OffsetBox` instance. Within the call, you should create relevant
artists (using relevant properties from the *legend* and/or
*orig_handle*) and add them into the *handlebox*. The artists need to
be scaled according to the *fontsize* (note that the size is in pixels,
i.e., this is dpi-scaled value).

This module includes definition of several legend handler classes
derived from the base class (HandlerBase) with the following method::

    def legend_artist(self, legend, orig_handle, fontsize, handlebox)
"""

from itertools import cycle  # 导入cycle函数，用于生成循环迭代器

import numpy as np  # 导入NumPy库，命名为np

from matplotlib import cbook  # 导入matplotlib中的cbook模块
from matplotlib.lines import Line2D  # 导入matplotlib中的Line2D类
from matplotlib.patches import Rectangle  # 导入matplotlib中的Rectangle类
import matplotlib.collections as mcoll  # 导入matplotlib中的mcoll模块


def update_from_first_child(tgt, src):
    """
    Update properties of target from its first child.

    Parameters
    ----------
    tgt : object
        Target object whose properties will be updated.
    src : object
        Source object from which properties are fetched.

    """
    first_child = next(iter(src.get_children()), None)  # 获取src对象的第一个子对象
    if first_child is not None:
        tgt.update_from(first_child)  # 更新tgt对象的属性从第一个子对象


class HandlerBase:
    """
    A base class for default legend handlers.

    The derived classes are meant to override *create_artists* method, which
    has the following signature::

      def create_artists(self, legend, orig_handle,
                         xdescent, ydescent, width, height, fontsize,
                         trans):

    The overridden method needs to create artists of the given
    transform that fits in the given dimension (xdescent, ydescent,
    width, height) that are scaled by fontsize if necessary.

    """
    def __init__(self, xpad=0., ypad=0., update_func=None):
        """
        Parameters
        ----------
        xpad : float, optional
            Padding in x-direction.
        ypad : float, optional
            Padding in y-direction.
        update_func : callable, optional
            Function for updating the legend handler properties from another
            legend handler, used by `~HandlerBase.update_prop`.
        """
        self._xpad, self._ypad = xpad, ypad  # 初始化xpad和ypad属性
        self._update_prop_func = update_func  # 初始化update_prop_func属性

    def _update_prop(self, legend_handle, orig_handle):
        """
        Update properties of legend handler from original handle.

        Parameters
        ----------
        legend_handle : object
            Legend handler object to update.
        orig_handle : object
            Original handle from which properties are fetched.

        """
        if self._update_prop_func is None:
            self._default_update_prop(legend_handle, orig_handle)
        else:
            self._update_prop_func(legend_handle, orig_handle)

    def _default_update_prop(self, legend_handle, orig_handle):
        """
        Default method to update properties of legend handler.

        Parameters
        ----------
        legend_handle : object
            Legend handler object to update.
        orig_handle : object
            Original handle from which properties are fetched.

        """
        legend_handle.update_from(orig_handle)  # 使用orig_handle更新legend_handle的属性
    # 更新图例句柄的属性
    def update_prop(self, legend_handle, orig_handle, legend):
        # 调用对象内部方法，更新图例句柄的属性
        self._update_prop(legend_handle, orig_handle)

        # 设置图例对象的艺术家属性为传入的图例句柄的属性
        legend._set_artist_props(legend_handle)
        # 清除图例句柄的剪裁框
        legend_handle.set_clip_box(None)
        # 清除图例句柄的剪裁路径
        legend_handle.set_clip_path(None)

    # 调整绘图区域的大小和位置
    def adjust_drawing_area(self, legend, orig_handle,
                            xdescent, ydescent, width, height, fontsize):
        # 根据内部属性调整 x 轴下降
        xdescent = xdescent - self._xpad * fontsize
        # 根据内部属性调整 y 轴下降
        ydescent = ydescent - self._ypad * fontsize
        # 根据内部属性调整宽度
        width = width - self._xpad * fontsize
        # 根据内部属性调整高度
        height = height - self._ypad * fontsize
        # 返回调整后的参数
        return xdescent, ydescent, width, height

    # 创建图例的艺术家对象
    def legend_artist(self, legend, orig_handle,
                      fontsize, handlebox):
        """
        返回此 HandlerBase 为给定原始艺术家/句柄生成的艺术家。

        Parameters
        ----------
        legend : `~matplotlib.legend.Legend`
            正在创建这些图例艺术家的图例。
        orig_handle : :class:`matplotlib.artist.Artist` or similar
            正在创建这些图例艺术家的对象。
        fontsize : int
            像素大小的字体。正在创建的艺术家应根据给定的字体大小进行缩放。
        handlebox : `~matplotlib.offsetbox.OffsetBox`
            已创建以容纳此图例条目的艺术家的框。在此方法内部必须将在 `legend_artist` 方法中创建的艺术家添加到此 handlebox 中。

        """
        # 调整绘图区域大小和位置
        xdescent, ydescent, width, height = self.adjust_drawing_area(
                 legend, orig_handle,
                 handlebox.xdescent, handlebox.ydescent,
                 handlebox.width, handlebox.height,
                 fontsize)
        # 创建艺术家对象列表
        artists = self.create_artists(legend, orig_handle,
                                      xdescent, ydescent, width, height,
                                      fontsize, handlebox.get_transform())

        # create_artists 方法将返回一个艺术家对象列表。
        for a in artists:
            # 将每个艺术家对象添加到 handlebox 中
            handlebox.add_artist(a)

        # 我们只返回第一个艺术家对象
        return artists[0]
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        """
        Return the legend artists generated.

        Parameters
        ----------
        legend : `~matplotlib.legend.Legend`
            表示正在创建图例艺术元素的图例对象。
        orig_handle : `~matplotlib.artist.Artist` or similar
            要为其创建图例艺术元素的对象。
        xdescent, ydescent, width, height : int
            描述图例艺术元素应适应其中的矩形区域（*xdescent*、*ydescent*、*width*、*height*）。
        fontsize : int
            以像素为单位的字体大小。应根据给定的字体大小缩放创建的图例艺术元素。
        trans : `~matplotlib.transforms.Transform`
            应用于正在创建的图例艺术元素的变换。通常从处理程序框中的单位坐标到屏幕坐标。

        Raises
        ------
        NotImplementedError
            派生类必须覆盖这个方法。
        """
        raise NotImplementedError('Derived must override')
# 定义一个名为 HandlerNpoints 的类，继承自 HandlerBase 类，用于处理图例条目，显示指定数量的点数。
class HandlerNpoints(HandlerBase):
    """
    A legend handler that shows *numpoints* points in the legend entry.
    """

    # 初始化方法，接受 marker_pad（点之间的间距）、numpoints（要在图例条目中显示的点数）和其他关键字参数
    def __init__(self, marker_pad=0.3, numpoints=None, **kwargs):
        """
        Parameters
        ----------
        marker_pad : float
            Padding between points in legend entry.
            图例条目中点之间的间距。
        numpoints : int
            Number of points to show in legend entry.
            图例条目中要显示的点的数量。
        **kwargs
            Keyword arguments forwarded to `.HandlerBase`.
            转发给 `.HandlerBase` 的关键字参数。
        """
        # 调用父类 HandlerBase 的初始化方法，传入所有关键字参数
        super().__init__(**kwargs)

        # 设置实例变量 _numpoints 为传入的 numpoints 参数
        self._numpoints = numpoints
        # 设置实例变量 _marker_pad 为传入的 marker_pad 参数
        self._marker_pad = marker_pad

    # 获取要显示的点数
    def get_numpoints(self, legend):
        if self._numpoints is None:
            return legend.numpoints
        else:
            return self._numpoints

    # 计算 x 轴数据
    def get_xdata(self, legend, xdescent, ydescent, width, height, fontsize):
        # 获取要显示的点数
        numpoints = self.get_numpoints(legend)
        if numpoints > 1:
            # 如果点数大于1，为了补偿标记的大小，在这里添加一些填充
            pad = self._marker_pad * fontsize
            # 在 x 轴上均匀分布点的位置
            xdata = np.linspace(-xdescent + pad,
                                -xdescent + width - pad,
                                numpoints)
            xdata_marker = xdata
        else:
            # 如果点数为1，将点的位置设置为线的两端
            xdata = [-xdescent, -xdescent + width]
            xdata_marker = [-xdescent + 0.5 * width]
        return xdata, xdata_marker


# 定义一个名为 HandlerNpointsYoffsets 的类，继承自 HandlerNpoints 类，用于显示在图例中显示的点，并允许它们在 y 方向上单独偏移。
class HandlerNpointsYoffsets(HandlerNpoints):
    """
    A legend handler that shows *numpoints* in the legend, and allows them to
    be individually offset in the y-direction.
    """

    # 初始化方法，接受 numpoints（图例条目中要显示的点数）、yoffsets（每个点在 y 方向上的偏移列表）和其他关键字参数
    def __init__(self, numpoints=None, yoffsets=None, **kwargs):
        """
        Parameters
        ----------
        numpoints : int
            Number of points to show in legend entry.
            图例条目中要显示的点的数量。
        yoffsets : array of floats
            Length *numpoints* list of y offsets for each point in
            legend entry.
            每个点在图例条目中 y 方向上的偏移量列表，长度为 *numpoints*。
        **kwargs
            Keyword arguments forwarded to `.HandlerNpoints`.
            转发给 `.HandlerNpoints` 的关键字参数。
        """
        # 调用父类 HandlerNpoints 的初始化方法，传入 numpoints 和其他关键字参数
        super().__init__(numpoints=numpoints, **kwargs)
        # 设置实例变量 _yoffsets 为传入的 yoffsets 参数
        self._yoffsets = yoffsets

    # 计算 y 轴数据
    def get_ydata(self, legend, xdescent, ydescent, width, height, fontsize):
        if self._yoffsets is None:
            # 如果 yoffsets 未设置，则使用 legend._scatteryoffsets 来计算 y 轴数据
            ydata = height * legend._scatteryoffsets
        else:
            # 否则，将 yoffsets 转换为 numpy 数组，并计算 y 轴数据
            ydata = height * np.asarray(self._yoffsets)

        return ydata


# 定义一个名为 HandlerLine2DCompound 的类，继承自 HandlerNpoints 类，用于处理 `.Line2D` 实例，通过结合仅具有线条的艺术家和仅具有标记的艺术家来实现。
class HandlerLine2DCompound(HandlerNpoints):
    """
    Original handler for `.Line2D` instances, that relies on combining
    a line-only with a marker-only artist.  May be deprecated in the future.
    """
    # 创建艺术家对象，用于图例显示，继承自父类的文档字符串
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        # 获取图例的 x 数据和标记 x 数据
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        # 创建一个与 xdata 相同长度的全为 ((height - ydescent) / 2) 的数组作为 y 数据
        ydata = np.full_like(xdata, ((height - ydescent) / 2))
        
        # 创建一个 Line2D 对象 legline，以 xdata 和 ydata 作为数据
        legline = Line2D(xdata, ydata)

        # 更新 legline 的属性，包括原始句柄和图例
        self.update_prop(legline, orig_handle, legend)
        
        # 设置 legline 的绘制风格为 'default'
        legline.set_drawstyle('default')
        
        # 设置 legline 的标记为无
        legline.set_marker("")

        # 使用 xdata_marker 和前面创建的 ydata 的前 xdata_marker 长度部分创建 Line2D 对象 legline_marker
        legline_marker = Line2D(xdata_marker, ydata[:len(xdata_marker)])
        
        # 更新 legline_marker 的属性，包括原始句柄和图例
        self.update_prop(legline_marker, orig_handle, legend)
        
        # 设置 legline_marker 的线型为 'None'
        legline_marker.set_linestyle('None')
        
        # 如果图例的标记缩放比例不为 1，则更新 legline_marker 的标记大小
        if legend.markerscale != 1:
            newsz = legline_marker.get_markersize() * legend.markerscale
            legline_marker.set_markersize(newsz)
        
        # 将 legline_marker 作为 legline 的属性 _legmarker，不将其添加到返回列表中，
        # 因为假设文本和句柄应一一对应。
        legline._legmarker = legline_marker

        # 设置 legline 和 legline_marker 的变换
        legline.set_transform(trans)
        legline_marker.set_transform(trans)

        # 返回包含 legline 和 legline_marker 的列表
        return [legline, legline_marker]
class HandlerLine2D(HandlerNpoints):
    """
    Handler for `.Line2D` instances.

    See Also
    --------
    HandlerLine2DCompound : An earlier handler implementation, which used one
                            artist for the line and another for the marker(s).
    """

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        # 获取 x 轴数据和标记点数据，继承自父类的文档字符串
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        markevery = None
        if self.get_numpoints(legend) == 1:
            # 特殊情况：只有一个标记点，将在中心处放置一个标记点，
            # 并且线会延伸到两侧。使用一个包含 3 个点的线段，
            # 但只标记第二个点（即中间点）。
            xdata = np.linspace(xdata[0], xdata[-1], 3)
            markevery = [1]

        # 创建 y 轴数据，使得线段位于图例框中心
        ydata = np.full_like(xdata, (height - ydescent) / 2)
        # 创建 Line2D 对象
        legline = Line2D(xdata, ydata, markevery=markevery)

        # 更新 Line2D 对象的属性
        self.update_prop(legline, orig_handle, legend)

        # 如果图例的标记比例不为 1，调整标记点的大小
        if legend.markerscale != 1:
            newsz = legline.get_markersize() * legend.markerscale
            legline.set_markersize(newsz)

        # 设置 Line2D 对象的变换
        legline.set_transform(trans)

        # 返回 Line2D 对象的列表形式
        return [legline]


class HandlerPatch(HandlerBase):
    """
    Handler for `.Patch` instances.
    """

    def __init__(self, patch_func=None, **kwargs):
        """
        Parameters
        ----------
        patch_func : callable, optional
            The function that creates the legend key artist.
            *patch_func* should have the signature::

                def patch_func(legend=legend, orig_handle=orig_handle,
                               xdescent=xdescent, ydescent=ydescent,
                               width=width, height=height, fontsize=fontsize)

            Subsequently, the created artist will have its ``update_prop``
            method called and the appropriate transform will be applied.

        **kwargs
            Keyword arguments forwarded to `.HandlerBase`.
        """
        # 调用父类构造函数，传递关键字参数
        super().__init__(**kwargs)
        # 存储 patch_func 函数的引用
        self._patch_func = patch_func

    def _create_patch(self, legend, orig_handle,
                      xdescent, ydescent, width, height, fontsize):
        # 如果 patch_func 未提供，创建一个矩形 Patch 对象
        if self._patch_func is None:
            p = Rectangle(xy=(-xdescent, -ydescent),
                          width=width, height=height)
        else:
            # 否则，使用提供的 patch_func 创建 Patch 对象
            p = self._patch_func(legend=legend, orig_handle=orig_handle,
                                 xdescent=xdescent, ydescent=ydescent,
                                 width=width, height=height, fontsize=fontsize)
        return p
    # 定义一个方法用于在图例中创建艺术家对象
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # 调用私有方法 `_create_patch` 创建一个艺术家对象 `p`
        p = self._create_patch(legend, orig_handle,
                               xdescent, ydescent, width, height, fontsize)
        # 调用 `update_prop` 方法更新艺术家对象 `p` 的属性
        self.update_prop(p, orig_handle, legend)
        # 设置艺术家对象 `p` 的变换矩阵为 `trans`
        p.set_transform(trans)
        # 返回包含艺术家对象 `p` 的列表
        return [p]
class HandlerStepPatch(HandlerBase):
    """
    Handler for `~.matplotlib.patches.StepPatch` instances.
    """

    @staticmethod
    def _create_patch(orig_handle, xdescent, ydescent, width, height):
        # 创建 StepPatch 的矩形对象，位置在左下角，大小由给定的参数确定
        return Rectangle(xy=(-xdescent, -ydescent), width=width,
                         height=height, color=orig_handle.get_facecolor())

    @staticmethod
    def _create_line(orig_handle, width, height):
        # 当 StepPatch 未填充时，显示为线条
        legline = Line2D([0, width], [height/2, height/2],
                         color=orig_handle.get_edgecolor(),
                         linestyle=orig_handle.get_linestyle(),
                         linewidth=orig_handle.get_linewidth(),
                         )
        # 手动设置绘图风格，因为图形和线条的属性不兼容
        legline.set_drawstyle('default')
        legline.set_marker("")
        return legline

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # 继承的文档字符串
        if orig_handle.get_fill() or (orig_handle.get_hatch() is not None):
            # 如果 StepPatch 是填充的或有阴影，则创建填充对象
            p = self._create_patch(orig_handle, xdescent, ydescent, width,
                                   height)
            self.update_prop(p, orig_handle, legend)
        else:
            # 否则创建线条对象
            p = self._create_line(orig_handle, width, height)
        p.set_transform(trans)
        return [p]


class HandlerLineCollection(HandlerLine2D):
    """
    Handler for `.LineCollection` instances.
    """

    def get_numpoints(self, legend):
        # 如果未设置点数，则返回图例的散点数
        if self._numpoints is None:
            return legend.scatterpoints
        else:
            return self._numpoints

    def _default_update_prop(self, legend_handle, orig_handle):
        # 更新图例的颜色、线型和线宽属性
        lw = orig_handle.get_linewidths()[0]
        dashes = orig_handle._us_linestyles[0]
        color = orig_handle.get_colors()[0]
        legend_handle.set_color(color)
        legend_handle.set_linestyle(dashes)
        legend_handle.set_linewidth(lw)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # 继承的文档字符串
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        # 创建 Line2D 对象，位置和数据根据给定参数确定
        ydata = np.full_like(xdata, (height - ydescent) / 2)
        legline = Line2D(xdata, ydata)

        self.update_prop(legline, orig_handle, legend)
        legline.set_transform(trans)

        return [legline]


class HandlerRegularPolyCollection(HandlerNpointsYoffsets):
    r"""Handler for `.RegularPolyCollection`\s."""

    def __init__(self, yoffsets=None, sizes=None, **kwargs):
        super().__init__(yoffsets=yoffsets, **kwargs)

        self._sizes = sizes

    def get_numpoints(self, legend):
        # 如果未设置点数，则返回图例的散点数
        if self._numpoints is None:
            return legend.scatterpoints
        else:
            return self._numpoints
    # 获取图例项的标记尺寸
    def get_sizes(self, legend, orig_handle,
                  xdescent, ydescent, width, height, fontsize):
        # 如果尺寸列表为空，则获取原始图例项的尺寸列表，若为空则默认为1
        if self._sizes is None:
            handle_sizes = orig_handle.get_sizes()
            if not len(handle_sizes):
                handle_sizes = [1]
            # 计算标记尺寸的最大和最小值
            size_max = max(handle_sizes) * legend.markerscale ** 2
            size_min = min(handle_sizes) * legend.markerscale ** 2

            # 获取图例项中的标记点数量
            numpoints = self.get_numpoints(legend)
            # 根据标记点数量选择尺寸值
            if numpoints < 4:
                sizes = [.5 * (size_max + size_min), size_max,
                         size_min][:numpoints]
            else:
                rng = (size_max - size_min)
                # 使用线性间隔生成标记点尺寸
                sizes = rng * np.linspace(0, 1, numpoints) + size_min
        else:
            sizes = self._sizes

        return sizes

    # 更新图例项的属性
    def update_prop(self, legend_handle, orig_handle, legend):
        # 调用内部方法更新属性
        self._update_prop(legend_handle, orig_handle)

        # 设置图例项的图形
        legend_handle.set_figure(legend.figure)
        # legend._set_artist_props(legend_handle)
        # 清除图例项的剪辑框
        legend_handle.set_clip_box(None)
        # 清除图例项的剪辑路径
        legend_handle.set_clip_path(None)

    # 创建集合对象
    def create_collection(self, orig_handle, sizes, offsets, offset_transform):
        # 根据原始图例项的类型创建新的集合对象
        return type(orig_handle)(
            orig_handle.get_numsides(),
            rotation=orig_handle.get_rotation(), sizes=sizes,
            offsets=offsets, offset_transform=offset_transform,
        )

    # 创建艺术家对象
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        # 继承的文档字符串
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        # 获取 y 数据
        ydata = self.get_ydata(legend, xdescent, ydescent,
                               width, height, fontsize)

        # 获取标记点尺寸
        sizes = self.get_sizes(legend, orig_handle, xdescent, ydescent,
                               width, height, fontsize)

        # 创建集合对象并设置其属性
        p = self.create_collection(
            orig_handle, sizes,
            offsets=list(zip(xdata_marker, ydata)), offset_transform=trans)

        # 更新集合对象的属性
        self.update_prop(p, orig_handle, legend)
        # 设置偏移变换
        p.set_offset_transform(trans)
        return [p]
class HandlerPathCollection(HandlerRegularPolyCollection):
    r"""Handler for `.PathCollection`\s, which are used by `~.Axes.scatter`."""

    def create_collection(self, orig_handle, sizes, offsets, offset_transform):
        # 创建一个新的 PathCollection 对象，包含原始 handle 的第一个路径
        return type(orig_handle)(
            [orig_handle.get_paths()[0]], sizes=sizes,
            offsets=offsets, offset_transform=offset_transform,
        )


class HandlerCircleCollection(HandlerRegularPolyCollection):
    r"""Handler for `.CircleCollection`\s."""

    def create_collection(self, orig_handle, sizes, offsets, offset_transform):
        # 创建一个新的 CircleCollection 对象，使用给定的 sizes, offsets 和 offset_transform
        return type(orig_handle)(
            sizes, offsets=offsets, offset_transform=offset_transform)


class HandlerErrorbar(HandlerLine2D):
    """Handler for Errorbars."""

    def __init__(self, xerr_size=0.5, yerr_size=None,
                 marker_pad=0.3, numpoints=None, **kwargs):
        # 初始化 Errorbar 处理器，设置 xerr_size, yerr_size, marker_pad 和 numpoints
        self._xerr_size = xerr_size
        self._yerr_size = yerr_size

        super().__init__(marker_pad=marker_pad, numpoints=numpoints, **kwargs)

    def get_err_size(self, legend, xdescent, ydescent,
                     width, height, fontsize):
        # 计算误差条的大小，根据 xerr_size 和 yerr_size 乘以 fontsize
        xerr_size = self._xerr_size * fontsize

        if self._yerr_size is None:
            yerr_size = xerr_size
        else:
            yerr_size = self._yerr_size * fontsize

        return xerr_size, yerr_size

class HandlerStem(HandlerNpointsYoffsets):
    """
    Handler for plots produced by `~.Axes.stem`.
    """

    def __init__(self, marker_pad=0.3, numpoints=None,
                 bottom=None, yoffsets=None, **kwargs):
        """
        Parameters
        ----------
        marker_pad : float, default: 0.3
            Padding between points in legend entry.
        numpoints : int, optional
            Number of points to show in legend entry.
        bottom : float, optional

        yoffsets : array of floats, optional
            Length *numpoints* list of y offsets for each point in
            legend entry.
        **kwargs
            Keyword arguments forwarded to `.HandlerNpointsYoffsets`.
        """
        # 初始化 Stem 图的处理器，设置 marker_pad, numpoints, yoffsets 和其他参数
        super().__init__(marker_pad=marker_pad, numpoints=numpoints,
                         yoffsets=yoffsets, **kwargs)
        self._bottom = bottom

    def get_ydata(self, legend, xdescent, ydescent, width, height, fontsize):
        # 获取 Y 数据用于 legend，根据 yoffsets 或者默认计算高度的一半
        if self._yoffsets is None:
            ydata = height * (0.5 * legend._scatteryoffsets + 0.5)
        else:
            ydata = height * np.asarray(self._yoffsets)

        return ydata
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        # 继承的文档字符串

        # 解构元组 orig_handle，获取 markerline, stemlines, baseline
        markerline, stemlines, baseline = orig_handle
        
        # 检查 stemlines 是否是 LineCollection 类型的实例
        using_linecoll = isinstance(stemlines, mcoll.LineCollection)
        
        # 调用 get_xdata 方法获取 xdata 和 xdata_marker
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        
        # 调用 get_ydata 方法获取 ydata
        ydata = self.get_ydata(legend, xdescent, ydescent,
                               width, height, fontsize)
        
        # 如果 self._bottom 为 None，则将 bottom 设置为 0.0，否则使用 self._bottom 的值
        if self._bottom is None:
            bottom = 0.
        else:
            bottom = self._bottom
        
        # 创建 Line2D 对象 leg_markerline，用于图例中的标记线
        leg_markerline = Line2D(xdata_marker, ydata[:len(xdata_marker)])
        # 调用 update_prop 方法更新 leg_markerline 的属性，传入 markerline 和 legend
        self.update_prop(leg_markerline, markerline, legend)
        
        # 创建 leg_stemlines 列表，每个元素是一个 Line2D 对象，用于图例中的线条
        leg_stemlines = [Line2D([x, x], [bottom, y])
                         for x, y in zip(xdata_marker, ydata)]
        
        # 如果 using_linecoll 为 True，则进入条件语句
        if using_linecoll:
            # 使用 cbook._setattr_cm 上下文管理器，设置 _update_prop_func 为 self._copy_collection_props 方法
            with cbook._setattr_cm(
                    self, _update_prop_func=self._copy_collection_props):
                # 遍历 leg_stemlines 列表，对每个 Line2D 对象调用 update_prop 方法更新属性，传入 stemlines 和 legend
                for line in leg_stemlines:
                    self.update_prop(line, stemlines, legend)
        
        # 否则，进入 else 分支
        else:
            # 遍历 leg_stemlines 和 stemlines，对应位置的元素调用 update_prop 方法更新属性
            for lm, m in zip(leg_stemlines, stemlines):
                self.update_prop(lm, m, legend)
        
        # 创建 leg_baseline 对象，用于图例中的基线
        leg_baseline = Line2D([np.min(xdata), np.max(xdata)],
                              [bottom, bottom])
        # 调用 update_prop 方法更新 leg_baseline 的属性，传入 baseline 和 legend
        self.update_prop(leg_baseline, baseline, legend)
        
        # 创建 artists 列表，包含 leg_stemlines、leg_baseline 和 leg_markerline
        artists = [*leg_stemlines, leg_baseline, leg_markerline]
        # 遍历 artists 列表，设置每个元素的 transform 属性为 trans
        for artist in artists:
            artist.set_transform(trans)
        
        # 返回 artists 列表作为函数的结果
        return artists

    def _copy_collection_props(self, legend_handle, orig_handle):
        """
        Copy properties from the `.LineCollection` *orig_handle* to the
        `.Line2D` *legend_handle*.
        """
        # 将 orig_handle 的颜色和线型属性复制给 legend_handle
        legend_handle.set_color(orig_handle.get_color()[0])
        legend_handle.set_linestyle(orig_handle.get_linestyle()[0])
class HandlerTuple(HandlerBase):
    """
    Handler for Tuple.
    """

    def __init__(self, ndivide=1, pad=None, **kwargs):
        """
        Parameters
        ----------
        ndivide : int or None, default: 1
            The number of sections to divide the legend area into.  If None,
            use the length of the input tuple.
        pad : float, default: :rc:`legend.borderpad`
            Padding in units of fraction of font size.
        **kwargs
            Keyword arguments forwarded to `.HandlerBase`.
        """
        # 初始化方法，设置处理器的属性
        self._ndivide = ndivide  # 设置要分割的部分数目
        self._pad = pad  # 设置填充值
        super().__init__(**kwargs)  # 调用父类的初始化方法，传递其他参数

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        # docstring inherited
        handler_map = legend.get_legend_handler_map()  # 获取图例的处理器映射表

        if self._ndivide is None:
            ndivide = len(orig_handle)  # 如果未指定分割数目，则使用原始处理对象的长度
        else:
            ndivide = self._ndivide  # 否则使用初始化时设置的分割数目

        if self._pad is None:
            pad = legend.borderpad * fontsize  # 如果未指定填充值，使用图例的边框填充乘以字体大小作为填充
        else:
            pad = self._pad * fontsize  # 否则使用初始化时设置的填充值乘以字体大小作为填充

        if ndivide > 1:
            width = (width - pad * (ndivide - 1)) / ndivide  # 如果分割数目大于1，重新计算宽度

        xds_cycle = cycle(xdescent - (width + pad) * np.arange(ndivide))  # 创建 x 坐标的循环迭代器

        a_list = []
        for handle1 in orig_handle:
            handler = legend.get_legend_handler(handler_map, handle1)  # 获取处理器对象
            _a_list = handler.create_artists(
                legend, handle1,
                next(xds_cycle), ydescent, width, height, fontsize, trans)  # 调用处理器的创建艺术对象方法
            a_list.extend(_a_list)  # 将创建的艺术对象列表扩展到结果列表中

        return a_list  # 返回创建的艺术对象列表


class HandlerPolyCollection(HandlerBase):
    """
    Handler for `.PolyCollection` used in `~.Axes.fill_between` and
    `~.Axes.stackplot`.
    """
    def _update_prop(self, legend_handle, orig_handle):
        def first_color(colors):
            # 如果颜色数组为空，则返回全透明的颜色
            if colors.size == 0:
                return (0, 0, 0, 0)
            # 否则返回颜色数组的第一个颜色，转换为元组格式
            return tuple(colors[0])

        def get_first(prop_array):
            # 如果属性数组非空，返回第一个属性；否则返回 None
            if len(prop_array):
                return prop_array[0]
            else:
                return None

        # orig_handle 是 PolyCollection 类型，legend_handle 是 Patch 类型。
        # 直接设置 Patch 的颜色属性（必须是 RGBA 元组）。
        legend_handle._facecolor = first_color(orig_handle.get_facecolor())
        legend_handle._edgecolor = first_color(orig_handle.get_edgecolor())
        legend_handle._original_facecolor = orig_handle._original_facecolor
        legend_handle._original_edgecolor = orig_handle._original_edgecolor
        legend_handle._fill = orig_handle.get_fill()
        legend_handle._hatch = orig_handle.get_hatch()
        # _hatch_color 的获取方式与其他属性不同。
        legend_handle._hatch_color = orig_handle._hatch_color
        # 剩余属性使用 setter 方法设置。
        legend_handle.set_linewidth(get_first(orig_handle.get_linewidths()))
        legend_handle.set_linestyle(get_first(orig_handle.get_linestyles()))
        legend_handle.set_transform(get_first(orig_handle.get_transforms()))
        legend_handle.set_figure(orig_handle.get_figure())
        # alpha 已经通过颜色属性考虑在内。

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # 继承的文档字符串
        # 创建一个 Rectangle 对象，位置为 (-xdescent, -ydescent)，宽度和高度分别为给定的 width 和 height
        p = Rectangle(xy=(-xdescent, -ydescent),
                      width=width, height=height)
        # 使用 update_prop 方法更新属性
        self.update_prop(p, orig_handle, legend)
        # 设置变换
        p.set_transform(trans)
        # 返回包含 p 的列表
        return [p]
```