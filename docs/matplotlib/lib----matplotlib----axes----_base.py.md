# `D:\src\scipysrc\matplotlib\lib\matplotlib\axes\_base.py`

```
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import re
import types

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms

_log = logging.getLogger(__name__)

# Helper class to generate Axes methods wrapping Axis methods.
class _axis_method_wrapper:
    """
    Helper to generate Axes methods wrapping Axis methods.

    After ::

        get_foo = _axis_method_wrapper("xaxis", "get_bar")

    (in the body of a class) ``get_foo`` is a method that forwards it arguments
    to the ``get_bar`` method of the ``xaxis`` attribute, and gets its
    signature and docstring from ``Axis.get_bar``.

    The docstring of ``get_foo`` is built by replacing "this Axis" by "the
    {attr_name}" (i.e., "the xaxis", "the yaxis") in the wrapped method's
    dedented docstring; additional replacements can be given in *doc_sub*.
    """

    def __init__(self, attr_name, method_name, *, doc_sub=None):
        self.attr_name = attr_name
        self.method_name = method_name
        # Immediately put the docstring in ``self.__doc__`` so that docstring
        # manipulations within the class body work as expected.
        doc = inspect.getdoc(getattr(maxis.Axis, method_name))
        self._missing_subs = []
        if doc:
            doc_sub = {"this Axis": f"the {self.attr_name}", **(doc_sub or {})}
            for k, v in doc_sub.items():
                if k not in doc:  # Delay raising error until we know qualname.
                    self._missing_subs.append(k)
                doc = doc.replace(k, v)
        self.__doc__ = doc
    def __set_name__(self, owner, name):
        # 这个方法在类体结束时被调用，作为 ``self.__set_name__(cls, name_under_which_self_is_assigned)``；
        # 我们依赖这一点来确保包装器拥有正确的 __name__/__qualname__。
        
        # 使用 attrgetter 获取特定属性的方法，属性名由 self.attr_name 和 self.method_name 构成
        get_method = attrgetter(f"{self.attr_name}.{self.method_name}")

        # 定义一个包装器函数，将调用转发给获取的方法
        def wrapper(self, *args, **kwargs):
            return get_method(self)(*args, **kwargs)

        # 设置包装器函数的模块名为 owner 的模块名
        wrapper.__module__ = owner.__module__
        # 设置包装器函数的名称为 name
        wrapper.__name__ = name
        # 设置包装器函数的限定名称为 owner 的限定名称加上 name
        wrapper.__qualname__ = f"{owner.__qualname__}.{name}"
        # 将包装器函数的文档字符串设置为当前对象的文档字符串
        wrapper.__doc__ = self.__doc__
        
        # 手动复制方法签名，而不是使用 functools.wraps，因为在请求 Axes 方法源代码时，
        # 显示 Axis 方法源代码会令人困惑。
        wrapper.__signature__ = inspect.signature(
            getattr(maxis.Axis, self.method_name))

        # 如果存在未找到的替代项，则引发 ValueError 异常
        if self._missing_subs:
            raise ValueError(
                "The definition of {} expected that the docstring of Axis.{} "
                "contains {!r} as substrings".format(
                    wrapper.__qualname__, self.method_name,
                    ", ".join(map(repr, self._missing_subs))))

        # 将包装器函数设置为 owner 类的属性，名称为 name
        setattr(owner, name, wrapper)
    """
    Axes locator for `.Axes.inset_axes` and similarly positioned Axes.

    The locator is a callable object used in `.Axes.set_aspect` to compute the
    Axes location depending on the renderer.
    """

    def __init__(self, bounds, transform):
        """
        *bounds* (a ``[l, b, w, h]`` rectangle) and *transform* together
        specify the position of the inset Axes.
        """
        # 初始化函数，设置对象的边界和变换信息
        self._bounds = bounds
        self._transform = transform

    def __call__(self, ax, renderer):
        # 在调用时，根据渲染器计算 Axes 的位置，通常涉及到对变换的反转
        # 在绘制时执行，因为 transSubfigure 可能在此之后会发生变化
        return mtransforms.TransformedBbox(
            mtransforms.Bbox.from_bounds(*self._bounds),
            self._transform - ax.figure.transSubfigure)
    # 当前索引 i 指向格式字符串 fmt 的位置，循环直到处理完整个格式字符串
    while i < len(fmt):
        # 获取当前位置字符 c
        c = fmt[i]
        
        # 检查当前位置和下一位置的两个字符是否是线型样式的标识符
        if fmt[i:i+2] in mlines.lineStyles:  # 首先检查两字符长的线型样式
            # 如果已经有了 linestyle，抛出错误，不能有两个线型样式符号
            if linestyle is not None:
                raise ValueError(errfmt.format(fmt, "two linestyle symbols"))
            # 记录当前的两字符长线型样式
            linestyle = fmt[i:i+2]
            i += 2  # 移动索引到下一个位置
        # 如果当前字符是单字符的线型样式的标识符
        elif c in mlines.lineStyles:
            # 如果已经有了 linestyle，抛出错误，不能有两个线型样式符号
            if linestyle is not None:
                raise ValueError(errfmt.format(fmt, "two linestyle symbols"))
            # 记录当前的单字符线型样式
            linestyle = c
            i += 1  # 移动索引到下一个位置
        # 如果当前字符是标记符号
        elif c in mlines.lineMarkers:
            # 如果已经有了 marker，抛出错误，不能有两个标记符号
            if marker is not None:
                raise ValueError(errfmt.format(fmt, "two marker symbols"))
            # 记录当前的标记符号
            marker = c
            i += 1  # 移动索引到下一个位置
        # 如果当前字符是颜色符号
        elif c in mcolors.get_named_colors_mapping():
            # 如果已经有了 color，抛出错误，不能有两个颜色符号
            if color is not None:
                raise ValueError(errfmt.format(fmt, "two color symbols"))
            # 记录当前的颜色符号
            color = c
            i += 1  # 移动索引到下一个位置
        # 如果当前字符是 "C"
        elif c == "C":
            # 匹配格式字符串中以 "C" 开头的颜色编号
            cn_color = re.match(r"C\d+", fmt[i:])
            # 如果没有匹配到颜色编号，抛出错误
            if not cn_color:
                raise ValueError(errfmt.format(fmt, "'C' must be followed by a number"))
            # 将颜色编号转换为 RGBA 格式的颜色，并记录
            color = mcolors.to_rgba(cn_color[0])
            i += len(cn_color[0])  # 移动索引到下一个位置
        else:
            # 如果遇到未识别的字符，抛出错误
            raise ValueError(errfmt.format(fmt, f"unrecognized character {c!r}"))
    
    # 如果既没有指定 linestyle 也没有指定 marker，则使用默认的配置
    if linestyle is None and marker is None:
        linestyle = mpl.rcParams['lines.linestyle']
    # 如果没有指定 linestyle，则设置为 'None'
    if linestyle is None:
        linestyle = 'None'
    # 如果没有指定 marker，则设置为 'None'
    if marker is None:
        marker = 'None'
    
    # 返回解析后的 linestyle、marker、color
    return linestyle, marker, color
class _process_plot_var_args:
    """
    Process variable length arguments to `~.Axes.plot`, to support ::

      plot(t, s)
      plot(t1, s1, t2, s2)
      plot(t1, s1, 'ko', t2, s2)
      plot(t1, s1, 'ko', t2, s2, 'r--', t3, e3)

    an arbitrary number of *x*, *y*, *fmt* are allowed
    """

    def __init__(self, command='plot'):
        # 初始化函数，设定命令字符串，默认为'plot'
        self.command = command
        # 调用方法设置属性循环
        self.set_prop_cycle(None)

    def set_prop_cycle(self, cycler):
        # 如果未提供属性循环，则使用默认的属性循环
        if cycler is None:
            cycler = mpl.rcParams['axes.prop_cycle']
        # 初始化循环索引为0
        self._idx = 0
        # 将属性循环的所有项放入列表中
        self._cycler_items = [*cycler]

    def get_next_color(self):
        """Return the next color in the cycle."""
        # 获取属性循环中下一个颜色
        entry = self._cycler_items[self._idx]
        # 如果循环项包含颜色信息，则返回颜色值并推进循环索引
        if "color" in entry:
            self._idx = (self._idx + 1) % len(self._cycler_items)  # Advance cycler.
            return entry["color"]
        else:
            # 否则返回黑色并不推进循环索引
            return "k"

    def _getdefaults(self, kw, ignore=frozenset()):
        """
        If some keys in the property cycle (excluding those in the set
        *ignore*) are absent or set to None in the dict *kw*, return a copy
        of the next entry in the property cycle, excluding keys in *ignore*.
        Otherwise, don't advance the property cycle, and return an empty dict.
        """
        # 获取默认值，如果需要的键在字典中缺失或为None，则返回属性循环中的下一个条目
        defaults = self._cycler_items[self._idx]
        if any(kw.get(k, None) is None for k in {*defaults} - ignore):
            self._idx = (self._idx + 1) % len(self._cycler_items)  # Advance cycler.
            # 返回新的字典副本，避免修改属性循环中的条目
            return {k: v for k, v in defaults.items() if k not in ignore}
        else:
            # 否则返回空字典，不推进循环索引
            return {}

    def _setdefaults(self, defaults, kw):
        """
        Add to the dict *kw* the entries in the dict *default* that are absent
        or set to None in *kw*.
        """
        # 将默认值中在kw中缺失或为None的条目添加到kw中
        for k in defaults:
            if kw.get(k, None) is None:
                kw[k] = defaults[k]

    def _makeline(self, axes, x, y, kw, kwargs):
        # 创建线段对象并返回
        kw = {**kw, **kwargs}  # Don't modify the original kw.
        # 将默认值应用于kw中的参数
        self._setdefaults(self._getdefaults(kw), kw)
        # 使用参数创建线段对象
        seg = mlines.Line2D(x, y, **kw)
        return seg, kw
    # Polygon 不直接支持单位化的输入。
    # 将 x 坐标转换为 Axes 对象中的单位
    x = axes.convert_xunits(x)
    # 将 y 坐标转换为 Axes 对象中的单位
    y = axes.convert_yunits(y)

    # 复制 kw 和 kwargs，避免修改原始字典
    kw = kw.copy()
    kwargs = kwargs.copy()

    # 忽略与 'marker' 相关的属性，因为它们不是 Polygon 的属性，
    # 而是 Line2D 的属性，很可能出现在默认的循环构建中。
    # 在 defaults 字典中执行这一步，而不是其他两个字典，因为我们希望捕获用户明确指定的 marker，这应该是一个错误。
    # 我们还希望在忽略给定属性后，阻止循环器的推进。
    ignores = ({'marker', 'markersize', 'markeredgecolor',
                'markerfacecolor', 'markeredgewidth'}
               # 同时忽略 kwargs 提供的任何内容。
               | {k for k, v in kwargs.items() if v is not None})

    # 仅使用第一个字典作为基础来获取默认值，以支持向后兼容。
    default_dict = self._getdefaults(kw, ignores)
    # 将获取的默认值应用到 kw 字典中
    self._setdefaults(default_dict, kw)

    # 看起来我们不希望 "color" 同时被解释为 facecolor 和 edgecolor。
    # 因此丢弃 'kw' 字典，并且只保留其 'color' 值，并将其作为 'facecolor' 翻译。
    # 这个设计可能需要重新审视，因为它增加了复杂性。
    facecolor = kw.get('color', None)

    # 从 default_dict 中移除 'color' 键，因为它现在作为 facecolor 处理
    default_dict.pop('color', None)

    # 修改 kwargs 字典以获取循环器设置的其他属性
    self._setdefaults(default_dict, kwargs)

    # 创建一个 Polygon 对象，使用 x 和 y 组成的列堆栈作为顶点坐标，
    # 设置 facecolor 为上面从 kw 获取的 color，fill 为 kwargs 中的 'fill'（默认为 True），
    # closed 为 kw 中的 'closed'（多边形是否闭合）
    seg = mpatches.Polygon(np.column_stack((x, y)),
                           facecolor=facecolor,
                           fill=kwargs.get('fill', True),
                           closed=kw['closed'])
    # 根据 kwargs 设置其他属性到 Polygon 对象中
    seg.set(**kwargs)
    # 返回创建的 Polygon 对象和更新后的 kwargs 字典
    return seg, kwargs
@_api.define_aliases({"facecolor": ["fc"]})
class _AxesBase(martist.Artist):
    # 类名
    name = "rectilinear"

    # 坐标轴名称是包含各自坐标轴属性的前缀；
    # 例如 'x' <-> self.xaxis，包含一个 XAxis。
    # 注意，PolarAxes 也使用这些属性，因此我们有 'x' <-> self.xaxis，包含一个 ThetaAxis。
    # 特别地，在 _axis_names 中没有 'theta'。
    # 实际上，对于所有的二维坐标轴，这些是 ('x', 'y')；对于 Axes3D，这些是 ('x', 'y', 'z')。
    _axis_names = ("x", "y")

    # 共享坐标轴，使用 cbook.Grouper() 进行初始化
    _shared_axes = {name: cbook.Grouper() for name in _axis_names}

    # 双生坐标轴，使用 cbook.Grouper() 进行初始化
    _twinned_axes = cbook.Grouper()

    # 子类是否使用了 `cla` 属性的标志，默认为 False
    _subclass_uses_cla = False

    @property
    def _axis_map(self):
        """返回坐标轴名称（如 'x'）到 `Axis` 实例的映射。"""
        return {name: getattr(self, f"{name}axis")
                for name in self._axis_names}

    def __str__(self):
        """返回对象的字符串表示，包含位置信息。"""
        return "{0}({1[0]:g},{1[1]:g};{1[2]:g}x{1[3]:g})".format(
            type(self).__name__, self._position.bounds)

    def __init_subclass__(cls, **kwargs):
        # 检查父类是否使用了 `cla` 属性
        parent_uses_cla = super(cls, cls)._subclass_uses_cla
        
        # 如果子类定义了 `cla` 属性，发出警告信息
        if 'cla' in cls.__dict__:
            _api.warn_deprecated(
                '3.6',
                pending=True,
                message=f'Overriding `Axes.cla` in {cls.__qualname__} is '
                'pending deprecation in %(since)s and will be fully '
                'deprecated in favor of `Axes.clear` in the future. '
                'Please report '
                f'this to the {cls.__module__!r} author.')
        
        # 更新子类是否使用了 `cla` 属性的标志
        cls._subclass_uses_cla = 'cla' in cls.__dict__ or parent_uses_cla
        super().__init_subclass__(**kwargs)

    def __getstate__(self):
        """获取对象的状态，用于序列化."""
        state = super().__getstate__()
        
        # 将共享和双生信息修剪为仅包含当前组的信息
        state["_shared_axes"] = {
            name: self._shared_axes[name].get_siblings(self)
            for name in self._axis_names if self in self._shared_axes[name]}
        
        state["_twinned_axes"] = (self._twinned_axes.get_siblings(self)
                                  if self in self._twinned_axes else None)
        return state

    def __setstate__(self, state):
        """设置对象的状态，用于反序列化."""
        # 将分组信息合并回全局分组器中
        shared_axes = state.pop("_shared_axes")
        for name, shared_siblings in shared_axes.items():
            self._shared_axes[name].join(*shared_siblings)
        
        twinned_siblings = state.pop("_twinned_axes")
        if twinned_siblings:
            self._twinned_axes.join(*twinned_siblings)
        
        # 恢复对象的状态
        self.__dict__ = state
        self._stale = True
    def __repr__(self):
        # 初始化一个空列表用于存储字段信息
        fields = []
        # 如果对象有标签，将标签信息添加到字段列表中
        if self.get_label():
            fields += [f"label={self.get_label()!r}"]
        # 如果对象具有名为 get_title 的属性方法
        if hasattr(self, "get_title"):
            # 初始化一个空字典用于存储标题信息
            titles = {}
            # 遍历三个可能的位置（left、center、right），获取对应位置的标题信息
            for k in ["left", "center", "right"]:
                title = self.get_title(loc=k)
                # 如果标题不为空，则将其加入到标题字典中
                if title:
                    titles[k] = title
            # 如果存在任何标题信息，则将其添加到字段列表中
            if titles:
                fields += [f"title={titles}"]
        # 遍历 _axis_map 中的每个条目，获取每个轴的标签文本，并添加到字段列表中
        for name, axis in self._axis_map.items():
            if axis.get_label() and axis.get_label().get_text():
                fields += [f"{name}label={axis.get_label().get_text()!r}"]
        # 返回一个包含类名和所有字段信息的字符串表示形式
        return f"<{self.__class__.__name__}: " + ", ".join(fields) + ">"

    def get_subplotspec(self):
        """Return the `.SubplotSpec` associated with the subplot, or None."""
        # 返回与子图关联的 SubplotSpec 对象，如果未设置则返回 None
        return self._subplotspec

    def set_subplotspec(self, subplotspec):
        """Set the `.SubplotSpec`. associated with the subplot."""
        # 设置与子图关联的 SubplotSpec 对象
        self._subplotspec = subplotspec
        # 根据给定的 subplotspec 更新 subplot 的位置信息
        self._set_position(subplotspec.get_position(self.figure))

    def get_gridspec(self):
        """Return the `.GridSpec` associated with the subplot, or None."""
        # 返回与子图关联的 GridSpec 对象，如果未设置则返回 None
        return self._subplotspec.get_gridspec() if self._subplotspec else None

    def get_window_extent(self, renderer=None):
        """
        Return the Axes bounding box in display space.

        This bounding box does not include the spines, ticks, ticklabels,
        or other labels.  For a bounding box including these elements use
        `~matplotlib.axes.Axes.get_tightbbox`.

        See Also
        --------
        matplotlib.axes.Axes.get_tightbbox
        matplotlib.axis.Axis.get_tightbbox
        matplotlib.spines.Spine.get_window_extent
        """
        # 返回 Axes 在显示空间中的边界框（bounding box）
        # 此边界框不包括脊柱（spines）、刻度线（ticks）、刻度标签（ticklabels）或其他标签
        # 若要包括这些元素，请使用 matplotlib.axes.Axes.get_tightbbox
        return self.bbox

    def _init_axis(self):
        # 这个方法从 __init__ 中移出来，因为不可分离的轴不会使用它
        # 初始化 x 轴对象，并将其注册到底部和顶部的脊柱上
        self.xaxis = maxis.XAxis(self, clear=False)
        self.spines.bottom.register_axis(self.xaxis)
        self.spines.top.register_axis(self.xaxis)
        # 初始化 y 轴对象，并将其注册到左侧和右侧的脊柱上
        self.yaxis = maxis.YAxis(self, clear=False)
        self.spines.left.register_axis(self.yaxis)
        self.spines.right.register_axis(self.yaxis)

    def set_figure(self, fig):
        # docstring inherited
        # 调用父类的 set_figure 方法，设置 subplot 所属的 figure
        super().set_figure(fig)

        # 根据当前 subplot 的位置信息（_position）和 figure 的转换关系（fig.transSubfigure）创建一个转换后的边界框（bbox）
        self.bbox = mtransforms.TransformedBbox(self._position,
                                                fig.transSubfigure)
        # 将 dataLim 初始化为空的 Bbox 对象，稍后会随数据的添加而更新
        self.dataLim = mtransforms.Bbox.null()
        # 将 _viewLim 初始化为单位 Bbox 对象
        self._viewLim = mtransforms.Bbox.unit()
        # 初始化一个用于缩放转换的 TransformWrapper 对象
        self.transScale = mtransforms.TransformWrapper(
            mtransforms.IdentityTransform())

        # 设置 subplot 的限制和转换信息
        self._set_lim_and_transforms()
    def _unstale_viewLim(self):
        """
        将视图限制信息存储在共享组中，而不是每个轴上都存储。

        根据需要检查是否需要缩放视图限制。对于每个视图限制名字，检查共享轴组中的所有兄弟轴，
        如果有任何一个兄弟轴的视图限制已过期（stale），则需要缩放。

        如果有需要缩放的视图限制，则将所有共享轴组中的兄弟轴的对应视图限制标记为非过期状态。

        最后，根据需要缩放的结果，调用 autoscale_view 方法进行视图自动缩放。
        """
        need_scale = {
            name: any(ax._stale_viewlims[name]
                      for ax in self._shared_axes[name].get_siblings(self))
            for name in self._axis_names}
        if any(need_scale.values()):
            for name in need_scale:
                for ax in self._shared_axes[name].get_siblings(self):
                    ax._stale_viewlims[name] = False
            self.autoscale_view(**{f"scale{name}": scale
                                   for name, scale in need_scale.items()})

    @property
    def viewLim(self):
        """
        获取视图限制属性。

        在返回视图限制属性之前，调用 _unstale_viewLim 方法，以确保视图限制是最新的。

        Returns
        -------
        视图限制属性 (_viewLim)
        """
        self._unstale_viewLim()
        return self._viewLim

    def _request_autoscale_view(self, axis="all", tight=None):
        """
        请求自动缩放视图。

        标记一个或多个轴的视图限制为过期，以便在下次自动缩放时重新计算。

        在下一次自动缩放之前不执行任何计算；因此，单独控制每个轴的调用对性能几乎没有影响。

        Parameters
        ----------
        axis : str, default: "all"
            要标记为过期的轴的名称，可以是 self._axis_names 中的元素，或者是 "all" 表示所有轴。
        tight : bool or None, default: None
            是否执行紧密布局的标志。
        """
        axis_names = _api.check_getitem(
            {**{k: [k] for k in self._axis_names}, "all": self._axis_names},
            axis=axis)
        for name in axis_names:
            self._stale_viewlims[name] = True
        if tight is not None:
            self._tight = tight
    def _set_lim_and_transforms(self):
        """
        Set the *_xaxis_transform*, *_yaxis_transform*, *transScale*,
        *transData*, *transLimits* and *transAxes* transformations.

        .. note::

            This method is primarily used by rectilinear projections of the
            `~matplotlib.axes.Axes` class, and is meant to be overridden by
            new kinds of projection Axes that need different transformations
            and limits. (See `~matplotlib.projections.polar.PolarAxes` for an
            example.)
        """
        # 设置transAxes变换，将当前对象的bbox应用到坐标系的变换
        self.transAxes = mtransforms.BboxTransformTo(self.bbox)

        # 设置transScale变换，对数据进行缩放变换，通常用于非线性比例尺（如对数坐标）
        self.transScale = mtransforms.TransformWrapper(
            mtransforms.IdentityTransform())

        # 设置transLimits变换，基于viewLim和transScale创建Bbox变换
        self.transLimits = mtransforms.BboxTransformFrom(
            mtransforms.TransformedBbox(self._viewLim, self.transScale))

        # 设置transData变换，组合transScale、transLimits和transAxes变换
        # 括号的分组对效率很重要 -- 它们将最后两个变换（通常是仿射变换）与第一个（具有对数缩放的变换）分开
        self.transData = self.transScale + (self.transLimits + self.transAxes)

        # 设置_xaxis_transform变换，混合transData和transAxes变换
        self._xaxis_transform = mtransforms.blended_transform_factory(
            self.transData, self.transAxes)

        # 设置_yaxis_transform变换，混合transAxes和transData变换
        self._yaxis_transform = mtransforms.blended_transform_factory(
            self.transAxes, self.transData)

    def get_xaxis_transform(self, which='grid'):
        """
        Get the transformation used for drawing x-axis labels, ticks
        and gridlines.  The x-direction is in data coordinates and the
        y-direction is in axis coordinates.

        .. note::

            This transformation is primarily used by the
            `~matplotlib.axis.Axis` class, and is meant to be
            overridden by new kinds of projections that may need to
            place axis elements in different locations.

        Parameters
        ----------
        which : {'grid', 'tick1', 'tick2'}
        """
        # 根据which参数返回相应的x轴变换
        if which == 'grid':
            return self._xaxis_transform
        elif which == 'tick1':
            # 对于笛卡尔投影，这是底部脊柱的变换
            return self.spines.bottom.get_spine_transform()
        elif which == 'tick2':
            # 对于笛卡尔投影，这是顶部脊柱的变换
            return self.spines.top.get_spine_transform()
        else:
            raise ValueError(f'unknown value for which: {which!r}')
    def get_xaxis_text1_transform(self, pad_points):
        """
        Returns
        -------
        transform : Transform
            The transform used for drawing x-axis labels, which will add
            *pad_points* of padding (in points) between the axis and the label.
            The x-direction is in data coordinates and the y-direction is in
            axis coordinates
        valign : {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
            The text vertical alignment.
        halign : {'center', 'left', 'right'}
            The text horizontal alignment.

        Notes
        -----
        This transformation is primarily used by the `~matplotlib.axis.Axis`
        class, and is meant to be overridden by new kinds of projections that
        may need to place axis elements in different locations.
        """
        # 获取 x 轴标签的绘制变换，添加指定的垂直偏移量，返回垂直对齐方式和水平对齐方式
        labels_align = mpl.rcParams["xtick.alignment"]
        return (self.get_xaxis_transform(which='tick1') +
                mtransforms.ScaledTranslation(0, -1 * pad_points / 72,
                                              self.figure.dpi_scale_trans),
                "top", labels_align)

    def get_xaxis_text2_transform(self, pad_points):
        """
        Returns
        -------
        transform : Transform
            The transform used for drawing secondary x-axis labels, which will
            add *pad_points* of padding (in points) between the axis and the
            label.  The x-direction is in data coordinates and the y-direction
            is in axis coordinates
        valign : {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
            The text vertical alignment.
        halign : {'center', 'left', 'right'}
            The text horizontal alignment.

        Notes
        -----
        This transformation is primarily used by the `~matplotlib.axis.Axis`
        class, and is meant to be overridden by new kinds of projections that
        may need to place axis elements in different locations.
        """
        # 获取次要 x 轴标签的绘制变换，添加指定的垂直偏移量，返回垂直对齐方式和水平对齐方式
        labels_align = mpl.rcParams["xtick.alignment"]
        return (self.get_xaxis_transform(which='tick2') +
                mtransforms.ScaledTranslation(0, pad_points / 72,
                                              self.figure.dpi_scale_trans),
                "bottom", labels_align)
    def get_yaxis_transform(self, which='grid'):
        """
        Get the transformation used for drawing y-axis labels, ticks
        and gridlines.  The x-direction is in axis coordinates and the
        y-direction is in data coordinates.

        .. note::

            This transformation is primarily used by the
            `~matplotlib.axis.Axis` class, and is meant to be
            overridden by new kinds of projections that may need to
            place axis elements in different locations.

        Parameters
        ----------
        which : {'grid', 'tick1', 'tick2'}
            Specifies which transformation to return:
            - 'grid': transformation for grid lines
            - 'tick1': transformation for bottom spine ticks (for Cartesian projection)
            - 'tick2': transformation for top spine ticks (for Cartesian projection)

        Returns
        -------
        transform : Transform
            The transformation object for the specified `which` value.

        Raises
        ------
        ValueError
            If an unknown value is provided for `which`.

        """
        if which == 'grid':
            return self._yaxis_transform
        elif which == 'tick1':
            # for cartesian projection, this is bottom spine
            return self.spines.left.get_spine_transform()
        elif which == 'tick2':
            # for cartesian projection, this is top spine
            return self.spines.right.get_spine_transform()
        else:
            raise ValueError(f'unknown value for which: {which!r}')

    def get_yaxis_text1_transform(self, pad_points):
        """
        Returns
        -------
        transform : Transform
            The transform used for drawing y-axis labels, which will add
            *pad_points* of padding (in points) between the axis and the label.
            The x-direction is in axis coordinates and the y-direction is in
            data coordinates.
        valign : {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
            The text vertical alignment.
        halign : {'center', 'left', 'right'}
            The text horizontal alignment.

        Notes
        -----
        This transformation is primarily used by the `~matplotlib.axis.Axis`
        class, and is meant to be overridden by new kinds of projections that
        may need to place axis elements in different locations.
        """
        labels_align = mpl.rcParams["ytick.alignment"]
        # Calculate the transformation for y-axis labels with specified padding
        return (self.get_yaxis_transform(which='tick1') +
                mtransforms.ScaledTranslation(-1 * pad_points / 72, 0,
                                              self.figure.dpi_scale_trans),
                labels_align, "right")
    def get_yaxis_text2_transform(self, pad_points):
        """
        Returns
        -------
        transform : Transform
            返回用于绘制第二Y轴标签的变换，该变换在轴与标签之间增加*pad_points*的填充（以点为单位）。
            x方向使用轴坐标，y方向使用数据坐标。
        valign : {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
            文本垂直对齐方式。
        halign : {'center', 'left', 'right'}
            文本水平对齐方式。

        Notes
        -----
        这个变换主要由 `~matplotlib.axis.Axis` 类使用，并且可以被新种类的投影重写，
        这些投影可能需要在不同位置放置轴元素。
        """
        labels_align = mpl.rcParams["ytick.alignment"]
        return (self.get_yaxis_transform(which='tick2') +
                mtransforms.ScaledTranslation(pad_points / 72, 0,
                                              self.figure.dpi_scale_trans),
                labels_align, "left")

    def _update_transScale(self):
        """
        Update the transformation used for scaling the Axes.

        This method sets the transformation (`transScale`) to a blended transform
        of x-axis and y-axis transforms (`xaxis.get_transform()` and
        `yaxis.get_transform()`).
        """
        self.transScale.set(
            mtransforms.blended_transform_factory(
                self.xaxis.get_transform(), self.yaxis.get_transform()))

    def get_position(self, original=False):
        """
        Return the position of the Axes within the figure as a `.Bbox`.

        Parameters
        ----------
        original : bool
            If ``True``, return the original position. Otherwise, return the
            active position. For an explanation of the positions see
            `.set_position`.

        Returns
        -------
        `.Bbox`
            The position of the Axes within the figure.
        """
        if original:
            return self._originalPosition.frozen()
        else:
            locator = self.get_axes_locator()
            if not locator:
                self.apply_aspect()
            return self._position.frozen()
    def set_position(self, pos, which='both'):
        """
        Set the Axes position.

        Axes have two position attributes. The 'original' position is the
        position allocated for the Axes. The 'active' position is the
        position the Axes is actually drawn at. These positions are usually
        the same unless a fixed aspect is set to the Axes. See
        `.Axes.set_aspect` for details.

        Parameters
        ----------
        pos : [left, bottom, width, height] or `~matplotlib.transforms.Bbox`
            The new position of the Axes in `.Figure` coordinates.

        which : {'both', 'active', 'original'}, default: 'both'
            Determines which position variables to change.

        See Also
        --------
        matplotlib.transforms.Bbox.from_bounds
        matplotlib.transforms.Bbox.from_extents
        """
        # 调用内部方法设置位置
        self._set_position(pos, which=which)
        # 因为此方法被外部调用，不允许在布局中使用
        self.set_in_layout(False)

    def _set_position(self, pos, which='both'):
        """
        Private version of set_position.

        Call this internally to get the same functionality of `set_position`,
        but not to take the axis out of the constrained_layout hierarchy.
        """
        # 如果位置不是 BboxBase 对象，从给定的左、下、宽、高创建一个 Bbox 对象
        if not isinstance(pos, mtransforms.BboxBase):
            pos = mtransforms.Bbox.from_bounds(*pos)
        # 获取所有与当前 Axes 共享相同 _twinned_axes 的 Axes 对象
        for ax in self._twinned_axes.get_siblings(self):
            # 根据 which 参数决定修改 'active' 或 'original' 位置变量
            if which in ('both', 'active'):
                ax._position.set(pos)
            if which in ('both', 'original'):
                ax._originalPosition.set(pos)
        # 标记为需要更新
        self.stale = True

    def reset_position(self):
        """
        Reset the active position to the original position.

        This undoes changes to the active position (as defined in
        `.set_position`) which may have been performed to satisfy fixed-aspect
        constraints.
        """
        # 获取所有与当前 Axes 共享相同 _twinned_axes 的 Axes 对象
        for ax in self._twinned_axes.get_siblings(self):
            # 获取原始位置
            pos = ax.get_position(original=True)
            # 将 'active' 位置重置为原始位置
            ax.set_position(pos, which='active')

    def set_axes_locator(self, locator):
        """
        Set the Axes locator.

        Parameters
        ----------
        locator : Callable[[Axes, Renderer], Bbox]
        """
        # 设置 Axes 的定位器
        self._axes_locator = locator
        # 标记为需要更新
        self.stale = True

    def get_axes_locator(self):
        """
        Return the axes_locator.
        """
        # 返回 Axes 的定位器
        return self._axes_locator

    def _set_artist_props(self, a):
        """Set the boilerplate props for artists added to Axes."""
        # 设置艺术家对象的基本属性
        a.set_figure(self.figure)
        # 如果尚未设置坐标系变换，则设置为 transData
        if not a.is_transform_set():
            a.set_transform(self.transData)

        a.axes = self
        # 如果启用了鼠标悬停效果，将艺术家对象添加到 _mouseover_set 集合中
        if a.get_mouseover():
            self._mouseover_set.add(a)
    def _gen_axes_patch(self):
        """
        Returns
        -------
        Patch
            返回用于绘制 Axes 背景的 Patch 对象。同时也作为 Axes 上数据元素的裁剪路径。

            在标准的 Axes 中，这是一个矩形，但在其他投影中可能不是。

        Notes
        -----
        用于新投影类型覆盖此方法。
        """
        return mpatches.Rectangle((0.0, 0.0), 1.0, 1.0)

    def _gen_axes_spines(self, locations=None, offset=0.0, units='inches'):
        """
        Returns
        -------
        dict
            返回一个字典，将脊柱名称映射到用于绘制 Axes 脊柱的 `.Line2D` 或 `.Patch` 实例。

            在标准的 Axes 中，脊柱是单独的线段，但在其他投影中可能不是。

        Notes
        -----
        用于新投影类型覆盖此方法。
        """
        return {side: mspines.Spine.linear_spine(self, side)
                for side in ['left', 'right', 'bottom', 'top']}

    def sharex(self, other):
        """
        Share the x-axis with *other*.

        This is equivalent to passing ``sharex=other`` when constructing the
        Axes, and cannot be used if the x-axis is already being shared with
        another Axes.  Note that it is not possible to unshare axes.
        """
        _api.check_isinstance(_AxesBase, other=other)
        if self._sharex is not None and other is not self._sharex:
            raise ValueError("x-axis is already shared")
        self._shared_axes["x"].join(self, other)
        self._sharex = other
        self.xaxis.major = other.xaxis.major  # Ticker instances holding
        self.xaxis.minor = other.xaxis.minor  # locator and formatter.
        x0, x1 = other.get_xlim()
        self.set_xlim(x0, x1, emit=False, auto=other.get_autoscalex_on())
        self.xaxis._scale = other.xaxis._scale

    def sharey(self, other):
        """
        Share the y-axis with *other*.

        This is equivalent to passing ``sharey=other`` when constructing the
        Axes, and cannot be used if the y-axis is already being shared with
        another Axes.  Note that it is not possible to unshare axes.
        """
        _api.check_isinstance(_AxesBase, other=other)
        if self._sharey is not None and other is not self._sharey:
            raise ValueError("y-axis is already shared")
        self._shared_axes["y"].join(self, other)
        self._sharey = other
        self.yaxis.major = other.yaxis.major  # Ticker instances holding
        self.yaxis.minor = other.yaxis.minor  # locator and formatter.
        y0, y1 = other.get_ylim()
        self.set_ylim(y0, y1, emit=False, auto=other.get_autoscaley_on())
        self.yaxis._scale = other.yaxis._scale
    def clear(self):
        """Clear the Axes."""
        # 如果子类使用 cla 方法，则调用 cla 方法来清空 Axes
        if self._subclass_uses_cla:
            self.cla()
        else:
            # 否则调用私有方法 __clear 来清空 Axes
            self.__clear()

    def cla(self):
        """Clear the Axes."""
        # 如果子类使用 cla 方法，则调用私有方法 __clear 来清空 Axes
        if self._subclass_uses_cla:
            self.__clear()
        else:
            # 否则调用 clear 方法来清空 Axes
            self.clear()
        """
        一个基于类型的 Axes 子列表。

        在 Matplotlib 3.7 中，基于类型的子列表已经变成了不可变对象。
        未来这些艺术家列表可能会被元组替代。可以像使用元组一样使用它们。
        """
        def __init__(self, axes, prop_name,
                     valid_types=None, invalid_types=None):
            """
            初始化函数，创建一个基于类型的艺术家子列表。

            Parameters
            ----------
            axes : `~matplotlib.axes.Axes`
                从这个 Axes 中获取子 Artists 的列表。
            prop_name : str
                用于从 Axes 中访问此子列表的属性名称；用于生成弃用警告。
            valid_types : list of type, optional
                决定哪些子 Artists 将被返回的类型列表。
                如果指定，则子列表中的 Artists 必须是这些类型的实例。
                如果未指定，则任何类型的 Artist 都是有效的（除非被 *invalid_types* 限制）。
            invalid_types : tuple, optional
                决定哪些子 Artists *不* 会被返回的类型列表。
                如果指定，则子列表中的 Artists 绝不会是这些类型的实例。
                否则，没有类型会被排除。

            Notes
            -----
            lambda 表达式 _type_check 用于检查艺术家的类型是否符合条件。
            """
            self._axes = axes
            self._prop_name = prop_name
            self._type_check = lambda artist: (
                (not valid_types or isinstance(artist, valid_types)) and
                (not invalid_types or not isinstance(artist, invalid_types))
            )

        def __repr__(self):
            """
            返回表示此对象的字符串表示形式。

            Returns
            -------
            str
                一个描述此对象的字符串，显示子列表的长度和属性名称。
            """
            return f'<Axes.ArtistList of {len(self)} {self._prop_name}>'

        def __len__(self):
            """
            返回此列表中符合类型条件的艺术家数量。

            Returns
            -------
            int
                符合类型条件的艺术家的数量。
            """
            return sum(self._type_check(artist)
                       for artist in self._axes._children)

        def __iter__(self):
            """
            迭代返回符合类型条件的艺术家。

            Yields
            ------
            Artist
                符合类型条件的艺术家对象。
            """
            for artist in list(self._axes._children):
                if self._type_check(artist):
                    yield artist

        def __getitem__(self, key):
            """
            返回符合类型条件的第 key 个艺术家对象。

            Parameters
            ----------
            key : int
                要获取的艺术家的索引。

            Returns
            -------
            list
                符合类型条件的艺术家列表中第 key 个艺术家对象。
            """
            return [artist
                    for artist in self._axes._children
                    if self._type_check(artist)][key]

        def __add__(self, other):
            """
            实现 '+' 运算符，将当前列表与另一个列表或者 ArtistList 对象相加。

            Parameters
            ----------
            other : list or ArtistList
                要与当前对象相加的另一个列表或 ArtistList 对象。

            Returns
            -------
            list or tuple
                如果 other 是 list 或者 ArtistList，则返回两者合并后的结果。
                否则返回 NotImplemented。
            """
            if isinstance(other, (list, _AxesBase.ArtistList)):
                return [*self, *other]
            if isinstance(other, (tuple, _AxesBase.ArtistList)):
                return (*self, *other)
            return NotImplemented

        def __radd__(self, other):
            """
            实现 '+' 运算符的反向操作，将当前列表与另一个列表或元组相加。

            Parameters
            ----------
            other : list or tuple
                要与当前对象相加的另一个列表或元组。

            Returns
            -------
            list or tuple
                如果 other 是 list，则返回 other 与当前列表的合并结果。
                如果 other 是 tuple，则返回 other 与当前列表的合并结果。
                否则返回 NotImplemented。
            """
            if isinstance(other, list):
                return other + list(self)
            if isinstance(other, tuple):
                return other + tuple(self)
            return NotImplemented
    # 返回一个包含所有艺术家元素的ArtistList对象，过滤掉指定的无效类型
    def artists(self):
        return self.ArtistList(self, 'artists', invalid_types=(
            mcoll.Collection, mimage.AxesImage, mlines.Line2D, mpatches.Patch,
            mtable.Table, mtext.Text))

    # 返回一个包含所有集合元素的ArtistList对象，只包括mcoll.Collection类型的有效元素
    @property
    def collections(self):
        return self.ArtistList(self, 'collections',
                               valid_types=mcoll.Collection)

    # 返回一个包含所有图像元素的ArtistList对象，只包括mimage.AxesImage类型的有效元素
    @property
    def images(self):
        return self.ArtistList(self, 'images', valid_types=mimage.AxesImage)

    # 返回一个包含所有线条元素的ArtistList对象，只包括mlines.Line2D类型的有效元素
    @property
    def lines(self):
        return self.ArtistList(self, 'lines', valid_types=mlines.Line2D)

    # 返回一个包含所有补丁元素的ArtistList对象，只包括mpatches.Patch类型的有效元素
    @property
    def patches(self):
        return self.ArtistList(self, 'patches', valid_types=mpatches.Patch)

    # 返回一个包含所有表格元素的ArtistList对象，只包括mtable.Table类型的有效元素
    @property
    def tables(self):
        return self.ArtistList(self, 'tables', valid_types=mtable.Table)

    # 返回一个包含所有文本元素的ArtistList对象，只包括mtext.Text类型的有效元素
    @property
    def texts(self):
        return self.ArtistList(self, 'texts', valid_types=mtext.Text)

    # 获取Axes的面板颜色
    def get_facecolor(self):
        return self.patch.get_facecolor()

    # 设置Axes的面板颜色
    def set_facecolor(self, color):
        """
        设置Axes的面板颜色。

        参数
        ----------
        color : :mpltype:`color`
        """
        self._facecolor = color
        self.stale = True
        return self.patch.set_facecolor(color)

    # 设置标题的偏移量，根据给定的标题偏移点数设置偏移量
    def _set_title_offset_trans(self, title_offset_points):
        """
        设置标题的偏移量，根据rc中的axes.titlepad或set_title的kwarg中的pad参数设置。
        """
        self.titleOffsetTrans = mtransforms.ScaledTranslation(
                0.0, title_offset_points / 72,
                self.figure.dpi_scale_trans)
        for _title in (self.title, self._left_title, self._right_title):
            _title.set_transform(self.transAxes + self.titleOffsetTrans)
            _title.set_clip_box(None)

    # 获取Axes的纵横比
    def get_aspect(self):
        """
        返回Axes的纵横比。

        可能是"auto"或一个浮点数，表示y/x比例。
        """
        return self._aspect
    def set_aspect(self, aspect, adjustable=None, anchor=None, share=False):
        """
        Set the aspect ratio of the Axes scaling, i.e. y/x-scale.

        Parameters
        ----------
        aspect : {'auto', 'equal'} or float
            Possible values:

            - 'auto': fill the position rectangle with data.
            - 'equal': same as ``aspect=1``, i.e. same scaling for x and y.
            - *float*: The displayed size of 1 unit in y-data coordinates will
              be *aspect* times the displayed size of 1 unit in x-data
              coordinates; e.g. for ``aspect=2`` a square in data coordinates
              will be rendered with a height of twice its width.

        adjustable : None or {'box', 'datalim'}, optional
            If not ``None``, this defines which parameter will be adjusted to
            meet the required aspect. See `.set_adjustable` for further
            details.

        anchor : None or str or (float, float), optional
            If not ``None``, this defines where the Axes will be drawn if there
            is extra space due to aspect constraints. The most common way
            to specify the anchor are abbreviations of cardinal directions:

            =====   =====================
            value   description
            =====   =====================
            'C'     centered
            'SW'    lower left corner
            'S'     middle of bottom edge
            'SE'    lower right corner
            etc.
            =====   =====================

            See `~.Axes.set_anchor` for further details.

        share : bool, default: False
            If ``True``, apply the settings to all shared Axes.

        See Also
        --------
        matplotlib.axes.Axes.set_adjustable
            Set how the Axes adjusts to achieve the required aspect ratio.
        matplotlib.axes.Axes.set_anchor
            Set the position in case of extra space.
        """
        # 如果 aspect 参数为字符串 'equal'，将其转换为数值 1，表示 x 和 y 的比例相等
        if cbook._str_equal(aspect, 'equal'):
            aspect = 1
        # 如果 aspect 参数不为字符串 'auto'，则将其转换为浮点数，如果不合法则抛出 ValueError
        if not cbook._str_equal(aspect, 'auto'):
            aspect = float(aspect)  # 如果需要的话会抛出 ValueError
            # 如果 aspect 不是正数或者不是有限数，则抛出 ValueError
            if aspect <= 0 or not np.isfinite(aspect):
                raise ValueError("aspect must be finite and positive ")

        # 如果 share 参数为 True，则获取所有共享 Axes 的列表，否则使用当前 Axes
        if share:
            axes = {sibling for name in self._axis_names
                    for sibling in self._shared_axes[name].get_siblings(self)}
        else:
            axes = [self]

        # 遍历所有的 Axes，设置其 _aspect 属性为给定的 aspect 值
        for ax in axes:
            ax._aspect = aspect

        # 如果 adjustable 参数为 None，则使用当前 Axes 的默认可调参数，否则使用给定的 adjustable
        if adjustable is None:
            adjustable = self._adjustable
        # 调用 set_adjustable 方法设置可调参数，处理共享情况
        self.set_adjustable(adjustable, share=share)  # 处理共享情况.

        # 如果 anchor 参数不为 None，则调用 set_anchor 方法设置锚点位置，处理共享情况
        if anchor is not None:
            self.set_anchor(anchor, share=share)
        # 设置 stale 属性为 True，表示需要重新绘制图形
        self.stale = True
    def get_adjustable(self):
        """
        Return whether the Axes will adjust its physical dimension ('box') or
        its data limits ('datalim') to achieve the desired aspect ratio.

        See Also
        --------
        matplotlib.axes.Axes.set_adjustable
            Set how the Axes adjusts to achieve the required aspect ratio.
        matplotlib.axes.Axes.set_aspect
            For a description of aspect handling.
        """
        # 返回 Axes 对象当前的调整方式，是基于物理尺寸 ('box') 还是数据范围 ('datalim')
        return self._adjustable

    def set_adjustable(self, adjustable, share=False):
        """
        Set how the Axes adjusts to achieve the required aspect ratio.

        Parameters
        ----------
        adjustable : {'box', 'datalim'}
            If 'box', change the physical dimensions of the Axes.
            If 'datalim', change the ``x`` or ``y`` data limits.

        share : bool, default: False
            If ``True``, apply the settings to all shared Axes.

        See Also
        --------
        matplotlib.axes.Axes.set_aspect
            For a description of aspect handling.

        Notes
        -----
        Shared Axes (of which twinned Axes are a special case)
        impose restrictions on how aspect ratios can be imposed.
        For twinned Axes, use 'datalim'.  For Axes that share both
        x and y, use 'box'.  Otherwise, either 'datalim' or 'box'
        may be used.  These limitations are partly a requirement
        to avoid over-specification, and partly a result of the
        particular implementation we are currently using, in
        which the adjustments for aspect ratios are done sequentially
        and independently on each Axes as it is drawn.
        """
        # 检查 adjustable 是否为合法取值
        _api.check_in_list(["box", "datalim"], adjustable=adjustable)
        
        # 根据 share 参数决定要设置调整方式的 Axes 列表
        if share:
            # 获取所有共享 Axes 的 siblings
            axs = {sibling for name in self._axis_names
                   for sibling in self._shared_axes[name].get_siblings(self)}
        else:
            # 如果不共享，只操作当前 Axes
            axs = [self]
        
        # 如果 adjustable 是 'datalim'，则需要检查所有 Axes 是否重写了 get_data_ratio 方法
        if (adjustable == "datalim"
                and any(getattr(ax.get_data_ratio, "__func__", None)
                        != _AxesBase.get_data_ratio
                        for ax in axs)):
            # 如果有 Axes 重写了 get_data_ratio 方法，则无法将 adjustable 设置为 'datalim'
            raise ValueError("Cannot set Axes adjustable to 'datalim' for "
                             "Axes which override 'get_data_ratio'")
        
        # 为每个需要设置 adjustable 的 Axes 设置对应的值
        for ax in axs:
            ax._adjustable = adjustable
        
        # 设置 self.stale 为 True，表示需要重新绘制
        self.stale = True
    def get_box_aspect(self):
        """
        Return the Axes box aspect, i.e. the ratio of height to width.

        The box aspect is ``None`` (i.e. chosen depending on the available
        figure space) unless explicitly specified.

        See Also
        --------
        matplotlib.axes.Axes.set_box_aspect
            for a description of box aspect.
        matplotlib.axes.Axes.set_aspect
            for a description of aspect handling.
        """
        # 返回当前 Axes 对象的盒子纵横比，即高度与宽度的比例
        return self._box_aspect

    def set_box_aspect(self, aspect=None):
        """
        Set the Axes box aspect, i.e. the ratio of height to width.

        This defines the aspect of the Axes in figure space and is not to be
        confused with the data aspect (see `~.Axes.set_aspect`).

        Parameters
        ----------
        aspect : float or None
            Changes the physical dimensions of the Axes, such that the ratio
            of the Axes height to the Axes width in physical units is equal to
            *aspect*. Defining a box aspect will change the *adjustable*
            property to 'datalim' (see `~.Axes.set_adjustable`).

            *None* will disable a fixed box aspect so that height and width
            of the Axes are chosen independently.

        See Also
        --------
        matplotlib.axes.Axes.set_aspect
            for a description of aspect handling.
        """
        # 获取与当前 Axes 相关联的所有同胞 Axes
        axs = {*self._twinned_axes.get_siblings(self),
               *self._twinned_axes.get_siblings(self)}

        if aspect is not None:
            aspect = float(aspect)
            # 当设置 box_aspect 时，adjustable 必须设置为 "datalim"
            for ax in axs:
                ax.set_adjustable("datalim")

        # 为所有相关 Axes 设置 box aspect
        for ax in axs:
            ax._box_aspect = aspect
            ax.stale = True

    def get_anchor(self):
        """
        Get the anchor location.

        See Also
        --------
        matplotlib.axes.Axes.set_anchor
            for a description of the anchor.
        matplotlib.axes.Axes.set_aspect
            for a description of aspect handling.
        """
        # 返回当前 Axes 对象的锚定位置
        return self._anchor
    def set_anchor(self, anchor, share=False):
        """
        Define the anchor location.

        The actual drawing area (active position) of the Axes may be smaller
        than the Bbox (original position) when a fixed aspect is required. The
        anchor defines where the drawing area will be located within the
        available space.

        Parameters
        ----------
        anchor : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', ...}
            Either an (*x*, *y*) pair of relative coordinates (0 is left or
            bottom, 1 is right or top), 'C' (center), or a cardinal direction
            ('SW', southwest, is bottom left, etc.).  str inputs are shorthands
            for (*x*, *y*) coordinates, as shown in the following diagram::

               ┌─────────────────┬─────────────────┬─────────────────┐
               │ 'NW' (0.0, 1.0) │ 'N' (0.5, 1.0)  │ 'NE' (1.0, 1.0) │
               ├─────────────────┼─────────────────┼─────────────────┤
               │ 'W'  (0.0, 0.5) │ 'C' (0.5, 0.5)  │ 'E'  (1.0, 0.5) │
               ├─────────────────┼─────────────────┼─────────────────┤
               │ 'SW' (0.0, 0.0) │ 'S' (0.5, 0.0)  │ 'SE' (1.0, 0.0) │
               └─────────────────┴─────────────────┴─────────────────┘

        share : bool, default: False
            If ``True``, apply the settings to all shared Axes.

        See Also
        --------
        matplotlib.axes.Axes.set_aspect
            for a description of aspect handling.
        """
        # 检查 anchor 参数是否合法，如果不是合法的 Bbox 系数或长度不为 2，则抛出 ValueError 异常
        if not (anchor in mtransforms.Bbox.coefs or len(anchor) == 2):
            raise ValueError('argument must be among %s' %
                             ', '.join(mtransforms.Bbox.coefs))
        
        # 如果 share 参数为 True，则获取所有共享 Axes 的 sibling Axes，否则将当前 Axes 放入列表中
        if share:
            axes = {sibling for name in self._axis_names
                    for sibling in self._shared_axes[name].get_siblings(self)}
        else:
            axes = [self]
        
        # 遍历 axes 列表，将每个 Axes 的 _anchor 属性设为指定的 anchor
        for ax in axes:
            ax._anchor = anchor

        # 将 self.stale 属性设为 True，表示需要重新绘制
        self.stale = True

    def get_data_ratio(self):
        """
        Return the aspect ratio of the scaled data.

        Notes
        -----
        This method is intended to be overridden by new projection types.
        """
        # 获取 x 轴和 y 轴的数据边界，将其转换为显示坐标系下的范围
        txmin, txmax = self.xaxis.get_transform().transform(self.get_xbound())
        tymin, tymax = self.yaxis.get_transform().transform(self.get_ybound())
        
        # 计算 x 和 y 轴数据的大小
        xsize = max(abs(txmax - txmin), 1e-30)
        ysize = max(abs(tymax - tymin), 1e-30)
        
        # 返回 y 轴数据大小与 x 轴数据大小的比值，即数据的纵横比
        return ysize / xsize

    def get_legend(self):
        """Return the `.Legend` instance, or None if no legend is defined."""
        # 返回当前 Axes 对象的图例实例，如果未定义图例，则返回 None
        return self.legend_

    def get_images(self):
        r"""Return a list of `.AxesImage`\s contained by the Axes."""
        # 返回当前 Axes 对象包含的所有 AxesImage 对象组成的列表
        return cbook.silent_list('AxesImage', self.images)

    def get_lines(self):
        """Return a list of lines contained by the Axes."""
        # 返回当前 Axes 对象包含的所有 Line2D 对象组成的列表
        return cbook.silent_list('Line2D', self.lines)
    def get_xaxis(self):
        """
        [*Discouraged*] Return the XAxis instance.

        .. admonition:: Discouraged

            The use of this function is discouraged. You should instead
            directly access the attribute ``ax.xaxis``.
        """
        return self.xaxis

    def get_yaxis(self):
        """
        [*Discouraged*] Return the YAxis instance.

        .. admonition:: Discouraged

            The use of this function is discouraged. You should instead
            directly access the attribute ``ax.yaxis``.
        """
        return self.yaxis

    # Wrapper methods for accessing grid lines and tick lines

    def get_xgridlines(self):
        """
        Return grid lines associated with the XAxis.

        This is a wrapper method that retrieves grid lines from the XAxis
        associated with the current Axes instance.
        """
        return self.xaxis.get_gridlines()

    def get_xticklines(self):
        """
        Return tick lines associated with the XAxis.

        This is a wrapper method that retrieves tick lines from the XAxis
        associated with the current Axes instance.
        """
        return self.xaxis.get_ticklines()

    def get_ygridlines(self):
        """
        Return grid lines associated with the YAxis.

        This is a wrapper method that retrieves grid lines from the YAxis
        associated with the current Axes instance.
        """
        return self.yaxis.get_gridlines()

    def get_yticklines(self):
        """
        Return tick lines associated with the YAxis.

        This is a wrapper method that retrieves tick lines from the YAxis
        associated with the current Axes instance.
        """
        return self.yaxis.get_ticklines()

    # Adding and tracking artists

    def _sci(self, im):
        """
        Set the current image.

        This function sets the current image attribute of the current Axes
        instance. It is used to designate the target of colormap functions
        and other image-related operations.
        """
        _api.check_isinstance((mcoll.Collection, mimage.AxesImage), im=im)
        if im not in self._children:
            raise ValueError("Argument must be an image or collection in this Axes")
        self._current_image = im

    def _gci(self):
        """
        Helper for `~matplotlib.pyplot.gci`; do not use elsewhere.

        This function retrieves the current image associated with the Axes.
        """
        return self._current_image

    def has_data(self):
        """
        Check if any artists have been added to the Axes.

        This function checks if any artists (such as collections, images,
        lines, or patches) have been added to the Axes instance.
        """
        return any(isinstance(a, (mcoll.Collection, mimage.AxesImage,
                                  mlines.Line2D, mpatches.Patch))
                   for a in self._children)

    def add_artist(self, a):
        """
        Add an `.Artist` to the Axes; return the artist.

        This function adds an artist to the Axes instance. It ensures the
        artist's axes attribute is set, updates properties, and manages
        the artist's lifecycle within the Axes.
        
        If the artist does not have a specified transform, it defaults to
        `ax.transData`.

        Args:
            a (`.Artist`): The artist to add.

        Returns:
            `.Artist`: The added artist.
        """
        a.axes = self
        self._children.append(a)
        a._remove_method = self._children.remove
        self._set_artist_props(a)
        if a.get_clip_path() is None:
            a.set_clip_path(self.patch)
        self.stale = True
        return a
    def add_child_axes(self, ax):
        """
        Add an `.AxesBase` to the Axes' children; return the child Axes.

        This is the lowlevel version.  See `.axes.Axes.inset_axes`.
        """

        # 将传入的 ax 对象的 _axes 属性设置为当前 Axes 对象 self
        ax._axes = self
        # 设置 ax 对象的 stale_callback 属性为 _stale_axes_callback 函数
        ax.stale_callback = martist._stale_axes_callback

        # 将 ax 添加到当前 Axes 对象的 child_axes 列表中
        self.child_axes.append(ax)
        # 设置 ax 对象的 _remove_method 属性为 self.figure._remove_axes 方法，使用 owners=[self.child_axes] 进行部分应用
        ax._remove_method = functools.partial(
            self.figure._remove_axes, owners=[self.child_axes])
        # 将当前 Axes 对象的 stale 属性设置为 True，表示需要更新
        self.stale = True
        # 返回添加的 child Axes 对象 ax
        return ax

    def add_collection(self, collection, autolim=True):
        """
        Add a `.Collection` to the Axes; return the collection.
        """
        # 检查 collection 是否为 mcoll.Collection 的实例
        _api.check_isinstance(mcoll.Collection, collection=collection)
        # 如果 collection 没有设置标签，自动设置一个标签
        if not collection.get_label():
            collection.set_label(f'_child{len(self._children)}')
        # 将 collection 添加到当前 Axes 对象的 _children 列表中
        self._children.append(collection)
        # 设置 collection 的 _remove_method 属性为 self._children.remove 方法
        collection._remove_method = self._children.remove
        # 设置 collection 的属性
        self._set_artist_props(collection)

        # 如果 collection 没有设置裁剪路径，将裁剪路径设置为当前 Axes 对象的 patch 对象
        if collection.get_clip_path() is None:
            collection.set_clip_path(self.patch)

        # 如果 autolim 为 True，自动调整视图限制
        if autolim:
            # 确保 viewLim 不过时
            self._unstale_viewLim()
            # 获取数据限制，使用 self.transData 转换
            datalim = collection.get_datalim(self.transData)
            # 获取数据限制的点集合
            points = datalim.get_points()
            # 如果 datalim 的 minpos 不是无穷大，则添加 minpos 到 points 中
            if not np.isinf(datalim.minpos).all():
                points = np.concatenate([points, [datalim.minpos]])
            # 更新数据限制
            self.update_datalim(points)

        # 设置 Axes 对象的 stale 属性为 True，表示需要更新
        self.stale = True
        # 返回添加的 collection 对象
        return collection

    def add_image(self, image):
        """
        Add an `.AxesImage` to the Axes; return the image.
        """
        # 检查 image 是否为 mimage.AxesImage 的实例
        _api.check_isinstance(mimage.AxesImage, image=image)
        # 设置 image 的属性
        self._set_artist_props(image)
        # 如果 image 没有设置标签，自动设置一个标签
        if not image.get_label():
            image.set_label(f'_child{len(self._children)}')
        # 将 image 添加到当前 Axes 对象的 _children 列表中
        self._children.append(image)
        # 设置 image 的 _remove_method 属性为 self._children.remove 方法
        image._remove_method = self._children.remove
        # 设置 Axes 对象的 stale 属性为 True，表示需要更新
        self.stale = True
        # 返回添加的 image 对象
        return image

    def _update_image_limits(self, image):
        xmin, xmax, ymin, ymax = image.get_extent()
        # 更新 Axes 对象的数据限制，使用 image 的 extent 返回的坐标范围
        self.axes.update_datalim(((xmin, ymin), (xmax, ymax)))
    # 将一个 `.Line2D` 对象添加到 Axes 中，并返回该对象
    def add_line(self, line):
        # 检查 line 是否是 `mlines.Line2D` 类型的实例
        _api.check_isinstance(mlines.Line2D, line=line)
        # 设置 line 的艺术家属性
        self._set_artist_props(line)
        # 如果 line 没有剪切路径，则设置其剪切路径为 Axes 的补丁路径
        if line.get_clip_path() is None:
            line.set_clip_path(self.patch)

        # 更新 Axes 的线条限制
        self._update_line_limits(line)
        # 如果 line 没有标签，设置其标签为 `_child` 后接当前子元素数量
        if not line.get_label():
            line.set_label(f'_child{len(self._children)}')
        # 将 line 添加到 Axes 的子元素列表中
        self._children.append(line)
        # 设置 line 的移除方法为从子元素列表中移除自己
        line._remove_method = self._children.remove
        # 将 Axes 标记为过时，需要重新绘制
        self.stale = True
        # 返回添加的 line 对象
        return line

    # 将一个 `.Text` 对象添加到 Axes 中，并返回该对象
    def _add_text(self, txt):
        # 检查 txt 是否是 `mtext.Text` 类型的实例
        _api.check_isinstance(mtext.Text, txt=txt)
        # 设置 txt 的艺术家属性
        self._set_artist_props(txt)
        # 将 txt 添加到 Axes 的子元素列表中
        self._children.append(txt)
        # 设置 txt 的移除方法为从子元素列表中移除自己
        txt._remove_method = self._children.remove
        # 将 Axes 标记为过时，需要重新绘制
        self.stale = True
        # 返回添加的 txt 对象
        return txt
    def _update_line_limits(self, line):
        """
        Figures out the data limit of the given line, updating self.dataLim.
        """
        # 获取线条对象的路径
        path = line.get_path()
        # 如果路径中没有顶点，直接返回
        if path.vertices.size == 0:
            return

        # 获取线条对象的变换
        line_trf = line.get_transform()

        # 如果线条变换与数据坐标变换相同，使用路径作为数据路径
        if line_trf == self.transData:
            data_path = path
        # 如果线条变换包含在数据坐标变换中的任何部分
        elif any(line_trf.contains_branch_seperately(self.transData)):
            # 计算从线条坐标到数据坐标的变换
            trf_to_data = line_trf - self.transData
            # 如果 transData 是仿射的，可以使用线条路径的缓存非仿射部分
            # 因为线条变换的非仿射部分完全包含在 trf_to_data 中
            if self.transData.is_affine:
                line_trans_path = line._get_transformed_path()
                na_path, _ = line_trans_path.get_transformed_path_and_affine()
                data_path = trf_to_data.transform_path_affine(na_path)
            else:
                data_path = trf_to_data.transform_path(path)
        else:
            # 对于向后兼容性，更新数据限制为给定路径的坐标范围
            # 即使坐标系统完全不同也要这样做，例如在绝对定位中传递 ax.transAxes 的情况
            data_path = path

        # 如果数据路径中没有顶点，直接返回
        if not data_path.vertices.size:
            return

        # 检查是否需要分别更新 x 和 y 轴
        updatex, updatey = line_trf.contains_branch_seperately(self.transData)
        
        # 如果不是 "rectilinear" 坐标系，特别处理 axvline 在极坐标图中的情况
        if self.name != "rectilinear":
            # 这个块主要处理极坐标图中的 axvline，其中 updatey 可能为 True
            if updatex and line_trf == self.get_yaxis_transform():
                updatex = False
            if updatey and line_trf == self.get_xaxis_transform():
                updatey = False

        # 更新 self.dataLim 的数据范围，忽略现有的数据限制
        self.dataLim.update_from_path(data_path,
                                      self.ignore_existing_data_limits,
                                      updatex=updatex, updatey=updatey)
        self.ignore_existing_data_limits = False

    def add_patch(self, p):
        """
        Add a `.Patch` to the Axes; return the patch.
        """
        # 检查 p 是否为 mpatches.Patch 类型的对象
        _api.check_isinstance(mpatches.Patch, p=p)
        # 设置艺术家属性
        self._set_artist_props(p)
        # 如果 p 没有剪辑路径，将当前图形的剪辑路径设置为 p 的剪辑路径
        if p.get_clip_path() is None:
            p.set_clip_path(self.patch)
        # 更新图形的路径限制
        self._update_patch_limits(p)
        # 将图形对象 p 添加到子对象列表中
        self._children.append(p)
        # 设置 p 的移除方法为从子对象列表中移除自己
        p._remove_method = self._children.remove
        # 返回添加的图形对象 p
        return p
    def _update_patch_limits(self, patch):
        """Update the data limits for the given patch."""
        # 检查传入的 patch 是否为矩形，并且其宽度和高度都不为零
        if (isinstance(patch, mpatches.Rectangle) and
                ((not patch.get_width()) and (not patch.get_height()))):
            return
        # 获取 patch 的路径对象
        p = patch.get_path()
        # 获取路径上的所有顶点
        vertices = []
        # 遍历路径上的每一段，以获取贝塞尔曲线部分的极值
        for curve, code in p.iter_bezier(simplify=False):
            # 获取贝塞尔曲线的轴向极值
            _, dzeros = curve.axis_aligned_extrema()
            # 计算起点、终点以及中间任何极值的顶点
            vertices.append(curve([0, *dzeros, 1]))

        # 如果存在顶点，则将它们堆叠成数组
        if len(vertices):
            vertices = np.vstack(vertices)

        # 获取 patch 的变换对象
        patch_trf = patch.get_transform()
        # 检查 patch 是否分别包含在 x 和 y 轴的分支上
        updatex, updatey = patch_trf.contains_branch_seperately(self.transData)
        # 如果不需要更新 x 或 y 轴，则直接返回
        if not (updatex or updatey):
            return
        # 如果当前坐标系不是矩形坐标系
        if self.name != "rectilinear":
            # 对于 axvspan，类似于 _update_line_limits 的处理
            if updatex and patch_trf == self.get_yaxis_transform():
                updatex = False
            if updatey and patch_trf == self.get_xaxis_transform():
                updatey = False
        # 计算从 patch 变换到数据坐标系的变换
        trf_to_data = patch_trf - self.transData
        # 将顶点坐标从变换坐标系转换到数据坐标系
        xys = trf_to_data.transform(vertices)
        # 更新数据限制
        self.update_datalim(xys, updatex=updatex, updatey=updatey)

    def add_table(self, tab):
        """
        Add a `.Table` to the Axes; return the table.
        """
        # 检查 tab 是否为 Table 类型的实例
        _api.check_isinstance(mtable.Table, tab=tab)
        # 设置图形对象的属性
        self._set_artist_props(tab)
        # 将表格对象添加到子对象列表中
        self._children.append(tab)
        # 如果表格没有剪裁路径，则使用 Axes 的 patch 对象作为其剪裁路径
        if tab.get_clip_path() is None:
            tab.set_clip_path(self.patch)
        # 设置表格对象的移除方法
        tab._remove_method = self._children.remove
        # 返回添加的表格对象
        return tab

    def add_container(self, container):
        """
        Add a `.Container` to the Axes' containers; return the container.
        """
        # 获取容器的标签
        label = container.get_label()
        # 如果容器没有标签，则设置一个默认标签
        if not label:
            container.set_label('_container%d' % len(self.containers))
        # 将容器添加到容器列表中
        self.containers.append(container)
        # 设置容器对象的移除方法
        container._remove_method = self.containers.remove
        # 返回添加的容器对象
        return container
    def _unit_change_handler(self, axis_name, event=None):
        """
        处理轴单位变化：请求更新数据和视图限制。
        """
        if event is None:  # 如果未提供事件，则返回部分函数
            return functools.partial(
                self._unit_change_handler, axis_name, event=object())
        _api.check_in_list(self._axis_map, axis_name=axis_name)
        # 对每条线重新计算缓存
        for line in self.lines:
            line.recache_always()
        # 重新计算数据限制
        self.relim()
        # 请求自动调整轴视图
        self._request_autoscale_view(axis_name)

    def relim(self, visible_only=False):
        """
        根据当前的图形对象重新计算数据限制。

        目前不支持 `.Collection` 实例。

        Parameters
        ----------
        visible_only : bool, default: False
            是否排除不可见的图形对象。
        """
        # 明确指出不支持集合对象（暂未支持）；参见 artists.py 中的 TODO 注释。
        self.dataLim.ignore(True)
        # 设置数据限制为空的矩形框
        self.dataLim.set_points(mtransforms.Bbox.null().get_points())
        self.ignore_existing_data_limits = True

        for artist in self._children:
            if not visible_only or artist.get_visible():
                if isinstance(artist, mlines.Line2D):
                    self._update_line_limits(artist)
                elif isinstance(artist, mpatches.Patch):
                    self._update_patch_limits(artist)
                elif isinstance(artist, mimage.AxesImage):
                    self._update_image_limits(artist)

    def update_datalim(self, xys, updatex=True, updatey=True):
        """
        扩展 `~.Axes.dataLim` 边界框以包含给定的点集。

        如果当前没有设置数据，则边界框将忽略其限制，并将边界设置为 xy 数据 (*xys*) 的边界。否则，将计算当前数据与 *xys* 数据的并集的边界。

        Parameters
        ----------
        xys : 2D array-like
            要包含在数据限制边界框中的点集。可以是 (x, y) 元组的列表或 (N, 2) 数组。

        updatex, updatey : bool, default: True
            是否更新 x/y 轴限制。
        """
        xys = np.asarray(xys)
        if not np.any(np.isfinite(xys)):
            return
        self.dataLim.update_from_data_xy(xys, self.ignore_existing_data_limits,
                                         updatex=updatex, updatey=updatey)
        self.ignore_existing_data_limits = False
    def _process_unit_info(self, datasets=None, kwargs=None, *, convert=True):
        """
        Set axis units based on *datasets* and *kwargs*, and optionally apply
        unit conversions to *datasets*.

        Parameters
        ----------
        datasets : list
            List of (axis_name, dataset) pairs (where the axis name is defined
            as in `._axis_map`).  Individual datasets can also be None
            (which gets passed through).
        kwargs : dict
            Other parameters from which unit info (i.e., the *xunits*,
            *yunits*, *zunits* (for 3D Axes), *runits* and *thetaunits* (for
            polar) entries) is popped, if present.  Note that this dict is
            mutated in-place!
        convert : bool, default: True
            Whether to return the original datasets or the converted ones.

        Returns
        -------
        list
            Either the original datasets if *convert* is False, or the
            converted ones if *convert* is True (the default).
        """
        # The API makes datasets a list of pairs rather than an axis_name to
        # dataset mapping because it is sometimes necessary to process multiple
        # datasets for a single axis, and concatenating them may be tricky
        datasets = datasets or []  # 如果 datasets 为 None，则设为空列表
        kwargs = kwargs or {}  # 如果 kwargs 为 None，则设为空字典
        axis_map = self._axis_map  # 获取当前对象的轴映射

        for axis_name, data in datasets:
            try:
                axis = axis_map[axis_name]  # 尝试从轴映射中获取指定轴名的轴对象
            except KeyError:
                raise ValueError(f"Invalid axis name: {axis_name!r}") from None
            # 如果轴已设置且数据不为空且轴尚未设置单位，则更新单位信息
            if axis is not None and data is not None and not axis.have_units():
                axis.update_units(data)

        for axis_name, axis in axis_map.items():
            # 如果轴未设置，则跳过当前轴的处理
            if axis is None:
                continue
            # 检查 kwargs 中是否存在与当前轴名相关的单位信息，并更新轴的单位
            units = kwargs.pop(f"{axis_name}units", axis.units)
            if self.name == "polar":
                # 特殊情况：极坐标支持 "thetaunits"/"runits"
                polar_units = {"x": "thetaunits", "y": "runits"}
                units = kwargs.pop(polar_units[axis_name], units)
            # 如果新的单位与当前轴的单位不同且不为 None，则设置新的单位，并检查是否需要更新转换器
            if units != axis.units and units is not None:
                axis.set_units(units)
                # 如果新单位需要使用不同的转换器，则需要再次更新单位信息
                for dataset_axis_name, data in datasets:
                    if dataset_axis_name == axis_name and data is not None:
                        axis.update_units(data)

        # 返回处理后的 datasets 列表，如果 convert=True 则进行单位转换
        return [axis_map[axis_name].convert_units(data)
                if convert and data is not None else data
                for axis_name, data in datasets]
    def in_axes(self, mouseevent):
        """
        Return whether the given event (in display coords) is in the Axes.
        """
        # 调用Axes对象的patch属性的contains方法来判断鼠标事件是否在Axes内部
        return self.patch.contains(mouseevent)[0]

    get_autoscalex_on = _axis_method_wrapper("xaxis", "_get_autoscale_on")
    get_autoscaley_on = _axis_method_wrapper("yaxis", "_get_autoscale_on")
    set_autoscalex_on = _axis_method_wrapper("xaxis", "_set_autoscale_on")
    set_autoscaley_on = _axis_method_wrapper("yaxis", "_set_autoscale_on")

    def get_autoscale_on(self):
        """Return True if each axis is autoscaled, False otherwise."""
        # 返回所有Axes对象的各个轴是否自动缩放的布尔值
        return all(axis._get_autoscale_on()
                   for axis in self._axis_map.values())

    def set_autoscale_on(self, b):
        """
        Set whether autoscaling is applied to each axis on the next draw or
        call to `.Axes.autoscale_view`.

        Parameters
        ----------
        b : bool
            True or False indicating whether autoscaling should be applied.
        """
        # 设置是否在下一次绘图或调用`.Axes.autoscale_view`时对每个轴进行自动缩放
        for axis in self._axis_map.values():
            axis._set_autoscale_on(b)

    @property
    def use_sticky_edges(self):
        """
        When autoscaling, whether to obey all `Artist.sticky_edges`.

        Default is ``True``.

        Setting this to ``False`` ensures that the specified margins
        will be applied, even if the plot includes an image, for
        example, which would otherwise force a view limit to coincide
        with its data limit.

        The changing this property does not change the plot until
        `autoscale` or `autoscale_view` is called.
        """
        # 返回是否使用粘性边界（sticky edges）来调整自动缩放行为的布尔值
        return self._use_sticky_edges

    @use_sticky_edges.setter
    def use_sticky_edges(self, b):
        """
        Set whether to use sticky edges when autoscaling.

        Parameters
        ----------
        b : bool
            True or False indicating whether to use sticky edges.

        Notes
        -----
        This change does not affect the plot until the next autoscaling operation,
        which will mark the Axes as needing update.
        """
        self._use_sticky_edges = bool(b)
        # 仅在下次自动缩放时有效，这会使Axes标记为需要更新

    def get_xmargin(self):
        """
        Retrieve autoscaling margin of the x-axis.

        .. versionadded:: 3.9

        Returns
        -------
        xmargin : float
            The current margin setting for autoscaling on the x-axis.

        See Also
        --------
        matplotlib.axes.Axes.set_xmargin
        """
        # 返回x轴的自动缩放边界
        return self._xmargin

    def get_ymargin(self):
        """
        Retrieve autoscaling margin of the y-axis.

        .. versionadded:: 3.9

        Returns
        -------
        ymargin : float
            The current margin setting for autoscaling on the y-axis.

        See Also
        --------
        matplotlib.axes.Axes.set_ymargin
        """
        # 返回y轴的自动缩放边界
        return self._ymargin
    # 设置X轴数据限制的填充量，以便在自动缩放之前使用
    def set_xmargin(self, m):
        """
        Set padding of X data limits prior to autoscaling.

        *m* times the data interval will be added to each end of that interval
        before it is used in autoscaling.  If *m* is negative, this will clip
        the data range instead of expanding it.

        For example, if your data is in the range [0, 2], a margin of 0.1 will
        result in a range [-0.2, 2.2]; a margin of -0.1 will result in a range
        of [0.2, 1.8].

        Parameters
        ----------
        m : float greater than -0.5
            Margin factor to adjust the data range.
        """
        if m <= -0.5:
            raise ValueError("margin must be greater than -0.5")
        # 设置X轴的填充量
        self._xmargin = m
        # 请求自动缩放视图更新
        self._request_autoscale_view("x")
        # 标记为过时，需要重新绘制
        self.stale = True

    # 设置Y轴数据限制的填充量，以便在自动缩放之前使用
    def set_ymargin(self, m):
        """
        Set padding of Y data limits prior to autoscaling.

        *m* times the data interval will be added to each end of that interval
        before it is used in autoscaling.  If *m* is negative, this will clip
        the data range instead of expanding it.

        For example, if your data is in the range [0, 2], a margin of 0.1 will
        result in a range [-0.2, 2.2]; a margin of -0.1 will result in a range
        of [0.2, 1.8].

        Parameters
        ----------
        m : float greater than -0.5
            Margin factor to adjust the data range.
        """
        if m <= -0.5:
            raise ValueError("margin must be greater than -0.5")
        # 设置Y轴的填充量
        self._ymargin = m
        # 请求自动缩放视图更新
        self._request_autoscale_view("y")
        # 标记为过时，需要重新绘制
        self.stale = True

    # 设置矢量图形输出的光栅化的zorder阈值
    def set_rasterization_zorder(self, z):
        """
        Set the zorder threshold for rasterization for vector graphics output.

        All artists with a zorder below the given value will be rasterized if
        they support rasterization.

        This setting is ignored for pixel-based output.

        See also :doc:`/gallery/misc/rasterization_demo`.

        Parameters
        ----------
        z : float or None
            The zorder below which artists are rasterized.
            If ``None`` rasterization based on zorder is deactivated.
        """
        # 设置光栅化的zorder阈值
        self._rasterization_zorder = z
        # 标记为过时，需要重新绘制
        self.stale = True

    # 获取光栅化zorder的阈值
    def get_rasterization_zorder(self):
        """Return the zorder value below which artists will be rasterized."""
        return self._rasterization_zorder
    def autoscale(self, enable=True, axis='both', tight=None):
        """
        Autoscale the axis view to the data (toggle).

        Convenience method for simple axis view autoscaling.
        It turns autoscaling on or off, and then,
        if autoscaling for either axis is on, it performs
        the autoscaling on the specified axis or Axes.

        Parameters
        ----------
        enable : bool or None, default: True
            True turns autoscaling on, False turns it off.
            None leaves the autoscaling state unchanged.
        axis : {'both', 'x', 'y'}, default: 'both'
            The axis on which to operate.  (For 3D Axes, *axis* can also be set
            to 'z', and 'both' refers to all three Axes.)
        tight : bool or None, default: None
            If True, first set the margins to zero.  Then, this argument is
            forwarded to `~.axes.Axes.autoscale_view` (regardless of
            its value); see the description of its behavior there.
        """
        # 根据 enable 参数确定是否开启自动缩放，如果 enable 为 None，则默认开启自动缩放
        if enable is None:
            scalex = True
            scaley = True
        else:
            # 根据 axis 参数设置 x 和 y 轴的自动缩放状态，并获取当前状态
            if axis in ['x', 'both']:
                self.set_autoscalex_on(bool(enable))
                scalex = self.get_autoscalex_on()
            else:
                scalex = False
            if axis in ['y', 'both']:
                self.set_autoscaley_on(bool(enable))
                scaley = self.get_autoscaley_on()
            else:
                scaley = False
        
        # 如果 tight 参数为 True，并且 x 轴需要自动缩放，则将 x 轴边缘设置为 0
        if tight and scalex:
            self._xmargin = 0
        # 如果 tight 参数为 True，并且 y 轴需要自动缩放，则将 y 轴边缘设置为 0
        if tight and scaley:
            self._ymargin = 0
        
        # 如果 x 轴需要自动缩放，则请求对 x 轴进行自动缩放视图调整
        if scalex:
            self._request_autoscale_view("x", tight=tight)
        # 如果 y 轴需要自动缩放，则请求对 y 轴进行自动缩放视图调整
        if scaley:
            self._request_autoscale_view("y", tight=tight)
    def draw(self, renderer):
        # 绘制函数，用于将图形元素渲染到指定的渲染器上

        # 如果未定义渲染器，抛出运行时错误
        if renderer is None:
            raise RuntimeError('No renderer defined')

        # 如果图形不可见，直接返回
        if not self.get_visible():
            return

        # 更新视图限制，确保视图是最新的
        self._unstale_viewLim()

        # 打开一个新的渲染分组，标记为 'axes'，使用当前对象的 GID
        renderer.open_group('axes', gid=self.get_gid())

        # 标记为过时，避免在绘制过程中触发回调
        self._stale = True

        # 根据当前的定位器对象，应用视角方面的调整
        locator = self.get_axes_locator()
        self.apply_aspect(locator(self, renderer) if locator else None)

        # 获取所有子元素，并移除背景补丁
        artists = self.get_children()
        artists.remove(self.patch)

        # 如果不需要绘制边框或者坐标轴，移除相关的元素
        if not (self.axison and self._frameon):
            for spine in self.spines.values():
                artists.remove(spine)

        # 更新标题位置
        self._update_title_position(renderer)

        # 如果不需要绘制坐标轴，移除相关的轴
        if not self.axison:
            for _axis in self._axis_map.values():
                artists.remove(_axis)

        # 如果不是在保存画布状态，筛选需要绘制的元素
        if not self.figure.canvas.is_saving():
            artists = [
                a for a in artists
                if not a.get_animated() or isinstance(a, mimage.AxesImage)
            ]

        # 根据绘制顺序排序所有艺术家元素
        artists = sorted(artists, key=attrgetter('zorder'))

        # 根据设置的负 zorder 值将需要栅格化的元素栅格化
        rasterization_zorder = self._rasterization_zorder
        if (rasterization_zorder is not None and
                artists and artists[0].zorder < rasterization_zorder):
            split_index = np.searchsorted(
                [art.zorder for art in artists],
                rasterization_zorder, side='right'
            )
            artists_rasterized = artists[:split_index]
            artists = artists[split_index:]
        else:
            artists_rasterized = []

        # 如果需要绘制坐标轴和边框，将栅格化的元素添加到需要绘制的列表中
        if self.axison and self._frameon:
            if artists_rasterized:
                artists_rasterized = [self.patch] + artists_rasterized
            else:
                artists = [self.patch] + artists

        # 如果有需要栅格化的元素，使用函数来绘制栅格化的图像
        if artists_rasterized:
            _draw_rasterized(self.figure, artists_rasterized, renderer)

        # 绘制合成图像
        mimage._draw_list_compositing_images(
            renderer, self, artists, self.figure.suppressComposite)

        # 关闭 'axes' 分组
        renderer.close_group('axes')

        # 标记为不过时
        self.stale = False

    def draw_artist(self, a):
        """
        高效地重绘单个艺术家对象。
        """
        a.draw(self.figure.canvas.get_renderer())
    # 重新绘制 Axes 数据，但不包括轴刻度、标签等内容，效率高
    def redraw_in_frame(self):
        """
        Efficiently redraw Axes data, but not axis ticks, labels, etc.
        """
        # 使用 ExitStack 确保所有艺术对象的可见性在退出时被恢复
        with ExitStack() as stack:
            # 将所有需要隐藏的艺术对象加入上下文管理器
            for artist in [*self._axis_map.values(),
                           self.title, self._left_title, self._right_title]:
                stack.enter_context(artist._cm_set(visible=False))
            # 根据给定的渲染器重新绘制当前 Axes
            self.draw(self.figure.canvas.get_renderer())

    # Axes 矩形特性

    # 获取 Axes 矩形补丁是否被绘制
    def get_frame_on(self):
        """Get whether the Axes rectangle patch is drawn."""
        return self._frameon

    # 设置 Axes 矩形补丁是否被绘制
    def set_frame_on(self, b):
        """
        Set whether the Axes rectangle patch is drawn.

        Parameters
        ----------
        b : bool
        """
        self._frameon = b
        self.stale = True

    # 获取轴刻度和网格线是否在大多数艺术对象之上还是之下
    def get_axisbelow(self):
        """
        Get whether axis ticks and gridlines are above or below most artists.

        Returns
        -------
        bool or 'line'

        See Also
        --------
        set_axisbelow
        """
        return self._axisbelow

    # 设置轴刻度和网格线是否在大多数艺术对象之上还是之下
    def set_axisbelow(self, b):
        """
        Set whether axis ticks and gridlines are above or below most artists.

        This controls the zorder of the ticks and gridlines. For more
        information on the zorder see :doc:`/gallery/misc/zorder_demo`.

        Parameters
        ----------
        b : bool or 'line'
            Possible values:

            - *True* (zorder = 0.5): Ticks and gridlines are below patches and
              lines, though still above images.
            - 'line' (zorder = 1.5): Ticks and gridlines are above patches
              (e.g. rectangles, with default zorder = 1) but still below lines
              and markers (with their default zorder = 2).
            - *False* (zorder = 2.5): Ticks and gridlines are above patches
              and lines / markers.

        Notes
        -----
        For more control, call the `~.Artist.set_zorder` method of each axis.

        See Also
        --------
        get_axisbelow
        """
        # 验证 b 的合法性，可以是 True、False 或 'line'
        self._axisbelow = axisbelow = validate_axisbelow(b)
        # 根据 axisbelow 的值设置 zorder
        zorder = {
            True: 0.5,
            'line': 1.5,
            False: 2.5,
        }[axisbelow]
        # 设置所有轴的 zorder
        for axis in self._axis_map.values():
            axis.set_zorder(zorder)
        self.stale = True

    # _docstring.dedent_interpd
    # 定义一个方法用于配置网格线的显示属性
    def grid(self, visible=None, which='major', axis='both', **kwargs):
        """
        Configure the grid lines.

        Parameters
        ----------
        visible : bool or None, optional
            Whether to show the grid lines.  If any *kwargs* are supplied, it
            is assumed you want the grid on and *visible* will be set to True.

            If *visible* is *None* and there are no *kwargs*, this toggles the
            visibility of the lines.

        which : {'major', 'minor', 'both'}, optional
            The grid lines to apply the changes on.

        axis : {'both', 'x', 'y'}, optional
            The axis to apply the changes on.

        **kwargs : `~matplotlib.lines.Line2D` properties
            Define the line properties of the grid, e.g.::

                grid(color='r', linestyle='-', linewidth=2)

            Valid keyword arguments are:

            %(Line2D:kwdoc)s

        Notes
        -----
        The axis is drawn as a unit, so the effective zorder for drawing the
        grid is determined by the zorder of each axis, not by the zorder of the
        `.Line2D` objects comprising the grid.  Therefore, to set grid zorder,
        use `.set_axisbelow` or, for more control, call the
        `~.Artist.set_zorder` method of each axis.
        """
        # 检查 axis 参数是否在有效取值列表中
        _api.check_in_list(['x', 'y', 'both'], axis=axis)
        # 如果 axis 是 'x' 或 'both'，则设置 x 轴的网格线属性
        if axis in ['x', 'both']:
            self.xaxis.grid(visible, which=which, **kwargs)
        # 如果 axis 是 'y' 或 'both'，则设置 y 轴的网格线属性
        if axis in ['y', 'both']:
            self.yaxis.grid(visible, which=which, **kwargs)
    def locator_params(self, axis='both', tight=None, **kwargs):
        """
        Control behavior of major tick locators.

        Because the locator is involved in autoscaling, `~.Axes.autoscale_view`
        is called automatically after the parameters are changed.

        Parameters
        ----------
        axis : {'both', 'x', 'y'}, default: 'both'
            The axis on which to operate.  (For 3D Axes, *axis* can also be
            set to 'z', and 'both' refers to all three axes.)
        tight : bool or None, optional
            Parameter passed to `~.Axes.autoscale_view`.
            Default is None, for no change.

        Other Parameters
        ----------------
        **kwargs
            Remaining keyword arguments are passed directly to the
            ``set_params()`` method of the locator. Supported keywords depend
            on the type of the locator. See for example
            `~.ticker.MaxNLocator.set_params` for the `.ticker.MaxNLocator`
            used by default for linear.

        Examples
        --------
        When plotting small subplots, one might want to reduce the maximum
        number of ticks and use tight bounds, for example::

            ax.locator_params(tight=True, nbins=4)

        """
        # 检查 axis 参数是否在允许的列表中
        _api.check_in_list([*self._axis_names, "both"], axis=axis)
        # 遍历所有轴名
        for name in self._axis_names:
            # 如果 axis 参数为当前轴名或者为 'both'
            if axis in [name, "both"]:
                # 获取当前轴主刻度定位器
                loc = self._axis_map[name].get_major_locator()
                # 调用定位器的 set_params 方法，传入 kwargs
                loc.set_params(**kwargs)
                # 请求自动缩放视图，包括 tight 参数
                self._request_autoscale_view(name, tight=tight)
        # 设置标记为需要重绘
        self.stale = True

    def set_axis_off(self):
        """
        Hide all visual components of the x- and y-axis.

        This sets a flag to suppress drawing of all axis decorations, i.e.
        axis labels, axis spines, and the axis tick component (tick markers,
        tick labels, and grid lines). Individual visibility settings of these
        components are ignored as long as `set_axis_off()` is in effect.
        """
        # 关闭坐标轴的所有可视组件
        self.axison = False
        # 设置标记为需要重绘
        self.stale = True

    def set_axis_on(self):
        """
        Do not hide all visual components of the x- and y-axis.

        This reverts the effect of a prior `.set_axis_off()` call. Whether the
        individual axis decorations are drawn is controlled by their respective
        visibility settings.

        This is on by default.
        """
        # 开启坐标轴的所有可视组件
        self.axison = True
        # 设置标记为需要重绘
        self.stale = True

    # data limits, ticks, tick labels, and formatting

    def get_xlabel(self):
        """
        Get the xlabel text string.
        """
        # 获取 x 轴标签的文本字符串
        label = self.xaxis.get_label()
        return label.get_text()
    def set_xlabel(self, xlabel, fontdict=None, labelpad=None, *,
                   loc=None, **kwargs):
        """
        Set the label for the x-axis.

        Parameters
        ----------
        xlabel : str
            The label text.

        labelpad : float, default: :rc:`axes.labelpad`
            Spacing in points from the Axes bounding box including ticks
            and tick labels.  If None, the previous value is left as is.

        loc : {'left', 'center', 'right'}, default: :rc:`xaxis.labellocation`
            The label position. This is a high-level alternative for passing
            parameters *x* and *horizontalalignment*.

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.text.Text` properties
            `.Text` properties control the appearance of the label.

        See Also
        --------
        text : Documents the properties supported by `.Text`.
        """
        if labelpad is not None:
            # 如果提供了labelpad参数，则更新x轴标签的间距
            self.xaxis.labelpad = labelpad
        
        protected_kw = ['x', 'horizontalalignment', 'ha']
        if {*kwargs} & {*protected_kw}:
            # 如果kwargs包含任何受保护的关键字参数，则不允许同时指定loc参数
            if loc is not None:
                raise TypeError(f"Specifying 'loc' is disallowed when any of "
                                f"its corresponding low level keyword "
                                f"arguments ({protected_kw}) are also "
                                f"supplied")
        else:
            # 否则，确定标签位置loc的值，或者使用默认配置
            loc = (loc if loc is not None
                   else mpl.rcParams['xaxis.labellocation'])
            _api.check_in_list(('left', 'center', 'right'), loc=loc)

            # 根据loc的不同值，更新kwargs以设置水平对齐和x位置
            x = {
                'left': 0,
                'center': 0.5,
                'right': 1,
            }[loc]
            kwargs.update(x=x, horizontalalignment=loc)

        # 最终设置x轴标签文本及其属性
        return self.xaxis.set_label_text(xlabel, fontdict, **kwargs)

    def invert_xaxis(self):
        """
        Invert the x-axis.

        See Also
        --------
        xaxis_inverted
        get_xlim, set_xlim
        get_xbound, set_xbound
        """
        # 反转x轴方向
        self.xaxis.set_inverted(not self.xaxis.get_inverted())

    xaxis_inverted = _axis_method_wrapper("xaxis", "get_inverted")

    def get_xbound(self):
        """
        Return the lower and upper x-axis bounds, in increasing order.

        See Also
        --------
        set_xbound
        get_xlim, set_xlim
        invert_xaxis, xaxis_inverted
        """
        # 获取当前x轴的边界值，并确保返回的是按增序排列的边界
        left, right = self.get_xlim()
        if left < right:
            return left, right
        else:
            return right, left
    def set_xbound(self, lower=None, upper=None):
        """
        Set the lower and upper numerical bounds of the x-axis.

        This method will honor axis inversion regardless of parameter order.
        It will not change the autoscaling setting (`.get_autoscalex_on()`).

        Parameters
        ----------
        lower, upper : float or None
            The lower and upper bounds. If *None*, the respective axis bound
            is not modified.

            .. ACCEPTS: (lower: float, upper: float)

        See Also
        --------
        get_xbound
        get_xlim, set_xlim
        invert_xaxis, xaxis_inverted
        """
        # 如果 upper 为 None 且 lower 是可迭代对象，则将 lower 和 upper 解包赋值
        if upper is None and np.iterable(lower):
            lower, upper = lower

        # 获取当前的 x 轴界限
        old_lower, old_upper = self.get_xbound()
        # 如果 lower 是 None，则使用旧的 lower
        if lower is None:
            lower = old_lower
        # 如果 upper 是 None，则使用旧的 upper
        if upper is None:
            upper = old_upper

        # 根据是否 x 轴被反转，决定如何排序 lower 和 upper，并应用于 x 轴界限
        self.set_xlim(sorted((lower, upper),
                             reverse=bool(self.xaxis_inverted())),
                      auto=None)

    def get_xlim(self):
        """
        Return the x-axis view limits.

        Returns
        -------
        left, right : (float, float)
            The current x-axis limits in data coordinates.

        See Also
        --------
        .Axes.set_xlim
        .Axes.set_xbound, .Axes.get_xbound
        .Axes.invert_xaxis, .Axes.xaxis_inverted

        Notes
        -----
        The x-axis may be inverted, in which case the *left* value will
        be greater than the *right* value.
        """
        # 返回当前 x 轴的视图限制
        return tuple(self.viewLim.intervalx)

    def _validate_converted_limits(self, limit, convert):
        """
        Raise ValueError if converted limits are non-finite.

        Note that this function also accepts None as a limit argument.

        Returns
        -------
        The limit value after call to convert(), or None if limit is None.
        """
        # 如果 limit 不为 None，则进行转换，并检查是否为有限数值，否则抛出 ValueError
        if limit is not None:
            converted_limit = convert(limit)
            if isinstance(converted_limit, np.ndarray):
                converted_limit = converted_limit.squeeze()
            if (isinstance(converted_limit, Real)
                    and not np.isfinite(converted_limit)):
                raise ValueError("Axis limits cannot be NaN or Inf")
            return converted_limit
    def set_xlim(self, left=None, right=None, *, emit=True, auto=False,
                 xmin=None, xmax=None):
        """
        Set the x-axis view limits.

        Parameters
        ----------
        left : float, optional
            The left xlim in data coordinates. Passing *None* leaves the
            limit unchanged.

            The left and right xlims may also be passed as the tuple
            (*left*, *right*) as the first positional argument (or as
            the *left* keyword argument).

            .. ACCEPTS: (left: float, right: float)

        right : float, optional
            The right xlim in data coordinates. Passing *None* leaves the
            limit unchanged.

        emit : bool, default: True
            Whether to notify observers of limit change.

        auto : bool or None, default: False
            Whether to turn on autoscaling of the x-axis. True turns on,
            False turns off, None leaves unchanged.

        xmin, xmax : float, optional
            They are equivalent to left and right respectively, and it is an
            error to pass both *xmin* and *left* or *xmax* and *right*.

        Returns
        -------
        left, right : (float, float)
            The new x-axis limits in data coordinates.

        See Also
        --------
        get_xlim
        set_xbound, get_xbound
        invert_xaxis, xaxis_inverted

        Notes
        -----
        The *left* value may be greater than the *right* value, in which
        case the x-axis values will decrease from left to right.

        Examples
        --------
        >>> set_xlim(left, right)
        >>> set_xlim((left, right))
        >>> left, right = set_xlim(left, right)

        One limit may be left unchanged.

        >>> set_xlim(right=right_lim)

        Limits may be passed in reverse order to flip the direction of
        the x-axis. For example, suppose *x* represents the number of
        years before present. The x-axis limits might be set like the
        following so 5000 years ago is on the left of the plot and the
        present is on the right.

        >>> set_xlim(5000, 0)
        """
        # 如果 right 参数为 None，并且 left 参数是可迭代对象，则解包 left 作为 left 和 right
        if right is None and np.iterable(left):
            left, right = left
        # 如果 xmin 参数不为 None，则检查 left 是否为 None，如果不是则抛出 TypeError
        if xmin is not None:
            if left is not None:
                raise TypeError("Cannot pass both 'left' and 'xmin'")
            left = xmin
        # 如果 xmax 参数不为 None，则检查 right 是否为 None，如果不是则抛出 TypeError
        if xmax is not None:
            if right is not None:
                raise TypeError("Cannot pass both 'right' and 'xmax'")
            right = xmax
        # 调用 self.xaxis._set_lim 方法来设置 x 轴的限制，并返回结果
        return self.xaxis._set_lim(left, right, emit=emit, auto=auto)

    # 获取 x 轴的比例尺度，通过 _axis_method_wrapper 方法实现
    get_xscale = _axis_method_wrapper("xaxis", "get_scale")
    # 设置 x 轴的比例尺度，通过 _axis_method_wrapper 方法实现
    set_xscale = _axis_method_wrapper("xaxis", "_set_axes_scale")
    # 获取 x 轴的刻度位置，通过 _axis_method_wrapper 方法实现
    get_xticks = _axis_method_wrapper("xaxis", "get_ticklocs")
    # 设置 x 轴的刻度位置，通过 _axis_method_wrapper 方法实现，并替换文档中的 set_ticks 为 set_xticks
    set_xticks = _axis_method_wrapper("xaxis", "set_ticks",
                                      doc_sub={'set_ticks': 'set_xticks'})
    # 使用辅助函数 _axis_method_wrapper 封装获取 x 轴主刻度标签的方法
    get_xmajorticklabels = _axis_method_wrapper("xaxis", "get_majorticklabels")
    # 使用辅助函数 _axis_method_wrapper 封装获取 x 轴次刻度标签的方法
    get_xminorticklabels = _axis_method_wrapper("xaxis", "get_minorticklabels")
    # 使用辅助函数 _axis_method_wrapper 封装获取 x 轴所有刻度标签的方法
    get_xticklabels = _axis_method_wrapper("xaxis", "get_ticklabels")
    # 使用辅助函数 _axis_method_wrapper 封装设置 x 轴所有刻度标签的方法，并通过 doc_sub 参数替换文档中的内容
    set_xticklabels = _axis_method_wrapper(
        "xaxis", "set_ticklabels",
        doc_sub={"Axis.set_ticks": "Axes.set_xticks"})

    def get_ylabel(self):
        """
        Get the ylabel text string.
        """
        # 获取 y 轴标签文本
        label = self.yaxis.get_label()
        return label.get_text()

    def set_ylabel(self, ylabel, fontdict=None, labelpad=None, *,
                   loc=None, **kwargs):
        """
        Set the label for the y-axis.

        Parameters
        ----------
        ylabel : str
            The label text.

        labelpad : float, default: :rc:`axes.labelpad`
            Spacing in points from the Axes bounding box including ticks
            and tick labels.  If None, the previous value is left as is.

        loc : {'bottom', 'center', 'top'}, default: :rc:`yaxis.labellocation`
            The label position. This is a high-level alternative for passing
            parameters *y* and *horizontalalignment*.

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.text.Text` properties
            `.Text` properties control the appearance of the label.

        See Also
        --------
        text : Documents the properties supported by `.Text`.
        """
        # 如果指定了 labelpad，则设置 y 轴标签的间距
        if labelpad is not None:
            self.yaxis.labelpad = labelpad
        # 保护关键字列表，用于检查是否有不兼容的关键字参数传入
        protected_kw = ['y', 'horizontalalignment', 'ha']
        if {*kwargs} & {*protected_kw}:
            # 如果传入了保护关键字参数且同时传入了 loc 参数，引发 TypeError
            if loc is not None:
                raise TypeError(f"Specifying 'loc' is disallowed when any of "
                                f"its corresponding low level keyword "
                                f"arguments ({protected_kw}) are also "
                                f"supplied")

        else:
            # 如果未传入保护关键字参数，则根据 loc 参数或默认配置确定 y 和水平对齐方式
            loc = (loc if loc is not None
                   else mpl.rcParams['yaxis.labellocation'])
            _api.check_in_list(('bottom', 'center', 'top'), loc=loc)

            y, ha = {
                'bottom': (0, 'left'),
                'center': (0.5, 'center'),
                'top': (1, 'right')
            }[loc]
            kwargs.update(y=y, horizontalalignment=ha)

        # 调用 y 轴的 set_label_text 方法设置标签文本，并传入相关参数
        return self.yaxis.set_label_text(ylabel, fontdict, **kwargs)

    def invert_yaxis(self):
        """
        Invert the y-axis.

        See Also
        --------
        yaxis_inverted
        get_ylim, set_ylim
        get_ybound, set_ybound
        """
        # 反转 y 轴方向
        self.yaxis.set_inverted(not self.yaxis.get_inverted())

    # 使用辅助函数 _axis_method_wrapper 封装获取 y 轴反转状态的方法
    yaxis_inverted = _axis_method_wrapper("yaxis", "get_inverted")
    def get_ybound(self):
        """
        Return the lower and upper y-axis bounds, in increasing order.

        See Also
        --------
        set_ybound
        get_ylim, set_ylim
        invert_yaxis, yaxis_inverted
        """
        # 获取当前 y 轴的数据视图上下限
        bottom, top = self.get_ylim()
        # 如果 bottom 小于 top，则按升序返回 bottom 和 top
        if bottom < top:
            return bottom, top
        else:
            return top, bottom

    def set_ybound(self, lower=None, upper=None):
        """
        Set the lower and upper numerical bounds of the y-axis.

        This method will honor axis inversion regardless of parameter order.
        It will not change the autoscaling setting (`.get_autoscaley_on()`).

        Parameters
        ----------
        lower, upper : float or None
            The lower and upper bounds. If *None*, the respective axis bound
            is not modified.

         .. ACCEPTS: (lower: float, upper: float)

        See Also
        --------
        get_ybound
        get_ylim, set_ylim
        invert_yaxis, yaxis_inverted
        """
        # 如果 upper 是 None 并且 lower 是一个可迭代对象，则解包 lower
        if upper is None and np.iterable(lower):
            lower, upper = lower

        # 获取当前的 y 轴界限
        old_lower, old_upper = self.get_ybound()
        # 如果 lower 是 None，则使用原始的 lower
        if lower is None:
            lower = old_lower
        # 如果 upper 是 None，则使用原始的 upper
        if upper is None:
            upper = old_upper

        # 设置 y 轴界限，按照是否反转 y 轴来决定排序顺序
        self.set_ylim(sorted((lower, upper),
                             reverse=bool(self.yaxis_inverted())),
                      auto=None)

    def get_ylim(self):
        """
        Return the y-axis view limits.

        Returns
        -------
        bottom, top : (float, float)
            The current y-axis limits in data coordinates.

        See Also
        --------
        .Axes.set_ylim
        .Axes.set_ybound, .Axes.get_ybound
        .Axes.invert_yaxis, .Axes.yaxis_inverted

        Notes
        -----
        The y-axis may be inverted, in which case the *bottom* value
        will be greater than the *top* value.
        """
        # 返回当前数据坐标中的 y 轴限制
        return tuple(self.viewLim.intervaly)
    def set_ylim(self, bottom=None, top=None, *, emit=True, auto=False,
                 ymin=None, ymax=None):
        """
        设置 y 轴的视图限制。

        Parameters
        ----------
        bottom : float, optional
            数据坐标中的下限 ylim。传递 *None* 表示保持限制不变。

            下限和上限也可以作为元组 (*bottom*, *top*) 作为第一个位置参数传递（或作为
            *bottom* 关键字参数）。

            .. ACCEPTS: (bottom: float, top: float)

        top : float, optional
            数据坐标中的上限 ylim。传递 *None* 表示保持限制不变。

        emit : bool, default: True
            是否通知观察者限制变化。

        auto : bool or None, default: False
            是否启用 y 轴的自动缩放。*True* 启用，*False* 禁用，*None* 表示保持不变。

        ymin, ymax : float, optional
            它们分别等同于 bottom 和 top，同时传递 *ymin* 和 *bottom* 或 *ymax* 和 *top*
            是错误的。

        Returns
        -------
        bottom, top : (float, float)
            新的数据坐标中的 y 轴限制。

        See Also
        --------
        get_ylim
        set_ybound, get_ybound
        invert_yaxis, yaxis_inverted

        Notes
        -----
        *bottom* 的值可以大于 *top* 的值，此时 y 轴的值将从 *bottom* 递减到 *top*。

        Examples
        --------
        >>> set_ylim(bottom, top)
        >>> set_ylim((bottom, top))
        >>> bottom, top = set_ylim(bottom, top)

        可以保持一个限制不变。

        >>> set_ylim(top=top_lim)

        可以反向传递限制以翻转 y 轴的方向。例如，假设 `y` 表示海洋的深度（以米为单位）。
        可以通过以下方式设置 y 轴限制，使得深度为 5000 米的底部在图中，而表面 0 米在顶部。

        >>> set_ylim(5000, 0)
        """
        # 如果 top 是 None 并且 bottom 是可迭代的 numpy 数组，则解压元组
        if top is None and np.iterable(bottom):
            bottom, top = bottom
        # 如果 ymin 不是 None，则处理它，并检查是否与 bottom 冲突
        if ymin is not None:
            if bottom is not None:
                raise TypeError("Cannot pass both 'bottom' and 'ymin'")
            bottom = ymin
        # 如果 ymax 不是 None，则处理它，并检查是否与 top 冲突
        if ymax is not None:
            if top is not None:
                raise TypeError("Cannot pass both 'top' and 'ymax'")
            top = ymax
        # 调用 y 轴对象的 _set_lim 方法来设置新的下限和上限，并返回结果
        return self.yaxis._set_lim(bottom, top, emit=emit, auto=auto)

    # 获取 y 轴的刻度比例
    get_yscale = _axis_method_wrapper("yaxis", "get_scale")
    # 设置 y 轴的刻度比例
    set_yscale = _axis_method_wrapper("yaxis", "_set_axes_scale")
    # 获取 y 轴的刻度位置
    get_yticks = _axis_method_wrapper("yaxis", "get_ticklocs")
    # 设置 y 轴的刻度位置
    set_yticks = _axis_method_wrapper("yaxis", "set_ticks",
                                      doc_sub={'set_ticks': 'set_yticks'})
    # 使用 _axis_method_wrapper 函数创建获取 y 轴主刻度标签的函数
    get_ymajorticklabels = _axis_method_wrapper("yaxis", "get_majorticklabels")
    # 使用 _axis_method_wrapper 函数创建获取 y 轴次刻度标签的函数
    get_yminorticklabels = _axis_method_wrapper("yaxis", "get_minorticklabels")
    # 使用 _axis_method_wrapper 函数创建获取 y 轴所有刻度标签的函数
    get_yticklabels = _axis_method_wrapper("yaxis", "get_ticklabels")
    # 使用 _axis_method_wrapper 函数创建设置 y 轴刻度标签的函数，并提供文档替换
    set_yticklabels = _axis_method_wrapper(
        "yaxis", "set_ticklabels",
        doc_sub={"Axis.set_ticks": "Axes.set_yticks"})

    # 使用 _axis_method_wrapper 函数创建 x 轴的 axis_date 方法的函数
    xaxis_date = _axis_method_wrapper("xaxis", "axis_date")
    # 使用 _axis_method_wrapper 函数创建 y 轴的 axis_date 方法的函数
    yaxis_date = _axis_method_wrapper("yaxis", "axis_date")

    def format_xdata(self, x):
        """
        Return *x* formatted as an x-value.

        This function will use the `.fmt_xdata` attribute if it is not None,
        else will fall back on the xaxis major formatter.
        """
        # 如果存在 fmt_xdata 属性，则使用该属性格式化 x 值；否则使用 x 轴的主要格式化器
        return (self.fmt_xdata if self.fmt_xdata is not None
                else self.xaxis.get_major_formatter().format_data_short)(x)

    def format_ydata(self, y):
        """
        Return *y* formatted as a y-value.

        This function will use the `.fmt_ydata` attribute if it is not None,
        else will fall back on the yaxis major formatter.
        """
        # 如果存在 fmt_ydata 属性，则使用该属性格式化 y 值；否则使用 y 轴的主要格式化器
        return (self.fmt_ydata if self.fmt_ydata is not None
                else self.yaxis.get_major_formatter().format_data_short)(y)

    def format_coord(self, x, y):
        """Return a format string formatting the *x*, *y* coordinates."""
        # 获取与当前轴成对的所有轴对象
        twins = self._twinned_axes.get_siblings(self)
        # 如果只有一个成对轴，则返回格式化后的坐标字符串
        if len(twins) == 1:
            return "(x, y) = ({}, {})".format(
                "???" if x is None else self.format_xdata(x),
                "???" if y is None else self.format_ydata(y))
        
        # 将数据坐标 (x, y) 转换为屏幕坐标
        screen_xy = self.transData.transform((x, y))
        xy_strs = []
        # 对所有成对轴进行排序，以便按照添加到图形中的顺序排序
        for ax in sorted(twins, key=attrgetter("zorder")):
            # 将屏幕坐标转换为当前成对轴的数据坐标
            data_x, data_y = ax.transData.inverted().transform(screen_xy)
            # 格式化成对轴的数据坐标，并加入列表中
            xy_strs.append(
                "({}, {})".format(ax.format_xdata(data_x), ax.format_ydata(data_y)))
        # 返回格式化后的多个成对轴的坐标字符串
        return "(x, y) = {}".format(" | ".join(xy_strs))

    def minorticks_on(self):
        """
        Display minor ticks on the Axes.

        Displaying minor ticks may reduce performance; you may turn them off
        using `minorticks_off()` if drawing speed is a problem.
        """
        # 打开 x 轴和 y 轴的次刻度
        self.xaxis.minorticks_on()
        self.yaxis.minorticks_on()

    def minorticks_off(self):
        """Remove minor ticks from the Axes."""
        # 关闭 x 轴和 y 轴的次刻度
        self.xaxis.minorticks_off()
        self.yaxis.minorticks_off()

    # Interactive manipulation

    def can_zoom(self):
        """
        Return whether this Axes supports the zoom box button functionality.
        """
        # 返回此 Axes 是否支持缩放框按钮功能
        return True

    def can_pan(self):
        """
        Return whether this Axes supports any pan/zoom button functionality.
        """
        # 返回此 Axes 是否支持任何平移/缩放按钮功能
        return True
    # 返回当前 Axes 对象是否响应导航命令的状态。
    def get_navigate(self):
        return self._navigate

    # 设置当前 Axes 对象是否响应导航工具栏命令。
    #
    # 参数
    # ------
    # b : bool
    #     布尔类型，表示是否响应导航工具栏命令。
    #
    # 参见
    # ------
    # matplotlib.axes.Axes.set_forward_navigation_events
    #
    def set_navigate(self, b):
        self._navigate = b

    # 返回当前 Axes 对象导航工具栏按钮的状态: 'PAN', 'ZOOM' 或 None。
    def get_navigate_mode(self):
        return self._navigate_mode

    # 设置当前 Axes 对象导航工具栏按钮的状态。
    #
    # .. warning::
    #     这不是用户API函数。
    #
    # 参数
    # ------
    # b : bool
    #     导航工具栏按钮的状态。
    #
    def set_navigate_mode(self, b):
        self._navigate_mode = b

    # 保存重现当前视图所需的信息。
    #
    # 在用户发起平移或缩放等视图更改操作之前调用此方法。
    # 返回一个描述当前视图的不透明对象，格式与 :meth:`_set_view` 兼容。
    #
    # 默认实现保存视图限制和自动缩放状态。
    # 子类可以根据需要重写此方法，只要同时调整 :meth:`_set_view` 方法即可。
    #
    def _get_view(self):
        return {
            "xlim": self.get_xlim(), "autoscalex_on": self.get_autoscalex_on(),
            "ylim": self.get_ylim(), "autoscaley_on": self.get_autoscaley_on(),
        }

    # 应用先前保存的视图。
    #
    # 在恢复视图时（使用 :meth:`_get_view` 的返回值作为参数），例如使用导航按钮时调用此方法。
    #
    # 如果子类重写了 :meth:`_get_view` 方法，还需要相应地重写此方法。
    #
    def _set_view(self, view):
        self.set(**view)
    # 更新视图以匹配选择框的边界框

    # 如果不是x轴方向的镜像且模式不是“y”，设置新的x轴边界
    self.set_xbound(new_xbound)
    # 关闭自动缩放x轴
    self.set_autoscalex_on(False)

    # 如果不是y轴方向的镜像且模式不是“x”，设置新的y轴边界
    self.set_ybound(new_ybound)
    # 关闭自动缩放y轴

    # 当开始一个平移操作时调用
    self._pan_start = types.SimpleNamespace(
        # 使用当前视图边界的冻结版本
        lim=self.viewLim.frozen(),
        # 使用数据到显示坐标的转换的冻结版本
        trans=self.transData.frozen(),
        # 使用数据到显示坐标的反向转换的冻结版本
        trans_inverse=self.transData.inverted().frozen(),
        # 使用冻结版本的bbox
        bbox=self.bbox.frozen(),
        # 设置鼠标的x坐标
        x=x,
        # 设置鼠标的y坐标
        y=y)

    # 当平移操作完成时调用
    del self._pan_start
    def _get_pan_points(self, button, key, x, y):
        """
        Helper function to return the new points after a pan.

        This helper function returns the points on the axis after a pan has
        occurred. This is a convenience method to abstract the pan logic
        out of the base setter.
        """
        # 定义一个内部函数，用于根据键盘按键和位移计算格式化后的位移量
        def format_deltas(key, dx, dy):
            # 如果按键是'control'，则优先水平或垂直方向等量移动
            if key == 'control':
                if abs(dx) > abs(dy):
                    dy = dx
                else:
                    dx = dy
            # 如果按键是'x'，则只在水平方向移动
            elif key == 'x':
                dy = 0
            # 如果按键是'y'，则只在垂直方向移动
            elif key == 'y':
                dx = 0
            # 如果按键是'shift'，则根据位移比例调整移动量
            elif key == 'shift':
                if 2 * abs(dx) < abs(dy):
                    dx = 0
                elif 2 * abs(dy) < abs(dx):
                    dy = 0
                elif abs(dx) > abs(dy):
                    dy = dy / abs(dy) * abs(dx)
                else:
                    dx = dx / abs(dx) * abs(dy)
            return dx, dy

        # 获取平移操作起始点
        p = self._pan_start
        dx = x - p.x
        dy = y - p.y
        # 如果位移量为零，则返回
        if dx == dy == 0:
            return
        # 如果鼠标左键被按下
        if button == 1:
            # 根据键盘按键和位移计算格式化后的位移量
            dx, dy = format_deltas(key, dx, dy)
            # 根据位移量更新边界框并进行逆变换
            result = p.bbox.translated(-dx, -dy).transformed(p.trans_inverse)
        # 如果鼠标右键被按下
        elif button == 3:
            try:
                # 计算归一化的位移量
                dx = -dx / self.bbox.width
                dy = -dy / self.bbox.height
                # 根据键盘按键和位移计算格式化后的位移量
                dx, dy = format_deltas(key, dx, dy)
                # 如果比例尺不是'auto'，则重新计算位移量
                if self.get_aspect() != 'auto':
                    dx = dy = 0.5 * (dx + dy)
                # 计算变换系数
                alpha = np.power(10.0, (dx, dy))
                start = np.array([p.x, p.y])
                oldpoints = p.lim.transformed(p.trans)
                # 根据变换系数计算新的点坐标
                newpoints = start + alpha * (oldpoints - start)
                # 将新的点坐标转换为边界框对象，并进行逆变换
                result = (mtransforms.Bbox(newpoints)
                          .transformed(p.trans_inverse))
            except OverflowError:
                # 溢出错误处理
                _api.warn_external('Overflow while panning')
                return
        else:
            return

        # 检查是否有效，排除无效限制（例如对数尺度下的下溢）
        valid = np.isfinite(result.transformed(p.trans))
        points = result.get_points().astype(object)
        # 忽略无效限制（通常是对数尺度下的下溢）
        points[~valid] = None
        # 返回更新后的点坐标
        return points

    def drag_pan(self, button, key, x, y):
        """
        Called when the mouse moves during a pan operation.

        Parameters
        ----------
        button : `.MouseButton`
            The pressed mouse button.
        key : str or None
            The pressed key, if any.
        x, y : float
            The mouse coordinates in display coords.

        Notes
        -----
        This is intended to be overridden by new projection types.
        """
        # 获取平移后的新点坐标
        points = self._get_pan_points(button, key, x, y)
        # 如果有新的点坐标，则更新X轴和Y轴的限制
        if points is not None:
            self.set_xlim(points[:, 0])
            self.set_ylim(points[:, 1])
    # 返回所有子元素的列表，包括子图、轴脊柱、轴映射、标题以及图例等
    def get_children(self):
        # docstring inherited.
        return [
            *self._children,                    # 返回子图列表
            *self.spines.values(),              # 返回所有轴脊柱对象
            *self._axis_map.values(),           # 返回所有轴映射对象
            self.title,                         # 返回标题对象
            self._left_title,                   # 返回左侧标题对象
            self._right_title,                  # 返回右侧标题对象
            *self.child_axes,                   # 返回所有子轴对象
            *([self.legend_] if self.legend_ is not None else []),  # 返回图例对象（如果存在）
            self.patch,                         # 返回图表的补丁对象
        ]

    # 检查鼠标事件是否在图表的补丁区域内
    def contains(self, mouseevent):
        # docstring inherited.
        return self.patch.contains(mouseevent)

    # 检查指定的像素坐标点是否在图表的补丁区域内
    def contains_point(self, point):
        """
        Return whether *point* (pair of pixel coordinates) is inside the Axes
        patch.
        """
        return self.patch.contains_point(point, radius=1.0)

    # 返回默认的用于边界框计算的艺术家列表
    def get_default_bbox_extra_artists(self):
        """
        Return a default list of artists that are used for the bounding box
        calculation.

        Artists are excluded either by not being visible or
        ``artist.set_in_layout(False)``.
        """

        artists = self.get_children()  # 获取所有子元素列表

        for axis in self._axis_map.values():
            # 如果轴是紧凑的边界框，则在布局计算中单独计算
            artists.remove(axis)

        if not (self.axison and self._frameon):
            # 如果不显示轴线或者图表边框不显示，则不计算轴脊柱的边界框
            for spine in self.spines.values():
                artists.remove(spine)

        # 从艺术家列表中排除标题对象
        artists.remove(self.title)
        artists.remove(self._left_title)
        artists.remove(self._right_title)

        # 始终包括不内部实现剪裁到轴的类型
        noclip = (_AxesBase, maxis.Axis,
                  offsetbox.AnnotationBbox, offsetbox.OffsetBox)
        # 返回可见且在布局中的艺术家列表，不包括完全剪裁到轴的艺术家
        return [a for a in artists if a.get_visible() and a.get_in_layout()
                and (isinstance(a, noclip) or not a._fully_clipped_to_axes())]

    @_api.make_keyword_only("3.8", "call_axes_locator")
    def _make_twin_axes(self, *args, **kwargs):
        """
        创建一个与当前 Axes 共享 x 轴或 y 轴的双轴图。这个函数被用于创建 twinx 和 twiny。

        Parameters:
        -----------
        *args, **kwargs : 可变参数和关键字参数
            传递给 add_subplot 或 add_axes 方法的参数，用于创建新的 Axes。

        Raises:
        -------
        ValueError
            如果 'sharex' 和 'sharey' 同时在 kwargs 中，并且它们都不是当前 Axes 的实例，则抛出异常。

        Returns:
        --------
        Axes
            新创建的 Axes 实例，可以共享 x 轴或 y 轴。

        Notes:
        ------
        当使用 twinx 进行图形选择（picking）时，只有最顶部的 Axes 的艺术品（artists）才会调用选择事件。
        """
        if 'sharex' in kwargs and 'sharey' in kwargs:
            # 在 v2.2 中添加以下行，以避免破坏 Seaborn，后者当前使用这个内部 API。
            if kwargs["sharex"] is not self and kwargs["sharey"] is not self:
                raise ValueError("Twinned Axes may share only one axis")

        # 获取当前 Axes 的子图规范
        ss = self.get_subplotspec()
        if ss:
            # 如果存在子图规范，则使用它来添加子图
            twin = self.figure.add_subplot(ss, *args, **kwargs)
        else:
            # 否则，使用当前 Axes 的位置信息来添加 Axes，同时使用 _TransformedBoundsLocator 定位器
            twin = self.figure.add_axes(
                self.get_position(True), *args, **kwargs,
                axes_locator=_TransformedBoundsLocator(
                    [0, 0, 1, 1], self.transAxes))

        # 设置调整方式为 'datalim'
        self.set_adjustable('datalim')
        twin.set_adjustable('datalim')
        # 设置 zorder（堆叠顺序）
        twin.set_zorder(self.zorder)

        # 将当前 Axes 和新创建的 twin Axes 进行连接
        self._twinned_axes.join(self, twin)
        return twin

    def twinx(self):
        """
        创建一个共享 x 轴的双轴图。

        创建一个新的 Axes，其 x 轴是不可见的，y 轴是独立的，并位于原始轴的对面（即右侧）。
        x 轴的自动缩放设置将继承自原始 Axes。要确保两个 y 轴的刻度标记对齐，请参阅 `~matplotlib.ticker.LinearLocator`。

        Returns:
        --------
        Axes
            新创建的 Axes 实例

        Notes:
        ------
        当使用 twinx 进行图形选择（picking）时，只有最顶部的 Axes 的艺术品（artists）才会调用选择事件。
        """
        # 调用 _make_twin_axes 方法创建一个共享 x 轴的 twin Axes
        ax2 = self._make_twin_axes(sharex=self)
        # 将 y 轴的刻度标记设置在右侧
        ax2.yaxis.tick_right()
        # 设置 y 轴标签位置为右侧
        ax2.yaxis.set_label_position('right')
        # 设置 y 轴的偏移位置为右侧
        ax2.yaxis.set_offset_position('right')
        # 设置 x 轴的自动缩放状态与当前 Axes 一致
        ax2.set_autoscalex_on(self.get_autoscalex_on())
        # 将当前 Axes 的 y 轴刻度标记设置在左侧
        self.yaxis.tick_left()
        # 设置 twin Axes 的 x 轴不可见
        ax2.xaxis.set_visible(False)
        # 隐藏 twin Axes 的图形补丁
        ax2.patch.set_visible(False)
        # 设置 twin Axes 的 x 轴单位与当前 Axes 的相同
        ax2.xaxis.units = self.xaxis.units

        return ax2
    def twiny(self):
        """
        Create a twin Axes sharing the yaxis.

        Create a new Axes with an invisible y-axis and an independent
        x-axis positioned opposite to the original one (i.e. at top). The
        y-axis autoscale setting will be inherited from the original Axes.
        To ensure that the tick marks of both x-axes align, see
        `~matplotlib.ticker.LinearLocator`.

        Returns
        -------
        Axes
            The newly created Axes instance

        Notes
        -----
        For those who are 'picking' artists while using twiny, pick
        events are only called for the artists in the top-most Axes.
        """
        # 创建一个共享y轴的双重坐标轴Axes实例
        ax2 = self._make_twin_axes(sharey=self)
        # 将新坐标轴的x轴设置在顶部
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')
        # 继承原始Axes的y轴自动缩放设置
        ax2.set_autoscaley_on(self.get_autoscaley_on())
        # 将原始Axes的x轴设置在底部
        self.xaxis.tick_bottom()
        # 隐藏新坐标轴的y轴
        ax2.yaxis.set_visible(False)
        # 设置新坐标轴的背景为不可见
        ax2.patch.set_visible(False)
        # 继承原始Axes的y轴单位设置
        ax2.yaxis.units = self.yaxis.units
        # 返回新创建的Axes实例
        return ax2

    def get_shared_x_axes(self):
        """Return an immutable view on the shared x-axes Grouper."""
        # 返回共享x轴的Grouper的不可变视图
        return cbook.GrouperView(self._shared_axes["x"])

    def get_shared_y_axes(self):
        """Return an immutable view on the shared y-axes Grouper."""
        # 返回共享y轴的Grouper的不可变视图
        return cbook.GrouperView(self._shared_axes["y"])

    def label_outer(self, remove_inner_ticks=False):
        """
        Only show "outer" labels and tick labels.

        x-labels are only kept for subplots on the last row (or first row, if
        labels are on the top side); y-labels only for subplots on the first
        column (or last column, if labels are on the right side).

        Parameters
        ----------
        remove_inner_ticks : bool, default: False
            If True, remove the inner ticks as well (not only tick labels).

            .. versionadded:: 3.8
        """
        # 仅显示“外部”标签和刻度标签
        # 对x轴标签，在最后一行的子图保留（如果标签在顶部，则保留在第一行）；
        # 对y轴标签，在第一列的子图保留（如果标签在右侧，则保留在最后一列）。
        self._label_outer_xaxis(skip_non_rectangular_axes=False,
                                remove_inner_ticks=remove_inner_ticks)
        self._label_outer_yaxis(skip_non_rectangular_axes=False,
                                remove_inner_ticks=remove_inner_ticks)
    def _label_outer_xaxis(self, *, skip_non_rectangular_axes,
                           remove_inner_ticks=False):
        # 如果设置跳过非矩形轴，并且当前图形不是矩形，则返回
        if skip_non_rectangular_axes and not isinstance(self.patch,
                                                        mpl.patches.Rectangle):
            return
        
        # 获取当前子图规范
        ss = self.get_subplotspec()
        if not ss:
            return
        
        # 获取x轴标签位置
        label_position = self.xaxis.get_label_position()
        
        # 如果不是第一行，则移除顶部的标签/刻度标签/偏移文本
        if not ss.is_first_row():
            if label_position == "top":
                self.set_xlabel("")  # 清空x轴标签
            top_kw = {'top': False} if remove_inner_ticks else {}
            self.xaxis.set_tick_params(
                which="both", labeltop=False, **top_kw)  # 设置x轴刻度参数
            if self.xaxis.offsetText.get_position()[1] == 1:
                self.xaxis.offsetText.set_visible(False)  # 如果偏移文本位置在顶部，隐藏偏移文本
        
        # 如果不是最后一行，则移除底部的标签/刻度标签/偏移文本
        if not ss.is_last_row():
            if label_position == "bottom":
                self.set_xlabel("")  # 清空x轴标签
            bottom_kw = {'bottom': False} if remove_inner_ticks else {}
            self.xaxis.set_tick_params(
                which="both", labelbottom=False, **bottom_kw)  # 设置x轴刻度参数
            if self.xaxis.offsetText.get_position()[1] == 0:
                self.xaxis.offsetText.set_visible(False)  # 如果偏移文本位置在底部，隐藏偏移文本

    def _label_outer_yaxis(self, *, skip_non_rectangular_axes,
                           remove_inner_ticks=False):
        # 如果设置跳过非矩形轴，并且当前图形不是矩形，则返回
        if skip_non_rectangular_axes and not isinstance(self.patch,
                                                        mpl.patches.Rectangle):
            return
        
        # 获取当前子图规范
        ss = self.get_subplotspec()
        if not ss:
            return
        
        # 获取y轴标签位置
        label_position = self.yaxis.get_label_position()
        
        # 如果不是第一列，则移除左侧的标签/刻度标签/偏移文本
        if not ss.is_first_col():
            if label_position == "left":
                self.set_ylabel("")  # 清空y轴标签
            left_kw = {'left': False} if remove_inner_ticks else {}
            self.yaxis.set_tick_params(
                which="both", labelleft=False, **left_kw)  # 设置y轴刻度参数
            if self.yaxis.offsetText.get_position()[0] == 0:
                self.yaxis.offsetText.set_visible(False)  # 如果偏移文本位置在左侧，隐藏偏移文本
        
        # 如果不是最后一列，则移除右侧的标签/刻度标签/偏移文本
        if not ss.is_last_col():
            if label_position == "right":
                self.set_ylabel("")  # 清空y轴标签
            right_kw = {'right': False} if remove_inner_ticks else {}
            self.yaxis.set_tick_params(
                which="both", labelright=False, **right_kw)  # 设置y轴刻度参数
            if self.yaxis.offsetText.get_position()[0] == 1:
                self.yaxis.offsetText.set_visible(False)  # 如果偏移文本位置在右侧，隐藏偏移文本
    # 设置此 Axes 对象如何将平移/缩放事件转发给位于其下方的其他 Axes。

    def set_forward_navigation_events(self, forward):
        """
        Set how pan/zoom events are forwarded to Axes below this one.

        Parameters
        ----------
        forward : bool or "auto"
            Possible values:

            - True: Forward events to other axes with lower or equal zorder.
            - False: Events are only executed on this axes.
            - "auto": Default behaviour (*True* for axes with an invisible
              patch and *False* otherwise)

        See Also
        --------
        matplotlib.axes.Axes.set_navigate

        """
        # 将传入的 forward 参数赋值给当前对象的 _forward_navigation_events 属性
        self._forward_navigation_events = forward

    # 获取此 Axes 对象如何将平移/缩放事件转发给位于其下方的其他 Axes 的设置。
    def get_forward_navigation_events(self):
        """Get how pan/zoom events are forwarded to Axes below this one."""
        # 返回当前对象的 _forward_navigation_events 属性，表示事件转发设置
        return self._forward_navigation_events
# 定义一个辅助函数，用于将一组艺术家对象进行光栅化处理
def _draw_rasterized(figure, artists, renderer):
    """
    A helper function for rasterizing the list of artists.

    The bookkeeping to track if we are or are not in rasterizing mode
    with the mixed-mode backends is relatively complicated and is now
    handled in the matplotlib.artist.allow_rasterization decorator.

    This helper defines the absolute minimum methods and attributes on a
    shim class to be compatible with that decorator and then uses it to
    rasterize the list of artists.

    This is maybe too-clever, but allows us to reuse the same code that is
    used on normal artists to participate in the "are we rasterizing"
    accounting.

    Please do not use this outside of the "rasterize below a given zorder"
    functionality of Axes.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        所有艺术家所属的图表（未检查）。因为我们可以在图表级别抑制合成并将每个光栅化的艺术家插入为自己的图像，所以我们需要这个参数。

    artists : List[matplotlib.artist.Artist]
        要光栅化的艺术家列表。假设这些艺术家都属于同一个图表。

    renderer : matplotlib.backendbases.RendererBase
        当前活动的渲染器对象

    Returns
    -------
    None
    """
    # 定义一个最小化的艺术家类，以兼容光栅化装饰器，这样我们可以重用在普通艺术家中使用的相同代码来参与光栅化状态的管理。
    class _MinimalArtist:
        # 返回True，表示支持光栅化
        def get_rasterized(self):
            return True

        # 返回None，表示不使用任何聚合过滤器
        def get_agg_filter(self):
            return None

        # 初始化方法，接受figure和artists参数，并将其存储在实例属性中
        def __init__(self, figure, artists):
            self.figure = figure
            self.artists = artists

        # 使用装饰器allow_rasterization进行修饰，实现光栅化绘制功能
        @martist.allow_rasterization
        def draw(self, renderer):
            # 遍历艺术家列表，调用每个艺术家的draw方法进行渲染
            for a in self.artists:
                a.draw(renderer)

    # 创建_MinimalArtist类的实例，并调用其draw方法进行光栅化绘制操作
    return _MinimalArtist(figure, artists).draw(renderer)
```