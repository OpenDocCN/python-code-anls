# `D:\src\scipysrc\matplotlib\lib\matplotlib\cm.py`

```py
"""
Builtin colormaps, colormap handling utilities, and the `ScalarMappable` mixin.

.. seealso::

  :doc:`/gallery/color/colormap_reference` for a list of builtin colormaps.

  :ref:`colormap-manipulation` for examples of how to make
  colormaps.

  :ref:`colormaps` an in-depth discussion of choosing
  colormaps.

  :ref:`colormapnorms` for more details about data normalization.
"""

# 导入必要的模块和库
from collections.abc import Mapping  # 导入 Mapping 类，用于创建映射类型的对象
import functools  # 导入 functools 模块，用于高阶函数和函数工具

import numpy as np  # 导入 numpy 库，并用 np 别名表示
from numpy import ma  # 从 numpy 导入 ma 模块，用于掩码数组的处理

import matplotlib as mpl  # 导入 matplotlib 库，并用 mpl 别名表示
from matplotlib import _api, colors, cbook, scale  # 导入 matplotlib 中的一些模块和类
from matplotlib._cm import datad  # 导入 matplotlib._cm 模块中的 datad 对象
from matplotlib._cm_listed import cmaps as cmaps_listed  # 导入 matplotlib._cm_listed 中的 cmaps_listed 对象


_LUTSIZE = mpl.rcParams['image.lut']  # 从 matplotlib 的全局配置中获取图像颜色表的大小

def _gen_cmap_registry():
    """
    Generate a dict mapping standard colormap names to standard colormaps, as
    well as the reversed colormaps.
    """
    # 创建一个空字典，将已列出的标准色彩映射对象添加进去
    cmap_d = {**cmaps_listed}
    
    # 遍历 datad.items() 中的每一项，创建颜色映射对象并加入 cmap_d 字典中
    for name, spec in datad.items():
        cmap_d[name] = (
            # 如果 spec 中包含 'red' 键，创建线性分段的颜色映射对象
            colors.LinearSegmentedColormap(name, spec, _LUTSIZE)
            if 'red' in spec else
            # 如果 spec 中包含 'listed' 键，创建列出颜色的颜色映射对象
            colors.ListedColormap(spec['listed'], name)
            if 'listed' in spec else
            # 否则，创建基于列表的颜色映射对象
            colors.LinearSegmentedColormap.from_list(name, spec, _LUTSIZE)
        )

    # 注册灰度别名映射
    aliases = {
        'grey': 'gray',
        'gist_grey': 'gist_gray',
        'gist_yerg': 'gist_yarg',
        'Grays': 'Greys',
    }
    for alias, original_name in aliases.items():
        # 复制原始颜色映射对象，并设置别名
        cmap = cmap_d[original_name].copy()
        cmap.name = alias
        cmap_d[alias] = cmap

    # 生成反转的颜色映射对象
    for cmap in list(cmap_d.values()):
        rmap = cmap.reversed()  # 创建颜色映射对象的反转版本
        cmap_d[rmap.name] = rmap  # 将反转的映射对象加入 cmap_d 字典中

    return cmap_d  # 返回包含所有颜色映射对象的字典


class ColormapRegistry(Mapping):
    r"""
    Container for colormaps that are known to Matplotlib by name.

    The universal registry instance is `matplotlib.colormaps`. There should be
    no need for users to instantiate `.ColormapRegistry` themselves.

    Read access uses a dict-like interface mapping names to `.Colormap`\s::

        import matplotlib as mpl
        cmap = mpl.colormaps['viridis']

    Returned `.Colormap`\s are copies, so that their modification does not
    change the global definition of the colormap.

    Additional colormaps can be added via `.ColormapRegistry.register`::

        mpl.colormaps.register(my_colormap)

    To get a list of all registered colormaps, you can do::

        from matplotlib import colormaps
        list(colormaps)
    """
    
    def __init__(self, cmaps):
        self._cmaps = cmaps  # 初始化注册的颜色映射字典
        self._builtin_cmaps = tuple(cmaps)  # 将所有已注册的颜色映射名称转为元组保存

    def __getitem__(self, item):
        try:
            return self._cmaps[item].copy()  # 返回指定名称的颜色映射对象的副本
        except KeyError:
            raise KeyError(f"{item!r} is not a known colormap name") from None

    def __iter__(self):
        return iter(self._cmaps)  # 返回颜色映射字典的迭代器

    def __len__(self):
        return len(self._cmaps)  # 返回颜色映射字典的长度
    def __str__(self):
        # 返回 ColormapRegistry 的字符串表示形式，列出所有注册的颜色映射名字
        return ('ColormapRegistry; available colormaps:\n' +
                ', '.join(f"'{name}'" for name in self))

    def __call__(self):
        """
        Return a list of the registered colormap names.

        This exists only for backward-compatibility in `.pyplot` which had a
        ``plt.colormaps()`` method. The recommended way to get this list is
        now ``list(colormaps)``.
        """
        # 返回一个包含所有已注册颜色映射名字的列表
        return list(self)

    def register(self, cmap, *, name=None, force=False):
        """
        Register a new colormap.

        The colormap name can then be used as a string argument to any ``cmap``
        parameter in Matplotlib. It is also available in ``pyplot.get_cmap``.

        The colormap registry stores a copy of the given colormap, so that
        future changes to the original colormap instance do not affect the
        registered colormap. Think of this as the registry taking a snapshot
        of the colormap at registration.

        Parameters
        ----------
        cmap : matplotlib.colors.Colormap
            The colormap to register.

        name : str, optional
            The name for the colormap. If not given, ``cmap.name`` is used.

        force : bool, default: False
            If False, a ValueError is raised if trying to overwrite an already
            registered name. True supports overwriting registered colormaps
            other than the builtin colormaps.
        """
        # 检查传入的 cmap 是否为 matplotlib.colors.Colormap 的实例
        _api.check_isinstance(colors.Colormap, cmap=cmap)

        # 如果未提供 name 参数，则使用 cmap 的默认名字
        name = name or cmap.name
        # 如果 name 已经存在于注册表中
        if name in self:
            # 如果 force 参数为 False，则不允许覆盖已经存在的 colormap
            if not force:
                raise ValueError(
                    f'A colormap named "{name}" is already registered.')
            # 如果 force 为 True 且 name 是内置 colormap 的名字，则不允许覆盖
            elif name in self._builtin_cmaps:
                raise ValueError("Re-registering the builtin cmap "
                                 f"{name!r} is not allowed.")

            # 发出警告，表明正在覆盖已经存在的 colormap
            _api.warn_external(f"Overwriting the cmap {name!r} "
                               "that was already in the registry.")

        # 在注册表中存储给定 colormap 的副本
        self._cmaps[name] = cmap.copy()
        # 如果注册的 colormap 的名字与原始的不同，更新注册的 colormap 的名字
        if self._cmaps[name].name != name:
            self._cmaps[name].name = name
    def unregister(self, name):
        """
        Remove a colormap from the registry.

        You cannot remove built-in colormaps.

        If the named colormap is not registered, returns with no error, raises
        if you try to de-register a default colormap.

        .. warning::

            Colormap names are currently a shared namespace that may be used
            by multiple packages. Use `unregister` only if you know you
            have registered that name before. In particular, do not
            unregister just in case to clean the name before registering a
            new colormap.

        Parameters
        ----------
        name : str
            The name of the colormap to be removed.

        Raises
        ------
        ValueError
            If you try to remove a default built-in colormap.
        """
        # Check if the colormap name is in the list of built-in colormaps
        if name in self._builtin_cmaps:
            # Raise an error if trying to unregister a built-in colormap
            raise ValueError(f"cannot unregister {name!r} which is a builtin "
                             "colormap.")
        # Remove the colormap from the colormap registry
        self._cmaps.pop(name, None)

    def get_cmap(self, cmap):
        """
        Return a color map specified through *cmap*.

        Parameters
        ----------
        cmap : str or `~matplotlib.colors.Colormap` or None

            - if a `.Colormap`, return it
            - if a string, look it up in ``mpl.colormaps``
            - if None, return the Colormap defined in :rc:`image.cmap`

        Returns
        -------
        Colormap
        """
        # get the default color map
        if cmap is None:
            # Return the default colormap defined in matplotlib configuration
            return self[mpl.rcParams["image.cmap"]]

        # if the user passed in a Colormap, simply return it
        if isinstance(cmap, colors.Colormap):
            # Return the passed colormap object
            return cmap
        if isinstance(cmap, str):
            # Ensure the colormap name is in the list of available colormaps
            _api.check_in_list(sorted(_colormaps), cmap=cmap)
            # otherwise, it must be a string so look it up
            return self[cmap]
        # Raise a type error if the input is not None, Colormap, or str
        raise TypeError(
            'get_cmap expects None or an instance of a str or Colormap . ' +
            f'you passed {cmap!r} of type {type(cmap)}'
        )
# 创建颜色映射注册表实例，并更新全局作用域
_colormaps = ColormapRegistry(_gen_cmap_registry())
globals().update(_colormaps)

# 这是 pyplot.get_cmap() 的精确副本。在 3.9 版本中被移除，但导致更多用户问题，因此在 3.9.1 版本中重新添加，
# 并延长了两个额外的小版本的弃用周期。
@_api.deprecated(
    '3.7',
    removal='3.11',
    alternative="``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap()``"
                " or ``pyplot.get_cmap()``"
    )
def get_cmap(name=None, lut=None):
    """
    获取一个颜色映射实例，默认情况下使用 rc 参数值（如果 name 为 None）。

    Parameters
    ----------
    name : `~matplotlib.colors.Colormap` or str or None, default: None
        如果是 `.Colormap` 实例，则返回该实例。否则，是 Matplotlib 中已知的颜色映射名称，
        将由 *lut* 重新采样。默认为 None，意味着使用 :rc:`image.cmap`。
    lut : int or None, default: None
        如果 *name* 不是 Colormap 实例且 *lut* 不为 None，则颜色映射将重新采样为 *lut* 个条目的查找表。

    Returns
    -------
    Colormap
    """
    if name is None:
        name = mpl.rcParams['image.cmap']
    if isinstance(name, colors.Colormap):
        return name
    _api.check_in_list(sorted(_colormaps), name=name)
    if lut is None:
        return _colormaps[name]
    else:
        return _colormaps[name].resampled(lut)


def _auto_norm_from_scale(scale_cls):
    """
    根据 *scale_cls* 自动生成一个规范化类。

    与 `.colors.make_norm_from_scale` 的不同点在于：

    - 本函数不是一个类装饰器，而是直接返回一个规范化类（就像装饰 `.Normalize` 一样）。
    - 如果支持这样的参数，自动构建的尺度将自动使用 ``nonpositive="mask"`` 来解决标准尺度
      （使用 "clip"）和规范化（使用 "mask"）之间默认设置的差异。

    注意，``make_norm_from_scale`` 会缓存生成的规范化类（而不是实例），并在后续调用中重用它们。
    例如，``type(_auto_norm_from_scale("log")) == LogNorm``。
    """
    # 尝试构建一个实例，以验证是否支持 ``nonpositive="mask"``。
    try:
        norm = colors.make_norm_from_scale(
            functools.partial(scale_cls, nonpositive="mask"))(
            colors.Normalize)()
    except TypeError:
        norm = colors.make_norm_from_scale(scale_cls)(
            colors.Normalize)()
    return type(norm)


class ScalarMappable:
    """
    一个混合类，用于将标量数据映射到 RGBA 颜色。

    ScalarMappable 在返回给定颜色映射的 RGBA 颜色之前应用数据规范化。
    """
    def __init__(self, norm=None, cmap=None):
        """
        Parameters
        ----------
        norm : `.Normalize` (or subclass thereof) or str or None
            The normalizing object which scales data, typically into the
            interval ``[0, 1]``.
            If a `str`, a `.Normalize` subclass is dynamically generated based
            on the scale with the corresponding name.
            If *None*, *norm* defaults to a *colors.Normalize* object which
            initializes its scaling based on the first data processed.
        cmap : str or `~matplotlib.colors.Colormap`
            The colormap used to map normalized data values to RGBA colors.
        """
        # 初始化函数，设置对象的初始状态
        self._A = None
        self._norm = None  # So that the setter knows we're initializing.
        self.set_norm(norm)  # 设置标准化对象的方法
        self.cmap = None  # So that the setter knows we're initializing.
        self.set_cmap(cmap)  # 设置颜色映射对象的方法
        #: The last colorbar associated with this ScalarMappable. May be None.
        self.colorbar = None  # 与此ScalarMappable关联的最后一个colorbar，可能为None
        self.callbacks = cbook.CallbackRegistry(signals=["changed"])  # 回调函数注册表，监听"changed"信号

    def _scale_norm(self, norm, vmin, vmax):
        """
        Helper for initial scaling.

        Used by public functions that create a ScalarMappable and support
        parameters *vmin*, *vmax* and *norm*. This makes sure that a *norm*
        will take precedence over *vmin*, *vmax*.

        Note that this method does not set the norm.
        """
        # 初始标准化的辅助函数，处理参数的初始缩放
        if vmin is not None or vmax is not None:
            self.set_clim(vmin, vmax)  # 设置颜色映射对象的数据范围
            if isinstance(norm, colors.Normalize):
                raise ValueError(
                    "Passing a Normalize instance simultaneously with "
                    "vmin/vmax is not supported.  Please pass vmin/vmax "
                    "directly to the norm when creating it.")
        
        # 始终解析自动缩放，确保有具体的限制而不是推迟到绘制时处理
        self.autoscale_None()

    def set_array(self, A):
        """
        Set the value array from array-like *A*.

        Parameters
        ----------
        A : array-like or None
            The values that are mapped to colors.

            The base class `.ScalarMappable` does not make any assumptions on
            the dimensionality and shape of the value array *A*.
        """
        # 从类似数组A中设置值数组
        if A is None:
            self._A = None
            return
        
        A = cbook.safe_masked_invalid(A, copy=True)  # 处理可能的无效数据和掩码
        if not np.can_cast(A.dtype, float, "same_kind"):
            raise TypeError(f"Image data of dtype {A.dtype} cannot be "
                            "converted to float")  # 如果数据类型不能转换为float，抛出类型错误
        
        self._A = A
        if not self.norm.scaled():
            self.norm.autoscale_None(A)  # 如果标准化对象未缩放，则根据数据自动缩放
    def get_array(self):
        """
        Return the array of values, that are mapped to colors.

        The base class `.ScalarMappable` does not make any assumptions on
        the dimensionality and shape of the array.
        """
        # 返回与颜色映射相关联的数值数组
        return self._A

    def get_cmap(self):
        """Return the `.Colormap` instance."""
        # 返回当前的颜色映射对象 `.Colormap` 的实例
        return self.cmap

    def get_clim(self):
        """
        Return the values (min, max) that are mapped to the colormap limits.
        """
        # 返回映射到颜色映射限制的值 (min, max)
        return self.norm.vmin, self.norm.vmax

    def set_clim(self, vmin=None, vmax=None):
        """
        Set the norm limits for image scaling.

        Parameters
        ----------
        vmin, vmax : float
             The limits.

             The limits may also be passed as a tuple (*vmin*, *vmax*) as a
             single positional argument.

             .. ACCEPTS: (vmin: float, vmax: float)
        """
        # 如果更新了规范的限制，则通过附加到规范的回调调用 self.changed()
        if vmax is None:
            try:
                vmin, vmax = vmin
            except (TypeError, ValueError):
                pass
        if vmin is not None:
            # 设置规范的最小值，对传入的 vmin 进行边界值处理
            self.norm.vmin = colors._sanitize_extrema(vmin)
        if vmax is not None:
            # 设置规范的最大值，对传入的 vmax 进行边界值处理
            self.norm.vmax = colors._sanitize_extrema(vmax)

    def get_alpha(self):
        """
        Returns
        -------
        float
            Always returns 1.
        """
        # 返回固定值 1，这个方法可以被 Artist 子类重写
        return 1.

    def set_cmap(self, cmap):
        """
        Set the colormap for luminance data.

        Parameters
        ----------
        cmap : `.Colormap` or str or None
        """
        # 检查是否处于初始化阶段
        in_init = self.cmap is None

        # 确保传入的 cmap 是有效的颜色映射对象 `.Colormap`，并设置为当前对象的颜色映射
        self.cmap = _ensure_cmap(cmap)
        if not in_init:
            self.changed()  # 事情还没有正确设置。

    @property
    def norm(self):
        # 返回规范化对象 self._norm
        return self._norm

    @norm.setter
    def norm(self, norm):
        # 检查 norm 参数是否为 Normalize 类型、字符串类型、或者 None
        _api.check_isinstance((colors.Normalize, str, None), norm=norm)
        if norm is None:
            # 如果 norm 为 None，则创建一个默认的 Normalize 实例
            norm = colors.Normalize()
        elif isinstance(norm, str):
            try:
                # 尝试从映射中获取对应的缩放类
                scale_cls = scale._scale_mapping[norm]
            except KeyError:
                # 如果找不到对应的缩放类，则抛出异常
                raise ValueError(
                    "Invalid norm str name; the following values are "
                    f"supported: {', '.join(scale._scale_mapping)}"
                ) from None
            # 根据缩放类创建自动规范化实例
            norm = _auto_norm_from_scale(scale_cls)()

        if norm is self.norm:
            # 如果传入的 norm 与当前的 self.norm 相同，则不进行更新
            # 我们不需要更新任何内容
            return

        in_init = self.norm is None
        # 如果不是在初始化阶段，则断开当前的回调并连接到新的 norm
        if not in_init:
            self.norm.callbacks.disconnect(self._id_norm)
        # 更新 self._norm 为新的 norm
        self._norm = norm
        # 连接到新的 norm 的 'changed' 信号的回调
        self._id_norm = self.norm.callbacks.connect('changed',
                                                    self.changed)
        # 如果不是在初始化阶段，则调用 self.changed() 方法
        if not in_init:
            self.changed()

    def set_norm(self, norm):
        """
        设置规范化实例。

        Parameters
        ----------
        norm : `.Normalize` or str or None

        Notes
        -----
        如果有任何色条使用此规范化的映射对象，设置映射对象的规范化会将色条上的规范化、定位器和格式化器重置为默认值。
        """
        self.norm = norm

    def autoscale(self):
        """
        根据当前数组自动调整规范化实例上的标量限制。
        """
        if self._A is None:
            # 如果 self._A 为 None，则抛出类型错误
            raise TypeError('You must first set_array for mappable')
        # 如果规范化实例的限制更新，则通过连接到规范化实例的回调调用 self.changed()
        self.norm.autoscale(self._A)

    def autoscale_None(self):
        """
        根据当前数组自动调整规范化实例上的标量限制，仅更改为 None 的限制。
        """
        if self._A is None:
            # 如果 self._A 为 None，则抛出类型错误
            raise TypeError('You must first set_array for mappable')
        # 如果规范化实例的限制更新，则通过连接到规范化实例的回调调用 self.changed()
        self.norm.autoscale_None(self._A)

    def changed(self):
        """
        每当映射对象更改时调用此方法，通知所有回调监听器 'changed' 信号。
        """
        # 处理所有 'changed' 信号的回调
        self.callbacks.process('changed', self)
        # 设置 stale 标志为 True，表示需要更新
        self.stale = True
# 更新 matplotlib 文档字符串插值字典，以便应用于所有相关方法
mpl._docstring.interpd.update(
    cmap_doc="""\
cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
    用于将标量数据映射到颜色的 Colormap 实例或已注册的颜色映射名称。
""",
    norm_doc="""\
norm : str or `~matplotlib.colors.Normalize`, optional
    在使用 *cmap* 映射到颜色之前，用于将标量数据缩放到 [0, 1] 范围的归一化方法。默认情况下，使用线性缩放，
    将最小值映射为 0，最大值映射为 1。

    如果提供，可以是以下之一：

    - `.Normalize` 实例或其子类之一（参见 :ref:`colormapnorms`）。
    - 缩放名称，例如 "linear"、"log"、"symlog"、"logit" 等。要获取可用缩放的列表，请调用 `matplotlib.scale.get_scale_names()`。
      在这种情况下，会动态生成并实例化适当的 `.Normalize` 子类。
""",
    vmin_vmax_doc="""\
vmin, vmax : float, optional
    在使用标量数据且没有显式 *norm* 时，*vmin* 和 *vmax* 定义颜色映射的数据范围。
    默认情况下，颜色映射涵盖提供数据的完整值范围。当给定 *norm* 实例时使用 *vmin*/*vmax* 会引发错误
    （但使用 `str` 类型的 *norm* 名称和 *vmin*/*vmax* 一起使用是可以接受的）。
""",
)


def _ensure_cmap(cmap):
    """
    确保我们有一个 `.Colormap` 对象。

    用于内部使用以保持错误的类型稳定性。

    Parameters
    ----------
    cmap : None, str, Colormap
        - 如果是 `Colormap`，则返回它
        - 如果是字符串，则在 mpl.colormaps 中查找
        - 如果为 None，则在 mpl.colormaps 中查找默认颜色映射

    Returns
    -------
    Colormap
        返回一个 `.Colormap` 对象
    """
    if isinstance(cmap, colors.Colormap):
        return cmap
    cmap_name = cmap if cmap is not None else mpl.rcParams["image.cmap"]
    # 使用 check_in_list 来确保内部使用时引发的异常的类型稳定性（ValueError vs KeyError）
    if cmap_name not in _colormaps:
        _api.check_in_list(sorted(_colormaps), cmap=cmap_name)
    return mpl.colormaps[cmap_name]
```