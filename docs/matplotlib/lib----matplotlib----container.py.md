# `D:\src\scipysrc\matplotlib\lib\matplotlib\container.py`

```py
# 从 matplotlib 库中导入 cbook 模块
from matplotlib import cbook
# 从 matplotlib.artist 模块中导入 Artist 类
from matplotlib.artist import Artist

# 定义一个继承自 tuple 的 Container 类
class Container(tuple):
    """
    Base class for containers.

    Containers are classes that collect semantically related Artists such as
    the bars of a bar plot.
    """

    # 重写 __repr__ 方法，返回类的描述字符串
    def __repr__(self):
        return f"<{type(self).__name__} object of {len(self)} artists>"

    # 重写 __new__ 方法，创建一个新的 Container 实例
    def __new__(cls, *args, **kwargs):
        return tuple.__new__(cls, args[0])

    # 初始化方法，接受 kl 和 label 作为参数
    def __init__(self, kl, label=None):
        # 初始化回调函数注册表
        self._callbacks = cbook.CallbackRegistry(signals=["pchanged"])
        # 初始化移除方法为 None
        self._remove_method = None
        # 如果 label 不为 None，则将 label 转换为字符串，否则为 None
        self._label = str(label) if label is not None else None

    # 移除方法，用于移除容器中的艺术家
    def remove(self):
        # 遍历容器中的艺术家，逐个移除
        for c in cbook.flatten(
                self, scalarp=lambda x: isinstance(x, Artist)):
            if c is not None:
                c.remove()
        # 如果存在移除方法，则调用它，并传入 self 作为参数
        if self._remove_method:
            self._remove_method(self)

    # 获取容器中的子元素（艺术家）
    def get_children(self):
        # 使用列表推导式获取容器中的非空子元素（艺术家）
        return [child for child in cbook.flatten(self) if child is not None]

    # 以下几个方法使用 Artist 类的对应方法，用于设置和获取标签、添加和移除回调函数、发送 pchanged 信号
    get_label = Artist.get_label
    set_label = Artist.set_label
    add_callback = Artist.add_callback
    remove_callback = Artist.remove_callback
    pchanged = Artist.pchanged

# 继承自 Container 类的 BarContainer 类
class BarContainer(Container):
    """
    Container for the artists of bar plots (e.g. created by `.Axes.bar`).

    The container can be treated as a tuple of the *patches* themselves.
    Additionally, you can access these and further parameters by the
    attributes.

    Attributes
    ----------
    patches : list of :class:`~matplotlib.patches.Rectangle`
        The artists of the bars.

    errorbar : None or :class:`~matplotlib.container.ErrorbarContainer`
        A container for the error bar artists if error bars are present.
        *None* otherwise.

    datavalues : None or array-like
        The underlying data values corresponding to the bars.

    orientation : {'vertical', 'horizontal'}, default: None
        If 'vertical', the bars are assumed to be vertical.
        If 'horizontal', the bars are assumed to be horizontal.

    """

    # 初始化方法，接受 patches、errorbar、datavalues、orientation 等参数
    def __init__(self, patches, errorbar=None, *, datavalues=None,
                 orientation=None, **kwargs):
        # 初始化 patches 属性为传入的 patches
        self.patches = patches
        # 初始化 errorbar 属性为传入的 errorbar
        self.errorbar = errorbar
        # 初始化 datavalues 属性为传入的 datavalues
        self.datavalues = datavalues
        # 初始化 orientation 属性为传入的 orientation
        self.orientation = orientation
        # 调用父类 Container 的初始化方法，传入 patches 和 kwargs
        super().__init__(patches, **kwargs)

# 继承自 Container 类的 ErrorbarContainer 类
class ErrorbarContainer(Container):
    """
    Container for the artists of error bars (e.g. created by `.Axes.errorbar`).

    The container can be treated as the *lines* tuple itself.
    Additionally, you can access these and further parameters by the
    attributes.

    Attributes
    ----------
    """
    # 定义一个类 ErrorbarContainer，用于管理和显示错误条线段和标记
    lines : tuple
        # 包含了数据线、上下限线和误差条的元组

        - data_line : `~matplotlib.lines.Line2D` 的实例，表示数据线的 x 和 y 的标记和/或线条
        - caplines : `~matplotlib.lines.Line2D` 实例的元组，表示误差条顶部和底部的线段
        - barlinecols : `~matplotlib.collections.LineCollection` 实例的元组，包含水平和垂直误差范围的线段集合

    has_xerr, has_yerr : bool
        # 如果误差条包含 x/y 方向的误差，则为 True

        ``True`` 表示误差条包含 x 方向或 y 方向的误差。

    """

    def __init__(self, lines, has_xerr=False, has_yerr=False, **kwargs):
        # 初始化 ErrorbarContainer 对象
        self.lines = lines
        self.has_xerr = has_xerr
        self.has_yerr = has_yerr
        # 调用父类的构造函数，初始化父类对象并传递 lines 和 kwargs 参数
        super().__init__(lines, **kwargs)
class StemContainer(Container):
    """
    Container for the artists created in a :meth:`.Axes.stem` plot.

    The container can be treated like a namedtuple ``(markerline, stemlines,
    baseline)``.

    Attributes
    ----------
    markerline : `~matplotlib.lines.Line2D`
        The artist of the markers at the stem heads.

    stemlines : `~matplotlib.collections.LineCollection`
        The artists of the vertical lines for all stems.

    baseline : `~matplotlib.lines.Line2D`
        The artist of the horizontal baseline.
    """

    def __init__(self, markerline_stemlines_baseline, **kwargs):
        """
        Parameters
        ----------
        markerline_stemlines_baseline : tuple
            Tuple of ``(markerline, stemlines, baseline)``.
            ``markerline`` contains the `.Line2D` of the markers,
            ``stemlines`` is a `.LineCollection` of the main lines,
            ``baseline`` is the `.Line2D` of the baseline.
        """
        # 从传入的元组中解包获取 markerline、stemlines 和 baseline
        markerline, stemlines, baseline = markerline_stemlines_baseline
        # 将解包得到的各个艺术家对象赋值给对象的属性
        self.markerline = markerline
        self.stemlines = stemlines
        self.baseline = baseline
        # 调用父类的初始化方法
        super().__init__(markerline_stemlines_baseline, **kwargs)
```