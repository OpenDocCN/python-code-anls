# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axes_grid1\axes_size.py`

```py
"""
Provides classes of simple units that will be used with `.AxesDivider`
class (or others) to determine the size of each Axes. The unit
classes define `get_size` method that returns a tuple of two floats,
meaning relative and absolute sizes, respectively.

Note that this class is nothing more than a simple tuple of two
floats. Take a look at the Divider class to see how these two
values are used.
"""

from numbers import Real  # 导入Real类，用于检查类型

from matplotlib import _api  # 导入Matplotlib内部的_api模块
from matplotlib.axes import Axes  # 导入Axes类，用于处理坐标轴


class _Base:
    def __rmul__(self, other):
        # 实现右乘运算符，返回other与self的分数
        return Fraction(other, self)

    def __add__(self, other):
        if isinstance(other, _Base):
            return Add(self, other)
        else:
            return Add(self, Fixed(other))

    def get_size(self, renderer):
        """
        Return two-float tuple with relative and absolute sizes.
        """
        # 获取尺寸，子类必须实现该方法
        raise NotImplementedError("Subclasses must implement")


class Add(_Base):
    """
    Sum of two sizes.
    """

    def __init__(self, a, b):
        self._a = a  # 存储第一个尺寸
        self._b = b  # 存储第二个尺寸

    def get_size(self, renderer):
        # 获取两个尺寸的相对和绝对大小
        a_rel_size, a_abs_size = self._a.get_size(renderer)
        b_rel_size, b_abs_size = self._b.get_size(renderer)
        return a_rel_size + b_rel_size, a_abs_size + b_abs_size


class Fixed(_Base):
    """
    Simple fixed size with absolute part = *fixed_size* and relative part = 0.
    """

    def __init__(self, fixed_size):
        _api.check_isinstance(Real, fixed_size=fixed_size)  # 检查fixed_size是否为Real类型
        self.fixed_size = fixed_size  # 存储固定的绝对尺寸

    def get_size(self, renderer):
        rel_size = 0.  # 相对尺寸为0
        abs_size = self.fixed_size  # 绝对尺寸为固定的尺寸
        return rel_size, abs_size


class Scaled(_Base):
    """
    Simple scaled(?) size with absolute part = 0 and
    relative part = *scalable_size*.
    """

    def __init__(self, scalable_size):
        self._scalable_size = scalable_size  # 存储可缩放的相对尺寸

    def get_size(self, renderer):
        rel_size = self._scalable_size  # 相对尺寸为可缩放的尺寸
        abs_size = 0.  # 绝对尺寸为0
        return rel_size, abs_size

Scalable = Scaled  # 别名


def _get_axes_aspect(ax):
    aspect = ax.get_aspect()  # 获取坐标轴的纵横比
    if aspect == "auto":
        aspect = 1.  # 如果纵横比为自动，则默认为1
    return aspect


class AxesX(_Base):
    """
    Scaled size whose relative part corresponds to the data width
    of the *axes* multiplied by the *aspect*.
    """

    def __init__(self, axes, aspect=1., ref_ax=None):
        self._axes = axes  # 存储坐标轴对象
        self._aspect = aspect  # 存储纵横比
        if aspect == "axes" and ref_ax is None:
            raise ValueError("ref_ax must be set when aspect='axes'")
        self._ref_ax = ref_ax  # 存储参考坐标轴对象

    def get_size(self, renderer):
        l1, l2 = self._axes.get_xlim()  # 获取坐标轴的x轴范围
        if self._aspect == "axes":
            ref_aspect = _get_axes_aspect(self._ref_ax)  # 获取参考坐标轴的纵横比
            aspect = ref_aspect / _get_axes_aspect(self._axes)  # 计算相对纵横比
        else:
            aspect = self._aspect  # 使用给定的纵横比

        rel_size = abs(l2-l1)*aspect  # 相对尺寸为数据宽度乘以纵横比
        abs_size = 0.  # 绝对尺寸为0
        return rel_size, abs_size


class AxesY(_Base):
    """
    """
    Scaled size whose relative part corresponds to the data height
    of the *axes* multiplied by the *aspect*.
    """

    # 定义一个类，用于计算尺寸，基于给定的坐标轴和纵横比
    def __init__(self, axes, aspect=1., ref_ax=None):
        # 初始化函数，设置类的初始属性
        self._axes = axes          # 将传入的 axes 参数赋给类的私有属性 _axes
        self._aspect = aspect      # 将传入的 aspect 参数赋给类的私有属性 _aspect

        # 如果 aspect 参数为 "axes" 且 ref_ax 为 None，则引发数值错误
        if aspect == "axes" and ref_ax is None:
            raise ValueError("ref_ax must be set when aspect='axes'")
        self._ref_ax = ref_ax      # 将传入的 ref_ax 参数赋给类的私有属性 _ref_ax

    def get_size(self, renderer):
        # 获取坐标轴的纵坐标范围
        l1, l2 = self._axes.get_ylim()

        # 根据不同的 aspect 类型计算纵向的相对尺寸
        if self._aspect == "axes":
            ref_aspect = _get_axes_aspect(self._ref_ax)   # 获取参考坐标轴的纵横比
            aspect = _get_axes_aspect(self._axes)         # 获取当前坐标轴的纵横比
        else:
            aspect = self._aspect   # 使用传入的 aspect 参数作为纵横比

        rel_size = abs(l2 - l1) * aspect   # 计算相对尺寸，即纵坐标范围的长度乘以纵横比
        abs_size = 0.                      # 绝对尺寸设为零
        return rel_size, abs_size
class MaxExtent(_Base):
    """
    Size whose absolute part is either the largest width or the largest height
    of the given *artist_list*.
    """

    def __init__(self, artist_list, w_or_h):
        self._artist_list = artist_list  # 初始化存储艺术家列表的属性
        _api.check_in_list(["width", "height"], w_or_h=w_or_h)  # 检查 w_or_h 参数是否在 ["width", "height"] 中
        self._w_or_h = w_or_h  # 存储 w_or_h 参数作为属性

    def add_artist(self, a):
        self._artist_list.append(a)  # 将艺术家对象 a 添加到艺术家列表中

    def get_size(self, renderer):
        rel_size = 0.  # 相对大小初始化为 0
        extent_list = [
            getattr(a.get_window_extent(renderer), self._w_or_h) / a.figure.dpi
            for a in self._artist_list]  # 获取每个艺术家对象的窗口范围在指定维度上的大小，并计算为绝对大小（单位是英寸）
        abs_size = max(extent_list, default=0)  # 计算最大的绝对大小，如果列表为空则默认为 0
        return rel_size, abs_size  # 返回相对大小和最大的绝对大小


class MaxWidth(MaxExtent):
    """
    Size whose absolute part is the largest width of the given *artist_list*.
    """

    def __init__(self, artist_list):
        super().__init__(artist_list, "width")  # 调用父类 MaxExtent 的初始化方法，指定宽度为属性


class MaxHeight(MaxExtent):
    """
    Size whose absolute part is the largest height of the given *artist_list*.
    """

    def __init__(self, artist_list):
        super().__init__(artist_list, "height")  # 调用父类 MaxExtent 的初始化方法，指定高度为属性


class Fraction(_Base):
    """
    An instance whose size is a *fraction* of the *ref_size*.

    >>> s = Fraction(0.3, AxesX(ax))
    """

    def __init__(self, fraction, ref_size):
        _api.check_isinstance(Real, fraction=fraction)  # 检查 fraction 是否为实数类型
        self._fraction_ref = ref_size  # 存储参考大小的属性
        self._fraction = fraction  # 存储分数大小的属性

    def get_size(self, renderer):
        if self._fraction_ref is None:
            return self._fraction, 0.  # 如果参考大小为空，则返回分数大小和绝对大小为 0
        else:
            r, a = self._fraction_ref.get_size(renderer)  # 调用参考大小对象的 get_size 方法获取相对大小 r 和绝对大小 a
            rel_size = r * self._fraction  # 计算相对大小
            abs_size = a * self._fraction  # 计算绝对大小
            return rel_size, abs_size  # 返回相对大小和绝对大小


def from_any(size, fraction_ref=None):
    """
    Create a Fixed unit when the first argument is a float, or a
    Fraction unit if that is a string that ends with %. The second
    argument is only meaningful when Fraction unit is created.

    >>> from mpl_toolkits.axes_grid1.axes_size import from_any
    >>> a = from_any(1.2) # => Fixed(1.2)
    >>> from_any("50%", a) # => Fraction(0.5, a)
    """
    if isinstance(size, Real):
        return Fixed(size)  # 如果 size 是实数，则创建 Fixed 单位对象
    elif isinstance(size, str):
        if size[-1] == "%":
            return Fraction(float(size[:-1]) / 100, fraction_ref)  # 如果 size 是以 "%" 结尾的字符串，则创建 Fraction 单位对象
    raise ValueError("Unknown format")  # 如果 size 不是有效的类型，则抛出 ValueError


class _AxesDecorationsSize(_Base):
    """
    Fixed size, corresponding to the size of decorations on a given Axes side.
    """

    _get_size_map = {
        "left":   lambda tight_bb, axes_bb: axes_bb.xmin - tight_bb.xmin,
        "right":  lambda tight_bb, axes_bb: tight_bb.xmax - axes_bb.xmax,
        "bottom": lambda tight_bb, axes_bb: axes_bb.ymin - tight_bb.ymin,
        "top":    lambda tight_bb, axes_bb: tight_bb.ymax - axes_bb.ymax,
    }
    # 初始化方法，接受一个Axes对象或者Axes对象列表ax和一个方向direction作为参数
    def __init__(self, ax, direction):
        # 调用_api模块中的check_in_list函数，验证self._get_size_map中包含direction指定的键
        _api.check_in_list(self._get_size_map, direction=direction)
        # 将direction参数赋值给实例变量self._direction
        self._direction = direction
        # 如果ax是Axes对象，则将其放入列表self._ax_list中，否则直接使用ax作为列表self._ax_list的值
        self._ax_list = [ax] if isinstance(ax, Axes) else ax

    # 获取尺寸信息的方法，接受一个renderer作为参数
    def get_size(self, renderer):
        # 计算所有Axes对象中，调用self._get_size_map[self._direction]方法后返回的最大尺寸sz
        sz = max([
            self._get_size_map[self._direction](
                ax.get_tightbbox(renderer, call_axes_locator=False), ax.bbox)
            for ax in self._ax_list])
        # 将renderer中的点转换为像素，计算每英寸的像素数dpi
        dpi = renderer.points_to_pixels(72)
        # 将计算出的尺寸sz转换为绝对尺寸abs_size，单位是英寸
        abs_size = sz / dpi
        # 相对尺寸rel_size初始化为0
        rel_size = 0
        # 返回相对尺寸和绝对尺寸
        return rel_size, abs_size
```