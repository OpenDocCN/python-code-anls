# `D:\src\scipysrc\pandas\pandas\_libs\interval.pyx`

```
import numbers
from operator import (
    le,  # 导入 operator 模块中的小于等于运算符
    lt,  # 导入 operator 模块中的小于运算符
)

from cpython.datetime cimport (
    PyDelta_Check,  # 导入 cpython.datetime 模块中的 PyDelta_Check 函数
    import_datetime,  # 导入 cpython.datetime 模块中的 import_datetime 函数
)

import_datetime()  # 调用 import_datetime 函数

cimport cython  # 导入 cython 声明

from cpython.object cimport PyObject_RichCompare  # 从 cpython.object 模块中导入 PyObject_RichCompare 函数
from cython cimport Py_ssize_t  # 从 cython 模块中导入 Py_ssize_t 类型

import numpy as np  # 导入 numpy 库，并使用 np 作为别名

cimport numpy as cnp  # 导入 cnp 声明
from numpy cimport (
    NPY_QUICKSORT,  # 从 numpy 模块中导入 NPY_QUICKSORT 常量
    PyArray_ArgSort,  # 从 numpy 模块中导入 PyArray_ArgSort 函数
    PyArray_Take,  # 从 numpy 模块中导入 PyArray_Take 函数
    float64_t,  # 从 numpy 模块中导入 float64_t 类型
    int64_t,  # 从 numpy 模块中导入 int64_t 类型
    ndarray,  # 从 numpy 模块中导入 ndarray 类型
    uint64_t,  # 从 numpy 模块中导入 uint64_t 类型
)

cnp.import_array()  # 调用 cnp.import_array 函数，导入 numpy C API

from pandas._libs cimport util  # 从 pandas._libs 模块中导入 util 声明
from pandas._libs.hashtable cimport Int64Vector  # 从 pandas._libs.hashtable 模块中导入 Int64Vector 声明
from pandas._libs.tslibs.timedeltas cimport _Timedelta  # 从 pandas._libs.tslibs.timedeltas 模块中导入 _Timedelta 声明
from pandas._libs.tslibs.timestamps cimport _Timestamp  # 从 pandas._libs.tslibs.timestamps 模块中导入 _Timestamp 声明
from pandas._libs.tslibs.timezones cimport tz_compare  # 从 pandas._libs.tslibs.timezones 模块中导入 tz_compare 函数
from pandas._libs.tslibs.util cimport (
    is_float_object,  # 从 pandas._libs.tslibs.util 模块中导入 is_float_object 函数
    is_integer_object,  # 从 pandas._libs.tslibs.util 模块中导入 is_integer_object 函数
)

VALID_CLOSED = frozenset(["left", "right", "both", "neither"])  # 定义一个不可变集合 VALID_CLOSED，包含指定字符串

cdef class IntervalMixin:  # 定义 Cython 扩展类 IntervalMixin

    @property
    def closed_left(self):  # 定义属性方法 closed_left
        """
        Check if the interval is closed on the left side.

        For the meaning of `closed` and `open` see :class:`~pandas.Interval`.

        Returns
        -------
        bool
            True if the Interval is closed on the left-side.

        See Also
        --------
        Interval.closed_right : Check if the interval is closed on the right side.
        Interval.open_left : Boolean inverse of closed_left.

        Examples
        --------
        >>> iv = pd.Interval(0, 5, closed='left')
        >>> iv.closed_left
        True

        >>> iv = pd.Interval(0, 5, closed='right')
        >>> iv.closed_left
        False
        """
        return self.closed in ("left", "both")  # 返回判断 Interval 是否在左侧闭合的布尔值

    @property
    def closed_right(self):  # 定义属性方法 closed_right
        """
        Check if the interval is closed on the right side.

        For the meaning of `closed` and `open` see :class:`~pandas.Interval`.

        Returns
        -------
        bool
            True if the Interval is closed on the left-side.

        See Also
        --------
        Interval.closed_left : Check if the interval is closed on the left side.
        Interval.open_right : Boolean inverse of closed_right.

        Examples
        --------
        >>> iv = pd.Interval(0, 5, closed='both')
        >>> iv.closed_right
        True

        >>> iv = pd.Interval(0, 5, closed='left')
        >>> iv.closed_right
        False
        """
        return self.closed in ("right", "both")  # 返回判断 Interval 是否在右侧闭合的布尔值

    @property
    def open_left(self):
        """
        Check if the interval is open on the left side.

        For the meaning of `closed` and `open` see :class:`~pandas.Interval`.

        Returns
        -------
        bool
            True if the Interval is not closed on the left-side.

        See Also
        --------
        Interval.open_right : Check if the interval is open on the right side.
        Interval.closed_left : Boolean inverse of open_left.

        Examples
        --------
        >>> iv = pd.Interval(0, 5, closed='neither')
        >>> iv.open_left
        True

        >>> iv = pd.Interval(0, 5, closed='both')
        >>> iv.open_left
        False
        """
        # 返回左侧是否非闭合的布尔值
        return not self.closed_left

    @property
    def open_right(self):
        """
        Check if the interval is open on the right side.

        For the meaning of `closed` and `open` see :class:`~pandas.Interval`.

        Returns
        -------
        bool
            True if the Interval is not closed on the right-side.

        See Also
        --------
        Interval.open_left : Check if the interval is open on the left side.
        Interval.closed_right : Boolean inverse of open_right.

        Examples
        --------
        >>> iv = pd.Interval(0, 5, closed='left')
        >>> iv.open_right
        True

        >>> iv = pd.Interval(0, 5)
        >>> iv.open_right
        False
        """
        # 返回右侧是否非闭合的布尔值
        return not self.closed_right

    @property
    def mid(self):
        """
        Return the midpoint of the Interval.

        See Also
        --------
        Interval.left : Return the left bound for the interval.
        Interval.right : Return the right bound for the interval.
        Interval.length : Return the length of the interval.

        Examples
        --------
        >>> iv = pd.Interval(0, 5)
        >>> iv.mid
        2.5
        """
        try:
            # 尝试计算并返回区间的中点
            return 0.5 * (self.left + self.right)
        except TypeError:
            # 处理日期时间安全的版本，返回区间的中点
            return self.left + 0.5 * self.length

    @property
    def length(self):
        """
        Return the length of the Interval.

        See Also
        --------
        Interval.is_empty : Indicates if an interval contains no points.

        Examples
        --------
        >>> interval = pd.Interval(left=1, right=2, closed='left')
        >>> interval
        Interval(1, 2, closed='left')
        >>> interval.length
        1
        """
        # 返回区间的长度
        return self.right - self.left
    def is_empty(self):
        """
        Indicates if an interval is empty, meaning it contains no points.

        Returns
        -------
        bool or ndarray
            A boolean indicating if a scalar :class:`Interval` is empty, or a
            boolean ``ndarray`` positionally indicating if an ``Interval`` in
            an :class:`~arrays.IntervalArray` or :class:`IntervalIndex` is
            empty.

        See Also
        --------
        Interval.length : Return the length of the Interval.

        Examples
        --------
        An :class:`Interval` that contains points is not empty:

        >>> pd.Interval(0, 1, closed='right').is_empty
        False

        An ``Interval`` that does not contain any points is empty:

        >>> pd.Interval(0, 0, closed='right').is_empty
        True
        >>> pd.Interval(0, 0, closed='left').is_empty
        True
        >>> pd.Interval(0, 0, closed='neither').is_empty
        True

        An ``Interval`` that contains a single point is not empty:

        >>> pd.Interval(0, 0, closed='both').is_empty
        False

        An :class:`~arrays.IntervalArray` or :class:`IntervalIndex` returns a
        boolean ``ndarray`` positionally indicating if an ``Interval`` is
        empty:

        >>> ivs = [pd.Interval(0, 0, closed='neither'),
        ...        pd.Interval(1, 2, closed='neither')]
        >>> pd.arrays.IntervalArray(ivs).is_empty
        array([ True, False])

        Missing values are not considered empty:

        >>> ivs = [pd.Interval(0, 0, closed='neither'), np.nan]
        >>> pd.IntervalIndex(ivs).is_empty
        array([ True, False])
        """
        # 检查间隔是否为空，即左右端点相等但不包含两端点的情况
        return (self.right == self.left) & (self.closed != "both")

    def _check_closed_matches(self, other, name="other"):
        """
        Check if the closed attribute of `other` matches.

        Note that 'left' and 'right' are considered different from 'both'.

        Parameters
        ----------
        other : Interval, IntervalIndex, IntervalArray
        name : str
            Name to use for 'other' in the error message.

        Raises
        ------
        ValueError
            When `other` is not closed exactly the same as self.
        """
        # 检查另一个间隔对象 `other` 的闭合属性是否与当前对象相同
        if self.closed != other.closed:
            raise ValueError(f"'{name}.closed' is {repr(other.closed)}, "
                             f"expected {repr(self.closed)}.")
# 定义一个 Cython 函数，用于检查对象是否类似于 Interval 对象
cdef bint _interval_like(other):
    # 检查对象是否具有 "left"、"right" 和 "closed" 属性
    return (hasattr(other, "left")
            and hasattr(other, "right")
            and hasattr(other, "closed"))


# 定义一个 Cython 类 Interval，实现不可变的 Interval 对象，类似于有界切片的区间
cdef class Interval(IntervalMixin):
    """
    Immutable object implementing an Interval, a bounded slice-like interval.

    Attributes
    ----------
    left : orderable scalar
        Left bound for the interval.
    right : orderable scalar
        Right bound for the interval.
    closed : {'right', 'left', 'both', 'neither'}, default 'right'
        Whether the interval is closed on the left-side, right-side, both or
        neither. See the Notes for more detailed explanation.

    See Also
    --------
    IntervalIndex : An Index of Interval objects that are all closed on the
        same side.
    cut : Convert continuous data into discrete bins (Categorical
        of Interval objects).
    qcut : Convert continuous data into bins (Categorical of Interval objects)
        based on quantiles.
    Period : Represents a period of time.

    Notes
    -----
    The parameters `left` and `right` must be from the same type, you must be
    able to compare them and they must satisfy ``left <= right``.

    A closed interval (in mathematics denoted by square brackets) contains
    its endpoints, i.e. the closed interval ``[0, 5]`` is characterized by the
    conditions ``0 <= x <= 5``. This is what ``closed='both'`` stands for.
    An open interval (in mathematics denoted by parentheses) does not contain
    its endpoints, i.e. the open interval ``(0, 5)`` is characterized by the
    conditions ``0 < x < 5``. This is what ``closed='neither'`` stands for.
    Intervals can also be half-open or half-closed, i.e. ``[0, 5)`` is
    described by ``0 <= x < 5`` (``closed='left'``) and ``(0, 5]`` is
    described by ``0 < x <= 5`` (``closed='right'``).

    Examples
    --------
    It is possible to build Intervals of different types, like numeric ones:

    >>> iv = pd.Interval(left=0, right=5)
    >>> iv
    Interval(0, 5, closed='right')

    You can check if an element belongs to it, or if it contains another interval:

    >>> 2.5 in iv
    True
    >>> pd.Interval(left=2, right=5, closed='both') in iv
    True

    You can test the bounds (``closed='right'``, so ``0 < x <= 5``):

    >>> 0 in iv
    False
    >>> 5 in iv
    True
    >>> 0.0001 in iv
    True

    Calculate its length

    >>> iv.length
    5

    You can operate with `+` and `*` over an Interval and the operation
    is applied to each of its bounds, so the result depends on the type
    of the bound elements

    >>> shifted_iv = iv + 3
    >>> shifted_iv
    Interval(3, 8, closed='right')
    >>> extended_iv = iv * 10.0
    >>> extended_iv
    Interval(0.0, 50.0, closed='right')

    To create a time interval you can use Timestamps as the bounds

    >>> year_2017 = pd.Interval(pd.Timestamp('2017-01-01 00:00:00'),
    ...                         pd.Timestamp('2018-01-01 00:00:00'),
    """
    pass  # 类定义的结尾，没有其他操作，用于描述一个有界的切片区间的不可变对象
    _typ = "interval"
    __array_priority__ = 1000


# 定义类属性 `_typ` 和 `__array_priority__`

cdef readonly object left


# 定义只读属性 `left`

Left bound for the interval.

See Also
--------
Interval.right : Return the right bound for the interval.
numpy.ndarray.left : A similar method in numpy for obtaining
    the left endpoint(s) of intervals.

Examples
--------
>>> interval = pd.Interval(left=1, right=2, closed='left')
>>> interval
Interval(1, 2, closed='left')
>>> interval.left
1


# `left` 是区间的左边界

cdef readonly object right


# 定义只读属性 `right`

Right bound for the interval.

See Also
--------
Interval.left : Return the left bound for the interval.
numpy.ndarray.right : A similar method in numpy for obtaining
    the right endpoint(s) of intervals.

Examples
--------
>>> interval = pd.Interval(left=1, right=2, closed='left')
>>> interval
Interval(1, 2, closed='left')
>>> interval.right
2


# `right` 是区间的右边界

cdef readonly str closed


# 定义只读属性 `closed`

String describing the inclusive side the intervals.

Either ``left``, ``right``, ``both`` or ``neither``.

See Also
--------
Interval.closed_left : Check if the interval is closed on the left side.
Interval.closed_right : Check if the interval is closed on the right side.
Interval.open_left : Check if the interval is open on the left side.
Interval.open_right : Check if the interval is open on the right side.

Examples
--------
>>> interval = pd.Interval(left=1, right=2, closed='left')
>>> interval
Interval(1, 2, closed='left')
>>> interval.closed
'left'


# `closed` 描述区间的闭合方式，可以是“left”、“right”、“both”或“neither”

def __init__(self, left, right, str closed="right"):
    # note: it is faster to just do these checks than to use a special
    # constructor (__cinit__/__new__) to avoid them

    self._validate_endpoint(left)
    self._validate_endpoint(right)

    if closed not in VALID_CLOSED:
        raise ValueError(f"invalid option for 'closed': {closed}")
    if not left <= right:
        raise ValueError("left side of interval must be <= right side")
    if (isinstance(left, _Timestamp) and
            not tz_compare(left.tzinfo, right.tzinfo)):
        # GH 18538
        raise ValueError("left and right must have the same time zone, got "
                         f"{repr(left.tzinfo)}' and {repr(right.tzinfo)}")
    self.left = left
    self.right = right
    self.closed = closed


# 定义初始化方法 `__init__`

初始化方法用于设置区间的左边界、右边界和闭合方式，并进行相应的验证。

- `_validate_endpoint(left)` 和 `_validate_endpoint(right)` 确保左右边界的有效性。
- `closed` 必须是预定义的闭合方式之一。
- 确保左边界小于等于右边界。
- 如果左边界是 `_Timestamp` 类型，并且左右边界的时区不同，则抛出异常。

self.left = left
self.right = right
self.closed = closed
    # 校验给定的端点是否合法
    # GH 23013
    def _validate_endpoint(self, endpoint):
        # 确保端点是整数、浮点数、_Timestamp 或 _Timedelta 类型之一
        if not (is_integer_object(endpoint) or is_float_object(endpoint) or
                isinstance(endpoint, (_Timestamp, _Timedelta))):
            raise ValueError("Only numeric, Timestamp and Timedelta endpoints "
                             "are allowed when constructing an Interval.")

    # 计算当前 Interval 对象的哈希值
    def __hash__(self):
        return hash((self.left, self.right, self.closed))

    # 检查指定的键是否在当前 Interval 对象中
    def __contains__(self, key) -> bool:
        if _interval_like(key):
            key_closed_left = key.closed in ("left", "both")
            key_closed_right = key.closed in ("right", "both")
            # 根据 Interval-like 对象的类型和开闭状态检查左右端点是否被包含
            if self.open_left and key_closed_left:
                left_contained = self.left < key.left
            else:
                left_contained = self.left <= key.left
            if self.open_right and key_closed_right:
                right_contained = key.right < self.right
            else:
                right_contained = key.right <= self.right
            return left_contained and right_contained
        # 对于普通的数值类型，根据开闭状态检查是否在当前 Interval 范围内
        return ((self.left < key if self.open_left else self.left <= key) and
                (key < self.right if self.open_right else key <= self.right))

    # 比较当前 Interval 对象与另一个对象，支持丰富的比较运算
    def __richcmp__(self, other, op: int):
        if isinstance(other, Interval):
            self_tuple = (self.left, self.right, self.closed)
            other_tuple = (other.left, other.right, other.closed)
            # 使用 PyObject_RichCompare 对比两个 Interval 对象的元组表示
            return PyObject_RichCompare(self_tuple, other_tuple, op)
        elif util.is_array(other):
            # 对于数组类型，逐个元素比较，并返回布尔数组结果
            return np.array(
                [PyObject_RichCompare(self, x, op) for x in other],
                dtype=bool,
            )

        return NotImplemented

    # 序列化当前 Interval 对象，返回用于重建对象的元组
    def __reduce__(self):
        args = (self.left, self.right, self.closed)
        return (type(self), args)

    # 返回当前 Interval 对象的字符串表示形式
    def __repr__(self) -> str:
        disp = str if isinstance(self.left, (np.generic, _Timestamp)) else repr
        name = type(self).__name__
        # 构造 Interval 对象的详细字符串表示，包括左右端点和闭合信息
        repr_str = f"{name}({disp(self.left)}, {disp(self.right)}, closed={repr(self.closed)})"  # noqa: E501
        return repr_str

    # 返回当前 Interval 对象的易读字符串表示形式
    def __str__(self) -> str:
        start_symbol = "[" if self.closed_left else "("
        end_symbol = "]" if self.closed_right else ")"
        # 构造易读的 Interval 字符串表示，包括左右端点和开闭符号
        return f"{start_symbol}{self.left}, {self.right}{end_symbol}"

    # 实现当前 Interval 对象与另一个对象的加法操作
    def __add__(self, y):
        if (
            isinstance(y, numbers.Number)
            or PyDelta_Check(y)
            or cnp.is_timedelta64_object(y)
        ):
            # 返回一个新的 Interval 对象，表示当前对象所有端点加上指定的数值
            return Interval(self.left + y, self.right + y, closed=self.closed)
        return NotImplemented

    # 实现其他对象与当前 Interval 对象的右加法操作
    def __radd__(self, other):
        if (
                isinstance(other, numbers.Number)
                or PyDelta_Check(other)
                or cnp.is_timedelta64_object(other)
        ):
            # 返回一个新的 Interval 对象，表示指定数值加上当前对象所有端点
            return Interval(self.left + other, self.right + other, closed=self.closed)
        return NotImplemented
    # 定义一个双减法运算符方法，用于处理 Interval 对象与数字或另一个对象的减法操作
    def __sub__(self, y):
        # 如果 y 是数字类型，或者是 PyDelta 对象，或者是 pandas 中的 timedelta64 对象
        if (
            isinstance(y, numbers.Number)
            or PyDelta_Check(y)
            or cnp.is_timedelta64_object(y)
        ):
            # 返回一个新的 Interval 对象，其左右端点减去 y，保持闭合性与原对象一致
            return Interval(self.left - y, self.right - y, closed=self.closed)
        return NotImplemented

    # 定义一个乘法运算符方法，用于处理 Interval 对象与数字的乘法操作
    def __mul__(self, y):
        # 如果 y 是数字类型
        if isinstance(y, numbers.Number):
            # 返回一个新的 Interval 对象，其左右端点分别乘以 y，保持闭合性与原对象一致
            return Interval(self.left * y, self.right * y, closed=self.closed)
        return NotImplemented

    # 定义一个右乘法运算符方法，用于处理数字与 Interval 对象的乘法操作
    def __rmul__(self, other):
        # 如果 other 是数字类型
        if isinstance(other, numbers.Number):
            # 返回一个新的 Interval 对象，其左右端点分别乘以 other，保持闭合性与原对象一致
            return Interval(self.left * other, self.right * other, closed=self.closed)
        return NotImplemented

    # 定义一个除法运算符方法，用于处理 Interval 对象与数字的除法操作
    def __truediv__(self, y):
        # 如果 y 是数字类型
        if isinstance(y, numbers.Number):
            # 返回一个新的 Interval 对象，其左右端点分别除以 y，保持闭合性与原对象一致
            return Interval(self.left / y, self.right / y, closed=self.closed)
        return NotImplemented

    # 定义一个整数除法运算符方法，用于处理 Interval 对象与数字的整数除法操作
    def __floordiv__(self, y):
        # 如果 y 是数字类型
        if isinstance(y, numbers.Number):
            # 返回一个新的 Interval 对象，其左右端点分别整数除以 y，保持闭合性与原对象一致
            return Interval(
                self.left // y, self.right // y, closed=self.closed)
        return NotImplemented

    # 定义一个方法用于检查两个 Interval 对象是否重叠
    def overlaps(self, other):
        """
        Check whether two Interval objects overlap.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Parameters
        ----------
        other : Interval
            Interval to check against for an overlap.

        Returns
        -------
        bool
            True if the two intervals overlap.

        Raises
        ------
        TypeError
            If `other` is not an Interval object.

        See Also
        --------
        IntervalArray.overlaps : The corresponding method for IntervalArray.
        IntervalIndex.overlaps : The corresponding method for IntervalIndex.

        Examples
        --------
        >>> i1 = pd.Interval(0, 2)
        >>> i2 = pd.Interval(1, 3)
        >>> i1.overlaps(i2)
        True
        >>> i3 = pd.Interval(4, 5)
        >>> i1.overlaps(i3)
        False

        Intervals that share closed endpoints overlap:

        >>> i4 = pd.Interval(0, 1, closed='both')
        >>> i5 = pd.Interval(1, 2, closed='both')
        >>> i4.overlaps(i5)
        True

        Intervals that only have an open endpoint in common do not overlap:

        >>> i6 = pd.Interval(1, 2, closed='neither')
        >>> i4.overlaps(i6)
        False
        """
        # 如果 other 不是 Interval 对象，则抛出类型错误异常
        if not isinstance(other, Interval):
            raise TypeError("`other` must be an Interval, "
                            f"got {type(other).__name__}")

        # 定义比较操作符，根据端点闭合性判断是否重叠
        op1 = le if (self.closed_left and other.closed_right) else lt
        op2 = le if (other.closed_left and self.closed_right) else lt

        # 判断两个 Interval 对象是否重叠，等价于判断它们是否不是分离的
        return op1(self.left, other.right) and op2(other.left, self.right)
# 设置 Cython 函数装饰器，禁用数组越界检查和负数索引检查
@cython.wraparound(False)
@cython.boundscheck(False)
def intervals_to_interval_bounds(ndarray intervals, bint validate_closed=True):
    """
    将间隔数组转换为左右边界数组和闭合类型。

    Parameters
    ----------
    intervals : ndarray
        包含间隔或空值的对象数组。

    validate_closed: bool, default True
        指示所有间隔是否必须在同一侧关闭的布尔值。
        如果为 True，不匹配的关闭将引发异常；否则对于不匹配的关闭返回 None。

    Returns
    -------
    tuple of
        left : ndarray
            左边界数组。
        right : ndarray
            右边界数组。
        closed: IntervalClosedType
            闭合类型。
    """
    cdef:
        object closed = None, interval  # 初始化闭合类型为 None
        Py_ssize_t i, n = len(intervals)  # 获取数组长度
        ndarray left, right  # 左右边界数组
        bint seen_closed = False  # 是否已经见过闭合类型的标志

    left = np.empty(n, dtype=intervals.dtype)  # 创建与间隔数组相同类型的空数组
    right = np.empty(n, dtype=intervals.dtype)  # 创建与间隔数组相同类型的空数组

    for i in range(n):
        interval = intervals[i]  # 获取当前间隔对象
        if interval is None or util.is_nan(interval):  # 处理空值或 NaN
            left[i] = np.nan  # 左边界设置为 NaN
            right[i] = np.nan  # 右边界设置为 NaN
            continue

        if not isinstance(interval, Interval):  # 检查间隔对象类型是否为 Interval
            raise TypeError(f"type {type(interval)} with value "
                            f"{interval} is not an interval")

        left[i] = interval.left  # 获取并记录左边界值
        right[i] = interval.right  # 获取并记录右边界值
        if not seen_closed:
            seen_closed = True
            closed = interval.closed  # 记录第一次见到的闭合类型
        elif closed != interval.closed:  # 检查闭合类型是否一致
            closed = None
            if validate_closed:
                raise ValueError("intervals must all be closed on the same side")  # 抛出异常，要求所有间隔在同一侧关闭

    return left, right, closed  # 返回左右边界数组和闭合类型


include "intervaltree.pxi"  # 导入 intervaltree.pxi 文件中的内容
```