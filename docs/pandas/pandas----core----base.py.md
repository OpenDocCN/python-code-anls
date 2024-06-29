# `D:\src\scipysrc\pandas\pandas\core\base.py`

```
# 导入未来版本支持的注解功能
from __future__ import annotations

# 导入标准库模块
import textwrap
# 从 typing 模块导入多个类型注解
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    cast,
    final,
    overload,
)

# 导入第三方库 numpy，并重命名为 np
import numpy as np

# 从 pandas._libs 中导入 lib 模块
from pandas._libs import lib
# 从 pandas._typing 中导入多个类型注解
from pandas._typing import (
    AxisInt,
    DtypeObj,
    IndexLabel,
    NDFrameT,
    Self,
    Shape,
    npt,
)
# 从 pandas.compat 中导入 PYPY 变量
from pandas.compat import PYPY
# 从 pandas.compat.numpy 中导入 function 函数，并重命名为 nv
from pandas.compat.numpy import function as nv
# 从 pandas.errors 中导入 AbstractMethodError 异常类
from pandas.errors import AbstractMethodError
# 从 pandas.util._decorators 中导入 cache_readonly 和 doc 修饰器函数
from pandas.util._decorators import (
    cache_readonly,
    doc,
)

# 从 pandas.core.dtypes.cast 中导入 can_hold_element 函数
from pandas.core.dtypes.cast import can_hold_element
# 从 pandas.core.dtypes.common 中导入多个函数
from pandas.core.dtypes.common import (
    is_object_dtype,
    is_scalar,
)
# 从 pandas.core.dtypes.dtypes 中导入 ExtensionDtype 类
from pandas.core.dtypes.dtypes import ExtensionDtype
# 从 pandas.core.dtypes.generic 中导入多个类
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCSeries,
)
# 从 pandas.core.dtypes.missing 中导入多个函数
from pandas.core.dtypes.missing import (
    isna,
    remove_na_arraylike,
)

# 从 pandas.core 中导入 algorithms、nanops 和 ops 模块
from pandas.core import (
    algorithms,
    nanops,
    ops,
)
# 从 pandas.core.accessor 中导入 DirNamesMixin 类
from pandas.core.accessor import DirNamesMixin
# 从 pandas.core.arraylike 中导入 OpsMixin 类
from pandas.core.arraylike import OpsMixin
# 从 pandas.core.arrays 中导入 ExtensionArray 类
from pandas.core.arrays import ExtensionArray
# 从 pandas.core.construction 中导入多个函数
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
)

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从 collections.abc 中导入 Hashable 和 Iterator 类型
    from collections.abc import (
        Hashable,
        Iterator,
    )
    # 从 pandas._typing 中导入多个类型注解
    from pandas._typing import (
        DropKeep,
        NumpySorter,
        NumpyValueArrayLike,
        ScalarLike_co,
    )
    # 从 pandas 中导入 DataFrame 和 Index 类
    from pandas import (
        DataFrame,
        Index,
        Series,
    )

# 创建一个空的共享文档字典
_shared_docs: dict[str, str] = {}


class PandasObject(DirNamesMixin):
    """
    pandas 对象的基类。
    """

    # 用于缓存只读属性方法调用结果的字典
    _cache: dict[str, Any]

    @property
    def _constructor(self) -> type[Self]:
        """
        返回当前对象的构造器（对于此类，即为 `__class__`）。
        """
        return type(self)

    def __repr__(self) -> str:
        """
        返回对象的字符串表示形式。
        """
        # 应该被子类重写
        return object.__repr__(self)

    def _reset_cache(self, key: str | None = None) -> None:
        """
        重置缓存属性。如果传入了 `key`，则仅清除该键。
        """
        # 如果对象没有 `_cache` 属性，则直接返回
        if not hasattr(self, "_cache"):
            return
        # 如果 `key` 为 `None`，则清空整个 `_cache`
        if key is None:
            self._cache.clear()
        else:
            # 否则从 `_cache` 中移除指定的 `key`
            self._cache.pop(key, None)

    def __sizeof__(self) -> int:
        """
        计算对象的内存使用量。
        """
        # 获取对象的 `memory_usage` 属性
        memory_usage = getattr(self, "memory_usage", None)
        # 如果 `memory_usage` 存在
        if memory_usage:
            # 调用 `memory_usage` 方法计算内存使用量（包括深层次）
            mem = memory_usage(deep=True)
            return int(mem if is_scalar(mem) else mem.sum())

        # 如果没有 `memory_usage` 属性，则调用父类的 `__sizeof__` 方法
        return super().__sizeof__()


class NoNewAttributesMixin:
    """
    用于阻止添加新属性的混合类。
    """
    # Mixin which prevents adding new attributes.

    # Prevents additional attributes via xxx.attribute = "something" after a
    # call to `self.__freeze()`. Mainly used to prevent the user from using
    # wrong attributes on an accessor (`Series.cat/.str/.dt`).

    # If you really want to add a new attribute at a later time, you need to use
    # `object.__setattr__(self, key, value)`.
    """

    # Prevents setting additional attributes.
    def _freeze(self) -> None:
        """
        Prevents setting additional attributes by setting '__frozen' to True.
        """
        object.__setattr__(self, "__frozen", True)

    # prevent adding any attribute via s.xxx.new_attribute = ...
    def __setattr__(self, key: str, value) -> None:
        """
        Overrides the __setattr__ method to prevent adding new attributes
        when '__frozen' attribute is True.

        Args:
            key (str): The attribute name.
            value: The value to set for the attribute.
        
        Raises:
            AttributeError: If '__frozen' is True and the attribute is not allowed.

        Notes:
            '_cache' is exempted from the restriction to support decorators.
            Both 'cls.__dict__' and 'getattr(self, key)' are checked to cover
            attributes that may raise errors or are defined in base classes.
        """
        if getattr(self, "__frozen", False) and not (
            key == "_cache"
            or key in type(self).__dict__
            or getattr(self, key, None) is not None
        ):
            raise AttributeError(f"You cannot add any new attribute '{key}'")
        object.__setattr__(self, key, value)
    """
    mixin implementing the selection & aggregation interface on a group-like
    object sub-classes need to define: obj, exclusions
    """

    # 泛型类 SelectionMixin，参数为泛型类型 NDFrameT
    obj: NDFrameT
    # _selection 用于存储选中的索引或标签，可以是单个值或列表
    _selection: IndexLabel | None = None
    # exclusions 是一个不可变的集合，存储要排除的键的哈希值
    exclusions: frozenset[Hashable]
    # _internal_names 存储内部名字的列表
    _internal_names = ["_cache", "__setstate__"]
    # _internal_names_set 是 _internal_names 的集合形式
    _internal_names_set = set(_internal_names)

    @final
    @property
    def _selection_list(self):
        # 如果 _selection 不是 list、tuple、ABCSeries、ABCIndex 或 np.ndarray 类型，将其包装为列表
        if not isinstance(
            self._selection, (list, tuple, ABCSeries, ABCIndex, np.ndarray)
        ):
            return [self._selection]
        return self._selection

    @cache_readonly
    def _selected_obj(self):
        # 如果 _selection 为 None 或者 obj 是 ABCSeries 类型，则返回 obj 本身
        if self._selection is None or isinstance(self.obj, ABCSeries):
            return self.obj
        else:
            # 否则，返回 obj 中选中的部分
            return self.obj[self._selection]

    @final
    @cache_readonly
    def ndim(self) -> int:
        # 返回 _selected_obj 的维度数
        return self._selected_obj.ndim

    @final
    @cache_readonly
    def _obj_with_exclusions(self):
        # 如果 obj 是 ABCSeries 类型，则返回 obj 本身
        if isinstance(self.obj, ABCSeries):
            return self.obj

        # 如果 _selection 不为 None，则返回 obj 中 _selection_list 所对应的部分
        if self._selection is not None:
            return self.obj[self._selection_list]

        # 如果 exclusions 集合中有元素，则从 obj 中移除这些列并返回结果
        if len(self.exclusions) > 0:
            # 相当于 self.obj.drop(self.exclusions, axis=1)，但是避免了复制操作
            return self.obj._drop_axis(self.exclusions, axis=1, only_slice=True)
        else:
            # 否则，返回 obj 本身
            return self.obj

    def __getitem__(self, key):
        # 如果 _selection 不为 None，则抛出索引错误，表示已经选择了列
        if self._selection is not None:
            raise IndexError(f"Column(s) {self._selection} already selected")

        # 如果 key 是 list、tuple、ABCSeries、ABCIndex 或 np.ndarray 类型
        if isinstance(key, (list, tuple, ABCSeries, ABCIndex, np.ndarray)):
            # 检查 key 中的列是否都存在于 obj 的列中
            if len(self.obj.columns.intersection(key)) != len(set(key)):
                bad_keys = list(set(key).difference(self.obj.columns))
                raise KeyError(f"Columns not found: {str(bad_keys)[1:-1]}")
            # 返回使用 key 获取的二维切片结果
            return self._gotitem(list(key), ndim=2)

        else:
            # 如果 key 不在 obj 的列中，则抛出 KeyError
            if key not in self.obj:
                raise KeyError(f"Column not found: {key}")
            # 获取 key 所对应列的维度并返回相应的切片结果
            ndim = self.obj[key].ndim
            return self._gotitem(key, ndim=ndim)

    def _gotitem(self, key, ndim: int, subset=None):
        """
        sub-classes to define
        return a sliced object

        Parameters
        ----------
        key : str / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        # 抽象方法错误，要求子类实现该方法
        raise AbstractMethodError(self)

    @final
    # 定义一个方法 `_infer_selection`，接受参数 `key` 和 `subset`，`subset` 可以是 Series 或 DataFrame 对象
    def _infer_selection(self, key, subset: Series | DataFrame):
        """
        Infer the `selection` to pass to our constructor in _gotitem.
        """
        # 初始化选择器为 None
        selection = None
        # 如果 subset 是二维且：
        # - key 是标量且在 subset 中
        # - 或者 key 是类列表对象
        if subset.ndim == 2 and (
            (lib.is_scalar(key) and key in subset) or lib.is_list_like(key)
        ):
            # 将选择器设置为 key
            selection = key
        # 如果 subset 是一维且 key 是标量且与 subset 的名称相同
        elif subset.ndim == 1 and lib.is_scalar(key) and key == subset.name:
            # 将选择器设置为 key
            selection = key
        # 返回推断出的选择器
        return selection

    # 定义一个抽象方法 `aggregate`，抛出抽象方法错误
    def aggregate(self, func, *args, **kwargs):
        raise AbstractMethodError(self)

    # 将 `aggregate` 方法别名 `agg` 指向 `aggregate` 方法
    agg = aggregate
    class IndexOpsMixin(OpsMixin):
        """
        Common ops mixin to support a unified interface / docs for Series / Index
        """

        # ndarray compatibility
        __array_priority__ = 1000
        _hidden_attrs: frozenset[str] = frozenset(
            ["tolist"]  # tolist is not deprecated, just suppressed in the __dir__
        )

        @property
        def dtype(self) -> DtypeObj:
            # 必须在此定义为属性以便于 mypy
            raise AbstractMethodError(self)

        @property
        def _values(self) -> ExtensionArray | np.ndarray:
            # 必须在此定义为属性以便于 mypy
            raise AbstractMethodError(self)

        @final
        def transpose(self, *args, **kwargs) -> Self:
            """
            返回转置，其定义即为 self。

            Returns
            -------
            %(klass)s
            """
            nv.validate_transpose(args, kwargs)
            return self

        T = property(
            transpose,
            doc="""
            返回转置，其定义即为 self。

            See Also
            --------
            Index : 用于索引和对齐的不可变序列。

            Examples
            --------
            对于 Series：

            >>> s = pd.Series(['Ant', 'Bear', 'Cow'])
            >>> s
            0     Ant
            1    Bear
            2     Cow
            dtype: object
            >>> s.T
            0     Ant
            1    Bear
            2     Cow
            dtype: object

            对于 Index：

            >>> idx = pd.Index([1, 2, 3])
            >>> idx.T
            Index([1, 2, 3], dtype='int64')
            """,
        )

        @property
        def shape(self) -> Shape:
            """
            返回基础数据的形状的元组。

            See Also
            --------
            Series.ndim : 基础数据的维数。
            Series.size : 返回基础数据中的元素数。
            Series.nbytes : 返回基础数据中的字节数。

            Examples
            --------
            >>> s = pd.Series([1, 2, 3])
            >>> s.shape
            (3,)
            """
            return self._values.shape

        def __len__(self) -> int:
            # 我们需要在此定义以便于 mypy
            raise AbstractMethodError(self)
    def ndim(self) -> Literal[1]:
        """
        Number of dimensions of the underlying data, by definition 1.

        See Also
        --------
        Series.size: Return the number of elements in the underlying data.
        Series.shape: Return a tuple of the shape of the underlying data.
        Series.dtype: Return the dtype object of the underlying data.
        Series.values: Return Series as ndarray or ndarray-like depending on the dtype.

        Examples
        --------
        >>> s = pd.Series(["Ant", "Bear", "Cow"])
        >>> s
        0     Ant
        1    Bear
        2     Cow
        dtype: object
        >>> s.ndim
        1

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.ndim
        1
        """
        return 1

    @final
    def item(self):
        """
        Return the first element of the underlying data as a Python scalar.

        Returns
        -------
        scalar
            The first element of Series or Index.

        Raises
        ------
        ValueError
            If the data is not length = 1.

        See Also
        --------
        Index.values : Returns an array representing the data in the Index.
        Series.head : Returns the first `n` rows.

        Examples
        --------
        >>> s = pd.Series([1])
        >>> s.item()
        1

        For an index:

        >>> s = pd.Series([1], index=["a"])
        >>> s.index.item()
        'a'
        """
        if len(self) == 1:
            return next(iter(self))
        raise ValueError("can only convert an array of size 1 to a Python scalar")

    @property
    def nbytes(self) -> int:
        """
        Return the number of bytes in the underlying data.

        See Also
        --------
        Series.ndim : Number of dimensions of the underlying data.
        Series.size : Return the number of elements in the underlying data.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["Ant", "Bear", "Cow"])
        >>> s
        0     Ant
        1    Bear
        2     Cow
        dtype: object
        >>> s.nbytes
        24

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.nbytes
        24
        """
        return self._values.nbytes
    # 返回基础数据中的元素数量
    """
    返回基础数据中的元素数量。

    See Also
    --------
    Series.ndim: 返回基础数据的维数，按定义为1。
    Series.shape: 返回基础数据的形状的元组。
    Series.dtype: 返回基础数据的dtype对象。
    Series.values: 根据dtype返回Series作为ndarray或类似ndarray的对象。

    Examples
    --------
    对于Series：

    >>> s = pd.Series(["Ant", "Bear", "Cow"])
    >>> s
    0     Ant
    1    Bear
    2     Cow
    dtype: object
    >>> s.size
    3

    对于Index：

    >>> idx = pd.Index([1, 2, 3])
    >>> idx
    Index([1, 2, 3], dtype='int64')
    >>> idx.size
    3
    """
    return len(self._values)
    def array(self) -> ExtensionArray:
        """
        The ExtensionArray of the data backing this Series or Index.

        Returns
        -------
        ExtensionArray
            An ExtensionArray of the values stored within. For extension
            types, this is the actual array. For NumPy native types, this
            is a thin (no copy) wrapper around :class:`numpy.ndarray`.

            ``.array`` differs from ``.values``, which may require converting
            the data to a different form.

        See Also
        --------
        Index.to_numpy : Similar method that always returns a NumPy array.
        Series.to_numpy : Similar method that always returns a NumPy array.

        Notes
        -----
        This table lays out the different array types for each extension
        dtype within pandas.

        ================== =============================
        dtype              array type
        ================== =============================
        category           Categorical
        period             PeriodArray
        interval           IntervalArray
        IntegerNA          IntegerArray
        string             StringArray
        boolean            BooleanArray
        datetime64[ns, tz] DatetimeArray
        ================== =============================

        For any 3rd-party extension types, the array type will be an
        ExtensionArray.

        For all remaining dtypes ``.array`` will be a
        :class:`arrays.NumpyExtensionArray` wrapping the actual ndarray
        stored within. If you absolutely need a NumPy array (possibly with
        copying / coercing data), then use :meth:`Series.to_numpy` instead.

        Examples
        --------
        For regular NumPy types like int, and float, a NumpyExtensionArray
        is returned.

        >>> pd.Series([1, 2, 3]).array
        <NumpyExtensionArray>
        [1, 2, 3]
        Length: 3, dtype: int64

        For extension types, like Categorical, the actual ExtensionArray
        is returned

        >>> ser = pd.Series(pd.Categorical(["a", "b", "a"]))
        >>> ser.array
        ['a', 'b', 'a']
        Categories (2, object): ['a', 'b']
        """
        # 抽象方法，由子类实现具体逻辑
        raise AbstractMethodError(self)

    # 将 Series 或 Index 转换为 NumPy 数组的方法
    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,  # 可选参数，指定返回的 NumPy 数组的数据类型
        copy: bool = False,  # 是否复制数据到新数组，默认为 False
        na_value: object = lib.no_default,  # 缺失值的替代值，默认为库定义的 no_default
        **kwargs,  # 其他关键字参数，传递给底层的转换函数
    ):
        @final  # 标记方法为最终方法，不允许子类重写
        @property  # 属性装饰器，使得该方法可以像属性一样调用，而非方法调用
    # 定义一个方法用于检查索引是否为空
    def empty(self) -> bool:
        """
        Indicator whether Index is empty.

        An Index is considered empty if it has no elements. This property can be
        useful for quickly checking the state of an Index, especially in data
        processing and analysis workflows where handling of empty datasets might
        be required.

        Returns
        -------
        bool
            If Index is empty, return True, if not return False.

        See Also
        --------
        Index.size : Return the number of elements in the underlying data.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.empty
        False

        >>> idx_empty = pd.Index([])
        >>> idx_empty
        Index([], dtype='object')
        >>> idx_empty.empty
        True

        If we only have NaNs in our DataFrame, it is not considered empty!

        >>> idx = pd.Index([np.nan, np.nan])
        >>> idx
        Index([nan, nan], dtype='float64')
        >>> idx.empty
        False
        """
        return not self.size

    @doc(op="max", oppose="min", value="largest")
    # 定义一个用于返回最大值索引的方法，并指定了相关的文档信息
    def argmax(
        self, axis: AxisInt | None = None, skipna: bool = True, *args, **kwargs
    ) -> int:
        """
        Return int position of the minimum value in the Series.

        If the minimum is achieved in multiple locations,
        the first row position is returned.

        Parameters
        ----------
        axis : {{None}}
            Unused. Parameter needed for compatibility with DataFrame.
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, or if ``skipna=False``
            and there is an NA value, this method will raise a ``ValueError``.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        int
            Row position of the minimum value.

        See Also
        --------
        Series.argmax : Return position of the maximum value.
        Series.argmax : Return position of the maximum value.
        numpy.ndarray.argmin : Equivalent method for numpy arrays.
        Series.idxmax : Return index label of the maximum values.
        Series.idxmin : Return index label of the minimum values.

        Examples
        --------
        Consider dataset containing cereal calories

        >>> s = pd.Series(
        ...     [100.0, 110.0, 120.0, 110.0],
        ...     index=[
        ...         "Corn Flakes",
        ...         "Almond Delight",
        ...         "Cinnamon Toast Crunch",
        ...         "Cocoa Puff",
        ...     ],
        ... )
        >>> s
        Corn Flakes              100.0
        Almond Delight           110.0
        Cinnamon Toast Crunch    120.0
        Cocoa Puff               110.0
        dtype: float64

        >>> s.argmax()
        2
        >>> s.argmin()
        0

        The maximum cereal calories is the third element and
        the minimum cereal calories is the first element,
        since series is zero-indexed.
        """
        # Delegate to the underlying values of the Series
        delegate = self._values
        # Validate the axis value (though it is unused here)
        nv.validate_minmax_axis(axis)
        # Validate skipna parameter based on arguments and keywords
        skipna = nv.validate_argmax_with_skipna(skipna, args, kwargs)

        # Check if the delegate is an ExtensionArray, then use its argmin method
        if isinstance(delegate, ExtensionArray):
            return delegate.argmin(skipna=skipna)
        else:
            # Otherwise, use nanops.nanargmin to find the argmin of the delegate
            result = nanops.nanargmin(delegate, skipna=skipna)
            # error: Incompatible return value type (got "Union[int, ndarray]", expected
            # "int")
            # Return the result, ignoring type checking for return value
            return result  # type: ignore[return-value]
    ) -> int:
        # 获取代理对象
        delegate = self._values
        # 验证轴的最小最大值
        nv.validate_minmax_axis(axis)
        # 验证 argmax 是否带有 skipna 参数
        skipna = nv.validate_argmax_with_skipna(skipna, args, kwargs)

        # 如果 delegate 是 ExtensionArray 对象
        if isinstance(delegate, ExtensionArray):
            # 返回 delegate 的 argmin 结果
            return delegate.argmin(skipna=skipna)
        else:
            # 否则，使用 nanops.nanargmin 函数计算结果
            result = nanops.nanargmin(delegate, skipna=skipna)
            # 错误: 返回值类型不兼容 (得到 "Union[int, ndarray]", 期望 "int")
            return result  # type: ignore[return-value]

    def tolist(self) -> list:
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        list
            List containing the values as Python or pandas scalars.

        See Also
        --------
        numpy.ndarray.tolist : Return the array as an a.ndim-levels deep
            nested list of Python scalars.

        Examples
        --------
        For Series

        >>> s = pd.Series([1, 2, 3])
        >>> s.to_list()
        [1, 2, 3]

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')

        >>> idx.to_list()
        [1, 2, 3]
        """
        return self._values.tolist()

    to_list = tolist

    def __iter__(self) -> Iterator:
        """
        Return an iterator of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        iterator
            An iterator yielding scalar values from the Series.

        See Also
        --------
        Series.items : Lazily iterate over (index, value) tuples.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> for x in s:
        ...     print(x)
        1
        2
        3
        """
        # 显式创建元素迭代器
        if not isinstance(self._values, np.ndarray):
            # 如果 _values 不是 ndarray 类型，则直接返回其迭代器
            return iter(self._values)
        else:
            # 否则，使用 map 函数将索引范围内的 _values.item 转换为迭代器
            return map(self._values.item, range(self._values.size))

    @cache_readonly
    def hasnans(self) -> bool:
        """
        Return True if there are any NaNs.

        Enables various performance speedups.

        Returns
        -------
        bool

        See Also
        --------
        Series.isna : Detect missing values.
        Series.notna : Detect existing (non-missing) values.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, None])
        >>> s
        0    1.0
        1    2.0
        2    3.0
        3    NaN
        dtype: float64
        >>> s.hasnans
        True
        """
        # 检查是否存在任何 NaN 值，并转换结果为布尔类型
        return bool(isna(self).any())  # type: ignore[union-attr]

    @final
    def _map_values(self, mapper, na_action=None):
        """
        An internal function that maps values using the input
        correspondence (which can be a dict, Series, or function).

        Parameters
        ----------
        mapper : function, dict, or Series
            The input correspondence object
        na_action : {None, 'ignore'}
            If 'ignore', propagate NA values, without passing them to the
            mapping function

        Returns
        -------
        Union[Index, MultiIndex], inferred
            The output of the mapping function applied to the index.
            If the function returns a tuple with more than one element
            a MultiIndex will be returned.
        """
        # 获取当前对象的值数组
        arr = self._values

        # 如果值数组是 ExtensionArray 类型，则调用其 map 方法进行映射
        if isinstance(arr, ExtensionArray):
            return arr.map(mapper, na_action=na_action)

        # 否则，调用算法模块中的 map_array 函数进行映射
        return algorithms.map_array(arr, mapper, na_action=na_action)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ) -> Series:
        """
        Return a Series containing counts of unique values.

        The resulting object will be in descending order so that the
        first element is the most frequently-occurring element.
        Excludes NA values by default.

        Parameters
        ----------
        normalize : bool, default False
            If True then the object returned will contain the relative
            frequencies of the unique values.
        sort : bool, default True
            Sort by frequencies when True. Preserve the order of the data when False.
        ascending : bool, default False
            Sort in ascending order.
        bins : int, optional
            Rather than count values, group them into half-open bins,
            a convenience for ``pd.cut``, only works with numeric data.
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        Series
            Series containing counts of unique values.

        See Also
        --------
        Series.count: Number of non-NA elements in a Series.
        DataFrame.count: Number of non-NA elements in a DataFrame.
        DataFrame.value_counts: Equivalent method on DataFrames.

        Examples
        --------
        >>> index = pd.Index([3, 1, 2, 3, 4, np.nan])
        >>> index.value_counts()
        3.0    2
        1.0    1
        2.0    1
        4.0    1
        Name: count, dtype: int64

        With `normalize` set to `True`, returns the relative frequency by
        dividing all values by the sum of values.

        >>> s = pd.Series([3, 1, 2, 3, 4, np.nan])
        >>> s.value_counts(normalize=True)
        3.0    0.4
        1.0    0.2
        2.0    0.2
        4.0    0.2
        Name: proportion, dtype: float64

        **bins**

        Bins can be useful for going from a continuous variable to a
        categorical variable; instead of counting unique
        apparitions of values, divide the index in the specified
        number of half-open bins.

        >>> s.value_counts(bins=3)
        (0.996, 2.0]    2
        (2.0, 3.0]      2
        (3.0, 4.0]      1
        Name: count, dtype: int64

        **dropna**

        With `dropna` set to `False` we can also see NaN index values.

        >>> s.value_counts(dropna=False)
        3.0    2
        1.0    1
        2.0    1
        4.0    1
        NaN    1
        Name: count, dtype: int64
        """
        调用内部方法 `algorithms.value_counts_internal` 计算 Series 中唯一值的计数，并返回结果。

    def unique(self):
        values = self._values
        if not isinstance(values, np.ndarray):
            # i.e. ExtensionArray
            result = values.unique()
        else:
            result = algorithms.unique1d(values)
        return result

    @final
    # 返回对象中唯一元素的数量
    def nunique(self, dropna: bool = True) -> int:
        """
        Return number of unique elements in the object.

        Excludes NA values by default.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the count.

        Returns
        -------
        int
            A integer indicating the number of unique elements in the object.

        See Also
        --------
        DataFrame.nunique: Method nunique for DataFrame.
        Series.count: Count non-NA/null observations in the Series.

        Examples
        --------
        >>> s = pd.Series([1, 3, 5, 7, 7])
        >>> s
        0    1
        1    3
        2    5
        3    7
        4    7
        dtype: int64

        >>> s.nunique()
        4
        """
        # 获取对象中的唯一值
        uniqs = self.unique()
        # 如果 dropna 参数为 True，则移除唯一值中的 NaN
        if dropna:
            uniqs = remove_na_arraylike(uniqs)
        # 返回唯一值的数量
        return len(uniqs)

    @property
    # 返回对象的值是否全部唯一
    def is_unique(self) -> bool:
        """
        Return True if values in the object are unique.

        Returns
        -------
        bool

        See Also
        --------
        Series.unique : Return unique values of Series object.
        Series.drop_duplicates : Return Series with duplicate values removed.
        Series.duplicated : Indicate duplicate Series values.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.is_unique
        True

        >>> s = pd.Series([1, 2, 3, 1])
        >>> s.is_unique
        False
        """
        # 检查对象的唯一值数量是否等于对象长度
        return self.nunique(dropna=False) == len(self)

    @property
    # 返回对象的值是否单调递增
    def is_monotonic_increasing(self) -> bool:
        """
        Return True if values in the object are monotonically increasing.

        Returns
        -------
        bool

        See Also
        --------
        Series.is_monotonic_decreasing : Return boolean if values in the object are
            monotonically decreasing.

        Examples
        --------
        >>> s = pd.Series([1, 2, 2])
        >>> s.is_monotonic_increasing
        True

        >>> s = pd.Series([3, 2, 1])
        >>> s.is_monotonic_increasing
        False
        """
        from pandas import Index

        # 使用 Index 类判断对象的值是否单调递增
        return Index(self).is_monotonic_increasing

    @property
    # 返回对象的值是否单调递减
    def is_monotonic_decreasing(self) -> bool:
        """
        Return True if values in the object are monotonically decreasing.

        Returns
        -------
        bool

        See Also
        --------
        Series.is_monotonic_increasing : Return boolean if values in the object are
            monotonically increasing.

        Examples
        --------
        >>> s = pd.Series([3, 2, 2, 1])
        >>> s.is_monotonic_decreasing
        True

        >>> s = pd.Series([1, 2, 3])
        >>> s.is_monotonic_decreasing
        False
        """
        from pandas import Index

        # 使用 Index 类判断对象的值是否单调递减
        return Index(self).is_monotonic_decreasing

    @final
    def _memory_usage(self, deep: bool = False) -> int:
        """
        返回值的内存使用情况。

        Parameters
        ----------
        deep : bool, default False
            是否深度检查数据，用于系统级内存消耗。

        Returns
        -------
        bytes used
            返回索引中值的内存使用情况（单位字节）。

        See Also
        --------
        numpy.ndarray.nbytes : 数组元素消耗的总字节数。

        Notes
        -----
        如果 deep=False 或者在 PyPy 上使用，内存使用情况不包括不是数组组成部分的元素消耗。

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.memory_usage()
        24
        """
        if hasattr(self.array, "memory_usage"):
            # 如果数组对象有 memory_usage 方法，则调用该方法返回内存使用情况
            return self.array.memory_usage(
                deep=deep,
            )

        # 否则，计算数组元素的 nbytes
        v = self.array.nbytes
        if deep and is_object_dtype(self.dtype) and not PYPY:
            # 如果 deep=True 并且数据类型为对象且不在 PyPy 上，则计算对象的内存使用情况
            values = cast(np.ndarray, self._values)
            v += lib.memory_usage_of_objects(values)
        return v

    @doc(
        algorithms.factorize,
        values="",
        order="",
        size_hint="",
        sort=textwrap.dedent(
            """\
            sort : bool, default False
                排序 `uniques` 并重排 `codes` 以保持它们之间的关系。
            """
        ),
    )
    def factorize(
        self,
        sort: bool = False,
        use_na_sentinel: bool = True,
    ) -> tuple[npt.NDArray[np.intp], Index]:
        """
        将索引的值因子化为整数编码和唯一值。

        Parameters
        ----------
        sort : bool, default False
            是否对唯一值进行排序并对编码进行重排以保持它们的关系。
        use_na_sentinel : bool, default True
            是否使用 NA 标记来编码缺失值。

        Returns
        -------
        tuple[npt.NDArray[np.intp], Index]
            返回编码和唯一值的元组。

        """
        # 调用 pandas 的 algorithms.factorize 方法来执行因子化
        codes, uniques = algorithms.factorize(
            self._values, sort=sort, use_na_sentinel=use_na_sentinel
        )

        if uniques.dtype == np.float16:
            # 如果唯一值的数据类型是 np.float16，则转换为 np.float32
            uniques = uniques.astype(np.float32)

        if isinstance(self, ABCIndex):
            # 如果是 ABCIndex 的实例，保留原始类型（例如 MultiIndex）
            uniques = self._constructor(uniques)
        else:
            # 否则，使用 pandas 中的 Index 类来构造唯一值的索引
            from pandas import Index
            uniques = Index(uniques)

        return codes, uniques
    # 将文档字符串添加到_shared_docs字典中，用于searchsorted方法的文档
    _shared_docs["searchsorted"] = """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted {klass} `self` such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        .. note::

            The {klass} *must* be monotonically sorted, otherwise
            wrong locations will likely be returned. Pandas does *not*
            check this for you.

        Parameters
        ----------
        value : array-like or scalar
            Values to insert into `self`.
        side : {{'left', 'right'}}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array-like, optional
            Optional array of integer indices that sort `self` into ascending
            order. They are typically the result of ``np.argsort``.

        Returns
        -------
        int or array of int
            A scalar or array of insertion points with the
            same shape as `value`.

        See Also
        --------
        sort_values : Sort by the values along either axis.
        numpy.searchsorted : Similar method from NumPy.

        Notes
        -----
        Binary search is used to find the required insertion points.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3])
        >>> ser
        0    1
        1    2
        2    3
        dtype: int64

        >>> ser.searchsorted(4)
        3

        >>> ser.searchsorted([0, 4])
        array([0, 3])

        >>> ser.searchsorted([1, 3], side='left')
        array([0, 2])

        >>> ser.searchsorted([1, 3], side='right')
        array([1, 3])

        >>> ser = pd.Series(pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000']))
        >>> ser
        0   2000-03-11
        1   2000-03-12
        2   2000-03-13
        dtype: datetime64[s]

        >>> ser.searchsorted('3/14/2000')
        3

        >>> ser = pd.Categorical(
        ...     ['apple', 'bread', 'bread', 'cheese', 'milk'], ordered=True
        ... )
        >>> ser
        ['apple', 'bread', 'bread', 'cheese', 'milk']
        Categories (4, object): ['apple' < 'bread' < 'cheese' < 'milk']

        >>> ser.searchsorted('bread')
        1

        >>> ser.searchsorted(['bread'], side='right')
        array([3])

        If the values are not monotonically sorted, wrong locations
        may be returned:

        >>> ser = pd.Series([2, 1, 3])
        >>> ser
        0    2
        1    1
        2    3
        dtype: int64

        >>> ser.searchsorted(1)  # doctest: +SKIP
        0  # wrong result, correct would be 1
        """

    # 这个重载是必需的，以便在
    # pandas.core.resample.TimeGrouper._get_period_bins picks the correct result

    # error: Overloaded function signatures 1 and 2 overlap with incompatible
    # return types
    @overload
    def searchsorted(  # type: ignore[overload-overlap]
        self,
        value: ScalarLike_co,
        side: Literal["left", "right"] = ...,
        sorter: NumpySorter = ...,
    ) -> np.intp: ...
    # 重载定义，用于标记类型检查忽略重载冲突问题

    @overload
    def searchsorted(
        self,
        value: npt.ArrayLike | ExtensionArray,
        side: Literal["left", "right"] = ...,
        sorter: NumpySorter = ...,
    ) -> npt.NDArray[np.intp]: ...
    # 重载定义，适用于数组和扩展数组，返回类型为整型数组

    @doc(_shared_docs["searchsorted"], klass="Index")
    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        if isinstance(value, ABCDataFrame):
            msg = (
                "Value must be 1-D array-like or scalar, "
                f"{type(value).__name__} is not supported"
            )
            raise ValueError(msg)
        # 检查输入值是否为数据帧，抛出值错误异常

        values = self._values
        if not isinstance(values, np.ndarray):
            # 直接通过 EA.searchsorted 优化性能 GH#38083
            return values.searchsorted(value, side=side, sorter=sorter)
        # 如果不是 NumPy 数组，则通过 values 的 searchsorted 方法查找值

        return algorithms.searchsorted(
            values,
            value,
            side=side,
            sorter=sorter,
        )
        # 否则，使用算法模块的 searchsorted 函数进行查找操作

    def drop_duplicates(self, *, keep: DropKeep = "first") -> Self:
        duplicated = self._duplicated(keep=keep)
        # error: Value of type "IndexOpsMixin" is not indexable
        return self[~duplicated]  # type: ignore[index]
        # 返回去除重复项后的结果，忽略类型检查中的索引错误提示

    @final
    def _duplicated(self, keep: DropKeep = "first") -> npt.NDArray[np.bool_]:
        arr = self._values
        if isinstance(arr, ExtensionArray):
            return arr.duplicated(keep=keep)
        # 如果是扩展数组，则调用扩展数组的 duplicated 方法

        return algorithms.duplicated(arr, keep=keep)
        # 否则，调用算法模块的 duplicated 函数进行重复项查找

    def _arith_method(self, other, op):
        res_name = ops.get_op_result_name(self, other)

        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)
        rvalues = ops.maybe_prepare_scalar_for_op(rvalues, lvalues.shape)
        rvalues = ensure_wrapped_if_datetimelike(rvalues)
        if isinstance(rvalues, range):
            rvalues = np.arange(rvalues.start, rvalues.stop, rvalues.step)
        # 准备右侧操作数，处理成适合进行算术操作的形式

        with np.errstate(all="ignore"):
            result = ops.arithmetic_op(lvalues, rvalues, op)
        # 执行算术操作，忽略所有的 NumPy 错误

        return self._construct_result(result, name=res_name)

    def _construct_result(self, result, name):
        """
        Construct an appropriately-wrapped result from the ArrayLike result
        of an arithmetic-like operation.
        """
        raise AbstractMethodError(self)
        # 构造算术操作的结果，由子类实现具体逻辑
```