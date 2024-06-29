# `D:\src\scipysrc\pandas\pandas\core\arrays\base.py`

```
"""
An interface for extending pandas with custom arrays.

.. warning::

   This is an experimental API and subject to breaking changes
   without warning.
"""

from __future__ import annotations

import operator
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    algos as libalgos,
    lib,
)
from pandas.compat import set_function_name
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
    Appender,
    Substitution,
    cache_readonly,
)
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
    validate_bool_kwarg,
    validate_insert_loc,
)

from pandas.core.dtypes.cast import maybe_cast_pointwise_result
from pandas.core.dtypes.common import (
    is_list_like,
    is_scalar,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

from pandas.core import (
    arraylike,
    missing,
    roperator,
)
from pandas.core.algorithms import (
    duplicated,
    factorize_array,
    isin,
    map_array,
    mode,
    rank,
    unique,
)
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.missing import _fill_limit_area_1d
from pandas.core.sorting import (
    nargminmax,
    nargsort,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterator,
        Sequence,
    )

    from pandas._libs.missing import NAType
    from pandas._typing import (
        ArrayLike,
        AstypeArg,
        AxisInt,
        Dtype,
        DtypeObj,
        FillnaOptions,
        InterpolateOptions,
        NumpySorter,
        NumpyValueArrayLike,
        PositionalIndexer,
        ScalarIndexer,
        Self,
        SequenceIndexer,
        Shape,
        SortKind,
        TakeIndexer,
        npt,
    )

    from pandas import Index

_extension_array_shared_docs: dict[str, str] = {}


class ExtensionArray:
    """
    Abstract base class for custom 1-D array types.

    pandas will recognize instances of this class as proper arrays
    with a custom type and will not attempt to coerce them to objects. They
    may be stored directly inside a :class:`DataFrame` or :class:`Series`.

    Attributes
    ----------
    dtype
        The data type of elements in the array.
    nbytes
        The total number of bytes consumed by the elements of the array.
    ndim
        The number of dimensions of the array.
    shape
        The shape of the array.

    Methods
    -------
    argsort
        Returns the indices that would sort the array.
    astype
        Casts the array to a specified dtype.
    copy
        Returns a copy of the array.
    dropna
        Returns an array with NaN values dropped.
    duplicated
        Returns a boolean array indicating duplicate values.
    factorize
        Returns unique values and an indexer.
    fillna
        Fills NaN values with a specified method.
    equals
        Compares if two arrays are equal.
    insert
        Inserts new values at specified locations.
    interpolate
        Interpolates values to fill NaNs.
    isin
        Returns a boolean array indicating whether each element is contained in the provided values.
    isna
        Returns a boolean array indicating whether each element is NaN.
    ravel
        Returns a flattened version of the array.
    repeat
        Repeats elements of the array.
    searchsorted
        Finds indices where elements should be inserted to maintain order.
    shift
        Shifts elements of the array.
    take
        Takes elements from the array at specified indices.
    tolist
        Converts the array to a Python list.
    unique
        Returns unique elements of the array.
    view
        Returns a view of the array.
    _accumulate
        Internal method for cumulative operations.
    _concat_same_type
        Internal method for concatenation.
    _explode
        Internal method for exploding array-like elements.
    _formatter
        Internal method for formatting.
    _from_factorized
        Internal method for creating an array from factorized data.
    _from_sequence
        Internal method for creating an array from a sequence.
    _from_sequence_of_strings
        Internal method for creating an array from a sequence of strings.
    _hash_pandas_object
        Internal method for hashing pandas objects.
    _pad_or_backfill
        Internal method for padding or backfilling.
    _reduce
        Internal method for reducing the array.
    _values_for_argsort
        Internal method for retrieving values to use for argsort.
    """
    pass
    _values_for_factorize
    # 定义了一个方法或者属性名为 _values_for_factorize

    See Also
    --------
    api.extensions.ExtensionDtype : A custom data type, to be paired with an
        ExtensionArray.
    api.extensions.ExtensionArray.dtype : An instance of ExtensionDtype.
    # 参见关联内容：api.extensions.ExtensionDtype，用于扩展数组的自定义数据类型；
    # api.extensions.ExtensionArray.dtype，ExtensionDtype 的实例。

    Notes
    -----
    The interface includes the following abstract methods that must be
    implemented by subclasses:
    # 下面列出了子类必须实现的抽象方法：

    * _from_sequence
    * _from_factorized
    * __getitem__
    * __len__
    * __eq__
    * dtype
    * nbytes
    * isna
    * take
    * copy
    * _concat_same_type
    * interpolate

    A default repr displaying the type, (truncated) data, length,
    and dtype is provided. It can be customized or replaced by
    by overriding:
    # 提供了默认的 __repr__ 方法，显示类型、（截断的）数据、长度和 dtype。
    # 可以通过重写来自定义或替换：

    * __repr__ : A default repr for the ExtensionArray.
    * _formatter : Print scalars inside a Series or DataFrame.

    Some methods require casting the ExtensionArray to an ndarray of Python
    objects with ``self.astype(object)``, which may be expensive. When
    performance is a concern, we highly recommend overriding the following
    methods:
    # 一些方法需要将 ExtensionArray 转换为 Python 对象的 ndarray，如 ``self.astype(object)``，这可能很昂贵。
    # 当性能是一个问题时，强烈建议重写以下方法：

    * fillna
    * _pad_or_backfill
    * dropna
    * unique
    * factorize / _values_for_factorize
    * argsort, argmax, argmin / _values_for_argsort
    * searchsorted
    * map

    The remaining methods implemented on this class should be performant,
    as they only compose abstract methods. Still, a more efficient
    implementation may be available, and these methods can be overridden.
    # 此类上实现的其余方法应该表现良好，因为它们仅仅是组成抽象方法。
    # 仍然可以重写这些方法以提高效率。

    One can implement methods to handle array accumulations or reductions.
    # 可以实现处理数组累加或减少的方法。

    * _accumulate
    * _reduce

    One can implement methods to handle parsing from strings that will be used
    in methods such as ``pandas.io.parsers.read_csv``.
    # 可以实现处理从字符串解析的方法，将在如 ``pandas.io.parsers.read_csv`` 中使用。

    * _from_sequence_of_strings

    This class does not inherit from 'abc.ABCMeta' for performance reasons.
    Methods and properties required by the interface raise
    ``pandas.errors.AbstractMethodError`` and no ``register`` method is
    provided for registering virtual subclasses.
    # 由于性能原因，此类没有继承 'abc.ABCMeta'。
    # 由接口需要的方法和属性引发 ``pandas.errors.AbstractMethodError``，且不提供 ``register`` 方法用于注册虚拟子类。

    ExtensionArrays are limited to 1 dimension.
    # ExtensionArrays 限制为 1 维数组。

    They may be backed by none, one, or many NumPy arrays. For example,
    ``pandas.Categorical`` is an extension array backed by two arrays,
    one for codes and one for categories. An array of IPv6 address may
    be backed by a NumPy structured array with two fields, one for the
    lower 64 bits and one for the upper 64 bits. Or they may be backed
    by some other storage type, like Python lists. Pandas makes no
    assumptions on how the data are stored, just that it can be converted
    to a NumPy array.
    # 它们可以由零个、一个或多个 NumPy 数组支持。
    # 例如，``pandas.Categorical`` 是由两个数组支持的扩展数组，一个用于编码，一个用于类别。
    # IPv6 地址数组可以由一个带有两个字段的 NumPy 结构化数组支持，一个用于低 64 位，一个用于高 64 位。
    # 或者它们可以由其他存储类型支持，如 Python 列表。
    # Pandas 不假设数据的存储方式，只是可以将其转换为 NumPy 数组。

    The ExtensionArray interface does not impose any rules on how this data
    is stored. However, currently, the backing data cannot be stored in
    attributes called ``.values`` or ``._values`` to ensure full compatibility
    with pandas internals. But other names as ``.data``, ``._data``,
    ``._items``, ... can be freely used.
    # ExtensionArray 接口对数据的存储方式没有施加任何规则。
    # 然而，当前情况下，为了确保与 pandas 内部的完全兼容性，支持数据不能存储在名为 ``.values`` 或 ``._values`` 的属性中。
    # 但其他名称如 ``.data``、``._data``、``._items`` 等可以自由使用。

    If implementing NumPy's ``__array_ufunc__`` interface, pandas expects
    that
    # 如果实现了 NumPy 的 ``__array_ufunc__`` 接口，pandas 期望
    # 返回 `NotImplemented`，如果输入中包含任何 Series 对象，则推迟处理。
    # Pandas 会提取数组并再次调用该 ufunc。
    You defer by returning ``NotImplemented`` when any Series are present
    in `inputs`. Pandas will extract the arrays and call the ufunc again.
    
    # 定义一个 `_HANDLED_TYPES` 元组作为类的属性。
    # Pandas 使用此元组来确定当前 ufunc 对所涉及的类型是否有效。
    You define a ``_HANDLED_TYPES`` tuple as an attribute on the class.
    Pandas inspects this to determine whether the ufunc is valid for the
    types present.
    
    # 默认情况下，ExtensionArrays 是不可哈希的。不可变的子类可以重写此行为。
    By default, ExtensionArrays are not hashable. Immutable subclasses may
    override this behavior.
    
    # ------------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------------
    
    # `_typ` 用于指定 pandas.core.dtypes.generic.ABCExtensionArray 类型。
    # 不要覆盖此属性。
    '_typ' is for pandas.core.dtypes.generic.ABCExtensionArray.
    Don't override this.
    
    # 类似于 __array_priority__，将 ExtensionArray 的优先级设置在 Index、Series 和 DataFrame 之后。
    # ExtensionArray 的子类可以重写此属性，以确定其优先级顺序。
    # 如果要重写，则值应始终严格小于 2000，以确保低于 Index.__pandas_priority__。
    similar to __array_priority__, positions ExtensionArray after Index,
    Series, and DataFrame. EA subclasses may override to choose which EA
    subclass takes priority. If overriding, the value should always be
    strictly less than 2000 to be below Index.__pandas_priority__.
    
    # ------------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------------
    
    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        """
        Construct a new ExtensionArray from a sequence of scalars.
    
        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type`` or be converted into this type in this method.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.
    
        Returns
        -------
        ExtensionArray
    
        See Also
        --------
        api.extensions.ExtensionArray._from_sequence_of_strings : Construct a new
            ExtensionArray from a sequence of strings.
        api.extensions.ExtensionArray._hash_pandas_object : Hook for
            hash_pandas_object.
    
        Examples
        --------
        >>> pd.arrays.IntegerArray._from_sequence([4, 5])
        <IntegerArray>
        [4, 5]
        Length: 2, dtype: Int64
        """
        raise AbstractMethodError(cls)
    
    
    这段代码是一个类的部分实现，涵盖了扩展数组（ExtensionArray）的一些属性和构造方法的声明。
    def _from_scalars(cls, scalars, *, dtype: DtypeObj) -> Self:
        """
        Strict analogue to _from_sequence, allowing only sequences of scalars
        that should be specifically inferred to the given dtype.

        Parameters
        ----------
        scalars : sequence
            Sequence of scalar values to be converted into an ExtensionArray.
        dtype : ExtensionDtype
            The specific data type to which the scalars should be inferred.

        Raises
        ------
        TypeError or ValueError
            Raised if the sequence of scalars cannot be converted to the
            specified dtype.

        Notes
        -----
        This is called in a try/except block when casting the result of a
        pointwise operation.
        """
        try:
            # Attempt to construct an ExtensionArray from the sequence of scalars
            return cls._from_sequence(scalars, dtype=dtype, copy=False)
        except (ValueError, TypeError):
            # Propagate specific exceptions TypeError or ValueError
            raise
        except Exception:
            # Issue a warning for unexpected exceptions
            warnings.warn(
                "_from_scalars should only raise ValueError or TypeError. "
                "Consider overriding _from_scalars where appropriate.",
                stacklevel=find_stack_level(),
            )
            # Raise the unexpected exception
            raise

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: ExtensionDtype, copy: bool = False
    ) -> Self:
        """
        Construct a new ExtensionArray from a sequence of strings.

        Parameters
        ----------
        strings : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : ExtensionDtype
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray
            A new ExtensionArray containing the data converted from strings.

        See Also
        --------
        api.extensions.ExtensionArray._from_sequence : Construct a new ExtensionArray
            from a sequence of scalars.
        api.extensions.ExtensionArray._from_factorized : Reconstruct an ExtensionArray
            after factorization.
        api.extensions.ExtensionArray._from_scalars : Strict analogue to _from_sequence,
            allowing only sequences of scalars that should be specifically inferred to
            the given dtype.

        Examples
        --------
        >>> pd.arrays.IntegerArray._from_sequence_of_strings(
        ...     ["1", "2", "3"], dtype=pd.Int64Dtype()
        ... )
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        raise AbstractMethodError(cls)

    @classmethod
    def _from_factorized(cls, values, original):
        """
        在因子化后重新构建 ExtensionArray。

        Parameters
        ----------
        values : ndarray
            包含因子化值的整数 ndarray。
        original : ExtensionArray
            调用 factorize 方法时的原始 ExtensionArray。

        See Also
        --------
        factorize : 分派到此处的顶级 factorize 方法。
        ExtensionArray.factorize : 将扩展数组编码为枚举类型。

        Examples
        --------
        >>> interv_arr = pd.arrays.IntervalArray(
        ...     [pd.Interval(0, 1), pd.Interval(1, 5), pd.Interval(1, 5)]
        ... )
        >>> codes, uniques = pd.factorize(interv_arr)
        >>> pd.arrays.IntervalArray._from_factorized(uniques, interv_arr)
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        """
        raise AbstractMethodError(cls)

    # ------------------------------------------------------------------------
    # Must be a Sequence
    # ------------------------------------------------------------------------
    
    @overload
    def __getitem__(self, item: ScalarIndexer) -> Any:
        """
        选择 self 的子集。

        Parameters
        ----------
        item : int
            获取 'self' 中的位置。

        Returns
        -------
        scalar
            适合数组类型的标量值，应该是 self.dtype.type 的实例。

        Notes
        -----
        对于标量 'item'，返回适合数组类型的标量值。

        Raises
        ------
        AbstractMethodError
            如果未在子类中实现，则引发抽象方法错误。
        """
        raise AbstractMethodError(self)

    @overload
    def __getitem__(self, item: SequenceIndexer) -> Self:
        """
        选择 self 的子集。

        Parameters
        ----------
        item : slice
            一个切片对象，其中 'start'、'stop' 和 'step' 都是整数或 None。

        Returns
        -------
        ExtensionArray
            返回一个 ExtensionArray 实例，即使切片的长度为 0 或 1。

        Notes
        -----
        对于切片 'key'，返回一个 ExtensionArray 实例。

        Raises
        ------
        AbstractMethodError
            如果未在子类中实现，则引发抽象方法错误。
        """
        raise AbstractMethodError(self)

    def __getitem__(self, item: PositionalIndexer) -> Self | Any:
        """
        选择 self 的子集。

        Parameters
        ----------
        item : ndarray or list[int]
            * ndarray: 一个与 'self' 长度相同的 1 维布尔 NumPy ndarray。
            * list[int]: 一个整数列表。

        Returns
        -------
        scalar or ExtensionArray
            如果 'item' 是布尔掩码，则返回筛选后的 ExtensionArray，否则返回标量值。

        Notes
        -----
        对于布尔掩码，返回筛选后的 ExtensionArray。

        Raises
        ------
        AbstractMethodError
            如果未在子类中实现，则引发抽象方法错误。
        """
        raise AbstractMethodError(self)
    def __setitem__(self, key, value) -> None:
        """
        Set one or more values inplace.

        This method is not required to satisfy the pandas extension array
        interface.

        Parameters
        ----------
        key : int, ndarray, or slice
            When called from, e.g. ``Series.__setitem__``, ``key`` will be
            one of

            * scalar int
            * ndarray of integers.
            * boolean ndarray
            * slice object

        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object
            value or values to be set of ``key``.

        Returns
        -------
        None
        """
        # 一些针对实现 ExtensionArray 的开发者的注解，他们可能会进入到这里。
        # 虽然这个方法并不是接口所必需的，但如果你*选择*实现 __setitem__，那么应该遵循一些语义：
        #
        # * 设置多个值：ExtensionArrays 应支持一次设置多个值，'key' 将是一个整数序列，'value' 将是一个相同长度的序列。
        #
        # * 广播：对于一个整数序列 'key' 和一个标量 'value'，应该将 'key' 中的每个位置设置为 'value'。
        #
        # * 强制类型转换：大多数用户希望基本的类型转换能够工作。例如，将字符串 '2018-01-01' 转换为 datetime 类型，当设置在一个 datetime64ns 数组上时。通常，如果 __init__ 方法能对该值进行转换，那么 __setitem__ 也应该能做到。
        # 注意，Series/DataFrame.where 在内部使用 __setitem__ 处理数据的副本。
        raise NotImplementedError(f"{type(self)} does not implement __setitem__.")

    def __len__(self) -> int:
        """
        Length of this array

        Returns
        -------
        length : int
        """
        # 返回该数组的长度
        raise AbstractMethodError(self)

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over elements of the array.
        """
        # 需要实现这个方法以便 pandas 将扩展数组识别为类似列表的对象。默认实现通过连续调用 ``__getitem__`` 实现，可能比必要的更慢。
        for i in range(len(self)):
            yield self[i]
    # error: Signature of "__contains__" incompatible with supertype "object"
    def __contains__(self, item: object) -> bool | np.bool_:
        """
        Return for `item in self`.
        """
        # GH37867
        # comparisons of any item to pd.NA always return pd.NA, so e.g. "a" in [pd.NA]
        # would raise a TypeError. The implementation below works around that.
        
        # 检查传入的item是否是标量并且为NA（缺失值）
        if is_scalar(item) and isna(item):
            # 如果数据结构不能容纳NA，则返回False
            if not self._can_hold_na:
                return False
            # 如果item等于数据类型的NA值或者是数据类型的实例，则返回self._hasna
            elif item is self.dtype.na_value or isinstance(item, self.dtype.type):
                return self._hasna
            else:
                return False
        else:
            # error: Item "ExtensionArray" of "Union[ExtensionArray, ndarray]" has no
            # attribute "any"
            # 如果item不是标量NA，则调用~(self == item).any()，即是否存在item在self中
            return (item == self).any()  # type: ignore[union-attr]

    # error: Signature of "__eq__" incompatible with supertype "object"
    def __eq__(self, other: object) -> ArrayLike:  # type: ignore[override]
        """
        Return for `self == other` (element-wise equality).
        """
        # Implementer note: this should return a boolean numpy ndarray or
        # a boolean ExtensionArray.
        # When `other` is one of Series, Index, or DataFrame, this method should
        # return NotImplemented (to ensure that those objects are responsible for
        # first unpacking the arrays, and then dispatch the operation to the
        # underlying arrays)
        
        # 抛出抽象方法错误，因为该方法需要在子类中实现
        raise AbstractMethodError(self)

    # error: Signature of "__ne__" incompatible with supertype "object"
    def __ne__(self, other: object) -> ArrayLike:  # type: ignore[override]
        """
        Return for `self != other` (element-wise in-equality).
        """
        # error: Unsupported operand type for ~ ("ExtensionArray")
        # 返回不等于操作的结果，即~(self == other)
        return ~(self == other)  # type: ignore[operator]

    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        """
        Convert to a NumPy ndarray.

        This is similar to :meth:`numpy.asarray`, but may provide additional control
        over how the conversion is done.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.
        na_value : Any, optional
            The value to use for missing values. The default value depends
            on `dtype` and the type of the array.

        Returns
        -------
        numpy.ndarray
        """
        # Convert the object to a NumPy ndarray with specified dtype
        result = np.asarray(self, dtype=dtype)
        # If copy is True or na_value is not default, ensure a copy of the array
        if copy or na_value is not lib.no_default:
            result = result.copy()
        # If na_value is provided and not default, replace NA values in result array with na_value
        if na_value is not lib.no_default:
            result[self.isna()] = na_value
        return result

    # ------------------------------------------------------------------------
    # Required attributes
    # ------------------------------------------------------------------------

    @property
    def dtype(self) -> ExtensionDtype:
        """
        An instance of ExtensionDtype.

        Examples
        --------
        >>> pd.array([1, 2, 3]).dtype
        Int64Dtype()
        """
        # This property is required to return the ExtensionDtype of the array
        raise AbstractMethodError(self)

    @property
    def shape(self) -> Shape:
        """
        Return a tuple of the array dimensions.

        See Also
        --------
        numpy.ndarray.shape : Similar attribute which returns the shape of an array.
        DataFrame.shape : Return a tuple representing the dimensionality of the
            DataFrame.
        Series.shape : Return a tuple representing the dimensionality of the Series.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.shape
        (3,)
        """
        # Return a tuple representing the dimensionality of the array
        return (len(self),)

    @property
    def size(self) -> int:
        """
        The number of elements in the array.
        """
        # Calculate and return the total number of elements in the array
        return np.prod(self.shape)  # type: ignore[return-value]

    @property
    def ndim(self) -> int:
        """
        Extension Arrays are only allowed to be 1-dimensional.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.ndim
        1
        """
        # Extension Arrays are always 1-dimensional
        return 1

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.

        Examples
        --------
        >>> pd.array([1, 2, 3]).nbytes
        27
        """
        # 如果计算成本较高，返回所需内存字节数的一个近似下界
        raise AbstractMethodError(self)

    # ------------------------------------------------------------------------
    # Additional Methods
    # ------------------------------------------------------------------------

    @overload
    def astype(self, dtype: npt.DTypeLike, copy: bool = ...) -> np.ndarray: ...
    
    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = ...) -> ExtensionArray: ...
    
    @overload
    def astype(self, dtype: AstypeArg, copy: bool = ...) -> ArrayLike: ...
    
    def astype(self, dtype: AstypeArg, copy: bool = True) -> ArrayLike:
        """
        Cast to a NumPy array or ExtensionArray with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        np.ndarray or pandas.api.extensions.ExtensionArray
            An ``ExtensionArray`` if ``dtype`` is ``ExtensionDtype``,
            otherwise a Numpy ndarray with ``dtype`` for its dtype.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64

        Casting to another ``ExtensionDtype`` returns an ``ExtensionArray``:

        >>> arr1 = arr.astype("Float64")
        >>> arr1
        <FloatingArray>
        [1.0, 2.0, 3.0]
        Length: 3, dtype: Float64
        >>> arr1.dtype
        Float64Dtype()

        Otherwise, we will get a Numpy ndarray:

        >>> arr2 = arr.astype("float64")
        >>> arr2
        array([1., 2., 3.])
        >>> arr2.dtype
        dtype('float64')
        """
        dtype = pandas_dtype(dtype)
        # 如果 dtype 与当前对象的数据类型相同
        if dtype == self.dtype:
            # 如果不需要复制数据
            if not copy:
                # 直接返回当前对象
                return self
            else:
                # 否则返回当前对象的副本
                return self.copy()

        # 如果 dtype 是 ExtensionDtype 类型
        if isinstance(dtype, ExtensionDtype):
            # 获取对应的 ExtensionArray 类
            cls = dtype.construct_array_type()
            # 从当前对象序列构建该类型的数组，并返回
            return cls._from_sequence(self, dtype=dtype, copy=copy)

        # 如果 dtype 是日期时间类型
        elif lib.is_np_dtype(dtype, "M"):
            from pandas.core.arrays import DatetimeArray

            # 从当前对象序列构建日期时间数组，并返回
            return DatetimeArray._from_sequence(self, dtype=dtype, copy=copy)

        # 如果 dtype 是时间增量类型
        elif lib.is_np_dtype(dtype, "m"):
            from pandas.core.arrays import TimedeltaArray

            # 从当前对象序列构建时间增量数组，并返回
            return TimedeltaArray._from_sequence(self, dtype=dtype, copy=copy)

        # 如果不需要复制数据
        if not copy:
            # 返回一个以 dtype 类型创建的 NumPy 数组
            return np.asarray(self, dtype=dtype)
        else:
            # 否则返回一个以 dtype 类型创建的 NumPy 数组的副本
            return np.array(self, dtype=dtype, copy=copy)
    def isna(self) -> np.ndarray | ExtensionArraySupportsAnyAll:
        """
        A 1-D array indicating if each value is missing.

        Returns
        -------
        numpy.ndarray or pandas.api.extensions.ExtensionArray
            In most cases, this should return a NumPy ndarray. For
            exceptional cases like ``SparseArray``, where returning
            an ndarray would be expensive, an ExtensionArray may be
            returned.

        Notes
        -----
        If returning an ExtensionArray, then

        * ``na_values._is_boolean`` should be True
        * ``na_values`` should implement :func:`ExtensionArray._reduce`
        * ``na_values`` should implement :func:`ExtensionArray._accumulate`
        * ``na_values.any`` and ``na_values.all`` should be implemented

        Examples
        --------
        >>> arr = pd.array([1, 2, np.nan, np.nan])
        >>> arr.isna()
        array([False, False,  True,  True])
        """
        # 抽象方法，需要在子类中实现；用于检查每个值是否缺失
        raise AbstractMethodError(self)

    @property
    def _hasna(self) -> bool:
        """
        Equivalent to `self.isna().any()`.

        Some ExtensionArray subclasses may be able to optimize this check.
        """
        # 返回是否存在缺失值，等同于 `self.isna().any()`
        return bool(self.isna().any())

    def _values_for_argsort(self) -> np.ndarray:
        """
        Return values for sorting.

        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.

        See Also
        --------
        ExtensionArray.argsort : Return the indices that would sort this array.

        Notes
        -----
        The caller is responsible for *not* modifying these values in-place, so
        it is safe for implementers to give views on ``self``.

        Functions that use this (e.g. ``ExtensionArray.argsort``) should ignore
        entries with missing values in the original array (according to
        ``self.isna()``). This means that the corresponding entries in the returned
        array don't need to be modified to sort correctly.

        Examples
        --------
        In most cases, this is the underlying Numpy array of the ``ExtensionArray``:

        >>> arr = pd.array([1, 2, 3])
        >>> arr._values_for_argsort()
        array([1, 2, 3])
        """
        # 返回用于排序的值数组，通常是 ExtensionArray 的基础 NumPy 数组
        # 注意：此方法在 `ExtensionArray.argsort/argmin/argmax` 中使用
        return np.array(self)

    def argsort(
        self,
        *,
        ascending: bool = True,
        kind: SortKind = "quicksort",
        na_position: str = "last",
        **kwargs,
    ):
        """
        Return the indices that would sort this array.

        Parameters
        ----------
        ascending : bool, default True
            If True, sort in ascending order.
            If False, sort in descending order.
        kind : str, default 'quicksort'
            Choice of sorting algorithm.
        na_position : {'first', 'last'}, default 'last'
            If 'first', NaNs are sorted at the beginning.
            If 'last', NaNs are sorted at the end.
        **kwargs
            Additional keyword arguments passed to the sorting algorithm.

        Returns
        -------
        numpy.ndarray
            Indices that would sort the array.

        See Also
        --------
        ExtensionArray._values_for_argsort : Return the values to be sorted.

        Notes
        -----
        This method should handle NaNs as per the value of `na_position`.

        Examples
        --------
        >>> arr = pd.array([3, 1, np.nan, 2])
        >>> arr.argsort()
        array([1, 3, 0, 2])
        """
        # 返回数组排序后的索引数组
        pass
    ) -> np.ndarray:
        """
        Return the indices that would sort this array.

        Parameters
        ----------
        ascending : bool, default True
            Whether the indices should result in an ascending
            or descending sort.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            Sorting algorithm.
        na_position : {'first', 'last'}, default 'last'
            If ``'first'``, put ``NaN`` values at the beginning.
            If ``'last'``, put ``NaN`` values at the end.
        **kwargs
            Passed through to :func:`numpy.argsort`.

        Returns
        -------
        np.ndarray[np.intp]
            Array of indices that sort ``self``. If NaN values are contained,
            NaN values are placed at the end.

        See Also
        --------
        numpy.argsort : Sorting implementation used internally.

        Examples
        --------
        >>> arr = pd.array([3, 1, 2, 5, 4])
        >>> arr.argsort()
        array([1, 2, 0, 4, 3])
        """
        # Implementer note: You have two places to override the behavior of
        # argsort.
        # 1. _values_for_argsort : construct the values passed to np.argsort
        # 2. argsort : total control over sorting. In case of overriding this,
        #    it is recommended to also override argmax/argmin
        ascending = nv.validate_argsort_with_ascending(ascending, (), kwargs)
        # 获取用于排序的值数组
        values = self._values_for_argsort()
        # 调用 nargsort 函数进行排序，并返回排序后的索引数组
        return nargsort(
            values,
            kind=kind,
            ascending=ascending,
            na_position=na_position,
            mask=np.asarray(self.isna()),
        )

    def argmin(self, skipna: bool = True) -> int:
        """
        Return the index of minimum value.

        In case of multiple occurrences of the minimum value, the index
        corresponding to the first occurrence is returned.

        Parameters
        ----------
        skipna : bool, default True
            Whether to skip NA/null values.

        Returns
        -------
        int
            Index of the minimum value.

        See Also
        --------
        ExtensionArray.argmax : Return the index of the maximum value.

        Examples
        --------
        >>> arr = pd.array([3, 1, 2, 5, 4])
        >>> arr.argmin()
        1
        """
        # Implementer note: You have two places to override the behavior of
        # argmin.
        # 1. _values_for_argsort : construct the values used in nargminmax
        # 2. argmin itself : total control over sorting.
        validate_bool_kwarg(skipna, "skipna")
        # 如果 skipna=False 且数组中包含 NA 值，则抛出 ValueError
        if not skipna and self._hasna:
            raise ValueError("Encountered an NA value with skipna=False")
        # 调用 nargminmax 函数，返回最小值的索引
        return nargminmax(self, "argmin")
    def argmax(self, skipna: bool = True) -> int:
        """
        Return the index of maximum value.

        In case of multiple occurrences of the maximum value, the index
        corresponding to the first occurrence is returned.

        Parameters
        ----------
        skipna : bool, default True
            If True, NA/null values are excluded during computation.

        Returns
        -------
        int
            Index of the maximum value in the array.

        See Also
        --------
        ExtensionArray.argmin : Return the index of the minimum value.

        Examples
        --------
        >>> arr = pd.array([3, 1, 2, 5, 4])
        >>> arr.argmax()
        3
        """
        # Validate the skipna parameter
        validate_bool_kwarg(skipna, "skipna")
        
        # Raise an error if skipna is False and there are NA values in the array
        if not skipna and self._hasna:
            raise ValueError("Encountered an NA value with skipna=False")
        
        # Call nargminmax function to compute the index of the maximum value
        return nargminmax(self, "argmax")

    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        axis: int,
        index: Index,
        limit,
        limit_direction,
        limit_area,
        copy: bool,
        **kwargs,
    ) -> Self:
        """
        See DataFrame.interpolate.__doc__.

        Examples
        --------
        >>> arr = pd.arrays.NumpyExtensionArray(np.array([0, 1, np.nan, 3]))
        >>> arr.interpolate(
        ...     method="linear",
        ...     limit=3,
        ...     limit_direction="forward",
        ...     index=pd.Index([1, 2, 3, 4]),
        ...     fill_value=1,
        ...     copy=False,
        ...     axis=0,
        ...     limit_area="inside",
        ... )
        <NumpyExtensionArray>
        [0.0, 1.0, 2.0, 3.0]
        Length: 4, dtype: float64
        """
        # Not implemented: raise NotImplementedError with a message
        raise NotImplementedError(
            f"{type(self).__name__} does not implement interpolate"
        )

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
        **kwargs,
    ) -> None:
        """
        Placeholder method for padding or backfilling values.

        Parameters
        ----------
        method : FillnaOptions
            Method used for filling missing values.
        limit : int | None, optional
            Maximum number of consecutive NaNs to fill. Default is None.
        limit_area : {"inside", "outside"} | None, optional
            Limiting direction for filling. Default is None.
        copy : bool, optional
            If True, return a copy of the object with filled values. Default is True.

        Notes
        -----
        This method serves as a placeholder and does not perform any actual filling.
        """
        # No implementation: method is currently a placeholder
        pass
    ) -> Self:
        """
        Pad or backfill values, used by Series/DataFrame ffill and bfill.

        Parameters
        ----------
        method : {'backfill', 'bfill', 'pad', 'ffill'}
            Method to use for filling holes in reindexed Series:

            * pad / ffill: propagate last valid observation forward to next valid.
            * backfill / bfill: use NEXT valid observation to fill gap.

        limit : int, default None
            This is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.

        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author's discretion whether to ignore "copy=False" or to raise.
            The base class implementation ignores the keyword if any NAs are
            present.

        Returns
        -------
        Same type as self
            Returns a Series or DataFrame with missing values filled according to
            the specified method and options.

        Examples
        --------
        >>> arr = pd.array([np.nan, np.nan, 2, 3, np.nan, np.nan])
        >>> arr._pad_or_backfill(method="backfill", limit=1)
        <IntegerArray>
        [<NA>, 2, 2, 3, <NA>, <NA>]
        Length: 6, dtype: Int64
        """
        # Create a boolean mask where True indicates NaN values
        mask = self.isna()

        # Check if there are any NaN values in the Series or DataFrame
        if mask.any():
            # Determine the appropriate filling method
            meth = missing.clean_fill_method(method)

            # Convert the mask to a NumPy array
            npmask = np.asarray(mask)

            # Perform limit area filling if specified and mask has non-all NaNs
            if limit_area is not None and not npmask.all():
                _fill_limit_area_1d(npmask, limit_area)

            # Handle the 'pad' or 'ffill' method
            if meth == "pad":
                # Get the indexer to fill NaNs forward
                indexer = libalgos.get_fill_indexer(npmask, limit=limit)
                return self.take(indexer, allow_fill=True)
            else:
                # Handle the 'backfill' or 'bfill' method
                # Reverse the mask and get the indexer to fill NaNs backward
                indexer = libalgos.get_fill_indexer(npmask[::-1], limit=limit)[::-1]
                return self[::-1].take(indexer, allow_fill=True)

        else:
            # If no NaNs are present
            if not copy:
                return self
            # Create a copy of self
            new_values = self.copy()
            return new_values
    ) -> Self:
        """
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like "value" can be given. It's expected
            that the array-like have the same length as 'self'.
        limit : int, default None
            The maximum number of entries where NA values will be filled.
        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author's discretion whether to ignore "copy=False" or to raise.

        Returns
        -------
        ExtensionArray
            With NA/NaN filled.

        Examples
        --------
        >>> arr = pd.array([np.nan, np.nan, 2, 3, np.nan, np.nan])
        >>> arr.fillna(0)
        <IntegerArray>
        [0, 0, 2, 3, 0, 0]
        Length: 6, dtype: Int64
        """
        # 创建一个布尔掩码，标记出缺失值的位置
        mask = self.isna()
        if limit is not None and limit < len(self):
            # 如果指定了限制，并且缺失值的数量超过限制，则执行以下操作
            # isna 可能返回 ExtensionArray，我们假设比较已经实现
            # mypy 不喜欢 mask 可以是 EA，EA 可能没有 `cumsum`
            modify = mask.cumsum() > limit  # type: ignore[union-attr]
            if modify.any():
                # 只在必要时复制掩码
                mask = mask.copy()
                mask[modify] = False
        # 检查并调整 value 的大小，以适应缺失值的掩码
        value = missing.check_value_size(
            value,
            mask,  # type: ignore[arg-type]
            len(self),
        )

        if mask.any():
            # 如果存在缺失值，则用指定的 value 填充这些位置
            if not copy:
                new_values = self[:]
            else:
                new_values = self.copy()
            new_values[mask] = value
        else:
            # 如果没有缺失值，则直接复制 self 的值
            if not copy:
                new_values = self[:]
            else:
                new_values = self.copy()
        return new_values

    def dropna(self) -> Self:
        """
        Return ExtensionArray without NA values.

        Returns
        -------
        ExtensionArray
            Without NA values.

        Examples
        --------
        >>> pd.array([1, 2, np.nan]).dropna()
        <IntegerArray>
        [1, 2]
        Length: 2, dtype: Int64
        """
        # 返回去除了 NA 值的 ExtensionArray
        return self[~self.isna()]  # type: ignore[operator]

    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    def duplicated(self, keep: Union[str, bool] = 'first') -> np.ndarray[np.bool_]:
        """
        返回一个布尔型的 ndarray，表示重复值的位置。

        Parameters
        ----------
        keep : {'first', 'last', False}, 默认为 'first'
            - ``first`` : 将除第一次出现外的重复值标记为 ``True``。
            - ``last`` : 将除最后一次出现外的重复值标记为 ``True``。
            - False : 将所有重复值标记为 ``True``。

        Returns
        -------
        ndarray[bool]
            布尔型的 ndarray，表示是否为重复值的数组。

        Examples
        --------
        >>> pd.array([1, 1, 2, 3, 3], dtype="Int64").duplicated()
        array([False,  True, False, False,  True])
        """
        # 创建一个布尔型的掩码，标记出缺失值位置
        mask = self.isna().astype(np.bool_, copy=False)
        # 调用 duplicated 函数计算重复值的布尔数组，并返回结果
        return duplicated(values=self, keep=keep, mask=mask)

    def shift(self, periods: int = 1, fill_value: object = None) -> ExtensionArray:
        """
        将值按指定数量进行位移。

        新引入的缺失值将使用 ``self.dtype.na_value`` 填充。

        Parameters
        ----------
        periods : int, 默认为 1
            要位移的周期数。允许使用负值向后位移。

        fill_value : object, 可选
            用于新引入的缺失值的标量值。默认为 ``self.dtype.na_value``。

        Returns
        -------
        ExtensionArray
            位移后的数组。

        See Also
        --------
        api.extensions.ExtensionArray.transpose : 返回数组的转置视图。
        api.extensions.ExtensionArray.factorize : 将扩展数组编码为枚举类型。

        Notes
        -----
        如果 ``self`` 是空的或者 ``periods`` 为 0，则返回 ``self`` 的副本。

        如果 ``periods > len(self)``，则返回一个大小为 len(self) 的数组，
        所有值都使用 ``self.dtype.na_value`` 填充。

        对于二维 ExtensionArrays，我们总是沿着 axis=0 进行位移。

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.shift(2)
        <IntegerArray>
        [<NA>, <NA>, 1]
        Length: 3, dtype: Int64
        """
        # 注意：此实现假定 `self.dtype.na_value` 可以用 `self.dtype` 的实例存储。
        if not len(self) or periods == 0:
            return self.copy()

        # 如果 fill_value 是缺失值，则使用 self.dtype.na_value
        if isna(fill_value):
            fill_value = self.dtype.na_value

        # 创建一个填充值为 fill_value 的空数组，长度为 abs(periods) 和 self 长度的最小值
        empty = self._from_sequence(
            [fill_value] * min(abs(periods), len(self)), dtype=self.dtype
        )

        if periods > 0:
            a = empty
            b = self[:-periods]  # 取 self 的倒数 periods 个元素
        else:
            a = self[abs(periods) :]  # 取 self 的绝对值 periods 开始到末尾的元素
            b = empty

        # 合并 a 和 b，并返回同类型的数组
        return self._concat_same_type([a, b])
    def unique(self) -> Self:
        """
        计算唯一值的 ExtensionArray。

        Returns
        -------
        pandas.api.extensions.ExtensionArray
            包含唯一值的 ExtensionArray 对象。

        Examples
        --------
        >>> arr = pd.array([1, 2, 3, 1, 2, 3])
        >>> arr.unique()
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        # 调用 unique 函数计算当前数组的唯一值
        uniques = unique(self.astype(object))
        # 使用 _from_sequence 方法创建新的 ExtensionArray 对象，包含唯一值
        return self._from_sequence(uniques, dtype=self.dtype)

    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        """
        查找应插入元素以保持顺序的索引。

        Parameters
        ----------
        value : array-like, list or scalar
            要插入到当前数组中的值。
        side : {'left', 'right'}, optional
            如果是 'left'，返回第一个适合的位置索引。
            如果是 'right'，返回最后一个适合的位置索引。
            如果没有适合的索引，则返回 0 或 N（其中 N 是当前数组的长度）。
        sorter : 1-D array-like, optional
            可选的整数索引数组，用于将数组按升序排序。通常是 argsort 的结果。

        Returns
        -------
        array of ints or int
            如果 value 是 array-like，则返回插入点数组。
            如果 value 是标量，则返回单个整数。

        See Also
        --------
        numpy.searchsorted : NumPy 中类似的方法。

        Examples
        --------
        >>> arr = pd.array([1, 2, 3, 5])
        >>> arr.searchsorted([4])
        array([3])
        """
        # 注意：pandas 提供的基本测试仅测试基本功能。
        # 我们不测试：
        # 1. 超出 data_for_sorting 范围的值
        # 2. 在 data_for_sorting 范围内值之间的值
        # 3. 缺失值。
        # 将数组转换为 object 类型，以便进行搜索排序
        arr = self.astype(object)
        # 如果 value 是 ExtensionArray，则也将其转换为 object 类型
        if isinstance(value, ExtensionArray):
            value = value.astype(object)
        # 调用数组的 searchsorted 方法进行搜索排序
        return arr.searchsorted(value, side=side, sorter=sorter)
    def equals(self, other: object) -> bool:
        """
        Return if another array is equivalent to this array.

        Equivalent means that both arrays have the same shape and dtype, and
        all values compare equal. Missing values in the same location are
        considered equal (in contrast with normal equality).

        Parameters
        ----------
        other : ExtensionArray
            Array to compare to this Array.

        Returns
        -------
        boolean
            Whether the arrays are equivalent.

        See Also
        --------
        numpy.array_equal : Equivalent method for numpy array.
        Series.equals : Equivalent method for Series.
        DataFrame.equals : Equivalent method for DataFrame.

        Examples
        --------
        >>> arr1 = pd.array([1, 2, np.nan])
        >>> arr2 = pd.array([1, 2, np.nan])
        >>> arr1.equals(arr2)
        True

        >>> arr1 = pd.array([1, 3, np.nan])
        >>> arr2 = pd.array([1, 2, np.nan])
        >>> arr1.equals(arr2)
        False
        """
        # 检查两个对象的类型是否相同
        if type(self) != type(other):
            return False
        # 将 `other` 强制转换为 ExtensionArray 类型
        other = cast(ExtensionArray, other)
        # 检查两个数组的数据类型是否相同
        if self.dtype != other.dtype:
            return False
        # 检查两个数组的长度是否相同
        elif len(self) != len(other):
            return False
        else:
            # 检查所有元素是否逐一相等，处理可能存在的缺失值情况
            equal_values = self == other
            if isinstance(equal_values, ExtensionArray):
                # 对于包含缺失值的布尔数组，将缺失值填充为 False
                equal_values = equal_values.fillna(False)
            # 检查缺失值的情况
            # 错误：不支持的左操作数类型（"ExtensionArray"）
            equal_na = self.isna() & other.isna()  # type: ignore[operator]
            # 返回最终是否全部相等的布尔值
            return bool((equal_values | equal_na).all())

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        """
        Pointwise comparison for set containment in the given values.

        Roughly equivalent to `np.array([x in values for x in self])`

        Parameters
        ----------
        values : np.ndarray or ExtensionArray

        Returns
        -------
        np.ndarray[bool]

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.isin([1])
        <BooleanArray>
        [True, False, False]
        Length: 3, dtype: boolean
        """
        # 使用 `isin` 函数检查每个元素是否在给定的 `values` 中
        return isin(np.asarray(self), values)
    # 定义一个方法 `_values_for_factorize`，返回一个包含数组和缺失值的元组，用于因子化操作
    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        """
        Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray
            An array suitable for factorization. This should maintain order
            and be a supported dtype (Float64, Int64, UInt64, String, Object).
            By default, the extension array is cast to object dtype.
        na_value : object
            The value in `values` to consider missing. This will be treated
            as NA in the factorization routines, so it will be coded as
            `-1` and not included in `uniques`. By default,
            ``np.nan`` is used.

        Notes
        -----
        The values returned by this method are also used in
        :func:`pandas.util.hash_pandas_object`. If needed, this can be
        overridden in the ``self._hash_pandas_object()`` method.

        Examples
        --------
        >>> pd.array([1, 2, 3])._values_for_factorize()
        (array([1, 2, 3], dtype=object), nan)
        """
        # 返回调用对象的 object 类型的副本和缺失值为 np.nan 的元组
        return self.astype(object), np.nan

    # 定义一个方法 `factorize`，用于对数据进行因子化处理
    def factorize(
        self,
        use_na_sentinel: bool = True,
        # 参数 use_na_sentinel: 是否使用 NA 哨兵值，默认为 True
        na_sentinel: int = -1
        # 参数 na_sentinel: 用于表示缺失值的哨兵值，默认为 -1
    ):
    ) -> tuple[np.ndarray, ExtensionArray]:
        """
        Encode the extension array as an enumerated type.

        Parameters
        ----------
        use_na_sentinel : bool, default True
            If True, the sentinel -1 will be used for NaN values. If False,
            NaN values will be encoded as non-negative integers and will not drop the
            NaN from the uniques of the values.

            .. versionadded:: 1.5.0

        Returns
        -------
        codes : ndarray
            An integer NumPy array that's an indexer into the original
            ExtensionArray.
        uniques : ExtensionArray
            An ExtensionArray containing the unique values of `self`.

            .. note::

               uniques will *not* contain an entry for the NA value of
               the ExtensionArray if there are any missing values present
               in `self`.

        See Also
        --------
        factorize : Top-level factorize method that dispatches here.

        Notes
        -----
        :meth:`pandas.factorize` offers a `sort` keyword as well.

        Examples
        --------
        >>> idx1 = pd.PeriodIndex(
        ...     ["2014-01", "2014-01", "2014-02", "2014-02", "2014-03", "2014-03"],
        ...     freq="M",
        ... )
        >>> arr, idx = idx1.factorize()
        >>> arr
        array([0, 0, 1, 1, 2, 2])
        >>> idx
        PeriodIndex(['2014-01', '2014-02', '2014-03'], dtype='period[M]')
        """
        # Implementer note: There are two ways to override the behavior of
        # pandas.factorize
        # 1. _values_for_factorize and _from_factorize.
        #    Specify the values passed to pandas' internal factorization
        #    routines, and how to convert from those values back to the
        #    original ExtensionArray.
        # 2. ExtensionArray.factorize.
        #    Complete control over factorization.

        # 调用 _values_for_factorize 方法获取用于因子化的值数组和 NA 值
        arr, na_value = self._values_for_factorize()

        # 调用 factorize_array 函数进行因子化处理，返回编码数组和唯一值数组
        codes, uniques = factorize_array(
            arr, use_na_sentinel=use_na_sentinel, na_value=na_value
        )

        # 调用 _from_factorized 方法将唯一值数组转换为 ExtensionArray
        uniques_ea = self._from_factorized(uniques, self)
        
        # 返回编码数组和转换后的唯一值 ExtensionArray
        return codes, uniques_ea
    # 将 "repeat" 方法的文档字符串添加到 _extension_array_shared_docs 字典中
    _extension_array_shared_docs["repeat"] = """
        Repeat elements of a %(klass)s.

        Returns a new %(klass)s where each element of the current %(klass)s
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            %(klass)s.
        axis : None
            Must be ``None``. Has no effect but is accepted for compatibility
            with numpy.

        Returns
        -------
        %(klass)s
            Newly created %(klass)s with repeated elements.

        See Also
        --------
        Series.repeat : Equivalent function for Series.
        Index.repeat : Equivalent function for Index.
        numpy.repeat : Similar method for :class:`numpy.ndarray`.
        ExtensionArray.take : Take arbitrary positions.

        Examples
        --------
        >>> cat = pd.Categorical(['a', 'b', 'c'])
        >>> cat
        ['a', 'b', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> cat.repeat(2)
        ['a', 'a', 'b', 'b', 'c', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> cat.repeat([1, 2, 3])
        ['a', 'b', 'b', 'c', 'c', 'c']
        Categories (3, object): ['a', 'b', 'c']
        """

    # 使用 @Appender 装饰器将 "repeat" 方法的文档字符串添加到当前方法的文档字符串末尾
    @Substitution(klass="ExtensionArray")
    @Appender(_extension_array_shared_docs["repeat"])
    # 定义 repeat 方法，用于重复数组元素
    def repeat(self, repeats: int | Sequence[int], axis: AxisInt | None = None) -> Self:
        # 使用 nv 模块验证重复操作的有效性
        nv.validate_repeat((), {"axis": axis})
        # 根据 repeats 参数生成索引数组 ind，重复元素的索引
        ind = np.arange(len(self)).repeat(repeats)
        # 返回当前数组中按照生成的索引取得的元素组成的新数组
        return self.take(ind)

    # ------------------------------------------------------------------------
    # Indexing methods
    # ------------------------------------------------------------------------

    # 定义 copy 方法，用于创建当前数组的副本
    def copy(self) -> Self:
        """
        Return a copy of the array.

        This method creates a copy of the `ExtensionArray` where modifying the
        data in the copy will not affect the original array. This is useful when
        you want to manipulate data without altering the original dataset.

        Returns
        -------
        ExtensionArray
            A new `ExtensionArray` object that is a copy of the current instance.

        See Also
        --------
        DataFrame.copy : Return a copy of the DataFrame.
        Series.copy : Return a copy of the Series.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr2 = arr.copy()
        >>> arr[0] = 2
        >>> arr2
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        # 抛出抽象方法错误，强制子类实现具体的复制逻辑
        raise AbstractMethodError(self)
    # 返回当前数组的视图。
    # 如果指定了 dtype 参数，则抛出 NotImplementedError。
    # 返回一个新的对象，引用相同的数据，而不是返回 self。
    def view(self, dtype: Dtype | None = None) -> ArrayLike:
        """
        Return a view on the array.

        Parameters
        ----------
        dtype : str, np.dtype, or ExtensionDtype, optional
            Default None.

        Returns
        -------
        ExtensionArray or np.ndarray
            A view on the :class:`ExtensionArray`'s data.

        Examples
        --------
        This gives view on the underlying data of an ``ExtensionArray`` and is not a
        copy. Modifications on either the view or the original ``ExtensionArray``
        will be reflectd on the underlying data:

        >>> arr = pd.array([1, 2, 3])
        >>> arr2 = arr.view()
        >>> arr[0] = 2
        >>> arr2
        <IntegerArray>
        [2, 2, 3]
        Length: 3, dtype: Int64
        """
        if dtype is not None:
            # 抛出未实现错误，如果指定了 dtype 参数
            raise NotImplementedError(dtype)
        # 返回当前对象的切片，即返回当前数组的视图
        return self[:]

    # ------------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.ndim > 1:
            # 如果数组维度大于 1，则调用 _repr_2d 方法生成表示多维数组的字符串
            return self._repr_2d()

        from pandas.io.formats.printing import format_object_summary

        # 使用 format_object_summary 格式化对象的摘要信息，去除末尾可能的逗号和换行符
        data = format_object_summary(
            self, self._formatter(), indent_for_name=False
        ).rstrip(", \n")
        # 构建类名字符串
        class_name = f"<{type(self).__name__}>\n"
        # 调用 _get_repr_footer 方法获取对象的尾部信息
        footer = self._get_repr_footer()
        # 返回格式化后的对象字符串表示，包括类名、数据和尾部信息
        return f"{class_name}{data}\n{footer}"

    def _get_repr_footer(self) -> str:
        # 如果数组维度大于 1，则返回包含形状和数据类型的字符串
        # 否则返回包含长度和数据类型的字符串
        if self.ndim > 1:
            return f"Shape: {self.shape}, dtype: {self.dtype}"
        return f"Length: {len(self)}, dtype: {self.dtype}"

    def _repr_2d(self) -> str:
        from pandas.io.formats.printing import format_object_summary

        # 使用 format_object_summary 格式化每个元素的摘要信息，去除末尾可能的逗号和换行符
        lines = [
            format_object_summary(x, self._formatter(), indent_for_name=False).rstrip(
                ", \n"
            )
            for x in self
        ]
        # 将格式化后的数据行连接为一个字符串
        data = ",\n".join(lines)
        # 构建类名字符串
        class_name = f"<{type(self).__name__}>"
        # 调用 _get_repr_footer 方法获取对象的尾部信息
        footer = self._get_repr_footer()
        # 返回格式化后的多维数组字符串表示，包括类名、数据和尾部信息
        return f"{class_name}\n[\n{data}\n]\n{footer}"
    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        """
        Formatting function for scalar values.

        This method defines a function that formats scalar values when they are converted to strings,
        particularly when used in the default '__repr__' method of the class.

        Parameters
        ----------
        boxed : bool, default False
            Indicates whether the scalar values are being formatted within a structured data type
            like Series, DataFrame, or Index (True), or independently (False). This can affect how
            scalar values are represented (e.g., quoted or not).

        Returns
        -------
        Callable[[Any], str]
            A callable function that takes instances of the scalar type and returns a formatted string.
            When `boxed=False`, it uses :func:`repr`; when `boxed=True`, it uses :func:`str`.

        See Also
        --------
        api.extensions.ExtensionArray._concat_same_type : Concatenate multiple arrays of this dtype.
        api.extensions.ExtensionArray._explode : Transform each element of list-like to a row.
        api.extensions.ExtensionArray._from_factorized : Reconstruct an ExtensionArray after factorization.
        api.extensions.ExtensionArray._from_sequence : Construct a new ExtensionArray from a sequence of scalars.

        Examples
        --------
        >>> class MyExtensionArray(pd.arrays.NumpyExtensionArray):
        ...     def _formatter(self, boxed=False):
        ...         return lambda x: "*" + str(x) + "*" if boxed else repr(x) + "*"
        >>> MyExtensionArray(np.array([1, 2, 3, 4]))
        <MyExtensionArray>
        [1*, 2*, 3*, 4*]
        Length: 4, dtype: int64
        """

        if boxed:
            # If `boxed` is True, return the `str` function
            return str
        else:
            # If `boxed` is False, return the `repr` function
            return repr

    # ------------------------------------------------------------------------
    # Reshaping
    # ------------------------------------------------------------------------

    def transpose(self, *axes: int) -> Self:
        """
        Return a transposed view on this array.

        Because ExtensionArrays are always 1D, this method is a no-op. It is included
        for compatibility with np.ndarray.

        Returns
        -------
        ExtensionArray
            Returns the current instance of ExtensionArray unchanged.

        Examples
        --------
        >>> pd.array([1, 2, 3]).transpose()
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        return self[:]  # Return a shallow copy of the current instance

    @property
    def T(self) -> Self:
        """
        Return a transposed view on this array.

        This property is equivalent to calling the `transpose()` method.

        Returns
        -------
        ExtensionArray
            Returns the current instance of ExtensionArray unchanged.
        """
        return self.transpose()
    def ravel(self, order: Literal["C", "F", "A", "K"] | None = "C") -> Self:
        """
        Return a flattened view on this array.

        Parameters
        ----------
        order : {None, 'C', 'F', 'A', 'K'}, default 'C'
            Specifies the order of flattening; ignored because ExtensionArrays are 1D-only.

        Returns
        -------
        ExtensionArray
            Returns self unchanged.

        Notes
        -----
        - Because ExtensionArrays are 1D-only, this method does not alter the array.
        - The "order" argument is ignored, used only for compatibility with NumPy.

        Examples
        --------
        >>> pd.array([1, 2, 3]).ravel()
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        return self

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:
        """
        Concatenate multiple arrays of this dtype.

        Parameters
        ----------
        to_concat : sequence of this type
            An array of the same dtype to concatenate.

        Returns
        -------
        ExtensionArray
            Concatenated array of the same dtype.

        See Also
        --------
        api.extensions.ExtensionArray._explode : Transform each element of
            list-like to a row.
        api.extensions.ExtensionArray._formatter : Formatting function for
            scalar values.
        api.extensions.ExtensionArray._from_factorized : Reconstruct an
            ExtensionArray after factorization.

        Examples
        --------
        >>> arr1 = pd.array([1, 2, 3])
        >>> arr2 = pd.array([4, 5, 6])
        >>> pd.arrays.IntegerArray._concat_same_type([arr1, arr2])
        <IntegerArray>
        [1, 2, 3, 4, 5, 6]
        Length: 6, dtype: Int64
        """
        # Implementer note: this method will only be called with a sequence of
        # ExtensionArrays of this class and with the same dtype as self. This
        # should allow "easy" concatenation (no upcasting needed), and result
        # in a new ExtensionArray of the same dtype.
        # Note: this strict behaviour is only guaranteed starting with pandas 1.1
        raise AbstractMethodError(cls)

    # The _can_hold_na attribute is set to True so that pandas internals
    # will use the ExtensionDtype.na_value as the NA value in operations
    # such as take(), reindex(), shift(), etc.  In addition, those results
    # will then be of the ExtensionArray subclass rather than an array
    # of objects
    @cache_readonly
    def _can_hold_na(self) -> bool:
        """
        Determine if NA values can be represented in this ExtensionArray.

        Returns
        -------
        bool
            True if NA values can be represented, False otherwise.
        """
        return self.dtype._can_hold_na

    def _accumulate(
        self, name: str, *, skipna: bool = True, **kwargs
    ):
        """
        Accumulate values along a given axis.

        Parameters
        ----------
        name : str
            The name of the accumulation function.
        skipna : bool, default True
            Whether to exclude NA/null values when accumulating.
        **kwargs
            Additional keyword arguments passed to the accumulation function.

        Notes
        -----
        This method is likely intended to perform cumulative operations
        on the ExtensionArray, accumulating values based on the specified
        function and arguments.
        """
        # Method definition is incomplete and should be continued in the actual code.
    def _accumulate(
        self, name: str, skipna: bool = True, **kwargs
    ) -> ExtensionArray:
        """
        Return an ExtensionArray performing an accumulation operation.

        The underlying data type might change.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            - cummin
            - cummax
            - cumsum
            - cumprod
        skipna : bool, default True
            If True, skip NA values.
        **kwargs
            Additional keyword arguments passed to the accumulation function.
            Currently, there is no supported kwarg.

        Returns
        -------
        array
            An array performing the accumulation operation.

        Raises
        ------
        NotImplementedError : subclass does not define accumulations

        See Also
        --------
        api.extensions.ExtensionArray._concat_same_type : Concatenate multiple
            array of this dtype.
        api.extensions.ExtensionArray.view : Return a view on the array.
        api.extensions.ExtensionArray._explode : Transform each element of
            list-like to a row.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr._accumulate(name="cumsum")
        <IntegerArray>
        [1, 3, 6]
        Length: 3, dtype: Int64
        """
        raise NotImplementedError(f"cannot perform {name} with type {self.dtype}")

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        """
        Return a scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        keepdims : bool, default False
            If False, a scalar is returned.
            If True, the result has dimension with size one along the reduced axis.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        scalar

        Raises
        ------
        TypeError : subclass does not define operations

        Examples
        --------
        >>> pd.array([1, 2, 3])._reduce("min")
        1
        """
        meth = getattr(self, name, None)
        # 如果找不到对应的方法，则抛出类型错误
        if meth is None:
            raise TypeError(
                f"'{type(self).__name__}' with dtype {self.dtype} "
                f"does not support operation '{name}'"
            )
        # 调用找到的方法进行操作，传递 skipna 和其他关键字参数
        result = meth(skipna=skipna, **kwargs)
        # 如果 keepdims 为 True，则将结果转换为长度为一的数组
        if keepdims:
            result = np.array([result])

        return result

    # https://github.com/python/typeshed/issues/2148#issuecomment-520783318
    # Incompatible types in assignment (expression has type "None", base class
    # "object" defined the type as "Callable[[object], int]")
    # 定义了类型为"Callable[[object], int]"的__hash__属性，标记为类型检查忽略赋值

    # ------------------------------------------------------------------------
    # Non-Optimized Default Methods; in the case of the private methods here,
    # these are not guaranteed to be stable across pandas versions.
    # ------------------------------------------------------------------------

    def _values_for_json(self) -> np.ndarray:
        """
        Specify how to render our entries in to_json.

        Notes
        -----
        The dtype on the returned ndarray is not restricted, but for non-native
        types that are not specifically handled in objToJSON.c, to_json is
        liable to raise. In these cases, it may be safer to return an ndarray
        of strings.
        """
        # 返回包含当前对象条目的 np.ndarray，用于序列化为 JSON

        return np.asarray(self)

    def _hash_pandas_object(
        self, *, encoding: str, hash_key: str, categorize: bool
    ) -> npt.NDArray[np.uint64]:
        """
        Hook for hash_pandas_object.

        Default is to use the values returned by _values_for_factorize.

        Parameters
        ----------
        encoding : str
            Encoding for data & key when strings.
        hash_key : str
            Hash_key for string key to encode.
        categorize : bool
            Whether to first categorize object arrays before hashing. This is more
            efficient when the array contains duplicate values.

        Returns
        -------
        np.ndarray[uint64]
            An array of hashed values.

        See Also
        --------
        api.extensions.ExtensionArray._values_for_factorize : Return an array and
            missing value suitable for factorization.
        util.hash_array : Given a 1d array, return an array of hashed values.

        Examples
        --------
        >>> pd.array([1, 2])._hash_pandas_object(
        ...     encoding="utf-8", hash_key="1000000000000000", categorize=False
        ... )
        array([ 6238072747940578789, 15839785061582574730], dtype=uint64)
        """
        # 用于 hash_pandas_object 的钩子方法

        from pandas.core.util.hashing import hash_array

        # 获取用于因子化的值
        values, _ = self._values_for_factorize()
        # 返回经哈希处理后的值数组
        return hash_array(
            values, encoding=encoding, hash_key=hash_key, categorize=categorize
        )
    def _explode(self) -> tuple[Self, npt.NDArray[np.uint64]]:
        """
        Transform each element of list-like to a row.

        For arrays that do not contain list-like elements the default
        implementation of this method just returns a copy and an array
        of ones (unchanged index).

        Returns
        -------
        ExtensionArray
            Array with the exploded values.
        np.ndarray[uint64]
            The original lengths of each list-like for determining the
            resulting index.

        See Also
        --------
        Series.explode : The method on the ``Series`` object that this
            extension array method is meant to support.

        Examples
        --------
        >>> import pyarrow as pa
        >>> a = pd.array(
        ...     [[1, 2, 3], [4], [5, 6]], dtype=pd.ArrowDtype(pa.list_(pa.int64()))
        ... )
        >>> a._explode()
        (<ArrowExtensionArray>
        [1, 2, 3, 4, 5, 6]
        Length: 6, dtype: int64[pyarrow], array([3, 1, 2], dtype=int32))
        """
        # 复制当前对象，以便进行操作
        values = self.copy()
        # 创建一个全为1的数组，长度等于当前对象的长度，用于表示每个列表元素的长度
        counts = np.ones(shape=(len(self),), dtype=np.uint64)
        # 返回处理后的对象和长度数组
        return values, counts

    def tolist(self) -> list:
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        list

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.tolist()
        [1, 2, 3]
        """
        # 如果当前对象的维度大于1，则将每个元素转换成列表形式
        if self.ndim > 1:
            return [x.tolist() for x in self]
        # 否则，直接将当前对象转换成列表
        return list(self)

    def delete(self, loc: PositionalIndexer) -> Self:
        # 根据位置索引 loc 删除元素
        indexer = np.delete(np.arange(len(self)), loc)
        return self.take(indexer)

    def insert(self, loc: int, item) -> Self:
        """
        Insert an item at the given position.

        Parameters
        ----------
        loc : int
            插入位置的索引
        item : scalar-like
            要插入的元素

        Returns
        -------
        same type as self
            与当前对象相同类型的对象

        Notes
        -----
        This method should be both type and dtype-preserving.  If the item
        cannot be held in an array of this type/dtype, either ValueError or
        TypeError should be raised.

        The default implementation relies on _from_sequence to raise on invalid
        items.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.insert(2, -1)
        <IntegerArray>
        [1, 2, -1, 3]
        Length: 4, dtype: Int64
        """
        # 验证插入位置是否有效
        loc = validate_insert_loc(loc, len(self))
        # 根据要插入的元素创建一个数组
        item_arr = type(self)._from_sequence([item], dtype=self.dtype)
        # 使用当前对象类型的方法连接原对象的前半部分、新元素和后半部分，形成新的对象
        return type(self)._concat_same_type([self[:loc], item_arr, self[loc:]])
    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        """
        Analogue to np.putmask(self, mask, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
            Boolean mask array indicating which elements to modify.
        value : scalar or listlike
            Value to place into self where mask is True. If listlike, it must
            be arraylike with the same length as self.

        Returns
        -------
        None

        Notes
        -----
        Unlike np.putmask, this method does not repeat listlike values with
        mismatched length. 'value' should either be a scalar or an arraylike
        with the same length as self.
        """
        # Check if 'value' is listlike; if so, select elements where 'mask' is True
        if is_list_like(value):
            val = value[mask]
        else:
            val = value

        # Update self with 'val' where 'mask' is True
        self[mask] = val

    def _where(self, mask: npt.NDArray[np.bool_], value) -> Self:
        """
        Analogue to np.where(mask, self, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
            Boolean mask array indicating elements to consider for replacement.
        value : scalar or listlike
            Value to place into self where mask is False. If listlike, it must
            be arraylike with the same length as self.

        Returns
        -------
        Self
            Returns a new instance of the same type as self with modified values.
        """
        # Create a copy of self
        result = self.copy()

        # Check if 'value' is listlike; if so, select elements where 'mask' is False
        if is_list_like(value):
            val = value[~mask]
        else:
            val = value

        # Update 'result' with 'val' where 'mask' is False
        result[~mask] = val
        return result

    def _rank(
        self,
        *,
        axis: AxisInt = 0,
        method: str = "average",
        na_option: str = "keep",
        ascending: bool = True,
        pct: bool = False,
    ):
        """
        See Series.rank.__doc__.

        Parameters
        ----------
        axis : AxisInt, default 0
            The axis to rank over.
        method : str, default 'average'
            The method used to assign ranks to tied elements.
        na_option : str, default 'keep'
            How to handle NA values.
        ascending : bool, default True
            Whether or not to sort ranks in ascending order.
        pct : bool, default False
            Whether or not to display ranks as percentages.

        Returns
        -------
        Series
            A Series containing the rank of elements along the specified axis.
        """
        # Only axis=0 is currently supported
        if axis != 0:
            raise NotImplementedError

        # Call rank function from pandas on self with specified parameters
        return rank(
            self,
            axis=axis,
            method=method,
            na_option=na_option,
            ascending=ascending,
            pct=pct,
        )

    @classmethod
    def _empty(cls, shape: Shape, dtype: ExtensionDtype):
        """
        Create an ExtensionArray with the given shape and dtype.

        Parameters
        ----------
        shape : Shape
            Shape of the array to create.
        dtype : ExtensionDtype
            Dtype of the array elements.

        Returns
        -------
        ExtensionArray
            An empty ExtensionArray of the specified shape and dtype.

        Notes
        -----
        This method is used internally by pandas to create empty ExtensionArray
        instances. It should not be called directly unless creating such arrays
        outside of pandas' normal operation.

        See also
        --------
        ExtensionDtype.empty
            ExtensionDtype.empty is the 'official' public version of this API.
        """
        # Create an empty ExtensionArray instance of the specified dtype
        obj = cls._from_sequence([], dtype=dtype)

        # Create a 'taker' array to index into 'obj' with shape 'shape'
        taker = np.broadcast_to(np.intp(-1), shape)

        # Take elements from 'obj' using 'taker', allowing fill if necessary
        result = obj.take(taker, allow_fill=True)

        # Validate the result type and dtype compatibility
        if not isinstance(result, cls) or dtype != result.dtype:
            raise NotImplementedError(
                f"Default 'empty' implementation is invalid for dtype='{dtype}'"
            )

        return result
    def _quantile(self, qs: npt.NDArray[np.float64], interpolation: str) -> Self:
        """
        Compute the quantiles of self for each quantile in `qs`.

        Parameters
        ----------
        qs : np.ndarray[float64]
            Array containing quantiles to compute.
        interpolation: str
            Specifies the interpolation method to use.

        Returns
        -------
        Self
            Returns the same type as self after computing quantiles.
        """
        # Create a boolean mask indicating NaN values in self
        mask = np.asarray(self.isna())
        # Convert self to a numpy array
        arr = np.asarray(self)
        # Define the fill value for NaNs
        fill_value = np.nan

        # Compute quantiles with respect to the mask and input parameters
        res_values = quantile_with_mask(arr, mask, fill_value, qs, interpolation)
        # Return a new instance of the same type as self from computed values
        return type(self)._from_sequence(res_values)

    def _mode(self, dropna: bool = True) -> Self:
        """
        Returns the mode(s) of the ExtensionArray.

        Always returns `ExtensionArray` even if only one value.

        Parameters
        ----------
        dropna : bool, default True
            If True, NA values are ignored when computing mode.

        Returns
        -------
        Self
            Returns the same type as self containing computed mode(s).
        """
        # Compute mode of the ExtensionArray with respect to dropna parameter
        return mode(self, dropna=dropna)  # type: ignore[return-value]

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        """
        Handles numpy ufuncs applied to the ExtensionArray.

        Parameters
        ----------
        ufunc : np.ufunc
            Numpy universal function.
        method : str
            Method of the ufunc to apply.
        inputs : tuple
            Input arguments for the ufunc.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Depends on operation
            Result of applying the ufunc to the ExtensionArray.
        """
        # Check if any input is an instance of ABCSeries, ABCIndex, or ABCDataFrame
        if any(
            isinstance(other, (ABCSeries, ABCIndex, ABCDataFrame)) for other in inputs
        ):
            return NotImplemented

        # Attempt to dispatch ufunc to corresponding dunder method
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        # Handle 'out' keyword argument for ufuncs
        if "out" in kwargs:
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        # Handle reduction ufuncs (e.g., sum, mean)
        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        # Default behavior for ufuncs on ExtensionArray
        return arraylike.default_array_ufunc(self, ufunc, method, *inputs, **kwargs)

    def map(self, mapper, na_action: Literal["ignore"] | None = None):
        """
        Map values using an input mapping or function.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NA values without passing them to mapper.
            If 'ignore' is not supported, raise NotImplementedError.

        Returns
        -------
        Union[ndarray, Index, ExtensionArray]
            Result of applying the mapping function to the array.
            Returns a MultiIndex if the function returns a tuple with more than one element.
        """
        # Call the map_array function to perform the mapping operation
        return map_array(self, mapper, na_action=na_action)

    # ------------------------------------------------------------------------
    # GroupBy Methods
    def _groupby_op(
        self,
        *,
        how: str,
        has_dropped_na: bool,
        min_count: int,
        ngroups: int,
        ids: npt.NDArray[np.intp],
        **kwargs,
    ) -> ArrayLike:
        """
        Dispatch GroupBy reduction or transformation operation.

        This is an *experimental* API to allow ExtensionArray authors to implement
        reductions and transformations. The API is subject to change.

        Parameters
        ----------
        how : {'any', 'all', 'sum', 'prod', 'min', 'max', 'mean', 'median',
               'median', 'var', 'std', 'sem', 'nth', 'last', 'ohlc',
               'cumprod', 'cumsum', 'cummin', 'cummax', 'rank'}
            Specifies the type of operation to be performed on the grouped data.
        has_dropped_na : bool
            Indicates whether missing values have been dropped from the group.
        min_count : int
            Minimum count of non-NA values required for the operation.
        ngroups : int
            Number of unique groups in the data.
        ids : np.ndarray[np.intp]
            Array where ids[i] gives the group label for self[i].
        **kwargs : operation-specific
            Additional keyword arguments specific to the operation.

        Returns
        -------
        np.ndarray or ExtensionArray
            Result of the group operation.
        """
        from pandas.core.arrays.string_ import StringDtype
        from pandas.core.groupby.ops import WrappedCythonOp

        # Determine the kind of operation based on 'how' parameter
        kind = WrappedCythonOp.get_kind_from_how(how)
        # Create a WrappedCythonOp object to perform the operation
        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)

        # GH#43682
        # Check if the dtype of 'self' is StringDtype
        if isinstance(self.dtype, StringDtype):
            # Convert StringArray to numpy array of objects
            npvalues = self.to_numpy(object, na_value=np.nan)
        else:
            # Raise error if the dtype is not supported
            raise NotImplementedError(
                f"function is not implemented for this dtype: {self.dtype}"
            )

        # Perform the operation using the Cython operation handler
        res_values = op._cython_op_ndim_compat(
            npvalues,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=ids,
            mask=None,
            **kwargs,
        )

        # Return early if the operation is in the cast_blocklist
        if op.how in op.cast_blocklist:
            return res_values

        # If the dtype is StringDtype, construct the result as a StringArray
        if isinstance(self.dtype, StringDtype):
            dtype = self.dtype
            string_array_cls = dtype.construct_array_type()
            return string_array_cls._from_sequence(res_values, dtype=dtype)

        else:
            # Raise error if the dtype is not supported
            raise NotImplementedError
class ExtensionArraySupportsAnyAll(ExtensionArray):
    @overload
    def any(self, *, skipna: Literal[True] = ...) -> bool: ...
    
    @overload
    def any(self, *, skipna: bool) -> bool | NAType: ...

    def any(self, *, skipna: bool = True) -> bool | NAType:
        # 抽象方法，用于在子类中实现。检查是否在数组中至少有一个非缺失值满足条件。
        raise AbstractMethodError(self)

    @overload
    def all(self, *, skipna: Literal[True] = ...) -> bool: ...

    @overload
    def all(self, *, skipna: bool) -> bool | NAType: ...

    def all(self, *, skipna: bool = True) -> bool | NAType:
        # 抽象方法，用于在子类中实现。检查是否数组中所有非缺失值都满足条件。
        raise AbstractMethodError(self)


class ExtensionOpsMixin:
    """
    A base class for linking the operators to their dunder names.

    .. note::

       You may want to set ``__array_priority__`` if you want your
       implementation to be called when involved in binary operations
       with NumPy arrays.
    """

    @classmethod
    def _create_arithmetic_method(cls, op):
        # 抽象方法，用于在子类中实现。创建指定算术操作的方法。
        raise AbstractMethodError(cls)

    @classmethod
    def _add_arithmetic_ops(cls) -> None:
        # 动态地将算术操作符关联到对应的算术方法上。
        setattr(cls, "__add__", cls._create_arithmetic_method(operator.add))
        setattr(cls, "__radd__", cls._create_arithmetic_method(roperator.radd))
        setattr(cls, "__sub__", cls._create_arithmetic_method(operator.sub))
        setattr(cls, "__rsub__", cls._create_arithmetic_method(roperator.rsub))
        setattr(cls, "__mul__", cls._create_arithmetic_method(operator.mul))
        setattr(cls, "__rmul__", cls._create_arithmetic_method(roperator.rmul))
        setattr(cls, "__pow__", cls._create_arithmetic_method(operator.pow))
        setattr(cls, "__rpow__", cls._create_arithmetic_method(roperator.rpow))
        setattr(cls, "__mod__", cls._create_arithmetic_method(operator.mod))
        setattr(cls, "__rmod__", cls._create_arithmetic_method(roperator.rmod))
        setattr(cls, "__floordiv__", cls._create_arithmetic_method(operator.floordiv))
        setattr(cls, "__rfloordiv__", cls._create_arithmetic_method(roperator.rfloordiv))
        setattr(cls, "__truediv__", cls._create_arithmetic_method(operator.truediv))
        setattr(cls, "__rtruediv__", cls._create_arithmetic_method(roperator.rtruediv))
        setattr(cls, "__divmod__", cls._create_arithmetic_method(divmod))
        setattr(cls, "__rdivmod__", cls._create_arithmetic_method(roperator.rdivmod))

    @classmethod
    def _create_comparison_method(cls, op):
        # 抽象方法，用于在子类中实现。创建指定比较操作的方法。
        raise AbstractMethodError(cls)

    @classmethod
    def _add_comparison_ops(cls) -> None:
        # 动态地将比较操作符关联到对应的比较方法上。
        setattr(cls, "__eq__", cls._create_comparison_method(operator.eq))
        setattr(cls, "__ne__", cls._create_comparison_method(operator.ne))
        setattr(cls, "__lt__", cls._create_comparison_method(operator.lt))
        setattr(cls, "__gt__", cls._create_comparison_method(operator.gt))
        setattr(cls, "__le__", cls._create_comparison_method(operator.le))
        setattr(cls, "__ge__", cls._create_comparison_method(operator.ge))
    # 定义一个类方法，用于创建逻辑操作方法，抛出抽象方法错误
    def _create_logical_method(cls, op):
        raise AbstractMethodError(cls)

    # 类方法，用于为当前类添加逻辑操作方法
    @classmethod
    def _add_logical_ops(cls) -> None:
        # 设置类的 "__and__" 属性为 _create_logical_method 方法使用 operator.and_ 创建的逻辑方法
        setattr(cls, "__and__", cls._create_logical_method(operator.and_))
        # 设置类的 "__rand__" 属性为 _create_logical_method 方法使用 roperator.rand_ 创建的逻辑方法
        setattr(cls, "__rand__", cls._create_logical_method(roperator.rand_))
        # 设置类的 "__or__" 属性为 _create_logical_method 方法使用 operator.or_ 创建的逻辑方法
        setattr(cls, "__or__", cls._create_logical_method(operator.or_))
        # 设置类的 "__ror__" 属性为 _create_logical_method 方法使用 roperator.ror_ 创建的逻辑方法
        setattr(cls, "__ror__", cls._create_logical_method(roperator.ror_))
        # 设置类的 "__xor__" 属性为 _create_logical_method 方法使用 operator.xor 创建的逻辑方法
        setattr(cls, "__xor__", cls._create_logical_method(operator.xor))
        # 设置类的 "__rxor__" 属性为 _create_logical_method 方法使用 roperator.rxor 创建的逻辑方法
        setattr(cls, "__rxor__", cls._create_logical_method(roperator.rxor))
# 定义了一个混合类，用于在 ExtensionArray 上定义操作。

class ExtensionScalarOpsMixin(ExtensionOpsMixin):
    """
    A mixin for defining ops on an ExtensionArray.

    It is assumed that the underlying scalar objects have the operators
    already defined.

    Notes
    -----
    If you have defined a subclass MyExtensionArray(ExtensionArray), then
    use MyExtensionArray(ExtensionArray, ExtensionScalarOpsMixin) to
    get the arithmetic operators.  After the definition of MyExtensionArray,
    insert the lines

    MyExtensionArray._add_arithmetic_ops()
    MyExtensionArray._add_comparison_ops()

    to link the operators to your class.

    .. note::

       You may want to set ``__array_priority__`` if you want your
       implementation to be called when involved in binary operations
       with NumPy arrays.
    """

    @classmethod
    @classmethod
    def _create_arithmetic_method(cls, op):
        # 创建一个类方法，用于生成特定操作符的算术方法。
        return cls._create_method(op)

    @classmethod
    def _create_comparison_method(cls, op):
        # 创建一个类方法，用于生成特定操作符的比较方法。
        # 在这里，设置 coerce_to_dtype=False 和 result_dtype=bool 可能是为了
        # 控制比较操作的类型转换和结果类型。
        return cls._create_method(op, coerce_to_dtype=False, result_dtype=bool)


这段代码定义了一个名为 `ExtensionScalarOpsMixin` 的混合类，用于在 `ExtensionArray` 上定义操作方法。其中提到，假设底层的标量对象已经定义了所需的操作符。同时，通过类方法 `_create_arithmetic_method` 和 `_create_comparison_method`，可以生成算术和比较方法，以便在特定的扩展数组类中添加这些操作。
```