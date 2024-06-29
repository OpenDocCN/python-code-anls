# `D:\src\scipysrc\pandas\pandas\core\dtypes\base.py`

```
# 导入未来的语言特性模块，用于支持类型注解中的 Self
from __future__ import annotations

# 引入类型检查相关模块和类型注解
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
    overload,
)

# 引入 numpy 库，并重命名为 np
import numpy as np

# 导入 pandas 库中的一些内部模块
from pandas._libs import missing as libmissing
from pandas._libs.hashtable import object_hash
from pandas._libs.properties import cache_readonly
from pandas.errors import AbstractMethodError

# 导入 pandas 核心数据类型模块
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCSeries,
)

# 如果是类型检查阶段，则进一步导入特定类型
if TYPE_CHECKING:
    from pandas._typing import (
        DtypeObj,
        Self,
        Shape,
        npt,
        type_t,
    )

    from pandas import Index
    from pandas.core.arrays import ExtensionArray

    # 用于同一 ExtensionDtype 进行参数化
    ExtensionDtypeT = TypeVar("ExtensionDtypeT", bound="ExtensionDtype")


class ExtensionDtype:
    """
    自定义数据类型，用于与 ExtensionArray 配对使用。

    参见
    --------
    extensions.register_extension_dtype: 通过类装饰器注册 ExtensionType 到 pandas。
    extensions.ExtensionArray: 自定义一维数组类型的抽象基类。

    注意
    -----
    接口包括子类必须实现的抽象方法：

    * type
    * name
    * construct_array_type

    下列属性和方法影响 dtype 在 pandas 操作中的行为：

    * _is_numeric
    * _is_boolean
    * _get_common_dtype

    `na_value` 类属性可用于设置此类型的默认 NA 值。默认使用 :attr:`numpy.nan`。

    ExtensionDtypes 要求是可哈希的。基类提供了默认实现，依赖于 ``_metadata`` 类属性。
    ``_metadata`` 应该是一个包含定义数据类型的字符串的元组。例如，对于 ``PeriodDtype``，这是 ``freq`` 属性。

    **如果您有一个带参数的 dtype，应该设置 ``_metadata`` 类属性**。

    理想情况下，``_metadata`` 中的属性将与您的 ``ExtensionDtype.__init__`` 的参数匹配（如果有）。
    如果 ``_metadata`` 中的任何属性没有实现标准的 ``__eq__`` 或 ``__hash__``，这里的默认实现将无法工作。

    示例
    --------

    与 Apache Arrow（pyarrow）交互时，可以实现一个 ``__from_arrow__`` 方法：
    此方法接收一个 pyarrow Array 或 ChunkedArray 作为唯一参数，并应返回适当的 pandas ExtensionArray
    用于此 dtype 和传递的值：

    >>> import pyarrow
    >>> from pandas.api.extensions import ExtensionArray
    >>> class ExtensionDtype:
    ...     def __from_arrow__(
    ...         self, array: pyarrow.Array | pyarrow.ChunkedArray
    ...     ) -> ExtensionArray: ...

    由于性能原因，此类不继承自 'abc.ABCMeta'。由接口要求的方法和属性引发
    """
    ``pandas.errors.AbstractMethodError`` and no ``register`` method is
    provided for registering virtual subclasses.
    """

    # _metadata 是一个元组类型的属性，用于存储元数据信息，初始为空元组
    _metadata: tuple[str, ...] = ()

    # 返回对象的字符串表示形式，这里直接返回对象的名称
    def __str__(self) -> str:
        return self.name

    # 判断对象是否等于另一个对象
    def __eq__(self, other: object) -> bool:
        """
        Check whether 'other' is equal to self.

        By default, 'other' is considered equal if either

        * it's a string matching 'self.name'.
        * it's an instance of this type and all of the attributes
          in ``self._metadata`` are equal between `self` and `other`.

        Parameters
        ----------
        other : Any

        Returns
        -------
        bool
        """
        # 如果 'other' 是字符串，尝试从字符串构造出一个新对象
        if isinstance(other, str):
            try:
                other = self.construct_from_string(other)
            except TypeError:
                return False
        # 如果 'other' 是当前类型的实例，比较所有元数据属性是否相等
        if isinstance(other, type(self)):
            return all(
                getattr(self, attr) == getattr(other, attr) for attr in self._metadata
            )
        return False

    # 返回对象的哈希值
    def __hash__(self) -> int:
        # 对于 Python >= 3.10，不同的 NaN 对象有不同的哈希值，
        # 这里使用旧版的哈希函数以避免这个问题
        return object_hash(tuple(getattr(self, attr) for attr in self._metadata))

    # 判断对象是否不等于另一个对象
    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    # 返回该类型的默认 NA 值
    @property
    def na_value(self) -> object:
        """
        Default NA value to use for this type.

        This is used in e.g. ExtensionArray.take. This should be the
        user-facing "boxed" version of the NA value, not the physical NA value
        for storage.  e.g. for JSONArray, this is an empty dictionary.
        """
        return np.nan

    # 返回该类型对应的标量类型
    @property
    def type(self) -> type_t[Any]:
        """
        The scalar type for the array, e.g. ``int``

        It's expected ``ExtensionArray[item]`` returns an instance
        of ``ExtensionDtype.type`` for scalar ``item``, assuming
        that value is valid (not NA). NA values do not need to be
        instances of `type`.
        """
        raise AbstractMethodError(self)

    # 返回该类型的标志符，用于表示转换为 ndarray 时的类型
    @property
    def kind(self) -> str:
        """
        A character code (one of 'biufcmMOSUV'), default 'O'

        This should match the NumPy dtype used when the array is
        converted to an ndarray, which is probably 'O' for object if
        the extension type cannot be represented as a built-in NumPy
        type.

        See Also
        --------
        numpy.dtype.kind
        """
        return "O"

    # 返回该类型的名称
    @property
    def name(self) -> str:
        """
        A string identifying the data type.

        Will be used for display in, e.g. ``Series.dtype``
        """
        raise AbstractMethodError(self)

    @property
    ```
    def names(self) -> list[str] | None:
        """
        Ordered list of field names, or None if there are no fields.

        This is for compatibility with NumPy arrays, and may be removed in the
        future.
        """
        # 返回一个空列表或者 None，用于与 NumPy 数组兼容，如果没有字段则返回 None
        return None

    @classmethod
    def construct_array_type(cls) -> type_t[ExtensionArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        # 抽象方法，应该由子类实现，返回与此数据类型关联的数组类型
        raise AbstractMethodError(cls)

    def empty(self, shape: Shape) -> ExtensionArray:
        """
        Construct an ExtensionArray of this dtype with the given shape.

        Analogous to numpy.empty.

        Parameters
        ----------
        shape : int or tuple[int]

        Returns
        -------
        ExtensionArray
        """
        # 使用给定的形状构造一个指定 dtype 的 ExtensionArray，类似于 numpy.empty
        cls = self.construct_array_type()
        return cls._empty(shape, dtype=self)

    @classmethod
    def construct_from_string(cls, string: str) -> Self:
        r"""
        Construct this type from a string.

        This is useful mainly for data types that accept parameters.
        For example, a period dtype accepts a frequency parameter that
        can be set as ``period[h]`` (where H means hourly frequency).

        By default, in the abstract class, just the name of the type is
        expected. But subclasses can overwrite this method to accept
        parameters.

        Parameters
        ----------
        string : str
            The name of the type, for example ``category``.

        Returns
        -------
        ExtensionDtype
            Instance of the dtype.

        Raises
        ------
        TypeError
            If a class cannot be constructed from this 'string'.

        Examples
        --------
        For extension dtypes with arguments the following may be an
        adequate implementation.

        >>> import re
        >>> @classmethod
        ... def construct_from_string(cls, string):
        ...     pattern = re.compile(r"^my_type\[(?P<arg_name>.+)\]$")
        ...     match = pattern.match(string)
        ...     if match:
        ...         return cls(**match.groupdict())
        ...     else:
        ...         raise TypeError(
        ...             f"Cannot construct a '{cls.__name__}' from '{string}'"
        ...         )
        """
        # 从字符串构造该数据类型的实例。对于接受参数的数据类型特别有用。
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        # 检查字符串是否与类名相符合，若不符则抛出 TypeError
        assert isinstance(cls.name, str), (cls, type(cls.name))
        if string != cls.name:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")
        return cls()

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        Check if we match 'dtype'.

        Parameters
        ----------
        dtype : object
            The object to check.

        Returns
        -------
        bool

        Notes
        -----
        The default implementation is True if

        1. ``cls.construct_from_string(dtype)`` is an instance
           of ``cls``.
        2. ``dtype`` is an object and is an instance of ``cls``
        3. ``dtype`` has a ``dtype`` attribute, and any of the above
           conditions is true for ``dtype.dtype``.
        """
        # 获取dtype对象的dtype属性，若不存在则返回dtype本身
        dtype = getattr(dtype, "dtype", dtype)

        # 如果dtype是ABCSeries、ABCIndex、ABCDataFrame、np.dtype类型的实例之一
        if isinstance(dtype, (ABCSeries, ABCIndex, ABCDataFrame, np.dtype)):
            # 返回False，避免将数据传递给`construct_from_string`，这可能导致numpy给出关于失败的逐元素比较的FutureWarning
            return False
        # 若dtype为None，则返回False
        elif dtype is None:
            return False
        # 若dtype是cls的实例，则返回True
        elif isinstance(dtype, cls):
            return True
        # 若dtype是字符串类型，尝试使用cls.construct_from_string检查，返回结果是否非空
        if isinstance(dtype, str):
            try:
                return cls.construct_from_string(dtype) is not None
            except TypeError:
                return False
        # 其余情况返回False
        return False

    @property
    def _is_numeric(self) -> bool:
        """
        Whether columns with this dtype should be considered numeric.

        By default ExtensionDtypes are assumed to be non-numeric.
        They'll be excluded from operations that exclude non-numeric
        columns, like (groupby) reductions, plotting, etc.
        """
        # 默认返回False，ExtensionDtypes被认为是非数值类型
        return False

    @property
    def _is_boolean(self) -> bool:
        """
        Whether this dtype should be considered boolean.

        By default, ExtensionDtypes are assumed to be non-numeric.
        Setting this to True will affect the behavior of several places,
        e.g.

        * is_bool
        * boolean indexing

        Returns
        -------
        bool
        """
        # 默认返回False，ExtensionDtypes被认为是非布尔类型
        return False
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        """
        Return the common dtype, if one exists.

        Used in `find_common_type` implementation. This is for example used
        to determine the resulting dtype in a concat operation.

        If no common dtype exists, return None (which gives the other dtypes
        the chance to determine a common dtype). If all dtypes in the list
        return None, then the common dtype will be "object" dtype (this means
        it is never needed to return "object" dtype from this method itself).

        Parameters
        ----------
        dtypes : list of dtypes
            The dtypes for which to determine a common dtype. This is a list
            of np.dtype or ExtensionDtype instances.

        Returns
        -------
        Common dtype (np.dtype or ExtensionDtype) or None
        """
        if len(set(dtypes)) == 1:
            # 如果所有的 dtypes 都相同，则返回这个 dtype 本身
            return self
        else:
            # 如果存在不同的 dtypes，则返回 None
            return None

    @property
    def _can_hold_na(self) -> bool:
        """
        Can arrays of this dtype hold NA values?
        """
        return True

    @property
    def _is_immutable(self) -> bool:
        """
        Can arrays with this dtype be modified with __setitem__? If not, return
        True.

        Immutable arrays are expected to raise TypeError on __setitem__ calls.
        """
        return False

    @cache_readonly
    def index_class(self) -> type_t[Index]:
        """
        The Index subclass to return from Index.__new__ when this dtype is
        encountered.
        """
        from pandas import Index

        return Index

    @property
    def _supports_2d(self) -> bool:
        """
        Do ExtensionArrays with this dtype support 2D arrays?

        Historically ExtensionArrays were limited to 1D. By returning True here,
        authors can indicate that their arrays support 2D instances. This can
        improve performance in some cases, particularly operations with `axis=1`.

        Arrays that support 2D values should:

            - implement Array.reshape
            - subclass the Dim2CompatTests in tests.extension.base
            - _concat_same_type should support `axis` keyword
            - _reduce and reductions should support `axis` keyword
        """
        return False

    @property
    def _can_fast_transpose(self) -> bool:
        """
        Is transposing an array with this dtype zero-copy?

        Only relevant for cases where _supports_2d is True.
        """
        return False
class StorageExtensionDtype(ExtensionDtype):
    """ExtensionDtype that may be backed by more than one implementation."""

    name: str  # 数据类型的名称

    # 元数据字段，指定此数据类型的存储方式
    _metadata = ("storage",)

    def __init__(self, storage: str | None = None) -> None:
        self.storage = storage  # 初始化存储方式

    def __repr__(self) -> str:
        return f"{self.name}[{self.storage}]"  # 返回数据类型的字符串表示形式

    def __str__(self) -> str:
        return self.name  # 返回数据类型的名称字符串

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str) and other == self.name:
            return True
        return super().__eq__(other)  # 比较两个数据类型对象是否相等

    def __hash__(self) -> int:
        # 自定义了 __eq__ 方法，因此必须重写 __hash__
        return super().__hash__()  # 计算数据类型对象的哈希值

    @property
    def na_value(self) -> libmissing.NAType:
        return libmissing.NA  # 返回此数据类型的缺失值表示


def register_extension_dtype(cls: type_t[ExtensionDtypeT]) -> type_t[ExtensionDtypeT]:
    """
    Register an ExtensionType with pandas as class decorator.

    This enables operations like ``.astype(name)`` for the name
    of the ExtensionDtype.

    Returns
    -------
    callable
        A class decorator.

    See Also
    --------
    api.extensions.ExtensionDtype : The base class for creating custom pandas
        data types.
    Series : One-dimensional array with axis labels.
    DataFrame : Two-dimensional, size-mutable, potentially heterogeneous
        tabular data.

    Examples
    --------
    >>> from pandas.api.extensions import register_extension_dtype, ExtensionDtype
    >>> @register_extension_dtype
    ... class MyExtensionDtype(ExtensionDtype):
    ...     name = "myextension"
    """
    _registry.register(cls)  # 将自定义数据类型注册到注册表中
    return cls


class Registry:
    """
    Registry for dtype inference.

    The registry allows one to map a string repr of a extension
    dtype to an extension dtype. The string alias can be used in several
    places, including

    * Series and Index constructors
    * :meth:`pandas.array`
    * :meth:`pandas.Series.astype`

    Multiple extension types can be registered.
    These are tried in order.
    """

    def __init__(self) -> None:
        self.dtypes: list[type_t[ExtensionDtype]] = []  # 初始化数据类型列表

    def register(self, dtype: type_t[ExtensionDtype]) -> None:
        """
        Parameters
        ----------
        dtype : ExtensionDtype class
        """
        if not issubclass(dtype, ExtensionDtype):
            raise ValueError("can only register pandas extension dtypes")
        
        self.dtypes.append(dtype)  # 向注册表中注册扩展数据类型

    @overload
    def find(self, dtype: type_t[ExtensionDtypeT]) -> type_t[ExtensionDtypeT]: ...

    @overload
    def find(self, dtype: ExtensionDtypeT) -> ExtensionDtypeT: ...

    @overload
    def find(self, dtype: str) -> ExtensionDtype | None: ...

    @overload
    def find(
        self, dtype: npt.DTypeLike
    ) -> type_t[ExtensionDtype] | ExtensionDtype | None: ...

    def find(
        self, dtype: type_t[ExtensionDtype] | ExtensionDtype | npt.DTypeLike
    ) -> type_t[ExtensionDtype] | ExtensionDtype | None:
        """
        Find and return the ExtensionDtype associated with the given dtype.

        Parameters
        ----------
        dtype : ExtensionDtype type, str, or numpy dtype
            The dtype to find.

        Returns
        -------
        ExtensionDtype or None
            The matching ExtensionDtype or None if not found.
        """
        # 实现根据不同输入类型查找对应的 ExtensionDtype 的功能
    ) -> type_t[ExtensionDtype] | ExtensionDtype | None:
        """
        Parameters
        ----------
        dtype : ExtensionDtype class or instance or str or numpy dtype or python type
        
        Returns
        -------
        return the first matching dtype, otherwise return None
        """
        # 如果 dtype 不是字符串类型
        if not isinstance(dtype, str):
            dtype_type: type_t
            # 如果 dtype 不是类类型，则获取其类型
            if not isinstance(dtype, type):
                dtype_type = type(dtype)
            else:
                dtype_type = dtype
            # 如果 dtype_type 是 ExtensionDtype 的子类
            if issubclass(dtype_type, ExtensionDtype):
                # 需要在这里进行类型转换，因为 mypy 不知道我们已经确认它是 ExtensionDtype 或 type_t[ExtensionDtype]
                return cast("ExtensionDtype | type_t[ExtensionDtype]", dtype)

            # 如果不是 ExtensionDtype 的子类，则返回 None
            return None

        # 如果 dtype 是字符串类型，则在 self.dtypes 中查找匹配的 dtype
        for dtype_type in self.dtypes:
            try:
                # 尝试从字符串构造 dtype_type
                return dtype_type.construct_from_string(dtype)
            except TypeError:
                pass

        # 如果没有找到匹配的 dtype，则返回 None
        return None
# 创建一个名为 _registry 的对象实例，使用 Registry 类的默认构造函数
_registry = Registry()
```