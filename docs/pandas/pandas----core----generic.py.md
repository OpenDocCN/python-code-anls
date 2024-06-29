# `D:\src\scipysrc\pandas\pandas\core\generic.py`

```
# pyright: reportPropertyTypeMismatch=false
# 导入 future 模块，使代码支持类型注解的类型别名
from __future__ import annotations

# 导入 collections 模块，用于支持额外的数据结构
import collections
# 从 copy 模块导入 deepcopy 函数，用于深度复制对象
from copy import deepcopy
# 导入 datetime 模块，并命名为 dt，用于处理日期和时间
import datetime as dt
# 从 functools 模块导入 partial 函数，用于创建偏函数
from functools import partial
# 从 json 模块导入 loads 函数，用于解析 JSON 数据
from json import loads
# 导入 operator 模块，用于操作符相关的函数
import operator
# 导入 pickle 模块，用于序列化和反序列化 Python 对象
import pickle
# 导入 re 模块，用于支持正则表达式操作
import re
# 导入 sys 模块，提供对 Python 解释器的访问
import sys
# 导入 typing 模块，用于类型提示
from typing import (
    TYPE_CHECKING,  # 引入 TYPE_CHECKING 常量，用于类型检查时的条件判断
    Any,  # 任意类型
    ClassVar,  # 类变量
    Literal,  # 字面常量类型
    NoReturn,  # 表示函数不会返回值
    cast,  # 类型转换函数
    final,  # 标记方法为 final，不可被子类重写
    overload,  # 重载函数的装饰器
)

# 导入 warnings 模块，用于警告处理
import warnings

# 导入 numpy 库，并命名为 np，用于数值计算
import numpy as np

# 从 pandas._config 模块导入 config 对象，用于配置 pandas 库
from pandas._config import config

# 从 pandas._libs 模块导入 lib 模块
from pandas._libs import lib
# 从 pandas._libs.lib 模块导入 is_range_indexer 函数，用于检查是否为范围索引
from pandas._libs.lib import is_range_indexer
# 从 pandas._libs.tslibs 模块导入 Period, Timestamp, to_offset 函数等，用于时间序列操作
from pandas._libs.tslibs import (
    Period,
    Timestamp,
    to_offset,
)
# 从 pandas._typing 模块导入各种类型提示
from pandas._typing import (
    AlignJoin,
    AnyArrayLike,
    ArrayLike,
    Axes,
    Axis,
    AxisInt,
    CompressionOptions,
    Concatenate,
    DtypeArg,
    DtypeBackend,
    DtypeObj,
    FilePath,
    FillnaOptions,
    FloatFormatType,
    FormattersType,
    Frequency,
    IgnoreRaise,
    IndexKeyFunc,
    IndexLabel,
    InterpolateOptions,
    IntervalClosedType,
    JSONSerializable,
    Level,
    ListLike,
    Manager,
    NaPosition,
    NDFrameT,
    OpenFileErrors,
    RandomState,
    ReindexMethod,
    Renamer,
    Scalar,
    Self,
    SequenceNotStr,
    SortKind,
    StorageOptions,
    Suffixes,
    T,
    TimeAmbiguous,
    TimedeltaConvertibleTypes,
    TimeNonexistent,
    TimestampConvertibleTypes,
    TimeUnit,
    ValueKeyFunc,
    WriteBuffer,
    WriteExcelBuffer,
    npt,
)
# 从 pandas.compat 模块导入 PYPY 常量，用于兼容性处理
from pandas.compat import PYPY
# 从 pandas.compat._constants 模块导入 REF_COUNT 常量
from pandas.compat._constants import REF_COUNT
# 从 pandas.compat._optional 模块导入 import_optional_dependency 函数
from pandas.compat._optional import import_optional_dependency
# 从 pandas.compat.numpy 模块导入 nv 函数，用于 numpy 兼容性
from pandas.compat.numpy import function as nv
# 从 pandas.errors 模块导入多个异常类
from pandas.errors import (
    AbstractMethodError,
    ChainedAssignmentError,
    InvalidIndexError,
)
# 从 pandas.errors.cow 模块导入 _chained_assignment_method_msg 常量
from pandas.errors.cow import _chained_assignment_method_msg
# 从 pandas.util._decorators 模块导入多个装饰器函数
from pandas.util._decorators import (
    deprecate_kwarg,
    doc,
)
# 从 pandas.util._exceptions 模块导入 find_stack_level 函数
from pandas.util._exceptions import find_stack_level
# 从 pandas.util._validators 模块导入多个验证函数
from pandas.util._validators import (
    check_dtype_backend,
    validate_ascending,
    validate_bool_kwarg,
    validate_inclusive,
)

# 从 pandas.core.dtypes.astype 模块导入 astype_is_view 函数
from pandas.core.dtypes.astype import astype_is_view
# 从 pandas.core.dtypes.common 模块导入多个通用函数
from pandas.core.dtypes.common import (
    ensure_object,
    ensure_platform_int,
    ensure_str,
    is_bool,
    is_bool_dtype,
    is_dict_like,
    is_extension_array_dtype,
    is_list_like,
    is_number,
    is_numeric_dtype,
    is_re_compilable,
    is_scalar,
    pandas_dtype,
)
# 从 pandas.core.dtypes.dtypes 模块导入多个数据类型类
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)
# 从 pandas.core.dtypes.generic 模块导入 ABCDataFrame, ABCSeries 等类
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
# 从 pandas.core.dtypes.inference 模块导入多个推断函数
from pandas.core.dtypes.inference import (
    is_hashable,
    is_nested_list_like,
)
# 从 pandas.core.dtypes.missing 模块导入 isna, notna 函数，用于缺失值检查
from pandas.core.dtypes.missing import (
    isna,
    notna,
)

# 从 pandas.core 模块导入多个子模块
from pandas.core import (
    algorithms as algos,
    arraylike,
    common,
    indexing,
    missing,
    nanops,
    sample,
)
# 从 pandas.core.array_algos.replace 模块导入 should_use_regex 函数
from pandas.core.array_algos.replace import should_use_regex
# 从 pandas.core.arrays 模块导入 ExtensionArray 类
from pandas.core.arrays import ExtensionArray
# 从 pandas.core.base 模块导入 PandasObject 基类
from pandas.core.base import PandasObject
# 从 pandas 库中导入需要的模块和函数
from pandas.core.construction import extract_array
from pandas.core.flags import Flags
from pandas.core.indexes.api import (
    DatetimeIndex,
    Index,
    MultiIndex,
    PeriodIndex,
    default_index,
    ensure_index,
)
from pandas.core.internals import BlockManager
from pandas.core.methods.describe import describe_ndframe
from pandas.core.missing import (
    clean_fill_method,
    clean_reindex_fill_method,
    find_valid_index,
)
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_indexer_indexer
from pandas.core.window import (
    Expanding,
    ExponentialMovingWindow,
    Rolling,
    Window,
)

# 从 pandas 库中导入格式化和打印相关的模块和函数
from pandas.io.formats.format import (
    DataFrameFormatter,
    DataFrameRenderer,
)
from pandas.io.formats.printing import pprint_thing

# 如果是类型检查，导入必要的类型信息
if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import (
        Hashable,
        Iterator,
        Mapping,
        Sequence,
    )
    from pandas._libs.tslibs import BaseOffset
    from pandas._typing import P
    from pandas import (
        DataFrame,
        ExcelWriter,
        HDFStore,
        Series,
    )
    from pandas.core.indexers.objects import BaseIndexer
    from pandas.core.resample import Resampler

# 导入 textwrap 模块
import textwrap

# 将 _shared_docs 复制一份存储到 _shared_docs 变量中
_shared_docs = {**_shared_docs}
# 定义共享文档的关键字参数
_shared_doc_kwargs = {
    "axes": "keywords for axes",  # 关于轴的关键字参数
    "klass": "Series/DataFrame",  # 类型为 Series 或 DataFrame
    "axes_single_arg": "{0 or 'index'} for Series, {0 or 'index', 1 or 'columns'} for DataFrame",  # 单个参数表示轴
    "inplace": """
    inplace : bool, default False
        If True, performs operation inplace and returns None.""",  # inplace 参数说明
    "optional_by": """
        by : str or list of str
            Name or list of names to sort by""",  # 可选的按参数说明
}


class NDFrame(PandasObject, indexing.IndexingMixin):
    """
    N-dimensional analogue of DataFrame. Store multi-dimensional in a
    size-mutable, labeled data structure

    Parameters
    ----------
    data : BlockManager
        数据的块管理器
    axes : list
        轴的列表
    copy : bool, default False
        是否复制数据
    """

    _internal_names: list[str] = [
        "_mgr",
        "_item_cache",
        "_cache",
        "_name",
        "_metadata",
        "_flags",
    ]  # 内部名称列表
    _internal_names_set: set[str] = set(_internal_names)  # 内部名称集合
    _accessors: set[str] = set()  # 访问器集合
    _hidden_attrs: frozenset[str] = frozenset([])  # 隐藏属性集合
    _metadata: list[str] = []  # 元数据列表
    _mgr: Manager  # 数据管理器
    _attrs: dict[Hashable, Any]  # 属性字典
    _typ: str  # 类型标识符

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(self, data: Manager) -> None:
        """
        初始化方法，接受一个数据管理器作为参数
        """
        object.__setattr__(self, "_mgr", data)  # 设置 _mgr 属性为传入的数据管理器
        object.__setattr__(self, "_attrs", {})  # 初始化 _attrs 属性为空字典
        object.__setattr__(self, "_flags", Flags(self, allows_duplicate_labels=True))  # 初始化 _flags 属性为 Flags 对象

    @final
    @classmethod
    def _init_mgr(
        cls,
        mgr: Manager,
        axes: dict[Literal["index", "columns"], Axes | None],
        dtype: DtypeObj | None = None,
        copy: bool = False,
    ) -> Manager:
        """Initialize a manager object with specified axes and optionally copy data.

        Parameters
        ----------
        mgr : Manager
            The manager object to initialize.
        axes : dict
            Dictionary containing axes ('index' or 'columns') and corresponding Axes objects or None.
        dtype : DtypeObj or None, optional
            Data type to coerce the manager to, by default None.
        copy : bool, optional
            If True, create a copy of the manager, by default False.

        Returns
        -------
        Manager
            The initialized manager object.
        """
        # Iterate over axes dictionary
        for a, axe in axes.items():
            # Ensure axe is an index object
            if axe is not None:
                axe = ensure_index(axe)
                # Determine the axis in the block manager and reindex it
                bm_axis = cls._get_block_manager_axis(a)
                mgr = mgr.reindex_axis(axe, axis=bm_axis)

        # Copy manager if explicitly requested
        if copy:
            mgr = mgr.copy()
        # Convert manager's dtype if specified
        if dtype is not None:
            # Check if dtype conversion is necessary to avoid unnecessary copies
            if (
                isinstance(mgr, BlockManager)
                and len(mgr.blocks) == 1
                and mgr.blocks[0].values.dtype == dtype
            ):
                pass
            else:
                mgr = mgr.astype(dtype=dtype)
        return mgr

    @final
    @classmethod
    def _from_mgr(cls, mgr: Manager, axes: list[Index]) -> Self:
        """
        Construct a new object of this type from a Manager object and axes.

        Parameters
        ----------
        mgr : Manager
            The Manager object from which to construct the new object.
            Must have the same ndim as cls.
        axes : list[Index]
            List of axes for the new object.

        Returns
        -------
        Self
            A new object of the current class constructed from the given Manager object and axes.
        """
        # Create a new instance of the class
        obj = cls.__new__(cls)
        # Initialize the new object with the provided Manager
        NDFrame.__init__(obj, mgr)
        return obj

    # ----------------------------------------------------------------------
    # attrs and flags

    @property
    def attrs(self) -> dict[Hashable, Any]:
        """
        Dictionary of global attributes of this dataset.

        .. warning::

           attrs is experimental and may change without warning.

        See Also
        --------
        DataFrame.flags : Global flags applying to this object.

        Notes
        -----
        Many operations that create new datasets will copy ``attrs``. Copies
        are always deep so that changing ``attrs`` will only affect the
        present dataset. ``pandas.concat`` copies ``attrs`` only if all input
        datasets have the same ``attrs``.

        Examples
        --------
        For Series:

        >>> ser = pd.Series([1, 2, 3])
        >>> ser.attrs = {"A": [10, 20, 30]}
        >>> ser.attrs
        {'A': [10, 20, 30]}

        For DataFrame:

        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.attrs = {"A": [10, 20, 30]}
        >>> df.attrs
        {'A': [10, 20, 30]}
        """
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Hashable, Any]) -> None:
        """
        Setter method for setting global attributes of this dataset.

        Parameters
        ----------
        value : Mapping[Hashable, Any]
            Dictionary-like object containing the new attributes.

        Returns
        -------
        None
        """
        self._attrs = dict(value)

    @final
    @property
    # 定义方法 `flags`，用于获取与此 pandas 对象关联的属性标志
    def flags(self) -> Flags:
        """
        Get the properties associated with this pandas object.

        The available flags are

        * :attr:`Flags.allows_duplicate_labels`

        See Also
        --------
        Flags : Flags that apply to pandas objects.
        DataFrame.attrs : Global metadata applying to this dataset.

        Notes
        -----
        "Flags" differ from "metadata". Flags reflect properties of the
        pandas object (the Series or DataFrame). Metadata refer to properties
        of the dataset, and should be stored in :attr:`DataFrame.attrs`.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2]})
        >>> df.flags
        <Flags(allows_duplicate_labels=True)>

        Flags can be get or set using ``.``

        >>> df.flags.allows_duplicate_labels
        True
        >>> df.flags.allows_duplicate_labels = False

        Or by slicing with a key

        >>> df.flags["allows_duplicate_labels"]
        False
        >>> df.flags["allows_duplicate_labels"] = True
        """
        return self._flags

    # 使用 `@final` 装饰器声明方法 `set_flags` 为不可重写的最终方法
    @final
    def set_flags(
        self,
        *,
        copy: bool | lib.NoDefault = lib.no_default,
        allows_duplicate_labels: bool | None = None,
    ):
    ) -> Self:
        """
        Return a new object with updated flags.

        Parameters
        ----------
        copy : bool, default False
            Specify if a copy of the object should be made.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``

            .. deprecated:: 3.0.0
        allows_duplicate_labels : bool, optional
            Whether the returned object allows duplicate labels.

        Returns
        -------
        Series or DataFrame
            The same type as the caller.

        See Also
        --------
        DataFrame.attrs : Global metadata applying to this dataset.
        DataFrame.flags : Global flags applying to this object.

        Notes
        -----
        This method returns a new object that's a view on the same data
        as the input. Mutating the input or the output values will be reflected
        in the other.

        This method is intended to be used in method chains.

        "Flags" differ from "metadata". Flags reflect properties of the
        pandas object (the Series or DataFrame). Metadata refer to properties
        of the dataset, and should be stored in :attr:`DataFrame.attrs`.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2]})
        >>> df.flags.allows_duplicate_labels
        True
        >>> df2 = df.set_flags(allows_duplicate_labels=False)
        >>> df2.flags.allows_duplicate_labels
        False
        """
        # Check for deprecated usage of `copy` keyword
        self._check_copy_deprecation(copy)
        # Create a shallow copy of the object
        df = self.copy(deep=False)
        # Update the 'allows_duplicate_labels' flag if provided
        if allows_duplicate_labels is not None:
            df.flags["allows_duplicate_labels"] = allows_duplicate_labels
        # Return the modified object
        return df

    @final
    @classmethod
    def _validate_dtype(cls, dtype) -> DtypeObj | None:
        """validate the passed dtype"""
        # Convert `dtype` to pandas dtype object if it's not None
        if dtype is not None:
            dtype = pandas_dtype(dtype)

            # Raise error if `dtype` is compound (not implemented)
            if dtype.kind == "V":
                raise NotImplementedError(
                    "compound dtypes are not implemented "
                    f"in the {cls.__name__} constructor"
                )

        return dtype

    # ----------------------------------------------------------------------
    # Construction

    # error: Signature of "_constructor" incompatible with supertype "PandasObject"
    @property


注释：
    def _constructor(self) -> Callable[..., Self]:  # type: ignore[override]
        """
        当一个操作结果与原始数据具有相同维度时使用。
        """
        raise AbstractMethodError(self)

    # ----------------------------------------------------------------------
    # Axis
    _AXIS_ORDERS: list[Literal["index", "columns"]]
    _AXIS_TO_AXIS_NUMBER: dict[Axis, AxisInt] = {0: 0, "index": 0, "rows": 0}
    _info_axis_number: int
    _info_axis_name: Literal["index", "columns"]
    _AXIS_LEN: int

    @final
    def _construct_axes_dict(
        self, axes: Sequence[Axis] | None = None, **kwargs: AxisInt
    ) -> dict:
        """
        返回一个包含轴信息的字典。
        """
        d = {a: self._get_axis(a) for a in (axes or self._AXIS_ORDERS)}
        # error: Argument 1 to "update" of "MutableMapping" has incompatible type
        # "Dict[str, Any]"; expected "SupportsKeysAndGetItem[Union[int, str], Any]"
        d.update(kwargs)  # type: ignore[arg-type]
        return d

    @final
    @classmethod
    def _get_axis_number(cls, axis: Axis) -> AxisInt:
        """
        返回给定轴名称对应的轴编号。
        """
        try:
            return cls._AXIS_TO_AXIS_NUMBER[axis]
        except KeyError as err:
            raise ValueError(
                f"No axis named {axis} for object type {cls.__name__}"
            ) from err

    @final
    @classmethod
    def _get_axis_name(cls, axis: Axis) -> Literal["index", "columns"]:
        """
        返回给定轴名称对应的轴类型。
        """
        axis_number = cls._get_axis_number(axis)
        return cls._AXIS_ORDERS[axis_number]

    @final
    def _get_axis(self, axis: Axis) -> Index:
        """
        返回给定轴名称对应的索引对象。
        """
        axis_number = self._get_axis_number(axis)
        assert axis_number in {0, 1}
        return self.index if axis_number == 0 else self.columns

    @final
    @classmethod
    def _get_block_manager_axis(cls, axis: Axis) -> AxisInt:
        """
        将数据对象的轴映射到数据块管理器的轴。
        """
        axis = cls._get_axis_number(axis)
        ndim = cls._AXIS_LEN
        if ndim == 2:
            # i.e. DataFrame
            return 1 - axis
        return axis

    @final
    def _get_axis_resolvers(self, axis: str) -> dict[str, Series | MultiIndex]:
        # 获取指定轴（索引或列）的索引对象
        axis_index = getattr(self, axis)
        d = {}
        prefix = axis[0]  # 从轴名称中获取前缀字符

        # 遍历轴索引对象的名称和位置信息
        for i, name in enumerate(axis_index.names):
            if name is not None:
                key = level = name  # 如果名称不为空，则使用名称作为键和级别
            else:
                # 如果名称为空，则使用特定前缀和位置信息构造键和级别
                # 例如，对于未命名的多级索引的第0级，需要以'i'或'c'为前缀
                key = f"{prefix}level_{i}"
                level = i

            # 获取级别对应的值，并转换为 Series 对象
            level_values = axis_index.get_level_values(level)
            s = level_values.to_series()
            s.index = axis_index  # 设置 Series 对象的索引为轴索引对象
            d[key] = s  # 将构建的 Series 对象添加到字典中

        # 将索引/列本身添加到字典中
        if isinstance(axis_index, MultiIndex):
            dindex = axis_index  # 如果是 MultiIndex，则直接使用
        else:
            dindex = axis_index.to_series()  # 否则转换为 Series 对象

        d[axis] = dindex  # 将索引/列对象添加到字典中
        return d  # 返回包含轴解析器的字典

    @final
    def _get_index_resolvers(self) -> dict[Hashable, Series | MultiIndex]:
        from pandas.core.computation.parsing import clean_column_name

        d: dict[str, Series | MultiIndex] = {}
        # 遍历所有轴的名称，获取并更新其对应的解析器
        for axis_name in self._AXIS_ORDERS:
            d.update(self._get_axis_resolvers(axis_name))

        # 清理列名中的特殊字符，并排除键为整数的项
        return {clean_column_name(k): v for k, v in d.items() if not isinstance(k, int)}

    @final
    def _get_cleaned_column_resolvers(self) -> dict[Hashable, Series]:
        """
        Return the special character free column resolvers of a DataFrame.

        Column names with special characters are 'cleaned up' so that they can
        be referred to by backtick quoting.
        Used in :meth:`DataFrame.eval`.
        """
        from pandas.core.computation.parsing import clean_column_name
        from pandas.core.series import Series

        if isinstance(self, ABCSeries):
            return {clean_column_name(self.name): self}

        # 清理列名中的特殊字符，并创建对应的 Series 对象
        return {
            clean_column_name(k): Series(
                v, copy=False, index=self.index, name=k, dtype=self.dtypes[k]
            ).__finalize__(self)
            for k, v in zip(self.columns, self._iter_column_arrays())
            if not isinstance(k, int)
        }

    @final
    @property
    def _info_axis(self) -> Index:
        # 返回指定的信息轴对象
        return getattr(self, self._info_axis_name)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return a tuple of axis dimensions
        """
        # 返回 DataFrame 的形状信息，即各轴的长度组成的元组
        return tuple(len(self._get_axis(a)) for a in self._AXIS_ORDERS)

    @property
    def axes(self) -> list[Index]:
        """
        Return index label(s) of the internal NDFrame
        """
        # 返回内部 NDFrame 的索引标签列表
        # 以这种方式实现是因为如果轴被反转，块管理器将显示它们的反向
        return [self._get_axis(a) for a in self._AXIS_ORDERS]
    def ndim(self) -> int:
        """
        Return an int representing the number of axes / array dimensions.

        Return 1 if Series. Otherwise return 2 if DataFrame.

        See Also
        --------
        ndarray.ndim : Number of array dimensions.

        Examples
        --------
        >>> s = pd.Series({"a": 1, "b": 2, "c": 3})
        >>> s.ndim
        1

        >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        >>> df.ndim
        2
        """
        # 返回当前对象的维度数，由底层数据管理器（_mgr）提供
        return self._mgr.ndim

    @final
    @property
    def size(self) -> int:
        """
        Return an int representing the number of elements in this object.

        Return the number of rows if Series. Otherwise return the number of
        rows times number of columns if DataFrame.

        See Also
        --------
        ndarray.size : Number of elements in the array.

        Examples
        --------
        >>> s = pd.Series({"a": 1, "b": 2, "c": 3})
        >>> s.size
        3

        >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        >>> df.size
        4
        """
        # 返回当前对象的元素总数，通过计算其形状的元素乘积得到
        return int(np.prod(self.shape))

    def set_axis(
        self,
        labels,
        *,
        axis: Axis = 0,
        copy: bool | lib.NoDefault = lib.no_default,
    ) -> Self:
        """
        Assign desired index to given axis.

        Indexes for%(extended_summary_sub)s row labels can be changed by assigning
        a list-like or Index.

        Parameters
        ----------
        labels : list-like, Index
            The values for the new index.

        axis : %(axes_single_arg)s, default 0
            The axis to update. The value 0 identifies the rows. For `Series`
            this parameter is unused and defaults to 0.

        copy : bool, default False
            Whether to make a copy of the underlying data.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``

            .. deprecated:: 3.0.0

        Returns
        -------
        %(klass)s
            An object of type %(klass)s.

        See Also
        --------
        %(klass)s.rename_axis : Alter the name of the index%(see_also_sub)s.
        """
        # 检查是否弃用复制操作，如果是，则发出警告
        self._check_copy_deprecation(copy)
        # 在不进行检查的情况下，设置指定轴的索引，返回操作后的对象
        return self._set_axis_nocheck(labels, axis, inplace=False)

    @overload
    def _set_axis_nocheck(
        self, labels, axis: Axis, inplace: Literal[False]
    ) -> Self: ...
    @overload
    def _set_axis_nocheck(self, labels, axis: Axis, inplace: Literal[True]) -> None:
        # 此方法的重载版本，用于在 inplace 模式下设置轴的标签，不返回任何值
        ...

    @overload
    def _set_axis_nocheck(self, labels, axis: Axis, inplace: bool) -> Self | None:
        # 此方法的重载版本，用于设置轴的标签，可以选择是否在原地修改，返回修改后的对象或 None
        ...

    @final
    def _set_axis_nocheck(self, labels, axis: Axis, inplace: bool) -> Self | None:
        # 如果选择 inplace 模式，则直接设置属性并返回 None
        if inplace:
            setattr(self, self._get_axis_name(axis), labels)
            return None
        # 如果不是 inplace 模式，则复制当前对象，设置属性并返回新对象
        obj = self.copy(deep=False)
        setattr(obj, obj._get_axis_name(axis), labels)
        return obj

    @final
    def _set_axis(self, axis: AxisInt, labels: AnyArrayLike | list) -> None:
        """
        从 Cython 代码直接设置 `index` 属性时调用此方法，例如 `series.index = [1, 2, 3]`。
        """
        # 确保标签是索引类型
        labels = ensure_index(labels)
        # 使用底层数据结构管理器设置轴的标签
        self._mgr.set_axis(axis, labels)

    @final
    @doc(klass=_shared_doc_kwargs["klass"])
    def droplevel(self, level: IndexLabel, axis: Axis = 0) -> Self:
        """
        从索引或列中删除请求的级别，并返回修改后的 {klass} 对象。

        Parameters
        ----------
        level : int, str, or list-like
            如果给定字符串，则必须是级别的名称。
            如果是类似列表，则元素必须是级别的名称或位置索引。

        axis : {{0 或 'index'，1 或 'columns'}}，默认 0
            指定删除级别的轴：

            * 0 或 'index'：删除列中的级别。
            * 1 或 'columns'：删除行中的级别。

            对于 `Series`，此参数未使用且默认为 0。

        Returns
        -------
        {klass}
            删除请求的索引或列级别后的 {klass} 对象。

        See Also
        --------
        DataFrame.replace : 用 `to_replace` 中的值替换值。
        DataFrame.pivot : 返回根据给定索引/列值重新组织的 DataFrame。

        Examples
        --------
        >>> df = (
        ...     pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        ...     .set_index([0, 1])
        ...     .rename_axis(["a", "b"])
        ... )

        >>> df.columns = pd.MultiIndex.from_tuples(
        ...     [("c", "e"), ("d", "f")], names=["level_1", "level_2"]
        ... )

        >>> df
        level_1   c   d
        level_2   e   f
        a b
        1 2      3   4
        5 6      7   8
        9 10    11  12

        >>> df.droplevel("a")
        level_1   c   d
        level_2   e   f
        b
        2        3   4
        6        7   8
        10      11  12

        >>> df.droplevel("level_2", axis=1)
        level_1   c   d
        a b
        1 2      3   4
        5 6      7   8
        9 10    11  12
        """
        # 获取指定轴上的标签
        labels = self._get_axis(axis)
        # 删除指定级别后，设置新的轴标签并返回结果
        new_labels = labels.droplevel(level)
        return self.set_axis(new_labels, axis=axis)

    def pop(self, item: Hashable) -> Series | Any:
        # 弹出指定项并返回其对应的值
        result = self[item]
        del self[item]
        return result
    # 定义 _rename 方法的装饰器，标记为最终方法，不允许子类重写
    @final
    # ----------------------------------------------------------------------
    # 以下是 _rename 方法的函数重载，用于不同的参数组合

    # 第一种函数重载定义
    @overload
    def _rename(
        self,
        mapper: Renamer | None = ...,
        *,
        index: Renamer | None = ...,
        columns: Renamer | None = ...,
        axis: Axis | None = ...,
        inplace: Literal[False] = ...,
        level: Level | None = ...,
        errors: str = ...,
    ) -> Self: ...

    # 第二种函数重载定义
    @overload
    def _rename(
        self,
        mapper: Renamer | None = ...,
        *,
        index: Renamer | None = ...,
        columns: Renamer | None = ...,
        axis: Axis | None = ...,
        inplace: Literal[True],
        level: Level | None = ...,
        errors: str = ...,
    ) -> None: ...

    # 第三种函数重载定义
    @overload
    def _rename(
        self,
        mapper: Renamer | None = ...,
        *,
        index: Renamer | None = ...,
        columns: Renamer | None = ...,
        axis: Axis | None = ...,
        inplace: bool,
        level: Level | None = ...,
        errors: str = ...,
    ) -> Self | None: ...

    # ----------------------------------------------------------------------
    # 具体的 _rename 方法定义，根据参数的默认值和类型进行重载
    @final
    def _rename(
        self,
        mapper: Renamer | None = None,
        *,
        index: Renamer | None = None,
        columns: Renamer | None = None,
        axis: Axis | None = None,
        inplace: bool = False,
        level: Level | None = None,
        errors: str = "ignore",
    ) -> Self | None: ...
    ) -> Self | None:
        # 由 Series.rename 和 DataFrame.rename 调用

        # 如果 mapper、index 和 columns 都为 None，则抛出类型错误
        if mapper is None and index is None and columns is None:
            raise TypeError("must pass an index to rename")

        # 如果 index 或 columns 至少有一个不为 None，则检查是否同时指定了 axis，若是则抛出类型错误
        if index is not None or columns is not None:
            if axis is not None:
                raise TypeError(
                    "Cannot specify both 'axis' and any of 'index' or 'columns'"
                )
            # 如果同时指定了 mapper，则抛出类型错误
            if mapper is not None:
                raise TypeError(
                    "Cannot specify both 'mapper' and any of 'index' or 'columns'"
                )
        else:
            # 否则使用 mapper 参数
            if axis and self._get_axis_number(axis) == 1:
                columns = mapper
            else:
                index = mapper

        # 检查 inplace 参数和是否允许重复标签
        self._check_inplace_and_allows_duplicate_labels(inplace)
        # 如果 inplace 为 True，则操作为就地修改，返回自身；否则复制当前对象并返回复制后的对象
        result = self if inplace else self.copy(deep=False)

        # 遍历 index 和 columns 替换的情况
        for axis_no, replacements in enumerate((index, columns)):
            if replacements is None:
                continue

            # 获取当前轴对象
            ax = self._get_axis(axis_no)
            # 获取重命名函数
            f = common.get_rename_function(replacements)

            # 如果指定了 level 参数，则获取其级别编号
            if level is not None:
                level = ax._get_level_number(level)

            # GH 13473
            # 如果 replacements 不是可调用对象，则进行索引操作
            if not callable(replacements):
                # 如果当前轴是多级索引且指定了 level，则获取相应的索引器；否则获取默认的索引器
                if ax._is_multi and level is not None:
                    indexer = ax.get_level_values(level).get_indexer_for(replacements)
                else:
                    indexer = ax.get_indexer_for(replacements)

                # 如果 errors 设置为 "raise" 并且有未找到的索引，则抛出 KeyError
                if errors == "raise" and len(indexer[indexer == -1]):
                    missing_labels = [
                        label
                        for index, label in enumerate(replacements)
                        if indexer[index] == -1
                    ]
                    raise KeyError(f"{missing_labels} not found in axis")

            # 对当前轴的索引进行转换并设置新的索引
            new_index = ax._transform_index(f, level=level)
            result._set_axis_nocheck(new_index, axis=axis_no, inplace=True)

        # 如果 inplace 为 True，则更新当前对象并返回 None；否则返回通过 __finalize__ 处理后的结果对象
        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result.__finalize__(self, method="rename")

    @overload
    def rename_axis(
        self,
        mapper: IndexLabel | lib.NoDefault = ...,
        *,
        index=...,
        columns=...,
        axis: Axis = ...,
        copy: bool | lib.NoDefault = lib.no_default,
        inplace: Literal[False] = ...,
    ) -> Self:
        ...

    @overload
    def rename_axis(
        self,
        mapper: IndexLabel | lib.NoDefault = ...,
        *,
        index=...,
        columns=...,
        axis: Axis = ...,
        copy: bool | lib.NoDefault = lib.no_default,
        inplace: Literal[True],
    ) -> None:
        ...

    @overload
    def rename_axis(
        self,
        mapper: IndexLabel | lib.NoDefault = ...,
        *,
        index=...,
        columns=...,
        axis: Axis = ...,
        copy: bool | lib.NoDefault = lib.no_default,
        inplace: bool = ...,
    @overload
    def rename_axis(
        self,
        mapper: IndexLabel | lib.NoDefault = lib.no_default,
        *,
        index=lib.no_default,
        columns=lib.NoDefault,
        axis: Axis = 0,
        copy: bool | lib.NoDefault = lib.no_default,
        inplace: bool = False,
    ) -> Self | None:
        """
        Overloaded method for renaming axis labels.

        Parameters
        ----------
        mapper : IndexLabel or lib.NoDefault, optional
            Object to map the axis labels to new labels.
        index : lib.NoDefault, optional
            Not used directly. Default is `lib.no_default`.
        columns : lib.NoDefault, optional
            Not used directly. Default is `lib.no_default`.
        axis : Axis, default 0
            Specifies the axis to rename: 0 or 'index' for index, 1 or 'columns' for columns.
        copy : bool or lib.NoDefault, optional
            Not used directly. Default is `lib.no_default`.
        inplace : bool, default False
            If `True`, perform the operation inplace.

        Returns
        -------
        Series, DataFrame, or None
            Returns the modified object if `inplace` is `False`, else returns `None`.

        See Also
        --------
        DataFrame.rename : Alter the axis labels of :class:`DataFrame`.
        Series.rename : Alter the index labels or set the index name of :class:`Series`.
        Index.rename : Set the name of :class:`Index` or :class:`MultiIndex`.

        Examples
        --------
        See the detailed examples in the docstring for usage scenarios.
        """
        """
        Set the name(s) of the axis.

        Parameters
        ----------
        name : str or list of str
            Name(s) to set.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to set the label. The value 0 or 'index' specifies index,
            and the value 1 or 'columns' specifies columns.
        inplace : bool, default False
            If `True`, do operation inplace and return None.

        Returns
        -------
        Series, DataFrame, or None
            The same type as the caller or `None` if `inplace` is `True`.

        See Also
        --------
        DataFrame.rename : Alter the axis labels of :class:`DataFrame`.
        Series.rename : Alter the index labels or set the index name
            of :class:`Series`.
        Index.rename : Set the name of :class:`Index` or :class:`MultiIndex`.

        Examples
        --------
        >>> df = pd.DataFrame({"num_legs": [4, 4, 2]}, ["dog", "cat", "monkey"])
        >>> df
                num_legs
        dog            4
        cat            4
        monkey         2
        >>> df._set_axis_name("animal")
                num_legs
        animal
        dog            4
        cat            4
        monkey         2
        >>> df.index = pd.MultiIndex.from_product(
        ...     [["mammal"], ["dog", "cat", "monkey"]]
        ... )
        >>> df._set_axis_name(["type", "name"])
                       num_legs
        type   name
        mammal dog        4
               cat        4
               monkey     2
        """
        axis = self._get_axis_number(axis)  # Determine the numerical axis position

        # Set names on the axis (index or columns)
        idx = self._get_axis(axis).set_names(name)

        # Validate the `inplace` argument
        inplace = validate_bool_kwarg(inplace, "inplace")

        # Create a copy or modify the object inplace
        renamed = self if inplace else self.copy(deep=False)

        # Update the index or columns with the new names
        if axis == 0:
            renamed.index = idx
        else:
            renamed.columns = idx

        # Return the modified object or None based on `inplace`
        if not inplace:
            return renamed
        return None
    def _indexed_same(self, other) -> bool:
        # 检查当前对象的每个轴是否与另一个对象的对应轴相等
        return all(
            self._get_axis(a).equals(other._get_axis(a)) for a in self._AXIS_ORDERS
        )

    @final
    # -------------------------------------------------------------------------
    # Unary Methods

    @final
    def __neg__(self) -> Self:
        # 定义一个用于处理块的函数，根据数据类型执行相应的负操作
        def blk_func(values: ArrayLike):
            if is_bool_dtype(values.dtype):
                # 如果数据类型是布尔型，则执行按位取反操作
                return operator.inv(values)  # type: ignore[arg-type]
            else:
                # 否则执行数值的负值操作
                return operator.neg(values)  # type: ignore[arg-type]

        # 应用块函数到数据管理器上，生成新的数据
        new_data = self._mgr.apply(blk_func)
        # 根据新数据创建一个新的对象，并保留轴信息
        res = self._constructor_from_mgr(new_data, axes=new_data.axes)
        return res.__finalize__(self, method="__neg__")

    @final
    def __pos__(self) -> Self:
        # 定义一个用于处理块的函数，根据数据类型执行相应的正操作
        def blk_func(values: ArrayLike):
            if is_bool_dtype(values.dtype):
                # 如果数据类型是布尔型，则直接复制值
                return values.copy()
            else:
                # 否则执行数值的正值操作
                return operator.pos(values)  # type: ignore[arg-type]

        # 应用块函数到数据管理器上，生成新的数据
        new_data = self._mgr.apply(blk_func)
        # 根据新数据创建一个新的对象，并保留轴信息
        res = self._constructor_from_mgr(new_data, axes=new_data.axes)
        return res.__finalize__(self, method="__pos__")

    @final
    def __invert__(self) -> Self:
        # 如果对象大小为零，则直接复制对象并返回
        if not self.size:
            return self.copy(deep=False)

        # 应用按位取反函数到数据管理器上，生成新的数据
        new_data = self._mgr.apply(operator.invert)
        # 根据新数据创建一个新的对象，并保留轴信息
        res = self._constructor_from_mgr(new_data, axes=new_data.axes)
        return res.__finalize__(self, method="__invert__")

    @final
    def __nonzero__(self) -> NoReturn:
        # 抛出值错误，因为对象的真值未定义，应使用其他方法来判断
        raise ValueError(
            f"The truth value of a {type(self).__name__} is ambiguous. "
            "Use a.empty, a.bool(), a.item(), a.any() or a.all()."
        )

    __bool__ = __nonzero__

    @final
    @final
    def abs(self) -> Self:
        """
        Return a Series/DataFrame with absolute numeric value of each element.

        This function only applies to elements that are all numeric.

        Returns
        -------
        abs
            Series/DataFrame containing the absolute value of each element.

        See Also
        --------
        numpy.absolute : Calculate the absolute value element-wise.

        Notes
        -----
        For ``complex`` inputs, ``1.2 + 1j``, the absolute value is
        :math:`\\sqrt{ a^2 + b^2 }`.

        Examples
        --------
        Absolute numeric values in a Series.

        >>> s = pd.Series([-1.10, 2, -3.33, 4])
        >>> s.abs()
        0    1.10
        1    2.00
        2    3.33
        3    4.00
        dtype: float64

        Absolute numeric values in a Series with complex numbers.

        >>> s = pd.Series([1.2 + 1j])
        >>> s.abs()
        0    1.56205
        dtype: float64

        Absolute numeric values in a Series with a Timedelta element.

        >>> s = pd.Series([pd.Timedelta("1 days")])
        >>> s.abs()
        0   1 days
        dtype: timedelta64[ns]

        Select rows with data closest to certain value using argsort (from
        `StackOverflow <https://stackoverflow.com/a/17758115>`__).

        >>> df = pd.DataFrame(
        ...     {"a": [4, 5, 6, 7], "b": [10, 20, 30, 40], "c": [100, 50, -30, -50]}
        ... )
        >>> df
             a    b    c
        0    4   10  100
        1    5   20   50
        2    6   30  -30
        3    7   40  -50
        >>> df.loc[(df.c - 43).abs().argsort()]
             a    b    c
        1    5   20   50
        0    4   10  100
        2    6   30  -30
        3    7   40  -50
        """
        # Apply numpy absolute function element-wise to the internal data manager
        res_mgr = self._mgr.apply(np.abs)
        # Return a new instance of the object constructed from the modified data manager
        return self._constructor_from_mgr(res_mgr, axes=res_mgr.axes).__finalize__(
            self, name="abs"
        )

    @final
    def __abs__(self) -> Self:
        """
        Override for absolute value computation using abs() method.

        Returns
        -------
        Self
            Absolute value of the Series/DataFrame.

        See Also
        --------
        abs : Absolute value calculation for Series/DataFrame.

        Notes
        -----
        This method directly calls the abs() method defined in the class.

        Examples
        --------
        Compute absolute values of a DataFrame.

        >>> df = pd.DataFrame({"A": [-1, 2, -3], "B": [4, -5, 6]})
        >>> abs(df)
           A  B
        0  1  4
        1  2  5
        2  3  6
        """
        return self.abs()

    @final
    def __round__(self, decimals: int = 0) -> Self:
        """
        Round the values in the Series/DataFrame to the specified number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to.

        Returns
        -------
        Self
            Series/DataFrame with rounded values.

        See Also
        --------
        round : Python built-in function for rounding.

        Notes
        -----
        This method rounds each element in the Series/DataFrame to the specified number of decimals.

        Examples
        --------
        Round the values in a Series to 2 decimal places.

        >>> s = pd.Series([1.234, 2.567, 3.789])
        >>> s.__round__(2)
        0    1.23
        1    2.57
        2    3.79
        dtype: float64

        Round the values in a DataFrame to 1 decimal place.

        >>> df = pd.DataFrame({"A": [1.234, 2.567], "B": [3.789, 4.891]})
        >>> df.__round__(1)
             A    B
        0  1.2  3.8
        1  2.6  4.9
        """
        # Round the values using the round method of the Series/DataFrame and finalize with method name
        return self.round(decimals).__finalize__(self, method="__round__")

    # -------------------------------------------------------------------------
    # Label or Level Combination Helpers
    #
    # A collection of helper methods for DataFrame/Series operations that
    # accept a combination of column/index labels and levels.  All such
    # operations should utilize/extend these methods when possible so that we
    # have consistent precedence and validation logic throughout the library.

    @final
    @final
    def _is_level_reference(self, key: Level, axis: Axis = 0) -> bool:
        """
        Test whether a key is a level reference for a given axis.

        To be considered a level reference, `key` must be a string that:
          - (axis=0): Matches the name of an index level and does NOT match
            a column label.
          - (axis=1): Matches the name of a column level and does NOT match
            an index label.

        Parameters
        ----------
        key : Hashable
            Potential level name for the given axis
        axis : int, default 0
            Axis that levels are associated with (0 for index, 1 for columns)

        Returns
        -------
        is_level : bool
        """
        axis_int = self._get_axis_number(axis)  # 获取指定轴的整数表示

        return (
            key is not None  # 确保 key 不为空
            and is_hashable(key)  # 确保 key 可哈希
            and key in self.axes[axis_int].names  # 检查 key 是否在指定轴的名称中
            and not self._is_label_reference(key, axis=axis_int)  # 确保 key 不是标签引用
        )

    @final
    def _is_label_reference(self, key: Level, axis: Axis = 0) -> bool:
        """
        Test whether a key is a label reference for a given axis.

        To be considered a label reference, `key` must be a string that:
          - (axis=0): Matches a column label
          - (axis=1): Matches an index label

        Parameters
        ----------
        key : Hashable
            Potential label name, i.e. Index entry.
        axis : int, default 0
            Axis perpendicular to the axis that labels are associated with
            (0 means search for column labels, 1 means search for index labels)

        Returns
        -------
        is_label: bool
        """
        axis_int = self._get_axis_number(axis)  # 获取指定轴的整数表示
        other_axes = (ax for ax in range(self._AXIS_LEN) if ax != axis_int)  # 获取除指定轴外的其他轴

        return (
            key is not None  # 确保 key 不为空
            and is_hashable(key)  # 确保 key 可哈希
            and any(key in self.axes[ax] for ax in other_axes)  # 检查 key 是否在其他轴的名称中
        )

    @final
    def _is_label_or_level_reference(self, key: Level, axis: AxisInt = 0) -> bool:
        """
        Test whether a key is a label or level reference for a given axis.

        To be considered either a label or a level reference, `key` must be a
        string that:
          - (axis=0): Matches a column label or an index level
          - (axis=1): Matches an index label or a column level

        Parameters
        ----------
        key : Hashable
            Potential label or level name
        axis : int, default 0
            Axis that levels are associated with (0 for index, 1 for columns)

        Returns
        -------
        bool
        """
        return self._is_level_reference(key, axis=axis) or self._is_label_reference(
            key, axis=axis
        )
    def _check_label_or_level_ambiguity(self, key: Level, axis: Axis = 0) -> None:
        """
        Check whether `key` is ambiguous.

        By ambiguous, we mean that it matches both a level of the input
        `axis` and a label of the other axis.

        Parameters
        ----------
        key : Hashable
            Label or level name.
        axis : int, default 0
            Axis that levels are associated with (0 for index, 1 for columns).

        Raises
        ------
        ValueError: `key` is ambiguous
        """
        # 将 axis 参数转换为轴的索引值
        axis_int = self._get_axis_number(axis)
        # 生成除了当前轴外的其他轴的迭代器
        other_axes = (ax for ax in range(self._AXIS_LEN) if ax != axis_int)

        # 检查 key 是否不为 None、可哈希，并且存在于当前轴的名称中，
        # 同时存在于其他轴中的任意一个轴中
        if (
            key is not None
            and is_hashable(key)
            and key in self.axes[axis_int].names
            and any(key in self.axes[ax] for ax in other_axes)
        ):
            # 构建一个具有信息量和语法正确性的警告消息
            level_article, level_type = (
                ("an", "index") if axis_int == 0 else ("a", "column")
            )

            label_article, label_type = (
                ("a", "column") if axis_int == 0 else ("an", "index")
            )

            # 构建异常消息，说明 key 同时是一个轴级别和另一个轴标签，导致歧义
            msg = (
                f"'{key}' is both {level_article} {level_type} level and "
                f"{label_article} {label_type} label, which is ambiguous."
            )
            # 抛出 ValueError 异常，传递消息
            raise ValueError(msg)

    @final
    def _get_label_or_level_values(self, key: Level, axis: AxisInt = 0) -> ArrayLike:
        """
        Return a 1-D array of values associated with `key`, a label or level
        from the given `axis`.

        Retrieval logic:
          - (axis=0): Return column values if `key` matches a column label.
            Otherwise return index level values if `key` matches an index
            level.
          - (axis=1): Return row values if `key` matches an index label.
            Otherwise return column level values if 'key' matches a column
            level

        Parameters
        ----------
        key : Hashable
            Label or level name.
        axis : int, default 0
            Axis that levels are associated with (0 for index, 1 for columns)

        Returns
        -------
        np.ndarray or ExtensionArray

        Raises
        ------
        KeyError
            if `key` matches neither a label nor a level
        ValueError
            if `key` matches multiple labels
        """
        # Determine the numerical axis representation based on the provided axis parameter
        axis = self._get_axis_number(axis)
        
        # Find the first axis other than the specified axis
        first_other_axes = next(
            (ax for ax in range(self._AXIS_LEN) if ax != axis), None
        )

        # Check if `key` is a label reference for the specified axis
        if self._is_label_reference(key, axis=axis):
            # Check for ambiguity if `key` matches multiple labels on the specified axis
            self._check_label_or_level_ambiguity(key, axis=axis)
            # Obtain values corresponding to `key` from the cross-section of data
            if first_other_axes is None:
                raise ValueError("axis matched all axes")
            values = self.xs(key, axis=first_other_axes)._values
        # Check if `key` is a level reference for the specified axis
        elif self._is_level_reference(key, axis=axis):
            # Obtain values corresponding to `key` from the level of the axis
            values = self.axes[axis].get_level_values(key)._values
        else:
            # Raise a KeyError if `key` matches neither a label nor a level
            raise KeyError(key)

        # Check for duplicates in the retrieved values
        if values.ndim > 1:
            # Provide additional guidance for multi-index structures if duplicates are found
            if first_other_axes is not None and isinstance(
                self._get_axis(first_other_axes), MultiIndex
            ):
                multi_message = (
                    "\n"
                    "For a multi-index, the label must be a "
                    "tuple with elements corresponding to each level."
                )
            else:
                multi_message = ""

            label_axis_name = "column" if axis == 0 else "index"
            # Raise a ValueError indicating non-uniqueness of the label
            raise ValueError(
                f"The {label_axis_name} label '{key}' is not unique.{multi_message}"
            )

        # Return the retrieved values associated with `key`
        return values
    def _drop_labels_or_levels(self, keys, axis: AxisInt = 0):
        """
        Drop labels and/or levels for the given `axis`.

        For each key in `keys`:
          - (axis=0): If key matches a column label then drop the column.
            Otherwise if key matches an index level then drop the level.
          - (axis=1): If key matches an index label then drop the row.
            Otherwise if key matches a column level then drop the level.

        Parameters
        ----------
        keys : str or list of str
            labels or levels to drop
        axis : int, default 0
            Axis that levels are associated with (0 for index, 1 for columns)

        Returns
        -------
        dropped: DataFrame

        Raises
        ------
        ValueError
            if any `keys` match neither a label nor a level
        """
        axis = self._get_axis_number(axis)

        # Validate keys
        keys = common.maybe_make_list(keys)
        # Filter out keys that are not valid labels or levels for the specified axis
        invalid_keys = [
            k for k in keys if not self._is_label_or_level_reference(k, axis=axis)
        ]

        if invalid_keys:
            raise ValueError(
                "The following keys are not valid labels or "
                f"levels for axis {axis}: {invalid_keys}"
            )

        # Compute levels and labels to drop
        # Identify which keys correspond to level references
        levels_to_drop = [k for k in keys if self._is_level_reference(k, axis=axis)]
        # Identify which keys correspond to label references
        labels_to_drop = [k for k in keys if not self._is_level_reference(k, axis=axis)]

        # Perform a shallow copy of the DataFrame to ensure only one copy is made
        dropped = self.copy(deep=False)

        if axis == 0:
            # Handle dropping index levels
            if levels_to_drop:
                # Drop specified levels from the index
                dropped.reset_index(levels_to_drop, drop=True, inplace=True)

            # Handle dropping column labels
            if labels_to_drop:
                # Drop specified labels from columns
                dropped.drop(labels_to_drop, axis=1, inplace=True)
        else:
            # Handle dropping column levels
            if levels_to_drop:
                if isinstance(dropped.columns, MultiIndex):
                    # Drop specified levels from the MultiIndex columns
                    dropped.columns = dropped.columns.droplevel(levels_to_drop)
                else:
                    # Replace the last level of Index with a RangeIndex
                    dropped.columns = default_index(dropped.columns.size)

            # Handle dropping index labels
            if labels_to_drop:
                # Drop specified labels from index
                dropped.drop(labels_to_drop, axis=0, inplace=True)

        return dropped

    # ----------------------------------------------------------------------
    # Iteration

    # Comment: Link provided as a reference for a known issue related to types in Python
    # and a specific issue in typeshed regarding incompatible assignments.
    # https://github.com/python/typeshed/issues/2148#issuecomment-520783318
    # Incompatible types in assignment (expression has type "None", base class
    # "object" defined the type as "Callable[[object], int]")
    __hash__: ClassVar[None]  # type: ignore[assignment]

    def __iter__(self) -> Iterator:
        """
        Iterate over info axis.

        Returns
        -------
        iterator
            Info axis as iterator.

        See Also
        --------
        DataFrame.items : Iterate over (column name, Series) pairs.
        DataFrame.itertuples : Iterate over DataFrame rows as namedtuples.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> for x in df:
        ...     print(x)
        A
        B
        """
        return iter(self._info_axis)

    # can we get a better explanation of this?
    def keys(self) -> Index:
        """
        Get the 'info axis' (see Indexing for more).

        This is index for Series, columns for DataFrame.

        Returns
        -------
        Index
            Info axis.

        See Also
        --------
        DataFrame.index : The index (row labels) of the DataFrame.
        DataFrame.columns: The column labels of the DataFrame.

        Examples
        --------
        >>> d = pd.DataFrame(
        ...     data={"A": [1, 2, 3], "B": [0, 4, 8]}, index=["a", "b", "c"]
        ... )
        >>> d
           A  B
        a  1  0
        b  2  4
        c  3  8
        >>> d.keys()
        Index(['A', 'B'], dtype='object')
        """
        return self._info_axis

    def items(self):
        """
        Iterate over (label, values) on info axis

        This is index for Series and columns for DataFrame.

        Returns
        -------
        Generator
            Generates tuples of (label, value) pairs.
        """
        for h in self._info_axis:
            yield h, self[h]

    def __len__(self) -> int:
        """Returns length of info axis"""
        return len(self._info_axis)

    @final
    def __contains__(self, key) -> bool:
        """True if the key is in the info axis"""
        return key in self._info_axis

    @property
    def empty(self) -> bool:
        """
        Indicator whether Series/DataFrame is empty.

        True if Series/DataFrame is entirely empty (no items), meaning any of the
        axes are of length 0.

        Returns
        -------
        bool
            If Series/DataFrame is empty, return True, if not return False.

        See Also
        --------
        Series.dropna : Return series without null values.
        DataFrame.dropna : Return DataFrame with labels on given axis omitted
            where (all or any) data are missing.

        Notes
        -----
        If Series/DataFrame contains only NaNs, it is still not considered empty. See
        the example below.

        Examples
        --------
        An example of an actual empty DataFrame. Notice the index is empty:

        >>> df_empty = pd.DataFrame({"A": []})
        >>> df_empty
        Empty DataFrame
        Columns: [A]
        Index: []
        >>> df_empty.empty
        True

        If we only have NaNs in our DataFrame, it is not considered empty! We
        will need to drop the NaNs to make the DataFrame empty:

        >>> df = pd.DataFrame({"A": [np.nan]})
        >>> df
            A
        0 NaN
        >>> df.empty
        False
        >>> df.dropna().empty
        True

        >>> ser_empty = pd.Series({"A": []})
        >>> ser_empty
        A    []
        dtype: object
        >>> ser_empty.empty
        False
        >>> ser_empty = pd.Series()
        >>> ser_empty.empty
        True
        """
        return any(len(self._get_axis(a)) == 0 for a in self._AXIS_ORDERS)




    # ----------------------------------------------------------------------
    # Array Interface

    # This is also set in IndexOpsMixin
    # GH#23114 Ensure ndarray.__op__(DataFrame) returns NotImplemented
    __array_priority__: int = 1000



    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """
        Convert the Series/DataFrame into a NumPy array.

        Parameters
        ----------
        dtype : data-type, optional
            Desired data-type for the array. If not specified, inferred from
            the Series/DataFrame's data.
        copy : bool, optional
            Whether to force a copy of the data. If not specified, behavior is
            determined by the internals of the data.

        Returns
        -------
        np.ndarray
            A NumPy array representation of the Series/DataFrame.

        Notes
        -----
        This method ensures that conversions are done efficiently and respects
        the underlying data structure's views when possible.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> arr = df.__array__()
        >>> arr
        array([[1, 4],
               [2, 5],
               [3, 6]])

        """
        values = self._values
        arr = np.asarray(values, dtype=dtype)
        if astype_is_view(values.dtype, arr.dtype) and self._mgr.is_single_block:
            # Check if both conversions can be done without a copy
            if astype_is_view(self.dtypes.iloc[0], values.dtype) and astype_is_view(
                values.dtype, arr.dtype
            ):
                arr = arr.view()
                arr.flags.writeable = False
        return arr



    @final
    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ):
        """
        Dispatch to handle Ufunc based numpy function operations.

        Parameters
        ----------
        ufunc : np.ufunc
            The numpy ufunc object.
        method : str
            The method to call on the ufunc (e.g., '__call__', 'reduce', etc.).
        *inputs : tuple
            The input arguments to the ufunc function.
        **kwargs : dict
            Additional keyword arguments passed to the ufunc function.

        Returns
        -------
        Result of the numpy ufunc function operation.

        Notes
        -----
        This method provides compatibility for numpy ufunc operations directly on
        Series/DataFrame objects.

        """
        return arraylike.array_ufunc(self, ufunc, method, *inputs, **kwargs)



    # ----------------------------------------------------------------------
    # Picklability

    @final
    # 返回对象的状态字典，包括元数据的键值对和指定的属性值
    def __getstate__(self) -> dict[str, Any]:
        # 从对象的元数据中获取每个键的当前属性值，并存储在字典中
        meta = {k: getattr(self, k, None) for k in self._metadata}
        # 返回对象的状态字典，包括内部管理器、类型、元数据、属性和标志等信息
        return {
            "_mgr": self._mgr,
            "_typ": self._typ,
            "_metadata": self._metadata,
            "attrs": self.attrs,
            "_flags": {k: self.flags[k] for k in self.flags._keys},
            **meta,
        }

    @final
    # 设置对象的状态，从给定的状态中恢复对象的各种属性
    def __setstate__(self, state) -> None:
        # 如果给定状态是 BlockManager 类的实例，则直接设置为内部管理器
        if isinstance(state, BlockManager):
            self._mgr = state
        # 如果给定状态是字典类型，则从中恢复对象的各种属性
        elif isinstance(state, dict):
            # 兼容旧版本的 pickle 文件，将 "_data" 转换为 "_mgr"
            if "_data" in state and "_mgr" not in state:
                state["_mgr"] = state.pop("_data")
            # 获取类型信息
            typ = state.get("_typ")
            if typ is not None:
                # 获取属性信息，默认为空字典
                attrs = state.get("_attrs", {})
                # 如果属性为 None，则设置为空字典以避免问题
                if attrs is None:  # should not happen, but better be on the safe side
                    attrs = {}
                # 将属性设置到对象中
                object.__setattr__(self, "_attrs", attrs)
                # 获取标志信息，默认为允许重复标签的 Flags 对象
                flags = state.get("_flags", {"allows_duplicate_labels": True})
                # 将标志设置到对象中
                object.__setattr__(self, "_flags", Flags(self, **flags))

                # 按照内部名称的顺序设置对象的各种属性，以避免递归定义问题
                meta = set(self._internal_names + self._metadata)
                for k in meta:
                    if k in state and k != "_flags":
                        v = state[k]
                        object.__setattr__(self, k, v)

                # 设置其余的状态值到对象中
                for k, v in state.items():
                    if k not in meta:
                        object.__setattr__(self, k, v)

            else:
                # 如果没有类型信息，则抛出未实现的错误
                raise NotImplementedError("Pre-0.12 pickles are no longer supported")
        # 如果状态的长度为 2，则抛出未实现的错误
        elif len(state) == 2:
            raise NotImplementedError("Pre-0.12 pickles are no longer supported")

    # ----------------------------------------------------------------------
    # 渲染方法

    # 返回对象的字符串表示，基于对对象进行迭代的结果
    def __repr__(self) -> str:
        prepr = f"[{','.join(map(pprint_thing, self))}]"
        return f"{type(self).__name__}({prepr})"

    @final
    # 返回对象的 LaTeX 表示形式，主要用于 nbconvert（将 Jupyter 笔记转换为 PDF）
    def _repr_latex_(self):
        """
        Returns a LaTeX representation for a particular object.
        Mainly for use with nbconvert (jupyter notebook conversion to pdf).
        """
        # 如果配置中指定了使用 LaTeX 渲染，则返回对象的 LaTeX 表示形式
        if config.get_option("styler.render.repr") == "latex":
            return self.to_latex()
        else:
            return None

    @final
    def _repr_data_resource_(self):
        """
        Not a real Jupyter special repr method, but we use the same
        naming convention.
        """
        # 如果配置中启用了 HTML 表格模式，取数据的前几行
        if config.get_option("display.html.table_schema"):
            data = self.head(config.get_option("display.max_rows"))

            # 将数据转换为 JSON 格式的表格形式
            as_json = data.to_json(orient="table")
            as_json = cast(str, as_json)
            # 将 JSON 字符串解析成有序字典对象
            return loads(as_json, object_pairs_hook=collections.OrderedDict)

    # ----------------------------------------------------------------------
    # I/O Methods

    @final
    @doc(
        klass="object",
        storage_options=_shared_docs["storage_options"],
        storage_options_versionadded="1.2.0",
        extra_parameters=textwrap.dedent(
            """\
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
    """
        ),
    )
    # 将数据框写入 Excel 文件的方法
    def to_excel(
        self,
        excel_writer: FilePath | WriteExcelBuffer | ExcelWriter,
        *,
        sheet_name: str = "Sheet1",
        na_rep: str = "",
        float_format: str | None = None,
        columns: Sequence[Hashable] | None = None,
        header: Sequence[Hashable] | bool = True,
        index: bool = True,
        index_label: IndexLabel | None = None,
        startrow: int = 0,
        startcol: int = 0,
        engine: Literal["openpyxl", "xlsxwriter"] | None = None,
        merge_cells: bool = True,
        inf_rep: str = "inf",
        freeze_panes: tuple[int, int] | None = None,
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ):
        pass

    @final
    @doc(
        storage_options=_shared_docs["storage_options"],
        compression_options=_shared_docs["compression_options"] % "path_or_buf",
    )
    # 将数据框写入 JSON 格式的方法
    def to_json(
        self,
        path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
        *,
        orient: Literal["split", "records", "index", "table", "columns", "values"]
        | None = None,
        date_format: str | None = None,
        double_precision: int = 10,
        force_ascii: bool = True,
        date_unit: TimeUnit = "ms",
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        lines: bool = False,
        compression: CompressionOptions = "infer",
        index: bool | None = None,
        indent: int | None = None,
        storage_options: StorageOptions | None = None,
        mode: Literal["a", "w"] = "w",
    ):
        pass
    @final
    def to_hdf(
        self,
        path_or_buf: FilePath | HDFStore,
        *,
        key: str,
        mode: Literal["a", "w", "r+"] = "a",
        complevel: int | None = None,
        complib: Literal["zlib", "lzo", "bzip2", "blosc"] | None = None,
        append: bool = False,
        format: Literal["fixed", "table"] | None = None,
        index: bool = True,
        min_itemsize: int | dict[str, int] | None = None,
        nan_rep=None,
        dropna: bool | None = None,
        data_columns: Literal[True] | list[str] | None = None,
        errors: OpenFileErrors = "strict",
        encoding: str = "UTF-8",
    ):
        """
        Final method to write data to HDF format.

        Args:
            path_or_buf (FilePath | HDFStore): File path or buffer to write.
            key (str): Identifier for the data within HDF.
            mode (Literal["a", "w", "r+"]): File mode for writing ('a' for append, 'w' for write, 'r+' for read/write).
            complevel (int | None): Compression level.
            complib (Literal["zlib", "lzo", "bzip2", "blosc"] | None): Compression library to use.
            append (bool): Whether to append to an existing file.
            format (Literal["fixed", "table"] | None): Format for storing data.
            index (bool): Whether to write DataFrame index.
            min_itemsize (int | dict[str, int] | None): Minimum item size for data.
            nan_rep (Any): Representation for NaN values.
            dropna (bool | None): Whether to drop NaN values before storing.
            data_columns (Literal[True] | list[str] | None): Columns to store as data.
            errors (OpenFileErrors): Error handling during file operations.
            encoding (str): Encoding format for strings.
        """
        ...

    @final
    def to_sql(
        self,
        name: str,
        con,
        *,
        schema: str | None = None,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = True,
        index_label: IndexLabel | None = None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
    ):
        """
        Final method to write data to a SQL database.

        Args:
            name (str): Name of the table.
            con: Database connection object.
            schema (str | None): Database schema.
            if_exists (Literal["fail", "replace", "append"]): Behavior if the table exists ('fail', 'replace', 'append').
            index (bool): Whether to include DataFrame index.
            index_label (IndexLabel | None): Column label(s) for index.
            chunksize (int | None): Number of rows to write at a time.
            dtype (DtypeArg | None): SQL column data types.
            method (Literal["multi"] | Callable | None): SQL insertion method.

        Notes:
            - This method is marked as final, indicating it cannot be overridden by subclasses.
        """
        ...

    @final
    @doc(
        storage_options=_shared_docs["storage_options"],
        compression_options=_shared_docs["compression_options"] % "path",
    )
    def to_pickle(
        self,
        path: FilePath | WriteBuffer[bytes],
        *,
        compression: CompressionOptions = "infer",
        protocol: int = pickle.HIGHEST_PROTOCOL,
        storage_options: StorageOptions | None = None,
    ):
        """
        Final method to serialize object to a pickle file.

        Args:
            path (FilePath | WriteBuffer[bytes]): File path or writable buffer.
            compression (CompressionOptions): Compression method.
            protocol (int): Pickle protocol version.
            storage_options (StorageOptions | None): Storage options for serialization.

        Notes:
            - This method is marked as final and includes documentation for storage and compression options.
        """
        ...
    ) -> None:
        """
        Pickle (serialize) object to file.

        Parameters
        ----------
        path : str, path object, or file-like object
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a binary ``write()`` function. File path where
            the pickled object will be stored.
        {compression_options}  # 描述压缩选项的参数部分
        protocol : int
            Int which indicates which protocol should be used by the pickler,
            default HIGHEST_PROTOCOL (see [1]_ paragraph 12.1.2). The possible
            values are 0, 1, 2, 3, 4, 5. A negative value for the protocol
            parameter is equivalent to setting its value to HIGHEST_PROTOCOL.
            协议参数，指示 pickler 应该使用的协议版本，默认为 HIGHEST_PROTOCOL。

            .. [1] https://docs.python.org/3/library/pickle.html.

        {storage_options}  # 描述存储选项的参数部分

        See Also
        --------
        read_pickle : Load pickled pandas object (or any object) from file.
        DataFrame.to_hdf : Write DataFrame to an HDF5 file.
        DataFrame.to_sql : Write DataFrame to a SQL database.
        DataFrame.to_parquet : Write a DataFrame to the binary parquet format.

        Examples
        --------
        >>> original_df = pd.DataFrame(
        ...     {{"foo": range(5), "bar": range(5, 10)}}
        ... )  # doctest: +SKIP
        >>> original_df  # doctest: +SKIP
           foo  bar
        0    0    5
        1    1    6
        2    2    7
        3    3    8
        4    4    9
        >>> original_df.to_pickle("./dummy.pkl")  # doctest: +SKIP

        >>> unpickled_df = pd.read_pickle("./dummy.pkl")  # doctest: +SKIP
        >>> unpickled_df  # doctest: +SKIP
           foo  bar
        0    0    5
        1    1    6
        2    2    7
        3    3    8
        4    4    9
        """
        from pandas.io.pickle import to_pickle

        to_pickle(
            self,
            path,
            compression=compression,
            protocol=protocol,
            storage_options=storage_options,
        )
    ) -> None:
        r"""
        Copy object to the system clipboard.

        Write a text representation of object to the system clipboard.
        This can be pasted into Excel, for example.

        Parameters
        ----------
        excel : bool, default True
            Produce output in a csv format for easy pasting into excel.

            - True, use the provided separator for csv pasting.
            - False, write a string representation of the object to the clipboard.

        sep : str, default ``'\t'``
            Field delimiter.
        **kwargs
            These parameters will be passed to DataFrame.to_csv.

        See Also
        --------
        DataFrame.to_csv : Write a DataFrame to a comma-separated values
            (csv) file.
        read_clipboard : Read text from clipboard and pass to read_csv.

        Notes
        -----
        Requirements for your platform.

          - Linux : `xclip`, or `xsel` (with `PyQt4` modules)
          - Windows : none
          - macOS : none

        This method uses the processes developed for the package `pyperclip`. A
        solution to render any output string format is given in the examples.

        Examples
        --------
        Copy the contents of a DataFrame to the clipboard.

        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])

        >>> df.to_clipboard(sep=",")  # doctest: +SKIP
        ... # Wrote the following to the system clipboard:
        ... # ,A,B,C
        ... # 0,1,2,3
        ... # 1,4,5,6

        We can omit the index by passing the keyword `index` and setting
        it to false.

        >>> df.to_clipboard(sep=",", index=False)  # doctest: +SKIP
        ... # Wrote the following to the system clipboard:
        ... # A,B,C
        ... # 1,2,3
        ... # 4,5,6

        Using the original `pyperclip` package for any string output format.

        .. code-block:: python

           import pyperclip

           html = df.style.to_html()
           pyperclip.copy(html)
        """
        # 导入 pandas 的剪贴板模块
        from pandas.io import clipboards
        # 使用 pandas 的剪贴板模块将对象复制到系统剪贴板
        clipboards.to_clipboard(self, excel=excel, sep=sep, **kwargs)

    @final
    @overload
    # 定义一个方法用于将数据框（DataFrame）转换为 LaTeX 格式的字符串表示
    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,  # 以下为关键字参数
        columns: Sequence[Hashable] | None = None,  # 要包含的列的顺序，若为 None 则包含所有列
        header: bool | SequenceNotStr[str] = True,  # 是否包含表头，或指定表头内容
        index: bool = True,  # 是否包含行索引
        na_rep: str = "NaN",  # 在表中显示的缺失值表示
        formatters: FormattersType | None = None,  # 列数据的格式化方式
        float_format: FloatFormatType | None = None,  # 浮点数显示格式
        sparsify: bool | None = None,  # 是否稀疏显示表格
        index_names: bool = True,  # 是否包含行索引的名称
        bold_rows: bool = False,  # 是否加粗行
        column_format: str | None = None,  # 列的格式
        longtable: bool | None = None,  # 是否使用长表格格式
        escape: bool | None = None,  # 是否转义特殊字符
        encoding: str | None = None,  # 输出文档的编码方式
        decimal: str = ".",  # 小数点的字符表示
        multicolumn: bool | None = None,  # 是否使用多列格式
        multicolumn_format: str | None = None,  # 多列格式的说明
        multirow: bool | None = None,  # 是否使用多行格式
        caption: str | tuple[str, str] | None = None,  # 表格标题
        label: str | None = None,  # 表格标签
        position: str | None = None,  # 表格的位置
    ) -> str:  # 方法返回字符串表示的 LaTeX 表格
    
    @overload
    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str],
        *,  # 同上，但要求指定输出文件路径或写入缓冲区
        columns: Sequence[Hashable] | None = ...,
        header: bool | SequenceNotStr[str] = ...,
        index: bool = ...,
        na_rep: str = ...,
        formatters: FormattersType | None = ...,
        float_format: FloatFormatType | None = ...,
        sparsify: bool | None = ...,
        index_names: bool = ...,
        bold_rows: bool = ...,
        column_format: str | None = ...,
        longtable: bool | None = ...,
        escape: bool | None = ...,
        encoding: str | None = ...,
        decimal: str = ...,
        multicolumn: bool | None = ...,
        multicolumn_format: str | None = ...,
        multirow: bool | None = ...,
        caption: str | tuple[str, str] | None = ...,
        label: str | None = ...,
        position: str | None = ...,
    ) -> None:  # 方法的重载，用于指定输出到文件或写入缓冲区
    
    @final
    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,  # 同上，但可选参数的默认值稍有不同
        columns: Sequence[Hashable] | None = None,
        header: bool | SequenceNotStr[str] = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: FormattersType | None = None,
        float_format: FloatFormatType | None = None,
        sparsify: bool | None = None,
        index_names: bool = True,
        bold_rows: bool = False,
        column_format: str | None = None,
        longtable: bool | None = None,
        escape: bool | None = None,
        encoding: str | None = None,
        decimal: str = ".",
        multicolumn: bool | None = None,
        multicolumn_format: str | None = None,
        multirow: bool | None = None,
        caption: str | tuple[str, str] | None = None,
        label: str | None = None,
        position: str | None = None,
    ):  # 最终定义的方法重载，综合了前两个版本的参数和默认值
    def _to_latex_via_styler(
        self,
        buf=None,
        *,
        hide: dict | list[dict] | None = None,
        relabel_index: dict | list[dict] | None = None,
        format: dict | list[dict] | None = None,
        format_index: dict | list[dict] | None = None,
        render_kwargs: dict | None = None,
    ):
        """
        Render object to a LaTeX tabular, longtable, or nested table.

        Uses the ``Styler`` implementation with the following, ordered, method chaining:

        .. code-block:: python
           styler = Styler(DataFrame)
           styler.hide(**hide)
           styler.relabel_index(**relabel_index)
           styler.format(**format)
           styler.format_index(**format_index)
           styler.to_latex(buf=buf, **render_kwargs)

        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        hide : dict, list of dict
            Keyword args to pass to the method call of ``Styler.hide``. If a list will
            call the method numerous times.
        relabel_index : dict, list of dict
            Keyword args to pass to the method of ``Styler.relabel_index``. If a list
            will call the method numerous times.
        format : dict, list of dict
            Keyword args to pass to the method call of ``Styler.format``. If a list will
            call the method numerous times.
        format_index : dict, list of dict
            Keyword args to pass to the method call of ``Styler.format_index``. If a
            list will call the method numerous times.
        render_kwargs : dict
            Keyword args to pass to the method call of ``Styler.to_latex``.

        Returns
        -------
        str or None
            If buf is None, returns the result as a string. Otherwise returns None.
        """
        # Import the Styler class from pandas.io.formats.style
        from pandas.io.formats.style import Styler

        # Cast self to DataFrame type
        self = cast("DataFrame", self)
        # Instantiate Styler object with DataFrame and an empty uuid
        styler = Styler(self, uuid="")

        # Iterate over ["hide", "relabel_index", "format", "format_index"]
        for kw_name in ["hide", "relabel_index", "format", "format_index"]:
            # Retrieve the keyword argument dictionary or list
            kw = vars()[kw_name]
            # If kw is a dictionary, call the respective method on styler with **kw
            if isinstance(kw, dict):
                getattr(styler, kw_name)(**kw)
            # If kw is a list, iterate through sub_kw dictionaries and call method on styler
            elif isinstance(kw, list):
                for sub_kw in kw:
                    getattr(styler, kw_name)(**sub_kw)

        # Ensure render_kwargs is not None; initialize as empty dict if so
        render_kwargs = {} if render_kwargs is None else render_kwargs
        # If 'bold_rows' is a key in render_kwargs and is True, apply text formatting
        if render_kwargs.pop("bold_rows"):
            styler.map_index(lambda v: "textbf:--rwrap;")

        # Generate LaTeX output using styler and return as string or None based on buf
        return styler.to_latex(buf=buf, **render_kwargs)

    @overload
    # 定义一个方法 to_csv，用于将数据框保存为 CSV 文件
    def to_csv(
        self,
        # 保存路径或缓冲区，可以是文件路径或写入缓冲区
        path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
        *,
        # CSV 文件的分隔符，默认为逗号
        sep: str = ",",
        # 缺失值表示方式，默认为空字符串
        na_rep: str = "",
        # 浮点数格式化的格式，可以是字符串或可调用对象，默认为 None
        float_format: str | Callable | None = None,
        # 要包含在 CSV 中的列名序列，默认为 None（表示所有列）
        columns: Sequence[Hashable] | None = None,
        # 是否包含列名作为头部，默认为 True
        header: bool | list[str] = True,
        # 是否包含索引，默认为 True
        index: bool = True,
        # 索引标签的名称，默认为 None
        index_label: IndexLabel | None = None,
        # 写入模式，默认为 "w"（覆盖写入）
        mode: str = "w",
        # 编码格式，默认为 None（系统默认编码）
        encoding: str | None = None,
        # 压缩选项，默认为 "infer"（自动推断）
        compression: CompressionOptions = "infer",
        # 引用风格，整数值或 None，默认为 None
        quoting: int | None = None,
        # 引用字符，默认为双引号
        quotechar: str = '"',
        # 行终止符，默认为 None（系统默认）
        lineterminator: str | None = None,
        # 分块写入的行数，默认为 None（全部一次性写入）
        chunksize: int | None = None,
        # 日期格式，默认为 None（不格式化日期）
        date_format: str | None = None,
        # 是否双引号转义，默认为 True
        doublequote: bool = True,
        # 转义字符，默认为 None（无转义字符）
        escapechar: str | None = None,
        # 十进制点表示，默认为 "."
        decimal: str = ".",
        # 文件打开错误处理方式，默认为 "strict"
        errors: OpenFileErrors = "strict",
        # 存储选项，默认为 None
        storage_options: StorageOptions | None = None,
    ) -> str:
        ...

    # 重载方法，用于特定参数的组合（文件路径或写入缓冲区），返回值为 None
    @overload
    def to_csv(
        self,
        path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str],
        *,
        sep: str = ...,
        na_rep: str = ...,
        float_format: str | Callable | None = ...,
        columns: Sequence[Hashable] | None = ...,
        header: bool | list[str] = ...,
        index: bool = ...,
        index_label: IndexLabel | None = ...,
        mode: str = ...,
        encoding: str | None = ...,
        compression: CompressionOptions = ...,
        quoting: int | None = ...,
        quotechar: str = ...,
        lineterminator: str | None = ...,
        chunksize: int | None = ...,
        date_format: str | None = ...,
        doublequote: bool = ...,
        escapechar: str | None = ...,
        decimal: str = ...,
        errors: OpenFileErrors = ...,
        storage_options: StorageOptions = ...,
    ) -> None:
        ...

    # 为 to_csv 方法添加最终定义，确保不被子类重写
    @final
    # 使用文档字符串装饰器 doc，提供关于存储选项和压缩选项的说明
    @doc(
        storage_options=_shared_docs["storage_options"],
        compression_options=_shared_docs["compression_options"] % "path_or_buf",
    )
    # 最终的 to_csv 方法定义，允许 path_or_buf 参数为文件路径、写入缓冲区或 None
    def to_csv(
        self,
        path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
        *,
        sep: str = ",",
        na_rep: str = "",
        float_format: str | Callable | None = None,
        columns: Sequence[Hashable] | None = None,
        header: bool | list[str] = True,
        index: bool = True,
        index_label: IndexLabel | None = None,
        mode: str = "w",
        encoding: str | None = None,
        compression: CompressionOptions = "infer",
        quoting: int | None = None,
        quotechar: str = '"',
        lineterminator: str | None = None,
        chunksize: int | None = None,
        date_format: str | None = None,
        doublequote: bool = True,
        escapechar: str | None = None,
        decimal: str = ".",
        errors: OpenFileErrors = "strict",
        storage_options: StorageOptions | None = None,
    ) -> None:
        ...
    def xs(
        self,
        key: IndexLabel,
        axis: Axis = 0,
        level: IndexLabel | None = None,
        drop_level: bool = True,
    ):
        """
        Returns cross-section from the DataFrame/Series.
        """
        raise AbstractMethodError(self)

    @final
    def _getitem_slice(self, key: slice) -> Self:
        """
        __getitem__ for the case where the key is a slice object.
        """
        # Determine if this slice is positional or label based,
        # and if the latter, convert to positional
        slobj = self.index._convert_slice_indexer(key, kind="getitem")
        if isinstance(slobj, np.ndarray):
            # Convert indices to slice objects, e.g., with DatetimeIndex
            indexer = lib.maybe_indices_to_slice(slobj.astype(np.intp), len(self))
            if isinstance(indexer, np.ndarray):
                # If conversion fails, fall back to using 'take'
                return self.take(indexer, axis=0)
            slobj = indexer
        return self._slice(slobj)

    def _slice(self, slobj: slice, axis: AxisInt = 0) -> Self:
        """
        Construct a slice of this container.

        Slicing with this method is *always* positional.
        """
        assert isinstance(slobj, slice), type(slobj)
        axis = self._get_block_manager_axis(axis)
        new_mgr = self._mgr.get_slice(slobj, axis=axis)
        result = self._constructor_from_mgr(new_mgr, axes=new_mgr.axes)
        result = result.__finalize__(self)
        return result

    @final
    def __delitem__(self, key) -> None:
        """
        Delete item
        """
        deleted = False

        maybe_shortcut = False
        if self.ndim == 2 and isinstance(self.columns, MultiIndex):
            try:
                # Check if key is in MultiIndex columns using engine's __contains__
                maybe_shortcut = key not in self.columns._engine
            except TypeError:
                pass

        if maybe_shortcut:
            # Allow shorthand to delete all columns whose first len(key) elements match key:
            if not isinstance(key, tuple):
                key = (key,)
            for col in self.columns:
                if isinstance(col, tuple) and col[: len(key)] == key:
                    del self[col]
                    deleted = True
        if not deleted:
            # If no match found during loop, raise appropriate exception:
            loc = self.axes[-1].get_loc(key)
            self._mgr = self._mgr.idelete(loc)

    # ----------------------------------------------------------------------
    # Unsorted

    @final
    def _check_inplace_and_allows_duplicate_labels(self, inplace: bool) -> None:
        # 检查是否指定了原地操作且不允许重复标签，若是则抛出 ValueError 异常
        if inplace and not self.flags.allows_duplicate_labels:
            raise ValueError(
                "Cannot specify 'inplace=True' when "
                "'self.flags.allows_duplicate_labels' is False."
            )

    @final
    def get(self, key, default=None):
        """
        Get item from object for given key (ex: DataFrame column).

        Returns ``default`` value if not found.

        Parameters
        ----------
        key : object
            Key for which item should be returned.
        default : object, default None
            Default value to return if key is not found.

        Returns
        -------
        same type as items contained in object
            Item for given key or ``default`` value, if key is not found.

        See Also
        --------
        DataFrame.get : Get item from object for given key (ex: DataFrame column).
        Series.get : Get item from object for given key (ex: DataFrame column).

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [
        ...         [24.3, 75.7, "high"],
        ...         [31, 87.8, "high"],
        ...         [22, 71.6, "medium"],
        ...         [35, 95, "medium"],
        ...     ],
        ...     columns=["temp_celsius", "temp_fahrenheit", "windspeed"],
        ...     index=pd.date_range(start="2014-02-12", end="2014-02-15", freq="D"),
        ... )

        >>> df
                    temp_celsius  temp_fahrenheit windspeed
        2014-02-12          24.3             75.7      high
        2014-02-13          31.0             87.8      high
        2014-02-14          22.0             71.6    medium
        2014-02-15          35.0             95.0    medium

        >>> df.get(["temp_celsius", "windspeed"])
                    temp_celsius windspeed
        2014-02-12          24.3      high
        2014-02-13          31.0      high
        2014-02-14          22.0    medium
        2014-02-15          35.0    medium

        >>> ser = df["windspeed"]
        >>> ser.get("2014-02-13")
        'high'

        If the key isn't found, the default value will be used.

        >>> df.get(["temp_celsius", "temp_kelvin"], default="default_value")
        'default_value'

        >>> ser.get("2014-02-10", "[unknown]")
        '[unknown]'
        """
        try:
            # 尝试获取指定 key 对应的项
            return self[key]
        except (KeyError, ValueError, IndexError):
            # 若 key 不存在，则返回默认值
            return default

    @staticmethod
    # 定义一个函数来检查 'copy' 参数的使用是否过时
    def _check_copy_deprecation(copy):
        # 如果 'copy' 参数不是 lib.no_default，发出警告
        if copy is not lib.no_default:
            warnings.warn(
                "The copy keyword is deprecated and will be removed in a future "
                "version. Copy-on-Write is active in pandas since 3.0 which utilizes "
                "a lazy copy mechanism that defers copies until necessary. Use "
                ".copy() to make an eager copy if necessary.",
                DeprecationWarning,
                # 获取调用栈的层级作为警告的堆栈级别
                stacklevel=find_stack_level(),
            )
    
    # issue 58667
    # 将 'method' 参数标记为过时
    @deprecate_kwarg("method", None)
    # 标记该方法为最终实现，不能被子类覆盖
    @final
    # 定义一个方法 reindex_like，用于根据另一个对象重新索引当前对象
    def reindex_like(
        self,
        other,
        # 重新索引时可选的填充方法，支持多种填充策略或 None
        method: Literal["backfill", "bfill", "pad", "ffill", "nearest"] | None = None,
        # 控制是否进行复制的标志，可以是布尔值或者特定的 lib.NoDefault 值
        copy: bool | lib.NoDefault = lib.no_default,
        # 限制填充的数量
        limit: int | None = None,
        # 容忍性参数，暂未指定类型
        tolerance=None,
    ):
        ...
    
    # 下面是 drop 方法的函数重载定义
    
    # 第一种重载：inplace=True，表示在原对象上进行操作，返回 None
    @overload
    def drop(
        self,
        labels: IndexLabel | ListLike = ...,
        *,
        axis: Axis = ...,
        index: IndexLabel | ListLike = ...,
        columns: IndexLabel | ListLike = ...,
        level: Level | None = ...,
        inplace: Literal[True],
        errors: IgnoreRaise = ...,
    ) -> None:
        ...
    
    # 第二种重载：inplace=False，表示返回操作后的新对象
    @overload
    def drop(
        self,
        labels: IndexLabel | ListLike = ...,
        *,
        axis: Axis = ...,
        index: IndexLabel | ListLike = ...,
        columns: IndexLabel | ListLike = ...,
        level: Level | None = ...,
        inplace: Literal[False] = ...,
        errors: IgnoreRaise = ...,
    ) -> Self:
        ...
    
    # 第三种重载：inplace 参数为布尔值，表示返回新对象或者原对象，取决于 inplace 参数
    @overload
    def drop(
        self,
        labels: IndexLabel | ListLike = ...,
        *,
        axis: Axis = ...,
        index: IndexLabel | ListLike = ...,
        columns: IndexLabel | ListLike = ...,
        level: Level | None = ...,
        inplace: bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Self | None:
        ...
    
    # 主体 drop 方法的实现
    def drop(
        self,
        labels: IndexLabel | ListLike = None,
        *,
        axis: Axis = 0,
        index: IndexLabel | ListLike = None,
        columns: IndexLabel | ListLike = None,
        level: Level | None = None,
        inplace: bool = False,
        errors: IgnoreRaise = "raise",
    ):
        ...
    # 确定是否要就地操作，并验证其有效性
    inplace = validate_bool_kwarg(inplace, "inplace")

    # 如果提供了标签（labels），则处理轴名称和标签的关系
    if labels is not None:
        # 如果同时指定了 'labels' 和 'index'/'columns'，则引发错误
        if index is not None or columns is not None:
            raise ValueError("Cannot specify both 'labels' and 'index'/'columns'")
        # 获取轴的名称（行或列）
        axis_name = self._get_axis_name(axis)
        # 创建包含轴名称和标签的字典
        axes = {axis_name: labels}
    # 如果未提供标签但提供了索引（index）或列（columns）
    elif index is not None or columns is not None:
        # 创建包含索引和列（如果为二维数据）的字典
        axes = {"index": index}
        if self.ndim == 2:
            axes["columns"] = columns
    else:
        # 如果既未提供标签，也未提供索引或列，则引发错误
        raise ValueError(
            "Need to specify at least one of 'labels', 'index' or 'columns'"
        )

    # 将操作对象初始化为当前对象
    obj = self

    # 遍历轴及其对应的标签，执行轴上的删除操作
    for axis, labels in axes.items():
        if labels is not None:
            obj = obj._drop_axis(labels, axis, level=level, errors=errors)

    # 如果指定了就地操作，则更新当前对象
    if inplace:
        self._update_inplace(obj)
        return None
    else:
        # 否则，返回新的对象副本
        return obj

@final
def _drop_axis(
    self,
    labels,
    axis,
    level=None,
    errors: IgnoreRaise = "raise",
    only_slice: bool = False,
):
    """
    在指定轴上删除标签。

    Parameters
    ----------
    labels : 标签或标签列表
    axis : 轴的名称（行或列）
    level : 层次索引级别（默认为None）
    errors : 处理错误方式（默认为"raise"）
    only_slice : 是否只接受切片作为输入（默认为False）
    """
    # 替换对象内部数据为新的结果
    self._mgr = result._mgr

@final
    def add_prefix(self, prefix: str, axis: Axis | None = None) -> Self:
        """
        Prefix labels with string `prefix`.

        For Series, the row labels are prefixed.
        For DataFrame, the column labels are prefixed.

        Parameters
        ----------
        prefix : str
            The string to add before each label.
        axis : {0 or 'index', 1 or 'columns', None}, default None
            Axis to add prefix on

            .. versionadded:: 2.0.0

        Returns
        -------
        Series or DataFrame
            New Series or DataFrame with updated labels.

        See Also
        --------
        Series.add_suffix: Suffix row labels with string `suffix`.
        DataFrame.add_suffix: Suffix column labels with string `suffix`.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> s.add_prefix("item_")
        item_0    1
        item_1    2
        item_2    3
        item_3    4
        dtype: int64

        >>> df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
        >>> df
           A  B
        0  1  3
        1  2  4
        2  3  5
        3  4  6

        >>> df.add_prefix("col_")
             col_A  col_B
        0       1       3
        1       2       4
        2       3       5
        3       4       6
        """
        # 使用 lambda 函数定义一个转换函数，用于在标签前添加指定前缀
        f = lambda x: f"{prefix}{x}"

        # 获取当前对象的标签轴名称，默认为行标签轴名称
        axis_name = self._info_axis_name
        # 如果指定了轴，则根据指定的轴类型获取对应的轴名称
        if axis is not None:
            axis_name = self._get_axis_name(axis)

        # 创建一个字典，将轴名称映射到前缀添加函数 f
        mapper = {axis_name: f}

        # 调用对象的 _rename 方法，根据映射字典对标签进行重命名
        # type: ignore[call-overload, misc] 是类型提示，指示忽略类型检查中的某些问题
        return self._rename(**mapper)  # type: ignore[call-overload, misc]

    @final
    def add_suffix(self, suffix: str, axis: Axis | None = None) -> Self:
        """
        Suffix labels with string `suffix`.

        For Series, the row labels are suffixed.
        For DataFrame, the column labels are suffixed.

        Parameters
        ----------
        suffix : str
            The string to add after each label.
        axis : {0 or 'index', 1 or 'columns', None}, default None
            Axis to add suffix on

            .. versionadded:: 2.0.0

        Returns
        -------
        Series or DataFrame
            New Series or DataFrame with updated labels.

        See Also
        --------
        Series.add_prefix: Prefix row labels with string `prefix`.
        DataFrame.add_prefix: Prefix column labels with string `prefix`.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> s.add_suffix("_item")
        0_item    1
        1_item    2
        2_item    3
        3_item    4
        dtype: int64

        >>> df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
        >>> df
           A  B
        0  1  3
        1  2  4
        2  3  5
        3  4  6

        >>> df.add_suffix("_col")
             A_col  B_col
        0       1       3
        1       2       4
        2       3       5
        3       4       6
        """
        # Define a lambda function `f` to concatenate suffix to labels
        f = lambda x: f"{x}{suffix}"

        # Determine the axis name based on the provided `axis` parameter or default axis
        axis_name = self._info_axis_name
        if axis is not None:
            axis_name = self._get_axis_name(axis)

        # Create a mapper dictionary with the axis name and the suffix function `f`
        mapper = {axis_name: f}
        
        # Call the `_rename` method with the mapper dictionary to suffix the labels
        return self._rename(**mapper)  # type: ignore[call-overload, misc]

    @overload
    def sort_values(
        self,
        *,
        axis: Axis = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: Literal[False] = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        ignore_index: bool = ...,
        key: ValueKeyFunc = ...,
    ) -> Self: ...

    @overload
    def sort_values(
        self,
        *,
        axis: Axis = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: Literal[True],
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        ignore_index: bool = ...,
        key: ValueKeyFunc = ...,
    ) -> None: ...

    @overload
    def sort_values(
        self,
        *,
        axis: Axis = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: bool = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        ignore_index: bool = ...,
        key: ValueKeyFunc = ...,
    ) -> Self | None: ...
    def sort_values(
        self,
        *,
        axis: Axis = 0,  # 指定排序的轴，默认为0，即按行排序
        ascending: bool | Sequence[bool] = True,  # 指定是否升序排序，可以是单个布尔值或布尔值序列，默认为True
        inplace: bool = False,  # 指定是否在原地排序，默认为False，即返回排序后的副本
        kind: SortKind = "quicksort",  # 指定排序算法的种类，默认为快速排序
        na_position: NaPosition = "last",  # 指定缺失值(NaN)的位置，默认放在排序结果的最后
        ignore_index: bool = False,  # 指定是否忽略索引，默认为False，即保留原索引
        key: ValueKeyFunc | None = None,  # 指定用于排序的函数，默认为None，即直接比较元素值
    ):
        ...

    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,  # 指定排序的轴，具体取决于重载函数的参数类型
        level: IndexLabel = ...,  # 指定排序的级别或标签
        ascending: bool | Sequence[bool] = ...,  # 指定是否升序排序，可以是单个布尔值或布尔值序列
        inplace: Literal[True],  # 指定是否在原地排序，此处必须为True
        kind: SortKind = ...,  # 指定排序算法的种类
        na_position: NaPosition = ...,  # 指定缺失值(NaN)的位置
        sort_remaining: bool = ...,  # 指定是否继续排序
        ignore_index: bool = ...,  # 指定是否忽略索引
        key: IndexKeyFunc = ...,  # 指定用于排序的函数
    ) -> None:  # 返回空值，因为此函数只进行原地排序，无返回值
        ...

    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,  # 指定排序的轴，具体取决于重载函数的参数类型
        level: IndexLabel = ...,  # 指定排序的级别或标签
        ascending: bool | Sequence[bool] = ...,  # 指定是否升序排序，可以是单个布尔值或布尔值序列
        inplace: Literal[False] = ...,  # 指定是否在原地排序，此处必须为False
        kind: SortKind = ...,  # 指定排序算法的种类
        na_position: NaPosition = ...,  # 指定缺失值(NaN)的位置
        sort_remaining: bool = ...,  # 指定是否继续排序
        ignore_index: bool = ...,  # 指定是否忽略索引
        key: IndexKeyFunc = ...,  # 指定用于排序的函数
    ) -> Self:  # 返回排序后的对象本身，因为此函数返回排序后的副本
        ...

    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,  # 指定排序的轴，具体取决于重载函数的参数类型
        level: IndexLabel = ...,  # 指定排序的级别或标签
        ascending: bool | Sequence[bool] = ...,  # 指定是否升序排序，可以是单个布尔值或布尔值序列
        inplace: bool = ...,  # 指定是否在原地排序
        kind: SortKind = ...,  # 指定排序算法的种类
        na_position: NaPosition = ...,  # 指定缺失值(NaN)的位置
        sort_remaining: bool = ...,  # 指定是否继续排序
        ignore_index: bool = ...,  # 指定是否忽略索引
        key: IndexKeyFunc = ...,  # 指定用于排序的函数
    ) -> Self | None:  # 返回排序后的对象本身或空值，因为此函数可返回原地排序后的对象或None
        ...

    def sort_index(
        self,
        *,
        axis: Axis = 0,  # 指定排序的轴，默认为0，即按行索引排序
        level: IndexLabel | None = None,  # 指定排序的级别或标签，可以为None
        ascending: bool | Sequence[bool] = True,  # 指定是否升序排序，可以是单个布尔值或布尔值序列，默认为True
        inplace: bool = False,  # 指定是否在原地排序，默认为False，即返回排序后的副本
        kind: SortKind = "quicksort",  # 指定排序算法的种类，默认为快速排序
        na_position: NaPosition = "last",  # 指定缺失值(NaN)的位置，默认放在排序结果的最后
        sort_remaining: bool = True,  # 指定是否继续排序，默认为True
        ignore_index: bool = False,  # 指定是否忽略索引，默认为False，即保留原索引
        key: IndexKeyFunc | None = None,  # 指定用于排序的函数，默认为None，即直接比较索引值
    ) -> Self | None:  # 返回排序后的对象本身或空值，因为此函数可以返回原地排序后的对象或None
        ...
    @final
    def _reindex_axes(
        self,
        axes,
        level: Level | None,
        limit: int | None,
        tolerance,
        method,
        fill_value: Scalar | None,
    ) -> Self:
        """Perform the reindex for all the axes."""
        # 将当前对象引用赋给变量 obj
        obj = self
        # 遍历所有轴的顺序
        for a in self._AXIS_ORDERS:
            # 获取当前轴的标签（索引或列）
            labels = axes[a]
            # 如果标签为空，跳过当前轴的处理
            if labels is None:
                continue

            # 获取当前轴对象
            ax = self._get_axis(a)
            # 对当前轴执行重新索引操作，获取新索引和索引器
            new_index, indexer = ax.reindex(
                labels, level=level, limit=limit, tolerance=tolerance, method=method
            )

            # 获取当前轴的编号
            axis = self._get_axis_number(a)
            # 使用新索引和索引器对 obj 进行重新索引
            obj = obj._reindex_with_indexers(
                {axis: [new_index, indexer]},
                fill_value=fill_value,
                allow_dups=False,
            )

        # 返回重新索引后的对象
        return obj
    # 检查是否需要进行多重索引重新索引
    def _needs_reindex_multi(self, axes, method, level: Level | None) -> bool:
        """Check if we do need a multi reindex."""
        return (
            # 判断所有轴是否都需要重新索引
            (common.count_not_none(*axes.values()) == self._AXIS_LEN)
            and method is None
            and level is None
            # 当 self._can_fast_transpose 为 True 时才执行 reindex_multi，以确保效率
            and self._can_fast_transpose
        )

    # 抛出抽象方法错误
    def _reindex_multi(self, axes, fill_value):
        raise AbstractMethodError(self)

    # 使用索引器进行重新索引
    @final
    def _reindex_with_indexers(
        self,
        reindexers,
        fill_value=None,
        allow_dups: bool = False,
    ) -> Self:
        """allow_dups indicates an internal call here"""
        # 对象中的新数据
        new_data = self._mgr
        # 遍历所有轴上的索引器
        for axis in sorted(reindexers.keys()):
            index, indexer = reindexers[axis]
            # 获取块管理器的轴
            baxis = self._get_block_manager_axis(axis)

            # 如果索引为空，则跳过
            if index is None:
                continue

            # 确保索引的正确性
            index = ensure_index(index)
            # 确保索引器是平台整数
            if indexer is not None:
                indexer = ensure_platform_int(indexer)

            # TODO: 在均匀的 DataFrame 对象上加速（参见 _reindex_multi）
            # 使用新索引和索引器对数据进行重新索引
            new_data = new_data.reindex_indexer(
                index,
                indexer,
                axis=baxis,
                fill_value=fill_value,
                allow_dups=allow_dups,
            )

        # 如果新数据与原始数据相同，则进行浅拷贝
        if new_data is self._mgr:
            new_data = new_data.copy(deep=False)

        # 从新数据中构造一个新对象，并最终化其状态
        return self._constructor_from_mgr(new_data, axes=new_data.axes).__finalize__(
            self
        )

    # 进行筛选操作
    def filter(
        self,
        items=None,
        like: str | None = None,
        regex: str | None = None,
        axis: Axis | None = None,
    ):
    # 定义一个方法 head，用于返回对象的前 `n` 行数据
    def head(self, n: int = 5) -> Self:
        """
        Return the first `n` rows.

        This function exhibits the same behavior as ``df[:n]``, returning the
        first ``n`` rows based on position. It is useful for quickly checking
        if your object has the right type of data in it.

        When ``n`` is positive, it returns the first ``n`` rows. For ``n`` equal to 0,
        it returns an empty object. When ``n`` is negative, it returns
        all rows except the last ``|n|`` rows, mirroring the behavior of ``df[:n]``.

        If ``n`` is larger than the number of rows, this function returns all rows.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.

        Returns
        -------
        same type as caller
            The first `n` rows of the caller object.

        See Also
        --------
        DataFrame.tail: Returns the last `n` rows.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "animal": [
        ...             "alligator",
        ...             "bee",
        ...             "falcon",
        ...             "lion",
        ...             "monkey",
        ...             "parrot",
        ...             "shark",
        ...             "whale",
        ...             "zebra",
        ...         ]
        ...     }
        ... )
        >>> df
              animal
        0  alligator
        1        bee
        2     falcon
        3       lion
        4     monkey
        5     parrot
        6      shark
        7      whale
        8      zebra

        Viewing the first 5 lines

        >>> df.head()
              animal
        0  alligator
        1        bee
        2     falcon
        3       lion
        4     monkey

        Viewing the first `n` lines (three in this case)

        >>> df.head(3)
              animal
        0  alligator
        1        bee
        2     falcon

        For negative values of `n`

        >>> df.head(-3)
              animal
        0  alligator
        1        bee
        2     falcon
        3       lion
        4     monkey
        5     parrot
        """
        # 返回对象的前 `n` 行数据的拷贝
        return self.iloc[:n].copy()

    @final
    def tail(self, n: int = 5) -> Self:
        """
        Return the last `n` rows.

        This function returns the last `n` rows from the object based on
        position. It is useful for quickly verifying data, for example,
        after sorting or appending rows.

        For negative values of `n`, this function returns all rows except
        the first `|n|` rows, equivalent to ``df[|n|:]``.

        If ``n`` is larger than the number of rows, this function returns all rows.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.

        Returns
        -------
        type of caller
            The last `n` rows of the caller object.

        See Also
        --------
        DataFrame.head : The first `n` rows of the caller object.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "animal": [
        ...             "alligator",
        ...             "bee",
        ...             "falcon",
        ...             "lion",
        ...             "monkey",
        ...             "parrot",
        ...             "shark",
        ...             "whale",
        ...             "zebra",
        ...         ]
        ...     }
        ... )
        >>> df
              animal
        0  alligator
        1        bee
        2     falcon
        3       lion
        4     monkey
        5     parrot
        6      shark
        7      whale
        8      zebra

        Viewing the last 5 lines

        >>> df.tail()
           animal
        4  monkey
        5  parrot
        6   shark
        7   whale
        8   zebra

        Viewing the last `n` lines (three in this case)

        >>> df.tail(3)
          animal
        6  shark
        7  whale
        8  zebra

        For negative values of `n`

        >>> df.tail(-3)
           animal
        3    lion
        4  monkey
        5  parrot
        6   shark
        7   whale
        8   zebra
        """
        # 如果 n 为 0，则返回一个空的DataFrame副本
        if n == 0:
            return self.iloc[0:0].copy()
        # 返回DataFrame中倒数第 n 行的副本
        return self.iloc[-n:].copy()

    @final
    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        replace: bool = False,
        weights=None,
        random_state: RandomState | None = None,
        axis: Axis | None = None,
        ignore_index: bool = False,
    ):
        """
        Return a random sample of items from an axis of object.

        This function returns a random sample of items from an axis of the object.
        The user can specify the number of items (`n`) or the fraction of total
        items (`frac`) to return. Additionally, the user can choose whether to
        sample with or without replacement, specify sampling weights, set a random
        seed for reproducibility (`random_state`), and determine whether the sample
        should be taken from rows or columns (`axis`).

        Parameters
        ----------
        n : int or None, default None
            Number of items to return. If None, `frac` must be specified.
        frac : float or None, default None
            Fraction of total items to return. If None, `n` must be specified.
        replace : bool, default False
            Whether to sample with or without replacement.
        weights : str or array-like of float, default None
            Sampling weights. If None, all rows are equally likely.
        random_state : RandomState or None, default None
            Seed for random number generation.
        axis : {0 or 'index', 1 or 'columns', None}, default None
            The axis to sample from. 0 or 'index' for rows, 1 or 'columns' for columns.
        ignore_index : bool, default False
            Whether to reset the index after sampling.

        Returns
        -------
        Same type as caller
            A random sample of items from the caller object.

        See Also
        --------
        DataFrame.head : The first `n` rows of the caller object.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "animal": [
        ...             "alligator",
        ...             "bee",
        ...             "falcon",
        ...             "lion",
        ...             "monkey",
        ...             "parrot",
        ...             "shark",
        ...             "whale",
        ...             "zebra",
        ...         ]
        ...     }
        ... )
        >>> df
              animal
        0  alligator
        1        bee
        2     falcon
        3       lion
        4     monkey
        5     parrot
        6      shark
        7      whale
        8      zebra

        Sampling 3 random items from the DataFrame

        >>> df.sample(n=3)
             animal
        4    monkey
        5    parrot
        6     shark

        Sampling 30% of the DataFrame

        >>> df.sample(frac=0.3)
             animal
        7     whale
        2    falcon
        5    parrot

        Sampling with replacement

        >>> df.sample(n=5, replace=True)
             animal
        0  alligator
        4     monkey
        3       lion
        7      whale
        0  alligator
        """
        # 返回调用者对象的随机样本
        pass

    @overload
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T: ...
    @overload
    def pipe(
        self,
        func: tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...

    @final
    @doc(klass=_shared_doc_kwargs["klass"])
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T] | tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ):
        """
        Apply a function to the DataFrame or Series.

        This function applies the given function (`func`) to the DataFrame or
        Series. It is useful for applying custom operations or transformations
        to the data without modifying the original object. The function `func`
        can be either a callable that takes a DataFrame or Series and returns
        a transformed result, or a tuple where the first element is a callable
        and the second element is a string representing additional arguments
        for the callable.

        Parameters
        ----------
        func : callable or tuple
            A callable function or a tuple where the first element is a callable
            and the second element is a string representing additional arguments
            for the callable.
        *args : any
            Additional positional arguments passed to `func`.
        **kwargs : any
            Additional keyword arguments passed to `func`.

        Returns
        -------
        any
            The result of applying `func` to the DataFrame or Series.

        See Also
        --------
        DataFrame.apply : Apply a function along an axis of the DataFrame.

        Examples
        --------
        >>> def add_two(df):
        ...     return df + 2
        >>> df = pd.DataFrame({"A": [1, 2, 3]})
        >>> df
           A
        0  1
        1  2
        2  3

        Applying `add_two` function using `pipe`

        >>> df.pipe(add_two)
           A
        0  3
        1  4
        2  5

        Applying a function with additional arguments using `pipe`

        >>> def multiply(df, factor):
        ...     return df * factor
        >>> df.pipe((multiply, 'factor'), factor=3)
           A
        0  3
        1  6
        2  9
        """
        # 将函数应用于DataFrame或Series
        pass

    # ----------------------------------------------------------------------
    # Attribute access
    @final
    def __finalize__(self, other, method: str | None = None, **kwargs) -> Self:
        """
        Propagate metadata from other to self.

        Parameters
        ----------
        other : the object from which to get the attributes that we are going
            to propagate
        method : str, optional
            A passed method name providing context on where ``__finalize__``
            was called.

            .. warning::

               The value passed as `method` are not currently considered
               stable across pandas releases.
        """
        # 检查 `other` 是否为 `NDFrame` 的实例
        if isinstance(other, NDFrame):
            # 如果 `other` 对象有属性 `attrs`
            if other.attrs:
                # 我们希望在不使用 attrs（即 attrs 是空字典）时，尽量减少 attrs 传播的性能影响。
                # 可以无条件地执行深拷贝，但空字典的深拷贝比空检查开销大 50 倍。
                # 因此，只有在 `other.attrs` 不为空时才执行深拷贝。
                self.attrs = deepcopy(other.attrs)

            # 将 `self.flags.allows_duplicate_labels` 设置为 `other.flags.allows_duplicate_labels`
            self.flags.allows_duplicate_labels = other.flags.allows_duplicate_labels

            # 对于使用 `_metadata` 的子类
            # 将 `self` 和 `other` 的共同属性复制到 `self`
            for name in set(self._metadata) & set(other._metadata):
                assert isinstance(name, str)
                object.__setattr__(self, name, getattr(other, name, None))

        # 如果 `method` 参数为 "concat"
        if method == "concat":
            # 仅当所有连接参数具有相同的 `attrs` 时才传播 `attrs`
            if all(bool(obj.attrs) for obj in other.objs):
                # 所有连接参数都具有非空的 `attrs`
                attrs = other.objs[0].attrs
                have_same_attrs = all(obj.attrs == attrs for obj in other.objs[1:])
                if have_same_attrs:
                    self.attrs = deepcopy(attrs)

            # 检查所有连接对象中 `flags.allows_duplicate_labels` 是否都相同
            allows_duplicate_labels = all(
                x.flags.allows_duplicate_labels for x in other.objs
            )
            # 将 `self.flags.allows_duplicate_labels` 设置为连接对象的相同标志
            self.flags.allows_duplicate_labels = allows_duplicate_labels

        # 返回更新后的 `self`
        return self
    def __setattr__(self, name: str, value) -> None:
        """
        After regular attribute access, try setting the name
        This allows simpler access to columns for interactive use.
        """
        # 首先尝试通过 __getattribute__ 进行常规属性访问，这样
        # 例如 ``obj.x`` 和 ``obj.x = 4`` 将始终引用/修改
        # 同一个属性。

        try:
            object.__getattribute__(self, name)
            return object.__setattr__(self, name, value)
        except AttributeError:
            pass

        # 如果上面的尝试失败，进入更复杂的属性设置过程
        # （注意，这与上面的 __getattr__ 匹配）。
        if name in self._internal_names_set:
            object.__setattr__(self, name, value)
        elif name in self._metadata:
            object.__setattr__(self, name, value)
        else:
            try:
                existing = getattr(self, name)
                if isinstance(existing, Index):
                    object.__setattr__(self, name, value)
                elif name in self._info_axis:
                    self[name] = value
                else:
                    object.__setattr__(self, name, value)
            except (AttributeError, TypeError):
                if isinstance(self, ABCDataFrame) and (is_list_like(value)):
                    warnings.warn(
                        "Pandas doesn't allow columns to be "
                        "created via a new attribute name - see "
                        "https://pandas.pydata.org/pandas-docs/"
                        "stable/indexing.html#attribute-access",
                        stacklevel=find_stack_level(),
                    )
                object.__setattr__(self, name, value)

    @final
    def _dir_additions(self) -> set[str]:
        """
        add the string-like attributes from the info_axis.
        If info_axis is a MultiIndex, its first level values are used.
        """
        # 获取基类（superclass）的 _dir_additions 结果
        additions = super()._dir_additions()
        # 如果 info_axis 可以包含字符串
        if self._info_axis._can_hold_strings:
            # 更新 additions，使用 _dir_additions_for_owner 来自 info_axis 的字符串属性
            additions.update(self._info_axis._dir_additions_for_owner)
        return additions

    # ----------------------------------------------------------------------
    # Consolidation of internals

    @final
    def _consolidate_inplace(self) -> None:
        """Consolidate data in place and return None"""
        
        # 在原地（inplace）对数据进行整理
        self._mgr = self._mgr.consolidate()

    @final
    def _consolidate(self):
        """
        Compute NDFrame with "consolidated" internals (data of each dtype
        grouped together in a single ndarray).

        Returns
        -------
        consolidated : same type as caller
        """
        # 对数据进行整理，使每种 dtype 的数据在单个 ndarray 中分组

        cons_data = self._mgr.consolidate()
        return self._constructor_from_mgr(cons_data, axes=cons_data.axes).__finalize__(
            self
        )

    @final
    @property
    def _is_mixed_type(self) -> bool:
        # 如果数据管理器只有一个块，说明数据都是同一类型（包括所有 Series 类型）
        if self._mgr.is_single_block:
            # 包括所有 Series 类型的情况
            return False

        # 如果数据管理器包含不同的扩展类型，即使它们具有相同的 dtype，也无法合并它们，因此假装这是“混合”的
        if self._mgr.any_extension_types:
            # 即使它们有相同的 dtype，我们也无法合并它们，所以我们假装这是“混合”的状态
            return True

        # 如果数据管理器中存在多种不同的 dtype，则返回 True，表示数据类型是混合的
        return self.dtypes.nunique() > 1

    @final
    def _get_numeric_data(self) -> Self:
        # 获取数据管理器中的数值型数据
        new_mgr = self._mgr.get_numeric_data()
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(self)

    @final
    def _get_bool_data(self):
        # 获取数据管理器中的布尔型数据
        new_mgr = self._mgr.get_bool_data()
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(self)

    # ----------------------------------------------------------------------
    # Internal Interface Methods

    @property
    def values(self):
        # 返回抽象方法错误，子类需要覆盖该属性
        raise AbstractMethodError(self)

    @property
    def _values(self) -> ArrayLike:
        """内部实现"""
        # 返回抽象方法错误，子类需要覆盖该属性
        raise AbstractMethodError(self)

    @property
    def dtypes(self):
        """
        返回 DataFrame 中的数据类型。

        返回一个 Series，其中包含每列的数据类型。
        结果的索引是原始 DataFrame 的列名。具有混合类型的列存储为“object”dtype。详细信息请参见用户指南中的文档。

        Returns
        -------
        pandas.Series
            每列的数据类型。

        See Also
        --------
        Series.dtypes : 返回底层数据的 dtype 对象。

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "float": [1.0],
        ...         "int": [1],
        ...         "datetime": [pd.Timestamp("20180310")],
        ...         "string": ["foo"],
        ...     }
        ... )
        >>> df.dtypes
        float              float64
        int                  int64
        datetime    datetime64[s]
        string              object
        dtype: object
        """
        # 获取数据管理器中的数据类型信息
        data = self._mgr.get_dtypes()
        return self._constructor_sliced(data, index=self._info_axis, dtype=np.object_)

    @final
    def astype(
        self,
        dtype,
        copy: bool | lib.NoDefault = lib.no_default,
        errors: IgnoreRaise = "raise",
    ):
        # 转换数据类型为指定的 dtype
        pass

    @final
    def __copy__(self, deep: bool = True) -> Self:
        # 创建当前对象的浅拷贝
        return self.copy(deep=deep)

    @final
    def __deepcopy__(self, memo=None) -> Self:
        """
        Parameters
        ----------
        memo, default None
            标准签名，未使用
        """
        # 创建当前对象的深拷贝
        return self.copy(deep=True)

    @final
    def _drop_axis(self, labels, axis: int):
        # 在指定的轴上删除标签
        pass
    @final
    # 声明一个实例方法，用于尝试推断对象列的更好数据类型
    def infer_objects(self, copy: bool | lib.NoDefault = lib.no_default) -> Self:
        """
        Attempt to infer better dtypes for object columns.

        Attempts soft conversion of object-dtyped
        columns, leaving non-object and unconvertible
        columns unchanged. The inference rules are the
        same as during normal Series/DataFrame construction.

        Parameters
        ----------
        copy : bool, default False
            Whether to make a copy for non-object or non-inferable columns
            or Series.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``

            .. deprecated:: 3.0.0

        Returns
        -------
        same type as input object
            Returns an object of the same type as the input object.

        See Also
        --------
        to_datetime : Convert argument to datetime.
        to_timedelta : Convert argument to timedelta.
        to_numeric : Convert argument to numeric type.
        convert_dtypes : Convert argument to best possible dtype.

        Examples
        --------
        >>> df = pd.DataFrame({"A": ["a", 1, 2, 3]})
        >>> df = df.iloc[1:]
        >>> df
           A
        1  1
        2  2
        3  3

        >>> df.dtypes
        A    object
        dtype: object

        >>> df.infer_objects().dtypes
        A    int64
        dtype: object
        """
        # 调用私有方法检查复制的过时警告
        self._check_copy_deprecation(copy)
        # 调用_mgr对象的convert方法，尝试推断对象列的更好数据类型
        new_mgr = self._mgr.convert()
        # 使用构造器创建一个新的实例，传入新的_mgr和轴信息
        res = self._constructor_from_mgr(new_mgr, axes=new_mgr.axes)
        # 使用__finalize__方法，确保返回的结果与当前对象类型一致
        return res.__finalize__(self, method="infer_objects")

    @final
    # 声明一个实例方法，用于将列的数据类型转换为最佳可能类型
    def convert_dtypes(
        self,
        infer_objects: bool = True,
        convert_string: bool = True,
        convert_integer: bool = True,
        convert_boolean: bool = True,
        convert_floating: bool = True,
        dtype_backend: DtypeBackend = "numpy_nullable",
    ):
        # ----------------------------------------------------------------------
        # Filling NA's

    @final
    # 声明一个实例方法，用于填充或回填缺失值
    def _pad_or_backfill(
        self,
        method: Literal["ffill", "bfill", "pad", "backfill"],
        *,
        axis: None | Axis = None,
        inplace: bool = False,
        limit: None | int = None,
        limit_area: Literal["inside", "outside"] | None = None,
    ):
        # 如果未指定轴，则默认为0轴
        if axis is None:
            axis = 0
        # 确定轴的编号
        axis = self._get_axis_number(axis)
        # 清理填充方法
        method = clean_fill_method(method)

        # 如果操作在1轴上
        if axis == 1:
            # 如果不是单一数据块且要求原地操作，则抛出未实现的错误
            if not self._mgr.is_single_block and inplace:
                raise NotImplementedError
            # 例如：test_align_fill_method
            # 对转置后的对象执行填充或反向填充操作
            result = self.T._pad_or_backfill(
                method=method, limit=limit, limit_area=limit_area
            ).T

            return result

        # 对数据管理器执行填充或反向填充操作
        new_mgr = self._mgr.pad_or_backfill(
            method=method,
            limit=limit,
            limit_area=limit_area,
            inplace=inplace,
        )
        # 从新数据管理器创建新对象
        result = self._constructor_from_mgr(new_mgr, axes=new_mgr.axes)
        # 如果是原地操作，则返回更新后的自身对象
        if inplace:
            return self._update_inplace(result)
        else:
            # 否则，返回由新对象完成填充操作后的结果
            return result.__finalize__(self, method="fillna")

    @overload
    def fillna(
        self,
        value: Hashable | Mapping | Series | DataFrame,
        *,
        axis: Axis | None = ...,
        inplace: Literal[False] = ...,
        limit: int | None = ...,
    ) -> Self: ...

    @overload
    def fillna(
        self,
        value: Hashable | Mapping | Series | DataFrame,
        *,
        axis: Axis | None = ...,
        inplace: Literal[True],
        limit: int | None = ...,
    ) -> None: ...

    @overload
    def fillna(
        self,
        value: Hashable | Mapping | Series | DataFrame,
        *,
        axis: Axis | None = ...,
        inplace: bool = ...,
        limit: int | None = ...,
    ) -> Self | None: ...

    @final
    @doc(
        klass=_shared_doc_kwargs["klass"],
        axes_single_arg=_shared_doc_kwargs["axes_single_arg"],
    )
    # 填充缺失值的方法，支持不同的参数组合
    def fillna(
        self,
        value: Hashable | Mapping | Series | DataFrame,
        *,
        axis: Axis | None = None,
        inplace: bool = False,
        limit: int | None = None,
    @overload
    # 向前填充方法，支持不同的参数组合
    def ffill(
        self,
        *,
        axis: None | Axis = ...,
        inplace: Literal[False] = ...,
        limit: None | int = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> Self: ...

    @overload
    def ffill(
        self,
        *,
        axis: None | Axis = ...,
        inplace: Literal[True],
        limit: None | int = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> None: ...

    @overload
    def ffill(
        self,
        *,
        axis: None | Axis = ...,
        inplace: bool = ...,
        limit: None | int = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> Self | None: ...

    @final
    @doc(
        klass=_shared_doc_kwargs["klass"],
        axes_single_arg=_shared_doc_kwargs["axes_single_arg"],
    )
    # 向前填充方法，支持不同的参数组合
    def ffill(
        self,
        *,
        axis: None | Axis = None,
        inplace: bool = False,
        limit: None | int = None,
        limit_area: Literal["inside", "outside"] | None = None,
    @overload
    # bfill 方法的重载定义，用于向后填充缺失的值
    @overload
    def bfill(
        self,
        *,
        axis: None | Axis = ...,
        inplace: Literal[False] = ...,
        limit: None | int = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> Self: ...
    
    # bfill 方法的重载定义，用于在原地填充缺失的值
    @overload
    def bfill(
        self,
        *,
        axis: None | Axis = ...,
        inplace: Literal[True],
        limit: None | int = ...,
    ) -> None: ...
    
    # bfill 方法的重载定义，用于向后填充缺失的值或在原地填充
    @overload
    def bfill(
        self,
        *,
        axis: None | Axis = ...,
        inplace: bool = ...,
        limit: None | int = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> Self | None: ...
    
    # 标记为最终实现的 bfill 方法，提供了详细的文档信息
    @final
    @doc(
        klass=_shared_doc_kwargs["klass"],
        axes_single_arg=_shared_doc_kwargs["axes_single_arg"],
    )
    def bfill(
        self,
        *,
        axis: None | Axis = None,
        inplace: bool = False,
        limit: None | int = None,
        limit_area: Literal["inside", "outside"] | None = None,
    
    
    
    # replace 方法的重载定义，用于替换特定的值
    @overload
    def replace(
        self,
        to_replace=...,
        value=...,
        *,
        inplace: Literal[False] = ...,
        regex: bool = ...,
    ) -> Self: ...
    
    # replace 方法的重载定义，用于在原地替换特定的值
    @overload
    def replace(
        self,
        to_replace=...,
        value=...,
        *,
        inplace: Literal[True],
        regex: bool = ...,
    ) -> None: ...
    
    # replace 方法的重载定义，用于替换特定的值或在原地替换
    @overload
    def replace(
        self,
        to_replace=...,
        value=...,
        *,
        inplace: bool = ...,
        regex: bool = ...,
    ) -> Self | None: ...
    
    # 标记为最终实现的 replace 方法，提供了详细的文档信息
    @final
    @doc(
        _shared_docs["replace"],
        klass=_shared_doc_kwargs["klass"],
        inplace=_shared_doc_kwargs["inplace"],
    )
    def replace(
        self,
        to_replace=None,
        value=lib.no_default,
        *,
        inplace: bool = False,
        regex: bool = False,
    
    
    
    # interpolate 方法的重载定义，用于插值处理缺失的值
    @overload
    def interpolate(
        self,
        method: InterpolateOptions = ...,
        *,
        axis: Axis = ...,
        limit: int | None = ...,
        inplace: Literal[False] = ...,
        limit_direction: Literal["forward", "backward", "both"] | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
        **kwargs,
    ) -> Self: ...
    
    # interpolate 方法的重载定义，用于在原地插值处理缺失的值
    @overload
    def interpolate(
        self,
        method: InterpolateOptions = ...,
        *,
        axis: Axis = ...,
        limit: int | None = ...,
        inplace: Literal[True],
        limit_direction: Literal["forward", "backward", "both"] | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
        **kwargs,
    ) -> None: ...
    
    # interpolate 方法的重载定义，用于插值处理缺失的值或在原地插值处理
    @overload
    def interpolate(
        self,
        method: InterpolateOptions = ...,
        *,
        axis: Axis = ...,
        limit: int | None = ...,
        inplace: bool = ...,
        limit_direction: Literal["forward", "backward", "both"] | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
        **kwargs,
    ) -> Self | None: ...
    
    # 标记为最终实现的 interpolate 方法，提供了详细的文档信息
    @final
    def interpolate(
        self,
        method: InterpolateOptions = "linear",
        *,
        axis: Axis = 0,
        limit: int | None = None,
        inplace: bool = False,
        limit_direction: Literal["forward", "backward", "both"] | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        **kwargs,
    ):
        """
        Interpolate values in the object.

        Parameters
        ----------
        method : InterpolateOptions, optional
            Interpolation technique (default is 'linear').
        axis : Axis, optional
            Axis to interpolate along (default is 0).
        limit : int or None, optional
            Maximum number of consecutive NaNs to fill (default is None).
        inplace : bool, optional
            Modify the object in place (default is False).
        limit_direction : {'forward', 'backward', 'both'} or None, optional
            Limit interpolation direction (default is None).
        limit_area : {'inside', 'outside'} or None, optional
            Interpolation limit area (default is None).
        **kwargs
            Additional keyword arguments for specific interpolation methods.

        Returns
        -------
        self
            The interpolated object.

        Notes
        -----
        This method performs interpolation to fill missing values based on the specified method and parameters.

        See Also
        --------
        pandas.DataFrame.interpolate : Interpolate DataFrame columns.
        pandas.Series.interpolate : Interpolate Series values.

        Examples
        --------
        Interpolate missing values in a DataFrame:

        >>> df = pd.DataFrame({'A': [1, 2, np.nan, 4]})
        >>> df.interpolate(method='linear')
             A
        0  1.0
        1  2.0
        2  3.0
        3  4.0

        Interpolate values in a Series:

        >>> ser = pd.Series([1, np.nan, 3])
        >>> ser.interpolate()
        0    1.0
        1    2.0
        2    3.0
        dtype: float64
        """
        return interpolate(self, method=method, axis=axis, limit=limit, inplace=inplace,
                           limit_direction=limit_direction, limit_area=limit_area, **kwargs)

    @final
    """
    This decorator marks a method as final, preventing it from being overridden by subclasses.
    """

    @doc(klass=_shared_doc_kwargs["klass"])
    """
    Decorator that attaches documentation from a shared documentation dictionary to the decorated method.
    """

    def isna(self) -> Self:
        """
        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA.
        NA values, such as None or :attr:`numpy.NaN`, gets mapped to True
        values.
        Everything else gets mapped to False values. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values.

        Returns
        -------
        {klass}
            Mask of bool values for each element in {klass} that
            indicates whether an element is an NA value.

        See Also
        --------
        {klass}.isnull : Alias of isna.
        {klass}.notna : Boolean inverse of isna.
        {klass}.dropna : Omit axes labels with missing values.
        isna : Top-level isna.

        Examples
        --------
        Show which entries in a DataFrame are NA.

        >>> df = pd.DataFrame(
        ...     dict(
        ...         age=[5, 6, np.nan],
        ...         born=[
        ...             pd.NaT,
        ...             pd.Timestamp("1939-05-27"),
        ...             pd.Timestamp("1940-04-25"),
        ...         ],
        ...         name=["Alfred", "Batman", ""],
        ...         toy=[None, "Batmobile", "Joker"],
        ...     )
        ... )
        >>> df
           age       born    name        toy
        0  5.0        NaT  Alfred       None
        1  6.0 1939-05-27  Batman  Batmobile
        2  NaN 1940-04-25              Joker

        >>> df.isna()
             age   born   name    toy
        0  False   True  False   True
        1  False  False  False  False
        2   True  False  False  False

        Show which entries in a Series are NA.

        >>> ser = pd.Series([5, 6, np.nan])
        >>> ser
        0    5.0
        1    6.0
        2    NaN
        dtype: float64

        >>> ser.isna()
        0    False
        1    False
        2     True
        dtype: bool
        """
        return isna(self).__finalize__(self, method="isna")

    @doc(isna, klass=_shared_doc_kwargs["klass"])
    """
    Decorator that attaches documentation from the shared documentation dictionary to the decorated method.
    """

    def isnull(self) -> Self:
        """
        Alias for isna method.

        Returns
        -------
        {klass}
            Mask of bool values for each element in {klass} that
            indicates whether an element is null.

        See Also
        --------
        {klass}.isna : Detect missing values.
        {klass}.notna : Boolean inverse of isna.
        {klass}.dropna : Omit axes labels with missing values.

        Examples
        --------
        Alias for isna method. Same behavior as isna.

        >>> df = pd.DataFrame({'A': [1, 2, np.nan]})
        >>> df.isnull()
              A
        0  False
        1  False
        2   True
        """
        return isna(self).__finalize__(self, method="isnull")

    @doc(klass=_shared_doc_kwargs["klass"])
    """
    Decorator that attaches documentation from a shared documentation dictionary to the decorated method.
    """
    def notna(self) -> Self:
        """
        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to True. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values.
        NA values, such as None or :attr:`numpy.NaN`, get mapped to False
        values.

        Returns
        -------
        {klass}
            Mask of bool values for each element in {klass} that
            indicates whether an element is not an NA value.

        See Also
        --------
        {klass}.notnull : Alias of notna.
        {klass}.isna : Boolean inverse of notna.
        {klass}.dropna : Omit axes labels with missing values.
        notna : Top-level notna.

        Examples
        --------
        Show which entries in a DataFrame are not NA.

        >>> df = pd.DataFrame(
        ...     dict(
        ...         age=[5, 6, np.nan],
        ...         born=[
        ...             pd.NaT,
        ...             pd.Timestamp("1939-05-27"),
        ...             pd.Timestamp("1940-04-25"),
        ...         ],
        ...         name=["Alfred", "Batman", ""],
        ...         toy=[None, "Batmobile", "Joker"],
        ...     )
        ... )
        >>> df
           age       born    name        toy
        0  5.0        NaT  Alfred       None
        1  6.0 1939-05-27  Batman  Batmobile
        2  NaN 1940-04-25              Joker

        >>> df.notna()
             age   born  name    toy
        0   True  False  True  False
        1   True   True  True   True
        2  False   True  True   True

        Show which entries in a Series are not NA.

        >>> ser = pd.Series([5, 6, np.nan])
        >>> ser
        0    5.0
        1    6.0
        2    NaN
        dtype: float64

        >>> ser.notna()
        0     True
        1     True
        2    False
        dtype: bool
        """
        # 调用 `notna` 函数检测非缺失值，并对结果进行最终化处理
        return notna(self).__finalize__(self, method="notna")

    @doc(notna, klass=_shared_doc_kwargs["klass"])
    def notnull(self) -> Self:
        """
        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to True. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values.
        NA values, such as None or :attr:`numpy.NaN`, get mapped to False
        values.

        Returns
        -------
        {klass}
            Mask of bool values for each element in {klass} that
            indicates whether an element is not an NA value.

        See Also
        --------
        {klass}.notnull : Alias of notna.
        {klass}.isna : Boolean inverse of notna.
        {klass}.dropna : Omit axes labels with missing values.
        notna : Top-level notna.

        Examples
        --------
        Show which entries in a DataFrame are not NA.

        >>> df = pd.DataFrame(
        ...     dict(
        ...         age=[5, 6, np.nan],
        ...         born=[
        ...             pd.NaT,
        ...             pd.Timestamp("1939-05-27"),
        ...             pd.Timestamp("1940-04-25"),
        ...         ],
        ...         name=["Alfred", "Batman", ""],
        ...         toy=[None, "Batmobile", "Joker"],
        ...     )
        ... )
        >>> df
           age       born    name        toy
        0  5.0        NaT  Alfred       None
        1  6.0 1939-05-27  Batman  Batmobile
        2  NaN 1940-04-25              Joker

        >>> df.notna()
             age   born  name    toy
        0   True  False  True  False
        1   True   True  True   True
        2  False   True  True   True

        Show which entries in a Series are not NA.

        >>> ser = pd.Series([5, 6, np.nan])
        >>> ser
        0    5.0
        1    6.0
        2    NaN
        dtype: float64

        >>> ser.notna()
        0     True
        1     True
        2    False
        dtype: bool
        """
        # 调用 `notna` 函数检测非缺失值，并对结果进行最终化处理
        return notna(self).__finalize__(self, method="notnull")

    @final
    def _clip_with_scalar(self, lower, upper, inplace: bool = False):
        """
        Clip values at threshold with scalar inputs.

        Parameters
        ----------
        lower : scalar
            Minimum threshold value. Values less than this will be set to this
            value. NA values are ignored if they are present in 'lower'.
        upper : scalar
            Maximum threshold value. Values greater than this will be set to this
            value. NA values are ignored if they are present in 'upper'.
        inplace : bool, default False
            If True, perform operation in place and return self.

        Returns
        -------
        {klass}
            Object with clipped values.

        Raises
        ------
        ValueError
            If 'lower' or 'upper' contain NA values.

        Notes
        -----
        If inplace=True, it modifies the object directly and returns None.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4, 5])
        >>> ser._clip_with_scalar(2, 4)
        0    2
        1    2
        2    3
        3    4
        4    4
        dtype: int64

        >>> ser = pd.Series([1, 2, 3, 4, 5])
        >>> ser._clip_with_scalar(2, 4, inplace=True)
        >>> ser
        0    2
        1    2
        2    3
        3    4
        4    4
        dtype: int64
        """
        # 检查 lower 和 upper 是否含有 NA 值，如果有则抛出 ValueError 异常
        if (lower is not None and np.any(isna(lower))) or (
            upper is not None and np.any(isna(upper))
        ):
            raise ValueError("Cannot use an NA value as a clip threshold")

        result = self
        mask = self.isna()

        # 对 lower 进行裁剪操作
        if lower is not None:
            cond = mask | (self >= lower)
            result = result.where(cond, lower, inplace=inplace)  # type: ignore[assignment]

        # 对 upper 进行裁剪操作
        if upper is not None:
            cond = mask | (self <= upper)
            result = self if inplace else result
            result = result.where(cond, upper, inplace=inplace)  # type: ignore[assignment]

        return result
    # 定义一个用于裁剪数据的方法，根据指定的阈值、方法、轴和是否原地操作来裁剪数据
    def _clip_with_one_bound(self, threshold, method, axis, inplace):
        # 如果指定了轴参数，则将轴参数转换为轴编号
        if axis is not None:
            axis = self._get_axis_number(axis)

        # 根据方法选择阈值进行裁剪，self.le 表示上界，self.ge 表示下界
        if is_scalar(threshold) and is_number(threshold):
            if method.__name__ == "le":
                # 调用 _clip_with_scalar 方法进行裁剪，指定上界阈值
                return self._clip_with_scalar(None, threshold, inplace=inplace)
            # 调用 _clip_with_scalar 方法进行裁剪，指定下界阈值
            return self._clip_with_scalar(threshold, None, inplace=inplace)

        # GH #15390
        # 为了让 where 方法能够工作，需要将阈值从其他数组结构转换为 NDFrame 结构
        if (not isinstance(threshold, ABCSeries)) and is_list_like(threshold):
            if isinstance(self, ABCSeries):
                # 如果当前对象是 Series 类型，则使用当前索引构造新的 Series 对象
                threshold = self._constructor(threshold, index=self.index)
            else:
                # 调用 _align_for_op 方法，将阈值与轴对齐，获取对齐后的阈值
                threshold = self._align_for_op(threshold, axis, flex=None)[1]

        # GH 40420
        # 处理缺失的阈值，将其视为无边界，即不对值进行裁剪
        if is_list_like(threshold):
            # 如果阈值是列表结构，则根据方法选择填充值为正无穷或负无穷
            fill_value = np.inf if method.__name__ == "le" else -np.inf
            # 使用填充值填充缺失值
            threshold_inf = threshold.fillna(fill_value)
        else:
            threshold_inf = threshold

        # 使用 method 方法对阈值进行比较操作，并将结果与缺失值检测结果进行或运算
        subset = method(threshold_inf, axis=axis) | isna(self)

        # GH 40420
        # 使用 where 方法根据 subset 结果对数据进行条件替换
        return self.where(subset, threshold, axis=axis, inplace=inplace)
    def at_time(self, time, asof: bool = False, axis: Axis | None = None) -> Self:
        """
        Select values at particular time of day (e.g., 9:30AM).

        Parameters
        ----------
        time : datetime.time or str
            The time of day to select values.
        asof : bool, default False
            This parameter is currently not supported.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            For `Series` this parameter is unused and defaults to 0.

        Returns
        -------
        Series or DataFrame
            A new Series or DataFrame with values at the specified time.

        Raises
        ------
        TypeError
            If the index is not a :class:`DatetimeIndex`.

        See Also
        --------
        between_time : Select values between particular times of the day.
        first : Select initial periods of time series based on a date offset.
        last : Select final periods of time series based on a date offset.
        DatetimeIndex.indexer_at_time : Get just the index locations for
            values at particular time of the day.

        Examples
        --------
        >>> i = pd.date_range("2018-04-09", periods=4, freq="12h")
        >>> ts = pd.DataFrame({"A": [1, 2, 3, 4]}, index=i)
        >>> ts
                             A
        2018-04-09 00:00:00  1
        2018-04-09 12:00:00  2
        2018-04-10 00:00:00  3
        2018-04-10 12:00:00  4

        >>> ts.at_time("12:00")
                             A
        2018-04-09 12:00:00  2
        2018-04-10 12:00:00  4
        """
        if axis is None:
            axis = 0  # 如果未指定轴向，则默认为0

        axis = self._get_axis_number(axis)  # 获取轴向的数值表示

        index = self._get_axis(axis)  # 获取指定轴向的索引

        if not isinstance(index, DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex")  # 如果索引不是 DatetimeIndex 类型，则抛出类型错误异常

        indexer = index.indexer_at_time(time, asof=asof)  # 获取特定时间的索引位置

        return self.take(indexer, axis=axis)  # 根据索引位置选取数据，返回新的 Series 或 DataFrame

    @final
    def between_time(
        self,
        start_time,
        end_time,
        inclusive: IntervalClosedType = "both",
        axis: Axis | None = None,
    ) -> Self:
        """
        Select values between particular times of the day (e.g., 9:00-9:30 AM).

        By setting ``start_time`` to be later than ``end_time``,
        you can get the times that are *not* between the two times.

        Parameters
        ----------
        start_time : datetime.time or str
            Initial time as a time filter limit.
        end_time : datetime.time or str
            End time as a time filter limit.
        inclusive : {"both", "neither", "left", "right"}, default "both"
            Include boundaries; whether to set each bound as closed or open.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Determine range time on index or columns value.
            For `Series` this parameter is unused and defaults to 0.

        Returns
        -------
        Series or DataFrame
            Data from the original object filtered to the specified dates range.

        Raises
        ------
        TypeError
            If the index is not  a :class:`DatetimeIndex`

        See Also
        --------
        at_time : Select values at a particular time of the day.
        first : Select initial periods of time series based on a date offset.
        last : Select final periods of time series based on a date offset.
        DatetimeIndex.indexer_between_time : Get just the index locations for
            values between particular times of the day.

        Examples
        --------
        >>> i = pd.date_range("2018-04-09", periods=4, freq="1D20min")
        >>> ts = pd.DataFrame({"A": [1, 2, 3, 4]}, index=i)
        >>> ts
                             A
        2018-04-09 00:00:00  1
        2018-04-10 00:20:00  2
        2018-04-11 00:40:00  3
        2018-04-12 01:00:00  4

        >>> ts.between_time("0:15", "0:45")
                             A
        2018-04-10 00:20:00  2
        2018-04-11 00:40:00  3

        You get the times that are *not* between two times by setting
        ``start_time`` later than ``end_time``:

        >>> ts.between_time("0:45", "0:15")
                             A
        2018-04-09 00:00:00  1
        2018-04-12 01:00:00  4
        """
        # 如果没有指定轴，则默认为0
        if axis is None:
            axis = 0
        # 确定轴的编号
        axis = self._get_axis_number(axis)

        # 获取轴上的索引
        index = self._get_axis(axis)
        # 如果索引不是 DatetimeIndex 类型，则抛出类型错误异常
        if not isinstance(index, DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex")

        # 确定包含左右边界的设置
        left_inclusive, right_inclusive = validate_inclusive(inclusive)
        # 获取符合条件的索引器
        indexer = index.indexer_between_time(
            start_time,
            end_time,
            include_start=left_inclusive,
            include_end=right_inclusive,
        )
        # 根据索引器获取数据，并返回结果
        return self.take(indexer, axis=axis)

    @final
    @doc(klass=_shared_doc_kwargs["klass"])
    # 定义一个方法 resample，用于重新采样时间序列数据
    def resample(
        self,
        rule,  # 重新采样的规则，例如 'D' 表示按日重新采样
        closed: Literal["right", "left"] | None = None,  # 控制重新采样中区间的闭合方式，可选为 'right' 或 'left'
        label: Literal["right", "left"] | None = None,  # 控制重新采样中标签的位置，可选为 'right' 或 'left'
        convention: Literal["start", "end", "s", "e"] | lib.NoDefault = lib.no_default,  # 确定重新采样时的起始时间点，默认无默认值
        on: Level | None = None,  # 指定重新采样的层级
        level: Level | None = None,  # 指定重新采样的级别
        origin: str | TimestampConvertibleTypes = "start_day",  # 确定重新采样中的起始时间点，默认为 "start_day"
        offset: TimedeltaConvertibleTypes | None = None,  # 指定重新采样的时间偏移量
        group_keys: bool = False,  # 是否根据索引的键分组
    ):
    
    # 标记方法为最终方法，不允许子类重写
    @final
    # 定义一个方法 rank，用于计算并返回数据的排名
    def rank(
        self,
        axis: Axis = 0,  # 指定计算排名的轴，默认为 0，表示沿行计算
        method: Literal["average", "min", "max", "first", "dense"] = "average",  # 指定计算排名的方法，例如 'average' 表示使用平均排名
        numeric_only: bool = False,  # 是否仅考虑数值型数据
        na_option: Literal["keep", "top", "bottom"] = "keep",  # 处理缺失值的方式，'keep' 表示保留在当前位置
        ascending: bool = True,  # 是否按升序排列
        pct: bool = False,  # 是否计算百分位排名
    ):
    
    # 使用文档字符串 _shared_docs["compare"] 和 klass=_shared_doc_kwargs["klass"] 注释 compare 方法
    @doc(_shared_docs["compare"], klass=_shared_doc_kwargs["klass"])
    # 定义一个方法 compare，用于比较当前对象与另一个对象的内容
    def compare(
        self,
        other: Self,  # 另一个用于比较的对象
        align_axis: Axis = 1,  # 对齐比较的轴，默认为 1，表示按列对齐
        keep_shape: bool = False,  # 是否保持形状一致
        keep_equal: bool = False,  # 是否要求对象相等
        result_names: Suffixes = ("self", "other"),  # 指定结果的名称后缀
    ):
    ):
        # 如果对象类型不同，则抛出类型错误异常
        if type(self) is not type(other):
            cls_self, cls_other = type(self).__name__, type(other).__name__
            raise TypeError(
                f"can only compare '{cls_self}' (not '{cls_other}') with '{cls_self}'"
            )

        # 计算掩码以排除 NaN 值的影响，同时确保只比较非 NaN 值
        mask = ~((self == other) | (self.isna() & other.isna()))  # type: ignore[operator]
        mask.fillna(True, inplace=True)

        # 如果不保留相等的值，则使用掩码过滤 self 和 other 对象
        if not keep_equal:
            self = self.where(mask)
            other = other.where(mask)

        # 如果不保留相同的形状，则根据对象类型调整 self 和 other
        if not keep_shape:
            if isinstance(self, ABCDataFrame):
                cmask = mask.any()
                rmask = mask.any(axis=1)
                self = self.loc[rmask, cmask]
                other = other.loc[rmask, cmask]
            else:
                self = self[mask]
                other = other[mask]

        # 如果 result_names 不是元组，则抛出类型错误异常
        if not isinstance(result_names, tuple):
            raise TypeError(
                f"Passing 'result_names' as a {type(result_names)} is not "
                "supported. Provide 'result_names' as a tuple instead."
            )

        # 确定对齐的轴
        if align_axis in (1, "columns"):  # This is needed for Series
            axis = 1
        else:
            axis = self._get_axis_number(align_axis)

        # 合并 self 和 other 对象，生成 diff 对象
        diff = concat(
            [self, other],  # type: ignore[list-item]
            axis=axis,
            keys=result_names,
        )

        # 如果轴超过对象的维度数，则直接返回 diff
        if axis >= self.ndim:
            return diff

        # 获取轴对象，并将索引名设置为位置以避免混淆
        ax = diff._get_axis(axis)
        ax_names = np.array(ax.names)
        ax.names = np.arange(len(ax_names))

        # 调整层级顺序以保持结构组织性
        order = list(range(1, ax.nlevels)) + [0]
        if isinstance(diff, ABCDataFrame):
            diff = diff.reorder_levels(order, axis=axis)
        else:
            diff = diff.reorder_levels(order)

        # 恢复原始的索引名顺序
        diff._get_axis(axis=axis).names = ax_names[order]

        # 重新排列轴以保持组织性
        indices = (
            np.arange(diff.shape[axis])
            .reshape([2, diff.shape[axis] // 2])
            .T.reshape(-1)
        )
        diff = diff.take(indices, axis=axis)

        return diff

    @final
    @doc(
        klass=_shared_doc_kwargs["klass"],
        axes_single_arg=_shared_doc_kwargs["axes_single_arg"],
    )
    def align(
        self,
        other: NDFrameT,
        join: AlignJoin = "outer",
        axis: Axis | None = None,
        level: Level | None = None,
        copy: bool | lib.NoDefault = lib.no_default,
        fill_value: Hashable | None = None,
    @final
    # 对齐当前对象和另一个 DataFrame 对象，返回对齐后的结果和对应的索引
    def _align_frame(
        self,
        other: DataFrame,
        join: AlignJoin = "outer",
        axis: Axis | None = None,
        level=None,
        fill_value=None,
    ) -> tuple[Self, DataFrame, Index | None]:
        # 默认初始化为 None
        join_index, join_columns = None, None
        ilidx, iridx = None, None
        clidx, cridx = None, None

        # 检查是否当前对象为 Series 类型
        is_series = isinstance(self, ABCSeries)

        # 如果轴为 None 或者为 0，并且当前对象的索引与另一个 DataFrame 的索引不相等
        if (axis is None or axis == 0) and not self.index.equals(other.index):
            # 进行索引的合并操作，并返回对齐后的索引及对应的索引器
            join_index, ilidx, iridx = self.index.join(
                other.index, how=join, level=level, return_indexers=True
            )

        # 如果轴为 None 或者为 1，并且当前对象不是 Series 类型，并且当前对象的列与另一个 DataFrame 的列不相等
        if (
            (axis is None or axis == 1)
            and not is_series
            and not self.columns.equals(other.columns)
        ):
            # 进行列的合并操作，并返回对齐后的列及对应的索引器
            join_columns, clidx, cridx = self.columns.join(
                other.columns, how=join, level=level, return_indexers=True
            )

        # 如果当前对象是 Series 类型
        if is_series:
            # 设置重新索引器字典，用于重建索引
            reindexers = {0: [join_index, ilidx]}
        else:
            # 设置重新索引器字典，分别用于索引和列的重建
            reindexers = {0: [join_index, ilidx], 1: [join_columns, clidx]}

        # 对当前对象进行重新索引操作，使用给定的索引器和填充值，允许重复值
        left = self._reindex_with_indexers(
            reindexers, fill_value=fill_value, allow_dups=True
        )

        # 确保 other 总是 DataFrame 类型，对其进行相同的重新索引操作
        right = other._reindex_with_indexers(
            {0: [join_index, iridx], 1: [join_columns, cridx]},
            fill_value=fill_value,
            allow_dups=True,
        )

        # 返回对齐后的左侧对象、右侧对象和对齐后的索引
        return left, right, join_index

    # 最终版本的方法，用于对齐当前对象和另一个 Series 对象
    @final
    def _align_series(
        self,
        other: Series,
        join: AlignJoin = "outer",
        axis: Axis | None = None,
        level=None,
        fill_value=None,
    ) -> tuple[Self, Series, Index | None]:
        # 判断当前对象是否为 Series 类型
        is_series = isinstance(self, ABCSeries)

        # 如果不是 Series 且未指定 axis 或 axis 不在 [None, 0, 1] 中，则抛出异常
        if (not is_series and axis is None) or axis not in [None, 0, 1]:
            raise ValueError("Must specify axis=0 or 1")

        # 如果是 Series 且 axis 为 1，则抛出异常，因为不能将一个 Series 对齐到另一个 Series 上的非 0 轴
        if is_series and axis == 1:
            raise ValueError("cannot align series to a series other than axis 0")

        # 如果 axis 为 0，进行以下操作
        # series/series compat, other must always be a Series
        if not axis:
            # 如果当前对象的索引与 other 的索引相等
            if self.index.equals(other.index):
                join_index, lidx, ridx = None, None, None
            else:
                # 将当前对象和 other 的索引进行合并，返回合并后的索引及其对应的位置索引
                join_index, lidx, ridx = self.index.join(
                    other.index, how=join, level=level, return_indexers=True
                )

            # 如果当前对象是 Series
            if is_series:
                # 使用合并后的索引和 lidx 重新索引当前对象
                left = self._reindex_indexer(join_index, lidx)
            # 如果 lidx 或 join_index 为 None，则直接复制当前对象
            elif lidx is None or join_index is None:
                left = self.copy(deep=False)
            else:
                # 根据合并后的索引和 lidx 重新索引当前对象的数据管理器
                new_mgr = self._mgr.reindex_indexer(join_index, lidx, axis=1)
                left = self._constructor_from_mgr(new_mgr, axes=new_mgr.axes)

            # 使用合并后的索引和 ridx 重新索引 other 对象
            right = other._reindex_indexer(join_index, ridx)

        # 如果 axis 不为 0，进行以下操作
        else:
            # 获取当前对象的数据管理器和其第二个轴的索引
            fdata = self._mgr
            join_index = self.axes[1]
            lidx, ridx = None, None
            # 如果当前对象的第二个轴的索引与 other 的索引不相等
            if not join_index.equals(other.index):
                # 将当前对象的第二个轴的索引和 other 的索引进行合并，返回合并后的索引及其对应的位置索引
                join_index, lidx, ridx = join_index.join(
                    other.index, how=join, level=level, return_indexers=True
                )

            # 如果 lidx 不为 None，根据当前对象的数据管理器的轴重新索引
            if lidx is not None:
                bm_axis = self._get_block_manager_axis(1)
                fdata = fdata.reindex_indexer(join_index, lidx, axis=bm_axis)

            # 根据重新索引后的数据管理器创建一个新的对象
            left = self._constructor_from_mgr(fdata, axes=fdata.axes)

            # 如果 ridx 为 None，则直接复制 other 对象
            if ridx is None:
                right = other.copy(deep=False)
            else:
                # 根据合并后的索引和 ridx 重新索引 other 对象
                right = other.reindex(join_index, level=level)

        # 填充缺失值
        fill_na = notna(fill_value)
        if fill_na:
            left = left.fillna(fill_value)
            right = right.fillna(fill_value)

        # 返回左右两个对象及合并后的索引
        return left, right, join_index
    # 定义 where 方法的签名，用于条件性地选择数据
    def where(
        self,
        cond,
        other=...,
        *,
        inplace: Literal[False] = ...,
        axis: Axis | None = ...,
        level: Level = ...,
    ) -> Self:
        ...

    # where 方法的重载，当 inplace 参数为 True 时，表示在原地进行操作
    @overload
    def where(
        self,
        cond,
        other=...,
        *,
        inplace: Literal[True],
        axis: Axis | None = ...,
        level: Level = ...,
    ) -> None:
        ...

    # where 方法的另一重载，当 inplace 参数为 bool 值时，返回自身或者 None
    @overload
    def where(
        self,
        cond,
        other=...,
        *,
        inplace: bool = ...,
        axis: Axis | None = ...,
        level: Level = ...,
    ) -> Self | None:
        ...

    # 装饰器标记为最终方法，并提供文档化注释信息
    @final
    @doc(
        klass=_shared_doc_kwargs["klass"],
        cond="True",
        cond_rev="False",
        name="where",
        name_other="mask",
    )
    # where 方法的实现，根据条件从数组中选择元素或进行置空处理
    def where(
        self,
        cond,
        other=np.nan,
        *,
        inplace: bool = False,
        axis: Axis | None = None,
        level: Level | None = None,
    ):
        # 确保 inplace 参数为布尔类型
        inplace = validate_bool_kwarg(inplace, "inplace")
        # 如果 inplace 为 True，则进行后续的原地修改验证
        if inplace:
            # 在非 PyPy 环境下，检查对象的引用计数是否低于阈值，发出警告
            if not PYPY:
                if sys.getrefcount(self) <= REF_COUNT:
                    warnings.warn(
                        _chained_assignment_method_msg,
                        ChainedAssignmentError,
                        stacklevel=2,
                    )

        # 对 cond 和 other 应用可能的函数
        cond = common.apply_if_callable(cond, self)
        other = common.apply_if_callable(other, self)

        # 见 gh-21891，确保 cond 对象具有 __invert__ 方法，否则转换为 NumPy 数组
        if not hasattr(cond, "__invert__"):
            cond = np.array(cond)

        # 调用私有方法 _where，根据条件执行选择操作
        return self._where(
            ~cond,
            other=other,
            inplace=inplace,
            axis=axis,
            level=level,
        )

    # mask 方法的签名，用于根据条件屏蔽数据
    @overload
    def mask(
        self,
        cond,
        other=...,
        *,
        inplace: Literal[False] = ...,
        axis: Axis | None = ...,
        level: Level = ...,
    ) -> Self:
        ...

    # mask 方法的重载，当 inplace 参数为 True 时，表示在原地进行操作
    @overload
    def mask(
        self,
        cond,
        other=...,
        *,
        inplace: Literal[True],
        axis: Axis | None = ...,
        level: Level = ...,
    ) -> None:
        ...

    # mask 方法的另一重载，当 inplace 参数为 bool 值时，返回自身或者 None
    @overload
    def mask(
        self,
        cond,
        other=...,
        *,
        inplace: bool = ...,
        axis: Axis | None = ...,
        level: Level = ...,
    ) -> Self | None:
        ...

    # 装饰器标记为最终方法，并提供文档化注释信息
    @final
    @doc(
        where,
        klass=_shared_doc_kwargs["klass"],
        cond="False",
        cond_rev="True",
        name="mask",
        name_other="where",
    )
    # mask 方法的实现，根据条件屏蔽数组中的元素
    def mask(
        self,
        cond,
        other=lib.no_default,
        *,
        inplace: bool = False,
        axis: Axis | None = None,
        level: Level | None = None,
    ) -> Self | None:
        # 确保 inplace 参数为布尔类型
        inplace = validate_bool_kwarg(inplace, "inplace")
        # 如果 inplace 为 True，则进行后续的原地修改验证
        if inplace:
            # 在非 PyPy 环境下，检查对象的引用计数是否低于阈值，发出警告
            if not PYPY:
                if sys.getrefcount(self) <= REF_COUNT:
                    warnings.warn(
                        _chained_assignment_method_msg,
                        ChainedAssignmentError,
                        stacklevel=2,
                    )

        # 对 cond 和 other 应用可能的函数
        cond = common.apply_if_callable(cond, self)
        other = common.apply_if_callable(other, self)

        # 返回调用私有方法 _where 的结果，根据条件进行屏蔽操作
        return self._where(
            cond,
            other=other,
            inplace=inplace,
            axis=axis,
            level=level,
        )

    # shift 方法的签名，用于按指定周期移动数据
    @doc(klass=_shared_doc_kwargs["klass"])
    def shift(
        self,
        periods: int | Sequence[int] = 1,
        freq=None,
        axis: Axis = 0,
        fill_value: Hashable = lib.no_default,
        suffix: str | None = None,
    ):
        ...
    def _shift_with_freq(self, periods: int, axis: int, freq) -> Self:
        # 定义一个函数 `_shift_with_freq`，用于在给定轴向上按指定频率进行数据位移操作
        # 查看 shift.__doc__ 获取更多信息
        # 当提供了 freq 参数时，索引会被位移，数据不会改变
        index = self._get_axis(axis)

        if freq == "infer":
            # 如果 freq 参数为 "infer"，尝试从索引中获取频率信息
            freq = getattr(index, "freq", None)

            if freq is None:
                # 如果索引中未设置频率信息，则尝试获取推断的频率信息
                freq = getattr(index, "inferred_freq", None)

            if freq is None:
                # 如果仍然无法获取频率信息，则抛出 ValueError 异常
                msg = "Freq was not set in the index hence cannot be inferred"
                raise ValueError(msg)

        elif isinstance(freq, str):
            # 如果 freq 是字符串类型，则尝试根据字符串转换为频率偏移量
            is_period = isinstance(index, PeriodIndex)
            freq = to_offset(freq, is_period=is_period)

        if isinstance(index, PeriodIndex):
            # 如果索引是 PeriodIndex 类型，则获取原始频率信息
            orig_freq = to_offset(index.freq)
            if freq != orig_freq:
                # 如果给定的频率不匹配原始频率，则引发 ValueError 异常
                assert orig_freq is not None  # 用于类型检查工具 mypy 的断言
                raise ValueError(
                    f"Given freq {PeriodDtype(freq)._freqstr} "
                    f"does not match PeriodIndex freq "
                    f"{PeriodDtype(orig_freq)._freqstr}"
                )
            # 在 PeriodIndex 上进行指定周期的位移操作
            new_ax: Index = index.shift(periods)
        else:
            # 在普通索引上进行指定周期和频率的位移操作
            new_ax = index.shift(periods, freq)

        # 将结果设置回对象的指定轴上，并使用 __finalize__ 方法保留对象的类型信息
        result = self.set_axis(new_ax, axis=axis)
        return result.__finalize__(self, method="shift")

    @final
    def truncate(
        self,
        before=None,
        after=None,
        axis: Axis | None = None,
        copy: bool | lib.NoDefault = lib.no_default,
    # 定义了一个函数 `truncate`，用于截断数据框或者序列对象的索引或列
    # 参数 before 和 after 用于指定截断的起始和结束位置
    # axis 用于指定操作的轴向，默认为 None 表示索引轴
    # copy 用于指定是否复制数据，默认为 lib.no_default

    @final
    @doc(klass=_shared_doc_kwargs["klass"])
    def tz_convert(
        self,
        tz,
        axis: Axis = 0,
        level=None,
        copy: bool | lib.NoDefault = lib.no_default,
    # 定义了一个函数 `tz_convert`，用于将对象的时区转换为指定时区
    # 参数 tz 用于指定目标时区
    # axis 用于指定操作的轴向，默认为 0 表示索引轴
    # level 用于指定多级索引中的级别
    # copy 用于指定是否复制数据，默认为 lib.no_default

    @final
    @doc(klass=_shared_doc_kwargs["klass"])
    def tz_localize(
        self,
        tz,
        axis: Axis = 0,
        level=None,
        copy: bool | lib.NoDefault = lib.no_default,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    # 定义了一个函数 `tz_localize`，用于为对象的时间数据设置本地化时区信息
    # 参数 tz 用于指定本地化的目标时区
    # axis 用于指定操作的轴向，默认为 0 表示索引轴
    # level 用于指定多级索引中的级别
    # copy 用于指定是否复制数据，默认为 lib.no_default
    # ambiguous 和 nonexistent 用于控制本地化过程中遇到歧义和不存在时间点时的处理方式

    # ----------------------------------------------------------------------
    # 数值方法

    @final
    def describe(
        self,
        percentiles=None,
        include=None,
        exclude=None,
    # 定义了一个函数 `describe`，用于生成数据框或者序列对象的描述性统计信息
    # 参数 percentiles 用于指定百分位数
    # include 用于指定要包含的数据类型
    # exclude 用于指定要排除的数据类型

    @final
    def pct_change(
        self,
        periods: int = 1,
        fill_method: None = None,
        freq=None,
        **kwargs,
    # 定义了一个函数 `pct_change`，用于计算数据框或者序列对象的百分比变化
    # 参数 periods 用于指定计算变化的时间跨度
    # fill_method 用于指定缺失值的填充方法，默认为 None
    # freq 用于指定时间频率

    @final
    def _logical_func(
        self,
        name: str,
        func,
        axis: Axis | None = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs,
    # 定义了一个函数 `_logical_func`，用于在数据框或者序列对象上应用逻辑函数
    # 参数 name 用于指定函数的名称
    # func 用于指定要应用的函数对象
    # axis 用于指定操作的轴向，默认为 0 表示索引轴
    # bool_only 用于指定是否只适用于布尔型数据，默认为 False
    # skipna 用于指定是否跳过 NaN 值，默认为 True
    ) -> Series | bool:
        nv.validate_logical_func((), kwargs, fname=name)
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)

        if self.ndim > 1 and axis is None:
            # 如果数据维度大于1且未指定轴向，则简化为对每个轴进行逻辑函数计算，以简化DataFrame._reduce的操作
            res = self._logical_func(
                name, func, axis=0, bool_only=bool_only, skipna=skipna, **kwargs
            )
            # 错误：类型 "Series | bool" 中的项 "bool" 没有 "_logical_func" 属性
            return res._logical_func(  # 错误：忽略联合类型的属性访问
                name, func, skipna=skipna, **kwargs
            )
        elif axis is None:
            axis = 0

        if (
            self.ndim > 1
            and axis == 1
            and len(self._mgr.blocks) > 1
            # TODO(EA2D): 不需要特殊处理
            and all(block.values.ndim == 2 for block in self._mgr.blocks)
            and not kwargs
        ):
            # 快速路径，避免潜在的昂贵转置操作
            obj = self
            if bool_only:
                obj = self._get_bool_data()
            return obj._reduce_axis1(name, func, skipna=skipna)

        return self._reduce(
            func,
            name=name,
            axis=axis,
            skipna=skipna,
            numeric_only=bool_only,
            filter_type="bool",
        )

    def any(
        self,
        *,
        axis: Axis | None = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs,
    ) -> Series | bool:
        return self._logical_func(
            "any", nanops.nanany, axis, bool_only, skipna, **kwargs
        )

    def all(
        self,
        *,
        axis: Axis = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs,
    ) -> Series | bool:
        return self._logical_func(
            "all", nanops.nanall, axis, bool_only, skipna, **kwargs
        )

    @final
    def _accum_func(
        self,
        name: str,
        func,
        axis: Axis | None = None,
        skipna: bool = True,
        *args,
        **kwargs,
    ):
        # 验证并获取处理跳过 NA 值的累积函数
        skipna = nv.validate_cum_func_with_skipna(skipna, args, kwargs, name)
        # 如果未指定 axis，则设定为默认的第 0 轴
        if axis is None:
            axis = 0
        else:
            # 否则，根据 axis 参数获取轴的编号
            axis = self._get_axis_number(axis)

        # 如果 axis 等于 1，则调用转置后的累积函数，并返回其转置结果
        if axis == 1:
            return self.T._accum_func(
                name,
                func,
                axis=0,
                skipna=skipna,
                *args,  # noqa: B026
                **kwargs,
            ).T

        # 定义块累积函数，处理块数据的累积操作
        def block_accum_func(blk_values):
            # 如果 blk_values 具有属性 "T"，则进行转置操作
            values = blk_values.T if hasattr(blk_values, "T") else blk_values

            result: np.ndarray | ExtensionArray
            # 如果 values 是 ExtensionArray 类型，则调用其 _accumulate 方法
            if isinstance(values, ExtensionArray):
                result = values._accumulate(name, skipna=skipna, **kwargs)
            else:
                # 否则，调用 nanops.na_accum_func 处理 NaN 安全的累积函数
                result = nanops.na_accum_func(values, func, skipna=skipna)

            # 如果结果具有属性 "T"，则进行转置操作
            result = result.T if hasattr(result, "T") else result
            return result

        # 对数据块应用块累积函数，并获取结果
        result = self._mgr.apply(block_accum_func)

        # 根据结果创建新的对象，同时设置方法名，并返回最终结果
        return self._constructor_from_mgr(result, axes=result.axes).__finalize__(
            self, method=name
        )

    # 计算累积最大值
    def cummax(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Self:
        return self._accum_func(
            "cummax", np.maximum.accumulate, axis, skipna, *args, **kwargs
        )

    # 计算累积最小值
    def cummin(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Self:
        return self._accum_func(
            "cummin", np.minimum.accumulate, axis, skipna, *args, **kwargs
        )

    # 计算累积和
    def cumsum(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Self:
        return self._accum_func("cumsum", np.cumsum, axis, skipna, *args, **kwargs)

    # 计算累积乘积
    def cumprod(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Self:
        return self._accum_func("cumprod", np.cumprod, axis, skipna, *args, **kwargs)

    # 定义一个带有 ddof 参数的统计函数
    @final
    def _stat_function_ddof(
        self,
        name: str,
        func,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | float:
        # 验证统计函数中的 ddof 参数
        nv.validate_stat_ddof_func((), kwargs, fname=name)
        # 验证并设置 skipna 参数的布尔值
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)

        # 调用 _reduce 方法执行函数计算，并返回结果
        return self._reduce(
            func, name, axis=axis, numeric_only=numeric_only, skipna=skipna, ddof=ddof
        )

    # 计算标准误差
    def sem(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | float:
        return self._stat_function_ddof(
            "sem", nanops.nansem, axis, skipna, ddof, numeric_only, **kwargs
        )

    # 计算方差
    def var(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | float:
        return self._stat_function_ddof(
            "var", nanops.nanvar, axis, skipna, ddof, numeric_only, **kwargs
        )
    # 使用指定的统计函数计算 Series 或 float 值的标准差
    def std(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | float:
        return self._stat_function_ddof(
            "std", nanops.nanstd, axis, skipna, ddof, numeric_only, **kwargs
        )

    # 最终函数，用于执行给定统计函数的计算，并返回结果
    @final
    def _stat_function(
        self,
        name: str,
        func,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        # 确保统计函数名在支持的列表中
        assert name in ["median", "mean", "min", "max", "kurt", "skew"], name
        # 验证参数与统计函数的匹配性
        nv.validate_func(name, (), kwargs)

        # 验证是否应该跳过 NaN 值
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)

        # 使用 reduce 方法执行统计函数的计算
        return self._reduce(
            func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only
        )

    # 计算 Series 或 float 值的最小值
    def min(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        return self._stat_function(
            "min",
            nanops.nanmin,
            axis,
            skipna,
            numeric_only,
            **kwargs,
        )

    # 计算 Series 或 float 值的最大值
    def max(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        return self._stat_function(
            "max",
            nanops.nanmax,
            axis,
            skipna,
            numeric_only,
            **kwargs,
        )

    # 计算 Series 或 float 值的均值
    def mean(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | float:
        return self._stat_function(
            "mean", nanops.nanmean, axis, skipna, numeric_only, **kwargs
        )

    # 计算 Series 或 float 值的中位数
    def median(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | float:
        return self._stat_function(
            "median", nanops.nanmedian, axis, skipna, numeric_only, **kwargs
        )

    # 计算 Series 或 float 值的偏度
    def skew(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | float:
        return self._stat_function(
            "skew", nanops.nanskew, axis, skipna, numeric_only, **kwargs
        )

    # 计算 Series 或 float 值的峰度
    def kurt(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | float:
        return self._stat_function(
            "kurt", nanops.nankurt, axis, skipna, numeric_only, **kwargs
        )

    # kurtosis 方法的别名，计算 Series 或 float 值的峰度
    kurtosis = kurt

    # 最终函数，带有最小计数限制的统计函数执行，计算 Series 或 float 值的结果
    @final
    def _min_count_stat_function(
        self,
        name: str,
        func,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs,
    ):
        # 确保 name 参数只能是 "sum" 或 "prod"，否则抛出异常
        assert name in ["sum", "prod"], name
        # 使用 nv 对象验证函数调用的合法性
        nv.validate_func(name, (), kwargs)

        # 验证 skipna 参数是否为布尔值，不允许为 None
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)

        # 调用 _reduce 方法进行数据减少操作
        return self._reduce(
            func,
            name=name,
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
        )

    def sum(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs,
    ):
        # 调用 _min_count_stat_function 方法进行 sum 操作
        return self._min_count_stat_function(
            "sum", nanops.nansum, axis, skipna, numeric_only, min_count, **kwargs
        )

    def prod(
        self,
        *,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs,
    ):
        # 调用 _min_count_stat_function 方法进行 prod 操作
        return self._min_count_stat_function(
            "prod",
            nanops.nanprod,
            axis,
            skipna,
            numeric_only,
            min_count,
            **kwargs,
        )

    product = prod

    @final
    @doc(Rolling)
    def rolling(
        self,
        window: int | dt.timedelta | str | BaseOffset | BaseIndexer,
        min_periods: int | None = None,
        center: bool = False,
        win_type: str | None = None,
        on: str | None = None,
        closed: IntervalClosedType | None = None,
        step: int | None = None,
        method: str = "single",
    ) -> Window | Rolling:
        # 如果指定了 win_type 参数，则返回 Window 对象
        if win_type is not None:
            return Window(
                self,
                window=window,
                min_periods=min_periods,
                center=center,
                win_type=win_type,
                on=on,
                closed=closed,
                step=step,
                method=method,
            )

        # 否则返回 Rolling 对象
        return Rolling(
            self,
            window=window,
            min_periods=min_periods,
            center=center,
            win_type=win_type,
            on=on,
            closed=closed,
            step=step,
            method=method,
        )

    @final
    @doc(Expanding)
    def expanding(
        self,
        min_periods: int = 1,
        method: Literal["single", "table"] = "single",
    ) -> Expanding:
        # 返回 Expanding 对象
        return Expanding(self, min_periods=min_periods, method=method)

    @final
    @doc(ExponentialMovingWindow)
    def ewm(
        self,
        com: float | None = None,
        span: float | None = None,
        halflife: float | TimedeltaConvertibleTypes | None = None,
        alpha: float | None = None,
        min_periods: int | None = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        times: np.ndarray | DataFrame | Series | None = None,
        method: Literal["single", "table"] = "single",
    ) -> ExponentialMovingWindow:
        return ExponentialMovingWindow(
            self,
            com=com,
            span=span,
            halflife=halflife,
            alpha=alpha,
            min_periods=min_periods,
            adjust=adjust,
            ignore_na=ignore_na,
            times=times,
            method=method,
        )

    # ----------------------------------------------------------------------
    # Arithmetic Methods

    @final
    def _inplace_method(self, other, op) -> Self:
        """
        Wrap arithmetic method to operate inplace.
        """
        # 调用给定的操作函数，并将结果存储在result中
        result = op(self, other)

        # 确保结果与输入对象保持对齐，进行原地更新
        self._update_inplace(result.reindex_like(self))
        return self

    @final
    def __iadd__(self, other) -> Self:
        # error: Unsupported left operand type for + ("Type[NDFrame]")
        # 调用_inplace_method方法，使用加法操作符
        return self._inplace_method(other, type(self).__add__)  # type: ignore[operator]

    @final
    def __isub__(self, other) -> Self:
        # error: Unsupported left operand type for - ("Type[NDFrame]")
        # 调用_inplace_method方法，使用减法操作符
        return self._inplace_method(other, type(self).__sub__)  # type: ignore[operator]

    @final
    def __imul__(self, other) -> Self:
        # error: Unsupported left operand type for * ("Type[NDFrame]")
        # 调用_inplace_method方法，使用乘法操作符
        return self._inplace_method(other, type(self).__mul__)  # type: ignore[operator]

    @final
    def __itruediv__(self, other) -> Self:
        # error: Unsupported left operand type for / ("Type[NDFrame]")
        # 调用_inplace_method方法，使用真除法操作符
        return self._inplace_method(
            other,
            type(self).__truediv__,  # type: ignore[operator]
        )

    @final
    def __ifloordiv__(self, other) -> Self:
        # error: Unsupported left operand type for // ("Type[NDFrame]")
        # 调用_inplace_method方法，使用整除操作符
        return self._inplace_method(
            other,
            type(self).__floordiv__,  # type: ignore[operator]
        )

    @final
    def __imod__(self, other) -> Self:
        # error: Unsupported left operand type for % ("Type[NDFrame]")
        # 调用_inplace_method方法，使用取模操作符
        return self._inplace_method(other, type(self).__mod__)  # type: ignore[operator]

    @final
    def __ipow__(self, other) -> Self:
        # error: Unsupported left operand type for ** ("Type[NDFrame]")
        # 调用_inplace_method方法，使用幂操作符
        return self._inplace_method(other, type(self).__pow__)  # type: ignore[operator]

    @final
    def __iand__(self, other) -> Self:
        # error: Unsupported left operand type for & ("Type[NDFrame]")
        # 调用_inplace_method方法，使用按位与操作符
        return self._inplace_method(other, type(self).__and__)  # type: ignore[operator]

    @final
    def __ior__(self, other) -> Self:
        # 调用_inplace_method方法，使用按位或操作符
        return self._inplace_method(other, type(self).__or__)

    @final
    def __ixor__(self, other) -> Self:
        # error: Unsupported left operand type for ^ ("Type[NDFrame]")
        # 调用_inplace_method方法，使用按位异或操作符
        return self._inplace_method(other, type(self).__xor__)  # type: ignore[operator]

    # ----------------------------------------------------------------------
    # Misc methods
    # 使用 @final 装饰器，表示该方法为最终方法，不能被子类重写
    def _find_valid_index(self, *, how: str) -> Hashable:
        """
        Retrieves the index of the first valid value.

        Parameters
        ----------
        how : {'first', 'last'}
            Use this parameter to change between the first or last valid index.

        Returns
        -------
        idx_first_valid : type of index
            Index of the first valid value, or None if no valid value is found.
        """
        # 获取一个布尔数组，指示哪些位置的值是非缺失的
        is_valid = self.notna().values
        # 调用 find_valid_index 函数，根据 how 参数和 is_valid 数组找到第一个有效值的位置
        idxpos = find_valid_index(how=how, is_valid=is_valid)
        # 如果找不到有效值，则返回 None
        if idxpos is None:
            return None
        # 返回 self.index 中对应的索引位置，即第一个有效值的索引
        return self.index[idxpos]

    # 使用 @final 装饰器，表示该方法为最终方法，不能被子类重写，并且使用文档字符串
    @final
    @doc(position="first", klass=_shared_doc_kwargs["klass"])
    def first_valid_index(self) -> Hashable:
        """
        Return index for first non-missing value or None, if no value is found.

        See the :ref:`User Guide <missing_data>` for more information
        on which values are considered missing.

        Returns
        -------
        type of index
            Index of first non-missing value.

        See Also
        --------
        DataFrame.last_valid_index : Return index for last non-NA value or None, if
            no non-NA value is found.
        Series.last_valid_index : Return index for last non-NA value or None, if no
            non-NA value is found.
        DataFrame.isna : Detect missing values.

        Examples
        --------
        Examples omitted due to length. Refer to docstring for detailed examples.
        """
        # 调用 _find_valid_index 方法，以获取第一个有效值的索引
        return self._find_valid_index(how="first")

    # 使用 @final 装饰器，表示该方法为最终方法，不能被子类重写，并且使用文档字符串
    @final
    @doc(first_valid_index, position="last", klass=_shared_doc_kwargs["klass"])
    # 定义一个方法 `last_valid_index`，返回一个可哈希对象
    def last_valid_index(self) -> Hashable:
        # 调用类中的 `_find_valid_index` 方法，以 "last" 参数调用，寻找最后一个有效索引
        return self._find_valid_index(how="last")
# _num_doc 文档字符串，描述了一个函数的参数和返回值
_num_doc = """
{desc}

Parameters
----------
axis : {axis_descr}
    Axis for the function to be applied on.
    For `Series` this parameter is unused and defaults to 0.

    For DataFrames, specifying ``axis=None`` will apply the aggregation
    across both axes.

    .. versionadded:: 2.0.0

skipna : bool, default True
    Exclude NA/null values when computing the result.
numeric_only : bool, default False
    Include only float, int, boolean columns.

{min_count}\
**kwargs
    Additional keyword arguments to be passed to the function.

Returns
-------
{name1} or scalar\
{see_also}\
{examples}
"""

# _sum_prod_doc 文档字符串，描述了一个函数的参数和返回值
_sum_prod_doc = """
{desc}

Parameters
----------
axis : {axis_descr}
    Axis for the function to be applied on.
    For `Series` this parameter is unused and defaults to 0.

    .. warning::

        The behavior of DataFrame.{name} with ``axis=None`` is deprecated,
        in a future version this will reduce over both axes and return a scalar
        To retain the old behavior, pass axis=0 (or do not pass axis).

    .. versionadded:: 2.0.0

skipna : bool, default True
    Exclude NA/null values when computing the result.
numeric_only : bool, default False
    Include only float, int, boolean columns. Not implemented for Series.

{min_count}\
**kwargs
    Additional keyword arguments to be passed to the function.

Returns
-------
{name1} or scalar\
{see_also}\
{examples}
"""

# _num_ddof_doc 文档字符串，描述了一个函数的参数和返回值
_num_ddof_doc = """
{desc}

Parameters
----------
axis : {axis_descr}
    For `Series` this parameter is unused and defaults to 0.

    .. warning::

        The behavior of DataFrame.{name} with ``axis=None`` is deprecated,
        in a future version this will reduce over both axes and return a scalar
        To retain the old behavior, pass axis=0 (or do not pass axis).

skipna : bool, default True
    Exclude NA/null values. If an entire row/column is NA, the result
    will be NA.
ddof : int, default 1
    Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
    where N represents the number of elements.
numeric_only : bool, default False
    Include only float, int, boolean columns. Not implemented for Series.

Returns
-------
{name1} or {name2} (if level specified) \
{notes}\
{examples}
"""

# _std_notes 注释说明，提供了函数的额外说明信息
_std_notes = """

Notes
-----
To have the same behaviour as `numpy.std`, use `ddof=0` (instead of the
default `ddof=1`)"""

# _std_examples 注释说明，提供了函数的使用示例
_std_examples = """

Examples
--------
>>> df = pd.DataFrame({'person_id': [0, 1, 2, 3],
...                    'age': [21, 25, 62, 43],
...                    'height': [1.61, 1.87, 1.49, 2.01]}
...                   ).set_index('person_id')
>>> df
           age  height
person_id
0           21    1.61
1           25    1.87
2           62    1.49
3           43    2.01

The standard deviation of the columns can be found as follows:

>>> df.std()
age       18.786076
height     0.237417
dtype: float64

Alternatively, `ddof=0` can be set to normalize by N instead of N-1:

>>> df.std(ddof=0)
age       16.269219
height     0.205609

"""
# 定义一个字符串变量，描述数据类型是 float64
dtype: float64"""

# 定义一个字符串变量，包含一些示例用法
_var_examples = """

Examples
--------
>>> df = pd.DataFrame({'person_id': [0, 1, 2, 3],
...                    'age': [21, 25, 62, 43],
...                    'height': [1.61, 1.87, 1.49, 2.01]}
...                   ).set_index('person_id')
>>> df
           age  height
person_id
0           21    1.61
1           25    1.87
2           62    1.49
3           43    2.01

>>> df.var()
age       352.916667
height      0.056367
dtype: float64

Alternatively, ``ddof=0`` can be set to normalize by N instead of N-1:

>>> df.var(ddof=0)
age       264.687500
height      0.042275
dtype: float64"""

# 定义一个字符串变量，包含布尔运算函数的文档字符串
_bool_doc = """
{desc}

Parameters
----------
axis : {{0 or 'index', 1 or 'columns', None}}, default 0
    Indicate which axis or axes should be reduced. For `Series` this parameter
    is unused and defaults to 0.

    * 0 / 'index' : reduce the index, return a Series whose index is the
      original column labels.
    * 1 / 'columns' : reduce the columns, return a Series whose index is the
      original index.
    * None : reduce all axes, return a scalar.

bool_only : bool, default False
    Include only boolean columns. Not implemented for Series.
skipna : bool, default True
    Exclude NA/null values. If the entire row/column is NA and skipna is
    True, then the result will be {empty_value}, as for an empty row/column.
    If skipna is False, then NA are treated as True, because these are not
    equal to zero.
**kwargs : any, default None
    Additional keywords have no effect but might be accepted for
    compatibility with NumPy.

Returns
-------
{name2} or {name1}
    If axis=None, then a scalar boolean is returned.
    Otherwise a Series is returned with index matching the index argument.

{see_also}
{examples}"""

# 定义一个字符串变量，包含布尔运算函数的描述信息
_all_desc = """\
Return whether all elements are True, potentially over an axis.

Returns True unless there at least one element within a series or
along a Dataframe axis that is False or equivalent (e.g. zero or
empty)."""

# 定义一个字符串变量，包含布尔运算函数的示例用法
_all_examples = """\
Examples
--------
**Series**

>>> pd.Series([True, True]).all()
True
>>> pd.Series([True, False]).all()
False
>>> pd.Series([], dtype="float64").all()
True
>>> pd.Series([np.nan]).all()
True
>>> pd.Series([np.nan]).all(skipna=False)
True

**DataFrames**

Create a DataFrame from a dictionary.

>>> df = pd.DataFrame({'col1': [True, True], 'col2': [True, False]})
>>> df
   col1   col2
0  True   True
1  True  False

Default behaviour checks if values in each column all return True.

>>> df.all()
col1     True
col2    False
dtype: bool

Specify ``axis='columns'`` to check if values in each row all return True.

>>> df.all(axis='columns')
0     True
1    False
dtype: bool

Or ``axis=None`` for whether every value is True.

>>> df.all(axis=None)
False
"""

# 定义一个字符串变量，包含布尔运算函数的相关参考信息
_all_see_also = """\
See Also
--------
Series.all : Return True if all elements are True.
DataFrame.any : Return True if one (or more) elements are True.
"""

# 定义一个字符串变量，空白行结束
_cnum_pd_doc = """
# 返回沿着 DataFrame 或 Series 轴的累积 {desc}。

# 返回一个大小相同的 DataFrame 或 Series，其中包含累积的 {desc}。

# 参数
# ----------
# axis : {{0 或 'index'，1 或 'columns'}}，默认为 0
#     轴的索引或名称。0 等效于 None 或 'index'。
#     对于 `Series`，此参数未使用，默认为 0。
# skipna : bool，默认为 True
#     排除 NA/空值。如果整行/整列为 NA，则结果将为 NA。
# numeric_only : bool，默认为 False
#     仅包括 float、int 和 boolean 类型的列。
# *args, **kwargs
#     额外的关键字参数无效，但可能会被 NumPy 接受以保持兼容性。

# 返回
# -------
# {name1} 或 {name2}
#     返回 {name1} 或 {name2} 的累积 {desc}。

# 参见
# --------
# core.window.expanding.Expanding.{accum_func_name} : 类似功能，但忽略 ``NaN`` 值。
# {name2}.{accum_func_name} : 返回 {name2} 轴上的 {desc}。
# {name2}.cummax : 返回 {name2} 轴上的累积最大值。
# {name2}.cummin : 返回 {name2} 轴上的累积最小值。
# {name2}.cumsum : 返回 {name2} 轴上的累积和。
# {name2}.cumprod : 返回 {name2} 轴上的累积乘积。

# {examples}
_cnum_series_doc = """
Return cumulative {desc} over a DataFrame or Series axis.

Returns a DataFrame or Series of the same size containing the cumulative
{desc}.

Parameters
----------
axis : {{0 or 'index', 1 or 'columns'}}, default 0
    The index or the name of the axis. 0 is equivalent to None or 'index'.
    For `Series` this parameter is unused and defaults to 0.
skipna : bool, default True
    Exclude NA/null values. If an entire row/column is NA, the result
    will be NA.
*args, **kwargs
    Additional keywords have no effect but might be accepted for
    compatibility with NumPy.

Returns
-------
{name1} or {name2}
    Return cumulative {desc} of {name1} or {name2}.

See Also
--------
core.window.expanding.Expanding.{accum_func_name} : Similar functionality
    but ignores ``NaN`` values.
{name2}.{accum_func_name} : Return the {desc} over
    {name2} axis.
{name2}.cummax : Return cumulative maximum over {name2} axis.
{name2}.cummin : Return cumulative minimum over {name2} axis.
{name2}.cumsum : Return cumulative sum over {name2} axis.
{name2}.cumprod : Return cumulative product over {name2} axis.

{examples}"""

# 累积最小值的示例
_cummin_examples = """
Examples
--------
**Series**

>>> s = pd.Series([2, np.nan, 5, -1, 0])
>>> s
0    2.0
1    NaN
2    5.0
3   -1.0
4    0.0
dtype: float64

默认情况下，忽略 NA 值。

>>> s.cummin()
0    2.0
1    NaN
2    2.0
3   -1.0
4   -1.0
dtype: float64

要包括 NA 值，请使用 ``skipna=False``

>>> s.cummin(skipna=False)
0    2.0
1    NaN
2    NaN
3    NaN
4    NaN
dtype: float64

**DataFrame**

>>> df = pd.DataFrame([[2.0, 1.0],
...                    [3.0, np.nan],
...                    [1.0, 0.0]],
...                   columns=list('AB'))
>>> df
     A    B
0  2.0  1.0
1  3.0  NaN
2  1.0  0.0


"""
# Examples for calculating cumulative minimum values across rows or columns in a DataFrame or Series.
_cummin_examples = """\
Examples
--------
**DataFrame**

>>> df = pd.DataFrame([[2.0, 1.0],
...                    [3.0, np.nan],
...                    [1.0, 0.0]],
...                   columns=list('AB'))
>>> df
     A    B
0  2.0  1.0
1  3.0  NaN
2  1.0  0.0

By default, iterates over rows and finds the minimum
in each column. This is equivalent to ``axis=None`` or ``axis='index'``.

>>> df.cummin()
     A    B
0  2.0  1.0
1  2.0  NaN
2  1.0  0.0

To iterate over columns and find the minimum in each row,
use ``axis=1``

>>> df.cummin(axis=1)
     A    B
0  2.0  1.0
1  3.0  NaN
2  1.0  0.0
"""

# Examples for calculating cumulative sum values across rows or columns in a DataFrame or Series.
_cumsum_examples = """\
Examples
--------
**Series**

>>> s = pd.Series([2, np.nan, 5, -1, 0])
>>> s
0    2.0
1    NaN
2    5.0
3   -1.0
4    0.0
dtype: float64

By default, NA values are ignored.

>>> s.cumsum()
0    2.0
1    NaN
2    7.0
3    6.0
4    6.0
dtype: float64

To include NA values in the operation, use ``skipna=False``

>>> s.cumsum(skipna=False)
0    2.0
1    NaN
2    NaN
3    NaN
4    NaN
dtype: float64

**DataFrame**

>>> df = pd.DataFrame([[2.0, 1.0],
...                    [3.0, np.nan],
...                    [1.0, 0.0]],
...                   columns=list('AB'))
>>> df
     A    B
0  2.0  1.0
1  3.0  NaN
2  1.0  0.0

By default, iterates over rows and finds the sum
in each column. This is equivalent to ``axis=None`` or ``axis='index'``.

>>> df.cumsum()
     A    B
0  2.0  1.0
1  5.0  NaN
2  6.0  1.0

To iterate over columns and find the sum in each row,
use ``axis=1``

>>> df.cumsum(axis=1)
     A    B
0  2.0  3.0
1  3.0  NaN
2  1.0  1.0
"""

# Examples for calculating cumulative product values across rows or columns in a DataFrame or Series.
_cumprod_examples = """\
Examples
--------
**Series**

>>> s = pd.Series([2, np.nan, 5, -1, 0])
>>> s
0    2.0
1    NaN
2    5.0
3   -1.0
4    0.0
dtype: float64

By default, NA values are ignored.

>>> s.cumprod()
0     2.0
1     NaN
2    10.0
3   -10.0
4    -0.0
dtype: float64

To include NA values in the operation, use ``skipna=False``

>>> s.cumprod(skipna=False)
0    2.0
1    NaN
2    NaN
3    NaN
4    NaN
dtype: float64

**DataFrame**

>>> df = pd.DataFrame([[2.0, 1.0],
...                    [3.0, np.nan],
...                    [1.0, 0.0]],
...                   columns=list('AB'))
>>> df
     A    B
0  2.0  1.0
1  3.0  NaN
2  1.0  0.0

By default, iterates over rows and finds the product
in each column. This is equivalent to ``axis=None`` or ``axis='index'``.

>>> df.cumprod()
     A    B
0  2.0  1.0
1  6.0  NaN
2  6.0  0.0

To iterate over columns and find the product in each row,
use ``axis=1``

>>> df.cumprod(axis=1)
     A    B
0  2.0  2.0
1  3.0  NaN
2  1.0  0.0
"""

# Examples for calculating cumulative maximum values across rows or columns in a DataFrame or Series.
_cummax_examples = """\
Examples
--------
**Series**

>>> s = pd.Series([2, np.nan, 5, -1, 0])
>>> s
0    2.0
1    NaN
2    5.0
3   -1.0
4    0.0
dtype: float64

By default, NA values are ignored.

>>> s.cummax()
0    2.0
1    NaN
2    5.0
3    5.0
4    5.0
dtype: float64

To include NA values in the operation, use ``skipna=False``

>>> s.cummax(skipna=False)
0    2.0
1    NaN
2    NaN
3    NaN
4    NaN
dtype: float64

**DataFrame**

>>> df = pd.DataFrame([[2.0, 1.0],
...                    [3.0, np.nan],
...                    [1.0, 0.0]],
...                   columns=list('AB'))
>>> df
     A    B
0  2.0  1.0
1  3.0  NaN
2  1.0  0.0
"""
By default, iterates over rows and finds the maximum
in each column. This is equivalent to ``axis=None`` or ``axis='index'``.

>>> df.cummax()
     A    B
0  2.0  1.0
1  3.0  NaN
2  3.0  1.0



# 默认情况下，迭代每行并找到每列中的最大值。这相当于 `axis=None` 或 `axis='index'`。
df.cummax()


To iterate over columns and find the maximum in each row,
use ``axis=1``

>>> df.cummax(axis=1)
     A    B
0  2.0  2.0
1  3.0  NaN
2  1.0  1.0



# 若要迭代每列并找到每行中的最大值，请使用 `axis=1`。
df.cummax(axis=1)


"""

_any_see_also = """\
See Also
--------
numpy.any : Numpy version of this method.
Series.any : Return whether any element is True.
Series.all : Return whether all elements are True.
DataFrame.any : Return whether any element is True over requested axis.
DataFrame.all : Return whether all elements are True over requested axis.
"""

_any_desc = """\
Return whether any element is True, potentially over an axis.

Returns False unless there is at least one element within a series or
along a Dataframe axis that is True or equivalent (e.g. non-zero or
non-empty)."""

_any_examples = """\
Examples
--------
**Series**

For Series input, the output is a scalar indicating whether any element
is True.

>>> pd.Series([False, False]).any()
False
>>> pd.Series([True, False]).any()
True
>>> pd.Series([], dtype="float64").any()
False
>>> pd.Series([np.nan]).any()
False
>>> pd.Series([np.nan]).any(skipna=False)
True

**DataFrame**

Whether each column contains at least one True element (the default).

>>> df = pd.DataFrame({"A": [1, 2], "B": [0, 2], "C": [0, 0]})
>>> df
   A  B  C
0  1  0  0
1  2  2  0

>>> df.any()
A     True
B     True
C    False
dtype: bool

Aggregating over the columns.

>>> df = pd.DataFrame({"A": [True, False], "B": [1, 2]})
>>> df
       A  B
0   True  1
1  False  2

>>> df.any(axis='columns')
0    True
1    True
dtype: bool

>>> df = pd.DataFrame({"A": [True, False], "B": [1, 0]})
>>> df
       A  B
0   True  1
1  False  0

>>> df.any(axis='columns')
0    True
1    False
dtype: bool

Aggregating over the entire DataFrame with ``axis=None``.

>>> df.any(axis=None)
True

`any` for an empty DataFrame is an empty Series.

>>> pd.DataFrame([]).any()
Series([], dtype: bool)
"""

_shared_docs["stat_func_example"] = """

Examples
--------
>>> idx = pd.MultiIndex.from_arrays([
...     ['warm', 'warm', 'cold', 'cold'],
...     ['dog', 'falcon', 'fish', 'spider']],
...     names=['blooded', 'animal'])
>>> s = pd.Series([4, 2, 0, 8], name='legs', index=idx)
>>> s
blooded  animal
warm     dog       4
         falcon    2
cold     fish      0
         spider    8
Name: legs, dtype: int64

>>> s.{stat_func}()
{default_output}"""

_sum_examples = _shared_docs["stat_func_example"].format(
    stat_func="sum", verb="Sum", default_output=14, level_output_0=6, level_output_1=8
)

_sum_examples += """

By default, the sum of an empty or all-NA Series is ``0``.

>>> pd.Series([], dtype="float64").sum()  # min_count=0 is the default
0.0

This can be controlled with the ``min_count`` parameter. For example, if
you'd like the sum of an empty series to be NaN, pass ``min_count=1``.
# 设置一个示例字符串，包含关于最大值函数的文档样例，包括函数名称、默认输出、输出级别等信息
_max_examples: str = _shared_docs["stat_func_example"].format(
    stat_func="max", verb="Max", default_output=8, level_output_0=4, level_output_1=8
)

# 设置一个示例字符串，包含关于最小值函数的文档样例，包括函数名称、默认输出、输出级别等信息
_min_examples: str = _shared_docs["stat_func_example"].format(
    stat_func="min", verb="Min", default_output=0, level_output_0=2, level_output_1=0
)

# 设置一个字符串，包含有关于统计函数的额外信息，包括与其他相关函数的链接
_stat_func_see_also = """
See Also
--------
Series.sum : Return the sum.
Series.min : Return the minimum.
Series.max : Return the maximum.
Series.idxmin : Return the index of the minimum.
Series.idxmax : Return the index of the maximum.
DataFrame.sum : Return the sum over the requested axis.
DataFrame.min : Return the minimum over the requested axis.
DataFrame.max : Return the maximum over the requested axis.
DataFrame.idxmin : Return the index of the minimum over the requested axis.
DataFrame.idxmax : Return the index of the maximum over the requested axis."""

# 设置一个字符串，包含有关于乘积函数的示例信息
_prod_examples = """
Examples
--------
By default, the product of an empty or all-NA Series is ``1``

>>> pd.Series([], dtype="float64").prod()
1.0

This can be controlled with the ``min_count`` parameter

>>> pd.Series([], dtype="float64").prod(min_count=1)
nan

Thanks to the ``skipna`` parameter, ``min_count`` handles all-NA and
empty series identically.

>>> pd.Series([np.nan]).prod()
1.0

>>> pd.Series([np.nan]).prod(min_count=1)
nan"""

# 设置一个字符串，包含有关于 min_count 参数的说明
_min_count_stub = """\
min_count : int, default 0
    The required number of valid values to perform the operation. If fewer than
    ``min_count`` non-NA values are present the result will be NA.
"""


def make_doc(name: str, ndim: int) -> str:
    """
    Generate the docstring for a Series/DataFrame reduction.
    """
    # 根据函数名称和维度生成相应的文档内容
    if ndim == 1:
        name1 = "scalar"
        name2 = "Series"
        axis_descr = "{index (0)}"
    else:
        name1 = "Series"
        name2 = "DataFrame"
        axis_descr = "{index (0), columns (1)}"

    # 根据函数名称选择不同的基础文档和描述
    if name == "any":
        base_doc = _bool_doc
        desc = _any_desc
        see_also = _any_see_also
        examples = _any_examples
        kwargs = {"empty_value": "False"}
    elif name == "all":
        base_doc = _bool_doc
        desc = _all_desc
        see_also = _all_see_also
        examples = _all_examples
        kwargs = {"empty_value": "True"}
    elif name == "min":
        base_doc = _num_doc
        desc = (
            "Return the minimum of the values over the requested axis.\n\n"
            "If you want the *index* of the minimum, use ``idxmin``. This is "
            "the equivalent of the ``numpy.ndarray`` method ``argmin``."
        )
        see_also = _stat_func_see_also
        examples = _min_examples
        kwargs = {"min_count": ""}
    elif name == "max":
        # 设置基础文档为数字函数文档
        base_doc = _num_doc
        # 设置描述信息，返回沿请求轴的值的最大值
        desc = (
            "Return the maximum of the values over the requested axis.\n\n"
            "If you want the *index* of the maximum, use ``idxmax``. This is "
            "the equivalent of the ``numpy.ndarray`` method ``argmax``."
        )
        # 设置参见信息，参考统计函数的相关链接
        see_also = _stat_func_see_also
        # 设置示例，使用最大值函数的示例
        examples = _max_examples
        # 设置关键字参数，此处为一个空字符串
        kwargs = {"min_count": ""}

    elif name == "sum":
        # 设置基础文档为求和和积的文档
        base_doc = _sum_prod_doc
        # 设置描述信息，返回沿请求轴的值的总和
        desc = (
            "Return the sum of the values over the requested axis.\n\n"
            "This is equivalent to the method ``numpy.sum``."
        )
        # 设置参见信息，参考统计函数的相关链接
        see_also = _stat_func_see_also
        # 设置示例，使用求和函数的示例
        examples = _sum_examples
        # 设置关键字参数，包括一个最小计数的占位符
        kwargs = {"min_count": _min_count_stub}

    elif name == "prod":
        # 设置基础文档为求和和积的文档
        base_doc = _sum_prod_doc
        # 设置描述信息，返回沿请求轴的值的乘积
        desc = "Return the product of the values over the requested axis."
        # 设置参见信息，参考统计函数的相关链接
        see_also = _stat_func_see_also
        # 设置示例，使用乘积函数的示例
        examples = _prod_examples
        # 设置关键字参数，包括一个最小计数的占位符
        kwargs = {"min_count": _min_count_stub}

    elif name == "median":
        # 设置基础文档为数字函数文档
        base_doc = _num_doc
        # 设置描述信息，返回沿请求轴的值的中位数
        desc = "Return the median of the values over the requested axis."
        # 设置参见信息，参考统计函数的相关链接
        see_also = _stat_func_see_also
        # 设置示例，包括多个使用中位数函数的示例
        examples = """

            Examples
            --------
            >>> s = pd.Series([1, 2, 3])
            >>> s.median()
            2.0

            With a DataFrame

            >>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, index=['tiger', 'zebra'])
            >>> df
                   a   b
            tiger  1   2
            zebra  2   3
            >>> df.median()
            a   1.5
            b   2.5
            dtype: float64

            Using axis=1

            >>> df.median(axis=1)
            tiger   1.5
            zebra   2.5
            dtype: float64

            In this case, `numeric_only` should be set to `True`
            to avoid getting an error.

            >>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']},
            ...                   index=['tiger', 'zebra'])
            >>> df.median(numeric_only=True)
            a   1.5
            dtype: float64"""
        # 设置关键字参数，此处为一个空字符串
        kwargs = {"min_count": ""}
    # 如果函数参数 name 等于 "mean"
    elif name == "mean":
        # 将基本文档设为数值函数文档
        base_doc = _num_doc
        # 描述设为"返回沿请求的轴上值的均值。"
        desc = "Return the mean of the values over the requested axis."
        # 参见相关统计函数的链接
        see_also = _stat_func_see_also
        # 设定示例代码块
        examples = """

            Examples
            --------
            >>> s = pd.Series([1, 2, 3])
            >>> s.mean()
            2.0

            With a DataFrame

            >>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, index=['tiger', 'zebra'])
            >>> df
                   a   b
            tiger  1   2
            zebra  2   3
            >>> df.mean()
            a   1.5
            b   2.5
            dtype: float64

            Using axis=1

            >>> df.mean(axis=1)
            tiger   1.5
            zebra   2.5
            dtype: float64

            In this case, `numeric_only` should be set to `True` to avoid
            getting an error.

            >>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']},
            ...                   index=['tiger', 'zebra'])
            >>> df.mean(numeric_only=True)
            a   1.5
            dtype: float64"""
        # 设定关键字参数字典，此处为空字符串
        kwargs = {"min_count": ""}

    # 如果函数参数 name 等于 "var"
    elif name == "var":
        # 将基本文档设为数值函数的偏差修正文档
        base_doc = _num_ddof_doc
        # 描述设为"返回请求轴上的无偏方差。\n\n默认情况下通过 N-1 进行归一化。可以使用 ddof 参数进行修改。"
        desc = (
            "Return unbiased variance over requested axis.\n\nNormalized by "
            "N-1 by default. This can be changed using the ddof argument."
        )
        # 示例设定为预定义的方差示例
        examples = _var_examples
        # 参见为空字符串
        see_also = ""
        # 设定关键字参数字典，带有空的 "notes" 键
        kwargs = {"notes": ""}

    # 如果函数参数 name 等于 "std"
    elif name == "std":
        # 将基本文档设为数值函数的偏差修正文档
        base_doc = _num_ddof_doc
        # 描述设为"返回请求轴上的样本标准差。\n\n默认情况下通过 N-1 进行归一化。可以使用 ddof 参数进行修改。"
        desc = (
            "Return sample standard deviation over requested axis."
            "\n\nNormalized by N-1 by default. This can be changed using the "
            "ddof argument."
        )
        # 示例设定为预定义的标准差示例
        examples = _std_examples
        # 参见为空字符串
        see_also = ""
        # 设定关键字参数字典，带有预定义的 "_std_notes" 键
        kwargs = {"notes": _std_notes}
    elif name == "sem":
        # 使用 _num_ddof_doc 作为基础文档
        base_doc = _num_ddof_doc
        # 描述信息，返回沿请求轴的无偏均值标准误差
        desc = (
            "Return unbiased standard error of the mean over requested "
            "axis.\n\nNormalized by N-1 by default. This can be changed "
            "using the ddof argument"
        )
        # 示例代码，演示如何使用 sem 函数计算标准误差
        examples = """

            Examples
            --------
            >>> s = pd.Series([1, 2, 3])
            >>> s.sem().round(6)
            0.57735

            With a DataFrame

            >>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, index=['tiger', 'zebra'])
            >>> df
                   a   b
            tiger  1   2
            zebra  2   3
            >>> df.sem()
            a   0.5
            b   0.5
            dtype: float64

            Using axis=1

            >>> df.sem(axis=1)
            tiger   0.5
            zebra   0.5
            dtype: float64

            In this case, `numeric_only` should be set to `True`
            to avoid getting an error.

            >>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']},
            ...                   index=['tiger', 'zebra'])
            >>> df.sem(numeric_only=True)
            a   0.5
            dtype: float64"""
        # 参见部分留空
        see_also = ""
        # 关键字参数，暂时没有特定的说明
        kwargs = {"notes": ""}

    elif name == "skew":
        # 使用 _num_doc 作为基础文档
        base_doc = _num_doc
        # 描述信息，返回沿请求轴的无偏偏度
        desc = "Return unbiased skew over requested axis.\n\nNormalized by N-1."
        # 示例代码，演示如何使用 skew 函数计算偏度
        examples = """

            Examples
            --------
            >>> s = pd.Series([1, 2, 3])
            >>> s.skew()
            0.0

            With a DataFrame

            >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4], 'c': [1, 3, 5]},
            ...                   index=['tiger', 'zebra', 'cow'])
            >>> df
                    a   b   c
            tiger   1   2   1
            zebra   2   3   3
            cow     3   4   5
            >>> df.skew()
            a   0.0
            b   0.0
            c   0.0
            dtype: float64

            Using axis=1

            >>> df.skew(axis=1)
            tiger   1.732051
            zebra  -1.732051
            cow     0.000000
            dtype: float64

            In this case, `numeric_only` should be set to `True` to avoid
            getting an error.

            >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['T', 'Z', 'X']},
            ...                   index=['tiger', 'zebra', 'cow'])
            >>> df.skew(numeric_only=True)
            a   0.0
            dtype: float64"""
        # 参见部分留空
        see_also = ""
        # 关键字参数，min_count 暂时没有特定的说明
        kwargs = {"min_count": ""}
    elif name == "kurt":
        # 如果函数名称为 "kurt"，选择合适的文档基础模板 _num_doc
        base_doc = _num_doc
        # 描述为返回请求轴上的无偏峰度
        desc = (
            "Return unbiased kurtosis over requested axis.\n\n"
            "Kurtosis obtained using Fisher's definition of\n"
            "kurtosis (kurtosis of normal == 0.0). Normalized "
            "by N-1."
        )
        # 没有相关内容参见
        see_also = ""
        # 展示例子的字符串
        examples = """

            Examples
            --------
            >>> s = pd.Series([1, 2, 2, 3], index=['cat', 'dog', 'dog', 'mouse'])
            >>> s
            cat    1
            dog    2
            dog    2
            mouse  3
            dtype: int64
            >>> s.kurt()
            1.5

            With a DataFrame

            >>> df = pd.DataFrame({'a': [1, 2, 2, 3], 'b': [3, 4, 4, 4]},
            ...                   index=['cat', 'dog', 'dog', 'mouse'])
            >>> df
                   a   b
              cat  1   3
              dog  2   4
              dog  2   4
            mouse  3   4
            >>> df.kurt()
            a   1.5
            b   4.0
            dtype: float64

            With axis=None

            >>> df.kurt(axis=None).round(6)
            -0.988693

            Using axis=1

            >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [3, 4], 'd': [1, 2]},
            ...                   index=['cat', 'dog'])
            >>> df.kurt(axis=1)
            cat   -6.0
            dog   -6.0
            dtype: float64"""
        # 关键字参数设为空字符串
        kwargs = {"min_count": ""}



    elif name == "cumsum":
        # 如果函数名称为 "cumsum"，根据数据维度选择适合的文档基础模板
        if ndim == 1:
            base_doc = _cnum_series_doc
        else:
            base_doc = _cnum_pd_doc
        # 描述为求和
        desc = "sum"
        # 没有相关内容参见
        see_also = ""
        # 使用预定义的累积求和示例
        examples = _cumsum_examples
        # 关键字参数设为累积函数名为 "sum"
        kwargs = {"accum_func_name": "sum"}



    elif name == "cumprod":
        # 如果函数名称为 "cumprod"，根据数据维度选择适合的文档基础模板
        if ndim == 1:
            base_doc = _cnum_series_doc
        else:
            base_doc = _cnum_pd_doc
        # 描述为乘积
        desc = "product"
        # 没有相关内容参见
        see_also = ""
        # 使用预定义的累积乘积示例
        examples = _cumprod_examples
        # 关键字参数设为累积函数名为 "prod"
        kwargs = {"accum_func_name": "prod"}



    elif name == "cummin":
        # 如果函数名称为 "cummin"，根据数据维度选择适合的文档基础模板
        if ndim == 1:
            base_doc = _cnum_series_doc
        else:
            base_doc = _cnum_pd_doc
        # 描述为最小值
        desc = "minimum"
        # 没有相关内容参见
        see_also = ""
        # 使用预定义的累积最小值示例
        examples = _cummin_examples
        # 关键字参数设为累积函数名为 "min"
        kwargs = {"accum_func_name": "min"}



    elif name == "cummax":
        # 如果函数名称为 "cummax"，根据数据维度选择适合的文档基础模板
        if ndim == 1:
            base_doc = _cnum_series_doc
        else:
            base_doc = _cnum_pd_doc
        # 描述为最大值
        desc = "maximum"
        # 没有相关内容参见
        see_also = ""
        # 使用预定义的累积最大值示例
        examples = _cummax_examples
        # 关键字参数设为累积函数名为 "max"
        kwargs = {"accum_func_name": "max"}



    else:
        # 如果函数名称不匹配已知的累积函数名称，则抛出未实现错误
        raise NotImplementedError

    # 根据模板和各种参数生成最终的文档字符串
    docstr = base_doc.format(
        desc=desc,
        name=name,
        name1=name1,
        name2=name2,
        axis_descr=axis_descr,
        see_also=see_also,
        examples=examples,
        **kwargs,
    )
    # 返回生成的文档字符串
    return docstr
```