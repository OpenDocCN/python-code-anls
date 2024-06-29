# `D:\src\scipysrc\pandas\pandas\core\series.py`

```
"""
Data structure for 1-dimensional cross-sectional and time series data
"""

# 导入必要的模块和库

from __future__ import annotations  # 使用未来版本的类型注解特性

from collections.abc import (  # 导入多个抽象基类
    Callable,  # 可调用对象
    Hashable,  # 可哈希对象
    Iterable,  # 可迭代对象
    Mapping,  # 映射类型
    Sequence,  # 序列类型
)
import operator  # 操作符模块
import sys  # 系统相关模块
from textwrap import dedent  # 文本格式化模块中的缩进调整功能
from typing import (  # 类型提示相关模块
    IO,  # IO操作相关类型
    TYPE_CHECKING,  # 类型检查标志
    Any,  # 任意类型
    Literal,  # 字面量类型
    cast,  # 类型转换函数
    overload,  # 函数重载装饰器
)
import warnings  # 警告模块

import numpy as np  # 导入NumPy库

from pandas._libs import (  # 导入Pandas底层库中的多个模块
    lib,  # 基础功能库
    properties,  # 属性相关
    reshape,  # 重塑相关
)
from pandas._libs.lib import is_range_indexer  # 导入判断是否为范围索引器的函数
from pandas.compat import PYPY  # 兼容性相关，判断是否为PyPy环境
from pandas.compat._constants import REF_COUNT  # 导入引用计数相关常量
from pandas.compat.numpy import function as nv  # 导入NumPy兼容相关函数
from pandas.errors import (  # 导入Pandas错误和异常类
    ChainedAssignmentError,  # 链式赋值错误
    InvalidIndexError,  # 无效索引错误
)
from pandas.errors.cow import (  # 导入Pandas COW模式相关异常信息
    _chained_assignment_method_msg,  # 链式赋值方法消息
    _chained_assignment_msg,  # 链式赋值消息
)
from pandas.util._decorators import (  # 导入Pandas工具装饰器
    Appender,  # 附加器装饰器
    Substitution,  # 替换装饰器
    deprecate_nonkeyword_arguments,  # 弃用非关键字参数装饰器
    doc,  # 文档字符串装饰器
)

from pandas.util._validators import (  # 导入Pandas工具验证器
    validate_ascending,  # 验证升序
    validate_bool_kwarg,  # 验证布尔关键字参数
    validate_percentile,  # 验证百分位数
)

from pandas.core.dtypes.astype import astype_is_view  # 导入数据类型转换相关函数
from pandas.core.dtypes.cast import (  # 导入数据类型转换相关函数和异常类
    LossySetitemError,  # 丢失设置项错误
    construct_1d_arraylike_from_scalar,  # 从标量构造1维数组类似对象
    find_common_type,  # 查找公共类型
    infer_dtype_from,  # 从数据推断数据类型
    maybe_box_native,  # 或许将原生数据装箱
    maybe_cast_pointwise_result,  # 或许按点转换结果
)
from pandas.core.dtypes.common import (  # 导入常用的数据类型判断函数
    is_dict_like,  # 判断是否类字典
    is_float,  # 判断是否浮点数
    is_integer,  # 判断是否整数
    is_iterator,  # 判断是否迭代器
    is_list_like,  # 判断是否类列表
    is_object_dtype,  # 判断是否对象类型数据
    is_scalar,  # 判断是否标量
    pandas_dtype,  # Pandas数据类型
    validate_all_hashable,  # 验证所有可哈希对象
)
from pandas.core.dtypes.dtypes import (  # 导入Pandas数据类型
    CategoricalDtype,  # 分类数据类型
    ExtensionDtype,  # 扩展数据类型
    SparseDtype,  # 稀疏数据类型
)
from pandas.core.dtypes.generic import (  # 导入Pandas通用数据结构
    ABCDataFrame,  # 抽象数据框架类
    ABCSeries,  # 抽象序列类
)
from pandas.core.dtypes.inference import is_hashable  # 导入判断是否可哈希函数
from pandas.core.dtypes.missing import (  # 导入处理缺失值相关函数
    isna,  # 判断是否为缺失值
    na_value_for_dtype,  # 获取给定数据类型的缺失值
    notna,  # 判断是否不是缺失值
    remove_na_arraylike,  # 移除类数组的缺失值
)

from pandas.core import (  # 导入Pandas核心模块
    algorithms,  # 算法模块
    base,  # 基础模块
    common as com,  # 公共模块别名
    nanops,  # 缺失值操作模块
    ops,  # 操作模块
    roperator,  # 反向操作模块
)
from pandas.core.accessor import Accessor  # 导入访问器基类
from pandas.core.apply import SeriesApply  # 导入序列应用类
from pandas.core.arrays import ExtensionArray  # 导入扩展数组类
from pandas.core.arrays.arrow import (  # 导入Arrow数组相关模块
    ListAccessor,  # 列表访问器
    StructAccessor,  # 结构体访问器
)
from pandas.core.arrays.categorical import CategoricalAccessor  # 导入分类数组访问器
from pandas.core.arrays.sparse import SparseAccessor  # 导入稀疏数组访问器
from pandas.core.arrays.string_ import StringDtype  # 导入字符串数据类型
from pandas.core.construction import (  # 导入数据结构构造函数
    array as pd_array,  # 数组构造函数别名
    extract_array,  # 提取数组
    sanitize_array,  # 清理数组
)
from pandas.core.generic import (  # 导入通用数据结构基类
    NDFrame,  # N维数据框架基类
    make_doc,  # 生成文档字符串函数
)
from pandas.core.indexers import (  # 导入索引器相关函数
    disallow_ndim_indexing,  # 禁止N维索引函数
    unpack_1tuple,  # 解包单个元组函数
)
from pandas.core.indexes.accessors import (  # 导入索引访问器
    CombinedDatetimelikeProperties,  # 组合日期时间属性
)
from pandas.core.indexes.api import (  # 导入索引API
    DatetimeIndex,  # 日期时间索引类
    Index,  # 索引基类
    MultiIndex,  # 多重索引类
    PeriodIndex,  # 时期索引类
    default_index,  # 默认索引
    ensure_index,  # 确保索引
    maybe_sequence_to_range,  # 或许将序列转换为范围
)
import pandas.core.indexes.base as ibase  # 导入索引基础模块
from pandas.core.indexes.multi import maybe_droplevels  # 导入或许丢弃层级函数
from pandas.core.indexing import (  # 导入索引操作相关模块
    check_bool_indexer,  # 检查布尔索引器
    # 导入函数 check_dict_or_set_indexers
)
# 导入需要的模块和函数

from pandas.core.internals import SingleBlockManager
from pandas.core.methods import selectn
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import (
    ensure_key_mapped,
    nargsort,
)
from pandas.core.strings.accessor import StringMethods
from pandas.core.tools.datetimes import to_datetime

import pandas.io.formats.format as fmt
from pandas.io.formats.info import (
    INFO_DOCSTRING,
    SeriesInfo,
    series_sub_kwargs,
)
import pandas.plotting

if TYPE_CHECKING:
    # 导入类型检查所需的类型定义
    from pandas._libs.internals import BlockValuesRefs
    from pandas._typing import (
        AggFuncType,
        AnyAll,
        AnyArrayLike,
        ArrayLike,
        Axis,
        AxisInt,
        CorrelationMethod,
        DropKeep,
        Dtype,
        DtypeObj,
        FilePath,
        Frequency,
        IgnoreRaise,
        IndexKeyFunc,
        IndexLabel,
        Level,
        ListLike,
        MutableMappingT,
        NaPosition,
        NumpySorter,
        NumpyValueArrayLike,
        QuantileInterpolation,
        ReindexMethod,
        Renamer,
        Scalar,
        Self,
        SortKind,
        StorageOptions,
        Suffixes,
        ValueKeyFunc,
        WriteBuffer,
        npt,
    )

    from pandas.core.frame import DataFrame
    from pandas.core.groupby.generic import SeriesGroupBy

__all__ = ["Series"]

_shared_doc_kwargs = {
    # 一些用于文档字符串的关键字参数定义
    "axes": "index",
    "klass": "Series",
    "axes_single_arg": "{0 or 'index'}",
    "axis": """axis : {0 or 'index'}
        Unused. Parameter needed for compatibility with DataFrame.""",
    "inplace": """inplace : bool, default False
        If True, performs operation inplace and returns None.""",
    "unique": "np.ndarray",
    "duplicated": "Series",
    "optional_by": "",
    "optional_reindex": """
index : array-like, optional
    New labels for the index. Preferably an Index object to avoid
    duplicating data.
axis : int or str, optional
    Unused.""",
}

# ----------------------------------------------------------------------
# Series class


# error: Cannot override final attribute "ndim" (previously declared in base
# class "NDFrame")
# error: Cannot override final attribute "size" (previously declared in base
# class "NDFrame")
# definition in base class "NDFrame"
class Series(base.IndexOpsMixin, NDFrame):  # type: ignore[misc]
    """
    One-dimensional ndarray with axis labels (including time series).

    Labels need not be unique but must be a hashable type. The object
    supports both integer- and label-based indexing and provides a host of
    methods for performing operations involving the index. Statistical
    methods from ndarray have been overridden to automatically exclude
    missing data (currently represented as NaN).

    Operations between Series (+, -, /, *, **) align values based on their
    associated index values-- they need not be the same length. The result
    index will be the sorted union of the two indexes.

    Parameters
    ----------
    # 数据：array-like、Iterable、dict 或标量数值
    # 包含存储在 Series 中的数据。如果数据是字典，则保持其顺序。不支持无序集合。
    data : array-like, Iterable, dict, or scalar value
    
    # 索引：array-like 或 Index（1d）
    # 值必须是可散列的，并且与 `data` 的长度相同。
    # 允许非唯一索引值。如果未提供索引，将默认使用 RangeIndex（0, 1, 2, ..., n）。
    # 如果数据类似于字典并且索引为 None，则使用数据中的键作为索引。
    # 如果索引不为 None，则生成的 Series 将重新索引为索引值。
    index : array-like or Index (1d)
    
    # dtype：str、numpy.dtype 或 ExtensionDtype，可选
    # 输出 Series 的数据类型。如果未指定，则将从 `data` 推断出数据类型。
    # 更多用法请参阅 :ref:`用户指南 <basics.dtypes>`。
    dtype : str, numpy.dtype, or ExtensionDtype, optional
    
    # name：Hashable，默认为 None
    # 要给 Series 指定的名称。
    name : Hashable, default None
    
    # copy：bool，默认为 False
    # 复制输入数据。仅影响 Series 或 1d ndarray 输入。参见示例。
    copy : bool, default False
    
    # 参见
    # --------
    # DataFrame：二维、大小可变、潜在异构的表格数据。
    # Index：用于索引和对齐的不可变序列。
    See Also
    --------
    DataFrame : Two-dimensional, size-mutable, potentially heterogeneous tabular data.
    Index : Immutable sequence used for indexing and alignment.
    
    # 注意
    # -----
    # 请参阅 :ref:`用户指南 <basics.series>` 获取更多信息。
    Notes
    -----
    Please reference the :ref:`User Guide <basics.series>` for more information.
    
    # 示例
    # --------
    # 使用指定索引从字典构造 Series
    Examples
    --------
    Constructing Series from a dictionary with an Index specified
    
    # >>> d = {"a": 1, "b": 2, "c": 3}
    # >>> ser = pd.Series(data=d, index=["a", "b", "c"])
    # >>> ser
    # a   1
    # b   2
    # c   3
    # dtype: int64
    
    # 字典的键与索引值匹配，因此索引值不会产生影响。
    
    # >>> d = {"a": 1, "b": 2, "c": 3}
    # >>> ser = pd.Series(data=d, index=["x", "y", "z"])
    # >>> ser
    # x   NaN
    # y   NaN
    # z   NaN
    # dtype: float64
    
    # 注意，索引首先由字典的键构建。之后，Series 重新索引为给定的索引值，因此结果为 NaN。
    
    # 使用 `copy=False` 从列表构造 Series。
    
    # >>> r = [1, 2]
    # >>> ser = pd.Series(r, copy=False)
    # >>> ser.iloc[0] = 999
    # >>> r
    # [1, 2]
    # >>> ser
    # 0    999
    # 1      2
    # dtype: int64
    
    # 由于输入数据类型的原因，即使 `copy=False`，Series 也具有原始数据的 `copy`，因此数据不会更改。
    
    # 使用 `copy=False` 从 1d ndarray 构造 Series。
    
    # >>> r = np.array([1, 2])
    # >>> ser = pd.Series(r, copy=False)
    # >>> ser.iloc[0] = 999
    # >>> r
    # array([999,   2])
    # >>> ser
    # 0    999
    # 1      2
    # dtype: int64
    
    # 由于输入数据类型的原因，Series 在原始数据上具有 `view`，因此数据也会更改。
    """

    # 定义类型标识为 "series"
    _typ = "series"
    
    # 处理的类型包括 Index、ExtensionArray 和 np.ndarray
    _HANDLED_TYPES = (Index, ExtensionArray, np.ndarray)
    
    # 定义 Series 的名称，类型为 Hashable，默认为 None
    _name: Hashable
    
    # 元数据列表，包括 "_name"
    _metadata: list[str] = ["_name"]
    
    # 内部名称集合，包括 "index"、"name" 和 NDFrame._internal_names_set 中的元素
    _internal_names_set = {"index", "name"} | NDFrame._internal_names_set
    
    # 访问器集合，包括 "dt"、"cat"、"str" 和 "sparse"
    _accessors = {"dt", "cat", "str", "sparse"}
    _hidden_attrs = (
        base.IndexOpsMixin._hidden_attrs | NDFrame._hidden_attrs | frozenset([])
    )
    # 将一些属性隐藏，包括 IndexOpsMixin 和 NDFrame 中的隐藏属性，目前为空集合

    # 类似于 __array_priority__，将 Series 放置在 DataFrame 之后，Index 和 ExtensionArray 之前。
    # 不应被子类覆盖。
    __pandas_priority__ = 3000

    # 重写 cache_readonly，因为 Series 是可变的
    # 错误：在赋值时类型不兼容（表达式类型为 "property"，
    # 基类 "IndexOpsMixin" 将类型定义为 "Callable[[IndexOpsMixin], bool]"）
    hasnans = property(  # type: ignore[assignment]
        # 错误："Callable[[IndexOpsMixin], bool]" 没有 "fget" 属性
        base.IndexOpsMixin.hasnans.fget,  # type: ignore[attr-defined]
        doc=base.IndexOpsMixin.hasnans.__doc__,
    )

    _mgr: SingleBlockManager
    # _mgr 属性声明为 SingleBlockManager 类型，用于管理单个数据块

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(
        self,
        data=None,
        index=None,
        dtype: Dtype | None = None,
        name=None,
        copy: bool | None = None,
    ):
        """
        从字典输入中派生新 Series 的 "_mgr" 和 "index" 属性。

        Parameters
        ----------
        data : dict or dict-like
            用于填充新 Series 的数据。
        index : Index or None, default None
            新 Series 的索引：如果为 None，则使用字典的键。
        dtype : np.dtype, ExtensionDtype, or None, default None
            新 Series 的数据类型：如果为 None，则从数据中推断。
        
        Returns
        -------
        _data : 新 Series 的 BlockManager
        index : 新 Series 的索引
        """

        # 如果存在数据，则迭代整个字典以查找 NaN（空值），并进行对齐操作
        if data:
            # GH:34717，问题在于使用 zip 从数据中提取键和值。
            # 使用生成器影响性能。
            # 下面是从数据中提取键和值的新方法
            keys = maybe_sequence_to_range(tuple(data.keys()))
            values = list(data.values())  # 生成值列表，更快的方法
        elif index is not None:
            # Series(data=None) 的快速路径。使用标量广播，而不是重新索引。
            if len(index) or dtype is not None:
                values = na_value_for_dtype(pandas_dtype(dtype), compat=False)
            else:
                values = []
            keys = index
        else:
            keys, values = default_index(0), []

        # 现在输入类似于列表，因此依赖于“标准”构建：
        s = Series(values, index=keys, dtype=dtype)

        # 确保尊重顺序，如果有的话
        if data and index is not None:
            s = s.reindex(index)
        
        return s._mgr, s.index
    # ----------------------------------------------------------------------

    @property
    def _constructor(self) -> type[Series]:
        # 返回Series类作为构造函数
        return Series

    def _constructor_from_mgr(self, mgr, axes):
        # 使用_mgr和axes参数从_mgr创建一个Series对象
        ser = Series._from_mgr(mgr, axes=axes)
        ser._name = None  # caller is responsible for setting real name

        if type(self) is Series:
            # 如果self是Series类，则直接返回ser对象
            # 这个检查稍微快一些，因此对于最常见的情况有所好处
            return ser

        # 如果self是Series的子类，则假设子类的__init__方法知道如何处理pd.Series对象
        return self._constructor(ser)

    @property
    def _constructor_expanddim(self) -> Callable[..., DataFrame]:
        """
        Used when a manipulation result has one higher dimension as the
        original, such as Series.to_frame()
        """
        from pandas.core.frame import DataFrame

        # 返回DataFrame类作为扩展维度时的构造函数
        return DataFrame

    def _constructor_expanddim_from_mgr(self, mgr, axes):
        from pandas.core.frame import DataFrame

        # 使用_mgr和axes参数从_mgr创建一个DataFrame对象
        df = DataFrame._from_mgr(mgr, axes=mgr.axes)

        if type(self) is Series:
            # 如果self是Series类，则直接返回df对象
            # 这个检查稍微快一些，因此对于最常见的情况有所好处
            return df

        # 如果self是Series的子类，则假设子类的__init__方法知道如何处理pd.DataFrame对象
        return self._constructor_expanddim(df)

    # types
    @property
    def _can_hold_na(self) -> bool:
        # 返回_mgr是否能够容纳缺失值（NA）
        return self._mgr._can_hold_na

    # ndarray compatibility
    @property
    def dtype(self) -> DtypeObj:
        """
        Return the dtype object of the underlying data.

        See Also
        --------
        Series.dtypes : Return the dtype object of the underlying data.
        Series.astype : Cast a pandas object to a specified dtype dtype.
        Series.convert_dtypes : Convert columns to the best possible dtypes using dtypes
            supporting pd.NA.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.dtype
        dtype('int64')
        """
        # 返回_mgr的数据类型对象
        return self._mgr.dtype

    @property
    def dtypes(self) -> DtypeObj:
        """
        Return the dtype object of the underlying data.

        See Also
        --------
        DataFrame.dtypes :  Return the dtypes in the DataFrame.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.dtypes
        dtype('int64')
        """
        # 返回_mgr的数据类型对象，与DataFrame兼容
        return self.dtype

    @property
    def name(self) -> Hashable:
        """
        Return the name of the Series.

        The name of a Series becomes its index or column name if it is used
        to form a DataFrame. It is also used whenever displaying the Series
        using the interpreter.

        Returns
        -------
        label (hashable object)
            The name of the Series, also the column name if part of a DataFrame.

        See Also
        --------
        Series.rename : Sets the Series name when given a scalar input.
        Index.name : Corresponding Index property.

        Examples
        --------
        The Series name can be set initially when calling the constructor.

        >>> s = pd.Series([1, 2, 3], dtype=np.int64, name="Numbers")
        >>> s
        0    1
        1    2
        2    3
        Name: Numbers, dtype: int64
        >>> s.name = "Integers"
        >>> s
        0    1
        1    2
        2    3
        Name: Integers, dtype: int64

        The name of a Series within a DataFrame is its column name.

        >>> df = pd.DataFrame(
        ...     [[1, 2], [3, 4], [5, 6]], columns=["Odd Numbers", "Even Numbers"]
        ... )
        >>> df
           Odd Numbers  Even Numbers
        0            1             2
        1            3             4
        2            5             6
        >>> df["Even Numbers"].name
        'Even Numbers'
        """
        return self._name

    @name.setter
    def name(self, value: Hashable) -> None:
        """
        Set the name of the Series.

        Parameters
        ----------
        value : hashable object
            The new name for the Series.

        Raises
        ------
        TypeError
            If the value is not hashable.

        See Also
        --------
        validate_all_hashable : Helper function to validate hashable inputs.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.name = "Numbers"
        >>> s.name
        'Numbers'
        """
        validate_all_hashable(value, error_name=f"{type(self).__name__}.name")
        object.__setattr__(self, "_name", value)

    @property
    def values(self):
        """
        Return Series as ndarray or ndarray-like depending on the dtype.

        .. warning::

           We recommend using :attr:`Series.array` or
           :meth:`Series.to_numpy`, depending on whether you need
           a reference to the underlying data or a NumPy array.

        Returns
        -------
        numpy.ndarray or ndarray-like
            The underlying data of the Series.

        See Also
        --------
        Series.array : Reference to the underlying data.
        Series.to_numpy : A NumPy array representing the underlying data.

        Examples
        --------
        >>> pd.Series([1, 2, 3]).values
        array([1, 2, 3])

        >>> pd.Series(list("aabc")).values
        array(['a', 'a', 'b', 'c'], dtype=object)

        >>> pd.Series(list("aabc")).astype("category").values
        ['a', 'a', 'b', 'c']
        Categories (3, object): ['a', 'b', 'c']

        Timezone aware datetime data is converted to UTC:

        >>> pd.Series(pd.date_range("20130101", periods=3, tz="US/Eastern")).values
        array(['2013-01-01T05:00:00.000000000',
               '2013-01-02T05:00:00.000000000',
               '2013-01-03T05:00:00.000000000'], dtype='datetime64[ns]')
        """
        return self._mgr.external_values()

    @property
    def _values(self):
        """
        Return the internal repr of this data (defined by Block.interval_values).
        This are the values as stored in the Block (ndarray or ExtensionArray
        depending on the Block class), with datetime64[ns] and timedelta64[ns]
        wrapped in ExtensionArrays to match Index._values behavior.

        Differs from the public ``.values`` for certain data types, because of
        historical backwards compatibility of the public attribute (e.g. period
        returns object ndarray and datetimetz a datetime64[ns] ndarray for
        ``.values`` while it returns an ExtensionArray for ``._values`` in those
        cases).

        Differs from ``.array`` in that this still returns the numpy array if
        the Block is backed by a numpy array (except for datetime64 and
        timedelta64 dtypes), while ``.array`` ensures to always return an
        ExtensionArray.

        Overview:

        dtype       | values        | _values       | array                 |
        ----------- | ------------- | ------------- | --------------------- |
        Numeric     | ndarray       | ndarray       | NumpyExtensionArray   |
        Category    | Categorical   | Categorical   | Categorical           |
        dt64[ns]    | ndarray[M8ns] | DatetimeArray | DatetimeArray         |
        dt64[ns tz] | ndarray[M8ns] | DatetimeArray | DatetimeArray         |
        td64[ns]    | ndarray[m8ns] | TimedeltaArray| TimedeltaArray        |
        Period      | ndarray[obj]  | PeriodArray   | PeriodArray           |
        Nullable    | EA            | EA            | EA                    |

        """
        return self._mgr.internal_values()  # 返回该对象内部数据的表示形式，使用了_mgr的internal_values方法

    @property
    def _references(self) -> BlockValuesRefs:
        """
        Property to access references of BlockValuesRefs type from _mgr._block.refs.

        """
        return self._mgr._block.refs  # 返回_mgr._block.refs属性，类型为BlockValuesRefs

    # error: Decorated property not supported
    @Appender(base.IndexOpsMixin.array.__doc__)  # type: ignore[misc]
    @property
    def array(self) -> ExtensionArray:
        """
        Property to return an ExtensionArray representation of array values from _mgr.

        """
        return self._mgr.array_values()  # 返回_mgr的array_values方法，表示为ExtensionArray类型

    def __len__(self) -> int:
        """
        Return the length of the Series.

        """
        return len(self._mgr)  # 返回_mgr的长度，即Series的长度

    # ----------------------------------------------------------------------
    # NDArray Compat
    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ):
        """
        Method to return the array representation with optional dtype and copy settings.

        """
    ) -> np.ndarray:
        """
        Return the values as a NumPy array.

        Users should not call this directly. Rather, it is invoked by
        :func:`numpy.array` and :func:`numpy.asarray`.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to use for the resulting NumPy array. By default,
            the dtype is inferred from the data.

        copy : bool or None, optional
            Unused.

        Returns
        -------
        numpy.ndarray
            The values in the series converted to a :class:`numpy.ndarray`
            with the specified `dtype`.

        See Also
        --------
        array : Create a new array from data.
        Series.array : Zero-copy view to the array backing the Series.
        Series.to_numpy : Series method for similar behavior.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3])
        >>> np.asarray(ser)
        array([1, 2, 3])

        For timezone-aware data, the timezones may be retained with
        ``dtype='object'``

        >>> tzser = pd.Series(pd.date_range("2000", periods=2, tz="CET"))
        >>> np.asarray(tzser, dtype="object")
        array([Timestamp('2000-01-01 00:00:00+0100', tz='CET'),
               Timestamp('2000-01-02 00:00:00+0100', tz='CET')],
              dtype=object)

        Or the values may be localized to UTC and the tzinfo discarded with
        ``dtype='datetime64[ns]'``

        >>> np.asarray(tzser, dtype="datetime64[ns]")  # doctest: +ELLIPSIS
        array(['1999-12-31T23:00:00.000000000', ...],
              dtype='datetime64[ns]')
        """
        values = self._values  # 获取Series对象的值
        arr = np.asarray(values, dtype=dtype)  # 将Series的值转换为NumPy数组，可以指定dtype
        if astype_is_view(values.dtype, arr.dtype):  # 检查是否为视图转换
            arr = arr.view()  # 转换为视图
            arr.flags.writeable = False  # 设置为只读视图
        return arr  # 返回转换后的NumPy数组

    # ----------------------------------------------------------------------

    # indexers
    @property
    def axes(self) -> list[Index]:
        """
        Return a list of the row axis labels.
        """
        return [self.index]  # 返回包含行轴标签的列表，这里只返回了索引对象

    # ----------------------------------------------------------------------
    # Indexing Methods

    def _ixs(self, i: int, axis: AxisInt = 0) -> Any:
        """
        Return the i-th value or values in the Series by location.

        Parameters
        ----------
        i : int
            The integer index location for retrieving the value.

        Returns
        -------
        scalar
            The value at the specified index location.
        """
        return self._values[i]  # 返回指定索引位置的值

    def _slice(self, slobj: slice, axis: AxisInt = 0) -> Series:
        # axis kwarg is retained for compat with NDFrame method
        #  _slice is *always* positional
        mgr = self._mgr.get_slice(slobj, axis=axis)  # 获取切片后的数据管理器
        out = self._constructor_from_mgr(mgr, axes=mgr.axes)  # 从数据管理器创建新的Series对象
        out._name = self._name  # 设置新Series对象的名称
        return out.__finalize__(self)  # 完成后返回新对象，并确保与原始对象一致的最终设置
    # 定义特殊方法 __getitem__，用于通过键来访问对象的元素
    def __getitem__(self, key):
        # 检查键是否是字典或集合的索引器
        check_dict_or_set_indexers(key)
        # 如果键是可调用的，则应用该函数到 self 上
        key = com.apply_if_callable(key, self)

        # 如果键是 Ellipsis，则返回一个浅拷贝
        if key is Ellipsis:
            return self.copy(deep=False)

        # 检查键是否是标量值
        key_is_scalar = is_scalar(key)
        if isinstance(key, (list, tuple)):
            # 如果键是列表或元组，解包为单个元素
            key = unpack_1tuple(key)

        elif key_is_scalar:
            # 注意：在版本 3.0 中，我们将 int 键始终视为标签，与 DataFrame 的行为一致。
            return self._get_value(key)

        # 将生成器转换为列表，以便在后续的可散列部分进行迭代以检查切片
        if is_iterator(key):
            key = list(key)

        # 如果键是可散列的且不是切片对象
        if is_hashable(key) and not isinstance(key, slice):
            try:
                # 获取键对应的值
                result = self._get_value(key)
                return result

            except (KeyError, TypeError, InvalidIndexError):
                # 对于无效索引或类型错误，例如生成器
                # 见 test_series_getitem_corner_generator
                if isinstance(key, tuple) and isinstance(self.index, MultiIndex):
                    # 处理在 MultiIndex 中第一级中的元组键的特殊情况
                    return self._get_values_tuple(key)

        # 如果键是切片对象，则处理切片操作
        if isinstance(key, slice):
            # 在检查是否是布尔索引器之前，先进行切片检查
            return self._getitem_slice(key)

        # 如果键是布尔索引器
        if com.is_bool_indexer(key):
            # 检查布尔索引器并返回相应行的数据
            key = check_bool_indexer(self.index, key)
            key = np.asarray(key, dtype=bool)
            return self._get_rows_with_mask(key)

        # 默认情况下，通过键进行访问
        return self._get_with(key)

    # 处理非常数索引或其他类型的特殊整数
    def _get_with(self, key):
        # 如果键是 DataFrame，则抛出 TypeError
        if isinstance(key, ABCDataFrame):
            raise TypeError(
                "Indexing a Series with DataFrame is not "
                "supported, use the appropriate DataFrame column"
            )
        # 如果键是元组，则调用 _get_values_tuple 方法处理
        elif isinstance(key, tuple):
            return self._get_values_tuple(key)

        # 否则通过 loc 方法获取相应的值
        return self.loc[key]
    # 获取以元组形式作为键的值
    def _get_values_tuple(self, key: tuple):
        # 处理 Matplotlib 的兼容性问题
        if com.any_none(*key):
            # 如果 key 中存在 None 值，执行以下操作来兼容 Matplotlib
            # 详情请参见 tests.series.timeseries.test_mpl_compat_hack
            # 使用 asarray 是为了避免返回一个二维的 DatetimeArray
            result = np.asarray(self._values[key])
            disallow_ndim_indexing(result)
            return result

        if not isinstance(self.index, MultiIndex):
            # 如果索引不是 MultiIndex 类型，则抛出 KeyError
            raise KeyError("key of type tuple not found and not a MultiIndex")

        # 如果 key 未被包含，此处应已返回
        # 获取位置索引和新的索引
        indexer, new_index = self.index.get_loc_level(key)
        # 根据索引获取新的 Series 对象
        new_ser = self._constructor(self._values[indexer], index=new_index, copy=False)
        if isinstance(indexer, slice):
            new_ser._mgr.add_references(self._mgr)
        return new_ser.__finalize__(self)

    # 根据掩码获取符合条件的行，并返回新的 Series 对象
    def _get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> Series:
        # 使用掩码获取新的 Manager 对象
        new_mgr = self._mgr.get_rows_with_mask(indexer)
        # 使用 Manager 对象构造新的 Series 对象并返回
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(self)

    # 快速检索指定索引标签处的单个值
    def _get_value(self, label, takeable: bool = False):
        """
        Quickly retrieve single value at passed index label.

        Parameters
        ----------
        label : object
            索引标签
        takeable : bool, optional
            是否将索引解释为索引器，默认为 False

        Returns
        -------
        scalar value
            单个标量值
        """
        if takeable:
            # 如果 takeable 为 True，直接返回对应 label 的值
            return self._values[label]

        # 类似于 Index.get_value，但不会回退到位置索引
        loc = self.index.get_loc(label)

        if is_integer(loc):
            # 如果 loc 是整数，则返回对应位置的值
            return self._values[loc]

        if isinstance(self.index, MultiIndex):
            # 如果索引是 MultiIndex 类型
            mi = self.index
            new_values = self._values[loc]
            if len(new_values) == 1 and mi.nlevels == 1:
                # 如果剩余超过一个级别，无法返回标量
                return new_values[0]

            # 获取新的索引
            new_index = mi[loc]
            new_index = maybe_droplevels(new_index, label)
            # 根据新的值和索引构造新的 Series 对象
            new_ser = self._constructor(
                new_values, index=new_index, name=self.name, copy=False
            )
            if isinstance(loc, slice):
                new_ser._mgr.add_references(self._mgr)
            return new_ser.__finalize__(self)

        else:
            # 否则，返回索引位置对应的值
            return self.iloc[loc]
    # 实现对象的赋值操作，当不是在 PyPy 环境下且对象的引用计数小于等于3时，发出警告
    def __setitem__(self, key, value) -> None:
        if not PYPY:
            if sys.getrefcount(self) <= 3:
                warnings.warn(
                    _chained_assignment_msg, ChainedAssignmentError, stacklevel=2
                )

        # 检查并规范化索引键
        check_dict_or_set_indexers(key)
        # 如果键是可调用的，则将其应用到对象上
        key = com.apply_if_callable(key, self)

        # 如果键是 Ellipsis 对象，则将其替换为 slice(None)
        if key is Ellipsis:
            key = slice(None)

        # 如果键是 slice 类型，则将其转换为适合获取数据的索引器，并调用 _set_values 方法设置值
        if isinstance(key, slice):
            indexer = self.index._convert_slice_indexer(key, kind="getitem")
            return self._set_values(indexer, value)

        # 否则，调用 _set_with_engine 方法设置键和值
        try:
            self._set_with_engine(key, value)
        except KeyError:
            # 如果出现 KeyError，则尝试将值设置到 self.loc[key] 上
            # 对于 MultiIndex 或者 object-dtype 的标量键，这里表现为标量键不在 self.index 中
            self.loc[key] = value

        except (TypeError, ValueError, LossySetitemError):
            # 如果键有效，但不能无损地设置值，则获取索引位置并调用 _set_values 方法设置值
            indexer = self.index.get_loc(key)
            self._set_values(indexer, value)

        except InvalidIndexError as err:
            # 如果键是元组类型且索引不是 MultiIndex，则抛出 KeyError
            if isinstance(key, tuple) and not isinstance(self.index, MultiIndex):
                raise KeyError(
                    "key of type tuple not found and not a MultiIndex"
                ) from err

            # 如果键是布尔索引器，则检查并规范化布尔索引器，然后在 key 非零的位置调用 _set_values 方法设置值
            if com.is_bool_indexer(key):
                key = check_bool_indexer(self.index, key)
                key = np.asarray(key, dtype=bool)

                # 如果值是列表或类列表且长度不匹配且不是 Series 且不是 object-dtype，则在 _where 调用中重新索引 Series
                if (
                    is_list_like(value)
                    and len(value) != len(self)
                    and not isinstance(value, Series)
                    and not is_object_dtype(self.dtype)
                ):
                    indexer = key.nonzero()[0]
                    self._set_values(indexer, value)
                    return

                # 否则，尝试在 _where 调用中处理 series[mask] = other 的情况
                try:
                    self._where(~key, value, inplace=True)
                except InvalidIndexError:
                    # 处理异常情况，如 test_where_dups，使用 iloc[key] 设置值
                    self.iloc[key] = value
                return

            else:
                # 其他情况下，调用 _set_with 方法设置键和值
                self._set_with(key, value)

    # 使用引擎方法设置键和值，获取键在索引中的位置，然后调用 _mgr.setitem_inplace 方法设置值
    def _set_with_engine(self, key, value) -> None:
        loc = self.index.get_loc(key)
        self._mgr.setitem_inplace(loc, value)
    # 我们通过处理 InvalidIndexError 的异常进入这里，因此 key 应该始终是类似列表的对象。
    assert not isinstance(key, tuple)

    # 如果 key 是迭代器，将其转换为列表，避免 infer_dtype 调用时消耗生成器。
    if is_iterator(key):
        key = list(key)

    # 调用 _set_labels 方法，将 key 和 value 设置到对象中。
    self._set_labels(key, value)




    # 将 key 转换为适合元组的数组表示形式。
    key = com.asarray_tuplesafe(key)
    
    # 使用 index 对象获取 key 的索引器。
    indexer: np.ndarray = self.index.get_indexer(key)
    
    # 检查索引器中是否有值为 -1 的项，如果有则引发 KeyError 异常。
    mask = indexer == -1
    if mask.any():
        raise KeyError(f"{key[mask]} not in index")
    
    # 调用 _set_values 方法，设置索引器对应位置的值为 value。
    self._set_values(indexer, value)



    # 如果 key 是 Index 或 Series 对象，则将其转换为其内部值的数组形式。
    if isinstance(key, (Index, Series)):
        key = key._values

    # 调用 _mgr 的 setitem 方法，设置索引器为 key 的位置为 value。
    self._mgr = self._mgr.setitem(indexer=key, value=value)



    """
    快速设置传递标签的单个值。

    如果标签不存在，则在结果索引的末尾创建一个新对象。

    Parameters
    ----------
    label : object
        不允许使用 MultiIndex 进行部分索引。
    value : object
        标量值。
    takeable : bool, 默认为 False
        将索引解释为索引器。

    """
    # 如果不允许 takeable，则尝试获取 label 在 index 中的位置。
    if not takeable:
        try:
            loc = self.index.get_loc(label)
        except KeyError:
            # 使用非递归方法设置
            self.loc[label] = value
            return
    else:
        loc = label

    # 调用 _set_values 方法，设置 loc 处的值为 value。
    self._set_values(loc, value)
    def repeat(self, repeats: int | Sequence[int], axis: None = None) -> Series:
        """
        Repeat elements of a Series.

        Returns a new Series where each element of the current Series
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            Series.
        axis : None
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            Newly created Series with repeated elements.

        See Also
        --------
        Index.repeat : Equivalent function for Index.
        numpy.repeat : Similar method for :class:`numpy.ndarray`.

        Examples
        --------
        >>> s = pd.Series(["a", "b", "c"])
        >>> s
        0    a
        1    b
        2    c
        dtype: object
        >>> s.repeat(2)
        0    a
        0    a
        1    b
        1    b
        2    c
        2    c
        dtype: object
        >>> s.repeat([1, 2, 3])
        0    a
        1    b
        1    b
        2    c
        2    c
        2    c
        dtype: object
        """
        # Validate the input parameters using nv.validate_repeat
        nv.validate_repeat((), {"axis": axis})
        
        # Create a new index by repeating the current Series' index based on 'repeats'
        new_index = self.index.repeat(repeats)
        
        # Repeat the values of the Series based on 'repeats'
        new_values = self._values.repeat(repeats)
        
        # Create and return a new Series object using the repeated values and index,
        # preserving metadata and indicating the method used for construction
        return self._constructor(new_values, index=new_index, copy=False).__finalize__(
            self, method="repeat"
        )

    @overload
    def reset_index(
        self,
        level: IndexLabel = ...,
        *,
        drop: Literal[False] = ...,
        name: Level = ...,
        inplace: Literal[False] = ...,
        allow_duplicates: bool = ...,
    ) -> DataFrame: ...
    
    @overload
    def reset_index(
        self,
        level: IndexLabel = ...,
        *,
        drop: Literal[True],
        name: Level = ...,
        inplace: Literal[False] = ...,
        allow_duplicates: bool = ...,
    ) -> Series: ...
    
    @overload
    def reset_index(
        self,
        level: IndexLabel = ...,
        *,
        drop: bool = ...,
        name: Level = ...,
        inplace: Literal[True],
        allow_duplicates: bool = ...,
    ) -> None: ...
    
    def reset_index(
        self,
        level: IndexLabel | None = None,
        *,
        drop: bool = False,
        name: Level = lib.no_default,
        inplace: bool = False,
        allow_duplicates: bool = False,
    ):
        """
        Reset the index of the Series.

        Parameters
        ----------
        level : IndexLabel or None, optional
            Level(s) of index to target for resetting. By default, resets all levels.
        drop : bool, default False
            Whether to drop the index levels being reset or not.
        name : Level, default lib.no_default
            The name to set for the newly created Series (if applicable).
        inplace : bool, default False
            Whether to modify the Series in place or return a new Series.
        allow_duplicates : bool, default False
            Whether to allow duplicate index values. If True, will raise if duplicates are found.

        Returns
        -------
        None or Series or DataFrame
            If inplace=True, modifies the Series in place and returns None.
            If inplace=False, returns a new Series or DataFrame with the index reset.
        """
        # Implementation details for resetting the index, handled by pandas internals
        # (not explicitly annotated here)
        pass
    
    # ----------------------------------------------------------------------
    # Rendering Methods

    def __repr__(self) -> str:
        """
        Return a string representation for a particular Series.
        """
        # Retrieve representation parameters specific to the Series
        repr_params = fmt.get_series_repr_params()
        
        # Convert the Series to a string representation using the retrieved parameters
        return self.to_string(**repr_params)

    @overload
    # 定义类方法 `to_string`，用于将对象转换为字符串表示形式
    def to_string(
        self,
        buf: None = ...,
        *,
        na_rep: str = ...,
        float_format: str | None = ...,
        header: bool = ...,
        index: bool = ...,
        length: bool = ...,
        dtype=...,
        name=...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
    ) -> str: ...
    
    # 方法重载，支持将对象转换为字符串并写入到指定文件路径或写缓冲区
    @overload
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        na_rep: str = ...,
        float_format: str | None = ...,
        header: bool = ...,
        index: bool = ...,
        length: bool = ...,
        dtype=...,
        name=...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
    ) -> None: ...
    
    # 使用装饰器标记该方法为过时，不再支持非关键字参数的版本，推荐使用关键字参数版本
    @deprecate_nonkeyword_arguments(
        version="4.0", allowed_args=["self", "buf"], name="to_string"
    )
    # 定义类方法 `to_string` 的具体实现，支持将对象转换为字符串并输出到指定的文件路径或写缓冲区
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        na_rep: str = "NaN",
        float_format: str | None = None,
        header: bool = True,
        index: bool = True,
        length: bool = False,
        dtype: bool = False,
        name: bool = False,
        max_rows: int | None = None,
        min_rows: int | None = None,
    ) -> None:
    # Render a string representation of the Series object.
    def to_string(
        buf: StringIO-like = ...,  # Buffer to write the string representation to
        na_rep: str = 'NaN',        # String representation of NaN values
        float_format: Callable[[float], str] | None = None,  # Custom formatter for floats
        header: bool = True,        # Whether to include the Series header (index name)
        index: bool = True,         # Whether to include index (row) labels
        length: bool = False,       # Whether to add the length of the Series
        dtype: bool = False,        # Whether to add the data type of the Series
        name: bool = False,         # Whether to add the name of the Series if not None
        max_rows: int | None = None,  # Maximum number of rows to display before truncating
        min_rows: int = 10,         # Number of rows to display in a truncated repr
    ) -> str | None:
        """
        Render a string representation of the Series.
    
        Parameters
        ----------
        buf : StringIO-like, optional
            Buffer to write to.
        na_rep : str, optional
            String representation of NaN to use, default 'NaN'.
        float_format : one-parameter function, optional
            Formatter function to apply to columns' elements if they are
            floats, default None.
        header : bool, default True
            Add the Series header (index name).
        index : bool, optional
            Add index (row) labels, default True.
        length : bool, default False
            Add the Series length.
        dtype : bool, default False
            Add the Series dtype.
        name : bool, default False
            Add the Series name if not None.
        max_rows : int, optional
            Maximum number of rows to show before truncating. If None, show
            all.
        min_rows : int, optional
            The number of rows to display in a truncated repr (when number
            of rows is above `max_rows`).
    
        Returns
        -------
        str or None
            String representation of Series if ``buf=None``, otherwise None.
    
        See Also
        --------
        Series.to_dict : Convert Series to dict object.
        Series.to_frame : Convert Series to DataFrame object.
        Series.to_markdown : Print Series in Markdown-friendly format.
        Series.to_timestamp : Cast to DatetimeIndex of Timestamps.
    
        Examples
        --------
        >>> ser = pd.Series([1, 2, 3]).to_string()
        >>> ser
        '0    1\\n1    2\\n2    3'
        """
    
        # Create a formatter object for rendering the Series to a string
        formatter = fmt.SeriesFormatter(
            self,
            name=name,
            length=length,
            header=header,
            index=index,
            dtype=dtype,
            na_rep=na_rep,
            float_format=float_format,
            min_rows=min_rows,
            max_rows=max_rows,
        )
    
        # Generate the string representation of the Series using the formatter
        result = formatter.to_string()
    
        # Verify that the result is a string; raise an error if it's not
        if not isinstance(result, str):
            raise AssertionError(
                "result must be of type str, type "
                f"of result is {type(result).__name__!r}"
            )
    
        # If buf is None, return the result; otherwise, write the result to buf
        if buf is None:
            return result
        else:
            if hasattr(buf, "write"):  # Check if buf has a 'write' attribute (is a writable buffer)
                buf.write(result)
            else:  # If buf is a file path, open it and write the result to the file
                with open(buf, "w", encoding="utf-8") as f:
                    f.write(result)
        return None
    ) -> None: ...


# 这是一个类型提示，表示该方法返回类型为 None。



    @overload
    def to_markdown(
        self,
        buf: IO[str] | None,
        *,
        mode: str = ...,
        index: bool = ...,
        storage_options: StorageOptions | None = ...,
        **kwargs,
    ) -> str | None: ...


# 方法签名的重载声明，用于静态类型检查，定义了可能的参数和返回类型。



    @doc(
        klass=_shared_doc_kwargs["klass"],
        storage_options=_shared_docs["storage_options"],
        examples=dedent(
            """Examples
            --------
            >>> s = pd.Series(["elk", "pig", "dog", "quetzal"], name="animal")
            >>> print(s.to_markdown())
            |    | animal   |
            |---:|:---------|
            |  0 | elk      |
            |  1 | pig      |
            |  2 | dog      |
            |  3 | quetzal  |

            Output markdown with a tabulate option.

            >>> print(s.to_markdown(tablefmt="grid"))
            +----+----------+
            |    | animal   |
            +====+==========+
            |  0 | elk      |
            +----+----------+
            |  1 | pig      |
            +----+----------+
            |  2 | dog      |
            +----+----------+
            |  3 | quetzal  |
            +----+----------+"""
        ),
    )


# 使用 @doc 装饰器为方法添加文档信息，包括类名（klass）、存储选项（storage_options）和示例（examples）。
# 示例展示了将 Series 对象转换为 Markdown 格式的示例输出。



    @deprecate_nonkeyword_arguments(
        version="4.0", allowed_args=["self", "buf"], name="to_markdown"
    )


# 使用 @deprecate_nonkeyword_arguments 装饰器标记方法 to_markdown 的非关键字参数（self 和 buf）已被弃用，从版本 4.0 开始。



    def to_markdown(
        self,
        buf: IO[str] | None = None,
        mode: str = "wt",
        index: bool = True,
        storage_options: StorageOptions | None = None,
        **kwargs,
    ) -> str | None:


# 定义方法 to_markdown，用于将对象以 Markdown 友好的格式打印出来。
# 参数包括 buf（写入缓冲区）、mode（打开文件的模式，默认为 "wt"）、index（是否添加索引标签，默认为 True）、storage_options（存储选项，默认为 None）和任意其他关键字参数。
# 返回类型为字符串或 None。



        """
        Print {klass} in Markdown-friendly format.

        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        mode : str, optional
            Mode in which file is opened, "wt" by default.
        index : bool, optional, default True
            Add index (row) labels.

        {storage_options}

        **kwargs
            These parameters will be passed to `tabulate \
                <https://pypi.org/project/tabulate>`_.

        Returns
        -------
        str
            {klass} in Markdown-friendly format.

        Notes
        -----
        Requires the `tabulate <https://pypi.org/project/tabulate>`_ package.

        {examples}
        """


# 方法的文档字符串，详细描述了方法的功能、参数和返回值。
# 文档字符串中包含了关于 buf、mode、index 和 storage_options 参数的说明，以及说明了 **kwargs 将被传递给 tabulate 包。
# 提供了返回值的描述，以及方法的一些注意事项和示例（examples）的引用。



        return self.to_frame().to_markdown(
            buf, mode=mode, index=index, storage_options=storage_options, **kwargs
        )


# 调用对象的 to_frame 方法，将结果转换为 DataFrame 后再调用 to_markdown 方法，将最终结果返回。



    # ----------------------------------------------------------------------


# 分隔符，用于标记下面的代码段属于不同的部分或功能。
    def items(self) -> Iterable[tuple[Hashable, Any]]:
        """
        Lazily iterate over (index, value) tuples.

        This method returns an iterable of tuples (index, value), representing
        index-value pairs of the Series. It allows lazy iteration over the Series,
        which is memory efficient for large datasets.

        Returns
        -------
        iterable
            An iterable of tuples containing the (index, value) pairs from a Series.

        See Also
        --------
        DataFrame.items : Iterate over (column name, Series) pairs.
        DataFrame.iterrows : Iterate over DataFrame rows as (index, Series) pairs.

        Examples
        --------
        >>> s = pd.Series(["A", "B", "C"])
        >>> for index, value in s.items():
        ...     print(f"Index : {index}, Value : {value}")
        Index : 0, Value : A
        Index : 1, Value : B
        Index : 2, Value : C
        """

    # ----------------------------------------------------------------------
    # Misc public methods

    def keys(self) -> Index:
        """
        Return alias for index.

        Returns
        -------
        Index
            The index (axis labels) of the Series.

        See Also
        --------
        Series.index : The index (axis labels) of the Series.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3], index=[0, 1, 2])
        >>> s.keys()
        Index([0, 1, 2], dtype='int64')
        """
        return self.index

    @overload
    def to_dict(
        self, *, into: type[MutableMappingT] | MutableMappingT
    ) -> MutableMappingT: ...

    @overload
    def to_dict(self, *, into: type[dict] = ...) -> dict: ...

    # error: Incompatible default for argument "into" (default has type "type[dict[Any, Any]]", argument has type "type[MutableMappingT] | MutableMappingT")
    def to_dict(
        self,
        *,
        into: type[MutableMappingT] | MutableMappingT = dict,  # type: ignore[assignment]
    ) -> MutableMappingT:
        """
        Convert the Series to a dictionary.

        Parameters
        ----------
        into : type[MutableMappingT] | MutableMappingT, default dict
            The type of the resulting dictionary or an instance of it.

        Returns
        -------
        MutableMappingT
            A dictionary representation of the Series.

        Notes
        -----
        This method converts a Series into a dictionary. The `into` parameter
        specifies the type of dictionary to return. By default, it returns a
        standard Python dictionary (`dict`).

        See Also
        --------
        Series.items : Lazily iterate over (index, value) pairs.
        Series.to_list : Convert the Series to a list.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        >>> s.to_dict()
        {'a': 1, 'b': 2, 'c': 3}
        """
    ) -> MutableMappingT:
        """
        Convert Series to {label -> value} dict or dict-like object.

        Parameters
        ----------
        into : class, default dict
            The collections.abc.MutableMapping subclass to use as the return
            object. Can be the actual class or an empty instance of the mapping
            type you want.  If you want a collections.defaultdict, you must
            pass it initialized.

        Returns
        -------
        collections.abc.MutableMapping
            Key-value representation of Series.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.to_dict()
        {0: 1, 1: 2, 2: 3, 3: 4}
        >>> from collections import OrderedDict, defaultdict
        >>> s.to_dict(into=OrderedDict)
        OrderedDict([(0, 1), (1, 2), (2, 3), (3, 4)])
        >>> dd = defaultdict(list)
        >>> s.to_dict(into=dd)
        defaultdict(<class 'list'>, {0: 1, 1: 2, 2: 3, 3: 4})
        """
        # 标准化 `into` 参数，确保其符合映射类型的标准
        into_c = com.standardize_mapping(into)

        # 如果 Series 的数据类型是对象型或者是扩展数据类型，则使用 maybe_box_native 将每对键值对包装成原生类型并放入 into_c 中
        if is_object_dtype(self.dtype) or isinstance(self.dtype, ExtensionDtype):
            return into_c((k, maybe_box_native(v)) for k, v in self.items())
        else:
            # 若数据类型不是对象型，则使用默认索引器返回原生 Python 类型
            return into_c(self.items())

    def to_frame(self, name: Hashable = lib.no_default) -> DataFrame:
        """
        Convert Series to DataFrame.

        Parameters
        ----------
        name : object, optional
            The passed name should substitute for the series name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame representation of Series.

        Examples
        --------
        >>> s = pd.Series(["a", "b", "c"], name="vals")
        >>> s.to_frame()
          vals
        0    a
        1    b
        2    c
        """
        # 初始化列索引
        columns: Index
        if name is lib.no_default:
            # 如果未提供 name 参数，则使用 Series 的名称作为列索引
            name = self.name
            if name is None:
                # 如果 Series 没有名称，则默认使用 [0] 作为列索引
                columns = default_index(1)
            else:
                # 使用提供的 name 参数作为列索引
                columns = Index([name])
        else:
            # 使用指定的 name 参数作为列索引
            columns = Index([name])

        # 将 Series 转换为二维管理器
        mgr = self._mgr.to_2d_mgr(columns)
        # 从管理器创建 DataFrame
        df = self._constructor_expanddim_from_mgr(mgr, axes=mgr.axes)
        # 保留 Series 的元数据，并返回 DataFrame
        return df.__finalize__(self, method="to_frame")

    def _set_name(
        self, name, inplace: bool = False, deep: bool | None = None
    ) -> Series:
        """
        Set the Series name.

        Parameters
        ----------
        name : str
            The name to set for the Series.
        inplace : bool
            Whether to modify `self` directly or return a copy.
        """
        # 验证 inplace 参数，确定是否直接修改 self 还是返回副本
        inplace = validate_bool_kwarg(inplace, "inplace")
        # 如果 inplace 为 True，则修改当前 Series 的名称
        ser = self if inplace else self.copy(deep=False)
        ser.name = name
        return ser
    @Appender(
        dedent(
            """
            Examples
            --------
            >>> ser = pd.Series([390., 350., 30., 20.],
            ...                 index=['Falcon', 'Falcon', 'Parrot', 'Parrot'],
            ...                 name="Max Speed")
            >>> ser
            Falcon    390.0
            Falcon    350.0
            Parrot     30.0
            Parrot     20.0
            Name: Max Speed, dtype: float64
            >>> ser.groupby(["a", "b", "a", "b"]).mean()
            a    210.0
            b    185.0
            Name: Max Speed, dtype: float64
            >>> ser.groupby(level=0).mean()
            Falcon    370.0
            Parrot     25.0
            Name: Max Speed, dtype: float64
            >>> ser.groupby(ser > 100).mean()
            Max Speed
            False     25.0
            True     370.0
            Name: Max Speed, dtype: float64
    
            **Grouping by Indexes**
    
            We can groupby different levels of a hierarchical index
            using the `level` parameter:
    
            >>> arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],
            ...           ['Captive', 'Wild', 'Captive', 'Wild']]
            >>> index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))
            >>> ser = pd.Series([390., 350., 30., 20.], index=index, name="Max Speed")
            >>> ser
            Animal  Type
            Falcon  Captive    390.0
                    Wild       350.0
            Parrot  Captive     30.0
                    Wild        20.0
            Name: Max Speed, dtype: float64
            >>> ser.groupby(level=0).mean()
            Animal
            Falcon    370.0
            Parrot     25.0
            Name: Max Speed, dtype: float64
            >>> ser.groupby(level="Type").mean()
            Type
            Captive    210.0
            Wild       185.0
            Name: Max Speed, dtype: float64
    
            We can also choose to include `NA` in group keys or not by defining
            `dropna` parameter, the default setting is `True`.
    
            >>> ser = pd.Series([1, 2, 3, 3], index=["a", 'a', 'b', np.nan])
            >>> ser.groupby(level=0).sum()
            a    3
            b    3
            dtype: int64
    
            >>> ser.groupby(level=0, dropna=False).sum()
            a    3
            b    3
            NaN  3
            dtype: int64
    
            >>> arrays = ['Falcon', 'Falcon', 'Parrot', 'Parrot']
            >>> ser = pd.Series([390., 350., 30., 20.], index=arrays, name="Max Speed")
            >>> ser.groupby(["a", "b", "a", np.nan]).mean()
            a    210.0
            b    350.0
            Name: Max Speed, dtype: float64
    
            >>> ser.groupby(["a", "b", "a", np.nan], dropna=False).mean()
            a    210.0
            b    350.0
            NaN   20.0
            Name: Max Speed, dtype: float64
            """
        )
    )
    @Appender(_shared_docs["groupby"] % _shared_doc_kwargs)
    def groupby(
        self,
        by=None,
        level: IndexLabel | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = False,
        dropna: bool = True,


注释：
这段代码定义了一个名为 `groupby` 的方法，用于数据分组操作。通过 `by` 参数或 `level` 参数指定分组依据，支持多种参数设置来控制分组行为，如是否排序、是否包含 NA 等。详细示例展示了不同方式的数据分组及其效果。
    # 返回类型声明为 SeriesGroupBy
    ) -> SeriesGroupBy:
        # 从 pandas 库导入 SeriesGroupBy 类
        from pandas.core.groupby.generic import SeriesGroupBy

        # 如果 level 和 by 均未指定，则抛出 TypeError 异常
        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        # 如果不使用 as_index=True，则抛出 TypeError 异常
        if not as_index:
            raise TypeError("as_index=False only valid with DataFrame")

        # 返回一个 SeriesGroupBy 对象，传入参数为当前对象的属性和相关参数
        return SeriesGroupBy(
            obj=self,
            keys=by,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna,
        )

    # ----------------------------------------------------------------------
    # 统计信息，重写的 ndarray 方法

    # TODO: integrate bottleneck
    def count(self) -> int:
        """
        返回 Series 中非缺失（非 NA/null）观测值的数量。

        返回
        -------
        int
            Series 中非缺失值的数量。

        参见
        --------
        DataFrame.count : 统计每列或每行中的非缺失单元格。

        示例
        --------
        >>> s = pd.Series([0.0, 1.0, np.nan])
        >>> s.count()
        2
        """
        # 返回非缺失值的数量，使用 notna 方法检查和求和，转换为 int64 类型
        return notna(self._values).sum().astype("int64")
    def mode(self, dropna: bool = True) -> Series:
        """
        Return the mode(s) of the Series.

        The mode is the value that appears most often. There can be multiple modes.

        Always returns Series even if only one value is returned.

        Parameters
        ----------
        dropna : bool, default True
            Don't consider counts of NaN/NaT.

        Returns
        -------
        Series
            Modes of the Series in sorted order.

        See Also
        --------
        numpy.mode : Equivalent numpy function for computing median.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.std : Standard deviation of the values.
        Series.var : Variance of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.

        Examples
        --------
        >>> s = pd.Series([2, 4, 2, 2, 4, None])
        >>> s.mode()
        0    2.0
        dtype: float64

        More than one mode:

        >>> s = pd.Series([2, 4, 8, 2, 4, None])
        >>> s.mode()
        0    2.0
        1    4.0
        dtype: float64

        With and without considering null value:

        >>> s = pd.Series([2, 4, None, None, 4, None])
        >>> s.mode(dropna=False)
        0   NaN
        dtype: float64
        >>> s = pd.Series([2, 4, None, None, 4, None])
        >>> s.mode()
        0    4.0
        dtype: float64
        """
        # TODO: Add option for bins like value_counts()
        # 获取 Series 的值
        values = self._values
        # 如果值是 numpy 数组
        if isinstance(values, np.ndarray):
            # 使用算法计算 mode
            res_values = algorithms.mode(values, dropna=dropna)
        else:
            # 否则调用 values 的 _mode 方法
            res_values = values._mode(dropna=dropna)

        # 确保索引是类型稳定的（应始终使用 int 索引）
        return self._constructor(
            res_values,
            index=range(len(res_values)),
            name=self.name,
            copy=False,
            dtype=self.dtype,
        ).__finalize__(self, method="mode")
    def unique(self) -> ArrayLike:
        """
        Return unique values of Series object.

        Uniques are returned in order of appearance. Hash table-based unique,
        therefore does NOT sort.

        Returns
        -------
        ndarray or ExtensionArray
            The unique values returned as a NumPy array. See Notes.

        See Also
        --------
        Series.drop_duplicates : Return Series with duplicate values removed.
        unique : Top-level unique method for any 1-d array-like object.
        Index.unique : Return Index with unique values from an Index object.

        Notes
        -----
        Returns the unique values as a NumPy array. In case of an
        extension-array backed Series, a new
        :class:`~api.extensions.ExtensionArray` of that type with just
        the unique values is returned. This includes

            * Categorical
            * Period
            * Datetime with Timezone
            * Datetime without Timezone
            * Timedelta
            * Interval
            * Sparse
            * IntegerNA

        See Examples section.

        Examples
        --------
        >>> pd.Series([2, 1, 3, 3], name="A").unique()
        array([2, 1, 3])

        >>> pd.Series([pd.Timestamp("2016-01-01") for _ in range(3)]).unique()
        <DatetimeArray>
        ['2016-01-01 00:00:00']
        Length: 1, dtype: datetime64[s]

        >>> pd.Series(
        ...     [pd.Timestamp("2016-01-01", tz="US/Eastern") for _ in range(3)]
        ... ).unique()
        <DatetimeArray>
        ['2016-01-01 00:00:00-05:00']
        Length: 1, dtype: datetime64[s, US/Eastern]

        An Categorical will return categories in the order of
        appearance and with the same dtype.

        >>> pd.Series(pd.Categorical(list("baabc"))).unique()
        ['b', 'a', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> pd.Series(
        ...     pd.Categorical(list("baabc"), categories=list("abc"), ordered=True)
        ... ).unique()
        ['b', 'a', 'c']
        Categories (3, object): ['a' < 'b' < 'c']
        """
        # 调用父类的 unique 方法，返回唯一值
        return super().unique()

    @overload
    def drop_duplicates(
        self,
        *,
        keep: DropKeep = ...,
        inplace: Literal[False] = ...,
        ignore_index: bool = ...,
    ) -> Series: ...
    
    @overload
    def drop_duplicates(
        self, *, keep: DropKeep = ..., inplace: Literal[True], ignore_index: bool = ...
    ) -> None: ...
    
    @overload
    def drop_duplicates(
        self, *, keep: DropKeep = ..., inplace: bool = ..., ignore_index: bool = ...
    ) -> Series | None: ...
    
    def drop_duplicates(
        self,
        *,
        keep: DropKeep = "first",
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> Series | None:
        """
        返回删除重复值后的 Series。

        Parameters
        ----------
        keep : {'first', 'last', ``False``}, 默认 'first'
            处理删除重复值的方法:

            - 'first' : 保留第一次出现的重复值，删除其余重复值。
            - 'last' : 保留最后一次出现的重复值，删除其余重复值。
            - ``False`` : 删除所有重复值。

        inplace : bool, 默认 ``False``
            如果为 ``True``，则在原对象上执行操作并返回 None。

        ignore_index : bool, 默认 ``False``
            如果为 ``True``，结果将使用默认的索引 0, 1, …, n - 1。

            .. versionadded:: 2.0.0

        Returns
        -------
        Series 或 None
            删除重复值后的 Series 或如果 ``inplace=True`` 则返回 None。

        See Also
        --------
        Index.drop_duplicates : Index 上的等效方法。
        DataFrame.drop_duplicates : DataFrame 上的等效方法。
        Series.duplicated : Series 上的相关方法，指示重复的 Series 值。
        Series.unique : 返回数组形式的唯一值。

        Examples
        --------
        生成一个带有重复条目的 Series。

        >>> s = pd.Series(
        ...     ["llama", "cow", "llama", "beetle", "llama", "hippo"], name="animal"
        ... )
        >>> s
        0     llama
        1       cow
        2     llama
        3    beetle
        4     llama
        5     hippo
        Name: animal, dtype: object

        使用 'keep' 参数可以改变重复值的选择行为。值 'first' 保留每组重复条目中的第一个出现的条目。默认的 keep 值是 'first'。

        >>> s.drop_duplicates()
        0     llama
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object

        参数 'keep' 设置为 'last' 时，保留每组重复条目中的最后一个出现的条目。

        >>> s.drop_duplicates(keep="last")
        1       cow
        3    beetle
        4     llama
        5     hippo
        Name: animal, dtype: object

        参数 'keep' 设置为 ``False`` 时，删除所有重复条目。

        >>> s.drop_duplicates(keep=False)
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object
        """
        # 将 inplace 参数转换为布尔值，确保其有效性
        inplace = validate_bool_kwarg(inplace, "inplace")
        # 调用父类的 drop_duplicates 方法，并返回结果
        result = super().drop_duplicates(keep=keep)

        # 如果 ignore_index 为 True，则重新设置结果的索引为默认索引
        if ignore_index:
            result.index = default_index(len(result))

        # 如果 inplace 为 True，则在原对象上进行就地更新，并返回 None
        if inplace:
            self._update_inplace(result)
            return None
        else:
            # 否则返回处理后的结果 Series
            return result
    def duplicated(self, keep: DropKeep = "first") -> Series:
        """
        Indicate duplicate Series values.

        Duplicated values are indicated as ``True`` values in the resulting
        Series. Either all duplicates, all except the first or all except the
        last occurrence of duplicates can be indicated.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            Method to handle dropping duplicates:

            - 'first' : Mark duplicates as ``True`` except for the first
              occurrence.
            - 'last' : Mark duplicates as ``True`` except for the last
              occurrence.
            - ``False`` : Mark all duplicates as ``True``.

        Returns
        -------
        Series[bool]
            Series indicating whether each value has occurred in the
            preceding values.

        See Also
        --------
        Index.duplicated : Equivalent method on pandas.Index.
        DataFrame.duplicated : Equivalent method on pandas.DataFrame.
        Series.drop_duplicates : Remove duplicate values from Series.

        Examples
        --------
        By default, for each set of duplicated values, the first occurrence is
        set on False and all others on True:

        >>> animals = pd.Series(["llama", "cow", "llama", "beetle", "llama"])
        >>> animals.duplicated()
        0    False
        1    False
        2     True
        3    False
        4     True
        dtype: bool

        which is equivalent to

        >>> animals.duplicated(keep="first")
        0    False
        1    False
        2     True
        3    False
        4     True
        dtype: bool

        By using 'last', the last occurrence of each set of duplicated values
        is set on False and all others on True:

        >>> animals.duplicated(keep="last")
        0     True
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        By setting keep on ``False``, all duplicates are True:

        >>> animals.duplicated(keep=False)
        0     True
        1    False
        2     True
        3    False
        4     True
        dtype: bool
        """
        # 调用内部方法 _duplicated 处理重复值，并传递 keep 参数
        res = self._duplicated(keep=keep)
        # 使用结果构造新的 Series，保留当前对象的索引，且不进行复制操作
        result = self._constructor(res, index=self.index, copy=False)
        # 返回最终的 Series 结果，并使用 __finalize__ 方法确保与原始 Series 关联
        return result.__finalize__(self, method="duplicated")
    def idxmin(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Hashable:
        """
        Return the row label of the minimum value.

        If multiple values equal the minimum, the first row label with that
        value is returned.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, or if ``skipna=False``
            and there is an NA value, this method will raise a ``ValueError``.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        Index
            Label of the minimum value.

        Raises
        ------
        ValueError
            If the Series is empty.

        See Also
        --------
        numpy.argmin : Return indices of the minimum values
            along the given axis.
        DataFrame.idxmin : Return index of first occurrence of minimum
            over requested axis.
        Series.idxmax : Return index *label* of the first occurrence
            of maximum of values.

        Notes
        -----
        This method is the Series version of ``ndarray.argmin``. This method
        returns the label of the minimum, while ``ndarray.argmin`` returns
        the position. To get the position, use ``series.values.argmin()``.

        Examples
        --------
        >>> s = pd.Series(data=[1, None, 4, 1], index=["A", "B", "C", "D"])
        >>> s
        A    1.0
        B    NaN
        C    4.0
        D    1.0
        dtype: float64

        >>> s.idxmin()
        'A'
        """
        # 获取轴的编号，这里是为了与 DataFrame 兼容而保留的参数
        axis = self._get_axis_number(axis)
        # 调用 argmin 方法找到最小值的位置索引
        iloc = self.argmin(axis, skipna, *args, **kwargs)
        # 返回最小值所对应的索引标签
        return self.index[iloc]
    def idxmax(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Hashable:
        """
        Return the row label of the maximum value.

        If multiple values equal the maximum, the first row label with that
        value is returned.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, or if ``skipna=False``
            and there is an NA value, this method will raise a ``ValueError``.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        Index
            Label of the maximum value.

        Raises
        ------
        ValueError
            If the Series is empty.

        See Also
        --------
        numpy.argmax : Return indices of the maximum values
            along the given axis.
        DataFrame.idxmax : Return index of first occurrence of maximum
            over requested axis.
        Series.idxmin : Return index *label* of the first occurrence
            of minimum of values.

        Notes
        -----
        This method is the Series version of ``ndarray.argmax``. This method
        returns the label of the maximum, while ``ndarray.argmax`` returns
        the position. To get the position, use ``series.values.argmax()``.

        Examples
        --------
        >>> s = pd.Series(data=[1, None, 4, 3, 4], index=["A", "B", "C", "D", "E"])
        >>> s
        A    1.0
        B    NaN
        C    4.0
        D    3.0
        E    4.0
        dtype: float64

        >>> s.idxmax()
        'C'
        """
        # 确定轴的编号，用于内部方法调用
        axis = self._get_axis_number(axis)
        # 调用 self.argmax 方法，获取最大值所在的位置
        iloc = self.argmax(axis, skipna, *args, **kwargs)
        # 根据位置获取对应的索引标签，并返回
        return self.index[iloc]
    # 定义 round 方法，用于将 Series 中的每个值四舍五入到指定小数位数。
    def round(self, decimals: int = 0, *args, **kwargs) -> Series:
        """
        Round each value in a Series to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        Series
            Rounded values of the Series.

        See Also
        --------
        numpy.around : Round values of an np.array.
        DataFrame.round : Round values of a DataFrame.

        Notes
        -----
        For values exactly halfway between rounded decimal values, pandas rounds
        to the nearest even value (e.g. -0.5 and 0.5 round to 0.0, 1.5 and 2.5
        round to 2.0, etc.).

        Examples
        --------
        >>> s = pd.Series([-0.5, 0.1, 2.5, 1.3, 2.7])
        >>> s.round()
        0   -0.0
        1    0.0
        2    2.0
        3    1.0
        4    3.0
        dtype: float64
        """
        # 验证并处理 round 方法的参数
        nv.validate_round(args, kwargs)
        # 使用底层数据管理器对 Series 中的值进行四舍五入操作
        new_mgr = self._mgr.round(decimals=decimals)
        # 使用新的数据管理器构造一个新的 Series 对象，并继承当前对象的一些属性
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(
            self, method="round"
        )

    # 定义 quantile 方法，用于计算 Series 的分位数
    @overload
    def quantile(
        self, q: float = ..., interpolation: QuantileInterpolation = ...
    ) -> float: ...

    @overload
    def quantile(
        self,
        q: Sequence[float] | AnyArrayLike,
        interpolation: QuantileInterpolation = ...,
    ) -> Series: ...

    @overload
    def quantile(
        self,
        q: float | Sequence[float] | AnyArrayLike = ...,
        interpolation: QuantileInterpolation = ...,
    ) -> float | Series: ...

    def quantile(
        self,
        q: float | Sequence[float] | AnyArrayLike = 0.5,
        interpolation: QuantileInterpolation = "linear",
    ) -> float | Series:
        """
        Return value at the given quantile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            The quantile(s) to compute, which can lie in range: 0 <= q <= 1.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points `i` and `j`:

                * linear: `i + (j - i) * (x-i)/(j-i)`, where `(x-i)/(j-i)` is
                  the fractional part of the index surrounded by `i > j`.
                * lower: `i`.
                * higher: `j`.
                * nearest: `i` or `j` whichever is nearest.
                * midpoint: (`i` + `j`) / 2.

        Returns
        -------
        float or Series
            If ``q`` is an array, a Series will be returned where the
            index is ``q`` and the values are the quantiles, otherwise
            a float will be returned.

        See Also
        --------
        core.window.Rolling.quantile : Calculate the rolling quantile.
        numpy.percentile : Returns the q-th percentile(s) of the array elements.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.quantile(0.5)
        2.5
        >>> s.quantile([0.25, 0.5, 0.75])
        0.25    1.75
        0.50    2.50
        0.75    3.25
        dtype: float64
        """
        validate_percentile(q)  # 调用函数验证 q 的有效性

        # We dispatch to DataFrame so that core.internals only has to worry
        #  about 2D cases.
        df = self.to_frame()  # 将当前 Series 转换为 DataFrame

        result = df.quantile(q=q, interpolation=interpolation, numeric_only=False)
        if result.ndim == 2:
            result = result.iloc[:, 0]  # 如果结果是二维的，只取第一列数据

        if is_list_like(q):
            result.name = self.name  # 设置结果的名称
            idx = Index(q, dtype=np.float64)  # 创建索引对象
            return self._constructor(result, index=idx, name=self.name)  # 返回一个新的 Series 对象
        else:
            # scalar
            return result.iloc[0]  # 返回结果的第一个元素

    def corr(
        self,
        other: Series,
        method: CorrelationMethod = "pearson",
        min_periods: int | None = None,
    ):
        """
        Compute correlation with another Series.

        Parameters
        ----------
        other : Series
            The other Series to compute correlation with.
        method : {'pearson', 'kendall', 'spearman'}
            The correlation method to use:

                * pearson : Standard correlation coefficient.
                * kendall : Kendall Tau correlation coefficient.
                * spearman : Spearman rank correlation.

            Default is 'pearson'.
        min_periods : int, optional
            Minimum number of observations required to have a valid result.
            Default is None, meaning the result will be calculated regardless of min_periods.

        Returns
        -------
        float
            The correlation coefficient between the Series and the other Series.

        See Also
        --------
        DataFrame.corr : Compute pairwise correlation of columns.
        numpy.corrcoef : Compute correlation coefficient for two arrays.

        Notes
        -----
        NaNs are automatically excluded from the calculation.

        Examples
        --------
        >>> s1 = pd.Series([1, 2, 3, 4])
        >>> s2 = pd.Series([2, 3, 4, 5])
        >>> s1.corr(s2)
        1.0
        """
        pass  # 计算与另一个 Series 的相关性，具体实现留空

    def cov(
        self,
        other: Series,
        min_periods: int | None = None,
        ddof: int | None = 1,
    ):
        """
        Compute covariance with another Series.

        Parameters
        ----------
        other : Series
            The other Series to compute covariance with.
        min_periods : int, optional
            Minimum number of observations required to have a valid result.
            Default is None, meaning the result will be calculated regardless of min_periods.
        ddof : int, optional
            Delta degrees of freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements. By default ddof is 1.

        Returns
        -------
        float
            The covariance between the Series and the other Series.

        See Also
        --------
        DataFrame.cov : Compute pairwise covariance of columns.
        numpy.cov : Compute covariance matrix for two arrays.

        Notes
        -----
        NaNs are automatically excluded from the calculation.

        Examples
        --------
        >>> s1 = pd.Series([1, 2, 3, 4])
        >>> s2 = pd.Series([2, 3, 4, 5])
        >>> s1.cov(s2)
        1.25
        """
        pass  # 计算与另一个 Series 的协方差，具体实现留空
    ) -> float:
        """
        Compute covariance with Series, excluding missing values.

        The two `Series` objects are not required to be the same length and
        will be aligned internally before the covariance is calculated.

        Parameters
        ----------
        other : Series
            Series with which to compute the covariance.
        min_periods : int, optional
            Minimum number of observations needed to have a valid result.
        ddof : int, default 1
            Delta degrees of freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.

        Returns
        -------
        float
            Covariance between Series and other normalized by N-1
            (unbiased estimator).

        See Also
        --------
        DataFrame.cov : Compute pairwise covariance of columns.

        Examples
        --------
        >>> s1 = pd.Series([0.90010907, 0.13484424, 0.62036035])
        >>> s2 = pd.Series([0.12528585, 0.26962463, 0.51111198])
        >>> s1.cov(s2)
        -0.01685762652715874
        """
        # Aligns this Series and the `other` Series, keeping only common indices
        this, other = self.align(other, join="inner")
        # If no common indices after alignment, return NaN
        if len(this) == 0:
            return np.nan
        # Convert aligned Series to numpy arrays with float dtype, handling NaNs
        this_values = this.to_numpy(dtype=float, na_value=np.nan, copy=False)
        other_values = other.to_numpy(dtype=float, na_value=np.nan, copy=False)
        # Compute covariance between aligned Series using nanops.nancov function
        return nanops.nancov(
            this_values, other_values, min_periods=min_periods, ddof=ddof
        )

    @doc(
        klass="Series",
        extra_params="",
        other_klass="DataFrame",
        examples=dedent(
            """
        Difference with previous row

        >>> s = pd.Series([1, 1, 2, 3, 5, 8])
        >>> s.diff()
        0    NaN
        1    0.0
        2    1.0
        3    1.0
        4    2.0
        5    3.0
        dtype: float64

        Difference with 3rd previous row

        >>> s.diff(periods=3)
        0    NaN
        1    NaN
        2    NaN
        3    2.0
        4    4.0
        5    6.0
        dtype: float64

        Difference with following row

        >>> s.diff(periods=-1)
        0    0.0
        1   -1.0
        2   -1.0
        3   -2.0
        4   -3.0
        5    NaN
        dtype: float64

        Overflow in input dtype

        >>> s = pd.Series([1, 0], dtype=np.uint8)
        >>> s.diff()
        0      NaN
        1    255.0
        dtype: float64"""
        ),
    )
    # 计算元素的第一个离散差分。

    # 根据 {klass} 元素与另一个 {klass} 元素的差值（默认为前一行的元素）。
    # periods：int，默认为1，用于计算差分的移动周期，可以接受负值。
    # {extra_params}

    # 返回
    # -------
    # {klass}
    #     Series 的第一个差分结果。

    # 参见
    # --------
    # {klass}.pct_change：给定周期的百分比变化。
    # {klass}.shift：将索引按所需周期移动，可选择时间频率。
    # {other_klass}.diff：对象的第一个离散差分。

    # 注意
    # -----
    # 对于布尔类型，使用 :meth:`operator.xor` 而不是 :meth:`operator.sub`。
    # 结果根据 {klass} 的当前数据类型计算，但结果的数据类型始终为 float64。

    # 示例
    # --------
    # {examples}
    def diff(self, periods: int = 1) -> Series:
        if not lib.is_integer(periods):
            if not (is_float(periods) and periods.is_integer()):
                raise ValueError("periods must be an integer")
        
        # 调用 algorithms 模块中的 diff 函数，计算 self._values 的差分结果
        result = algorithms.diff(self._values, periods)
        
        # 返回用差分结果构造的新 Series 对象，保留原索引，不复制数据，最终通过 diff 方法完成对象初始化
        return self._constructor(result, index=self.index, copy=False).__finalize__(
            self, method="diff"
        )

    # 计算滞后为 N 的自相关系数。

    # 此方法计算 Series 与其自身的滞后版本之间的 Pearson 相关系数。

    # 参数
    # ----------
    # lag : int，默认为1，应用于计算自相关之前的滞后数量。

    # 返回
    # -------
    # float
    #     self 与 self.shift(lag) 之间的 Pearson 相关系数。

    # 参见
    # --------
    # Series.corr：计算两个 Series 之间的相关系数。
    # Series.shift：按所需的周期移动索引。
    # DataFrame.corr：计算列的成对相关性。
    # DataFrame.corrwith：计算两个 DataFrame 对象的行或列之间的成对相关性。

    # 注意
    # -----
    # 如果 Pearson 相关系数无法定义，则返回 'NaN'。

    # 示例
    # --------
    # >>> s = pd.Series([0.25, 0.5, 0.2, -0.05])
    # >>> s.autocorr()  # doctest: +ELLIPSIS
    # 0.10355...
    # >>> s.autocorr(lag=2)  # doctest: +ELLIPSIS
    # -0.99999...

    # 如果 Pearson 相关系数无法定义，则返回 'NaN'。

    # >>> s = pd.Series([1, 0, 0, 0])
    # >>> s.autocorr()
    # nan
    def autocorr(self, lag: int = 1) -> float:
        # 调用 self.shift(lag) 方法，计算 self 与其滞后版本之间的相关系数，并返回结果
        return self.corr(cast(Series, self.shift(lag)))
    def dot(self, other: AnyArrayLike | DataFrame) -> Series | np.ndarray:
        """
        Compute the dot product between the Series and the columns of other.

        This method computes the dot product between the Series and another
        one, or the Series and each column of a DataFrame, or the Series and
        each column of an array.

        It can also be called using `self @ other`.

        Parameters
        ----------
        other : Series, DataFrame or array-like
            The other object to compute the dot product with its columns.

        Returns
        -------
        scalar, Series or numpy.ndarray
            Return the dot product of the Series and other if other is a
            Series, the Series of the dot product of Series and each row of
            other if other is a DataFrame or a numpy.ndarray between the Series
            and each column of the numpy array.

        See Also
        --------
        DataFrame.dot: Compute the matrix product with the DataFrame.
        Series.mul: Multiplication of series and other, element-wise.

        Notes
        -----
        The Series and other have to share the same index if other is a Series
        or a DataFrame.

        Examples
        --------
        >>> s = pd.Series([0, 1, 2, 3])
        >>> other = pd.Series([-1, 2, -3, 4])
        >>> s.dot(other)
        8
        >>> s @ other
        8
        >>> df = pd.DataFrame([[0, 1], [-2, 3], [4, -5], [6, 7]])
        >>> s.dot(df)
        0    24
        1    14
        dtype: int64
        >>> arr = np.array([[0, 1], [-2, 3], [4, -5], [6, 7]])
        >>> s.dot(arr)
        array([24, 14])
        """
        # 根据 other 的类型判断需要进行的操作
        if isinstance(other, (Series, ABCDataFrame)):
            # 获取 Series 和 other 共同的索引
            common = self.index.union(other.index)
            # 如果共同索引大于任何一个对象的索引长度，抛出异常
            if len(common) > len(self.index) or len(common) > len(other.index):
                raise ValueError("matrices are not aligned")

            # 根据共同索引重新索引 Series 和 other
            left = self.reindex(index=common)
            right = other.reindex(index=common)
            # 获取左右对象的数值数组
            lvals = left.values
            rvals = right.values
        else:
            # 如果 other 是数组或其他类型，直接使用其数值
            lvals = self.values
            rvals = np.asarray(other)
            # 检查形状是否匹配
            if lvals.shape[0] != rvals.shape[0]:
                raise Exception(
                    f"Dot product shape mismatch, {lvals.shape} vs {rvals.shape}"
                )

        # 根据 other 的具体类型执行不同的 dot product 操作
        if isinstance(other, ABCDataFrame):
            # 如果 other 是 DataFrame，返回一个新的 Series 对象，其值为 dot product 结果
            return self._constructor(
                np.dot(lvals, rvals), index=other.columns, copy=False
            ).__finalize__(self, method="dot")
        elif isinstance(other, Series):
            # 如果 other 是 Series，返回 dot product 的标量结果
            return np.dot(lvals, rvals)
        elif isinstance(rvals, np.ndarray):
            # 如果 rvals 是 numpy 数组，返回 dot product 的 numpy 数组结果
            return np.dot(lvals, rvals)
        else:  # pragma: no cover
            # 处理未支持的其他类型情况
            raise TypeError(f"unsupported type: {type(other)}")

    def __matmul__(self, other):
        """
        Matrix multiplication using binary `@` operator.
        """
        # 通过 @ 操作符调用 dot 方法实现矩阵乘法
        return self.dot(other)
    def __rmatmul__(self, other):
        """
        Matrix multiplication using binary `@` operator.
        """
        # Perform matrix multiplication of self with the transpose of `other`
        return self.dot(np.transpose(other))

    @doc(base.IndexOpsMixin.searchsorted, klass="Series")
    # This method overrides the `searchsorted` method from `base.IndexOpsMixin`.
    # The signature is adjusted to accept `NumpyValueArrayLike` or `ExtensionArray` for `value`,
    # and optionally `side` and `sorter` parameters.
    def searchsorted(  # type: ignore[override]
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        # Delegate the searchsorted operation to the base class `IndexOpsMixin`
        return base.IndexOpsMixin.searchsorted(self, value, side=side, sorter=sorter)

    # -------------------------------------------------------------------
    # Combination

    def _append(
        self, to_append, ignore_index: bool = False, verify_integrity: bool = False
    ):
        from pandas.core.reshape.concat import concat

        # Check if `to_append` is a list or tuple; if so, concatenate `self` with each item
        if isinstance(to_append, (list, tuple)):
            to_concat = [self]
            to_concat.extend(to_append)
        else:
            to_concat = [self, to_append]
        
        # Check if any element in `to_concat` (excluding `self`) is an instance of `ABCDataFrame`
        if any(isinstance(x, (ABCDataFrame,)) for x in to_concat[1:]):
            msg = "to_append should be a Series or list/tuple of Series, got DataFrame"
            raise TypeError(msg)
        
        # Concatenate all elements in `to_concat` into a single DataFrame using `concat` function
        return concat(
            to_concat, ignore_index=ignore_index, verify_integrity=verify_integrity
        )
    @doc(
        _shared_docs["compare"],
        dedent(
            """
            Returns
            -------
            Series or DataFrame
                If axis is 0 or 'index' the result will be a Series.
                The resulting index will be a MultiIndex with 'self' and 'other'
                stacked alternately at the inner level.

                If axis is 1 or 'columns' the result will be a DataFrame.
                It will have two columns namely 'self' and 'other'.

            See Also
            --------
            DataFrame.compare : Compare with another DataFrame and show differences.

            Notes
            -----
            Matching NaNs will not appear as a difference.

            Examples
            --------
            >>> s1 = pd.Series(["a", "b", "c", "d", "e"])
            >>> s2 = pd.Series(["a", "a", "c", "b", "e"])

            Align the differences on columns

            >>> s1.compare(s2)
              self other
            1    b     a
            3    d     b

            Stack the differences on indices

            >>> s1.compare(s2, align_axis=0)
            1  self     b
               other    a
            3  self     d
               other    b
            dtype: object

            Keep all original rows

            >>> s1.compare(s2, keep_shape=True)
              self other
            0  NaN   NaN
            1    b     a
            2  NaN   NaN
            3    d     b
            4  NaN   NaN

            Keep all original rows and also all original values

            >>> s1.compare(s2, keep_shape=True, keep_equal=True)
              self other
            0    a     a
            1    b     a
            2    c     c
            3    d     b
            4    e     e
            """
        ),
        klass=_shared_doc_kwargs["klass"],
    )
    # 定义一个文档化的函数装饰器，用于为 compare 方法添加文档说明
    def compare(
        self,
        other: Series,
        align_axis: Axis = 1,  # 指定对齐轴，默认为列对齐
        keep_shape: bool = False,  # 是否保持原始形状，默认为 False
        keep_equal: bool = False,  # 是否保持相等性，默认为 False
        result_names: Suffixes = ("self", "other"),  # 结果命名后缀，默认为 ("self", "other")
    ) -> DataFrame | Series:  # 返回值可以是 DataFrame 或 Series
        # 调用父类的 compare 方法，并返回其结果
        return super().compare(
            other=other,
            align_axis=align_axis,
            keep_shape=keep_shape,
            keep_equal=keep_equal,
            result_names=result_names,
        )

    # 定义一个函数 combine，用于组合操作
    def combine(
        self,
        other: Series | Hashable,  # 可以是 Series 或可散列对象
        func: Callable[[Hashable, Hashable], Hashable],  # 接受两个可散列对象并返回可散列对象的函数
        fill_value: Hashable | None = None,  # 填充值，默认为 None
        axis: Optional[Axis] = None,  # 操作轴，默认为 None
    ) -> DataFrame | Series:  # 返回值可以是 DataFrame 或 Series
    def combine_first(self, other) -> Series:
        """
        Update null elements with value in the same location in 'other'.

        Combine two Series objects by filling null values in one Series with
        non-null values from the other Series. Result index will be the union
        of the two indexes.

        Parameters
        ----------
        other : Series
            The value(s) to be used for filling null values.

        Returns
        -------
        Series
            The result of combining the provided Series with the other object.

        See Also
        --------
        Series.combine : Perform element-wise operation on two Series
            using a given function.

        Examples
        --------
        >>> s1 = pd.Series([1, np.nan])
        >>> s2 = pd.Series([3, 4, 5])
        >>> s1.combine_first(s2)
        0    1.0
        1    4.0
        2    5.0
        dtype: float64

        Null values still persist if the location of that null value
        does not exist in `other`

        >>> s1 = pd.Series({"falcon": np.nan, "eagle": 160.0})
        >>> s2 = pd.Series({"eagle": 200.0, "duck": 30.0})
        >>> s1.combine_first(s2)
        duck       30.0
        eagle     160.0
        falcon      NaN
        dtype: float64
        """
        
        # 导入 pandas.core.reshape.concat 模块中的 concat 函数
        from pandas.core.reshape.concat import concat
        
        # 如果两个 Series 的数据类型相同
        if self.dtype == other.dtype:
            # 并且两个 Series 的索引完全相同
            if self.index.equals(other.index):
                # 将当前 Series 中的空值位置使用 `other` 中的值填充，并返回结果
                return self.mask(self.isna(), other)
            # 如果索引不相同，但当前 Series 支持空值，并且不是稀疏数据类型
            elif self._can_hold_na and not isinstance(self.dtype, SparseDtype):
                # 将两个 Series 按照外连接方式对齐
                this, other = self.align(other, join="outer")
                # 使用 `other` 中的值填充当前 Series 中的空值位置，并返回结果
                return this.mask(this.isna(), other)
        
        # 获取两个 Series 索引的并集
        new_index = self.index.union(other.index)
        
        # 保存当前 Series 到 `this`
        this = self
        
        # 确定要保留的每个 Series 的索引子集
        keep_other = other.index.difference(this.index[notna(this)])
        keep_this = this.index.difference(keep_other)
        
        # 使用索引子集重新索引当前 Series 和 `other`
        this = this.reindex(keep_this)
        other = other.reindex(keep_other)
        
        # 如果当前 Series 的数据类型是日期时间类型，而 `other` 的数据类型不是日期时间类型
        if this.dtype.kind == "M" and other.dtype.kind != "M":
            # 尝试将 `other` 转换为日期时间类型
            other = to_datetime(other)
        
        # 将 `this` 和 `other` 进行连接，并重新索引为 `new_index`
        combined = concat([this, other])
        combined = combined.reindex(new_index)
        
        # 返回最终合并结果，并使用 `self` 进行最终化处理
        return combined.__finalize__(self, method="combine_first")
    # ----------------------------------------------------------------------
    # Reindexing, sorting

    @overload
    # 定义重载方法，用于对 Series 进行排序
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
    ) -> Series: ...
    # 返回排序后的 Series 对象

    @overload
    # 定义重载方法，用于对 Series 进行排序（原地排序）
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
    # 不返回任何值（原地修改）

    @overload


这段代码定义了 `sort_values` 方法的多个重载版本，用于对 Pandas Series 进行排序操作。
    # 定义 Series 对象的排序方法，按值排序
    def sort_values(
        self,
        *,
        axis: Axis = 0,                           # 指定排序的轴，默认为第0轴（行索引）
        ascending: bool | Sequence[bool] = True,  # 控制排序顺序，True 表示升序，默认为 True
        inplace: bool = False,                    # 是否原地修改，默认为 False
        kind: SortKind = "quicksort",             # 指定排序算法，默认为快速排序
        na_position: NaPosition = "last",         # 指定缺失值的处理位置，默认在排序后的最后
        ignore_index: bool = False,               # 是否忽略索引，默认为 False
        key: ValueKeyFunc | None = None,          # 指定排序键的函数，默认为 None
    ) -> Series | None:                          # 返回排序后的 Series 对象或 None

    # 定义 Series 对象的索引排序方法，按索引排序
    @overload
    def sort_index(
        self,
        *,
        axis: Axis = 0,                           # 指定排序的轴，默认为第0轴（行索引）
        level: IndexLabel | None = None,          # 指定多级索引的级别，默认为 None
        ascending: bool | Sequence[bool] = True,  # 控制排序顺序，True 表示升序，默认为 True
        inplace: Literal[True],                   # 是否原地修改，必须为 True
        kind: SortKind = "quicksort",             # 指定排序算法，默认为快速排序
        na_position: NaPosition = "last",         # 指定缺失值的处理位置，默认在排序后的最后
        sort_remaining: bool = True,              # 是否对未指定的轴进行排序，默认为 True
        ignore_index: bool = False,               # 是否忽略索引，默认为 False
        key: IndexKeyFunc = ...,                  # 指定排序键的函数
    ) -> None:                                   # 返回 None，表示原地修改

    @overload
    def sort_index(
        self,
        *,
        axis: Axis = 0,                           # 指定排序的轴，默认为第0轴（行索引）
        level: IndexLabel | None = None,          # 指定多级索引的级别，默认为 None
        ascending: bool | Sequence[bool] = True,  # 控制排序顺序，True 表示升序，默认为 True
        inplace: Literal[False] = ...,            # 是否原地修改，必须为 False
        kind: SortKind = ...,                     # 指定排序算法，默认为快速排序
        na_position: NaPosition = ...,            # 指定缺失值的处理位置，默认在排序后的最后
        sort_remaining: bool = ...,               # 是否对未指定的轴进行排序
        ignore_index: bool = ...,                 # 是否忽略索引
        key: IndexKeyFunc = ...,                  # 指定排序键的函数
    ) -> Series:                                 # 返回排序后的 Series 对象

    @overload
    def sort_index(
        self,
        *,
        axis: Axis = 0,                           # 指定排序的轴，默认为第0轴（行索引）
        level: IndexLabel | None = None,          # 指定多级索引的级别，默认为 None
        ascending: bool | Sequence[bool] = True,  # 控制排序顺序，True 表示升序，默认为 True
        inplace: bool = ...,                      # 是否原地修改
        kind: SortKind = ...,                     # 指定排序算法，默认为快速排序
        na_position: NaPosition = ...,            # 指定缺失值的处理位置，默认在排序后的最后
        sort_remaining: bool = ...,               # 是否对未指定的轴进行排序
        ignore_index: bool = ...,                 # 是否忽略索引
        key: IndexKeyFunc = ...,                  # 指定排序键的函数
    ) -> Series | None:                          # 返回排序后的 Series 对象或 None

    def sort_index(
        self,
        *,
        axis: Axis = 0,                           # 指定排序的轴，默认为第0轴（行索引）
        level: IndexLabel | None = None,          # 指定多级索引的级别，默认为 None
        ascending: bool | Sequence[bool] = True,  # 控制排序顺序，True 表示升序，默认为 True
        inplace: bool = False,                    # 是否原地修改，默认为 False
        kind: SortKind = "quicksort",             # 指定排序算法，默认为快速排序
        na_position: NaPosition = "last",         # 指定缺失值的处理位置，默认在排序后的最后
        sort_remaining: bool = True,              # 是否对未指定的轴进行排序，默认为 True
        ignore_index: bool = False,               # 是否忽略索引，默认为 False
        key: IndexKeyFunc | None = None,          # 指定排序键的函数，默认为 None
    ) -> Series | None:                          # 返回排序后的 Series 对象或 None

    # 定义 Series 对象的按索引排序方法，返回索引排序的顺序
    def argsort(
        self,
        axis: Axis = 0,                           # 指定排序的轴，默认为第0轴（行索引）
        kind: SortKind = "quicksort",             # 指定排序算法，默认为快速排序
        order: None = None,                       # 保留为 None
        stable: None = None,                      # 保留为 None
    ) -> Series:
        """
        Return the integer indices that would sort the Series values.

        Override ndarray.argsort. Argsorts the value, omitting NA/null values,
        and places the result in the same locations as the non-NA values.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        kind : {'mergesort', 'quicksort', 'heapsort', 'stable'}, default 'quicksort'
            Choice of sorting algorithm. See :func:`numpy.sort` for more
            information. 'mergesort' and 'stable' are the only stable algorithms.
        order : None
            Has no effect but is accepted for compatibility with numpy.
        stable : None
            Has no effect but is accepted for compatibility with numpy.

        Returns
        -------
        Series[np.intp]
            Positions of values within the sort order with -1 indicating
            nan values.

        See Also
        --------
        numpy.ndarray.argsort : Returns the indices that would sort this array.

        Examples
        --------
        >>> s = pd.Series([3, 2, 1])
        >>> s.argsort()
        0    2
        1    1
        2    0
        dtype: int64
        """
        # 如果 axis 不为 -1，则获取轴的编号
        if axis != -1:
            self._get_axis_number(axis)

        # 使用 self.array.argsort() 方法对值进行排序，返回排序后的索引
        result = self.array.argsort(kind=kind)

        # 使用排序后的索引创建新的 Series 对象
        res = self._constructor(
            result, index=self.index, name=self.name, dtype=np.intp, copy=False
        )
        # 返回最终的排序结果，并继承原始 Series 的属性
        return res.__finalize__(self, method="argsort")

    def nlargest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ):
        """
        Return the largest `n` elements in the Series.

        Parameters
        ----------
        n : int, default 5
            Number of largest elements to select.
        keep : {'first', 'last', 'all'}, default 'first'
            Determine which elements to keep when duplicates exist.
            - 'first': Keep the first occurrence.
            - 'last': Keep the last occurrence.
            - 'all': Keep all occurrences.

        Returns
        -------
        Series
            Subset of the Series containing the largest `n` elements.

        Notes
        -----
        The Series is sorted in descending order based on its values.

        Examples
        --------
        >>> s = pd.Series([3, 1, 2, 2, 4])
        >>> s.nlargest(2)
        4    4
        0    3
        dtype: int64
        """
        # 省略部分代码，用于返回指定数量的最大元素

    def nsmallest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ):
        """
        Return the smallest `n` elements in the Series.

        Parameters
        ----------
        n : int, default 5
            Number of smallest elements to select.
        keep : {'first', 'last', 'all'}, default 'first'
            Determine which elements to keep when duplicates exist.
            - 'first': Keep the first occurrence.
            - 'last': Keep the last occurrence.
            - 'all': Keep all occurrences.

        Returns
        -------
        Series
            Subset of the Series containing the smallest `n` elements.

        Notes
        -----
        The Series is sorted in ascending order based on its values.

        Examples
        --------
        >>> s = pd.Series([3, 1, 2, 2, 4])
        >>> s.nsmallest(2)
        1    1
        2    2
        dtype: int64
        """
        # 省略部分代码，用于返回指定数量的最小元素

    def swaplevel(
        self, i: Level = -2, j: Level = -1, copy: bool | lib.NoDefault = lib.no_default
    ):
        """
        Swap levels i and j in a MultiIndex.

        Parameters
        ----------
        i, j : int, str, or level name
            Levels of the MultiIndex to be swapped. Can pass level number
            or name.
        copy : bool, default lib.no_default
            Not supported yet.

        Returns
        -------
        Series
            A Series with levels swapped.

        Raises
        ------
        NotImplementedError
            If copy=True.

        Examples
        --------
        >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        >>> index = pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
        >>> s = pd.Series([10, 20, 30, 40], index=index)
        >>> s.swaplevel('number', 'color')
        color  number
        red    1        10
        blue   1        20
        red    2        30
        blue   2        40
        """
        # 省略部分代码，用于交换 MultiIndex 中的级别
    def reorder_levels(self, order: Sequence[Level]) -> Series:
        """
        Rearrange index levels using input order.

        May not drop or duplicate levels.

        Parameters
        ----------
        order : list of int representing new level order
            Reference level by number or key.

        Returns
        -------
        type of caller (new object)

        Examples
        --------
        >>> arrays = [
        ...     np.array(["dog", "dog", "cat", "cat", "bird", "bird"]),
        ...     np.array(["white", "black", "white", "black", "white", "black"]),
        ... ]
        >>> s = pd.Series([1, 2, 3, 3, 5, 2], index=arrays)
        >>> s
        dog   white    1
              black    2
        cat   white    3
              black    3
        bird  white    5
              black    2
        dtype: int64
        >>> s.reorder_levels([1, 0])
        white  dog     1
        black  dog     2
        white  cat     3
        black  cat     3
        white  bird    5
        black  bird    2
        dtype: int64
        """
        # 检查索引是否为多级索引，如果不是则抛出异常
        if not isinstance(self.index, MultiIndex):  # pragma: no cover
            raise Exception("Can only reorder levels on a hierarchical axis.")

        # 复制当前对象以确保不修改原对象，而是在副本上操作
        result = self.copy(deep=False)
        # 确保结果对象的索引是多级索引
        assert isinstance(result.index, MultiIndex)
        # 使用给定的顺序重排索引的层级
        result.index = result.index.reorder_levels(order)
        # 返回重排后的结果对象
        return result
    def explode(self, ignore_index: bool = False) -> Series:
        """
        Transform each element of a list-like to a row.

        Parameters
        ----------
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.

        Returns
        -------
        Series
            Exploded lists to rows; index will be duplicated for these rows.

        See Also
        --------
        Series.str.split : Split string values on specified separator.
        Series.unstack : Unstack, a.k.a. pivot, Series with MultiIndex
            to produce DataFrame.
        DataFrame.melt : Unpivot a DataFrame from wide format to long format.
        DataFrame.explode : Explode a DataFrame from list-like
            columns to long format.

        Notes
        -----
        This routine will explode list-likes including lists, tuples, sets,
        Series, and np.ndarray. The result dtype of the subset rows will
        be object. Scalars will be returned unchanged, and empty list-likes will
        result in a np.nan for that row. In addition, the ordering of elements in
        the output will be non-deterministic when exploding sets.

        Reference :ref:`the user guide <reshaping.explode>` for more examples.

        Examples
        --------
        >>> s = pd.Series([[1, 2, 3], "foo", [], [3, 4]])
        >>> s
        0    [1, 2, 3]
        1          foo
        2           []
        3       [3, 4]
        dtype: object

        >>> s.explode()
        0      1
        0      2
        0      3
        1    foo
        2    NaN
        3      3
        3      4
        dtype: object
        """
        # 如果数据类型是扩展数据类型，则调用私有方法 _explode() 处理
        if isinstance(self.dtype, ExtensionDtype):
            values, counts = self._values._explode()
        # 如果 Series 长度不为零且数据类型是 object 类型，则调用 reshape 模块的 explode 函数处理
        elif len(self) and is_object_dtype(self.dtype):
            values, counts = reshape.explode(np.asarray(self._values))
        else:
            # 如果不符合上述条件，直接复制当前 Series 对象
            result = self.copy()
            # 如果 ignore_index 为 True，则重置索引为默认的 0, 1, ..., n-1 样式
            return result.reset_index(drop=True) if ignore_index else result

        # 根据 ignore_index 的值决定生成的索引方式
        if ignore_index:
            index: Index = default_index(len(values))
        else:
            index = self.index.repeat(counts)

        # 根据处理后的 values 和索引创建新的 Series 对象，并返回
        return self._constructor(values, index=index, name=self.name, copy=False)

    def unstack(
        self,
        level: IndexLabel = -1,
        fill_value: Hashable | None = None,
        sort: bool = True,
    ) -> DataFrame:
        """
        Unstack, also known as pivot, Series with MultiIndex to produce DataFrame.

        Parameters
        ----------
        level : int, str, or list of these, default last level
            Level(s) to unstack, can pass level name.
        fill_value : scalar value, default None
            Value to use when replacing NaN values.
        sort : bool, default True
            Sort the level(s) in the resulting MultiIndex columns.

        Returns
        -------
        DataFrame
            Unstacked Series.

        See Also
        --------
        DataFrame.unstack : Pivot the MultiIndex of a DataFrame.

        Notes
        -----
        Reference :ref:`the user guide <reshaping.stacking>` for more examples.

        Examples
        --------
        >>> s = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.MultiIndex.from_product([["one", "two"], ["a", "b"]]),
        ... )
        >>> s
        one  a    1
             b    2
        two  a    3
             b    4
        dtype: int64

        >>> s.unstack(level=-1)
             a  b
        one  1  2
        two  3  4

        >>> s.unstack(level=0)
           one  two
        a    1    3
        b    2    4
        """
        # 导入 unstack 函数用于执行 Series 的展开操作
        from pandas.core.reshape.reshape import unstack

        # 调用 unstack 函数，对当前的 Series 对象进行展开操作
        return unstack(self, level, fill_value, sort)

    # ----------------------------------------------------------------------
    # function application

    def map(
        self,
        arg: Callable | Mapping | Series,
        na_action: Literal["ignore"] | None = None,
    ) -> Series:
        """
        Map values of Series according to an input mapping or function.

        Used for substituting each value in a Series with another value,
        that may be derived from a function, a ``dict`` or
        a :class:`Series`.

        Parameters
        ----------
        arg : function, collections.abc.Mapping subclass or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NaN values, without passing them to the
            mapping correspondence.

        Returns
        -------
        Series
            Same index as caller.

        See Also
        --------
        Series.apply : For applying more complex functions on a Series.
        Series.replace: Replace values given in `to_replace` with `value`.
        DataFrame.apply : Apply a function row-/column-wise.
        DataFrame.map : Apply a function elementwise on a whole DataFrame.

        Notes
        -----
        When ``arg`` is a dictionary, values in Series that are not in the
        dictionary (as keys) are converted to ``NaN``. However, if the
        dictionary is a ``dict`` subclass that defines ``__missing__`` (i.e.
        provides a method for default values), then this default is used
        rather than ``NaN``.

        Examples
        --------
        >>> s = pd.Series(["cat", "dog", np.nan, "rabbit"])
        >>> s
        0      cat
        1      dog
        2      NaN
        3   rabbit
        dtype: object

        ``map`` accepts a ``dict`` or a ``Series``. Values that are not found
        in the ``dict`` are converted to ``NaN``, unless the dict has a default
        value (e.g. ``defaultdict``):

        >>> s.map({"cat": "kitten", "dog": "puppy"})
        0   kitten
        1    puppy
        2      NaN
        3      NaN
        dtype: object

        It also accepts a function:

        >>> s.map("I am a {}".format)
        0       I am a cat
        1       I am a dog
        2       I am a nan
        3    I am a rabbit
        dtype: object

        To avoid applying the function to missing values (and keep them as
        ``NaN``) ``na_action='ignore'`` can be used:

        >>> s.map("I am a {}".format, na_action="ignore")
        0     I am a cat
        1     I am a dog
        2            NaN
        3  I am a rabbit
        dtype: object
        """
        # 使用 `_map_values` 方法根据参数 `arg` 和 `na_action` 映射 Series 的值
        new_values = self._map_values(arg, na_action=na_action)
        # 使用 `_constructor` 构造一个新的 Series 对象，保持索引不变
        # 最后调用 `__finalize__` 方法进行最终初始化
        return self._constructor(new_values, index=self.index, copy=False).__finalize__(
            self, method="map"
        )

    def _gotitem(self, key, ndim, subset=None) -> Self:
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
            The key or list of keys to select the data.
        ndim : {1, 2}
            The dimensionality of the result.
        subset : object, default None
            The subset of data to act on.

        Returns
        -------
        Self
            A sliced object as defined by sub-classes.
        """
        # 返回当前对象自身，子类需覆盖此方法实现具体的切片行为
        return self
    _agg_see_also_doc = dedent(
        """
    See Also
    --------
    Series.apply : Invoke function on a Series.
    Series.transform : Transform function producing a Series with like indexes.
    """
    )

    _agg_examples_doc = dedent(
        """
    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4])
    >>> s
    0    1
    1    2
    2    3
    3    4
    dtype: int64

    >>> s.agg('min')
    1

    >>> s.agg(['min', 'max'])
    min   1
    max   4
    dtype: int64
    """
    )

    @doc(
        _shared_docs["aggregate"],  # 使用共享文档中的 "aggregate" 部分作为文档字符串
        klass=_shared_doc_kwargs["klass"],  # 使用共享文档参数中的类别参数
        axis=_shared_doc_kwargs["axis"],  # 使用共享文档参数中的轴参数
        see_also=_agg_see_also_doc,  # 引用上面定义的 "_agg_see_also_doc"，作为相关链接文档
        examples=_agg_examples_doc,  # 引用上面定义的 "_agg_examples_doc"，作为示例文档
    )
    def aggregate(self, func=None, axis: Axis = 0, *args, **kwargs):
        # Validate the axis parameter
        self._get_axis_number(axis)  # 调用对象方法，验证轴参数的有效性

        # if func is None, will switch to user-provided "named aggregation" kwargs
        if func is None:
            func = dict(kwargs.items())  # 如果 func 为 None，则使用用户提供的命名聚合 kwargs

        op = SeriesApply(self, func, args=args, kwargs=kwargs)  # 创建 SeriesApply 对象实例
        result = op.agg()  # 执行聚合操作
        return result  # 返回聚合结果

    agg = aggregate  # 将 aggregate 方法赋值给 agg，作为其别名

    @doc(
        _shared_docs["transform"],  # 使用共享文档中的 "transform" 部分作为文档字符串
        klass=_shared_doc_kwargs["klass"],  # 使用共享文档参数中的类别参数
        axis=_shared_doc_kwargs["axis"],  # 使用共享文档参数中的轴参数
    )
    def transform(
        self, func: AggFuncType, axis: Axis = 0, *args, **kwargs
    ) -> DataFrame | Series:
        # Validate axis argument
        self._get_axis_number(axis)  # 调用对象方法，验证轴参数的有效性
        ser = self.copy(deep=False)  # 浅复制当前 Series 对象
        result = SeriesApply(ser, func=func, args=args, kwargs=kwargs).transform()  # 创建 SeriesApply 对象实例并执行变换操作
        return result  # 返回变换后的结果

    def apply(
        self,
        func: AggFuncType,
        args: tuple[Any, ...] = (),
        *,
        by_row: Literal[False, "compat"] = "compat",
        **kwargs,
    ):
        # apply 方法定义，接受一个函数作为参数，以及其他可选参数

    def _reindex_indexer(
        self,
        new_index: Index | None,
        indexer: npt.NDArray[np.intp] | None,
    ) -> Series:
        # Note: new_index is None iff indexer is None
        # if not None, indexer is np.intp
        if indexer is None and (
            new_index is None or new_index.names == self.index.names
        ):
            return self.copy(deep=False)  # 如果 indexer 和 new_index 都为 None，或者 new_index 的名称与当前索引名称相同，则返回当前 Series 对象的浅复制

        new_values = algorithms.take_nd(
            self._values, indexer, allow_fill=True, fill_value=None
        )  # 使用 take_nd 函数重新索引数据
        return self._constructor(new_values, index=new_index, copy=False)  # 使用新值和索引构造一个新的 Series 对象，并返回

    def _needs_reindex_multi(self, axes, method, level) -> bool:
        """
        Check if we do need a multi reindex; this is for compat with
        higher dims.
        """
        return False  # 返回 False，表示不需要多重索引重建

    @overload
    def rename(
        self,
        index: Renamer | Hashable | None = ...,
        *,
        axis: Axis | None = ...,
        copy: bool | lib.NoDefault = ...,
        inplace: Literal[True],
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> None: ...
    # rename 方法的重载版本，定义了不同的参数组合和返回类型，用于重命名操作
    @Appender(
        """
        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        dtype: int64

        >>> s.set_axis(['a', 'b', 'c'], axis=0)
        a    1
        b    2
        c    3
        dtype: int64
        """
    )
    # 将示例代码追加到文档字符串中，展示如何使用 set_axis 方法
    @Substitution(
        klass=_shared_doc_kwargs["klass"],
        axes_single_arg=_shared_doc_kwargs["axes_single_arg"],
        extended_summary_sub="",
        axis_description_sub="",
        see_also_sub="",
    )
    # 根据 _shared_doc_kwargs 提供的参数，进行文档字符串的替换
    @Appender(NDFrame.set_axis.__doc__)
    # 追加 NDFrame.set_axis 方法的文档字符串
    def set_axis(
        self,
        labels,
        *,
        axis: Axis = 0,
        copy: bool | lib.NoDefault = lib.no_default,
    ) -> Series:
        # 调用父类方法设置轴标签，并返回结果
        return super().set_axis(labels, axis=axis, copy=copy)

    # error: Cannot determine type of 'reindex'
    # 错误：无法确定 'reindex' 方法的类型
    @doc(
        NDFrame.reindex,  # type: ignore[has-type]
        klass=_shared_doc_kwargs["klass"],
        optional_reindex=_shared_doc_kwargs["optional_reindex"],
    )
    # 使用 doc 装饰器将文档附加到 reindex 方法上
    def reindex(  # type: ignore[override]
        self,
        index=None,
        *,
        axis: Axis | None = None,
        method: ReindexMethod | None = None,
        copy: bool | lib.NoDefault = lib.no_default,
        level: Level | None = None,
        fill_value: Scalar | None = None,
        limit: int | None = None,
        tolerance=None,
    ) -> Series:
        # 调用父类方法 reindex 并返回结果
        return super().reindex(
            index=index,
            method=method,
            level=level,
            fill_value=fill_value,
            limit=limit,
            tolerance=tolerance,
            copy=copy,
        )

    @overload  # type: ignore[override]
    # 重载声明，覆盖基类方法
    def rename_axis(
        self,
        mapper: IndexLabel | lib.NoDefault = ...,
        *,
        index=...,
        axis: Axis = ...,
        copy: bool | lib.NoDefault = ...,
        inplace: Literal[True],
    ) -> None: ...
    # 对 rename_axis 方法的注释未完成
    # 定义一个方法 `rename_axis`，用于重命名轴索引。
    def rename_axis(
        # 参数 `mapper` 用于指定新的索引标签映射关系，支持索引标签或无默认值。
        mapper: IndexLabel | lib.NoDefault = lib.no_default,
        # 参数 `index` 指定要操作的索引，默认无默认值。
        *,
        index=lib.no_default,
        # 参数 `axis` 指定操作的轴，默认为0。
        axis: Axis = 0,
        # 参数 `copy` 指定是否复制数据，默认无默认值。
        copy: bool | lib.NoDefault = lib.no_default,
        # 参数 `inplace` 指定是否原地修改，默认为False。
        inplace: bool = False,
    ) -> Self:
        ...
    ) -> Self | None:
        """
        Set the name of the axis for the index.

        Parameters
        ----------
        mapper : scalar, list-like, optional
            Value to set the axis name attribute.

            Use either ``mapper`` and ``axis`` to
            specify the axis to target with ``mapper``, or ``index``.

        index : scalar, list-like, dict-like or function, optional
            A scalar, list-like, dict-like or functions transformations to
            apply to that axis' values.
        
        axis : {0 or 'index'}, default 0
            The axis to rename. For `Series` this parameter is unused and defaults to 0.
        
        copy : bool, default False
            Also copy underlying data.

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
        
        inplace : bool, default False
            Modifies the object directly, instead of creating a new Series
            or DataFrame.
        
        Returns
        -------
        Series, or None
            The same type as the caller or None if ``inplace=True``.

        See Also
        --------
        Series.rename : Alter Series index labels or name.
        DataFrame.rename : Alter DataFrame index labels or name.
        Index.rename : Set new names on index.

        Examples
        --------

        >>> s = pd.Series(["dog", "cat", "monkey"])
        >>> s
        0       dog
        1       cat
        2    monkey
        dtype: object
        >>> s.rename_axis("animal")
        animal
        0    dog
        1    cat
        2    monkey
        dtype: object
        """
        return super().rename_axis(
            mapper=mapper,
            index=index,
            axis=axis,
            inplace=inplace,
            copy=copy,
        )
    # 定义一个方法 drop，用于从 Series 中删除指定的标签或索引
    def drop(
        self,
        labels: IndexLabel | ListLike = ...,  # 要删除的标签或索引，可以是单个值或列表
        *,
        axis: Axis = ...,  # 操作的轴向，默认为...
        index: IndexLabel | ListLike = ...,  # 要删除的行索引，可以是单个值或列表
        columns: IndexLabel | ListLike = ...,  # 要删除的列索引，可以是单个值或列表
        level: Level | None = ...,  # 如果 labels 是多级索引的一部分，则指定级别
        inplace: Literal[False] = ...,  # 是否在原地进行修改，默认不是
        errors: IgnoreRaise = ...,  # 如果标签不存在，如何处理错误情况
    ) -> Series: ...  # 返回一个 Series 对象

    @overload
    def drop(
        self,
        labels: IndexLabel | ListLike = ...,  # 要删除的标签或索引，可以是单个值或列表
        *,
        axis: Axis = ...,  # 操作的轴向，默认为...
        index: IndexLabel | ListLike = ...,  # 要删除的行索引，可以是单个值或列表
        columns: IndexLabel | ListLike = ...,  # 要删除的列索引，可以是单个值或列表
        level: Level | None = ...,  # 如果 labels 是多级索引的一部分，则指定级别
        inplace: bool = ...,  # 是否在原地进行修改
        errors: IgnoreRaise = ...,  # 如果标签不存在，如何处理错误情况
    ) -> Series | None: ...  # 返回一个可能为 None 的 Series 对象

    # 定义方法 drop 的重载版本，支持 inplace 参数为 bool 类型

    def drop(
        self,
        labels: IndexLabel | ListLike = None,  # 要删除的标签或索引，默认为 None
        *,
        axis: Axis = 0,  # 操作的轴向，默认为0（行）
        index: IndexLabel | ListLike = None,  # 要删除的行索引，默认为 None
        columns: IndexLabel | ListLike = None,  # 要删除的列索引，默认为 None
        level: Level | None = None,  # 如果 labels 是多级索引的一部分，则指定级别，默认为 None
        inplace: bool = False,  # 是否在原地进行修改，默认为 False
        errors: IgnoreRaise = "raise",  # 如果标签不存在时如何处理错误，默认抛出异常
    ):  # 方法体开始


        # 定义方法 pop，从 Series 中弹出指定标签对应的元素并返回，如果标签不存在则抛出 KeyError
        def pop(self, item: Hashable) -> Any:  # item 是要弹出的标签，返回任意类型的数据
            """
            Return item and drops from series. Raise KeyError if not found.

            Parameters
            ----------
            item : label
                Index of the element that needs to be removed.

            Returns
            -------
            scalar
                Value that is popped from series.

            Examples
            --------
            >>> ser = pd.Series([1, 2, 3])

            >>> ser.pop(0)
            1

            >>> ser
            1    2
            2    3
            dtype: int64
            """
            return super().pop(item=item)  # 调用父类的 pop 方法执行实际的弹出操作


    @doc(INFO_DOCSTRING, **series_sub_kwargs)
    # 修饰 info 方法，提供关于 Series 对象的详细信息，使用 INFO_DOCSTRING 和其他参数
    def info(
        self,
        verbose: bool | None = None,  # 是否显示详细信息，默认为 None
        buf: IO[str] | None = None,  # 输出信息的缓冲区，默认为 None
        max_cols: int | None = None,  # 允许显示的最大列数，默认为 None
        memory_usage: bool | str | None = None,  # 是否显示内存使用情况，默认为 None
        show_counts: bool = True,  # 是否显示计数，默认为 True
    ) -> None:  # 返回类型为 None
        return SeriesInfo(self, memory_usage).render(
            buf=buf,
            max_cols=max_cols,
            verbose=verbose,
            show_counts=show_counts,
        )  # 调用 SeriesInfo 类的 render 方法，渲染 Series 的信息并输出到指定的 buf
    # 返回 Series 对象的内存使用量

    def memory_usage(self, index: bool = True, deep: bool = False) -> int:
        """
        Return the memory usage of the Series.

        The memory usage can optionally include the contribution of
        the index and of elements of `object` dtype.

        Parameters
        ----------
        index : bool, default True
            Specifies whether to include the memory usage of the Series index.
        deep : bool, default False
            If True, introspect the data deeply by interrogating
            `object` dtypes for system-level memory consumption, and include
            it in the returned value.

        Returns
        -------
        int
            Bytes of memory consumed.

        See Also
        --------
        numpy.ndarray.nbytes : Total bytes consumed by the elements of the
            array.
        DataFrame.memory_usage : Bytes consumed by a DataFrame.

        Examples
        --------
        >>> s = pd.Series(range(3))
        >>> s.memory_usage()
        152

        Not including the index gives the size of the rest of the data, which
        is necessarily smaller:

        >>> s.memory_usage(index=False)
        24

        The memory footprint of `object` values is ignored by default:

        >>> s = pd.Series(["a", "b"])
        >>> s.values
        array(['a', 'b'], dtype=object)
        >>> s.memory_usage()
        144
        >>> s.memory_usage(deep=True)
        244
        """
        
        # 调用内部方法 `_memory_usage` 计算 Series 数据的内存消耗
        v = self._memory_usage(deep=deep)
        
        # 如果 index 参数为 True，则计算并加上 Series 索引的内存使用量
        if index:
            v += self.index.memory_usage(deep=deep)
        
        # 返回计算得到的总内存消耗
        return v
    def isin(self, values) -> Series:
        """
        Whether elements in Series are contained in `values`.

        Return a boolean Series showing whether each element in the Series
        matches an element in the passed sequence of `values` exactly.

        Parameters
        ----------
        values : set or list-like
            The sequence of values to test. Passing in a single string will
            raise a ``TypeError``. Instead, turn a single string into a
            list of one element.

        Returns
        -------
        Series
            Series of booleans indicating if each element is in values.

        Raises
        ------
        TypeError
          * If `values` is a string

        See Also
        --------
        DataFrame.isin : Equivalent method on DataFrame.

        Examples
        --------
        >>> s = pd.Series(
        ...     ["llama", "cow", "llama", "beetle", "llama", "hippo"], name="animal"
        ... )
        >>> s.isin(["cow", "llama"])
        0     True
        1     True
        2     True
        3    False
        4     True
        5    False
        Name: animal, dtype: bool

        To invert the boolean values, use the ``~`` operator:

        >>> ~s.isin(["cow", "llama"])
        0    False
        1    False
        2    False
        3     True
        4    False
        5     True
        Name: animal, dtype: bool

        Passing a single string as ``s.isin('llama')`` will raise an error. Use
        a list of one element instead:

        >>> s.isin(["llama"])
        0     True
        1    False
        2     True
        3    False
        4     True
        5    False
        Name: animal, dtype: bool

        Strings and integers are distinct and are therefore not comparable:

        >>> pd.Series([1]).isin(["1"])
        0    False
        dtype: bool
        >>> pd.Series([1.1]).isin(["1.1"])
        0    False
        dtype: bool
        """
        # 调用底层算法实现元素是否在指定的值序列中的逻辑
        result = algorithms.isin(self._values, values)
        # 根据结果构造新的 Series 对象，并保留原有的索引和元数据
        return self._constructor(result, index=self.index, copy=False).__finalize__(
            self, method="isin"
        )
    ) -> Series:
        """
        Return boolean Series equivalent to left <= series <= right.

        This function returns a boolean vector containing `True` wherever the
        corresponding Series element is between the boundary values `left` and
        `right`. NA values are treated as `False`.

        Parameters
        ----------
        left : scalar or list-like
            Left boundary.
        right : scalar or list-like
            Right boundary.
        inclusive : {"both", "neither", "left", "right"}
            Include boundaries. Whether to set each bound as closed or open.

            .. versionchanged:: 1.3.0

        Returns
        -------
        Series
            Series representing whether each element is between left and
            right (inclusive).

        See Also
        --------
        Series.gt : Greater than of series and other.
        Series.lt : Less than of series and other.

        Notes
        -----
        This function is equivalent to ``(left <= ser) & (ser <= right)``

        Examples
        --------
        >>> s = pd.Series([2, 0, 4, 8, np.nan])

        Boundary values are included by default:

        >>> s.between(1, 4)
        0     True
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        With `inclusive` set to ``"neither"`` boundary values are excluded:

        >>> s.between(1, 4, inclusive="neither")
        0     True
        1    False
        2    False
        3    False
        4    False
        dtype: bool

        `left` and `right` can be any scalar value:

        >>> s = pd.Series(["Alice", "Bob", "Carol", "Eve"])
        >>> s.between("Anna", "Daniel")
        0    False
        1     True
        2     True
        3    False
        dtype: bool
        """
        if inclusive == "both":
            # Create mask for elements greater than or equal to left boundary
            lmask = self >= left
            # Create mask for elements less than or equal to right boundary
            rmask = self <= right
        elif inclusive == "left":
            # Create mask for elements greater than or equal to left boundary
            lmask = self >= left
            # Create mask for elements less than right boundary
            rmask = self < right
        elif inclusive == "right":
            # Create mask for elements greater than left boundary
            lmask = self > left
            # Create mask for elements less than or equal to right boundary
            rmask = self <= right
        elif inclusive == "neither":
            # Create mask for elements greater than left boundary
            lmask = self > left
            # Create mask for elements less than right boundary
            rmask = self < right
        else:
            # Raise an error if inclusive parameter is not one of the specified options
            raise ValueError(
                "Inclusive has to be either string of 'both',"
                "'left', 'right', or 'neither'."
            )

        # Return boolean Series indicating elements that satisfy both left and right masks
        return lmask & rmask

    def case_when(
        self,
        caselist: list[
            tuple[
                ArrayLike | Callable[[Series], Series | np.ndarray | Sequence[bool]],
                ArrayLike | Scalar | Callable[[Series], Series | np.ndarray],
            ],
        ],
    ):
        """
        Apply conditional logic to Series elements based on a list of conditions and values.

        Parameters
        ----------
        caselist : list of tuples
            List where each tuple consists of a condition and a corresponding value or function.

        Returns
        -------
        Series
            Series with values computed based on the first satisfied condition in `caselist`.

        Notes
        -----
        This method is a vectorized implementation of conditional logic similar to SQL's
        CASE WHEN statement.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.case_when([
        ...     (s < 2, "small"),
        ...     (s >= 3, "large"),
        ...     (s == 2, "medium")
        ... ])
        0     small
        1    medium
        2     large
        3     large
        dtype: object
        """
        # Implementation of the case_when method is typically defined elsewhere

    @doc(NDFrame.isna, klass=_shared_doc_kwargs["klass"])  # type: ignore[has-type]
    def isna(self) -> Series:
        """
        Detect missing values for Series.

        Returns
        -------
        Series
            Series containing boolean values indicating whether each element is NaN.

        See Also
        --------
        Series.notna : Negation of isna.

        Notes
        -----
        This method is equivalent to `pd.isna`.

        Examples
        --------
        >>> s = pd.Series([1, np.nan, None, "string"])
        >>> s.isna()
        0    False
        1     True
        2     True
        3    False
        dtype: bool
        """
        return NDFrame.isna(self)
    # 使用 @doc 装饰器将 NDFrame.isna 方法的文档注入到当前方法中，klass 参数来自 _shared_doc_kwargs["klass"]
    @doc(NDFrame.isna, klass=_shared_doc_kwargs["klass"])  # type: ignore[has-type]
    def isnull(self) -> Series:
        """
        Series.isnull is an alias for Series.isna.
        """
        # 调用父类的 isnull 方法
        return super().isnull()

    # error: Cannot determine type of 'notna'
    # 使用 @doc 装饰器将 NDFrame.notna 方法的文档注入到当前方法中，klass 参数来自 _shared_doc_kwargs["klass"]
    @doc(NDFrame.notna, klass=_shared_doc_kwargs["klass"])  # type: ignore[has-type]
    def notna(self) -> Series:
        # 调用父类的 notna 方法
        return super().notna()

    # error: Cannot determine type of 'notna'
    # 使用 @doc 装饰器将 NDFrame.notna 方法的文档注入到当前方法中，klass 参数来自 _shared_doc_kwargs["klass"]
    def notnull(self) -> Series:
        """
        Series.notnull is an alias for Series.notna.
        """
        # 调用父类的 notnull 方法
        return super().notnull()

    # 重载方法：当 inplace 参数为 False 时返回 Series 对象
    @overload
    def dropna(
        self,
        *,
        axis: Axis = ...,
        inplace: Literal[False] = ...,
        how: AnyAll | None = ...,
        ignore_index: bool = ...,
    ) -> Series: ...

    # 重载方法：当 inplace 参数为 True 时返回 None
    @overload
    def dropna(
        self,
        *,
        axis: Axis = ...,
        inplace: Literal[True],
        how: AnyAll | None = ...,
        ignore_index: bool = ...,
    ) -> None: ...

    # 实现方法：根据参数删除缺失值，根据 inplace 参数决定是否原地修改
    def dropna(
        self,
        *,
        axis: Axis = 0,
        inplace: bool = False,
        how: AnyAll | None = None,
        ignore_index: bool = False,
    ) -> Series | None:
        """
        Return a new Series with missing values removed.

        See the :ref:`User Guide <missing_data>` for more on which values are
        considered missing, and how to work with missing data.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        inplace : bool, default False
            If True, do operation inplace and return None.
        how : str, optional
            Not in use. Kept for compatibility.
        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

            .. versionadded:: 2.0.0

        Returns
        -------
        Series or None
            Series with NA entries dropped from it or None if ``inplace=True``.

        See Also
        --------
        Series.isna: Indicate missing values.
        Series.notna : Indicate existing (non-missing) values.
        Series.fillna : Replace missing values.
        DataFrame.dropna : Drop rows or columns which contain NA values.
        Index.dropna : Drop missing indices.

        Examples
        --------
        >>> ser = pd.Series([1.0, 2.0, np.nan])
        >>> ser
        0    1.0
        1    2.0
        2    NaN
        dtype: float64

        Drop NA values from a Series.

        >>> ser.dropna()
        0    1.0
        1    2.0
        dtype: float64

        Empty strings are not considered NA values. ``None`` is considered an
        NA value.

        >>> ser = pd.Series([np.nan, 2, pd.NaT, "", None, "I stay"])
        >>> ser
        0       NaN
        1         2
        2       NaT
        3
        4      None
        5    I stay
        dtype: object
        >>> ser.dropna()
        1         2
        3
        5    I stay
        dtype: object
        """
        # Validate and convert inplace and ignore_index parameters to boolean
        inplace = validate_bool_kwarg(inplace, "inplace")
        ignore_index = validate_bool_kwarg(ignore_index, "ignore_index")
        
        # Validate the axis parameter, determining the axis number
        self._get_axis_number(axis or 0)

        # Check if the Series can hold NA values
        if self._can_hold_na:
            # Call function to remove NA values from the Series
            result = remove_na_arraylike(self)
        else:
            # If inplace is False, create a shallow copy of the Series
            if not inplace:
                result = self.copy(deep=False)
            else:
                result = self

        # If ignore_index is True, reset the index of the result
        if ignore_index:
            result.index = default_index(len(result))

        # If inplace is True, update self in place and return None
        if inplace:
            return self._update_inplace(result)
        else:
            # Otherwise, return the modified Series
            return result

    # ----------------------------------------------------------------------
    # Time series-oriented methods

    def to_timestamp(
        self,
        freq: Frequency | None = None,
        how: Literal["s", "e", "start", "end"] = "start",
        copy: bool | lib.NoDefault = lib.no_default,
    ) -> Series:
        """
        Cast to DatetimeIndex of Timestamps, at *beginning* of period.

        This can be changed to the *end* of the period, by specifying `how="e"`.

        Parameters
        ----------
        freq : str, default frequency of PeriodIndex
            Desired frequency.
        how : {'s', 'e', 'start', 'end'}
            Convention for converting period to timestamp; start of period
            vs. end.
        copy : bool, default False
            Whether or not to return a copy.

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
        Series with DatetimeIndex
            Series with the PeriodIndex cast to DatetimeIndex.

        See Also
        --------
        Series.to_period: Inverse method to cast DatetimeIndex to PeriodIndex.
        DataFrame.to_timestamp: Equivalent method for DataFrame.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")
        >>> s1 = pd.Series([1, 2, 3], index=idx)
        >>> s1
        2023    1
        2024    2
        2025    3
        Freq: Y-DEC, dtype: int64

        The resulting frequency of the Timestamps is `YearBegin`

        >>> s1 = s1.to_timestamp()
        >>> s1
        2023-01-01    1
        2024-01-01    2
        2025-01-01    3
        Freq: YS-JAN, dtype: int64

        Using `freq` which is the offset that the Timestamps will have

        >>> s2 = pd.Series([1, 2, 3], index=idx)
        >>> s2 = s2.to_timestamp(freq="M")
        >>> s2
        2023-01-31    1
        2024-01-31    2
        2025-01-31    3
        Freq: YE-JAN, dtype: int64
        """
        # Check for deprecation of the `copy` keyword
        self._check_copy_deprecation(copy)
        
        # Ensure that the index of the Series is of type PeriodIndex; raise TypeError if not
        if not isinstance(self.index, PeriodIndex):
            raise TypeError(f"unsupported Type {type(self.index).__name__}")

        # Create a shallow copy of the Series object
        new_obj = self.copy(deep=False)
        
        # Convert the PeriodIndex of the Series to DatetimeIndex using specified frequency and how
        new_index = self.index.to_timestamp(freq=freq, how=how)
        
        # Set the modified index back to the copied Series object
        setattr(new_obj, "index", new_index)
        
        # Return the modified Series object with DatetimeIndex
        return new_obj
    # ----------------------------------------------------------------------
    # Add index

    # 定义索引顺序列表，指示对象索引在轴上的顺序，这里默认为 ["index"]
    _AXIS_ORDERS: list[Literal["index", "columns"]] = ["index"]

    # 计算索引顺序列表的长度并赋值给 _AXIS_LEN
    _AXIS_LEN = len(_AXIS_ORDERS)

    # 指定信息轴的编号，这里固定为0，表示主索引
    _info_axis_number: Literal[0] = 0

    # 指定信息轴的名称，这里固定为 "index"
    _info_axis_name: Literal["index"] = "index"
    # 创建一个名为index的属性，类型为properties.AxisProperty，表示Series的索引（轴标签）
    index = properties.AxisProperty(
        axis=0,
        doc="""
        The index (axis labels) of the Series.

        The index of a Series is used to label and identify each element of the
        underlying data. The index can be thought of as an immutable ordered set
        (technically a multi-set, as it may contain duplicate labels), and is
        used to index and align data in pandas.

        Returns
        -------
        Index
            The index labels of the Series.

        See Also
        --------
        Series.reindex : Conform Series to new index.
        Index : The base pandas index type.

        Notes
        -----
        For more information on pandas indexing, see the `indexing user guide
        <https://pandas.pydata.org/docs/user_guide/indexing.html>`__.

        Examples
        --------
        To create a Series with a custom index and view the index labels:

        >>> cities = ['Kolkata', 'Chicago', 'Toronto', 'Lisbon']
        >>> populations = [14.85, 2.71, 2.93, 0.51]
        >>> city_series = pd.Series(populations, index=cities)
        >>> city_series.index
        Index(['Kolkata', 'Chicago', 'Toronto', 'Lisbon'], dtype='object')

        To change the index labels of an existing Series:

        >>> city_series.index = ['KOL', 'CHI', 'TOR', 'LIS']
        >>> city_series.index
        Index(['KOL', 'CHI', 'TOR', 'LIS'], dtype='object')
        """
    )

    # ----------------------------------------------------------------------
    # Accessor Methods
    # ----------------------------------------------------------------------

    # 创建多个属性访问器，用于访问Series对象的不同功能
    str = Accessor("str", StringMethods)
    dt = Accessor("dt", CombinedDatetimelikeProperties)
    cat = Accessor("cat", CategoricalAccessor)
    plot = Accessor("plot", pandas.plotting.PlotAccessor)
    sparse = Accessor("sparse", SparseAccessor)
    struct = Accessor("struct", StructAccessor)
    list = Accessor("list", ListAccessor)

    # ----------------------------------------------------------------------
    # Add plotting methods to Series

    # 将pandas.plotting.hist_series函数赋值给hist属性，用于Series对象的直方图绘制
    hist = pandas.plotting.hist_series

    # ----------------------------------------------------------------------
    # Template-Based Arithmetic/Comparison Methods

    # 创建一个私有方法_cmp_method，用于执行模板化的比较方法
    def _cmp_method(self, other, op):
        # 获取操作结果的名称
        res_name = ops.get_op_result_name(self, other)

        # 如果other是Series对象且索引不同，抛出异常
        if isinstance(other, Series) and not self._indexed_same(other):
            raise ValueError("Can only compare identically-labeled Series objects")

        # 获取self的值和other的值
        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)

        # 执行比较操作，得到结果值
        res_values = ops.comparison_op(lvalues, rvalues, op)

        # 构造并返回结果Series对象
        return self._construct_result(res_values, name=res_name)
    def _logical_method(self, other, op):
        # 获取操作结果的名称，用于结果命名
        res_name = ops.get_op_result_name(self, other)
        # 将 self 和 other 对齐以便进行操作，确保以对象的形式进行对齐
        self, other = self._align_for_op(other, align_asobject=True)

        # 获取 self 的值
        lvalues = self._values
        # 从 other 中提取数组，使用 numpy 和范围提取方式
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)

        # 使用逻辑操作符进行逻辑运算
        res_values = ops.logical_op(lvalues, rvalues, op)
        # 构造结果对象，并指定结果名称
        return self._construct_result(res_values, name=res_name)

    def _arith_method(self, other, op):
        # 将 self 和 other 对齐以便进行操作
        self, other = self._align_for_op(other)
        # 调用基类的算术方法进行运算
        return base.IndexOpsMixin._arith_method(self, other, op)

    def _align_for_op(self, right, align_asobject: bool = False):
        """align lhs and rhs Series"""
        # TODO: Different from DataFrame._align_for_op, list, tuple and ndarray
        # are not coerced here
        # because Series has inconsistencies described in GH#13637
        left = self

        # 如果 right 是 Series 类型
        if isinstance(right, Series):
            # 避免重复对齐
            if not left.index.equals(right.index):
                # 如果需要作为对象对齐
                if align_asobject:
                    # 如果 left 和 right 的数据类型不是 object 或者 np.bool_
                    if left.dtype not in (object, np.bool_) or right.dtype not in (
                        object,
                        np.bool_,
                    ):
                        pass
                        # GH#52538 no longer cast in these cases
                    else:
                        # 为了保留布尔运算的原始值数据类型
                        left = left.astype(object)
                        right = right.astype(object)

                # 对齐 left 和 right
                left, right = left.align(right)

        return left, right

    def _binop(self, other: Series, func, level=None, fill_value=None) -> Series:
        """
        Perform generic binary operation with optional fill value.

        Parameters
        ----------
        other : Series
        func : binary operator
        fill_value : float or object
            Value to substitute for NA/null values. If both Series are NA in a
            location, the result will be NA regardless of the passed fill value.
        level : int or level name, default None
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.

        Returns
        -------
        Series
        """
        this = self

        # 如果 self 的索引和 other 的索引不相等
        if not self.index.equals(other.index):
            # 对 self 和 other 进行对齐，使用外连接方式对齐指定级别的索引值
            this, other = self.align(other, level=level, join="outer")

        # 使用指定的填充值进行二进制操作
        this_vals, other_vals = ops.fill_binop(this._values, other._values, fill_value)

        # 忽略所有的 numpy 错误
        with np.errstate(all="ignore"):
            # 使用指定的二元操作函数进行计算
            result = func(this_vals, other_vals)

        # 获取操作结果的名称
        name = ops.get_op_result_name(self, other)
        # 构造结果对象，并指定结果名称
        out = this._construct_result(result, name)
        return cast(Series, out)

    def _construct_result(
        self, result: ArrayLike | tuple[ArrayLike, ArrayLike], name: Hashable
    ):
        """
        Construct result Series or tuple of Series.

        Parameters
        ----------
        result : ArrayLike or tuple[ArrayLike, ArrayLike]
            Resulting array(s) from the operation.
        name : Hashable
            Name to be assigned to the resulting Series.

        Returns
        -------
        Series or tuple of Series
            Constructed Series or tuple of Series.
        """
        # 构造结果 Series 或者 Series 元组
        return cast(Series, self._constructor(result, index=self.index, name=name))
    ) -> Series | tuple[Series, Series]:
        """
        Construct an appropriately-labelled Series from the result of an op.

        Parameters
        ----------
        result : ndarray or ExtensionArray
            The result of an operation, which can be an ndarray or ExtensionArray.
        name : Label
            The label to assign to the resulting Series.

        Returns
        -------
        Series
            In the case of __divmod__ or __rdivmod__, a 2-tuple of Series.
        """
        if isinstance(result, tuple):
            # produced by divmod or rdivmod

            # Construct Series from the first element of the tuple
            res1 = self._construct_result(result[0], name=name)
            # Construct Series from the second element of the tuple
            res2 = self._construct_result(result[1], name=name)

            # GH#33427 assertions to keep mypy happy
            assert isinstance(res1, Series)
            assert isinstance(res2, Series)
            return (res1, res2)

        # TODO: result should always be ArrayLike, but this fails for some
        #  JSONArray tests
        # Determine dtype of the result
        dtype = getattr(result, "dtype", None)
        # Create a Series or derived class instance using the result data
        out = self._constructor(result, index=self.index, dtype=dtype, copy=False)
        # Finalize construction of the Series instance
        out = out.__finalize__(self)

        # Set the result's name after __finalize__ is called because __finalize__
        #  would set it back to self.name
        out.name = name
        return out

    def _flex_method(self, other, op, *, level=None, fill_value=None, axis: Axis = 0):
        if axis is not None:
            # Check and retrieve the axis number
            self._get_axis_number(axis)

        # Determine the resulting Series name for the operation
        res_name = ops.get_op_result_name(self, other)

        if isinstance(other, Series):
            # Perform binary operation with another Series
            return self._binop(other, op, level=level, fill_value=fill_value)
        elif isinstance(other, (np.ndarray, list, tuple)):
            if len(other) != len(self):
                # Ensure lengths are equal for ndarray, list, or tuple
                raise ValueError("Lengths must be equal")
            # Convert other to a Series-like object with the same index
            other = self._constructor(other, self.index, copy=False)
            # Perform binary operation with the converted Series-like object
            result = self._binop(other, op, level=level, fill_value=fill_value)
            result._name = res_name
            return result
        else:
            if fill_value is not None:
                # Handle fill_value scenarios
                if isna(other):
                    return op(self, fill_value)
                # Fill NaN values in self with fill_value
                self = self.fillna(fill_value)

            # Perform binary operation with a scalar or non-Series object
            return op(self, other)

    def eq(
        self,
        other,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: Axis = 0,
        ):
        """
        Compare this Series with another for equality.

        Parameters
        ----------
        other : Series or scalar
            The object to compare with.
        level : int, default None
            If the target is a MultiIndex, level to use for comparison.
        fill_value : float, default None
            Value to use for missing values in this Series and other before comparing.
        axis : Axis, default 0
            The axis to use for comparison.

        """
    @Appender(ops.make_flex_doc("ne", "series"))
    # 使用装饰器 `Appender` 将生成 "ne"（不等于）操作的文档追加到现有文档中
    def ne(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        # 调用内部方法 `_flex_method`，执行二元运算符 `operator.ne`，实现不等于操作
        return self._flex_method(
            other, operator.ne, level=level, fill_value=fill_value, axis=axis
        )
    def le(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        """
        Return Less than or equal to of series and other, \
        element-wise (binary operator `le`).

        Equivalent to ``series <= other``, but with support to substitute a
        fill_value for missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
            The second operand in this operation.
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.ge : Return elementwise Greater than or equal to of series and other.
        Series.lt : Return elementwise Less than of series and other.
        Series.gt : Return elementwise Greater than of series and other.
        Series.eq : Return elementwise equal to of series and other.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan, 1], index=['a', 'b', 'c', 'd', 'e'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        e    1.0
        dtype: float64
        >>> b = pd.Series([0, 1, 2, np.nan, 1], index=['a', 'b', 'c', 'd', 'f'])
        >>> b
        a    0.0
        b    1.0
        c    2.0
        d    NaN
        f    1.0
        dtype: float64
        >>> a.le(b, fill_value=0)
        a    False
        b     True
        c     True
        d    False
        e    False
        f     True
        dtype: bool
        """
        # 调用灵活方法处理操作，使用 operator.le 执行小于等于比较
        return self._flex_method(
            other, operator.le, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("lt", "series"))
    def lt(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        """
        Return Less than of series and other, element-wise (binary operator `lt`).

        Equivalent to ``series < other``, but with support to substitute a
        fill_value for missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
            The second operand in this operation.
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.ge : Return elementwise Greater than or equal to of series and other.
        Series.le : Return elementwise Less than or equal to of series and other.
        Series.gt : Return elementwise Greater than of series and other.
        Series.eq : Return elementwise equal to of series and other.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan, 1], index=['a', 'b', 'c', 'd', 'e'])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        e    1.0
        dtype: float64
        >>> b = pd.Series([0, 1, 2, np.nan, 1], index=['a', 'b', 'c', 'd', 'f'])
        >>> b
        a    0.0
        b    1.0
        c    2.0
        d    NaN
        f    1.0
        dtype: float64
        >>> a.lt(b, fill_value=0)
        a    False
        b    False
        c     True
        d    False
        e    False
        f     True
        dtype: bool
        """
        # 调用灵活方法处理操作，使用 operator.lt 执行小于比较
        return self._flex_method(
            other, operator.lt, level=level, fill_value=fill_value, axis=axis
        )
    def ge(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        """
        Return Greater than or equal to of series and other, \
        element-wise (binary operator `ge`).

        Equivalent to ``series >= other``, but with support to substitute a
        fill_value for missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
            The second operand in this operation.
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.gt : Greater than comparison, element-wise.
        Series.le : Less than or equal to comparison, element-wise.
        Series.lt : Less than comparison, element-wise.
        Series.eq : Equal to comparison, element-wise.
        Series.ne : Not equal to comparison, element-wise.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan, 1], index=["a", "b", "c", "d", "e"])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        e    1.0
        dtype: float64
        >>> b = pd.Series([0, 1, 2, np.nan, 1], index=["a", "b", "c", "d", "f"])
        >>> b
        a    0.0
        b    1.0
        c    2.0
        d    NaN
        f    1.0
        dtype: float64
        >>> a.ge(b, fill_value=0)
        a     True
        b     True
        c    False
        d    False
        e     True
        f    False
        dtype: bool
        """
        # 调用 _flex_method 方法，执行大于等于（ge）操作
        return self._flex_method(
            other, operator.ge, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("gt", "series"))
    def gt(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        """
        Return Greater than of series and other, element-wise (binary operator `gt`).

        Equivalent to ``series > other``, but with support to substitute a
        fill_value for missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
            The second operand in this operation.
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.ge : Greater than or equal to comparison, element-wise.
        Series.le : Less than or equal to comparison, element-wise.
        Series.lt : Less than comparison, element-wise.
        Series.eq : Equal to comparison, element-wise.
        Series.ne : Not equal to comparison, element-wise.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan, 1], index=["a", "b", "c", "d", "e"])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        e    1.0
        dtype: float64
        >>> b = pd.Series([0, 1, 2, np.nan, 1], index=["a", "b", "c", "d", "f"])
        >>> b
        a    0.0
        b    1.0
        c    2.0
        d    NaN
        f    1.0
        dtype: float64
        >>> a.gt(b, fill_value=0)
        a     True
        b    False
        c    False
        d    False
        e     True
        f     True
        dtype: bool
        """
        # 调用 _flex_method 方法，执行大于（gt）操作
        return self._flex_method(
            other, operator.gt, level=level, fill_value=fill_value, axis=axis
        )
    def add(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        """
        Return Addition of series and other, element-wise (binary operator `add`).

        Equivalent to ``series + other``, but with support to substitute a fill_value
        for missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
            With which to compute the addition.
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.radd : Reverse of the Addition operator, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=["a", "b", "c", "d"])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=["a", "b", "d", "e"])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.add(b, fill_value=0)
        a    2.0
        b    1.0
        c    1.0
        d    1.0
        e    NaN
        dtype: float64
        """
        # 使用灵活方法 `_flex_method` 进行操作，此处调用加法操作 `operator.add`
        return self._flex_method(
            other, operator.add, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("radd", "series"))
    def radd(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        # 使用灵活方法 `_flex_method` 进行反向加法操作 `roperator.radd`
        return self._flex_method(
            other, roperator.radd, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("sub", "series"))
    def sub(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        # 使用灵活方法 `_flex_method` 进行减法操作 `operator.sub`
        return self._flex_method(
            other, operator.sub, level=level, fill_value=fill_value, axis=axis
        )

    subtract = sub

    @Appender(ops.make_flex_doc("rsub", "series"))
    def rsub(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        # 使用灵活方法 `_flex_method` 进行反向减法操作 `roperator.rsub`
        return self._flex_method(
            other, roperator.rsub, level=level, fill_value=fill_value, axis=axis
        )

    def mul(
        self,
        other,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: Axis = 0,
    @Appender(ops.make_flex_doc("rmul", "series"))
    def rmul(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        """
        Append documentation for 'rmul' operation to the existing documentation.

        Parameters
        ----------
        other : Series or scalar value
            The series or scalar value to multiply with the current series.
        level : int or name, optional
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}, optional
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.mul : Multiplication operation, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=["a", "b", "c", "d"])
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=["a", "b", "d", "e"])
        >>> a.rmul(b, fill_value=0)
        a    1.0
        b    0.0
        c    0.0
        d    0.0
        e    NaN
        dtype: float64
        """
        return self._flex_method(
            other, roperator.rmul, level=level, fill_value=fill_value, axis=axis
        )
    def truediv(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        """
        Return Floating division of series and other, \
        element-wise (binary operator `truediv`).

        Equivalent to ``series / other``, but with support to substitute a
        fill_value for missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
            Series with which to compute division.
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.

        See Also
        --------
        Series.rtruediv : Reverse of the Floating division operator, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.

        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=["a", "b", "c", "d"])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=["a", "b", "d", "e"])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.divide(b, fill_value=0)
        a    1.0
        b    inf
        c    inf
        d    0.0
        e    NaN
        dtype: float64
        """
        # 使用 _flex_method 方法调用 operator.truediv 进行灵活的方法调用
        return self._flex_method(
            other, operator.truediv, level=level, fill_value=fill_value, axis=axis
        )

    # 将 truediv 方法别名为 div 和 divide
    div = truediv
    divide = truediv

    @Appender(ops.make_flex_doc("rtruediv", "series"))
    def rtruediv(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        """
        Reverse of the Floating division operator for Series.

        Parameters
        ----------
        other : Series or scalar value
            Series with which to compute division.
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.
        """
        # 使用 _flex_method 方法调用 roperator.rtruediv 进行灵活的方法调用
        return self._flex_method(
            other, roperator.rtruediv, level=level, fill_value=fill_value, axis=axis
        )

    # 将 rtruediv 方法别名为 rdiv
    rdiv = rtruediv

    @Appender(ops.make_flex_doc("floordiv", "series"))
    def floordiv(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        """
        Return Integer division of series and other, \
        element-wise (binary operator `floordiv`).

        Equivalent to ``series // other``, but with support to substitute a
        fill_value for missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
            Series with which to compute division.
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.
        """
        # 使用 _flex_method 方法调用 operator.floordiv 进行灵活的方法调用
        return self._flex_method(
            other, operator.floordiv, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("rfloordiv", "series"))
    def rfloordiv(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        """
        Return Reverse Integer division of series and other, \
        element-wise (binary operator `rfloordiv`).

        Equivalent to ``other // series``, but with support to substitute a
        fill_value for missing data in either one of the inputs.

        Parameters
        ----------
        other : Series or scalar value
            Series with which to compute division.
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            The result of the operation.
        """
        # 使用 _flex_method 方法调用 roperator.rfloordiv 进行灵活的方法调用
        return self._flex_method(
            other, roperator.rfloordiv, level=level, fill_value=fill_value, axis=axis
        )
    # 定义一个方法用于计算系列与另一个值的模数，支持填充缺失数据，并指定操作的轴向
    def mod(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        """
        Return Modulo of series and other, element-wise (binary operator `mod`).
    
        Equivalent to ``series % other``, but with support to substitute a
        fill_value for missing data in either one of the inputs.
    
        Parameters
        ----------
        other : Series or scalar value
            Series with which to compute modulo.
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
    
        Returns
        -------
        Series
            The result of the operation.
    
        See Also
        --------
        Series.rmod : Reverse of the Modulo operator, see
            `Python documentation
            <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
            for more details.
    
        Examples
        --------
        >>> a = pd.Series([1, 1, 1, np.nan], index=["a", "b", "c", "d"])
        >>> a
        a    1.0
        b    1.0
        c    1.0
        d    NaN
        dtype: float64
        >>> b = pd.Series([1, np.nan, 1, np.nan], index=["a", "b", "d", "e"])
        >>> b
        a    1.0
        b    NaN
        d    1.0
        e    NaN
        dtype: float64
        >>> a.mod(b, fill_value=0)
        a    0.0
        b    NaN
        c    NaN
        d    0.0
        e    NaN
        dtype: float64
        """
        # 调用内部方法执行灵活操作，传入操作符和其他参数
        return self._flex_method(
            other, operator.mod, level=level, fill_value=fill_value, axis=axis
        )
    
    @Appender(ops.make_flex_doc("rmod", "series"))
    def rmod(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        """
        Return Reverse Modulo of series and other, element-wise.
    
        Equivalent to ``other % series``, but with support to substitute a
        fill_value for missing data in either one of the inputs.
    
        Parameters
        ----------
        other : Series or scalar value
            Series with which to compute modulo.
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
    
        Returns
        -------
        Series
            The result of the operation.
        """
        # 调用内部方法执行灵活操作，传入反向操作符和其他参数
        return self._flex_method(
            other, roperator.rmod, level=level, fill_value=fill_value, axis=axis
        )
    
    @Appender(ops.make_flex_doc("pow", "series"))
    def pow(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        """
        Return Exponential power of series and other, element-wise.
    
        Equivalent to ``series ** other``, but with support to substitute a
        fill_value for missing data in either one of the inputs.
    
        Parameters
        ----------
        other : Series or scalar value
            Series with which to compute power.
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
    
        Returns
        -------
        Series
            The result of the operation.
        """
        # 调用内部方法执行灵活操作，传入操作符和其他参数
        return self._flex_method(
            other, operator.pow, level=level, fill_value=fill_value, axis=axis
        )
    
    @Appender(ops.make_flex_doc("rpow", "series"))
    def rpow(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        """
        Return Reverse Exponential power of series and other, element-wise.
    
        Equivalent to ``other ** series``, but with support to substitute a
        fill_value for missing data in either one of the inputs.
    
        Parameters
        ----------
        other : Series or scalar value
            Series with which to compute power.
        level : int or name
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : None or float value, default None (NaN)
            Fill existing missing (NaN) values, and any new element needed for
            successful Series alignment, with this value before computation.
            If data in both corresponding Series locations is missing
            the result of filling (at that location) will be missing.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
    
        Returns
        -------
        Series
            The result of the operation.
        """
        # 调用内部方法执行灵活操作，传入反向操作符和其他参数
        return self._flex_method(
            other, roperator.rpow, level=level, fill_value=fill_value, axis=axis
        )
    
    @Appender(ops.make_flex_doc("divmod", "series"))
    def divmod(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        """
        Perform element-wise divmod between this Series and another object.

        Parameters:
        - other: Another object to perform divmod with.
        - level: Level of MultiIndex if Series contains MultiIndex.
        - fill_value: Value to use for missing data.
        - axis: Axis along which to perform divmod.

        Returns:
        Series: Result of divmod operation.
        """
        return self._flex_method(
            other, divmod, level=level, fill_value=fill_value, axis=axis
        )

    @Appender(ops.make_flex_doc("rdivmod", "series"))
    def rdivmod(self, other, level=None, fill_value=None, axis: Axis = 0) -> Series:
        """
        Right-side variant of divmod operation for Series.

        Parameters:
        - other: Another object to perform divmod with.
        - level: Level of MultiIndex if Series contains MultiIndex.
        - fill_value: Value to use for missing data.
        - axis: Axis along which to perform divmod.

        Returns:
        Series: Result of rdivmod operation.
        """
        return self._flex_method(
            other, roperator.rdivmod, level=level, fill_value=fill_value, axis=axis
        )

    # ----------------------------------------------------------------------
    # Reductions

    def _reduce(
        self,
        op,
        name: str,  # type: ignore[valid-type]
        *,
        axis: Axis = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        filter_type=None,
        **kwds,
    ):
        """
        Perform a reduction operation.

        If we have an ndarray as a value, then simply perform the operation,
        otherwise delegate to the object.

        Parameters:
        - op: Reduction operation function.
        - name: Name of the reduction operation.
        - axis: Axis along which to perform the reduction.
        - skipna: Whether to skip NaN values.
        - numeric_only: Whether to restrict to numeric types only.
        - filter_type: Type of data to filter during reduction.
        - kwds: Additional keyword arguments.

        Returns:
        Depends on the reduction operation.
        """
        delegate = self._values

        if axis is not None:
            self._get_axis_number(axis)

        if isinstance(delegate, ExtensionArray):
            # dispatch to ExtensionArray interface
            return delegate._reduce(name, skipna=skipna, **kwds)

        else:
            # dispatch to numpy arrays
            if numeric_only and self.dtype.kind not in "iufcb":
                # i.e. not is_numeric_dtype(self.dtype)
                kwd_name = "numeric_only"
                if name in ["any", "all"]:
                    kwd_name = "bool_only"
                # GH#47500 - change to TypeError to match other methods
                raise TypeError(
                    f"Series.{name} does not allow {kwd_name}={numeric_only} "
                    "with non-numeric dtypes."
                )
            return op(delegate, skipna=skipna, **kwds)

    @Appender(make_doc("any", ndim=1))
    def any(
        self,
        *,
        axis: Axis = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs,
    ) -> bool:
        """
        Return whether any element is True over requested axis.

        Parameters:
        - axis: Axis along which to check for truthiness.
        - bool_only: Whether to consider only boolean data.
        - skipna: Whether to skip NaN values.
        - kwargs: Additional keyword arguments.

        Returns:
        bool: True if any element is True, False otherwise.
        """
        nv.validate_logical_func((), kwargs, fname="any")
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._reduce(
            nanops.nanany,
            name="any",
            axis=axis,
            numeric_only=bool_only,
            skipna=skipna,
            filter_type="bool",
        )

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="all")
    @Appender(make_doc("all", ndim=1))
    def all(
        self,
        axis: Axis = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs,
    ) -> bool:
        """
        Return whether all elements are True over requested axis.

        Parameters:
        - axis: Axis along which to check for truthiness.
        - bool_only: Whether to consider only boolean data.
        - skipna: Whether to skip NaN values.
        - kwargs: Additional keyword arguments.

        Returns:
        bool: True if all elements are True, False otherwise.
        """
    ) -> bool:
        # 使用 nv.validate_logical_func() 验证传入的参数，确保它们符合逻辑函数的要求
        nv.validate_logical_func((), kwargs, fname="all")
        # 验证 skipna 参数是否为布尔类型，确保不允许传入 None
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        # 调用 _reduce 方法，计算沿指定轴的逻辑“全部”操作结果
        return self._reduce(
            nanops.nanall,
            name="all",
            axis=axis,
            numeric_only=bool_only,
            skipna=skipna,
            filter_type="bool",
        )

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="min")
    def min(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        """
        Return the minimum of the values over the requested axis.

        If you want the *index* of the minimum, use ``idxmin``.
        This is the equivalent of the ``numpy.ndarray`` method ``argmin``.

        Parameters
        ----------
        axis : {index (0)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        scalar or Series (if level specified)
            The maximum of the values in the Series.

        See Also
        --------
        numpy.min : Equivalent numpy function for arrays.
        Series.min : Return the minimum.
        Series.max : Return the maximum.
        Series.idxmin : Return the index of the minimum.
        Series.idxmax : Return the index of the maximum.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_arrays(
        ...     [["warm", "warm", "cold", "cold"], ["dog", "falcon", "fish", "spider"]],
        ...     names=["blooded", "animal"],
        ... )
        >>> s = pd.Series([4, 2, 0, 8], name="legs", index=idx)
        >>> s
        blooded  animal
        warm     dog       4
                 falcon    2
        cold     fish      0
                 spider    8
        Name: legs, dtype: int64

        >>> s.min()
        0
        """
        # 调用 NDFrame.min() 方法计算沿指定轴的最小值
        return NDFrame.min(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="max")
    # 定义一个方法用于计算沿指定轴的最大值
    def max(
        self,
        axis: Axis | None = 0,  # 轴参数，默认为0，适用于Series
        skipna: bool = True,    # 是否跳过NaN值，默认跳过
        numeric_only: bool = False,  # 是否仅包括数值类型列，默认为False
        **kwargs,                # 其他关键字参数
    ):
        """
        Return the maximum of the values over the requested axis.

        If you want the *index* of the maximum, use ``idxmax``.
        This is the equivalent of the ``numpy.ndarray`` method ``argmax``.

        Parameters
        ----------
        axis : {index (0)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        scalar or Series (if level specified)
            The maximum of the values in the Series.

        See Also
        --------
        numpy.max : Equivalent numpy function for arrays.
        Series.min : Return the minimum.
        Series.max : Return the maximum.
        Series.idxmin : Return the index of the minimum.
        Series.idxmax : Return the index of the maximum.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_arrays(
        ...     [["warm", "warm", "cold", "cold"], ["dog", "falcon", "fish", "spider"]],
        ...     names=["blooded", "animal"],
        ... )
        >>> s = pd.Series([4, 2, 0, 8], name="legs", index=idx)
        >>> s
        blooded  animal
        warm     dog       4
                 falcon    2
        cold     fish      0
                 spider    8
        Name: legs, dtype: int64

        >>> s.max()
        8
        """
        # 调用父类 NDFrame 的 max 方法来计算最大值
        return NDFrame.max(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="sum")
    # 定义一个方法用于计算沿指定轴的和，同时标记为已弃用，将在版本4.0移除
    def sum(
        self,
        axis: Axis | None = None,  # 轴参数，默认为None，适用于Series和DataFrame
        skipna: bool = True,       # 是否跳过NaN值，默认跳过
        numeric_only: bool = False,  # 是否仅包括数值类型列，默认为False
        min_count: int = 0,        # 最小非NaN值的数量，默认为0
        **kwargs,                  # 其他关键字参数
    ):
        """
        Return the sum of the values over the requested axis.

        This is equivalent to the method ``numpy.sum``.

        Parameters
        ----------
        axis : {index (0)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.sum with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer than
            ``min_count`` non-NA values are present the result will be NA.
        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        scalar or Series (if level specified)
            Median of the values for the requested axis.

        See Also
        --------
        numpy.sum : Equivalent numpy function for computing sum.
        Series.mean : Mean of the values.
        Series.median : Median of the values.
        Series.std : Standard deviation of the values.
        Series.var : Variance of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_arrays(
        ...     [["warm", "warm", "cold", "cold"], ["dog", "falcon", "fish", "spider"]],
        ...     names=["blooded", "animal"],
        ... )
        >>> s = pd.Series([4, 2, 0, 8], name="legs", index=idx)
        >>> s
        blooded  animal
        warm     dog       4
                 falcon    2
        cold     fish      0
                 spider    8
        Name: legs, dtype: int64

        >>> s.sum()
        14

        By default, the sum of an empty or all-NA Series is ``0``.

        >>> pd.Series([], dtype="float64").sum()  # min_count=0 is the default
        0.0

        This can be controlled with the ``min_count`` parameter. For example, if
        you'd like the sum of an empty series to be NaN, pass ``min_count=1``.

        >>> pd.Series([], dtype="float64").sum(min_count=1)
        nan

        Thanks to the ``skipna`` parameter, ``min_count`` handles all-NA and
        empty series identically.

        >>> pd.Series([np.nan]).sum()
        0.0

        >>> pd.Series([np.nan]).sum(min_count=1)
        nan
        """
        # 调用父类 NDFrame 的 sum 方法来计算沿指定轴的值的总和，并返回结果
        return NDFrame.sum(
            self,
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )
    # 使用装饰器将不推荐使用的非关键字参数转为关键字参数，版本为4.0，允许的参数包括self，函数名为"prod"
    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="prod")
    # 使用make_doc函数生成文档，并应用到"prod"函数上，指定ndim=1
    @doc(make_doc("prod", ndim=1))
    # 定义名为prod的函数，计算沿指定轴的乘积
    def prod(
        self,
        axis: Axis | None = None,  # 指定计算乘积的轴，默认为None
        skipna: bool = True,  # 是否跳过NaN值，默认为True
        numeric_only: bool = False,  # 是否仅包括数值列，默认为False
        min_count: int = 0,  # 最小非NaN值数量，默认为0
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类NDFrame的prod方法，传入参数进行计算
        return NDFrame.prod(
            self,
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    # 使用装饰器将不推荐使用的非关键字参数转为关键字参数，版本为4.0，允许的参数包括self，函数名为"mean"
    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="mean")
    # 定义名为mean的函数，计算沿指定轴的均值
    def mean(
        self,
        axis: Axis | None = 0,  # 指定计算均值的轴，默认为0
        skipna: bool = True,  # 是否跳过NaN值，默认为True
        numeric_only: bool = False,  # 是否仅包括数值列，默认为False
        **kwargs,  # 其他关键字参数
    ) -> Any:
        """
        返回沿请求的轴的值的均值。

        参数
        ----------
        axis : {index (0)}
            应用函数的轴。
            对于`Series`，此参数未使用，默认为0。

            对于DataFrames，指定``axis=None``将在两个轴上应用聚合函数。

            .. versionadded:: 2.0.0

        skipna : bool，默认 True
            计算结果时是否排除NA/null值。
        numeric_only : bool，默认 False
            是否仅包括浮点数、整数和布尔值列。
        **kwargs
            要传递给函数的其他关键字参数。

        返回
        -------
        scalar或Series（如果指定了level）
            请求轴上值的均值。

        另请参阅
        --------
        numpy.median：用于计算中位数的等效numpy函数。
        Series.sum：值的总和。
        Series.median：值的中位数。
        Series.std：值的标准偏差。
        Series.var：值的方差。
        Series.min：最小值。
        Series.max：最大值。

        示例
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.mean()
        2.0
        """
        # 调用父类NDFrame的mean方法，传入参数进行计算
        return NDFrame.mean(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    # 使用装饰器将不推荐使用的非关键字参数转为关键字参数，版本为4.0，允许的参数包括self，函数名为"median"
    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="median")
    # 定义名为median的函数，计算沿指定轴的中位数
    def median(
        self,
        axis: Axis | None = 0,  # 指定计算中位数的轴，默认为0
        skipna: bool = True,  # 是否跳过NaN值，默认为True
        numeric_only: bool = False,  # 是否仅包括数值列，默认为False
        **kwargs,  # 其他关键字参数

    ) -> Any:
        """
        返回沿请求的轴的值的中位数。

        参数
        ----------
        axis : {index (0)}
            应用函数的轴。
            对于`Series`，此参数未使用，默认为0。

            对于DataFrames，指定``axis=None``将在两个轴上应用聚合函数。

        skipna : bool，默认 True
            计算结果时是否排除NA/null值。
        numeric_only : bool，默认 False
            是否仅包括浮点数、整数和布尔值列。
        **kwargs
            要传递给函数的其他关键字参数。

        返回
        -------
        scalar或Series（如果指定了level）
            请求轴上值的中位数。

        另请参阅
        --------
        numpy.median：用于计算中位数的等效numpy函数。
        Series.sum：值的总和。
        Series.mean：值的均值。
        Series.std：值的标准偏差。
        Series.var：值的方差。
        Series.min：最小值。
        Series.max：最大值。
        """
        # 调用父类NDFrame的median方法，传入参数进行计算
        return NDFrame.median(
            self,
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            **kwargs,
        )
    ) -> Any:
        """
        Return the median of the values over the requested axis.

        Parameters
        ----------
        axis : {index (0)}
            要应用函数的轴。
            对于 `Series`，此参数未使用且默认为 0。
            
            对于 DataFrames，指定 ``axis=None`` 将应用于两个轴。

            .. versionadded:: 2.0.0

        skipna : bool, default True
            在计算结果时排除 NA/null 值。
        numeric_only : bool, default False
            仅包括 float、int 和 boolean 类型的列。
        **kwargs
            要传递给函数的额外关键字参数。

        Returns
        -------
        scalar or Series (if level specified)
            请求轴上值的中位数。

        See Also
        --------
        numpy.median : 计算中位数的等效 numpy 函数。
        Series.sum : 值的总和。
        Series.median : 值的中位数。
        Series.std : 值的标准差。
        Series.var : 值的方差。
        Series.min : 最小值。
        Series.max : 最大值。

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.median()
        2.0

        使用 DataFrame

        >>> df = pd.DataFrame({"a": [1, 2], "b": [2, 3]}, index=["tiger", "zebra"])
        >>> df
               a   b
        tiger  1   2
        zebra  2   3
        >>> df.median()
        a   1.5
        b   2.5
        dtype: float64

        使用 axis=1

        >>> df.median(axis=1)
        tiger   1.5
        zebra   2.5
        dtype: float64

        在这种情况下，应将 `numeric_only` 设置为 `True`，以避免出现错误。

        >>> df = pd.DataFrame({"a": [1, 2], "b": ["T", "Z"]}, index=["tiger", "zebra"])
        >>> df.median(numeric_only=True)
        a   1.5
        dtype: float64
        """
        return NDFrame.median(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="sem")
    @doc(make_doc("sem", ndim=1))
    def sem(
        self,
        axis: Axis | None = None,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ):
        """
        计算标准误差（SEM）。

        Parameters
        ----------
        axis : Axis or None, optional
            计算的轴。默认为 None。
        skipna : bool, optional
            是否排除 NA/null 值。默认为 True。
        ddof : int, optional
            自由度的修正值。默认为 1。
        numeric_only : bool, optional
            是否仅包含 float、int 和 boolean 类型的列。默认为 False。
        **kwargs
            传递给函数的额外关键字参数。

        Returns
        -------
        scalar or Series (if level specified)
            请求轴上值的标准误差（SEM）。

        See Also
        --------
        numpy.std : 计算标准差的等效 numpy 函数。
        Series.std : 值的标准差。
        Series.var : 值的方差。

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.sem()
        0.5773502691896257

        使用 DataFrame

        >>> df = pd.DataFrame({"a": [1, 2], "b": [2, 3]}, index=["tiger", "zebra"])
        >>> df
               a   b
        tiger  1   2
        zebra  2   3
        >>> df.sem()
        a    0.5
        b    0.5
        dtype: float64
        """
        return NDFrame.sem(
            self,
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="var")
    def var(
        self,
        axis: Axis | None = None,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ):
        """
        计算方差。

        Parameters
        ----------
        axis : Axis or None, optional
            计算的轴。默认为 None。
        skipna : bool, optional
            是否排除 NA/null 值。默认为 True。
        ddof : int, optional
            自由度的修正值。默认为 1。
        numeric_only : bool, optional
            是否仅包含 float、int 和 boolean 类型的列。默认为 False。
        **kwargs
            传递给函数的额外关键字参数。

        Returns
        -------
        scalar or Series (if level specified)
            请求轴上值的方差。

        See Also
        --------
        numpy.var : 计算方差的等效 numpy 函数。
        Series.std : 值的标准差。
        Series.sem : 值的标准误差（SEM）。

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.var()
        1.0

        使用 DataFrame

        >>> df = pd.DataFrame({"a": [1, 2], "b": [2, 3]}, index=["tiger", "zebra"])
        >>> df
               a   b
        tiger  1   2
        zebra  2   3
        >>> df.var()
        a    0.5
        b    0.5
        dtype: float64
        """
    ):
        """
        Return unbiased variance over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        Parameters
        ----------
        axis : {index (0)}
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.var with ``axis=None`` is deprecated,
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
        **kwargs :
            Additional keywords passed.

        Returns
        -------
        scalar or Series (if level specified)
            Unbiased variance over requested axis.

        See Also
        --------
        numpy.var : Equivalent function in NumPy.
        Series.std : Returns the standard deviation of the Series.
        DataFrame.var : Returns the variance of the DataFrame.
        DataFrame.std : Return standard deviation of the values over
            the requested axis.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "person_id": [0, 1, 2, 3],
        ...         "age": [21, 25, 62, 43],
        ...         "height": [1.61, 1.87, 1.49, 2.01],
        ...     }
        ... ).set_index("person_id")
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
        dtype: float64
        """
        return NDFrame.var(
            self,
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="std")
    @doc(make_doc("std", ndim=1))
    def std(
        self,
        axis: Axis | None = None,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ):
        """
        Return unbiased standard deviation over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        Parameters
        ----------
        axis : {index (0)}
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.std with ``axis=None`` is deprecated,
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
        **kwargs :
            Additional keywords passed.

        Returns
        -------
        scalar or Series (if level specified)
            Unbiased standard deviation over requested axis.

        See Also
        --------
        numpy.std : Equivalent function in NumPy.
        Series.var : Returns the variance of the Series.
        DataFrame.std : Returns the standard deviation of the DataFrame.
        DataFrame.var : Return variance of the values over
            the requested axis.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "person_id": [0, 1, 2, 3],
        ...         "age": [21, 25, 62, 43],
        ...         "height": [1.61, 1.87, 1.49, 2.01],
        ...     }
        ... ).set_index("person_id")
        >>> df
                   age  height
        person_id
        0           21    1.61
        1           25    1.87
        2           62    1.49
        3           43    2.01

        >>> df.std()
        age       18.772316
        height     0.237410
        dtype: float64
        """
        return NDFrame.std(
            self,
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )
    # 使用装饰器 @deprecate_nonkeyword_arguments 标记方法为已废弃，版本为 "4.0"，只允许 "self" 作为参数
    # 使用装饰器 @doc(make_doc("skew", ndim=1)) 添加文档说明，指定方法为 "skew"，维度为 1
    def skew(
        self,
        axis: Axis | None = 0,  # 定义轴参数，默认为 0
        skipna: bool = True,  # 定义是否跳过 NaN 值，默认为 True
        numeric_only: bool = False,  # 定义是否只包含数字类型列，默认为 False
        **kwargs,  # 允许额外的关键字参数
    ):
        # 调用 NDFrame 类的 skew 方法，传递参数并返回结果
        return NDFrame.skew(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    # 使用装饰器 @deprecate_nonkeyword_arguments 标记方法为已废弃，版本为 "4.0"，只允许 "self" 作为参数
    def kurt(
        self,
        axis: Axis | None = 0,  # 定义轴参数，默认为 0
        skipna: bool = True,  # 定义是否跳过 NaN 值，默认为 True
        numeric_only: bool = False,  # 定义是否只包含数字类型列，默认为 False
        **kwargs,  # 允许额外的关键字参数
    ):
        """
        返回请求轴上的无偏峰度。

        使用 Fisher 的定义计算峰度（正态分布的峰度 == 0.0）。由 N-1 规范化。

        参数
        ----------
        axis : {index (0)}
            应用函数的轴。
            对于 `Series`，此参数未使用且默认为 0。

            对于 DataFrames，指定 ``axis=None`` 将在两个轴上应用聚合。

            .. versionadded:: 2.0.0

        skipna : bool, default True
            计算结果时排除 NA/null 值。
        numeric_only : bool, default False
            仅包含 float、int 和 boolean 类型列。

        **kwargs
            传递给函数的额外关键字参数。

        返回
        -------
        scalar
            无偏峰度。

        另请参阅
        --------
        Series.skew : 返回请求轴上的无偏偏度。
        Series.var : 返回请求轴上的无偏方差。
        Series.std : 返回请求轴上的无偏标准差。

        示例
        --------
        >>> s = pd.Series([1, 2, 2, 3], index=["cat", "dog", "dog", "mouse"])
        >>> s
        cat    1
        dog    2
        dog    2
        mouse  3
        dtype: int64
        >>> s.kurt()
        1.5
        """
        # 调用 NDFrame 类的 kurt 方法，传递参数并返回结果
        return NDFrame.kurt(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    # 将 kurt 方法重命名为 kurtosis 方法
    kurtosis = kurt

    # 将 prod 方法赋值给 product
    product = prod

    # 使用装饰器 @doc(make_doc("cummin", ndim=1)) 添加文档说明，指定方法为 "cummin"，维度为 1
    def cummin(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Self:
        # 调用 NDFrame 类的 cummin 方法，传递参数并返回结果
        return NDFrame.cummin(self, axis, skipna, *args, **kwargs)

    # 使用装饰器 @doc(make_doc("cummax", ndim=1)) 添加文档说明，指定方法为 "cummax"，维度为 1
    def cummax(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Self:
        # 调用 NDFrame 类的 cummax 方法，传递参数并返回结果
        return NDFrame.cummax(self, axis, skipna, *args, **kwargs)

    # 使用装饰器 @doc(make_doc("cumsum", ndim=1)) 添加文档说明，指定方法为 "cumsum"，维度为 1
    def cumsum(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Self:
        # 调用 NDFrame 类的 cumsum 方法，传递参数并返回结果
        return NDFrame.cumsum(self, axis, skipna, *args, **kwargs)

    # 使用装饰器 @doc(make_doc("cumprod", 1)) 添加文档说明，指定方法为 "cumprod"，维度为 1
    def cumprod(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Self:
        # 调用 NDFrame 类的 cumprod 方法，传递参数并返回结果
        return NDFrame.cumprod(self, axis, skipna, *args, **kwargs)
```