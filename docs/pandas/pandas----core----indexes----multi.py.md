# `D:\src\scipysrc\pandas\pandas\core\indexes\multi.py`

```
from __future__ import annotations
# 导入未来版本兼容的 annotations 特性

from collections.abc import (
    Callable,
    Collection,
    Generator,
    Hashable,
    Iterable,
    Sequence,
)
# 导入抽象基类中的几种集合类型

from functools import wraps
# 导入 wraps 装饰器，用于复制函数的元数据

from sys import getsizeof
# 导入 sys 模块中的 getsizeof 函数，用于获取对象占用的内存大小

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)
# 导入类型提示相关的模块和类型

import warnings
# 导入警告模块，用于显示警告消息

import numpy as np
# 导入 NumPy 库，通常用 np 别名引用

from pandas._config import get_option
# 从 pandas 内部模块 _config 中导入 get_option 函数

from pandas._libs import (
    algos as libalgos,
    index as libindex,
    lib,
)
# 从 pandas 内部模块 _libs 中导入 algos、index 和 lib 等

from pandas._libs.hashtable import duplicated
# 从 pandas 内部模块 _libs.hashtable 中导入 duplicated 函数

from pandas._typing import (
    AnyAll,
    AnyArrayLike,
    Axis,
    DropKeep,
    DtypeObj,
    F,
    IgnoreRaise,
    IndexLabel,
    IndexT,
    Scalar,
    Self,
    Shape,
    npt,
)
# 从 pandas 内部模块 _typing 中导入多个类型别名

from pandas.compat.numpy import function as nv
# 从 pandas 兼容模块 numpy 中导入 function 函数并命名为 nv

from pandas.errors import (
    InvalidIndexError,
    PerformanceWarning,
    UnsortedIndexError,
)
# 从 pandas errors 模块中导入特定的异常类

from pandas.util._decorators import (
    Appender,
    cache_readonly,
    doc,
)
# 从 pandas.util._decorators 中导入装饰器和辅助功能

from pandas.util._exceptions import find_stack_level
# 从 pandas.util._exceptions 中导入 find_stack_level 函数

from pandas.core.dtypes.cast import coerce_indexer_dtype
# 从 pandas.core.dtypes.cast 中导入 coerce_indexer_dtype 函数

from pandas.core.dtypes.common import (
    ensure_int64,
    ensure_platform_int,
    is_hashable,
    is_integer,
    is_iterator,
    is_list_like,
    is_object_dtype,
    is_scalar,
    pandas_dtype,
)
# 从 pandas.core.dtypes.common 中导入多个通用函数

from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
)
# 从 pandas.core.dtypes.dtypes 中导入 CategoricalDtype 和 ExtensionDtype 类

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
# 从 pandas.core.dtypes.generic 中导入 ABCDataFrame 和 ABCSeries 类

from pandas.core.dtypes.inference import is_array_like
# 从 pandas.core.dtypes.inference 中导入 is_array_like 函数

from pandas.core.dtypes.missing import (
    array_equivalent,
    isna,
)
# 从 pandas.core.dtypes.missing 中导入 array_equivalent 和 isna 函数

import pandas.core.algorithms as algos
# 导入 pandas 核心算法模块，并使用 algos 别名引用

from pandas.core.array_algos.putmask import validate_putmask
# 从 pandas.core.array_algos.putmask 中导入 validate_putmask 函数

from pandas.core.arrays import (
    Categorical,
    ExtensionArray,
)
# 从 pandas.core.arrays 中导入 Categorical 和 ExtensionArray 类

from pandas.core.arrays.categorical import (
    factorize_from_iterables,
    recode_for_categories,
)
# 从 pandas.core.arrays.categorical 中导入 factorize_from_iterables 和 recode_for_categories 函数

import pandas.core.common as com
# 导入 pandas 核心通用模块，并使用 com 别名引用

from pandas.core.construction import sanitize_array
# 从 pandas.core.construction 中导入 sanitize_array 函数

import pandas.core.indexes.base as ibase
# 导入 pandas 核心索引基类模块，并使用 ibase 别名引用

from pandas.core.indexes.base import (
    Index,
    _index_shared_docs,
    ensure_index,
    get_unanimous_names,
)
# 从 pandas.core.indexes.base 中导入 Index 类和几个相关函数

from pandas.core.indexes.frozen import FrozenList
# 从 pandas.core.indexes.frozen 中导入 FrozenList 类

from pandas.core.ops.invalid import make_invalid_op
# 从 pandas.core.ops.invalid 中导入 make_invalid_op 函数

from pandas.core.sorting import (
    get_group_index,
    lexsort_indexer,
)
# 从 pandas.core.sorting 中导入 get_group_index 和 lexsort_indexer 函数

from pandas.io.formats.printing import pprint_thing
# 从 pandas.io.formats.printing 中导入 pprint_thing 函数

if TYPE_CHECKING:
    from pandas import (
        CategoricalIndex,
        DataFrame,
        Series,
    )
# 如果 TYPE_CHECKING 为真，则从 pandas 中导入 CategoricalIndex、DataFrame 和 Series 类

_index_doc_kwargs = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update(
    {"klass": "MultiIndex", "target_klass": "MultiIndex or list of tuples"}
)
# 创建 _index_doc_kwargs 字典并更新其内容

class MultiIndexUInt64Engine(libindex.BaseMultiIndexCodesEngine, libindex.UInt64Engine):
    """Manages a MultiIndex by mapping label combinations to positive integers.

    The number of possible label combinations must not overflow the 64 bits integers.
    """

    _base = libindex.UInt64Engine
    _codes_dtype = "uint64"
    # 定义 MultiIndexUInt64Engine 类，继承自 libindex.BaseMultiIndexCodesEngine 和 libindex.UInt64Engine

class MultiIndexUInt32Engine(libindex.BaseMultiIndexCodesEngine, libindex.UInt32Engine):
    # 定义 MultiIndexUInt32Engine 类，继承自 libindex.BaseMultiIndexCodesEngine 和 libindex.UInt32Engine
    """Manages a MultiIndex by mapping label combinations to positive integers.
    
    The number of possible label combinations must not overflow the 32 bits integers.
    """
    
    # 设定基础引擎为UInt32Engine，用于管理MultiIndex，并保证标签组合数量不会导致32位整数溢出
    _base = libindex.UInt32Engine
    
    # 指定索引的编码数据类型为uint32，即32位无符号整数
    _codes_dtype = "uint32"
class MultiIndexUInt16Engine(libindex.BaseMultiIndexCodesEngine, libindex.UInt16Engine):
    """Manages a MultiIndex by mapping label combinations to positive integers.

    The number of possible label combinations must not overflow the 16 bits integers.
    """

    # 继承自 BaseMultiIndexCodesEngine 和 UInt16Engine
    _base = libindex.UInt16Engine
    # 指定编码数据类型为 uint16
    _codes_dtype = "uint16"


class MultiIndexUInt8Engine(libindex.BaseMultiIndexCodesEngine, libindex.UInt8Engine):
    """Manages a MultiIndex by mapping label combinations to positive integers.

    The number of possible label combinations must not overflow the 8 bits integers.
    """

    # 继承自 BaseMultiIndexCodesEngine 和 UInt8Engine
    _base = libindex.UInt8Engine
    # 指定编码数据类型为 uint8
    _codes_dtype = "uint8"


class MultiIndexPyIntEngine(libindex.BaseMultiIndexCodesEngine, libindex.ObjectEngine):
    """Manages a MultiIndex by mapping label combinations to positive integers.

    This class manages those (extreme) cases in which the number of possible
    label combinations overflows the 64 bits integers, and uses an ObjectEngine
    containing Python integers.
    """

    # 继承自 BaseMultiIndexCodesEngine 和 ObjectEngine
    _base = libindex.ObjectEngine
    # 指定编码数据类型为 object
    _codes_dtype = "object"


def names_compat(meth: F) -> F:
    """
    A decorator to allow either `name` or `names` keyword but not both.

    This makes it easier to share code with base class.
    """

    @wraps(meth)
    def new_meth(self_or_cls, *args, **kwargs):
        if "name" in kwargs and "names" in kwargs:
            raise TypeError("Can only provide one of `names` and `name`")
        if "name" in kwargs:
            kwargs["names"] = kwargs.pop("name")

        return meth(self_or_cls, *args, **kwargs)

    return cast(F, new_meth)


class MultiIndex(Index):
    """
    A multi-level, or hierarchical, index object for pandas objects.

    Parameters
    ----------
    levels : sequence of arrays
        The unique labels for each level.
    codes : sequence of arrays
        Integers for each level designating which label at each location.
    sortorder : optional int
        Level of sortedness (must be lexicographically sorted by that
        level).
    names : optional sequence of objects
        Names for each of the index levels. (name is accepted for compat).
    dtype : Numpy dtype or pandas type, optional
        Data type for the MultiIndex.
    copy : bool, default False
        Copy the meta-data.
    name : Label
        Kept for compatibility with 1-dimensional Index. Should not be used.
    verify_integrity : bool, default True
        Check that the levels/codes are consistent and valid.

    Attributes
    ----------
    names
    levels
    codes
    nlevels
    levshape
    dtypes

    Methods
    -------
    from_arrays
    from_tuples
    from_product
    from_frame
    set_levels
    set_codes
    to_frame
    to_flat_index
    sortlevel
    droplevel
    swaplevel
    reorder_levels
    remove_unused_levels
    get_level_values
    get_indexer
    get_loc
    get_locs
    get_loc_level
    drop

    See Also
    --------

    """

    # 管理 pandas 对象的多级索引
    def __init__(
        self,
        levels=None,
        codes=None,
        sortorder=None,
        names=None,
        dtype=None,
        copy=False,
        name=None,  # 与一维索引兼容性保留的参数，不推荐使用
        verify_integrity=True,
    ):
        super().__init__()
        # 初始化 MultiIndex 对象
        pass
    # _hidden_attrs 是 Index 类型的隐藏属性的一个空集合
    _hidden_attrs = Index._hidden_attrs | frozenset()
    
    # 初始化为长度为零的元组列表，以确保所有操作正常运行
    _typ = "multiindex"
    _names: list[Hashable | None] = []  # 存储 MultiIndex 的名称
    _levels = FrozenList()  # 存储 MultiIndex 的层级信息
    _codes = FrozenList()   # 存储 MultiIndex 的编码信息
    _comparables = ["names"]  # 指定 MultiIndex 可比较的部分为名称
    
    sortorder: int | None  # MultiIndex 的排序顺序，可以为整数或 None
    
    # --------------------------------------------------------------------
    # 构造函数
    
    def __new__(
        cls,
        levels=None,
        codes=None,
        sortorder=None,
        names=None,
        dtype=None,
        copy: bool = False,
        name=None,
        verify_integrity: bool = True,
    ) -> Self:
        # 兼容 Index 类型
        if name is not None:
            names = name
        # 检查 levels 和 codes 是否都已传入
        if levels is None or codes is None:
            raise TypeError("Must pass both levels and codes")
        # 检查 levels 和 codes 的长度是否一致
        if len(levels) != len(codes):
            raise ValueError("Length of levels and codes must be the same.")
        # 检查 levels 和 codes 的长度是否大于 0
        if len(levels) == 0:
            raise ValueError("Must pass non-zero number of levels/codes")
    
        result = object.__new__(cls)
        result._cache = {}  # 创建结果对象的缓存字典
    
        # 因为已经验证了 levels 和 codes，所以在此处进行快捷处理
        result._set_levels(levels, copy=copy, validate=False)  # 设置 MultiIndex 的层级信息
        result._set_codes(codes, copy=copy, validate=False)    # 设置 MultiIndex 的编码信息
    
        result._names = [None] * len(levels)  # 设置 MultiIndex 的名称列表为 None
        if names is not None:
            # 处理名称的验证和设置
            result._set_names(names)  # 设置 MultiIndex 的名称
    
        if sortorder is not None:
            result.sortorder = int(sortorder)  # 设置 MultiIndex 的排序顺序为整数值
        else:
            result.sortorder = sortorder  # 设置 MultiIndex 的排序顺序为 None
    
        if verify_integrity:
            new_codes = result._verify_integrity()  # 验证 MultiIndex 的完整性，并返回新的编码
            result._codes = new_codes  # 更新 MultiIndex 的编码信息
    
        result._reset_identity()  # 重置 MultiIndex 的标识信息
        result._references = None  # 初始化 MultiIndex 的引用信息为 None
    
        return result  # 返回构造好的 MultiIndex 对象
    def _validate_codes(self, level: Index, code: np.ndarray) -> np.ndarray:
        """
        Reassign code values as -1 if their corresponding levels are NaN.

        Parameters
        ----------
        code : Index
            Code to reassign.
        level : np.ndarray
            Level to check for missing values (NaN, NaT, None).

        Returns
        -------
        np.ndarray
            New code where code value = -1 if it corresponds
            to a level with missing values (NaN, NaT, None).
        """
        # 创建一个布尔掩码，标识哪些 level 是 NaN
        null_mask = isna(level)
        # 如果存在任何 NaN 的 level，则更新 code 数组中对应位置的值为 -1
        if np.any(null_mask):
            code = np.where(null_mask[code], -1, code)
        # 返回更新后的 code 数组
        return code

    def _verify_integrity(
        self,
        codes: list | None = None,
        levels: list | None = None,
        levels_to_verify: list[int] | range | None = None,
    ) -> FrozenList:
        """
        Parameters
        ----------
        codes : optional list
            Codes to check for validity. Defaults to current codes.
        levels : optional list
            Levels to check for validity. Defaults to current levels.
        levels_to_validate: optional list
            Specifies the levels to verify.

        Raises
        ------
        ValueError
            If length of levels and codes don't match, if the codes for any
            level would exceed level bounds, or there are any duplicate levels.

        Returns
        -------
        new codes where code value = -1 if it corresponds to a
        NaN level.
        """
        # NOTE: Currently does not check, among other things, that cached
        # nlevels matches nor that sortorder matches actually sortorder.
        
        # 如果没有提供 codes 参数，则使用当前对象的 codes
        codes = codes or self.codes
        # 如果没有提供 levels 参数，则使用当前对象的 levels
        levels = levels or self.levels
        # 如果未指定 levels_to_verify 参数，则验证所有 levels 的索引
        if levels_to_verify is None:
            levels_to_verify = range(len(levels))

        # 检查 levels 和 codes 的长度是否相同
        if len(levels) != len(codes):
            raise ValueError(
                "Length of levels and codes must match. NOTE: "
                "this index is in an inconsistent state."
            )
        
        # 检查 codes 中每个元素的长度是否相同
        codes_length = len(codes[0])
        for i in levels_to_verify:
            level = levels[i]
            level_codes = codes[i]

            if len(level_codes) != codes_length:
                raise ValueError(
                    f"Unequal code lengths: {[len(code_) for code_ in codes]}"
                )
            # 检查 level_codes 中的最大值是否超过 level 的长度
            if len(level_codes) and level_codes.max() >= len(level):
                raise ValueError(
                    f"On level {i}, code max ({level_codes.max()}) >= length of "
                    f"level ({len(level)}). NOTE: this index is in an "
                    "inconsistent state"
                )
            # 检查 level_codes 中的最小值是否小于 -1
            if len(level_codes) and level_codes.min() < -1:
                raise ValueError(f"On level {i}, code value ({level_codes.min()}) < -1")
            # 检查 level 是否包含重复的值
            if not level.is_unique:
                raise ValueError(
                    f"Level values must be unique: {list(level)} on level {i}"
                )
        
        # 如果 sortorder 不为 None，则检查其是否小于等于 _lexsort_depth 函数返回的值
        if self.sortorder is not None:
            if self.sortorder > _lexsort_depth(self.codes, self.nlevels):
                raise ValueError(
                    "Value for sortorder must be inferior or equal to actual "
                    f"lexsort_depth: sortorder {self.sortorder} "
                    f"with lexsort_depth {_lexsort_depth(self.codes, self.nlevels)}"
                )

        # 对每个 level 进行代码验证，将验证后的结果存入 result_codes 中
        result_codes = []
        for i in range(len(levels)):
            if i in levels_to_verify:
                result_codes.append(self._validate_codes(levels[i], codes[i]))
            else:
                result_codes.append(codes[i])

        # 将结果转为不可变列表并返回
        new_codes = FrozenList(result_codes)
        return new_codes
    # 定义一个类方法，用于从数组转换为多级索引对象 MultiIndex
    def from_arrays(
        cls,
        arrays,  # arrays 参数是一个包含数组的列表或序列，每个数组代表一个级别的值
        sortorder: int | None = None,  # sortorder 参数用于指定排序顺序，可以是整数或 None
        names: Sequence[Hashable] | Hashable | lib.NoDefault = lib.no_default,  # names 参数用于指定索引级别的名称，可以是字符串序列或单个字符串，缺省时使用 lib.no_default
    ) -> MultiIndex:
        """
        Convert arrays to MultiIndex.

        Parameters
        ----------
        arrays : list / sequence of array-likes
            Each array-like gives one level's value for each data point.
            len(arrays) is the number of levels.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> arrays = [[1, 1, 2, 2], ["red", "blue", "red", "blue"]]
        >>> pd.MultiIndex.from_arrays(arrays, names=("number", "color"))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """

        error_msg = "Input must be a list / sequence of array-likes."

        # 检查 arrays 是否是列表或序列类型，如果不是则抛出 TypeError 异常
        if not is_list_like(arrays):
            raise TypeError(error_msg)

        # 如果 arrays 是迭代器，则将其转换为列表
        if is_iterator(arrays):
            arrays = list(arrays)

        # 检查 arrays 中每个元素是否为列表或类似列表的类型，如果不是则抛出 TypeError 异常
        for array in arrays:
            if not is_list_like(array):
                raise TypeError(error_msg)

        # 检查 arrays 中所有数组的长度是否相等，如果不相等则抛出 ValueError 异常
        for i in range(1, len(arrays)):
            if len(arrays[i]) != len(arrays[i - 1]):
                raise ValueError("all arrays must be same length")

        # 使用 factorize_from_iterables 函数从 arrays 中获取 codes 和 levels
        codes, levels = factorize_from_iterables(arrays)

        # 如果 names 参数为 lib.no_default，则尝试从 arrays 中获取每个数组的名称作为 names
        if names is lib.no_default:
            names = [getattr(arr, "name", None) for arr in arrays]

        # 返回通过类的构造方法创建的 MultiIndex 对象，传入 levels、codes、sortorder 和 names 等参数
        return cls(
            levels=levels,
            codes=codes,
            sortorder=sortorder,
            names=names,
            verify_integrity=False,
        )

    @classmethod
    @names_compat
    def from_tuples(
        cls,
        tuples: Iterable[tuple[Hashable, ...]],
        sortorder: int | None = None,
        names: Sequence[Hashable] | Hashable | None = None,
    ) -> MultiIndex:
        """
        Convert list of tuples to MultiIndex.

        Parameters
        ----------
        tuples : list / sequence of tuple-likes
            Each tuple is the index of one row/column.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> tuples = [(1, "red"), (1, "blue"), (2, "red"), (2, "blue")]
        >>> pd.MultiIndex.from_tuples(tuples, names=("number", "color"))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        if not is_list_like(tuples):
            raise TypeError("Input must be a list / sequence of tuple-likes.")
        if is_iterator(tuples):
            tuples = list(tuples)
        tuples = cast(Collection[tuple[Hashable, ...]], tuples)

        # handling the empty tuple cases
        if len(tuples) and all(isinstance(e, tuple) and not e for e in tuples):
            # Create codes and levels for an empty tuple scenario
            codes = [np.zeros(len(tuples))]  # Generate zero-filled codes
            levels = [Index(com.asarray_tuplesafe(tuples, dtype=np.dtype("object")))]  # Convert tuples to an Index object
            return cls(
                levels=levels,
                codes=codes,
                sortorder=sortorder,
                names=names,
                verify_integrity=False,
            )

        arrays: list[Sequence[Hashable]]
        if len(tuples) == 0:
            if names is None:
                raise TypeError("Cannot infer number of levels from empty list")
            # error: Argument 1 to "len" has incompatible type "Hashable";
            # expected "Sized"
            arrays = [[]] * len(names)  # type: ignore[arg-type]
        elif isinstance(tuples, (np.ndarray, Index)):
            if isinstance(tuples, Index):
                tuples = np.asarray(tuples._values)

            # Convert tuples to object arrays for numpy operations
            arrays = list(lib.tuples_to_object_array(tuples).T)
        elif isinstance(tuples, list):
            # Convert tuples to object arrays using pandas's internal function
            arrays = list(lib.to_object_array_tuples(tuples).T)
        else:
            # Assume tuples is an iterator yielding sequences of hashable objects
            arrs = zip(*tuples)
            arrays = cast(list[Sequence[Hashable]], arrs)

        return cls.from_arrays(arrays, sortorder=sortorder, names=names)
    ) -> MultiIndex:
        """
        从多个可迭代对象的笛卡尔积生成一个 MultiIndex。

        Parameters
        ----------
        iterables : list / sequence of iterables
            每个可迭代对象都为索引的每个级别提供唯一标签。
        sortorder : int or None
            排序级别（必须按照该级别的词典顺序排序）。
        names : list / sequence of str, optional
            索引中每个级别的名称。
            如果未明确提供，名称将从可迭代对象的元素中推断，如果元素具有名称属性的话。

        Returns
        -------
        MultiIndex
            生成的 MultiIndex 对象。

        See Also
        --------
        MultiIndex.from_arrays : 从数组列表生成 MultiIndex。
        MultiIndex.from_tuples : 从元组列表生成 MultiIndex。
        MultiIndex.from_frame : 从 DataFrame 生成 MultiIndex。

        Examples
        --------
        >>> numbers = [0, 1, 2]
        >>> colors = ["green", "purple"]
        >>> pd.MultiIndex.from_product([numbers, colors], names=["number", "color"])
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        """
        from pandas.core.reshape.util import cartesian_product

        if not is_list_like(iterables):
            raise TypeError("Input must be a list / sequence of iterables.")
        if is_iterator(iterables):
            iterables = list(iterables)

        # 使用 factorize_from_iterables 函数处理 iterables，返回 codes 和 levels
        codes, levels = factorize_from_iterables(iterables)
        
        # 如果未提供 names 参数，则从 iterables 的元素中推断名称
        if names is lib.no_default:
            names = [getattr(it, "name", None) for it in iterables]

        # codes 全部为 ndarray，因此 cartesian_product 是无损的
        codes = cartesian_product(codes)
        
        # 返回使用 levels、codes、sortorder 和 names 构建的 MultiIndex 对象
        return cls(levels, codes, sortorder=sortorder, names=names)
    ) -> MultiIndex:
        """
        从 DataFrame 创建 MultiIndex。

        Parameters
        ----------
        df : DataFrame
            要转换为 MultiIndex 的 DataFrame。
        sortorder : int, optional
            排序级别（必须按该级别的词典顺序排序）。
        names : list-like, optional
            如果未提供名称，则使用列名，或者如果列是 MultiIndex，则使用列名元组。如果是序列，则使用给定序列覆盖名称。

        Returns
        -------
        MultiIndex
            给定 DataFrame 的 MultiIndex 表示。

        See Also
        --------
        MultiIndex.from_arrays : 将数组列表转换为 MultiIndex。
        MultiIndex.from_tuples : 将元组列表转换为 MultiIndex。
        MultiIndex.from_product : 从可迭代对象的笛卡尔积创建 MultiIndex。

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [["HI", "Temp"], ["HI", "Precip"], ["NJ", "Temp"], ["NJ", "Precip"]],
        ...     columns=["a", "b"],
        ... )
        >>> df
              a       b
        0    HI    Temp
        1    HI  Precip
        2    NJ    Temp
        3    NJ  Precip

        >>> pd.MultiIndex.from_frame(df)
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['a', 'b'])

        使用显式名称，而不是列名

        >>> pd.MultiIndex.from_frame(df, names=["state", "observation"])
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['state', 'observation'])
        """
        # 检查输入是否为 DataFrame 类型，如果不是则引发 TypeError
        if not isinstance(df, ABCDataFrame):
            raise TypeError("Input must be a DataFrame")

        # 将 DataFrame 列名和列值分别解压缩成两个元组
        column_names, columns = zip(*df.items())
        # 如果未提供 names 参数，则使用列名作为 names
        names = column_names if names is None else names
        # 调用类方法 from_arrays 创建 MultiIndex 对象，并返回
        return cls.from_arrays(columns, sortorder=sortorder, names=names)

    # --------------------------------------------------------------------
    def _values(self) -> np.ndarray:
        # We override here, since our parent uses _data, which we don't use.
        # 重写 _values 方法，因为父类使用 _data，而我们不使用它。

        values = []

        for i in range(self.nlevels):
            # Iterate over levels to process each level's index and codes
            index = self.levels[i]
            codes = self.codes[i]

            vals = index
            if isinstance(vals.dtype, CategoricalDtype):
                # If the index values are of CategoricalDtype, convert them to CategoricalIndex
                vals = cast("CategoricalIndex", vals)
                # Fetch internal values from CategoricalIndex
                vals = vals._data._internal_get_values()

            if isinstance(vals.dtype, ExtensionDtype) or lib.is_np_dtype(
                vals.dtype, "mM"
            ):
                # If values are of ExtensionDtype or datetime64/timedelta64, convert to object dtype
                vals = vals.astype(object)

            # Convert values to a NumPy array
            array_vals = np.asarray(vals)
            # Reconstruct values using codes and fill missing values with index's na_value
            array_vals = algos.take_nd(array_vals, codes, fill_value=index._na_value)
            # Append processed array_vals to values list
            values.append(array_vals)

        # Zip processed values into a structured NumPy array and return
        arr = lib.fast_zip(values)
        return arr

    @property
    def values(self) -> np.ndarray:
        # Property to return the values obtained from _values method
        return self._values

    @property
    def array(self):
        """
        Raises a ValueError for `MultiIndex` because there's no single
        array backing a MultiIndex.

        Raises
        ------
        ValueError
        """
        # Property that raises an error indicating MultiIndex doesn't have a single backing array
        raise ValueError(
            "MultiIndex has no single backing array. Use "
            "'MultiIndex.to_numpy()' to get a NumPy array of tuples."
        )

    @cache_readonly
    def dtypes(self) -> Series:
        """
        Return the dtypes as a Series for the underlying MultiIndex.

        See Also
        --------
        Index.dtype : Return the dtype object of the underlying data.
        Series.dtypes : Return the data type of the underlying Series.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_product(
        ...     [(0, 1, 2), ("green", "purple")], names=["number", "color"]
        ... )
        >>> idx
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        >>> idx.dtypes
        number     int64
        color     object
        dtype: object
        """
        # Method to return a Series of dtypes for each level in the MultiIndex
        from pandas import Series

        # Fill missing names and create a Series of dtypes with corresponding Index
        names = com.fill_missing_names([level.name for level in self.levels])
        return Series([level.dtype for level in self.levels], index=Index(names))

    def __len__(self) -> int:
        # Return the length of the first level codes, which represents the number of elements in the MultiIndex
        return len(self.codes[0])

    @property
    def size(self) -> int:
        """
        Return the number of elements in the underlying data.
        """
        # Property to return the size of the MultiIndex, equivalent to its length
        # override Index.size to avoid materializing _values
        return len(self)

    # --------------------------------------------------------------------
    # Levels Methods

    @cache_readonly
    def levels(self) -> FrozenList:
        """
        Levels of the MultiIndex.

        Levels refer to the different hierarchical levels or layers in a MultiIndex.
        In a MultiIndex, each level represents a distinct dimension or category of
        the index.

        To access the levels, you can use the levels attribute of the MultiIndex,
        which returns a tuple of Index objects. Each Index object represents a
        level in the MultiIndex and contains the unique values found in that
        specific level.

        If a MultiIndex is created with levels A, B, C, and the DataFrame using
        it filters out all rows of the level C, MultiIndex.levels will still
        return A, B, C.

        See Also
        --------
        MultiIndex.codes : The codes of the levels in the MultiIndex.
        MultiIndex.get_level_values : Return vector of label values for requested
            level.

        Examples
        --------
        >>> index = pd.MultiIndex.from_product(
        ...     [["mammal"], ("goat", "human", "cat", "dog")],
        ...     names=["Category", "Animals"],
        ... )
        >>> leg_num = pd.DataFrame(data=(4, 2, 4, 4), index=index, columns=["Legs"])
        >>> leg_num
                          Legs
        Category Animals
        mammal   goat        4
                 human       2
                 cat         4
                 dog         4

        >>> leg_num.index.levels
        FrozenList([['mammal'], ['cat', 'dog', 'goat', 'human']])

        MultiIndex levels will not change even if the DataFrame using the MultiIndex
        does not contain all of them anymore.
        See how "human" is not in the DataFrame, but it is still in levels:

        >>> large_leg_num = leg_num[leg_num.Legs > 2]
        >>> large_leg_num
                          Legs
        Category Animals
        mammal   goat        4
                 cat         4
                 dog         4

        >>> large_leg_num.index.levels
        FrozenList([['mammal'], ['cat', 'dog', 'goat', 'human']])
        """
        # Use cache_readonly to ensure that self.get_locs doesn't repeatedly
        # create new IndexEngine
        # https://github.com/pandas-dev/pandas/issues/31648
        # 通过 cache_readonly 确保 self.get_locs 不会重复创建新的 IndexEngine
        result = [x._rename(name=name) for x, name in zip(self._levels, self._names)]
        for level in result:
            # disallow midx.levels[0].name = "foo"
            # 禁止 midx.levels[0].name = "foo"
            level._no_setting_name = True
        # 返回冻结的列表作为结果
        return FrozenList(result)
    ) -> None:
        """
        Set the levels of the MultiIndex.

        Parameters
        ----------
        levels : list-like
            New levels to be set.
        level : list-like or None, optional
            Specific levels to set.
        verify_integrity : bool, default True
            Whether to verify the integrity of the MultiIndex after setting levels.

        Notes
        -----
        This method adjusts the levels of the MultiIndex. If `verify_integrity` is True,
        it ensures that the MultiIndex remains consistent after modification.

        Raises
        ------
        ValueError
            - If no levels are provided.
            - If the length of `levels` does not match `self.nlevels` when `level` is None.
            - If the lengths of `levels` and `level` do not match when `level` is provided.

        """
        # Validate the input levels if requested
        if validate:
            if len(levels) == 0:
                raise ValueError("Must set non-zero number of levels.")
            if level is None and len(levels) != self.nlevels:
                raise ValueError("Length of levels must match number of levels.")
            if level is not None and len(levels) != len(level):
                raise ValueError("Length of levels must match length of level.")

        # Compute new levels based on the input
        if level is None:
            # Create a new FrozenList of levels
            new_levels = FrozenList(
                ensure_index(lev, copy=copy)._view() for lev in levels
            )
            # Determine level numbers to be affected
            level_numbers: range | list[int] = range(len(new_levels))
        else:
            # Get the level numbers corresponding to the provided levels
            level_numbers = [self._get_level_number(lev) for lev in level]
            # Update the existing levels with new values
            new_levels_list = list(self._levels)
            for lev_num, lev in zip(level_numbers, levels):
                new_levels_list[lev_num] = ensure_index(lev, copy=copy)._view()
            new_levels = FrozenList(new_levels_list)

        # Verify integrity if requested
        if verify_integrity:
            new_codes = self._verify_integrity(
                levels=new_levels, levels_to_verify=level_numbers
            )
            self._codes = new_codes

        # Update the names and levels of the MultiIndex
        names = self.names
        self._levels = new_levels
        if any(names):
            self._set_names(names)

        # Reset any cached attributes
        self._reset_cache()

    def set_levels(
        self, levels, *, level=None, verify_integrity: bool = True
    ):
        """
        Set the levels of the MultiIndex.

        Parameters
        ----------
        levels : list-like
            New levels to be set.
        level : list-like or None, optional
            Specific levels to set.
        verify_integrity : bool, default True
            Whether to verify the integrity of the MultiIndex after setting levels.

        Notes
        -----
        This method adjusts the levels of the MultiIndex. If `verify_integrity` is True,
        it ensures that the MultiIndex remains consistent after modification.

        Raises
        ------
        ValueError
            - If no levels are provided.
            - If the length of `levels` does not match `self.nlevels` when `level` is None.
            - If the lengths of `levels` and `level` do not match when `level` is provided.

        """
        # Code for setting levels has been implemented in the function above
        pass

    @property
    def nlevels(self) -> int:
        """
        Integer number of levels in this MultiIndex.

        See Also
        --------
        MultiIndex.levels : Get the levels of the MultiIndex.
        MultiIndex.codes : Get the codes of the MultiIndex.
        MultiIndex.from_arrays : Convert arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([["a"], ["b"], ["c"]])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.nlevels
        3
        """
        return len(self._levels)

    @property
    def levshape(self) -> Shape:
        """
        A tuple with the length of each level.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([["a"], ["b"], ["c"]])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.levshape
        (1, 1, 1)
        """
        return tuple(len(x) for x in self.levels)

    # --------------------------------------------------------------------
    # Codes Methods

    @property
    def codes(self) -> FrozenList:
        """
        Codes of the MultiIndex.

        Codes are the position of the index value in the list of level values
        for each level.

        Returns
        -------
        tuple of numpy.ndarray
            The codes of the MultiIndex. Each array in the tuple corresponds
            to a level in the MultiIndex.

        See Also
        --------
        MultiIndex.set_codes : Set new codes on MultiIndex.

        Examples
        --------
        >>> arrays = [[1, 1, 2, 2], ["red", "blue", "red", "blue"]]
        >>> mi = pd.MultiIndex.from_arrays(arrays, names=("number", "color"))
        >>> mi.codes
        FrozenList([[0, 0, 1, 1], [1, 0, 1, 0]])
        """
        return self._codes


    def _set_codes(
        self,
        codes,
        *,
        level=None,
        copy: bool = False,
        validate: bool = True,
        verify_integrity: bool = False,
    ) -> None:
        if validate:
            # 如果需要验证，则检查 codes 的长度是否与层级数量匹配
            if level is None and len(codes) != self.nlevels:
                raise ValueError("Length of codes must match number of levels")
            if level is not None and len(codes) != len(level):
                raise ValueError("Length of codes must match length of levels.")

        level_numbers: list[int] | range
        if level is None:
            # 如果未指定 level，则按照每个层级对应的 codes 创建新的 FrozenList
            new_codes = FrozenList(
                _coerce_indexer_frozen(level_codes, lev, copy=copy).view()
                for lev, level_codes in zip(self._levels, codes)
            )
            level_numbers = range(len(new_codes))
        else:
            # 否则，根据指定的 level，更新对应层级的 codes
            level_numbers = [self._get_level_number(lev) for lev in level]
            new_codes_list = list(self._codes)
            for lev_num, level_codes in zip(level_numbers, codes):
                lev = self.levels[lev_num]
                new_codes_list[lev_num] = _coerce_indexer_frozen(
                    level_codes, lev, copy=copy
                )
            new_codes = FrozenList(new_codes_list)

        if verify_integrity:
            # 如果需要验证数据完整性，则调用 _verify_integrity 方法验证新的 codes
            new_codes = self._verify_integrity(
                codes=new_codes, levels_to_verify=level_numbers
            )

        self._codes = new_codes  # 更新 MultiIndex 的 codes

        self._reset_cache()  # 重置缓存


    def set_codes(
        self, codes, *, level=None, verify_integrity: bool = True
    ):
        # 设置 MultiIndex 的 codes
        # 参数 codes: 新的 codes
        # 参数 level: 指定要设置的层级
        # 参数 verify_integrity: 是否验证数据完整性，默认为 True
    ) -> MultiIndex:
        """
        Set new codes on MultiIndex. Defaults to returning new index.

        Parameters
        ----------
        codes : sequence or list of sequence
            New codes to apply.
        level : int, level name, or sequence of int/level names (default None)
            Level(s) to set (None for all levels).
        verify_integrity : bool, default True
            If True, checks that levels and codes are compatible.

        Returns
        -------
        new index (of same type and class...etc) or None
            The same type as the caller or None if ``inplace=True``.

        See Also
        --------
        MultiIndex.set_levels : Set new levels on MultiIndex.
        MultiIndex.codes : Get the codes of the levels in the MultiIndex.
        MultiIndex.levels : Get the levels of the MultiIndex.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [(1, "one"), (1, "two"), (2, "one"), (2, "two")], names=["foo", "bar"]
        ... )
        >>> idx
        MultiIndex([(1, 'one'),
            (1, 'two'),
            (2, 'one'),
            (2, 'two')],
           names=['foo', 'bar'])

        >>> idx.set_codes([[1, 0, 1, 0], [0, 0, 1, 1]])
        MultiIndex([(2, 'one'),
                    (1, 'one'),
                    (2, 'two'),
                    (1, 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_codes([1, 0, 1, 0], level=0)
        MultiIndex([(2, 'one'),
                    (1, 'two'),
                    (2, 'one'),
                    (1, 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_codes([0, 0, 1, 1], level="bar")
        MultiIndex([(1, 'one'),
                    (1, 'one'),
                    (2, 'two'),
                    (2, 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_codes([[1, 0, 1, 0], [0, 0, 1, 1]], level=[0, 1])
        MultiIndex([(2, 'one'),
                    (1, 'one'),
                    (2, 'two'),
                    (1, 'two')],
                   names=['foo', 'bar'])
        """

        # 需要确保 level 和 codes 参数是列表或列表的列表
        level, codes = _require_listlike(level, codes, "Codes")
        # 创建当前 MultiIndex 的视图
        idx = self._view()
        # 重置视图的标识信息
        idx._reset_identity()
        # 设置新的 codes 到 MultiIndex，支持指定的 level，并可选验证完整性
        idx._set_codes(codes, level=level, verify_integrity=verify_integrity)
        # 返回设置完 codes 后的新 MultiIndex 对象
        return idx

    # --------------------------------------------------------------------
    # Index Internals

    @cache_readonly
    # 计算每个层级中表示标签所需的位数，使用每个级别大小的对数（以2为底）向上取整：
    # NaN 值被移到1，其他缺失值在计算索引时被移到0
    sizes = np.ceil(
        np.log2(
            [len(level) + libindex.multiindex_nulls_shift for level in self.levels]
        )
    )

    # 从右侧开始累积位数，以获取偏移量，确保按字典顺序排序
    lev_bits = np.cumsum(sizes[::-1])[::-1]

    # 为了获得偏移量，将每个级别的位移量连接起来，以便排序组合的移位代码等同于按字典顺序排序代码本身
    # 注意，每个级别都需要按前面级别所需的位数进行移位
    offsets = np.concatenate([lev_bits[1:], [0]])
    # 如果可能的话，将类型降级，以防在移位代码时进行类型提升
    offsets = offsets.astype(np.min_scalar_type(int(offsets[0])))

    # 检查表示所需的总位数：
    if lev_bits[0] > 64:
        # 级别将溢出64位整数 - 使用 Python 整数：
        return MultiIndexPyIntEngine(self.levels, self.codes, offsets)
    if lev_bits[0] > 32:
        # 级别将溢出32位整数 - 使用 uint64
        return MultiIndexUInt64Engine(self.levels, self.codes, offsets)
    if lev_bits[0] > 16:
        # 级别将溢出16位整数 - 使用 uint32
        return MultiIndexUInt32Engine(self.levels, self.codes, offsets)
    if lev_bits[0] > 8:
        # 级别将溢出8位整数 - 使用 uint16
        return MultiIndexUInt16Engine(self.levels, self.codes, offsets)
    # 级别适合于8位整数 - 使用 uint8
    return MultiIndexUInt8Engine(self.levels, self.codes, offsets)
    # error: Signature of "copy" incompatible with supertype "Index"
    def copy(  # type: ignore[override]
        self,
        names=None,
        deep: bool = False,
        name=None,
    ) -> Self:
        """
        Make a copy of this object.

        Names, dtype, levels and codes can be passed and will be set on new copy.

        Parameters
        ----------
        names : sequence, optional
            Sequence of names to set on the new copy.
        deep : bool, default False
            If True, perform deep copy of levels and codes.
        name : Label
            Kept for compatibility with 1-dimensional Index. Should not be used.

        Returns
        -------
        MultiIndex
            A new instance of MultiIndex with copied attributes.

        Notes
        -----
        In most cases, there should be no functional difference from using
        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.
        This could be potentially expensive on large MultiIndex objects.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([["a"], ["b"], ["c"]])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.copy()
        MultiIndex([('a', 'b', 'c')],
                   )
        """
        # Validate and retrieve names to use
        names = self._validate_names(name=name, names=names, deep=deep)
        keep_id = not deep
        levels, codes = None, None

        # Perform deep copy if deep is True
        if deep:
            from copy import deepcopy

            levels = deepcopy(self.levels)
            codes = deepcopy(self.codes)

        # Ensure levels and codes are initialized correctly
        levels = levels if levels is not None else self.levels
        codes = codes if codes is not None else self.codes

        # Create a new instance of the same type (MultiIndex)
        new_index = type(self)(
            levels=levels,
            codes=codes,
            sortorder=self.sortorder,
            names=names,
            verify_integrity=False,
        )

        # Copy cache attributes from the original instance
        new_index._cache = self._cache.copy()
        new_index._cache.pop("levels", None)  # GH32669

        # Preserve identity if deep copy is not performed
        if keep_id:
            new_index._id = self._id

        return new_index

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """the array interface, return my values"""
        # Return the values of the MultiIndex as an array
        return self.values

    def view(self, cls=None) -> Self:
        """this is defined as a copy with the same identity"""
        # Create a view of the MultiIndex by making a shallow copy
        result = self.copy()
        result._id = self._id  # Maintain the same identity
        return result

    @doc(Index.__contains__)
    def __contains__(self, key: Any) -> bool:
        # Check if the key exists in the MultiIndex
        hash(key)
        try:
            self.get_loc(key)
            return True
        except (LookupError, TypeError, ValueError):
            return False

    @cache_readonly
    def dtype(self) -> np.dtype:
        # Return the data type of the MultiIndex, which is object (O)
        return np.dtype("O")

    @cache_readonly
    def _is_memory_usage_qualified(self) -> bool:
        """return a boolean if we need a qualified .info display"""

        def f(level) -> bool:
            # Check if the inferred type of the level contains mixed, string, or unicode types
            return "mixed" in level or "string" in level or "unicode" in level

        # Check if any level in the MultiIndex requires qualified memory usage
        return any(f(level.inferred_type) for level in self.levels)

    # Cannot determine type of "memory_usage"
    @doc(Index.memory_usage)  # type: ignore[has-type]
    # 覆盖基类方法，计算对象的内存使用量（不包括深层对象）
    def memory_usage(self, deep: bool = False) -> int:
        # 避免在此处计算.values，以避免不必要地实现元组表示
        return self._nbytes(deep)

    @cache_readonly
    # 返回底层数据的字节数
    def nbytes(self) -> int:
        """返回底层数据的字节数"""
        return self._nbytes(False)

    # 返回底层数据的字节数
    def _nbytes(self, deep: bool = False) -> int:
        """
        返回底层数据的字节数
        如果deep=True，则深度检查数据级别

        包括引擎哈希表

        *这是一个内部例程*
        """
        # 对于没有有用的getsizeof实现（如PyPy）
        objsize = 24

        # 计算数据级别的总字节数
        level_nbytes = sum(i.memory_usage(deep=deep) for i in self.levels)
        # 计算代码级别的总字节数
        label_nbytes = sum(i.nbytes for i in self.codes)
        # 计算名称列表的总字节数
        names_nbytes = sum(getsizeof(i, objsize) for i in self.names)
        result = level_nbytes + label_nbytes + names_nbytes

        # 如果_engine已缓存，则包括引擎哈希表
        if "_engine" in self._cache:
            result += self._engine.sizeof(deep=deep)
        return result

    # --------------------------------------------------------------------
    # 渲染方法

    def _formatter_func(self, tup):
        """
        根据其级别的格式化函数格式化tup中的每个项目。
        """
        # 获取每个级别的格式化函数并应用于tup中的值
        formatter_funcs = (level._formatter_func for level in self.levels)
        return tuple(func(val) for func, val in zip(formatter_funcs, tup))

    def _get_values_for_csv(
        self, *, na_rep: str = "nan", **kwargs
    ) -> npt.NDArray[np.object_]:
        new_levels = []
        new_codes = []

        # 遍历级别并格式化它们
        for level, level_codes in zip(self.levels, self.codes):
            # 获取用于CSV的级别值，并根据需要填充nan_rep
            level_strs = level._get_values_for_csv(na_rep=na_rep, **kwargs)
            # 如果存在NaN值，添加nan_rep
            mask = level_codes == -1
            if mask.any():
                nan_index = len(level_strs)
                # numpy 1.21不再支持隐式字符串转换
                level_strs = level_strs.astype(str)
                level_strs = np.append(level_strs, na_rep)
                # 确保level_codes不可写入，需要进行复制
                assert not level_codes.flags.writeable  # 即需要复制
                level_codes = level_codes.copy()  # 使其可写
                level_codes[mask] = nan_index
            new_levels.append(level_strs)
            new_codes.append(level_codes)

        if len(new_levels) == 1:
            # 单级别多索引
            return Index(new_levels[0].take(new_codes[0]))._get_values_for_csv()
        else:
            # 重建多索引
            mi = MultiIndex(
                levels=new_levels,
                codes=new_codes,
                names=self.names,
                sortorder=self.sortorder,
                verify_integrity=False,
            )
            return mi._values
    # 定义一个方法，用于格式化多级索引的显示
    def _format_multi(
        self,
        *,
        include_names: bool,  # 是否包含名称信息
        sparsify: bool | None | lib.NoDefault,  # 是否稀疏显示的选项
        formatter: Callable | None = None,  # 自定义格式化函数，默认为None
    ) -> list:
        # 如果当前对象为空，则返回空列表
        if len(self) == 0:
            return []

        # 存储每个级别的字符串化表示
        stringified_levels = []
        # 遍历每个级别和对应的编码
        for lev, level_codes in zip(self.levels, self.codes):
            # 获取NA值的表示
            na = _get_na_rep(lev.dtype)

            # 如果级别不为空
            if len(lev) > 0:
                # 根据编码取出相应的子集
                taken = formatted = lev.take(level_codes)
                # 进行扁平化格式化，不包含名称信息
                formatted = taken._format_flat(include_name=False, formatter=formatter)

                # 检查是否有NA值存在
                mask = level_codes == -1
                if mask.any():
                    formatted = np.array(formatted, dtype=object)
                    formatted[mask] = na
                    formatted = formatted.tolist()

            else:
                # 特殊情况，所有值都是NA
                formatted = [
                    pprint_thing(na if isna(x) else x, escape_chars=("\t", "\r", "\n"))
                    for x in algos.take_nd(lev._values, level_codes)
                ]
            stringified_levels.append(formatted)

        # 存储最终结果的列表
        result_levels = []
        # 遍历每个级别的字符串化表示和对应的名称
        for lev, lev_name in zip(stringified_levels, self.names):
            level = []

            # 如果需要包含名称信息
            if include_names:
                level.append(
                    pprint_thing(lev_name, escape_chars=("\t", "\r", "\n"))
                    if lev_name is not None
                    else ""
                )

            # 将级别的字符串化表示加入到结果列表中
            level.extend(np.array(lev, dtype=object))
            result_levels.append(level)

        # 如果sparsify为None，则根据设置获取是否稀疏显示
        if sparsify is None:
            sparsify = get_option("display.multi_sparse")

        # 如果需要进行稀疏显示
        if sparsify:
            sentinel: Literal[""] | bool | lib.NoDefault = ""
            # 对sparsify的值进行断言，如果是lib.no_default，则将其作为sentinel
            assert isinstance(sparsify, bool) or sparsify is lib.no_default
            if sparsify is lib.no_default:
                sentinel = sparsify
            # 对结果级别进行稀疏化处理
            result_levels = sparsify_labels(
                result_levels, start=int(include_names), sentinel=sentinel
            )

        # 返回最终的结果级别列表
        return result_levels

# --------------------------------------------------------------------
# Names Methods

    # 返回对象的名称列表作为FrozenList对象
    def _get_names(self) -> FrozenList:
        return FrozenList(self._names)
    # 设置新的索引名称。每个名称必须是可哈希类型。
    def _set_names(self, names, *, level=None, validate: bool = True) -> None:
        """
        Set new names on index. Each name has to be a hashable type.

        Parameters
        ----------
        values : str or sequence
            name(s) to set
        level : int, level name, or sequence of int/level names (default None)
            If the index is a MultiIndex (hierarchical), level(s) to set (None
            for all levels).  Otherwise level must be None
        validate : bool, default True
            validate that the names match level lengths

        Raises
        ------
        TypeError if each name is not hashable.

        Notes
        -----
        sets names on levels. WARNING: mutates!

        Note that you generally want to set this *after* changing levels, so
        that it only acts on copies
        """
        # GH 15110
        # 不允许在 MultiIndex 中使用单个字符串作为名称
        if names is not None and not is_list_like(names):
            raise ValueError("Names should be list-like for a MultiIndex")
        names = list(names)

        if validate:
            if level is not None and len(names) != len(level):
                raise ValueError("Length of names must match length of level.")
            if level is None and len(names) != self.nlevels:
                raise ValueError(
                    "Length of names must match number of levels in MultiIndex."
                )

        if level is None:
            level = range(self.nlevels)
        else:
            level = (self._get_level_number(lev) for lev in level)

        # 设置名称
        for lev, name in zip(level, names):
            if name is not None:
                # GH 20527
                # 所有 'names' 中的项都必须是可哈希的：
                if not is_hashable(name):
                    raise TypeError(
                        f"{type(self).__name__}.name must be a hashable type"
                    )
            self._names[lev] = name

        # 如果已访问过 .levels 属性，则缓存中的名称将变得过时。
        self._reset_cache()

    # 设置属性 'names'，用于获取和设置 MultiIndex 中各级别的名称。
    names = property(
        fset=_set_names,
        fget=_get_names,
        doc="""
        Names of levels in MultiIndex.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays(
        ...     [[1, 2], [3, 4], [5, 6]], names=['x', 'y', 'z']
        ... )
        >>> mi
        MultiIndex([(1, 3, 5),
                    (2, 4, 6)],
                   names=['x', 'y', 'z'])
        >>> mi.names
        FrozenList(['x', 'y', 'z'])
        """,
    )

    # --------------------------------------------------------------------

    # 使用缓存的只读属性装饰器，返回字符串 "mixed"
    @cache_readonly
    def inferred_type(self) -> str:
        return "mixed"
    # 返回指定级别名称在列表中出现的次数
    def _get_level_number(self, level) -> int:
        count = self.names.count(level)
        # 如果名称出现多次且不是整数，则引发 ValueError 异常
        if (count > 1) and not is_integer(level):
            raise ValueError(
                f"The name {level} occurs multiple times, use a level number"
            )
        try:
            # 获取指定级别名称在列表中的索引位置
            level = self.names.index(level)
        except ValueError as err:
            # 如果不是整数且名称未找到，则引发 KeyError 异常
            if not is_integer(level):
                raise KeyError(f"Level {level} not found") from err
            # 处理负数索引的情况
            if level < 0:
                level += self.nlevels
                # 如果负数索引超出范围，则引发 IndexError 异常
                if level < 0:
                    orig_level = level - self.nlevels
                    raise IndexError(
                        f"Too many levels: Index has only {self.nlevels} levels, "
                        f"{orig_level} is not a valid level number"
                    ) from err
            # 注意：级别是从零开始计数
            elif level >= self.nlevels:
                # 如果索引超出范围，则引发 IndexError 异常
                raise IndexError(
                    f"Too many levels: Index has only {self.nlevels} levels, "
                    f"not {level + 1}"
                ) from err
        # 返回级别的索引位置
        return level

    @cache_readonly
    # 返回布尔值，指示值是否相等或递增
    def is_monotonic_increasing(self) -> bool:
        """
        Return a boolean if the values are equal or increasing.
        """
        # 如果任意一个代码列表中包含 -1，则返回 False
        if any(-1 in code for code in self.codes):
            return False

        # 如果所有级别都是单调递增的，则操作直接作用于代码列表
        if all(level.is_monotonic_increasing for level in self.levels):
            # 如果每个级别都已排序，则可以直接操作代码列表
            return libalgos.is_lexsorted(
                [x.astype("int64", copy=False) for x in self.codes]
            )

        # reversed() 是因为 lexsort() 要求最重要的键位于最后
        values = [
            self._get_level_values(i)._values for i in reversed(range(len(self.levels)))
        ]
        try:
            # 使用 np.lexsort 对值进行排序，返回排序后的索引顺序
            sort_order = np.lexsort(values)  # type: ignore[arg-type]
            return Index(sort_order).is_monotonic_increasing
        except TypeError:
            # 如果值类型混合且 np.lexsort 无法处理，则使用默认值排序
            return Index(self._values).is_monotonic_increasing

    @cache_readonly
    # 返回布尔值，指示值是否相等或递减
    def is_monotonic_decreasing(self) -> bool:
        """
        Return a boolean if the values are equal or decreasing.
        """
        # 值单调递减的条件是逆序后的值单调递增
        return self[::-1].is_monotonic_increasing

    # 文档引用：Index.duplicated
    @doc(Index.duplicated)
    def duplicated(self, keep: DropKeep = "first") -> npt.NDArray[np.bool_]:
        # 计算每个层级的长度，形成一个元组作为 shape
        shape = tuple(len(lev) for lev in self.levels)
        # 根据 shape 和 self.codes 获取分组索引 ids，不进行排序，排除空值
        ids = get_group_index(self.codes, shape, sort=False, xnull=False)

        # 调用外部函数 duplicated，返回重复项的布尔数组
        return duplicated(ids, keep)

    # 错误：无法覆盖 final 属性 "_duplicated"
    # （该属性在基类 "IndexOpsMixin" 中已声明）
    _duplicated = duplicated  # type: ignore[misc]

    def fillna(self, value):
        """
        fillna is not implemented for MultiIndex
        """
        # 对于 MultiIndex，fillna 方法未实现，抛出 NotImplementedError
        raise NotImplementedError("isna is not defined for MultiIndex")

    @doc(Index.dropna)
    def dropna(self, how: AnyAll = "any") -> MultiIndex:
        # 获取每个层级中的空值布尔数组
        nans = [level_codes == -1 for level_codes in self.codes]
        if how == "any":
            # 如果 how 为 "any"，则索引器是任意层级中存在空值的位置
            indexer = np.any(nans, axis=0)
        elif how == "all":
            # 如果 how 为 "all"，则索引器是所有层级都存在空值的位置
            indexer = np.all(nans, axis=0)
        else:
            # 如果 how 参数既不是 "any" 也不是 "all"，抛出值错误异常
            raise ValueError(f"invalid how option: {how}")

        # 根据索引器过滤后的代码，创建新的代码列表 new_codes
        new_codes = [level_codes[~indexer] for level_codes in self.codes]
        # 调用 set_codes 方法，返回一个新的 MultiIndex 实例
        return self.set_codes(codes=new_codes)

    def _get_level_values(self, level: int, unique: bool = False) -> Index:
        """
        Return vector of label values for requested level,
        equal to the length of the index

        **this is an internal method**

        Parameters
        ----------
        level : int
            要请求的层级
        unique : bool, default False
            如果为 True，则去除重复的值

        Returns
        -------
        Index
            包含请求层级的标签值的向量
        """
        # 获取指定层级的标签数组 lev 和层级代码 level_codes，以及层级名称 name
        lev = self.levels[level]
        level_codes = self.codes[level]
        name = self._names[level]
        if unique:
            # 如果 unique 为 True，则调用 algos.unique 去除重复值
            level_codes = algos.unique(level_codes)
        # 使用 algos.take_nd 方法根据 level_codes 从 lev._values 中取值填充，使用 lev._na_value 填充缺失值
        filled = algos.take_nd(lev._values, level_codes, fill_value=lev._na_value)
        # 返回浅拷贝的 lev 实例，填充后的数据，同时更新名称为指定的 name
        return lev._shallow_copy(filled, name=name)

    # 错误："get_level_values" 的签名与超类型 "Index" 不兼容
    def get_level_values(self, level) -> Index:  # type: ignore[override]
        """
        Return vector of label values for requested level.

        Length of returned vector is equal to the length of the index.

        Parameters
        ----------
        level : int or str
            ``level`` is either the integer position of the level in the
            MultiIndex, or the name of the level.

        Returns
        -------
        Index
            Values is a level of this MultiIndex converted to
            a single :class:`Index` (or subclass thereof).

        Notes
        -----
        If the level contains missing values, the result may be casted to
        ``float`` with missing values specified as ``NaN``. This is because
        the level is converted to a regular ``Index``.

        Examples
        --------
        Create a MultiIndex:

        >>> mi = pd.MultiIndex.from_arrays((list("abc"), list("def")))
        >>> mi.names = ["level_1", "level_2"]

        Get level values by supplying level as either integer or name:

        >>> mi.get_level_values(0)
        Index(['a', 'b', 'c'], dtype='object', name='level_1')
        >>> mi.get_level_values("level_2")
        Index(['d', 'e', 'f'], dtype='object', name='level_2')

        If a level contains missing values, the return type of the level
        may be cast to ``float``.

        >>> pd.MultiIndex.from_arrays([[1, None, 2], [3, 4, 5]]).dtypes
        level_0    int64
        level_1    int64
        dtype: object
        >>> pd.MultiIndex.from_arrays([[1, None, 2], [3, 4, 5]]).get_level_values(0)
        Index([1.0, nan, 2.0], dtype='float64')
        """
        # 获取要操作的级别的编号
        level = self._get_level_number(level)
        # 获取指定级别的值
        values = self._get_level_values(level)
        # 返回值作为 Index 对象
        return values

    @doc(Index.unique)
    def unique(self, level=None):
        """
        Return unique values at the requested level.

        If `level` is None, returns unique values of the entire MultiIndex.

        Parameters
        ----------
        level : int or str, optional
            If provided, specifies the level to compute unique values for.
            Can be the integer position or name of the level.

        Returns
        -------
        Index
            Unique values of the specified level as an :class:`Index`.

        Notes
        -----
        This method delegates to ``self.drop_duplicates()`` when `level` is None.

        Examples
        --------
        Create a MultiIndex:

        >>> mi = pd.MultiIndex.from_arrays((list("abc"), list("def")))
        >>> mi.names = ["level_1", "level_2"]

        Get unique values at a specific level:

        >>> mi.unique(level=0)
        Index(['a', 'b', 'c'], dtype='object', name='level_1')
        >>> mi.unique(level="level_2")
        Index(['d', 'e', 'f'], dtype='object', name='level_2')

        See Also
        --------
        Index.drop_duplicates : Return Index with duplicate elements removed.
        """
        if level is None:
            # 如果 level 为 None，则返回整个 MultiIndex 的唯一值
            return self.drop_duplicates()
        else:
            # 否则，获取指定级别的编号
            level = self._get_level_number(level)
            # 返回指定级别的唯一值
            return self._get_level_values(level=level, unique=True)

    def to_frame(
        self,
        index: bool = True,
        name=lib.no_default,
        allow_duplicates: bool = False,
        ):
        """
        Convert MultiIndex to DataFrame.

        Parameters
        ----------
        index : bool, default True
            Whether to include the MultiIndex values as columns in the DataFrame.
        name : object, optional
            Name of the resulting DataFrame, if None, derived from MultiIndex.
        allow_duplicates : bool, default False
            Whether to allow duplicate values. If False, an error is raised if
            the resulting DataFrame would have duplicate columns or index.

        Returns
        -------
        DataFrame
            DataFrame representation of the MultiIndex.

        Notes
        -----
        If `index` is False, only the data values are returned without the MultiIndex
        structure.

        Examples
        --------
        Create a MultiIndex:

        >>> mi = pd.MultiIndex.from_arrays((list("abc"), list("def")))
        >>> mi.names = ["level_1", "level_2"]

        Convert MultiIndex to a DataFrame:

        >>> mi.to_frame()
                    level_1 level_2
        a   d
        b   e
        c   f

        See Also
        --------
        DataFrame.set_index : Set the DataFrame index using one or more existing columns.
        """
    ) -> DataFrame:
        """
        Create a DataFrame with the levels of the MultiIndex as columns.

        Column ordering is determined by the DataFrame constructor with data as
        a dict.

        Parameters
        ----------
        index : bool, default True
            Set the index of the returned DataFrame as the original MultiIndex.

        name : list / sequence of str, optional
            The passed names should substitute index level names.

        allow_duplicates : bool, optional default False
            Allow duplicate column labels to be created.

            .. versionadded:: 1.5.0

        Returns
        -------
        DataFrame
            The DataFrame object containing MultiIndex levels as columns.

        See Also
        --------
        DataFrame : Two-dimensional, size-mutable, potentially heterogeneous
            tabular data.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([["a", "b"], ["c", "d"]])
        >>> mi
        MultiIndex([('a', 'c'),
                    ('b', 'd')],
                   )

        >>> df = mi.to_frame()
        >>> df
             0  1
        a c  a  c
        b d  b  d

        >>> df = mi.to_frame(index=False)
        >>> df
           0  1
        0  a  c
        1  b  d

        >>> df = mi.to_frame(name=["x", "y"])
        >>> df
             x  y
        a c  a  c
        b d  b  d
        """
        from pandas import DataFrame

        if name is not lib.no_default:
            if not is_list_like(name):
                raise TypeError("'name' must be a list / sequence of column names.")

            if len(name) != len(self.levels):
                raise ValueError(
                    "'name' should have same length as number of levels on index."
                )
            idx_names = name
        else:
            idx_names = self._get_level_names()

        if not allow_duplicates and len(set(idx_names)) != len(idx_names):
            raise ValueError(
                "Cannot create duplicate column labels if allow_duplicates is False"
            )

        # Guarantee resulting column order - PY36+ dict maintains insertion order
        # 创建一个 DataFrame 对象，其中每个列对应 MultiIndex 的一个级别值
        result = DataFrame(
            {level: self._get_level_values(level) for level in range(len(self.levels))},
            copy=False,
        )
        # 将列名设置为传入的 idx_names
        result.columns = idx_names

        if index:
            # 如果 index 参数为 True，则将结果的索引设置为原始的 MultiIndex
            result.index = self
        return result

    # error: Return type "Index" of "to_flat_index" incompatible with return type
    # "MultiIndex" in supertype "Index"
    def to_flat_index(self) -> Index:  # type: ignore[override]
        """
        Convert a MultiIndex to an Index of Tuples containing the level values.

        Returns
        -------
        pd.Index
            Index with the MultiIndex data represented in Tuples.

        See Also
        --------
        MultiIndex.from_tuples : Convert flat index back to MultiIndex.

        Notes
        -----
        This method will simply return the caller if called by anything other
        than a MultiIndex.

        Examples
        --------
        >>> index = pd.MultiIndex.from_product(
        ...     [["foo", "bar"], ["baz", "qux"]], names=["a", "b"]
        ... )
        >>> index.to_flat_index()
        Index([('foo', 'baz'), ('foo', 'qux'),
               ('bar', 'baz'), ('bar', 'qux')],
              dtype='object')
        """
        # 将多级索引转换为包含级别值元组的索引
        return Index(self._values, tupleize_cols=False)

    def _is_lexsorted(self) -> bool:
        """
        Return True if the codes are lexicographically sorted.

        Returns
        -------
        bool

        Examples
        --------
        In the below examples, the first level of the MultiIndex is sorted because
        a<b<c, so there is no need to look at the next level.

        >>> pd.MultiIndex.from_arrays(
        ...     [["a", "b", "c"], ["d", "e", "f"]]
        ... )._is_lexsorted()
        True
        >>> pd.MultiIndex.from_arrays(
        ...     [["a", "b", "c"], ["d", "f", "e"]]
        ... )._is_lexsorted()
        True

        In case there is a tie, the lexicographical sorting looks
        at the next level of the MultiIndex.

        >>> pd.MultiIndex.from_arrays([[0, 1, 1], ["a", "b", "c"]])._is_lexsorted()
        True
        >>> pd.MultiIndex.from_arrays([[0, 1, 1], ["a", "c", "b"]])._is_lexsorted()
        False
        >>> pd.MultiIndex.from_arrays(
        ...     [["a", "a", "b", "b"], ["aa", "bb", "aa", "bb"]]
        ... )._is_lexsorted()
        True
        >>> pd.MultiIndex.from_arrays(
        ...     [["a", "a", "b", "b"], ["bb", "aa", "aa", "bb"]]
        ... )._is_lexsorted()
        False
        """
        # 如果代码在字典序上排序，则返回True
        return self._lexsort_depth == self.nlevels

    @cache_readonly
    def _lexsort_depth(self) -> int:
        """
        Compute and return the lexsort_depth, the number of levels of the
        MultiIndex that are sorted lexically

        Returns
        -------
        int
        """
        # 计算并返回字典序排序的深度，即按字典序排序的多级索引的级别数
        if self.sortorder is not None:
            return self.sortorder
        return _lexsort_depth(self.codes, self.nlevels)
    def _sort_levels_monotonic(self, raise_if_incomparable: bool = False) -> MultiIndex:
        """
        This is an *internal* function.

        Create a new MultiIndex from the current to monotonically sorted
        items IN the levels. This does not actually make the entire MultiIndex
        monotonic, JUST the levels.

        The resulting MultiIndex will have the same outward
        appearance, meaning the same .values and ordering. It will also
        be .equals() to the original.

        Returns
        -------
        MultiIndex
            A new MultiIndex object with levels sorted in a monotonic order.

        Examples
        --------
        >>> mi = pd.MultiIndex(
        ...     levels=[["a", "b"], ["bb", "aa"]], codes=[[0, 0, 1, 1], [0, 1, 0, 1]]
        ... )
        >>> mi
        MultiIndex([('a', 'bb'),
                    ('a', 'aa'),
                    ('b', 'bb'),
                    ('b', 'aa')],
                   )

        >>> mi.sort_values()
        MultiIndex([('a', 'aa'),
                    ('a', 'bb'),
                    ('b', 'aa'),
                    ('b', 'bb')],
                   )
        """
        # 如果已经按字典序排序且是单调递增的，直接返回当前对象
        if self._is_lexsorted() and self.is_monotonic_increasing:
            return self

        # 创建新的 levels 和 codes 列表用于存储排序后的数据
        new_levels = []
        new_codes = []

        # 遍历当前 MultiIndex 的 levels 和 codes
        for lev, level_codes in zip(self.levels, self.codes):
            # 如果当前 level 不是单调递增的
            if not lev.is_monotonic_increasing:
                try:
                    # 使用 lev.argsort() 获取排序后的索引
                    indexer = lev.argsort()
                except TypeError:
                    # 如果 lev.argsort() 抛出 TypeError
                    if raise_if_incomparable:
                        raise
                else:
                    # 使用排序后的索引重新排列 lev
                    lev = lev.take(indexer)

                    # 确保 indexer 是平台整数类型，并获取逆向索引
                    indexer = ensure_platform_int(indexer)
                    ri = lib.get_reverse_indexer(indexer, len(indexer))
                    # 使用逆向索引重排 level_codes
                    level_codes = algos.take_nd(ri, level_codes, fill_value=-1)

            # 将更新后的 lev 和 level_codes 添加到新的 levels 和 codes 中
            new_levels.append(lev)
            new_codes.append(level_codes)

        # 返回一个新的 MultiIndex 对象，其中包含排序后的 levels 和 codes
        return MultiIndex(
            new_levels,
            new_codes,
            names=self.names,
            sortorder=self.sortorder,
            verify_integrity=False,
        )
    def remove_unused_levels(self) -> MultiIndex:
        """
        Create new MultiIndex from current that removes unused levels.

        Unused level(s) means levels that are not expressed in the
        labels. The resulting MultiIndex will have the same outward
        appearance, meaning the same .values and ordering. It will
        also be .equals() to the original.

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex.from_product([range(2), list("ab")])
        >>> mi
        MultiIndex([(0, 'a'),
                    (0, 'b'),
                    (1, 'a'),
                    (1, 'b')],
                   )

        >>> mi[2:]
        MultiIndex([(1, 'a'),
                    (1, 'b')],
                   )

        The 0 from the first level is not represented
        and can be removed

        >>> mi2 = mi[2:].remove_unused_levels()
        >>> mi2.levels
        FrozenList([[1], ['a', 'b']])
        """

        # Initialize new levels and codes
        new_levels = []
        new_codes = []

        # Flag to track if any changes are made
        changed = False

        # Iterate through each level and its corresponding codes
        for lev, level_codes in zip(self.levels, self.codes):
            # Calculate unique values in level_codes using bincount
            uniques = np.where(np.bincount(level_codes + 1) > 0)[0] - 1
            # Check if there are NaN values in uniques
            has_na = int(len(uniques) and (uniques[0] == -1))

            # Check if the number of unique values matches the length of lev
            if len(uniques) != len(lev) + has_na:
                if lev.isna().any() and len(uniques) == len(lev):
                    break

                # Set changed flag to True since there are unused levels
                changed = True

                # Calculate unique values preserving order
                uniques = algos.unique(level_codes)

                # Adjust order of -1 (NaN) if present
                if has_na:
                    na_idx = np.where(uniques == -1)[0]
                    uniques[[0, na_idx[0]]] = uniques[[na_idx[0], 0]]

                # Map codes to new indices based on unique values
                code_mapping = np.zeros(len(lev) + has_na)
                code_mapping[uniques] = np.arange(len(uniques)) - has_na
                level_codes = code_mapping[level_codes]

                # Trim lev to only include used levels
                lev = lev.take(uniques[has_na:])

            # Append modified level and codes to new_levels and new_codes
            new_levels.append(lev)
            new_codes.append(level_codes)

        # Create a view of the current MultiIndex
        result = self.view()

        # If changes were made, reset identity and update levels and codes
        if changed:
            result._reset_identity()
            result._set_levels(new_levels, validate=False)
            result._set_codes(new_codes, validate=False)

        return result
    # --------------------------------------------------------------------
    # Pickling Methods

    def __reduce__(self):
        """Necessary for making this object picklable"""
        # 将当前对象转换为可序列化的字典形式，以便进行序列化操作
        d = {
            "levels": list(self.levels),    # 将 levels 属性转换为列表
            "codes": list(self.codes),      # 将 codes 属性转换为列表
            "sortorder": self.sortorder,    # 保留 sortorder 属性的当前值
            "names": list(self.names),      # 将 names 属性转换为列表
        }
        # 返回用于反序列化对象的元组，包括对象的构造函数和状态信息字典
        return ibase._new_Index, (type(self), d), None

    # --------------------------------------------------------------------

    def __getitem__(self, key):
        if is_scalar(key):
            key = com.cast_scalar_indexer(key)

            retval = []
            # 遍历 levels 和 codes 属性，根据 key 获取相应的值并组成元组返回
            for lev, level_codes in zip(self.levels, self.codes):
                if level_codes[key] == -1:
                    retval.append(np.nan)   # 如果索引值为 -1，则返回 NaN
                else:
                    retval.append(lev[level_codes[key]])   # 根据索引值获取对应的 level 值

            return tuple(retval)
        else:
            # 如果 key 不是标量，根据不同类型的索引进行处理
            sortorder = None
            if com.is_bool_indexer(key):
                key = np.asarray(key, dtype=bool)
                sortorder = self.sortorder   # 使用当前对象的 sortorder 属性
            elif isinstance(key, slice):
                if key.step is None or key.step > 0:
                    sortorder = self.sortorder   # 使用当前对象的 sortorder 属性
            elif isinstance(key, Index):
                key = np.asarray(key)

            # 根据 key 获取新的 codes 列表，创建一个新的 MultiIndex 对象并返回
            new_codes = [level_codes[key] for level_codes in self.codes]

            return MultiIndex(
                levels=self.levels,
                codes=new_codes,
                names=self.names,
                sortorder=sortorder,
                verify_integrity=False,
            )

    def _getitem_slice(self: MultiIndex, slobj: slice) -> MultiIndex:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        sortorder = None
        if slobj.step is None or slobj.step > 0:
            sortorder = self.sortorder   # 根据 slice 的 step 属性设置 sortorder

        # 根据 slice 对象 slobj 获取新的 codes 列表，创建一个新的 MultiIndex 对象并返回
        new_codes = [level_codes[slobj] for level_codes in self.codes]

        return type(self)(
            levels=self.levels,
            codes=new_codes,
            names=self._names,   # 使用当前对象的 _names 属性
            sortorder=sortorder,
            verify_integrity=False,
        )

    @Appender(_index_shared_docs["take"] % _index_doc_kwargs)
    def take(
        self: MultiIndex,
        indices,
        axis: Axis = 0,
        allow_fill: bool = True,
        fill_value=None,
        **kwargs,
        # 参考 take 方法的文档字符串中的参数说明
        nv.validate_take((), kwargs)
        # 使用 nv 对象的 validate_take 方法验证空元组 () 和关键字参数 kwargs
        indices = ensure_platform_int(indices)
        # 将 indices 转换为平台整数类型

        # 只有在传递了非 None 的 fill_value 时才进行填充操作
        allow_fill = self._maybe_disallow_fill(allow_fill, fill_value, indices)

        # 如果 indices 的维度为 1 并且是一个范围索引器，则返回当前对象的副本
        if indices.ndim == 1 and lib.is_range_indexer(indices, len(self)):
            return self.copy()

        na_value = -1

        # 对每个 self.codes 中的标签按照 indices 进行取值操作，形成列表 taken
        taken = [lab.take(indices) for lab in self.codes]

        # 如果允许填充操作
        if allow_fill:
            # 创建一个表示是否为 -1 的掩码
            mask = indices == -1
            if mask.any():
                masked = []
                # 遍历 taken 中的每个新标签
                for new_label in taken:
                    label_values = new_label
                    # 将掩码位置的值设为 na_value
                    label_values[mask] = na_value
                    masked.append(np.asarray(label_values))
                taken = masked

        # 返回一个新的 MultiIndex 对象，使用 taken 作为 codes 参数
        return MultiIndex(
            levels=self.levels, codes=taken, names=self.names, verify_integrity=False
        )

    def append(self, other):
        """
        Append a collection of Index options together.

        Parameters
        ----------
        other : Index or list/tuple of indices

        Returns
        -------
        Index
            The combined index.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([["a"], ["b"]])
        >>> mi
        MultiIndex([('a', 'b')],
                   )
        >>> mi.append(mi)
        MultiIndex([('a', 'b'), ('a', 'b')],
                   )
        """
        # 如果 other 不是 Index 或 indices 的列表/元组，则将其转换为列表
        if not isinstance(other, (list, tuple)):
            other = [other]

        # 如果 other 中所有元素都是 MultiIndex 并且其 nlevels 大于等于当前对象的 nlevels
        if all(
            (isinstance(o, MultiIndex) and o.nlevels >= self.nlevels) for o in other
        ):
            codes = []
            levels = []
            names = []
            # 遍历当前对象的各级别
            for i in range(self.nlevels):
                level_values = self.levels[i]
                # 合并 other 中每个 MultiIndex 对象的各级别的值
                for mi in other:
                    level_values = level_values.union(mi.levels[i])
                # 对每个级别的 codes 执行重新编码，以适应合并后的 level_values
                level_codes = [
                    recode_for_categories(
                        mi.codes[i], mi.levels[i], level_values, copy=False
                    )
                    for mi in ([self, *other])
                ]
                level_name = self.names[i]
                # 如果 other 中有任何 MultiIndex 对象的级别名称与当前对象不同，则设为 None
                if any(mi.names[i] != level_name for mi in other):
                    level_name = None
                codes.append(np.concatenate(level_codes))
                levels.append(level_values)
                names.append(level_name)
            # 返回一个新的 MultiIndex 对象，合并后的 codes、levels 和 names 作为参数
            return MultiIndex(
                codes=codes, levels=levels, names=names, verify_integrity=False
            )

        # 如果 other 中所有元素都是 Index 类型，则执行下列操作
        to_concat = (self._values,) + tuple(k._values for k in other)
        new_tuples = np.concatenate(to_concat)

        # 尝试从新的元组数组创建一个新的 MultiIndex 对象
        try:
            return MultiIndex.from_tuples(new_tuples)
        # 如果出现 TypeError 或 IndexError 异常，则创建一个新的 Index 对象
        except (TypeError, IndexError):
            return Index(new_tuples)
    # 返回按照特定顺序排序的索引数组
    def argsort(
        self, *args, na_position: str = "last", **kwargs
    ) -> npt.NDArray[np.intp]:
        # 获得经过排序后的目标 MultiIndex 对象，如果存在不可比较的情况则抛出异常
        target = self._sort_levels_monotonic(raise_if_incomparable=True)
        # 获取用于排序的各级别的编码数组
        keys = [lev.codes for lev in target._get_codes_for_sorting()]
        # 使用 lexsort_indexer 函数对编码数组进行排序，并返回排序后的索引数组
        return lexsort_indexer(keys, na_position=na_position, codes_given=True)

    @Appender(_index_shared_docs["repeat"] % _index_doc_kwargs)
    def repeat(self, repeats: int, axis=None) -> MultiIndex:
        # 验证重复次数参数的有效性
        nv.validate_repeat((), {"axis": axis})
        # 确保重复次数为平台整数类型，忽略类型检查错误
        repeats = ensure_platform_int(repeats)  # type: ignore[assignment]
        # 创建一个新的 MultiIndex 对象，其中各级别的编码数组被重复指定次数
        return MultiIndex(
            levels=self.levels,
            codes=[
                level_codes.view(np.ndarray).astype(np.intp, copy=False).repeat(repeats)
                for level_codes in self.codes
            ],
            names=self.names,
            sortorder=self.sortorder,
            verify_integrity=False,
        )

    # Signature of "drop" incompatible with supertype "Index" 的错误提示
    def drop(  # type: ignore[override]
        self,
        codes,
        level: Index | np.ndarray | Iterable[Hashable] | None = None,
        errors: IgnoreRaise = "raise",
    ):
        # 将索引标签转换为数组
        codes = com.index_labels_to_array(codes)
        # 获取指定级别的编号
        i = self._get_level_number(level)
        # 获取该级别的索引对象
        index = self.levels[i]
        # 获取要从索引中删除的值的索引
        values = index.get_indexer(codes)
        
        # 如果需要删除 NaN，则值为 -1。检查哪些值不是 NaN 且等于 -1，这意味着它们在索引中缺失
        nan_codes = isna(codes)
        values[(np.equal(nan_codes, False)) & (values == -1)] = -2
        
        # 如果索引的长度等于 MultiIndex 的长度，则表示没有找到对应的值
        if index.shape[0] == self.shape[0]:
            values[np.equal(nan_codes, True)] = -2
        
        # 检查在索引中未找到的标签，并根据 errors 参数决定是否抛出 KeyError
        not_found = codes[values == -2]
        if len(not_found) != 0 and errors != "ignore":
            raise KeyError(f"labels {not_found} not found in level")
        
        # 生成一个掩码，用于从 MultiIndex 中移除指定的值
        mask = ~algos.isin(self.codes[i], values)
        
        # 返回根据掩码移除指定值后的新 MultiIndex 对象
        return self[mask]
    def swaplevel(self, i=-2, j=-1) -> MultiIndex:
        """
        Swap level i with level j.

        Calling this method does not change the ordering of the values.

        Parameters
        ----------
        i : int, str, default -2
            First level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.
        j : int, str, default -1
            Second level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.

        Returns
        -------
        MultiIndex
            A new MultiIndex.

        See Also
        --------
        Series.swaplevel : Swap levels i and j in a MultiIndex.
        DataFrame.swaplevel : Swap levels i and j in a MultiIndex on a
            particular axis.

        Examples
        --------
        >>> mi = pd.MultiIndex(
        ...     levels=[["a", "b"], ["bb", "aa"]], codes=[[0, 0, 1, 1], [0, 1, 0, 1]]
        ... )
        >>> mi
        MultiIndex([('a', 'bb'),
                    ('a', 'aa'),
                    ('b', 'bb'),
                    ('b', 'aa')],
                   )
        >>> mi.swaplevel(0, 1)
        MultiIndex([('bb', 'a'),
                    ('aa', 'a'),
                    ('bb', 'b'),
                    ('aa', 'b')],
                   )
        """
        # 创建新的列表，复制当前的 levels、codes、names
        new_levels = list(self.levels)
        new_codes = list(self.codes)
        new_names = list(self.names)

        # 获取要交换的级别 i 和 j 的序号
        i = self._get_level_number(i)
        j = self._get_level_number(j)

        # 交换 levels 中的 i 和 j 的位置
        new_levels[i], new_levels[j] = new_levels[j], new_levels[i]
        # 交换 codes 中的 i 和 j 的位置
        new_codes[i], new_codes[j] = new_codes[j], new_codes[i]
        # 交换 names 中的 i 和 j 的位置
        new_names[i], new_names[j] = new_names[j], new_names[i]

        # 返回一个新的 MultiIndex 对象，使用新的 levels、codes、names，关闭完整性检查
        return MultiIndex(
            levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
        )

    def reorder_levels(self, order) -> MultiIndex:
        """
        Rearrange levels using input order. May not drop or duplicate levels.

        Parameters
        ----------
        order : list of int or list of str
            List representing new level order. Reference level by number
            (position) or by key (label).

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([[1, 2], [3, 4]], names=["x", "y"])
        >>> mi
        MultiIndex([(1, 3),
                    (2, 4)],
                   names=['x', 'y'])

        >>> mi.reorder_levels(order=[1, 0])
        MultiIndex([(3, 1),
                    (4, 2)],
                   names=['y', 'x'])

        >>> mi.reorder_levels(order=["y", "x"])
        MultiIndex([(3, 1),
                    (4, 2)],
                   names=['y', 'x'])
        """
        # 将 order 中的级别标识符转换为其对应的序号
        order = [self._get_level_number(i) for i in order]
        # 使用给定的顺序重新排列级别
        result = self._reorder_ilevels(order)
        # 返回重新排列后的 MultiIndex 结果
        return result
    def _reorder_ilevels(self, order) -> MultiIndex:
        # 检查给定的排序顺序列表长度是否与当前多级索引的级数相同
        if len(order) != self.nlevels:
            raise AssertionError(
                f"Length of order must be same as number of levels ({self.nlevels}), "
                f"got {len(order)}"
            )
        # 根据给定的顺序重新排列当前多级索引的级别、代码和名称
        new_levels = [self.levels[i] for i in order]
        new_codes = [self.codes[i] for i in order]
        new_names = [self.names[i] for i in order]

        # 返回一个新的 MultiIndex 对象，其中包含重新排列后的级别、代码和名称
        return MultiIndex(
            levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
        )

    def _recode_for_new_levels(
        self, new_levels, copy: bool = True
    ) -> Generator[np.ndarray, None, None]:
        # 检查新级别列表的长度是否不超过当前多级索引的级数
        if len(new_levels) > self.nlevels:
            raise AssertionError(
                f"Length of new_levels ({len(new_levels)}) "
                f"must be <= self.nlevels ({self.nlevels})"
            )
        # 使用 recode_for_categories 函数为新级别重新编码当前索引的代码数组
        for i in range(len(new_levels)):
            yield recode_for_categories(
                self.codes[i], self.levels[i], new_levels[i], copy=copy
            )

    def _get_codes_for_sorting(self) -> list[Categorical]:
        """
        准备用于排序的代码列表，按照所有可用的类别（包括未观察到的）进行分类，
        排除任何缺失的类别（-1），这是为了确保排序时能够正确处理 -1 不是有效值的情况
        """

        def cats(level_codes):
            return np.arange(
                np.array(level_codes).max() + 1 if len(level_codes) else 0,
                dtype=level_codes.dtype,
            )

        # 返回一个 Categorical 对象的列表，每个对象表示当前多级索引的一个级别的分类结果
        return [
            Categorical.from_codes(level_codes, cats(level_codes), True, validate=False)
            for level_codes in self.codes
        ]

    def sortlevel(
        self,
        level: IndexLabel = 0,
        ascending: bool | list[bool] = True,
        sort_remaining: bool = True,
        na_position: str = "first",
    ):
        # 此方法尚未完全定义，需要进一步添加代码以实现索引方法

    def _wrap_reindex_result(self, target, indexer, preserve_names: bool):
        # 如果目标不是 MultiIndex 类型，则根据条件转换目标索引
        if not isinstance(target, MultiIndex):
            if indexer is None:
                target = self
            elif (indexer >= 0).all():
                target = self.take(indexer)
            else:
                try:
                    target = MultiIndex.from_tuples(target)
                except TypeError:
                    # 如果不能全部转换为元组，则返回原始的目标对象
                    return target

        # 根据条件决定是否保留索引的名称，并返回处理后的目标对象
        target = self._maybe_preserve_names(target, preserve_names)
        return target

    def _maybe_preserve_names(self, target: IndexT, preserve_names: bool) -> IndexT:
        # 如果条件允许并且目标索引与当前索引级数相同但名称不同，则复制目标索引并修改名称为当前索引的名称
        if (
            preserve_names
            and target.nlevels == self.nlevels
            and target.names != self.names
        ):
            target = target.copy(deep=False)
            target.names = self.names
        return target

    # --------------------------------------------------------------------
    # Indexing Methods
    def _check_indexing_error(self, key) -> None:
        # 检查索引错误，如果键不可散列或者是迭代器，则抛出无效索引错误
        if not is_hashable(key) or is_iterator(key):
            # 允许元组作为键，前提是元组可散列；其他索引子类要求标量
            # 需要明确排除生成器，因为生成器是可散列的
            raise InvalidIndexError(key)

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        """
        是否将整数键视为位置索引？
        """
        # GH#33355
        # 返回第一个级别的对象是否应该回退到位置索引
        return self.levels[0]._should_fallback_to_positional

    def _get_indexer_strict(
        self, key, axis_name: str
    ) -> tuple[Index, npt.NDArray[np.intp]]:
        # 获取严格索引器
        keyarr = key
        if not isinstance(keyarr, Index):
            # 将键转换为安全的元组形式数组
            keyarr = com.asarray_tuplesafe(keyarr)

        if len(keyarr) and not isinstance(keyarr[0], tuple):
            # 如果键数组非空且第一个元素不是元组，则获取第一级别的索引器
            indexer = self._get_indexer_level_0(keyarr)

            # 如果索引器中有缺失的位置，抛出键错误
            self._raise_if_missing(key, indexer, axis_name)
            return self[indexer], indexer

        # 否则调用父类方法获取索引器
        return super()._get_indexer_strict(key, axis_name)

    def _raise_if_missing(self, key, indexer, axis_name: str) -> None:
        # 如果键不是索引对象，则将键转换为安全的元组形式数组
        keyarr = key
        if not isinstance(key, Index):
            keyarr = com.asarray_tuplesafe(key)

        if len(keyarr) and not isinstance(keyarr[0], tuple):
            # 即MultiIndex._get_indexer_strict的特殊情况条件

            # 创建缺失位置的布尔掩码
            mask = indexer == -1
            if mask.any():
                # 检查第一个级别的索引器是否包含键数组中的索引
                check = self.levels[0].get_indexer(keyarr)
                cmask = check == -1
                if cmask.any():
                    # 如果检查中有缺失的索引，抛出键错误
                    raise KeyError(f"{keyarr[cmask]} not in index")
                # 当级别仍包含实际上不再在索引中的值时，到达此处
                raise KeyError(f"{keyarr} not in index")
        else:
            # 否则调用父类方法抛出缺失键错误
            return super()._raise_if_missing(key, indexer, axis_name)

    def _get_indexer_level_0(self, target) -> npt.NDArray[np.intp]:
        """
        优化版本的`self.get_level_values(0).get_indexer_for(target)`。
        """
        # 获取第0级别的索引器
        lev = self.levels[0]
        codes = self._codes[0]
        # 从编码创建分类变量
        cat = Categorical.from_codes(codes=codes, categories=lev, validate=False)
        ci = Index(cat)
        # 返回目标的索引器
        return ci.get_indexer_for(target)

    def get_slice_bound(
        self,
        label: Hashable | Sequence[Hashable],
        side: Literal["left", "right"],
    ) -> int:
        """
        根据有序的 MultiIndex，计算对应于给定标签的切片边界位置。

        返回给定标签的最左边位置（如果 side=='right'，则是最右边位置的下一个）。

        Parameters
        ----------
        label : object or tuple of objects
            标签或标签元组
        side : {'left', 'right'}
            边界位置，是左边界还是右边界

        Returns
        -------
        int
            标签的索引位置

        Notes
        -----
        仅当 MultiIndex 的第 0 级索引是按词典顺序排列时，此方法才有效。

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list("abbc"), list("gefd")])

        获取第一级中最左边的 'b' 开始直到 MultiIndex 的末尾位置：

        >>> mi.get_slice_bound("b", side="left")
        1

        类似上述，但如果获取第一级中最右边的 'b' 和第二级中的 'f' 的位置：

        >>> mi.get_slice_bound(("b", "f"), side="right")
        3

        See Also
        --------
        MultiIndex.get_loc : 获取标签或标签元组的位置。
        MultiIndex.get_locs : 获取标签、切片、列表、掩码或其序列的位置。
        """
        if not isinstance(label, tuple):
            label = (label,)
        # 调用内部方法 _partial_tup_index 处理标签元组的部分索引，根据给定的 side 返回结果
        return self._partial_tup_index(label, side=side)
    # 定义一个方法 slice_locs，用于计算有序 MultiIndex 中输入标签的切片位置
    def slice_locs(self, start=None, end=None, step=None) -> tuple[int, int]:
        """
        For an ordered MultiIndex, compute the slice locations for input
        labels.

        The input labels can be tuples representing partial levels, e.g. for a
        MultiIndex with 3 levels, you can pass a single value (corresponding to
        the first level), or a 1-, 2-, or 3-tuple.

        Parameters
        ----------
        start : label or tuple, default None
            If None, defaults to the beginning
        end : label or tuple
            If None, defaults to the end
        step : int or None
            Slice step

        Returns
        -------
        (start, end) : (int, int)

        Notes
        -----
        This method only works if the MultiIndex is properly lexsorted. So,
        if only the first 2 levels of a 3-level MultiIndex are lexsorted,
        you can only pass two levels to ``.slice_locs``.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays(
        ...     [list("abbd"), list("deff")], names=["A", "B"]
        ... )

        Get the slice locations from the beginning of 'b' in the first level
        until the end of the multiindex:

        >>> mi.slice_locs(start="b")
        (1, 4)

        Like above, but stop at the end of 'b' in the first level and 'f' in
        the second level:

        >>> mi.slice_locs(start="b", end=("b", "f"))
        (1, 3)

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.
        """
        # 调用父类的 slice_locs 方法来获取切片的起始位置和结束位置
        return super().slice_locs(start, end, step)
    # 定义一个方法，用于在多级索引中查找部分元组的位置索引
    def _partial_tup_index(self, tup: tuple, side: Literal["left", "right"] = "left"):
        # 如果元组长度大于多级索引的排序深度，则引发未排序索引错误
        if len(tup) > self._lexsort_depth:
            raise UnsortedIndexError(
                f"Key length ({len(tup)}) was greater than MultiIndex lexsort depth "
                f"({self._lexsort_depth})"
            )

        # 初始化变量 n 为元组的长度
        n = len(tup)
        # 初始化搜索范围的起始和结束位置
        start, end = 0, len(self)
        # 使用 zip 函数将元组 tup 与 MultiIndex 的 levels 和 codes 组合成三元组
        zipped = zip(tup, self.levels, self.codes)
        # 遍历 zipped 中的每个元素，k 为索引，lab 为元组中的元素，lev 为 levels 中的标签，level_codes 为 codes 中的数据
        for k, (lab, lev, level_codes) in enumerate(zipped):
            # 获取当前层级范围内的 section
            section = level_codes[start:end]

            # 定义 loc 变量，用于存储位置索引
            loc: npt.NDArray[np.intp] | np.intp | int
            # 如果 lab 不在 lev 中且不是 NaN 值
            if lab not in lev and not isna(lab):
                # 短路返回，尝试在 lev 中搜索 lab 的位置
                try:
                    loc = algos.searchsorted(lev, lab, side=side)
                except TypeError as err:
                    # 类型不匹配错误，例如 test_slice_locs_with_type_mismatch
                    raise TypeError(f"Level type mismatch: {lab}") from err
                # 如果 loc 不是整数类型
                if not is_integer(loc):
                    # 类型不匹配错误，例如 test_groupby_example
                    raise TypeError(f"Level type mismatch: {lab}")
                # 如果 side 是 "right" 并且 loc 大于等于 0，则 loc 减一
                if side == "right" and loc >= 0:
                    loc -= 1
                # 返回 start 加上 section 中 loc 的搜索结果的索引位置
                return start + algos.searchsorted(section, loc, side=side)

            # 获取单个层级索引的位置 idx
            idx = self._get_loc_single_level_index(lev, lab)
            # 如果 idx 是切片类型并且 k 小于 n - 1
            if isinstance(idx, slice) and k < n - 1:
                # 从切片中获取起始和结束值，当输入为非整数间隔时需要，例如 GH#37707
                start = idx.start
                end = idx.stop
            elif k < n - 1:
                # 错误：赋值时不兼容的类型（表达式类型为 "Union[ndarray[Any, dtype[signedinteger[Any]]]）
                # 使用 algos.searchsorted 在 section 中查找 idx 的右侧位置，更新 end
                end = start + algos.searchsorted(  # type: ignore[assignment]
                    section, idx, side="right"
                )
                # 错误：赋值时不兼容的类型（表达式类型为 "Union[ndarray[Any, dtype[signedinteger[Any]]]）
                # 使用 algos.searchsorted 在 section 中查找 idx 的左侧位置，更新 start
                start = start + algos.searchsorted(  # type: ignore[assignment]
                    section, idx, side="left"
                )
            elif isinstance(idx, slice):
                # 如果 idx 是切片类型，则获取其起始值
                idx = idx.start
                # 返回 start 加上 section 中 idx 的搜索结果的索引位置
                return start + algos.searchsorted(section, idx, side=side)
            else:
                # 返回 start 加上 section 中 idx 的搜索结果的索引位置
                return start + algos.searchsorted(section, idx, side=side)
    def _get_loc_single_level_index(self, level_index: Index, key: Hashable) -> int:
        """
        If key is NA value, location of index unify as -1.

        Parameters
        ----------
        level_index: Index
            The index to search within.
        key : label
            The label whose location is to be found.

        Returns
        -------
        loc : int
            If key is NA value, loc is -1
            Else, location of key in index.

        See Also
        --------
        Index.get_loc : The get_loc method for (single-level) index.
        """
        if is_scalar(key) and isna(key):
            # TODO: need is_valid_na_for_dtype(key, level_index.dtype)
            # If key is a scalar NA value, return -1
            return -1
        else:
            # Otherwise, return the location of the key in the index
            return level_index.get_loc(key)

    def get_loc_level(self, key, level: IndexLabel = 0, drop_level: bool = True):
        """
        Get location and sliced index for requested label(s)/level(s).

        Parameters
        ----------
        key : label or sequence of labels
            The label or labels whose locations are to be found.
        level : int/level name or list thereof, optional
            The level(s) of the index to search within.
        drop_level : bool, default True
            If ``False``, the resulting index will not drop any level.

        Returns
        -------
        tuple
            A 2-tuple where the elements :

            Element 0: int, slice object or boolean array.
                Location(s) of the key(s) in the index.
            Element 1: The resulting sliced multiindex/index. If the key
                contains all levels, this will be ``None``.

        See Also
        --------
        MultiIndex.get_loc  : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list("abb"), list("def")], names=["A", "B"])

        >>> mi.get_loc_level("b")
        (slice(1, 3, None), Index(['e', 'f'], dtype='object', name='B'))

        >>> mi.get_loc_level("e", level="B")
        (array([False,  True, False]), Index(['b'], dtype='object', name='A'))

        >>> mi.get_loc_level(["b", "e"])
        (1, None)
        """
        if not isinstance(level, (range, list, tuple)):
            level = self._get_level_number(level)
        else:
            level = [self._get_level_number(lev) for lev in level]

        loc, mi = self._get_loc_level(key, level=level)
        if not drop_level:
            if lib.is_integer(loc):
                # Slice index must be an integer or None
                mi = self[loc : loc + 1]
            else:
                mi = self[loc]
        return loc, mi

    def _get_level_indexer(
        self, key, level: int = 0, indexer: npt.NDArray[np.bool_] | None = None
    ):
        """
        Placeholder function, likely to return indexer.

        Parameters
        ----------
        key : label or array-like
            The label or array-like object to query.
        level : int, optional, default 0
            The level within the index.
        indexer : ndarray of bools or None
            An optional array of boolean values.

        Returns
        -------
        """
        # This function is likely intended to return an indexer based on the key and level
        pass

    def _reorder_indexer(
        self,
        seq: tuple[Scalar | Iterable | AnyArrayLike, ...],
        indexer: npt.NDArray[np.intp],
    ):
        """
        Placeholder function for reordering indexer.

        Parameters
        ----------
        seq : tuple
            A tuple containing scalars, iterables, or array-like objects.
        indexer : ndarray of integers
            An array of integer values representing indices.

        Returns
        -------
        """
        # This function is likely intended to reorder the indexer based on seq
        pass
    def truncate(self, before=None, after=None) -> MultiIndex:
        """
        Slice index between two labels / tuples, return new MultiIndex.

        Parameters
        ----------
        before : label or tuple, can be partial. Default None
            None defaults to start.
        after : label or tuple, can be partial. Default None
            None defaults to end.

        Returns
        -------
        MultiIndex
            The truncated MultiIndex.

        See Also
        --------
        DataFrame.truncate : Truncate a DataFrame before and after some index values.
        Series.truncate : Truncate a Series before and after some index values.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([["a", "b", "c"], ["x", "y", "z"]])
        >>> mi
        MultiIndex([('a', 'x'), ('b', 'y'), ('c', 'z')],
                   )
        >>> mi.truncate(before="a", after="b")
        MultiIndex([('a', 'x'), ('b', 'y')],
                   )
        """
        # 如果指定的 after 小于 before，抛出 ValueError 异常
        if after and before and after < before:
            raise ValueError("after < before")

        # 获取 before 和 after 所在的切片位置
        i, j = self.levels[0].slice_locs(before, after)
        left, right = self.slice_locs(before, after)

        # 复制当前 MultiIndex 的 levels 列表
        new_levels = list(self.levels)
        # 更新第一层级的数据，只保留切片范围内的数据
        new_levels[0] = new_levels[0][i:j]

        # 复制当前 MultiIndex 的 codes 列表，只更新第一层级的代码
        new_codes = [level_codes[left:right] for level_codes in self.codes]
        # 调整第一层级的代码，以便正确映射到新的 levels
        new_codes[0] = new_codes[0] - i

        # 返回一个新的 MultiIndex 对象，基于更新后的 levels 和 codes
        return MultiIndex(
            levels=new_levels,
            codes=new_codes,
            names=self._names,
            verify_integrity=False,
        )
    def equals(self, other: object) -> bool:
        """
        Determines if two MultiIndex objects have the same labeling information
        (the levels themselves do not necessarily have to be the same)

        See Also
        --------
        equal_levels
        """
        # 检查对象是否和自身相同，如果是则返回True
        if self.is_(other):
            return True

        # 如果other不是Index的实例，则返回False
        if not isinstance(other, Index):
            return False

        # 如果self和other的长度不相等，则返回False
        if len(self) != len(other):
            return False

        # 如果other不是MultiIndex的实例
        if not isinstance(other, MultiIndex):
            # 对于d级别的MultiIndex，可以和d元组的Index相等
            if not self._should_compare(other):
                # 对象Index或Categorical[object]可能包含元组，返回False
                return False
            # 检查self._values和other._values是否数组等价
            return array_equivalent(self._values, other._values)

        # 如果self和other的层级数不相等，则返回False
        if self.nlevels != other.nlevels:
            return False

        # 遍历每个层级
        for i in range(self.nlevels):
            self_codes = self.codes[i]  # 获取self的第i层级的codes
            other_codes = other.codes[i]  # 获取other的第i层级的codes
            self_mask = self_codes == -1  # 获取self的第i层级的掩码
            other_mask = other_codes == -1  # 获取other的第i层级的掩码
            # 如果self_mask和other_mask不是数组等价，则返回False
            if not np.array_equal(self_mask, other_mask):
                return False
            self_level = self.levels[i]  # 获取self的第i层级的levels
            other_level = other.levels[i]  # 获取other的第i层级的levels
            # 根据other_level和self_level对other_codes进行重新编码，不进行复制
            new_codes = recode_for_categories(
                other_codes, other_level, self_level, copy=False
            )
            # 如果self_codes和new_codes不是数组等价，则返回False
            if not np.array_equal(self_codes, new_codes):
                return False
            # 如果self_level和other_level的第一个元素不相等，则返回False
            if not self_level[:0].equals(other_level[:0]):
                # 例如，Int64 != int64
                return False
        # 如果以上条件都满足，则返回True
        return True

    def equal_levels(self, other: MultiIndex) -> bool:
        """
        Return True if the levels of both MultiIndex objects are the same

        """
        # 如果self和other的层级数不相等，则返回False
        if self.nlevels != other.nlevels:
            return False

        # 遍历每个层级
        for i in range(self.nlevels):
            # 如果self的第i层级的levels和other的第i层级的levels不相等，则返回False
            if not self.levels[i].equals(other.levels[i]):
                return False
        # 如果以上条件都满足，则返回True
        return True

    # --------------------------------------------------------------------
    # Set Methods
    # 对象方法，用于执行 MultiIndex 对象的并集操作
    def _union(self, other, sort) -> MultiIndex:
        # 将 other 转换为适合进行集合操作的形式，并获取结果的名称列表
        other, result_names = self._convert_can_do_setop(other)
        
        # 如果 other 包含重复项，执行特定的并集操作；否则直接使用 difference 方法更快地计算
        if other.has_duplicates:
            result = super()._union(other, sort)
            
            # 如果结果是 MultiIndex 类型，则直接返回结果
            if isinstance(result, MultiIndex):
                return result
            # 否则从结果的元组列表中创建 MultiIndex 对象
            return MultiIndex.from_arrays(
                zip(*result), sortorder=None, names=result_names
            )
        else:
            # 获取 other 相对于 self 的差集，并将其附加到 self 上
            right_missing = other.difference(self, sort=False)
            if len(right_missing):
                result = self.append(right_missing)
            else:
                # 如果差集为空，则获取与 other 协调的名称对象
                result = self._get_reconciled_name_object(other)
            
            # 如果 sort 参数不为 False，则尝试对结果进行排序
            if sort is not False:
                try:
                    result = result.sort_values()
                except TypeError:
                    # 如果排序失败且 sort 为 True，则引发异常；否则发出警告
                    if sort is True:
                        raise
                    warnings.warn(
                        "The values in the array are unorderable. "
                        "Pass `sort=False` to suppress this warning.",
                        RuntimeWarning,
                        stacklevel=find_stack_level(),
                    )
            return result

    # 检查给定的 dtype 是否是可比较的对象数据类型
    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        return is_object_dtype(dtype)

    # 根据操作结果确定返回的名称对象
    def _get_reconciled_name_object(self, other) -> MultiIndex:
        """
        If the result of a set operation will be self,
        return self, unless the names change, in which
        case make a shallow copy of self.
        """
        # 尝试匹配操作结果的名称，如果名称发生变化则返回 self 的重命名副本
        names = self._maybe_match_names(other)
        if self.names != names:
            # 如果名称发生变化，返回重命名后的 self 对象
            return self.rename(names)  # type: ignore[has-type]
        # 否则直接返回 self 对象
        return self

    # 尝试匹配操作结果的名称，返回一个共识名称列表或 None 列表
    def _maybe_match_names(self, other):
        """
        Try to find common names to attach to the result of an operation between
        a and b. Return a consensus list of names if they match at least partly
        or list of None if they have completely different names.
        """
        # 如果 self 和 other 的名称列表长度不同，返回与 self 相同长度的 None 列表
        if len(self.names) != len(other.names):
            return [None] * len(self.names)
        names = []
        # 遍历 self 和 other 的名称列表，尝试找到共同的名称
        for a_name, b_name in zip(self.names, other.names):
            if a_name == b_name:
                names.append(a_name)
            else:
                # 如果名称不同，返回 None
                names.append(None)
        return names

    # 包装交集操作的结果，设置结果的名称为 result_names
    def _wrap_intersection_result(self, other, result) -> MultiIndex:
        _, result_names = self._convert_can_do_setop(other)
        return result.set_names(result_names)

    # 包装差集操作的结果，设置结果的名称为 result_names，并根据结果的长度决定是否移除未使用的级别
    def _wrap_difference_result(self, other, result: MultiIndex) -> MultiIndex:
        _, result_names = self._convert_can_do_setop(other)
        
        # 如果结果长度为 0，则移除未使用的级别并设置结果的名称
        if len(result) == 0:
            return result.remove_unused_levels().set_names(result_names)
        else:
            # 否则直接设置结果的名称为 result_names
            return result.set_names(result_names)
    # 定义一个方法用于执行可以进行集合操作的转换
    def _convert_can_do_setop(self, other):
        # 将结果名称设为自身的名称列表
        result_names = self.names

        # 如果 other 不是 Index 类型
        if not isinstance(other, Index):
            # 如果 other 是空的，返回一个空的 MultiIndex 和结果名称
            if len(other) == 0:
                return self[:0], self.names
            else:
                # 否则，生成错误信息
                msg = "other must be a MultiIndex or a list of tuples"
                try:
                    # 尝试将 other 转换为 MultiIndex 类型，并使用当前对象的名称
                    other = MultiIndex.from_tuples(other, names=self.names)
                except (ValueError, TypeError) as err:
                    # 如果转换失败，则抛出 TypeError，附带上原始错误信息
                    raise TypeError(msg) from err
        else:
            # 如果 other 是 Index 类型，则获取与当前对象一致的名称
            result_names = get_unanimous_names(self, other)

        # 返回处理后的 other 和结果名称
        return other, result_names

    # --------------------------------------------------------------------

    # 使用 Index.astype 方法的文档注释
    @doc(Index.astype)
    def astype(self, dtype, copy: bool = True):
        # 将 dtype 转换为 Pandas 数据类型
        dtype = pandas_dtype(dtype)

        # 如果 dtype 是 CategoricalDtype 类型，则抛出未实现的错误
        if isinstance(dtype, CategoricalDtype):
            msg = "> 1 ndim Categorical are not supported at this time"
            raise NotImplementedError(msg)

        # 如果 dtype 不是对象类型，则抛出类型错误
        if not is_object_dtype(dtype):
            raise TypeError(
                "Setting a MultiIndex dtype to anything other than object "
                "is not supported"
            )

        # 如果 copy 是 True，则返回当前对象的视图
        if copy is True:
            return self._view()
        
        # 否则，返回当前对象本身
        return self

    # 定义一个方法用于验证填充值的有效性
    def _validate_fill_value(self, item):
        # 如果 item 是 MultiIndex 类型
        if isinstance(item, MultiIndex):
            # 检查 item 的级数是否与当前对象的级数相同
            if item.nlevels != self.nlevels:
                raise ValueError("Item must have length equal to number of levels.")
            # 返回 item 的值数组
            return item._values
        
        # 如果 item 不是元组类型
        elif not isinstance(item, tuple):
            # 如果键的较低级别未指定，则使用空字符串填充键：
            item = (item,) + ("",) * (self.nlevels - 1)
        
        # 否则，如果 item 的长度与当前对象的级数不相同，则抛出值错误
        elif len(item) != self.nlevels:
            raise ValueError("Item must have length equal to number of levels.")
        
        # 返回处理后的 item
        return item
    # 使用给定的掩码和值更新 MultiIndex，并返回一个新的 MultiIndex 对象
    def putmask(self, mask, value: MultiIndex) -> MultiIndex:
        """
        Return a new MultiIndex of the values set with the mask.

        Parameters
        ----------
        mask : array like
            用于指定更新位置的布尔数组
        value : MultiIndex
            必须与当前 MultiIndex 的长度相同或长度为一

        Returns
        -------
        MultiIndex
        """
        # 验证掩码，并检查是否无需更新
        mask, noop = validate_putmask(self, mask)
        if noop:
            return self.copy()

        # 根据掩码长度决定如何更新值
        if len(mask) == len(value):
            subset = value[mask].remove_unused_levels()
        else:
            subset = value.remove_unused_levels()

        new_levels = []
        new_codes = []

        # 遍历当前 MultiIndex 的各级别，更新各级别的值
        for i, (value_level, level, level_codes) in enumerate(
            zip(subset.levels, self.levels, self.codes)
        ):
            # 合并新值和当前值的级别
            new_level = level.union(value_level, sort=False)
            # 获取新值在新级别中的索引，并更新当前代码
            value_codes = new_level.get_indexer_for(subset.get_level_values(i))
            new_code = ensure_int64(level_codes)
            new_code[mask] = value_codes
            new_levels.append(new_level)
            new_codes.append(new_code)

        # 返回更新后的 MultiIndex 对象
        return MultiIndex(
            levels=new_levels, codes=new_codes, names=self.names, verify_integrity=False
        )

    # 在指定位置插入新项目，返回新的 MultiIndex 对象
    def insert(self, loc: int, item) -> MultiIndex:
        """
        Make new MultiIndex inserting new item at location

        Parameters
        ----------
        loc : int
            要插入新项目的位置
        item : tuple
            必须与 MultiIndex 中级别的数量相同

        Returns
        -------
        new_index : Index
        """
        # 验证并填充要插入的项目
        item = self._validate_fill_value(item)

        new_levels = []
        new_codes = []
        # 遍历 MultiIndex 的各级别，根据需求进行插入
        for k, level, level_codes in zip(item, self.levels, self.codes):
            if k not in level:
                # 如果 k 不在当前级别中，则必须插入到末尾
                # 否则，必须插入到当前位置，否则需要重新计算所有其他代码
                if isna(k):  # GH 59003
                    lev_loc = -1
                else:
                    lev_loc = len(level)
                    level = level.insert(lev_loc, k)
            else:
                lev_loc = level.get_loc(k)

            new_levels.append(level)
            new_codes.append(np.insert(ensure_int64(level_codes), loc, lev_loc))

        # 返回更新后的 MultiIndex 对象
        return MultiIndex(
            levels=new_levels, codes=new_codes, names=self.names, verify_integrity=False
        )

    # 删除指定位置的元素，返回新的 MultiIndex 对象
    def delete(self, loc) -> MultiIndex:
        """
        Make new index with passed location deleted

        Returns
        -------
        new_index : MultiIndex
        """
        # 删除指定位置的代码，并生成新的代码列表
        new_codes = [np.delete(level_codes, loc) for level_codes in self.codes]
        # 返回更新后的 MultiIndex 对象
        return MultiIndex(
            levels=self.levels,
            codes=new_codes,
            names=self.names,
            verify_integrity=False,
        )

    @doc(Index.isin)
    # 添加 Index 类中 isin 方法的文档
    # 检查值是否存在于对象中，返回布尔类型的 NumPy 数组
    def isin(self, values, level=None) -> npt.NDArray[np.bool_]:
        # 如果传入的 values 是生成器，将其转换为列表
        if isinstance(values, Generator):
            values = list(values)

        # 如果未指定 level 参数
        if level is None:
            # 如果 values 为空列表，则返回一个全部为 False 的布尔数组，长度与 self 对象相同
            if len(values) == 0:
                return np.zeros((len(self),), dtype=np.bool_)
            # 如果 values 不是 MultiIndex 对象，则将其转换为 MultiIndex 对象
            if not isinstance(values, MultiIndex):
                values = MultiIndex.from_tuples(values)
            # 返回 values 唯一值的索引器与 self 的比较结果，不为 -1 则表示存在于 self 中
            return values.unique().get_indexer_for(self) != -1
        else:
            # 获取指定 level 的级别号码
            num = self._get_level_number(level)
            # 获取指定级别的级别值
            levs = self.get_level_values(num)

            # 如果级别值数组大小为 0，则返回一个全部为 False 的布尔数组
            if levs.size == 0:
                return np.zeros(len(levs), dtype=np.bool_)
            # 返回级别值数组中的值是否存在于给定的 values 中的布尔数组
            return levs.isin(values)

    # 将 Index 类的 set_names 方法赋值给 rename 变量，并忽略类型检查错误
    rename = Index.set_names  # type: ignore[assignment]

    # ---------------------------------------------------------------
    # 禁用的算术/数值方法

    # 将 __add__ 方法设置为无效操作
    __add__ = make_invalid_op("__add__")
    # 将 __radd__ 方法设置为无效操作
    __radd__ = make_invalid_op("__radd__")
    # 将 __iadd__ 方法设置为无效操作
    __iadd__ = make_invalid_op("__iadd__")
    # 将 __sub__ 方法设置为无效操作
    __sub__ = make_invalid_op("__sub__")
    # 将 __rsub__ 方法设置为无效操作
    __rsub__ = make_invalid_op("__rsub__")
    # 将 __isub__ 方法设置为无效操作
    __isub__ = make_invalid_op("__isub__")
    # 将 __pow__ 方法设置为无效操作
    __pow__ = make_invalid_op("__pow__")
    # 将 __rpow__ 方法设置为无效操作
    __rpow__ = make_invalid_op("__rpow__")
    # 将 __mul__ 方法设置为无效操作
    __mul__ = make_invalid_op("__mul__")
    # 将 __rmul__ 方法设置为无效操作
    __rmul__ = make_invalid_op("__rmul__")
    # 将 __floordiv__ 方法设置为无效操作
    __floordiv__ = make_invalid_op("__floordiv__")
    # 将 __rfloordiv__ 方法设置为无效操作
    __rfloordiv__ = make_invalid_op("__rfloordiv__")
    # 将 __truediv__ 方法设置为无效操作
    __truediv__ = make_invalid_op("__truediv__")
    # 将 __rtruediv__ 方法设置为无效操作
    __rtruediv__ = make_invalid_op("__rtruediv__")
    # 将 __mod__ 方法设置为无效操作
    __mod__ = make_invalid_op("__mod__")
    # 将 __rmod__ 方法设置为无效操作
    __rmod__ = make_invalid_op("__rmod__")
    # 将 __divmod__ 方法设置为无效操作
    __divmod__ = make_invalid_op("__divmod__")
    # 将 __rdivmod__ 方法设置为无效操作
    __rdivmod__ = make_invalid_op("__rdivmod__")
    # 禁用的一元方法
    # 将 __neg__ 方法设置为无效操作
    __neg__ = make_invalid_op("__neg__")
    # 将 __pos__ 方法设置为无效操作
    __pos__ = make_invalid_op("__pos__")
    # 将 __abs__ 方法设置为无效操作
    __abs__ = make_invalid_op("__abs__")
    # 将 __invert__ 方法设置为无效操作
    __invert__ = make_invalid_op("__invert__")
# 计算代码数组列表的词法排序深度（最多为 `nlevels`）。
def _lexsort_depth(codes: list[np.ndarray], nlevels: int) -> int:
    # 将每个代码数组转换为确保为 int64 类型的数组
    int64_codes = [ensure_int64(level_codes) for level_codes in codes]
    # 从 `nlevels` 向 1 递减，检查是否词法排序
    for k in range(nlevels, 0, -1):
        # 如果前 k 个数组词法排序，则返回 k
        if libalgos.is_lexsorted(int64_codes[:k]):
            return k
    # 如果都不是词法排序，则返回 0
    return 0


# 将标签列表转换为稀疏表示的标签列表
def sparsify_labels(label_list, start: int = 0, sentinel: object = ""):
    # 将标签列表进行转置
    pivoted = list(zip(*label_list))
    k = len(label_list)

    # 初始化结果列表为起始位置及之前的部分
    result = pivoted[: start + 1]
    prev = pivoted[start]

    # 遍历从起始位置之后的标签
    for cur in pivoted[start + 1 :]:
        sparse_cur = []

        # 逐个比较前后标签，生成稀疏表示
        for i, (p, t) in enumerate(zip(prev, cur)):
            if i == k - 1:
                sparse_cur.append(t)
                result.append(sparse_cur)  # 添加稀疏表示到结果中
                break

            if p == t:
                sparse_cur.append(sentinel)
            else:
                sparse_cur.extend(cur[i:])
                result.append(sparse_cur)  # 添加稀疏表示到结果中
                break

        prev = cur

    # 返回转置后的结果列表
    return list(zip(*result))


# 获取指定数据类型的 NA 表示
def _get_na_rep(dtype: DtypeObj) -> str:
    if isinstance(dtype, ExtensionDtype):
        return f"{dtype.na_value}"  # 如果是扩展数据类型，返回其 NA 值的字符串表示
    else:
        dtype_type = dtype.type

    # 根据数据类型返回对应的 NA 表示
    return {np.datetime64: "NaT", np.timedelta64: "NaT"}.get(dtype_type, "NaN")


# 尝试从给定的索引中删除级别或多个级别
def maybe_droplevels(index: Index, key) -> Index:
    """
    Attempt to drop level or levels from the given index.

    Parameters
    ----------
    index: Index
        原始索引对象
    key : scalar or tuple
        要删除的级别或级别的键

    Returns
    -------
    Index
        返回处理后的索引对象
    """
    # 保存原始索引
    original_index = index
    if isinstance(key, tuple):
        # 如果 key 是元组，逐个尝试删除对应级别
        for _ in key:
            try:
                index = index._drop_level_numbers([0])
            except ValueError:
                # 如果删除过多导致错误，返回原始索引
                return original_index
    else:
        try:
            # 尝试删除第一个级别
            index = index._drop_level_numbers([0])
        except ValueError:
            pass

    # 返回处理后的索引
    return index


# 强制转换数组索引器为可以编码所有给定类别的最小整数类型
def _coerce_indexer_frozen(array_like, categories, copy: bool = False) -> np.ndarray:
    """
    Coerce the array-like indexer to the smallest integer dtype that can encode all
    of the given categories.

    Parameters
    ----------
    array_like : array-like
        数组索引器
    categories : array-like
        给定的类别数组
    copy : bool
        是否复制数组

    Returns
    -------
    np.ndarray
        返回不可写的数组
    """
    # 确保数组索引器的数据类型可以编码所有给定的类别
    array_like = coerce_indexer_dtype(array_like, categories)
    if copy:
        array_like = array_like.copy()  # 如果需要复制，进行复制操作
    array_like.flags.writeable = False  # 设置数组为不可写
    return array_like


# 确保级别是 None 或类似列表的形式
def _require_listlike(level, arr, arrname: str):
    """
    Ensure that level is either None or listlike, and arr is list-of-listlike.
    """
    # 如果 level 不是 None 并且不是列表类型（is_list_like 函数用于检查是否为列表类型）
    if level is not None and not is_list_like(level):
        # 如果 arr 不是列表类型，则抛出 TypeError 异常，指示 arrname 必须是类似列表的对象
        if not is_list_like(arr):
            raise TypeError(f"{arrname} must be list-like")
        # 如果 arr 的长度大于 0 并且 arr[0] 不是列表类型，则抛出 TypeError 异常，指示 arrname 必须是类似列表的对象
        if len(arr) > 0 and is_list_like(arr[0]):
            raise TypeError(f"{arrname} must be list-like")
        # 将 level 和 arr 分别转换为列表类型
        level = [level]
        arr = [arr]
    # 如果 level 是 None 或者是列表类型（is_list_like 函数用于检查是否为列表类型）
    elif level is None or is_list_like(level):
        # 如果 arr 不是列表类型，或者 arr[0] 不是列表类型，则抛出 TypeError 异常，指示 arrname 必须是类似列表的列表对象
        if not is_list_like(arr) or not is_list_like(arr[0]):
            raise TypeError(f"{arrname} must be list of lists-like")
    # 返回经过检查和调整后的 level 和 arr
    return level, arr
```