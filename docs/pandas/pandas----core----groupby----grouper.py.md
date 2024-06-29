# `D:\src\scipysrc\pandas\pandas\core\groupby\grouper.py`

```
"""
Provide user facing operators for doing the split part of the
split-apply-combine paradigm.
"""

# 导入必要的模块和类
from __future__ import annotations  # 支持将注解作为返回类型使用

from typing import (
    TYPE_CHECKING,  # 用于类型检查的特殊标志
    final,  # 用于声明不可重写的类和方法
)

import numpy as np  # 导入 NumPy 库

# 导入 pandas 中的特定错误和装饰器
from pandas._libs.tslibs import OutOfBoundsDatetime  # 时间序列相关的异常
from pandas.errors import InvalidIndexError  # 索引相关的异常
from pandas.util._decorators import cache_readonly  # 用于装饰缓存只读属性的装饰器

# 导入 pandas 中与数据类型和常用函数相关的模块
from pandas.core.dtypes.common import (
    is_list_like,  # 检查对象是否类列表
    is_scalar,  # 检查对象是否标量
)
from pandas.core.dtypes.dtypes import CategoricalDtype  # 分类数据类型

# 导入 pandas 中的算法和数组相关模块
from pandas.core import algorithms  # pandas 中的算法
from pandas.core.arrays import (
    Categorical,  # pandas 中的分类数组
    ExtensionArray,  # 扩展数组
)
import pandas.core.common as com  # pandas 中的通用功能
from pandas.core.frame import DataFrame  # pandas 中的数据帧类
from pandas.core.groupby import ops  # pandas 中的分组操作
from pandas.core.groupby.categorical import recode_for_groupby  # 分类数据的分组重编码
from pandas.core.indexes.api import (
    Index,  # pandas 中的索引类
    MultiIndex,  # pandas 中的多级索引类
    default_index,  # pandas 中默认索引的创建函数
)
from pandas.core.series import Series  # pandas 中的序列类

# 导入 pandas 中的格式化打印函数
from pandas.io.formats.printing import pprint_thing  # 用于美化打印输出的函数

# 如果在类型检查模式下，导入额外的类型
if TYPE_CHECKING:
    from collections.abc import (
        Hashable,  # 可散列对象的抽象基类
        Iterator,  # 迭代器的抽象基类
    )

    from pandas._typing import (
        ArrayLike,  # 类数组对象的类型
        NDFrameT,  # pandas 数据结构的类型
        npt,  # NumPy 类型
    )

    from pandas.core.generic import NDFrame  # pandas 中通用数据结构的抽象基类


class Grouper:
    """
    A Grouper allows the user to specify a groupby instruction for an object.

    This specification will select a column via the key parameter, or if the
    level parameter is given, a level of the index of the target
    object.

    If ``level`` is passed as a keyword to both `Grouper` and
    `groupby`, the values passed to `Grouper` take precedence.

    Parameters
    ----------
    *args
        Currently unused, reserved for future use.
    **kwargs
        Dictionary of the keyword arguments to pass to Grouper.
    key : str, defaults to None
        Groupby key, which selects the grouping column of the target.
    level : name/number, defaults to None
        The level for the target index.
    freq : str / frequency object, defaults to None
        This will groupby the specified frequency if the target selection
        (via key or level) is a datetime-like object. For full specification
        of available frequencies, please see :ref:`here<timeseries.offset_aliases>`.
    sort : bool, default to False
        Whether to sort the resulting labels.
    closed : {'left' or 'right'}
        Closed end of interval. Only when `freq` parameter is passed.
    label : {'left' or 'right'}
        Interval boundary to use for labeling.
        Only when `freq` parameter is passed.
    convention : {'start', 'end', 'e', 's'}
        If grouper is PeriodIndex and `freq` parameter is passed.
    """
    origin : Timestamp or str, default 'start_day'
        # 参数 origin 表示时间分组的起点，可以是 Timestamp 对象或字符串，字符串取值如下：
        # - 'epoch': origin 是 1970-01-01
        # - 'start': origin 是时间序列的第一个值
        # - 'start_day': origin 是时间序列的第一天午夜时刻

        # - 'end': origin 是时间序列的最后一个值
        # - 'end_day': origin 是时间序列最后一天的午夜时刻

        # .. versionadded:: 1.3.0
        # 版本 1.3.0 新增功能

    offset : Timedelta or str, default is None
        # 偏移量，可以是 Timedelta 对象或字符串，默认为 None

    dropna : bool, default True
        # 如果为 True，并且分组键包含 NA 值，则会将 NA 值与对应的行/列一起删除。
        # 如果为 False，NA 值也将作为键进行分组。

    Returns
    -------
    Grouper or pandas.api.typing.TimeGrouper
        # 如果 freq 不是 None，则返回一个 TimeGrouper 对象；否则返回一个 Grouper 对象。

    See Also
    --------
    Series.groupby : 对 Series 应用分组函数。
    DataFrame.groupby : 对 DataFrame 应用分组函数。

    Examples
    --------
    ``df.groupby(pd.Grouper(key="Animal"))`` 等同于 ``df.groupby('Animal')``
        # 对于 DataFrame df，根据 'Animal' 列进行分组

    >>> df = pd.DataFrame(
    ...     {
    ...         "Animal": ["Falcon", "Parrot", "Falcon", "Falcon", "Parrot"],
    ...         "Speed": [100, 5, 200, 300, 15],
    ...     }
    ... )
    >>> df
       Animal  Speed
    0  Falcon    100
    1  Parrot      5
    2  Falcon    200
    3  Falcon    300
    4  Parrot     15
    >>> df.groupby(pd.Grouper(key="Animal")).mean()
            Speed
    Animal
    Falcon  200.0
    Parrot   10.0

    Specify a resample operation on the column 'Publish date'

    >>> df = pd.DataFrame(
    ...     {
    ...         "Publish date": [
    ...             pd.Timestamp("2000-01-02"),
    ...             pd.Timestamp("2000-01-02"),
    ...             pd.Timestamp("2000-01-09"),
    ...             pd.Timestamp("2000-01-16"),
    ...         ],
    ...         "ID": [0, 1, 2, 3],
    ...         "Price": [10, 20, 30, 40],
    ...     }
    ... )
    >>> df
      Publish date  ID  Price
    0   2000-01-02   0     10
    1   2000-01-02   1     20
    2   2000-01-09   2     30
    3   2000-01-16   3     40
    >>> df.groupby(pd.Grouper(key="Publish date", freq="1W")).mean()
                   ID  Price
    Publish date
    2000-01-02    0.5   15.0
    2000-01-09    2.0   30.0
    2000-01-16    3.0   40.0

    If you want to adjust the start of the bins based on a fixed timestamp:

    >>> start, end = "2000-10-01 23:30:00", "2000-10-02 00:30:00"
    >>> rng = pd.date_range(start, end, freq="7min")
    >>> ts = pd.Series(np.arange(len(rng)) * 3, index=rng)
    >>> ts
    2000-10-01 23:30:00     0
    2000-10-01 23:37:00     3
    2000-10-01 23:44:00     6
    2000-10-01 23:51:00     9
    2000-10-01 23:58:00    12
        # 创建一个时间序列 ts，以 7 分钟为频率进行索引
    """
    sort: bool
    dropna: bool
    _grouper: Index | None
    
    _attributes: tuple[str, ...] = ("key", "level", "freq", "sort", "dropna")
    定义类的属性和变量，包括排序标志，缺失值处理标志，以及一个私有变量 _grouper 和一个元组 _attributes，包含了属性的名称。
    
    def __new__(cls, *args, **kwargs):
        如果传入参数中包含 freq 关键字，则动态改变类的类型为 TimeGrouper 类。
        这里通过检查 kwargs 中的 freq 参数来决定是否切换类的类型。
        from pandas.core.resample import TimeGrouper
        cls = TimeGrouper
        return super().__new__(cls)
    动态创建对象时，用来决定类的实际类型，基于是否传入了 freq 参数。
    
    def __init__(
        self,
        key=None,
        level=None,
        freq=None,
        sort: bool = False,
        dropna: bool = True,
    ) -> None:
        初始化函数，用来设置对象的属性。
        self.key = key
        self.level = level
        self.freq = freq
        self.sort = sort
        self.dropna = dropna
        初始化对象的 key、level、freq、sort 和 dropna 属性。
    
    self._indexer_deprecated: npt.NDArray[np.intp] | None = None
    self.binner = None
    self._grouper = None
    self._indexer: npt.NDArray[np.intp] | None = None
    初始化对象的几个私有属性，分别为 _indexer_deprecated、binner、_grouper 和 _indexer。
    
    def _get_grouper(
        self, obj: NDFrameT, validate: bool = True
        获取 grouper 对象的方法。
        obj: NDFrameT 表示传入的数据对象。
        validate: bool 表示是否需要验证数据。
    
    """
    @final
    # 定义一个不可继承的方法，用于返回对象的字符串表示形式
    def __repr__(self) -> str:
        # 生成对象属性的字符串表示形式列表
        attrs_list = (
            f"{attr_name}={getattr(self, attr_name)!r}"
            for attr_name in self._attributes
            # 过滤掉属性值为 None 的属性
            if getattr(self, attr_name) is not None
        )
        # 将属性列表用逗号连接成一个字符串
        attrs = ", ".join(attrs_list)
        # 获取当前对象的类名
        cls_name = type(self).__name__
        # 返回格式化后的对象字符串表示形式
        return f"{cls_name}({attrs})"
@final
class Grouping:
    """
    Holds the grouping information for a single key

    Parameters
    ----------
    index : Index
        The index to use for grouping
    grouper :
        Not specified in the code snippet
    obj : DataFrame or Series
        The data object (DataFrame or Series) to be grouped
    name : Label
        The label associated with the grouping
    level :
        Not specified in the code snippet
    observed : bool, default False
        If True, uses observed values for Categorical data
    in_axis : bool
        Indicates if the Grouping is a column in self.obj and hence among Groupby.exclusions list
    dropna : bool, default True
        Whether to drop NA groups
    uniques : Array-like, optional
        Array of unique values; enables including empty groups for a BinGrouper

    Attributes
    -------
    indices : dict
        Mapping of {group -> index_list}
        Holds mappings from each group to its corresponding index list
    codes : ndarray
        Group codes
        NumPy array containing group codes
    group_index : Index or None
        Unique groups
        Represents unique groups as an Index object or None if not applicable
    groups : dict
        Mapping of {group -> label_list}
        Holds mappings from each group to its corresponding label list
    """

    _codes: npt.NDArray[np.signedinteger] | None = None
    _orig_cats: Index | None
    _index: Index

    def __init__(
        self,
        index: Index,
        grouper=None,
        obj: NDFrame | None = None,
        level=None,
        sort: bool = True,
        observed: bool = False,
        in_axis: bool = False,
        dropna: bool = True,
        uniques: ArrayLike | None = None,
    ):
        """
        Initialize Grouping object.

        Parameters
        ----------
        index : Index
            The index to use for grouping
        grouper :
            Not specified in the code snippet
        obj : NDFrame or None, optional
            The data object (DataFrame or Series) to be grouped
        level :
            Not specified in the code snippet
        sort : bool, default True
            Whether to sort the groups
        observed : bool, default False
            If True, uses observed values for Categorical data
        in_axis : bool, default False
            Indicates if the Grouping is a column in self.obj and hence among Groupby.exclusions list
        dropna : bool, default True
            Whether to drop NA groups
        uniques : ArrayLike or None, optional
            Array of unique values; enables including empty groups for a BinGrouper
        """
        pass

    def __repr__(self) -> str:
        """
        Return a string representation of the Grouping object.
        """
        return f"Grouping({self.name})"

    def __iter__(self) -> Iterator:
        """
        Return an iterator over the indices of the groups.
        """
        return iter(self.indices)

    @cache_readonly
    def _passed_categorical(self) -> bool:
        """
        Check if the grouping vector is of CategoricalDtype.
        """
        dtype = getattr(self.grouping_vector, "dtype", None)
        return isinstance(dtype, CategoricalDtype)

    @cache_readonly
    def name(self) -> Hashable:
        """
        Return the name associated with the Grouping.

        Returns
        -------
        Hashable or None
            Name of the Grouping or None if no name is available.
        """
        ilevel = self._ilevel
        if ilevel is not None:
            return self._index.names[ilevel]

        if isinstance(self._orig_grouper, (Index, Series)):
            return self._orig_grouper.name

        elif isinstance(self.grouping_vector, ops.BaseGrouper):
            return self.grouping_vector.result_index.name

        elif isinstance(self.grouping_vector, Index):
            return self.grouping_vector.name

        # otherwise we have ndarray or ExtensionArray -> no name
        return None

    @cache_readonly
    def _ilevel(self) -> int | None:
        """
        Convert index level name to index level position if necessary.

        Returns
        -------
        int or None
            Index level position or None if level is not found.
        """
        level = self.level
        if level is None:
            return None
        if not isinstance(level, int):
            index = self._index
            if level not in index.names:
                raise AssertionError(f"Level {level} not in index")
            return index.names.index(level)
        return level

    @property
    def ngroups(self) -> int:
        """
        Return the number of unique groups.

        Returns
        -------
        int
            Number of unique groups.
        """
        return len(self.uniques)

    @cache_readonly
    # 返回一个字典，其键是可散列对象，值是 NumPy 数组（整数类型）
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        # 如果 self.grouping_vector 是 ops.BaseGrouper 类型的实例
        if isinstance(self.grouping_vector, ops.BaseGrouper):
            # 直接返回 self.grouping_vector 的 indices 属性
            return self.grouping_vector.indices

        # 否则，将 self.grouping_vector 转换为 Categorical 对象
        values = Categorical(self.grouping_vector)
        # 返回 values 对象的 _reverse_indexer 方法的结果
        return values._reverse_indexer()

    # 返回一个 NumPy 数组（有符号整数类型），即 self._codes_and_uniques 的第一个元素
    @property
    def codes(self) -> npt.NDArray[np.signedinteger]:
        return self._codes_and_uniques[0]

    # 返回一个类似数组（ArrayLike），即 self._codes_and_uniques 的第二个元素
    @property
    def uniques(self) -> ArrayLike:
        return self._codes_and_uniques[1]

    # 表示一个缓存只读属性（cache_readonly）
    def _codes_and_uniques(self) -> tuple[npt.NDArray[np.signedinteger], ArrayLike]:
        uniques: ArrayLike
        # 如果传入了分类数据
        if self._passed_categorical:
            # 创建一个基于分类数据的 CategoricalIndex，
            # 保留其分类和有序属性；
            # 目前不支持（GH#46909）处理 dropna=False
            cat = self.grouping_vector
            categories = cat.categories

            # 如果观察过数据
            if self._observed:
                # 获取分类的唯一值代码并删除值为 -1 的条目
                ucodes = algorithms.unique1d(cat.codes)
                ucodes = ucodes[ucodes != -1]
                # 如果需要排序，对唯一值代码进行排序
                if self._sort:
                    ucodes = np.sort(ucodes)
            else:
                # 否则直接生成从0到分类数目的代码
                ucodes = np.arange(len(categories))

            has_dropped_na = False
            # 如果不丢弃 NA 值
            if not self._dropna:
                na_mask = cat.isna()
                # 如果存在 NA 值
                if np.any(na_mask):
                    has_dropped_na = True
                    if self._sort:
                        # 将 NA 值放在末尾，代码为“最大非 NA 代码 + 1”
                        na_code = len(categories)
                    else:
                        # 根据首次出现的位置插入 NA 值，需要之前唯一代码的数目
                        na_idx = na_mask.argmax()
                        na_code = algorithms.nunique_ints(cat.codes[:na_idx])
                    ucodes = np.insert(ucodes, na_code, -1)

            # 创建一个基于唯一代码和分类的 Categorical 对象
            uniques = Categorical.from_codes(
                codes=ucodes, categories=categories, ordered=cat.ordered, validate=False
            )
            codes = cat.codes

            # 如果有丢弃的 NA 值
            if has_dropped_na:
                if not self._sort:
                    # 根据首次出现的位置插入 NA 代码，增加高代码的值
                    codes = np.where(codes >= na_code, codes + 1, codes)
                codes = np.where(na_mask, na_code, codes)

            return codes, uniques

        # 如果 grouping_vector 是 ops.BaseGrouper 的实例
        elif isinstance(self.grouping_vector, ops.BaseGrouper):
            # 我们有一个分组向量列表
            codes = self.grouping_vector.codes_info
            uniques = self.grouping_vector.result_index._values

        # 如果已经存在 _uniques
        elif self._uniques is not None:
            # 使用 _uniques 来分组 grouping_vector；
            # 允许包含不在 grouping_vector 中的唯一值。
            cat = Categorical(self.grouping_vector, categories=self._uniques)
            codes = cat.codes
            uniques = self._uniques

        else:
            # GH35667, 将 dropna=False 替换为 use_na_sentinel=False
            # 错误：分配中类型不兼容（表达式的类型为 "Union[ndarray[Any, Any], Index]"，变量的类型为 "Categorical"）
            codes, uniques = algorithms.factorize(  # type: ignore[assignment]
                self.grouping_vector, sort=self._sort, use_na_sentinel=self._dropna
            )

        return codes, uniques

    @cache_readonly
    # 返回一个字典，键为可哈希的对象，值为 Index 对象
    def groups(self) -> dict[Hashable, Index]:
        # 解构赋值，获取 self._codes_and_uniques 的返回值
        codes, uniques = self._codes_and_uniques
        # 创建新的 Index 对象，并使用 uniques 参数进行推断，设置对象的名称为 self.name
        uniques = Index._with_infer(uniques, name=self.name)
        # 使用 codes 和 uniques 创建一个 Categorical 对象，禁用验证
        cats = Categorical.from_codes(codes, uniques, validate=False)
        # 根据 cats 对象对 self._index 进行分组，返回分组结果字典
        return self._index.groupby(cats)

    # 返回一个 Grouping 对象，用于表示观察到的分组情况
    @property
    def observed_grouping(self) -> Grouping:
        # 如果 self._observed 为真，则返回自身对象
        if self._observed:
            return self
        # 否则返回 self._observed_grouping 对象
        return self._observed_grouping

    # 返回一个缓存只读的 _observed_grouping 对象，用于表示观察到的分组情况
    @cache_readonly
    def _observed_grouping(self) -> Grouping:
        # 创建一个 Grouping 对象，使用以下参数初始化：
        # self._index, self._orig_grouper, self.obj, self.level, self._sort,
        # observed=True, self.in_axis, self._dropna, self._uniques
        grouping = Grouping(
            self._index,
            self._orig_grouper,
            obj=self.obj,
            level=self.level,
            sort=self._sort,
            observed=True,
            in_axis=self.in_axis,
            dropna=self._dropna,
            uniques=self._uniques,
        )
        # 返回创建的 Grouping 对象
        return grouping
def get_grouper(
    obj: NDFrameT,
    key=None,
    level=None,
    sort: bool = True,
    observed: bool = False,
    validate: bool = True,
    dropna: bool = True,
) -> tuple[ops.BaseGrouper, frozenset[Hashable], NDFrameT]:
    """
    Create and return a BaseGrouper, which is an internal
    mapping of how to create the grouper indexers.
    This may be composed of multiple Grouping objects, indicating
    multiple groupers

    Groupers are ultimately index mappings. They can originate as:
    index mappings, keys to columns, functions, or Groupers

    Groupers enable local references to level,sort, while
    the passed in level, and sort are 'global'.

    This routine tries to figure out what the passing in references
    are and then creates a Grouping for each one, combined into
    a BaseGrouper.

    If observed & we have a categorical grouper, only show the observed
    values.

    If validate, then check for key/level overlaps.

    """
    group_axis = obj.index

    # validate that the passed single level is compatible with the passed
    # index of the object
    if level is not None:
        # Handle the case where obj.index is a MultiIndex
        if isinstance(group_axis, MultiIndex):
            # If level is a list-like object with one element, use that element
            if is_list_like(level) and len(level) == 1:
                level = level[0]

            # If key is None and level is a scalar, get the level values from group_axis
            if key is None and is_scalar(level):
                key = group_axis.get_level_values(level)
                level = None

        else:
            # Handle the case where obj.index is not a MultiIndex
            if is_list_like(level):
                nlevels = len(level)
                # If level is a length-one list-like object, use that element
                if nlevels == 1:
                    level = level[0]
                elif nlevels == 0:
                    raise ValueError("No group keys passed!")
                else:
                    raise ValueError("multiple levels only valid with MultiIndex")

            # If level is a string, check if it matches the index name of obj
            if isinstance(level, str):
                if obj.index.name != level:
                    raise ValueError(f"level name {level} is not the name of the index")
            # If level is an integer, it should be 0 or -1 for a non-MultiIndex
            elif level > 0 or level < -1:
                raise ValueError("level > 0 or level < -1 only valid with MultiIndex")

            # Both `group_axis` and `group_axis.get_level_values(level)` are essentially
            # the same in this section, so set level to None and key to group_axis
            level = None
            key = group_axis

    # If a passed-in Grouper exists, directly convert it
    # 如果 key 是 Grouper 类型的实例
    if isinstance(key, Grouper):
        # 从 key 获取 grouper 和 obj，不进行验证
        grouper, obj = key._get_grouper(obj, validate=False)
        # 如果 key.key 为 None，返回 grouper、空 frozenset 和 obj
        if key.key is None:
            return grouper, frozenset(), obj
        else:
            # 否则返回 grouper、包含 key.key 的 frozenset 和 obj
            return grouper, frozenset({key.key}), obj

    # 如果 key 已经是 BaseGrouper 类型的实例，直接返回 key、空 frozenset 和 obj
    elif isinstance(key, ops.BaseGrouper):
        return key, frozenset(), obj

    # 如果 key 不是 list 类型，将其转化为列表 keys，match_axis_length 设为 False
    if not isinstance(key, list):
        keys = [key]
        match_axis_length = False
    else:
        # 否则直接使用 key 作为 keys，match_axis_length 设为 keys 和 group_axis 长度是否相等
        keys = key
        match_axis_length = len(keys) == len(group_axis)

    # 确定 keys 中是否有可调用对象（callable）、Grouper 或 Grouping 类型的实例（any_groupers）、
    # 数组或类数组对象（any_arraylike）
    any_callable = any(callable(g) or isinstance(g, dict) for g in keys)
    any_groupers = any(isinstance(g, (Grouper, Grouping)) for g in keys)
    any_arraylike = any(
        isinstance(g, (list, tuple, Series, Index, np.ndarray)) for g in keys
    )

    # 如果不是可调用对象、数组或类数组对象、Grouper 或 Grouping 实例，并且匹配 axis 长度，
    # level 为 None，判断是否为索引替换
    if (
        not any_callable
        and not any_arraylike
        and not any_groupers
        and match_axis_length
        and level is None
    ):
        # 如果 obj 是 DataFrame
        if isinstance(obj, DataFrame):
            # 检查 keys 是否都在 obj 的列或索引名称中
            all_in_columns_index = all(
                g in obj.columns or g in obj.index.names for g in keys
            )
        else:
            # 否则，obj 应为 Series，检查 keys 是否都在 obj 的索引名称中
            assert isinstance(obj, Series)
            all_in_columns_index = all(g in obj.index.names for g in keys)

        # 如果不都在列或索引名称中，将 keys 转化为单元素元组的数组
        if not all_in_columns_index:
            keys = [com.asarray_tuplesafe(keys)]

    # 如果 level 是元组或列表，且 key 为 None，将 keys 设为长度为 level 的列表
    if isinstance(level, (tuple, list)):
        if key is None:
            keys = [None] * len(level)
        levels = level
    else:
        # 否则，将 levels 设为长度为 keys 的 level 列表
        levels = [level] * len(keys)

    # 初始化 groupings 为 Grouping 类型的列表，exclusions 为空的集合
    groupings: list[Grouping] = []
    exclusions: set[Hashable] = set()

    # 判断是否应该是 obj[key] 的 grouper
    def is_in_axis(key) -> bool:
        # 如果 key 不是标签样式的，且 obj 是一维的，返回 False
        if not _is_label_like(key):
            if obj.ndim == 1:
                return False

            # 对于 DataFrame，获取最后一个轴的 items（列名）；对于 Series，获取索引
            items = obj.axes[-1]
            try:
                # 尝试获取 key 在 items 中的位置
                items.get_loc(key)
            except (KeyError, TypeError, InvalidIndexError):
                # 如果出现 KeyError、TypeError 或 InvalidIndexError，返回 False
                return False

        return True

    # 判断是否应该是 obj[name] 的 grouper
    def is_in_obj(gpr) -> bool:
        # 如果 gpr 没有 "name" 属性，返回 False
        if not hasattr(gpr, "name"):
            return False
        # 检查 obj[gpr.name] 是否存在，确定该系列是否属于对象的一部分
        try:
            obj_gpr_column = obj[gpr.name]
        except (KeyError, IndexError, InvalidIndexError, OutOfBoundsDatetime):
            return False
        # 如果 gpr 和 obj_gpr_column 都是 Series，并且引用相同的值，返回 True
        if isinstance(gpr, Series) and isinstance(obj_gpr_column, Series):
            return gpr._mgr.references_same_values(obj_gpr_column._mgr, 0)
        return False
    # 遍历给定的 keys 和 levels 列表，同时迭代它们的元素和对应的索引
    for gpr, level in zip(keys, levels):
        # 检查 gpr 是否在对象中作为分组条件
        if is_in_obj(gpr):  # df.groupby(df['name'])
            # 如果是，则设置 in_axis 标志为 True，并将 gpr.name 添加到 exclusions 集合中
            in_axis = True
            exclusions.add(gpr.name)

        # 如果 gpr 是轴上的分组条件（字符串形式）
        elif is_in_axis(gpr):  # df.groupby('name')
            # 如果对象不是一维的且 gpr 在对象中
            if obj.ndim != 1 and gpr in obj:
                # 如果需要验证，则检查标签或层次的歧义性
                if validate:
                    obj._check_label_or_level_ambiguity(gpr, axis=0)
                # 设置 in_axis 为 True，name 为 gpr，gpr 为 obj[gpr]
                in_axis, name, gpr = True, gpr, obj[gpr]
                # 如果 gpr 不是一维的，则抛出 ValueError 异常
                if gpr.ndim != 1:
                    raise ValueError(f"Grouper for '{name}' not 1-dimensional")
                # 将 name 添加到 exclusions 集合中
                exclusions.add(name)
            # 如果 gpr 是轴上的层次索引参考
            elif obj._is_level_reference(gpr, axis=0):
                # 设置 in_axis 为 False，level 为 gpr，gpr 为 None
                in_axis, level, gpr = False, gpr, None
            else:
                # 否则抛出 KeyError 异常
                raise KeyError(gpr)

        # 如果 gpr 是 Grouper 类型且其 key 不为 None
        elif isinstance(gpr, Grouper) and gpr.key is not None:
            # 将 gpr.key 添加到 exclusions 集合中
            exclusions.add(gpr.key)
            # 设置 in_axis 为 True
            in_axis = True

        else:
            # 否则设置 in_axis 为 False
            in_axis = False

        # 创建 Grouping 对象 ping
        # 允许我们将实际的 Grouping 作为 gpr 传递进去
        ping = (
            Grouping(
                group_axis,
                gpr,
                obj=obj,
                level=level,
                sort=sort,
                observed=observed,
                in_axis=in_axis,
                dropna=dropna,
            )
            if not isinstance(gpr, Grouping)
            else gpr
        )

        # 将 ping 添加到 groupings 列表中
        groupings.append(ping)

    # 如果 groupings 列表长度为 0 且对象 obj 长度不为 0，则抛出 ValueError 异常
    if len(groupings) == 0 and len(obj):
        raise ValueError("No group keys passed!")
    # 如果 groupings 列表长度为 0，则将空的 Grouping 对象添加到 groupings 中
    if len(groupings) == 0:
        groupings.append(Grouping(default_index(0), np.array([], dtype=np.intp)))

    # 创建内部的 grouper 对象
    grouper = ops.BaseGrouper(group_axis, groupings, sort=sort, dropna=dropna)
    # 返回 grouper 对象、exclusions 的不可变集合和对象 obj
    return grouper, frozenset(exclusions), obj
# 判断参数 `val` 是否类似标签，返回布尔值
def _is_label_like(val) -> bool:
    return isinstance(val, (str, tuple)) or (val is not None and is_scalar(val))


# 将分组器 `grouper` 转换为适当的形式，以便进行分组操作
def _convert_grouper(axis: Index, grouper):
    # 如果 `grouper` 是字典，则返回其 `get` 方法
    if isinstance(grouper, dict):
        return grouper.get
    # 如果 `grouper` 是 Series 类型
    elif isinstance(grouper, Series):
        # 如果 Series 的索引与 `axis` 相同，则返回其值数组
        if grouper.index.equals(axis):
            return grouper._values
        else:
            # 否则，重新索引到 `axis` 并返回其值数组
            return grouper.reindex(axis)._values
    # 如果 `grouper` 是 MultiIndex 类型，则返回其值数组
    elif isinstance(grouper, MultiIndex):
        return grouper._values
    # 如果 `grouper` 是列表、元组、索引、分类类型或 ndarray
    elif isinstance(grouper, (list, tuple, Index, Categorical, np.ndarray)):
        # 检查 `grouper` 和 `axis` 的长度是否相同，否则引发 ValueError
        if len(grouper) != len(axis):
            raise ValueError("Grouper and axis must be same length")

        # 如果 `grouper` 是列表或元组，则转换为安全的数组并返回
        if isinstance(grouper, (list, tuple)):
            grouper = com.asarray_tuplesafe(grouper)
        return grouper
    # 如果以上条件均不满足，则直接返回 `grouper`
    else:
        return grouper
```