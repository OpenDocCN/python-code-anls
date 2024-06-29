# `D:\src\scipysrc\pandas\pandas\core\dtypes\dtypes.py`

```
"""
Define extension dtypes.
"""

# 导入必要的模块和类
from __future__ import annotations

from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from decimal import Decimal
import re
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
import warnings

import numpy as np
import pytz

from pandas._config.config import get_option

# 导入 pandas 内部库
from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas._libs.interval import Interval
from pandas._libs.properties import cache_readonly
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    NaTType,
    Period,
    Timedelta,
    Timestamp,
    timezones,
    to_offset,
    tz_compare,
)
from pandas._libs.tslibs.dtypes import (
    PeriodDtypeBase,
    abbrev_to_npy_unit,
)
from pandas._libs.tslibs.offsets import BDay
from pandas.compat import pa_version_under10p1
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level

# 导入 pandas 核心数据类型相关模块和类
from pandas.core.dtypes.base import (
    ExtensionDtype,
    StorageExtensionDtype,
    register_extension_dtype,
)
from pandas.core.dtypes.generic import (
    ABCCategoricalIndex,
    ABCIndex,
    ABCRangeIndex,
)
from pandas.core.dtypes.inference import (
    is_bool,
    is_list_like,
)

# 如果未安装较旧版本的 pyarrow，导入 pyarrow 库
if not pa_version_under10p1:
    import pyarrow as pa

# 如果是类型检查阶段，导入额外的类型
if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from datetime import tzinfo

    import pyarrow as pa  # noqa: TCH004

    from pandas._typing import (
        Dtype,
        DtypeObj,
        IntervalClosedType,
        Ordered,
        Scalar,
        Self,
        npt,
        type_t,
    )

    from pandas import (
        Categorical,
        CategoricalIndex,
        DatetimeIndex,
        Index,
        IntervalIndex,
        PeriodIndex,
    )
    from pandas.core.arrays import (
        BaseMaskedArray,
        DatetimeArray,
        IntervalArray,
        NumpyExtensionArray,
        PeriodArray,
        SparseArray,
    )
    from pandas.core.arrays.arrow import ArrowExtensionArray

# 设置字符串类型的别名
str_type = str


class PandasExtensionDtype(ExtensionDtype):
    """
    A np.dtype duck-typed class, suitable for holding a custom dtype.

    THIS IS NOT A REAL NUMPY DTYPE
    """

    type: Any
    kind: Any
    # The Any type annotations above are here only because mypy seems to have a
    # problem dealing with multiple inheritance from PandasExtensionDtype
    # and ExtensionDtype's @properties in the subclasses below. The kind and
    # type variables in those subclasses are explicitly typed below.
    subdtype = None
    str: str_type
    num = 100
    shape: tuple[int, ...] = ()
    itemsize = 8
    base: DtypeObj | None = None
    isbuiltin = 0
    isnative = 0
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}

    def __repr__(self) -> str_type:
        """
        Return a string representation for a particular object.
        """
        return str(self)
    # 定义对象的哈希方法，子类应该实现一个 __hash__ 方法
    def __hash__(self) -> int:
        raise NotImplementedError("sub-classes should implement an __hash__ method")

    # 返回对象的状态字典以供 pickle 使用；不包含缓存项
    def __getstate__(self) -> dict[str_type, Any]:
        return {k: getattr(self, k, None) for k in self._metadata}

    # 类方法：重置缓存，清空缓存中的数据类型信息字典
    @classmethod
    def reset_cache(cls) -> None:
        """clear the cache"""
        cls._cache_dtypes = {}
class CategoricalDtypeType(type):
    """
    定义CategoricalDtype的类型，这个元类确定子类的能力
    """


@register_extension_dtype
class CategoricalDtype(PandasExtensionDtype, ExtensionDtype):
    """
    表示具有类别和顺序性的分类数据类型。

    Parameters
    ----------
    categories : sequence, optional
        必须是唯一的，不能包含任何空值。
        类别存储在一个索引中，如果提供了索引，将使用该索引的dtype。
    ordered : bool or None, default False
        是否将此分类视为有序分类。
        None可用于在操作中保持现有分类的有序值，例如astype，
        如果没有现有的有序值，则解析为False。

    Attributes
    ----------
    categories
    ordered

    Methods
    -------
    None

    See Also
    --------
    Categorical : 以经典的R / S-plus方式表示分类变量。

    Notes
    -----
    此类用于独立于值指定“Categorical”的类型。参见 :ref:`categorical.categoricaldtype`
    了解更多信息。

    Examples
    --------
    >>> t = pd.CategoricalDtype(categories=["b", "a"], ordered=True)
    >>> pd.Series(["a", "b", "a", "c"], dtype=t)
    0      a
    1      b
    2      a
    3    NaN
    dtype: category
    Categories (2, object): ['b' < 'a']

    可通过提供空索引创建具有特定dtype的空CategoricalDtype。如下所示，

    >>> pd.CategoricalDtype(pd.DatetimeIndex([])).categories.dtype
    dtype('<M8[s]')
    """

    # TODO: Document public vs. private API
    name = "category"
    type: type[CategoricalDtypeType] = CategoricalDtypeType
    kind: str_type = "O"
    str = "|O08"
    base = np.dtype("O")
    _metadata = ("categories", "ordered")
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}
    _supports_2d = False
    _can_fast_transpose = False

    def __init__(self, categories=None, ordered: Ordered = False) -> None:
        self._finalize(categories, ordered, fastpath=False)

    @classmethod
    def _from_fastpath(
        cls, categories=None, ordered: bool | None = None
    ) -> CategoricalDtype:
        """
        从快速路径创建CategoricalDtype对象。

        Parameters
        ----------
        categories : sequence, optional
            分类的类别。
        ordered : bool or None, default None
            是否有序。

        Returns
        -------
        CategoricalDtype
            返回新创建的CategoricalDtype对象。
        """
        self = cls.__new__(cls)
        self._finalize(categories, ordered, fastpath=True)
        return self

    @classmethod
    def _from_categorical_dtype(
        cls, dtype: CategoricalDtype, categories=None, ordered: Ordered | None = None
    ) -> CategoricalDtype:
        """
        从现有的CategoricalDtype对象创建新的CategoricalDtype对象。

        Parameters
        ----------
        dtype : CategoricalDtype
            现有的CategoricalDtype对象。
        categories : sequence, optional
            分类的类别。
        ordered : bool or None, default None
            是否有序。

        Returns
        -------
        CategoricalDtype
            返回新创建的CategoricalDtype对象。
        """
        if categories is ordered is None:
            return dtype
        if categories is None:
            categories = dtype.categories
        if ordered is None:
            ordered = dtype.ordered
        return cls(categories, ordered)

    @classmethod
    @classmethod
    def construct_from_string(cls, string: str_type) -> CategoricalDtype:
        """
        Construct a CategoricalDtype from a string.

        Parameters
        ----------
        string : str
            Must be the string "category" in order to be successfully constructed.

        Returns
        -------
        CategoricalDtype
            Instance of the dtype.

        Raises
        ------
        TypeError
            If a CategoricalDtype cannot be constructed from the input.
        """
        # 检查输入参数是否为字符串类型
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        # 检查字符串是否与当前类的名称匹配，如果不匹配则引发类型错误
        if string != cls.name:
            raise TypeError(f"Cannot construct a 'CategoricalDtype' from '{string}'")

        # 需要设置 ordered=None，以确保对 dtype="category" 的操作不会覆盖现有分类数据的有序性
        return cls(ordered=None)

    def _finalize(self, categories, ordered: Ordered, fastpath: bool = False) -> None:
        if ordered is not None:
            # 如果提供了 ordered 参数，则验证其值是否符合要求
            self.validate_ordered(ordered)

        if categories is not None:
            # 如果提供了 categories 参数，则验证并设置有效的分类数据
            categories = self.validate_categories(categories, fastpath=fastpath)

        # 将验证后的 categories 和 ordered 分配给实例的私有属性
        self._categories = categories
        self._ordered = ordered

    def __setstate__(self, state: MutableMapping[str_type, Any]) -> None:
        # 为了兼容 pickle，需要在 PandasExtensionDtype 超类中定义 __get_state__，
        # 并使用公共属性进行 pickle。需要在这里设置私有可设置的属性 (见 GH26067)。
        self._categories = state.pop("categories", None)
        self._ordered = state.pop("ordered", False)

    def __hash__(self) -> int:
        # _hash_categories 返回一个 uint64，因此使用负空间来处理未知分类，避免冲突
        if self.categories is None:
            if self.ordered:
                return -1
            else:
                return -2
        # 确保在哈希计算中包含真实的 self.ordered 值
        return int(self._hash_categories)
    def __eq__(self, other: object) -> bool:
        """
        Rules for CDT equality:
        1) Any CDT is equal to the string 'category'
        2) Any CDT is equal to itself
        3) Any CDT is equal to a CDT with categories=None regardless of ordered
        4) A CDT with ordered=True is only equal to another CDT with
           ordered=True and identical categories in the same order
        5) A CDT with ordered={False, None} is only equal to another CDT with
           ordered={False, None} and identical categories, but same order is
           not required. There is no distinction between False/None.
        6) Any other comparison returns False
        """
        # Check if `other` is a string and equal to the name of the CDT
        if isinstance(other, str):
            return other == self.name
        # Check if `other` is the same object as `self`
        elif other is self:
            return True
        # Check if `other` has the attributes 'ordered' and 'categories'
        elif not (hasattr(other, "ordered") and hasattr(other, "categories")):
            return False
        # Handle cases where `self.categories` or `other.categories` is None
        elif self.categories is None or other.categories is None:
            # For cases where categories are None, only equal if both are None
            return self.categories is other.categories
        # Handle cases where `self.ordered` or `other.ordered` is True
        elif self.ordered or other.ordered:
            # Both must have ordered=True and identical categories in the same order
            return (self.ordered == other.ordered) and self.categories.equals(
                other.categories
            )
        else:
            # Cases where neither `self.ordered` nor `other.ordered` is True
            left = self.categories
            right = other.categories

            # Check for quick mismatches based on dtype
            if not left.dtype == right.dtype:
                return False

            # Check if lengths of categories are different
            if len(left) != len(right):
                return False

            # Check if categories are identical (same elements in potentially different order)
            if self.categories.equals(other.categories):
                return True

            # For non-object dtype, use indexer to check bijection between `left` and `right`
            if left.dtype != object:
                indexer = left.get_indexer(right)
                return (indexer != -1).all()

            # For object dtype, compare sets of categories
            return set(left) == set(right)
    def __repr__(self) -> str_type:
        # 如果没有定义分类(categories)，则将数据和数据类型设置为 "None"
        if self.categories is None:
            data = "None"
            dtype = "None"
        else:
            # 否则，格式化分类数据并移除末尾的逗号和空格
            data = self.categories._format_data(name=type(self).__name__)
            # 如果分类是范围索引(ABCRangeIndex)类型，则直接使用范围索引的字符串表示
            if isinstance(self.categories, ABCRangeIndex):
                data = str(self.categories._range)
            data = data.rstrip(", ")
            dtype = self.categories.dtype

        # 返回表示该对象的字符串，包括分类数据、是否有序和分类数据类型
        return (
            f"CategoricalDtype(categories={data}, ordered={self.ordered}, "
            f"categories_dtype={dtype})"
        )

    @cache_readonly
    def _hash_categories(self) -> int:
        from pandas.core.util.hashing import (
            combine_hash_arrays,
            hash_array,
            hash_tuples,
        )

        # 获取分类和是否有序的属性
        categories = self.categories
        ordered = self.ordered

        if len(categories) and isinstance(categories[0], tuple):
            # 如果第一个分类是元组，则假设所有分类都是元组，使用 hash_tuples 进行哈希
            cat_list = list(categories)  # 如果是 np.array 类型的分类则会中断
            cat_array = hash_tuples(cat_list)
        else:
            if categories.dtype == "O" and len({type(x) for x in categories}) != 1:
                # 如果分类的 dtype 是对象并且包含不同类型的元素，则暂时无法处理，返回哈希值
                hashed = hash((tuple(categories), ordered))
                return hashed

            if DatetimeTZDtype.is_dtype(categories.dtype):
                # 如果是日期时间类型，则转换为 datetime64[ns] 类型以避免未来警告
                categories = categories.view("datetime64[ns]")

            # 对分类数组进行哈希处理，categorize=False 表示不进行分类处理
            cat_array = hash_array(np.asarray(categories), categorize=False)

        if ordered:
            # 如果分类是有序的，则将分类数组与从 0 到数组长度的整数数组垂直堆叠
            cat_array = np.vstack(
                [cat_array, np.arange(len(cat_array), dtype=cat_array.dtype)]
            )
        else:
            # 如果分类是无序的，则将分类数组转换为单行数组
            cat_array = np.array([cat_array])

        # 将组合的哈希数组进行混合处理并返回最终的位异或结果
        combined_hashed = combine_hash_arrays(iter(cat_array), num_items=len(cat_array))
        return np.bitwise_xor.reduce(combined_hashed)

    @classmethod
    def construct_array_type(cls) -> type_t[Categorical]:
        """
        返回与此数据类型关联的数组类型。

        Returns
        -------
        type
            pandas 中的 Categorical 类型
        """
        from pandas import Categorical

        return Categorical

    @staticmethod
    def validate_ordered(ordered: Ordered) -> None:
        """
        验证是否有有效的有序参数。如果不是布尔值，则引发 TypeError。

        Parameters
        ----------
        ordered : object
            要验证的参数。

        Raises
        ------
        TypeError
            如果 'ordered' 不是布尔值。
        """
        # 如果 ordered 不是布尔值，则引发类型错误异常
        if not is_bool(ordered):
            raise TypeError("'ordered' must either be 'True' or 'False'")
    @staticmethod
    def validate_categories(categories, fastpath: bool = False) -> Index:
        """
        Validates that we have good categories

        Parameters
        ----------
        categories : array-like
            The input categories to be validated, expected to be array-like.
        fastpath : bool
            Whether to skip nan and uniqueness checks. Defaults to False.

        Returns
        -------
        categories : Index
            Validated and potentially transformed categories as an Index object.
        """
        from pandas.core.indexes.base import Index

        if not fastpath and not is_list_like(categories):
            # Raise an error if categories is not list-like and fastpath is not enabled
            raise TypeError(
                f"Parameter 'categories' must be list-like, was {categories!r}"
            )
        
        if not isinstance(categories, ABCIndex):
            # Convert categories to an Index object if it's not already of type ABCIndex
            categories = Index._with_infer(categories, tupleize_cols=False)

        if not fastpath:
            # Perform additional checks if fastpath is False
            if categories.hasnans:
                # Raise an error if categories contain NaN values
                raise ValueError("Categorical categories cannot be null")

            if not categories.is_unique:
                # Raise an error if categories are not unique
                raise ValueError("Categorical categories must be unique")

        if isinstance(categories, ABCCategoricalIndex):
            # Extract the categories from ABCCategoricalIndex if categories is of this type
            categories = categories.categories

        return categories

    def update_dtype(self, dtype: str_type | CategoricalDtype) -> CategoricalDtype:
        """
        Returns a CategoricalDtype with categories and ordered taken from dtype
        if specified, otherwise falling back to self if unspecified

        Parameters
        ----------
        dtype : CategoricalDtype or str
            The dtype to update to. If str, should be 'category'.

        Returns
        -------
        new_dtype : CategoricalDtype
            Updated CategoricalDtype object.
        """
        if isinstance(dtype, str) and dtype == "category":
            # Return self if dtype is 'category' since it should not change anything
            return self
        elif not self.is_dtype(dtype):
            # Raise an error if dtype is not a valid CategoricalDtype
            raise ValueError(
                f"a CategoricalDtype must be passed to perform an update, "
                f"got {dtype!r}"
            )
        else:
            # dtype is now a valid CategoricalDtype
            dtype = cast(CategoricalDtype, dtype)

        # Update categories/ordered unless explicitly passed as None
        new_categories = (
            dtype.categories if dtype.categories is not None else self.categories
        )
        new_ordered = dtype.ordered if dtype.ordered is not None else self.ordered

        return CategoricalDtype(new_categories, new_ordered)

    @property
    def categories(self) -> Index:
        """
        An ``Index`` containing the unique categories allowed.

        Returns
        -------
        categories : Index
            Index object containing unique categories.
        """
        return self._categories

    @property
    # 返回当前分类类型对象是否具有有序关系的属性
    def ordered(self) -> Ordered:
        """
        Whether the categories have an ordered relationship.

        See Also
        --------
        categories : An Index containing the unique categories allowed.

        Examples
        --------
        >>> cat_type = pd.CategoricalDtype(categories=["a", "b"], ordered=True)
        >>> cat_type.ordered
        True

        >>> cat_type = pd.CategoricalDtype(categories=["a", "b"], ordered=False)
        >>> cat_type.ordered
        False
        """
        return self._ordered

    @property
    # 返回当前分类类型对象的 `categories` 是否为布尔类型的属性
    def _is_boolean(self) -> bool:
        from pandas.core.dtypes.common import is_bool_dtype

        return is_bool_dtype(self.categories)

    # 返回一组数据类型中共同的分类数据类型，或者返回 None
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # 检查是否所有的数据类型都是分类数据类型且具有相同的分类
        if all(isinstance(x, CategoricalDtype) for x in dtypes):
            first = dtypes[0]
            if all(first == other for other in dtypes[1:]):
                return first

        # 特殊情况：未初始化的分类数据类型
        non_init_cats = [
            isinstance(x, CategoricalDtype) and x.categories is None for x in dtypes
        ]
        if all(non_init_cats):
            return self
        elif any(non_init_cats):
            return None

        # 如果是 SparseDtype，则提取其子数据类型
        subtypes = (x.subtype if isinstance(x, SparseDtype) else x for x in dtypes)
        # 提取分类数据类型的分类数据类型
        non_cat_dtypes = [
            x.categories.dtype if isinstance(x, CategoricalDtype) else x
            for x in subtypes
        ]
        from pandas.core.dtypes.cast import find_common_type

        # 返回这些数据类型的最常见公共类型
        return find_common_type(non_cat_dtypes)

    @cache_readonly
    # 返回 CategoricalIndex 类型的索引类
    def index_class(self) -> type_t[CategoricalIndex]:
        from pandas import CategoricalIndex

        return CategoricalIndex
    @register_extension_dtype
    class DatetimeTZDtype(PandasExtensionDtype):
        """
        An ExtensionDtype for timezone-aware datetime data.

        **This is not an actual numpy dtype**, but a duck type.

        Parameters
        ----------
        unit : str, default "ns"
            The precision of the datetime data. Valid options are
            ``"s"``, ``"ms"``, ``"us"``, ``"ns"``.
        tz : str, int, or datetime.tzinfo
            The timezone.

        Attributes
        ----------
        unit
        tz

        Methods
        -------
        None

        Raises
        ------
        ZoneInfoNotFoundError
            When the requested timezone cannot be found.

        See Also
        --------
        numpy.datetime64 : Numpy data type for datetime.
        datetime.datetime : Python datetime object.

        Examples
        --------
        >>> from zoneinfo import ZoneInfo
        >>> pd.DatetimeTZDtype(tz=ZoneInfo("UTC"))
        datetime64[ns, UTC]

        >>> pd.DatetimeTZDtype(tz=ZoneInfo("Europe/Paris"))
        datetime64[ns, Europe/Paris]
        """

        type: type[Timestamp] = Timestamp
        kind: str_type = "M"
        num = 101
        _metadata = ("unit", "tz")
        _match = re.compile(r"(datetime64|M8)\[(?P<unit>.+), (?P<tz>.+)\]")
        _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}
        _supports_2d = True
        _can_fast_transpose = True

        @property
        def na_value(self) -> NaTType:
            return NaT

        @cache_readonly
        def base(self) -> DtypeObj:  # type: ignore[override]
            return np.dtype(f"M8[{self.unit}]")

        # error: Signature of "str" incompatible with supertype "PandasExtensionDtype"
        @cache_readonly
        def str(self) -> str:  # type: ignore[override]
            return f"|M8[{self.unit}]"

        def __init__(self, unit: str_type | DatetimeTZDtype = "ns", tz=None) -> None:
            if isinstance(unit, DatetimeTZDtype):
                # error: "str" has no attribute "tz"
                unit, tz = unit.unit, unit.tz  # type: ignore[attr-defined]

            if unit != "ns":
                if isinstance(unit, str) and tz is None:
                    # maybe a string like datetime64[ns, tz], which we support for
                    # now.
                    result = type(self).construct_from_string(unit)
                    unit = result.unit
                    tz = result.tz
                    msg = (
                        f"Passing a dtype alias like 'datetime64[ns, {tz}]' "
                        "to DatetimeTZDtype is no longer supported. Use "
                        "'DatetimeTZDtype.construct_from_string()' instead."
                    )
                    raise ValueError(msg)
                if unit not in ["s", "ms", "us", "ns"]:
                    raise ValueError("DatetimeTZDtype only supports s, ms, us, ns units")

            if tz:
                tz = timezones.maybe_get_tz(tz)
                tz = timezones.tz_standardize(tz)
            elif tz is not None:
                raise pytz.UnknownTimeZoneError(tz)
            if tz is None:
                raise TypeError("A 'tz' is required.")

            self._unit = unit
            self._tz = tz

        @cache_readonly
        def _get_default_freq(cls) -> str:
            return "ns"
    def _creso(self) -> int:
        """
        返回与此数据类型的分辨率对应的NPY_DATETIMEUNIT。
        """
        return abbrev_to_npy_unit(self.unit)

    @property
    def unit(self) -> str_type:
        """
        返回日期时间数据的精度。

        See Also
        --------
        DatetimeTZDtype.tz : 获取时区信息。

        Examples
        --------
        >>> from zoneinfo import ZoneInfo
        >>> dtype = pd.DatetimeTZDtype(tz=ZoneInfo("America/Los_Angeles"))
        >>> dtype.unit
        'ns'
        """
        return self._unit

    @property
    def tz(self) -> tzinfo:
        """
        返回时区信息。

        See Also
        --------
        DatetimeTZDtype.unit : 获取日期时间数据的精度。

        Examples
        --------
        >>> from zoneinfo import ZoneInfo
        >>> dtype = pd.DatetimeTZDtype(tz=ZoneInfo("America/Los_Angeles"))
        >>> dtype.tz
        zoneinfo.ZoneInfo(key='America/Los_Angeles')
        """
        return self._tz

    @classmethod
    def construct_array_type(cls) -> type_t[DatetimeArray]:
        """
        返回与该数据类型相关联的数组类型。

        Returns
        -------
        type
        """
        from pandas.core.arrays import DatetimeArray

        return DatetimeArray

    @classmethod
    def construct_from_string(cls, string: str_type) -> DatetimeTZDtype:
        """
        从字符串构造DatetimeTZDtype对象。

        Parameters
        ----------
        string : str
            此DatetimeTZDtype的字符串别名。
            应该格式为 ``datetime64[ns, <tz>]``,
            其中 ``<tz>`` 是时区名称。

        Examples
        --------
        >>> DatetimeTZDtype.construct_from_string("datetime64[ns, UTC]")
        datetime64[ns, UTC]
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        msg = f"Cannot construct a 'DatetimeTZDtype' from '{string}'"
        match = cls._match.match(string)
        if match:
            d = match.groupdict()
            try:
                return cls(unit=d["unit"], tz=d["tz"])
            except (KeyError, TypeError, ValueError) as err:
                # 如果 maybe_get_tz 尝试并且无法获取 pytz 时区 (实际上是 pytz.UnknownTimeZoneError) 时出现 KeyError；
                # 如果传入了无效的时区时出现 TypeError；
                # 如果传入了除 "ns" 以外的单位时出现 ValueError。
                raise TypeError(msg) from err
        raise TypeError(msg)

    def __str__(self) -> str_type:
        """
        返回DatetimeTZDtype对象的字符串表示形式。
        """
        return f"datetime64[{self.unit}, {self.tz}]"

    @property
    def name(self) -> str_type:
        """返回该数据类型的字符串表示形式。"""
        return str(self)

    def __hash__(self) -> int:
        """
        使自身成为可哈希的对象。
        TODO: 更新此方法。
        """
        return hash(str(self))
    def __eq__(self, other: object) -> bool:
        # 如果 other 是字符串类型
        if isinstance(other, str):
            # 如果 other 以 "M8[" 开头，则修改其格式为 f"datetime64[{other[3:]}]"
            if other.startswith("M8["):
                other = f"datetime64[{other[3:]}"
            # 返回比较结果 other 是否等于 self.name
            return other == self.name

        # 如果 other 不是字符串类型，则执行以下逻辑
        return (
            # 如果 other 是 DatetimeTZDtype 类型
            isinstance(other, DatetimeTZDtype)
            # 并且 self.unit 等于 other.unit
            and self.unit == other.unit
            # 并且比较时区的函数 tz_compare 返回 True
            and tz_compare(self.tz, other.tz)
        )

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> DatetimeArray:
        """
        Construct DatetimeArray from pyarrow Array/ChunkedArray.

        Note: If the units in the pyarrow Array are the same as this
        DatetimeDtype, then values corresponding to the integer representation
        of ``NaT`` (e.g. one nanosecond before :attr:`pandas.Timestamp.min`)
        are converted to ``NaT``, regardless of the null indicator in the
        pyarrow array.

        Parameters
        ----------
        array : pyarrow.Array or pyarrow.ChunkedArray
            The Arrow array to convert to DatetimeArray.

        Returns
        -------
        extension array : DatetimeArray
        """
        import pyarrow

        from pandas.core.arrays import DatetimeArray

        # 将 pyarrow Array 转换为指定单位的时间戳格式
        array = array.cast(pyarrow.timestamp(unit=self._unit), safe=True)

        # 根据 array 的类型将其转换为 numpy 数组
        if isinstance(array, pyarrow.Array):
            np_arr = array.to_numpy(zero_copy_only=False)
        else:
            np_arr = array.to_numpy()

        # 使用 DatetimeArray 类的 _simple_new 方法创建新的 DatetimeArray 对象，并返回
        return DatetimeArray._simple_new(np_arr, dtype=self)

    def __setstate__(self, state) -> None:
        # 用于 pickle 兼容性，从 state 中设置私有属性 _tz 和 _unit
        self._tz = state["tz"]
        self._unit = state["unit"]

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # 如果所有的 dtypes 都是 DatetimeTZDtype 类型，并且它们的时区都等于 self 的时区
        if all(isinstance(t, DatetimeTZDtype) and t.tz == self.tz for t in dtypes):
            # 找出 dtypes 中的最大 np.dtype，并获取其单位
            np_dtype = np.max([cast(DatetimeTZDtype, t).base for t in [self, *dtypes]])
            unit = np.datetime_data(np_dtype)[0]
            # 返回一个新的 DatetimeTZDtype 对象，单位为 unit，时区为 self 的时区
            return type(self)(unit=unit, tz=self.tz)
        # 如果上述条件不满足，则调用父类的 _get_common_dtype 方法返回结果
        return super()._get_common_dtype(dtypes)

    @cache_readonly
    def index_class(self) -> type_t[DatetimeIndex]:
        from pandas import DatetimeIndex

        # 返回 DatetimeIndex 类型
        return DatetimeIndex
@register_extension_dtype
class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype):
    """
    An ExtensionDtype for Period data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    freq : str or DateOffset
        The frequency of this PeriodDtype.

    Attributes
    ----------
    freq

    Methods
    -------
    None

    Examples
    --------
    >>> pd.PeriodDtype(freq="D")
    period[D]

    >>> pd.PeriodDtype(freq=pd.offsets.MonthEnd())
    period[M]
    """

    type: type[Period] = Period  # 指定 Period 类型
    kind: str_type = "O"  # 指定 dtype 的 kind
    str = "|O08"  # dtype 对象的字符串表示形式
    base = np.dtype("O")  # 基本的 numpy dtype
    num = 102  # dtype 对象的编号
    _metadata = ("freq",)  # 元数据，存储频率信息

    # error: Incompatible types in assignment (expression has type
    # "Dict[int, PandasExtensionDtype]", base class "PandasExtensionDtype"
    # defined the type as "Dict[str, PandasExtensionDtype]")  [assignment]
    _cache_dtypes: dict[BaseOffset, int] = {}  # type: ignore[assignment]
    # 缓存不同频率对应的 dtype 编号的字典，忽略类型检查

    __hash__ = PeriodDtypeBase.__hash__  # 继承基类 PeriodDtypeBase 的哈希方法

    _freq: BaseOffset  # 周期频率对象
    _supports_2d = True  # 支持二维操作
    _can_fast_transpose = True  # 可以进行快速转置

    def __new__(cls, freq) -> PeriodDtype:  # noqa: PYI034
        """
        Parameters
        ----------
        freq : PeriodDtype, BaseOffset, or string
        """
        if isinstance(freq, PeriodDtype):
            return freq

        if not isinstance(freq, BaseOffset):
            freq = cls._parse_dtype_strict(freq)  # 解析频率字符串

        if isinstance(freq, BDay):
            # GH#53446
            # TODO(3.0): enforcing this will close GH#10575
            warnings.warn(
                "PeriodDtype[B] is deprecated and will be removed in a future "
                "version. Use a DatetimeIndex with freq='B' instead",
                FutureWarning,
                stacklevel=find_stack_level(),  # 获取当前堆栈层级
            )

        try:
            dtype_code = cls._cache_dtypes[freq]  # 尝试从缓存中获取 dtype 编号
        except KeyError:
            dtype_code = freq._period_dtype_code  # 获取频率对象的 dtype 编号
            cls._cache_dtypes[freq] = dtype_code  # 将新的频率对象和对应的 dtype 编号加入缓存
        u = PeriodDtypeBase.__new__(cls, dtype_code, freq.n)  # 创建新的 PeriodDtypeBase 对象
        u._freq = freq  # 设置频率对象
        return u  # 返回新创建的 PeriodDtype 对象

    def __reduce__(self) -> tuple[type_t[Self], tuple[str_type]]:
        return type(self), (self.name,)  # 返回用于 pickle 序列化的数据

    @property
    def freq(self) -> BaseOffset:
        """
        The frequency object of this PeriodDtype.

        Examples
        --------
        >>> dtype = pd.PeriodDtype(freq="D")
        >>> dtype.freq
        <Day>
        """
        return self._freq  # 返回周期频率对象

    @classmethod
    # 解析严格的数据类型字符串，返回对应的时间偏移对象
    def _parse_dtype_strict(cls, freq: str_type) -> BaseOffset:
        if isinstance(freq, str):  # 注意：freq 已经是字符串类型！
            # 如果字符串以 "Period[" 或 "period[" 开头，则尝试匹配频率字符串
            if freq.startswith(("Period[", "period[")):
                m = cls._match.search(freq)
                if m is not None:
                    freq = m.group("freq")

            # 将频率字符串转换为时间偏移对象
            freq_offset = to_offset(freq, is_period=True)
            if freq_offset is not None:
                return freq_offset

        # 如果不是字符串或无法转换为时间偏移对象，则抛出类型错误
        raise TypeError(
            "PeriodDtype argument should be string or BaseOffset, "
            f"got {type(freq).__name__}"
        )

    @classmethod
    def construct_from_string(cls, string: str_type) -> PeriodDtype:
        """
        从字符串严格构造 PeriodDtype 对象，如果无法构造则抛出 TypeError
        """
        if (
            isinstance(string, str)
            and (string.startswith(("period[", "Period[")))
            or isinstance(string, BaseOffset)
        ):
            # 避免像 'U' 这样的字符串被误认为 period[U]
            # 避免元组被视为频率
            try:
                return cls(freq=string)
            except ValueError:
                pass
        if isinstance(string, str):
            msg = f"Cannot construct a 'PeriodDtype' from '{string}'"
        else:
            msg = f"'construct_from_string' expects a string, got {type(string)}"
        raise TypeError(msg)

    def __str__(self) -> str_type:
        # 返回对象的名称表示为字符串
        return self.name

    @property
    def name(self) -> str_type:
        # 返回对象的名称属性，格式为 period[频率字符串]
        return f"period[{self._freqstr}]"

    @property
    def na_value(self) -> NaTType:
        # 返回该类型的缺失值表示
        return NaT

    def __eq__(self, other: object) -> bool:
        # 判断当前对象是否与另一个对象相等
        if isinstance(other, str):
            return other[:1].lower() + other[1:] == self.name

        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        # 判断当前对象是否与另一个对象不相等
        return not self.__eq__(other)

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        判断传入的对象是否为有效的 dtype 类型（可以通过字符串或类型进行匹配）
        """
        if isinstance(dtype, str):
            # PeriodDtype 可以从频率字符串如 "U" 实例化，但不将 freq 字符串 "U" 视为 dtype
            if dtype.startswith(("period[", "Period[")):
                try:
                    return cls._parse_dtype_strict(dtype) is not None
                except ValueError:
                    return False
            else:
                return False
        return super().is_dtype(dtype)

    @classmethod
    def construct_array_type(cls) -> type_t[PeriodArray]:
        """
        返回与此 dtype 关联的数组类型 PeriodArray
        """
        from pandas.core.arrays import PeriodArray

        return PeriodArray
    # 从 pyarrow Array/ChunkedArray 构建 PeriodArray。
    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> PeriodArray:
        """
        Construct PeriodArray from pyarrow Array/ChunkedArray.
        """
        # 导入必要的模块和类
        import pyarrow
        from pandas.core.arrays import PeriodArray
        from pandas.core.arrays.arrow._arrow_utils import (
            pyarrow_array_to_numpy_and_mask,
        )

        # 根据输入类型确定数据块
        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            chunks = array.chunks

        # 初始化结果列表
        results = []
        # 遍历每个数据块
        for arr in chunks:
            # 转换 pyarrow Array 到 NumPy 数组和掩码数组
            data, mask = pyarrow_array_to_numpy_and_mask(arr, dtype=np.dtype(np.int64))
            # 使用数据创建 PeriodArray 对象，copy=False 表示不复制数据
            parr = PeriodArray(data.copy(), dtype=self, copy=False)
            # 错误: 为 "PeriodArray" 使用了无效的索引类型 "ndarray[Any, dtype[bool_]]"；
            # 预期类型为 "Union[int, Sequence[int], Sequence[bool], slice]"
            parr[~mask] = NaT  # type: ignore[index]
            # 将 PeriodArray 添加到结果列表
            results.append(parr)

        # 如果结果列表为空，则返回一个空的 PeriodArray 对象
        if not results:
            return PeriodArray(np.array([], dtype="int64"), dtype=self, copy=False)
        # 合并同一类型的 PeriodArray 对象并返回
        return PeriodArray._concat_same_type(results)

    # 返回 PeriodIndex 类型对象的缓存只读属性
    @cache_readonly
    def index_class(self) -> type_t[PeriodIndex]:
        """
        Returns the cached read-only property for the PeriodIndex class.
        """
        # 导入 PeriodIndex 类
        from pandas import PeriodIndex

        # 返回 PeriodIndex 类型对象
        return PeriodIndex
# 注册为 Pandas 扩展数据类型的装饰器函数
@register_extension_dtype
# 定义 IntervalDtype 类，继承自 PandasExtensionDtype
class IntervalDtype(PandasExtensionDtype):
    """
    An ExtensionDtype for Interval data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    subtype : str, np.dtype
        The dtype of the Interval bounds.
    closed : {'right', 'left', 'both', 'neither'}, default 'right'
        Whether the interval is closed on the left-side, right-side, both or
        neither. See the Notes for more detailed explanation.

    Attributes
    ----------
    subtype : str or np.dtype
        The data type of the Interval bounds.
    """

    # 类的名称属性为 "interval"
    name = "interval"
    # 类的 kind 属性为字符串类型 "O"
    kind: str_type = "O"
    # 类的 str 属性为 "|O08"
    str = "|O08"
    # 类的基础类型为 numpy 的对象类型 "O"
    base = np.dtype("O")
    # 类的编号为 103
    num = 103
    # 类的元数据属性包括 subtype 和 closed
    _metadata = (
        "subtype",
        "closed",
    )

    # 正则表达式模式，用于匹配 IntervalDtype 字符串表示形式
    _match = re.compile(
        r"(I|i)nterval\[(?P<subtype>[^,]+(\[.+\])?)"
        r"(, (?P<closed>(right|left|both|neither)))?\]"
    )

    # 缓存不同 str_type 类型的 PandasExtensionDtype 对象
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}
    # 子类型的数据类型，初始为 None
    _subtype: None | np.dtype
    # 区间的闭合类型，初始为 None
    _closed: IntervalClosedType | None
    # 初始化方法，用于创建 IntervalDtype 对象
    def __init__(self, subtype=None, closed: IntervalClosedType | None = None) -> None:
        # 导入必要的模块和函数
        from pandas.core.dtypes.common import (
            is_string_dtype,
            pandas_dtype,
        )

        # 检查 closed 参数是否有效
        if closed is not None and closed not in {"right", "left", "both", "neither"}:
            raise ValueError("closed must be one of 'right', 'left', 'both', 'neither'")

        # 如果 subtype 是 IntervalDtype 类型，则根据 closed 参数进行匹配
        if isinstance(subtype, IntervalDtype):
            if closed is not None and closed != subtype.closed:
                raise ValueError(
                    "dtype.closed and 'closed' do not match. "
                    "Try IntervalDtype(dtype.subtype, closed) instead."
                )
            # 设置当前对象的 subtype 和 closed 属性
            self._subtype = subtype._subtype
            self._closed = subtype._closed
        elif subtype is None:
            # 如果 subtype 为 None，则作为空构造函数调用，通常为了兼容 pickle
            self._subtype = None
            self._closed = closed
        elif isinstance(subtype, str) and subtype.lower() == "interval":
            # 如果 subtype 是字符串 "interval"，则设置 subtype 为 None，closed 为指定的 closed 参数
            self._subtype = None
            self._closed = closed
        else:
            # 处理其他情况的 subtype
            if isinstance(subtype, str):
                # 使用正则表达式匹配 subtype
                m = IntervalDtype._match.search(subtype)
                if m is not None:
                    gd = m.groupdict()
                    subtype = gd["subtype"]
                    if gd.get("closed", None) is not None:
                        if closed is not None:
                            # 检查 closed 参数是否与 dtype 字符串中的值匹配
                            if closed != gd["closed"]:
                                raise ValueError(
                                    "'closed' keyword does not match value "
                                    "specified in dtype string"
                                )
                        closed = gd["closed"]  # type: ignore[assignment]

            try:
                # 尝试根据 subtype 构造 pandas 数据类型
                subtype = pandas_dtype(subtype)
            except TypeError as err:
                raise TypeError("could not construct IntervalDtype") from err
            # 检查 subtype 是否为分类或字符串类型，这些类型不支持 IntervalDtype
            if CategoricalDtype.is_dtype(subtype) or is_string_dtype(subtype):
                # 抛出类型错误异常
                msg = (
                    "category, object, and string subtypes are not supported "
                    "for IntervalDtype"
                )
                raise TypeError(msg)
            # 设置当前对象的 subtype 和 closed 属性
            self._subtype = subtype
            self._closed = closed

    # 缓存修饰符，用于表示只读的 _can_hold_na 方法
    @cache_readonly
    def _can_hold_na(self) -> bool:
        # 获取当前对象的 subtype 属性
        subtype = self._subtype
        # 如果 subtype 为 None，表示对象部分初始化，抛出未实现的异常
        if subtype is None:
            raise NotImplementedError(
                "_can_hold_na is not defined for partially-initialized IntervalDtype"
            )
        # 如果 subtype 的种类为整数或无符号整数，返回 False；否则返回 True
        if subtype.kind in "iu":
            return False
        return True

    # 属性方法，用于获取对象的 closed 属性
    @property
    def closed(self) -> IntervalClosedType:
        return self._closed  # type: ignore[return-value]

    # 属性方法，用于获取对象的 subtype 属性
    @property
    def subtype(self):
        """
        The dtype of the Interval bounds.

        See Also
        --------
        IntervalDtype: An ExtensionDtype for Interval data.

        Examples
        --------
        >>> dtype = pd.IntervalDtype(subtype="int64", closed="both")
        >>> dtype.subtype
        dtype('int64')
        """
        # 返回 Interval 对象的 subtype 属性
        return self._subtype

    @classmethod
    def construct_array_type(cls) -> type[IntervalArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        # 导入 IntervalArray 类并返回其类型
        from pandas.core.arrays import IntervalArray
        return IntervalArray

    @classmethod
    def construct_from_string(cls, string: str_type) -> IntervalDtype:
        """
        attempt to construct this type from a string, raise a TypeError
        if its not possible
        """
        # 如果输入不是字符串，抛出 TypeError 异常
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        # 检查字符串是否是 "interval" 或者符合格式 Interval[dtype]
        if string.lower() == "interval" or cls._match.search(string) is not None:
            return cls(string)

        # 如果字符串格式不正确，抛出详细的 TypeError 异常
        msg = (
            f"Cannot construct a 'IntervalDtype' from '{string}'.\n\n"
            "Incorrectly formatted string passed to constructor. "
            "Valid formats include Interval or Interval[dtype] "
            "where dtype is numeric, datetime, or timedelta"
        )
        raise TypeError(msg)

    @property
    def type(self) -> type[Interval]:
        # 返回 Interval 类型
        return Interval

    def __str__(self) -> str_type:
        # 返回 IntervalDtype 对象的字符串表示形式
        if self.subtype is None:
            return "interval"
        if self.closed is None:
            # 只部分初始化的情况下返回的字符串，可能会有 GH#38394
            return f"interval[{self.subtype}]"
        return f"interval[{self.subtype}, {self.closed}]"

    def __hash__(self) -> int:
        # 返回对象的哈希值，使得对象可以作为字典的键
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        # 比较对象是否相等
        if isinstance(other, str):
            return other.lower() in (self.name.lower(), str(self).lower())
        elif not isinstance(other, IntervalDtype):
            return False
        elif self.subtype is None or other.subtype is None:
            # None 应该与任何 subtype 匹配
            return True
        elif self.closed != other.closed:
            return False
        else:
            return self.subtype == other.subtype

    def __setstate__(self, state) -> None:
        # 用于 pickle 兼容性，__get_state__ 在 PandasExtensionDtype 超类中定义，
        # 使用公共属性进行 pickle，需要在此设置可设置的私有属性 (见 GH26067)
        self._subtype = state["subtype"]

        # 向后兼容旧的 pickle 可能没有 "closed" 键
        self._closed = state.pop("closed", None)

    @classmethod
    # 检查传入的类型是否是有效的数据类型，返回布尔值
    def is_dtype(cls, dtype: object) -> bool:
        """
        Return a boolean if we if the passed type is an actual dtype that we
        can match (via string or type)
        """
        # 如果传入的数据类型是字符串
        if isinstance(dtype, str):
            # 如果字符串以"interval"开头
            if dtype.lower().startswith("interval"):
                try:
                    # 尝试从字符串构造数据类型对象，并判断是否成功
                    return cls.construct_from_string(dtype) is not None
                except (ValueError, TypeError):
                    return False
            else:
                return False
        # 调用父类的 is_dtype 方法进行进一步检查
        return super().is_dtype(dtype)

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> IntervalArray:
        """
        Construct IntervalArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow

        from pandas.core.arrays import IntervalArray

        # 如果传入的是 pyarrow.Array 对象
        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            chunks = array.chunks

        results = []
        # 遍历数组的各个 chunk
        for arr in chunks:
            # 如果是扩展数组，获取其存储部分
            if isinstance(arr, pyarrow.ExtensionArray):
                arr = arr.storage
            # 从字段 "left" 和 "right" 中获取数据，作为区间的左右端点
            left = np.asarray(arr.field("left"), dtype=self.subtype)
            right = np.asarray(arr.field("right"), dtype=self.subtype)
            # 根据左右端点数组创建 IntervalArray 对象，并设置闭合属性
            iarr = IntervalArray.from_arrays(left, right, closed=self.closed)
            results.append(iarr)

        # 如果结果列表为空，返回一个空的 IntervalArray
        if not results:
            return IntervalArray.from_arrays(
                np.array([], dtype=self.subtype),
                np.array([], dtype=self.subtype),
                closed=self.closed,
            )
        # 合并结果列表中的 IntervalArray 对象，并返回
        return IntervalArray._concat_same_type(results)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # 如果不是所有的元素都是 IntervalDtype 类型，则返回 None
        if not all(isinstance(x, IntervalDtype) for x in dtypes):
            return None

        # 获取第一个元素的闭合属性
        closed = cast("IntervalDtype", dtypes[0]).closed
        # 如果不是所有元素的闭合属性都相同，则返回通用对象类型
        if not all(cast("IntervalDtype", x).closed == closed for x in dtypes):
            return np.dtype(object)

        from pandas.core.dtypes.cast import find_common_type

        # 找到所有 IntervalDtype 类型中的公共子类型
        common = find_common_type([cast("IntervalDtype", x).subtype for x in dtypes])
        # 如果公共子类型是 object，则返回通用对象类型
        if common == object:
            return np.dtype(object)
        # 返回包含公共子类型和闭合属性的 IntervalDtype 对象
        return IntervalDtype(common, closed=closed)

    @cache_readonly
    def index_class(self) -> type_t[IntervalIndex]:
        # 导入 IntervalIndex 类型并返回
        from pandas import IntervalIndex

        return IntervalIndex
    """
    A base class for defining custom Pandas ExtensionDtypes.

    This class is designed to be inherited and extended by specific dtype implementations.

    Attributes
    ----------
    base : NoneType
        Placeholder attribute for potential future use.
    type : type
        Type object representing the scalar type associated with this dtype.
    _internal_fill_value : Scalar
        Internal fill value used for masked arrays.

    Properties
    ----------
    _truthy_value : Scalar
        Determines the truthy value used for 'any' operations based on dtype kind.
    """

    # Fill values used for 'any'
    @property
    def _truthy_value(self):
        if self.kind == "f":
            return 1.0
        if self.kind in "iu":
            return 1
        return True
    def _falsey_value(self):
        # 如果对象类型为 'f'，返回浮点数 0.0
        if self.kind == "f":
            return 0.0
        # 如果对象类型为 'i' 或 'u'，返回整数 0
        if self.kind in "iu":
            return 0
        # 默认返回 False
        return False

    @property
    def na_value(self) -> libmissing.NAType:
        # 返回缺失值 NA 的实例
        return libmissing.NA

    @cache_readonly
    def numpy_dtype(self) -> np.dtype:
        """Return an instance of our numpy dtype"""
        # 返回当前对象的 numpy 数据类型实例
        return np.dtype(self.type)

    @cache_readonly
    def kind(self) -> str:
        # 返回当前对象的 numpy 数据类型的种类字符
        return self.numpy_dtype.kind

    @cache_readonly
    def itemsize(self) -> int:
        """Return the number of bytes in this dtype"""
        # 返回当前 numpy 数据类型所占字节数
        return self.numpy_dtype.itemsize

    @classmethod
    def construct_array_type(cls) -> type_t[BaseMaskedArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        # 抽象方法，返回与此 dtype 相关联的数组类型，由子类实现
        raise NotImplementedError

    @classmethod
    def from_numpy_dtype(cls, dtype: np.dtype) -> BaseMaskedDtype:
        """
        Construct the MaskedDtype corresponding to the given numpy dtype.
        """
        # 根据给定的 numpy dtype 构造对应的 MaskedDtype
        if dtype.kind == "b":
            from pandas.core.arrays.boolean import BooleanDtype

            return BooleanDtype()
        elif dtype.kind in "iu":
            from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE

            return NUMPY_INT_TO_DTYPE[dtype]
        elif dtype.kind == "f":
            from pandas.core.arrays.floating import NUMPY_FLOAT_TO_DTYPE

            return NUMPY_FLOAT_TO_DTYPE[dtype]
        else:
            # 抛出未实现的错误，提示尚未支持的 dtype 类型
            raise NotImplementedError(dtype)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # 解包所有的 MaskedDtype，找到它们共同的数据类型
        from pandas.core.dtypes.cast import find_common_type

        # 找到 dtypes 中所有非 MaskedDtype 的 numpy 数据类型，计算它们的共同类型
        new_dtype = find_common_type(
            [
                dtype.numpy_dtype if isinstance(dtype, BaseMaskedDtype) else dtype
                for dtype in dtypes
            ]
        )
        if not isinstance(new_dtype, np.dtype):
            # 如果找不到共同类型，返回 None
            return None
        try:
            # 尝试通过共同类型构造一个新的 MaskedDtype 实例
            return type(self).from_numpy_dtype(new_dtype)
        except (KeyError, NotImplementedError):
            # 处理可能的错误情况，返回 None
            return None
# 在 Pandas 中注册稀疏数据类型的装饰器，用于扩展数据类型的功能
@register_extension_dtype
class SparseDtype(ExtensionDtype):
    """
    Dtype for data stored in :class:`SparseArray`.

    ``SparseDtype`` is used as the data type for :class:`SparseArray`, enabling
    more efficient storage of data that contains a significant number of
    repetitive values typically represented by a fill value. It supports any
    scalar dtype as the underlying data type of the non-fill values.

    Parameters
    ----------
    dtype : str, ExtensionDtype, numpy.dtype, type, default numpy.float64
        The dtype of the underlying array storing the non-fill value values.
    fill_value : scalar, optional
        The scalar value not stored in the SparseArray. By default, this
        depends on ``dtype``.

        =========== ==========
        dtype       na_value
        =========== ==========
        float       ``np.nan``
        complex     ``np.nan``
        int         ``0``
        bool        ``False``
        datetime64  ``pd.NaT``
        timedelta64 ``pd.NaT``
        =========== ==========

        The default value may be overridden by specifying a ``fill_value``.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    arrays.SparseArray : The array structure that uses SparseDtype
        for data representation.

    Examples
    --------
    >>> ser = pd.Series([1, 0, 0], dtype=pd.SparseDtype(dtype=int, fill_value=0))
    >>> ser
    0    1
    1    0
    2    0
    dtype: Sparse[int64, 0]
    >>> ser.sparse.density
    0.3333333333333333
    """

    _is_immutable = True

    # We include `_is_na_fill_value` in the metadata to avoid hash collisions
    # between SparseDtype(float, 0.0) and SparseDtype(float, nan).
    # Without is_na_fill_value in the comparison, those would be equal since
    # hash(nan) is (sometimes?) 0.
    _metadata = ("_dtype", "_fill_value", "_is_na_fill_value")

    def __init__(self, dtype: Dtype = np.float64, fill_value: Any = None) -> None:
        # 如果传入的 dtype 是 SparseDtype 自身的实例，则将 fill_value 设为传入 dtype 的 fill_value
        if isinstance(dtype, type(self)):
            if fill_value is None:
                fill_value = dtype.fill_value
            dtype = dtype.subtype

        # 导入必要的函数和模块
        from pandas.core.dtypes.common import (
            is_string_dtype,
            pandas_dtype,
        )
        from pandas.core.dtypes.missing import na_value_for_dtype

        # 将 dtype 转换为 pandas 的 dtype 对象
        dtype = pandas_dtype(dtype)
        # 如果 dtype 是字符串类型，则将其转换为对象类型
        if is_string_dtype(dtype):
            dtype = np.dtype("object")
        # 如果 dtype 不是 numpy 的 dtype，则抛出类型错误
        if not isinstance(dtype, np.dtype):
            # GH#53160
            raise TypeError("SparseDtype subtype must be a numpy dtype")

        # 如果 fill_value 未指定，则根据 dtype 获取默认的 fill_value
        if fill_value is None:
            fill_value = na_value_for_dtype(dtype)

        # 初始化 SparseDtype 的属性
        self._dtype = dtype
        self._fill_value = fill_value
        self._check_fill_value()

    def __hash__(self) -> int:
        # Python3 不会继承 __hash__ 如果基类重写了 __eq__，因此在此显式地进行继承
        return super().__hash__()
    def __eq__(self, other: object) -> bool:
        # 重写 __eq__ 方法来处理 _metadata 中的 NA 值。
        # 基类进行简单的 == 检查，对于 NA 值会失败。
        
        # 如果 other 是字符串，则尝试使用 construct_from_string 方法转换
        if isinstance(other, str):
            try:
                other = self.construct_from_string(other)
            except TypeError:
                return False

        # 如果 other 是当前对象的同一类型
        if isinstance(other, type(self)):
            # 检查 subtype 是否相等
            subtype = self.subtype == other.subtype
            # 如果当前对象或者 other 对象的 fill_value 是 NA 值
            if self._is_na_fill_value or other._is_na_fill_value:
                # 这种情况较为复杂：
                # SparseDtype(float, float(nan)) == SparseDtype(float, np.nan)
                # SparseDtype(float, np.nan)     != SparseDtype(float, pd.NaT)
                # 我们希望任何浮点数的 NaN 被视为相等，但浮点数的 NaN 与日期时间的 NaT 不等。
                fill_value = isinstance(
                    self.fill_value, type(other.fill_value)
                ) or isinstance(other.fill_value, type(self.fill_value))
            else:
                # 忽略 numpy 的警告
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        "elementwise comparison failed",
                        category=DeprecationWarning,
                    )
                    # 检查 fill_value 是否相等
                    fill_value = self.fill_value == other.fill_value

            # 返回 subtype 和 fill_value 的比较结果
            return subtype and fill_value
        # 如果 other 不是当前对象的同一类型，则返回 False
        return False

    @property
    def fill_value(self):
        """
        The fill value of the array.

        Converting the SparseArray to a dense ndarray will fill the
        array with this value.

        .. warning::

           It's possible to end up with a SparseArray that has ``fill_value``
           values in ``sp_values``. This can occur, for example, when setting
           ``SparseArray.fill_value`` directly.
        """
        # 返回 SparseArray 的填充值
        return self._fill_value
    def _check_fill_value(self) -> None:
        # 检查填充值是否为标量，如果不是则引发 ValueError 异常
        if not lib.is_scalar(self._fill_value):
            raise ValueError(
                f"fill_value must be a scalar. Got {self._fill_value} instead"
            )

        # 导入必要的函数和模块
        from pandas.core.dtypes.cast import can_hold_element
        from pandas.core.dtypes.missing import (
            is_valid_na_for_dtype,
            isna,
        )

        from pandas.core.construction import ensure_wrapped_if_datetimelike

        # GH#23124 要求填充值和子类型必须匹配
        val = self._fill_value
        if isna(val):
            # 如果填充值是 NA，则检查其是否为 SparseDtype.subtype 的有效值
            if not is_valid_na_for_dtype(val, self.subtype):
                raise ValueError(
                    # GH#53043
                    "fill_value must be a valid value for the SparseDtype.subtype"
                )
        else:
            # 创建一个空的 NumPy 数组以验证填充值是否有效
            dummy = np.empty(0, dtype=self.subtype)
            dummy = ensure_wrapped_if_datetimelike(dummy)

            # 如果填充值不符合数组的要求，则引发 ValueError 异常
            if not can_hold_element(dummy, val):
                raise ValueError(
                    # GH#53043
                    "fill_value must be a valid value for the SparseDtype.subtype"
                )

    @property
    def _is_na_fill_value(self) -> bool:
        # 检查填充值是否为 NA
        from pandas import isna
        return isna(self.fill_value)

    @property
    def _is_numeric(self) -> bool:
        # 检查子类型是否为数值类型，而不是对象类型
        return not self.subtype == object

    @property
    def _is_boolean(self) -> bool:
        # 检查子类型是否为布尔类型
        return self.subtype.kind == "b"

    @property
    def kind(self) -> str:
        """
        稀疏类型。可能是 'integer' 或 'block'。
        """
        return self.subtype.kind

    @property
    def type(self):
        # 返回子类型的类型对象
        return self.subtype.type

    @property
    def subtype(self):
        # 返回当前稀疏类型的子类型
        return self._dtype

    @property
    def name(self) -> str:
        # 返回稀疏类型的名称，包括子类型和填充值
        return f"Sparse[{self.subtype.name}, {self.fill_value!r}]"

    def __repr__(self) -> str:
        # 返回稀疏类型的字符串表示形式
        return self.name

    @classmethod
    def construct_array_type(cls) -> type_t[SparseArray]:
        """
        返回与此 dtype 关联的数组类型。

        Returns
        -------
        type
        """
        # 导入 SparseArray 类并返回
        from pandas.core.arrays.sparse.array import SparseArray
        return SparseArray

    @classmethod
    def construct_from_string(cls, string: str) -> SparseDtype:
        """
        Construct a SparseDtype from a string form.

        Parameters
        ----------
        string : str
            Can take the following forms.

            string           dtype
            ================ ============================
            'int'            SparseDtype[np.int64, 0]
            'Sparse'         SparseDtype[np.float64, nan]
            'Sparse[int]'    SparseDtype[np.int64, 0]
            'Sparse[int, 0]' SparseDtype[np.int64, 0]
            ================ ============================

            It is not possible to specify non-default fill values
            with a string. An argument like ``'Sparse[int, 1]'``
            will raise a ``TypeError`` because the default fill value
            for integers is 0.

        Returns
        -------
        SparseDtype
            A SparseDtype object constructed based on the input string.

        Raises
        ------
        TypeError
            If the input `string` is not a valid string.
            If the construction fails due to non-default fill values.

        """
        # 检查输入是否为字符串，如果不是则抛出类型错误
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        
        msg = f"Cannot construct a 'SparseDtype' from '{string}'"
        
        # 如果字符串以"Sparse"开头
        if string.startswith("Sparse"):
            try:
                # 尝试解析子类型和是否有填充值
                sub_type, has_fill_value = cls._parse_subtype(string)
            except ValueError as err:
                # 如果解析失败，则抛出类型错误
                raise TypeError(msg) from err
            else:
                # 构建 SparseDtype 对象
                result = SparseDtype(sub_type)
                # 如果有非默认填充值并且结果不等于输入字符串，则抛出类型错误
                msg = (
                    f"Cannot construct a 'SparseDtype' from '{string}'.\n\nIt "
                    "looks like the fill_value in the string is not "
                    "the default for the dtype. Non-default fill_values "
                    "are not supported. Use the 'SparseDtype()' "
                    "constructor instead."
                )
                if has_fill_value and str(result) != string:
                    raise TypeError(msg)
                return result
        else:
            # 如果字符串不以"Sparse"开头，则抛出类型错误
            raise TypeError(msg)

    @staticmethod
    def _parse_subtype(dtype: str) -> tuple[str, bool]:
        """
        Parse a string to get the subtype

        Parameters
        ----------
        dtype : str
            A string like

            * Sparse[subtype]
            * Sparse[subtype, fill_value]

        Returns
        -------
        subtype : str
            The subtype extracted from the input string.
        has_fill_value : bool
            Indicates whether the string contains a fill value.

        Raises
        ------
        ValueError
            When the subtype cannot be extracted or the string format is invalid.
        """
        # 使用正则表达式解析字符串来提取子类型和填充值信息
        xpr = re.compile(r"Sparse\[(?P<subtype>[^,]*)(, )?(?P<fill_value>.*?)?\]$")
        m = xpr.match(dtype)
        has_fill_value = False
        if m:
            subtype = m.groupdict()["subtype"]
            has_fill_value = bool(m.groupdict()["fill_value"])
        elif dtype == "Sparse":
            subtype = "float64"
        else:
            raise ValueError(f"Cannot parse {dtype}")
        return subtype, has_fill_value

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        # 获取 dtype 对象的 dtype 属性，如果不存在则不变
        dtype = getattr(dtype, "dtype", dtype)
        # 如果 dtype 是字符串并且以 "Sparse" 开头
        if isinstance(dtype, str) and dtype.startswith("Sparse"):
            # 解析子类型并创建新的 np.dtype 对象
            sub_type, _ = cls._parse_subtype(dtype)
            dtype = np.dtype(sub_type)
        # 如果 dtype 是 cls 类型的实例，则返回 True
        elif isinstance(dtype, cls):
            return True
        # 返回 dtype 是否是 np.dtype 类型或者等于 "Sparse"
        return isinstance(dtype, np.dtype) or dtype == "Sparse"

    def update_dtype(self, dtype) -> SparseDtype:
        """
        Convert the SparseDtype to a new dtype.

        This takes care of converting the ``fill_value``.

        Parameters
        ----------
        dtype : Union[str, numpy.dtype, SparseDtype]
            The new dtype to use.

            * For a SparseDtype, it is simply returned
            * For a NumPy dtype (or str), the current fill value
              is converted to the new dtype, and a SparseDtype
              with `dtype` and the new fill value is returned.

        Returns
        -------
        SparseDtype
            A new SparseDtype with the correct `dtype` and fill value
            for that `dtype`.

        Raises
        ------
        ValueError
            When the current fill value cannot be converted to the
            new `dtype` (e.g. trying to convert ``np.nan`` to an
            integer dtype).


        Examples
        --------
        >>> SparseDtype(int, 0).update_dtype(float)
        Sparse[float64, 0.0]

        >>> SparseDtype(int, 1).update_dtype(SparseDtype(float, np.nan))
        Sparse[float64, nan]
        """
        from pandas.core.dtypes.astype import astype_array
        from pandas.core.dtypes.common import pandas_dtype

        # 获取当前对象的类型
        cls = type(self)
        # 将 dtype 转换为 pandas 的数据类型
        dtype = pandas_dtype(dtype)

        # 如果 dtype 不是当前类的实例
        if not isinstance(dtype, cls):
            # 如果 dtype 不是 np.dtype 类型，则抛出类型错误
            if not isinstance(dtype, np.dtype):
                raise TypeError("sparse arrays of extension dtypes not supported")

            # 将 fill_value 转换为指定 dtype，并取出第一个值作为 fill_value
            fv_asarray = np.atleast_1d(np.array(self.fill_value))
            fvarr = astype_array(fv_asarray, dtype)
            # 注意：不使用 fv_0d.item()，因为它将 dt64 转换为 int
            fill_value = fvarr[0]
            # 创建新的 SparseDtype 对象
            dtype = cls(dtype, fill_value=fill_value)

        return dtype

    @property
    def _subtype_with_str(self):
        """
        Whether the SparseDtype's subtype should be considered ``str``.

        Typically, pandas will store string data in an object-dtype array.
        When converting values to a dtype, e.g. in ``.astype``, we need to
        be more specific, we need the actual underlying type.

        Returns
        -------
        >>> SparseDtype(int, 1)._subtype_with_str
        dtype('int64')

        >>> SparseDtype(object, 1)._subtype_with_str
        dtype('O')

        >>> dtype = SparseDtype(str, "")
        >>> dtype.subtype
        dtype('O')

        >>> dtype._subtype_with_str
        <class 'str'>
        """
        # 如果 fill_value 是字符串，则返回其类型
        if isinstance(self.fill_value, str):
            return type(self.fill_value)
        # 否则返回 subtype 属性
        return self.subtype
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # TODO for now only handle SparseDtypes and numpy dtypes => extend
        # with other compatible extension dtypes
        # 导入 numpy 的查找公共类型函数
        from pandas.core.dtypes.cast import np_find_common_type

        # 检查是否存在不是 SparseDtype 的 ExtensionDtype 对象
        if any(
            isinstance(x, ExtensionDtype) and not isinstance(x, SparseDtype)
            for x in dtypes
        ):
            return None

        # 获取所有 SparseDtype 对象的填充值
        fill_values = [x.fill_value for x in dtypes if isinstance(x, SparseDtype)]
        fill_value = fill_values[0]

        # 导入 pandas 的 isna 函数
        from pandas import isna

        # 如果启用了性能警告并且填充值不唯一且不全为 NA，则发出警告
        if get_option("performance_warnings") and (
            not (len(set(fill_values)) == 1 or isna(fill_values).all())
        ):
            warnings.warn(
                "Concatenating sparse arrays with multiple fill "
                f"values: '{fill_values}'. Picking the first and "
                "converting the rest.",
                PerformanceWarning,
                stacklevel=find_stack_level(),
            )

        # 提取所有非 SparseDtype 对象的 subtype 或者直接的 dtypes
        np_dtypes = (x.subtype if isinstance(x, SparseDtype) else x for x in dtypes)
        
        # 返回公共 SparseDtype，使用 numpy 的公共类型查找函数
        return SparseDtype(np_find_common_type(*np_dtypes), fill_value=fill_value)
# 注册为扩展数据类型的装饰器函数
@register_extension_dtype
# 定义 ArrowDtype 类，继承自 StorageExtensionDtype 类
class ArrowDtype(StorageExtensionDtype):
    """
    An ExtensionDtype for PyArrow data types.

    .. warning::

       ArrowDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    While most ``dtype`` arguments can accept the "string"
    constructor, e.g. ``"int64[pyarrow]"``, ArrowDtype is useful
    if the data type contains parameters like ``pyarrow.timestamp``.

    Parameters
    ----------
    pyarrow_dtype : pa.DataType
        An instance of a `pyarrow.DataType <https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions>`__.

    Attributes
    ----------
    pyarrow_dtype : pa.DataType
        The underlying PyArrow data type instance.

    Methods
    -------
    None

    Returns
    -------
    ArrowDtype
        A new instance of ArrowDtype.

    See Also
    --------
    DataFrame.convert_dtypes : Convert columns to the best possible dtypes.

    Examples
    --------
    >>> import pyarrow as pa
    >>> pd.ArrowDtype(pa.int64())
    int64[pyarrow]

    Types with parameters must be constructed with ArrowDtype.

    >>> pd.ArrowDtype(pa.timestamp("s", tz="America/New_York"))
    timestamp[s, tz=America/New_York][pyarrow]
    >>> pd.ArrowDtype(pa.list_(pa.int64()))
    list<item: int64>[pyarrow]
    """

    # 定义元数据 _metadata 为元组 ("storage", "pyarrow_dtype")
    _metadata = ("storage", "pyarrow_dtype")  # type: ignore[assignment]

    # 初始化方法，接受一个 pyarrow.DataType 类型的参数 pyarrow_dtype
    def __init__(self, pyarrow_dtype: pa.DataType) -> None:
        # 调用父类的初始化方法，传递参数 "pyarrow"
        super().__init__("pyarrow")
        # 如果当前 pyarrow 版本低于 10.0.1，则抛出 ImportError 异常
        if pa_version_under10p1:
            raise ImportError("pyarrow>=10.0.1 is required for ArrowDtype")
        # 如果 pyarrow_dtype 不是 pa.DataType 类型，则抛出 ValueError 异常
        if not isinstance(pyarrow_dtype, pa.DataType):
            raise ValueError(
                f"pyarrow_dtype ({pyarrow_dtype}) must be an instance "
                f"of a pyarrow.DataType. Got {type(pyarrow_dtype)} instead."
            )
        # 将传入的 pyarrow_dtype 赋值给实例变量 self.pyarrow_dtype
        self.pyarrow_dtype = pyarrow_dtype

    # 定义 __repr__ 方法，返回实例的名称
    def __repr__(self) -> str:
        return self.name

    # 定义 __hash__ 方法，使实例可以作为字典的键
    def __hash__(self) -> int:
        # 返回当前实例的哈希值
        return hash(str(self))

    # 定义 __eq__ 方法，用于比较两个 ArrowDtype 实例是否相等
    def __eq__(self, other: object) -> bool:
        # 如果 other 不是当前类的实例，则调用父类的 __eq__ 方法比较
        if not isinstance(other, type(self)):
            return super().__eq__(other)
        # 比较两个实例的 pyarrow_dtype 属性是否相等
        return self.pyarrow_dtype == other.pyarrow_dtype

    # 定义属性装饰器 @property，但后续代码未提供完整实现
    @property
    def type(self):
        """
        Returns associated scalar type.
        """
        # 获取当前字段对应的 PyArrow 数据类型
        pa_type = self.pyarrow_dtype
        # 判断是否为整数类型
        if pa.types.is_integer(pa_type):
            return int
        # 判断是否为浮点数类型
        elif pa.types.is_floating(pa_type):
            return float
        # 判断是否为字符串类型或大字符串类型
        elif pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
            return str
        # 判断是否为二进制类型、固定大小二进制类型或大二进制类型
        elif (
            pa.types.is_binary(pa_type)
            or pa.types.is_fixed_size_binary(pa_type)
            or pa.types.is_large_binary(pa_type)
        ):
            return bytes
        # 判断是否为布尔类型
        elif pa.types.is_boolean(pa_type):
            return bool
        # 判断是否为时间间隔类型
        elif pa.types.is_duration(pa_type):
            # 若单位为纳秒，则返回 Timedelta 类型；否则返回 timedelta 类型
            if pa_type.unit == "ns":
                return Timedelta
            else:
                return timedelta
        # 判断是否为时间戳类型
        elif pa.types.is_timestamp(pa_type):
            # 若单位为纳秒，则返回 Timestamp 类型；否则返回 datetime 类型
            if pa_type.unit == "ns":
                return Timestamp
            else:
                return datetime
        # 判断是否为日期类型
        elif pa.types.is_date(pa_type):
            return date
        # 判断是否为时间类型
        elif pa.types.is_time(pa_type):
            return time
        # 判断是否为十进制类型
        elif pa.types.is_decimal(pa_type):
            return Decimal
        # 判断是否为字典类型
        elif pa.types.is_dictionary(pa_type):
            # TODO: 可能更改此处及 CategoricalDtype.type 为更符合标量类型的表示形式
            return CategoricalDtypeType
        # 判断是否为列表类型或大列表类型
        elif pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type):
            return list
        # 判断是否为固定大小列表类型
        elif pa.types.is_fixed_size_list(pa_type):
            return list
        # 判断是否为映射类型
        elif pa.types.is_map(pa_type):
            return list
        # 判断是否为结构体类型
        elif pa.types.is_struct(pa_type):
            return dict
        # 判断是否为 null 类型
        elif pa.types.is_null(pa_type):
            # TODO: None? pd.NA? pa.null?
            return type(pa_type)
        # 若为扩展类型，则返回其存储类型对应的 type 属性的类型
        elif isinstance(pa_type, pa.ExtensionType):
            return type(self)(pa_type.storage_type).type
        # 若没有匹配的类型，则抛出未实现错误
        raise NotImplementedError(pa_type)

    @property
    def name(self) -> str:  # type: ignore[override]
        """
        A string identifying the data type.
        """
        # 返回描述数据类型的字符串，包括 PyArrow 数据类型和存储类型
        return f"{self.pyarrow_dtype!s}[{self.storage}]"

    @cache_readonly
    def numpy_dtype(self) -> np.dtype:
        """Return an instance of the related numpy dtype"""
        if pa.types.is_timestamp(self.pyarrow_dtype):
            # 如果数据类型是时间戳类型，根据单位创建对应的 numpy datetime64 dtype
            # pa.timestamp(unit).to_pandas_dtype() 总是返回 ns 单位，不考虑 pyarrow 的时间戳单位。
            # 这段代码可以在 pyarrow 修复以下问题时移除：
            # https://github.com/apache/arrow/issues/34462
            return np.dtype(f"datetime64[{self.pyarrow_dtype.unit}]")
        if pa.types.is_duration(self.pyarrow_dtype):
            # 如果数据类型是持续时间类型，根据单位创建对应的 numpy timedelta64 dtype
            # pa.duration(unit).to_pandas_dtype() 总是返回 ns 单位，不考虑 pyarrow 的持续时间单位。
            # 这段代码可以在 pyarrow 修复以下问题时移除：
            # https://github.com/apache/arrow/issues/34462
            return np.dtype(f"timedelta64[{self.pyarrow_dtype.unit}]")
        if pa.types.is_string(self.pyarrow_dtype) or pa.types.is_large_string(
            self.pyarrow_dtype
        ):
            # 如果数据类型是字符串或大字符串类型，返回 numpy 的 str dtype
            # pa.string().to_pandas_dtype() 返回 object 类型，这里我们想要 str 类型。
            return np.dtype(str)
        try:
            # 尝试将 pyarrow 的 dtype 转换为 pandas 的 dtype，并返回对应的 numpy dtype
            return np.dtype(self.pyarrow_dtype.to_pandas_dtype())
        except (NotImplementedError, TypeError):
            # 如果转换不成功，返回 object dtype
            return np.dtype(object)

    @cache_readonly
    def kind(self) -> str:
        if pa.types.is_timestamp(self.pyarrow_dtype):
            # 如果数据类型是时间戳类型，返回 'M'，以匹配 DatetimeTZDtype
            return "M"
        # 对于其他数据类型，返回对应 numpy dtype 的 kind 属性
        return self.numpy_dtype.kind

    @cache_readonly
    def itemsize(self) -> int:
        """Return the number of bytes in this dtype"""
        # 返回 numpy dtype 的 itemsize 属性，即数据类型所占字节数
        return self.numpy_dtype.itemsize

    @classmethod
    def construct_array_type(cls) -> type_t[ArrowExtensionArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        # 返回与此数据类型相关联的数组类型 ArrowExtensionArray
        from pandas.core.arrays.arrow import ArrowExtensionArray

        return ArrowExtensionArray
    # 从字符串构造 ArrowDtype 对象的类方法
    def construct_from_string(cls, string: str) -> ArrowDtype:
        """
        Construct this type from a string.

        Parameters
        ----------
        string : str
            string should follow the format f"{pyarrow_type}[pyarrow]"
            e.g. int64[pyarrow]
        """
        # 检查输入是否为字符串类型，若不是则抛出类型错误
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        # 检查字符串是否以 '[pyarrow]' 结尾，若不是则抛出类型错误
        if not string.endswith("[pyarrow]"):
            raise TypeError(f"'{string}' must end with '[pyarrow]'")
        # 特殊情况：如果字符串是 'string[pyarrow]'，则抛出类型错误
        if string == "string[pyarrow]":
            raise TypeError("string[pyarrow] should be constructed by StringDtype")

        # 去除字符串末尾的 '[pyarrow]'，获取基础类型
        base_type = string[:-9]  # get rid of "[pyarrow]"
        
        try:
            # 尝试使用 pyarrow 的类型别名获取对应的数据类型
            pa_dtype = pa.type_for_alias(base_type)
        except ValueError as err:
            # 如果类型别名无法找到，检查是否有参数，尝试处理常见的时间类型
            has_parameters = re.search(r"[\[\(].*[\]\)]", base_type)
            if has_parameters:
                # 如果有参数，尝试解析为常见的时间类型
                try:
                    return cls._parse_temporal_dtype_string(base_type)
                except (NotImplementedError, ValueError):
                    # 如果解析失败，则通过下面的异常消息提供更好的提示
                    pass

                raise NotImplementedError(
                    "Passing pyarrow type specific parameters "
                    f"({has_parameters.group()}) in the string is not supported. "
                    "Please construct an ArrowDtype object with a pyarrow_dtype "
                    "instance with specific parameters."
                ) from err
            # 如果既不是别名也不是时间类型，则抛出类型错误
            raise TypeError(f"'{base_type}' is not a valid pyarrow data type.") from err
        
        # 使用获取到的 pyarrow 数据类型构造 ArrowDtype 对象并返回
        return cls(pa_dtype)

    # TODO(arrow#33642): This can be removed once supported by pyarrow
    # 类方法，用于解析时间类型的字符串表示，构造对应的 ArrowDtype 对象
    @classmethod
    def _parse_temporal_dtype_string(cls, string: str) -> ArrowDtype:
        """
        Construct a temporal ArrowDtype from string.
        """
        # 假设：
        #  1) 字符串末尾已经去除了 "[pyarrow]"
        #  2) 我们知道 "[" 存在
        
        # 将字符串分割为头部和尾部，头部是类型名，尾部是参数列表
        head, tail = string.split("[", 1)

        # 如果尾部不以 "]" 结尾，则抛出值错误
        if not tail.endswith("]"):
            raise ValueError
        # 去除尾部的 "]"，获取参数列表
        tail = tail[:-1]

        # 如果头部是 "timestamp"
        if head == "timestamp":
            assert "," in tail  # 否则 type_for_alias 应该可以工作
            # 分割参数列表，第一个是单位，第二个是时区
            unit, tz = tail.split(",", 1)
            unit = unit.strip()
            tz = tz.strip()
            # 如果时区以 "tz=" 开头，则去除开头的 "tz="
            if tz.startswith("tz="):
                tz = tz[3:]

            # 使用 pyarrow 的 timestamp 方法构造时间戳类型
            pa_type = pa.timestamp(unit, tz=tz)
            # 使用构造的数据类型构造 ArrowDtype 对象并返回
            dtype = cls(pa_type)
            return dtype

        # 如果头部不是 "timestamp"，则抛出未实现的错误
        raise NotImplementedError(string)

    @property
    def _is_numeric(self) -> bool:
        """
        Whether columns with this dtype should be considered numeric.
        """
        # TODO: pa.types.is_boolean?
        # 检查当前数据类型是否应视为数值型
        return (
            pa.types.is_integer(self.pyarrow_dtype)
            or pa.types.is_floating(self.pyarrow_dtype)
            or pa.types.is_decimal(self.pyarrow_dtype)
        )

    @property
    def _is_boolean(self) -> bool:
        """
        Whether this dtype should be considered boolean.
        """
        # 返回当前数据类型是否应视为布尔型
        return pa.types.is_boolean(self.pyarrow_dtype)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # We unwrap any masked dtypes, find the common dtype we would use
        #  for that, then re-mask the result.
        # Mirrors BaseMaskedDtype
        from pandas.core.dtypes.cast import find_common_type

        null_dtype = type(self)(pa.null())

        # 寻找一组数据类型的公共类型
        new_dtype = find_common_type(
            [
                dtype.numpy_dtype if isinstance(dtype, ArrowDtype) else dtype
                for dtype in dtypes
                if dtype != null_dtype
            ]
        )
        if not isinstance(new_dtype, np.dtype):
            return None
        try:
            # 尝试将 numpy 的 dtype 转换为 pyarrow 的数据类型
            pa_dtype = pa.from_numpy_dtype(new_dtype)
            return type(self)(pa_dtype)
        except NotImplementedError:
            return None

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> ArrowExtensionArray:
        """
        Construct IntegerArray/FloatingArray from pyarrow Array/ChunkedArray.
        """
        # 根据 pyarrow 的 Array/ChunkedArray 构建 IntegerArray 或 FloatingArray
        array_class = self.construct_array_type()
        arr = array.cast(self.pyarrow_dtype, safe=True)
        return array_class(arr)
```