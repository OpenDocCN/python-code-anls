# `D:\src\scipysrc\pandas\pandas\core\indexes\category.py`

```
from __future__ import annotations  # 允许在类型提示中使用当前模块的类作为返回类型

from typing import (  # 导入类型提示相关模块
    TYPE_CHECKING,  # 类型检查标志
    Any,  # 任意类型
    Literal,  # 字面量类型
    cast,  # 强制类型转换函数
)

import numpy as np  # 导入NumPy库

from pandas._libs import index as libindex  # 导入Pandas内部索引模块
from pandas.util._decorators import (  # 导入Pandas实用装饰器
    cache_readonly,  # 只读缓存装饰器
    doc,  # 文档字符串装饰器
)

from pandas.core.dtypes.common import is_scalar  # 导入Pandas通用数据类型检查函数
from pandas.core.dtypes.concat import concat_compat  # 导入Pandas数据合并兼容函数
from pandas.core.dtypes.dtypes import CategoricalDtype  # 导入Pandas分类数据类型
from pandas.core.dtypes.missing import (  # 导入Pandas缺失值处理函数
    is_valid_na_for_dtype,  # 检查缺失值是否适用于数据类型
    isna,  # 检查对象是否为缺失值
)

from pandas.core.arrays.categorical import (  # 导入Pandas分类数组相关函数和类
    Categorical,  # 分类数据结构
    contains,  # 检查分类中是否包含某个值
)
from pandas.core.construction import extract_array  # 导入Pandas从结构中提取数组的函数
from pandas.core.indexes.base import (  # 导入Pandas基本索引类
    Index,  # 索引基类
    maybe_extract_name,  # 尝试从对象中提取名称的函数
)
from pandas.core.indexes.extension import (  # 导入Pandas扩展索引类
    NDArrayBackedExtensionIndex,  # 支持数组作为后端的扩展索引
    inherit_names,  # 继承方法名称的装饰器
)

if TYPE_CHECKING:  # 如果在类型检查模式下
    from collections.abc import Hashable  # 导入集合模块的可散列类型
    from pandas._typing import (  # 导入Pandas类型提示
        Dtype,  # 数据类型
        DtypeObj,  # 数据类型对象
        Self,  # 自身类型
        npt,  # NumPy类型
    )


@inherit_names(  # 继承Categorical类的方法名称作为本类的方法
    [
        "argsort",  # 排序参数
        "tolist",  # 转换为列表
        "codes",  # 码值
        "categories",  # 分类
        "ordered",  # 是否有序
        "_reverse_indexer",  # 反向索引器
        "searchsorted",  # 搜索排序
        "min",  # 最小值
        "max",  # 最大值
    ],
    Categorical,  # 继承自Categorical类
)
@inherit_names(  # 继承Categorical类特定方法名称，包装在本类中
    [
        "rename_categories",  # 重命名分类
        "reorder_categories",  # 重新排序分类
        "add_categories",  # 添加分类
        "remove_categories",  # 移除分类
        "remove_unused_categories",  # 移除未使用的分类
        "set_categories",  # 设置分类
        "as_ordered",  # 转为有序
        "as_unordered",  # 转为无序
    ],
    Categorical,
    wrap=True,  # 包装为装饰器
)
class CategoricalIndex(NDArrayBackedExtensionIndex):
    """
    基于底层的Categorical创建索引。

    CategoricalIndex类似于Categorical，只能包含有限数量（通常是固定的）可能值（'categories'）。
    同样地，它可能有顺序，但不支持数值操作（如加法、除法等）。

    Parameters
    ----------
    data : array-like（1维）
        分类的值。如果给定了`categories`，不在`categories`中的值将被替换为NaN。
    categories : index-like，可选
        分类的分类。项需要唯一。如果这里没有给出分类（也不在`dtype`中），它们将从`data`中推断出来。
    ordered : bool，可选
        是否将此分类视为有序分类。如果此处或`dtype`中未给出，生成的分类将是无序的。
    dtype : CategoricalDtype或"category"，可选
        如果是CategoricalDtype，则不能与`categories`或`ordered`一起使用。
    copy : bool，默认为False
        复制输入ndarray。
    name : object，可选
        要存储在索引中的名称。

    Attributes
    ----------
    codes
    categories
    ordered

    Methods
    -------
    rename_categories
    reorder_categories
    add_categories
    remove_categories
    remove_unused_categories
    set_categories
    as_ordered

    """
    # _typ 属性用于指示对象类型为 categoricalindex
    _typ = "categoricalindex"
    # _data_cls 属性指定数据类别为 Categorical
    _data_cls = Categorical

    @property
    def _can_hold_strings(self):
        # 返回一个布尔值，指示分类索引是否能够容纳字符串
        return self.categories._can_hold_strings

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        # 返回一个布尔值，指示是否应该回退到使用位置索引
        return self.categories._should_fallback_to_positional

    # codes 属性声明为 numpy 数组类型，存储索引编码
    codes: np.ndarray
    # categories 属性声明为 Index 类型，存储分类的标签
    categories: Index
    # ordered 属性声明为布尔值或 None，指示分类是否有序
    ordered: bool | None
    # _data 属性声明为 Categorical 类型，存储实际的分类数据
    _data: Categorical
    # _values 属性声明为 Categorical 类型，存储分类的值
    _values: Categorical

    @property
    def _engine_type(self) -> type[libindex.IndexEngine]:
        # 返回一个索引引擎类型，根据 self.codes 的数据类型确定返回的具体类型
        # 可能的数据类型包括 np.int8, np.int16, np.int32, np.int64
        return {
            np.int8: libindex.Int8Engine,
            np.int16: libindex.Int16Engine,
            np.int32: libindex.Int32Engine,
            np.int64: libindex.Int64Engine,
        }[self.codes.dtype.type]

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data=None,
        categories=None,
        ordered=None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ) -> Self:
        # 根据传入的参数创建新的实例
        # 如果 data 是标量，则抛出异常
        name = maybe_extract_name(name, data, cls)

        if is_scalar(data):
            # 如果 data 是标量，则抛出异常，不支持直接使用标量创建
            cls._raise_scalar_data_error(data)

        # 创建 Categorical 对象来处理数据
        data = Categorical(
            data, categories=categories, ordered=ordered, dtype=dtype, copy=copy
        )

        # 返回创建的新实例
        return cls._simple_new(data, name=name)
    # --------------------------------------------------------------------
    
    # 定义一个内部非公开方法用于检查数据类型是否兼容
    def _is_dtype_compat(self, other: Index) -> Categorical:
        """
        *this is an internal non-public method*

        提供对比 self 和 other 的数据类型是否兼容（如有必要，进行强制转换）

        Parameters
        ----------
        other : Index
            另一个索引对象

        Returns
        -------
        Categorical
            返回一个分类数据对象

        Raises
        ------
        TypeError
            如果数据类型不兼容时引发异常
        """
        # 如果 other 的数据类型是 CategoricalDtype
        if isinstance(other.dtype, CategoricalDtype):
            # 提取 other 中的分类数据，并进行强制类型转换为 Categorical
            cat = extract_array(other)
            cat = cast(Categorical, cat)
            # 如果其分类与当前对象的值不匹配，则引发异常
            if not cat._categories_match_up_to_permutation(self._values):
                raise TypeError(
                    "categories must match existing categories when appending"
                )

        # 如果 other 是多级索引
        elif other._is_multi:
            # 预先避免在 isna 调用时引发 NotImplementedError
            raise TypeError("MultiIndex is not dtype-compatible with CategoricalIndex")
        
        # 否则处理其他情况
        else:
            values = other

            # 使用 other 创建一个新的 Categorical 对象
            cat = Categorical(other, dtype=self.dtype)
            # 将其转换为 CategoricalIndex 类型
            other = CategoricalIndex(cat)
            # 如果 values 中存在任何非分类项，则引发异常
            if not other.isin(values).all():
                raise TypeError(
                    "cannot append a non-category item to a CategoricalIndex"
                )
            # 获取 other 的值
            cat = other._values

            # 如果其值与 values 不匹配或者存在缺失值，则引发异常
            if not ((cat == values) | (isna(cat) & isna(values))).all():
                # GH#37667 参见 test_equals_non_category
                raise TypeError(
                    "categories must match existing categories when appending"
                )

        # 返回最终确定的 cat 变量
        return cat
    def equals(self, other: object) -> bool:
        """
        Determine if two CategoricalIndex objects contain the same elements.

        The order and orderedness of elements matters. The categories matter,
        but the order of the categories matters only when ``ordered=True``.

        Parameters
        ----------
        other : object
            The CategoricalIndex object to compare with.

        Returns
        -------
        bool
            ``True`` if two :class:`pandas.CategoricalIndex` objects have equal
            elements, ``False`` otherwise.

        See Also
        --------
        Categorical.equals : Returns True if categorical arrays are equal.

        Examples
        --------
        >>> ci = pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"])
        >>> ci2 = pd.CategoricalIndex(pd.Categorical(["a", "b", "c", "a", "b", "c"]))
        >>> ci.equals(ci2)
        True

        The order of elements matters.

        >>> ci3 = pd.CategoricalIndex(["c", "b", "a", "a", "b", "c"])
        >>> ci.equals(ci3)
        False

        The orderedness also matters.

        >>> ci4 = ci.as_ordered()
        >>> ci.equals(ci4)
        False

        The categories matter, but the order of the categories matters only when
        ``ordered=True``.

        >>> ci5 = ci.set_categories(["a", "b", "c", "d"])
        >>> ci.equals(ci5)
        False

        >>> ci6 = ci.set_categories(["b", "c", "a"])
        >>> ci.equals(ci6)
        True
        >>> ci_ordered = pd.CategoricalIndex(
        ...     ["a", "b", "c", "a", "b", "c"], ordered=True
        ... )
        >>> ci2_ordered = ci_ordered.set_categories(["b", "c", "a"])
        >>> ci_ordered.equals(ci2_ordered)
        False
        """
        # 检查对象引用是否相同，如果是，则直接返回 True
        if self.is_(other):
            return True

        # 如果 other 不是 Index 的实例，则返回 False
        if not isinstance(other, Index):
            return False

        # 尝试将 other 转换为与 self 相容的类型，如果转换时出现错误，则返回 False
        try:
            other = self._is_dtype_compat(other)
        except (TypeError, ValueError):
            return False

        # 调用 _data 属性的 equals 方法来比较数据是否相等
        return self._data.equals(other)

    # --------------------------------------------------------------------
    # Rendering Methods

    @property
    def _formatter_func(self):
        # 返回 categories 属性的 _formatter_func 方法
        return self.categories._formatter_func

    def _format_attrs(self):
        """
        Return a list of tuples of the (attr,formatted_value)
        """
        # 声明 attrs 变量的类型注释
        attrs: list[tuple[str, str | int | bool | None]]

        # 创建包含分类数据和有序性的属性列表
        attrs = [
            (
                "categories",
                f"[{', '.join(self._data._repr_categories())}]",
            ),
            ("ordered", self.ordered),
        ]
        # 调用超类的 _format_attrs 方法获取额外的属性信息
        extra = super()._format_attrs()
        # 将额外的属性信息添加到 attrs 中并返回
        return attrs + extra

    # --------------------------------------------------------------------

    @property
    def inferred_type(self) -> str:
        # 返回字符串 "categorical" 表示推断的类型为分类数据
        return "categorical"

    @doc(Index.__contains__)
    def __contains__(self, key: Any) -> bool:
        # 如果 key 是 NaN，则检查 self 中是否有任何 NaN。
        if is_valid_na_for_dtype(key, self.categories.dtype):
            return self.hasnans

        # 调用 contains 函数，检查 key 是否存在于 self._engine 中
        return contains(self, key, container=self._engine)

    def reindex(
        self, target, method=None, level=None, limit: int | None = None, tolerance=None
    ) -> tuple[Index, npt.NDArray[np.intp] | None]:
        """
        使用 target 的值创建索引（根据需要移动/添加/删除值）

        返回
        -------
        new_index : pd.Index
            结果索引
        indexer : np.ndarray[np.intp] or None
            输出值在原始索引中的索引

        """
        # 如果 method 不为 None，则抛出未实现的错误
        if method is not None:
            raise NotImplementedError(
                "argument method is not implemented for CategoricalIndex.reindex"
            )
        # 如果 level 不为 None，则抛出未实现的错误
        if level is not None:
            raise NotImplementedError(
                "argument level is not implemented for CategoricalIndex.reindex"
            )
        # 如果 limit 不为 None，则抛出未实现的错误
        if limit is not None:
            raise NotImplementedError(
                "argument limit is not implemented for CategoricalIndex.reindex"
            )
        # 调用父类的 reindex 方法，并返回结果
        return super().reindex(target)

    # --------------------------------------------------------------------
    # 索引方法

    def _maybe_cast_indexer(self, key) -> int:
        # GH#41933: 我们必须这样做而不是使用 self._data._validate_scalar，
        # 因为这样可以正确地在 Interval 类别上进行部分索引
        try:
            return self._data._unbox_scalar(key)
        except KeyError:
            # 如果 key 是 NaN 并且与 self.categories 的 dtype 兼容，则返回 -1
            if is_valid_na_for_dtype(key, self.categories.dtype):
                return -1
            # 否则引发 KeyError
            raise

    def _maybe_cast_listlike_indexer(self, values) -> CategoricalIndex:
        if isinstance(values, CategoricalIndex):
            values = values._data
        if isinstance(values, Categorical):
            # 如果 values 是 Categorical 类型，则使用当前实例的 categories 进行编码
            cat = self._data._encode_with_my_categories(values)
            codes = cat._codes
        else:
            # 否则，使用当前实例的 categories 获取 values 的索引
            codes = self.categories.get_indexer(values)
            # 将 codes 转换为与 self.codes 相同的 dtype
            codes = codes.astype(self.codes.dtype, copy=False)
            # 使用 codes 创建新的 CategoricalIndex 实例
            cat = self._data._from_backing_data(codes)
        # 返回类型为 self 的新实例
        return type(self)._simple_new(cat)

    # --------------------------------------------------------------------

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        # 检查 self.categories 是否与给定的 dtype 是可比较的 dtype
        return self.categories._is_comparable_dtype(dtype)
    def map(self, mapper, na_action: Literal["ignore"] | None = None):
        """
        Map values using input an input mapping or function.

        Maps the values (their categories, not the codes) of the index to new
        categories. If the mapping correspondence is one-to-one the result is a
        :class:`~pandas.CategoricalIndex` which has the same order property as
        the original, otherwise an :class:`~pandas.Index` is returned.

        If a `dict` or :class:`~pandas.Series` is used any unmapped category is
        mapped to `NaN`. Note that if this happens an :class:`~pandas.Index`
        will be returned.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence. This can be a function, a dictionary, or a
            pandas Series defining how to map the index values.
        na_action : {None, 'ignore'}, default 'ignore'
            If 'ignore', propagate NaN values, without passing them to
            the mapping correspondence. This parameter controls how NaN values
            are handled during the mapping process.

        Returns
        -------
        pandas.CategoricalIndex or pandas.Index
            Mapped index. Returns a CategoricalIndex if the mapping is one-to-one
            preserving order, otherwise returns an Index.

        See Also
        --------
        Index.map : Apply a mapping correspondence on an
            :class:`~pandas.Index`.
        Series.map : Apply a mapping correspondence on a
            :class:`~pandas.Series`.
        Series.apply : Apply more complex functions on a
            :class:`~pandas.Series`.

        Examples
        --------
        >>> idx = pd.CategoricalIndex(["a", "b", "c"])
        >>> idx
        CategoricalIndex(['a', 'b', 'c'], categories=['a', 'b', 'c'],
                          ordered=False, dtype='category')
        >>> idx.map(lambda x: x.upper())
        CategoricalIndex(['A', 'B', 'C'], categories=['A', 'B', 'C'],
                         ordered=False, dtype='category')
        >>> idx.map({"a": "first", "b": "second", "c": "third"})
        CategoricalIndex(['first', 'second', 'third'], categories=['first',
                         'second', 'third'], ordered=False, dtype='category')

        If the mapping is one-to-one the ordering of the categories is
        preserved:

        >>> idx = pd.CategoricalIndex(["a", "b", "c"], ordered=True)
        >>> idx
        CategoricalIndex(['a', 'b', 'c'], categories=['a', 'b', 'c'],
                         ordered=True, dtype='category')
        >>> idx.map({"a": 3, "b": 2, "c": 1})
        CategoricalIndex([3, 2, 1], categories=[3, 2, 1], ordered=True,
                         dtype='category')

        If the mapping is not one-to-one an :class:`~pandas.Index` is returned:

        >>> idx.map({"a": "first", "b": "second", "c": "first"})
        Index(['first', 'second', 'first'], dtype='object')

        If a `dict` is used, all unmapped categories are mapped to `NaN` and
        the result is an :class:`~pandas.Index`:

        >>> idx.map({"a": "first", "b": "second"})
        Index(['first', 'second', nan], dtype='object')
        """
        # 使用 self._values 的 map 方法进行实际的映射操作，传入 mapper 和 na_action 参数
        mapped = self._values.map(mapper, na_action=na_action)
        # 返回一个新的 Index 对象，使用映射后的结果作为值，保留当前对象的名称作为新 Index 的名称
        return Index(mapped, name=self.name)
    # 定义一个方法 `_concat`，用于合并索引对象。
    # `to_concat` 是一个待合并的索引对象列表，类型为 `list[Index]`。
    # `name` 是合并后的索引对象的名称，类型为 `Hashable`。
    def _concat(self, to_concat: list[Index], name: Hashable) -> Index:
        # 尝试将 `to_concat` 中的索引对象合并成相同类型的分类对象 `cat`。
        try:
            cat = Categorical._concat_same_type(
                [self._is_dtype_compat(c) for c in to_concat]
            )
        except TypeError:
            # 如果无法合并（可能是因为不是所有的 `to_concat` 元素都属于我们的类别（或者为 NA））
            
            # 调用 `concat_compat` 方法，将 `to_concat` 中各对象的值合并成一个结果 `res`。
            res = concat_compat([x._values for x in to_concat])
            # 返回一个新的索引对象 `Index`，其中包含合并后的值 `res`，并指定名称为 `name`。
            return Index(res, name=name)
        else:
            # 如果成功合并为相同类型的分类对象 `cat`，则创建并返回一个新的同类型的索引对象。
            return type(self)._simple_new(cat, name=name)
```