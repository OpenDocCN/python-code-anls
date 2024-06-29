# `D:\src\scipysrc\pandas\pandas\core\arrays\categorical.py`

```
# 从未来导入注解，以支持类型注解中的类型自引用
from __future__ import annotations

# 导入 csv 模块中的 QUOTE_NONNUMERIC 常量，用于指示 CSV 格式中的非数值引用
from csv import QUOTE_NONNUMERIC

# 导入 functools 模块中的 partial 函数，用于部分应用函数操作
from functools import partial

# 导入 operator 模块，用于函数操作符
import operator

# 导入 shutil 模块中的 get_terminal_size 函数，用于获取终端窗口大小
from shutil import get_terminal_size

# 导入 typing 模块中的类型相关工具
from typing import (
    TYPE_CHECKING,  # 类型检查标志，用于在类型注解中检测循环引用
    Literal,        # 字面值类型
    cast,           # 类型强制转换函数
    overload        # 函数重载装饰器
)

# 导入 numpy 库
import numpy as np

# 从 pandas._config 模块中导入 get_option 函数，用于获取 pandas 配置选项
from pandas._config import get_option

# 从 pandas._libs 模块中导入 NaT（Not a Time）对象和算法函数库
from pandas._libs import (
    NaT,            # 日期时间中的缺失值表示
    algos as libalgos,  # pandas 库中的算法函数
    lib             # pandas 库中的基础函数库
)

# 从 pandas._libs.arrays 模块中导入 NDArrayBacked 类
from pandas._libs.arrays import NDArrayBacked

# 从 pandas.compat.numpy 模块中导入 nv 函数，用于兼容处理 numpy 函数
from pandas.compat.numpy import function as nv

# 从 pandas.util._validators 模块中导入 validate_bool_kwarg 函数，用于验证布尔类型的关键字参数
from pandas.util._validators import validate_bool_kwarg

# 从 pandas.core.dtypes.cast 模块中导入类型强制转换相关函数
from pandas.core.dtypes.cast import (
    coerce_indexer_dtype,   # 强制转换索引器的数据类型
    find_common_type        # 查找公共数据类型
)

# 从 pandas.core.dtypes.common 模块中导入通用数据类型相关函数
from pandas.core.dtypes.common import (
    ensure_int64,                   # 确保整数类型为 int64
    ensure_platform_int,            # 确保平台相关整数类型
    is_any_real_numeric_dtype,      # 判断是否为任意实数数值类型
    is_bool_dtype,                  # 判断是否为布尔类型
    is_dict_like,                   # 判断是否类字典对象
    is_hashable,                    # 判断是否可散列
    is_integer_dtype,               # 判断是否为整数类型
    is_list_like,                   # 判断是否类列表对象
    is_scalar,                      # 判断是否为标量
    needs_i8_conversion,            # 判断是否需要进行 int64 转换
    pandas_dtype                    # 获取 pandas 对象的数据类型
)

# 从 pandas.core.dtypes.dtypes 模块中导入具体数据类型相关类
from pandas.core.dtypes.dtypes import (
    ArrowDtype,             # 箭头数据类型
    CategoricalDtype,       # 分类数据类型
    CategoricalDtypeType,   # 分类数据类型的类型
    ExtensionDtype          # 扩展数据类型
)

# 从 pandas.core.dtypes.generic 模块中导入通用类
from pandas.core.dtypes.generic import (
    ABCIndex,       # 抽象基类索引
    ABCSeries       # 抽象基类序列
)

# 从 pandas.core.dtypes.missing 模块中导入缺失值处理相关函数
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,  # 判断缺失值是否适用于指定数据类型
    isna                    # 判断是否为缺失值
)

# 从 pandas.core 模块中导入核心算法、数组操作、运算等
from pandas.core import (
    algorithms,     # pandas 核心算法
    arraylike,      # 类数组操作
    ops             # 运算
)

# 从 pandas.core.accessor 模块中导入 PandasDelegate 类、委托名称相关函数
from pandas.core.accessor import (
    PandasDelegate,     # Pandas 委托基类
    delegate_names      # 委托名称
)

# 从 pandas.core.algorithms 模块中导入因子化、多维索引取值等相关函数
from pandas.core.algorithms import (
    factorize,      # 因子化
    take_nd         # 多维索引取值
)

# 从 pandas.core.arrays._mixins 模块中导入 NDArrayBackedExtensionArray 类、拉直兼容函数
from pandas.core.arrays._mixins import (
    NDArrayBackedExtensionArray,    # 基于 NDArray 的扩展数组
    ravel_compat                    # 拉直兼容
)

# 从 pandas.core.base 模块中导入扩展数组、无新属性混合类、Pandas 对象等
from pandas.core.base import (
    ExtensionArray,         # 扩展数组
    NoNewAttributesMixin,   # 无新属性混合类
    PandasObject            # Pandas 对象基类
)

# 导入 pandas.core.common 模块，并重命名为 com，用于通用功能
import pandas.core.common as com

# 从 pandas.core.construction 模块中导入数组提取、数组清理等相关函数
from pandas.core.construction import (
    extract_array,      # 数组提取
    sanitize_array      # 数组清理
)

# 从 pandas.core.ops.common 模块中导入解包零维并延迟处理函数
from pandas.core.ops.common import unpack_zerodim_and_defer

# 从 pandas.core.sorting 模块中导入 nargsort 函数，用于排序参数获取
from pandas.core.sorting import nargsort

# 从 pandas.core.strings.object_array 模块中导入 ObjectStringArrayMixin 类，用于对象字符串数组混合
from pandas.core.strings.object_array import ObjectStringArrayMixin

# 从 pandas.io.formats 模块中导入 console，用于控制台输出格式
from pandas.io.formats import console

# 如果支持类型检查，则从 collections.abc 模块中导入 Callable、Hashable、Iterator、Sequence 等类型
if TYPE_CHECKING:
    from collections.abc import (
        Callable,       # 可调用对象类型
        Hashable,       # 可散列对象类型
        Iterator,       # 迭代器类型
        Sequence        # 序列类型
    )

    # 从 pandas._typing 模块中导入数组样式、类型参数等类型
    from pandas._typing import (
        ArrayLike,      # 类似数组对象
        AstypeArg,      # 强制类型转换参数
        AxisInt,        # 轴整数类型
        Dtype,          # 数据类型
        DtypeObj,       # 数据类型对象
        NpDtype,        # numpy 数据类型
        Ordered,        # 有序对象
        Self,           # 自身类型
        Shape,          # 形状类型
        SortKind,       # 排序类型
        npt             # numpy 类型
    )

    # 从 pandas 模块中导入 DataFrame、Index、Series 等核心数据结构
    from pandas import (
        DataFrame,      # 数据帧
        Index,          # 索引对象
        Series          # 系列对象
    )


# 定义一个函数 _cat_compare_op，用于比较操作符的分类处理
def _cat_compare_op(op):
    # 获取操作符名称，形如 '__操作符名__'
    opname = f"__{op.__name__}__"
    # 如果操作符为 operator.ne，则填充值为 True
    fill_value = op is operator.ne

    # 解包零维并延迟处理操作符名
    @unpack_zerodim_and_defer(opname)
    # 定义一个方法 func(self, other)，用于处理分类数据对象的比较操作
    def func(self, other):
        # 检查 other 是否可哈希
        hashable = is_hashable(other)
        # 如果 other 是类似列表的可迭代对象，并且长度不等于 self 的长度且不可哈希，则抛出 ValueError
        if is_list_like(other) and len(other) != len(self) and not hashable:
            raise ValueError("Lengths must match.")

        # 如果 self 不是有序的分类数据对象
        if not self.ordered:
            # 如果操作名在 ["__lt__", "__gt__", "__le__", "__ge__"] 中，抛出 TypeError
            if opname in ["__lt__", "__gt__", "__le__", "__ge__"]:
                raise TypeError(
                    "Unordered Categoricals can only compare equality or not"
                )

        # 如果 other 是 Categorical 对象
        if isinstance(other, Categorical):
            # 两个 Categorical 对象只能比较如果它们的类别相同（可能顺序不同，取决于是否有序）
            
            msg = "Categoricals can only be compared if 'categories' are the same."
            # 如果类别不匹配，抛出 TypeError
            if not self._categories_match_up_to_permutation(other):
                raise TypeError(msg)

            # 如果 self 不是有序的且类别不同，进行类别重新编码以匹配 self 的类别
            if not self.ordered and not self.categories.equals(other.categories):
                other_codes = recode_for_categories(
                    other.codes, other.categories, self.categories, copy=False
                )
            else:
                other_codes = other._codes

            # 对比 self 和 other 的代码，返回比较结果
            ret = op(self._codes, other_codes)
            # 找出 self 或 other 中代码为 -1 的位置，将对应位置的结果设为 fill_value
            mask = (self._codes == -1) | (other_codes == -1)
            if mask.any():
                ret[mask] = fill_value
            return ret

        # 如果 other 是可哈希的
        if hashable:
            # 如果 other 在 self 的类别中
            if other in self.categories:
                # 获取 other 在 self 中的位置
                i = self._unbox_scalar(other)
                # 对比 self 和 i 的代码，返回比较结果
                ret = op(self._codes, i)

                # 如果操作名不在 {"__eq__", "__ge__", "__gt__"} 中，对比结果中未匹配的代码位置设为 fill_value
                if opname not in {"__eq__", "__ge__", "__gt__"}:
                    mask = self._codes == -1
                    ret[mask] = fill_value
                return ret
            else:
                # 否则，返回无效比较的操作结果
                return ops.invalid_comparison(self, other, op)
        else:
            # 允许分类数据与对象类型数组进行比较操作，仅限于位置比较
            # 如果操作名不在 ["__eq__", "__ne__"] 中，抛出 TypeError
            if opname not in ["__eq__", "__ne__"]:
                raise TypeError(
                    f"Cannot compare a Categorical for op {opname} with "
                    f"type {type(other)}.\nIf you want to compare values, "
                    "use 'np.asarray(cat) <op> other'."
                )

            # 如果 other 是 ExtensionArray 且需要转换为 int64 类型
            if isinstance(other, ExtensionArray) and needs_i8_conversion(other.dtype):
                # 返回 other 与 self 的比较结果
                return op(other, self)
            # 否则，将 self 转换为 numpy 数组，再使用对应的操作
            return getattr(np.array(self), opname)(np.array(other))

    # 将 func 方法的名称设置为 opname
    func.__name__ = opname

    # 返回 func 方法
    return func
# 定义了一个名为 contains 的函数，用于检查 key 是否存在于 cat.categories 中，并且其位置在 container 中

def contains(cat, key, container) -> bool:
    """
    Helper for membership check for ``key`` in ``cat``.

    This is a helper method for :method:`__contains__`
    and :class:`CategoricalIndex.__contains__`.

    Returns True if ``key`` is in ``cat.categories`` and the
    location of ``key`` in ``categories`` is in ``container``.

    Parameters
    ----------
    cat : :class:`Categorical`or :class:`categoricalIndex`
        The categorical object or index to check membership against.
    key : a hashable object
        The key to check membership for.
    container : Container (e.g. list-like or mapping)
        The container to check for membership in.

    Returns
    -------
    is_in : bool
        True if ``key`` is in ``self.categories`` and location of
        ``key`` in ``categories`` is in ``container``, else False.

    Notes
    -----
    This method does not check for NaN values. Do that separately
    before calling this method.
    """

    hash(key)  # 计算 key 的哈希值

    # 获取 key 在 categories 中的位置。
    # 如果抛出 KeyError 或 TypeError，则 key 不在 categories 中，因此也不可能在 container 中。
    try:
        loc = cat.categories.get_loc(key)
    except (KeyError, TypeError):
        return False

    # loc 是 key 在 categories 中的位置，同时也是 container 中的值。
    # 因此，`key` 可能在 categories 中，但仍然不在 `container` 中。
    # 例如 ('b' 在 categories 中，但不在 container 中的情况)：
    # 'b' in Categorical(['a'], categories=['a', 'b'])  # False
    if is_scalar(loc):  # 检查 loc 是否是标量值
        return loc in container
    else:
        # 如果 categories 是一个 IntervalIndex，loc 将是一个数组。
        # 检查 loc_ 是否在 container 中的任何一个元素中。
        return any(loc_ in container for loc_ in loc)


# error: Definition of "delete/ravel/T/repeat/copy" in base class "NDArrayBacked"
# is incompatible with definition in base class "ExtensionArray"
class Categorical(NDArrayBackedExtensionArray, PandasObject, ObjectStringArrayMixin):  # type: ignore[misc]
    """
    Represent a categorical variable in classic R / S-plus fashion.

    `Categoricals` can only take on a limited, and usually fixed, number
    of possible values (`categories`). In contrast to statistical categorical
    variables, a `Categorical` might have an order, but numerical operations
    (additions, divisions, ...) are not possible.

    All values of the `Categorical` are either in `categories` or `np.nan`.
    Assigning values outside of `categories` will raise a `ValueError`. Order
    is defined by the order of the `categories`, not lexical order of the
    values.

    Parameters
    ----------
    values : list-like
        The values of the categorical. If categories are given, values not in
        categories will be replaced with NaN.
    categories : Index-like (unique), optional
        The unique categories for this categorical. If not given, the
        categories are assumed to be the unique values of `values` (sorted, if
        possible, otherwise in the order in which they appear).
    """
    # 用于指定分类数据是否按顺序排列的布尔值，默认为 False
    ordered : bool, default False
        Whether or not this categorical is treated as a ordered categorical.
        If True, the resulting categorical will be ordered.
        An ordered categorical respects, when sorted, the order of its
        `categories` attribute (which in turn is the `categories` argument, if
        provided).
    
    # 用于指定此分类数据使用的 CategoricalDtype 类型
    dtype : CategoricalDtype
        An instance of ``CategoricalDtype`` to use for this categorical.
    
    # 控制是否在代码未更改的情况下进行复制的布尔值，默认为 True
    copy : bool, default True
        Whether to copy if the codes are unchanged.

    Attributes
    ----------
    
    # 此分类数据的分类列表，类型为 Index
    categories : Index
        The categories of this categorical.
    
    # 此分类数据的代码（指向分类的整数位置），只读的 ndarray
    codes : ndarray
        The codes (integer positions, which point to the categories) of this
        categorical, read only.
    
    # 指示此 Categorical 是否按顺序排列的布尔值
    ordered : bool
        Whether or not this Categorical is ordered.
    
    # 存储分类和排序信息的 CategoricalDtype 的实例
    dtype : CategoricalDtype
        The instance of ``CategoricalDtype`` storing the ``categories``
        and ``ordered``.

    Methods
    -------
    
    # 从指定的 codes 创建 Categorical 对象
    from_codes
    # 将此 Categorical 转换为有序状态
    as_ordered
    # 将此 Categorical 转换为无序状态
    as_unordered
    # 设置此 Categorical 的分类列表
    set_categories
    # 重命名此 Categorical 的分类
    rename_categories
    # 重新排序此 Categorical 的分类
    reorder_categories
    # 添加新的分类到此 Categorical
    add_categories
    # 移除指定的分类从此 Categorical
    remove_categories
    # 移除未使用的分类从此 Categorical
    remove_unused_categories
    # 映射数据到此 Categorical
    map
    # 返回此 Categorical 对象的 ndarray 表示
    __array__

    Raises
    ------
    
    # 如果分类无效时抛出 ValueError 异常
    ValueError
        If the categories do not validate.
    
    # 如果明确指定了 `ordered=True`，但没有提供 `categories` 或值不可排序时抛出 TypeError 异常
    TypeError
        If an explicit ``ordered=True`` is given but no `categories` and the
        `values` are not sortable.

    See Also
    --------
    
    # 用于表示分类数据的类型 CategoricalDtype
    CategoricalDtype : Type for categorical data.
    
    # 带有基础 ``Categorical`` 的 Index
    CategoricalIndex : An Index with an underlying ``Categorical``.

    Notes
    -----
    
    # 查看用户指南获取更多信息
    See the `user guide
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html>`__
    for more.

    Examples
    --------
    
    # 创建整数类型的分类数据
    >>> pd.Categorical([1, 2, 3, 1, 2, 3])
    [1, 2, 3, 1, 2, 3]
    Categories (3, int64): [1, 2, 3]

    # 创建对象类型的分类数据
    >>> pd.Categorical(["a", "b", "c", "a", "b", "c"])
    ['a', 'b', 'c', 'a', 'b', 'c']
    Categories (3, object): ['a', 'b', 'c']

    # 不包含缺失值的分类数据
    Missing values are not included as a category.

    # 创建包含缺失值的分类数据，缺失值的 code 为 `-1`
    >>> c = pd.Categorical([1, 2, 3, 1, 2, 3, np.nan])
    >>> c
    [1, 2, 3, 1, 2, 3, NaN]
    Categories (3, int64): [1, 2, 3]

    # 有序分类数据按照自定义的顺序进行排序，并且可以获取最小值和最大值
    >>> c = pd.Categorical(
    ...     ["a", "b", "c", "a", "b", "c"], ordered=True, categories=["c", "b", "a"]
    ... )
    >>> c
    ['a', 'b', 'c', 'a', 'b', 'c']
    Categories (3, object): ['c' < 'b' < 'a']
    >>> c.min()
    'c'
    """

    # 用于比较操作，使得 numpy 使用我们的实现
    __array_priority__ = 1000
    # tolist 实际上并未被弃用，只是在 __dir__ 中被隐藏了
    _hidden_attrs = PandasObject._hidden_attrs | frozenset(["tolist"])
    # 数据类型标记为 categorical
    _typ = "categorical"

    _dtype: CategoricalDtype

    @classmethod
    # 定义一个类型忽略的方法 _simple_new，用于创建新的 Categorical 对象
    # 参数:
    #   cls: 类型本身，用于类方法
    #   codes: np.ndarray，用于存储分类编码的数组
    #   dtype: CategoricalDtype，指定分类数据类型
    # 返回:
    #   Self: 新创建的 Categorical 对象
    def _simple_new(  # type: ignore[override]
        cls, codes: np.ndarray, dtype: CategoricalDtype
    ) -> Self:
        # 强制将 codes 的数据类型转换为 dtype.categories 的类型
        codes = coerce_indexer_dtype(codes, dtype.categories)
        # 更新 dtype，确保其有序性为 False
        dtype = CategoricalDtype(ordered=False).update_dtype(dtype)
        # 调用父类的 _simple_new 方法创建新的对象
        return super()._simple_new(codes, dtype)

    # 初始化方法，用于创建一个新的 Categorical 对象
    def __init__(
        self,
        values,
        categories=None,
        ordered=None,
        dtype: Dtype | None = None,
        copy: bool = True,
    ):

    @property
    # 返回当前 Categorical 对象的数据类型
    def dtype(self) -> CategoricalDtype:

    @property
    # 返回内部填充值，用于处理所有元素均为 NA 的情况
    def _internal_fill_value(self) -> int:

    # 类方法，从标量序列创建 Categorical 对象
    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:

    # 类方法，从标量创建 Categorical 对象
    @classmethod
    def _from_scalars(cls, scalars, *, dtype: DtypeObj) -> Self:

    # 方法重载，用于将当前对象转换为指定的数据类型
    @overload
    def astype(self, dtype: npt.DTypeLike, copy: bool = ...) -> np.ndarray: ...

    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = ...) -> ExtensionArray: ...

    @overload
    def astype(self, dtype: AstypeArg, copy: bool = ...) -> ArrayLike: ...
    def astype(self, dtype: AstypeArg, copy: bool = True) -> ArrayLike:
        """
        Coerce this type to another dtype
        
        Parameters
        ----------
        dtype : numpy dtype or pandas type
            The target data type to which the current data should be converted.
        copy : bool, default True
            By default, astype always returns a newly allocated object.
            If copy is set to False and dtype is categorical, the original
            object is returned.
        """
        # Convert dtype to a pandas dtype object for consistency
        dtype = pandas_dtype(dtype)
        
        # Declare result variable with potential types of Categorical or np.ndarray
        result: Categorical | np.ndarray
        
        # Check if current dtype matches the target dtype
        if self.dtype is dtype:
            # Return a copy if copy is True, otherwise return self (original object)
            result = self.copy() if copy else self
        
        # Handle case where dtype is an instance of CategoricalDtype
        elif isinstance(dtype, CategoricalDtype):
            # Update dtype of self to match the new categorical dtype
            dtype = self.dtype.update_dtype(dtype)
            # Create a copy of self if copy is True, otherwise use self
            self = self.copy() if copy else self
            # Set the updated dtype for self and assign to result
            result = self._set_dtype(dtype)
        
        # Handle case where dtype is an ExtensionDtype (subclass of dtype)
        elif isinstance(dtype, ExtensionDtype):
            # Delegate to superclass's astype method for ExtensionDtype handling
            return super().astype(dtype, copy=copy)
        
        # Handle cases where dtype is 'integer' or 'unsigned integer' and self contains NaN values
        elif dtype.kind in "iu" and self.isna().any():
            raise ValueError("Cannot convert float NaN to integer")
        
        # Handle cases where either codes or categories are empty
        elif len(self.codes) == 0 or len(self.categories) == 0:
            # Convert self to a numpy array with specified dtype and copy settings
            result = np.array(
                self,
                dtype=dtype,
                copy=copy,
            )
        
        else:
            # Performance optimization: Convert category codes instead of converting the entire array
            new_cats = self.categories._values
            
            try:
                # Attempt to convert new_cats to the specified dtype with copy settings
                new_cats = new_cats.astype(dtype=dtype, copy=copy)
                
                # Determine the fill value for NaNs based on dtype compatibility
                fill_value = self.categories._na_value
                if not is_valid_na_for_dtype(fill_value, dtype):
                    fill_value = lib.item_from_zerodim(
                        np.array(self.categories._na_value).astype(dtype)
                    )
            
            except (
                TypeError,  # downstream error msg for CategoricalIndex is misleading
                ValueError,
            ) as err:
                # Raise an error if conversion fails with a descriptive message
                msg = f"Cannot cast {self.categories.dtype} dtype to {dtype}"
                raise ValueError(msg) from err
            
            # Convert new category codes using take_nd function with platform-specific integer conversion
            result = take_nd(
                new_cats, ensure_platform_int(self._codes), fill_value=fill_value
            )
        
        # Return the coerced data with the appropriate dtype
        return result
    ) -> Self:
        """
        从推断的值构造一个分类变量。

        对于推断的类别（`dtype` 为 None），类别将会被排序。
        对于显式指定的 `dtype`，`inferred_categories` 将会被转换为适当的类型。

        Parameters
        ----------
        inferred_categories : Index
            推断的类别
        inferred_codes : Index
            推断的编码
        dtype : CategoricalDtype or 'category'
            分类变量的数据类型或者 'category' 字符串
        true_values : list, optional
            真值列表，可选的，默认为 ["True", "TRUE", "true"]。

        Returns
        -------
        Categorical
            返回构造的分类变量对象
        """
        from pandas import (
            Index,
            to_datetime,
            to_numeric,
            to_timedelta,
        )

        # 使用推断的类别创建一个索引对象
        cats = Index(inferred_categories)
        
        # 检查是否已知类别
        known_categories = (
            isinstance(dtype, CategoricalDtype) and dtype.categories is not None
        )

        if known_categories:
            # 如果已知类别，根据 dtype 的具体类型进行转换
            if is_any_real_numeric_dtype(dtype.categories.dtype):
                cats = to_numeric(inferred_categories, errors="coerce")
            elif lib.is_np_dtype(dtype.categories.dtype, "M"):
                cats = to_datetime(inferred_categories, errors="coerce")
            elif lib.is_np_dtype(dtype.categories.dtype, "m"):
                cats = to_timedelta(inferred_categories, errors="coerce")
            elif is_bool_dtype(dtype.categories.dtype):
                if true_values is None:
                    true_values = ["True", "TRUE", "true"]

                # 将 cats 转换为布尔类型的数组，并忽略类型检查的赋值错误
                cats = cats.isin(true_values)  # type: ignore[assignment]

        if known_categories:
            # 根据观察顺序重新编码为 dtype.categories 的顺序
            categories = dtype.categories
            codes = recode_for_categories(inferred_codes, cats, categories)
        elif not cats.is_monotonic_increasing:
            # 如果类别不是单调递增的，对其进行排序并重新编码
            unsorted = cats.copy()
            categories = cats.sort_values()

            codes = recode_for_categories(inferred_codes, unsorted, categories)
            dtype = CategoricalDtype(categories, ordered=False)
        else:
            # 如果是未知类别，创建一个无序的分类数据类型
            dtype = CategoricalDtype(cats, ordered=False)
            codes = inferred_codes

        # 返回使用类方法构造的新实例
        return cls._simple_new(codes, dtype=dtype)

    @classmethod
    def from_codes(
        cls,
        codes,
        categories=None,
        ordered=None,
        dtype: Dtype | None = None,
        validate: bool = True,
    ) -> Self:
        """
        从给定的代码和类别或数据类型创建一个分类类型。

        如果您已经具有代码和类别或数据类型，且不需要（计算密集型的）因子化步骤，
        则此构造函数非常有用，通常在构造函数中完成。

        如果您的数据不符合此约定，请使用普通构造函数。

        Parameters
        ----------
        codes : array-like of int
            整数数组，其中每个整数指向类别或dtype.categories中的一个类别，或者是-1表示NaN。
        categories : index-like, 可选
            分类的类别。项目需要是唯一的。
            如果这里没有给出类别，则必须在`dtype`中提供。
        ordered : bool, 可选
            是否将此分类视为有序分类。如果在这里或在`dtype`中未给出，则生成的分类将是无序的。
        dtype : CategoricalDtype 或 "category", 可选
            如果是 :class:`CategoricalDtype`，则不能与`categories`或`ordered`一起使用。
        validate : bool, 默认 True
            如果为True，则验证代码是否对该dtype有效。
            如果为False，则不验证代码的有效性。跳过验证可能会导致严重问题，例如段错误。

            .. versionadded:: 2.1.0

        Returns
        -------
        Categorical

        See Also
        --------
        codes : 分类的类别代码。
        CategoricalIndex : 具有底层``Categorical``的索引。

        Examples
        --------
        >>> dtype = pd.CategoricalDtype(["a", "b"], ordered=True)
        >>> pd.Categorical.from_codes(codes=[0, 1, 0, 1], dtype=dtype)
        ['a', 'b', 'a', 'b']
        Categories (2, object): ['a' < 'b']
        """
        dtype = CategoricalDtype._from_values_or_dtype(
            categories=categories, ordered=ordered, dtype=dtype
        )
        if dtype.categories is None:
            msg = (
                "必须在 'categories' 或 'dtype' 中提供类别。两者都为None。"
            )
            raise ValueError(msg)

        if validate:
            # 注意：非有效的代码可能会导致段错误
            codes = cls._validate_codes_for_dtype(codes, dtype=dtype)

        return cls._simple_new(codes, dtype=dtype)

    # ------------------------------------------------------------------
    # Categories/Codes/Ordered

    @property
    # 返回该分类变量的所有类别
    def categories(self) -> Index:
        """
        The categories of this categorical.

        Setting assigns new values to each category (effectively a rename of
        each individual category).

        The assigned value has to be a list-like object. All items must be
        unique and the number of items in the new categories must be the same
        as the number of items in the old categories.

        Raises
        ------
        ValueError
            If the new categories do not validate as categories or if the
            number of new categories is unequal the number of old categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(["a", "b", "c", "a"], dtype="category")
        >>> ser.cat.categories
        Index(['a', 'b', 'c'], dtype='object')

        >>> raw_cat = pd.Categorical(["a", "b", "c", "a"], categories=["b", "c", "d"])
        >>> ser = pd.Series(raw_cat)
        >>> ser.cat.categories
        Index(['b', 'c', 'd'], dtype='object')

        For :class:`pandas.Categorical`:

        >>> cat = pd.Categorical(["a", "b"], ordered=True)
        >>> cat.categories
        Index(['a', 'b'], dtype='object')

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(["a", "c", "b", "a", "c", "b"])
        >>> ci.categories
        Index(['a', 'b', 'c'], dtype='object')

        >>> ci = pd.CategoricalIndex(["a", "c"], categories=["c", "b", "a"])
        >>> ci.categories
        Index(['c', 'b', 'a'], dtype='object')
        """
        # 返回该分类变量的所有类别
        return self.dtype.categories

    @property
    def ordered(self) -> Ordered:
        """
        Whether the categories have an ordered relationship.

        See Also
        --------
        set_ordered : Set the ordered attribute.
        as_ordered : Set the Categorical to be ordered.
        as_unordered : Set the Categorical to be unordered.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(["a", "b", "c", "a"], dtype="category")
        >>> ser.cat.ordered
        False

        >>> raw_cat = pd.Categorical(["a", "b", "c", "a"], ordered=True)
        >>> ser = pd.Series(raw_cat)
        >>> ser.cat.ordered
        True

        For :class:`pandas.Categorical`:

        >>> cat = pd.Categorical(["a", "b"], ordered=True)
        >>> cat.ordered
        True

        >>> cat = pd.Categorical(["a", "b"], ordered=False)
        >>> cat.ordered
        False

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(["a", "b"], ordered=True)
        >>> ci.ordered
        True

        >>> ci = pd.CategoricalIndex(["a", "b"], ordered=False)
        >>> ci.ordered
        False
        """
        # 返回当前分类数据类型是否有序
        return self.dtype.ordered

    @property
    def codes(self) -> np.ndarray:
        """
        The category codes of this categorical index.

        Codes are an array of integers which are the positions of the actual
        values in the categories array.

        There is no setter, use the other categorical methods and the normal item
        setter to change values in the categorical.

        Returns
        -------
        ndarray[int]
            A non-writable view of the ``codes`` array.

        See Also
        --------
        Categorical.from_codes : Make a Categorical from codes.
        CategoricalIndex : An Index with an underlying ``Categorical``.

        Examples
        --------
        For :class:`pandas.Categorical`:

        >>> cat = pd.Categorical(["a", "b"], ordered=True)
        >>> cat.codes
        array([0, 1], dtype=int8)

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"])
        >>> ci.codes
        array([0, 1, 2, 0, 1, 2], dtype=int8)

        >>> ci = pd.CategoricalIndex(["a", "c"], categories=["c", "b", "a"])
        >>> ci.codes
        array([2, 0], dtype=int8)
        """
        # 获取当前分类索引的代码，返回一个不可写的视图
        v = self._codes.view()
        v.flags.writeable = False
        return v
    def _set_categories(self, categories, fastpath: bool = False) -> None:
        """
        Sets new categories inplace

        Parameters
        ----------
        fastpath : bool, default False
           Don't perform validation of the categories for uniqueness or nulls

        Examples
        --------
        >>> c = pd.Categorical(["a", "b"])
        >>> c
        ['a', 'b']
        Categories (2, object): ['a', 'b']

        >>> c._set_categories(pd.Index(["a", "c"]))
        >>> c
        ['a', 'c']
        Categories (2, object): ['a', 'c']
        """
        # 根据 fastpath 参数决定是否使用快速路径创建新的分类类型
        if fastpath:
            new_dtype = CategoricalDtype._from_fastpath(categories, self.ordered)
        else:
            # 使用给定的 categories 创建新的分类类型
            new_dtype = CategoricalDtype(categories, ordered=self.ordered)
        
        # 如果不使用快速路径，且原始分类类型不为空，新的分类类型长度必须与原始分类类型长度相同
        if (
            not fastpath
            and self.dtype.categories is not None
            and len(new_dtype.categories) != len(self.dtype.categories)
        ):
            raise ValueError(
                "new categories need to have the same number of "
                "items as the old categories!"
            )

        # 调用父类的初始化方法，用新的 dtype 初始化对象
        super().__init__(self._ndarray, new_dtype)

    def _set_dtype(self, dtype: CategoricalDtype) -> Self:
        """
        Internal method for directly updating the CategoricalDtype

        Parameters
        ----------
        dtype : CategoricalDtype

        Notes
        -----
        We don't do any validation here. It's assumed that the dtype is
        a (valid) instance of `CategoricalDtype`.
        """
        # 根据传入的 dtype 重编码当前的 codes，以适应新的 categories
        codes = recode_for_categories(self.codes, self.categories, dtype.categories)
        # 使用 _simple_new 方法创建一个新的对象，更新 dtype
        return type(self)._simple_new(codes, dtype=dtype)

    def set_ordered(self, value: bool) -> Self:
        """
        Set the ordered attribute to the boolean value.

        Parameters
        ----------
        value : bool
           Set whether this categorical is ordered (True) or not (False).
        """
        # 根据传入的 value 值创建一个新的有序的 CategoricalDtype
        new_dtype = CategoricalDtype(self.categories, ordered=value)
        # 复制当前对象，初始化为 NDArrayBacked 类的实例，并用新的 dtype 初始化
        cat = self.copy()
        NDArrayBacked.__init__(cat, cat._ndarray, new_dtype)
        # 返回新的对象
        return cat

    def as_ordered(self) -> Self:
        """
        Set the Categorical to be ordered.

        Returns
        -------
        Categorical
            Ordered Categorical.

        See Also
        --------
        as_unordered : Set the Categorical to be unordered.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(["a", "b", "c", "a"], dtype="category")
        >>> ser.cat.ordered
        False
        >>> ser = ser.cat.as_ordered()
        >>> ser.cat.ordered
        True

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(["a", "b", "c", "a"])
        >>> ci.ordered
        False
        >>> ci = ci.as_ordered()
        >>> ci.ordered
        True
        """
        # 调用 set_ordered 方法，将当前 Categorical 设置为有序
        return self.set_ordered(True)
    # 将当前的分类对象设置为无序状态。
    def as_unordered(self) -> Self:
        """
        Set the Categorical to be unordered.

        Returns
        -------
        Categorical
            Unordered Categorical.

        See Also
        --------
        as_ordered : Set the Categorical to be ordered.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> raw_cat = pd.Categorical(["a", "b", "c", "a"], ordered=True)
        >>> ser = pd.Series(raw_cat)
        >>> ser.cat.ordered
        True
        >>> ser = ser.cat.as_unordered()
        >>> ser.cat.ordered
        False

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(["a", "b", "c", "a"], ordered=True)
        >>> ci.ordered
        True
        >>> ci = ci.as_unordered()
        >>> ci.ordered
        False
        """
        # 调用 set_ordered 方法，将有序状态设置为 False，并返回调用结果
        return self.set_ordered(False)

    def set_categories(
        self, new_categories, ordered=None, rename: bool = False
    def rename_categories(self, new_categories) -> Self:
        """
        Rename categories.

        Parameters
        ----------
        new_categories : list-like, dict-like or callable
            New categories which will replace old categories.

            * list-like: all items must be unique and the number of items in
              the new categories must match the existing number of categories.

            * dict-like: specifies a mapping from
              old categories to new. Categories not contained in the mapping
              are passed through and extra categories in the mapping are
              ignored.

            * callable : a callable that is called on all items in the old
              categories and whose return values comprise the new categories.

        Returns
        -------
        Categorical
            Categorical with renamed categories.

        Raises
        ------
        ValueError
            If new categories are list-like and do not have the same number of
            items than the current categories or do not validate as categories

        See Also
        --------
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(["a", "a", "b"])
        >>> c.rename_categories([0, 1])
        [0, 0, 1]
        Categories (2, int64): [0, 1]

        For dict-like ``new_categories``, extra keys are ignored and
        categories not in the dictionary are passed through

        >>> c.rename_categories({"a": "A", "c": "C"})
        ['A', 'A', 'b']
        Categories (2, object): ['A', 'b']

        You may also provide a callable to create the new categories

        >>> c.rename_categories(lambda x: x.upper())
        ['A', 'A', 'B']
        Categories (2, object): ['A', 'B']
        """

        # 如果 new_categories 是类似字典的对象，则根据映射替换旧的类别
        if is_dict_like(new_categories):
            new_categories = [
                new_categories.get(item, item) for item in self.categories
            ]
        # 如果 new_categories 是可调用对象，则对旧类别中的每个项调用该函数，并生成新的类别列表
        elif callable(new_categories):
            new_categories = [new_categories(item) for item in self.categories]

        # 复制当前的 Categorical 对象
        cat = self.copy()
        # 使用新的类别列表来设置类别
        cat._set_categories(new_categories)
        # 返回具有重命名类别的新 Categorical 对象
        return cat
    def reorder_categories(self, new_categories, ordered=None) -> Self:
        """
        Reorder categories as specified in new_categories.

        ``new_categories`` need to include all old categories and no new category
        items.

        Parameters
        ----------
        new_categories : Index-like
           The categories in new order.
        ordered : bool, optional
           Whether or not the categorical is treated as an ordered categorical.
           If not given, do not change the ordered information.

        Returns
        -------
        Categorical
            Categorical with reordered categories.

        Raises
        ------
        ValueError
            If the new categories do not contain all old category items or any
            new ones

        See Also
        --------
        rename_categories : Rename categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(["a", "b", "c", "a"], dtype="category")
        >>> ser = ser.cat.reorder_categories(["c", "b", "a"], ordered=True)
        >>> ser
        0   a
        1   b
        2   c
        3   a
        dtype: category
        Categories (3, object): ['c' < 'b' < 'a']

        >>> ser.sort_values()
        2   c
        1   b
        0   a
        3   a
        dtype: category
        Categories (3, object): ['c' < 'b' < 'a']

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(["a", "b", "c", "a"])
        >>> ci
        CategoricalIndex(['a', 'b', 'c', 'a'], categories=['a', 'b', 'c'],
                         ordered=False, dtype='category')
        >>> ci.reorder_categories(["c", "b", "a"], ordered=True)
        CategoricalIndex(['a', 'b', 'c', 'a'], categories=['c', 'b', 'a'],
                         ordered=True, dtype='category')
        """
        # 检查新分类列表是否包含所有旧的分类，并且不包含任何新的分类项
        if (
            len(self.categories) != len(new_categories)
            or not self.categories.difference(new_categories).empty
        ):
            raise ValueError(
                "items in new_categories are not the same as in old categories"
            )
        # 调用 set_categories 方法重新设置分类，包括可选的有序信息
        return self.set_categories(new_categories, ordered=ordered)
    def add_categories(self, new_categories) -> Self:
        """
        Add new categories.

        `new_categories` will be included at the last/highest place in the
        categories and will be unused directly after this call.

        Parameters
        ----------
        new_categories : category or list-like of category
            The new categories to be included.

        Returns
        -------
        Categorical
            Categorical with new categories added.

        Raises
        ------
        ValueError
            If the new categories include old categories or do not validate as
            categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(["c", "b", "c"])
        >>> c
        ['c', 'b', 'c']
        Categories (2, object): ['b', 'c']

        >>> c.add_categories(["d", "a"])
        ['c', 'b', 'c']
        Categories (4, object): ['b', 'c', 'd', 'a']
        """

        # 如果 `new_categories` 不是类列表对象，则转换为列表
        if not is_list_like(new_categories):
            new_categories = [new_categories]
        
        # 检查新类别中是否包含已经存在的类别
        already_included = set(new_categories) & set(self.dtype.categories)
        if len(already_included) != 0:
            # 如果存在已经包含的类别，抛出 ValueError 异常
            raise ValueError(
                f"new categories must not include old categories: {already_included}"
            )

        # 如果 `new_categories` 具有 dtype 属性，使用 find_common_type 确定最合适的数据类型
        if hasattr(new_categories, "dtype"):
            from pandas import Series

            dtype = find_common_type(
                [self.dtype.categories.dtype, new_categories.dtype]
            )
            # 创建新的 Series 对象，包含原始类别和新类别，并指定数据类型
            new_categories = Series(
                list(self.dtype.categories) + list(new_categories), dtype=dtype
            )
        else:
            # 否则，将新类别添加到现有的类别列表中
            new_categories = list(self.dtype.categories) + list(new_categories)

        # 创建新的 CategoricalDtype 对象，包含更新后的类别和排序信息
        new_dtype = CategoricalDtype(new_categories, self.ordered)
        # 复制当前对象
        cat = self.copy()
        # 强制转换索引器的数据类型以匹配新的类别数据类型
        codes = coerce_indexer_dtype(cat._ndarray, new_dtype.categories)
        # 使用新的数据初始化 NDArrayBacked 对象
        NDArrayBacked.__init__(cat, codes, new_dtype)
        # 返回更新后的 Categorical 对象
        return cat
    def remove_categories(self, removals) -> Self:
        """
        Remove the specified categories.

        `removals` must be included in the old categories. Values which were in
        the removed categories will be set to NaN

        Parameters
        ----------
        removals : category or list of categories
           The categories which should be removed.

        Returns
        -------
        Categorical
            Categorical with removed categories.

        Raises
        ------
        ValueError
            If the removals are not contained in the categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(["a", "c", "b", "c", "d"])
        >>> c
        ['a', 'c', 'b', 'c', 'd']
        Categories (4, object): ['a', 'b', 'c', 'd']

        >>> c.remove_categories(["d", "a"])
        [NaN, 'c', 'b', 'c', NaN]
        Categories (2, object): ['b', 'c']
        """

        # Import the Index class from pandas
        from pandas import Index
        
        # If `removals` is not list-like, convert it into a list
        if not is_list_like(removals):
            removals = [removals]

        # Create an Index object from `removals`, drop any NaN values, and keep only unique values
        removals = Index(removals).unique().dropna()

        # Determine new categories based on whether the dtype is ordered or not
        new_categories = (
            self.dtype.categories.difference(removals, sort=False)
            if self.dtype.ordered is True
            else self.dtype.categories.difference(removals)
        )

        # Find categories in `removals` that are not present in current categories
        not_included = removals.difference(self.dtype.categories)

        # If there are categories in `removals` not present in current categories, raise an error
        if len(not_included) != 0:
            not_included = set(not_included)
            raise ValueError(f"removals must all be in old categories: {not_included}")

        # Return the result of setting the categories to `new_categories` with specified options
        return self.set_categories(new_categories, ordered=self.ordered, rename=False)
    # 定义一个方法来移除未使用的分类
    def remove_unused_categories(self) -> Self:
        """
        Remove categories which are not used.

        Returns
        -------
        Categorical
            Categorical with unused categories dropped.

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(["a", "c", "b", "c", "d"])
        >>> c
        ['a', 'c', 'b', 'c', 'd']
        Categories (4, object): ['a', 'b', 'c', 'd']

        >>> c[2] = "a"
        >>> c[4] = "c"
        >>> c
        ['a', 'c', 'a', 'c', 'c']
        Categories (4, object): ['a', 'b', 'c', 'd']

        >>> c.remove_unused_categories()
        ['a', 'c', 'a', 'c', 'c']
        Categories (2, object): ['a', 'c']
        """
        # 获取分类代码的唯一值和反向索引
        idx, inv = np.unique(self._codes, return_inverse=True)

        # 如果唯一值数组不为空且第一个元素为-1（表示缺失值的标记），则去除该元素
        if idx.size != 0 and idx[0] == -1:  # na sentinel
            idx, inv = idx[1:], inv - 1

        # 根据新的分类代码创建新的分类类别
        new_categories = self.dtype.categories.take(idx)
        # 根据新的类别和是否有序创建新的数据类型
        new_dtype = CategoricalDtype._from_fastpath(
            new_categories, ordered=self.ordered
        )
        # 将旧的分类代码转换为新的数据类型的索引数组
        new_codes = coerce_indexer_dtype(inv, new_dtype.categories)

        # 复制当前的分类对象
        cat = self.copy()
        # 使用新的分类代码和数据类型初始化复制后的分类对象
        NDArrayBacked.__init__(cat, new_codes, new_dtype)
        # 返回更新后的分类对象
        return cat

    # ------------------------------------------------------------------

    # 定义用于映射操作的魔术方法，通过操作符实现对分类的比较操作
    def map(
        self,
        mapper,
        na_action: Literal["ignore"] | None = None,
    __eq__ = _cat_compare_op(operator.eq)
    __ne__ = _cat_compare_op(operator.ne)
    __lt__ = _cat_compare_op(operator.lt)
    __gt__ = _cat_compare_op(operator.gt)
    __le__ = _cat_compare_op(operator.le)
    __ge__ = _cat_compare_op(operator.ge)

    # -------------------------------------------------------------

    # 验证器函数，用于验证要设置的值是否可哈希
    def _validate_setitem_value(self, value):
        if not is_hashable(value):
            # 如果值不可哈希，则将标量或者可哈希列表包装成列表
            return self._validate_listlike(value)
        else:
            # 如果值可哈希，则直接验证标量
            return self._validate_scalar(value)
    def _validate_scalar(self, fill_value):
        """
        Convert a user-facing fill_value to a representation to use with our
        underlying ndarray, raising TypeError if this is not possible.

        Parameters
        ----------
        fill_value : object
            The value to be validated and converted.

        Returns
        -------
        fill_value : int
            Validated and converted integer representation of fill_value.

        Raises
        ------
        TypeError
            If fill_value cannot be converted to a valid representation.
        """

        if is_valid_na_for_dtype(fill_value, self.categories.dtype):
            # If fill_value is a valid NA value for the dtype, set it to -1
            fill_value = -1
        elif fill_value in self.categories:
            # If fill_value is in the existing categories, convert it to its internal representation
            fill_value = self._unbox_scalar(fill_value)
        else:
            # Raise TypeError if fill_value is neither a valid NA nor in the current categories
            raise TypeError(
                "Cannot setitem on a Categorical with a new "
                f"category ({fill_value}), set the categories first"
            ) from None
        return fill_value

    @classmethod
    def _validate_codes_for_dtype(cls, codes, *, dtype: CategoricalDtype) -> np.ndarray:
        """
        Validate and convert codes to numpy array with dtype=np.int64.

        Parameters
        ----------
        codes : object
            Codes to be validated and converted to numpy array.
        dtype : CategoricalDtype
            Expected dtype for the codes.

        Returns
        -------
        np.ndarray
            Validated and converted numpy array of codes.

        Raises
        ------
        ValueError
            If codes contain NA values, are not array-like integers,
            or are out of bounds relative to dtype categories.
        """

        if isinstance(codes, ExtensionArray) and is_integer_dtype(codes.dtype):
            # If codes is an ExtensionArray and of integer type, convert to numpy array
            if isna(codes).any():
                raise ValueError("codes cannot contain NA values")
            codes = codes.to_numpy(dtype=np.int64)
        else:
            # Otherwise, treat codes as a general array and convert to numpy array
            codes = np.asarray(codes)

        if len(codes) and codes.dtype.kind not in "iu":
            # Ensure codes are array-like integers
            raise ValueError("codes need to be array-like integers")

        if len(codes) and (codes.max() >= len(dtype.categories) or codes.min() < -1):
            # Check if codes are within valid bounds relative to dtype categories
            raise ValueError("codes need to be between -1 and len(categories)-1")

        return codes

    # -------------------------------------------------------------

    @ravel_compat
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """
        The numpy array interface for Categorical objects.

        Users should not call this directly. It is invoked by functions like
        :func:`numpy.array` and :func:`numpy.asarray`.

        Parameters
        ----------
        dtype : np.dtype or None
            Specifies the dtype for the resulting numpy array.
        copy : bool or None, optional
            Unused parameter.

        Returns
        -------
        numpy.array
            A numpy array representing the categorical data.

        See Also
        --------
        numpy.asarray : Convert input to numpy.ndarray.

        Examples
        --------
        >>> cat = pd.Categorical(["a", "b"], ordered=True)

        The following calls `cat.__array__`

        >>> np.asarray(cat)
        array(['a', 'b'], dtype=object)
        """

        # Retrieve the corresponding values from categories using _codes
        ret = take_nd(self.categories._values, self._codes)

        if dtype and np.dtype(dtype) != self.categories.dtype:
            # Convert ret to specified dtype if dtype is provided and different from categories' dtype
            return np.asarray(ret, dtype)

        # Ensure that __array__ gets all the way to an ndarray for Categorical[ExtensionArray] types
        return np.asarray(ret)
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # 对于二元操作，使用我们自定义的双下划线方法来处理
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            # 例如，test_numpy_ufuncs_out
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        if method == "reduce":
            # 例如，TestCategoricalAnalytics::test_min_max_ordered
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        # 对于所有其他情况，暂时抛出异常（类似于 Series.__array_prepare__ 中的处理）
        raise TypeError(
            f"Object with dtype {self.dtype} cannot perform "
            f"the numpy op {ufunc.__name__}"
        )

    def __setstate__(self, state) -> None:
        """
        使对象可被 pickle 序列化所必需的方法
        """
        if not isinstance(state, dict):
            return super().__setstate__(state)

        if "_dtype" not in state:
            state["_dtype"] = CategoricalDtype(state["_categories"], state["_ordered"])

        if "_codes" in state and "_ndarray" not in state:
            # 向后兼容，改变了属性和属性的定义
            state["_ndarray"] = state.pop("_codes")

        super().__setstate__(state)

    @property
    def nbytes(self) -> int:
        """
        返回由 .codes 和 .dtype.categories.values 所占用的内存字节数
        """
        return self._codes.nbytes + self.dtype.categories.values.nbytes

    def memory_usage(self, deep: bool = False) -> int:
        """
        计算对象值所占用的内存大小

        Parameters
        ----------
        deep : bool
            是否深入检查数据，用于查询系统级内存消耗

        Returns
        -------
        使用的字节数

        Notes
        -----
        如果 deep=False，则内存使用不包括非数组组成部分消耗的内存

        See Also
        --------
        numpy.ndarray.nbytes
        """
        return self._codes.nbytes + self.dtype.categories.memory_usage(deep=deep)

    def isna(self) -> npt.NDArray[np.bool_]:
        """
        检测缺失值

        检测到缺失值（.codes 中的 -1）。

        Returns
        -------
        np.ndarray[bool]，指示值是否为空

        See Also
        --------
        isna : 顶级 isna。
        isnull : isna 的别名。
        Categorical.notna : Categorical.isna 的布尔反转。

        """
        return self._codes == -1

    isnull = isna
    # 返回当前对象的非空值掩码，通过调用 isna 方法的结果取反得到
    def notna(self) -> npt.NDArray[np.bool_]:
        """
        Inverse of isna

        Both missing values (-1 in .codes) and NA as a category are detected as
        null.

        Returns
        -------
        np.ndarray[bool] of whether my values are not null

        See Also
        --------
        notna : Top-level notna.
        notnull : Alias of notna.
        Categorical.isna : Boolean inverse of Categorical.notna.

        """
        return ~self.isna()

    # notnull 是 notna 方法的别名
    notnull = notna

    # 返回每个类别的计数信息组成的 Series 对象
    def value_counts(self, dropna: bool = True) -> Series:
        """
        Return a Series containing counts of each category.

        Every category will have an entry, even those with a count of 0.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
        # 导入 pandas 中的 CategoricalIndex 和 Series 类
        from pandas import (
            CategoricalIndex,
            Series,
        )

        # 获取类别编码和类别列表
        code, cat = self._codes, self.categories
        ncat, mask = (len(cat), code >= 0)
        ix, clean = np.arange(ncat), mask.all()

        # 根据 dropna 和 clean 条件确定是否需要过滤 NaN 值或者整体都是有效值
        if dropna or clean:
            obs = code if clean else code[mask]
            count = np.bincount(obs, minlength=ncat or 0)
        else:
            count = np.bincount(np.where(mask, code, ncat))
            ix = np.append(ix, -1)

        # 强制将 ix 转换为适合当前 dtype 类别的索引类型
        ix = coerce_indexer_dtype(ix, self.dtype.categories)
        ix_categorical = self._from_backing_data(ix)

        # 返回计数结果的 Series 对象
        return Series(
            count,
            index=CategoricalIndex(ix_categorical),
            dtype="int64",
            name="count",
            copy=False,
        )

    # 类方法 _empty 返回一个空数组，类似于 np.empty，但使用指定的类别 dtype
    # 注意，使用 np.empty 而不是 np.zeros，因为后者可能会生成不被当前 dtype 支持的编码
    # 这样会导致 repr(result) 可能出现段错误
    @classmethod
    def _empty(  # type: ignore[override]
        cls, shape: Shape, dtype: CategoricalDtype
    ) -> Self:
        """
        Analogous to np.empty(shape, dtype=dtype)

        Parameters
        ----------
        shape : tuple[int]
        dtype : CategoricalDtype
        """
        # 使用空序列创建一个 arr 对象，指定 dtype
        arr = cls._from_sequence([], dtype=dtype)

        # 使用 np.zeros 而不是 np.empty，以避免出现不被当前 dtype 支持的编码
        backing = np.zeros(shape, dtype=arr._ndarray.dtype)

        # 从 backing 数据创建一个新的 arr 对象并返回
        return arr._from_backing_data(backing)
    # 定义一个方法，返回分类数据的值数组或扩展数组，用于内部与 pandas 格式兼容
    def _internal_get_values(self) -> ArrayLike:
        """
        Return the values.

        For internal compatibility with pandas formatting.

        Returns
        -------
        np.ndarray or ExtensionArray
            A numpy array or ExtensionArray of the same dtype as
            categorical.categories.dtype.
        """
        # 如果是 datetime 和 period 索引，则返回 Index 以保留元数据
        if needs_i8_conversion(self.categories.dtype):
            # 返回根据 self._codes 获取的 categories 的值，填充值为 NaT
            return self.categories.take(self._codes, fill_value=NaT)._values
        # 如果分类数据类型是整数且 _codes 中包含 -1
        elif is_integer_dtype(self.categories.dtype) and -1 in self._codes:
            # 将 categories 转换为对象类型后，根据 _codes 获取值，填充值为 np.nan
            return (
                self.categories.astype("object")
                .take(self._codes, fill_value=np.nan)
                ._values
            )
        # 默认情况下返回 self 的 numpy 数组形式
        return np.array(self)

    # 检查分类数据是否是有序的，如果不是则抛出 TypeError 异常
    def check_for_ordered(self, op) -> None:
        """assert that we are ordered"""
        if not self.ordered:
            raise TypeError(
                f"Categorical is not ordered for operation {op}\n"
                "you can use .as_ordered() to change the "
                "Categorical to an ordered one\n"
            )

    # 返回对分类数据进行排序的索引数组
    def argsort(
        self, *, ascending: bool = True, kind: SortKind = "quicksort", **kwargs
    ) -> npt.NDArray[np.intp]:
        """
        Return the indices that would sort the Categorical.

        Missing values are sorted at the end.

        Parameters
        ----------
        ascending : bool, default True
            Whether the indices should result in an ascending
            or descending sort.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            Sorting algorithm.
        **kwargs:
            passed through to :func:`numpy.argsort`.

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        numpy.ndarray.argsort

        Notes
        -----
        While an ordering is applied to the category values, arg-sorting
        in this context refers more to organizing and grouping together
        based on matching category values. Thus, this function can be
        called on an unordered Categorical instance unlike the functions
        'Categorical.min' and 'Categorical.max'.

        Examples
        --------
        >>> pd.Categorical(["b", "b", "a", "c"]).argsort()
        array([2, 0, 1, 3])

        >>> cat = pd.Categorical(
        ...     ["b", "b", "a", "c"], categories=["c", "b", "a"], ordered=True
        ... )
        >>> cat.argsort()
        array([3, 0, 1, 2])

        Missing values are placed at the end

        >>> cat = pd.Categorical([2, None, 1])
        >>> cat.argsort()
        array([2, 0, 1])
        """
        # 调用父类的 argsort 方法，返回排序后的索引数组
        return super().argsort(ascending=ascending, kind=kind, **kwargs)

    # 排序方法的类型注解，指示支持两种重载方式
    @overload
    def sort_values(
        self,
        *,
        inplace: Literal[False] = ...,
        ascending: bool = ...,
        na_position: str = ...,
    ) -> Self: ...
    # 定义一个方法来对类别值进行排序操作
    def sort_values(
        self,
        *,
        inplace: bool = False,  # 是否原地排序，默认为False
        ascending: bool = True,  # 是否升序排列，默认为True
        na_position: str = "last",  # 缺失值NaN的位置，默认放在最后
    ) -> Self | None:
        """
        Sort the Categorical by category value returning a new
        Categorical by default.

        While an ordering is applied to the category values, sorting in this
        context refers more to organizing and grouping together based on
        matching category values. Thus, this function can be called on an
        unordered Categorical instance unlike the functions 'Categorical.min'
        and 'Categorical.max'.

        Parameters
        ----------
        inplace : bool, default False
            Do operation in place. 是否原地操作。
        ascending : bool, default True
            Order ascending. 是否升序排列。False则降序排列。
        na_position : {'first', 'last'} (optional, default='last')
            'first' puts NaNs at the beginning
            'last' puts NaNs at the end
            控制NaN值的位置，'first'将NaN值放在最前面，'last'放在最后面。

        Returns
        -------
        Categorical or None
        返回一个新的分类对象或者None。

        See Also
        --------
        Categorical.sort
        Series.sort_values

        Examples
        --------
        >>> c = pd.Categorical([1, 2, 2, 1, 5])
        >>> c
        [1, 2, 2, 1, 5]
        Categories (3, int64): [1, 2, 5]
        >>> c.sort_values()
        [1, 1, 2, 2, 5]
        Categories (3, int64): [1, 2, 5]
        >>> c.sort_values(ascending=False)
        [5, 2, 2, 1, 1]
        Categories (3, int64): [1, 2, 5]

        >>> c = pd.Categorical([1, 2, 2, 1, 5])

        'sort_values' behaviour with NaNs. Note that 'na_position'
        is independent of the 'ascending' parameter:

        >>> c = pd.Categorical([np.nan, 2, 2, np.nan, 5])
        >>> c
        [NaN, 2, 2, NaN, 5]
        Categories (2, int64): [2, 5]
        >>> c.sort_values()
        [2, 2, 5, NaN, NaN]
        Categories (2, int64): [2, 5]
        >>> c.sort_values(ascending=False)
        [5, 2, 2, NaN, NaN]
        Categories (2, int64): [2, 5]
        >>> c.sort_values(na_position="first")
        [NaN, NaN, 2, 2, 5]
        Categories (2, int64): [2, 5]
        >>> c.sort_values(ascending=False, na_position="first")
        [NaN, NaN, 5, 2, 2]
        Categories (2, int64): [2, 5]
        """
        inplace = validate_bool_kwarg(inplace, "inplace")  # 确保 inplace 参数是有效的布尔值

        if na_position not in ["last", "first"]:
            raise ValueError(f"invalid na_position: {na_position!r}")  # 如果na_position不在指定的列表中，则引发值错误异常

        sorted_idx = nargsort(self, ascending=ascending, na_position=na_position)  # 使用指定参数对self进行排序，并返回排序后的索引

        if not inplace:
            codes = self._codes[sorted_idx]  # 如果不是原地排序，则根据排序后的索引获取对应的代码值
            return self._from_backing_data(codes)  # 根据排序后的代码值返回一个新的分类对象
        self._codes[:] = self._codes[sorted_idx]  # 如果是原地排序，则直接将排序后的代码值赋值回self._codes
        return None  # 原地排序返回None
    def _rank(
        self,
        *,
        axis: AxisInt = 0,  # 定义默认参数 axis 为整数类型，表示沿着哪个轴进行排名，默认为 0
        method: str = "average",  # 定义默认参数 method 为字符串类型，表示排名方法，默认为 "average"
        na_option: str = "keep",  # 定义默认参数 na_option 为字符串类型，表示处理缺失值的选项，默认为 "keep"
        ascending: bool = True,  # 定义默认参数 ascending 为布尔类型，表示是否升序排列，默认为 True
        pct: bool = False,  # 定义默认参数 pct 为布尔类型，表示是否返回百分位排名，默认为 False
    ):
        """
        See Series.rank.__doc__.
        """
        if axis != 0:  # 如果 axis 不等于 0，则抛出 NotImplementedError 异常
            raise NotImplementedError
        vff = self._values_for_rank()  # 调用 _values_for_rank() 方法，获取用于排名的值
        return algorithms.rank(  # 调用算法进行排名，返回排名结果
            vff,
            axis=axis,
            method=method,
            na_option=na_option,
            ascending=ascending,
            pct=pct,
        )

    def _values_for_rank(self) -> np.ndarray:
        """
        For correctly ranking ordered categorical data. See GH#15420

        Ordered categorical data should be ranked on the basis of
        codes with -1 translated to NaN.

        Returns
        -------
        numpy.array

        """
        from pandas import Series  # 导入 Series 类

        if self.ordered:  # 如果是有序分类数据
            values = self.codes  # 获取分类编码作为排名的基础
            mask = values == -1  # 创建掩码，标记为 -1 的值
            if mask.any():  # 如果掩码中有任何 True 值
                values = values.astype("float64")  # 将 values 数组转换为 float64 类型
                values[mask] = np.nan  # 将掩码中为 True 的位置设置为 NaN
        elif is_any_real_numeric_dtype(self.categories.dtype):  # 如果不是有序分类但是数据类型为实数
            values = np.array(self)  # 将当前对象转换为 numpy 数组
        else:  # 否则，重新排序分类（以便排名使用浮点数编码）
            values = np.array(
                self.rename_categories(
                    Series(self.categories, copy=False).rank().values
                )
            )
        return values  # 返回用于排名的值数组

    def _hash_pandas_object(
        self, *, encoding: str, hash_key: str, categorize: bool
    ) -> npt.NDArray[np.uint64]:
        """
        Hash a Categorical by hashing its categories, and then mapping the codes
        to the hashes.

        Parameters
        ----------
        encoding : str
        hash_key : str
        categorize : bool
            Ignored for Categorical.

        Returns
        -------
        np.ndarray[uint64]
        """
        # Note we ignore categorize, as we are already Categorical.
        from pandas.core.util.hashing import hash_array  # 导入 hash_array 函数

        # Convert ExtensionArrays to ndarrays
        values = np.asarray(self.categories._values)  # 获取分类的值，并转换为 ndarray

        # 使用 hash_array 函数对分类的值进行哈希处理，返回哈希值数组
        hashed = hash_array(values, encoding, hash_key, categorize=False)

        # 我们使用 uint64 类型，因为我们不直接支持缺失值
        # 我们不希望使用 take_nd 函数，因为它会强制转换为 float
        # 而是直接使用最大的 np.uint64 构造结果，作为缺失值指示符
        #
        # TODO: GH#15362

        mask = self.isna()  # 获取是否为缺失值的布尔掩码
        if len(hashed):
            result = hashed.take(self._codes)  # 使用 _codes 数组获取相应的哈希值
        else:
            result = np.zeros(len(mask), dtype="uint64")  # 如果 hashed 为空，创建全零数组作为结果

        if mask.any():  # 如果存在缺失值
            result[mask] = lib.u8max  # 将缺失值位置设为最大的 uint64 值

        return result  # 返回最终的哈希结果数组
    # 返回 self._ndarray 数组作为结果
    def _codes(self) -> np.ndarray:
        return self._ndarray

    # 根据索引 i 返回对应的分类值，如果 i 为 -1，则返回 np.nan
    def _box_func(self, i: int):
        if i == -1:
            return np.nan
        return self.categories[i]

    # 根据 key 返回其在 categories 中的位置（索引），并转换为与 self.codes 相同的数据类型以提高性能
    def _unbox_scalar(self, key) -> int:
        # searchsorted 非常关键性能，将 codes 转换为与 self.codes 相同的数据类型以获得更快的性能
        code = self.categories.get_loc(key)
        code = self._ndarray.dtype.type(code)
        return code

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator:
        """
        返回一个迭代器，遍历此分类数据的值。
        """
        if self.ndim == 1:
            return iter(self._internal_get_values().tolist())
        else:
            return (self[n] for n in range(len(self)))

    def __contains__(self, key) -> bool:
        """
        如果 `key` 在此分类数据中，则返回 True。
        """
        # 如果 key 是 NaN，则检查 self 中是否有任何 NaN。
        if is_valid_na_for_dtype(key, self.categories.dtype):
            return bool(self.isna().any())

        return contains(self, key, container=self._codes)

    # ------------------------------------------------------------------
    # 渲染方法

    # error: Return type "None" of "_formatter" incompatible with return
    # type "Callable[[Any], str | None]" in supertype "ExtensionArray"
    def _formatter(self, boxed: bool = False) -> None:  # type: ignore[override]
        """
        返回 None，将导致 format_array 进行推断。
        """
        return None

    def _repr_categories(self) -> list[str]:
        """
        返回 categories 的基本 repr 列表。
        """
        max_categories = (
            10
            if get_option("display.max_categories") == 0
            else get_option("display.max_categories")
        )
        from pandas.io.formats import format as fmt

        format_array = partial(
            fmt.format_array, formatter=None, quoting=QUOTE_NONNUMERIC
        )
        if len(self.categories) > max_categories:
            num = max_categories // 2
            head = format_array(self.categories[:num]._values)
            tail = format_array(self.categories[-num:]._values)
            category_strs = head + ["..."] + tail
        else:
            category_strs = format_array(self.categories._values)

        # 去除所有列前面的空格，这些空格是 format_array 添加的...
        category_strs = [x.strip() for x in category_strs]
        return category_strs
    # 返回一个字符串表示的对象尾部信息
    def _get_repr_footer(self) -> str:
        # 调用 _repr_categories 方法获取类别字符串列表
        category_strs = self._repr_categories()
        # 将类别的数据类型转换为字符串
        dtype = str(self.categories.dtype)
        # 构建类别头信息
        levheader = f"Categories ({len(self.categories)}, {dtype}): "
        # 获取终端的宽度和高度信息
        width, _ = get_terminal_size()
        # 获取显示宽度的设置，或者使用终端的宽度
        max_width = get_option("display.width") or width
        # 如果在 IPython 前端，则最大宽度为 0，即不换行
        if console.in_ipython_frontend():
            max_width = 0
        # 初始化类别字符串为空
        levstring = ""
        # 是否是第一行
        start = True
        # 初始化当前列长度为头部长度
        cur_col_len = len(levheader)  # header
        # 确定分隔符的长度和内容
        sep_len, sep = (3, " < ") if self.ordered else (2, ", ")
        # 构建行分隔符
        linesep = f"{sep.rstrip()}\n"  # remove whitespace
        # 遍历类别字符串列表
        for val in category_strs:
            # 如果最大宽度不为 0，且当前列长度加上分隔符和当前值的长度大于最大宽度
            if max_width != 0 and cur_col_len + sep_len + len(val) > max_width:
                # 添加行分隔符和额外空格
                levstring += linesep + (" " * (len(levheader) + 1))
                # 更新当前列长度为头部长度加一个空格
                cur_col_len = len(levheader) + 1  # header + a whitespace
            # 如果不是第一行，则添加分隔符
            elif not start:
                levstring += sep
                cur_col_len += len(val)
            # 添加当前值到类别字符串
            levstring += val
            # 更新 start 为 False，表示不是第一行
            start = False
        # 替换字符串中的特定内容以节省空间
        return f"{levheader}[{levstring.replace(' < ... < ', ' ... ')}]"

    # 返回对象的值的字符串表示
    def _get_values_repr(self) -> str:
        # 导入格式化模块
        from pandas.io.formats import format as fmt

        # 断言长度大于 0
        assert len(self) > 0

        # 获取内部值
        vals = self._internal_get_values()
        # 使用格式化模块格式化值数组
        fmt_values = fmt.format_array(
            vals,
            None,
            float_format=None,
            na_rep="NaN",
            quoting=QUOTE_NONNUMERIC,
        )

        # 去除每个格式化值的首尾空格
        fmt_values = [i.strip() for i in fmt_values]
        # 使用逗号连接格式化后的值数组
        joined = ", ".join(fmt_values)
        # 构建结果字符串
        result = "[" + joined + "]"
        # 返回结果字符串
        return result

    # 返回对象的字符串表示
    def __repr__(self) -> str:
        """
        String representation.
        """
        # 获取尾部字符串表示
        footer = self._get_repr_footer()
        # 获取对象的长度
        length = len(self)
        # 设置最大长度为 10
        max_len = 10
        # 如果长度大于最大长度
        if length > max_len:
            # 在长情况下，不显示所有条目，而是在 __repr__ 中添加长度信息
            num = max_len // 2
            # 获取头部和尾部的字符串表示
            head = self[:num]._get_values_repr()
            tail = self[-(max_len - num) :]._get_values_repr()
            # 构建显示部分内容
            body = f"{head[:-1]}, ..., {tail[1:]}"
            # 构建长度信息字符串
            length_info = f"Length: {len(self)}"
            # 构建最终结果字符串
            result = f"{body}\n{length_info}\n{footer}"
        # 如果长度大于 0
        elif length > 0:
            # 获取值的字符串表示
            body = self._get_values_repr()
            # 构建结果字符串
            result = f"{body}\n{footer}"
        else:
            # 如果长度为 0，使用逗号而不是换行符获取更紧凑的 __repr__
            body = "[]"
            # 构建结果字符串
            result = f"{body}, {footer}"

        # 返回最终结果字符串
        return result
    def _validate_listlike(self, value):
        # NB: here we assume scalar-like tuples have already been excluded
        # 要求输入的 value 转换为数组，并确保排除了类似标量的元组
        value = extract_array(value, extract_numpy=True)

        # require identical categories set
        # 如果输入的 value 是 Categorical 类型
        if isinstance(value, Categorical):
            # 检查当前实例的 dtype 是否与 value 的 dtype 相同
            if self.dtype != value.dtype:
                raise TypeError(
                    "Cannot set a Categorical with another, "
                    "without identical categories"
                )
            # 当 dtype 相同时，意味着类别相匹配，可以使用当前实例的类别对 value 进行编码
            value = self._encode_with_my_categories(value)
            return value._codes

        from pandas import Index

        # tupleize_cols=False for e.g. test_fillna_iterable_category GH#41914
        # 使用 value 创建 Index 对象，并确保不将其作为列元组处理
        to_add = Index._with_infer(value, tupleize_cols=False).difference(
            self.categories
        )

        # no assignments of values not in categories, but it's always ok to set
        # something to np.nan
        # 如果 to_add 中存在不在当前实例类别中的值，则抛出异常
        if len(to_add) and not isna(to_add).all():
            raise TypeError(
                "Cannot setitem on a Categorical with a new "
                "category, set the categories first"
            )

        # 获取 value 的编码，并将其转换为与实例的 dtype 相同的类型
        codes = self.categories.get_indexer(value)
        return codes.astype(self._ndarray.dtype, copy=False)

    def _reverse_indexer(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        """
        Compute the inverse of a categorical, returning
        a dict of categories -> indexers.

        *This is an internal function*

        Returns
        -------
        Dict[Hashable, np.ndarray[np.intp]]
            dict of categories -> indexers

        Examples
        --------
        >>> c = pd.Categorical(list("aabca"))
        >>> c
        ['a', 'a', 'b', 'c', 'a']
        Categories (3, object): ['a', 'b', 'c']
        >>> c.categories
        Index(['a', 'b', 'c'], dtype='object')
        >>> c.codes
        array([0, 0, 1, 2, 0], dtype=int8)
        >>> c._reverse_indexer()
        {'a': array([0, 1, 4]), 'b': array([2]), 'c': array([3])}

        """
        categories = self.categories
        # 使用 groupsort_indexer 函数对 self.codes 进行分组排序索引
        r, counts = libalgos.groupsort_indexer(
            ensure_platform_int(self.codes), categories.size
        )
        counts = ensure_int64(counts).cumsum()
        # 生成 categories 到索引数组的映射字典
        _result = (r[start:end] for start, end in zip(counts, counts[1:]))
        return dict(zip(categories, _result))

    # ------------------------------------------------------------------
    # Reductions

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        # 调用超类的 _reduce 方法，返回结果
        result = super()._reduce(name, skipna=skipna, keepdims=keepdims, **kwargs)
        # 如果操作名称为 "argmax" 或 "argmin"，直接返回结果而不用 Categorical 类包装
        if name in ["argmax", "argmin"]:
            return result
        # 如果 keepdims 为 True，则返回一个新的类型相同的 Categorical 实例
        if keepdims:
            return type(self)(result, dtype=self.dtype)
        else:
            return result
    # 确定轴的有效性，确保在范围内
    nv.validate_minmax_axis(kwargs.get("axis", 0))
    # 验证最小值函数参数，确保正确性
    nv.validate_min((), kwargs)
    # 检查当前分类数据是否为有序，否则引发TypeError异常
    self.check_for_ordered("min")

    # 如果 _codes 长度为零，返回数据类型的 NA 值
    if not len(self._codes):
        return self.dtype.na_value

    # 创建布尔掩码以确定有效数据点
    good = self._codes != -1
    # 如果存在无效数据点且允许跳过无效数据，则计算有效数据点的最小值
    if not good.all():
        if skipna and good.any():
            pointer = self._codes[good].min()
        else:
            # 如果不允许跳过无效数据或没有有效数据点，则返回 NaN
            return np.nan
    else:
        # 计算所有数据点的最小值
        pointer = self._codes.min()

    # 将最终结果封装并返回
    return self._wrap_reduction_result(None, pointer)



    # 确定轴的有效性，确保在范围内
    nv.validate_minmax_axis(kwargs.get("axis", 0))
    # 验证最大值函数参数，确保正确性
    nv.validate_max((), kwargs)
    # 检查当前分类数据是否为有序，否则引发TypeError异常
    self.check_for_ordered("max")

    # 如果 _codes 长度为零，返回数据类型的 NA 值
    if not len(self._codes):
        return self.dtype.na_value

    # 创建布尔掩码以确定有效数据点
    good = self._codes != -1
    # 如果存在无效数据点且允许跳过无效数据，则计算有效数据点的最大值
    if not good.all():
        if skipna and good.any():
            pointer = self._codes[good].max()
        else:
            # 如果不允许跳过无效数据或没有有效数据点，则返回 NaN
            return np.nan
    else:
        # 计算所有数据点的最大值
        pointer = self._codes.max()

    # 将最终结果封装并返回
    return self._wrap_reduction_result(None, pointer)



    # 获取分类数据的众数，即出现频率最高的数据值
    codes = self._codes
    mask = None
    # 如果指定删除缺失值，则生成缺失值掩码
    if dropna:
        mask = self.isna()

    # 调用算法以计算数据的众数
    res_codes = algorithms.mode(codes, mask=mask)
    res_codes = cast(np.ndarray, res_codes)
    # 确保计算结果的数据类型与原始数据相同
    assert res_codes.dtype == codes.dtype
    # 根据计算结果创建新的分类数据对象并返回
    res = self._from_backing_data(res_codes)
    return res
    def unique(self) -> Self:
        """
        Return the ``Categorical`` which ``categories`` and ``codes`` are
        unique.

        .. versionchanged:: 1.3.0

            Previously, unused categories were dropped from the new categories.

        Returns
        -------
        Categorical

        See Also
        --------
        pandas.unique
        CategoricalIndex.unique
        Series.unique : Return unique values of Series object.

        Examples
        --------
        >>> pd.Categorical(list("baabc")).unique()
        ['b', 'a', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> pd.Categorical(list("baab"), categories=list("abc"), ordered=True).unique()
        ['b', 'a']
        Categories (3, object): ['a' < 'b' < 'c']
        """
        # 调用父类方法返回当前Categorical对象的唯一值
        return super().unique()

    def _cast_quantile_result(self, res_values: np.ndarray) -> np.ndarray:
        # 确保结果值的数据类型与当前Categorical对象的数据类型一致
        assert res_values.dtype == self._ndarray.dtype
        return res_values

    def equals(self, other: object) -> bool:
        """
        Returns True if categorical arrays are equal.

        Parameters
        ----------
        other : `Categorical`

        Returns
        -------
        bool
        """
        # 如果other不是Categorical对象，则返回False
        if not isinstance(other, Categorical):
            return False
        # 如果分类的类别与排列一致，则对other进行编码后比较_codes数组是否相等
        elif self._categories_match_up_to_permutation(other):
            other = self._encode_with_my_categories(other)
            return np.array_equal(self._codes, other._codes)
        return False

    def _accumulate(self, name: str, skipna: bool = True, **kwargs) -> Self:
        # 根据name参数选择累积函数，如cummin或cummax
        func: Callable
        if name == "cummin":
            func = np.minimum.accumulate
        elif name == "cummax":
            func = np.maximum.accumulate
        else:
            # 如果name参数不支持当前Categorical对象的累积操作，则引发异常
            raise TypeError(f"Accumulation {name} not supported for {type(self)}")
        # 检查当前Categorical对象是否已排序
        self.check_for_ordered(name)

        # 复制当前对象的codes数组
        codes = self.codes.copy()
        # 标记NaN值的位置
        mask = self.isna()
        # 对于cummin函数，将NaN位置的codes设置为最大整数值
        if func == np.minimum.accumulate:
            codes[mask] = np.iinfo(codes.dtype.type).max
        # 对于cummax函数，不需要更改NaN位置的codes，因为codes[mask]已经是-1
        if not skipna:
            # 如果不跳过NaN值，则对mask进行累积操作
            mask = np.maximum.accumulate(mask)

        # 应用累积函数到codes数组
        codes = func(codes)
        # 将NaN位置的codes设置为-1
        codes[mask] = -1
        # 返回一个新的Categorical对象，使用累积后的codes数组和指定的数据类型
        return self._simple_new(codes, dtype=self._dtype)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self], axis: AxisInt = 0) -> Self:
        # 导入 pandas 库中的 union_categoricals 函数
        from pandas.core.dtypes.concat import union_categoricals

        # 取第一个对象作为参考对象
        first = to_concat[0]
        # 检查指定的轴是否超出参考对象的维度
        if axis >= first.ndim:
            raise ValueError(
                f"axis {axis} is out of bounds for array of dimension {first.ndim}"
            )

        # 如果指定轴为1，则执行以下操作
        if axis == 1:
            # 检查所有对象是否都是二维的
            if not all(x.ndim == 2 for x in to_concat):
                raise ValueError

            # 将所有对象按列展平，并放入列表 tc_flat 中
            tc_flat = []
            for obj in to_concat:
                tc_flat.extend([obj[:, i] for i in range(obj.shape[1])])

            # 调用 _concat_same_type 函数，按照轴0进行连接
            res_flat = cls._concat_same_type(tc_flat, axis=0)

            # 将结果按指定顺序重新排列成二维数组
            result = res_flat.reshape(len(first), -1, order="F")
            return result

        # 如果指定轴不为1，则执行以下操作
        # 使用 union_categoricals 函数将对象列表合并为一个对象
        result = union_categoricals(to_concat)  # type: ignore[assignment]
        return result

    # ------------------------------------------------------------------

    def _encode_with_my_categories(self, other: Categorical) -> Categorical:
        """
        Re-encode another categorical using this Categorical's categories.

        Notes
        -----
        This assumes we have already checked
        self._categories_match_up_to_permutation(other).
        """
        # 使用 recode_for_categories 函数对传入的 other 对象进行重新编码
        codes = recode_for_categories(
            other.codes, other.categories, self.categories, copy=False
        )
        # 调用 _from_backing_data 方法，基于编码数据创建新的 Categorical 对象
        return self._from_backing_data(codes)

    def _categories_match_up_to_permutation(self, other: Categorical) -> bool:
        """
        Returns True if categoricals are the same dtype
          same categories, and same ordered

        Parameters
        ----------
        other : Categorical

        Returns
        -------
        bool
        """
        # 检查两个 Categorical 对象的数据类型、类别和顺序是否完全匹配
        return hash(self.dtype) == hash(other.dtype)

    def describe(self) -> DataFrame:
        """
        Describes this Categorical

        Returns
        -------
        description: `DataFrame`
            A dataframe with frequency and counts by category.
        """
        # 计算每个类别的计数和频率
        counts = self.value_counts(dropna=False)
        freqs = counts / counts.sum()

        # 导入 Index 和 concat 函数
        from pandas import Index
        from pandas.core.reshape.concat import concat

        # 将计数和频率合并成一个 DataFrame 对象
        result = concat([counts, freqs], ignore_index=True, axis=1)
        # 设置结果 DataFrame 的列名和索引名
        result.columns = Index(["counts", "freqs"])
        result.index.name = "categories"

        return result
    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        """
        Check whether `values` are contained in Categorical.

        Return a boolean NumPy Array showing whether each element in
        the Categorical matches an element in the passed sequence of
        `values` exactly.

        Parameters
        ----------
        values : np.ndarray or ExtensionArray
            The sequence of values to test. Passing in a single string will
            raise a ``TypeError``. Instead, turn a single string into a
            list of one element.

        Returns
        -------
        np.ndarray[bool]

        Raises
        ------
        TypeError
          * If `values` is not a set or list-like

        See Also
        --------
        pandas.Series.isin : Equivalent method on Series.

        Examples
        --------
        >>> s = pd.Categorical(["llama", "cow", "llama", "beetle", "llama", "hippo"])
        >>> s.isin(["cow", "llama"])
        array([ True,  True,  True, False,  True, False])

        Passing a single string as ``s.isin('llama')`` will raise an error. Use
        a list of one element instead:

        >>> s.isin(["llama"])
        array([ True, False,  True, False,  True, False])
        """
        # 将传入的 values 转换为布尔值的 NumPy 数组，标记其中的空值
        null_mask = np.asarray(isna(values))
        # 获取 values 在分类变量中的索引
        code_values = self.categories.get_indexer_for(values)
        # 过滤空值或有效索引值，以获取最终用于匹配的值列表
        code_values = code_values[null_mask | (code_values >= 0)]
        # 使用算法库中的 isin 方法，返回匹配结果的布尔数组
        return algorithms.isin(self.codes, code_values)

    # ------------------------------------------------------------------------
    # String methods interface
    def _str_map(
        self, f, na_value=np.nan, dtype=np.dtype("object"), convert: bool = True
    ):
        """
        Optimization to apply the callable `f` to the categories once
        and rebuild the result by `take`ing from the result with the codes.
        Returns the same type as the object-dtype implementation though.
        """
        # 导入必要的 NumpyExtensionArray 类
        from pandas.core.arrays import NumpyExtensionArray

        # 获取分类变量的 categories 和 codes
        categories = self.categories
        codes = self.codes
        # 使用 NumpyExtensionArray 对象的 _str_map 方法，应用函数 f 到 categories，然后通过 codes 重新构建结果
        result = NumpyExtensionArray(categories.to_numpy())._str_map(f, na_value, dtype)
        # 使用 take_nd 函数根据 codes 取值，并填充空值为 na_value，返回结果
        return take_nd(result, codes, fill_value=na_value)

    def _str_get_dummies(self, sep: str = "|"):
        """
        Generates dummy variables for each unique category in the Categorical,
        with separator `sep`.

        Parameters
        ----------
        sep : str, default '|'
            Separator to use in the dummy variable column names.

        Returns
        -------
        DataFrame
            A DataFrame containing dummy variable columns.

        Notes
        -----
        `sep` should not be a category in the Categorical.
        """
        # 导入必要的 NumpyExtensionArray 类
        from pandas.core.arrays import NumpyExtensionArray

        # 使用 NumpyExtensionArray 对象的 _str_get_dummies 方法，生成分类变量中每个唯一类别的虚拟变量
        return NumpyExtensionArray(self.astype(str))._str_get_dummies(sep)

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
        ):
        # 导入必要的模块和类
        from pandas.core.groupby.ops import WrappedCythonOp

        # 根据指定的聚合方式获取操作类型
        kind = WrappedCythonOp.get_kind_from_how(how)
        # 创建 WrappedCythonOp 实例
        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)

        # 获取当前对象的数据类型
        dtype = self.dtype
        # 如果聚合方式为 ["sum", "prod", "cumsum", "cumprod", "skew"] 中的一种，则抛出类型错误异常
        if how in ["sum", "prod", "cumsum", "cumprod", "skew"]:
            raise TypeError(f"{dtype} type does not support {how} operations")
        # 如果聚合方式为 ["min", "max", "rank", "idxmin", "idxmax"] 中的一种并且数据类型未排序，则抛出类型错误异常
        if how in ["min", "max", "rank", "idxmin", "idxmax"] and not dtype.ordered:
            # 抛出类型错误异常，以确保不会继续进行分组路径，因为空组情况下会失败而不会抛出异常
            raise TypeError(f"Cannot perform {how} with non-ordered Categorical")
        # 如果聚合方式不在支持的列表中，则根据操作类型抛出类型错误异常
        if how not in [
            "rank",
            "any",
            "all",
            "first",
            "last",
            "min",
            "max",
            "idxmin",
            "idxmax",
        ]:
            if kind == "transform":
                raise TypeError(f"{dtype} type does not support {how} operations")
            raise TypeError(f"{dtype} dtype does not support aggregation '{how}'")

        # 初始化结果掩码为 None
        result_mask = None
        # 获取当前对象的缺失值掩码
        mask = self.isna()
        # 如果聚合方式为 "rank"
        if how == "rank":
            # 确保数据类型已排序，这在之前已经检查过
            assert self.ordered  # checked earlier
            # 使用内部 ndarray 作为 npvalues
            npvalues = self._ndarray
        # 如果聚合方式为 ["first", "last", "min", "max", "idxmin", "idxmax"] 中的一种
        elif how in ["first", "last", "min", "max", "idxmin", "idxmax"]:
            # 使用内部 ndarray 作为 npvalues
            npvalues = self._ndarray
            # 初始化结果掩码为布尔型全零数组
            result_mask = np.zeros(ngroups, dtype=bool)
        else:
            # 对于聚合方式为 "any" 或 "all"
            # 将当前对象转换为布尔型数据
            npvalues = self.astype(bool)

        # 调用 _cython_op_ndim_compat 方法进行操作
        res_values = op._cython_op_ndim_compat(
            npvalues,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=ids,
            mask=mask,
            result_mask=result_mask,
            **kwargs,
        )

        # 如果聚合方式在 op.cast_blocklist 中，则返回结果值
        if how in op.cast_blocklist:
            return res_values
        # 如果聚合方式为 ["first", "last", "min", "max"] 中的一种
        elif how in ["first", "last", "min", "max"]:
            # 将结果值中对应结果掩码为 1 的位置设置为 -1
            res_values[result_mask == 1] = -1
        # 返回使用 res_values 创建的新对象
        return self._from_backing_data(res_values)
# 定义一个代理类，用于访问 Series 值的分类属性
@delegate_names(
    delegate=Categorical, accessors=["categories", "ordered"], typ="property"
)
# 添加代理方法名称，用于操作分类数据，如重命名、重新排序、添加、删除、设置、转换有序等
@delegate_names(
    delegate=Categorical,
    accessors=[
        "rename_categories",
        "reorder_categories",
        "add_categories",
        "remove_categories",
        "remove_unused_categories",
        "set_categories",
        "as_ordered",
        "as_unordered",
    ],
    typ="method",
)
class CategoricalAccessor(PandasDelegate, PandasObject, NoNewAttributesMixin):
    """
    表示 Series 值的分类属性访问器对象。

    Parameters
    ----------
    data : Series or CategoricalIndex
        被附加分类访问器的对象。

    See Also
    --------
    Series.dt : Series 值的日期时间属性访问器对象。
    Series.sparse : 稀疏矩阵数据类型访问器。

    Examples
    --------
    >>> s = pd.Series(list("abbccc")).astype("category")
    >>> s
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): ['a', 'b', 'c']

    >>> s.cat.categories
    Index(['a', 'b', 'c'], dtype='object')

    >>> s.cat.rename_categories(list("cba"))
    0    c
    1    b
    2    b
    3    a
    4    a
    5    a
    dtype: category
    Categories (3, object): ['c', 'b', 'a']

    >>> s.cat.reorder_categories(list("cba"))
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): ['c', 'b', 'a']

    >>> s.cat.add_categories(["d", "e"])
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (5, object): ['a', 'b', 'c', 'd', 'e']

    >>> s.cat.remove_categories(["a", "c"])
    0    NaN
    1      b
    2      b
    3    NaN
    4    NaN
    5    NaN
    dtype: category
    Categories (1, object): ['b']

    >>> s1 = s.cat.add_categories(["d", "e"])
    >>> s1.cat.remove_unused_categories()
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): ['a', 'b', 'c']

    >>> s.cat.set_categories(list("abcde"))
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (5, object): ['a', 'b', 'c', 'd', 'e']

    >>> s.cat.as_ordered()
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): ['a' < 'b' < 'c']

    >>> s.cat.as_unordered()
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): ['a', 'b', 'c']
    """

    def __init__(self, data) -> None:
        # 验证输入数据的有效性
        self._validate(data)
        # 获取父对象的值、索引和名称
        self._parent = data.values
        self._index = data.index
        self._name = data.name
        # 冻结对象，防止新增属性
        self._freeze()

    @staticmethod
    # 定义一个私有方法 _validate，用于验证数据是否为分类数据类型
    def _validate(data) -> None:
        # 如果数据的 dtype 不是 CategoricalDtype 类型，则抛出 AttributeError 异常
        if not isinstance(data.dtype, CategoricalDtype):
            raise AttributeError("Can only use .cat accessor with a 'category' dtype")

    # 定义一个私有方法 _delegate_property_get，用于获取属性值并委托给父对象
    def _delegate_property_get(self, name: str):
        # 返回 self._parent 对象的 name 属性值
        return getattr(self._parent, name)

    # 定义一个私有方法 _delegate_property_set，用于设置属性值并委托给父对象
    # 这里使用 type: ignore[override] 来忽略类型检查错误
    def _delegate_property_set(self, name: str, new_values) -> None:
        # 设置 self._parent 对象的 name 属性为 new_values
        setattr(self._parent, name, new_values)

    # 定义一个属性方法 codes，返回一个 Series 对象
    @property
    def codes(self) -> Series:
        """
        返回包含代码和索引的 Series 对象。

        参见
        --------
        Series.cat.categories : 返回该分类的所有类别。
        Series.cat.as_ordered : 设置分类为有序。
        Series.cat.as_unordered : 设置分类为无序。

        示例
        --------
        >>> raw_cate = pd.Categorical(["a", "b", "c", "a"], categories=["a", "b"])
        >>> ser = pd.Series(raw_cate)
        >>> ser.cat.codes
        0   0
        1   1
        2  -1
        3   0
        dtype: int8
        """
        from pandas import Series

        # 返回一个 Series 对象，包含 self._parent.codes 的数据，以及当前对象的索引
        return Series(self._parent.codes, index=self._index)

    # 定义一个私有方法 _delegate_method，用于委托调用父对象的方法
    def _delegate_method(self, name: str, *args, **kwargs):
        from pandas import Series

        # 获取 self._parent 对象的 name 方法
        method = getattr(self._parent, name)
        # 调用该方法并返回结果
        res = method(*args, **kwargs)
        # 如果结果不为 None，则返回一个 Series 对象，包含 res 的数据，以及当前对象的索引和名称
        if res is not None:
            return Series(res, index=self._index, name=self._name)
# utility routines


def _get_codes_for_values(
    values: Index | Series | ExtensionArray | np.ndarray,
    categories: Index,
) -> np.ndarray:
    """
    utility routine to turn values into codes given the specified categories

    If `values` is known to be a Categorical, use recode_for_categories instead.
    """
    # 获取给定值在指定类别中的索引，返回索引数组
    codes = categories.get_indexer_for(values)
    return coerce_indexer_dtype(codes, categories)


def recode_for_categories(
    codes: np.ndarray, old_categories, new_categories, copy: bool = True
) -> np.ndarray:
    """
    Convert a set of codes for to a new set of categories

    Parameters
    ----------
    codes : np.ndarray
    old_categories, new_categories : Index
    copy: bool, default True
        Whether to copy if the codes are unchanged.

    Returns
    -------
    new_codes : np.ndarray[np.int64]

    Examples
    --------
    >>> old_cat = pd.Index(["b", "a", "c"])
    >>> new_cat = pd.Index(["a", "b"])
    >>> codes = np.array([0, 1, 1, 2])
    >>> recode_for_categories(codes, old_cat, new_cat)
    array([ 1,  0,  0, -1], dtype=int8)
    """
    if len(old_categories) == 0:
        # All null anyway, so just retain the nulls
        if copy:
            return codes.copy()
        return codes
    elif new_categories.equals(old_categories):
        # Same categories, so no need to actually recode
        if copy:
            return codes.copy()
        return codes

    # 获取新类别在旧类别中的索引，生成一个转换索引数组
    indexer = coerce_indexer_dtype(
        new_categories.get_indexer_for(old_categories), new_categories
    )
    # 使用转换索引数组将旧的代码转换为新的代码
    new_codes = take_nd(indexer, codes, fill_value=-1)
    return new_codes


def factorize_from_iterable(values) -> tuple[np.ndarray, Index]:
    """
    Factorize an input `values` into `categories` and `codes`. Preserves
    categorical dtype in `categories`.

    Parameters
    ----------
    values : list-like

    Returns
    -------
    codes : ndarray
    categories : Index
        If `values` has a categorical dtype, then `categories` is
        a CategoricalIndex keeping the categories and order of `values`.
    """
    from pandas import CategoricalIndex

    if not is_list_like(values):
        raise TypeError("Input must be list-like")

    categories: Index

    vdtype = getattr(values, "dtype", None)
    if isinstance(vdtype, CategoricalDtype):
        values = extract_array(values)
        # 构建一个与 values 相同类别但代码从 [0, ..., len(n_categories) - 1] 的分类
        cat_codes = np.arange(len(values.categories), dtype=values.codes.dtype)
        cat = Categorical.from_codes(cat_codes, dtype=values.dtype, validate=False)

        # 使用构建的分类创建分类索引
        categories = CategoricalIndex(cat)
        codes = values.codes
    else:
        # 如果不是字典类型，创建一个分类变量对象 Categorical
        # 参数 ordered 设为 False，因为我们只关心类别的结果，不关心顺序
        # 这是为了解决 GH（GitHub）上的问题 #15457
        cat = Categorical(values, ordered=False)
        # 获取分类变量对象的类别
        categories = cat.categories
        # 获取分类变量对象的编码（对应类别的索引）
        codes = cat.codes
    # 返回分类变量的编码和类别
    return codes, categories
# 从多个可迭代对象中因子化元素，并返回因子化后的结果
def factorize_from_iterables(iterables) -> tuple[list[np.ndarray], list[Index]]:
    """
    A higher-level wrapper over `factorize_from_iterable`.

    Parameters
    ----------
    iterables : list-like of list-likes
        由多个可迭代对象组成的列表

    Returns
    -------
    codes : list of ndarrays
        包含因子化结果的列表，每个元素是一个 ndarray
    categories : list of Indexes
        包含因子化后的索引对象的列表

    Notes
    -----
    See `factorize_from_iterable` for more info.
    """
    # 如果输入的可迭代对象为空，则返回两个空列表以保持一致性
    if len(iterables) == 0:
        return [], []

    # 使用生成器表达式对每个可迭代对象进行因子化，并将结果分别存储在 codes 和 categories 中
    codes, categories = zip(*(factorize_from_iterable(it) for it in iterables))
    # 将元组转换为列表，并返回结果
    return list(codes), list(categories)
```