# `D:\src\scipysrc\pandas\pandas\core\util\hashing.py`

```
"""
data hash pandas / numpy objects
"""

# 引入未来的注释语法支持
from __future__ import annotations

# 引入模块
import itertools
from typing import TYPE_CHECKING

# 引入 NumPy 库
import numpy as np

# 导入 pandas 库中的 hash_object_array 函数
from pandas._libs.hashing import hash_object_array

# 导入 pandas 库中的类型检查和通用数据类型定义
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCExtensionArray,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)

# 如果是类型检查，则引入特定类型
if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Iterator,
    )

    from pandas._typing import (
        ArrayLike,
        npt,
    )

    from pandas import (
        DataFrame,
        Index,
        MultiIndex,
        Series,
    )


# 16 字节长的哈希键
_default_hash_key = "0123456789123456"


def combine_hash_arrays(
    arrays: Iterator[np.ndarray], num_items: int
) -> npt.NDArray[np.uint64]:
    """
    Parameters
    ----------
    arrays : Iterator[np.ndarray]
        迭代器，包含要合并哈希的 NumPy 数组
    num_items : int
        数组中的元素数量

    Returns
    -------
    np.ndarray[uint64]
        合并后的哈希数组，每个元素为 uint64 类型

    Should be the same as CPython's tupleobject.c
    """
    try:
        first = next(arrays)
    except StopIteration:
        return np.array([], dtype=np.uint64)

    arrays = itertools.chain([first], arrays)

    mult = np.uint64(1000003)
    out = np.zeros_like(first) + np.uint64(0x345678)
    last_i = 0
    for i, a in enumerate(arrays):
        inverse_i = num_items - i
        out ^= a
        out *= mult
        mult += np.uint64(82520 + inverse_i + inverse_i)
        last_i = i
    assert last_i + 1 == num_items, "Fed in wrong num_items"
    out += np.uint64(97531)
    return out


def hash_pandas_object(
    obj: Index | DataFrame | Series,
    index: bool = True,
    encoding: str = "utf8",
    hash_key: str | None = _default_hash_key,
    categorize: bool = True,
) -> Series:
    """
    Return a data hash of the Index/Series/DataFrame.

    Parameters
    ----------
    obj : Index, Series, or DataFrame
        要进行哈希的对象，可以是索引、序列或数据框
    index : bool, default True
        是否包含索引在哈希中（对于 Series/DataFrame）。
    encoding : str, default 'utf8'
        当数据和键为字符串时的编码方式。
    hash_key : str, default _default_hash_key
        用于编码字符串键的哈希密钥。
    categorize : bool, default True
        是否在进行哈希之前对对象数组进行分类。当数组包含重复值时，这样更有效率。

    Returns
    -------
    Series of uint64
        与输入对象相同长度的 uint64 类型的哈希值序列。

    Examples
    --------
    >>> pd.util.hash_pandas_object(pd.Series([1, 2, 3]))
    0    14639053686158035780
    1     3869563279212530728
    2      393322362522515241
    dtype: uint64
    """
    from pandas import Series

    # 如果哈希键为 None，则使用默认的哈希键
    if hash_key is None:
        hash_key = _default_hash_key

    # 如果对象是 ABCMultiIndex 类型，则返回元组的哈希序列
    if isinstance(obj, ABCMultiIndex):
        return Series(hash_tuples(obj, encoding, hash_key), dtype="uint64", copy=False)
    elif isinstance(obj, ABCIndex):
        # 如果 obj 是 ABCIndex 的实例，则执行以下操作
        # 对 obj 的 _values 属性进行哈希处理，生成 uint64 类型的数组 h
        h = hash_array(obj._values, encoding, hash_key, categorize).astype(
            "uint64", copy=False
        )
        # 创建一个 Series 对象 ser，使用 h 作为数据，obj 作为索引，数据类型为 uint64
        ser = Series(h, index=obj, dtype="uint64", copy=False)

    elif isinstance(obj, ABCSeries):
        # 如果 obj 是 ABCSeries 的实例，则执行以下操作
        # 对 obj 的 _values 属性进行哈希处理，生成 uint64 类型的数组 h
        h = hash_array(obj._values, encoding, hash_key, categorize).astype(
            "uint64", copy=False
        )
        # 如果 index 参数为真，则对 obj 的索引进行哈希处理，并将结果与 h 合并
        if index:
            # 生成一个迭代器，对 obj 的索引进行哈希处理
            index_iter = (
                hash_pandas_object(
                    obj.index,
                    index=False,
                    encoding=encoding,
                    hash_key=hash_key,
                    categorize=categorize,
                )._values
                for _ in [None]
            )
            # 将 h 和 index_iter 合并成数组，并进行二次哈希处理
            arrays = itertools.chain([h], index_iter)
            h = combine_hash_arrays(arrays, 2)

        # 创建一个 Series 对象 ser，使用 h 作为数据，obj 的索引作为索引，数据类型为 uint64
        ser = Series(h, index=obj.index, dtype="uint64", copy=False)

    elif isinstance(obj, ABCDataFrame):
        # 如果 obj 是 ABCDataFrame 的实例，则执行以下操作
        # 生成一个生成器 hashes，对 obj 的每列进行哈希处理
        hashes = (
            hash_array(series._values, encoding, hash_key, categorize)
            for _, series in obj.items()
        )
        # 记录 obj 的列数
        num_items = len(obj.columns)
        # 如果 index 参数为真，则对 obj 的索引进行哈希处理，并将结果与 hashes 合并
        if index:
            # 生成一个迭代器，对 obj 的索引进行哈希处理
            index_hash_generator = (
                hash_pandas_object(
                    obj.index,
                    index=False,
                    encoding=encoding,
                    hash_key=hash_key,
                    categorize=categorize,
                )._values
                for _ in [None]
            )
            # 增加哈希处理的项目数
            num_items += 1

            # 将 hashes 和 index_hash_generator 合并成生成器 _hashes
            _hashes = itertools.chain(hashes, index_hash_generator)
            hashes = (x for x in _hashes)

        # 将 hashes 中的哈希数组和 num_items 合并成一个数组，并进行哈希处理
        h = combine_hash_arrays(hashes, num_items)

        # 创建一个 Series 对象 ser，使用 h 作为数据，obj 的索引作为索引，数据类型为 uint64
        ser = Series(h, index=obj.index, dtype="uint64", copy=False)
    else:
        # 如果 obj 类型不是 ABCIndex、ABCSeries 或 ABCDataFrame 中的任何一个，则抛出类型错误
        raise TypeError(f"Unexpected type for hashing {type(obj)}")

    # 返回生成的 Series 对象 ser
    return ser
def hash_tuples(
    vals: MultiIndex | Iterable[tuple[Hashable, ...]],  # 参数vals可以是MultiIndex或元组的可迭代对象，元组中的元素必须是可哈希的
    encoding: str = "utf8",  # 编码方式，默认为'utf8'
    hash_key: str = _default_hash_key,  # 哈希键值，默认为默认哈希键
) -> npt.NDArray[np.uint64]:  # 返回类型为NumPy数组，其中元素类型为uint64

    """
    Hash an MultiIndex / listlike-of-tuples efficiently.

    Parameters
    ----------
    vals : MultiIndex or listlike-of-tuples  # 输入参数可以是MultiIndex或者元组的列表
    encoding : str, default 'utf8'  # 编码方式，默认为'utf8'
    hash_key : str, default _default_hash_key  # 哈希键值，默认为默认哈希键

    Returns
    -------
    ndarray[np.uint64] of hashed values  # 返回哈希值的NumPy数组，元素类型为uint64
    """

    if not is_list_like(vals):  # 如果vals不是列表形式
        raise TypeError("must be convertible to a list-of-tuples")  # 抛出类型错误异常，要求可以转换为元组列表

    from pandas import (  # 导入pandas模块中的Categorical和MultiIndex类
        Categorical,
        MultiIndex,
    )

    if not isinstance(vals, ABCMultiIndex):  # 如果vals不是ABCMultiIndex的实例
        mi = MultiIndex.from_tuples(vals)  # 将vals转换为MultiIndex对象
    else:
        mi = vals  # 否则直接使用vals作为MultiIndex对象

    # create a list-of-Categoricals
    cat_vals = [  # 创建一个Categorical对象的列表
        Categorical._simple_new(  # 使用Categorical类的_simple_new方法创建对象
            mi.codes[level],  # 传入当前级别的编码数据
            CategoricalDtype(categories=mi.levels[level], ordered=False),  # 设置类别和有序性参数
        )
        for level in range(mi.nlevels)  # 对于MultiIndex对象的每一个级别
    ]

    # hash the list-of-ndarrays
    hashes = (  # 哈希列表中的ndarray对象
        cat._hash_pandas_object(encoding=encoding, hash_key=hash_key, categorize=False)  # 对Categorical对象进行哈希计算
        for cat in cat_vals  # 对于cat_vals中的每个Categorical对象
    )
    h = combine_hash_arrays(hashes, len(cat_vals))  # 将哈希数组组合成一个数组

    return h  # 返回哈希值数组


def hash_array(
    vals: ArrayLike,  # 参数vals可以是类数组对象
    encoding: str = "utf8",  # 编码方式，默认为'utf8'
    hash_key: str = _default_hash_key,  # 哈希键值，默认为默认哈希键
    categorize: bool = True,  # 是否对对象数组进行分类，默认为True
) -> npt.NDArray[np.uint64]:  # 返回类型为NumPy数组，其中元素类型为uint64

    """
    Given a 1d array, return an array of deterministic integers.

    Parameters
    ----------
    vals : ndarray or ExtensionArray  # 输入参数可以是ndarray或ExtensionArray对象
        The input array to hash.
    encoding : str, default 'utf8'  # 编码方式，默认为'utf8'
        Encoding for data & key when strings.
    hash_key : str, default _default_hash_key  # 哈希键值，默认为默认哈希键
        Hash_key for string key to encode.
    categorize : bool, default True  # 是否先对对象数组进行分类，默认为True
        Whether to first categorize object arrays before hashing.

    Returns
    -------
    ndarray[np.uint64, ndim=1]  # 返回哈希值的NumPy数组，维度为1
        Hashed values, same length as the vals.

    See Also
    --------
    util.hash_pandas_object : Return a data hash of the Index/Series/DataFrame.
    util.hash_tuples : Hash an MultiIndex / listlike-of-tuples efficiently.

    Examples
    --------
    >>> pd.util.hash_array(np.array([1, 2, 3]))
    array([ 6238072747940578789, 15839785061582574730,  2185194620014831856],
      dtype=uint64)
    """

    if not hasattr(vals, "dtype"):  # 如果vals没有dtype属性
        raise TypeError("must pass a ndarray-like")  # 抛出类型错误异常，要求传入类似ndarray的对象

    if isinstance(vals, ABCExtensionArray):  # 如果vals是ABCExtensionArray的实例
        return vals._hash_pandas_object(  # 调用ExtensionArray的_hash_pandas_object方法进行哈希计算
            encoding=encoding, hash_key=hash_key, categorize=categorize
        )

    if not isinstance(vals, np.ndarray):  # 如果vals不是np.ndarray的实例
        # GH#42003
        raise TypeError(
            "hash_array requires np.ndarray or ExtensionArray, not "
            f"{type(vals).__name__}. Use hash_pandas_object instead."
        )  # 抛出类型错误异常，提示应使用hash_pandas_object函数而不是hash_array

    return _hash_ndarray(vals, encoding, hash_key, categorize)


def _hash_ndarray(
    vals: np.ndarray,
    # 设置一个字符串类型的变量，用于指定编码方式为UTF-8
    encoding: str = "utf8",
    # 设置一个字符串类型的变量，用于指定哈希键，默认使用预定义的默认哈希键
    hash_key: str = _default_hash_key,
    # 设置一个布尔类型的变量，用于指定是否进行分类，默认为True，即进行分类
    categorize: bool = True,
# 定义函数签名，指定函数返回类型为 numpy.uint64 的 NumPy 数组
def hash_array(vals: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """
    使用 hash_array.__doc__ 中的说明。
    """
    # 获取数组的数据类型
    dtype = vals.dtype

    # 如果数据类型是复数且为 128 位，则按部分处理
    if np.issubdtype(dtype, np.complex128):
        # 分别对实部和虚部进行哈希处理，然后返回它们的加权和
        hash_real = _hash_ndarray(vals.real, encoding, hash_key, categorize)
        hash_imag = _hash_ndarray(vals.imag, encoding, hash_key, categorize)
        return hash_real + 23 * hash_imag

    # 首先，尝试将数组转换为无符号 64 位整数类型
    if dtype == bool:
        vals = vals.astype("u8")
    elif issubclass(dtype.type, (np.datetime64, np.timedelta64)):
        vals = vals.view("i8").astype("u8", copy=False)
    elif issubclass(dtype.type, np.number) and dtype.itemsize <= 8:
        vals = vals.view(f"u{vals.dtype.itemsize}").astype("u8")
    else:
        # 当数组包含重复值时，通过分类对象数据类型可以显著提高性能，
        # 然后进行哈希处理并重命名分类。在已知或可能是唯一值时，允许跳过分类过程。
        if categorize:
            from pandas import (
                Categorical,
                Index,
                factorize,
            )

            # 使用 pandas 的 factorize 函数对值进行分类编码
            codes, categories = factorize(vals, sort=False)
            dtype = CategoricalDtype(categories=Index(categories), ordered=False)
            cat = Categorical._simple_new(codes, dtype)
            # 使用 pandas 的哈希函数处理分类数据对象
            return cat._hash_pandas_object(
                encoding=encoding, hash_key=hash_key, categorize=False
            )

        try:
            # 尝试对对象数组进行哈希处理
            vals = hash_object_array(vals, hash_key, encoding)
        except TypeError:
            # 如果数组包含混合类型，则先转换为字符串再进行哈希处理
            vals = hash_object_array(
                vals.astype(str).astype(object), hash_key, encoding
            )

    # 对 64 位整数进行位移和混合操作，以增加哈希的随机性
    vals ^= vals >> 30
    vals *= np.uint64(0xBF58476D1CE4E5B9)
    vals ^= vals >> 27
    vals *= np.uint64(0x94D049BB133111EB)
    vals ^= vals >> 31
    # 返回最终的哈希值
    return vals
```