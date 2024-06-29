# `D:\src\scipysrc\pandas\pandas\tests\dtypes\test_inference.py`

```
# 引入 collections 模块，用于处理集合数据类型
import collections
# 从 collections 模块中引入 namedtuple 类型
from collections import namedtuple
# 从 collections.abc 模块中引入 Iterator 抽象基类
from collections.abc import Iterator
# 从 datetime 模块中分别引入 date、datetime、time、timedelta、timezone 类型
from datetime import (
    date,
    datetime,
    time,
    timedelta,
    timezone,
)
# 从 decimal 模块中引入 Decimal 类型
from decimal import Decimal
# 从 fractions 模块中引入 Fraction 类型
from fractions import Fraction
# 从 io 模块中引入 StringIO 类型
from io import StringIO
# 引入 itertools 模块，用于高效循环和组合操作
import itertools
# 从 numbers 模块中引入 Number 抽象基类
from numbers import Number
# 引入 re 模块，用于正则表达式操作
import re
# 引入 sys 模块，用于系统相关操作
import sys
# 从 typing 模块中引入 Generic、TypeVar 类型
from typing import (
    Generic,
    TypeVar,
)

# 引入 numpy 库，并重命名为 np
import numpy as np
# 引入 pytest 测试框架
import pytest

# 从 pandas._libs 模块中引入 lib、libmissing、libops 别名
from pandas._libs import (
    lib,
    missing as libmissing,
    ops as libops,
)

# 从 pandas.compat.numpy 模块中引入 np_version_gt2 函数
from pandas.compat.numpy import np_version_gt2

# 从 pandas.core.dtypes 模块中引入 inference 类型
from pandas.core.dtypes import inference
# 从 pandas.core.dtypes.cast 模块中引入 find_result_type 函数
from pandas.core.dtypes.cast import find_result_type
# 从 pandas.core.dtypes.common 模块中引入多个函数和变量，如 ensure_int32、is_bool 等
from pandas.core.dtypes.common import (
    ensure_int32,
    is_bool,
    is_complex,
    is_datetime64_any_dtype,
    is_datetime64_dtype,
    is_datetime64_ns_dtype,
    is_datetime64tz_dtype,
    is_float,
    is_integer,
    is_number,
    is_scalar,
    is_scipy_sparse,
    is_timedelta64_dtype,
    is_timedelta64_ns_dtype,
)

# 引入 pandas 库，并重命名为 pd
import pandas as pd
# 从 pandas 模块中引入多个类型，如 Categorical、DataFrame、Series 等
from pandas import (
    Categorical,
    DataFrame,
    DateOffset,
    DatetimeIndex,
    Index,
    Interval,
    Period,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
)
# 引入 pandas._testing 模块，并重命名为 tm
import pandas._testing as tm
# 从 pandas.core.arrays 模块中引入多个数组类型，如 BooleanArray、FloatingArray 等
from pandas.core.arrays import (
    BooleanArray,
    FloatingArray,
    IntegerArray,
)

# 定义 pytest 的 fixture，用于参数化测试
@pytest.fixture(params=[True, False], ids=str)
def coerce(request):
    return request.param


# 定义一个模拟的类，类似于 numpy 数组，但不是真正的 numpy 数组
class MockNumpyLikeArray:
    """
    A class which is numpy-like (e.g. Pint's Quantity) but not actually numpy

    The key is that it is not actually a numpy array so
    ``util.is_array(mock_numpy_like_array_instance)`` returns ``False``. Other
    important properties are that the class defines a :meth:`__iter__` method
    (so that ``isinstance(abc.Iterable)`` returns ``True``) and has a
    :meth:`ndim` property, as pandas special-cases 0-dimensional arrays in some
    cases.

    We expect pandas to behave with respect to such duck arrays exactly as
    with real numpy arrays. In particular, a 0-dimensional duck array is *NOT*
    a scalar (`is_scalar(np.array(1)) == False`), but it is not list-like either.
    """

    def __init__(self, values) -> None:
        self._values = values

    # 定义迭代器方法，使得该类实例可以迭代
    def __iter__(self) -> Iterator:
        iter_values = iter(self._values)

        def it_outer():
            yield from iter_values

        return it_outer()

    # 定义长度方法，返回存储值的长度
    def __len__(self) -> int:
        return len(self._values)

    # 定义转为数组的方法，使用 numpy 的 asarray 将值转为数组
    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._values, dtype=dtype)

    # 定义属性，返回值数组的维度
    @property
    def ndim(self):
        return self._values.ndim

    # 定义属性，返回值数组的数据类型
    @property
    def dtype(self):
        return self._values.dtype

    # 定义属性，返回值数组的大小
    @property
    def size(self):
        return self._values.size

    # 定义属性，返回值数组的形状
    @property
    def shape(self):
        return self._values.shape
# 定义包含所有待测试对象的列表，每个元素是一个包含对象、期望值和标识的元组
ll_params = [
    ([1], True, "list"),                        # 包含一个元素的列表，预期为可迭代对象
    ([], True, "list-empty"),                   # 空列表，预期为可迭代对象
    ((1,), True, "tuple"),                      # 包含一个元素的元组，预期为可迭代对象
    ((), True, "tuple-empty"),                  # 空元组，预期为可迭代对象
    ({"a": 1}, True, "dict"),                   # 字典，预期为可迭代对象
    ({}, True, "dict-empty"),                   # 空字典，预期为可迭代对象
    ({"a", 1}, "set", "set"),                   # 集合，特殊处理为 set 类型，预期为非可迭代对象
    (set(), "set", "set-empty"),                # 空集合，特殊处理为 set 类型，预期为非可迭代对象
    (frozenset({"a", 1}), "set", "frozenset"),  # 不可变集合，特殊处理为 set 类型，预期为非可迭代对象
    (frozenset(), "set", "frozenset-empty"),    # 空不可变集合，特殊处理为 set 类型，预期为非可迭代对象
    (iter([1, 2]), True, "iterator"),           # 迭代器，预期为可迭代对象
    (iter([]), True, "iterator-empty"),         # 空迭代器，预期为可迭代对象
    ((x for x in [1, 2]), True, "generator"),   # 生成器，预期为可迭代对象
    ((_ for _ in []), True, "generator-empty"), # 空生成器，预期为可迭代对象
    (Series([1]), True, "Series"),              # Pandas Series 对象，预期为可迭代对象
    (Series([], dtype=object), True, "Series-empty"),  # 空的 Pandas Series 对象，预期为可迭代对象
    (Series(["a"]).str, True, "StringMethods"), # Pandas Series 的字符串方法，预期为可迭代对象
    (Series([], dtype="O").str, True, "StringMethods-empty"),  # 空的 Pandas Series 的字符串方法，预期为可迭代对象
    (Index([1]), True, "Index"),                # Pandas Index 对象，预期为可迭代对象
    (Index([]), True, "Index-empty"),           # 空的 Pandas Index 对象，预期为可迭代对象
    (DataFrame([[1]]), True, "DataFrame"),      # Pandas DataFrame 对象，预期为可迭代对象
    (DataFrame(), True, "DataFrame-empty"),     # 空的 Pandas DataFrame 对象，预期为可迭代对象
    (np.ndarray((2,) * 1), True, "ndarray-1d"), # NumPy 1 维数组，预期为可迭代对象
    (np.array([]), True, "ndarray-1d-empty"),   # 空的 NumPy 1 维数组，预期为可迭代对象
    (np.ndarray((2,) * 2), True, "ndarray-2d"), # NumPy 2 维数组，预期为可迭代对象
    (np.array([[]]), True, "ndarray-2d-empty"), # 空的 NumPy 2 维数组，预期为可迭代对象
    (np.ndarray((2,) * 3), True, "ndarray-3d"), # NumPy 3 维数组，预期为可迭代对象
    (np.array([[[]]]), True, "ndarray-3d-empty"), # 空的 NumPy 3 维数组，预期为可迭代对象
    (np.ndarray((2,) * 4), True, "ndarray-4d"), # NumPy 4 维数组，预期为可迭代对象
    (np.array([[[[]]]]), True, "ndarray-4d-empty"), # 空的 NumPy 4 维数组，预期为可迭代对象
    (np.array(2), False, "ndarray-0d"),         # NumPy 0 维数组，预期为非可迭代对象
    (MockNumpyLikeArray(np.ndarray((2,) * 1)), True, "duck-ndarray-1d"),  # 模拟 NumPy 1 维数组的对象，预期为可迭代对象
    (MockNumpyLikeArray(np.array([])), True, "duck-ndarray-1d-empty"),   # 模拟空的 NumPy 1 维数组的对象，预期为可迭代对象
    (MockNumpyLikeArray(np.ndarray((2,) * 2)), True, "duck-ndarray-2d"),  # 模拟 NumPy 2 维数组的对象，预期为可迭代对象
    (MockNumpyLikeArray(np.array([[]])), True, "duck-ndarray-2d-empty"), # 模拟空的 NumPy 2 维数组的对象，预期为可迭代对象
    (MockNumpyLikeArray(np.ndarray((2,) * 3)), True, "duck-ndarray-3d"),  # 模拟 NumPy 3 维数组的对象，预期为可迭代对象
    (MockNumpyLikeArray(np.array([[[]]])), True, "duck-ndarray-3d-empty"), # 模拟空的 NumPy 3 维数组的对象，预期为可迭代对象
    (MockNumpyLikeArray(np.ndarray((2,) * 4)), True, "duck-ndarray-4d"),  # 模拟 NumPy 4 维数组的对象，预期为可迭代对象
    (MockNumpyLikeArray(np.array([[[[]]]])), True, "duck-ndarray-4d-empty"),  # 模拟空的 NumPy 4 维数组的对象，预期为可迭代对象
    (MockNumpyLikeArray(np.array(2)), False, "duck-ndarray-0d"),  # 模拟 NumPy 0 维数组的对象，预期为非可迭代对象
    (1, False, "int"),                          # 整数，预期为非可迭代对象
    (b"123", False, "bytes"),                   # 字节串，预期为非可迭代对象
    (b"", False, "bytes-empty"),                # 空字节串，预期为非可迭代对象
    ("123", False, "string"),                   # 字符串，预期为非可迭代对象
    ("", False, "string-empty"),                # 空字符串，预期为非可迭代对象
    (str, False, "string-type"),                # 字符串类型，预期为非可迭代对象
    (object(), False, "object"),                # 对象实例，预期为非可迭代对象
    (np.nan, False, "NaN"),                     # NaN 值，预期为非可迭代对象
    (None, False, "None"),                      # None 值，预期为非可迭代对象
]
objs, expected, ids = zip(*ll_params)

# 为测试准备数据的夹具，参数化使用 objs 和 expected，通过 ids 标识不同情况
@pytest.fixture(params=zip(objs, expected), ids=ids)
def maybe_list_like(request):
    return request.param

# 测试函数：检查是否对象可视为可迭代对象
def test_is_list_like(maybe_list_like):
    obj, expected = maybe_list_like
    # 如果期望值为 "set"，则预期结果为 True
    expected = True if expected == "set" else expected
    assert inference.is_list_like(obj) == expected

# 测试函数：检查是否对象可视为可迭代对象，不允许集合
def test_is_list_like_disallow_sets(maybe_list_like):
    obj, expected = maybe_list_like
    # 如果期望值为 "set"，则预期结果为 False
    expected = False if expected == "set" else expected
    assert inference.is_list_like(obj, allow_sets=False) == expected
def test_is_list_like_recursion():
    # GH 33721
    # interpreter would crash with SIGABRT
    # 定义递归函数 list_like，用于测试 is_list_like 函数在递归调用时的行为
    def list_like():
        # 调用 is_list_like 函数，传入空列表作为参数
        inference.is_list_like([])
        # 递归调用 list_like 函数自身
        list_like()

    # 获取当前系统的递归深度限制
    rec_limit = sys.getrecursionlimit()
    try:
        # 临时将递归深度限制设置为 100，以避免在 Windows CI 上发生堆栈溢出
        sys.setrecursionlimit(100)
        # 使用 external_error_raised 上下文管理器，捕获 RecursionError 异常
        with tm.external_error_raised(RecursionError):
            # 调用 list_like 函数，期望触发 RecursionError 异常
            list_like()
    finally:
        # 恢复系统的递归深度限制
        sys.setrecursionlimit(rec_limit)


def test_is_list_like_iter_is_none():
    # GH 43373
    # is_list_like was yielding false positives with __iter__ == None
    # 定义一个类 NotListLike，模拟一个不像列表的对象
    class NotListLike:
        def __getitem__(self, item):
            return self

        __iter__ = None

    # 断言 NotListLike 的实例不被视为列表样式
    assert not inference.is_list_like(NotListLike())


def test_is_list_like_generic():
    # GH 49649
    # is_list_like was yielding false positives for Generic classes in python 3.11
    # 定义一个泛型类 MyDataFrame，继承自 DataFrame
    T = TypeVar("T")

    class MyDataFrame(DataFrame, Generic[T]): ...

    # 创建 MyDataFrame 的实例 tstc 和 tst，分别传入不同的类型参数
    tstc = MyDataFrame[int]
    tst = MyDataFrame[int]({"x": [1, 2, 3]})

    # 断言对于 MyDataFrame[int] 类型，不被视为列表样式
    assert not inference.is_list_like(tstc)
    # 断言 tst 是 DataFrame 的实例
    assert isinstance(tst, DataFrame)
    # 断言 tst 被视为列表样式
    assert inference.is_list_like(tst)


def test_is_sequence():
    # 测试 is_sequence 函数的行为
    is_seq = inference.is_sequence
    # 断言元组和列表被视为序列
    assert is_seq((1, 2))
    assert is_seq([1, 2])
    # 断言字符串和 np.int64 不被视为序列
    assert not is_seq("abcd")
    assert not is_seq(np.int64)

    # 定义一个类 A，实现 __getitem__ 方法
    class A:
        def __getitem__(self, item):
            return 1

    # 断言类 A 的实例不被视为序列
    assert not is_seq(A())


def test_is_array_like():
    # 断言不同类型的对象是否被视为数组样式
    assert inference.is_array_like(Series([], dtype=object))
    assert inference.is_array_like(Series([1, 2]))
    assert inference.is_array_like(np.array(["a", "b"]))
    assert inference.is_array_like(Index(["2016-01-01"]))
    assert inference.is_array_like(np.array([2, 3]))
    assert inference.is_array_like(MockNumpyLikeArray(np.array([2, 3])))

    # 定义一个继承自 list 的类 DtypeList，并添加一个 dtype 属性
    class DtypeList(list):
        dtype = "special"

    # 断言 DtypeList 的实例被视为数组样式
    assert inference.is_array_like(DtypeList())

    # 断言普通列表、元组、字符串和整数不被视为数组样式
    assert not inference.is_array_like([1, 2, 3])
    assert not inference.is_array_like(())
    assert not inference.is_array_like("foo")
    assert not inference.is_array_like(123)


@pytest.mark.parametrize(
    "inner",
    [
        [],
        [1],
        (1,),
        (1, 2),
        {"a": 1},
        {1, "a"},
        Series([1]),
        Series([], dtype=object),
        Series(["a"]).str,
        (x for x in range(5)),
    ],
)
@pytest.mark.parametrize("outer", [list, Series, np.array, tuple])
def test_is_nested_list_like_passes(inner, outer):
    # 参数化测试，测试 is_list_like 是否正确识别嵌套列表样式
    result = outer([inner for _ in range(5)])
    assert inference.is_list_like(result)


@pytest.mark.parametrize(
    "obj",
    [
        "abc",
        [],
        [1],
        (1,),
        ["a"],
        "a",
        {"a"},
        [1, 2, 3],
        Series([1]),
        DataFrame({"A": [1]}),
        ([1, 2] for _ in range(5)),
    ],
)
def test_is_nested_list_like_fails(obj):
    # 参数化测试，测试 is_list_like 是否正确识别非嵌套列表样式
    assert not inference.is_nested_list_like(obj)
def test_is_dict_like_passes(ll):
    # 调用断言，验证参数 ll 是否被判断为字典式结构
    assert inference.is_dict_like(ll)


@pytest.mark.parametrize(
    "ll",
    [
        "1",        # 测试字符串是否字典式
        1,          # 测试整数是否字典式
        [1, 2],     # 测试列表是否字典式
        (1, 2),     # 测试元组是否字典式
        range(2),   # 测试 range 对象是否字典式
        Index([1]), # 测试 Index 对象是否字典式
        dict,       # 测试 dict 类型本身是否字典式
        collections.defaultdict,   # 测试 defaultdict 类型本身是否字典式
        Series,     # 测试 Series 类型本身是否字典式
    ],
)
def test_is_dict_like_fails(ll):
    # 调用断言，验证参数 ll 是否被判断为非字典式结构
    assert not inference.is_dict_like(ll)


@pytest.mark.parametrize("has_keys", [True, False])
@pytest.mark.parametrize("has_getitem", [True, False])
@pytest.mark.parametrize("has_contains", [True, False])
def test_is_dict_like_duck_type(has_keys, has_getitem, has_contains):
    class DictLike:
        def __init__(self, d) -> None:
            self.d = d

        if has_keys:
            # 如果有 keys 方法，则返回内部字典的 keys
            def keys(self):
                return self.d.keys()

        if has_getitem:
            # 如果有 __getitem__ 方法，则调用内部字典的 __getitem__ 方法
            def __getitem__(self, key):
                return self.d.__getitem__(key)

        if has_contains:
            # 如果有 __contains__ 方法，则调用内部字典的 __contains__ 方法
            def __contains__(self, key) -> bool:
                return self.d.__contains__(key)

    d = DictLike({1: 2})
    # 调用断言，验证 DictLike 实例是否被判断为字典式结构
    result = inference.is_dict_like(d)
    expected = has_keys and has_getitem and has_contains
    assert result is expected


def test_is_file_like():
    class MockFile:
        pass

    is_file = inference.is_file_like

    data = StringIO("data")
    # 断言 StringIO 实例被判断为文件式对象
    assert is_file(data)

    # 没有 read / write 属性
    m = MockFile()
    # 断言 MockFile 实例不被判断为文件式对象
    assert not is_file(m)

    MockFile.write = lambda self: 0

    # 有 write 属性但不是迭代器
    m = MockFile()
    # 断言 MockFile 实例不被判断为文件式对象
    assert not is_file(m)

    # gh-16530: 有效的迭代器只需具有 __iter__ 属性即可
    MockFile.__iter__ = lambda self: self

    # 有效的只写文件
    m = MockFile()
    # 断言 MockFile 实例被判断为文件式对象
    assert is_file(m)

    del MockFile.write
    MockFile.read = lambda self: 0

    # 有效的只读文件
    m = MockFile()
    # 断言 MockFile 实例被判断为文件式对象
    assert is_file(m)

    # 迭代器但没有 read / write 属性
    data = [1, 2, 3]
    # 断言列表不被判断为文件式对象
    assert not is_file(data)


test_tuple = collections.namedtuple("test_tuple", ["a", "b", "c"])


@pytest.mark.parametrize("ll", [test_tuple(1, 2, 3)])
def test_is_names_tuple_passes(ll):
    # 断言 namedtuple 实例被判断为命名元组
    assert inference.is_named_tuple(ll)


@pytest.mark.parametrize("ll", [(1, 2, 3), "a", Series({"pi": 3.14})])
def test_is_names_tuple_fails(ll):
    # 断言非命名元组对象不被判断为命名元组
    assert not inference.is_named_tuple(ll)


def test_is_hashable():
    # 所有新式类默认是可哈希的
    class HashableClass:
        pass

    class UnhashableClass1:
        __hash__ = None

    class UnhashableClass2:
        def __hash__(self):
            raise TypeError("Not hashable")

    hashable = (1, 3.14, np.float64(3.14), "a", (), (1,), HashableClass())
    not_hashable = ([], UnhashableClass1())
    abc_hashable_not_really_hashable = (([],), UnhashableClass2())

    for i in hashable:
        # 断言可哈希对象返回 True
        assert inference.is_hashable(i)
    for i in not_hashable:
        # 断言不可哈希对象返回 False
        assert not inference.is_hashable(i)
    # 对于 abc_hashable_not_really_hashable 中的每个元素，断言其不可哈希
    for i in abc_hashable_not_really_hashable:
        assert not inference.is_hashable(i)

    # numpy.array 在 https://github.com/numpy/numpy/pull/5326 中不再是 collections.abc.Hashable，
    # 因此需要测试 is_hashable() 函数来验证其不可哈希性
    assert not inference.is_hashable(np.array([]))
# 使用 pytest 的装饰器标记参数化测试函数，测试参数为编译后的正则表达式对象，验证推理模块的 is_re 方法
@pytest.mark.parametrize("ll", [re.compile("ad")])
def test_is_re_passes(ll):
    assert inference.is_re(ll)

# 使用 pytest 的装饰器标记参数化测试函数，测试参数包括非正则表达式对象（字符串、整数、对象），验证推理模块的 is_re 方法不接受非正则表达式对象
@pytest.mark.parametrize("ll", ["x", 2, 3, object()])
def test_is_re_fails(ll):
    assert not inference.is_re(ll)

# 使用 pytest 的装饰器标记参数化测试函数，测试参数为可编译的正则表达式字符串或对象，验证推理模块的 is_re_compilable 方法
@pytest.mark.parametrize(
    "ll", [r"a", "x", r"asdf", re.compile("adsf"), r"\u2233\s*", re.compile(r"")]
)
def test_is_recompilable_passes(ll):
    assert inference.is_re_compilable(ll)

# 使用 pytest 的装饰器标记参数化测试函数，测试参数包括不可编译的对象（整数、空列表、对象），验证推理模块的 is_re_compilable 方法不接受不可编译的对象
@pytest.mark.parametrize("ll", [1, [], object()])
def test_is_recompilable_fails(ll):
    assert not inference.is_re_compilable(ll)

# 定义测试类 TestInference，测试推理模块的不同功能
class TestInference:
    # 使用 pytest 的装饰器标记参数化测试函数，测试参数为字节类型的数组，验证库中 infer_dtype 方法对字节类型的推断
    @pytest.mark.parametrize(
        "arr",
        [
            np.array(list("abc"), dtype="S1"),
            np.array(list("abc"), dtype="S1").astype(object),
            [b"a", np.nan, b"c"],
        ],
    )
    def test_infer_dtype_bytes(self, arr):
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "bytes"

    # 使用 pytest 的装饰器标记参数化测试函数，测试参数为不同的标量值，验证库中 libmissing 模块的 isposinf_scalar 方法
    @pytest.mark.parametrize(
        "value, expected",
        [
            (float("inf"), True),
            (np.inf, True),
            (-np.inf, False),
            (1, False),
            ("a", False),
        ],
    )
    def test_isposinf_scalar(self, value, expected):
        # GH 11352
        result = libmissing.isposinf_scalar(value)
        assert result is expected

    # 使用 pytest 的装饰器标记参数化测试函数，测试参数为不同的标量值，验证库中 libmissing 模块的 isneginf_scalar 方法
    @pytest.mark.parametrize(
        "value, expected",
        [
            (float("-inf"), True),
            (-np.inf, True),
            (np.inf, False),
            (1, False),
            ("a", False),
        ],
    )
    def test_isneginf_scalar(self, value, expected):
        result = libmissing.isneginf_scalar(value)
        assert result is expected

    # 使用 pytest 的装饰器标记参数化测试函数，测试参数为可能需要转换为 masked nullable 的布尔数组，验证库中 libops 模块的 maybe_convert_bool 方法
    @pytest.mark.parametrize(
        "convert_to_masked_nullable, exp",
        [
            (
                True,
                BooleanArray(
                    np.array([True, False], dtype="bool"), np.array([False, True])
                ),
            ),
            (False, np.array([True, np.nan], dtype="object")),
        ],
    )
    def test_maybe_convert_nullable_boolean(self, convert_to_masked_nullable, exp):
        # GH 40687
        arr = np.array([True, np.nan], dtype=object)
        result = libops.maybe_convert_bool(
            arr, set(), convert_to_masked_nullable=convert_to_masked_nullable
        )
        if convert_to_masked_nullable:
            tm.assert_extension_array_equal(BooleanArray(*result), exp)
        else:
            result = result[0]
            tm.assert_numpy_array_equal(result, exp)

    # 使用 pytest 的装饰器标记参数化测试函数，测试参数为转换为 masked nullable 的布尔数组和是否强制转换为数值的标志，验证库中 libops 模块的 maybe_convert_numeric_infinities 方法
    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    @pytest.mark.parametrize("coerce_numeric", [True, False])
    @pytest.mark.parametrize(
        "infinity", ["inf", "inF", "iNf", "Inf", "iNF", "InF", "INf", "INF"]
    )
    @pytest.mark.parametrize("prefix", ["", "-", "+"])
    def test_maybe_convert_numeric_infinities(
        self, coerce_numeric, infinity, prefix, convert_to_masked_nullable
    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    # 使用 pytest 的参数化装饰器，测试 convert_to_masked_nullable 参数为 True 和 False 时的情况
    def test_maybe_convert_numeric_infinities_raises(self, convert_to_masked_nullable):
        # 设置错误消息，用于捕获预期的 ValueError 异常
        msg = "Unable to parse string"
        # 使用 pytest 的断言来验证是否会引发 ValueError 异常，并检查错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            # 调用 lib.maybe_convert_numeric 函数，传入包含 "foo_inf" 的数组等参数
            lib.maybe_convert_numeric(
                np.array(["foo_inf"], dtype=object),
                # 设置可能的 NA 值集合
                na_values={"", "NULL", "nan"},
                # 禁止强制类型转换
                coerce_numeric=False,
                # 传入 convert_to_masked_nullable 参数
                convert_to_masked_nullable=convert_to_masked_nullable,
            )

    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    # 使用 pytest 的参数化装饰器，测试 convert_to_masked_nullable 参数为 True 和 False 时的情况
    def test_maybe_convert_numeric_post_floatify_nan(
        self, coerce, convert_to_masked_nullable
    ):
        # see gh-13314
        # 设置包含浮点数字符串的数据数组
        data = np.array(["1.200", "-999.000", "4.500"], dtype=object)
        # 设置期望的浮点数数组
        expected = np.array([1.2, np.nan, 4.5], dtype=np.float64)
        # 设置 NaN 值的集合
        nan_values = {-999, -999.0}

        # 调用 lib.maybe_convert_numeric 函数，传入相关参数
        out = lib.maybe_convert_numeric(
            data,
            nan_values,
            coerce,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )
        # 根据 convert_to_masked_nullable 参数选择不同的断言方式来验证结果
        if convert_to_masked_nullable:
            # 如果 convert_to_masked_nullable 为 True，则使用 FloatingArray 进行比较
            expected = FloatingArray(expected, np.isnan(expected))
            tm.assert_extension_array_equal(expected, FloatingArray(*out))
        else:
            # 如果 convert_to_masked_nullable 为 False，则直接比较 numpy 数组
            out = out[0]
            tm.assert_numpy_array_equal(out, expected)

    def test_convert_infs(self):
        # 设置包含 "inf" 的数组
        arr = np.array(["inf", "inf", "inf"], dtype="O")
        # 调用 lib.maybe_convert_numeric 函数，传入相关参数
        result, _ = lib.maybe_convert_numeric(arr, set(), False)
        # 断言结果的数据类型为 np.float64
        assert result.dtype == np.float64

        # 设置包含 "-inf" 的数组
        arr = np.array(["-inf", "-inf", "-inf"], dtype="O")
        # 调用 lib.maybe_convert_numeric 函数，传入相关参数
        result, _ = lib.maybe_convert_numeric(arr, set(), False)
        # 断言结果的数据类型为 np.float64
        assert result.dtype == np.float64

    def test_scientific_no_exponent(self):
        # See PR 12215
        # 设置包含科学计数法但没有指数的字符串数组
        arr = np.array(["42E", "2E", "99e", "6e"], dtype="O")
        # 调用 lib.maybe_convert_numeric 函数，传入相关参数
        result, _ = lib.maybe_convert_numeric(arr, set(), False, True)
        # 断言所有结果均为 NaN
        assert np.all(np.isnan(result))

    def test_convert_non_hashable(self):
        # GH13324
        # 确保我们处理非可哈希对象
        # 设置包含不可哈希对象的数组
        arr = np.array([[10.0, 2], 1.0, "apple"], dtype=object)
        # 调用 lib.maybe_convert_numeric 函数，传入相关参数
        result, _ = lib.maybe_convert_numeric(arr, set(), False, True)
        # 使用 pytest 的断言来比较结果数组
        tm.assert_numpy_array_equal(result, np.array([np.nan, 1.0, np.nan]))
    # 定义测试方法，用于测试将对象数组转换为 uint64 类型的功能
    def test_convert_numeric_uint64(self):
        # 创建一个包含 2^63 的对象数组 arr，dtype 设置为 object 类型
        arr = np.array([2**63], dtype=object)
        # 创建期望结果 exp，包含一个 uint64 类型的数组，数值为 2^63
        exp = np.array([2**63], dtype=np.uint64)
        # 使用 numpy.testing 模块的 assert_numpy_array_equal 方法，验证转换后的结果
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)

        # 创建一个包含字符串 '2^63' 的对象数组 arr，dtype 设置为 object 类型
        arr = np.array([str(2**63)], dtype=object)
        # 再次创建期望结果 exp，同样包含一个 uint64 类型的数组，数值为 2^63
        exp = np.array([2**63], dtype=np.uint64)
        # 验证字符串 '2^63' 能否正确转换为 uint64 类型
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)

        # 创建一个包含 np.uint64 类型对象的数组 arr，dtype 设置为 object 类型
        arr = np.array([np.uint64(2**63)], dtype=object)
        # 再次创建期望结果 exp，同样包含一个 uint64 类型的数组，数值为 2^63
        exp = np.array([2**63], dtype=np.uint64)
        # 验证 np.uint64 类型对象能否正确转换为 uint64 类型
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)

    # 使用 pytest.mark.parametrize 装饰器，定义测试参数化方法，测试带 NaN 值的 uint64 转换
    @pytest.mark.parametrize(
        "arr",
        [
            np.array([2**63, np.nan], dtype=object),  # 包含一个 NaN 值的对象数组
            np.array([str(2**63), np.nan], dtype=object),  # 包含一个 NaN 值的对象数组
            np.array([np.nan, 2**63], dtype=object),  # 包含一个 NaN 值的对象数组
            np.array([np.nan, str(2**63)], dtype=object),  # 包含一个 NaN 值的对象数组
        ],
    )
    # 定义测试方法，用于验证带 NaN 值的 uint64 转换是否正确
    def test_convert_numeric_uint64_nan(self, coerce, arr):
        # 如果 coerce 参数为 True，则将 arr 转换为 float 类型，并赋值给 expected
        expected = arr.astype(float) if coerce else arr.copy()
        # 调用 lib.maybe_convert_numeric 方法进行转换，获取转换后的结果 result 和 mask
        result, _ = lib.maybe_convert_numeric(arr, set(), coerce_numeric=coerce)
        # 使用 numpy.testing 模块的 assert_almost_equal 方法验证转换后的结果
        tm.assert_almost_equal(result, expected)

    # 使用 pytest.mark.parametrize 装饰器，定义测试参数化方法，测试带 NaN 值的 uint64 数组转换
    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    # 定义测试方法，验证带 NaN 值的 uint64 数组转换是否正确
    def test_convert_numeric_uint64_nan_values(
        self, coerce, convert_to_masked_nullable
    ):
        # 创建一个包含多个 uint64 值的对象数组 arr，dtype 设置为 object 类型
        arr = np.array([2**63, 2**63 + 1], dtype=object)
        # 创建 NaN 值集合 na_values，包含值 2^63
        na_values = {2**63}

        # 如果 coerce 参数为 True，则将 arr 转换为 float 类型，并赋值给 expected
        expected = np.array([np.nan, 2**63 + 1], dtype=float) if coerce else arr.copy()
        # 调用 lib.maybe_convert_numeric 方法进行转换，传入额外参数及标志
        result = lib.maybe_convert_numeric(
            arr,
            na_values,
            coerce_numeric=coerce,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )
        # 根据 convert_to_masked_nullable 和 coerce 的值，进行不同的结果处理
        if convert_to_masked_nullable and coerce:
            # 创建 IntegerArray 对象，以 64 位无符号整数数组和布尔掩码数组初始化
            expected = IntegerArray(
                np.array([0, 2**63 + 1], dtype="u8"),
                np.array([True, False], dtype="bool"),
            )
            # 使用 IntegerArray 类型封装 result 的第一个元素
            result = IntegerArray(*result)
        else:
            # 丢弃 result 的掩码部分，仅保留数值部分
            result = result[0]  # discard mask
        # 使用 numpy.testing 模块的 assert_almost_equal 方法验证最终的结果
        tm.assert_almost_equal(result, expected)

    # 使用 pytest.mark.parametrize 装饰器，定义测试参数化方法，测试带有 int64 和 uint64 混合值的转换
    @pytest.mark.parametrize(
        "case",
        [
            np.array([2**63, -1], dtype=object),  # 包含一个 int64 值和一个负数的对象数组
            np.array([str(2**63), -1], dtype=object),  # 包含一个字符串和一个负数的对象数组
            np.array([str(2**63), str(-1)], dtype=object),  # 包含两个字符串的对象数组
            np.array([-1, 2**63], dtype=object),  # 包含一个负数和一个 uint64 值的对象数组
            np.array([-1, str(2**63)], dtype=object),  # 包含一个负数和一个字符串的对象数组
            np.array([str(-1), str(2**63)], dtype=object),  # 包含两个负数字符串的对象数组
        ],
    )
    # 定义测试方法，用于验证 int64 和 uint64 混合值的转换是否正确
    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_convert_numeric_int64_uint64(
        self, case, coerce, convert_to_masked_nullable
    ):
        # 如果 coerce 参数为 True，则将 case 转换为 float 类型，并赋值给 expected
        expected = case.astype(float) if coerce else case.copy()
        # 调用 lib.maybe_convert_numeric 方法进行转换，获取转换后的结果 result 和 mask
        result, _ = lib.maybe_convert_numeric(
            case,
            set(),
            coerce_numeric=coerce,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )

        # 使用 numpy.testing 模块的 assert_almost_equal 方法验证转换后的结果
        tm.assert_almost_equal(result, expected)
    # 使用 pytest 的参数化装饰器，测试函数的多个输入情况，用于测试 convert_to_masked_nullable 参数
    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_convert_numeric_string_uint64(self, convert_to_masked_nullable):
        # 标识为 GH32394 的测试用例
        result = lib.maybe_convert_numeric(
            np.array(["uint64"], dtype=object),
            set(),
            coerce_numeric=True,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )
        if convert_to_masked_nullable:
            # 如果 convert_to_masked_nullable 为 True，则将结果转换为 FloatingArray 类型
            result = FloatingArray(*result)
        else:
            # 否则，取结果的第一个元素
            result = result[0]
        # 断言结果应为 NaN
        assert np.isnan(result)

    # 使用 pytest 的参数化装饰器，测试函数的多个输入情况，用于测试 value 参数
    @pytest.mark.parametrize("value", [-(2**63) - 1, 2**64])
    def test_convert_int_overflow(self, value):
        # 查看 gh-18584 的相关信息
        arr = np.array([value], dtype=object)
        # 调用 lib.maybe_convert_objects 函数处理数组
        result = lib.maybe_convert_objects(arr)
        # 断言处理后的结果与原始数组相等
        tm.assert_numpy_array_equal(arr, result)

    # 使用 pytest 的参数化装饰器，测试函数的多个输入情况，用于测试 val 和 dtype 参数
    @pytest.mark.parametrize("val", [None, np.nan, float("nan")])
    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_maybe_convert_objects_nat_inference(self, val, dtype):
        dtype = np.dtype(dtype)
        vals = np.array([pd.NaT, val], dtype=object)
        # 调用 lib.maybe_convert_objects 函数处理 vals 数组
        result = lib.maybe_convert_objects(
            vals,
            convert_non_numeric=True,
            dtype_if_all_nat=dtype,
        )
        # 断言处理后结果的 dtype 为预期的 dtype
        assert result.dtype == dtype
        # 断言处理后结果中所有元素均为 NaT（Not a Time）
        assert np.isnat(result).all()

        # 逆序处理 vals 数组
        result = lib.maybe_convert_objects(
            vals[::-1],
            convert_non_numeric=True,
            dtype_if_all_nat=dtype,
        )
        # 再次断言处理后结果的 dtype 为预期的 dtype
        assert result.dtype == dtype
        # 再次断言处理后结果中所有元素均为 NaT
        assert np.isnat(result).all()

    # 使用 pytest 的参数化装饰器，测试函数的多个输入情况，用于测试 value 和 expected_dtype 参数
    @pytest.mark.parametrize(
        "value, expected_dtype",
        [
            # 查看 gh-4471 的相关信息
            ([2**63], np.uint64),
            # NumPy 的一个 bug：不能将 uint64 类型与 int64 类型比较，因为结果会强制转换为 float64，
            # 因此此函数需要确保对此情况具有鲁棒性
            ([np.uint64(2**63)], np.uint64),
            ([2, -1], np.int64),
            ([2**63, -1], object),
            # 查看 GH#47294 的相关信息
            ([np.uint8(1)], np.uint8),
            ([np.uint16(1)], np.uint16),
            ([np.uint32(1)], np.uint32),
            ([np.uint64(1)], np.uint64),
            ([np.uint8(2), np.uint16(1)], np.uint16),
            ([np.uint32(2), np.uint16(1)], np.uint32),
            ([np.uint32(2), -1], object),
            ([np.uint32(2), 1], np.uint64),
            ([np.uint32(2), np.int32(1)], object),
        ],
    )
    def test_maybe_convert_objects_uint(self, value, expected_dtype):
        # 创建包含给定值的 numpy 数组，数据类型为 object
        arr = np.array(value, dtype=object)
        # 创建包含预期数据类型的 numpy 数组
        exp = np.array(value, dtype=expected_dtype)
        # 调用 lib.maybe_convert_objects 函数处理 arr 数组
        tm.assert_numpy_array_equal(lib.maybe_convert_objects(arr), exp)
    # 定义测试函数 test_maybe_convert_objects_datetime，用于测试 maybe_convert_objects 方法的日期时间转换功能
    def test_maybe_convert_objects_datetime(self):
        # GH27438：测试用例标识符
        arr = np.array(
            [np.datetime64("2000-01-01"), np.timedelta64(1, "s")], dtype=object
        )
        exp = arr.copy()  # 复制 arr 数组作为期望输出
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)  # 调用方法进行对象转换
        tm.assert_numpy_array_equal(out, exp)  # 断言转换后的输出与期望输出相等

        arr = np.array([pd.NaT, np.timedelta64(1, "s")], dtype=object)
        exp = np.array([np.timedelta64("NaT"), np.timedelta64(1, "s")], dtype="m8[ns]")
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, exp)

        # with convert_non_numeric=True, the nan is a valid NA value for td64
        arr = np.array([np.timedelta64(1, "s"), np.nan], dtype=object)
        exp = exp[::-1]  # 反转 exp 数组
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, exp)

    # 定义测试函数 test_maybe_convert_objects_dtype_if_all_nat，用于测试 maybe_convert_objects 方法处理全为 NaT 时的类型转换情况
    def test_maybe_convert_objects_dtype_if_all_nat(self):
        arr = np.array([pd.NaT, pd.NaT], dtype=object)
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        # no dtype_if_all_nat passed -> we dont guess
        tm.assert_numpy_array_equal(out, arr)  # 断言转换后的输出与输入相等

        out = lib.maybe_convert_objects(
            arr,
            convert_non_numeric=True,
            dtype_if_all_nat=np.dtype("timedelta64[ns]"),
        )
        exp = np.array(["NaT", "NaT"], dtype="timedelta64[ns]")
        tm.assert_numpy_array_equal(out, exp)

        out = lib.maybe_convert_objects(
            arr,
            convert_non_numeric=True,
            dtype_if_all_nat=np.dtype("datetime64[ns]"),
        )
        exp = np.array(["NaT", "NaT"], dtype="datetime64[ns]")
        tm.assert_numpy_array_equal(out, exp)

    # 定义测试函数 test_maybe_convert_objects_dtype_if_all_nat_invalid，用于测试 maybe_convert_objects 方法处理全为 NaT 时传入无效类型的情况
    def test_maybe_convert_objects_dtype_if_all_nat_invalid(self):
        # we accept datetime64[ns], timedelta64[ns], and EADtype
        arr = np.array([pd.NaT, pd.NaT], dtype=object)

        with pytest.raises(ValueError, match="int64"):
            lib.maybe_convert_objects(
                arr,
                convert_non_numeric=True,
                dtype_if_all_nat=np.dtype("int64"),
            )

    # 使用 pytest.mark.parametrize 标记参数化测试，测试 maybe_convert_objects 方法处理日期时间溢出的安全性
    @pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
    def test_maybe_convert_objects_datetime_overflow_safe(self, dtype):
        stamp = datetime(2363, 10, 4)  # Enterprise-D launch date
        if dtype == "timedelta64[ns]":
            stamp = stamp - datetime(1970, 1, 1)
        arr = np.array([stamp], dtype=object)

        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        # no OutOfBoundsDatetime/OutOfBoundsTimedeltas
        if dtype == "datetime64[ns]":
            expected = np.array(["2363-10-04"], dtype="M8[us]")
        else:
            expected = arr
        tm.assert_numpy_array_equal(out, expected)
    # 定义测试方法，用于测试混合日期时间对象的转换功能
    def test_maybe_convert_objects_mixed_datetimes(self):
        # 创建 Timestamp 对象
        ts = Timestamp("now")
        # 准备包含不同类型值的列表
        vals = [ts, ts.to_pydatetime(), ts.to_datetime64(), pd.NaT, np.nan, None]

        # 遍历值的所有排列组合
        for data in itertools.permutations(vals):
            # 将数据转换为包含对象类型的 NumPy 数组
            data = np.array(list(data), dtype=object)
            # 创建预期的 DatetimeIndex 的内部 ndarray
            expected = DatetimeIndex(data)._data._ndarray
            # 调用库函数进行对象转换，并获取结果
            result = lib.maybe_convert_objects(data, convert_non_numeric=True)
            # 断言 NumPy 数组相等性
            tm.assert_numpy_array_equal(result, expected)

    # 定义测试方法，用于测试 timedelta64("NaT", "ns") 对象的转换功能
    def test_maybe_convert_objects_timedelta64_nat(self):
        # 创建 timedelta64("NaT", "ns") 对象
        obj = np.timedelta64("NaT", "ns")
        # 创建包含对象类型的 NumPy 数组
        arr = np.array([obj], dtype=object)
        # 断言数组的首个元素与原对象相等
        assert arr[0] is obj

        # 调用库函数进行对象转换，并获取结果
        result = lib.maybe_convert_objects(arr, convert_non_numeric=True)

        # 创建预期的 timedelta64 数组
        expected = np.array([obj], dtype="m8[ns]")
        # 断言 NumPy 数组相等性
        tm.assert_numpy_array_equal(result, expected)

    # 使用 pytest 的参数化装饰器标记，定义测试方法，测试可空整数类型的对象转换功能
    @pytest.mark.parametrize(
        "exp",
        [
            IntegerArray(np.array([2, 0], dtype="i8"), np.array([False, True])),
            IntegerArray(np.array([2, 0], dtype="int64"), np.array([False, True])),
        ],
    )
    def test_maybe_convert_objects_nullable_integer(self, exp):
        # GH27335
        # 创建包含对象类型的 NumPy 数组，包括整数和 NaN 值
        arr = np.array([2, np.nan], dtype=object)
        # 调用库函数进行对象转换，并获取结果
        result = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)

        # 断言扩展数组相等性
        tm.assert_extension_array_equal(result, exp)

    # 使用 pytest 的参数化装饰器标记，定义测试方法，测试可空类型为 None 的对象转换功能
    @pytest.mark.parametrize(
        "dtype, val", [("int64", 1), ("uint64", np.iinfo(np.int64).max + 1)]
    )
    def test_maybe_convert_objects_nullable_none(self, dtype, val):
        # GH#50043
        # 创建包含对象类型的 NumPy 数组，包括整数、None 和另一个整数
        arr = np.array([val, None, 3], dtype="object")
        # 调用库函数进行对象转换，并获取结果
        result = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        # 创建预期的 IntegerArray
        expected = IntegerArray(
            np.array([val, 0, 3], dtype=dtype), np.array([False, True, False])
        )
        # 断言扩展数组相等性
        tm.assert_extension_array_equal(result, expected)

    # 使用 pytest 的参数化装饰器标记，定义测试方法，测试可空整数类型的数值转换功能
    @pytest.mark.parametrize(
        "convert_to_masked_nullable, exp",
        [
            (True, IntegerArray(np.array([2, 0], dtype="i8"), np.array([False, True]))),
            (False, np.array([2, np.nan], dtype="float64")),
        ],
    )
    def test_maybe_convert_numeric_nullable_integer(
        self, convert_to_masked_nullable, exp
    ):
        # GH 40687
        # 创建包含对象类型的 NumPy 数组，包括整数和 NaN 值
        arr = np.array([2, np.nan], dtype=object)
        # 调用库函数进行数值转换，并获取结果
        result = lib.maybe_convert_numeric(
            arr, set(), convert_to_masked_nullable=convert_to_masked_nullable
        )
        if convert_to_masked_nullable:
            # 如果需要转换为可掩码的可空整数类型，转换结果为 IntegerArray
            result = IntegerArray(*result)
            # 断言扩展数组相等性
            tm.assert_extension_array_equal(result, exp)
        else:
            # 否则，直接获取结果的第一个元素
            result = result[0]
            # 断言 NumPy 数组相等性
            tm.assert_numpy_array_equal(result, exp)

    # 使用 pytest 的参数化装饰器标记，定义测试方法，测试可空浮点数类型的数值转换功能
    @pytest.mark.parametrize(
        "convert_to_masked_nullable, exp",
        [
            (
                True,
                FloatingArray(
                    np.array([2.0, 0.0], dtype="float64"), np.array([False, True])
                ),
            ),
            (False, np.array([2.0, np.nan], dtype="float64")),
        ],
    )
    # 定义测试函数，用于测试可能转换为数字浮点数组的情况
    def test_maybe_convert_numeric_floating_array(
        self, convert_to_masked_nullable, exp
    ):
        # GH 40687
        # 创建一个包含浮点数和NaN的numpy数组，数据类型为object
        arr = np.array([2.0, np.nan], dtype=object)
        # 调用库函数maybe_convert_numeric，处理数组arr
        result = lib.maybe_convert_numeric(
            arr, set(), convert_to_masked_nullable=convert_to_masked_nullable
        )
        # 如果参数convert_to_masked_nullable为True，断言处理后的结果与期望值exp相等
        if convert_to_masked_nullable:
            tm.assert_extension_array_equal(FloatingArray(*result), exp)
        else:
            # 否则，获取处理结果的第一个元素，并断言其与期望值exp相等
            result = result[0]
            tm.assert_numpy_array_equal(result, exp)

    # 定义测试函数，用于测试对象数组转换为布尔值和NaN的情况
    def test_maybe_convert_objects_bool_nan(self):
        # GH32146
        # 创建一个包含True、False和NaN的Index对象，数据类型为object
        ind = Index([True, False, np.nan], dtype=object)
        # 创建一个期望的numpy数组，与ind相同
        exp = np.array([True, False, np.nan], dtype=object)
        # 调用库函数maybe_convert_objects，处理ind的值
        out = lib.maybe_convert_objects(ind.values, safe=1)
        # 断言处理后的结果与期望值exp相等
        tm.assert_numpy_array_equal(out, exp)

    # 定义测试函数，用于测试对象数组转换为可空布尔类型的情况
    def test_maybe_convert_objects_nullable_boolean(self):
        # GH50047
        # 创建一个包含True和False的numpy数组，数据类型为object
        arr = np.array([True, False], dtype=object)
        # 创建一个期望的BooleanArray对象
        exp = BooleanArray._from_sequence([True, False], dtype="boolean")
        # 调用库函数maybe_convert_objects，将arr转换为可空数据类型
        out = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        # 断言处理后的结果与期望值exp相等
        tm.assert_extension_array_equal(out, exp)

        # 创建一个包含True、False和pd.NaT的numpy数组，数据类型为object
        arr = np.array([True, False, pd.NaT], dtype=object)
        # 创建一个与arr相同的期望numpy数组
        exp = np.array([True, False, pd.NaT], dtype=object)
        # 再次调用库函数maybe_convert_objects，将arr转换为可空数据类型
        out = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        # 断言处理后的结果与期望值exp相等
        tm.assert_numpy_array_equal(out, exp)

    # 使用pytest的参数化装饰器，测试可能的空布尔类型值的情况
    @pytest.mark.parametrize("val", [None, np.nan])
    def test_maybe_convert_objects_nullable_boolean_na(self, val):
        # GH50047
        # 创建一个包含True、False和val的numpy数组，数据类型为object
        arr = np.array([True, False, val], dtype=object)
        # 创建一个期望的BooleanArray对象
        exp = BooleanArray(
            np.array([True, False, False]), np.array([False, False, True])
        )
        # 调用库函数maybe_convert_objects，将arr转换为可空数据类型
        out = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        # 断言处理后的结果与期望值exp相等
        tm.assert_extension_array_equal(out, exp)

    # 使用pytest的参数化装饰器，参数化data0和data1，测试不同类型数据的情况
    @pytest.mark.parametrize(
        "data0",
        [
            True,
            1,
            1.0,
            1.0 + 1.0j,
            np.int8(1),
            np.int16(1),
            np.int32(1),
            np.int64(1),
            np.float16(1),
            np.float32(1),
            np.float64(1),
            np.complex64(1),
            np.complex128(1),
        ],
    )
    @pytest.mark.parametrize(
        "data1",
        [
            True,
            1,
            1.0,
            1.0 + 1.0j,
            np.int8(1),
            np.int16(1),
            np.int32(1),
            np.int64(1),
            np.float16(1),
            np.float32(1),
            np.float64(1),
            np.complex64(1),
            np.complex128(1),
        ],
    )
    # GH 40908
    # 将传入的数据 data0 和 data1 组成列表 data
    data = [data0, data1]
    # 使用列表 data 创建 NumPy 数组 arr，数据类型为 "object"
    arr = np.array(data, dtype="object")

    # 确定 data0 和 data1 的公共数据类型种类
    common_kind = np.result_type(type(data0), type(data1)).kind
    # 确定 data0 的数据类型种类，如果没有 dtype 属性，则为 "python"
    kind0 = "python" if not hasattr(data0, "dtype") else data0.dtype.kind
    # 确定 data1 的数据类型种类，如果没有 dtype 属性，则为 "python"
    kind1 = "python" if not hasattr(data1, "dtype") else data1.dtype.kind

    # 根据不同情况确定数据类型种类 kind 和数据项大小 itemsize
    if kind0 != "python" and kind1 != "python":
        kind = common_kind
        itemsize = max(data0.dtype.itemsize, data1.dtype.itemsize)
    elif is_bool(data0) or is_bool(data1):
        kind = "bool" if (is_bool(data0) and is_bool(data1)) else "object"
        itemsize = ""
    elif is_complex(data0) or is_complex(data1):
        kind = common_kind
        itemsize = 16
    else:
        kind = common_kind
        itemsize = 8

    # 根据确定的 kind 和 itemsize 创建预期的 NumPy 数组 expected
    expected = np.array(data, dtype=f"{kind}{itemsize}")
    # 调用 lib.maybe_convert_objects 函数处理 arr，得到处理后的结果 result
    result = lib.maybe_convert_objects(arr)
    # 使用测试工具 tm 断言 result 和 expected 数组相等
    tm.assert_numpy_array_equal(result, expected)


    # GH14956
    # 创建包含混合数据类型的 NumPy 数组 arr，数据类型为 "object"
    arr = np.array([datetime(2015, 1, 1, tzinfo=timezone.utc), 1], dtype=object)
    # 调用 lib.maybe_convert_objects 函数处理 arr，设置 convert_non_numeric=True
    result = lib.maybe_convert_objects(arr, convert_non_numeric=True)
    # 使用测试工具 tm 断言 result 和 arr 数组相等
    tm.assert_numpy_array_equal(result, arr)


    @pytest.mark.parametrize(
        "idx",
        [
            # 创建 IntervalIndex 对象 idx，使用范围为 [0, 1, 2, 3, 4] 的断点，闭区间
            pd.IntervalIndex.from_breaks(range(5), closed="both"),
            # 创建 period_range 对象 idx，从 "2016-01-01" 开始的三个日期，每天频率
            pd.period_range("2016-01-01", periods=3, freq="D"),
        ],
    )
    # 测试 lib.maybe_convert_objects 处理 idx 对象的情况
    def test_maybe_convert_objects_ea(self, idx):
        # 调用 lib.maybe_convert_objects 函数处理 idx 数组，设置 convert_non_numeric=True
        result = lib.maybe_convert_objects(
            np.array(idx, dtype=object),
            convert_non_numeric=True,
        )
        # 使用测试工具 tm 断言 result 和 idx._data 扩展数组相等
        tm.assert_extension_array_equal(result, idx._data)
# 定义一个用于测试Python对象的虚拟类
class TestTypeInference:
    class Dummy:
        pass

    # 测试从夹具中推断数据类型
    def test_inferred_dtype_fixture(self, any_skipna_inferred_dtype):
        # 从夹具中获取推断的数据类型和值
        inferred_dtype, values = any_skipna_inferred_dtype
        # 确保夹具推断的数据类型与请求的一致
        assert inferred_dtype == lib.infer_dtype(values, skipna=True)

    # 测试长度为零的情况
    def test_length_zero(self, skipna):
        # 对空的NumPy数组推断数据类型为整数
        result = lib.infer_dtype(np.array([], dtype="i4"), skipna=skipna)
        assert result == "integer"

        # 对空列表推断数据类型为空
        result = lib.infer_dtype([], skipna=skipna)
        assert result == "empty"

        # GH 18004
        # 对包含空数组的数组推断数据类型为空
        arr = np.array([np.array([], dtype=object), np.array([], dtype=object)])
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == "empty"

    # 测试整数情况
    def test_integers(self):
        # 对混合类型的NumPy数组推断数据类型为整数
        arr = np.array([1, 2, 3, np.int64(4), np.int32(5)], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "integer"

        # 对混合整数和字符串类型的NumPy数组推断数据类型为混合整数
        arr = np.array([1, 2, 3, np.int64(4), np.int32(5), "foo"], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "mixed-integer"

        # 对整数类型的NumPy数组推断数据类型为整数
        arr = np.array([1, 2, 3, 4, 5], dtype="i4")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "integer"

    # 使用参数化测试不同情况下的整数与NA值推断数据类型
    @pytest.mark.parametrize(
        "arr, skipna",
        [
            ([1, 2, np.nan, np.nan, 3], False),
            ([1, 2, np.nan, np.nan, 3], True),
            ([1, 2, 3, np.int64(4), np.int32(5), np.nan], False),
            ([1, 2, 3, np.int64(4), np.int32(5), np.nan], True),
        ],
    )
    def test_integer_na(self, arr, skipna):
        # GH 27392
        # 对包含NA值的对象类型的NumPy数组推断数据类型为整数或整数NA
        result = lib.infer_dtype(np.array(arr, dtype="O"), skipna=skipna)
        expected = "integer" if skipna else "integer-na"
        assert result == expected

    # 测试推断数据类型时，skipna默认值的情况
    def test_infer_dtype_skipna_default(self):
        # infer_dtype `skipna` 默认值在 GH#24050 被弃用，
        # 在 GH#29876 中更改为True
        arr = np.array([1, 2, 3, np.nan], dtype=object)
        # 推断包含NA值的对象类型的NumPy数组数据类型为整数
        result = lib.infer_dtype(arr)
        assert result == "integer"
    # 定义一个测试函数，用于测试布尔类型数据推断
    def test_bools(self):
        # 创建一个包含布尔值的 NumPy 数组，dtype 设置为对象类型
        arr = np.array([True, False, True, True, True], dtype="O")
        # 调用 lib.infer_dtype 函数，推断数组的数据类型，跳过 NaN 值
        result = lib.infer_dtype(arr, skipna=True)
        # 断言推断结果应为 "boolean"
        assert result == "boolean"

        # 创建一个包含 np.bool_ 类型的 NumPy 数组，dtype 设置为对象类型
        arr = np.array([np.bool_(True), np.bool_(False)], dtype="O")
        # 再次调用推断函数
        result = lib.infer_dtype(arr, skipna=True)
        # 断言推断结果应为 "boolean"
        assert result == "boolean"

        # 创建一个包含布尔值和字符串的混合类型 NumPy 数组，dtype 设置为对象类型
        arr = np.array([True, False, True, "foo"], dtype="O")
        # 再次调用推断函数
        result = lib.infer_dtype(arr, skipna=True)
        # 断言推断结果应为 "mixed"
        assert result == "mixed"

        # 创建一个包含布尔值的 NumPy 数组，dtype 设置为 bool 类型
        arr = np.array([True, False, True], dtype=bool)
        # 再次调用推断函数
        result = lib.infer_dtype(arr, skipna=True)
        # 断言推断结果应为 "boolean"
        assert result == "boolean"

        # 创建一个包含布尔值和 NaN 值的 NumPy 数组，dtype 设置为对象类型
        arr = np.array([True, np.nan, False], dtype="O")
        # 再次调用推断函数
        result = lib.infer_dtype(arr, skipna=True)
        # 断言推断结果应为 "boolean"
        assert result == "boolean"

        # 再次调用推断函数，但这次不跳过 NaN 值
        result = lib.infer_dtype(arr, skipna=False)
        # 断言推断结果应为 "mixed"
        assert result == "mixed"

    # 定义一个测试函数，用于测试浮点数类型数据推断
    def test_floats(self):
        # 创建一个包含浮点数的 NumPy 数组，dtype 设置为对象类型
        arr = np.array([1.0, 2.0, 3.0, np.float64(4), np.float32(5)], dtype="O")
        # 调用 lib.infer_dtype 函数，推断数组的数据类型，跳过 NaN 值
        result = lib.infer_dtype(arr, skipna=True)
        # 断言推断结果应为 "floating"
        assert result == "floating"

        # 创建一个包含整数、浮点数和字符串的混合类型 NumPy 数组，dtype 设置为对象类型
        arr = np.array([1, 2, 3, np.float64(4), np.float32(5), "foo"], dtype="O")
        # 再次调用推断函数
        result = lib.infer_dtype(arr, skipna=True)
        # 断言推断结果应为 "mixed-integer"
        assert result == "mixed-integer"

        # 创建一个包含浮点数的 NumPy 数组，dtype 设置为 f4 浮点数类型
        arr = np.array([1, 2, 3, 4, 5], dtype="f4")
        # 再次调用推断函数
        result = lib.infer_dtype(arr, skipna=True)
        # 断言推断结果应为 "floating"
        assert result == "floating"

        # 创建一个包含浮点数的 NumPy 数组，dtype 设置为 f8 浮点数类型
        arr = np.array([1, 2, 3, 4, 5], dtype="f8")
        # 再次调用推断函数
        result = lib.infer_dtype(arr, skipna=True)
        # 断言推断结果应为 "floating"
        assert result == "floating"

    # 定义一个测试函数，用于测试十进制数类型数据推断
    def test_decimals(self):
        # GH15690
        # 创建一个包含 Decimal 类型的 NumPy 数组
        arr = np.array([Decimal(1), Decimal(2), Decimal(3)])
        # 调用 lib.infer_dtype 函数，推断数组的数据类型，跳过 NaN 值
        result = lib.infer_dtype(arr, skipna=True)
        # 断言推断结果应为 "decimal"
        assert result == "decimal"

        # 创建一个包含浮点数和 Decimal 类型的混合类型 NumPy 数组
        arr = np.array([1.0, 2.0, Decimal(3)])
        # 再次调用推断函数
        result = lib.infer_dtype(arr, skipna=True)
        # 断言推断结果应为 "mixed"
        assert result == "mixed"

        # 将数组倒序后再次调用推断函数
        result = lib.infer_dtype(arr[::-1], skipna=True)
        # 断言推断结果应为 "mixed"
        assert result == "mixed"

        # 创建一个包含 Decimal 类型和 NaN 值的 NumPy 数组，dtype 设置为对象类型
        arr = np.array([Decimal(1), Decimal("NaN"), Decimal(3)])
        # 再次调用推断函数
        result = lib.infer_dtype(arr, skipna=True)
        # 断言推断结果应为 "decimal"
        assert result == "decimal"

        # 创建一个包含 Decimal 类型和 NaN 值的 NumPy 数组，dtype 设置为对象类型
        arr = np.array([Decimal(1), np.nan, Decimal(3)], dtype="O")
        # 再次调用推断函数
        result = lib.infer_dtype(arr, skipna=True)
        # 断言推断结果应为 "decimal"
        assert result == "decimal"
    # 测试复杂数据类型推断函数的功能，使用不同的输入数组进行测试

    # 创建包含浮点数和复数的 NumPy 数组，根据 skipna 参数推断数据类型，应返回 "complex"
    arr = np.array([1.0, 2.0, 1 + 1j])
    result = lib.infer_dtype(arr, skipna=skipna)
    assert result == "complex"

    # 创建包含浮点数和复数的 NumPy 数组，指定 dtype="O"，根据 skipna 参数推断数据类型，应返回 "mixed"
    arr = np.array([1.0, 2.0, 1 + 1j], dtype="O")
    result = lib.infer_dtype(arr, skipna=skipna)
    assert result == "mixed"

    # 使用切片反转数组，根据 skipna 参数推断数据类型，应返回 "mixed"
    result = lib.infer_dtype(arr[::-1], skipna=skipna)
    assert result == "mixed"

    # 创建包含整数、NaN 和复数的 NumPy 数组，根据 skipna 参数推断数据类型，应返回 "complex"
    arr = np.array([1, np.nan, 1 + 1j])
    result = lib.infer_dtype(arr, skipna=skipna)
    assert result == "complex"

    # 创建包含浮点数、NaN 和复数的 NumPy 数组，指定 dtype="O"，根据 skipna 参数推断数据类型，应返回 "mixed"
    arr = np.array([1.0, np.nan, 1 + 1j], dtype="O")
    result = lib.infer_dtype(arr, skipna=skipna)
    assert result == "mixed"

    # 创建包含复数、NaN 和复数的 NumPy 数组，指定 dtype="O"，根据 skipna 参数推断数据类型，应返回 "complex"
    arr = np.array([1 + 1j, np.nan, 3 + 3j], dtype="O")
    result = lib.infer_dtype(arr, skipna=skipna)
    assert result == "complex"

    # 创建包含复数、NaN 和复数的 NumPy 数组，指定 dtype=np.complex64，根据 skipna 参数推断数据类型，应返回 "complex"
    arr = np.array([1 + 1j, np.nan, 3 + 3j], dtype=np.complex64)
    result = lib.infer_dtype(arr, skipna=skipna)
    assert result == "complex"
    
    # 测试字符串类型数据推断函数
    def test_string(self):
        pass

    # 测试Unicode类型数据推断函数
    def test_unicode(self):
        # 创建包含字符串和 NaN 的数组，根据 skipna 参数推断数据类型，应返回 "mixed"
        arr = ["a", np.nan, "c"]
        result = lib.infer_dtype(arr, skipna=False)
        # 当前返回 "mixed"，但不确定这是最佳选择
        assert result == "mixed"

        # 创建包含字符串和 NaN 的数组，根据 skipna 参数推断数据类型，应返回 "string"
        arr = ["a", np.nan, "c"]
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "string"

        # 创建包含字符串和 pd.NA 的数组，根据 skipna 参数推断数据类型，应返回 "string"
        arr = ["a", pd.NA, "c"]
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "string"

        # 创建包含字符串和 pd.NaT 的数组，根据 skipna 参数推断数据类型，应返回 "mixed"
        arr = ["a", pd.NaT, "c"]
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "mixed"

        # 创建仅包含字符串的数组，根据 skipna 参数推断数据类型，应返回 "string"
        arr = ["a", "c"]
        result = lib.infer_dtype(arr, skipna=False)
        assert result == "string"

    # 使用参数化测试来测试对象为空时的推断行为
    @pytest.mark.parametrize(
        "dtype, missing, skipna, expected",
        [
            (float, np.nan, False, "floating"),
            (float, np.nan, True, "floating"),
            (object, np.nan, False, "floating"),
            (object, np.nan, True, "empty"),
            (object, None, False, "mixed"),
            (object, None, True, "empty"),
        ],
    )
    @pytest.mark.parametrize("box", [Series, np.array])
    def test_object_empty(self, box, missing, dtype, skipna, expected):
        # GH 23421
        # 使用指定的缺失值和数据类型创建数组，根据 skipna 参数推断数据类型，应返回 expected 中指定的类型
        arr = box([missing, missing], dtype=dtype)

        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == expected

    # 测试日期时间类型数据推断函数
    def test_datetime(self):
        # 创建包含日期时间的列表，创建 Index 对象后，断言推断的数据类型应为 "datetime64"
        dates = [datetime(2012, 1, x) for x in range(1, 20)]
        index = Index(dates)
        assert index.inferred_type == "datetime64"
    # 测试函数，用于测试推断包含 datetime64 类型的数组的数据类型推断功能
    def test_infer_dtype_datetime64(self):
        # 创建包含两个 datetime64 类型数据的 numpy 数组
        arr = np.array(
            [np.datetime64("2011-01-01"), np.datetime64("2011-01-01")], dtype=object
        )
        # 断言推断结果为 "datetime64"
        assert lib.infer_dtype(arr, skipna=True) == "datetime64"

    @pytest.mark.parametrize("na_value", [pd.NaT, np.nan])
    # 参数化测试函数，用于测试在包含 NaN 值时的 datetime64 类型数据类型推断功能
    def test_infer_dtype_datetime64_with_na(self, na_value):
        # 创建包含 NaN 值和一个 datetime64 类型数据的 numpy 数组
        arr = np.array([na_value, np.datetime64("2011-01-02")])
        # 断言推断结果为 "datetime64"
        assert lib.infer_dtype(arr, skipna=True) == "datetime64"

        # 创建包含 NaN 值、一个 datetime64 类型数据和再次 NaN 值的 numpy 数组
        arr = np.array([na_value, np.datetime64("2011-01-02"), na_value])
        # 断言推断结果为 "datetime64"
        assert lib.infer_dtype(arr, skipna=True) == "datetime64"

    @pytest.mark.parametrize(
        "arr",
        [
            np.array(
                [np.timedelta64("nat"), np.datetime64("2011-01-02")], dtype=object
            ),
            np.array(
                [np.datetime64("2011-01-02"), np.timedelta64("nat")], dtype=object
            ),
            np.array([np.datetime64("2011-01-01"), Timestamp("2011-01-02")]),
            np.array([Timestamp("2011-01-02"), np.datetime64("2011-01-01")]),
            np.array([np.nan, Timestamp("2011-01-02"), 1.1]),
            np.array([np.nan, "2011-01-01", Timestamp("2011-01-02")], dtype=object),
            np.array([np.datetime64("nat"), np.timedelta64(1, "D")], dtype=object),
            np.array([np.timedelta64(1, "D"), np.datetime64("nat")], dtype=object),
        ],
    )
    # 参数化测试函数，用于测试混合类型数据（包含 datetime64）的数据类型推断功能
    def test_infer_datetimelike_dtype_mixed(self, arr):
        # 断言推断结果为 "mixed"
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

    # 测试函数，用于测试推断包含 mixed-integer 类型的数组的数据类型推断功能
    def test_infer_dtype_mixed_integer(self):
        # 创建包含 NaN 值、Timestamp 类型数据和整数 1 的 numpy 数组
        arr = np.array([np.nan, Timestamp("2011-01-02"), 1])
        # 断言推断结果为 "mixed-integer"
        assert lib.infer_dtype(arr, skipna=True) == "mixed-integer"

    @pytest.mark.parametrize(
        "arr",
        [
            [Timestamp("2011-01-01"), Timestamp("2011-01-02")],
            [datetime(2011, 1, 1), datetime(2012, 2, 1)],
            [datetime(2011, 1, 1), Timestamp("2011-01-02")],
        ],
    )
    # 参数化测试函数，用于测试推断包含 datetime 类型的数组的数据类型推断功能
    def test_infer_dtype_datetime(self, arr):
        # 断言推断结果为 "datetime"
        assert lib.infer_dtype(np.array(arr), skipna=True) == "datetime"

    @pytest.mark.parametrize("na_value", [pd.NaT, np.nan])
    @pytest.mark.parametrize(
        "time_stamp", [Timestamp("2011-01-01"), datetime(2011, 1, 1)]
    )
    # 参数化测试函数，用于测试在包含 NaN 值时的 datetime 类型数据类型推断功能
    def test_infer_dtype_datetime_with_na(self, na_value, time_stamp):
        # 创建包含 NaN 值和一个 datetime 类型数据的 numpy 数组
        arr = np.array([na_value, time_stamp])
        # 断言推断结果为 "datetime"
        assert lib.infer_dtype(arr, skipna=True) == "datetime"

        # 创建包含 NaN 值、一个 datetime 类型数据和再次 NaN 值的 numpy 数组
        arr = np.array([na_value, time_stamp, na_value])
        # 断言推断结果为 "datetime"
        assert lib.infer_dtype(arr, skipna=True) == "datetime"

    @pytest.mark.parametrize(
        "arr",
        [
            np.array([Timedelta("1 days"), Timedelta("2 days")]),
            np.array([np.timedelta64(1, "D"), np.timedelta64(2, "D")], dtype=object),
            np.array([timedelta(1), timedelta(2)]),
        ],
    )
    # 参数化测试函数，用于测试推断包含 timedelta 类型的数组的数据类型推断功能
    def test_infer_dtype_timedelta(self, arr):
        # 断言推断结果为 "timedelta"
        assert lib.infer_dtype(arr, skipna=True) == "timedelta"
    @pytest.mark.parametrize("na_value", [pd.NaT, np.nan])
    @pytest.mark.parametrize(
        "delta", [Timedelta("1 days"), np.timedelta64(1, "D"), timedelta(1)]
    )
    # 使用 pytest 的 parametrize 装饰器，对测试用例进行参数化，na_value 和 delta 是参数
    def test_infer_dtype_timedelta_with_na(self, na_value, delta):
        # 创建包含 NaN 值和时间增量的 NumPy 数组
        arr = np.array([na_value, delta])
        # 调用 lib.infer_dtype 函数，验证数组类型推断为 "timedelta"
        assert lib.infer_dtype(arr, skipna=True) == "timedelta"

        # 添加另一个 NaN 值，再次调用类型推断函数，验证数组类型仍为 "timedelta"
        arr = np.array([na_value, delta, na_value])
        assert lib.infer_dtype(arr, skipna=True) == "timedelta"

    def test_infer_dtype_period(self):
        # GH 13664
        # 创建包含 Period 对象的 NumPy 数组，频率为 "D"
        arr = np.array([Period("2011-01", freq="D"), Period("2011-02", freq="D")])
        # 验证 lib.infer_dtype 函数推断结果为 "period"
        assert lib.infer_dtype(arr, skipna=True) == "period"

        # 创建包含频率不同的 Period 对象数组，验证推断结果为 "mixed"
        arr = np.array([Period("2011-01", freq="D"), Period("2011-02", freq="M")])
        assert lib.infer_dtype(arr, skipna=True) == "mixed"

    def test_infer_dtype_period_array(self, index_or_series_or_array, skipna):
        klass = index_or_series_or_array
        # https://github.com/pandas-dev/pandas/issues/23553
        # 创建包含 Period 对象和 NaT 的数组
        values = klass(
            [
                Period("2011-01-01", freq="D"),
                Period("2011-01-02", freq="D"),
                pd.NaT,
            ]
        )
        # 验证 lib.infer_dtype 函数推断结果为 "period"
        assert lib.infer_dtype(values, skipna=skipna) == "period"

        # 创建包含频率不同的 Period 对象和 NaT 的数组
        values = klass(
            [
                Period("2011-01-01", freq="D"),
                Period("2011-01-02", freq="M"),
                pd.NaT,
            ]
        )
        # 根据 klass 类型，验证 lib.infer_dtype 函数推断结果为 "unknown-array" 或 "mixed"
        exp = "unknown-array" if klass is pd.array else "mixed"
        assert lib.infer_dtype(values, skipna=skipna) == exp

    def test_infer_dtype_period_mixed(self):
        # 创建包含 Period 对象和 np.datetime64("nat") 的数组，dtype 为 object
        arr = np.array(
            [Period("2011-01", freq="M"), np.datetime64("nat")], dtype=object
        )
        # 验证 lib.infer_dtype 函数推断结果为 "mixed"
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        # 创建包含 np.datetime64("nat") 和 Period 对象的数组，dtype 为 object
        arr = np.array(
            [np.datetime64("nat"), Period("2011-01", freq="M")], dtype=object
        )
        # 验证 lib.infer_dtype 函数推断结果为 "mixed"
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

    @pytest.mark.parametrize("na_value", [pd.NaT, np.nan])
    def test_infer_dtype_period_with_na(self, na_value):
        # 创建包含 NaN 值和 Period 对象的数组
        arr = np.array([na_value, Period("2011-01", freq="D")])
        # 验证 lib.infer_dtype 函数推断结果为 "period"
        assert lib.infer_dtype(arr, skipna=True) == "period"

        # 添加另一个 NaN 值，再次验证推断结果为 "period"
        arr = np.array([na_value, Period("2011-01", freq="D"), na_value])
        assert lib.infer_dtype(arr, skipna=True) == "period"
    # 定义测试函数 test_infer_dtype_all_nan_nat_like，用于测试推断数组类型的函数
    def test_infer_dtype_all_nan_nat_like(self):
        # 创建包含两个 NaN 的 NumPy 数组 arr
        arr = np.array([np.nan, np.nan])
        # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "floating"
        assert lib.infer_dtype(arr, skipna=True) == "floating"

        # 创建包含 NaN 和 None 的混合数组 arr
        arr = np.array([np.nan, np.nan, None])
        # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "empty"（跳过 NaN，空值类型为空）
        assert lib.infer_dtype(arr, skipna=True) == "empty"
        # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "mixed"（不跳过 NaN，混合类型）
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        # 创建包含 None 和 NaN 的混合数组 arr
        arr = np.array([None, np.nan, np.nan])
        # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "empty"（跳过 NaN，空值类型为空）
        assert lib.infer_dtype(arr, skipna=True) == "empty"
        # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "mixed"（不跳过 NaN，混合类型）
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        # 创建包含 pd.NaT 的数组 arr
        arr = np.array([pd.NaT])
        # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "datetime"（日期时间类型）
        assert lib.infer_dtype(arr, skipna=False) == "datetime"

        # 创建包含 pd.NaT 和 NaN 的数组 arr
        arr = np.array([pd.NaT, np.nan])
        # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "datetime"（日期时间类型）
        assert lib.infer_dtype(arr, skipna=False) == "datetime"

        # 创建包含 NaN 和 pd.NaT 的数组 arr
        arr = np.array([np.nan, pd.NaT])
        # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "datetime"（日期时间类型）
        assert lib.infer_dtype(arr, skipna=False) == "datetime"

        # 创建包含 NaN、pd.NaT 和 NaN 的数组 arr
        arr = np.array([np.nan, pd.NaT, np.nan])
        # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "datetime"（日期时间类型）
        assert lib.infer_dtype(arr, skipna=False) == "datetime"

        # 创建包含 None、pd.NaT 和 None 的数组 arr
        arr = np.array([None, pd.NaT, None])
        # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "datetime"（日期时间类型）
        assert lib.infer_dtype(arr, skipna=False) == "datetime"

        # 创建包含 np.datetime64("nat") 的数组 arr
        arr = np.array([np.datetime64("nat")])
        # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "datetime64"（日期时间类型）
        assert lib.infer_dtype(arr, skipna=False) == "datetime64"

        # 遍历包含 np.nan、pd.NaT 和 None 的元素 n 的列表
        for n in [np.nan, pd.NaT, None]:
            # 创建包含 n、np.datetime64("nat") 和 n 的数组 arr
            arr = np.array([n, np.datetime64("nat"), n])
            # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "datetime64"（日期时间类型）
            assert lib.infer_dtype(arr, skipna=False) == "datetime64"

            # 创建包含 pd.NaT、n、np.datetime64("nat") 和 n 的数组 arr
            arr = np.array([pd.NaT, n, np.datetime64("nat"), n])
            # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "datetime64"（日期时间类型）
            assert lib.infer_dtype(arr, skipna=False) == "datetime64"

        # 创建包含 np.timedelta64("nat") 的数组 arr，并指定为 object 类型
        arr = np.array([np.timedelta64("nat")], dtype=object)
        # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "timedelta"（时间增量类型）
        assert lib.infer_dtype(arr, skipna=False) == "timedelta"

        # 遍历包含 np.nan、pd.NaT 和 None 的元素 n 的列表
        for n in [np.nan, pd.NaT, None]:
            # 创建包含 n、np.timedelta64("nat") 和 n 的数组 arr
            arr = np.array([n, np.timedelta64("nat"), n])
            # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "timedelta"（时间增量类型）
            assert lib.infer_dtype(arr, skipna=False) == "timedelta"

            # 创建包含 pd.NaT、n、np.timedelta64("nat") 和 n 的数组 arr
            arr = np.array([pd.NaT, n, np.timedelta64("nat"), n])
            # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "timedelta"（时间增量类型）
            assert lib.infer_dtype(arr, skipna=False) == "timedelta"

        # 创建包含 pd.NaT、np.datetime64("nat")、np.timedelta64("nat") 和 np.nan 的数组 arr
        arr = np.array([pd.NaT, np.datetime64("nat"), np.timedelta64("nat"), np.nan])
        # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "mixed"（混合类型）
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        # 创建包含 np.timedelta64("nat") 和 np.datetime64("nat") 的数组 arr，并指定为 object 类型
        arr = np.array([np.timedelta64("nat"), np.datetime64("nat")], dtype=object)
        # 断言调用 lib.infer_dtype 函数对 arr 推断出的类型为 "mixed"（混合类型）
        assert lib.infer_dtype(arr, skipna=False) == "mixed"
    # 定义测试函数，检查输入数组是否全部为 NaN 或 NaT 或 nat 类型
    def test_is_datetimelike_array_all_nan_nat_like(self):
        # 创建包含 NaN、NaT 和 nat 类型的 NumPy 数组
        arr = np.array([np.nan, pd.NaT, np.datetime64("nat")])
        # 断言数组是否为日期时间数组
        assert lib.is_datetime_array(arr)
        # 断言数组是否为 datetime64 类型数组
        assert lib.is_datetime64_array(arr)
        # 断言数组不是 timedelta 或 timedelta64 类型数组
        assert not lib.is_timedelta_or_timedelta64_array(arr)

        # 创建包含 NaN、NaT 和 nat 类型的 NumPy 数组
        arr = np.array([np.nan, pd.NaT, np.timedelta64("nat")])
        # 断言数组不是日期时间数组
        assert not lib.is_datetime_array(arr)
        # 断言数组不是 datetime64 类型数组
        assert not lib.is_datetime64_array(arr)
        # 断言数组是 timedelta 或 timedelta64 类型数组
        assert lib.is_timedelta_or_timedelta64_array(arr)

        # 创建包含 NaN、NaT、nat 类型的 NumPy 数组和 timedelta64 类型的数组
        arr = np.array([np.nan, pd.NaT, np.datetime64("nat"), np.timedelta64("nat")])
        # 断言数组不是日期时间数组
        assert not lib.is_datetime_array(arr)
        # 断言数组不是 datetime64 类型数组
        assert not lib.is_datetime64_array(arr)
        # 断言数组不是 timedelta 或 timedelta64 类型数组
        assert not lib.is_timedelta_or_timedelta64_array(arr)

        # 创建包含 NaN 和 NaT 类型的 NumPy 数组
        arr = np.array([np.nan, pd.NaT])
        # 断言数组是日期时间数组
        assert lib.is_datetime_array(arr)
        # 断言数组是 datetime64 类型数组
        assert lib.is_datetime64_array(arr)
        # 断言数组是 timedelta 或 timedelta64 类型数组
        assert lib.is_timedelta_or_timedelta64_array(arr)

        # 创建包含 NaN 类型的 NumPy 数组，数据类型为 object
        arr = np.array([np.nan, np.nan], dtype=object)
        # 断言数组不是日期时间数组
        assert not lib.is_datetime_array(arr)
        # 断言数组不是 datetime64 类型数组
        assert not lib.is_datetime64_array(arr)
        # 断言数组不是 timedelta 或 timedelta64 类型数组
        assert not lib.is_timedelta_or_timedelta64_array(arr)

        # 断言数组是带单一时区的日期时间数组
        assert lib.is_datetime_with_singletz_array(
            np.array(
                [
                    Timestamp("20130101", tz="US/Eastern"),
                    Timestamp("20130102", tz="US/Eastern"),
                ],
                dtype=object,
            )
        )
        # 断言数组不是带单一时区的日期时间数组
        assert not lib.is_datetime_with_singletz_array(
            np.array(
                [
                    Timestamp("20130101", tz="US/Eastern"),
                    Timestamp("20130102", tz="CET"),
                ],
                dtype=object,
            )
        )

    # 使用 pytest 的参数化装饰器，测试 lib 模块中各种数据类型的数组
    @pytest.mark.parametrize(
        "func",
        [
            "is_datetime_array",
            "is_datetime64_array",
            "is_bool_array",
            "is_timedelta_or_timedelta64_array",
            "is_date_array",
            "is_time_array",
            "is_interval_array",
        ],
    )
    # 定义测试函数，检查不同函数对于非日期时间类型的数组的判断
    def test_other_dtypes_for_array(self, func):
        # 获取 lib 模块中相应的函数对象
        func = getattr(lib, func)
        # 创建字符串数组
        arr = np.array(["foo", "bar"])
        # 断言函数对字符串数组的判断结果为 False
        assert not func(arr)
        # 断言函数对重塑后的字符串数组的判断结果为 False
        assert not func(arr.reshape(2, 1))

        # 创建整数数组
        arr = np.array([1, 2])
        # 断言函数对整数数组的判断结果为 False
        assert not func(arr)
        # 断言函数对重塑后的整数数组的判断结果为 False
        assert not func(arr.reshape(2, 1))

    # 定义测试日期处理的函数
    def test_date(self):
        # 创建日期对象列表
        dates = [date(2012, 1, day) for day in range(1, 20)]
        # 创建日期索引对象
        index = Index(dates)
        # 断言索引对象推断的类型为 "date"
        assert index.inferred_type == "date"

        # 创建包含日期对象和 NaN 的日期对象列表
        dates = [date(2012, 1, day) for day in range(1, 20)] + [np.nan]
        # 使用 lib 模块的函数推断数据类型，跳过 NaN 值
        result = lib.infer_dtype(dates, skipna=False)
        # 断言推断的数据类型为 "mixed"
        assert result == "mixed"

        # 使用 lib 模块的函数推断数据类型，包含 NaN 值
        result = lib.infer_dtype(dates, skipna=True)
        # 断言推断的数据类型为 "date"
        assert result == "date"

    # 使用 pytest 的参数化装饰器，测试不同的日期和时间值
    @pytest.mark.parametrize(
        "values",
        [
            [date(2020, 1, 1), Timestamp("2020-01-01")],
            [Timestamp("2020-01-01"), date(2020, 1, 1)],
            [date(2020, 1, 1), pd.NaT],
            [pd.NaT, date(2020, 1, 1)],
        ],
    )
    def test_infer_dtype_date_order_invariant(self, values, skipna):
        # 根据 GitHub 上的 issue 33741，测试推断数据类型是否不受日期顺序影响
        result = lib.infer_dtype(values, skipna=skipna)
        # 断言推断结果为 "date"
        assert result == "date"

    def test_is_numeric_array(self):
        # 断言给定数组是否为浮点数数组
        assert lib.is_float_array(np.array([1, 2.0]))
        assert lib.is_float_array(np.array([1, 2.0, np.nan]))
        assert not lib.is_float_array(np.array([1, 2]))

        # 断言给定数组是否为整数数组
        assert lib.is_integer_array(np.array([1, 2]))
        assert not lib.is_integer_array(np.array([1, 2.0]))

    def test_is_string_array(self):
        # 当 skipna=True 时，仅接受 pd.NA、np.nan、其他浮点数 NaN（例如 float('nan')）
        assert lib.is_string_array(np.array(["foo", "bar"]))
        assert not lib.is_string_array(
            np.array(["foo", "bar", pd.NA], dtype=object), skipna=False
        )
        assert lib.is_string_array(
            np.array(["foo", "bar", pd.NA], dtype=object), skipna=True
        )
        # 在 StringArray 构造函数中允许 NaN/None，因此这里也允许
        assert lib.is_string_array(
            np.array(["foo", "bar", None], dtype=object), skipna=True
        )
        assert lib.is_string_array(
            np.array(["foo", "bar", np.nan], dtype=object), skipna=True
        )
        # 但不允许例如 datetime 类型或 Decimal 的 NA
        assert not lib.is_string_array(
            np.array(["foo", "bar", pd.NaT], dtype=object), skipna=True
        )
        assert not lib.is_string_array(
            np.array(["foo", "bar", np.datetime64("NaT")], dtype=object), skipna=True
        )
        assert not lib.is_string_array(
            np.array(["foo", "bar", Decimal("NaN")], dtype=object), skipna=True
        )

        assert not lib.is_string_array(
            np.array(["foo", "bar", None], dtype=object), skipna=False
        )
        assert not lib.is_string_array(
            np.array(["foo", "bar", np.nan], dtype=object), skipna=False
        )
        assert not lib.is_string_array(np.array([1, 2]))

    def test_to_object_array_tuples(self):
        r = (5, 6)
        values = [r]
        lib.to_object_array_tuples(values)

        # 确保记录数组正常工作
        record = namedtuple("record", "x y")
        r = record(5, 6)
        values = [r]
        lib.to_object_array_tuples(values)

    def test_object(self):
        # GitHub issue 7431
        # 由于只有单个元素，不能推断更多
        arr = np.array([None], dtype="O")
        result = lib.infer_dtype(arr, skipna=False)
        # 断言推断结果为 "mixed"
        assert result == "mixed"
        result = lib.infer_dtype(arr, skipna=True)
        # 断言推断结果为 "empty"
        assert result == "empty"
    # 定义一个测试方法，用于测试将列表转换为对象数组的功能
    def test_to_object_array_width(self):
        # 测试用例的标识，参考 GitHub issue-13320
        rows = [[1, 2, 3], [4, 5, 6]]

        # 期望的输出是一个 NumPy 对象数组，数据类型为 object
        expected = np.array(rows, dtype=object)
        # 调用被测试的函数 to_object_array，并获取其输出
        out = lib.to_object_array(rows)
        # 使用测试工具函数验证 out 是否等于 expected
        tm.assert_numpy_array_equal(out, expected)

        # 再次测试 to_object_array，这次指定最小宽度为 1
        out = lib.to_object_array(rows, min_width=1)
        # 验证输出是否与预期相等
        tm.assert_numpy_array_equal(out, expected)

        # 测试 to_object_array，指定最小宽度为 5
        expected = np.array(
            [[1, 2, 3, None, None], [4, 5, 6, None, None]], dtype=object
        )
        out = lib.to_object_array(rows, min_width=5)
        # 验证输出是否与预期相等
        tm.assert_numpy_array_equal(out, expected)

    # 定义一个测试方法，用于测试分类数据的推断
    def test_categorical(self):
        # GitHub issue 8974
        # 创建一个分类数据对象 arr，包含字符 'abc'
        arr = Categorical(list("abc"))
        # 调用库函数 infer_dtype，验证对 arr 的数据类型推断结果
        result = lib.infer_dtype(arr, skipna=True)
        # 使用断言验证推断结果是否为 "categorical"
        assert result == "categorical"

        # 将 arr 包装成 Series 对象，再次进行数据类型推断
        result = lib.infer_dtype(Series(arr), skipna=True)
        # 使用断言验证推断结果是否为 "categorical"
        assert result == "categorical"

        # 创建一个具有指定分类和顺序的分类数据对象 arr
        arr = Categorical(list("abc"), categories=["cegfab"], ordered=True)
        # 再次进行数据类型推断
        result = lib.infer_dtype(arr, skipna=True)
        # 使用断言验证推断结果是否为 "categorical"
        assert result == "categorical"

        # 将 arr 包装成 Series 对象，再次进行数据类型推断
        result = lib.infer_dtype(Series(arr), skipna=True)
        # 使用断言验证推断结果是否为 "categorical"
        assert result == "categorical"

    # 使用 pytest 的参数化装饰器定义一个测试方法，测试区间数据类型的推断
    @pytest.mark.parametrize("asobject", [True, False])
    def test_interval(self, asobject):
        # 创建一个 IntervalIndex 对象 idx，其区间为 [0, 1, 2, 3, 4]，闭区间
        idx = pd.IntervalIndex.from_breaks(range(5), closed="both")
        # 如果 asobject 为 True，则将 idx 转换为 object 类型
        if asobject:
            idx = idx.astype(object)

        # 对 idx 进行数据类型推断，跳过 NaN 值检查
        inferred = lib.infer_dtype(idx, skipna=False)
        # 使用断言验证推断结果是否为 "interval"
        assert inferred == "interval"

        # 对 idx 的数据进行推断，跳过 NaN 值检查
        inferred = lib.infer_dtype(idx._data, skipna=False)
        # 使用断言验证推断结果是否为 "interval"
        assert inferred == "interval"

        # 将 idx 包装成 Series 对象，再次进行数据类型推断
        inferred = lib.infer_dtype(Series(idx, dtype=idx.dtype), skipna=False)
        # 使用断言验证推断结果是否为 "interval"
        assert inferred == "interval"

    # 使用 pytest 的参数化装饰器定义一个测试方法，测试区间类型不匹配的推断
    @pytest.mark.parametrize("value", [Timestamp(0), Timedelta(0), 0, 0.0])
    def test_interval_mismatched_closed(self, value):
        # 创建两个具有不同闭合方式的 Interval 对象
        first = Interval(value, value, closed="left")
        second = Interval(value, value, closed="right")

        # 如果两个 Interval 对象的闭合方式匹配，则推断结果应为 "interval"
        arr = np.array([first, first], dtype=object)
        # 使用断言验证推断结果是否为 "interval"
        assert lib.infer_dtype(arr, skipna=False) == "interval"

        # 如果两个 Interval 对象的闭合方式不匹配，则推断结果应为 "mixed"
        arr2 = np.array([first, second], dtype=object)
        # 使用断言验证推断结果是否为 "mixed"
        assert lib.infer_dtype(arr2, skipna=False) == "mixed"
    # 测试不匹配的子类型间隔
    def test_interval_mismatched_subtype(self):
        # 创建整数区间对象，左闭右开
        first = Interval(0, 1, closed="left")
        # 创建时间戳区间对象，左闭右开
        second = Interval(Timestamp(0), Timestamp(1), closed="left")
        # 创建时间增量区间对象，左闭右开
        third = Interval(Timedelta(0), Timedelta(1), closed="left")

        # 将区间对象放入 NumPy 数组中
        arr = np.array([first, second])
        # 确定数组类型为混合类型
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        # 将不同类型的区间对象放入 NumPy 数组中
        arr = np.array([second, third])
        # 确定数组类型为混合类型
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        # 将整数和时间增量类型的区间对象放入 NumPy 数组中
        arr = np.array([first, third])
        # 确定数组类型为混合类型
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        # 浮点数与整数子类型是兼容的
        # 创建浮点数区间对象，左闭右开
        flt_interval = Interval(1.5, 2.5, closed="left")
        # 将整数和浮点数区间对象放入 NumPy 数组中，使用对象类型
        arr = np.array([first, flt_interval], dtype=object)
        # 确定数组类型为区间类型
        assert lib.infer_dtype(arr, skipna=False) == "interval"

    # 使用字符串数据类型的测试参数化
    @pytest.mark.parametrize("data", [["a", "b", "c"], ["a", "b", pd.NA]])
    def test_string_dtype(
        self, data, skipna, index_or_series_or_array, nullable_string_dtype
    ):
        # 创建 StringArray 对象
        val = index_or_series_or_array(data, dtype=nullable_string_dtype)
        # 推断数据类型
        inferred = lib.infer_dtype(val, skipna=skipna)
        # 确保推断出的数据类型为字符串
        assert inferred == "string"

    # 使用布尔数据类型的测试参数化
    @pytest.mark.parametrize("data", [[True, False, True], [True, False, pd.NA]])
    def test_boolean_dtype(self, data, skipna, index_or_series_or_array):
        # 创建 BooleanArray 对象
        val = index_or_series_or_array(data, dtype="boolean")
        # 推断数据类型
        inferred = lib.infer_dtype(val, skipna=skipna)
        # 确保推断出的数据类型为布尔值
        assert inferred == "boolean"
class TestNumberScalar:
    # 测试函数，验证输入是否为数值类型
    def test_is_number(self):
        # 断言以下为数值类型
        assert is_number(True)
        assert is_number(1)
        assert is_number(1.1)
        assert is_number(1 + 3j)
        assert is_number(np.int64(1))
        assert is_number(np.float64(1.1))
        assert is_number(np.complex128(1 + 3j))
        assert is_number(np.nan)

        # 断言以下不为数值类型
        assert not is_number(None)
        assert not is_number("x")
        assert not is_number(datetime(2011, 1, 1))
        assert not is_number(np.datetime64("2011-01-01"))
        assert not is_number(Timestamp("2011-01-01"))
        assert not is_number(Timestamp("2011-01-01", tz="US/Eastern"))
        assert not is_number(timedelta(1000))
        assert not is_number(Timedelta("1 days"))

        # 疑问情况
        assert not is_number(np.bool_(False))
        assert is_number(np.timedelta64(1, "D"))

    # 测试函数，验证输入是否为布尔类型
    def test_is_bool(self):
        # 断言以下为布尔类型
        assert is_bool(True)
        assert is_bool(False)
        assert is_bool(np.bool_(False))

        # 断言以下不为布尔类型
        assert not is_bool(1)
        assert not is_bool(1.1)
        assert not is_bool(1 + 3j)
        assert not is_bool(np.int64(1))
        assert not is_bool(np.float64(1.1))
        assert not is_bool(np.complex128(1 + 3j))
        assert not is_bool(np.nan)
        assert not is_bool(None)
        assert not is_bool("x")
        assert not is_bool(datetime(2011, 1, 1))
        assert not is_bool(np.datetime64("2011-01-01"))
        assert not is_bool(Timestamp("2011-01-01"))
        assert not is_bool(Timestamp("2011-01-01", tz="US/Eastern"))
        assert not is_bool(timedelta(1000))
        assert not is_bool(np.timedelta64(1, "D"))
        assert not is_bool(Timedelta("1 days"))

    # 测试函数，验证输入是否为整数类型
    def test_is_integer(self):
        # 断言以下为整数类型
        assert is_integer(1)
        assert is_integer(np.int64(1))

        # 断言以下不为整数类型
        assert not is_integer(True)
        assert not is_integer(1.1)
        assert not is_integer(1 + 3j)
        assert not is_integer(False)
        assert not is_integer(np.bool_(False))
        assert not is_integer(np.float64(1.1))
        assert not is_integer(np.complex128(1 + 3j))
        assert not is_integer(np.nan)
        assert not is_integer(None)
        assert not is_integer("x")
        assert not is_integer(datetime(2011, 1, 1))
        assert not is_integer(np.datetime64("2011-01-01"))
        assert not is_integer(Timestamp("2011-01-01"))
        assert not is_integer(Timestamp("2011-01-01", tz="US/Eastern"))
        assert not is_integer(timedelta(1000))
        assert not is_integer(Timedelta("1 days"))
        assert not is_integer(np.timedelta64(1, "D"))
    # 测试函数，用于检查 is_float 函数的行为
    def test_is_float(self):
        # 检查浮点数和 np.float64 类型的输入是否返回 True
        assert is_float(1.1)
        assert is_float(np.float64(1.1))
        # 检查 NaN 是否返回 True
        assert is_float(np.nan)

        # 检查布尔值、整数、复数、布尔数组、整数数组、复数数组、None、字符串、日期时间等是否返回 False
        assert not is_float(True)
        assert not is_float(1)
        assert not is_float(1 + 3j)
        assert not is_float(False)
        assert not is_float(np.bool_(False))
        assert not is_float(np.int64(1))
        assert not is_float(np.complex128(1 + 3j))
        assert not is_float(None)
        assert not is_float("x")
        assert not is_float(datetime(2011, 1, 1))
        assert not is_float(np.datetime64("2011-01-01"))
        assert not is_float(Timestamp("2011-01-01"))
        assert not is_float(Timestamp("2011-01-01", tz="US/Eastern"))
        assert not is_float(timedelta(1000))
        assert not is_float(np.timedelta64(1, "D"))
        assert not is_float(Timedelta("1 days"))

    # 测试函数，用于检查日期时间类型的函数行为
    def test_is_datetime_dtypes(self):
        # 创建一个日期时间序列和带时区的日期时间序列
        ts = pd.date_range("20130101", periods=3)
        tsa = pd.date_range("20130101", periods=3, tz="US/Eastern")

        # 警告信息
        msg = "is_datetime64tz_dtype is deprecated"

        # 检查不同类型的日期时间对象是否返回 True 或 False
        assert is_datetime64_dtype("datetime64")
        assert is_datetime64_dtype("datetime64[ns]")
        assert is_datetime64_dtype(ts)
        assert not is_datetime64_dtype(tsa)

        assert not is_datetime64_ns_dtype("datetime64")
        assert is_datetime64_ns_dtype("datetime64[ns]")
        assert is_datetime64_ns_dtype(ts)
        assert is_datetime64_ns_dtype(tsa)

        assert is_datetime64_any_dtype("datetime64")
        assert is_datetime64_any_dtype("datetime64[ns]")
        assert is_datetime64_any_dtype(ts)
        assert is_datetime64_any_dtype(tsa)

        # 检查带有时区信息的日期时间类型，同时检查警告是否会被触发
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert not is_datetime64tz_dtype("datetime64")
            assert not is_datetime64tz_dtype("datetime64[ns]")
            assert not is_datetime64tz_dtype(ts)
            assert is_datetime64tz_dtype(tsa)

    # 使用参数化测试进行时区相关的日期时间类型检查
    @pytest.mark.parametrize("tz", ["US/Eastern", "UTC"])
    def test_is_datetime_dtypes_with_tz(self, tz):
        # 构造带有时区信息的 datetime64[ns, tz] 类型字符串，进行类型检查
        dtype = f"datetime64[ns, {tz}]"
        assert not is_datetime64_dtype(dtype)

        # 检查带有时区信息的日期时间类型是否触发警告，同时检查其它日期时间类型的函数行为
        msg = "is_datetime64tz_dtype is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(dtype)
        assert is_datetime64_ns_dtype(dtype)
        assert is_datetime64_any_dtype(dtype)
    # 测试函数，用于测试时间增量数据类型的判断函数
    def test_is_timedelta(self):
        # 断言 "timedelta64" 是时间增量数据类型
        assert is_timedelta64_dtype("timedelta64")
        # 断言 "timedelta64[ns]" 是时间增量数据类型
        assert is_timedelta64_dtype("timedelta64[ns]")
        # 断言 "timedelta64" 不是纳秒精度的时间增量数据类型
        assert not is_timedelta64_ns_dtype("timedelta64")
        # 断言 "timedelta64[ns]" 是纳秒精度的时间增量数据类型
        assert is_timedelta64_ns_dtype("timedelta64[ns]")

        # 创建一个时间增量索引对象 tdi，包含两个时间增量值，数据类型为 "timedelta64[ns]"
        tdi = TimedeltaIndex([1e14, 2e14], dtype="timedelta64[ns]")
        # 断言 tdi 是时间增量数据类型
        assert is_timedelta64_dtype(tdi)
        # 断言 tdi 是纳秒精度的时间增量数据类型
        assert is_timedelta64_ns_dtype(tdi)
        # 将 tdi 转换为 "timedelta64[ns]" 数据类型后，断言其是纳秒精度的时间增量数据类型
        assert is_timedelta64_ns_dtype(tdi.astype("timedelta64[ns]"))

        # 创建一个空的索引对象，数据类型为 np.float64，断言它不是纳秒精度的时间增量数据类型
        assert not is_timedelta64_ns_dtype(Index([], dtype=np.float64))
        # 创建一个空的索引对象，数据类型为 np.int64，断言它不是纳秒精度的时间增量数据类型
        assert not is_timedelta64_ns_dtype(Index([], dtype=np.int64))
class TestIsScalar:
    # 测试函数：测试是否为标量（单一数值类型）
    def test_is_scalar_builtin_scalars(self):
        # 断言：判断是否为标量
        assert is_scalar(None)
        assert is_scalar(True)
        assert is_scalar(False)
        assert is_scalar(Fraction())
        assert is_scalar(0.0)
        assert is_scalar(1)
        assert is_scalar(complex(2))
        assert is_scalar(float("NaN"))
        assert is_scalar(np.nan)
        assert is_scalar("foobar")
        assert is_scalar(b"foobar")
        assert is_scalar(datetime(2014, 1, 1))
        assert is_scalar(date(2014, 1, 1))
        assert is_scalar(time(12, 0))
        assert is_scalar(timedelta(hours=1))
        assert is_scalar(pd.NaT)
        assert is_scalar(pd.NA)

    # 测试函数：测试非标量内置数据类型
    def test_is_scalar_builtin_nonscalars(self):
        assert not is_scalar({})
        assert not is_scalar([])
        assert not is_scalar([1])
        assert not is_scalar(())
        assert not is_scalar((1,))
        assert not is_scalar(slice(None))
        assert not is_scalar(Ellipsis)

    # 测试函数：测试 numpy 数组标量
    def test_is_scalar_numpy_array_scalars(self):
        assert is_scalar(np.int64(1))
        assert is_scalar(np.float64(1.0))
        assert is_scalar(np.int32(1))
        assert is_scalar(np.complex64(2))
        assert is_scalar(np.object_("foobar"))
        assert is_scalar(np.str_("foobar"))
        assert is_scalar(np.bytes_(b"foobar"))
        assert is_scalar(np.datetime64("2014-01-01"))
        assert is_scalar(np.timedelta64(1, "h"))

    # 参数化测试：测试 numpy 零维数组
    @pytest.mark.parametrize(
        "zerodim",
        [
            1,
            "foobar",
            np.datetime64("2014-01-01"),
            np.timedelta64(1, "h"),
            np.datetime64("NaT"),
        ],
    )
    def test_is_scalar_numpy_zerodim_arrays(self, zerodim):
        zerodim = np.array(zerodim)
        assert not is_scalar(zerodim)
        assert is_scalar(lib.item_from_zerodim(zerodim))

    # 参数化测试：测试 numpy 数组
    @pytest.mark.parametrize("arr", [np.array([]), np.array([[]])])
    def test_is_scalar_numpy_arrays(self, arr):
        assert not is_scalar(arr)
        assert not is_scalar(MockNumpyLikeArray(arr))

    # 测试函数：测试 pandas 标量
    def test_is_scalar_pandas_scalars(self):
        assert is_scalar(Timestamp("2014-01-01"))
        assert is_scalar(Timedelta(hours=1))
        assert is_scalar(Period("2014-01-01"))
        assert is_scalar(Interval(left=0, right=1))
        assert is_scalar(DateOffset(days=1))
        assert is_scalar(pd.offsets.Minute(3))

    # 测试函数：测试 pandas 容器类型
    def test_is_scalar_pandas_containers(self):
        assert not is_scalar(Series(dtype=object))
        assert not is_scalar(Series([1]))
        assert not is_scalar(DataFrame())
        assert not is_scalar(DataFrame([[1]]))
        assert not is_scalar(Index([]))
        assert not is_scalar(Index([1]))
        assert not is_scalar(Categorical([]))
        assert not is_scalar(DatetimeIndex([])._data)
        assert not is_scalar(TimedeltaIndex([])._data)
        assert not is_scalar(DatetimeIndex([])._data.to_period("D"))
        assert not is_scalar(pd.array([1, 2, 3]))
    def test_is_scalar_number(self):
        # 定义了一个测试函数 test_is_scalar_number，用于测试 is_scalar 函数对数字类型的判断
        # Number() 由于未被 PyNumber_Check 所识别，因此扩展而言也不被 is_scalar 函数所识别，
        # 但是非抽象子类的实例则被认为是标量（scalar）。

        # 定义一个 Numeric 类，继承自 Number 类
        class Numeric(Number):
            def __init__(self, value) -> None:
                self.value = value

            # 定义 __int__ 方法，返回实例的整数值
            def __int__(self) -> int:
                return self.value

        # 创建 Numeric 类的实例 num，值为 1
        num = Numeric(1)
        # 断言 num 是标量
        assert is_scalar(num)
# 使用 pytest 的 parametrize 装饰器来多次运行此测试函数，分别使用 "ms", "us", "ns" 作为参数
@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
def test_datetimeindex_from_empty_datetime64_array(unit):
    # 创建一个空的 DatetimeIndex 对象，数据类型为指定的 datetime64 单位
    idx = DatetimeIndex(np.array([], dtype=f"datetime64[{unit}]"))
    # 断言空的 DatetimeIndex 的长度为 0
    assert len(idx) == 0


# 测试 Pandas DataFrame 中 NaN 和 NaT 的转换
def test_nan_to_nat_conversions():
    # 创建一个 DataFrame 对象，包含两列：一列是 float64 数组，一列是指定的时间戳
    df = DataFrame({"A": np.asarray(range(10), dtype="float64"), "B": Timestamp("20010101")})
    # 将第三到第六行的所有列设置为 NaN
    df.iloc[3:6, :] = np.nan
    # 从 DataFrame 中提取特定位置的值，预期结果为 pd.NaT
    result = df.loc[4, "B"]
    assert result is pd.NaT

    # 复制 DataFrame 中的一列，并将其中第八行设置为 NaN
    s = df["B"].copy()
    s[8:9] = np.nan
    # 断言第八行的值为 pd.NaT
    assert s[8] is pd.NaT


# 使用 pytest 的 parametrize 装饰器来多次运行此测试函数，分别测试不同的 scipy 稀疏矩阵类型
@pytest.mark.parametrize("spmatrix", ["bsr", "coo", "csc", "csr", "dia", "dok", "lil"])
def test_is_scipy_sparse(spmatrix):
    # 导入 scipy.sparse 模块，如果导入失败则跳过测试
    sparse = pytest.importorskip("scipy.sparse")

    # 根据 spmatrix 字符串动态获取对应的稀疏矩阵类
    klass = getattr(sparse, spmatrix + "_matrix")
    # 断言一个包含 [0, 1] 的矩阵是 scipy 稀疏矩阵
    assert is_scipy_sparse(klass([[0, 1]]))
    # 断言一个 numpy 数组不是 scipy 稀疏矩阵
    assert not is_scipy_sparse(np.array([1]))


# 测试确保数组元素为 int32 类型的函数
def test_ensure_int32():
    # 创建一个从 0 到 9 的整数数组，数据类型为 np.int32
    values = np.arange(10, dtype=np.int32)
    # 调用函数 ensure_int32 处理数组，并断言返回结果的数据类型为 np.int32
    result = ensure_int32(values)
    assert result.dtype == np.int32

    # 创建一个从 0 到 9 的整数数组，数据类型为 np.int64
    values = np.arange(10, dtype=np.int64)
    # 再次调用 ensure_int32 处理数组，并断言返回结果的数据类型为 np.int32
    result = ensure_int32(values)
    assert result.dtype == np.int32


# 使用 pytest 的 parametrize 装饰器来多次运行此测试函数，测试不同的数据类型组合
@pytest.mark.parametrize(
    "right,result",
    [
        (0, np.uint8),
        (-1, np.int16),
        (300, np.uint16),
        # 对于浮点数，直接升级到 float64 而不是寻找更小的浮点数数据类型
        (300.0, np.uint16),  # 对于整数浮点数，将其转换为整数
        (300.1, np.float64),
        (np.int16(300), np.int16 if np_version_gt2 else np.uint16),
    ],
)
def test_find_result_type_uint_int(right, result):
    # 定义左侧数据类型为 uint8
    left_dtype = np.dtype("uint8")
    # 断言根据 left_dtype 和 right 计算得到的数据类型为 result
    assert find_result_type(left_dtype, right) == result


# 使用 pytest 的 parametrize 装饰器来多次运行此测试函数，测试不同的数据类型组合
@pytest.mark.parametrize(
    "right,result",
    [
        (0, np.int8),
        (-1, np.int8),
        (300, np.int16),
        # 对于浮点数，直接升级到 float64 而不是寻找更小的浮点数数据类型
        (300.0, np.int16),  # 对于整数浮点数，将其转换为整数
        (300.1, np.float64),
        (np.int16(300), np.int16),
    ],
)
def test_find_result_type_int_int(right, result):
    # 定义左侧数据类型为 int8
    left_dtype = np.dtype("int8")
    # 断言根据 left_dtype 和 right 计算得到的数据类型为 result
    assert find_result_type(left_dtype, right) == result


# 使用 pytest 的 parametrize 装饰器来多次运行此测试函数，测试不同的数据类型组合
@pytest.mark.parametrize(
    "right,result",
    [
        (300.0, np.float64),
        (np.float32(300), np.float32),
    ],
)
def test_find_result_type_floats(right, result):
    # 定义左侧数据类型为 float16
    left_dtype = np.dtype("float16")
    # 断言根据 left_dtype 和 right 计算得到的数据类型为 result
    assert find_result_type(left_dtype, right) == result
```