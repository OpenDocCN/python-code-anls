# `D:\src\scipysrc\pandas\pandas\tests\indexes\test_indexing.py`

```
"""
test_indexing tests the following Index methods:
    __getitem__
    get_loc
    get_value
    __contains__
    take
    where
    get_indexer
    get_indexer_for
    slice_locs
    asof_locs

The corresponding tests.indexes.[index_type].test_indexing files
contain tests for the corresponding methods specific to those Index subclasses.
"""

# 导入所需的模块和库
import numpy as np  # 导入NumPy库
import pytest  # 导入pytest库用于单元测试

from pandas.errors import InvalidIndexError  # 导入Pandas错误处理模块

from pandas.core.dtypes.common import (  # 导入Pandas通用数据类型模块
    is_float_dtype,  # 判断是否为浮点型数据类型
    is_scalar,  # 判断是否为标量
)

from pandas import (  # 导入Pandas核心模块
    NA,  # 缺失值
    DatetimeIndex,  # 时间日期索引
    Index,  # 标准索引
    IntervalIndex,  # 区间索引
    MultiIndex,  # 多级索引
    NaT,  # 不可用的时间戳
    PeriodIndex,  # 时期索引
    TimedeltaIndex,  # 时间增量索引
)
import pandas._testing as tm  # 导入Pandas测试工具模块


class TestTake:
    def test_take_invalid_kwargs(self, index):
        indices = [1, 2]

        # 测试是否捕获到意外的关键字参数异常
        msg = r"take\(\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            index.take(indices, foo=2)

        # 测试是否捕获到不支持 'out' 参数的异常
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            index.take(indices, out=indices)

        # 测试是否捕获到不支持 'mode' 参数的异常
        msg = "the 'mode' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            index.take(indices, mode="clip")

    def test_take(self, index):
        indexer = [4, 3, 0, 2]
        if len(index) < 5:
            pytest.skip("Test doesn't make sense since not enough elements")

        # 执行索引取值操作，比较结果是否与预期一致
        result = index.take(indexer)
        expected = index[indexer]
        assert result.equals(expected)

        if not isinstance(index, (DatetimeIndex, PeriodIndex, TimedeltaIndex)):
            # GH 10791
            # 如果索引不是时间日期索引、时期索引或时间增量索引，测试是否捕获到属性错误异常
            msg = r"'(.*Index)' object has no attribute 'freq'"
            with pytest.raises(AttributeError, match=msg):
                index.freq

    def test_take_indexer_type(self):
        # GH#42875
        integer_index = Index([0, 1, 2, 3])
        scalar_index = 1
        msg = "Expected indices to be array-like"
        with pytest.raises(TypeError, match=msg):
            integer_index.take(scalar_index)

    def test_take_minus1_without_fill(self, index):
        # 当不允许填充时，-1不会被视为NA
        if len(index) == 0:
            pytest.skip("Test doesn't make sense for empty index")

        # 执行索引取值操作，比较结果是否与预期一致
        result = index.take([0, 0, -1])
        expected = index.take([0, 0, len(index) - 1])
        tm.assert_index_equal(result, expected)


class TestContains:
    @pytest.mark.parametrize(
        "index,val",
        [
            ([0, 1, 2], 2),
            ([0, 1, "2"], "2"),
            ([0, 1, 2, np.inf, 4], 4),
            ([0, 1, 2, np.nan, 4], 4),
            ([0, 1, 2, np.inf], np.inf),
            ([0, 1, 2, np.nan], np.nan),
        ],
    )
    def test_index_contains(self, index, val):
        index = Index(index)
        # 测试索引对象是否包含特定的值
        assert val in index
    @pytest.mark.parametrize(
        "index,val",
        [  # 使用 pytest.mark.parametrize 装饰器定义参数化测试，参数为 index 和 val
            (Index([0, 1, 2]), "2"),  # 测试情况：Index 包含整数 0、1、2，val 为字符串 "2"
            (Index([0, 1, "2"]), 2),  # 测试情况：Index 包含整数 0、1 和字符串 "2"，val 为整数 2
            (Index([0, 1, 2, np.inf]), 4),  # 测试情况：Index 包含整数 0、1、2 和 np.inf，val 为整数 4
            (Index([0, 1, 2, np.nan]), 4),  # 测试情况：Index 包含整数 0、1、2 和 np.nan，val 为整数 4
            (Index([0, 1, 2, np.inf]), np.nan),  # 测试情况：Index 包含整数 0、1、2 和 np.inf，val 为 np.nan
            (Index([0, 1, 2, np.nan]), np.inf),  # 测试情况：Index 包含整数 0、1、2 和 np.nan，val 为 np.inf
            # 检查 np.inf 在 int64 Index 中不会引发 OverflowError，相关于 GH 16957
            (Index([0, 1, 2], dtype=np.int64), np.inf),  # 测试情况：dtype 为 np.int64 的 Index 包含整数 0、1、2，val 为 np.inf
            (Index([0, 1, 2], dtype=np.int64), np.nan),  # 测试情况：dtype 为 np.int64 的 Index 包含整数 0、1、2，val 为 np.nan
            (Index([0, 1, 2], dtype=np.uint64), np.inf),  # 测试情况：dtype 为 np.uint64 的 Index 包含整数 0、1、2，val 为 np.inf
            (Index([0, 1, 2], dtype=np.uint64), np.nan),  # 测试情况：dtype 为 np.uint64 的 Index 包含整数 0、1、2，val 为 np.nan
        ],
    )
    def test_index_not_contains(self, index, val):
        assert val not in index  # 断言：val 不在 index 中

    @pytest.mark.parametrize("val", [0, "2"])
    def test_mixed_index_contains(self, val):
        # GH#19860
        index = Index([0, 1, "2"])  # 创建包含整数 0、1 和字符串 "2" 的 Index
        assert val in index  # 断言：val 在 index 中

    @pytest.mark.parametrize("val", ["1", 2])
    def test_mixed_index_not_contains(self, index, val):
        # GH#19860
        index = Index([0, 1, "2"])  # 创建包含整数 0、1 和字符串 "2" 的 Index
        assert val not in index  # 断言：val 不在 index 中

    def test_contains_with_float_index(self, any_real_numpy_dtype):
        # GH#22085
        dtype = any_real_numpy_dtype
        data = [0, 1, 2, 3] if not is_float_dtype(dtype) else [0.1, 1.1, 2.2, 3.3]
        index = Index(data, dtype=dtype)  # 使用指定的数据和数据类型创建 Index

        if not is_float_dtype(index.dtype):
            assert 1.1 not in index  # 断言：1.1 不在 index 中
            assert 1.0 in index  # 断言：1.0 在 index 中
            assert 1 in index  # 断言：整数 1 在 index 中
        else:
            assert 1.1 in index  # 断言：1.1 在 index 中
            assert 1.0 not in index  # 断言：1.0 不在 index 中
            assert 1 not in index  # 断言：整数 1 不在 index 中

    def test_contains_requires_hashable_raises(self, index):
        if isinstance(index, MultiIndex):
            return  # 如果 index 是 MultiIndex 类型，则跳过测试

        msg = "unhashable type: 'list'"
        with pytest.raises(TypeError, match=msg):
            [] in index  # 断言：空列表不可哈希

        msg = "|".join(
            [
                r"unhashable type: 'dict'",
                r"must be real number, not dict",
                r"an integer is required",
                r"\{\}",
                r"pandas\._libs\.interval\.IntervalTree' is not iterable",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            {} in index._engine  # 断言：空字典或其他不可迭代对象在 index._engine 中引发 TypeError
class TestGetLoc:
    # 测试获取位置对于非可哈希对象的行为
    def test_get_loc_non_hashable(self, index):
        # 使用 pytest 检查是否会抛出 InvalidIndexError 异常，异常信息匹配 "[0, 1]"
        with pytest.raises(InvalidIndexError, match="[0, 1]"):
            index.get_loc([0, 1])

    # 测试获取位置对于非标量可哈希对象的行为
    def test_get_loc_non_scalar_hashable(self, index):
        # GH52877
        from enum import Enum
        
        class E(Enum):
            X1 = "x1"
        
        assert not is_scalar(E.X1)
        
        exc = KeyError
        msg = "<E.X1: 'x1'>"
        if isinstance(
            index,
            (
                DatetimeIndex,
                TimedeltaIndex,
                PeriodIndex,
                IntervalIndex,
            ),
        ):
            # 对于特定的索引类型，将异常类型设为 InvalidIndexError，消息设为 "E.X1"
            exc = InvalidIndexError
            msg = "E.X1"
        with pytest.raises(exc, match=msg):
            index.get_loc(E.X1)

    # 测试获取位置对于生成器对象的行为
    def test_get_loc_generator(self, index):
        exc = KeyError
        if isinstance(
            index,
            (
                DatetimeIndex,
                TimedeltaIndex,
                PeriodIndex,
                IntervalIndex,
                MultiIndex,
            ),
        ):
            # 对于特定的索引类型，将异常类型设为 InvalidIndexError
            exc = InvalidIndexError
        with pytest.raises(exc, match="generator object"):
            # MultiIndex 特别检查生成器对象；其他索引类型检查标量对象
            index.get_loc(x for x in range(5))

    # 测试获取位置对于掩码、重复值、缺失值的行为
    def test_get_loc_masked_duplicated_na(self):
        # GH#48411
        idx = Index([1, 2, NA, NA], dtype="Int64")
        result = idx.get_loc(NA)
        expected = np.array([False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)


class TestGetIndexer:
    # 测试获取索引器基本行为
    def test_get_indexer_base(self, index):
        if index._index_as_unique:
            # 如果索引标记为唯一，则期望索引器是一个整数范围数组
            expected = np.arange(index.size, dtype=np.intp)
            actual = index.get_indexer(index)
            tm.assert_numpy_array_equal(expected, actual)
        else:
            # 否则，期望抛出 InvalidIndexError 异常，异常消息为 "Reindexing only valid with uniquely valued Index objects"
            msg = "Reindexing only valid with uniquely valued Index objects"
            with pytest.raises(InvalidIndexError, match=msg):
                index.get_indexer(index)

        # 期望抛出 ValueError 异常，异常消息为 "Invalid fill method"
        with pytest.raises(ValueError, match="Invalid fill method"):
            index.get_indexer(index, method="invalid")

    # 测试获取索引器一致性行为
    def test_get_indexer_consistency(self, index):
        # See GH#16819

        if index._index_as_unique:
            # 如果索引标记为唯一，则获取索引器应返回一个整数类型的 NumPy 数组
            indexer = index.get_indexer(index[0:2])
            assert isinstance(indexer, np.ndarray)
            assert indexer.dtype == np.intp
        else:
            # 否则，期望抛出 InvalidIndexError 异常，异常消息为 "Reindexing only valid with uniquely valued Index objects"
            msg = "Reindexing only valid with uniquely valued Index objects"
            with pytest.raises(InvalidIndexError, match=msg):
                index.get_indexer(index[0:2])

        # 获取非唯一值索引器，期望返回一个整数类型的 NumPy 数组
        indexer, _ = index.get_indexer_non_unique(index[0:2])
        assert isinstance(indexer, np.ndarray)
        assert indexer.dtype == np.intp
    # 定义一个测试函数，用于测试处理索引时的特定情况
    def test_get_indexer_masked_duplicated_na(self):
        # 该测试用例关注GitHub上的问题编号GH#48411

        # 创建一个索引对象，包含整数和缺失值
        idx = Index([1, 2, NA, NA], dtype="Int64")
        
        # 对创建的索引对象调用get_indexer_for方法，使用另一个索引对象作为参数
        result = idx.get_indexer_for(Index([1, NA], dtype="Int64"))
        
        # 预期的结果是一个NumPy数组，用于标识给定索引在原索引中的位置
        expected = np.array([0, 2, 3], dtype=result.dtype)
        
        # 使用测试工具方法验证result与expected是否相等
        tm.assert_numpy_array_equal(result, expected)
class TestConvertSliceIndexer:
    def test_convert_almost_null_slice(self, index):
        # 创建一个带有非整数步长的切片对象
        key = slice(None, None, "foo")

        # 如果 index 是 IntervalIndex 类型，测试对 IntervalIndex 的不支持
        if isinstance(index, IntervalIndex):
            msg = "label-based slicing with step!=1 is not supported for IntervalIndex"
            # 确保抛出 ValueError 异常，并匹配特定的错误消息
            with pytest.raises(ValueError, match=msg):
                index._convert_slice_indexer(key, "loc")
        else:
            msg = "'>=' not supported between instances of 'str' and 'int'"
            # 确保抛出 TypeError 异常，并匹配特定的错误消息
            with pytest.raises(TypeError, match=msg):
                index._convert_slice_indexer(key, "loc")


class TestPutmask:
    def test_putmask_with_wrong_mask(self, index):
        # 跳过测试，如果 index 是空的，因为对空索引进行此测试没有意义
        if not len(index):
            pytest.skip("Test doesn't make sense for empty index")

        # 获取索引中的第一个元素作为填充值
        fill = index[0]

        # 确保对于过长的掩码数组，putmask 函数会引发 ValueError 异常
        msg = "putmask: mask and data must be the same size"
        with pytest.raises(ValueError, match=msg):
            index.putmask(np.ones(len(index) + 1, np.bool_), fill)

        # 确保对于过短的掩码数组，putmask 函数会引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            index.putmask(np.ones(len(index) - 1, np.bool_), fill)

        # 确保对于非布尔类型的掩码，putmask 函数会引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            index.putmask("foo", fill)


@pytest.mark.parametrize("idx", [[1, 2, 3], [0.1, 0.2, 0.3], ["a", "b", "c"]])
def test_getitem_deprecated_float(idx):
    # https://github.com/pandas-dev/pandas/issues/34191
    # 创建一个 Index 对象用于测试
    idx = Index(idx)
    # 确保索引使用浮点数会引发 IndexError 异常
    msg = "Indexing with a float is no longer supported"
    with pytest.raises(IndexError, match=msg):
        idx[1.0]


@pytest.mark.parametrize(
    "idx,target,expected",
    [
        ([np.nan, "var1", np.nan], [np.nan], np.array([0, 2], dtype=np.intp)),
        (
            [np.nan, "var1", np.nan],
            [np.nan, "var1"],
            np.array([0, 2, 1], dtype=np.intp),
        ),
        (
            np.array([np.nan, "var1", np.nan], dtype=object),
            [np.nan],
            np.array([0, 2], dtype=np.intp),
        ),
        (
            DatetimeIndex(["2020-08-05", NaT, NaT]),
            [NaT],
            np.array([1, 2], dtype=np.intp),
        ),
        (["a", "b", "a", np.nan], [np.nan], np.array([3], dtype=np.intp)),
        (
            np.array(["b", np.nan, float("NaN"), "b"], dtype=object),
            Index([np.nan], dtype=object),
            np.array([1, 2], dtype=np.intp),
        ),
    ],
)
def test_get_indexer_non_unique_multiple_nans(idx, target, expected):
    # GH 35392
    # 创建一个 Index 对象用于测试
    axis = Index(idx)
    # 获取目标索引数组在当前索引中的索引位置
    actual = axis.get_indexer_for(target)
    # 断言实际结果与预期结果相等
    tm.assert_numpy_array_equal(actual, expected)


def test_get_indexer_non_unique_nans_in_object_dtype_target(nulls_fixture):
    # 创建一个 Index 对象用于测试
    idx = Index([1.0, 2.0])
    # 创建一个 Index 对象用于测试，其中包含对象类型的空值
    target = Index([1, nulls_fixture], dtype="object")

    # 获取在 idx 中非唯一值的索引及缺失值的索引
    result_idx, result_missing = idx.get_indexer_non_unique(target)
    # 断言结果与预期的 NumPy 数组相等
    tm.assert_numpy_array_equal(result_idx, np.array([0, -1], dtype=np.intp))
    tm.assert_numpy_array_equal(result_missing, np.array([1], dtype=np.intp))
```