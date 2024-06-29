# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_take.py`

```
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

# 定义一个测试函数，用于测试 idx 对象的 take 方法
def test_take(idx):
    # 创建索引列表
    indexer = [4, 3, 0, 2]
    # 调用 take 方法，获取索引列表对应的结果
    result = idx.take(indexer)
    # 使用普通索引方式获取期望结果
    expected = idx[indexer]
    # 断言结果与期望是否相等
    assert result.equals(expected)

    # 测试 GH 10791，验证在 pytest 中是否正确捕获 AttributeError 异常
    msg = "'MultiIndex' object has no attribute 'freq'"
    with pytest.raises(AttributeError, match=msg):
        idx.freq


# 定义测试函数，测试在不同参数下 idx 对象的 take 方法是否能够正确抛出异常
def test_take_invalid_kwargs(idx):
    indices = [1, 2]

    # 测试是否能捕获到 TypeError 异常，并检查异常消息是否匹配
    msg = r"take\(\) got an unexpected keyword argument 'foo'"
    with pytest.raises(TypeError, match=msg):
        idx.take(indices, foo=2)

    # 测试是否能捕获到 ValueError 异常，并检查异常消息是否匹配
    msg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=msg):
        idx.take(indices, out=indices)

    # 测试是否能捕获到 ValueError 异常，并检查异常消息是否匹配
    msg = "the 'mode' parameter is not supported"
    with pytest.raises(ValueError, match=msg):
        idx.take(indices, mode="clip")


# 定义测试函数，测试在不同参数下 MultiIndex 对象的 take 方法是否正确工作
def test_take_fill_value():
    # 创建一个 MultiIndex 对象
    vals = [["A", "B"], [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-02")]]
    idx = pd.MultiIndex.from_product(vals, names=["str", "dt"])

    # 测试普通情况下的 take 方法结果是否符合预期
    result = idx.take(np.array([1, 0, -1]))
    exp_vals = [
        ("A", pd.Timestamp("2011-01-02")),
        ("A", pd.Timestamp("2011-01-01")),
        ("B", pd.Timestamp("2011-01-02")),
    ]
    expected = pd.MultiIndex.from_tuples(exp_vals, names=["str", "dt"])
    tm.assert_index_equal(result, expected)

    # 测试带有 fill_value 参数的 take 方法结果是否符合预期
    result = idx.take(np.array([1, 0, -1]), fill_value=True)
    exp_vals = [
        ("A", pd.Timestamp("2011-01-02")),
        ("A", pd.Timestamp("2011-01-01")),
        (np.nan, pd.NaT),
    ]
    expected = pd.MultiIndex.from_tuples(exp_vals, names=["str", "dt"])
    tm.assert_index_equal(result, expected)

    # 测试带有 allow_fill=False 参数的 take 方法结果是否符合预期
    result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
    exp_vals = [
        ("A", pd.Timestamp("2011-01-02")),
        ("A", pd.Timestamp("2011-01-01")),
        ("B", pd.Timestamp("2011-01-02")),
    ]
    expected = pd.MultiIndex.from_tuples(exp_vals, names=["str", "dt"])
    tm.assert_index_equal(result, expected)

    # 测试带有 fill_value=True 但索引值超出范围的情况，是否能捕获到 ValueError 异常
    msg = "When allow_fill=True and fill_value is not None, all indices must be >= -1"
    with pytest.raises(ValueError, match=msg):
        idx.take(np.array([1, 0, -2]), fill_value=True)
    with pytest.raises(ValueError, match=msg):
        idx.take(np.array([1, 0, -5]), fill_value=True)

    # 测试带有索引超出范围的情况，是否能捕获到 IndexError 异常
    msg = "index -5 is out of bounds for( axis 0 with)? size 4"
    with pytest.raises(IndexError, match=msg):
        idx.take(np.array([1, -5]))
```