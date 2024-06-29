# `D:\src\scipysrc\pandas\pandas\tests\indexing\test_check_indexer.py`

```
# 导入必要的库：numpy（命名为np）、pytest、pandas（命名为pd）、pandas测试工具（命名为tm）、pandas的索引器检查函数（check_array_indexer）
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.indexers import check_array_indexer

# 使用pytest的@parametrize装饰器定义参数化测试函数，用于测试check_array_indexer函数的不同输入情况

@pytest.mark.parametrize(
    "indexer, expected",
    [
        # integer
        ([1, 2], np.array([1, 2], dtype=np.intp)),  # 传入整数列表作为索引器，期望返回整数类型的numpy数组
        (np.array([1, 2], dtype="int64"), np.array([1, 2], dtype=np.intp)),  # 传入int64类型的numpy数组作为索引器，期望返回整数类型的numpy数组
        (pd.array([1, 2], dtype="Int32"), np.array([1, 2], dtype=np.intp)),  # 传入Int32类型的pandas数组作为索引器，期望返回整数类型的numpy数组
        (pd.Index([1, 2]), np.array([1, 2], dtype=np.intp)),  # 传入整数列表作为索引器，期望返回整数类型的numpy数组
        # boolean
        ([True, False, True], np.array([True, False, True], dtype=np.bool_)),  # 传入布尔值列表作为索引器，期望返回布尔类型的numpy数组
        (np.array([True, False, True]), np.array([True, False, True], dtype=np.bool_)),  # 传入布尔类型的numpy数组作为索引器，期望返回布尔类型的numpy数组
        (
            pd.array([True, False, True], dtype="boolean"),
            np.array([True, False, True], dtype=np.bool_),
        ),  # 传入布尔类型的pandas数组作为索引器，期望返回布尔类型的numpy数组
        # other
        ([], np.array([], dtype=np.intp)),  # 传入空列表作为索引器，期望返回空的整数类型的numpy数组
    ],
)
def test_valid_input(indexer, expected):
    arr = np.array([1, 2, 3])  # 创建一个numpy数组
    result = check_array_indexer(arr, indexer)  # 调用check_array_indexer函数，传入数组和索引器
    tm.assert_numpy_array_equal(result, expected)  # 使用测试工具tm检查结果数组与期望数组是否相等


@pytest.mark.parametrize(
    "indexer", [[True, False, None], pd.array([True, False, None], dtype="boolean")]
)
def test_boolean_na_returns_indexer(indexer):
    # https://github.com/pandas-dev/pandas/issues/31503
    arr = np.array([1, 2, 3])  # 创建一个numpy数组

    result = check_array_indexer(arr, indexer)  # 调用check_array_indexer函数，传入数组和索引器
    expected = np.array([True, False, False], dtype=bool)  # 创建期望的布尔类型的numpy数组

    tm.assert_numpy_array_equal(result, expected)  # 使用测试工具tm检查结果数组与期望数组是否相等


@pytest.mark.parametrize(
    "indexer",
    [
        [True, False],
        pd.array([True, False], dtype="boolean"),
        np.array([True, False], dtype=np.bool_),
    ],
)
def test_bool_raise_length(indexer):
    arr = np.array([1, 2, 3])  # 创建一个numpy数组

    msg = "Boolean index has wrong length"  # 出错时的错误信息
    with pytest.raises(IndexError, match=msg):  # 使用pytest检查是否抛出预期的IndexError异常并匹配错误信息
        check_array_indexer(arr, indexer)  # 调用check_array_indexer函数，传入数组和索引器


@pytest.mark.parametrize(
    "indexer", [[0, 1, None], pd.array([0, 1, pd.NA], dtype="Int64")]
)
def test_int_raise_missing_values(indexer):
    arr = np.array([1, 2, 3])  # 创建一个numpy数组

    msg = "Cannot index with an integer indexer containing NA values"  # 出错时的错误信息
    with pytest.raises(ValueError, match=msg):  # 使用pytest检查是否抛出预期的ValueError异常并匹配错误信息
        check_array_indexer(arr, indexer)  # 调用check_array_indexer函数，传入数组和索引器


@pytest.mark.parametrize(
    "indexer",
    [
        [0.0, 1.0],
        np.array([1.0, 2.0], dtype="float64"),
        np.array([True, False], dtype=object),
        pd.Index([True, False], dtype=object),
    ],
)
def test_raise_invalid_array_dtypes(indexer):
    arr = np.array([1, 2, 3])  # 创建一个numpy数组

    msg = "arrays used as indices must be of integer or boolean type"  # 出错时的错误信息
    with pytest.raises(IndexError, match=msg):  # 使用pytest检查是否抛出预期的IndexError异常并匹配错误信息
        check_array_indexer(arr, indexer)  # 调用check_array_indexer函数，传入数组和索引器


def test_raise_nullable_string_dtype(nullable_string_dtype):
    indexer = pd.array(["a", "b"], dtype=nullable_string_dtype)  # 创建一个带有可空字符串数据类型的pandas数组
    arr = np.array([1, 2, 3])  # 创建一个numpy数组

    msg = "arrays used as indices must be of integer or boolean type"  # 出错时的错误信息
    with pytest.raises(IndexError, match=msg):  # 使用pytest检查是否抛出预期的IndexError异常并匹配错误信息
        check_array_indexer(arr, indexer)  # 调用check_array_indexer函数，传入数组和索引器
# 使用 pytest.mark.parametrize 装饰器为 test_pass_through_non_array_likes 函数参数 indexer 添加参数化测试
@pytest.mark.parametrize("indexer", [None, Ellipsis, slice(0, 3), (None,)])
# 定义测试函数 test_pass_through_non_array_likes，接收参数 indexer
def test_pass_through_non_array_likes(indexer):
    # 创建一个包含元素 1, 2, 3 的 NumPy 数组
    arr = np.array([1, 2, 3])

    # 调用 check_array_indexer 函数，传入数组 arr 和 indexer 参数，返回结果赋值给 result
    result = check_array_indexer(arr, indexer)
    # 断言 result 的值等于 indexer
    assert result == indexer
```