# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_drop.py`

```
import pytest  # 导入 pytest 库

from pandas import (  # 从 pandas 库中导入 Index 和 Series 类
    Index,
    Series,
)
import pandas._testing as tm  # 导入 pandas 测试模块，命名为 tm
from pandas.api.types import is_bool_dtype  # 导入 pandas 的类型检查函数 is_bool_dtype


@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，用于参数化测试用例
    "data, index, drop_labels, axis, expected_data, expected_index",  # 参数化的参数名称列表
    [
        # Unique Index
        ([1, 2], ["one", "two"], ["two"], 0, [1], ["one"]),  # 参数化测试用例1
        ([1, 2], ["one", "two"], ["two"], "rows", [1], ["one"]),  # 参数化测试用例2
        ([1, 1, 2], ["one", "two", "one"], ["two"], 0, [1, 2], ["one", "one"]),  # 参数化测试用例3
        # GH 5248 Non-Unique Index
        ([1, 1, 2], ["one", "two", "one"], "two", 0, [1, 2], ["one", "one"]),  # 参数化测试用例4
        ([1, 1, 2], ["one", "two", "one"], ["one"], 0, [1], ["two"]),  # 参数化测试用例5
        ([1, 1, 2], ["one", "two", "one"], "one", 0, [1], ["two"]),  # 参数化测试用例6
    ],
)
def test_drop_unique_and_non_unique_index(
    data, index, axis, drop_labels, expected_data, expected_index
):
    ser = Series(data=data, index=index)  # 创建 Series 对象
    result = ser.drop(drop_labels, axis=axis)  # 调用 drop 方法删除指定标签
    expected = Series(data=expected_data, index=expected_index)  # 创建期望的 Series 对象
    tm.assert_series_equal(result, expected)  # 使用测试模块的函数比较结果和期望


@pytest.mark.parametrize(  # 参数化装饰器，用于参数化异常测试用例
    "drop_labels, axis, error_type, error_desc",  # 参数化的参数名称列表
    [
        # single string/tuple-like
        ("bc", 0, KeyError, "not found in axis"),  # 参数化异常测试用例1
        # bad axis
        (("a",), 0, KeyError, "not found in axis"),  # 参数化异常测试用例2
        ("one", "columns", ValueError, "No axis named columns"),  # 参数化异常测试用例3
    ],
)
def test_drop_exception_raised(drop_labels, axis, error_type, error_desc):
    ser = Series(range(3), index=list("abc"))  # 创建带索引的 Series 对象
    with pytest.raises(error_type, match=error_desc):  # 使用 pytest 断言异常抛出
        ser.drop(drop_labels, axis=axis)  # 调用 drop 方法并指定轴


def test_drop_with_ignore_errors():
    # errors='ignore'
    ser = Series(range(3), index=list("abc"))  # 创建带索引的 Series 对象
    result = ser.drop("bc", errors="ignore")  # 调用 drop 方法删除指定标签并忽略错误
    tm.assert_series_equal(result, ser)  # 比较结果和原始 Series 对象
    result = ser.drop(["a", "d"], errors="ignore")  # 调用 drop 方法删除多个标签并忽略错误
    expected = ser.iloc[1:]  # 期望的结果为去除标签后的 Series 对象
    tm.assert_series_equal(result, expected)  # 比较结果和期望的 Series 对象

    # GH 8522
    ser = Series([2, 3], index=[True, False])  # 创建带布尔索引的 Series 对象
    assert is_bool_dtype(ser.index)  # 断言索引是布尔类型
    assert ser.index.dtype == bool  # 断言索引的数据类型是布尔型
    result = ser.drop(True)  # 调用 drop 方法删除布尔索引为 True 的元素
    expected = Series([3], index=[False])  # 期望的结果为删除指定元素后的 Series 对象
    tm.assert_series_equal(result, expected)  # 比较结果和期望的 Series 对象


@pytest.mark.parametrize("index", [[1, 2, 3], [1, 1, 3]])  # 参数化测试用例的索引
@pytest.mark.parametrize("drop_labels", [[], [1], [3]])  # 参数化测试用例的删除标签
def test_drop_empty_list(index, drop_labels):
    # GH 21494
    expected_index = [i for i in index if i not in drop_labels]  # 创建期望的索引列表
    series = Series(index=index, dtype=object).drop(drop_labels)  # 创建 Series 对象并调用 drop 方法删除指定标签
    expected = Series(index=expected_index, dtype=object)  # 创建期望的 Series 对象
    tm.assert_series_equal(series, expected)  # 比较结果和期望的 Series 对象


@pytest.mark.parametrize(  # 参数化装饰器，用于参数化非空列表删除测试用例
    "data, index, drop_labels",  # 参数化的参数名称列表
    [
        (None, [1, 2, 3], [1, 4]),  # 参数化非空列表删除测试用例1
        (None, [1, 2, 2], [1, 4]),  # 参数化非空列表删除测试用例2
        ([2, 3], [0, 1], [False, True]),  # 参数化非空列表删除测试用例3
    ],
)
def test_drop_non_empty_list(data, index, drop_labels):
    # GH 21494 and GH 16877
    dtype = object if data is None else None  # 根据 data 是否为 None 决定 dtype 类型
    ser = Series(data=data, index=index, dtype=dtype)  # 创建指定数据和索引的 Series 对象
    with pytest.raises(KeyError, match="not found in axis"):  # 使用 pytest 断言预期的 KeyError 异常
        ser.drop(drop_labels)  # 调用 drop 方法删除指定标签
# 定义一个测试函数，用于测试在特定条件下的索引删除操作
def test_drop_index_ea_dtype(any_numeric_ea_dtype):
    # GH#45860: 标识GitHub上的问题编号
    # 创建一个 Series 对象，所有元素都是 100，指定索引为 [1, 2, 2]，索引类型为给定的数值类型
    df = Series(100, index=Index([1, 2, 2], dtype=any_numeric_ea_dtype))
    # 从 df 的索引中获取第二个索引，并创建一个新的索引对象
    idx = Index([df.index[1]])
    # 删除 df 中指定的索引 idx 对应的行，并得到结果
    result = df.drop(idx)
    # 创建一个预期的 Series 对象，所有元素都是 100，指定索引为 [1]，索引类型为给定的数值类型
    expected = Series(100, index=Index([1], dtype=any_numeric_ea_dtype))
    # 断言函数，比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
```