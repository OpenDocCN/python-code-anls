# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_compare.py`

```
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


# 定义测试函数，用于比较不同轴上的 Pandas Series 对象
def test_compare_axis(axis):
    # 创建两个 Pandas Series 对象
    s1 = pd.Series(["a", "b", "c"])
    s2 = pd.Series(["x", "b", "z"])

    # 使用 compare 方法比较两个 Series 对象，并指定对齐轴
    result = s1.compare(s2, align_axis=axis)

    # 根据轴的不同，生成预期的比较结果
    if axis in (1, "columns"):
        # 如果是按列比较，则预期结果是一个 DataFrame
        indices = pd.Index([0, 2])
        columns = pd.Index(["self", "other"])
        expected = pd.DataFrame(
            [["a", "x"], ["c", "z"]], index=indices, columns=columns
        )
        # 使用测试模块中的函数验证结果是否与预期一致
        tm.assert_frame_equal(result, expected)
    else:
        # 否则按行比较，预期结果是一个带多级索引的 Series
        indices = pd.MultiIndex.from_product([[0, 2], ["self", "other"]])
        expected = pd.Series(["a", "x", "c", "z"], index=indices)
        tm.assert_series_equal(result, expected)


# 使用参数化装饰器，定义多种格式的测试用例
@pytest.mark.parametrize(
    "keep_shape, keep_equal",
    [
        (True, False),
        (False, True),
        (True, True),
        # 已经在 test_compare_axis 中覆盖了 False, False 的情况
    ],
)
def test_compare_various_formats(keep_shape, keep_equal):
    # 创建两个 Pandas Series 对象
    s1 = pd.Series(["a", "b", "c"])
    s2 = pd.Series(["x", "b", "z"])

    # 使用 compare 方法比较两个 Series 对象，并指定 keep_shape 和 keep_equal 参数
    result = s1.compare(s2, keep_shape=keep_shape, keep_equal=keep_equal)

    # 根据参数值生成预期的比较结果
    if keep_shape:
        indices = pd.Index([0, 1, 2])
        columns = pd.Index(["self", "other"])
        if keep_equal:
            expected = pd.DataFrame(
                [["a", "x"], ["b", "b"], ["c", "z"]], index=indices, columns=columns
            )
        else:
            expected = pd.DataFrame(
                [["a", "x"], [np.nan, np.nan], ["c", "z"]],
                index=indices,
                columns=columns,
            )
    else:
        indices = pd.Index([0, 2])
        columns = pd.Index(["self", "other"])
        expected = pd.DataFrame(
            [["a", "x"], ["c", "z"]], index=indices, columns=columns
        )
    # 使用测试模块中的函数验证结果是否与预期一致
    tm.assert_frame_equal(result, expected)


# 测试比较带有相等的 NaN 值的情况
def test_compare_with_equal_nulls():
    # 创建两个 Pandas Series 对象，其中包含 NaN 值
    s1 = pd.Series(["a", "b", np.nan])
    s2 = pd.Series(["x", "b", np.nan])

    # 使用 compare 方法比较两个 Series 对象
    result = s1.compare(s2)
    expected = pd.DataFrame([["a", "x"]], columns=["self", "other"])
    # 使用测试模块中的函数验证结果是否与预期一致
    tm.assert_frame_equal(result, expected)


# 测试比较带有不相等的 NaN 值的情况
def test_compare_with_non_equal_nulls():
    # 创建两个 Pandas Series 对象，其中一个包含 NaN 值
    s1 = pd.Series(["a", "b", "c"])
    s2 = pd.Series(["x", "b", np.nan])

    # 使用 compare 方法比较两个 Series 对象，并指定对齐轴
    result = s1.compare(s2, align_axis=0)

    # 生成预期的比较结果，这里是一个带多级索引的 Series
    indices = pd.MultiIndex.from_product([[0, 2], ["self", "other"]])
    expected = pd.Series(["a", "x", "c", np.nan], index=indices)
    # 使用测试模块中的函数验证结果是否与预期一致
    tm.assert_series_equal(result, expected)


# 测试比较带有多级索引的 Series 对象
def test_compare_multi_index():
    # 创建带有多级索引的 Pandas Series 对象
    index = pd.MultiIndex.from_arrays([[0, 0, 1], [0, 1, 2]])
    s1 = pd.Series(["a", "b", "c"], index=index)
    s2 = pd.Series(["x", "b", "z"], index=index)

    # 使用 compare 方法比较两个 Series 对象，并指定对齐轴
    result = s1.compare(s2, align_axis=0)

    # 生成预期的比较结果，这里是一个带多级索引的 Series
    indices = pd.MultiIndex.from_arrays(
        [[0, 0, 1, 1], [0, 0, 2, 2], ["self", "other", "self", "other"]]
    )
    # 注意：此处没有使用 tm.assert_series_equal，因为 expected 是 DataFrame 而不是 Series
    # 创建预期的 Pandas Series 对象，包含指定的值和索引
    expected = pd.Series(["a", "x", "c", "z"], index=indices)
    # 使用 Pandas 测试工具（`tm.assert_series_equal`）比较结果 Series (`result`) 和预期的 Series (`expected`)
    tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试当Series对象的索引不同时的比较行为
def test_compare_different_indices():
    # 错误消息，用于验证是否会引发 ValueError
    msg = "Can only compare identically-labeled Series objects"
    # 创建两个Series对象，分别设置不同的索引
    ser1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
    ser2 = pd.Series([1, 2, 3], index=["a", "b", "d"])
    # 使用 pytest 来验证是否会引发 ValueError，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        ser1.compare(ser2)


# 定义一个测试函数，用于测试当Series对象的长度不同时的比较行为
def test_compare_different_lengths():
    # 错误消息，用于验证是否会引发 ValueError
    msg = "Can only compare identically-labeled Series objects"
    # 创建两个Series对象，分别设置不同的长度
    ser1 = pd.Series([1, 2, 3])
    ser2 = pd.Series([1, 2, 3, 4])
    # 使用 pytest 来验证是否会引发 ValueError，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        ser1.compare(ser2)


# 定义一个测试函数，用于测试比较包含 datetime64 和 string 类型的 DataFrame 的行为
def test_compare_datetime64_and_string():
    # GitHub 上的问题链接，用于描述此测试的背景
    # Issue https://github.com/pandas-dev/pandas/issues/45506
    # 当比较 datetime64 和 string 类型的数据时，捕获 OverflowError
    # 创建包含不同数据类型的字典列表
    data = [
        {"a": "2015-07-01", "b": "08335394550"},
        {"a": "2015-07-02", "b": "+49 (0) 0345 300033"},
        {"a": "2015-07-03", "b": "+49(0)2598 04457"},
        {"a": "2015-07-04", "b": "0741470003"},
        {"a": "2015-07-05", "b": "04181 83668"},
    ]
    # 指定每列的数据类型
    dtypes = {"a": "datetime64[ns]", "b": "string"}
    # 创建 DataFrame，并将其类型转换为指定的数据类型
    df = pd.DataFrame(data=data).astype(dtypes)

    # 进行 Series 对比操作，比较两列的相等性和不相等性
    result_eq1 = df["a"].eq(df["b"])   # 使用 .eq() 方法比较相等性
    result_eq2 = df["a"] == df["b"]    # 使用 == 运算符比较相等性
    result_neq = df["a"] != df["b"]    # 使用 != 运算符比较不相等性

    # 期望的结果：对于 .eq() 和 == 操作，都应该返回 False
    expected_eq = pd.Series([False] * 5)
    # 期望的结果：对于 != 操作，都应该返回 True
    expected_neq = pd.Series([True] * 5)

    # 使用 pandas 的 tm 模块来断言 Series 是否相等
    tm.assert_series_equal(result_eq1, expected_eq)
    tm.assert_series_equal(result_eq2, expected_eq)
    tm.assert_series_equal(result_neq, expected_neq)
```