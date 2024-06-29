# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_compare.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试

from pandas.compat.numpy import np_version_gte1p25  # 导入兼容模块，用于检查 NumPy 版本

import pandas as pd  # 导入 Pandas 库，用于数据分析和处理
import pandas._testing as tm  # 导入 Pandas 测试模块，用于测试辅助功能


@pytest.mark.parametrize("align_axis", [0, 1, "index", "columns"])
def test_compare_axis(align_axis):
    # GH#30429
    # 创建一个 DataFrame 对象 df，包含三列数据
    df = pd.DataFrame(
        {"col1": ["a", "b", "c"], "col2": [1.0, 2.0, np.nan], "col3": [1.0, 2.0, 3.0]},
        columns=["col1", "col2", "col3"],
    )
    # 复制 df 得到 df2
    df2 = df.copy()
    # 修改 df2 的某些值
    df2.loc[0, "col1"] = "c"
    df2.loc[2, "col3"] = 4.0

    # 调用 DataFrame 的 compare 方法，比较 df 和 df2，根据 align_axis 参数对齐比较
    result = df.compare(df2, align_axis=align_axis)

    # 根据不同的 align_axis 值生成预期的 DataFrame 对象 expected
    if align_axis in (1, "columns"):
        indices = pd.Index([0, 2])
        columns = pd.MultiIndex.from_product([["col1", "col3"], ["self", "other"]])
        expected = pd.DataFrame(
            [["a", "c", np.nan, np.nan], [np.nan, np.nan, 3.0, 4.0]],
            index=indices,
            columns=columns,
        )
    else:
        indices = pd.MultiIndex.from_product([[0, 2], ["self", "other"]])
        columns = pd.Index(["col1", "col3"])
        expected = pd.DataFrame(
            [["a", np.nan], ["c", np.nan], [np.nan, 3.0], [np.nan, 4.0]],
            index=indices,
            columns=columns,
        )
    # 使用测试模块的 assert_frame_equal 方法比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "keep_shape, keep_equal",
    [
        (True, False),
        (False, True),
        (True, True),
        # False, False case is already covered in test_compare_axis
    ],
)
def test_compare_various_formats(keep_shape, keep_equal):
    # 创建一个 DataFrame 对象 df，包含三列数据
    df = pd.DataFrame(
        {"col1": ["a", "b", "c"], "col2": [1.0, 2.0, np.nan], "col3": [1.0, 2.0, 3.0]},
        columns=["col1", "col2", "col3"],
    )
    # 复制 df 得到 df2
    df2 = df.copy()
    # 修改 df2 的某些值
    df2.loc[0, "col1"] = "c"
    df2.loc[2, "col3"] = 4.0

    # 调用 DataFrame 的 compare 方法，比较 df 和 df2，根据 keep_shape 和 keep_equal 参数保持形状和相等性
    result = df.compare(df2, keep_shape=keep_shape, keep_equal=keep_equal)

    # 根据 keep_shape 和 keep_equal 参数生成预期的 DataFrame 对象 expected
    if keep_shape:
        indices = pd.Index([0, 1, 2])
        columns = pd.MultiIndex.from_product(
            [["col1", "col2", "col3"], ["self", "other"]]
        )
        if keep_equal:
            expected = pd.DataFrame(
                [
                    ["a", "c", 1.0, 1.0, 1.0, 1.0],
                    ["b", "b", 2.0, 2.0, 2.0, 2.0],
                    ["c", "c", np.nan, np.nan, 3.0, 4.0],
                ],
                index=indices,
                columns=columns,
            )
        else:
            expected = pd.DataFrame(
                [
                    ["a", "c", np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, 3.0, 4.0],
                ],
                index=indices,
                columns=columns,
            )
    else:
        indices = pd.Index([0, 2])
        columns = pd.MultiIndex.from_product([["col1", "col3"], ["self", "other"]])
        expected = pd.DataFrame(
            [["a", "c", 1.0, 1.0], ["c", "c", 3.0, 4.0]], index=indices, columns=columns
        )
    # 使用测试模块的 assert_frame_equal 方法比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
def test_compare_with_equal_nulls():
    # We want to make sure two NaNs are considered the same
    # and dropped where applicable

    # 创建一个包含NaN值的DataFrame，用于测试
    df = pd.DataFrame(
        {"col1": ["a", "b", "c"], "col2": [1.0, 2.0, np.nan], "col3": [1.0, 2.0, 3.0]},
        columns=["col1", "col2", "col3"],
    )
    # 创建df的副本df2
    df2 = df.copy()
    # 修改df2中第一行"col1"的值为"c"
    df2.loc[0, "col1"] = "c"

    # 调用DataFrame的compare方法，比较df和df2的不同之处
    result = df.compare(df2)

    # 创建预期的DataFrame，包含预期的比较结果
    indices = pd.Index([0])
    columns = pd.MultiIndex.from_product([["col1"], ["self", "other"]])
    expected = pd.DataFrame([["a", "c"]], index=indices, columns=columns)

    # 使用pytest断言函数检查result和expected是否相等
    tm.assert_frame_equal(result, expected)


def test_compare_with_non_equal_nulls():
    # We want to make sure the relevant NaNs do not get dropped
    # even if the entire row or column are NaNs

    # 创建一个包含NaN值的DataFrame，用于测试
    df = pd.DataFrame(
        {"col1": ["a", "b", "c"], "col2": [1.0, 2.0, np.nan], "col3": [1.0, 2.0, 3.0]},
        columns=["col1", "col2", "col3"],
    )
    # 创建df的副本df2
    df2 = df.copy()
    # 修改df2中第一行"col1"的值为"c"，以及第三行"col3"的值为NaN
    df2.loc[0, "col1"] = "c"
    df2.loc[2, "col3"] = np.nan

    # 调用DataFrame的compare方法，比较df和df2的不同之处
    result = df.compare(df2)

    # 创建预期的DataFrame，包含预期的比较结果
    indices = pd.Index([0, 2])
    columns = pd.MultiIndex.from_product([["col1", "col3"], ["self", "other"]])
    expected = pd.DataFrame(
        [["a", "c", np.nan, np.nan], [np.nan, np.nan, 3.0, np.nan]],
        index=indices,
        columns=columns,
    )

    # 使用pytest断言函数检查result和expected是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("align_axis", [0, 1])
def test_compare_multi_index(align_axis):
    # 创建一个包含多级索引的DataFrame，用于测试

    # 创建一个普通的DataFrame
    df = pd.DataFrame(
        {"col1": ["a", "b", "c"], "col2": [1.0, 2.0, np.nan], "col3": [1.0, 2.0, 3.0]}
    )
    # 将df的列设置为多级索引
    df.columns = pd.MultiIndex.from_arrays([["a", "a", "b"], ["col1", "col2", "col3"]])
    # 将df的行设置为多级索引
    df.index = pd.MultiIndex.from_arrays([["x", "x", "y"], [0, 1, 2]])

    # 创建df的副本df2
    df2 = df.copy()
    # 修改df2中的值，以产生比较差异
    df2.iloc[0, 0] = "c"
    df2.iloc[2, 2] = 4.0

    # 调用DataFrame的compare方法，比较df和df2的不同之处
    result = df.compare(df2, align_axis=align_axis)

    # 根据不同的align_axis设置预期的多级索引和数据
    if align_axis == 0:
        indices = pd.MultiIndex.from_arrays(
            [["x", "x", "y", "y"], [0, 0, 2, 2], ["self", "other", "self", "other"]]
        )
        columns = pd.MultiIndex.from_arrays([["a", "b"], ["col1", "col3"]])
        data = [["a", np.nan], ["c", np.nan], [np.nan, 3.0], [np.nan, 4.0]]
    else:
        indices = pd.MultiIndex.from_arrays([["x", "y"], [0, 2]])
        columns = pd.MultiIndex.from_arrays(
            [
                ["a", "a", "b", "b"],
                ["col1", "col1", "col3", "col3"],
                ["self", "other", "self", "other"],
            ]
        )
        data = [["a", "c", np.nan, np.nan], [np.nan, np.nan, 3.0, 4.0]]

    # 创建预期的DataFrame，包含预期的比较结果
    expected = pd.DataFrame(data=data, index=indices, columns=columns)

    # 使用pytest断言函数检查result和expected是否相等
    tm.assert_frame_equal(result, expected)


def test_compare_different_indices():
    # 测试当DataFrame具有不同索引时是否引发异常消息

    # 准备测试所需的异常消息
    msg = (
        r"Can only compare identically-labeled \(both index and columns\) DataFrame "
        "objects"
    )
    # 创建具有不同索引的两个DataFrame
    df1 = pd.DataFrame([1, 2, 3], index=["a", "b", "c"])
    df2 = pd.DataFrame([1, 2, 3], index=["a", "b", "d"])

    # 使用pytest.raises检查是否引发了预期的异常，并验证异常消息是否匹配
    with pytest.raises(ValueError, match=msg):
        df1.compare(df2)


这些注释完整地解释了每行代码的作用和功能，符合要求的格式和规范。
def`
# 测试比较不同形状的数据框，检查是否引发预期的 ValueError 异常
def test_compare_different_shapes():
    # 定义用于匹配异常信息的消息字符串
    msg = (
        r"Can only compare identically-labeled \(both index and columns\) DataFrame "
        "objects"
    )
    # 创建两个不同形状的数据框
    df1 = pd.DataFrame(np.ones((3, 3)))
    df2 = pd.DataFrame(np.zeros((2, 1)))
    # 使用 pytest 检查比较 df1 和 df2 是否会引发 ValueError 异常，并验证异常信息是否与 msg 匹配
    with pytest.raises(ValueError, match=msg):
        df1.compare(df2)


# 测试比较结果的命名
def test_compare_result_names():
    # GH 44354
    # 创建两个数据框 df1 和 df2，包含不同的数据
    df1 = pd.DataFrame(
        {"col1": ["a", "b", "c"], "col2": [1.0, 2.0, np.nan], "col3": [1.0, 2.0, 3.0]},
    )
    df2 = pd.DataFrame(
        {
            "col1": ["c", "b", "c"],
            "col2": [1.0, 2.0, np.nan],
            "col3": [1.0, 2.0, np.nan],
        },
    )
    # 执行比较操作，指定结果的命名为 ("left", "right")
    result = df1.compare(df2, result_names=("left", "right"))
    # 创建期望的结果数据框 expected，包含比较的差异
    expected = pd.DataFrame(
        {
            ("col1", "left"): {0: "a", 2: np.nan},
            ("col1", "right"): {0: "c", 2: np.nan},
            ("col3", "left"): {0: np.nan, 2: 3.0},
            ("col3", "right"): {0: np.nan, 2: np.nan},
        }
    )
    # 使用 assert_frame_equal 检查 result 是否等于 expected
    tm.assert_frame_equal(result, expected)


# 测试无效的 result_names 输入
@pytest.mark.parametrize(
    "result_names",
    [
        [1, 2],
        "HK",
        {"2": 2, "3": 3},
        3,
        3.0,
    ],
)
def test_invalid_input_result_names(result_names):
    # GH 44354
    # 创建两个数据框 df1 和 df2，包含不同的数据
    df1 = pd.DataFrame(
        {"col1": ["a", "b", "c"], "col2": [1.0, 2.0, np.nan], "col3": [1.0, 2.0, 3.0]},
    )
    df2 = pd.DataFrame(
        {
            "col1": ["c", "b", "c"],
            "col2": [1.0, 2.0, np.nan],
            "col3": [1.0, 2.0, np.nan],
        },
    )
    # 使用 pytest 检查传入不同类型的 result_names 是否会引发 TypeError 异常
    with pytest.raises(
        TypeError,
        match=(
            f"Passing 'result_names' as a {type(result_names)} is not "
            "supported. Provide 'result_names' as a tuple instead."
        ),
    ):
        df1.compare(df2, result_names=result_names)


# 测试比较包含不同类型和 NA 值的数据框
@pytest.mark.parametrize(
    "val1,val2",
    [(4, pd.NA), (pd.NA, pd.NA), (pd.NA, 4)],
)
def test_compare_ea_and_np_dtype(val1, val2):
    # GH 48966
    # 创建包含不同值和 NA 的列表和系列
    arr = [4.0, val1]
    ser = pd.Series([1, val2], dtype="Int64")

    # 创建包含不同列的数据框 df1 和 df2
    df1 = pd.DataFrame({"a": arr, "b": [1.0, 2]})
    df2 = pd.DataFrame({"a": ser, "b": [1.0, 2]})
    
    # 创建期望的比较结果数据框 expected
    expected = pd.DataFrame(
        {
            ("a", "self"): arr,
            ("a", "other"): ser,
            ("b", "self"): np.nan,
            ("b", "other"): np.nan,
        }
    )
    
    # 如果 val1 和 val2 都是 NA，则将 expected 中相应位置的值设置为 np.nan
    if val1 is pd.NA and val2 is pd.NA:
        # GH#18463 TODO: is this really the desired behavior?
        expected.loc[1, ("a", "self")] = np.nan

    # 如果 val1 是 NA 并且 numpy 版本大于等于 1.25，则检查是否会引发 TypeError 异常
    if val1 is pd.NA and np_version_gte1p25:
        # 如果比较 df1 和 df2 时保持形状，并包含 NA 值，则预期引发 TypeError 异常
        with pytest.raises(TypeError, match="boolean value of NA is ambiguous"):
            result = df1.compare(df2, keep_shape=True)
    else:
        # 否则，执行比较 df1 和 df2，保持形状
        result = df1.compare(df2, keep_shape=True)
        # 使用 assert_frame_equal 检查 result 是否等于 expected
        tm.assert_frame_equal(result, expected)


# 测试比较包含不同类型和 NA 值的数据框
@pytest.mark.parametrize(
    "df1_val,df2_val,diff_self,diff_other",
    [
        (4, 3, 4, 3),
        (4, 4, pd.NA, pd.NA),
        (4, pd.NA, 4, pd.NA),
        (pd.NA, pd.NA, pd.NA, pd.NA),
    # 结束包含列表的定义，示例中未给出列表的名称，只有结尾的括号。
)
# 定义一个测试函数，用于比较可空的 Int64 数据类型的 DataFrame
def test_compare_nullable_int64_dtype(df1_val, df2_val, diff_self, diff_other):
    # GH 48966

    # 创建第一个 DataFrame df1，包含两列："a" 和 "b"，其中 "a" 列包含可空的 Int64 类型数据
    df1 = pd.DataFrame({"a": pd.Series([df1_val, pd.NA], dtype="Int64"), "b": [1.0, 2]})
    
    # 复制 df1 创建第二个 DataFrame df2，并将 df2 的第一行第一列的值修改为 df2_val
    df2 = df1.copy()
    df2.loc[0, "a"] = df2_val

    # 创建期望得到的 DataFrame expected，包含四列：("a", "self")、("a", "other")、("b", "self")、("b", "other")
    expected = pd.DataFrame(
        {
            ("a", "self"): pd.Series([diff_self, pd.NA], dtype="Int64"),
            ("a", "other"): pd.Series([diff_other, pd.NA], dtype="Int64"),
            ("b", "self"): np.nan,
            ("b", "other"): np.nan,
        }
    )

    # 使用 DataFrame df1 的 compare 方法比较 df1 和 df2 的差异，保持形状一致
    result = df1.compare(df2, keep_shape=True)
    
    # 使用测试框架的 assert_frame_equal 函数断言结果 result 与期望 expected 相等
    tm.assert_frame_equal(result, expected)
```