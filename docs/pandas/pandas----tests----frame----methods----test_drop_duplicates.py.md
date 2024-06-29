# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_drop_duplicates.py`

```
`
# 从 datetime 模块导入 datetime 类
from datetime import datetime
# 导入 re 模块，用于正则表达式操作
import re

# 导入 numpy 库，并使用 np 别名
import numpy as np
# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 从 pandas 库中导入 DataFrame、NaT（Not a Time）、concat 函数
from pandas import (
    DataFrame,
    NaT,
    concat,
)
# 导入 pandas 内部的测试工具模块，使用 tm 别名
import pandas._testing as tm


# 使用 pytest 的 parametrize 装饰器，对 test_drop_duplicates_with_misspelled_column_name 函数进行参数化测试
@pytest.mark.parametrize("subset", ["a", ["a"], ["a", "B"]])
def test_drop_duplicates_with_misspelled_column_name(subset):
    # 创建一个 DataFrame 对象 df，包含列"A"、"B"、"C"，每列有三行数据
    df = DataFrame({"A": [0, 0, 1], "B": [0, 0, 1], "C": [0, 0, 1]})
    # 根据正则表达式生成一个用于匹配错误 KeyError 消息的字符串
    msg = re.escape("Index(['a'], dtype=")

    # 使用 pytest 的 raises 方法，验证在执行 df.drop_duplicates(subset) 时是否会抛出 KeyError 异常，并匹配消息 msg
    with pytest.raises(KeyError, match=msg):
        df.drop_duplicates(subset)


# 定义一个名为 test_drop_duplicates 的测试函数
def test_drop_duplicates():
    # 创建一个 DataFrame 对象 df，包含列"AAA"、"B"、"C"、"D"，每列有八行数据
    df = DataFrame(
        {
            "AAA": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    
    # 使用单列"AAA"进行去重操作，保存结果到 result
    result = df.drop_duplicates("AAA")
    # 期望的结果为 df 的前两行
    expected = df[:2]
    # 使用 pandas 的 assert_frame_equal 方法验证 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)

    # 使用单列"AAA"进行去重操作，保留最后一条重复记录
    result = df.drop_duplicates("AAA", keep="last")
    # 期望的结果为 df 的第 6、7 行
    expected = df.loc[[6, 7]]
    tm.assert_frame_equal(result, expected)

    # 使用单列"AAA"进行去重操作，删除所有重复记录
    result = df.drop_duplicates("AAA", keep=False)
    # 期望的结果为空的 DataFrame
    expected = df.loc[[]]
    tm.assert_frame_equal(result, expected)
    # 验证结果 DataFrame 的长度是否为 0
    assert len(result) == 0

    # 使用多列["AAA", "B"]进行去重操作
    expected = df.loc[[0, 1, 2, 3]]
    result = df.drop_duplicates(np.array(["AAA", "B"]))
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates(["AAA", "B"])
    tm.assert_frame_equal(result, expected)

    # 使用多列("AAA", "B")进行去重操作，保留最后一条重复记录
    result = df.drop_duplicates(("AAA", "B"), keep="last")
    # 期望的结果为 df 的第 0、5、6、7 行
    expected = df.loc[[0, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    # 使用多列("AAA", "B")进行去重操作，删除所有重复记录
    result = df.drop_duplicates(("AAA", "B"), keep=False)
    # 期望的结果为 df 的第 0 行
    expected = df.loc[[0]]
    tm.assert_frame_equal(result, expected)

    # 使用所有列进行去重操作
    df2 = df.loc[:, ["AAA", "B", "C"]]

    result = df2.drop_duplicates()
    # 期望的结果为按["AAA", "B"]列进行去重后的 df2
    expected = df2.drop_duplicates(["AAA", "B"])
    tm.assert_frame_equal(result, expected)

    result = df2.drop_duplicates(keep="last")
    # 期望的结果为按["AAA", "B"]列进行去重并保留最后一条记录后的 df2
    expected = df2.drop_duplicates(["AAA", "B"], keep="last")
    tm.assert_frame_equal(result, expected)

    result = df2.drop_duplicates(keep=False)
    # 期望的结果为按["AAA", "B"]列进行去重并删除所有重复记录后的 df2
    expected = df2.drop_duplicates(["AAA", "B"], keep=False)
    tm.assert_frame_equal(result, expected)

    # 使用整数列"C"进行去重操作
    result = df.drop_duplicates("C")
    # 期望的结果为 df 的第 0、2 行
    expected = df.iloc[[0, 2]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates("C", keep="last")
    # 期望的结果为 df 的倒数第 2、最后一行
    expected = df.iloc[[-2, -1]]
    tm.assert_frame_equal(result, expected)

    # 将列"C"转换为整型列"E"后，使用"E"列进行去重操作
    df["E"] = df["C"].astype("int8")
    result = df.drop_duplicates("E")
    # 期望的结果为 df 的第 0、2 行
    expected = df.iloc[[0, 2]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates("E", keep="last")
    # 期望的结果为 df 的倒数第 2、最后一行
    expected = df.iloc[[-2, -1]]
    tm.assert_frame_equal(result, expected)

    # GH 11376
    # 创建一个 DataFrame 对象 df，包含两列"x"、"y"，每列有七行数据
    df = DataFrame({"x": [7, 6, 3, 3, 4, 8, 0], "y": [0, 6, 5, 5, 9, 1, 2]})
    # 期望的结果为剔除索引为 3 的行后的 df
    expected = df.loc[df.index != 3]
    tm.assert_frame_equal(df.drop_duplicates(), expected)

    # 创建一个二维列表数据创建 DataFrame 对象 df
    df = DataFrame([[1, 0], [0, 2]])
    # 检查 DataFrame 中是否有重复行，并要求二者完全相等
    tm.assert_frame_equal(df.drop_duplicates(), df)

    # 创建一个新的 DataFrame，包含两行数据，每行都有一个重复的值
    df = DataFrame([[-2, 0], [0, -4]])
    # 检查 DataFrame 中是否有重复行，并要求二者完全相等
    tm.assert_frame_equal(df.drop_duplicates(), df)

    # 计算 np.int64 的最大值的两倍除以三，然后再乘以二
    x = np.iinfo(np.int64).max / 3 * 2
    # 创建一个新的 DataFrame，包含两行数据，第一行包含一个负的 x 值和一个正的 x 值
    df = DataFrame([[-x, x], [0, x + 4]])
    # 检查 DataFrame 中是否有重复行，并要求二者完全相等
    tm.assert_frame_equal(df.drop_duplicates(), df)

    # 创建一个新的 DataFrame，包含两行数据，每行都包含 x 和 x+4 两个相同的值
    df = DataFrame([[-x, x], [x, x + 4]])
    # 检查 DataFrame 中是否有重复行，并要求二者完全相等
    tm.assert_frame_equal(df.drop_duplicates(), df)

    # GH 11864：测试特定情况下的 DataFrame
    # 创建一个 DataFrame，其中包含 16 行，每行都重复 9 次不同的索引值 i
    df = DataFrame([i for i in range(16)] * 9)
    # 创建一个包含一行的 DataFrame，该行的第一个元素为 1，其余元素为 0
    df = concat([df, DataFrame([[1] + [0] * 8])], ignore_index=True)

    # 针对不同的 keep 参数（first、last、False），检查 DataFrame 中是否有重复行
    for keep in ["first", "last", False]:
        # 断言在给定 keep 参数条件下，DataFrame 中没有重复行
        assert df.duplicated(keep=keep).sum() == 0
def test_drop_duplicates_with_duplicate_column_names():
    # 测试用例：处理具有重复列名的DataFrame
    # 创建一个DataFrame，包含重复的列名"a"，以及列名"b"
    df = DataFrame([[1, 2, 5], [3, 4, 6], [3, 4, 7]], columns=["a", "a", "b")

    # 调用drop_duplicates()函数，不指定任何参数，期望返回原始DataFrame
    result0 = df.drop_duplicates()
    tm.assert_frame_equal(result0, df)

    # 调用drop_duplicates()函数，指定列"a"，期望返回前两行（去除重复值）
    result1 = df.drop_duplicates("a")
    expected1 = df[:2]
    tm.assert_frame_equal(result1, expected1)


def test_drop_duplicates_for_take_all():
    # 测试用例：处理包含不同情况下的DataFrame
    df = DataFrame(
        {
            "AAA": ["foo", "bar", "baz", "bar", "foo", "bar", "qux", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )

    # 单列情况下的不同参数调用
    result = df.drop_duplicates("AAA")
    expected = df.iloc[[0, 1, 2, 6]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("AAA", keep="last")
    expected = df.iloc[[2, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates("AAA", keep=False)
    expected = df.iloc[[2, 6]]
    tm.assert_frame_equal(result, expected)

    # 多列情况下的不同参数调用
    result = df.drop_duplicates(["AAA", "B"])
    expected = df.iloc[[0, 1, 2, 3, 4, 6]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(["AAA", "B"], keep="last")
    expected = df.iloc[[0, 1, 2, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(["AAA", "B"], keep=False)
    expected = df.iloc[[0, 1, 2, 6]]
    tm.assert_frame_equal(result, expected)


def test_drop_duplicates_tuple():
    # 测试用例：处理包含元组索引的DataFrame
    df = DataFrame(
        {
            ("AA", "AB"): ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )

    # 单列情况下的不同参数调用
    result = df.drop_duplicates(("AA", "AB"))
    expected = df[:2]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(("AA", "AB"), keep="last")
    expected = df.loc[[6, 7]]
    tm.assert_frame_equal(result, expected)

    result = df.drop_duplicates(("AA", "AB"), keep=False)
    expected = df.loc[[]]  # 空DataFrame
    assert len(result) == 0
    tm.assert_frame_equal(result, expected)

    # 多列情况下的不同参数调用
    expected = df.loc[[0, 1, 2, 3]]
    result = df.drop_duplicates((("AA", "AB"), "B"))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "df",
    [
        DataFrame(),
        DataFrame(columns=[]),
        DataFrame(columns=["A", "B", "C"]),
        DataFrame(index=[]),
        DataFrame(index=["A", "B", "C"]),
    ],
)
def test_drop_duplicates_empty(df):
    # 测试用例：处理空DataFrame的情况
    # 调用drop_duplicates()函数，期望返回原始空DataFrame
    result = df.drop_duplicates()
    tm.assert_frame_equal(result, df)

    # 使用inplace=True方式调用drop_duplicates()函数，期望返回原始空DataFrame
    result = df.copy()
    result.drop_duplicates(inplace=True)
    tm.assert_frame_equal(result, df)


def test_drop_duplicates_NA():
    # 测试用例：处理没有缺失值的情况
    # 目前无需特殊处理
    # 创建一个 DataFrame 对象，包含四列数据
    df = DataFrame(
        {
            "A": [None, None, "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1, 1.0],
            "D": range(8),
        }
    )
    # 按照列"A"的值去除重复行，返回结果赋给result
    result = df.drop_duplicates("A")
    # 预期结果是df中索引为0、2、3的行组成的DataFrame
    expected = df.loc[[0, 2, 3]]
    # 使用测试框架tm检查result和expected是否相等
    tm.assert_frame_equal(result, expected)

    # 按照列"A"的值去除重复行，保留最后一个重复项
    result = df.drop_duplicates("A", keep="last")
    # 预期结果是df中索引为1、6、7的行组成的DataFrame
    expected = df.loc[[1, 6, 7]]
    tm.assert_frame_equal(result, expected)

    # 按照列"A"的值去除所有重复行，预期返回一个空的DataFrame
    result = df.drop_duplicates("A", keep=False)
    expected = df.loc[[]]  # 空的DataFrame
    tm.assert_frame_equal(result, expected)
    # 使用断言检查result的长度是否为0
    assert len(result) == 0

    # 按照列"A"和"B"的值去除重复行
    result = df.drop_duplicates(["A", "B"])
    # 预期结果是df中索引为0、2、3、6的行组成的DataFrame
    expected = df.loc[[0, 2, 3, 6]]
    tm.assert_frame_equal(result, expected)

    # 按照列"A"和"B"的值去除重复行，保留最后一个重复项
    result = df.drop_duplicates(["A", "B"], keep="last")
    # 预期结果是df中索引为1、5、6、7的行组成的DataFrame
    expected = df.loc[[1, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    # 按照列"A"和"B"的值去除所有重复行，预期返回一个包含索引为6的行的DataFrame
    result = df.drop_duplicates(["A", "B"], keep=False)
    expected = df.loc[[6]]
    tm.assert_frame_equal(result, expected)

    # nan值处理
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1, 1.0],
            "D": range(8),
        }
    )
    # 按照列"C"的值去除重复行
    result = df.drop_duplicates("C")
    # 预期结果是df中前两行组成的DataFrame
    expected = df[:2]
    tm.assert_frame_equal(result, expected)

    # 按照列"C"的值去除重复行，保留最后一个重复项
    result = df.drop_duplicates("C", keep="last")
    # 预期结果是df中索引为3、7的行组成的DataFrame
    expected = df.loc[[3, 7]]
    tm.assert_frame_equal(result, expected)

    # 按照列"C"的值去除所有重复行，预期返回一个空的DataFrame
    result = df.drop_duplicates("C", keep=False)
    expected = df.loc[[]]  # 空的DataFrame
    tm.assert_frame_equal(result, expected)
    assert len(result) == 0

    # 按照列"C"和"B"的值去除重复行
    result = df.drop_duplicates(["C", "B"])
    # 预期结果是df中索引为0、1、2、4的行组成的DataFrame
    expected = df.loc[[0, 1, 2, 4]]
    tm.assert_frame_equal(result, expected)

    # 按照列"C"和"B"的值去除重复行，保留最后一个重复项
    result = df.drop_duplicates(["C", "B"], keep="last")
    # 预期结果是df中索引为1、3、6、7的行组成的DataFrame
    expected = df.loc[[1, 3, 6, 7]]
    tm.assert_frame_equal(result, expected)

    # 按照列"C"和"B"的值去除所有重复行，预期返回一个包含索引为1的行的DataFrame
    result = df.drop_duplicates(["C", "B"], keep=False)
    expected = df.loc[[1]]
    tm.assert_frame_equal(result, expected)
def test_drop_duplicates_NA_for_take_all():
    # 定义一个包含空值的数据框
    df = DataFrame(
        {
            "A": [None, None, "foo", "bar", "foo", "baz", "bar", "qux"],
            "C": [1.0, np.nan, np.nan, np.nan, 1.0, 2.0, 3, 1.0],
        }
    )

    # 根据'A'列去重，保留第一个出现的值
    result = df.drop_duplicates("A")
    expected = df.iloc[[0, 2, 3, 5, 7]]
    tm.assert_frame_equal(result, expected)

    # 根据'A'列去重，保留最后一个出现的值
    result = df.drop_duplicates("A", keep="last")
    expected = df.iloc[[1, 4, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    # 根据'A'列去重，不保留任何重复项
    result = df.drop_duplicates("A", keep=False)
    expected = df.iloc[[5, 7]]
    tm.assert_frame_equal(result, expected)

    # nan情况

    # 根据'C'列去重，保留第一个出现的值
    result = df.drop_duplicates("C")
    expected = df.iloc[[0, 1, 5, 6]]
    tm.assert_frame_equal(result, expected)

    # 根据'C'列去重，保留最后一个出现的值
    result = df.drop_duplicates("C", keep="last")
    expected = df.iloc[[3, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)

    # 根据'C'列去重，不保留任何重复项
    result = df.drop_duplicates("C", keep=False)
    expected = df.iloc[[5, 6]]
    tm.assert_frame_equal(result, expected)


def test_drop_duplicates_inplace():
    orig = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    # 单列去重，原地修改
    df = orig.copy()
    return_value = df.drop_duplicates("A", inplace=True)
    expected = orig[:2]
    result = df
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    # 单列去重，保留最后一个出现的值，原地修改
    df = orig.copy()
    return_value = df.drop_duplicates("A", keep="last", inplace=True)
    expected = orig.loc[[6, 7]]
    result = df
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    # 单列去重，不保留任何重复项，原地修改
    df = orig.copy()
    return_value = df.drop_duplicates("A", keep=False, inplace=True)
    expected = orig.loc[[]]
    result = df
    tm.assert_frame_equal(result, expected)
    assert len(df) == 0
    assert return_value is None

    # 多列去重，原地修改
    df = orig.copy()
    return_value = df.drop_duplicates(["A", "B"], inplace=True)
    expected = orig.loc[[0, 1, 2, 3]]
    result = df
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    # 多列去重，保留最后一个出现的值，原地修改
    df = orig.copy()
    return_value = df.drop_duplicates(["A", "B"], keep="last", inplace=True)
    expected = orig.loc[[0, 5, 6, 7]]
    result = df
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    # 多列去重，不保留任何重复项，原地修改
    df = orig.copy()
    return_value = df.drop_duplicates(["A", "B"], keep=False, inplace=True)
    expected = orig.loc[[0]]
    result = df
    tm.assert_frame_equal(result, expected)
    assert return_value is None

    # 考虑所有列去重，原地修改
    orig2 = orig.loc[:, ["A", "B", "C"]].copy()

    df2 = orig2.copy()
    return_value = df2.drop_duplicates(inplace=True)
    # 只考虑'A', 'B'列的去重情况
    expected = orig2.drop_duplicates(["A", "B"])
    result = df2
    tm.assert_frame_equal(result, expected)
    # 断言返回值应为 None，确保没有意外的返回值
    assert return_value is None
    
    # 复制原始 DataFrame `orig2`，以避免直接修改原始数据
    df2 = orig2.copy()
    # 在副本 DataFrame `df2` 上应用 drop_duplicates 方法，保留最后出现的重复行，并就地修改
    return_value = df2.drop_duplicates(keep="last", inplace=True)
    # 根据列"A"和"B"在原始 DataFrame `orig2` 上进行 drop_duplicates 操作，保留最后出现的重复行
    expected = orig2.drop_duplicates(["A", "B"], keep="last")
    # 将结果保存在变量 `result` 中，此时 `df2` 已经被修改
    result = df2
    # 使用 assert_frame_equal 函数比较 `result` 和 `expected` 是否相等
    tm.assert_frame_equal(result, expected)
    # 再次断言返回值应为 None，确保没有意外的返回值
    assert return_value is None
    
    # 复制原始 DataFrame `orig2`，以避免直接修改原始数据
    df2 = orig2.copy()
    # 在副本 DataFrame `df2` 上应用 drop_duplicates 方法，删除所有重复行，并就地修改
    return_value = df2.drop_duplicates(keep=False, inplace=True)
    # 根据列"A"和"B"在原始 DataFrame `orig2` 上进行 drop_duplicates 操作，删除所有重复行
    expected = orig2.drop_duplicates(["A", "B"], keep=False)
    # 将结果保存在变量 `result` 中，此时 `df2` 已经被修改
    result = df2
    # 使用 assert_frame_equal 函数比较 `result` 和 `expected` 是否相等
    tm.assert_frame_equal(result, expected)
    # 再次断言返回值应为 None，确保没有意外的返回值
    assert return_value is None
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize(
    "origin_dict, output_dict, ignore_index, output_index",
    [
        # 定义测试参数和期望输出，包括原始字典、输出字典、是否忽略索引以及输出索引
        ({"A": [2, 2, 3]}, {"A": [2, 3]}, True, [0, 1]),  # 示例1：忽略索引，期望输出为{"A": [2, 3]}
        ({"A": [2, 2, 3]}, {"A": [2, 3]}, False, [0, 2]),  # 示例2：不忽略索引，期望输出为{"A": [2, 3]}
        ({"A": [2, 2, 3], "B": [2, 2, 4]}, {"A": [2, 3], "B": [2, 4]}, True, [0, 1]),  # 示例3：多列，忽略索引
        ({"A": [2, 2, 3], "B": [2, 2, 4]}, {"A": [2, 3], "B": [2, 4]}, False, [0, 2]),  # 示例4：多列，不忽略索引
    ],
)
def test_drop_duplicates_ignore_index(
    inplace, origin_dict, output_dict, ignore_index, output_index
):
    # GH 30114
    # 使用原始字典创建 DataFrame 对象
    df = DataFrame(origin_dict)
    # 使用输出字典和输出索引创建预期的 DataFrame 对象
    expected = DataFrame(output_dict, index=output_index)

    if inplace:
        # 如果 inplace 为 True，则对 DataFrame 执行就地修改，并返回修改后的结果
        result_df = df.copy()
        result_df.drop_duplicates(ignore_index=ignore_index, inplace=inplace)
    else:
        # 如果 inplace 为 False，则返回新的修改后的 DataFrame，原始 DataFrame 不变
        result_df = df.drop_duplicates(ignore_index=ignore_index, inplace=inplace)

    # 断言修改后的 DataFrame 和预期的 DataFrame 相等
    tm.assert_frame_equal(result_df, expected)
    # 断言原始 DataFrame 未被修改
    tm.assert_frame_equal(df, DataFrame(origin_dict))


def test_drop_duplicates_null_in_object_column(nulls_fixture):
    # https://github.com/pandas-dev/pandas/issues/32992
    # 使用包含空值的列创建 DataFrame
    df = DataFrame([[1, nulls_fixture], [2, "a"]], dtype=object)
    # 对 DataFrame 执行去重操作
    result = df.drop_duplicates()
    # 断言去重后的结果与原始 DataFrame 相等
    tm.assert_frame_equal(result, df)


def test_drop_duplicates_series_vs_dataframe(keep):
    # GH#14192
    # 创建包含不同数据类型和空值的 DataFrame
    df = DataFrame(
        {
            "a": [1, 1, 1, "one", "one"],
            "b": [2, 2, np.nan, np.nan, np.nan],
            "c": [3, 3, np.nan, np.nan, "three"],
            "d": [1, 2, 3, 4, 4],
            "e": [
                datetime(2015, 1, 1),
                datetime(2015, 1, 1),
                datetime(2015, 2, 1),
                NaT,
                NaT,
            ],
        }
    )
    # 遍历 DataFrame 的每一列
    for column in df.columns:
        # 对单独的列执行去重操作，返回 DataFrame
        dropped_frame = df[[column]].drop_duplicates(keep=keep)
        # 对单独的列执行去重操作，返回 Series
        dropped_series = df[column].drop_duplicates(keep=keep)
        # 断言两种方式去重后的结果相等
        tm.assert_frame_equal(dropped_frame, dropped_series.to_frame())


@pytest.mark.parametrize("arg", [[1], 1, "True", [], 0])
def test_drop_duplicates_non_boolean_ignore_index(arg):
    # GH#38274
    # 创建包含重复值的 DataFrame
    df = DataFrame({"a": [1, 2, 1, 3]})
    # 定义错误的 ignore_index 参数类型，并期望引发 ValueError 异常
    msg = '^For argument "ignore_index" expected type bool, received type .*.$'
    with pytest.raises(ValueError, match=msg):
        # 尝试使用错误类型的 ignore_index 参数执行去重操作，预期引发异常
        df.drop_duplicates(ignore_index=arg)
```