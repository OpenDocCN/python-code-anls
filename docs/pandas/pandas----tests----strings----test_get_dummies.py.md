# `D:\src\scipysrc\pandas\pandas\tests\strings\test_get_dummies.py`

```
# 导入 NumPy 库，用于处理数组和数值计算
import numpy as np

# 从 pandas 库中导入以下模块：
# - DataFrame：用于处理二维表格数据
# - Index：用于表示 pandas 数据结构中的索引
# - MultiIndex：用于表示多层次索引
# - Series：用于处理一维数据结构
# - _testing as tm：导入 _testing 模块并重命名为 tm
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
)


# 定义测试函数 test_get_dummies，参数为 any_string_dtype
def test_get_dummies(any_string_dtype):
    # 创建 Series 对象 s，包含字符串数组和缺失值，数据类型为 any_string_dtype
    s = Series(["a|b", "a|c", np.nan], dtype=any_string_dtype)
    # 对 Series 中的字符串进行分割并生成哑变量（dummy variables），分隔符为 "|"
    result = s.str.get_dummies("|")
    # 预期的 DataFrame 结果，表示哑变量的二维表格
    expected = DataFrame([[1, 1, 0], [1, 0, 1], [0, 0, 0]], columns=list("abc"))
    # 断言结果与预期相等
    tm.assert_frame_equal(result, expected)

    # 创建新的 Series 对象 s，包含不同类型的元素，数据类型为 any_string_dtype
    s = Series(["a;b", "a", 7], dtype=any_string_dtype)
    # 对 Series 中的字符串进行分割并生成哑变量（dummy variables），分隔符为 ";"
    result = s.str.get_dummies(";")
    # 预期的 DataFrame 结果，表示哑变量的二维表格
    expected = DataFrame([[0, 1, 1], [0, 1, 0], [1, 0, 0]], columns=list("7ab"))
    # 断言结果与预期相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数 test_get_dummies_index
def test_get_dummies_index():
    # 测试特定 GitHub 问题 GH9980 和 GH8028
    # 创建 Index 对象 idx，包含字符串数组，使用 "|" 作为分隔符
    idx = Index(["a|b", "a|c", "b|c"])
    # 对 Index 中的字符串进行分割并生成哑变量（dummy variables），分隔符为 "|"
    result = idx.str.get_dummies("|")

    # 预期的 MultiIndex 结果，表示哑变量的多层次索引
    expected = MultiIndex.from_tuples(
        [(1, 1, 0), (1, 0, 1), (0, 1, 1)], names=("a", "b", "c")
    )
    # 断言结果与预期相等
    tm.assert_index_equal(result, expected)


# 定义测试函数 test_get_dummies_with_name_dummy，参数为 any_string_dtype
def test_get_dummies_with_name_dummy(any_string_dtype):
    # 测试特定 GitHub 问题 GH 12180
    # 创建 Series 对象 s，包含不同类型的元素和包含 "name" 的字符串，数据类型为 any_string_dtype
    s = Series(["a", "b,name", "b"], dtype=any_string_dtype)
    # 对 Series 中的字符串进行分割并生成哑变量（dummy variables），分隔符为 ","
    result = s.str.get_dummies(",")
    # 预期的 DataFrame 结果，表示哑变量的二维表格
    expected = DataFrame([[1, 0, 0], [0, 1, 1], [0, 1, 0]], columns=["a", "b", "name"])
    # 断言结果与预期相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数 test_get_dummies_with_name_dummy_index
def test_get_dummies_with_name_dummy_index():
    # 测试特定 GitHub 问题 GH 12180
    # 创建 Index 对象 idx，包含字符串数组，使用 "|" 作为分隔符，并包含 "name" 的字符串
    idx = Index(["a|b", "name|c", "b|name"])
    # 对 Index 中的字符串进行分割并生成哑变量（dummy variables），分隔符为 "|"
    result = idx.str.get_dummies("|")

    # 预期的 MultiIndex 结果，表示哑变量的多层次索引
    expected = MultiIndex.from_tuples(
        [(1, 1, 0, 0), (0, 0, 1, 1), (0, 1, 0, 1)], names=("a", "b", "c", "name")
    )
    # 断言结果与预期相等
    tm.assert_index_equal(result, expected)
```