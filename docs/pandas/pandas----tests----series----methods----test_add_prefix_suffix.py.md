# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_add_prefix_suffix.py`

```
# 导入pytest模块，用于测试框架
import pytest

# 从pandas库中导入Index类
from pandas import Index

# 导入pandas._testing模块，命名为tm，用于测试辅助功能
import pandas._testing as tm


# 定义测试函数test_add_prefix_suffix，测试字符串系列的前缀和后缀添加操作
def test_add_prefix_suffix(string_series):
    # 对字符串系列添加前缀"foo#"
    with_prefix = string_series.add_prefix("foo#")
    # 构建预期的Index对象，其元素为在原索引基础上添加"foo#"前缀的字符串
    expected = Index([f"foo#{c}" for c in string_series.index])
    # 断言添加前缀后的索引与预期的索引相等
    tm.assert_index_equal(with_prefix.index, expected)

    # 对字符串系列添加后缀"#foo"
    with_suffix = string_series.add_suffix("#foo")
    # 构建预期的Index对象，其元素为在原索引基础上添加"#foo"后缀的字符串
    expected = Index([f"{c}#foo" for c in string_series.index])
    # 断言添加后缀后的索引与预期的索引相等
    tm.assert_index_equal(with_suffix.index, expected)

    # 对字符串系列添加前缀"%"
    with_pct_prefix = string_series.add_prefix("%")
    # 构建预期的Index对象，其元素为在原索引基础上添加"%"前缀的字符串
    expected = Index([f"%{c}" for c in string_series.index])
    # 断言添加前缀后的索引与预期的索引相等
    tm.assert_index_equal(with_pct_prefix.index, expected)

    # 对字符串系列添加后缀"%"
    with_pct_suffix = string_series.add_suffix("%")
    # 构建预期的Index对象，其元素为在原索引基础上添加"%"后缀的字符串
    expected = Index([f"{c}%" for c in string_series.index])
    # 断言添加后缀后的索引与预期的索引相等
    tm.assert_index_equal(with_pct_suffix.index, expected)


# 定义测试函数test_add_prefix_suffix_axis，测试带有轴参数的字符串系列前缀和后缀添加操作
def test_add_prefix_suffix_axis(string_series):
    # GH 47819
    # 对字符串系列添加前缀"foo#"，指定轴为0（行索引）
    with_prefix = string_series.add_prefix("foo#", axis=0)
    # 构建预期的Index对象，其元素为在原行索引基础上添加"foo#"前缀的字符串
    expected = Index([f"foo#{c}" for c in string_series.index])
    # 断言添加前缀后的索引与预期的索引相等
    tm.assert_index_equal(with_prefix.index, expected)

    # 对字符串系列添加后缀"#foo"，指定轴为0（行索引）
    with_pct_suffix = string_series.add_suffix("#foo", axis=0)
    # 构建预期的Index对象，其元素为在原行索引基础上添加"#foo"后缀的字符串
    expected = Index([f"{c}#foo" for c in string_series.index])
    # 断言添加后缀后的索引与预期的索引相等
    tm.assert_index_equal(with_pct_suffix.index, expected)


# 定义测试函数test_add_prefix_suffix_invalid_axis，测试使用无效轴参数时的异常情况
def test_add_prefix_suffix_invalid_axis(string_series):
    # 使用pytest断言捕获期望的异常类型和匹配的异常消息
    with pytest.raises(ValueError, match="No axis named 1 for object type Series"):
        # 尝试在轴1上对字符串系列添加前缀"foo#"，应触发值错误异常
        string_series.add_prefix("foo#", axis=1)

    with pytest.raises(ValueError, match="No axis named 1 for object type Series"):
        # 尝试在轴1上对字符串系列添加后缀"foo#"，应触发值错误异常
        string_series.add_suffix("foo#", axis=1)
```