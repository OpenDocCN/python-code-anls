# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_add_prefix_suffix.py`

```
# 导入 pytest 库，用于单元测试
import pytest

# 导入 pandas 库中的 Index 类和 pandas._testing 模块中的 tm 对象
from pandas import Index
import pandas._testing as tm


# 定义测试函数 test_add_prefix_suffix，参数为 float_frame
def test_add_prefix_suffix(float_frame):
    # 在列名前添加前缀 "foo#"
    with_prefix = float_frame.add_prefix("foo#")
    # 期望的列名列表，每列名前加上 "foo#"
    expected = Index([f"foo#{c}" for c in float_frame.columns])
    # 断言 with_prefix 的列名与期望的列名列表相同
    tm.assert_index_equal(with_prefix.columns, expected)

    # 在列名后添加后缀 "#foo"
    with_suffix = float_frame.add_suffix("#foo")
    # 期望的列名列表，每列名后加上 "#foo"
    expected = Index([f"{c}#foo" for c in float_frame.columns])
    # 断言 with_suffix 的列名与期望的列名列表相同
    tm.assert_index_equal(with_suffix.columns, expected)

    # 在列名前添加前缀 "%"
    with_pct_prefix = float_frame.add_prefix("%")
    # 期望的列名列表，每列名前加上 "%"
    expected = Index([f"%{c}" for c in float_frame.columns])
    # 断言 with_pct_prefix 的列名与期望的列名列表相同
    tm.assert_index_equal(with_pct_prefix.columns, expected)

    # 在列名后添加后缀 "%"
    with_pct_suffix = float_frame.add_suffix("%")
    # 期望的列名列表，每列名后加上 "%"
    expected = Index([f"{c}%" for c in float_frame.columns])
    # 断言 with_pct_suffix 的列名与期望的列名列表相同
    tm.assert_index_equal(with_pct_suffix.columns, expected)


# 定义测试函数 test_add_prefix_suffix_axis，参数为 float_frame
def test_add_prefix_suffix_axis(float_frame):
    # GH 47819
    # 在行索引前添加前缀 "foo#"
    with_prefix = float_frame.add_prefix("foo#", axis=0)
    # 期望的行索引列表，每行索引前加上 "foo#"
    expected = Index([f"foo#{c}" for c in float_frame.index])
    # 断言 with_prefix 的行索引与期望的行索引列表相同
    tm.assert_index_equal(with_prefix.index, expected)

    # 在列名前添加前缀 "foo#"
    with_prefix = float_frame.add_prefix("foo#", axis=1)
    # 期望的列名列表，每列名前加上 "foo#"
    expected = Index([f"foo#{c}" for c in float_frame.columns])
    # 断言 with_prefix 的列名与期望的列名列表相同
    tm.assert_index_equal(with_prefix.columns, expected)

    # 在行索引后添加后缀 "#foo"
    with_pct_suffix = float_frame.add_suffix("#foo", axis=0)
    # 期望的行索引列表，每行索引后加上 "#foo"
    expected = Index([f"{c}#foo" for c in float_frame.index])
    # 断言 with_pct_suffix 的行索引与期望的行索引列表相同
    tm.assert_index_equal(with_pct_suffix.index, expected)

    # 在列名后添加后缀 "#foo"
    with_pct_suffix = float_frame.add_suffix("#foo", axis=1)
    # 期望的列名列表，每列名后加上 "#foo"
    expected = Index([f"{c}#foo" for c in float_frame.columns])
    # 断言 with_pct_suffix 的列名与期望的列名列表相同
    tm.assert_index_equal(with_pct_suffix.columns, expected)


# 定义测试函数 test_add_prefix_suffix_invalid_axis，参数为 float_frame
def test_add_prefix_suffix_invalid_axis(float_frame):
    # 测试在 DataFrame 中使用不存在的 axis=2 抛出 ValueError 异常
    with pytest.raises(ValueError, match="No axis named 2 for object type DataFrame"):
        float_frame.add_prefix("foo#", axis=2)

    # 测试在 DataFrame 中使用不存在的 axis=2 抛出 ValueError 异常
    with pytest.raises(ValueError, match="No axis named 2 for object type DataFrame"):
        float_frame.add_suffix("foo#", axis=2)
```