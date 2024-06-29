# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_index_as_string.py`

```
# 导入所需的库
import numpy as np
import pytest

# 导入 pandas 库及其测试模块
import pandas as pd
import pandas._testing as tm

# 使用 pytest 的 parametrize 装饰器定义多组参数化测试用例
@pytest.mark.parametrize(
    "key_strs,groupers",
    [
        ("inner", pd.Grouper(level="inner")),  # 使用字符串作为索引级别
        (["inner"], [pd.Grouper(level="inner")]),  # 使用字符串列表作为索引级别
        (["B", "inner"], ["B", pd.Grouper(level="inner")]),  # 同时指定列名和索引级别
        (["inner", "B"], [pd.Grouper(level="inner"), "B"]),  # 同时指定索引级别和列名
    ],
)
@pytest.mark.parametrize("levels", [["inner"], ["inner", "outer"]])
def test_grouper_index_level_as_string(levels, key_strs, groupers):
    # 创建一个 DataFrame 对象
    frame = pd.DataFrame(
        {
            "outer": ["a", "a", "a", "b", "b", "b"],
            "inner": [1, 2, 3, 1, 2, 3],
            "A": np.arange(6),
            "B": ["one", "one", "two", "two", "one", "one"],
        }
    )
    # 将 DataFrame 设置索引
    frame = frame.set_index(levels)
    
    # 根据条件选择使用不同的分组器（groupers）计算结果
    if "B" not in key_strs or "outer" in frame.columns:
        result = frame.groupby(key_strs).mean(numeric_only=True)
        expected = frame.groupby(groupers).mean(numeric_only=True)
    else:
        result = frame.groupby(key_strs).mean()
        expected = frame.groupby(groupers).mean()
    
    # 使用测试模块中的 assert_frame_equal 函数比较结果是否一致
    tm.assert_frame_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器定义多组参数化测试用例
@pytest.mark.parametrize(
    "levels",
    [
        "inner",
        "outer",
        "B",
        ["inner"],
        ["outer"],
        ["B"],
        ["inner", "outer"],
        ["outer", "inner"],
        ["inner", "outer", "B"],
        ["B", "outer", "inner"],
    ],
)
def test_grouper_index_level_as_string_series(levels):
    # 创建一个 DataFrame 对象
    df = pd.DataFrame(
        {
            "outer": ["a", "a", "a", "b", "b", "b"],
            "inner": [1, 2, 3, 1, 2, 3],
            "A": np.arange(6),
            "B": ["one", "one", "two", "two", "one", "one"],
        }
    )
    # 从 DataFrame 中选择特定的 Series 对象，并设置复合索引
    series = df.set_index(["outer", "inner", "B"])["A"]
    
    # 根据 levels 参数类型选择适当的分组器（groupers）
    if isinstance(levels, list):
        groupers = [pd.Grouper(level=lv) for lv in levels]
    else:
        groupers = pd.Grouper(level=levels)

    # 计算期望结果
    expected = series.groupby(groupers).mean()

    # 计算并检查实际结果
    result = series.groupby(levels).mean()
    # 使用测试模块中的 assert_series_equal 函数比较结果是否一致
    tm.assert_series_equal(result, expected)
```