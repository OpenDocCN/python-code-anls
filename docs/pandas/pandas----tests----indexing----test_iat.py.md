# `D:\src\scipysrc\pandas\pandas\tests\indexing\test_iat.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

from pandas import (  # 从 Pandas 库中导入以下模块：
    DataFrame,      # DataFrame 数据结构，用于表格数据
    Series,         # Series 数据结构，用于一维标签化数组
    period_range,   # period_range 函数，用于创建时间周期范围
)


def test_iat(float_frame):
    # 遍历 float_frame 的索引和列标签
    for i, row in enumerate(float_frame.index):
        for j, col in enumerate(float_frame.columns):
            # 使用 iat 获取指定位置的元素
            result = float_frame.iat[i, j]
            # 使用 at 获取指定行列标签的元素
            expected = float_frame.at[row, col]
            # 断言 iat 获取的结果与 at 获取的结果相等
            assert result == expected


def test_iat_duplicate_columns():
    # 测试处理重复列标签的情况
    # 参考 GitHub issue：https://github.com/pandas-dev/pandas/issues/11754
    df = DataFrame([[1, 2]], columns=["x", "x"])
    # 断言使用 iat 获取重复列标签的第一个元素为 1
    assert df.iat[0, 0] == 1


def test_iat_getitem_series_with_period_index():
    # 测试在 Series 中使用 iat 获取具有周期索引的元素
    # 参考 GitHub issue：https://github.com/pandas-dev/pandas/issues/4390
    index = period_range("1/1/2001", periods=10)
    # 创建一个具有周期索引的 Series 对象
    ser = Series(np.random.default_rng(2).standard_normal(10), index=index)
    # 使用 at 获取指定周期索引的元素
    expected = ser[index[0]]
    # 使用 iat 获取第一个位置的元素
    result = ser.iat[0]
    # 断言使用 iat 获取的结果与 at 获取的结果相等
    assert expected == result
```