# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_repeat.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 从 pandas 库中导入以下模块：
    MultiIndex,  # 多级索引
    Series,  # Series 数据结构
)
import pandas._testing as tm  # 导入 pandas 测试工具模块


class TestRepeat:  # 定义测试类 TestRepeat
    def test_repeat(self):  # 定义测试方法 test_repeat
        ser = Series(np.random.default_rng(2).standard_normal(3), index=["a", "b", "c"])
        # 创建一个包含随机标准正态分布值的 Series，指定索引为 ["a", "b", "c"]

        reps = ser.repeat(5)
        # 将 Series 中的每个元素重复 5 次
        exp = Series(ser.values.repeat(5), index=ser.index.values.repeat(5))
        # 创建一个期望的 Series，其值是原始 Series 值重复 5 次，索引也重复 5 次
        tm.assert_series_equal(reps, exp)
        # 使用测试工具检查重复后的 Series 是否与期望的 Series 相等

        to_rep = [2, 3, 4]
        reps = ser.repeat(to_rep)
        # 将 Series 中的每个元素根据列表 to_rep 中的对应值重复
        exp = Series(ser.values.repeat(to_rep), index=ser.index.values.repeat(to_rep))
        # 创建一个期望的 Series，其值是原始 Series 值根据 to_rep 中的值重复，索引也相应重复
        tm.assert_series_equal(reps, exp)
        # 使用测试工具检查重复后的 Series 是否与期望的 Series 相等

    def test_numpy_repeat(self):  # 定义测试方法 test_numpy_repeat
        ser = Series(np.arange(3), name="x")
        # 创建一个 Series，其值为从 0 到 2 的整数，指定名称为 "x"

        expected = Series(
            ser.values.repeat(2), name="x", index=ser.index.values.repeat(2)
        )
        # 创建一个期望的 Series，其值是原始 Series 值重复 2 次，索引也重复 2 次，指定名称为 "x"
        tm.assert_series_equal(np.repeat(ser, 2), expected)
        # 使用测试工具检查使用 NumPy 的 repeat 函数后的 Series 是否与期望的 Series 相等

        msg = "the 'axis' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.repeat(ser, 2, axis=0)
        # 使用 pytest 的断言检查调用 NumPy 的 repeat 函数时是否引发 ValueError 异常，并验证异常消息是否匹配

    def test_repeat_with_multiindex(self):  # 定义测试方法 test_repeat_with_multiindex
        # GH#9361, fixed by  GH#7891
        m_idx = MultiIndex.from_tuples([(1, 2), (3, 4), (5, 6), (7, 8)])
        # 创建一个多级索引对象，由元组列表构成

        data = ["a", "b", "c", "d"]
        # 创建一个包含字符串的列表 data

        m_df = Series(data, index=m_idx)
        # 创建一个 Series，将 data 作为值，m_idx 作为索引

        assert m_df.repeat(3).shape == (3 * len(data),)
        # 断言：重复 m_df 中的每个元素 3 次后，结果的形状应为 (3 * len(data),)
```