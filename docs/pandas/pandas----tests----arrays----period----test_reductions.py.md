# `D:\src\scipysrc\pandas\pandas\tests\arrays\period\test_reductions.py`

```
import pandas as pd  # 导入 pandas 库，通常用 pd 别名表示
from pandas.core.arrays import period_array  # 从 pandas 库的 core.arrays 模块导入 period_array 函数

class TestReductions:
    def test_min_max(self):
        arr = period_array(  # 使用 period_array 函数创建 PeriodArray 对象 arr
            [
                "2000-01-03",
                "2000-01-03",
                "NaT",
                "2000-01-02",
                "2000-01-05",
                "2000-01-04",
            ],
            freq="D",  # 指定频率为每天（"D"）
        )

        result = arr.min()  # 计算 arr 中的最小值
        expected = pd.Period("2000-01-02", freq="D")  # 创建预期的 Period 对象
        assert result == expected  # 断言计算结果与预期结果相等

        result = arr.max()  # 计算 arr 中的最大值
        expected = pd.Period("2000-01-05", freq="D")  # 创建预期的 Period 对象
        assert result == expected  # 断言计算结果与预期结果相等

        result = arr.min(skipna=False)  # 在不跳过 NaN 值的情况下计算 arr 中的最小值
        assert result is pd.NaT  # 断言结果为 NaT（Not a Time）

        result = arr.max(skipna=False)  # 在不跳过 NaN 值的情况下计算 arr 中的最大值
        assert result is pd.NaT  # 断言结果为 NaT（Not a Time）

    def test_min_max_empty(self, skipna):
        arr = period_array([], freq="D")  # 创建一个空的 PeriodArray 对象 arr，频率为每天（"D"）
        result = arr.min(skipna=skipna)  # 在指定 skipna 参数的情况下计算 arr 中的最小值
        assert result is pd.NaT  # 断言结果为 NaT（Not a Time）

        result = arr.max(skipna=skipna)  # 在指定 skipna 参数的情况下计算 arr 中的最大值
        assert result is pd.NaT  # 断言结果为 NaT（Not a Time）
```