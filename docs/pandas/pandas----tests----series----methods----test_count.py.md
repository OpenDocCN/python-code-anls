# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_count.py`

```
# 导入 NumPy 库，并使用别名 np
import numpy as np

# 从 pandas 库中导入 Categorical 和 Series 类
from pandas import (
    Categorical,
    Series,
)


# 定义一个测试类 TestSeriesCount
class TestSeriesCount:
    # 定义测试方法 test_count，接受参数 datetime_series
    def test_count(self, datetime_series):
        # 断言 datetime_series 的 count 方法返回的结果等于其长度
        assert datetime_series.count() == len(datetime_series)

        # 将 datetime_series 中每隔两个元素设为 NaN
        datetime_series[::2] = np.nan

        # 再次断言 datetime_series 的 count 方法返回的结果等于非 NaN 值的数量
        assert datetime_series.count() == np.isfinite(datetime_series).sum()

    # 定义测试方法 test_count_categorical，无参数
    def test_count_categorical(self):
        # 创建一个 Series 对象 ser，其中的值是一个 Categorical 对象
        ser = Series(
            Categorical(
                [np.nan, 1, 2, np.nan],  # 使用 Categorical 创建包含 NaN 的序列
                categories=[5, 4, 3, 2, 1],  # 指定分类的顺序
                ordered=True  # 指定分类是否有序
            )
        )
        # 调用 ser 的 count 方法，将结果赋给变量 result
        result = ser.count()
        # 断言 result 的值等于 2
        assert result == 2
```