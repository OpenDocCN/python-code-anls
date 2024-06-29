# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_reindex_like.py`

```
# 从 datetime 模块中导入 datetime 类
from datetime import datetime

# 导入 numpy 库并用 np 别名表示
import numpy as np

# 从 pandas 库中导入 Series 类
from pandas import Series

# 导入 pandas._testing 模块并用 tm 别名表示
import pandas._testing as tm


# 定义函数 test_reindex_like，接收一个 datetime_series 参数
def test_reindex_like(datetime_series):
    # 从 datetime_series 中取出索引为偶数的子序列
    other = datetime_series[::2]
    # 断言 datetime_series 重新索引到 other 索引后与 datetime_series.reindex_like(other) 相等
    tm.assert_series_equal(
        datetime_series.reindex(other.index), datetime_series.reindex_like(other)
    )

    # GH#7179
    # 创建三个 datetime 对象
    day1 = datetime(2013, 3, 5)
    day2 = datetime(2013, 5, 5)
    day3 = datetime(2014, 3, 5)

    # 创建两个 Series 对象，指定索引和数据
    series1 = Series([5, None, None], [day1, day2, day3])
    series2 = Series([None, None], [day1, day3])

    # 使用 assert_produces_warning 检查 FutureWarning
    with tm.assert_produces_warning(FutureWarning):
        # 使用 pad 方法，将 series1 重新索引到与 series2 相同的索引上
        result = series1.reindex_like(series2, method="pad")
    
    # 创建预期的 Series 对象，指定索引和数据
    expected = Series([5, np.nan], index=[day1, day3])
    # 断言 result 与 expected 相等
    tm.assert_series_equal(result, expected)


# 定义函数 test_reindex_like_nearest，无参数
def test_reindex_like_nearest():
    # 创建一个 Series 对象，包含整数值 0 到 9
    ser = Series(np.arange(10, dtype="int64"))

    # 创建目标索引列表 target
    target = [0.1, 0.9, 1.5, 2.0]
    # 使用 nearest 方法，将 ser 重新索引到 target 索引上
    other = ser.reindex(target, method="nearest")
    # 创建预期的 Series 对象，指定数据和索引
    expected = Series(np.around(target).astype("int64"), index=target)

    # 使用 assert_produces_warning 检查 FutureWarning
    with tm.assert_produces_warning(FutureWarning):
        # 将 ser 重新索引到与 other 相同的索引上，使用 nearest 方法
        result = ser.reindex_like(other, method="nearest")
    # 断言 result 与 expected 相等
    tm.assert_series_equal(expected, result)

    # 使用 assert_produces_warning 检查 FutureWarning
    with tm.assert_produces_warning(FutureWarning):
        # 将 ser 重新索引到与 other 相同的索引上，使用 nearest 方法和指定的 tolerance 参数
        result = ser.reindex_like(other, method="nearest", tolerance=1)
    # 断言 result 与 expected 相等
    tm.assert_series_equal(expected, result)
    
    # 使用 assert_produces_warning 检查 FutureWarning
    with tm.assert_produces_warning(FutureWarning):
        # 将 ser 重新索引到与 other 相同的索引上，使用 nearest 方法和指定的 tolerance 列表参数
        result = ser.reindex_like(other, method="nearest", tolerance=[1, 2, 3, 4])
    # 断言 result 与 expected 相等
    tm.assert_series_equal(expected, result)
```