# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\test_reindex.py`

```
from datetime import timedelta  # 导入 timedelta 类

import numpy as np  # 导入 NumPy 库

from pandas import (  # 从 pandas 库中导入 DatetimeIndex 和 date_range 函数
    DatetimeIndex,
    date_range,
)
import pandas._testing as tm  # 导入 pandas 测试模块


class TestDatetimeIndexReindex:  # 定义测试类 TestDatetimeIndexReindex
    def test_reindex_preserves_tz_if_target_is_empty_list_or_array(self):  # 定义测试方法 test_reindex_preserves_tz_if_target_is_empty_list_or_array
        # 创建一个带时区 "US/Eastern" 的日期范围索引
        index = date_range("2013-01-01", periods=3, tz="US/Eastern")
        # 断言空列表重新索引后第一个元素的时区仍为 "US/Eastern"
        assert str(index.reindex([])[0].tz) == "US/Eastern"
        # 断言空 NumPy 数组重新索引后第一个元素的时区仍为 "US/Eastern"
        assert str(index.reindex(np.array([]))[0].tz) == "US/Eastern"

    def test_reindex_with_same_tz_nearest(self):  # 定义测试方法 test_reindex_with_same_tz_nearest
        # 创建两个日期范围索引 rng_a 和 rng_b，时区均为 "utc"
        rng_a = date_range("2010-01-01", "2010-01-02", periods=24, tz="utc")
        rng_b = date_range("2010-01-01", "2010-01-02", periods=23, tz="utc")
        # 使用 "nearest" 方法和 20 秒的容差对 rng_a 进行重新索引，得到两个结果 result1 和 result2
        result1, result2 = rng_a.reindex(
            rng_b, method="nearest", tolerance=timedelta(seconds=20)
        )
        # 期望的结果列表1，包含预期的日期时间字符串
        expected_list1 = [
            "2010-01-01 00:00:00",
            "2010-01-01 01:05:27.272727272",
            "2010-01-01 02:10:54.545454545",
            "2010-01-01 03:16:21.818181818",
            "2010-01-01 04:21:49.090909090",
            "2010-01-01 05:27:16.363636363",
            "2010-01-01 06:32:43.636363636",
            "2010-01-01 07:38:10.909090909",
            "2010-01-01 08:43:38.181818181",
            "2010-01-01 09:49:05.454545454",
            "2010-01-01 10:54:32.727272727",
            "2010-01-01 12:00:00",
            "2010-01-01 13:05:27.272727272",
            "2010-01-01 14:10:54.545454545",
            "2010-01-01 15:16:21.818181818",
            "2010-01-01 16:21:49.090909090",
            "2010-01-01 17:27:16.363636363",
            "2010-01-01 18:32:43.636363636",
            "2010-01-01 19:38:10.909090909",
            "2010-01-01 20:43:38.181818181",
            "2010-01-01 21:49:05.454545454",
            "2010-01-01 22:54:32.727272727",
            "2010-01-02 00:00:00",
        ]
        # 期望的结果1，包含预期的日期时间索引对象，带有 "datetime64[ns, UTC]" 数据类型
        expected1 = DatetimeIndex(
            expected_list1, dtype="datetime64[ns, UTC]", freq=None
        )
        # 期望的结果2，包含预期的 NumPy 数组，用于检验 reindex 后的索引位置
        expected2 = np.array([0] + [-1] * 21 + [23], dtype=np.dtype("intp"))
        # 断言 result1 和 expected1 相等
        tm.assert_index_equal(result1, expected1)
        # 断言 result2 和 expected2 相等
        tm.assert_numpy_array_equal(result2, expected2)
```