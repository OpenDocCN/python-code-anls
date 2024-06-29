# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\methods\test_repeat.py`

```
import numpy as np  # 导入NumPy库，用于科学计算

from pandas import (  # 从Pandas库中导入以下模块：
    TimedeltaIndex,   # 时间增量索引模块
    timedelta_range,  # 时间增量范围模块
)
import pandas._testing as tm  # 导入Pandas的测试模块

class TestRepeat:
    def test_repeat(self):
        index = timedelta_range("1 days", periods=2, freq="D")  # 创建一个时间增量索引，从"1 days"开始，2个时间点，频率为每日
        exp = TimedeltaIndex(["1 days", "1 days", "2 days", "2 days"])  # 期望的时间增量索引，包含指定的时间增量
        for res in [index.repeat(2), np.repeat(index, 2)]:  # 遍历两种方式生成的结果：Pandas的repeat方法和NumPy的repeat方法
            tm.assert_index_equal(res, exp)  # 使用测试模块检查结果是否与期望一致
            assert res.freq is None  # 断言结果的频率属性为None，即不设定频率

        index = TimedeltaIndex(["1 days", "NaT", "3 days"])  # 创建另一个时间增量索引，包含"1 days"、"NaT"和"3 days"
        exp = TimedeltaIndex(  # 期望的时间增量索引，包含指定的时间增量和NaT
            [
                "1 days",
                "1 days",
                "1 days",
                "NaT",
                "NaT",
                "NaT",
                "3 days",
                "3 days",
                "3 days",
            ]
        )
        for res in [index.repeat(3), np.repeat(index, 3)]:  # 遍历两种方式生成的结果：Pandas的repeat方法和NumPy的repeat方法
            tm.assert_index_equal(res, exp)  # 使用测试模块检查结果是否与期望一致
            assert res.freq is None  # 断言结果的频率属性为None，即不设定频率
```