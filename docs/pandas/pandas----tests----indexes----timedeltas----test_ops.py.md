# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\test_ops.py`

```
from pandas import (
    TimedeltaIndex,       # 导入 TimedeltaIndex 类
    timedelta_range,      # 导入 timedelta_range 函数
)
import pandas._testing as tm  # 导入 pandas 测试模块


class TestTimedeltaIndexOps:
    def test_infer_freq(self, freq_sample):
        # GH#11018
        # 使用 timedelta_range 函数创建一个时间增量范围索引 idx，
        # 其中起始时间为 "1"，频率为 freq_sample，包含 10 个周期
        idx = timedelta_range("1", freq=freq_sample, periods=10)
        
        # 使用 asi8 属性创建 TimedeltaIndex 对象 result，
        # 设置其频率为 "infer"
        result = TimedeltaIndex(idx.asi8, freq="infer")
        
        # 断言 idx 和 result 是相等的时间增量索引
        tm.assert_index_equal(idx, result)
        
        # 断言 result 的频率与 freq_sample 相同
        assert result.freq == freq_sample
```