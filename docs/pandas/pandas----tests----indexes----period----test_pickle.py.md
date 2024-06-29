# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\test_pickle.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值计算
import pytest  # 导入 Pytest 库，用于编写和运行单元测试

from pandas import (  # 从 Pandas 库中导入以下模块：
    NaT,  # NaT 表示不可用的时间戳
    PeriodIndex,  # PeriodIndex 用于表示时间段的索引
    period_range,  # period_range 用于生成时间段的范围
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

from pandas.tseries import offsets  # 从 Pandas 时间序列模块中导入 offsets

class TestPickle:
    @pytest.mark.parametrize("freq", ["D", "M", "Y"])
    def test_pickle_round_trip(self, freq):
        idx = PeriodIndex(["2016-05-16", "NaT", NaT, np.nan], freq=freq)  # 创建 PeriodIndex 对象，包含指定频率的时间段
        result = tm.round_trip_pickle(idx)  # 对 idx 进行序列化和反序列化操作
        tm.assert_index_equal(result, idx)  # 断言反序列化后的结果与原始对象 idx 相等

    def test_pickle_freq(self):
        # GH#2891
        prng = period_range("1/1/2011", "1/1/2012", freq="M")  # 创建一个按月频率的时间段范围
        new_prng = tm.round_trip_pickle(prng)  # 对 prng 进行序列化和反序列化操作
        assert new_prng.freq == offsets.MonthEnd()  # 断言反序列化后的对象的频率为月末
        assert new_prng.freqstr == "M"  # 断言反序列化后的对象的频率字符串为 "M"
```