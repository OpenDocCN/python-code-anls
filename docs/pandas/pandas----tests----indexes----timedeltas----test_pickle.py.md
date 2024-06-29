# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\test_pickle.py`

```
from pandas import timedelta_range  # 导入pandas模块中的timedelta_range函数
import pandas._testing as tm  # 导入pandas._testing模块并命名为tm，用于测试辅助函数


class TestPickle:
    def test_pickle_after_set_freq(self):
        tdi = timedelta_range("1 day", periods=4, freq="s")
        tdi = tdi._with_freq(None)  # 调用_timedelta_range类的_with_freq方法，设置频率为None

        res = tm.round_trip_pickle(tdi)  # 使用tm模块的round_trip_pickle函数对tdi进行pickle序列化和反序列化
        tm.assert_index_equal(res, tdi)  # 使用tm模块的assert_index_equal函数断言序列化前后的索引对象相等
```