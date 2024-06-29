# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\methods\test_fillna.py`

```
from pandas import (
    Index,         # 导入 Index 类，用于处理索引
    NaT,           # 导入 NaT 对象，表示缺失时间
    Timedelta,     # 导入 Timedelta 类，用于处理时间差
    TimedeltaIndex,  # 导入 TimedeltaIndex 类，表示时间差索引
)
import pandas._testing as tm  # 导入 pandas 内部测试模块


class TestFillNA:
    def test_fillna_timedelta(self):
        # GH#11343
        # 创建 TimedeltaIndex 对象，包含三个时间差字符串
        idx = TimedeltaIndex(["1 day", NaT, "3 day"])

        # 预期的 TimedeltaIndex 对象，填充缺失值为 "2 day"
        exp = TimedeltaIndex(["1 day", "2 day", "3 day"])
        tm.assert_index_equal(idx.fillna(Timedelta("2 day")), exp)

        # 用 Timedelta("3 hour") 填充缺失值
        # 这行代码没有赋值给任何变量，仅修改了 idx 对象本身
        exp = TimedeltaIndex(["1 day", "3 hour", "3 day"])
        idx.fillna(Timedelta("3 hour"))

        # 预期的 Index 对象，填充缺失值为 "x"
        exp = Index([Timedelta("1 day"), "x", Timedelta("3 day")], dtype=object)
        tm.assert_index_equal(idx.fillna("x"), exp)
```