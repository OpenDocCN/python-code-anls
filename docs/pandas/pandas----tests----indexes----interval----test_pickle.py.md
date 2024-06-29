# `D:\src\scipysrc\pandas\pandas\tests\indexes\interval\test_pickle.py`

```
from pandas import IntervalIndex  # 导入 IntervalIndex 类
import pandas._testing as tm  # 导入 pandas 内部测试模块


class TestPickle:
    def test_pickle_round_trip_closed(self, closed):
        # 定义一个 IntervalIndex 对象，包含闭合属性，参考：https://github.com/pandas-dev/pandas/issues/35658
        idx = IntervalIndex.from_tuples([(1, 2), (2, 3)], closed=closed)
        # 对 idx 进行序列化和反序列化，返回序列化后的结果
        result = tm.round_trip_pickle(idx)
        # 使用测试模块中的函数检查两个索引对象是否相等
        tm.assert_index_equal(result, idx)
```