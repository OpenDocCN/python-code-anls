# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\test_npfuncs.py`

```
import numpy as np

# 从 pandas 库中导入 date_range 函数
from pandas import date_range

# 导入 pandas 内部测试模块，命名为 tm
import pandas._testing as tm

# 定义一个测试类 TestSplit
class TestSplit:
    
    # 定义测试方法 test_split_non_utc
    def test_split_non_utc(self):
        # 用例 GH#14042
        # 创建一个日期范围，从 "2016-01-01 00:00:00+0200" 开始，频率为每秒一次，共生成 10 个时间点
        indices = date_range("2016-01-01 00:00:00+0200", freq="s", periods=10)
        
        # 使用 numpy 的 split 函数将 indices 数组按照空列表（即不分割）分割，取第一个分割结果
        result = np.split(indices, indices_or_sections=[])[0]
        
        # 期望的结果是将 indices 的频率设为 None
        expected = indices._with_freq(None)
        
        # 使用测试模块 tm 的 assert_index_equal 方法，比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected)
```