# `D:\src\scipysrc\pandas\pandas\tests\arrays\floating\test_contains.py`

```
# 导入 NumPy 库，用于科学计算
import numpy as np

# 导入 Pandas 库，用于数据处理和分析
import pandas as pd

# 定义一个函数，用于测试数组中是否包含 NaN 值的情况
def test_contains_nan():
    # 创建一个 Pandas 的数组对象，其中包含一个除以零的操作，意图是生成 NaN 值
    arr = pd.array(range(5)) / 0
    
    # 断言第一个元素的值在 arr 对象的内部数据表示中是 NaN
    assert np.isnan(arr._data[0])
    
    # 断言 arr 对象的第一个元素不是 NaN
    assert not arr.isna()[0]
    
    # 断言 NaN 值在 arr 对象中
    assert np.nan in arr
```