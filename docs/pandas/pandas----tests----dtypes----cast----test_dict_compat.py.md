# `D:\src\scipysrc\pandas\pandas\tests\dtypes\cast\test_dict_compat.py`

```
# 导入 numpy 库，并将其命名为 np
import numpy as np

# 从 pandas 库中导入 dict_compat 函数
from pandas.core.dtypes.cast import dict_compat

# 从 pandas 库中导入 Timestamp 类
from pandas import Timestamp

# 定义一个测试函数，用于测试 dict_compat 函数的功能
def test_dict_compat():
    # 创建一个包含 np.datetime64 对象作为键的字典
    data_datetime64 = {np.datetime64("1990-03-15"): 1, np.datetime64("2015-03-15"): 2}
    
    # 创建一个普通的不变字典
    data_unchanged = {1: 2, 3: 4, 5: 6}
    
    # 创建一个预期输出是 Timestamp 对象作为键的字典
    expected = {Timestamp("1990-3-15"): 1, Timestamp("2015-03-15"): 2}
    
    # 断言 dict_compat 函数对 data_datetime64 的处理结果等于预期的 expected 字典
    assert dict_compat(data_datetime64) == expected
    
    # 断言 dict_compat 函数对已经转换为 Timestamp 键的 expected 字典不会改变
    assert dict_compat(expected) == expected
    
    # 断言 dict_compat 函数对普通的不变字典 data_unchanged 不会改变其内容
    assert dict_compat(data_unchanged) == data_unchanged
```