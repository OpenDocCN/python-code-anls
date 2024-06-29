# `D:\src\scipysrc\pandas\pandas\tests\indexes\base_class\test_pickle.py`

```
# 从 pandas 库中导入 Index 类
from pandas import Index
# 从 pandas._testing 中导入 tm 模块
import pandas._testing as tm

# 定义一个测试函数，用于测试 pickle 是否保留对象类型的数据
def test_pickle_preserves_object_dtype():
    # 创建一个 Index 对象，包含整数 1, 2, 3，数据类型为 object
    # GH#43188, GH#43155 表明不要推断为数值数据类型
    index = Index([1, 2, 3], dtype=object)

    # 使用 pandas._testing 的 round_trip_pickle 函数进行 pickle 往返操作
    result = tm.round_trip_pickle(index)
    
    # 断言 pickle 得到的结果的数据类型仍然是 object
    assert result.dtype == object
    
    # 使用 tm.assert_index_equal 函数断言原始 Index 和 pickle 后的结果是相等的
    tm.assert_index_equal(index, result)
```