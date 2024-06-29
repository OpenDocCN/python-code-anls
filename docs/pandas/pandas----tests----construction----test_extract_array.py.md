# `D:\src\scipysrc\pandas\pandas\tests\construction\test_extract_array.py`

```
# 从 pandas 库中导入 Index 类
from pandas import Index
# 导入 pandas 内部测试模块
import pandas._testing as tm
# 从 pandas 核心构建模块中导入 extract_array 函数
from pandas.core.construction import extract_array

# 定义一个测试函数，用于测试 extract_array 函数对 RangeIndex 的处理
def test_extract_array_rangeindex():
    # 创建一个 RangeIndex 对象，包含从 0 到 4 的索引
    ri = Index(range(5))

    # 获取 RangeIndex 对象的内部数值作为期望结果
    expected = ri._values
    # 调用 extract_array 函数，提取 RangeIndex 对象的数据作为 numpy 数组，并断言结果与期望相等
    res = extract_array(ri, extract_numpy=True, extract_range=True)
    tm.assert_numpy_array_equal(res, expected)
    
    # 再次调用 extract_array 函数，不使用 numpy 数组提取选项，但依然提取 RangeIndex 对象的数据作为范围索引，并断言结果与期望相等
    res = extract_array(ri, extract_numpy=False, extract_range=True)
    tm.assert_numpy_array_equal(res, expected)

    # 使用 extract_array 函数提取 RangeIndex 对象的数据，返回结果应与原始 RangeIndex 对象相等，并断言结果与期望相等
    res = extract_array(ri, extract_numpy=True, extract_range=False)
    tm.assert_index_equal(res, ri)
    
    # 再次使用 extract_array 函数，不提取 numpy 数组也不提取范围索引，直接返回原始 RangeIndex 对象，并断言结果与期望相等
    res = extract_array(ri, extract_numpy=False, extract_range=False)
    tm.assert_index_equal(res, ri)
```