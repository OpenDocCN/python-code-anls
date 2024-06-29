# `D:\src\scipysrc\pandas\pandas\tests\indexes\base_class\test_where.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数组和数值计算

from pandas import Index  # 导入 pandas 库中的 Index 类，用于操作索引对象
import pandas._testing as tm  # 导入 pandas 内部测试模块，用于测试辅助函数和类


class TestWhere:
    def test_where_intlike_str_doesnt_cast_ints(self):
        idx = Index(range(3))  # 创建一个包含整数范围的索引对象 idx: Index([0, 1, 2])
        mask = np.array([True, False, True])  # 创建一个布尔类型的 NumPy 数组 mask: [True, False, True]
        res = idx.where(mask, "2")  # 使用 mask 条件，将符合条件的索引值保留，不符合的替换为 "2"，结果 res: Index([0, '2', 2])
        expected = Index([0, "2", 2])  # 预期的索引对象 expected: Index([0, '2', 2])
        tm.assert_index_equal(res, expected)  # 断言函数，验证 res 和 expected 是否相等
```