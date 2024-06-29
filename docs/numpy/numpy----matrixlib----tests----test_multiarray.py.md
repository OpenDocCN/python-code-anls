# `.\numpy\numpy\matrixlib\tests\test_multiarray.py`

```
import numpy as np  # 导入 NumPy 库
from numpy.testing import assert_, assert_equal, assert_array_equal  # 导入测试工具函数

class TestView:
    def test_type(self):
        x = np.array([1, 2, 3])  # 创建一个 NumPy 数组
        assert_(isinstance(x.view(np.matrix), np.matrix))  # 断言转换为矩阵类型后的类型判断

    def test_keywords(self):
        x = np.array([(1, 2)], dtype=[('a', np.int8), ('b', np.int8)])  # 创建一个结构化数组
        # 我们必须在这里明确指定字节顺序：
        y = x.view(dtype='<i2', type=np.matrix)  # 将数组视图转换为指定类型（矩阵），指定小端序
        assert_array_equal(y, [[513]])  # 断言视图转换后的数组内容

        assert_(isinstance(y, np.matrix))  # 断言 y 是 np.matrix 类型
        assert_equal(y.dtype, np.dtype('<i2'))  # 断言 y 的数据类型为小端序的 16 位整数
```