# `.\numpy\numpy\lib\tests\test_array_utils.py`

```
import numpy as np  # 导入 NumPy 库

from numpy.lib import array_utils  # 从 NumPy 库中导入 array_utils 模块
from numpy.testing import assert_equal  # 从 NumPy 测试模块导入断言函数 assert_equal


class TestByteBounds:
    def test_byte_bounds(self):
        # pointer difference matches size * itemsize
        # due to contiguity
        a = np.arange(12).reshape(3, 4)  # 创建一个 3x4 的数组 a，包含 0 到 11 的连续整数
        low, high = array_utils.byte_bounds(a)  # 调用 byte_bounds 函数计算数组 a 的字节边界
        assert_equal(high - low, a.size * a.itemsize)  # 使用 assert_equal 断言计算的字节边界符合预期

    def test_unusual_order_positive_stride(self):
        a = np.arange(12).reshape(3, 4)  # 创建一个 3x4 的数组 a，包含 0 到 11 的连续整数
        b = a.T  # b 是 a 的转置数组
        low, high = array_utils.byte_bounds(b)  # 调用 byte_bounds 函数计算数组 b 的字节边界
        assert_equal(high - low, b.size * b.itemsize)  # 使用 assert_equal 断言计算的字节边界符合预期

    def test_unusual_order_negative_stride(self):
        a = np.arange(12).reshape(3, 4)  # 创建一个 3x4 的数组 a，包含 0 到 11 的连续整数
        b = a.T[::-1]  # b 是 a 的逆序转置数组
        low, high = array_utils.byte_bounds(b)  # 调用 byte_bounds 函数计算数组 b 的字节边界
        assert_equal(high - low, b.size * b.itemsize)  # 使用 assert_equal 断言计算的字节边界符合预期

    def test_strided(self):
        a = np.arange(12)  # 创建一个包含 0 到 11 的一维数组 a
        b = a[::2]  # b 是 a 的步长为 2 的切片数组
        low, high = array_utils.byte_bounds(b)  # 调用 byte_bounds 函数计算数组 b 的字节边界
        # the largest pointer address is lost (even numbers only in the
        # stride), and compensate addresses for striding by 2
        assert_equal(high - low, b.size * 2 * b.itemsize - b.itemsize)
        # 使用 assert_equal 断言计算的字节边界符合预期，考虑到步长为 2 的地址调整
```