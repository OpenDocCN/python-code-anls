# `D:\src\scipysrc\scipy\scipy\signal\tests\test_array_tools.py`

```
# 导入 numpy 库，并将其命名为 np
import numpy as np

# 从 numpy.testing 模块中导入 assert_array_equal 函数
# 从 pytest 模块中导入 raises 函数并重命名为 assert_raises

# 从 scipy.signal._arraytools 模块中导入以下函数：
# axis_slice, axis_reverse, odd_ext, even_ext, const_ext, zero_ext

# 定义一个测试类 TestArrayTools
class TestArrayTools:

    # 定义测试方法 test_axis_slice
    def test_axis_slice(self):
        # 创建一个 3x4 的数组 a，元素为 0 到 11
        a = np.arange(12).reshape(3, 4)

        # 测试 axis_slice 函数在 axis=0 轴上的切片操作
        s = axis_slice(a, start=0, stop=1, axis=0)
        assert_array_equal(s, a[0:1, :])

        # 测试 axis_slice 函数在 axis=0 轴上的反向切片操作
        s = axis_slice(a, start=-1, axis=0)
        assert_array_equal(s, a[-1:, :])

        # 测试 axis_slice 函数在 axis=1 轴上的切片操作
        s = axis_slice(a, start=0, stop=1, axis=1)
        assert_array_equal(s, a[:, 0:1])

        # 测试 axis_slice 函数在 axis=1 轴上的反向切片操作
        s = axis_slice(a, start=-1, axis=1)
        assert_array_equal(s, a[:, -1:])

        # 测试 axis_slice 函数在 axis=0 轴上的步长为2的切片操作
        s = axis_slice(a, start=0, step=2, axis=0)
        assert_array_equal(s, a[::2, :])

        # 测试 axis_slice 函数在 axis=1 轴上的步长为2的切片操作
        s = axis_slice(a, start=0, step=2, axis=1)
        assert_array_equal(s, a[:, ::2])

    # 定义测试方法 test_axis_reverse
    def test_axis_reverse(self):
        # 创建一个 3x4 的数组 a，元素为 0 到 11
        a = np.arange(12).reshape(3, 4)

        # 测试 axis_reverse 函数在 axis=0 轴上的反转操作
        r = axis_reverse(a, axis=0)
        assert_array_equal(r, a[::-1, :])

        # 测试 axis_reverse 函数在 axis=1 轴上的反转操作
        r = axis_reverse(a, axis=1)
        assert_array_equal(r, a[:, ::-1])

    # 定义测试方法 test_odd_ext
    def test_odd_ext(self):
        # 创建一个 2x5 的数组 a
        a = np.array([[1, 2, 3, 4, 5],
                      [9, 8, 7, 6, 5]])

        # 测试 odd_ext 函数在 axis=1 轴上的奇数扩展操作
        odd = odd_ext(a, 2, axis=1)
        expected = np.array([[-1, 0, 1, 2, 3, 4, 5, 6, 7],
                             [11, 10, 9, 8, 7, 6, 5, 4, 3]])
        assert_array_equal(odd, expected)

        # 测试 odd_ext 函数在 axis=0 轴上的奇数扩展操作
        odd = odd_ext(a, 1, axis=0)
        expected = np.array([[-7, -4, -1, 2, 5],
                             [1, 2, 3, 4, 5],
                             [9, 8, 7, 6, 5],
                             [17, 14, 11, 8, 5]])
        assert_array_equal(odd, expected)

        # 测试 odd_ext 函数在 axis=0 轴上超出边界时是否抛出 ValueError 异常
        assert_raises(ValueError, odd_ext, a, 2, axis=0)
        # 测试 odd_ext 函数在 axis=1 轴上超出边界时是否抛出 ValueError 异常
        assert_raises(ValueError, odd_ext, a, 5, axis=1)

    # 定义测试方法 test_even_ext
    def test_even_ext(self):
        # 创建一个 2x5 的数组 a
        a = np.array([[1, 2, 3, 4, 5],
                      [9, 8, 7, 6, 5]])

        # 测试 even_ext 函数在 axis=1 轴上的偶数扩展操作
        even = even_ext(a, 2, axis=1)
        expected = np.array([[3, 2, 1, 2, 3, 4, 5, 4, 3],
                             [7, 8, 9, 8, 7, 6, 5, 6, 7]])
        assert_array_equal(even, expected)

        # 测试 even_ext 函数在 axis=0 轴上的偶数扩展操作
        even = even_ext(a, 1, axis=0)
        expected = np.array([[9, 8, 7, 6, 5],
                             [1, 2, 3, 4, 5],
                             [9, 8, 7, 6, 5],
                             [1, 2, 3, 4, 5]])
        assert_array_equal(even, expected)

        # 测试 even_ext 函数在 axis=0 轴上超出边界时是否抛出 ValueError 异常
        assert_raises(ValueError, even_ext, a, 2, axis=0)
        # 测试 even_ext 函数在 axis=1 轴上超出边界时是否抛出 ValueError 异常
        assert_raises(ValueError, even_ext, a, 5, axis=1)
    # 定义测试函数 test_const_ext，测试在给定数组上进行常量扩展的功能
    def test_const_ext(self):
        # 创建一个二维 NumPy 数组 a
        a = np.array([[1, 2, 3, 4, 5],
                      [9, 8, 7, 6, 5]])

        # 在 axis=1 轴向上对数组 a 进行常量扩展，生成 const 数组
        const = const_ext(a, 2, axis=1)
        # 期望得到的结果数组 expected
        expected = np.array([[1, 1, 1, 2, 3, 4, 5, 5, 5],
                             [9, 9, 9, 8, 7, 6, 5, 5, 5]])
        # 断言 const 数组与 expected 数组相等
        assert_array_equal(const, expected)

        # 在 axis=0 轴向上对数组 a 进行常量扩展，生成 const 数组
        const = const_ext(a, 1, axis=0)
        # 期望得到的结果数组 expected
        expected = np.array([[1, 2, 3, 4, 5],
                             [1, 2, 3, 4, 5],
                             [9, 8, 7, 6, 5],
                             [9, 8, 7, 6, 5]])
        # 断言 const 数组与 expected 数组相等
        assert_array_equal(const, expected)

    # 定义测试函数 test_zero_ext，测试在给定数组上进行零扩展的功能
    def test_zero_ext(self):
        # 创建一个二维 NumPy 数组 a
        a = np.array([[1, 2, 3, 4, 5],
                      [9, 8, 7, 6, 5]])

        # 在 axis=1 轴向上对数组 a 进行零扩展，生成 zero 数组
        zero = zero_ext(a, 2, axis=1)
        # 期望得到的结果数组 expected
        expected = np.array([[0, 0, 1, 2, 3, 4, 5, 0, 0],
                             [0, 0, 9, 8, 7, 6, 5, 0, 0]])
        # 断言 zero 数组与 expected 数组相等
        assert_array_equal(zero, expected)

        # 在 axis=0 轴向上对数组 a 进行零扩展，生成 zero 数组
        zero = zero_ext(a, 1, axis=0)
        # 期望得到的结果数组 expected
        expected = np.array([[0, 0, 0, 0, 0],
                             [1, 2, 3, 4, 5],
                             [9, 8, 7, 6, 5],
                             [0, 0, 0, 0, 0]])
        # 断言 zero 数组与 expected 数组相等
        assert_array_equal(zero, expected)
```