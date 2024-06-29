# `.\numpy\numpy\fft\tests\test_helper.py`

```
# 导入必要的库
import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy import fft, pi

# 定义一个测试类 TestFFTShift
class TestFFTShift:

    # 定义测试函数 test_definition，测试 fftshift 和 ifftshift 的定义
    def test_definition(self):
        # 定义输入数组 x 和预期输出数组 y
        x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
        y = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        # 断言 fftshift(x) 的输出与 y 几乎相等
        assert_array_almost_equal(fft.fftshift(x), y)
        # 断言 ifftshift(y) 的输出与 x 几乎相等
        assert_array_almost_equal(fft.ifftshift(y), x)

        # 重新定义输入数组 x 和预期输出数组 y
        x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        y = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        # 断言 fftshift(x) 的输出与 y 几乎相等
        assert_array_almost_equal(fft.fftshift(x), y)
        # 断言 ifftshift(y) 的输出与 x 几乎相等
        assert_array_almost_equal(fft.ifftshift(y), x)

    # 定义测试函数 test_inverse，测试 fftshift 和 ifftshift 的逆操作
    def test_inverse(self):
        # 对于不同的 n 值，生成随机数组 x 进行测试
        for n in [1, 4, 9, 100, 211]:
            x = np.random.random((n,))
            # 断言 ifftshift(fftshift(x)) 的输出与 x 几乎相等
            assert_array_almost_equal(fft.ifftshift(fft.fftshift(x)), x)

    # 定义测试函数 test_axes_keyword，测试带有 axes 关键字参数的 fftshift 和 ifftshift
    def test_axes_keyword(self):
        # 定义频率数组 freqs 和预期输出数组 shifted
        freqs = [[0, 1, 2], [3, 4, -4], [-3, -2, -1]]
        shifted = [[-1, -3, -2], [2, 0, 1], [-4, 3, 4]]
        # 断言 fftshift(freqs, axes=(0, 1)) 的输出与 shifted 几乎相等
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0, 1)), shifted)
        # 断言 fftshift(freqs, axes=0) 的输出与 fftshift(freqs, axes=(0,)) 几乎相等
        assert_array_almost_equal(fft.fftshift(freqs, axes=0),
                                  fft.fftshift(freqs, axes=(0,)))
        # 断言 ifftshift(shifted, axes=(0, 1)) 的输出与 freqs 几乎相等
        assert_array_almost_equal(fft.ifftshift(shifted, axes=(0, 1)), freqs)
        # 断言 ifftshift(shifted, axes=0) 的输出与 ifftshift(shifted, axes=(0,)) 几乎相等
        assert_array_almost_equal(fft.ifftshift(shifted, axes=0),
                                  fft.ifftshift(shifted, axes=(0,)))

        # 断言 fftshift(freqs) 的输出与 shifted 几乎相等
        assert_array_almost_equal(fft.fftshift(freqs), shifted)
        # 断言 ifftshift(shifted) 的输出与 freqs 几乎相等
        assert_array_almost_equal(fft.ifftshift(shifted), freqs)
    def test_uneven_dims(self):
        """ Test 2D input, which has uneven dimension sizes """
        freqs = [
            [0, 1],
            [2, 3],
            [4, 5]
        ]

        # shift in dimension 0
        # 在维度 0 上进行移位
        shift_dim0 = [
            [4, 5],
            [0, 1],
            [2, 3]
        ]
        assert_array_almost_equal(fft.fftshift(freqs, axes=0), shift_dim0)
        assert_array_almost_equal(fft.ifftshift(shift_dim0, axes=0), freqs)
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0,)), shift_dim0)
        assert_array_almost_equal(fft.ifftshift(shift_dim0, axes=[0]), freqs)

        # shift in dimension 1
        # 在维度 1 上进行移位
        shift_dim1 = [
            [1, 0],
            [3, 2],
            [5, 4]
        ]
        assert_array_almost_equal(fft.fftshift(freqs, axes=1), shift_dim1)
        assert_array_almost_equal(fft.ifftshift(shift_dim1, axes=1), freqs)

        # shift in both dimensions
        # 在两个维度上同时进行移位
        shift_dim_both = [
            [5, 4],
            [1, 0],
            [3, 2]
        ]
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0, 1)), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=(0, 1)), freqs)
        assert_array_almost_equal(fft.fftshift(freqs, axes=[0, 1]), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=[0, 1]), freqs)

        # axes=None (default) shift in all dimensions
        # axes=None（默认）在所有维度上进行移位
        assert_array_almost_equal(fft.fftshift(freqs, axes=None), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=None), freqs)
        assert_array_almost_equal(fft.fftshift(freqs), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both), freqs)
    def test_equal_to_original(self):
        """ Test that the new (>=v1.15) implementation (see #10073) is equal to the original (<=v1.14) """
        # 导入必要的函数和模块
        from numpy._core import asarray, concatenate, arange, take

        def original_fftshift(x, axes=None):
            """ How fftshift was implemented in v1.14"""
            # 将输入数组转换为数组
            tmp = asarray(x)
            # 获取数组的维度
            ndim = tmp.ndim
            # 如果 axes 未指定，将其设为所有维度的范围列表
            if axes is None:
                axes = list(range(ndim))
            # 如果 axes 是整数，则转换为元组
            elif isinstance(axes, int):
                axes = (axes,)
            # 初始值设为输入数组
            y = tmp
            # 对每一个轴进行操作
            for k in axes:
                # 获取当前轴的长度
                n = tmp.shape[k]
                # 计算中间点位置
                p2 = (n + 1) // 2
                # 创建平移后的索引列表
                mylist = concatenate((arange(p2, n), arange(p2)))
                # 在当前轴上进行取值
                y = take(y, mylist, k)
            return y

        def original_ifftshift(x, axes=None):
            """ How ifftshift was implemented in v1.14 """
            # 将输入数组转换为数组
            tmp = asarray(x)
            # 获取数组的维度
            ndim = tmp.ndim
            # 如果 axes 未指定，将其设为所有维度的范围列表
            if axes is None:
                axes = list(range(ndim))
            # 如果 axes 是整数，则转换为元组
            elif isinstance(axes, int):
                axes = (axes,)
            # 初始值设为输入数组
            y = tmp
            # 对每一个轴进行操作
            for k in axes:
                # 获取当前轴的长度
                n = tmp.shape[k]
                # 计算中间点位置
                p2 = n - (n + 1) // 2
                # 创建逆平移后的索引列表
                mylist = concatenate((arange(p2, n), arange(p2)))
                # 在当前轴上进行取值
                y = take(y, mylist, k)
            return y

        # create possible 2d array combinations and try all possible keywords
        # compare output to original functions
        # 针对可能的二维数组组合和所有可能的关键字进行测试
        for i in range(16):
            for j in range(16):
                for axes_keyword in [0, 1, None, (0,), (0, 1)]:
                    # 生成随机输入数组
                    inp = np.random.rand(i, j)

                    # 检查 fftshift 的输出与原始函数的输出是否几乎相等
                    assert_array_almost_equal(fft.fftshift(inp, axes_keyword),
                                              original_fftshift(inp, axes_keyword))

                    # 检查 ifftshift 的输出与原始函数的输出是否几乎相等
                    assert_array_almost_equal(fft.ifftshift(inp, axes_keyword),
                                              original_ifftshift(inp, axes_keyword))
# 定义一个名为 TestFFTFreq 的测试类
class TestFFTFreq:

    # 定义测试方法 test_definition
    def test_definition(self):
        # 输入数据 x
        x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
        # 断言使用 fft.fftfreq() 函数计算的结果与预期结果 x 几乎相等
        assert_array_almost_equal(9*fft.fftfreq(9), x)
        # 断言使用 fft.fftfreq() 函数计算的结果与预期结果 x 几乎相等，带有自定义的 pi 参数
        assert_array_almost_equal(9*pi*fft.fftfreq(9, pi), x)

        # 更新输入数据 x
        x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        # 断言使用 fft.fftfreq() 函数计算的结果与预期结果 x 几乎相等
        assert_array_almost_equal(10*fft.fftfreq(10), x)
        # 断言使用 fft.fftfreq() 函数计算的结果与预期结果 x 几乎相等，带有自定义的 pi 参数
        assert_array_almost_equal(10*pi*fft.fftfreq(10, pi), x)


# 定义一个名为 TestRFFTFreq 的测试类
class TestRFFTFreq:

    # 定义测试方法 test_definition
    def test_definition(self):
        # 输入数据 x
        x = [0, 1, 2, 3, 4]
        # 断言使用 fft.rfftfreq() 函数计算的结果与预期结果 x 几乎相等
        assert_array_almost_equal(9*fft.rfftfreq(9), x)
        # 断言使用 fft.rfftfreq() 函数计算的结果与预期结果 x 几乎相等，带有自定义的 pi 参数
        assert_array_almost_equal(9*pi*fft.rfftfreq(9, pi), x)

        # 更新输入数据 x
        x = [0, 1, 2, 3, 4, 5]
        # 断言使用 fft.rfftfreq() 函数计算的结果与预期结果 x 几乎相等
        assert_array_almost_equal(10*fft.rfftfreq(10), x)
        # 断言使用 fft.rfftfreq() 函数计算的结果与预期结果 x 几乎相等，带有自定义的 pi 参数
        assert_array_almost_equal(10*pi*fft.rfftfreq(10, pi), x)


# 定义一个名为 TestIRFFTN 的测试类
class TestIRFFTN:

    # 定义测试方法 test_not_last_axis_success
    def test_not_last_axis_success(self):
        # 创建一个大小为 (2, 16, 8, 32) 的随机复数数组 ar 和 ai
        ar, ai = np.random.random((2, 16, 8, 32))
        # 将 ar 和 ai 组合成复数数组 a
        a = ar + 1j*ai

        # 指定轴向参数 axes = (-2)
        axes = (-2,)

        # 应该不会引发错误，执行 fft.irfftn() 函数
        fft.irfftn(a, axes=axes)
```