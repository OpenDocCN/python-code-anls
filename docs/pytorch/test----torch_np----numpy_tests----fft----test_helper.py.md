# `.\pytorch\test\torch_np\numpy_tests\fft\test_helper.py`

```py
# Owner(s): ["module: dynamo"]
# 模块所有者为 "module: dynamo"

"""Test functions for fftpack.helper module

Copied from fftpack.helper by Pearu Peterson, October 2005

"""
# 导入测试所需模块和函数
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)

# 根据是否使用 TorchDynamo 运行测试，选择相应的 numpy 和测试函数
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import fft, pi
    from numpy.testing import assert_array_almost_equal
else:
    import torch._numpy as np
    from torch._numpy import fft, pi
    from torch._numpy.testing import assert_array_almost_equal


# 定义测试类 TestFFTShift，继承自 TestCase 类
class TestFFTShift(TestCase):

    # 定义测试 fft.fftshift 函数的行为
    def test_definition(self):
        x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
        y = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        assert_array_almost_equal(fft.fftshift(x), y)  # 检查 fft.fftshift 函数的结果是否与预期相符
        assert_array_almost_equal(fft.ifftshift(y), x)  # 检查 fft.ifftshift 函数的结果是否与预期相符
        x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        y = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        assert_array_almost_equal(fft.fftshift(x), y)  # 再次检查 fft.fftshift 函数的结果是否与预期相符
        assert_array_almost_equal(fft.ifftshift(y), x)  # 再次检查 fft.ifftshift 函数的结果是否与预期相符

    # 定义测试 fft.ifftshift 函数的逆操作是否与原始操作相等
    def test_inverse(self):
        for n in [1, 4, 9, 100, 211]:
            x = np.random.random((n,))
            assert_array_almost_equal(fft.ifftshift(fft.fftshift(x)), x)  # 检查逆操作结果是否与原始数据相等

    # 定义测试 fft.fftshift 和 fft.ifftshift 函数的关于 axes 参数的行为
    def test_axes_keyword(self):
        freqs = [[0, 1, 2], [3, 4, -4], [-3, -2, -1]]
        shifted = [[-1, -3, -2], [2, 0, 1], [-4, 3, 4]]
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0, 1)), shifted)  # 检查指定轴的 fft.fftshift 结果是否与预期相符
        assert_array_almost_equal(
            fft.fftshift(freqs, axes=0), fft.fftshift(freqs, axes=(0,))
        )  # 检查不同形式指定轴的 fft.fftshift 结果是否一致
        assert_array_almost_equal(fft.ifftshift(shifted, axes=(0, 1)), freqs)  # 检查指定轴的 fft.ifftshift 结果是否与预期相符
        assert_array_almost_equal(
            fft.ifftshift(shifted, axes=0), fft.ifftshift(shifted, axes=(0,))
        )  # 检查不同形式指定轴的 fft.ifftshift 结果是否一致

        assert_array_almost_equal(fft.fftshift(freqs), shifted)  # 检查未指定轴的 fft.fftshift 结果是否与预期相符
        assert_array_almost_equal(fft.ifftshift(shifted), freqs)  # 检查未指定轴的 fft.ifftshift 结果是否与预期相符
    def test_uneven_dims(self):
        """Test 2D input, which has uneven dimension sizes"""
        # 定义一个二维数组，表示频率信息
        freqs = [[0, 1], [2, 3], [4, 5]]

        # 在第 0 维度进行移位
        shift_dim0 = [[4, 5], [0, 1], [2, 3]]
        # 对频率数组进行傅里叶移位操作，指定轴为第 0 维度
        assert_array_almost_equal(fft.fftshift(freqs, axes=0), shift_dim0)
        # 对移位后的数组进行逆傅里叶移位操作，指定轴为第 0 维度
        assert_array_almost_equal(fft.ifftshift(shift_dim0, axes=0), freqs)
        # 以元组形式指定轴为第 0 维度进行傅里叶移位操作
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0,)), shift_dim0)
        # 以列表形式指定轴为第 0 维度进行逆傅里叶移位操作
        assert_array_almost_equal(fft.ifftshift(shift_dim0, axes=[0]), freqs)

        # 在第 1 维度进行移位
        shift_dim1 = [[1, 0], [3, 2], [5, 4]]
        # 对频率数组进行傅里叶移位操作，指定轴为第 1 维度
        assert_array_almost_equal(fft.fftshift(freqs, axes=1), shift_dim1)
        # 对移位后的数组进行逆傅里叶移位操作，指定轴为第 1 维度
        assert_array_almost_equal(fft.ifftshift(shift_dim1, axes=1), freqs)

        # 在两个维度同时进行移位
        shift_dim_both = [[5, 4], [1, 0], [3, 2]]
        # 对频率数组进行傅里叶移位操作，指定轴为两个维度
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0, 1)), shift_dim_both)
        # 对移位后的数组进行逆傅里叶移位操作，指定轴为两个维度
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=(0, 1)), freqs)
        # 以列表形式指定轴为两个维度进行傅里叶移位操作
        assert_array_almost_equal(fft.fftshift(freqs, axes=[0, 1]), shift_dim_both)
        # 以列表形式指定轴为两个维度进行逆傅里叶移位操作
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=[0, 1]), freqs)

        # axes=None（默认值），在所有维度上进行移位
        assert_array_almost_equal(fft.fftshift(freqs, axes=None), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=None), freqs)
        assert_array_almost_equal(fft.fftshift(freqs), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both), freqs)
    def test_equal_to_original(self):
        """Test that the new (>=v1.15) implementation (see #10073) is equal to the original (<=v1.14)"""
        # 如果使用 TorchDynamo 进行测试，则导入相关函数和模块
        if TEST_WITH_TORCHDYNAMO:
            from numpy import arange, asarray, concatenate, take
        else:
            # 否则使用 Torch 内部的 numpy 替代品
            from torch._numpy import arange, asarray, concatenate, take

        def original_fftshift(x, axes=None):
            """How fftshift was implemented in v1.14"""
            # 将输入数组转换为 numpy 数组
            tmp = asarray(x)
            # 获取数组的维度
            ndim = tmp.ndim
            # 如果未指定轴，则默认使用所有维度的索引
            if axes is None:
                axes = list(range(ndim))
            # 如果指定的轴是一个整数，则转换为元组
            elif isinstance(axes, int):
                axes = (axes,)
            # 初始 y 为输入数组
            y = tmp
            # 对每一个指定的轴执行 fftshift 操作
            for k in axes:
                n = tmp.shape[k]
                p2 = (n + 1) // 2
                # 创建索引列表以进行 fftshift 操作
                mylist = concatenate((arange(p2, n), arange(p2)))
                # 在指定轴上应用索引列表进行切片
                y = take(y, mylist, k)
            return y

        def original_ifftshift(x, axes=None):
            """How ifftshift was implemented in v1.14"""
            # 将输入数组转换为 numpy 数组
            tmp = asarray(x)
            # 获取数组的维度
            ndim = tmp.ndim
            # 如果未指定轴，则默认使用所有维度的索引
            if axes is None:
                axes = list(range(ndim))
            # 如果指定的轴是一个整数，则转换为元组
            elif isinstance(axes, int):
                axes = (axes,)
            # 初始 y 为输入数组
            y = tmp
            # 对每一个指定的轴执行 ifftshift 操作
            for k in axes:
                n = tmp.shape[k]
                p2 = n - (n + 1) // 2
                # 创建索引列表以进行 ifftshift 操作
                mylist = concatenate((arange(p2, n), arange(p2)))
                # 在指定轴上应用索引列表进行切片
                y = take(y, mylist, k)
            return y

        # create possible 2d array combinations and try all possible keywords
        # compare output to original functions
        # 遍历可能的二维数组组合，并尝试所有可能的关键字
        for i in range(16):
            for j in range(16):
                for axes_keyword in [0, 1, None, (0,), (0, 1)]:
                    # 生成随机的输入数组
                    inp = np.random.rand(i, j)

                    # 断言新实现的 fftshift 函数的输出与原始版本一致
                    assert_array_almost_equal(
                        fft.fftshift(inp, axes_keyword),
                        original_fftshift(inp, axes_keyword),
                    )

                    # 断言新实现的 ifftshift 函数的输出与原始版本一致
                    assert_array_almost_equal(
                        fft.ifftshift(inp, axes_keyword),
                        original_ifftshift(inp, axes_keyword),
                    )
class TestFFTFreq(TestCase):
    def test_definition(self):
        # 定义输入数组 x
        x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
        # 断言：使用 fft.fftfreq 函数计算频率，乘以 9 后与 x 数组近似相等
        assert_array_almost_equal(9 * fft.fftfreq(9), x)
        # 断言：使用 fft.fftfreq 函数计算频率，乘以 9*pi 后与 x 数组近似相等
        assert_array_almost_equal(9 * pi * fft.fftfreq(9, pi), x)
        
        # 重新定义输入数组 x
        x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        # 断言：使用 fft.fftfreq 函数计算频率，乘以 10 后与 x 数组近似相等
        assert_array_almost_equal(10 * fft.fftfreq(10), x)
        # 断言：使用 fft.fftfreq 函数计算频率，乘以 10*pi 后与 x 数组近似相等
        assert_array_almost_equal(10 * pi * fft.fftfreq(10, pi), x)


class TestRFFTFreq(TestCase):
    def test_definition(self):
        # 定义输入数组 x
        x = [0, 1, 2, 3, 4]
        # 断言：使用 fft.rfftfreq 函数计算实数信号的频率，乘以 9 后与 x 数组近似相等
        assert_array_almost_equal(9 * fft.rfftfreq(9), x)
        # 断言：使用 fft.rfftfreq 函数计算实数信号的频率，乘以 9*pi 后与 x 数组近似相等
        assert_array_almost_equal(9 * pi * fft.rfftfreq(9, pi), x)
        
        # 重新定义输入数组 x
        x = [0, 1, 2, 3, 4, 5]
        # 断言：使用 fft.rfftfreq 函数计算实数信号的频率，乘以 10 后与 x 数组近似相等
        assert_array_almost_equal(10 * fft.rfftfreq(10), x)
        # 断言：使用 fft.rfftfreq 函数计算实数信号的频率，乘以 10*pi 后与 x 数组近似相等
        assert_array_almost_equal(10 * pi * fft.rfftfreq(10, pi), x)


class TestIRFFTN(TestCase):
    def test_not_last_axis_success(self):
        # 创建一个复数数组 a，实部和虚部都是随机的
        ar, ai = np.random.random((2, 16, 8, 32))
        a = ar + 1j * ai

        # 定义轴参数为 (-2)
        axes = (-2,)

        # 应当不引发错误：使用 fft.irfftn 函数对复数数组 a 进行逆快速傅里叶变换，指定轴参数为 (-2)
        fft.irfftn(a, axes=axes)


if __name__ == "__main__":
    run_tests()
```