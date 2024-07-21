# `.\pytorch\test\torch_np\numpy_tests\fft\test_pocketfft.py`

```
# Owner(s): ["module: dynamo"]

# 导入必要的库和模块
import functools
import queue
import threading
from unittest import skipIf as skipif, SkipTest  # 从 unittest 中导入 skipIf 和 SkipTest

import pytest  # 导入 pytest 测试框架
from pytest import raises as assert_raises  # 导入 pytest 的 raises 断言函数

# 导入 Torch Dynamo 相关的测试工具和类
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)

if TEST_WITH_TORCHDYNAMO:
    import numpy as np  # 如果 TEST_WITH_TORCHDYNAMO 为真，则导入 numpy 库
    from numpy.random import random  # 导入 numpy 的 random 函数
    from numpy.testing import assert_allclose  # 导入 numpy.testing 的 assert_allclose 函数
    # from numpy.testing import IS_WASM  # 注释掉的导入语句
else:
    import torch._numpy as np  # 如果 TEST_WITH_TORCHDYNAMO 为假，则导入 torch._numpy 库作为 numpy
    from torch._numpy.random import random  # 导入 torch._numpy 的 random 函数
    from torch._numpy.testing import assert_allclose  # 导入 torch._numpy.testing 的 assert_allclose 函数
    # from torch._numpy.testing import IS_WASM  # 注释掉的导入语句

# 定义 skip 函数，使用 functools.partial 将 skipIf 函数的第一个参数设为 True
skip = functools.partial(skipif, True)

# 初始化 IS_WASM 变量为 False
IS_WASM = False


def fft1(x):
    # 计算输入向量 x 的长度
    L = len(x)
    # 计算相位，这里用到了 numpy 的 arange、pi 和 reshape 函数
    phase = -2j * np.pi * (np.arange(L) / L)
    phase = np.arange(L).reshape(-1, 1) * phase
    # 执行快速傅里叶变换，并返回结果
    return np.sum(x * np.exp(phase), axis=1)


class TestFFTShift(TestCase):
    def test_fft_n(self):
        # 断言调用 np.fft.fft 函数抛出 ValueError 或 RuntimeError 异常
        assert_raises((ValueError, RuntimeError), np.fft.fft, [1, 2, 3], 0)


# 使用装饰器 instantiate_parametrized_tests 包装的 TestFFT1D 类
@instantiate_parametrized_tests
class TestFFT1D(TestCase):
    def setUp(self):
        # 设置测试的前置条件，这里是调用父类的 setUp 方法和设置随机种子
        super().setUp()
        np.random.seed(123456)

    def test_identity(self):
        # 测试傅里叶变换的恒等性质
        maxlen = 512
        x = random(maxlen) + 1j * random(maxlen)
        xr = random(maxlen)
        for i in range(1, maxlen):
            # 断言 np.fft.ifft(np.fft.fft(x[0:i])) 得到的结果与 x[0:i] 很接近
            assert_allclose(np.fft.ifft(np.fft.fft(x[0:i])), x[0:i], atol=1e-12)
            # 断言 np.fft.irfft(np.fft.rfft(xr[0:i]), i) 得到的结果与 xr[0:i] 很接近
            assert_allclose(np.fft.irfft(np.fft.rfft(xr[0:i]), i), xr[0:i], atol=1e-12)

    def test_fft(self):
        # 测试自定义的 fft1 函数与 np.fft.fft 函数的结果是否接近
        np.random.seed(1234)
        x = random(30) + 1j * random(30)
        assert_allclose(fft1(x), np.fft.fft(x), atol=3e-5)
        assert_allclose(fft1(x), np.fft.fft(x, norm="backward"), atol=3e-5)
        assert_allclose(fft1(x) / np.sqrt(30), np.fft.fft(x, norm="ortho"), atol=5e-6)
        assert_allclose(fft1(x) / 30.0, np.fft.fft(x, norm="forward"), atol=5e-6)

    @parametrize("norm", (None, "backward", "ortho", "forward"))
    def test_ifft(self, norm):
        # 测试 np.fft.ifft 函数
        x = random(30) + 1j * random(30)
        assert_allclose(x, np.fft.ifft(np.fft.fft(x, norm=norm), norm=norm), atol=1e-6)

        # 确保捕获到正确的错误消息
        # 注意：在 Dynamo 和 eager 模式下，确切的错误消息略有不同
        with pytest.raises((ValueError, RuntimeError), match="Invalid number of"):
            np.fft.ifft([], norm=norm)

    def test_fft2(self):
        # 测试二维情况下的快速傅里叶变换
        x = random((30, 20)) + 1j * random((30, 20))
        assert_allclose(
            np.fft.fft(np.fft.fft(x, axis=1), axis=0), np.fft.fft2(x), atol=1e-6
        )
        assert_allclose(np.fft.fft2(x), np.fft.fft2(x, norm="backward"), atol=1e-6)
        assert_allclose(
            np.fft.fft2(x) / np.sqrt(30 * 20), np.fft.fft2(x, norm="ortho"), atol=1e-6
        )
        assert_allclose(
            np.fft.fft2(x) / (30.0 * 20.0), np.fft.fft2(x, norm="forward"), atol=1e-6
        )
    def test_ifft2(self):
        # 创建一个随机复数数组 x，形状为 (30, 20)
        x = random((30, 20)) + 1j * random((30, 20))
        # 使用 np.fft.ifft 进行两次傅里叶逆变换，分别在 axis=1 和 axis=0 上，与 np.fft.ifft2(x) 的结果进行比较
        assert_allclose(
            np.fft.ifft(np.fft.ifft(x, axis=1), axis=0), np.fft.ifft2(x), atol=1e-6
        )
        # 比较 np.fft.ifft2(x) 的结果与 np.fft.ifft2(x, norm="backward") 的结果
        assert_allclose(np.fft.ifft2(x), np.fft.ifft2(x, norm="backward"), atol=1e-6)
        # 比较 np.fft.ifft2(x) 乘以 sqrt(30 * 20) 与 np.fft.ifft2(x, norm="ortho") 的结果
        assert_allclose(
            np.fft.ifft2(x) * np.sqrt(30 * 20), np.fft.ifft2(x, norm="ortho"), atol=1e-6
        )
        # 比较 np.fft.ifft2(x) 乘以 (30.0 * 20.0) 与 np.fft.ifft2(x, norm="forward") 的结果
        assert_allclose(
            np.fft.ifft2(x) * (30.0 * 20.0), np.fft.ifft2(x, norm="forward"), atol=1e-6
        )

    def test_fftn(self):
        # 创建一个三维随机复数数组 x，形状为 (30, 20, 10)
        x = random((30, 20, 10)) + 1j * random((30, 20, 10))
        # 使用 np.fft.fft 进行三次傅里叶变换，分别在 axis=2, axis=1 和 axis=0 上，与 np.fft.fftn(x) 的结果进行比较
        assert_allclose(
            np.fft.fft(np.fft.fft(np.fft.fft(x, axis=2), axis=1), axis=0),
            np.fft.fftn(x),
            atol=1e-6,
        )
        # 比较 np.fft.fftn(x) 的结果与 np.fft.fftn(x, norm="backward") 的结果
        assert_allclose(np.fft.fftn(x), np.fft.fftn(x, norm="backward"), atol=1e-6)
        # 比较 np.fft.fftn(x) 除以 sqrt(30 * 20 * 10) 与 np.fft.fftn(x, norm="ortho") 的结果
        assert_allclose(
            np.fft.fftn(x) / np.sqrt(30 * 20 * 10),
            np.fft.fftn(x, norm="ortho"),
            atol=1e-6,
        )
        # 比较 np.fft.fftn(x) 除以 (30.0 * 20.0 * 10.0) 与 np.fft.fftn(x, norm="forward") 的结果
        assert_allclose(
            np.fft.fftn(x) / (30.0 * 20.0 * 10.0),
            np.fft.fftn(x, norm="forward"),
            atol=1e-6,
        )

    def test_ifftn(self):
        # 创建一个三维随机复数数组 x，形状为 (30, 20, 10)
        x = random((30, 20, 10)) + 1j * random((30, 20, 10))
        # 使用 np.fft.ifft 进行三次傅里叶逆变换，分别在 axis=2, axis=1 和 axis=0 上，与 np.fft.ifftn(x) 的结果进行比较
        assert_allclose(
            np.fft.ifft(np.fft.ifft(np.fft.ifft(x, axis=2), axis=1), axis=0),
            np.fft.ifftn(x),
            atol=1e-6,
        )
        # 比较 np.fft.ifftn(x) 的结果与 np.fft.ifftn(x, norm="backward") 的结果
        assert_allclose(np.fft.ifftn(x), np.fft.ifftn(x, norm="backward"), atol=1e-6)
        # 比较 np.fft.ifftn(x) 乘以 sqrt(30 * 20 * 10) 与 np.fft.ifftn(x, norm="ortho") 的结果
        assert_allclose(
            np.fft.ifftn(x) * np.sqrt(30 * 20 * 10),
            np.fft.ifftn(x, norm="ortho"),
            atol=1e-6,
        )
        # 比较 np.fft.ifftn(x) 乘以 (30.0 * 20.0 * 10.0) 与 np.fft.ifftn(x, norm="forward") 的结果
        assert_allclose(
            np.fft.ifftn(x) * (30.0 * 20.0 * 10.0),
            np.fft.ifftn(x, norm="forward"),
            atol=1e-6,
        )

    def test_rfft(self):
        # 创建一个一维随机数组 x，长度为 30
        x = random(30)
        # 对于 n 取 x.size 和 2 * x.size
        for n in [x.size, 2 * x.size]:
            # 对于不同的 norm 参数值，比较 np.fft.fft(x, n=n, norm=norm)[: (n // 2 + 1)] 与 np.fft.rfft(x, n=n, norm=norm) 的结果
            for norm in [None, "backward", "ortho", "forward"]:
                assert_allclose(
                    np.fft.fft(x, n=n, norm=norm)[: (n // 2 + 1)],
                    np.fft.rfft(x, n=n, norm=norm),
                    atol=1e-6,
                )
            # 比较 np.fft.rfft(x, n=n) 的结果与 np.fft.rfft(x, n=n, norm="backward") 的结果
            assert_allclose(
                np.fft.rfft(x, n=n), np.fft.rfft(x, n=n, norm="backward"), atol=1e-6
            )
            # 比较 np.fft.rfft(x, n=n) 除以 sqrt(n) 与 np.fft.rfft(x, n=n, norm="ortho") 的结果
            assert_allclose(
                np.fft.rfft(x, n=n) / np.sqrt(n),
                np.fft.rfft(x, n=n, norm="ortho"),
                atol=1e-6,
            )
            # 比较 np.fft.rfft(x, n=n) 除以 n 与 np.fft.rfft(x, n=n, norm="forward") 的结果
            assert_allclose(
                np.fft.rfft(x, n=n) / n, np.fft.rfft(x, n=n, norm="forward"), atol=1e-6
            )
    # 定义一个测试函数，测试逆实数快速傅里叶变换（IRFFT）
    def test_irfft(self):
        # 生成长度为30的随机数组 x
        x = random(30)
        # 断言：x 与其傅里叶变换的逆变换结果非常接近，允许误差为 1e-6
        assert_allclose(x, np.fft.irfft(np.fft.rfft(x)), atol=1e-6)
        # 断言：x 与其傅里叶变换的逆变换（反向归一化）结果非常接近，允许误差为 1e-6
        assert_allclose(
            x, np.fft.irfft(np.fft.rfft(x, norm="backward"), norm="backward"), atol=1e-6
        )
        # 断言：x 与其傅里叶变换的逆变换（正交归一化）结果非常接近，允许误差为 1e-6
        assert_allclose(
            x, np.fft.irfft(np.fft.rfft(x, norm="ortho"), norm="ortho"), atol=1e-6
        )
        # 断言：x 与其傅里叶变换的逆变换（前向归一化）结果非常接近，允许误差为 1e-6
        assert_allclose(
            x, np.fft.irfft(np.fft.rfft(x, norm="forward"), norm="forward"), atol=1e-6
        )

    # 定义一个测试函数，测试二维实数快速傅里叶变换（RFFT2）
    def test_rfft2(self):
        # 生成形状为 (30, 20) 的随机数组 x
        x = random((30, 20))
        # 断言：x 的二维傅里叶变换（FFT2）的部分结果与二维实数傅里叶变换（RFFT2）结果非常接近，允许误差为 1e-6
        assert_allclose(np.fft.fft2(x)[:, :11], np.fft.rfft2(x), atol=1e-6)
        # 断言：x 的二维实数傅里叶变换（RFFT2）结果与其反向归一化的结果非常接近，允许误差为 1e-6
        assert_allclose(np.fft.rfft2(x), np.fft.rfft2(x, norm="backward"), atol=1e-6)
        # 断言：x 的二维实数傅里叶变换（RFFT2）结果除以其尺寸的平方根，与其正交归一化的结果非常接近，允许误差为 1e-6
        assert_allclose(
            np.fft.rfft2(x) / np.sqrt(30 * 20),
            np.fft.rfft2(x, norm="ortho"),
            atol=1e-6,
        )
        # 断言：x 的二维实数傅里叶变换（RFFT2）结果除以其尺寸乘积，与其前向归一化的结果非常接近，允许误差为 1e-6
        assert_allclose(
            np.fft.rfft2(x) / (30.0 * 20.0), np.fft.rfft2(x, norm="forward"), atol=1e-6
        )

    # 定义一个测试函数，测试二维逆实数快速傅里叶变换（IRFFT2）
    def test_irfft2(self):
        # 生成形状为 (30, 20) 的随机数组 x
        x = random((30, 20))
        # 断言：x 与其二维傅里叶变换的逆变换结果非常接近，允许误差为 1e-6
        assert_allclose(x, np.fft.irfft2(np.fft.rfft2(x)), atol=1e-6)
        # 断言：x 与其二维傅里叶变换的逆变换（反向归一化）结果非常接近，允许误差为 1e-6
        assert_allclose(
            x,
            np.fft.irfft2(np.fft.rfft2(x, norm="backward"), norm="backward"),
            atol=1e-6,
        )
        # 断言：x 与其二维傅里叶变换的逆变换（正交归一化）结果非常接近，允许误差为 1e-6
        assert_allclose(
            x, np.fft.irfft2(np.fft.rfft2(x, norm="ortho"), norm="ortho"), atol=1e-6
        )
        # 断言：x 与其二维傅里叶变换的逆变换（前向归一化）结果非常接近，允许误差为 1e-6
        assert_allclose(
            x, np.fft.irfft2(np.fft.rfft2(x, norm="forward"), norm="forward"), atol=1e-6
        )

    # 定义一个测试函数，测试多维实数快速傅里叶变换（RFFTN）
    def test_rfftn(self):
        # 生成形状为 (30, 20, 10) 的随机数组 x
        x = random((30, 20, 10))
        # 断言：x 的多维傅里叶变换（FFTN）的部分结果与多维实数傅里叶变换（RFFTN）结果非常接近，允许误差为 1e-6
        assert_allclose(np.fft.fftn(x)[:, :, :6], np.fft.rfftn(x), atol=1e-6)
        # 断言：x 的多维实数傅里叶变换（RFFTN）结果与其反向归一化的结果非常接近，允许误差为 1e-6
        assert_allclose(np.fft.rfftn(x), np.fft.rfftn(x, norm="backward"), atol=1e-6)
        # 断言：x 的多维实数傅里叶变换（RFFTN）结果除以其尺寸的平方根，与其正交归一化的结果非常接近，允许误差为 1e-6
        assert_allclose(
            np.fft.rfftn(x) / np.sqrt(30 * 20 * 10),
            np.fft.rfftn(x, norm="ortho"),
            atol=1e-6,
        )
        # 断言：x 的多维实数傅里叶变换（RFFTN）结果除以其尺寸乘积，与其前向归一化的结果非常接近，允许误差为 1e-6
        assert_allclose(
            np.fft.rfftn(x) / (30.0 * 20.0 * 10.0),
            np.fft.rfftn(x, norm="forward"),
            atol=1e-6,
        )

    # 定义一个测试函数，测试多维逆实数快速傅里叶变换（IRFFTN）
    def test_irfftn(self):
        # 生成形状为 (30, 20, 10) 的随机数组 x
        x = random((30, 20, 10))
        # 断言：x 与其多维傅里叶变换的逆变换结果非常接近，允许误差为 1e-6
        assert_allclose(x, np.fft.irfftn(np.fft.rfftn(x)), atol=1e-6)
        # 断言：x 与其多维傅里叶变换的逆变换（反向归一化）结果非常接近，允许误差为 1e-6
        assert_allclose(
            x,
            np.fft.irfftn(np.fft.rfftn(x, norm="backward"), norm="backward"),
            atol=1e-6,
        )
        # 断言：x 与其多维傅里叶变换的逆变换（正交归一化）结果非常接近
    # 定义一个测试函数，用于测试 np.fft.hfft 函数的功能
    def test_hfft(self):
        # 创建一个包含随机复数的长度为 14 的数组
        x = random(14) + 1j * random(14)
        # 在数组前后各加一个随机复数，形成厄米特对称的数组
        x_herm = np.concatenate((random(1), x, random(1)))
        # 将数组 x_herm 与其翻转并共轭的结果拼接，形成输入数组 x
        x = np.concatenate((x_herm, np.flip(x).conj()))
        # 断言 np.fft.fft(x) 和 np.fft.hfft(x_herm) 的结果在给定精度内相等
        assert_allclose(np.fft.fft(x), np.fft.hfft(x_herm), atol=1e-6)
        # 断言 np.fft.hfft(x_herm) 和 np.fft.hfft(x_herm, norm="backward") 的结果在给定精度内相等
        assert_allclose(
            np.fft.hfft(x_herm), np.fft.hfft(x_herm, norm="backward"), atol=1e-6
        )
        # 断言 np.fft.hfft(x_herm) 除以 sqrt(30) 和 np.fft.hfft(x_herm, norm="ortho") 的结果在给定精度内相等
        assert_allclose(
            np.fft.hfft(x_herm) / np.sqrt(30),
            np.fft.hfft(x_herm, norm="ortho"),
            atol=1e-6,
        )
        # 断言 np.fft.hfft(x_herm) 除以 30.0 和 np.fft.hfft(x_herm, norm="forward") 的结果在给定精度内相等
        assert_allclose(
            np.fft.hfft(x_herm) / 30.0, np.fft.hfft(x_herm, norm="forward"), atol=1e-6
        )

    # 定义一个测试函数，用于测试 np.fft.ihfft 函数的功能
    def test_ihfft(self):
        # 创建一个包含随机复数的长度为 14 的数组
        x = random(14) + 1j * random(14)
        # 在数组前后各加一个随机复数，形成厄米特对称的数组
        x_herm = np.concatenate((random(1), x, random(1)))
        # 将数组 x_herm 与其翻转并共轭的结果拼接，形成输入数组 x
        x = np.concatenate((x_herm, np.flip(x).conj()))
        # 断言 np.fft.ihfft(np.fft.hfft(x_herm)) 的结果与 x_herm 在给定精度内相等
        assert_allclose(x_herm, np.fft.ihfft(np.fft.hfft(x_herm)), atol=1e-6)
        # 断言 np.fft.ihfft(np.fft.hfft(x_herm, norm="backward"), norm="backward") 的结果与 x_herm 在给定精度内相等
        assert_allclose(
            x_herm,
            np.fft.ihfft(np.fft.hfft(x_herm, norm="backward"), norm="backward"),
            atol=1e-6,
        )
        # 断言 np.fft.ihfft(np.fft.hfft(x_herm, norm="ortho"), norm="ortho") 的结果与 x_herm 在给定精度内相等
        assert_allclose(
            x_herm,
            np.fft.ihfft(np.fft.hfft(x_herm, norm="ortho"), norm="ortho"),
            atol=1e-6,
        )
        # 断言 np.fft.ihfft(np.fft.hfft(x_herm, norm="forward"), norm="forward") 的结果与 x_herm 在给定精度内相等
        assert_allclose(
            x_herm,
            np.fft.ihfft(np.fft.hfft(x_herm, norm="forward"), norm="forward"),
            atol=1e-6,
        )

    # 定义一个测试函数，用于测试多维数组在不同轴上的 FFT 和 IFFT 操作是否正确
    @parametrize("op", [np.fft.fftn, np.fft.ifftn, np.fft.rfftn, np.fft.irfftn])
    def test_axes(self, op):
        # 创建一个形状为 (30, 20, 10) 的随机多维数组
        x = random((30, 20, 10))
        # 定义多个轴的排列方式
        axes = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        for a in axes:
            # 对数组 x 按照轴 a 进行转置，然后对转置后的结果进行 op 操作
            op_tr = op(np.transpose(x, a))
            # 先对数组 x 进行 op 操作，然后再按照轴 a 进行转置
            tr_op = np.transpose(op(x, axes=a), a)
            # 断言转置后的结果 op_tr 和 tr_op 在给定精度内相等
            assert_allclose(op_tr, tr_op, atol=1e-6)

    # 定义一个测试函数，验证所有一维 FFT 和 IFFT 转换是否保持向量的范数不变
    def test_all_1d_norm_preserving(self):
        # 创建一个长度为 30 的随机数组 x
        x = random(30)
        # 计算数组 x 的范数
        x_norm = np.linalg.norm(x)
        # 定义需要测试的函数对 (forward, backward) 和 (ortho, forward) 的组合
        func_pairs = [
            (np.fft.fft, np.fft.ifft),
            (np.fft.rfft, np.fft.irfft),
            # hfft: 顺序是第一个函数取 x.size 个样本（与上面的 x_norm 比较必要）
            (np.fft.ihfft, np.fft.hfft),
        ]
        for forw, back in func_pairs:
            for n in [x.size, 2 * x.size]:
                for norm in [None, "backward", "ortho", "forward"]:
                    # 对数组 x 进行 forward 变换，再进行 backward 变换，验证范数不变
                    tmp = forw(x, n=n, norm=norm)
                    tmp = back(tmp, n=n, norm=norm)
                    assert_allclose(x_norm, np.linalg.norm(tmp), atol=1e-6)

    # 定义一个测试函数，用于验证不同精度输入在转换过程中是否被转换为 64 位精度
    @parametrize("dtype", [np.half, np.single, np.double])
    def test_dtypes(self, dtype):
        # 创建一个长度为 30 的随机数组 x，并将其类型转换为指定的 dtype
        x = random(30).astype(dtype)
        # 断言对 x 进行 fft 和 ifft 后得到的结果与 x 在给定精度内相等
        assert_allclose(np.fft.ifft(np.fft.fft(x)), x, atol=1e-6)
        # 断言对 x 进行 rfft 和 irfft 后得到的结果与 x 在给定精度内相等
        assert_allclose(np.fft.irfft(np.fft.rfft(x)), x, atol=1e-6)
    # 使用参数化装饰器，依次测试不同数据类型和顺序的 FFT 变换函数
    @parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    @parametrize("order", ["F", "non-contiguous"])
    @parametrize(
        "fft",
        [np.fft.fft, np.fft.fft2, np.fft.fftn, np.fft.ifft, np.fft.ifft2, np.fft.ifftn],
    )
    # 定义 FFT 测试函数，接受数据类型、顺序和 FFT 函数作为参数
    def test_fft_with_order(self, dtype, order, fft):
        # 检查 FFT/IFFT 是否对 C、Fortran 和非连续数组产生相同结果
        #   rng = np.random.RandomState(42)
        rng = np.random  # 使用默认随机数生成器
        # 创建一个形状为 (8, 7, 13) 的随机数组 X，并将其类型转换为指定的 dtype
        X = rng.rand(8, 7, 13).astype(dtype)  # , copy=False)
        # 根据讨论在 pull/14178 中的内容，计算容差 _tol
        _tol = float(8.0 * np.sqrt(np.log2(X.size)) * np.finfo(X.dtype).eps)
        
        if order == "F":
            # 如果顺序为 'F'，跳过测试（因为是 Fortran 排序的数组）
            raise SkipTest("Fortran order arrays")
            Y = np.asfortranarray(X)  # 将 X 转换为 Fortran 排序的数组
        else:
            # 创建一个非连续的数组 Z，并用 X 的值填充其中的偶数索引位置
            Z = np.empty((16, 7, 13), dtype=X.dtype)
            Z[::2] = X
            Y = Z[::2]  # Y 是 Z 的偶数索引位置的视图
            X = Y.copy()  # 将 X 更新为 Y 的副本
        
        if fft.__name__.endswith("fft"):
            # 如果 FFT 函数以 "fft" 结尾，对每个轴执行 FFT 变换
            for axis in range(3):
                X_res = fft(X, axis=axis)  # 对 X 执行 FFT 变换，指定轴
                Y_res = fft(Y, axis=axis)  # 对 Y 执行 FFT 变换，指定轴
                # 断言 X_res 和 Y_res 之间的近似性，使用指定的容差 _tol
                assert_allclose(X_res, Y_res, atol=_tol, rtol=_tol)
        elif fft.__name__.endswith(("fft2", "fftn")):
            # 如果 FFT 函数以 "fft2" 或 "fftn" 结尾，对指定的轴组合执行 FFT 变换
            axes = [(0, 1), (1, 2), (0, 2)]
            if fft.__name__.endswith("fftn"):
                axes.extend([(0,), (1,), (2,), None])
            for ax in axes:
                X_res = fft(X, axes=ax)  # 对 X 执行 FFT 变换，指定轴组合
                Y_res = fft(Y, axes=ax)  # 对 Y 执行 FFT 变换，指定轴组合
                # 断言 X_res 和 Y_res 之间的近似性，使用指定的容差 _tol
                assert_allclose(X_res, Y_res, atol=_tol, rtol=_tol)
        else:
            # 如果 FFT 函数名称不符合预期，抛出 ValueError
            raise ValueError
# 如果运行环境是 WebAssembly，跳过该测试类，因为无法启动线程
@skipif(IS_WASM, reason="Cannot start thread")
# 定义一个测试类 TestFFTThreadSafe，继承自 TestCase
class TestFFTThreadSafe(TestCase):
    # 线程数
    threads = 16
    # 输入数据的形状
    input_shape = (800, 200)

    # 定义一个内部方法 _test_mtsame，用于多线程测试同一函数
    def _test_mtsame(self, func, *args):
        # 定义一个工作线程函数，将函数的返回值放入队列 q 中
        def worker(args, q):
            q.put(func(*args))

        # 创建一个队列 q
        q = queue.Queue()
        # 计算预期结果
        expected = func(*args)

        # 启动 threads 个线程，每个线程调用 worker 方法
        t = [
            threading.Thread(target=worker, args=(args, q)) for i in range(self.threads)
        ]
        # 启动所有线程
        [x.start() for x in t]

        # 等待所有线程结束
        [x.join() for x in t]

        # 确保所有线程返回了正确的值
        for i in range(self.threads):
            # 在 torch.dynamo 下，使用 assert_array_equal 会因相对误差约为 1.5e-14 而失败。
            # 因此替换为 assert_allclose(..., rtol=2e-14)
            assert_allclose(
                q.get(timeout=5),
                expected,
                atol=2e-14
                # msg="Function returned wrong value in multithreaded context",
            )

    # 测试 FFT 函数
    def test_fft(self):
        # 创建一个复数数组 a，全为 1
        a = np.ones(self.input_shape) * 1 + 0j
        # 测试 _test_mtsame 方法，传入 np.fft.fft 函数和数组 a
        self._test_mtsame(np.fft.fft, a)

    # 测试 IFFT 函数
    def test_ifft(self):
        # 创建一个复数数组 a，全为 1
        a = np.ones(self.input_shape) * 1 + 0j
        # 测试 _test_mtsame 方法，传入 np.fft.ifft 函数和数组 a
        self._test_mtsame(np.fft.ifft, a)

    # 测试 Real FFT 函数
    def test_rfft(self):
        # 创建一个实数数组 a，全为 1
        a = np.ones(self.input_shape)
        # 测试 _test_mtsame 方法，传入 np.fft.rfft 函数和数组 a
        self._test_mtsame(np.fft.rfft, a)

    # 测试 Inverse Real FFT 函数
    def test_irfft(self):
        # 创建一个复数数组 a，全为 1
        a = np.ones(self.input_shape) * 1 + 0j
        # 测试 _test_mtsame 方法，传入 np.fft.irfft 函数和数组 a
        self._test_mtsame(np.fft.irfft, a)


# 如果当前脚本被直接执行，则运行所有测试
if __name__ == "__main__":
    run_tests()
```