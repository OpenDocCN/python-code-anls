# `.\pytorch\test\torch_np\numpy_tests\core\test_einsum.py`

```py
# Owner(s): ["module: dynamo"]

# 导入 functools 模块，用于创建偏函数
import functools
# 导入 itertools 模块，用于创建迭代器相关操作
import itertools

# 导入 unittest 模块的 expectedFailure 别名为 xfail，
# 以及 skipIf 别名为 skipif，和 SkipTest 异常类
from unittest import expectedFailure as xfail, skipIf as skipif, SkipTest

# 导入 pytest 的 raises 别名为 assert_raises
from pytest import raises as assert_raises

# 导入 torch._numpy 模块别名为 np
import torch._numpy as np
# 从 torch._numpy.testing 导入多个测试函数和装饰器
from torch._numpy.testing import (
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
    suppress_warnings,
)
# 从 torch.testing._internal.common_utils 导入多个实用函数和类
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

# 定义 functools.partial 函数的别名 skip，用于条件性跳过测试
skip = functools.partial(skipif, True)

# 设置 optimize einsum 的字符集合和对应的尺寸数组
chars = "abcdefghij"
sizes = np.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3])
global_size_dict = dict(zip(chars, sizes))

# 使用 instantiate_parametrized_tests 装饰器创建参数化测试的测试类 TestEinsum
@instantiate_parametrized_tests
class TestEinsum(TestCase):
    # 标记该测试函数为预期失败，原因是一个视图转换到其他内容
    @xfail  # (reason="a view into smth else")
    # 标记该测试函数为预期失败，原因是在 numpy 和 pytorch 中整数溢出不同
    @xfail  # (reason="int overflow differs in numpy and pytorch")
    def test_einsum_sums_int8(self):
        # 调用父类方法检查 einsum 对于 int8 类型的求和操作
        self.check_einsum_sums("i1")

    # 标记该测试函数为预期失败，原因是在 numpy 和 pytorch 中整数溢出不同
    @xfail  # (reason="int overflow differs in numpy and pytorch")
    def test_einsum_sums_uint8(self):
        # 调用父类方法检查 einsum 对于 uint8 类型的求和操作
        self.check_einsum_sums("u1")

    # 标记该测试函数为预期失败，原因是在 numpy 和 pytorch 中整数溢出不同
    @xfail  # (reason="int overflow differs in numpy and pytorch")
    def test_einsum_sums_int16(self):
        # 调用父类方法检查 einsum 对于 int16 类型的求和操作
        self.check_einsum_sums("i2")

    # 测试函数，检查 einsum 对于 int32 类型的求和操作
    def test_einsum_sums_int32(self):
        self.check_einsum_sums("i4")   # 调用父类方法检查非累加和
        self.check_einsum_sums("i4", True)   # 调用父类方法检查累加和

    # 测试函数，检查 einsum 对于 int64 类型的求和操作
    def test_einsum_sums_int64(self):
        self.check_einsum_sums("i8")   # 调用父类方法检查非累加和

    # 标记该测试函数为预期失败，原因是 np.float16(4641) == 4640.0
    @xfail  # (reason="np.float16(4641) == 4640.0")
    def test_einsum_sums_float16(self):
        # 调用父类方法检查 einsum 对于 float16 类型的求和操作
        self.check_einsum_sums("f2")

    # 测试函数，检查 einsum 对于 float32 类型的求和操作
    def test_einsum_sums_float32(self):
        self.check_einsum_sums("f4")   # 调用父类方法检查非累加和

    # 测试函数，检查 einsum 对于 float64 类型的求和操作
    def test_einsum_sums_float64(self):
        self.check_einsum_sums("f8")   # 调用父类方法检查非累加和
        self.check_einsum_sums("f8", True)   # 调用父类方法检查累加和

    # 测试函数，检查 einsum 对于 cfloat64 类型的求和操作
    def test_einsum_sums_cfloat64(self):
        self.check_einsum_sums("c8")   # 调用父类方法检查非累加和
        self.check_einsum_sums("c8", True)   # 调用父类方法检查累加和

    # 测试函数，检查 einsum 对于 cfloat128 类型的求和操作
    def test_einsum_sums_cfloat128(self):
        self.check_einsum_sums("c16")   # 调用父类方法检查非累加和
    def test_einsum_misc(self):
        # 这个调用曾因 PyArray_AssignZero 的 bug 而崩溃
        a = np.ones((1, 2))  # 创建一个形状为 (1, 2) 的全为1的数组 a
        b = np.ones((2, 2, 1))  # 创建一个形状为 (2, 2, 1) 的全为1的数组 b
        assert_equal(np.einsum("ij...,j...->i...", a, b), [[[2], [2]]])  # 使用 einsum 计算张量乘积并比较结果
        assert_equal(np.einsum("ij...,j...->i...", a, b, optimize=True), [[[2], [2]]])  # 使用优化后的 einsum 计算并比较结果

        # 问题 #10369 的回归测试（测试 Python 2 中的 Unicode 输入）
        assert_equal(np.einsum("ij...,j...->i...", a, b), [[[2], [2]]])  # 再次使用 einsum 计算张量乘积并比较结果
        assert_equal(np.einsum("...i,...i", [1, 2, 3], [2, 3, 4]), 20)  # 使用 einsum 计算并比较结果
        assert_equal(
            np.einsum("...i,...i", [1, 2, 3], [2, 3, 4], optimize="greedy"), 20
        )  # 使用贪婪优化的 einsum 计算并比较结果

        # 迭代器在缓冲此规约时存在问题
        a = np.ones((5, 12, 4, 2, 3), np.int64)  # 创建一个形状为 (5, 12, 4, 2, 3) 的全为1的 int64 类型数组 a
        b = np.ones((5, 12, 11), np.int64)  # 创建一个形状为 (5, 12, 11) 的全为1的 int64 类型数组 b
        assert_equal(
            np.einsum("ijklm,ijn,ijn->", a, b, b), np.einsum("ijklm,ijn->", a, b)
        )  # 使用 einsum 计算张量乘积并比较结果
        assert_equal(
            np.einsum("ijklm,ijn,ijn->", a, b, b, optimize=True),
            np.einsum("ijklm,ijn->", a, b, optimize=True),
        )  # 使用优化后的 einsum 计算并比较结果

        # 问题 #2027，内部循环实现在连续的三参数中存在问题
        a = np.arange(1, 3)  # 创建一个值从 1 到 2 的数组 a
        b = np.arange(1, 5).reshape(2, 2)  # 创建一个形状为 (2, 2) 的数组 b
        c = np.arange(1, 9).reshape(4, 2)  # 创建一个形状为 (4, 2) 的数组 c
        assert_equal(
            np.einsum("x,yx,zx->xzy", a, b, c),
            [
                [[1, 3], [3, 9], [5, 15], [7, 21]],
                [[8, 16], [16, 32], [24, 48], [32, 64]],
            ],
        )  # 使用 einsum 计算张量乘积并比较结果
        assert_equal(
            np.einsum("x,yx,zx->xzy", a, b, c, optimize=True),
            [
                [[1, 3], [3, 9], [5, 15], [7, 21]],
                [[8, 16], [16, 32], [24, 48], [32, 64]],
            ],
        )  # 使用优化后的 einsum 计算并比较结果

        # 确保明确设置 out=None 不会导致错误
        # 参见问题 gh-15776 和问题 gh-15256
        assert_equal(np.einsum("i,j", [1], [2], out=None), [[2]])

    def test_subscript_range(self):
        # 问题 #7741，确保从数组创建一个下标时可以使用拉丁字母表的所有字母（大写和小写）
        a = np.ones((2, 3))  # 创建一个形状为 (2, 3) 的全为1的数组 a
        b = np.ones((3, 4))  # 创建一个形状为 (3, 4) 的全为1的数组 b
        np.einsum(a, [0, 20], b, [20, 2], [0, 2], optimize=False)  # 使用 einsum 计算
        np.einsum(a, [0, 27], b, [27, 2], [0, 2], optimize=False)  # 使用 einsum 计算
        np.einsum(a, [0, 51], b, [51, 2], [0, 2], optimize=False)  # 使用 einsum 计算
        assert_raises(
            ValueError,
            lambda: np.einsum(a, [0, 52], b, [52, 2], [0, 2], optimize=False),
        )  # 检查异常是否被引发
        assert_raises(
            ValueError,
            lambda: np.einsum(a, [-1, 5], b, [5, 2], [-1, 2], optimize=False),
        )  # 检查异常是否被引发
    def test_einsum_broadcast(self):
        # 定义测试函数 test_einsum_broadcast
        # 处理 ellipsis 的变化问题，参见 Issue #2455
        # 修复 'middle broadcast' 错误
        # 在 prepare_op_axes 中仅使用 'RIGHT' 迭代
        # 在左侧添加自动广播，属于其正确使用位置
        # 右侧的广播必须显式指定
        # 我们需要测试优化后的解析功能

        A = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        B = np.arange(3)
        ref = np.einsum("ijk,j->ijk", A, B, optimize=False)
        for opt in [True, False]:
            assert_equal(np.einsum("ij...,j...->ij...", A, B, optimize=opt), ref)
            assert_equal(np.einsum("ij...,...j->ij...", A, B, optimize=opt), ref)
            assert_equal(
                np.einsum("ij...,j->ij...", A, B, optimize=opt), ref
            )  # 曾经引发错误

        A = np.arange(12).reshape((4, 3))
        B = np.arange(6).reshape((3, 2))
        ref = np.einsum("ik,kj->ij", A, B, optimize=False)
        for opt in [True, False]:
            assert_equal(np.einsum("ik...,k...->i...", A, B, optimize=opt), ref)
            assert_equal(np.einsum("ik...,...kj->i...j", A, B, optimize=opt), ref)
            assert_equal(
                np.einsum("...k,kj", A, B, optimize=opt), ref
            )  # 曾经引发错误
            assert_equal(
                np.einsum("ik,k...->i...", A, B, optimize=opt), ref
            )  # 曾经引发错误

        dims = [2, 3, 4, 5]
        a = np.arange(np.prod(dims)).reshape(dims)
        v = np.arange(dims[2])
        ref = np.einsum("ijkl,k->ijl", a, v, optimize=False)
        for opt in [True, False]:
            assert_equal(np.einsum("ijkl,k", a, v, optimize=opt), ref)
            assert_equal(
                np.einsum("...kl,k", a, v, optimize=opt), ref
            )  # 曾经引发错误
            assert_equal(np.einsum("...kl,k...", a, v, optimize=opt), ref)

        J, K, M = 160, 160, 120
        A = np.arange(J * K * M).reshape(1, 1, 1, J, K, M)
        B = np.arange(J * K * M * 3).reshape(J, K, M, 3)
        ref = np.einsum("...lmn,...lmno->...o", A, B, optimize=False)
        for opt in [True, False]:
            assert_equal(
                np.einsum("...lmn,lmno->...o", A, B, optimize=opt), ref
            )  # 曾经引发错误
    def test_einsum_fixedstridebug(self):
        # Issue #4485 obscure einsum bug
        # This case revealed a bug in nditer where it reported a stride
        # as 'fixed' (0) when it was in fact not fixed during processing
        # (0 or 4). The reason for the bug was that the check for a fixed
        # stride was using the information from the 2D inner loop reuse
        # to restrict the iteration dimensions it had to validate to be
        # the same, but that 2D inner loop reuse logic is only triggered
        # during the buffer copying step, and hence it was invalid to
        # rely on those values. The fix is to check all the dimensions
        # of the stride in question, which in the test case reveals that
        # the stride is not fixed.
        #
        # NOTE: This test is triggered by the fact that the default buffersize,
        #       used by einsum, is 8192, and 3*2731 = 8193, is larger than that
        #       and results in a mismatch between the buffering and the
        #       striding for operand A.
        
        # 创建一个 2x3 的浮点数数组 A
        A = np.arange(2 * 3).reshape(2, 3).astype(np.float32)
        # 创建一个 2x3x2731 的整数数组 B
        B = np.arange(2 * 3 * 2731).reshape(2, 3, 2731).astype(np.int16)
        # 使用 einsum 函数计算 "cl, cpx->lpx" 并存储结果在 es 中
        es = np.einsum("cl, cpx->lpx", A, B)
        # 使用 tensordot 函数计算 A 和 B 的张量积，轴为 (0, 0)，并存储结果在 tp 中
        tp = np.tensordot(A, B, axes=(0, 0))
        # 断言 es 和 tp 相等
        assert_equal(es, tp)
        
        # 下面是报告中的原始测试用例，通过将随机数组更改为 aranges 来使其可重复
        # 创建一个 3x3 的双精度浮点数数组 A
        A = np.arange(3 * 3).reshape(3, 3).astype(np.float64)
        # 创建一个 3x3x64x64 的单精度浮点数数组 B
        B = np.arange(3 * 3 * 64 * 64).reshape(3, 3, 64, 64).astype(np.float32)
        # 使用 einsum 函数计算 "cl, cpxy->lpxy" 并存储结果在 es 中
        es = np.einsum("cl, cpxy->lpxy", A, B)
        # 使用 tensordot 函数计算 A 和 B 的张量积，轴为 (0, 0)，并存储结果在 tp 中
        tp = np.tensordot(A, B, axes=(0, 0))
        # 断言 es 和 tp 相等
        assert_equal(es, tp)

    def test_einsum_fixed_collapsingbug(self):
        # Issue #5147.
        # The bug only occurred when output argument of einssum was used.
        # 创建一个形状为 (5, 5, 5, 5) 的正态分布随机数组 x
        x = np.random.normal(0, 1, (5, 5, 5, 5))
        # 创建一个形状为 (5, 5) 的全零数组 y1
        y1 = np.zeros((5, 5))
        # 使用 einsum 函数计算 "aabb->ab"，将结果存储在 y1 中
        np.einsum("aabb->ab", x, out=y1)
        # 创建一个索引数组 idx，包含 [0, 1, 2, 3, 4]
        idx = np.arange(5)
        # 根据索引数组 idx，从 x 中获取数据并存储在 y2 中
        y2 = x[idx[:, None], idx[:, None], idx, idx]
        # 断言 y1 和 y2 相等
        assert_equal(y1, y2)

    def test_einsum_failed_on_p9_and_s390x(self):
        # Issues gh-14692 and gh-12689
        # Bug with signed vs unsigned char errored on power9 and s390x Linux
        # 创建一个形状为 (10, 10, 10, 10) 的随机数组 tensor
        tensor = np.random.random_sample((10, 10, 10, 10))
        # 使用 einsum 函数计算 "ijij->"，并将结果存储在 x 中
        x = np.einsum("ijij->", tensor)
        # 计算 tensor 沿轴1和轴2的迹，再计算其迹，将结果存储在 y 中
        y = tensor.trace(axis1=0, axis2=2).trace()
        # 断言 x 和 y 的近似相等性
        assert_allclose(x, y)

    @xfail  # (reason="no base")
    # 定义一个测试函数，测试 `np.einsum` 函数在特定情况下的输出
    def test_einsum_all_contig_non_contig_output(self):
        # 解决问题 gh-5907，测试全连续特殊情况确实检查了输出的连续性
        # 创建一个 5x5 的全 1 数组
        x = np.ones((5, 5))
        # 创建一个长度为 10 的全 1 数组，每隔一个元素取值，得到 out
        out = np.ones(10)[::2]
        # 创建一个长度为 10 的全 1 数组作为正确的基础数组
        correct_base = np.ones(10)
        # 将正确基础数组每隔一个元素设置为 5
        correct_base[::2] = 5
        
        # 使用 np.einsum 函数，计算并将结果存入 out，内部迭代使用 0 步长
        np.einsum("mi,mi,mi->m", x, x, x, out=out)
        # 断言 out 的基础数组与正确基础数组相等
        assert_array_equal(out.base, correct_base)
        
        # 示例 1：
        # 将 out 重新设置为长度为 10 的全 1 数组，每隔一个元素取值
        out = np.ones(10)[::2]
        # 使用 np.einsum 函数，计算并将结果存入 out
        np.einsum("im,im,im->m", x, x, x, out=out)
        # 断言 out 的基础数组与正确基础数组相等
        assert_array_equal(out.base, correct_base)
        
        # 示例 2，缓冲区导致 x 连续，但特殊情况未捕捉到操作前：
        # 创建一个 shape 为 (2, 2, 2) 的数组，取出最后一维的切片作为 out
        out = np.ones((2, 2, 2))[..., 0]
        # 创建一个 shape 为 (2, 2, 2) 的全 1 数组作为正确的基础数组
        correct_base = np.ones((2, 2, 2))
        # 将正确基础数组的最后一维切片设置为 2
        correct_base[..., 0] = 2
        # 创建一个 shape 为 (2, 2) 的全 1 数组 x，并指定其数据类型为 np.float32
        x = np.ones((2, 2), np.float32)
        # 使用 np.einsum 函数，计算并将结果存入 out
        np.einsum("ij,jk->ik", x, x, out=out)
        # 断言 out 的基础数组与正确基础数组相等
        assert_array_equal(out.base, correct_base)

    # 使用参数化装饰器 parametrize，为 np.typecodes 中所有的浮点数和整数类型添加参数化测试
    @parametrize("dtype", np.typecodes["AllFloat"] + np.typecodes["AllInteger"])
    def test_different_paths(self, dtype):
        # Test originally added to cover broken float16 path: gh-20305
        # Likely most are covered elsewhere, at least partially.
        dtype = np.dtype(dtype)
        # 将输入的 dtype 转换为 NumPy 的数据类型对象

        arr = (np.arange(7) + 0.5).astype(dtype)
        # 创建一个长度为 7 的数组，元素为从 0.5 开始的浮点数序列，并转换为指定的 dtype 类型
        scalar = np.array(2, dtype=dtype)
        # 创建一个包含单个元素 2 的数组，数据类型为指定的 dtype

        # contig -> scalar:
        res = np.einsum("i->", arr)
        # 对数组 arr 执行 Einstein Summation 表达式 "i->"，将数组中的元素求和为标量
        assert res == arr.sum()
        # 断言：计算结果 res 应该与 arr.sum() 相等

        # contig, contig -> contig:
        res = np.einsum("i,i->i", arr, arr)
        # 对数组 arr 执行 Einstein Summation 表达式 "i,i->i"，对数组中对应位置的元素进行逐元素乘法
        assert_array_equal(res, arr * arr)
        # 断言：计算结果 res 应该与 arr * arr 相等，即每个元素的平方

        # noncontig, noncontig -> contig:
        res = np.einsum("i,i->i", arr.repeat(2)[::2], arr.repeat(2)[::2])
        # 对数组 arr 的扩展版本进行 Einstein Summation 表达式 "i,i->i" 的计算，对应位置元素逐元素乘法
        assert_array_equal(res, arr * arr)
        # 断言：计算结果 res 应该与 arr * arr 相等

        # contig + contig -> scalar
        assert np.einsum("i,i->", arr, arr) == (arr * arr).sum()
        # 断言：Einstein Summation 表达式 "i,i->" 的结果应该与 (arr * arr).sum() 相等，即两个数组的点积

        # contig + scalar -> contig (with out)
        out = np.ones(7, dtype=dtype)
        # 创建一个包含全为 1 的长度为 7 的数组，数据类型为指定的 dtype
        res = np.einsum("i,->i", arr, dtype.type(2), out=out)
        # 对数组 arr 和标量进行 Einstein Summation 表达式 "i,->i" 的计算，结果存储到指定的 out 数组中
        assert_array_equal(res, arr * dtype.type(2))
        # 断言：计算结果 res 应该与 arr * dtype.type(2) 相等

        # scalar + contig -> contig (with out)
        res = np.einsum(",i->i", scalar, arr)
        # 对标量和数组 arr 执行 Einstein Summation 表达式 ",i->i" 的计算
        assert_array_equal(res, arr * dtype.type(2))
        # 断言：计算结果 res 应该与 arr * dtype.type(2) 相等

        # scalar + contig -> scalar
        res = np.einsum(",i->", scalar, arr)
        # 对标量和数组 arr 执行 Einstein Summation 表达式 ",i->" 的计算
        # Use einsum to compare to not have difference due to sum round-offs:
        assert res == np.einsum("i->", scalar * arr)
        # 断言：计算结果 res 应该与 np.einsum("i->", scalar * arr) 相等，避免由于求和时的舍入误差导致差异

        # contig + scalar -> scalar
        res = np.einsum("i,->", arr, scalar)
        # 对数组 arr 和标量进行 Einstein Summation 表达式 "i,->" 的计算
        # Use einsum to compare to not have difference due to sum round-offs:
        assert res == np.einsum("i->", scalar * arr)
        # 断言：计算结果 res 应该与 np.einsum("i->", scalar * arr) 相等，避免由于求和时的舍入误差导致差异

        # contig + contig + contig -> scalar
        if dtype in ["e", "B", "b"]:
            # FIXME make xfail
            raise SkipTest("overflow differs in pytorch and numpy")
            # 如果 dtype 是 "e", "B", "b" 中的一个，抛出跳过测试的异常，因为在 PyTorch 和 NumPy 中溢出的处理不同

        arr = np.array([0.5, 0.5, 0.25, 4.5, 3.0], dtype=dtype)
        # 创建一个包含特定数据和数据类型的数组

        res = np.einsum("i,i,i->", arr, arr, arr)
        # 对数组 arr 执行 Einstein Summation 表达式 "i,i,i->" 的计算，将数组中元素的乘积求和为标量
        assert_array_equal(res, (arr * arr * arr).sum())
        # 断言：计算结果 res 应该与 (arr * arr * arr).sum() 相等，即数组元素的立方和

        # four arrays:
        res = np.einsum("i,i,i,i->", arr, arr, arr, arr)
        # 对四个数组 arr 执行 Einstein Summation 表达式 "i,i,i,i->" 的计算，将数组中元素的乘积求和为标量
        assert_array_equal(res, (arr * arr * arr * arr).sum())
        # 断言：计算结果 res 应该与 (arr * arr * arr * arr).sum() 相等，即数组元素的四次方和

    def test_small_boolean_arrays(self):
        # See gh-5946.
        # Use array of True embedded in False.
        a = np.zeros((16, 1, 1), dtype=np.bool_)[:2]
        # 创建一个形状为 (2, 1, 1) 的布尔类型的数组，初始值为 False
        a[...] = True
        # 将数组 a 中所有元素的值设置为 True
        out = np.zeros((16, 1, 1), dtype=np.bool_)[:2]
        # 创建一个形状为 (2, 1, 1) 的布尔类型的数组，初始值为 False
        tgt = np.ones((2, 1, 1), dtype=np.bool_)
        # 创建一个形状为 (2, 1, 1) 的布尔类型的数组，所有元素为 True
        res = np.einsum("...ij,...jk->...ik", a, a, out=out)
        # 对两个数组 a 和 a 执行 Einstein Summation 表达式 "...ij,...jk->...ik" 的计算，结果存储到 out 数组中
        assert_equal(res, tgt)
        # 断言：计算结果 res 应该与 tgt 相等，即所有元素都为 True 的布尔类型的数组

    def test_out_is_res(self):
        a = np.arange(9).reshape(3, 3)
        # 创建一个形状为 (3, 3) 的数组，元素值为 0 到 8 的整数
        res = np.einsum("...ij,...jk->...ik", a, a, out=a)
        # 对两个数组 a 和 a 执行 Einstein Summation 表达式 "...ij,...jk->...ik" 的计算，将结果存储到数组 a 中
        assert res is a
        # 断言：计算结果 res 应该与数组 a 相同
    def optimize_compare(self, subscripts, operands=None):
        # 对优化函数的所有路径进行测试，与常规的 einsum 进行比较
        
        # 如果没有给定操作数，则根据子串创建参数列表
        if operands is None:
            args = [subscripts]
            # 解析子串，获取其中的各项
            terms = subscripts.split("->")[0].split(",")
            for term in terms:
                # 根据全局大小字典获取各维度大小，并生成随机数组
                dims = [global_size_dict[x] for x in term]
                args.append(np.random.rand(*dims))
        else:
            # 如果给定了操作数，则将子串与操作数一起作为参数列表
            args = [subscripts] + operands

        # 在不优化的情况下执行 einsum 运算
        noopt = np.einsum(*args, optimize=False)
        # 使用贪婪优化策略执行 einsum 运算
        opt = np.einsum(*args, optimize="greedy")
        # 断言贪婪优化后的结果与未优化结果几乎相等
        assert_almost_equal(opt, noopt)
        # 使用最优化策略执行 einsum 运算
        opt = np.einsum(*args, optimize="optimal")
        # 断言最优化后的结果与未优化结果几乎相等
        assert_almost_equal(opt, noopt)

    def test_hadamard_like_products(self):
        # 测试哈达玛积类似的外积
        
        # 调用 optimize_compare 方法进行测试
        self.optimize_compare("a,ab,abc->abc")
        self.optimize_compare("a,b,ab->ab")

    def test_index_transformations(self):
        # 测试简单的索引变换情况
        
        # 调用 optimize_compare 方法进行测试
        self.optimize_compare("ea,fb,gc,hd,abcd->efgh")
        self.optimize_compare("ea,fb,abcd,gc,hd->efgh")
        self.optimize_compare("abcd,ea,fb,gc,hd->efgh")

    def test_complex(self):
        # 测试长的测试用例
        
        # 调用 optimize_compare 方法进行测试
        self.optimize_compare("acdf,jbje,gihb,hfac,gfac,gifabc,hfac")
        self.optimize_compare("acdf,jbje,gihb,hfac,gfac,gifabc,hfac")
        self.optimize_compare("cd,bdhe,aidb,hgca,gc,hgibcd,hgac")
        self.optimize_compare("abhe,hidj,jgba,hiab,gab")
        self.optimize_compare("bde,cdh,agdb,hica,ibd,hgicd,hiac")
        self.optimize_compare("chd,bde,agbc,hiad,hgc,hgi,hiad")
        self.optimize_compare("chd,bde,agbc,hiad,bdi,cgh,agdb")
        self.optimize_compare("bdhe,acad,hiab,agac,hibd")

    def test_collapse(self):
        # 测试内积
        
        # 调用 optimize_compare 方法进行测试
        self.optimize_compare("ab,ab,c->")
        self.optimize_compare("ab,ab,c->c")
        self.optimize_compare("ab,ab,cd,cd->")
        self.optimize_compare("ab,ab,cd,cd->ac")
        self.optimize_compare("ab,ab,cd,cd->cd")
        self.optimize_compare("ab,ab,cd,cd,ef,ef->")

    def test_expand(self):
        # 测试外积
        
        # 调用 optimize_compare 方法进行测试
        self.optimize_compare("ab,cd,ef->abcdef")
        self.optimize_compare("ab,cd,ef->acdf")
        self.optimize_compare("ab,cd,de->abcde")
        self.optimize_compare("ab,cd,de->be")
        self.optimize_compare("ab,bcd,cd->abcd")
        self.optimize_compare("ab,bcd,cd->abd")
    # 测试边界情况的方法
    def test_edge_cases(self):
        # 对优化算法具有挑战性的边界情况
        self.optimize_compare("eb,cb,fb->cef")
        self.optimize_compare("dd,fb,be,cdb->cef")
        self.optimize_compare("bca,cdb,dbf,afc->")
        self.optimize_compare("dcc,fce,ea,dbf->ab")
        self.optimize_compare("fdf,cdd,ccd,afe->ae")
        self.optimize_compare("abcd,ad")
        self.optimize_compare("ed,fcd,ff,bcf->be")
        self.optimize_compare("baa,dcf,af,cde->be")
        self.optimize_compare("bd,db,eac->ace")
        self.optimize_compare("fff,fae,bef,def->abd")
        self.optimize_compare("efc,dbc,acf,fd->abe")
        self.optimize_compare("ba,ac,da->bcd")

    # 测试内积操作的方法
    def test_inner_product(self):
        # 内积操作
        self.optimize_compare("ab,ab")
        self.optimize_compare("ab,ba")
        self.optimize_compare("abc,abc")
        self.optimize_compare("abc,bac")
        self.optimize_compare("abc,cba")

    # 测试随机生成的用例
    def test_random_cases(self):
        # 随机生成的测试用例
        self.optimize_compare("aab,fa,df,ecc->bde")
        self.optimize_compare("ecb,fef,bad,ed->ac")
        self.optimize_compare("bcf,bbb,fbf,fc->")
        self.optimize_compare("bb,ff,be->e")
        self.optimize_compare("bcb,bb,fc,fff->")
        self.optimize_compare("fbb,dfd,fc,fc->")
        self.optimize_compare("afd,ba,cc,dc->bf")
        self.optimize_compare("adb,bc,fa,cfc->d")
        self.optimize_compare("bbd,bda,fc,db->acf")
        self.optimize_compare("dba,ead,cad->bce")
        self.optimize_compare("aef,fbc,dca->bde")

    # 测试组合视图映射的方法
    def test_combined_views_mapping(self):
        # GitHub issue #10792
        a = np.arange(9).reshape(1, 1, 3, 1, 3)
        b = np.einsum("bbcdc->d", a)
        assert_equal(b, [12])

    # 测试广播点积的用例
    def test_broadcasting_dot_cases(self):
        # 确保广播情况不会被误认为是通用矩阵乘法

        # 创建随机数组
        a = np.random.rand(1, 5, 4)
        b = np.random.rand(4, 6)
        c = np.random.rand(5, 6)
        d = np.random.rand(10)

        # 执行优化比较
        self.optimize_compare("ijk,kl,jl", operands=[a, b, c])
        self.optimize_compare("ijk,kl,jl,i->i", operands=[a, b, c, d])

        # 创建另一组随机数组
        e = np.random.rand(1, 1, 5, 4)
        f = np.random.rand(7, 7)

        # 执行优化比较
        self.optimize_compare("abjk,kl,jl", operands=[e, b, c])
        self.optimize_compare("abjk,kl,jl,ab->ab", operands=[e, b, c, f])

        # GitHub issue #11308 中的边缘情况
        g = np.arange(64).reshape(2, 4, 8)
        self.optimize_compare("obk,ijk->ioj", operands=[g, g])

    @xfail  # (reason="order='F' not supported")
    def test_output_order(self):
        # 定义一个测试方法，用于验证输出顺序在优化情况下的正确性，以下压缩应该产生一个重塑的张量视图
        # gh-16415

        # 创建一个列优先顺序（Fortran顺序）的全1张量a，形状为(2, 3, 5)
        a = np.ones((2, 3, 5), order="F")
        # 创建一个列优先顺序（Fortran顺序）的全1张量b，形状为(4, 3)
        b = np.ones((4, 3), order="F")

        # 遍历优化选项True和False
        for opt in [True, False]:
            # 使用einsum函数计算张量乘积，结果应该是Fortran顺序的
            tmp = np.einsum("...ft,mf->...mt", a, b, order="a", optimize=opt)
            assert_(tmp.flags.f_contiguous)

            # 使用einsum函数计算张量乘积，结果应该是Fortran顺序的
            tmp = np.einsum("...ft,mf->...mt", a, b, order="f", optimize=opt)
            assert_(tmp.flags.f_contiguous)

            # 使用einsum函数计算张量乘积，结果应该是C顺序的
            tmp = np.einsum("...ft,mf->...mt", a, b, order="c", optimize=opt)
            assert_(tmp.flags.c_contiguous)

            # 使用einsum函数计算张量乘积，结果应该是C顺序的
            tmp = np.einsum("...ft,mf->...mt", a, b, order="k", optimize=opt)
            assert_(tmp.flags.c_contiguous is False)
            assert_(tmp.flags.f_contiguous is False)

            # 使用einsum函数计算张量乘积，结果应该既不是C顺序也不是Fortran顺序的
            tmp = np.einsum("...ft,mf->...mt", a, b, optimize=opt)
            assert_(tmp.flags.c_contiguous is False)
            assert_(tmp.flags.f_contiguous is False)

        # 创建一个行优先顺序（C顺序）的全1张量c，形状为(4, 3)
        c = np.ones((4, 3), order="C")
        # 遍历优化选项True和False
        for opt in [True, False]:
            # 使用einsum函数计算张量乘积，结果应该是C顺序的
            tmp = np.einsum("...ft,mf->...mt", a, c, order="a", optimize=opt)
            assert_(tmp.flags.c_contiguous)

        # 创建一个行优先顺序（C顺序）的全1张量d，形状为(2, 3, 5)
        d = np.ones((2, 3, 5), order="C")
        # 遍历优化选项True和False
        for opt in [True, False]:
            # 使用einsum函数计算张量乘积，结果应该是C顺序的
            tmp = np.einsum("...ft,mf->...mt", d, c, order="a", optimize=opt)
            assert_(tmp.flags.c_contiguous)
@skip(reason="no pytorch analog")
class TestEinsumPath(TestCase):
    # 定义一个测试类 TestEinsumPath，继承自 TestCase

    def build_operands(self, string, size_dict=global_size_dict):
        # 根据给定字符串构建操作数列表
        # 初始操作数列表包含字符串本身
        operands = [string]
        # 将字符串按 "->" 分割，取第一部分再按逗号分割得到各个操作数的名称
        terms = string.split("->")[0].split(",")
        # 根据操作数名称从 size_dict 中获取维度大小，并随机生成对应维度的数组作为操作数添加到 operands 中
        for term in terms:
            dims = [size_dict[x] for x in term]
            operands.append(np.random.rand(*dims))

        return operands
        # 返回构建好的操作数列表

    def assert_path_equal(self, comp, benchmark):
        # 断言路径是否相等的方法

        # 检查 comp 和 benchmark 的长度是否相等
        ret = len(comp) == len(benchmark)
        assert_(ret)
        # 遍历比较每一个位置的元素是否相等
        for pos in range(len(comp) - 1):
            ret &= isinstance(comp[pos + 1], tuple)
            ret &= comp[pos + 1] == benchmark[pos + 1]
        assert_(ret)

    def test_memory_contraints(self):
        # 测试内存约束是否满足的方法

        # 构建简单的测试操作数
        outer_test = self.build_operands("a,b,c->abc")

        # 使用 "greedy" 算法优化计算路径，并检查路径是否符合预期
        path, path_str = np.einsum_path(*outer_test, optimize=("greedy", 0))
        self.assert_path_equal(path, ["einsum_path", (0, 1, 2)])

        # 使用 "optimal" 算法优化计算路径，并检查路径是否符合预期
        path, path_str = np.einsum_path(*outer_test, optimize=("optimal", 0))
        self.assert_path_equal(path, ["einsum_path", (0, 1, 2)])

        # 构建复杂的测试操作数
        long_test = self.build_operands("acdf,jbje,gihb,hfac")

        # 使用 "greedy" 算法优化计算路径，并检查路径是否符合预期
        path, path_str = np.einsum_path(*long_test, optimize=("greedy", 0))
        self.assert_path_equal(path, ["einsum_path", (0, 1, 2, 3)])

        # 使用 "optimal" 算法优化计算路径，并检查路径是否符合预期
        path, path_str = np.einsum_path(*long_test, optimize=("optimal", 0))
        self.assert_path_equal(path, ["einsum_path", (0, 1, 2, 3)])

    def test_long_paths(self):
        # 测试长且复杂路径的方法

        # 长测试 1
        long_test1 = self.build_operands("acdf,jbje,gihb,hfac,gfac,gifabc,hfac")

        # 使用 "greedy" 算法优化计算路径，并检查路径是否符合预期
        path, path_str = np.einsum_path(*long_test1, optimize="greedy")
        self.assert_path_equal(
            path, ["einsum_path", (3, 6), (3, 4), (2, 4), (2, 3), (0, 2), (0, 1)]
        )

        # 使用 "optimal" 算法优化计算路径，并检查路径是否符合预期
        path, path_str = np.einsum_path(*long_test1, optimize="optimal")
        self.assert_path_equal(
            path, ["einsum_path", (3, 6), (3, 4), (2, 4), (2, 3), (0, 2), (0, 1)]
        )

        # 长测试 2
        long_test2 = self.build_operands("chd,bde,agbc,hiad,bdi,cgh,agdb")

        # 使用 "greedy" 算法优化计算路径，并检查路径是否符合预期
        path, path_str = np.einsum_path(*long_test2, optimize="greedy")
        self.assert_path_equal(
            path, ["einsum_path", (3, 4), (0, 3), (3, 4), (1, 3), (1, 2), (0, 1)]
        )

        # 使用 "optimal" 算法优化计算路径，并检查路径是否符合预期
        path, path_str = np.einsum_path(*long_test2, optimize="optimal")
        self.assert_path_equal(
            path, ["einsum_path", (0, 5), (1, 4), (3, 4), (1, 3), (1, 2), (0, 1)]
        )
    def test_edge_paths(self):
        # Difficult edge cases

        # Edge test1
        edge_test1 = self.build_operands("eb,cb,fb->cef")
        # 调用 np.einsum_path 函数计算给定操作数的最佳路径和路径字符串，使用贪婪算法优化
        path, path_str = np.einsum_path(*edge_test1, optimize="greedy")
        # 断言路径是否符合预期结果
        self.assert_path_equal(path, ["einsum_path", (0, 2), (0, 1)])

        # 使用最优算法优化路径
        path, path_str = np.einsum_path(*edge_test1, optimize="optimal")
        # 断言路径是否符合预期结果
        self.assert_path_equal(path, ["einsum_path", (0, 2), (0, 1)])

        # Edge test2
        edge_test2 = self.build_operands("dd,fb,be,cdb->cef")
        # 使用贪婪算法优化路径
        path, path_str = np.einsum_path(*edge_test2, optimize="greedy")
        # 断言路径是否符合预期结果
        self.assert_path_equal(path, ["einsum_path", (0, 3), (0, 1), (0, 1)])

        # 使用最优算法优化路径
        path, path_str = np.einsum_path(*edge_test2, optimize="optimal")
        # 断言路径是否符合预期结果
        self.assert_path_equal(path, ["einsum_path", (0, 3), (0, 1), (0, 1)])

        # Edge test3
        edge_test3 = self.build_operands("bca,cdb,dbf,afc->")
        # 使用贪婪算法优化路径
        path, path_str = np.einsum_path(*edge_test3, optimize="greedy")
        # 断言路径是否符合预期结果
        self.assert_path_equal(path, ["einsum_path", (1, 2), (0, 2), (0, 1)])

        # 使用最优算法优化路径
        path, path_str = np.einsum_path(*edge_test3, optimize="optimal")
        # 断言路径是否符合预期结果
        self.assert_path_equal(path, ["einsum_path", (1, 2), (0, 2), (0, 1)])

        # Edge test4
        edge_test4 = self.build_operands("dcc,fce,ea,dbf->ab")
        # 使用贪婪算法优化路径
        path, path_str = np.einsum_path(*edge_test4, optimize="greedy")
        # 断言路径是否符合预期结果
        self.assert_path_equal(path, ["einsum_path", (1, 2), (0, 1), (0, 1)])

        # 使用最优算法优化路径
        path, path_str = np.einsum_path(*edge_test4, optimize="optimal")
        # 断言路径是否符合预期结果
        self.assert_path_equal(path, ["einsum_path", (1, 2), (0, 2), (0, 1)])

        # Edge test5
        edge_test4 = self.build_operands(
            "a,ac,ab,ad,cd,bd,bc->", size_dict={"a": 20, "b": 20, "c": 20, "d": 20}
        )
        # 使用贪婪算法优化路径
        path, path_str = np.einsum_path(*edge_test4, optimize="greedy")
        # 断言路径是否符合预期结果
        self.assert_path_equal(path, ["einsum_path", (0, 1), (0, 1, 2, 3, 4, 5)])

        # 使用最优算法优化路径
        path, path_str = np.einsum_path(*edge_test4, optimize="optimal")
        # 断言路径是否符合预期结果
        self.assert_path_equal(path, ["einsum_path", (0, 1), (0, 1, 2, 3, 4, 5)])

    def test_path_type_input(self):
        # Test explicit path handling
        path_test = self.build_operands("dcc,fce,ea,dbf->ab")

        # 关闭优化，直接使用传入的路径
        path, path_str = np.einsum_path(*path_test, optimize=False)
        # 断言路径是否符合预期结果
        self.assert_path_equal(path, ["einsum_path", (0, 1, 2, 3)])

        # 使用最优算法优化路径
        path, path_str = np.einsum_path(*path_test, optimize=True)
        # 断言路径是否符合预期结果
        self.assert_path_equal(path, ["einsum_path", (1, 2), (0, 1), (0, 1)])

        exp_path = ["einsum_path", (0, 2), (0, 2), (0, 1)]
        # 使用显式传入的路径优化
        path, path_str = np.einsum_path(*path_test, optimize=exp_path)
        # 断言路径是否符合预期结果
        self.assert_path_equal(path, exp_path)

        # 检查不优化的 einsum 是否与优化的结果相等
        noopt = np.einsum(*path_test, optimize=False)
        opt = np.einsum(*path_test, optimize=exp_path)
        assert_almost_equal(noopt, opt)
    # 定义测试方法，验证路径类型输入在内部追踪时的行为
    def test_path_type_input_internal_trace(self):
        # 标识：gh-20962
        # 构建运算操作数，生成 "cab,cdd->ab" 的路径测试
        path_test = self.build_operands("cab,cdd->ab")
        # 期望的优化路径
        exp_path = ["einsum_path", (1,), (0, 1)]
        
        # 调用 np.einsum_path 函数获取实际路径和路径字符串
        path, path_str = np.einsum_path(*path_test, optimize=exp_path)
        # 断言实际路径与期望路径相等
        self.assert_path_equal(path, exp_path)
        
        # 再次验证 einsum 在给定路径下的工作情况
        # 未优化版本
        noopt = np.einsum(*path_test, optimize=False)
        # 使用期望路径进行优化
        opt = np.einsum(*path_test, optimize=exp_path)
        # 断言两者几乎相等
        assert_almost_equal(noopt, opt)

    # 定义测试方法，验证路径类型输入无效时的行为
    def test_path_type_input_invalid(self):
        # 构建运算操作数，生成 "ab,bc,cd,de->ae" 的路径测试
        path_test = self.build_operands("ab,bc,cd,de->ae")
        # 期望的优化路径
        exp_path = ["einsum_path", (2, 3), (0, 1)]
        
        # 断言在给定的优化路径下，np.einsum 函数会引发 RuntimeError 异常
        assert_raises(RuntimeError, np.einsum, *path_test, optimize=exp_path)
        # 断言在给定的优化路径下，np.einsum_path 函数会引发 RuntimeError 异常
        assert_raises(RuntimeError, np.einsum_path, *path_test, optimize=exp_path)

        # 构建运算操作数，生成 "a,a,a->a" 的路径测试
        path_test = self.build_operands("a,a,a->a")
        # 期望的优化路径
        exp_path = ["einsum_path", (1,), (0, 1)]
        
        # 断言在给定的优化路径下，np.einsum 函数会引发 RuntimeError 异常
        assert_raises(RuntimeError, np.einsum, *path_test, optimize=exp_path)
        # 断言在给定的优化路径下，np.einsum_path 函数会引发 RuntimeError 异常
        assert_raises(RuntimeError, np.einsum_path, *path_test, optimize=exp_path)

    # 定义测试方法，验证空格的处理
    def test_spaces(self):
        # 标识：gh-10794
        arr = np.array([[1]])
        # 遍历所有空格组合的迭代器
        for sp in itertools.product(["", " "], repeat=4):
            # 生成带有不同空格的 einsum 表达式，例如 "{}...a{}->{}...a{}"
            # 在这些表达式下，不应该出现任何错误
            np.einsum("{}...a{}->{}...a{}".format(*sp), arr)
class TestMisc(TestCase):
    # 定义测试类 TestMisc，继承自 TestCase

    def test_overlap(self):
        # 定义测试方法 test_overlap

        a = np.arange(9, dtype=int).reshape(3, 3)
        # 创建一个 3x3 的整数数组 a，内容为 [0, 1, 2, 3, 4, 5, 6, 7, 8]

        b = np.arange(9, dtype=int).reshape(3, 3)
        # 创建一个 3x3 的整数数组 b，内容与 a 相同

        d = np.dot(a, b)
        # 计算矩阵乘积 a 和 b，结果存储在 d 中

        # sanity check
        c = np.einsum("ij,jk->ik", a, b)
        # 使用 Einstein 求和约定计算矩阵乘积，结果存储在 c 中

        assert_equal(c, d)
        # 断言 c 和 d 相等，用于验证计算的正确性

        # gh-10080, out overlaps one of the operands
        c = np.einsum("ij,jk->ik", a, b, out=b)
        # 使用 Einstein 求和约定计算矩阵乘积，将结果存储在 b 中，确保 b 与计算的其中一个操作数重叠

        assert_equal(c, d)
        # 再次断言 c 和 d 相等，验证计算的正确性及重叠的处理是否正确


if __name__ == "__main__":
    run_tests()
    # 如果脚本作为主程序运行，则执行测试
```