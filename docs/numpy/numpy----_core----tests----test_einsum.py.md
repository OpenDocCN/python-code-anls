# `.\numpy\numpy\_core\tests\test_einsum.py`

```
# 导入 itertools 库，用于生成迭代器的工具函数
import itertools
# 导入 sys 库，提供对 Python 解释器的访问和操作功能
import sys
# 导入 platform 库，提供访问平台特定属性的功能
import platform

# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 numpy 库，并从中导入一系列用于测试的函数和类
import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_almost_equal,
    assert_raises, suppress_warnings, assert_raises_regex, assert_allclose
    )

# 为优化 einsum 函数设置参数
# 字符串 chars 表示用于 einsum 函数的标记
chars = 'abcdefghij'
# 大小数组 sizes 表示与每个标记相关联的尺寸
sizes = np.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3])
# 全局大小字典 global_size_dict 将每个标记映射到其对应的大小
global_size_dict = dict(zip(chars, sizes))

# 定义测试类 TestEinsum
class TestEinsum:
    # 使用 pytest.mark.parametrize 装饰器，为测试函数参数化
    @pytest.mark.parametrize("do_opt", [True, False])
    @pytest.mark.parametrize("einsum_fn", [np.einsum, np.einsum_path])
   python
    def test_einsum_errors(self, do_opt, einsum_fn):
        # 检查参数数量是否足够
        assert_raises(ValueError, einsum_fn, optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "", optimize=do_opt)

        # 子脚本必须是字符串
        assert_raises(TypeError, einsum_fn, 0, 0, optimize=do_opt)

        # 问题 4528 显示此调用会导致段错误
        assert_raises(TypeError, einsum_fn, *(None,)*63, optimize=do_opt)

        # 操作数数量必须与子脚本字符串中的计数匹配
        assert_raises(ValueError, einsum_fn, "", 0, 0, optimize=do_opt)
        assert_raises(ValueError, einsum_fn, ",", 0, [0], [0],
                      optimize=do_opt)
        assert_raises(ValueError, einsum_fn, ",", [0], optimize=do_opt)

        # 子脚本不能比操作数的维度更多
        assert_raises(ValueError, einsum_fn, "i", 0, optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "ij", [0, 0], optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "...i", 0, optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "i...j", [0, 0], optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "i...", 0, optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "ij...", [0, 0], optimize=do_opt)

        # 无效的省略号
        assert_raises(ValueError, einsum_fn, "i..", [0, 0], optimize=do_opt)
        assert_raises(ValueError, einsum_fn, ".i...", [0, 0], optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "j->..j", [0, 0], optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "j->.j...", [0, 0],
                      optimize=do_opt)

        # 无效的子脚本字符
        assert_raises(ValueError, einsum_fn, "i%...", [0, 0], optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "...j$", [0, 0], optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "i->&", [0, 0], optimize=do_opt)

        # 输出子脚本必须在输入中出现
        assert_raises(ValueError, einsum_fn, "i->ij", [0, 0], optimize=do_opt)

        # 输出子脚本只能指定一次
        assert_raises(ValueError, einsum_fn, "ij->jij", [[0, 0], [0, 0]],
                      optimize=do_opt)

        # 当被折叠时，维度必须匹配
        assert_raises(ValueError, einsum_fn, "ii",
                      np.arange(6).reshape(2, 3), optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "ii->i",
                      np.arange(6).reshape(2, 3), optimize=do_opt)

        with assert_raises_regex(ValueError, "'b'"):
            # gh-11221 - 错误消息中错误地出现了 'c'
            a = np.ones((3, 3, 4, 5, 6))
            b = np.ones((3, 4, 5))
            einsum_fn('aabcb,abc', a, b)

    @pytest.mark.parametrize("do_opt", [True, False])
    # 定义测试函数，用于检查 np.einsum 函数在特定错误情况下的行为
    def test_einsum_specific_errors(self, do_opt):
        # 断言：out 参数必须是一个数组类型
        assert_raises(TypeError, np.einsum, "", 0, out='test',
                      optimize=do_opt)

        # 断言：order 参数必须是有效的顺序字符串
        assert_raises(ValueError, np.einsum, "", 0, order='W',
                      optimize=do_opt)

        # 断言：casting 参数必须是有效的转换规则
        assert_raises(ValueError, np.einsum, "", 0, casting='blah',
                      optimize=do_opt)

        # 断言：dtype 参数必须是有效的数据类型
        assert_raises(TypeError, np.einsum, "", 0, dtype='bad_data_type',
                      optimize=do_opt)

        # 断言：不接受其他关键字参数
        assert_raises(TypeError, np.einsum, "", 0, bad_arg=0, optimize=do_opt)

        # 断言：必须显式启用对新维度的广播
        assert_raises(ValueError, np.einsum, "i", np.arange(6).reshape(2, 3),
                      optimize=do_opt)
        assert_raises(ValueError, np.einsum, "i->i", [[0, 1], [0, 1]],
                      out=np.arange(4).reshape(2, 2), optimize=do_opt)

        # 检查 order 关键字参数，asanyarray 允许 1 维数组通过
        assert_raises(ValueError, np.einsum, "i->i",
                      np.arange(6).reshape(-1, 1), optimize=do_opt, order='d')
    # 测试 einsum 方法对对象错误的处理能力
    def test_einsum_object_errors(self):
        # 定义自定义异常类，用于处理对象算术引发的异常
        class CustomException(Exception):
            pass

        # 定义一个特殊的盒子类，支持加法和乘法运算，当值超过特定阈值时抛出自定义异常
        class DestructoBox:

            def __init__(self, value, destruct):
                self._val = value
                self._destruct = destruct

            def __add__(self, other):
                # 定义加法运算，如果结果值超过设定的破坏阈值，则抛出自定义异常
                tmp = self._val + other._val
                if tmp >= self._destruct:
                    raise CustomException
                else:
                    self._val = tmp
                    return self

            def __radd__(self, other):
                # 定义反向加法运算，处理特殊情况下的加法
                if other == 0:
                    return self
                else:
                    return self.__add__(other)

            def __mul__(self, other):
                # 定义乘法运算，如果结果值超过设定的破坏阈值，则抛出自定义异常
                tmp = self._val * other._val
                if tmp >= self._destruct:
                    raise CustomException
                else:
                    self._val = tmp
                    return self

            def __rmul__(self, other):
                # 定义反向乘法运算，处理特殊情况下的乘法
                if other == 0:
                    return self
                else:
                    return self.__mul__(other)

        # 创建一个包含 DestructoBox 实例的 NumPy 对象数组，并进行形状重塑
        a = np.array([DestructoBox(i, 5) for i in range(1, 10)],
                     dtype='object').reshape(3, 3)

        # 断言异常应该由 np.einsum 方法抛出，计算字符串为 "ij->i"，操作数组为 a
        assert_raises(CustomException, np.einsum, "ij->i", a)

        # 创建一个包含 DestructoBox 实例的三维 NumPy 对象数组，并进行形状重塑
        b = np.array([DestructoBox(i, 100) for i in range(0, 27)],
                     dtype='object').reshape(3, 3, 3)

        # 断言异常应该由 np.einsum 方法抛出，计算字符串为 "i...k->..."，操作数组为 b
        assert_raises(CustomException, np.einsum, "i...k->...", b)

        # 创建一个包含 DestructoBox 实例的一维 NumPy 对象数组 b
        b = np.array([DestructoBox(i, 55) for i in range(1, 4)],
                     dtype='object')

        # 断言异常应该由 np.einsum 方法抛出，计算字符串为 "ij, j"，操作数组分别为 a 和 b
        assert_raises(CustomException, np.einsum, "ij, j", a, b)

        # 断言异常应该由 np.einsum 方法抛出，计算字符串为 "ij, jh"，操作数组分别为 a 和 a
        assert_raises(CustomException, np.einsum, "ij, jh", a, a)

        # 断言异常应该由 np.einsum 方法抛出，计算字符串为 "ij->"，操作数组为 a
        assert_raises(CustomException, np.einsum, "ij->", a)

    @np._no_nep50_warning()
    # 使用装饰器，禁止在 NEP 50 警告下运行的 einsum 方法测试
    def test_einsum_sums_int8(self):
        # 调用 check_einsum_sums 方法，检查特定类型（'i1'）的 einsum 运算
        self.check_einsum_sums('i1')

    # 调用 check_einsum_sums 方法，检查特定类型（'u1'）的 einsum 运算
    def test_einsum_sums_uint8(self):
        self.check_einsum_sums('u1')

    # 调用 check_einsum_sums 方法，检查特定类型（'i2'）的 einsum 运算
    def test_einsum_sums_int16(self):
        self.check_einsum_sums('i2')

    # 调用 check_einsum_sums 方法，检查特定类型（'u2'）的 einsum 运算
    def test_einsum_sums_uint16(self):
        self.check_einsum_sums('u2')

    # 调用 check_einsum_sums 方法，检查特定类型（'i4'）的 einsum 运算，并进行额外的检查
    def test_einsum_sums_int32(self):
        self.check_einsum_sums('i4')
        self.check_einsum_sums('i4', True)

    # 调用 check_einsum_sums 方法，检查特定类型（'u4'）的 einsum 运算，并进行额外的检查
    def test_einsum_sums_uint32(self):
        self.check_einsum_sums('u4')
        self.check_einsum_sums('u4', True)

    # 调用 check_einsum_sums 方法，检查特定类型（'i8'）的 einsum 运算
    def test_einsum_sums_int64(self):
        self.check_einsum_sums('i8')

    # 调用 check_einsum_sums 方法，检查特定类型（'u8'）的 einsum 运算
    def test_einsum_sums_uint64(self):
        self.check_einsum_sums('u8')

    # 调用 check_einsum_sums 方法，检查特定类型（'f2'）的 einsum 运算
    def test_einsum_sums_float16(self):
        self.check_einsum_sums('f2')

    # 调用 check_einsum_sums 方法，检查特定类型（'f4'）的 einsum 运算
    def test_einsum_sums_float32(self):
        self.check_einsum_sums('f4')
    # 测试 np.einsum 在浮点数类型 'f8' 上的求和功能
    def test_einsum_sums_float64(self):
        self.check_einsum_sums('f8')  # 调用检查函数，对 'f8' 类型进行测试
        self.check_einsum_sums('f8', True)  # 同上，使用优化参数进行测试

    # 测试 np.einsum 在 np.longdouble 类型上的求和功能
    def test_einsum_sums_longdouble(self):
        self.check_einsum_sums(np.longdouble)  # 调用检查函数，对 np.longdouble 类型进行测试

    # 测试 np.einsum 在复数浮点数 'c8' 上的求和功能
    def test_einsum_sums_cfloat64(self):
        self.check_einsum_sums('c8')  # 调用检查函数，对 'c8' 类型进行测试
        self.check_einsum_sums('c8', True)  # 同上，使用优化参数进行测试

    # 测试 np.einsum 在更大精度的复数浮点数 'c16' 上的求和功能
    def test_einsum_sums_cfloat128(self):
        self.check_einsum_sums('c16')  # 调用检查函数，对 'c16' 类型进行测试

    # 测试 np.einsum 在 np.clongdouble 类型上的求和功能
    def test_einsum_sums_clongdouble(self):
        self.check_einsum_sums(np.clongdouble)  # 调用检查函数，对 np.clongdouble 类型进行测试

    # 测试 np.einsum 在对象 'object' 类型上的其他功能
    def test_einsum_sums_object(self):
        self.check_einsum_sums('object')  # 调用检查函数，对 'object' 类型进行测试
        self.check_einsum_sums('object', True)  # 同上，使用优化参数进行测试

    # 测试 np.einsum 的各种其他用途和特殊情况
    def test_einsum_misc(self):
        # 检验之前因 PyArray_AssignZero 错误而导致崩溃的情况
        a = np.ones((1, 2))
        b = np.ones((2, 2, 1))
        assert_equal(np.einsum('ij...,j...->i...', a, b), [[[2], [2]]])
        assert_equal(np.einsum('ij...,j...->i...', a, b, optimize=True), [[[2], [2]]])

        # 对于问题 #10369 的回归测试（测试 Python 2 中的 Unicode 输入）
        assert_equal(np.einsum('ij...,j...->i...', a, b), [[[2], [2]]])
        assert_equal(np.einsum('...i,...i', [1, 2, 3], [2, 3, 4]), 20)
        assert_equal(np.einsum('...i,...i', [1, 2, 3], [2, 3, 4],
                               optimize='greedy'), 20)

        # 迭代器在缓冲这种归约过程时存在问题
        a = np.ones((5, 12, 4, 2, 3), np.int64)
        b = np.ones((5, 12, 11), np.int64)
        assert_equal(np.einsum('ijklm,ijn,ijn->', a, b, b),
                     np.einsum('ijklm,ijn->', a, b))
        assert_equal(np.einsum('ijklm,ijn,ijn->', a, b, b, optimize=True),
                     np.einsum('ijklm,ijn->', a, b, optimize=True))

        # 问题 #2027，内部循环实现中连续的三参数存在问题
        a = np.arange(1, 3)
        b = np.arange(1, 5).reshape(2, 2)
        c = np.arange(1, 9).reshape(4, 2)
        assert_equal(np.einsum('x,yx,zx->xzy', a, b, c),
                     [[[1,  3], [3,  9], [5, 15], [7, 21]],
                     [[8, 16], [16, 32], [24, 48], [32, 64]]])
        assert_equal(np.einsum('x,yx,zx->xzy', a, b, c, optimize=True),
                     [[[1,  3], [3,  9], [5, 15], [7, 21]],
                     [[8, 16], [16, 32], [24, 48], [32, 64]]])

        # 确保明确设置 out=None 不会导致错误
        # 参见问题 gh-15776 和问题 gh-15256
        assert_equal(np.einsum('i,j', [1], [2], out=None), [[2]])

    # 测试包含对象的循环情况
    def test_object_loop(self):

        # 定义一个类 Mult，实现乘法操作返回 42
        class Mult:
            def __mul__(self, other):
                return 42

        # 创建包含 Mult 对象的 np.array
        objMult = np.array([Mult()])

        # 创建一个包含空对象的 np.ndarray
        objNULL = np.ndarray(buffer=b'\0' * np.intp(0).itemsize, shape=1, dtype=object)

        # 使用 pytest 的断言捕获 TypeError 异常
        with pytest.raises(TypeError):
            np.einsum("i,j", [1], objNULL)
        with pytest.raises(TypeError):
            np.einsum("i,j", objNULL, [1])

        # 检查 np.einsum 在 objMult 对象上的乘法操作是否返回了预期值 42
        assert np.einsum("i,j", objMult, objMult) == 42
    def test_subscript_range(self):
        # 对于拉丁字母表的所有字母（包括大写和小写），确保可以在创建数组的下标时使用
        # 创建两个全为1的数组，形状分别为 (2, 3) 和 (3, 4)
        a = np.ones((2, 3))
        b = np.ones((3, 4))
        # 使用 np.einsum 函数执行张量运算，不优化，指定下标操作
        np.einsum(a, [0, 20], b, [20, 2], [0, 2], optimize=False)
        np.einsum(a, [0, 27], b, [27, 2], [0, 2], optimize=False)
        np.einsum(a, [0, 51], b, [51, 2], [0, 2], optimize=False)
        # 断言语句，检查是否引发 ValueError 异常，lambda 函数用于调用 np.einsum
        assert_raises(ValueError, lambda: np.einsum(a, [0, 52], b, [52, 2], [0, 2], optimize=False))
        assert_raises(ValueError, lambda: np.einsum(a, [-1, 5], b, [5, 2], [-1, 2], optimize=False))

    def test_einsum_broadcast(self):
        # 处理省略号时的变化，修复“中间广播”错误
        # 仅在 prepare_op_axes 中使用“RIGHT”迭代
        # 在左侧自动广播，在右侧必须显式广播
        # 我们需要测试优化解析功能

        # 创建两个数组 A 和 B，分别为形状为 (2, 3, 4) 和 (3,)
        A = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        B = np.arange(3)
        # 创建参考结果 ref，使用 np.einsum 执行张量运算，指定下标操作，不优化
        ref = np.einsum('ijk,j->ijk', A, B, optimize=False)
        # 循环测试优化和非优化的解析功能
        for opt in [True, False]:
            assert_equal(np.einsum('ij...,j...->ij...', A, B, optimize=opt), ref)
            assert_equal(np.einsum('ij...,...j->ij...', A, B, optimize=opt), ref)
            assert_equal(np.einsum('ij...,j->ij...', A, B, optimize=opt), ref)  # 曾引发错误

        # 创建两个数组 A 和 B，形状分别为 (4, 3) 和 (3, 2)
        A = np.arange(12).reshape((4, 3))
        B = np.arange(6).reshape((3, 2))
        # 创建参考结果 ref，使用 np.einsum 执行张量运算，指定下标操作，不优化
        ref = np.einsum('ik,kj->ij', A, B, optimize=False)
        # 循环测试优化和非优化的解析功能
        for opt in [True, False]:
            assert_equal(np.einsum('ik...,k...->i...', A, B, optimize=opt), ref)
            assert_equal(np.einsum('ik...,...kj->i...j', A, B, optimize=opt), ref)
            assert_equal(np.einsum('...k,kj', A, B, optimize=opt), ref)  # 曾引发错误
            assert_equal(np.einsum('ik,k...->i...', A, B, optimize=opt), ref)  # 曾引发错误

        # 创建维度列表 dims，形状为 [2, 3, 4, 5]
        dims = [2, 3, 4, 5]
        # 创建数组 a 和 v，形状分别为 dims 和 (4,)
        a = np.arange(np.prod(dims)).reshape(dims)
        v = np.arange(dims[2])
        # 创建参考结果 ref，使用 np.einsum 执行张量运算，指定下标操作，不优化
        ref = np.einsum('ijkl,k->ijl', a, v, optimize=False)
        # 循环测试优化和非优化的解析功能
        for opt in [True, False]:
            assert_equal(np.einsum('ijkl,k', a, v, optimize=opt), ref)
            assert_equal(np.einsum('...kl,k', a, v, optimize=opt), ref)  # 曾引发错误
            assert_equal(np.einsum('...kl,k...', a, v, optimize=opt), ref)

        # 设置 J、K、M 的值分别为 160
        J, K, M = 160, 160, 120
        # 创建数组 A 和 B，形状分别为 (1, 1, 1, 160, 160, 120) 和 (160, 160, 120, 3)
        A = np.arange(J * K * M).reshape(1, 1, 1, J, K, M)
        B = np.arange(J * K * M * 3).reshape(J, K, M, 3)
        # 创建参考结果 ref，使用 np.einsum 执行张量运算，指定下标操作，不优化
        ref = np.einsum('...lmn,...lmno->...o', A, B, optimize=False)
        # 循环测试优化和非优化的解析功能
        for opt in [True, False]:
            assert_equal(np.einsum('...lmn,lmno->...o', A, B, optimize=opt), ref)  # 曾引发错误
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
        
        # 创建一个2行3列的浮点型数组A，用于测试
        A = np.arange(2 * 3).reshape(2, 3).astype(np.float32)
        # 创建一个2行3列2731深度的整型数组B，用于测试
        B = np.arange(2 * 3 * 2731).reshape(2, 3, 2731).astype(np.int16)
        # 使用einsum函数对A和B进行操作，计算结果存入es中
        es = np.einsum('cl, cpx->lpx', A, B)
        # 使用tensordot函数对A和B进行操作，计算结果存入tp中
        tp = np.tensordot(A, B, axes=(0, 0))
        # 断言es与tp的值相等
        assert_equal(es, tp)
        
        # 以下是原始的出现bug的测试用例，通过使用aranges函数使其可重复性
        # 创建一个3行3列的双精度浮点型数组A，用于测试
        A = np.arange(3 * 3).reshape(3, 3).astype(np.float64)
        # 创建一个3行3列64x64深度的单精度浮点型数组B，用于测试
        B = np.arange(3 * 3 * 64 * 64).reshape(3, 3, 64, 64).astype(np.float32)
        # 使用einsum函数对A和B进行操作，计算结果存入es中
        es = np.einsum('cl, cpxy->lpxy', A, B)
        # 使用tensordot函数对A和B进行操作，计算结果存入tp中
        tp = np.tensordot(A, B, axes=(0, 0))
        # 断言es与tp的值相等
        assert_equal(es, tp)

    def test_einsum_fixed_collapsingbug(self):
        # Issue #5147.
        # The bug only occurred when output argument of einssum was used.
        
        # 创建一个5x5x5x5的正态分布随机数组x，用于测试
        x = np.random.normal(0, 1, (5, 5, 5, 5))
        # 创建一个5x5的零数组y1，用于存放结果
        y1 = np.zeros((5, 5))
        # 使用einsum函数计算x的特定索引操作，结果存入y1中
        np.einsum('aabb->ab', x, out=y1)
        # 创建一个1维数组idx，用于后续索引操作
        idx = np.arange(5)
        # 使用索引数组idx对x进行多维索引操作，结果存入y2中
        y2 = x[idx[:, None], idx[:, None], idx, idx]
        # 断言y1与y2的值相等
        assert_equal(y1, y2)

    def test_einsum_failed_on_p9_and_s390x(self):
        # Issues gh-14692 and gh-12689
        # Bug with signed vs unsigned char errored on power9 and s390x Linux
        
        # 创建一个10x10x10x10的随机数组tensor，用于测试
        tensor = np.random.random_sample((10, 10, 10, 10))
        # 使用einsum函数对tensor进行特定操作，结果存入x中
        x = np.einsum('ijij->', tensor)
        # 使用trace方法计算tensor在指定轴的轨迹，结果存入y中
        y = tensor.trace(axis1=0, axis2=2).trace()
        # 断言x与y的值在允许误差范围内相等
        assert_allclose(x, y)
    def test_einsum_all_contig_non_contig_output(self):
        # Issue gh-5907, tests that the all contiguous special case
        # actually checks the contiguity of the output
        
        # 创建一个 5x5 的全1数组
        x = np.ones((5, 5))
        
        # 创建一个长度为10的全1数组，取出其偶数索引位置的切片作为输出
        out = np.ones(10)[::2]
        
        # 创建一个长度为10的全1数组作为正确的基础参考
        correct_base = np.ones(10)
        correct_base[::2] = 5
        
        # np.einsum的使用例子1：使用 x, x, x 进行矩阵乘法运算，并将结果输出到 out 中
        np.einsum('mi,mi,mi->m', x, x, x, out=out)
        
        # 断言，检查 out 的基础数据是否与正确的基础数据相等
        assert_array_equal(out.base, correct_base)
        
        # np.einsum的使用例子2：使用 x, x 进行矩阵乘法运算，并将结果输出到 out 中
        np.einsum('im,im,im->m', x, x, x, out=out)
        
        # 断言，检查 out 的基础数据是否与正确的基础数据相等
        assert_array_equal(out.base, correct_base)
        
        # 重新初始化 out，创建一个形状为 (2, 2, 2) 的数组，并取其最后一个维度的切片作为输出
        out = np.ones((2, 2, 2))[..., 0]
        
        # 创建一个形状为 (2, 2, 2) 的全1数组作为正确的基础参考
        correct_base = np.ones((2, 2, 2))
        correct_base[..., 0] = 2
        
        # 创建一个形状为 (2, 2) 的全1浮点数数组 x
        x = np.ones((2, 2), np.float32)
        
        # np.einsum的使用例子3：使用 x, x 进行矩阵乘法运算，并将结果输出到 out 中
        np.einsum('ij,jk->ik', x, x, out=out)
        
        # 断言，检查 out 的基础数据是否与正确的基础数据相等
        assert_array_equal(out.base, correct_base)

    @pytest.mark.parametrize("dtype",
             np.typecodes["AllFloat"] + np.typecodes["AllInteger"])
    def test_different_paths(self, dtype):
        # Test originally added to cover broken float16 path: gh-20305
        # Likely most are covered elsewhere, at least partially.
        dtype = np.dtype(dtype)
        # 创建一个包含指定数据类型的 NumPy dtype 对象

        arr = (np.arange(7) + 0.5).astype(dtype)
        # 创建一个包含 7 个元素的数组，元素为从 0.5 开始的浮点数序列，并转换为指定数据类型

        scalar = np.array(2, dtype=dtype)
        # 创建一个包含单个整数 2 的数组，并使用指定的数据类型

        # contig -> scalar:
        res = np.einsum('i->', arr)
        # 对数组进行 einsum 操作，求和所有元素，返回一个标量
        assert res == arr.sum()
        # 断言求和结果与 arr.sum() 相等

        # contig, contig -> contig:
        res = np.einsum('i,i->i', arr, arr)
        # 对两个数组进行 einsum 操作，对应元素相乘，返回一个数组
        assert_array_equal(res, arr * arr)
        # 断言结果数组与 arr * arr 相等

        # noncontig, noncontig -> contig:
        res = np.einsum('i,i->i', arr.repeat(2)[::2], arr.repeat(2)[::2])
        # 对两个非连续的数组进行 einsum 操作，对应元素相乘，返回一个数组
        assert_array_equal(res, arr * arr)
        # 断言结果数组与 arr * arr 相等

        # contig + contig -> scalar
        assert np.einsum('i,i->', arr, arr) == (arr * arr).sum()
        # 对两个数组进行 einsum 操作，求和所有对应元素的乘积，返回一个标量

        # contig + scalar -> contig (with out)
        out = np.ones(7, dtype=dtype)
        # 创建一个包含全部为 1 的数组，使用指定的数据类型
        res = np.einsum('i,->i', arr, dtype.type(2), out=out)
        # 对数组和标量进行 einsum 操作，每个元素乘以标量，将结果存储在指定的输出数组中
        assert_array_equal(res, arr * dtype.type(2))
        # 断言结果数组与 arr * dtype.type(2) 相等

        # scalar + contig -> contig (with out)
        res = np.einsum(',i->i', scalar, arr)
        # 对标量和数组进行 einsum 操作，标量乘以每个数组元素，返回一个数组
        assert_array_equal(res, arr * dtype.type(2))
        # 断言结果数组与 arr * dtype.type(2) 相等

        # scalar + contig -> scalar
        res = np.einsum(',i->', scalar, arr)
        # 对标量和数组进行 einsum 操作，标量乘以数组所有元素之和，返回一个标量
        # 使用 einsum 来比较，避免由于求和过程中的舍入误差导致差异
        assert res == np.einsum('i->', scalar * arr)
        # 断言结果与 np.einsum('i->', scalar * arr) 相等

        # contig + scalar -> scalar
        res = np.einsum('i,->', arr, scalar)
        # 对数组和标量进行 einsum 操作，数组乘以标量，返回一个标量
        # 使用 einsum 来比较，避免由于求和过程中的舍入误差导致差异
        assert res == np.einsum('i->', scalar * arr)
        # 断言结果与 np.einsum('i->', scalar * arr) 相等

        # contig + contig + contig -> scalar
        arr = np.array([0.5, 0.5, 0.25, 4.5, 3.], dtype=dtype)
        # 创建一个包含指定数据类型的数组，包含指定的元素
        res = np.einsum('i,i,i->', arr, arr, arr)
        # 对三个数组进行 einsum 操作，对应元素相乘并求和，返回一个标量
        assert_array_equal(res, (arr * arr * arr).sum())
        # 断言结果与 (arr * arr * arr).sum() 相等

        # four arrays:
        res = np.einsum('i,i,i,i->', arr, arr, arr, arr)
        # 对四个数组进行 einsum 操作，对应元素相乘并求和，返回一个标量
        assert_array_equal(res, (arr * arr * arr * arr).sum())
        # 断言结果与 (arr * arr * arr * arr).sum() 相等

    def test_small_boolean_arrays(self):
        # See gh-5946.
        # Use array of True embedded in False.
        a = np.zeros((16, 1, 1), dtype=np.bool)[:2]
        # 创建一个形状为 (2, 1, 1) 的布尔类型数组，初始化为 False
        a[...] = True
        # 将数组所有元素设置为 True
        out = np.zeros((16, 1, 1), dtype=np.bool)[:2]
        # 创建一个形状为 (2, 1, 1) 的布尔类型数组，初始化为 False
        tgt = np.ones((2, 1, 1), dtype=np.bool)
        # 创建一个形状为 (2, 1, 1) 的布尔类型数组，初始化为 True
        res = np.einsum('...ij,...jk->...ik', a, a, out=out)
        # 对两个布尔类型数组进行 einsum 操作，对应元素相乘，并将结果存储在指定的输出数组中
        assert_equal(res, tgt)
        # 断言结果与目标数组相等

    def test_out_is_res(self):
        a = np.arange(9).reshape(3, 3)
        # 创建一个形状为 (3, 3) 的数组，包含 0 到 8 的整数
        res = np.einsum('...ij,...jk->...ik', a, a, out=a)
        # 对两个数组进行 einsum 操作，对应元素相乘，并将结果存储在输入的数组中
        assert res is a
        # 断言结果与输入的数组是同一个对象
    def optimize_compare(self, subscripts, operands=None):
        # 对优化函数的所有路径进行测试，与常规的 einsum 进行比较

        # 如果没有操作数，则从子表达式中提取参数并生成随机数组
        if operands is None:
            args = [subscripts]
            terms = subscripts.split('->')[0].split(',')
            for term in terms:
                dims = [global_size_dict[x] for x in term]
                args.append(np.random.rand(*dims))
        else:
            args = [subscripts] + operands
        
        # 使用 optimize=False 参数执行 einsum 操作
        noopt = np.einsum(*args, optimize=False)
        
        # 使用 optimize='greedy' 参数执行 einsum 操作
        opt = np.einsum(*args, optimize='greedy')
        
        # 断言优化后的结果与未优化结果近似相等
        assert_almost_equal(opt, noopt)
        
        # 使用 optimize='optimal' 参数执行 einsum 操作
        opt = np.einsum(*args, optimize='optimal')
        
        # 断言优化后的结果与未优化结果近似相等
        assert_almost_equal(opt, noopt)

    def test_hadamard_like_products(self):
        # 测试 Hadamard 外积类似的乘积
        self.optimize_compare('a,ab,abc->abc')
        self.optimize_compare('a,b,ab->ab')

    def test_index_transformations(self):
        # 测试简单的索引转换情况
        self.optimize_compare('ea,fb,gc,hd,abcd->efgh')
        self.optimize_compare('ea,fb,abcd,gc,hd->efgh')
        self.optimize_compare('abcd,ea,fb,gc,hd->efgh')

    def test_complex(self):
        # 测试长的复杂用例
        self.optimize_compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.optimize_compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.optimize_compare('cd,bdhe,aidb,hgca,gc,hgibcd,hgac')
        self.optimize_compare('abhe,hidj,jgba,hiab,gab')
        self.optimize_compare('bde,cdh,agdb,hica,ibd,hgicd,hiac')
        self.optimize_compare('chd,bde,agbc,hiad,hgc,hgi,hiad')
        self.optimize_compare('chd,bde,agbc,hiad,bdi,cgh,agdb')
        self.optimize_compare('bdhe,acad,hiab,agac,hibd')

    def test_collapse(self):
        # 测试内积
        self.optimize_compare('ab,ab,c->')
        self.optimize_compare('ab,ab,c->c')
        self.optimize_compare('ab,ab,cd,cd->')
        self.optimize_compare('ab,ab,cd,cd->ac')
        self.optimize_compare('ab,ab,cd,cd->cd')
        self.optimize_compare('ab,ab,cd,cd,ef,ef->')

    def test_expand(self):
        # 测试外积
        self.optimize_compare('ab,cd,ef->abcdef')
        self.optimize_compare('ab,cd,ef->acdf')
        self.optimize_compare('ab,cd,de->abcde')
        self.optimize_compare('ab,cd,de->be')
        self.optimize_compare('ab,bcd,cd->abcd')
        self.optimize_compare('ab,bcd,cd->abd')
    def test_edge_cases(self):
        # Difficult edge cases for optimization
        # 调用 optimize_compare 方法进行优化比较，传入字符串参数
        self.optimize_compare('eb,cb,fb->cef')
        self.optimize_compare('dd,fb,be,cdb->cef')
        self.optimize_compare('bca,cdb,dbf,afc->')
        self.optimize_compare('dcc,fce,ea,dbf->ab')
        self.optimize_compare('fdf,cdd,ccd,afe->ae')
        self.optimize_compare('abcd,ad')
        self.optimize_compare('ed,fcd,ff,bcf->be')
        self.optimize_compare('baa,dcf,af,cde->be')
        self.optimize_compare('bd,db,eac->ace')
        self.optimize_compare('fff,fae,bef,def->abd')
        self.optimize_compare('efc,dbc,acf,fd->abe')
        self.optimize_compare('ba,ac,da->bcd')

    def test_inner_product(self):
        # Inner products
        # 调用 optimize_compare 方法进行优化比较，传入字符串参数
        self.optimize_compare('ab,ab')
        self.optimize_compare('ab,ba')
        self.optimize_compare('abc,abc')
        self.optimize_compare('abc,bac')
        self.optimize_compare('abc,cba')

    def test_random_cases(self):
        # Randomly built test cases
        # 调用 optimize_compare 方法进行优化比较，传入字符串参数
        self.optimize_compare('aab,fa,df,ecc->bde')
        self.optimize_compare('ecb,fef,bad,ed->ac')
        self.optimize_compare('bcf,bbb,fbf,fc->')
        self.optimize_compare('bb,ff,be->e')
        self.optimize_compare('bcb,bb,fc,fff->')
        self.optimize_compare('fbb,dfd,fc,fc->')
        self.optimize_compare('afd,ba,cc,dc->bf')
        self.optimize_compare('adb,bc,fa,cfc->d')
        self.optimize_compare('bbd,bda,fc,db->acf')
        self.optimize_compare('dba,ead,cad->bce')
        self.optimize_compare('aef,fbc,dca->bde')

    def test_combined_views_mapping(self):
        # gh-10792
        # 创建一个 5x3 的 NumPy 数组，并使用 einsum 计算特定乘积
        a = np.arange(9).reshape(1, 1, 3, 1, 3)
        b = np.einsum('bbcdc->d', a)
        # 断言计算结果与预期值相等
        assert_equal(b, [12])

    def test_broadcasting_dot_cases(self):
        # Ensures broadcasting cases are not mistaken for GEMM

        # 创建不同形状的随机 NumPy 数组
        a = np.random.rand(1, 5, 4)
        b = np.random.rand(4, 6)
        c = np.random.rand(5, 6)
        d = np.random.rand(10)

        # 调用 optimize_compare 方法进行优化比较，传入字符串参数和 operands 参数列表
        self.optimize_compare('ijk,kl,jl', operands=[a, b, c])
        self.optimize_compare('ijk,kl,jl,i->i', operands=[a, b, c, d])

        # 创建不同形状的随机 NumPy 数组，并调用 optimize_compare 方法进行优化比较，传入字符串参数和 operands 参数列表
        e = np.random.rand(1, 1, 5, 4)
        f = np.random.rand(7, 7)
        self.optimize_compare('abjk,kl,jl', operands=[e, b, c])
        self.optimize_compare('abjk,kl,jl,ab->ab', operands=[e, b, c, f])

        # 在 gh-11308 中发现的边缘情况
        # 创建一个形状为 (2, 4, 8) 的 NumPy 数组 g，然后进行优化比较，传入字符串参数和 operands 参数列表
        g = np.arange(64).reshape(2, 4, 8)
        self.optimize_compare('obk,ijk->ioj', operands=[g, g])
    def test_output_order(self):
        # 定义一个测试方法，用于验证输出顺序在优化情况下是否被尊重，以下压缩应该产生重塑的张量视图
        # 问题编号 gh-16415

        # 创建一个列序（列主序）的全为1的3维数组
        a = np.ones((2, 3, 5), order='F')
        # 创建一个列序（列主序）的全为1的2维数组
        b = np.ones((4, 3), order='F')

        # 对于优化和非优化两种情况进行循环
        for opt in [True, False]:
            # 使用 einsum 函数按照指定的公式 '...ft,mf->...mt' 计算结果 tmp
            tmp = np.einsum('...ft,mf->...mt', a, b, order='a', optimize=opt)
            # 断言 tmp 是列序连续的
            assert_(tmp.flags.f_contiguous)

            # 同上，但是指定了不同的输出顺序为 'f'
            tmp = np.einsum('...ft,mf->...mt', a, b, order='f', optimize=opt)
            # 断言 tmp 是列序连续的
            assert_(tmp.flags.f_contiguous)

            # 同上，但是指定了输出顺序为 'c'
            tmp = np.einsum('...ft,mf->...mt', a, b, order='c', optimize=opt)
            # 断言 tmp 是行序连续的
            assert_(tmp.flags.c_contiguous)

            # 同上，但是指定了输出顺序为 'k'
            tmp = np.einsum('...ft,mf->...mt', a, b, order='k', optimize=opt)
            # 断言 tmp 既不是行序连续的也不是列序连续的
            assert_(tmp.flags.c_contiguous is False)
            assert_(tmp.flags.f_contiguous is False)

            # 未指定输出顺序的情况下进行 einsum 计算
            tmp = np.einsum('...ft,mf->...mt', a, b, optimize=opt)
            # 断言 tmp 既不是行序连续的也不是列序连续的
            assert_(tmp.flags.c_contiguous is False)
            assert_(tmp.flags.f_contiguous is False)

        # 创建一个行序（行主序）的全为1的2维数组
        c = np.ones((4, 3), order='C')
        # 对于优化和非优化两种情况进行循环
        for opt in [True, False]:
            # 使用 einsum 函数按照指定的公式 '...ft,mf->...mt' 计算结果 tmp
            tmp = np.einsum('...ft,mf->...mt', a, c, order='a', optimize=opt)
            # 断言 tmp 是行序连续的
            assert_(tmp.flags.c_contiguous)

        # 创建一个行序（行主序）的全为1的3维数组
        d = np.ones((2, 3, 5), order='C')
        # 对于优化和非优化两种情况进行循环
        for opt in [True, False]:
            # 使用 einsum 函数按照指定的公式 '...ft,mf->...mt' 计算结果 tmp
            tmp = np.einsum('...ft,mf->...mt', d, c, order='a', optimize=opt)
            # 断言 tmp 是行序连续的
            assert_(tmp.flags.c_contiguous)
    # 定义一个测试类 TestEinsumPath，用于测试 np.einsum_path 函数的路径优化功能
class TestEinsumPath:
    # 根据给定的字符串和大小字典构建操作数列表
    def build_operands(self, string, size_dict=global_size_dict):
        # 初始化操作数列表，第一个元素为输入字符串本身
        operands = [string]
        # 解析输入字符串中的各个操作数
        terms = string.split('->')[0].split(',')
        # 根据每个操作数的维度在大小字典中查找并创建随机数组
        for term in terms:
            dims = [size_dict[x] for x in term]
            operands.append(np.random.rand(*dims))

        return operands

    # 断言两个路径列表是否相等
    def assert_path_equal(self, comp, benchmark):
        # 检查路径列表的长度是否相等
        ret = (len(comp) == len(benchmark))
        assert_(ret)
        # 逐个比较路径中的每个元素是否相等
        for pos in range(len(comp) - 1):
            ret &= isinstance(comp[pos + 1], tuple)
            ret &= (comp[pos + 1] == benchmark[pos + 1])
        assert_(ret)

    # 测试内存约束是否满足
    def test_memory_contraints(self):
        # 使用简单的操作数构建测试
        outer_test = self.build_operands('a,b,c->abc')

        # 使用 'greedy' 优化方法计算路径和路径字符串
        path, path_str = np.einsum_path(*outer_test, optimize=('greedy', 0))
        # 断言计算得到的路径与预期路径相等
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2)])

        # 使用 'optimal' 优化方法计算路径和路径字符串
        path, path_str = np.einsum_path(*outer_test, optimize=('optimal', 0))
        # 断言计算得到的路径与预期路径相等
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2)])

        # 使用复杂的操作数构建测试
        long_test = self.build_operands('acdf,jbje,gihb,hfac')

        # 使用 'greedy' 优化方法计算路径和路径字符串
        path, path_str = np.einsum_path(*long_test, optimize=('greedy', 0))
        # 断言计算得到的路径与预期路径相等
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])

        # 使用 'optimal' 优化方法计算路径和路径字符串
        path, path_str = np.einsum_path(*long_test, optimize=('optimal', 0))
        # 断言计算得到的路径与预期路径相等
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])

    # 测试长而复杂的路径情况
    def test_long_paths(self):
        # 长而复杂的测试 1
        long_test1 = self.build_operands('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        # 使用 'greedy' 优化方法计算路径和路径字符串
        path, path_str = np.einsum_path(*long_test1, optimize='greedy')
        # 断言计算得到的路径与预期路径相等
        self.assert_path_equal(path, ['einsum_path',
                                      (3, 6), (3, 4), (2, 4), (2, 3), (0, 2), (0, 1)])

        # 使用 'optimal' 优化方法计算路径和路径字符串
        path, path_str = np.einsum_path(*long_test1, optimize='optimal')
        # 断言计算得到的路径与预期路径相等
        self.assert_path_equal(path, ['einsum_path',
                                      (3, 6), (3, 4), (2, 4), (2, 3), (0, 2), (0, 1)])

        # 长而复杂的测试 2
        long_test2 = self.build_operands('chd,bde,agbc,hiad,bdi,cgh,agdb')
        # 使用 'greedy' 优化方法计算路径和路径字符串
        path, path_str = np.einsum_path(*long_test2, optimize='greedy')
        # 断言计算得到的路径与预期路径相等
        self.assert_path_equal(path, ['einsum_path',
                                      (3, 4), (0, 3), (3, 4), (1, 3), (1, 2), (0, 1)])

        # 使用 'optimal' 优化方法计算路径和路径字符串
        path, path_str = np.einsum_path(*long_test2, optimize='optimal')
        # 断言计算得到的路径与预期路径相等
        self.assert_path_equal(path, ['einsum_path',
                                      (0, 5), (1, 4), (3, 4), (1, 3), (1, 2), (0, 1)])
    def test_edge_paths(self):
        # Difficult edge cases

        # Edge test1
        edge_test1 = self.build_operands('eb,cb,fb->cef')
        # 计算最优的求和路径和路径字符串，使用贪婪算法优化
        path, path_str = np.einsum_path(*edge_test1, optimize='greedy')
        # 断言路径是否符合预期
        self.assert_path_equal(path, ['einsum_path', (0, 2), (0, 1)])

        # 使用最优算法优化求和路径
        path, path_str = np.einsum_path(*edge_test1, optimize='optimal')
        # 断言路径是否符合预期
        self.assert_path_equal(path, ['einsum_path', (0, 2), (0, 1)])

        # Edge test2
        edge_test2 = self.build_operands('dd,fb,be,cdb->cef')
        # 计算最优的求和路径和路径字符串，使用贪婪算法优化
        path, path_str = np.einsum_path(*edge_test2, optimize='greedy')
        # 断言路径是否符合预期
        self.assert_path_equal(path, ['einsum_path', (0, 3), (0, 1), (0, 1)])

        # 使用最优算法优化求和路径
        path, path_str = np.einsum_path(*edge_test2, optimize='optimal')
        # 断言路径是否符合预期
        self.assert_path_equal(path, ['einsum_path', (0, 3), (0, 1), (0, 1)])

        # Edge test3
        edge_test3 = self.build_operands('bca,cdb,dbf,afc->')
        # 计算最优的求和路径和路径字符串，使用贪婪算法优化
        path, path_str = np.einsum_path(*edge_test3, optimize='greedy')
        # 断言路径是否符合预期
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])

        # 使用最优算法优化求和路径
        path, path_str = np.einsum_path(*edge_test3, optimize='optimal')
        # 断言路径是否符合预期
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])

        # Edge test4
        edge_test4 = self.build_operands('dcc,fce,ea,dbf->ab')
        # 计算最优的求和路径和路径字符串，使用贪婪算法优化
        path, path_str = np.einsum_path(*edge_test4, optimize='greedy')
        # 断言路径是否符合预期
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 1), (0, 1)])

        # 使用最优算法优化求和路径
        path, path_str = np.einsum_path(*edge_test4, optimize='optimal')
        # 断言路径是否符合预期
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])

        # Edge test5
        edge_test4 = self.build_operands('a,ac,ab,ad,cd,bd,bc->',
                                         size_dict={"a": 20, "b": 20, "c": 20, "d": 20})
        # 计算最优的求和路径和路径字符串，使用贪婪算法优化
        path, path_str = np.einsum_path(*edge_test4, optimize='greedy')
        # 断言路径是否符合预期
        self.assert_path_equal(path, ['einsum_path', (0, 1), (0, 1, 2, 3, 4, 5)])

        # 使用最优算法优化求和路径
        path, path_str = np.einsum_path(*edge_test4, optimize='optimal')
        # 断言路径是否符合预期
        self.assert_path_equal(path, ['einsum_path', (0, 1), (0, 1, 2, 3, 4, 5)])

    def test_path_type_input(self):
        # Test explicit path handling
        path_test = self.build_operands('dcc,fce,ea,dbf->ab')

        # 计算不使用优化的求和路径和路径字符串
        path, path_str = np.einsum_path(*path_test, optimize=False)
        # 断言路径是否符合预期
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])

        # 使用最优算法优化求和路径
        path, path_str = np.einsum_path(*path_test, optimize=True)
        # 断言路径是否符合预期
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 1), (0, 1)])

        # 预期的求和路径
        exp_path = ['einsum_path', (0, 2), (0, 2), (0, 1)]
        # 使用预期的路径进行优化
        path, path_str = np.einsum_path(*path_test, optimize=exp_path)
        # 断言路径是否符合预期
        self.assert_path_equal(path, exp_path)

        # Double check einsum works on the input path
        # 没有优化的求和
        noopt = np.einsum(*path_test, optimize=False)
        # 使用预期的路径进行优化求和
        opt = np.einsum(*path_test, optimize=exp_path)
        # 断言两种求和结果几乎相等
        assert_almost_equal(noopt, opt)
    # 定义测试函数，用于测试路径类型输入的内部追踪功能
    def test_path_type_input_internal_trace(self):
        #gh-20962
        # 创建测试用例，构建操作数
        path_test = self.build_operands('cab,cdd->ab')
        # 预期的优化路径
        exp_path = ['einsum_path', (1,), (0, 1)]

        # 调用 np.einsum_path 函数计算路径和路径字符串
        path, path_str = np.einsum_path(*path_test, optimize=exp_path)
        # 使用断言检查计算得到的路径是否与预期路径相等
        self.assert_path_equal(path, exp_path)

        # 双重检查 einsum 是否能够在输入路径上正常工作
        # 无优化的 einsum 计算结果
        noopt = np.einsum(*path_test, optimize=False)
        # 使用预期路径优化的 einsum 计算结果
        opt = np.einsum(*path_test, optimize=exp_path)
        # 使用断言检查两种计算结果是否几乎相等
        assert_almost_equal(noopt, opt)

    # 定义测试函数，用于测试路径类型输入的无效情况
    def test_path_type_input_invalid(self):
        # 构建操作数，创建测试用例
        path_test = self.build_operands('ab,bc,cd,de->ae')
        # 预期的优化路径
        exp_path = ['einsum_path', (2, 3), (0, 1)]
        # 使用断言检查是否抛出 RuntimeError 异常
        assert_raises(RuntimeError, np.einsum, *path_test, optimize=exp_path)
        assert_raises(
            RuntimeError, np.einsum_path, *path_test, optimize=exp_path)

        # 构建操作数，创建测试用例
        path_test = self.build_operands('a,a,a->a')
        # 预期的优化路径
        exp_path = ['einsum_path', (1,), (0, 1)]
        # 使用断言检查是否抛出 RuntimeError 异常
        assert_raises(RuntimeError, np.einsum, *path_test, optimize=exp_path)
        assert_raises(
            RuntimeError, np.einsum_path, *path_test, optimize=exp_path)

    # 定义测试函数，用于测试 np.einsum 中的空格情况
    def test_spaces(self):
        #gh-10794
        # 创建一个包含单个元素的二维数组
        arr = np.array([[1]])
        # 使用 itertools.product 生成所有可能的空格组合，长度为 4
        for sp in itertools.product(['', ' '], repeat=4):
            # 构造 einsum 的表达式字符串，测试不同的空格组合
            # 没有错误应该发生
            np.einsum('{}...a{}->{}...a{}'.format(*sp), arr)
# 定义一个函数用于测试矩阵乘法的重叠情况
def test_overlap():
    # 创建一个3x3的整数数组a，其中元素为0到8的序列
    a = np.arange(9, dtype=int).reshape(3, 3)
    # 创建一个3x3的整数数组b，也是0到8的序列
    b = np.arange(9, dtype=int).reshape(3, 3)
    # 计算矩阵a和b的矩阵乘法结果并赋值给d
    d = np.dot(a, b)
    # 进行一致性检查
    c = np.einsum('ij,jk->ik', a, b)
    assert_equal(c, d)
    # 通过使用已有的操作数之一来重叠输出，用于验证gh-10080问题
    c = np.einsum('ij,jk->ik', a, b, out=b)
    assert_equal(c, d)
```