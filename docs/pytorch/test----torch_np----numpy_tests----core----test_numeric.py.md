# `.\pytorch\test\torch_np\numpy_tests\core\test_numeric.py`

```py
# Owner(s): ["module: dynamo"]

# 导入必要的库和模块
import functools  # 导入 functools 模块，用于创建偏函数
import itertools  # 导入 itertools 模块，用于迭代操作
import math  # 导入 math 模块，提供数学运算函数
import platform  # 导入 platform 模块，用于获取平台信息
import sys  # 导入 sys 模块，用于系统相关的操作
import warnings  # 导入 warnings 模块，用于警告控制

import numpy  # 导入 numpy 库，科学计算的核心库

import pytest  # 导入 pytest 库，用于编写和运行测试

# 初始化一些全局变量
IS_WASM = False  # 设置 WASM 环境标志为 False
HAS_REFCOUNT = True  # 设置引用计数标志为 True

# 导入操作符模块
import operator

# 导入测试相关模块和函数
from unittest import expectedFailure as xfail, skipIf as skipif, SkipTest

# 导入 hypothesis 库中的模块和函数
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp

# 导入 pytest 中的异常断言函数
from pytest import raises as assert_raises

# 导入 torch.testing._internal.common_utils 模块中的函数和类
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfailIfTorchDynamo,
    xpassIfTorchDynamo,
)

# 根据 TEST_WITH_TORCHDYNAMO 的值选择导入 numpy 或 torch._numpy 相应的模块和函数
if TEST_WITH_TORCHDYNAMO:
    import numpy as np  # 导入 numpy 库，并重命名为 np
    from numpy.random import rand, randint, randn  # 导入随机数生成相关函数
    from numpy.testing import (
        assert_,  # 导入断言函数
        assert_allclose,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
        assert_warns,  # 导入警告断言函数
    )
else:
    import torch._numpy as np  # 导入 torch._numpy 库，并重命名为 np
    from torch._numpy.random import rand, randint, randn  # 导入 torch._numpy 中的随机数生成相关函数
    from torch._numpy.testing import (
        assert_,  # 导入断言函数
        assert_allclose,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
        assert_warns,  # 导入警告断言函数
    )

# 创建一个 skip 函数，其功能是跳过测试
skip = functools.partial(skipif, True)

# 定义一个测试类 TestResize，继承自 TestCase
@instantiate_parametrized_tests
class TestResize(TestCase):
    # 定义测试方法 test_copies，测试数组的复制操作
    def test_copies(self):
        A = np.array([[1, 2], [3, 4]])  # 创建一个 numpy 数组 A
        Ar1 = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])  # 期望的结果数组 Ar1
        assert_equal(np.resize(A, (2, 4)), Ar1)  # 断言 resize 操作的结果与 Ar1 相等

        Ar2 = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])  # 期望的结果数组 Ar2
        assert_equal(np.resize(A, (4, 2)), Ar2)  # 断言 resize 操作的结果与 Ar2 相等

        Ar3 = np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]])  # 期望的结果数组 Ar3
        assert_equal(np.resize(A, (4, 3)), Ar3)  # 断言 resize 操作的结果与 Ar3 相等

    # 定义测试方法 test_repeats，测试数组的重复操作
    def test_repeats(self):
        A = np.array([1, 2, 3])  # 创建一个 numpy 数组 A
        Ar1 = np.array([[1, 2, 3, 1], [2, 3, 1, 2]])  # 期望的结果数组 Ar1
        assert_equal(np.resize(A, (2, 4)), Ar1)  # 断言 resize 操作的结果与 Ar1 相等

        Ar2 = np.array([[1, 2], [3, 1], [2, 3], [1, 2]])  # 期望的结果数组 Ar2
        assert_equal(np.resize(A, (4, 2)), Ar2)  # 断言 resize 操作的结果与 Ar2 相等

        Ar3 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])  # 期望的结果数组 Ar3
        assert_equal(np.resize(A, (4, 3)), Ar3)  # 断言 resize 操作的结果与 Ar3 相等

    # 定义测试方法 test_zeroresize，测试零尺寸数组的 resize 操作
    def test_zeroresize(self):
        A = np.array([[1, 2], [3, 4]])  # 创建一个 numpy 数组 A
        Ar = np.resize(A, (0,))  # 执行 resize 操作，期望结果是零长度的数组
        assert_array_equal(Ar, np.array([]))  # 断言 resize 操作的结果为空数组
        assert_equal(A.dtype, Ar.dtype)  # 断言 resize 操作后数组的数据类型与原数组相同

        Ar = np.resize(A, (0, 2))  # 执行 resize 操作，期望结果是零行的二维数组
        assert_equal(Ar.shape, (0, 2))  # 断言 resize 操作的结果形状为 (0, 2)

        Ar = np.resize(A, (2, 0))  # 执行 resize 操作，期望结果是两行零列的二维数组
        assert_equal(Ar.shape, (2, 0))  # 断言 resize 操作的结果形状为 (2, 0)

    # 定义测试方法 test_reshape_from_zero，测试从零长度数组进行 reshape 操作
    def test_reshape_from_zero(self):
        A = np.zeros(0, dtype=np.float32)  # 创建一个零长度的浮点数数组 A
        Ar = np.resize(A, (2, 1))  # 执行 resize 操作，将 A 调整为形状 (2, 1) 的数组
        assert_array_equal(Ar, np.zeros((2, 1), Ar.dtype))  # 断言 resize 操作的结果与期望的数组相等
        assert_equal(A.dtype, Ar.dtype)  # 断言 resize 操作后数组的数据类型与原数组相同
    # 定义一个测试方法，用于测试负数调整数组大小的情况
    def test_negative_resize(self):
        # 创建一个从0到9的浮点数数组
        A = np.arange(0, 10, dtype=np.float32)
        # 定义新的形状为(-10, -1)，负数形状通常会引发异常
        new_shape = (-10, -1)
        # 使用 pytest 的上下文管理器来检测是否会抛出 RuntimeError 或 ValueError 异常
        with pytest.raises((RuntimeError, ValueError)):
            # 调用 numpy 的 resize 函数尝试调整数组 A 到新的形状 new_shape
            np.resize(A, new_shape=new_shape)
@instantiate_parametrized_tests
class TestNonarrayArgs(TestCase):
    # 测试非数组参数传递给函数时是否被包装成数组
    def test_choose(self):
        choices = [[0, 1, 2], [3, 4, 5], [5, 6, 7]]
        tgt = [5, 1, 5]  # 目标结果数组
        a = [2, 0, 1]  # 选择索引数组

        out = np.choose(a, choices)  # 使用选择索引从 choices 数组中选择元素
        assert_equal(out, tgt)  # 断言输出是否与目标结果一致

    def test_clip(self):
        arr = [-1, 5, 2, 3, 10, -4, -9]  # 输入数组
        out = np.clip(arr, 2, 7)  # 将输入数组中小于2的元素替换为2，大于7的元素替换为7
        tgt = [2, 5, 2, 3, 7, 2, 2]  # 目标结果数组
        assert_equal(out, tgt)  # 断言输出是否与目标结果一致

    @xpassIfTorchDynamo  # 标记为需要跳过的测试，理由是 "TODO implement compress(...)"
    def test_compress(self):
        arr = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]  # 输入数组
        tgt = [[5, 6, 7, 8, 9]]  # 目标结果数组
        out = np.compress([0, 1], arr, axis=0)  # 在指定轴上压缩数组
        assert_equal(out, tgt)  # 断言输出是否与目标结果一致

    def test_count_nonzero(self):
        arr = [[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]]  # 输入数组
        tgt = np.array([2, 3])  # 目标结果数组
        out = np.count_nonzero(arr, axis=1)  # 计算每行非零元素的个数
        assert_equal(out, tgt)  # 断言输出是否与目标结果一致

    def test_cumproduct(self):
        A = [[1, 2, 3], [4, 5, 6]]  # 输入数组
        assert_(np.all(np.cumproduct(A) == np.array([1, 2, 6, 24, 120, 720])))  # 断言累积乘积的结果是否符合预期

    def test_diagonal(self):
        a = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]  # 输入数组
        out = np.diagonal(a)  # 提取数组的对角线元素
        tgt = [0, 5, 10]  # 目标结果数组
        assert_equal(out, tgt)  # 断言输出是否与目标结果一致

    def test_mean(self):
        A = [[1, 2, 3], [4, 5, 6]]  # 输入数组
        assert_(np.mean(A) == 3.5)  # 断言数组的平均值是否为3.5
        assert_(np.all(np.mean(A, 0) == np.array([2.5, 3.5, 4.5])))  # 断言按列计算平均值的结果是否符合预期
        assert_(np.all(np.mean(A, 1) == np.array([2.0, 5.0])))  # 断言按行计算平均值的结果是否符合预期
        assert_(np.isnan(np.mean([])))  # 断言空数组的平均值是否为 NaN

    def test_ptp(self):
        a = [3, 4, 5, 10, -3, -5, 6.0]  # 输入数组
        assert_equal(np.ptp(a, axis=0), 15.0)  # 计算数组在指定轴上的峰值与谷值之差

    def test_prod(self):
        arr = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]  # 输入数组
        tgt = [24, 1890, 600]  # 目标结果数组
        assert_equal(np.prod(arr, axis=-1), tgt)  # 计算数组在指定轴上的乘积

    def test_ravel(self):
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]  # 输入数组
        tgt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 目标结果数组
        assert_equal(np.ravel(a), tgt)  # 将多维数组展平为一维数组

    def test_repeat(self):
        a = [1, 2, 3]  # 输入数组
        tgt = [1, 1, 2, 2, 3, 3]  # 目标结果数组

        out = np.repeat(a, 2)  # 对数组中的元素进行重复
        assert_equal(out, tgt)  # 断言输出是否与目标结果一致

    def test_reshape(self):
        arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]  # 输入数组
        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]  # 目标结果数组
        assert_equal(np.reshape(arr, (2, 6)), tgt)  # 将数组重塑成指定形状

    def test_round(self):
        arr = [1.56, 72.54, 6.35, 3.25]  # 输入数组
        tgt = [1.6, 72.5, 6.4, 3.2]  # 目标结果数组
        assert_equal(np.around(arr, decimals=1), tgt)  # 将数组元素四舍五入到指定小数位数
        s = np.float64(1.0)
        assert_equal(s.round(), 1.0)  # 对标量进行四舍五入

    def test_round_2(self):
        s = np.float64(1.0)
        assert_(isinstance(s.round(), (np.float64, np.ndarray)))  # 断言结果是 np.float64 或 np.ndarray 类型

    @xpassIfTorchDynamo  # 标记为需要跳过的测试，理由是 "scalar instances"
    @parametrize(
        "dtype",
        [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.float16,
            np.float32,
            np.float64,
        ],
    )
    # 参数化测试函数，用于测试不同的数据类型
    def test_dunder_round(self, dtype):
        # 创建指定类型的数据对象
        s = dtype(1)
        # 断言：round() 函数返回的结果是整数
        assert_(isinstance(round(s), int))
        # 断言：round() 函数对 None 值的处理结果是整数
        assert_(isinstance(round(s, None), int))
        # 断言：round() 函数对 ndigits=None 的处理结果是整数
        assert_(isinstance(round(s, ndigits=None), int))
        # 断言：round() 函数的默认行为
        assert_equal(round(s), 1)
        # 断言：round() 函数对 None 值的处理结果
        assert_equal(round(s, None), 1)
        # 断言：round() 函数对 ndigits=None 的处理结果
        assert_equal(round(s, ndigits=None), 1)

    @parametrize(
        "val, ndigits",
        [
            # pytest.param(
            #    2**31 - 1, -1, marks=pytest.mark.xfail(reason="Out of range of int32")
            # ),
            # 子测试：对特定值和小数位数进行测试，使用 xpassIfTorchDynamo 装饰器
            subtest((2**31 - 1, -1), decorators=[xpassIfTorchDynamo]),
            subtest(
                (2**31 - 1, 1 - math.ceil(math.log10(2**31 - 1))),
                decorators=[xpassIfTorchDynamo],
            ),
            subtest(
                (2**31 - 1, -math.ceil(math.log10(2**31 - 1))),
                decorators=[xpassIfTorchDynamo],
            ),
        ],
    )
    # 参数化测试函数，用于测试 round 函数在边缘情况下的行为
    def test_dunder_round_edgecases(self, val, ndigits):
        # 断言：round 函数对于给定值和小数位数的处理结果与 np.int32 相同
        assert_equal(round(val, ndigits), round(np.int32(val), ndigits))

    @xfail  # (reason="scalar instances")
    # 预期测试失败：测试 round 函数的精度
    def test_dunder_round_accuracy(self):
        f = np.float64(5.1 * 10**73)
        # 断言：round 函数处理浮点数并保持精度
        assert_(isinstance(round(f, -73), np.float64))
        # 断言：验证 round 函数处理浮点数后的最大 ULP（最小单位最后一位误差）
        assert_array_max_ulp(round(f, -73), 5.0 * 10**73)
        # 断言：round 函数对于指定精度的处理结果类型
        assert_(isinstance(round(f, ndigits=-73), np.float64))
        # 断言：验证 round 函数对于指定精度的处理结果的最大 ULP
        assert_array_max_ulp(round(f, ndigits=-73), 5.0 * 10**73)

        i = np.int64(501)
        # 断言：round 函数对整数类型的处理
        assert_(isinstance(round(i, -2), np.int64))
        # 断言：验证 round 函数对整数类型的处理结果
        assert_array_max_ulp(round(i, -2), 500)
        # 断言：round 函数对于指定精度的处理结果类型
        assert_(isinstance(round(i, ndigits=-2), np.int64))
        # 断言：验证 round 函数对于指定精度的处理结果
        assert_array_max_ulp(round(i, ndigits=-2), 500)

    @xfail  # (raises=AssertionError, reason="gh-15896")
    # 预期测试失败：验证 round 函数在 Python 中的一致性
    def test_round_py_consistency(self):
        f = 5.1 * 10**73
        # 断言：验证 round 函数在 numpy.float64 和普通浮点数上的一致性
        assert_equal(round(np.float64(f), -73), round(f, -73))

    # 测试 np.searchsorted 函数
    def test_searchsorted(self):
        arr = [-8, -5, -1, 3, 6, 10]
        # 执行搜索并断言搜索结果
        out = np.searchsorted(arr, 0)
        assert_equal(out, 3)

    # 测试 np.size 函数
    def test_size(self):
        A = [[1, 2, 3], [4, 5, 6]]
        # 断言：验证 np.size 函数返回的数组元素数量是否正确
        assert_(np.size(A) == 6)
        # 断言：验证 np.size 函数返回的指定轴的长度是否正确
        assert_(np.size(A, 0) == 2)
        assert_(np.size(A, 1) == 3)
    def test_squeeze(self):
        # 创建一个三维数组 A
        A = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
        # 对数组 A 进行 squeeze 操作，断言其形状为 (3, 3)
        assert_equal(np.squeeze(A).shape, (3, 3))
        # 对一个形状为 (1, 3, 1) 的零数组进行 squeeze 操作，断言其形状为 (3,)
        assert_equal(np.squeeze(np.zeros((1, 3, 1))).shape, (3,))
        # 对一个形状为 (1, 3, 1) 的零数组进行 squeeze 操作，指定 axis=0，断言其形状为 (3, 1)
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=0).shape, (3, 1))
        # 对一个形状为 (1, 3, 1) 的零数组进行 squeeze 操作，指定 axis=-1，断言其形状为 (1, 3)
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=-1).shape, (1, 3))
        # 对一个形状为 (1, 3, 1) 的零数组进行 squeeze 操作，指定 axis=2，断言其形状为 (1, 3)
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=2).shape, (1, 3))
        # 对一个包含一个形状为 (3, 1) 的零数组的列表进行 squeeze 操作，断言其形状为 (3,)
        assert_equal(np.squeeze([np.zeros((3, 1))]).shape, (3,))
        # 对一个包含一个形状为 (3, 1) 的零数组的列表进行 squeeze 操作，指定 axis=0，断言其形状为 (3, 1)
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=0).shape, (3, 1))
        # 对一个包含一个形状为 (3, 1) 的零数组的列表进行 squeeze 操作，指定 axis=2，断言其形状为 (1, 3)
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=2).shape, (1, 3))
        # 对一个包含一个形状为 (3, 1) 的零数组的列表进行 squeeze 操作，指定 axis=-1，断言其形状为 (1, 3)

    def test_std(self):
        # 定义一个二维数组 A
        A = [[1, 2, 3], [4, 5, 6]]
        # 断言计算数组 A 的标准差，约为 1.707825127659933
        assert_almost_equal(np.std(A), 1.707825127659933)
        # 断言计算数组 A 沿 axis=0 方向的标准差，应为 [1.5, 1.5, 1.5]
        assert_almost_equal(np.std(A, 0), np.array([1.5, 1.5, 1.5]))
        # 断言计算数组 A 沿 axis=1 方向的标准差，应为 [0.81649658, 0.81649658]
        assert_almost_equal(np.std(A, 1), np.array([0.81649658, 0.81649658]))
        # 使用空数组调用 np.std()，断言返回值为 NaN
        assert_(np.isnan(np.std([])))

    def test_swapaxes(self):
        # 定义目标数组 tgt 和源数组 a
        tgt = [[[0, 4], [2, 6]], [[1, 5], [3, 7]]]
        a = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        # 对数组 a 进行 swapaxes 操作，交换 axis=0 和 axis=2，断言结果与 tgt 相等
        out = np.swapaxes(a, 0, 2)
        assert_equal(out, tgt)

    def test_sum(self):
        # 定义一个二维数组 m 和目标数组 tgt
        m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        tgt = [[6], [15], [24]]
        # 对数组 m 沿 axis=1 方向进行求和，并保持维度，断言结果与 tgt 相等
        out = np.sum(m, axis=1, keepdims=True)
        assert_equal(tgt, out)

    def test_take(self):
        # 定义目标数组 tgt 和索引数组 indices，以及源数组 a
        tgt = [2, 3, 5]
        indices = [1, 2, 4]
        a = [1, 2, 3, 4, 5]
        # 对数组 a 根据 indices 中的索引取值，断言结果与 tgt 相等
        out = np.take(a, indices)
        assert_equal(out, tgt)

    def test_trace(self):
        # 定义一个二维数组 c
        c = [[1, 2], [3, 4], [5, 6]]
        # 计算数组 c 的迹（对角线元素的和），断言结果为 5
        assert_equal(np.trace(c), 5)

    def test_transpose(self):
        # 定义二维数组 arr 和目标数组 tgt
        arr = [[1, 2], [3, 4], [5, 6]]
        tgt = [[1, 3, 5], [2, 4, 6]]
        # 对数组 arr 进行转置，指定轴顺序为 (1, 0)，断言结果与 tgt 相等
        assert_equal(np.transpose(arr, (1, 0)), tgt)

    def test_var(self):
        # 定义一个二维数组 A
        A = [[1, 2, 3], [4, 5, 6]]
        # 断言计算数组 A 的方差，约为 2.9166666666666665
        assert_almost_equal(np.var(A), 2.9166666666666665)
        # 断言计算数组 A 沿 axis=0 方向的方差，应为 [2.25, 2.25, 2.25]
        assert_almost_equal(np.var(A, 0), np.array([2.25, 2.25, 2.25]))
        # 断言计算数组 A 沿 axis=1 方向的方差，应为 [0.66666667, 0.66666667]
        assert_almost_equal(np.var(A, 1), np.array([0.66666667, 0.66666667]))
        # 使用空数组调用 np.var()，断言返回值为 NaN
        assert_(np.isnan(np.var([])))
# 使用装饰器 @xfail 标记此测试类，表示这些测试预期会失败，原因是还未完成（TODO）
@xfail  # (reason="TODO")
class TestIsscalar(TestCase):
    # 测试 np.isscalar 函数的行为
    def test_isscalar(self):
        # 断言 3.1 是标量
        assert_(np.isscalar(3.1))
        # 断言 np.int16(12345) 是标量
        assert_(np.isscalar(np.int16(12345)))
        # 断言 False 是标量
        assert_(np.isscalar(False))
        # 断言 "numpy" 是标量
        assert_(np.isscalar("numpy"))
        # 断言 [3.1] 不是标量
        assert_(not np.isscalar([3.1]))
        # 断言 None 不是标量
        assert_(not np.isscalar(None))

        # 导入 Fraction 类
        from fractions import Fraction
        # 断言 Fraction(5, 17) 是标量
        assert_(np.isscalar(Fraction(5, 17)))

        # 导入 Number 类
        from numbers import Number
        # 断言 Number() 是标量
        assert_(np.isscalar(Number()))


class TestBoolScalar(TestCase):
    # 测试逻辑运算与布尔标量
    def test_logical(self):
        f = np.False_
        t = np.True_
        s = "xyz"
        # 断言 (True and "xyz") 的结果是 "xyz"
        assert_((t and s) is s)
        # 断言 (False and "xyz") 的结果是 False
        assert_((f and s) is f)

    # 测试位运算中的按位或操作
    def test_bitwise_or_eq(self):
        f = np.False_
        t = np.True_
        # 断言 (True | True) 等于 True
        assert_((t | t) == t)
        # 断言 (False | True) 等于 True
        assert_((f | t) == t)
        # 断言 (True | False) 等于 True
        assert_((t | f) == t)
        # 断言 (False | False) 等于 False
        assert_((f | f) == f)

    # 测试位运算中的按位或操作的布尔结果
    def test_bitwise_or_is(self):
        f = np.False_
        t = np.True_
        # 断言 bool(True | True) 的结果是 bool(True)
        assert_(bool(t | t) is bool(t))
        # 断言 bool(False | True) 的结果是 bool(True)
        assert_(bool(f | t) is bool(t))
        # 断言 bool(True | False) 的结果是 bool(True)
        assert_(bool(t | f) is bool(t))
        # 断言 bool(False | False) 的结果是 bool(False)
        assert_(bool(f | f) is bool(f))

    # 测试位运算中的按位与操作
    def test_bitwise_and_eq(self):
        f = np.False_
        t = np.True_
        # 断言 (True & True) 等于 True
        assert_((t & t) == t)
        # 断言 (False & True) 等于 False
        assert_((f & t) == f)
        # 断言 (True & False) 等于 False
        assert_((t & f) == f)
        # 断言 (False & False) 等于 False
        assert_((f & f) == f)

    # 测试位运算中的按位与操作的布尔结果
    def test_bitwise_and_is(self):
        f = np.False_
        t = np.True_
        # 断言 bool(True & True) 的结果是 bool(True)
        assert_(bool(t & t) is bool(t))
        # 断言 bool(False & True) 的结果是 bool(False)
        assert_(bool(f & t) is bool(f))
        # 断言 bool(True & False) 的结果是 bool(False)
        assert_(bool(t & f) is bool(f))
        # 断言 bool(False & False) 的结果是 bool(False)
        assert_(bool(f & f) is bool(f))

    # 测试位运算中的按位异或操作
    def test_bitwise_xor_eq(self):
        f = np.False_
        t = np.True_
        # 断言 (True ^ True) 等于 False
        assert_((t ^ t) == f)
        # 断言 (False ^ True) 等于 True
        assert_((f ^ t) == t)
        # 断言 (True ^ False) 等于 True
        assert_((t ^ f) == t)
        # 断言 (False ^ False) 等于 False
        assert_((f ^ f) == f)

    # 测试位运算中的按位异或操作的布尔结果
    def test_bitwise_xor_is(self):
        f = np.False_
        t = np.True_
        # 断言 bool(True ^ True) 的结果是 bool(False)
        assert_(bool(t ^ t) is bool(f))
        # 断言 bool(False ^ True) 的结果是 bool(True)
        assert_(bool(f ^ t) is bool(t))
        # 断言 bool(True ^ False) 的结果是 bool(True)
        assert_(bool(t ^ f) is bool(t))
        # 断言 bool(False ^ False) 的结果是 bool(False)
        assert_(bool(f ^ f) is bool(f))


class TestBoolArray(TestCase):
    # 设置测试环境，在 simd 测试中使用偏移量
    def setUp(self):
        super().setUp()
        self.t = np.array([True] * 41, dtype=bool)[1::]
        self.f = np.array([False] * 41, dtype=bool)[1::]
        self.o = np.array([False] * 42, dtype=bool)[2::]
        self.nm = self.f.copy()
        self.im = self.t.copy()
        self.nm[3] = True
        self.nm[-2] = True
        self.im[3] = False
        self.im[-2] = False
    # 测试所有和任意函数
    def test_all_any(self):
        # 断言self.t中所有元素为True
        assert_(self.t.all())
        # 断言self.t中任意元素为True
        assert_(self.t.any())
        # 断言self.f中所有元素为False
        assert_(not self.f.all())
        # 断言self.f中任意元素为False
        assert_(not self.f.any())
        # 断言self.nm中任意元素为True
        assert_(self.nm.any())
        # 断言self.im中任意元素为True
        assert_(self.im.any())
        # 断言self.nm中所有元素为False
        assert_(not self.nm.all())
        # 断言self.im中所有元素为False
        assert_(not self.im.all())
        
        # 检查在所有位置放置错误元素
        for i in range(256 - 7):
            # 创建一个包含256个False的布尔数组，并从索引7开始设置为True
            d = np.array([False] * 256, dtype=bool)[7::]
            d[i] = True
            # 断言数组d中至少有一个True值
            assert_(np.any(d))
            
            # 创建一个包含256个True的布尔数组，并从索引7开始设置为False
            e = np.array([True] * 256, dtype=bool)[7::]
            e[i] = False
            # 断言数组e中没有所有元素为True
            assert_(not np.all(e))
            # 断言e数组与d数组的逻辑非结果相等
            assert_array_equal(e, ~d)
        
        # 大数组测试，用于检查阻塞的libc循环
        for i in list(range(9, 6000, 507)) + [7764, 90021, -10]:
            # 创建一个包含100043个False的布尔数组，并将第i个元素设置为True
            d = np.array([False] * 100043, dtype=bool)
            d[i] = True
            # 断言数组d中至少有一个True值
            assert_(np.any(d), msg=f"{i!r}")
            
            # 创建一个包含100043个True的布尔数组，并将第i个元素设置为False
            e = np.array([True] * 100043, dtype=bool)
            e[i] = False
            # 断言数组e中没有所有元素为True
            assert_(not np.all(e), msg=f"{i!r}")

    # 测试逻辑非和绝对值函数
    def test_logical_not_abs(self):
        # 断言逻辑非操作后的self.t数组与self.f数组相等
        assert_array_equal(~self.t, self.f)
        # 断言逻辑非操作后再取绝对值的结果与self.f数组相等
        assert_array_equal(np.abs(~self.t), self.f)
        # 断言逻辑非操作后再取绝对值的结果与self.t数组相等
        assert_array_equal(np.abs(~self.f), self.t)
        # 断言对self.f数组取绝对值的结果与self.f数组相等
        assert_array_equal(np.abs(self.f), self.f)
        # 断言对self.f数组取绝对值后再进行逻辑非操作的结果与self.t数组相等
        assert_array_equal(~np.abs(self.f), self.t)
        # 断言对self.t数组取绝对值后再进行逻辑非操作的结果与self.f数组相等
        assert_array_equal(~np.abs(self.t), self.f)
        # 断言对逻辑非操作后再取绝对值的self.nm数组与self.im数组相等
        assert_array_equal(np.abs(~self.nm), self.im)
        
        # 在self.o中使用逻辑非操作，将结果与self.f数组相等
        np.logical_not(self.t, out=self.o)
        assert_array_equal(self.o, self.f)
        
        # 在self.o中使用绝对值操作，将结果与self.t数组相等
        np.abs(self.t, out=self.o)
        assert_array_equal(self.o, self.t)
    # 断言两个布尔数组的逻辑或操作结果等于第一个数组
    assert_array_equal(self.t | self.t, self.t)
    # 断言两个布尔数组的逻辑或操作结果等于第二个数组
    assert_array_equal(self.f | self.f, self.f)
    # 断言两个布尔数组的逻辑或操作结果等于第一个数组
    assert_array_equal(self.t | self.f, self.t)
    # 断言两个布尔数组的逻辑或操作结果等于第一个数组
    assert_array_equal(self.f | self.t, self.t)
    # 使用 numpy 的 logical_or 函数进行逻辑或操作，并将结果存入预定义的输出数组 o
    np.logical_or(self.t, self.t, out=self.o)
    # 断言输出数组 o 等于第一个数组
    assert_array_equal(self.o, self.t)
    # 断言两个布尔数组的逻辑与操作结果等于第一个数组
    assert_array_equal(self.t & self.t, self.t)
    # 断言两个布尔数组的逻辑与操作结果等于第二个数组
    assert_array_equal(self.f & self.f, self.f)
    # 断言两个布尔数组的逻辑与操作结果等于第二个数组
    assert_array_equal(self.t & self.f, self.f)
    # 断言两个布尔数组的逻辑与操作结果等于第二个数组
    assert_array_equal(self.f & self.t, self.f)
    # 使用 numpy 的 logical_and 函数进行逻辑与操作，并将结果存入预定义的输出数组 o
    np.logical_and(self.t, self.t, out=self.o)
    # 断言输出数组 o 等于第一个数组
    assert_array_equal(self.o, self.t)
    # 断言两个布尔数组的逻辑异或操作结果等于第二个数组
    assert_array_equal(self.t ^ self.t, self.f)
    # 断言两个布尔数组的逻辑异或操作结果等于第二个数组
    assert_array_equal(self.f ^ self.f, self.f)
    # 断言两个布尔数组的逻辑异或操作结果等于第一个数组
    assert_array_equal(self.t ^ self.f, self.t)
    # 断言两个布尔数组的逻辑异或操作结果等于第一个数组
    assert_array_equal(self.f ^ self.t, self.t)
    # 使用 numpy 的 logical_xor 函数进行逻辑异或操作，并将结果存入预定义的输出数组 o
    np.logical_xor(self.t, self.t, out=self.o)
    # 断言输出数组 o 等于第二个数组
    assert_array_equal(self.o, self.f)

    # 断言一个布尔数组与另一个数组的逻辑与操作结果等于第一个数组
    assert_array_equal(self.nm & self.t, self.nm)
    # 断言一个布尔数组与另一个数组的逻辑与操作结果等于 False
    assert_array_equal(self.im & self.f, False)
    # 断言一个布尔数组与 True 的逻辑与操作结果等于第一个数组
    assert_array_equal(self.nm & True, self.nm)
    # 断言一个布尔数组与 False 的逻辑与操作结果等于第二个数组
    assert_array_equal(self.im & False, self.f)
    # 断言一个布尔数组与另一个数组的逻辑或操作结果等于第一个数组
    assert_array_equal(self.nm | self.t, self.t)
    # 断言一个布尔数组与另一个数组的逻辑或操作结果等于第二个数组
    assert_array_equal(self.im | self.f, self.im)
    # 断言一个布尔数组与 True 的逻辑或操作结果等于第一个数组
    assert_array_equal(self.nm | True, self.t)
    # 断言一个布尔数组与 False 的逻辑或操作结果等于第二个数组
    assert_array_equal(self.im | False, self.im)
    # 断言一个布尔数组与另一个数组的逻辑异或操作结果等于第二个数组
    assert_array_equal(self.nm ^ self.t, self.im)
    # 断言一个布尔数组与另一个数组的逻辑异或操作结果等于第二个数组
    assert_array_equal(self.im ^ self.f, self.im)
    # 断言一个布尔数组与 True 的逻辑异或操作结果等于第二个数组
    assert_array_equal(self.nm ^ True, self.im)
    # 断言一个布尔数组与 False 的逻辑异或操作结果等于第二个数组
    assert_array_equal(self.im ^ False, self.im)
# 定义一个测试类 TestBoolCmp，用于测试布尔比较操作
@xfailIfTorchDynamo
class TestBoolCmp(TestCase):
    # 在每个测试方法执行前设置初始化操作
    def setUp(self):
        super().setUp()
        # 创建长度为 256 的浮点数数组 self.f，所有元素为 1.0
        self.f = np.ones(256, dtype=np.float32)
        # 创建与 self.f 同样长度的布尔数组 self.ef，所有元素为 True
        self.ef = np.ones(self.f.size, dtype=bool)
        # 创建长度为 128 的双精度浮点数数组 self.d，所有元素为 1.0
        self.d = np.ones(128, dtype=np.float64)
        # 创建与 self.d 同样长度的布尔数组 self.ed，所有元素为 True
        self.ed = np.ones(self.d.size, dtype=bool)
        
        # 生成所有 256bit SIMD 向量的所有排列的值
        s = 0
        for i in range(32):
            # 将每个 8 个元素的子数组 self.f[s:s+8] 设置为当前 i 的二进制表示
            self.f[s : s + 8] = [i & 2**x for x in range(8)]
            # 将对应的布尔数组 self.ef[s:s+8] 设置为判断当前 i 的二进制表示是否为真
            self.ef[s : s + 8] = [(i & 2**x) != 0 for x in range(8)]
            s += 8
        
        # 重新初始化 s 为 0，生成所有 128bit SIMD 向量的所有排列的值
        s = 0
        for i in range(16):
            # 将每个 4 个元素的子数组 self.d[s:s+4] 设置为当前 i 的二进制表示
            self.d[s : s + 4] = [i & 2**x for x in range(4)]
            # 将对应的布尔数组 self.ed[s:s+4] 设置为判断当前 i 的二进制表示是否为真
            self.ed[s : s + 4] = [(i & 2**x) != 0 for x in range(4)]
            s += 4

        # 复制 self.f 和 self.d 到 self.nf 和 self.nd
        self.nf = self.f.copy()
        self.nd = self.d.copy()
        # 将 self.nf 中满足 self.ef 的元素设为 NaN
        self.nf[self.ef] = np.nan
        # 将 self.nd 中满足 self.ed 的元素设为 NaN

        self.nd[self.ed] = np.nan

        # 复制 self.f 和 self.d 到 self.inff 和 self.infd
        self.inff = self.f.copy()
        self.infd = self.d.copy()
        # 在 self.inff 中每隔 3 个元素中满足 self.ef 的位置设为正无穷
        self.inff[::3][self.ef[::3]] = np.inf
        # 在 self.infd 中每隔 3 个元素中满足 self.ed 的位置设为正无穷
        self.infd[::3][self.ed[::3]] = np.inf
        # 在 self.inff 中每隔 3 个元素中满足 self.ef 的位置设为负无穷
        self.inff[1::3][self.ef[1::3]] = -np.inf
        # 在 self.infd 中每隔 3 个元素中满足 self.ed 的位置设为负无穷
        self.infd[1::3][self.ed[1::3]] = -np.inf
        # 在 self.inff 中每隔 3 个元素中满足 self.ef 的位置设为 NaN
        self.inff[2::3][self.ef[2::3]] = np.nan
        # 在 self.infd 中每隔 3 个元素中满足 self.ed 的位置设为 NaN
        self.infd[2::3][self.ed[2::3]] = np.nan
        
        # 复制 self.ef 到 self.efnonan
        self.efnonan = self.ef.copy()
        # 将 self.efnonan 中每隔 3 个元素中的位置设为 False
        self.efnonan[2::3] = False
        # 复制 self.ed 到 self.ednonan
        self.ednonan = self.ed.copy()
        # 将 self.ednonan 中每隔 3 个元素中的位置设为 False
        self.ednonan[2::3] = False
        
        # 复制 self.f 和 self.d 到 self.signf 和 self.signd
        self.signf = self.f.copy()
        self.signd = self.d.copy()
        # 在 self.signf 中满足 self.ef 的位置的元素乘以 -1.0
        self.signf[self.ef] *= -1.0
        # 在 self.signd 中满足 self.ed 的位置的元素乘以 -1.0
        self.signd[self.ed] *= -1.0
        # 在 self.signf 中每隔 6 个元素中满足 self.ef 的位置设为负无穷
        self.signf[1::6][self.ef[1::6]] = -np.inf
        # 在 self.signd 中每隔 6 个元素中满足 self.ed 的位置设为负无穷
        self.signd[1::6][self.ed[1::6]] = -np.inf
        # 在 self.signf 中每隔 6 个元素中满足 self.ef 的位置设为 NaN
        self.signf[3::6][self.ef[3::6]] = -np.nan
        # 在 self.signd 中每隔 6 个元素中满足 self.ed 的位置设为 NaN
        self.signd[3::6][self.ed[3::6]] = -np.nan
        # 在 self.signf 中每隔 6 个元素中满足 self.ef 的位置设为 -0.0
        self.signf[4::6][self.ef[4::6]] = -0.0
        # 在 self.signd 中每隔 6 个元素中满足 self.ed 的位置设为 -0.0
        self.signd[4::6][self.ed[4::6]] = -0.0
    # 测试单精度浮点数的方法
    def test_float(self):
        # 偏移量，用于对齐测试
        for i in range(4):
            # 断言数组是否相等：self.f[i:] > 0 与 self.ef[i:]
            assert_array_equal(self.f[i:] > 0, self.ef[i:])
            # 断言数组是否相等：self.f[i:] - 1 >= 0 与 self.ef[i:]
            assert_array_equal(self.f[i:] - 1 >= 0, self.ef[i:])
            # 断言数组是否相等：self.f[i:] == 0 与 ~self.ef[i:]
            assert_array_equal(self.f[i:] == 0, ~self.ef[i:])
            # 断言数组是否相等：-self.f[i:] < 0 与 self.ef[i:]
            assert_array_equal(-self.f[i:] < 0, self.ef[i:])
            # 断言数组是否相等：-self.f[i:] + 1 <= 0 与 self.ef[i:]
            assert_array_equal(-self.f[i:] + 1 <= 0, self.ef[i:])
            
            # 将 self.f[i:] != 0 的结果保存到 r 中
            r = self.f[i:] != 0
            # 断言数组是否相等：r 与 self.ef[i:]
            assert_array_equal(r, self.ef[i:])
            
            # 使用 np.zeros_like 创建与 self.f[i:] 相同形状的数组，并比较 self.f[i:] != np.zeros_like(self.f[i:])
            r2 = self.f[i:] != np.zeros_like(self.f[i:])
            # 与 0 != self.f[i:] 进行比较
            r3 = 0 != self.f[i:]
            # 断言数组是否相等：r 与 r2，r 与 r3
            assert_array_equal(r, r2)
            assert_array_equal(r, r3)
            
            # 检查布尔值是否等于 0x1 的整数值
            assert_array_equal(r.view(np.int8), r.astype(np.int8))
            assert_array_equal(r2.view(np.int8), r2.astype(np.int8))
            assert_array_equal(r3.view(np.int8), r3.astype(np.int8))
            
            # 对 amd64 平台上的 NaN 使用相同的代码路径进行检查
            assert_array_equal(np.isnan(self.nf[i:]), self.ef[i:])
            assert_array_equal(np.isfinite(self.nf[i:]), ~self.ef[i:])
            assert_array_equal(np.isfinite(self.inff[i:]), ~self.ef[i:])
            assert_array_equal(np.isinf(self.inff[i:]), self.efnonan[i:])
            assert_array_equal(np.signbit(self.signf[i:]), self.ef[i:])

    # 测试双精度浮点数的方法
    def test_double(self):
        # 偏移量，用于对齐测试
        for i in range(2):
            # 断言数组是否相等：self.d[i:] > 0 与 self.ed[i:]
            assert_array_equal(self.d[i:] > 0, self.ed[i:])
            # 断言数组是否相等：self.d[i:] - 1 >= 0 与 self.ed[i:]
            assert_array_equal(self.d[i:] - 1 >= 0, self.ed[i:])
            # 断言数组是否相等：self.d[i:] == 0 与 ~self.ed[i:]
            assert_array_equal(self.d[i:] == 0, ~self.ed[i:])
            # 断言数组是否相等：-self.d[i:] < 0 与 self.ed[i:]
            assert_array_equal(-self.d[i:] < 0, self.ed[i:])
            # 断言数组是否相等：-self.d[i:] + 1 <= 0 与 self.ed[i:]
            assert_array_equal(-self.d[i:] + 1 <= 0, self.ed[i:])
            
            # 将 self.d[i:] != 0 的结果保存到 r 中
            r = self.d[i:] != 0
            # 断言数组是否相等：r 与 self.ed[i:]
            assert_array_equal(r, self.ed[i:])
            
            # 使用 np.zeros_like 创建与 self.d[i:] 相同形状的数组，并比较 self.d[i:] != np.zeros_like(self.d[i:])
            r2 = self.d[i:] != np.zeros_like(self.d[i:])
            # 与 0 != self.d[i:] 进行比较
            r3 = 0 != self.d[i:]
            # 断言数组是否相等：r 与 r2，r 与 r3
            assert_array_equal(r, r2)
            assert_array_equal(r, r3)
            
            # 检查布尔值是否等于 0x1 的整数值
            assert_array_equal(r.view(np.int8), r.astype(np.int8))
            assert_array_equal(r2.view(np.int8), r2.astype(np.int8))
            assert_array_equal(r3.view(np.int8), r3.astype(np.int8))
            
            # 对 amd64 平台上的 NaN 使用相同的代码路径进行检查
            assert_array_equal(np.isnan(self.nd[i:]), self.ed[i:])
            assert_array_equal(np.isfinite(self.nd[i:]), ~self.ed[i:])
            assert_array_equal(np.isfinite(self.infd[i:]), ~self.ed[i:])
            assert_array_equal(np.isinf(self.infd[i:]), self.ednonan[i:])
            assert_array_equal(np.signbit(self.signd[i:]), self.ed[i:])
# 定义一个测试类 TestSeterr，用于测试 NumPy 的错误处理机制
@xpassIfTorchDynamo  # (reason="TODO")
class TestSeterr(TestCase):

    # 测试默认的错误处理设置
    def test_default(self):
        # 获取当前的错误处理设置
        err = np.geterr()
        # 断言默认的错误处理设置
        assert_equal(
            err, dict(divide="warn", invalid="warn", over="warn", under="ignore")
        )

    # 测试设置错误处理选项
    def test_set(self):
        # 获取当前的错误处理设置
        err = np.seterr()
        # 设置 divide 错误处理为打印信息，并记录旧的设置
        old = np.seterr(divide="print")
        # 断言设置前后的错误处理设置相同
        assert_(err == old)
        # 获取新的错误处理设置
        new = np.seterr()
        # 断言新的 divide 错误处理设置为打印信息
        assert_(new["divide"] == "print")
        # 设置 over 错误处理为抛出异常
        np.seterr(over="raise")
        # 断言当前 over 错误处理设置为抛出异常
        assert_(np.geterr()["over"] == "raise")
        # 再次断言 divide 错误处理设置仍为打印信息
        assert_(new["divide"] == "print")
        # 恢复到旧的错误处理设置
        np.seterr(**old)
        # 断言当前的错误处理设置与旧的设置相同
        assert_(np.geterr() == old)

    # 测试除零错误处理
    @xfail
    @skipif(IS_WASM, reason="no wasm fp exception support")
    @skipif(platform.machine() == "armv5tel", reason="See gh-413.")
    def test_divide_err(self):
        # 使用 assert_raises 断言浮点异常被引发
        with assert_raises(FloatingPointError):
            np.array([1.0]) / np.array([0.0])

        # 设置 divide 错误处理为忽略
        np.seterr(divide="ignore")
        # 再次进行除零操作，不应该引发异常
        np.array([1.0]) / np.array([0.0])

    # 测试错误对象处理
    @skipif(IS_WASM, reason="no wasm fp exception support")
    def test_errobj(self):
        # 获取当前的错误对象
        olderrobj = np.geterrobj()
        # 初始化一个计数器
        self.called = 0
        try:
            # 使用 warnings 模块捕获警告
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # 设置错误对象为指定的值
                np.seterrobj([20000, 1, None])
                # 执行可能引发错误的操作
                np.array([1.0]) / np.array([0.0])
                # 断言捕获到一条警告
                assert_equal(len(w), 1)

            # 定义一个日志错误的函数
            def log_err(*args):
                self.called += 1
                extobj_err = args
                assert_(len(extobj_err) == 2)
                assert_("divide" in extobj_err[0])

            # 设置错误对象为指定的值，并执行可能引发错误的操作
            np.seterrobj([20000, 3, log_err])
            np.array([1.0]) / np.array([0.0])
            # 断言 log_err 函数被调用一次
            assert_equal(self.called, 1)

            # 恢复到旧的错误对象设置
            np.seterrobj(olderrobj)
            # 使用指定的错误对象执行除法操作
            np.divide(1.0, 0.0, extobj=[20000, 3, log_err])
            # 断言 log_err 函数被调用两次
            assert_equal(self.called, 2)
        finally:
            # 最终恢复到旧的错误对象设置
            np.seterrobj(olderrobj)
            del self.called


# 定义一个测试类 TestFloatExceptions，用于测试浮点异常
@xfail  # (reason="TODO")
@instantiate_parametrized_tests
class TestFloatExceptions(TestCase):

    # 断言特定的浮点异常被引发
    def assert_raises_fpe(self, fpeerr, flop, x, y):
        ftype = type(x)
        try:
            flop(x, y)
            assert_(False, f"Type {ftype} did not raise fpe error '{fpeerr}'.")
        except FloatingPointError as exc:
            assert_(
                str(exc).find(fpeerr) >= 0,
                f"Type {ftype} raised wrong fpe error '{exc}'.",
            )

    # 断言操作引发特定的浮点异常
    def assert_op_raises_fpe(self, fpeerr, flop, sc1, sc2):
        # 检查是否引发了浮点异常
        #
        # 给定一个浮点操作 `flop` 和两个标量值，检查操作是否引发了由 `fpeerr` 指定的浮点异常。
        # 对所有的 0 维数组标量也进行测试。

        self.assert_raises_fpe(fpeerr, flop, sc1, sc2)
        self.assert_raises_fpe(fpeerr, flop, sc1[()], sc2)
        self.assert_raises_fpe(fpeerr, flop, sc1, sc2[()])
        self.assert_raises_fpe(fpeerr, flop, sc1[()], sc2[()])
    # Test for all real and complex float types
    @skipif(IS_WASM, reason="no wasm fp exception support")
    @parametrize("typecode", np.typecodes["AllFloat"])
    def test_floating_exceptions(self, typecode):
        # Test basic arithmetic function errors
        ftype = np.obj2sctype(typecode)
        if np.dtype(ftype).kind == "f":
            # Get some extreme values for the type
            fi = np.finfo(ftype)
            ft_tiny = fi._machar.tiny  # 获取当前浮点类型的最小正数值
            ft_max = fi.max  # 获取当前浮点类型的最大值
            ft_eps = fi.eps  # 获取当前浮点类型的机器精度
            underflow = "underflow"  # 表示下溢出异常类型
            divbyzero = "divide by zero"  # 表示除以零异常类型
        else:
            # 'c', complex, corresponding real dtype
            rtype = type(ftype(0).real)
            fi = np.finfo(rtype)
            ft_tiny = ftype(fi._machar.tiny)  # 获取复数对应的最小正数值
            ft_max = ftype(fi.max)  # 获取复数对应的最大值
            ft_eps = ftype(fi.eps)  # 获取复数对应的机器精度
            # The complex types raise different exceptions
            underflow = ""  # 复数类型无下溢出异常
            divbyzero = ""  # 复数类型无除以零异常
        overflow = "overflow"  # 表示上溢出异常类型
        invalid = "invalid"  # 表示无效数值异常类型

        # The value of tiny for double double is NaN, so we need to
        # pass the assert
        if not np.isnan(ft_tiny):
            self.assert_raises_fpe(underflow, operator.truediv, ft_tiny, ft_max)  # 断言检查下溢出异常
            self.assert_raises_fpe(underflow, operator.mul, ft_tiny, ft_tiny)  # 断言检查下溢出异常
        self.assert_raises_fpe(overflow, operator.mul, ft_max, ftype(2))  # 断言检查上溢出异常
        self.assert_raises_fpe(overflow, operator.truediv, ft_max, ftype(0.5))  # 断言检查上溢出异常
        self.assert_raises_fpe(overflow, operator.add, ft_max, ft_max * ft_eps)  # 断言检查上溢出异常
        self.assert_raises_fpe(overflow, operator.sub, -ft_max, ft_max * ft_eps)  # 断言检查上溢出异常
        self.assert_raises_fpe(overflow, np.power, ftype(2), ftype(2**fi.nexp))  # 断言检查上溢出异常
        self.assert_raises_fpe(divbyzero, operator.truediv, ftype(1), ftype(0))  # 断言检查除以零异常
        self.assert_raises_fpe(invalid, operator.truediv, ftype(np.inf), ftype(np.inf))  # 断言检查无效数值异常
        self.assert_raises_fpe(invalid, operator.truediv, ftype(0), ftype(0))  # 断言检查无效数值异常
        self.assert_raises_fpe(invalid, operator.sub, ftype(np.inf), ftype(np.inf))  # 断言检查无效数值异常
        self.assert_raises_fpe(invalid, operator.add, ftype(np.inf), ftype(-np.inf))  # 断言检查无效数值异常
        self.assert_raises_fpe(invalid, operator.mul, ftype(0), ftype(np.inf))  # 断言检查无效数值异常

    @skipif(IS_WASM, reason="no wasm fp exception support")
    def test_warnings(self):
        # test warning code path
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            np.divide(1, 0.0)  # 产生除以零的警告
            assert_equal(len(w), 1)  # 断言检查警告数量
            assert_("divide by zero" in str(w[0].message))  # 断言检查警告内容
            np.array(1e300) * np.array(1e300)  # 产生上溢出的警告
            assert_equal(len(w), 2)  # 断言检查警告数量
            assert_("overflow" in str(w[-1].message))  # 断言检查警告内容
            np.array(np.inf) - np.array(np.inf)  # 产生无效数值的警告
            assert_equal(len(w), 3)  # 断言检查警告数量
            assert_("invalid value" in str(w[-1].message))  # 断言检查警告内容
            np.array(1e-300) * np.array(1e-300)  # 产生下溢出的警告
            assert_equal(len(w), 4)  # 断言检查警告数量
            assert_("underflow" in str(w[-1].message))  # 断言检查警告内容
# 定义一个名为 TestTypes 的测试类，继承自 TestCase 类
class TestTypes(TestCase):
    
    # 定义一个方法 check_promotion_cases，用于测试类型提升函数的行为
    def check_promotion_cases(self, promote_func):
        # tests that the scalars get coerced correctly.
        
        # 定义不同类型的标量变量
        b = np.bool_(0)  # 布尔类型标量
        i8, i16, i32, i64 = np.int8(0), np.int16(0), np.int32(0), np.int64(0)  # 整数类型标量
        u8 = np.uint8(0)  # 无符号整数类型标量
        f32, f64 = np.float32(0), np.float64(0)  # 浮点数类型标量
        c64, c128 = np.complex64(0), np.complex128(0)  # 复数类型标量
        
        # coercion within the same kind
        # 测试相同类型间的类型提升
        assert_equal(promote_func(i8, i16), np.dtype(np.int16))
        assert_equal(promote_func(i32, i8), np.dtype(np.int32))
        assert_equal(promote_func(i16, i64), np.dtype(np.int64))
        assert_equal(promote_func(f32, f64), np.dtype(np.float64))
        assert_equal(promote_func(c128, c64), np.dtype(np.complex128))
        
        # coercion between kinds
        # 测试不同类型间的类型提升
        assert_equal(promote_func(b, i32), np.dtype(np.int32))
        assert_equal(promote_func(b, u8), np.dtype(np.uint8))
        assert_equal(promote_func(i8, u8), np.dtype(np.int16))
        assert_equal(promote_func(u8, i32), np.dtype(np.int32))
        
        assert_equal(promote_func(f32, i16), np.dtype(np.float32))
        assert_equal(promote_func(f32, c64), np.dtype(np.complex64))
        assert_equal(promote_func(c128, f32), np.dtype(np.complex128))
        
        # coercion between scalars and 1-D arrays
        # 测试标量与一维数组间的类型提升
        assert_equal(promote_func(np.array([b]), i8), np.dtype(np.int8))
        assert_equal(promote_func(np.array([b]), u8), np.dtype(np.uint8))
        assert_equal(promote_func(np.array([b]), i32), np.dtype(np.int32))
        assert_equal(promote_func(c64, np.array([f64])), np.dtype(np.complex128))
        assert_equal(promote_func(np.complex64(3j), np.array([f64])), np.dtype(np.complex128))
        
        # coercion between scalars and 1-D arrays, where
        # the scalar has greater kind than the array
        # 测试标量与一维数组间的类型提升，其中标量类型高于数组类型
        assert_equal(promote_func(np.array([b]), f64), np.dtype(np.float64))
        assert_equal(promote_func(np.array([b]), i64), np.dtype(np.int64))
        assert_equal(promote_func(np.array([i8]), f64), np.dtype(np.float64))
    def check_promotion_cases_2(self, promote_func):
        # 这些测试失败是因为“标量不会升级数组”的规则
        # 前两个测试 (i32 + f32 -> f64, 和 i64+f32 -> f64) 失败
        # 直到通用函数实现适当的类型提升（通用函数循环？）

        # 创建布尔类型的标量对象
        b = np.bool_(0)
        # 创建不同整数类型的标量对象
        i8, i16, i32, i64 = np.int8(0), np.int16(0), np.int32(0), np.int64(0)
        # 创建无符号整数类型的标量对象
        u8 = np.uint8(0)
        # 创建不同浮点数类型的标量对象
        f32, f64 = np.float32(0), np.float64(0)
        # 创建不同复数类型的标量对象
        c64, c128 = np.complex64(0), np.complex128(0)

        # 断言不同类型参数经过提升函数后的结果是否为预期的浮点数类型
        assert_equal(promote_func(i32, f32), np.dtype(np.float64))
        assert_equal(promote_func(i64, f32), np.dtype(np.float64))

        # 断言提升函数处理数组和标量组合时的类型提升结果是否为预期的整数类型
        assert_equal(promote_func(np.array([i8]), i64), np.dtype(np.int8))
        # 断言提升函数处理标量和数组组合时的类型提升结果是否为预期的浮点数类型
        assert_equal(promote_func(f64, np.array([f32])), np.dtype(np.float32))

        # 浮点数和复数在数组标量提升目的上被视为相同的“种类”，
        # 这样可以执行 (0j + float32array) 来获取 complex64 数组而不是 complex128 数组。
        assert_equal(promote_func(np.array([f32]), c128), np.dtype(np.complex64))
    # 定义测试函数 test_coercion，测试类型强制转换的情况
    def test_coercion(self):
        
        # 定义一个内部函数 res_type，返回两个数组相加后的数据类型
        def res_type(a, b):
            return np.add(a, b).dtype

        # 调用类中的方法 check_promotion_cases 来检查类型提升的情况
        self.check_promotion_cases(res_type)

        # Use-case: float/complex scalar * bool/int8 array
        #           shouldn't narrow the float/complex type
        # 针对不同类型的数组进行测试，确保在乘法运算后不会缩小浮点数或复数类型的位宽
        for a in [np.array([True, False]), np.array([-3, 12], dtype=np.int8)]:
            # 测试乘法运算后的结果数据类型是否为 np.float64
            b = 1.234 * a
            assert_equal(b.dtype, np.dtype("f8"), f"array type {a.dtype}")
            # 测试乘法运算后的结果数据类型是否为 np.float64
            b = np.float64(1.234) * a
            assert_equal(b.dtype, np.dtype("f8"), f"array type {a.dtype}")
            # 测试乘法运算后的结果数据类型是否为 np.float32
            b = np.float32(1.234) * a
            assert_equal(b.dtype, np.dtype("f4"), f"array type {a.dtype}")
            # 测试乘法运算后的结果数据类型是否为 np.float16
            b = np.float16(1.234) * a
            assert_equal(b.dtype, np.dtype("f2"), f"array type {a.dtype}")

            # 测试乘法运算后的结果数据类型是否为 np.complex128
            b = 1.234j * a
            assert_equal(b.dtype, np.dtype("c16"), f"array type {a.dtype}")
            # 测试乘法运算后的结果数据类型是否为 np.complex128
            b = np.complex128(1.234j) * a
            assert_equal(b.dtype, np.dtype("c16"), f"array type {a.dtype}")
            # 测试乘法运算后的结果数据类型是否为 np.complex64
            b = np.complex64(1.234j) * a
            assert_equal(b.dtype, np.dtype("c8"), f"array type {a.dtype}")

        # The following use-case is problematic, and to resolve its
        # tricky side-effects requires more changes.
        #
        # Use-case: (1-t)*a, where 't' is a boolean array and 'a' is
        #            a float32, shouldn't promote to float64
        #
        # a = np.array([1.0, 1.5], dtype=np.float32)
        # t = np.array([True, False])
        # b = t*a
        # assert_equal(b, [1.0, 0.0])
        # assert_equal(b.dtype, np.dtype('f4'))
        # b = (1-t)*a
        # assert_equal(b, [0.0, 1.5])
        # assert_equal(b.dtype, np.dtype('f4'))
        #
        # Probably ~t (bitwise negation) is more proper to use here,
        # but this is arguably less intuitive to understand at a glance, and
        # would fail if 't' is actually an integer array instead of boolean:
        #
        # b = (~t)*a
        # assert_equal(b, [0.0, 1.5])
        # assert_equal(b.dtype, np.dtype('f4'))

    @xpassIfTorchDynamo  # (reason="'Scalars do not upcast arrays' rule")
    # 定义测试函数 test_coercion_2，用于测试类型强制转换的另一个情况
    def test_coercion_2(self):
        
        # 定义一个内部函数 res_type，返回两个数组相加后的数据类型
        def res_type(a, b):
            return np.add(a, b).dtype

        # 调用类中的方法 check_promotion_cases_2 来检查类型提升的情况
        self.check_promotion_cases_2(res_type)

    # 定义测试函数 test_result_type，测试结果类型的情况
    def test_result_type(self):
        # 调用类中的方法 check_promotion_cases 来检查结果类型的情况
        self.check_promotion_cases(np.result_type)

    @skip(reason="array(None) not supported")
    # 定义测试函数 test_tesult_type_2，测试结果类型的另一种情况
    def test_tesult_type_2(self):
        # 断言空值的结果类型应为 None
        assert_(np.result_type(None) == np.dtype(None))

    @skip(reason="no endianness in dtypes")
    # 定义测试函数 test_promote_types_endian，测试类型提升的字节序问题
    def test_promote_types_endian(self):
        # 测试 promote_types 是否总是返回本机字节序类型
        assert_equal(np.promote_types("<i8", "<i8"), np.dtype("i8"))
        assert_equal(np.promote_types(">i8", ">i8"), np.dtype("i8"))
    # 定义一个测试方法，用于测试数据类型转换的功能
    def test_can_cast(self):
        # 断言 np.int32 可以转换为 np.int64
        assert_(np.can_cast(np.int32, np.int64))
        # 断言 np.float64 可以转换为 complex 复数类型
        assert_(np.can_cast(np.float64, complex))
        # 断言 complex 复数类型不能转换为 float 浮点数类型
        assert_(not np.can_cast(complex, float))

        # 断言 "i8" 字符串类型可以转换为 "f8" 浮点数类型
        assert_(np.can_cast("i8", "f8"))
        # 断言 "i8" 字符串类型不能转换为 "f4" 浮点数类型
        assert_(not np.can_cast("i8", "f4"))

        # 断言 "i8" 字符串类型可以转换为 "i8"，使用 "no" 模式
        assert_(np.can_cast("i8", "i8", "no"))

    # 标记为跳过测试，原因是 dtype 中没有字节序信息
    @skip(reason="no endianness in dtypes")
    def test_can_cast_2(self):
        # 断言 "<i8" 小端整数类型不能转换为 ">i8" 大端整数类型，使用 "no" 模式
        assert_(not np.can_cast("<i8", ">i8", "no"))

        # 断言 "<i8" 小端整数类型可以转换为 ">i8" 大端整数类型，使用 "equiv" 模式
        assert_(np.can_cast("<i8", ">i8", "equiv"))
        # 断言 "<i4" 小端整数类型不能转换为 ">i8" 大端整数类型，使用 "equiv" 模式
        assert_(not np.can_cast("<i4", ">i8", "equiv"))

        # 断言 "<i4" 小端整数类型可以安全转换为 ">i8" 大端整数类型，使用 "safe" 模式
        assert_(np.can_cast("<i4", ">i8", "safe"))
        # 断言 "<i8" 小端整数类型不能安全转换为 ">i4" 大端整数类型，使用 "safe" 模式
        assert_(not np.can_cast("<i8", ">i4", "safe"))

        # 断言 "<i8" 小端整数类型可以与 ">i4" 大端整数类型属于相同类型，使用 "same_kind" 模式
        assert_(np.can_cast("<i8", ">i4", "same_kind"))
        # 断言 "<i8" 小端整数类型不能与 ">u4" 大端无符号整数类型属于相同类型，使用 "same_kind" 模式
        assert_(not np.can_cast("<i8", ">u4", "same_kind"))

        # 断言 "<i8" 小端整数类型可以不安全转换为 ">u4" 大端无符号整数类型，使用 "unsafe" 模式
        assert_(np.can_cast("<i8", ">u4", "unsafe"))

        # 断言 TypeError 异常被触发，因为 "i4" 类型不能转换为 None 类型
        assert_raises(TypeError, np.can_cast, "i4", None)
        # 断言 TypeError 异常被触发，因为 None 类型不能转换为 "i4" 类型
        assert_raises(TypeError, np.can_cast, None, "i4")

        # 也测试关键字参数的使用
        # 断言 np.int32 可以转换为 np.int64，使用 from_= 和 to= 关键字
        assert_(np.can_cast(from_=np.int32, to=np.int64))

    # 标记为跳过测试，原因可能是基于值的转换
    @xpassIfTorchDynamo  # (reason="value-based casting?")
    def test_can_cast_values(self):
        # 对于所有整数类型和无符号整数类型进行测试
        for dt in np.sctypes["int"] + np.sctypes["uint"]:
            # 获取当前整数类型的信息
            ii = np.iinfo(dt)
            # 断言当前类型的最小值可以转换为当前类型
            assert_(np.can_cast(ii.min, dt))
            # 断言当前类型的最大值可以转换为当前类型
            assert_(np.can_cast(ii.max, dt))
            # 断言当前类型的最小值减一不能转换为当前类型
            assert_(not np.can_cast(ii.min - 1, dt))
            # 断言当前类型的最大值加一不能转换为当前类型
            assert_(not np.can_cast(ii.max + 1, dt))

        # 对于所有浮点数类型进行测试
        for dt in np.sctypes["float"]:
            # 获取当前浮点数类型的信息
            fi = np.finfo(dt)
            # 断言当前类型的最小正值可以转换为当前类型
            assert_(np.can_cast(fi.min, dt))
            # 断言当前类型的最大正值可以转换为当前类型
            assert_(np.can_cast(fi.max, dt))
# Custom exception class to test exception propagation in fromiter
class NIterError(Exception):
    pass


@skip(reason="NP_VER: fails on CI")
@xpassIfTorchDynamo  # (reason="TODO")
@instantiate_parametrized_tests
class TestFromiter(TestCase):
    # 生成一个生成器，产生 0 到 23 的平方的序列
    def makegen(self):
        return (x**2 for x in range(24))

    # 测试从生成器中创建不同类型的数组
    def test_types(self):
        ai32 = np.fromiter(self.makegen(), np.int32)
        ai64 = np.fromiter(self.makegen(), np.int64)
        af = np.fromiter(self.makegen(), float)
        # 断言生成的数组的数据类型与预期一致
        assert_(ai32.dtype == np.dtype(np.int32))
        assert_(ai64.dtype == np.dtype(np.int64))
        assert_(af.dtype == np.dtype(float))

    # 测试从生成器中创建的数组的长度
    def test_lengths(self):
        expected = np.array(list(self.makegen()))
        a = np.fromiter(self.makegen(), int)
        a20 = np.fromiter(self.makegen(), int, 20)
        # 断言生成的数组长度与预期一致
        assert_(len(a) == len(expected))
        assert_(len(a20) == 20)
        # 测试当传入的 count 大于生成器产生的元素数量时是否会抛出 ValueError 异常
        assert_raises(ValueError, np.fromiter, self.makegen(), int, len(expected) + 10)

    # 测试从生成器中创建的数组的数值
    def test_values(self):
        expected = np.array(list(self.makegen()))
        a = np.fromiter(self.makegen(), int)
        a20 = np.fromiter(self.makegen(), int, 20)
        # 断言生成的数组中的值与预期一致
        assert_(np.alltrue(a == expected, axis=0))
        assert_(np.alltrue(a20 == expected[:20], axis=0))

    # 辅助方法，用于测试第 2592 个问题
    def load_data(self, n, eindex):
        # 在迭代器中的特定索引位置抛出异常
        for e in range(n):
            if e == eindex:
                raise NIterError(f"error at index {eindex}")
            yield e

    # 参数化测试，测试异常迭代是否能正确抛出
    @parametrize("dtype", [int])
    @parametrize("count, error_index", [(10, 5), (10, 9)])
    def test_2592(self, count, error_index, dtype):
        # 测试迭代异常是否能正确抛出。数据/生成器有 `count` 元素，但在 `error_index` 处发生错误
        iterable = self.load_data(count, error_index)
        with pytest.raises(NIterError):
            np.fromiter(iterable, dtype=dtype, count=count)

    # 跳过测试，NP_VER: 在 CI 上失败
    @skip(reason="NP_VER: fails on CI")
    def test_empty_result(self):
        # 定义一个空迭代器类
        class MyIter:
            def __length_hint__(self):
                return 10

            def __iter__(self):
                return iter([])  # 实际迭代器为空。

        # 测试从空迭代器中创建的数组的形状和数据类型
        res = np.fromiter(MyIter(), dtype="d")
        assert res.shape == (0,)
        assert res.dtype == "d"

    # 测试从迭代器中创建数组时，元素数量过少的情况
    def test_too_few_items(self):
        msg = "iterator too short: Expected 10 but iterator had only 3 items."
        with pytest.raises(ValueError, match=msg):
            np.fromiter([1, 2, 3], count=10, dtype=int)

    # 测试从迭代器中创建数组时，元素类型设置错误的情况
    def test_failed_itemsetting(self):
        with pytest.raises(TypeError):
            np.fromiter([1, None, 3], dtype=int)

        # 测试从生成器中创建数组时，出现更复杂的代码路径
        iterable = ((2, 3, 4) for i in range(5))
        with pytest.raises(ValueError):
            np.fromiter(iterable, dtype=np.dtype((int, 2)))


@instantiate_parametrized_tests
class TestNonzeroAndCountNonzero(TestCase):
    pass
    # 测试函数，用于测试 np.count_nonzero 函数在列表中的行为
    def test_count_nonzero_list(self):
        # 创建一个包含两个子列表的列表
        lst = [[0, 1, 2, 3], [1, 0, 0, 6]]
        # 断言：整个列表中非零元素的数量应为 5
        assert np.count_nonzero(lst) == 5
        # 断言：沿着列的方向，计算每列非零元素的数量，并与给定的数组进行比较
        assert_array_equal(np.count_nonzero(lst, axis=0), np.array([1, 1, 1, 2]))
        # 断言：沿着行的方向，计算每行非零元素的数量，并与给定的数组进行比较
        assert_array_equal(np.count_nonzero(lst, axis=1), np.array([3, 2]))

    # 测试函数，测试在各种情况下 np.count_nonzero 和 np.nonzero 的行为
    def test_nonzero_trivial(self):
        # 断言：空数组的非零元素数量应为 0
        assert_equal(np.count_nonzero(np.array([])), 0)
        # 断言：空数组（布尔类型）的非零元素数量应为 0
        assert_equal(np.count_nonzero(np.array([], dtype="?")), 0)
        # 断言：空数组的非零元素的索引应为空元组
        assert_equal(np.nonzero(np.array([])), ([],))

        # 断言：包含一个零元素的数组的非零元素数量应为 0
        assert_equal(np.count_nonzero(np.array([0])), 0)
        # 断言：包含一个零元素的数组（布尔类型）的非零元素数量应为 0
        assert_equal(np.count_nonzero(np.array([0], dtype="?")), 0)
        # 断言：包含一个零元素的数组的非零元素的索引应为空元组
        assert_equal(np.nonzero(np.array([0])), ([],))

        # 断言：包含一个非零元素的数组的非零元素数量应为 1
        assert_equal(np.count_nonzero(np.array([1])), 1)
        # 断言：包含一个非零元素的数组（布尔类型）的非零元素数量应为 1
        assert_equal(np.count_nonzero(np.array([1], dtype="?")), 1)
        # 断言：包含一个非零元素的数组的非零元素的索引应为 [0]
        assert_equal(np.nonzero(np.array([1])), ([0],))

    # 测试函数，测试特殊情况下的返回值类型
    def test_nonzero_trivial_differs():
        # 断言：np.count_nonzero 返回值的类型应为 np.ndarray
        assert isinstance(np.count_nonzero([]), np.ndarray)

    # 测试函数，测试一维数组的各种情况
    def test_nonzero_zerod(self):
        # 断言：包含一个零元素的一维数组的非零元素数量应为 0
        assert_equal(np.count_nonzero(np.array(0)), 0)
        # 断言：包含一个零元素的一维数组（布尔类型）的非零元素数量应为 0
        assert_equal(np.count_nonzero(np.array(0, dtype="?")), 0)

        # 断言：包含一个非零元素的一维数组的非零元素数量应为 1
        assert_equal(np.count_nonzero(np.array(1)), 1)
        # 断言：包含一个非零元素的一维数组（布尔类型）的非零元素数量应为 1
        assert_equal(np.count_nonzero(np.array(1, dtype="?")), 1)

    # 测试函数，测试特殊情况下的返回值类型
    def test_nonzero_zerod_differs():
        # 断言：np.count_nonzero 返回值的类型应为 np.ndarray
        assert isinstance(np.count_nonzero(np.array(1)), np.ndarray)

    # 测试函数，测试一维数组的非零元素计算
    def test_nonzero_onedim(self):
        # 创建一个包含多个整数的一维数组
        x = np.array([1, 0, 2, -1, 0, 0, 8])
        # 断言：一维数组中非零元素的数量应为 4
        assert_equal(np.count_nonzero(x), 4)
        # 断言：一维数组中非零元素的索引应为 [0, 2, 3, 6]
        assert_equal(np.nonzero(x), ([0, 2, 3, 6],))

    # 测试函数，测试特殊情况下的返回值类型
    def test_nonzero_onedim_differs():
        # 创建一个包含多个整数的一维数组
        x = np.array([1, 0, 2, -1, 0, 0, 8])
        # 断言：np.count_nonzero 返回值的类型应为 np.ndarray
        assert isinstance(np.count_nonzero(x), np.ndarray)

    # 测试函数，测试二维数组的非零元素计算
    def test_nonzero_twodim(self):
        # 创建一个二维数组
        x = np.array([[0, 1, 0], [2, 0, 3]])
        # 断言：将数组转换为指定数据类型后，二维数组中非零元素的数量应为 3
        assert_equal(np.count_nonzero(x.astype("i1")), 3)
        assert_equal(np.count_nonzero(x.astype("i2")), 3)
        assert_equal(np.count_nonzero(x.astype("i4")), 3)
        assert_equal(np.count_nonzero(x.astype("i8")), 3)
        # 断言：二维数组中非零元素的索引应为 ([0, 1, 1], [1, 0, 2])

        # 创建一个单位矩阵
        x = np.eye(3)
        # 断言：将数组转换为指定数据类型后，单位矩阵中非零元素的数量应为 3
        assert_equal(np.count_nonzero(x.astype("i1")), 3)
        assert_equal(np.count_nonzero(x.astype("i2")), 3)
        assert_equal(np.count_nonzero(x.astype("i4")), 3)
        assert_equal(np.count_nonzero(x.astype("i8")), 3)
        # 断言：单位矩阵中非零元素的索引应为 ([0, 1, 2], [0, 1, 2])
    def test_sparse(self):
        # 针对稀疏条件的特殊测试，布尔型数组路径
        for i in range(20):
            # 创建长度为 200 的布尔型全零数组
            c = np.zeros(200, dtype=bool)
            # 每隔 20 个位置设为 True
            c[i::20] = True
            # 断言非零元素的索引数组与预期数组相等
            assert_equal(np.nonzero(c)[0], np.arange(i, 200 + i, 20))

            # 创建长度为 400 的布尔型全零数组
            c = np.zeros(400, dtype=bool)
            # 在指定范围内设为 True
            c[10 + i : 20 + i] = True
            # 设置单个位置为 True
            c[20 + i * 2] = True
            # 断言非零元素的索引数组与预期数组相等
            assert_equal(
                np.nonzero(c)[0],
                np.concatenate((np.arange(10 + i, 20 + i), [20 + i * 2])),
            )

    def test_count_nonzero_axis(self):
        # 基本功能检查
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        # 期望结果数组
        expected = np.array([1, 1, 1, 1, 1])
        # 断言沿 axis=0 方向非零元素的数量与期望结果数组相等
        assert_array_equal(np.count_nonzero(m, axis=0), expected)

        # 期望结果数组
        expected = np.array([2, 3])
        # 断言沿 axis=1 方向非零元素的数量与期望结果数组相等
        assert_array_equal(np.count_nonzero(m, axis=1), expected)

        # 断言返回结果类型为 ndarray
        assert isinstance(np.count_nonzero(m, axis=1), np.ndarray)

        # 断言引发异常：ValueError，由于 axis 参数为 (1, 1)
        assert_raises(ValueError, np.count_nonzero, m, axis=(1, 1))
        # 断言引发异常：TypeError，由于 axis 参数为 "foo"
        assert_raises(TypeError, np.count_nonzero, m, axis="foo")
        # 断言引发异常：AxisError，由于 axis 参数为 3
        assert_raises(np.AxisError, np.count_nonzero, m, axis=3)
        # 断言引发异常：TypeError，由于 axis 参数为 ndarray 类型
        assert_raises(TypeError, np.count_nonzero, m, axis=np.array([[1], [2]]))

    @parametrize("typecode", "efdFDBbhil?")
    def test_count_nonzero_axis_all_dtypes(self, typecode):
        # 对所有数据类型进行更彻底的测试，确保 axis 参数的正确响应
        # 当 axis 参数为整数或元组时的行为正确

        # 创建指定数据类型的全零数组
        m = np.zeros((3, 3), dtype=typecode)
        # 创建指定数据类型的全一数组
        n = np.ones(1, dtype=typecode)

        # 在数组中设置非零值
        m[0, 0] = n[0]
        m[1, 0] = n[0]

        # 期望结果数组
        expected = np.array([2, 0, 0], dtype=np.intp)
        # 断言沿 axis=0 方向非零元素的数量与期望结果数组相等
        result = np.count_nonzero(m, axis=0)
        assert_array_equal(result, expected)
        # 断言结果数组的数据类型与期望结果数组相等
        assert expected.dtype == result.dtype

        # 期望结果数组
        expected = np.array([1, 1, 0], dtype=np.intp)
        # 断言沿 axis=1 方向非零元素的数量与期望结果数组相等
        result = np.count_nonzero(m, axis=1)
        assert_array_equal(result, expected)
        # 断言结果数组的数据类型与期望结果数组相等
        assert expected.dtype == result.dtype

        # 期望结果为标量 2
        expected = np.array(2)
        # 断言沿 axis=(0, 1) 方向非零元素的数量与期望结果相等
        assert_array_equal(np.count_nonzero(m, axis=(0, 1)), expected)
        # 断言沿 axis=None 方向非零元素的数量与期望结果相等
        assert_array_equal(np.count_nonzero(m, axis=None), expected)
        # 断言全局数组的非零元素数量与期望结果相等
        assert_array_equal(np.count_nonzero(m), expected)

    def test_countnonzero_axis_empty(self):
        # 对空 axis 的情况进行测试
        a = np.array([[0, 0, 1], [1, 0, 1]])
        # 断言沿空 axis 的非零元素数量与转换为布尔型后的数组相等
        assert_equal(np.count_nonzero(a, axis=()), a.astype(bool))

    def test_countnonzero_keepdims(self):
        # 对 keepdims 参数进行测试
        a = np.array([[0, 0, 1, 0], [0, 3, 5, 0], [7, 9, 2, 0]])
        # 断言沿 axis=0 方向保持维度的非零元素数量结果与期望结果相等
        assert_array_equal(np.count_nonzero(a, axis=0, keepdims=True), [[1, 2, 3, 0]])
        # 断言沿 axis=1 方向保持维度的非零元素数量结果与期望结果相等
        assert_array_equal(np.count_nonzero(a, axis=1, keepdims=True), [[1], [2], [3]])
        # 断言全局数组的非零元素数量结果与期望结果相等
        assert_array_equal(np.count_nonzero(a, keepdims=True), [[6]])
        # 断言沿 axis=1 方向保持维度的非零元素数量结果为 ndarray 类型
        assert isinstance(np.count_nonzero(a, axis=1, keepdims=True), np.ndarray)
# 定义一个测试类 TestIndex，继承自 TestCase
class TestIndex(TestCase):
    
    # 定义测试方法 test_boolean
    def test_boolean(self):
        # 使用 rand 函数创建一个 3x5x8 的随机数组 a
        a = rand(3, 5, 8)
        # 使用 rand 函数创建一个 5x8 的随机数组 V
        V = rand(5, 8)
        # 使用 randint 函数生成一个包含 15 个元素的随机整数数组 g1，取值范围在 [0, 5)
        g1 = randint(0, 5, size=15)
        # 使用 randint 函数生成一个包含 15 个元素的随机整数数组 g2，取值范围在 [0, 8)
        g2 = randint(0, 8, size=15)
        # 根据索引 g1 和 g2 将数组 V 中对应位置的元素取相反数
        V[g1, g2] = -V[g1, g2]
        # 断言：判断条件成立，验证条件中的数组切片与原数组 a 切片相等
        assert_(
            (np.array([a[0][V > 0], a[1][V > 0], a[2][V > 0]]) == a[:, V > 0]).all()
        )

    # 定义测试方法 test_boolean_edgecase
    def test_boolean_edgecase(self):
        # 创建一个空的 int32 类型数组 a
        a = np.array([], dtype="int32")
        # 创建一个空的布尔类型数组 b
        b = np.array([], dtype="bool")
        # 根据布尔数组 b 从数组 a 中选择元素，形成数组 c
        c = a[b]
        # 断言：验证数组 c 为空数组
        assert_equal(c, [])
        # 断言：验证数组 c 的数据类型为 int32
        assert_equal(c.dtype, np.dtype("int32"))


# 为 TestBinaryRepr 类添加装饰器 @xpassIfTorchDynamo
@xpassIfTorchDynamo  # (reason="TODO")
class TestBinaryRepr(TestCase):
    
    # 定义测试方法 test_zero
    def test_zero(self):
        # 断言：验证二进制表示的 0 应为字符串 "0"
        assert_equal(np.binary_repr(0), "0")

    # 定义测试方法 test_positive
    def test_positive(self):
        # 断言：验证十进制数 10 的二进制表示应为字符串 "1010"
        assert_equal(np.binary_repr(10), "1010")
        # 断言：验证十进制数 12522 的二进制表示应为字符串 "11000011101010"
        assert_equal(np.binary_repr(12522), "11000011101010")
        # 断言：验证十进制数 10736848 的二进制表示应为字符串 "101000111101010011010000"
        assert_equal(np.binary_repr(10736848), "101000111101010011010000")

    # 定义测试方法 test_negative
    def test_negative(self):
        # 断言：验证十进制数 -1 的二进制表示应为字符串 "-1"
        assert_equal(np.binary_repr(-1), "-1")
        # 断言：验证十进制数 -10 的二进制表示应为字符串 "-1010"
        assert_equal(np.binary_repr(-10), "-1010")
        # 断言：验证十进制数 -12522 的二进制表示应为字符串 "-11000011101010"
        assert_equal(np.binary_repr(-12522), "-11000011101010")
        # 断言：验证十进制数 -10736848 的二进制表示应为字符串 "-101000111101010011010000"
        assert_equal(np.binary_repr(-10736848), "-101000111101010011010000")

    # 定义测试方法 test_sufficient_width
    def test_sufficient_width(self):
        # 断言：验证二进制表示的 0，宽度为 5 时应为字符串 "00000"
        assert_equal(np.binary_repr(0, width=5), "00000")
        # 断言：验证十进制数 10，宽度为 7 时的二进制表示应为字符串 "0001010"
        assert_equal(np.binary_repr(10, width=7), "0001010")
        # 断言：验证十进制数 -5，宽度为 7 时的二进制表示应为字符串 "1111011"
        assert_equal(np.binary_repr(-5, width=7), "1111011")

    # 定义测试方法 test_neg_width_boundaries
    def test_neg_width_boundaries(self):
        # 注释：检查 GitHub 问题编号 8670

        # 断言：验证对于宽度为 8 的情况下，十进制数 -128 的二进制表示应为字符串 "10000000"
        assert_equal(np.binary_repr(-128, width=8), "10000000")

        # 使用循环遍历宽度从 1 到 10 的范围
        for width in range(1, 11):
            # 计算负数 -(2 ** (width - 1)) 的二进制表示期望值，例如宽度为 5 时为 "10000"
            exp = "1" + (width - 1) * "0"
            # 断言：验证负数 -(2 ** (width - 1)) 的二进制表示是否与期望值相同
            assert_equal(np.binary_repr(-(2 ** (width - 1)), width=width), exp)

    # 定义测试方法 test_large_neg_int64
    def test_large_neg_int64(self):
        # 注释：检查 GitHub 问题编号 14289.

        # 断言：验证 int64 类型的 -(2**62) 的二进制表示为字符串 "11" + "0" * 62
        assert_equal(np.binary_repr(np.int64(-(2**62)), width=64), "11" + "0" * 62)


# 为 TestBaseRepr 类添加装饰器 @xpassIfTorchDynamo
@xpassIfTorchDynamo  # (reason="TODO")
class TestBaseRepr(TestCase):
    
    # 定义测试方法 test_base3
    def test_base3(self):
        # 断言：验证 3 的 5 次方的三进制表示应为字符串 "100000"
        assert_equal(np.base_repr(3**5, 3), "100000")

    # 定义测试方法 test_positive
    def test_positive(self):
        # 断言：验证十进制数 12 的十进制表示应为字符串 "12"
        assert_equal(np.base_repr(12, 10), "12")
        # 断言：验证十进制数 12 的十进制表示，宽度为 4 时应为字符串 "000012"
        assert_equal(np.base_repr(12, 10, 4), "000012")
        # 断言：验证十进制数 12 的 4 进制表示应为字符串 "30"
        assert_equal(np.base_repr(12, 4), "30")
        # 断言：验证十进制数 3731624803700888 的 36 进制表示应为字符串 "10QR0ROFCEW"
        assert_equal(np.base_repr(3731624803700888, 36), "10QR0ROFCEW")

    # 定义测试方法 test_negative
    def test_negative(self):
        # 断言：验证十进制数 -12 的十进制表示应为字符串 "-12"
        assert_equal(np.base_repr(-12, 10), "-12")
        # 断言：验证十进制数 -12 的十进制表示，宽度为 4 时应为字符串 "-000012"
        assert_equal(np.base_repr(-12, 10, 4), "-000012")
        # 断言：验证十进制数 -12 的 4 进制表示应为字符串 "-30"
        assert_equal(np.base_repr(-12, 4), "-30")

    # 定义测试方法 test_base_range
    def test_base_range(self):
        # 断言：验证当 base 参数为 1 时，抛出 ValueError 异常
        with assert_raises(ValueError):
            np.base_repr(1, 1)
        # 断言：验证当 base 参数为 37 时，抛出 ValueError 异常
        with assert_raises(ValueError):
            np.base_repr(1, 37)


# 定义测试类 TestArrayComparisons，继承自 TestCase
class TestArrayComparisons(TestCase):
    # 留空，用于未来可能的测试添加
    # 定义测试方法，用于测试 np.array_equal 函数的行为
    def test_array_equal(self):
        # 测试两个相同数组是否相等
        res = np.array_equal(np.array([1, 2]), np.array([1, 2]))
        # 断言结果为真
        assert_(res)
        # 断言结果类型为布尔值
        assert_(type(res) is bool)
        # 测试两个长度不同的数组是否相等
        res = np.array_equal(np.array([1, 2]), np.array([1, 2, 3]))
        # 断言结果为假
        assert_(not res)
        # 断言结果类型为布尔值
        assert_(type(res) is bool)
        # 测试两个不同数组是否相等
        res = np.array_equal(np.array([1, 2]), np.array([3, 4]))
        # 断言结果为假
        assert_(not res)
        # 断言结果类型为布尔值
        assert_(type(res) is bool)
        # 测试两个部分相同数组是否相等
        res = np.array_equal(np.array([1, 2]), np.array([1, 3]))
        # 断言结果为假
        assert_(not res)
        # 断言结果类型为布尔值
        assert_(type(res) is bool)

    # 定义测试方法，测试带有 equal_nan 参数的 np.array_equal 函数
    def test_array_equal_equal_nan(self):
        # 测试数组包含 NaN 值的情况
        a1 = np.array([1, 2, np.nan])
        a2 = np.array([1, np.nan, 2])
        a3 = np.array([1, 2, np.inf])

        # 默认 equal_nan=False 的情况
        assert_(not np.array_equal(a1, a1))
        # 测试 equal_nan=True 的情况
        assert_(np.array_equal(a1, a1, equal_nan=True))
        # 测试不相同顺序但包含相同值的数组，使用 equal_nan=True
        assert_(not np.array_equal(a1, a2, equal_nan=True))
        # 测试 NaN 值不与 inf 值混淆的情况，使用 equal_nan=True
        assert_(not np.array_equal(a1, a3, equal_nan=True))
        # 测试 0 维数组的情况
        a = np.array(np.nan)
        assert_(not np.array_equal(a, a))
        assert_(np.array_equal(a, a, equal_nan=True))
        # 测试非浮点数类型的数组，equal_nan 参数不应影响结果
        a = np.array([1, 2, 3], dtype=int)
        assert_(np.array_equal(a, a))
        assert_(np.array_equal(a, a, equal_nan=True))
        # 测试多维数组的情况
        a = np.array([[0, 1], [np.nan, 1]])
        assert_(not np.array_equal(a, a))
        assert_(np.array_equal(a, a, equal_nan=True))
        # 测试复数值的情况
        a, b = [np.array([1 + 1j])] * 2
        a.real, b.imag = np.nan, np.nan
        assert_(not np.array_equal(a, b, equal_nan=False))
        assert_(np.array_equal(a, b, equal_nan=True))

    # 定义测试方法，测试 numpy 数组与 None 的逐元素比较
    def test_none_compares_elementwise(self):
        # 创建一个包含全 1 的数组
        a = np.ones(3)
        # 断言数组与 None 逐元素比较的结果
        assert_equal(a.__eq__(None), [False, False, False])
        assert_equal(a.__ne__(None), [True, True, True])
    # 定义一个测试方法，用于验证两个 NumPy 数组是否在形状和元素上等效
    def test_array_equiv(self):
        # 调用 np.array_equiv() 检查两个相同数组是否等效
        res = np.array_equiv(np.array([1, 2]), np.array([1, 2]))
        # 使用 assert_() 断言 res 应为 True
        assert_(res)
        # 使用 assert_() 断言 res 的类型应为 bool
        assert_(type(res) is bool)

        # 再次调用 np.array_equiv() 检查不同长度的数组是否等效
        res = np.array_equiv(np.array([1, 2]), np.array([1, 2, 3]))
        # 使用 assert_() 断言 res 应为 False
        assert_(not res)
        # 使用 assert_() 断言 res 的类型应为 bool
        assert_(type(res) is bool)

        # 检查不同元素的数组是否等效
        res = np.array_equiv(np.array([1, 2]), np.array([3, 4]))
        # 使用 assert_() 断言 res 应为 False
        assert_(not res)
        # 使用 assert_() 断言 res 的类型应为 bool
        assert_(type(res) is bool)

        # 检查顺序不同的数组是否等效
        res = np.array_equiv(np.array([1, 2]), np.array([1, 3]))
        # 使用 assert_() 断言 res 应为 False
        assert_(not res)
        # 使用 assert_() 断言 res 的类型应为 bool
        assert_(type(res) is bool)

        # 检查重复元素数组与不同形状的数组是否等效
        res = np.array_equiv(np.array([1, 1]), np.array([1]))
        # 使用 assert_() 断言 res 应为 True
        assert_(res)
        # 使用 assert_() 断言 res 的类型应为 bool
        assert_(type(res) is bool)

        # 检查重复元素数组与包含重复的不同形状的数组是否等效
        res = np.array_equiv(np.array([1, 1]), np.array([[1], [1]]))
        # 使用 assert_() 断言 res 应为 True
        assert_(res)
        # 使用 assert_() 断言 res 的类型应为 bool
        assert_(type(res) is bool)

        # 检查数组与包含不同元素的数组是否等效
        res = np.array_equiv(np.array([1, 2]), np.array([2]))
        # 使用 assert_() 断言 res 应为 False
        assert_(not res)
        # 使用 assert_() 断言 res 的类型应为 bool
        assert_(type(res) is bool)

        # 检查数组与包含不同形状的数组是否等效
        res = np.array_equiv(np.array([1, 2]), np.array([[1], [2]]))
        # 使用 assert_() 断言 res 应为 False
        assert_(not res)
        # 使用 assert_() 断言 res 的类型应为 bool
        assert_(type(res) is bool)

        # 检查数组与形状不同的二维数组是否等效
        res = np.array_equiv(
            np.array([1, 2]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        )
        # 使用 assert_() 断言 res 应为 False
        assert_(not res)
        # 使用 assert_() 断言 res 的类型应为 bool
        assert_(type(res) is bool)
@instantiate_parametrized_tests
class TestClip(TestCase):
    # 设置测试环境，在每个测试方法执行前调用
    def setUp(self):
        super().setUp()
        self.nr = 5  # 定义行数
        self.nc = 3  # 定义列数

    # 快速剪切函数
    def fastclip(self, a, m, M, out=None, casting=None):
        if out is None:
            if casting is None:
                return a.clip(m, M)  # 如果没有指定输出和类型转换方式，则直接使用 ndarray 的 clip 方法
            else:
                return a.clip(m, M, casting=casting)  # 如果没有指定输出但指定了类型转换方式，则使用指定的类型转换方式进行剪切
        else:
            if casting is None:
                return a.clip(m, M, out)  # 如果指定了输出但没有指定类型转换方式，则使用指定的输出进行剪切
            else:
                return a.clip(m, M, out, casting=casting)  # 如果指定了输出和类型转换方式，则使用指定的输出和类型转换方式进行剪切

    # 使用慢速剪切方法
    def clip(self, a, m, M, out=None):
        # use slow-clip
        selector = np.less(a, m) + 2 * np.greater(a, M)  # 使用 np.less 和 np.greater 创建选择器
        return selector.choose((a, m, M), out=out)  # 使用选择器进行剪切，将结果存储到指定的输出中

    # 生成测试数据的辅助函数
    def _generate_data(self, n, m):
        return randn(n, m)  # 返回一个 n 行 m 列的随机数组成的 ndarray

    def _generate_data_complex(self, n, m):
        return randn(n, m) + 1.0j * rand(n, m)  # 返回一个包含实部和虚部的随机复数数组

    def _generate_flt_data(self, n, m):
        return (randn(n, m)).astype(np.float32)  # 返回一个 n 行 m 列的随机浮点数数组，并将其转换为 np.float32 类型

    def _neg_byteorder(self, a):
        a = np.asarray(a)  # 将输入转换为 ndarray 类型
        if sys.byteorder == "little":
            a = a.astype(a.dtype.newbyteorder(">"))  # 如果系统字节序为小端，则将数组类型转换为大端字节序
        else:
            a = a.astype(a.dtype.newbyteorder("<"))  # 否则将数组类型转换为小端字节序
        return a  # 返回转换后的数组

    def _generate_non_native_data(self, n, m):
        data = randn(n, m)  # 生成一个 n 行 m 列的随机数组成的 ndarray
        data = self._neg_byteorder(data)  # 将数据转换为非本机字节顺序
        assert_(not data.dtype.isnative)  # 断言数据类型不是本机字节顺序
        return data  # 返回转换后的数据

    def _generate_int_data(self, n, m):
        return (10 * rand(n, m)).astype(np.int64)  # 生成一个 n 行 m 列的随机整数数组，并将其转换为 np.int64 类型

    def _generate_int32_data(self, n, m):
        return (10 * rand(n, m)).astype(np.int32)  # 生成一个 n 行 m 列的随机整数数组，并将其转换为 np.int32 类型

    # 真实测试用例开始

    @parametrize("dtype", "?bhilBfd")
    def test_ones_pathological(self, dtype):
        # 为了保留 gh-12519 中描述的行为，将 amin 大于 amax 的行为进行测试
        arr = np.ones(10, dtype=dtype)  # 生成一个包含 10 个元素的全为 1 的数组，数据类型为 dtype
        expected = np.zeros(10, dtype=dtype)  # 生成一个包含 10 个元素的全为 0 的数组，数据类型为 dtype
        actual = np.clip(arr, 1, 0)  # 对 arr 数组进行剪切，amin=1，amax=0
        assert_equal(actual, expected)  # 断言剪切后的结果与预期相等

    @parametrize("dtype", "eFD")
    def test_ones_pathological_2(self, dtype):
        if dtype in "FD":
            # FIXME: make xfail
            raise SkipTest("torch.clamp not implemented for complex types")  # 如果 dtype 是复数类型，则跳过测试
        # 为了保留 gh-12519 中描述的行为，将 amin 大于 amax 的行为进行测试
        arr = np.ones(10, dtype=dtype)  # 生成一个包含 10 个元素的全为 1 的数组，数据类型为 dtype
        expected = np.zeros(10, dtype=dtype)  # 生成一个包含 10 个元素的全为 0 的数组，数据类型为 dtype
        actual = np.clip(arr, 1, 0)  # 对 arr 数组进行剪切，amin=1，amax=0
        assert_equal(actual, expected)  # 断言剪切后的结果与预期相等

    def test_simple_double(self):
        # 测试输入为本机 double 类型的数组，并使用标量 min/max 进行剪切
        a = self._generate_data(self.nr, self.nc)  # 生成一个大小为 nr x nc 的随机数组成的 ndarray
        m = 0.1  # 定义最小值
        M = 0.6  # 定义最大值
        ac = self.fastclip(a, m, M)  # 使用 fastclip 方法进行剪切
        act = self.clip(a, m, M)  # 使用 clip 方法进行剪切
        assert_array_equal(ac, act)  # 断言两种剪切方法的结果相等
    def test_simple_int(self):
        # Test native int input with scalar min/max.
        # 生成一个整数类型的测试数据
        a = self._generate_int_data(self.nr, self.nc)
        # 将数据类型转换为整数类型
        a = a.astype(int)
        # 设置最小值和最大值
        m = -2
        M = 4
        # 使用快速剪裁函数进行剪裁操作
        ac = self.fastclip(a, m, M)
        # 使用标准剪裁函数进行剪裁操作
        act = self.clip(a, m, M)
        # 断言剪裁后的数组是否相等
        assert_array_equal(ac, act)

    def test_array_double(self):
        # Test native double input with array min/max.
        # 生成一个双精度浮点数类型的测试数据
        a = self._generate_data(self.nr, self.nc)
        # 生成一个全零数组作为最小值数组
        m = np.zeros(a.shape)
        # 将最小值数组的每个元素增加0.5作为最大值数组
        M = m + 0.5
        # 使用快速剪裁函数进行剪裁操作
        ac = self.fastclip(a, m, M)
        # 使用标准剪裁函数进行剪裁操作
        act = self.clip(a, m, M)
        # 断言剪裁后的数组是否相等
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo  # (reason="byteorder not supported in torch")
    def test_simple_nonnative(self):
        # Test non native double input with scalar min/max.
        # Test native double input with non native double scalar min/max.
        # 生成一个非本地（可能不是C语言的内存布局）双精度浮点数类型的测试数据
        a = self._generate_non_native_data(self.nr, self.nc)
        # 设置最小值和最大值
        m = -0.5
        M = 0.6
        # 使用快速剪裁函数进行剪裁操作
        ac = self.fastclip(a, m, M)
        # 使用标准剪裁函数进行剪裁操作
        act = self.clip(a, m, M)
        # 断言剪裁后的数组是否相等
        assert_array_equal(ac, act)

        # Test native double input with non native double scalar min/max.
        # 生成一个双精度浮点数类型的测试数据
        a = self._generate_data(self.nr, self.nc)
        # 设置最小值和通过_neg_byteorder函数得到的最大值
        m = -0.5
        M = self._neg_byteorder(0.6)
        # 断言最大值的数据类型是否为非本地
        assert_(not M.dtype.isnative)
        # 使用快速剪裁函数进行剪裁操作
        ac = self.fastclip(a, m, M)
        # 使用标准剪裁函数进行剪裁操作
        act = self.clip(a, m, M)
        # 断言剪裁后的数组是否相等
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo  # (reason="clamp not supported for complex")
    def test_simple_complex(self):
        # Test native complex input with native double scalar min/max.
        # Test native input with complex double scalar min/max.
        # 生成一个复数类型的测试数据
        a = 3 * self._generate_data_complex(self.nr, self.nc)
        # 设置最小值和最大值
        m = -0.5
        M = 1.0
        # 使用快速剪裁函数进行剪裁操作
        ac = self.fastclip(a, m, M)
        # 使用标准剪裁函数进行剪裁操作
        act = self.clip(a, m, M)
        # 断言剪裁后的数组是否相等
        assert_array_equal(ac, act)

        # Test native input with complex double scalar min/max.
        # 生成一个双精度浮点数类型的测试数据
        a = 3 * self._generate_data(self.nr, self.nc)
        # 设置最小值和最大值为复数
        m = -0.5 + 1.0j
        M = 1.0 + 2.0j
        # 使用快速剪裁函数进行剪裁操作
        ac = self.fastclip(a, m, M)
        # 使用标准剪裁函数进行剪裁操作
        act = self.clip(a, m, M)
        # 断言剪裁后的数组是否相等
        assert_array_equal(ac, act)

    @xfail  # (reason="clamp not supported for complex")
    def test_clip_complex(self):
        # Address Issue gh-5354 for clipping complex arrays
        # Test native complex input without explicit min/max
        # ie, either min=None or max=None
        # 生成一个复数类型的测试数据
        a = np.ones(10, dtype=complex)
        # 计算数组的最小值和最大值
        m = a.min()
        M = a.max()
        # 使用快速剪裁函数进行剪裁操作
        am = self.fastclip(a, m, None)
        aM = self.fastclip(a, None, M)
        # 断言剪裁后的数组是否相等
        assert_array_equal(am, a)
        assert_array_equal(aM, a)

    def test_clip_non_contig(self):
        # Test clip for non contiguous native input and native scalar min/max.
        # 生成一个双精度浮点数类型的测试数据
        a = self._generate_data(self.nr * 2, self.nc * 3)
        # 取出部分数据，形成非连续数组
        a = a[::2, ::3]
        # 断言数组是否为列优先存储（Fortran风格）或行优先存储（C风格）
        assert_(not a.flags["F_CONTIGUOUS"])
        assert_(not a.flags["C_CONTIGUOUS"])
        # 使用快速剪裁函数进行剪裁操作
        ac = self.fastclip(a, -1.6, 1.7)
        # 使用标准剪裁函数进行剪裁操作
        act = self.clip(a, -1.6, 1.7)
        # 断言剪裁后的数组是否相等
        assert_array_equal(ac, act)
    # 定义测试方法，用于测试简单的输出
    def test_simple_out(self):
        # 生成测试数据，self.nr 和 self.nc 是行数和列数
        a = self._generate_data(self.nr, self.nc)
        # 设置最小值和最大值
        m = -0.5
        M = 0.6
        # 初始化两个与 a 形状相同的零数组
        ac = np.zeros(a.shape)
        act = np.zeros(a.shape)
        # 使用 fastclip 方法对 a 进行裁剪操作，结果存入 ac
        self.fastclip(a, m, M, ac)
        # 使用 clip 方法对 a 进行裁剪操作，结果存入 act
        self.clip(a, m, M, act)
        # 断言 ac 和 act 数组相等
        assert_array_equal(ac, act)

    # 参数化测试方法，用于测试不同参数的情况
    @parametrize(
        "casting",
        [
            subtest(None, decorators=[xfail]),  # 设置为 None 的子测试，预期会失败
            subtest("unsafe", decorators=[xpassIfTorchDynamo]),  # 设置为 "unsafe" 的子测试
        ],
    )
    def test_simple_int32_inout(self, casting):
        # 测试输入为 native int32 类型，输出为 int32 类型的情况
        a = self._generate_int32_data(self.nr, self.nc)
        # 设置最小值和最大值为 np.float64 类型
        m = np.float64(0)
        M = np.float64(2)
        # 初始化一个 int32 类型的零数组 ac，并复制给 act
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        # 如果 casting 不为 None，则使用 fastclip 方法进行裁剪操作
        if casting is not None:
            self.fastclip(a, m, M, ac, casting=casting)
        # 使用 clip 方法对 a 进行裁剪操作，结果存入 act
        self.clip(a, m, M, act)
        # 断言 ac 和 act 数组相等
        assert_array_equal(ac, act)

    # 测试方法，测试输入为 native int32 类型，输出为 int64 类型的情况
    def test_simple_int64_out(self):
        a = self._generate_int32_data(self.nr, self.nc)
        # 设置 int32 类型的最小值和最大值
        m = np.int32(-1)
        M = np.int32(1)
        # 初始化一个 int64 类型的零数组 ac，并复制给 act
        ac = np.zeros(a.shape, dtype=np.int64)
        act = ac.copy()
        # 使用 fastclip 方法对 a 进行裁剪操作，结果存入 ac
        self.fastclip(a, m, M, ac)
        # 使用 clip 方法对 a 进行裁剪操作，结果存入 act
        self.clip(a, m, M, act)
        # 断言 ac 和 act 数组相等
        assert_array_equal(ac, act)

    # 标记为预期失败的测试方法
    @xfail  # (reason="FIXME arrays not equal")
    def test_simple_int64_inout(self):
        # 测试输入为 native int32 类型，输出为 int32 类型的情况
        a = self._generate_int32_data(self.nr, self.nc)
        # 设置最小值为全零的浮点数组，最大值为 np.float64(1)
        m = np.zeros(a.shape, np.float64)
        M = np.float64(1)
        # 初始化一个 int32 类型的零数组 ac，并复制给 act
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        # 使用 clip 方法对 a 进行裁剪操作，结果存入 act
        self.clip(a, m, M, act)
        # 断言 ac 和 act 数组相等
        assert_array_equal(ac, act)

    # 标记为预期失败的测试方法
    @xfail  # (reason="FIXME arrays not equal")
    def test_simple_int32_out(self):
        # 测试输入为 native double 类型，输出为 int 类型的情况
        a = self._generate_data(self.nr, self.nc)
        # 设置最小值和最大值
        m = -1.0
        M = 2.0
        # 初始化一个 int32 类型的零数组 ac，并复制给 act
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        # 使用 clip 方法对 a 进行裁剪操作，结果存入 act
        self.clip(a, m, M, act)
        # 断言 ac 和 act 数组相等
        assert_array_equal(ac, act)

    # 测试方法，测试 native double 输入，使用 array 形式的最小值和最大值进行就地操作
    def test_simple_inplace_01(self):
        a = self._generate_data(self.nr, self.nc)
        # 复制 a 到 ac
        ac = a.copy()
        # 设置最小值为全零的数组，最大值为 1.0
        m = np.zeros(a.shape)
        M = 1.0
        # 使用 fastclip 方法对 a 进行就地裁剪操作
        self.fastclip(a, m, M, a)
        # 使用 clip 方法对 ac 进行裁剪操作
        self.clip(a, m, M, ac)
        # 断言 a 和 ac 数组相等
        assert_array_equal(a, ac)

    # 测试方法，测试 native double 输入，使用 scalar 形式的最小值和最大值进行就地操作
    def test_simple_inplace_02(self):
        a = self._generate_data(self.nr, self.nc)
        # 复制 a 到 ac
        ac = a.copy()
        # 设置最小值和最大值
        m = -0.5
        M = 0.6
        # 使用 fastclip 方法对 a 进行就地裁剪操作
        self.fastclip(a, m, M, a)
        # 使用 clip 方法对 ac 进行裁剪操作
        self.clip(ac, m, M, ac)
        # 断言 a 和 ac 数组相等
        assert_array_equal(a, ac)
    def test_noncontig_inplace(self):
        # 测试非连续的双输入，使用双标量最小/最大值进行原地操作。
        # 生成一个大小为 self.nr * 2，self.nc * 3 的数据数组
        a = self._generate_data(self.nr * 2, self.nc * 3)
        # 按照步长为2和3对数组 a 进行切片操作
        a = a[::2, ::3]
        # 断言数组 a 不是 F 风格连续的
        assert_(not a.flags["F_CONTIGUOUS"])
        # 断言数组 a 不是 C 风格连续的
        assert_(not a.flags["C_CONTIGUOUS"])
        # 创建数组 a 的副本 ac
        ac = a.copy()
        # 设置最小值 m 和最大值 M
        m = -0.5
        M = 0.6
        # 使用 self.fastclip 方法进行裁剪操作，原地修改数组 a
        self.fastclip(a, m, M, a)
        # 使用 self.clip 方法对数组 ac 进行裁剪操作
        self.clip(ac, m, M, ac)
        # 断言数组 a 和 ac 相等
        assert_array_equal(a, ac)

    def test_type_cast_01(self):
        # 测试原生双精度输入与标量最小/最大值。
        # 生成一个大小为 self.nr, self.nc 的数据数组 a
        a = self._generate_data(self.nr, self.nc)
        # 设置最小值 m 和最大值 M
        m = -0.5
        M = 0.6
        # 使用 self.fastclip 方法对数组 a 进行裁剪操作，返回结果给 ac
        ac = self.fastclip(a, m, M)
        # 使用 self.clip 方法对数组 a 进行裁剪操作，返回结果给 act
        act = self.clip(a, m, M)
        # 断言数组 ac 和 act 相等
        assert_array_equal(ac, act)

    def test_type_cast_02(self):
        # 测试原生 int32 输入与 int32 标量最小/最大值。
        # 生成一个大小为 self.nr, self.nc 的整数数据数组 a
        a = self._generate_int_data(self.nr, self.nc)
        # 将数组 a 转换为 int32 类型
        a = a.astype(np.int32)
        # 设置最小值 m 和最大值 M
        m = -2
        M = 4
        # 使用 self.fastclip 方法对数组 a 进行裁剪操作，返回结果给 ac
        ac = self.fastclip(a, m, M)
        # 使用 self.clip 方法对数组 a 进行裁剪操作，返回结果给 act
        act = self.clip(a, m, M)
        # 断言数组 ac 和 act 相等
        assert_array_equal(ac, act)

    def test_type_cast_03(self):
        # 测试原生 int32 输入与 float64 标量最小/最大值。
        # 生成一个大小为 self.nr, self.nc 的 int32 类型数据数组 a
        a = self._generate_int32_data(self.nr, self.nc)
        # 设置最小值 m 和最大值 M，转换为 float64 类型
        m = -2
        M = 4
        # 使用 self.fastclip 方法对数组 a 进行裁剪操作，传入 float64 类型的 m 和 M
        ac = self.fastclip(a, np.float64(m), np.float64(M))
        # 使用 self.clip 方法对数组 a 进行裁剪操作，传入 float64 类型的 m 和 M
        act = self.clip(a, np.float64(m), np.float64(M))
        # 断言数组 ac 和 act 相等
        assert_array_equal(ac, act)

    def test_type_cast_04(self):
        # 测试原生 int32 输入与 float32 标量最小/最大值。
        # 生成一个大小为 self.nr, self.nc 的 int32 类型数据数组 a
        a = self._generate_int32_data(self.nr, self.nc)
        # 设置最小值 m 和最大值 M，转换为 float32 类型
        m = np.float32(-2)
        M = np.float32(4)
        # 使用 self.fastclip 方法对数组 a 进行裁剪操作，传入 float32 类型的 m 和 M
        act = self.fastclip(a, m, M)
        # 使用 self.clip 方法对数组 a 进行裁剪操作，传入 float32 类型的 m 和 M
        ac = self.clip(a, m, M)
        # 断言数组 ac 和 act 相等
        assert_array_equal(ac, act)

    def test_type_cast_05(self):
        # 测试原生 int32 输入与双精度数组最小/最大值。
        # 生成一个大小为 self.nr, self.nc 的整数数据数组 a
        a = self._generate_int_data(self.nr, self.nc)
        # 设置最小值 m 和最大值 M
        m = -0.5
        M = 1.0
        # 使用 self.fastclip 方法对数组 a 进行裁剪操作，传入 m 乘以全零数组的结果和 M
        ac = self.fastclip(a, m * np.zeros(a.shape), M)
        # 使用 self.clip 方法对数组 a 进行裁剪操作，传入 m 乘以全零数组的结果和 M
        act = self.clip(a, m * np.zeros(a.shape), M)
        # 断言数组 ac 和 act 相等
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo  # (reason="newbyteorder not supported")
    def test_type_cast_06(self):
        # 测试原生输入与非原生标量最小/最大值。
        # 生成一个大小为 self.nr, self.nc 的数据数组 a
        a = self._generate_data(self.nr, self.nc)
        # 设置最小值 m 和经过 _neg_byteorder 处理后的 m_s
        m = 0.5
        m_s = self._neg_byteorder(m)
        # 设置最大值 M
        M = 1.0
        # 使用 self.clip 方法对数组 a 进行裁剪操作，传入 m_s 和 M
        act = self.clip(a, m_s, M)
        # 使用 self.fastclip 方法对数组 a 进行裁剪操作，传入 m_s 和 M
        ac = self.fastclip(a, m_s, M)
        # 断言数组 ac 和 act 相等
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo  # (reason="newbyteorder not supported")
    def test_type_cast_07(self):
        # 测试非原生输入与原生数组最小/最大值。
        # 生成一个大小为 self.nr, self.nc 的数据数组 a
        a = self._generate_data(self.nr, self.nc)
        # 设置最小值 m 为 -0.5 乘以全 1 数组的结果
        m = -0.5 * np.ones(a.shape)
        # 设置最大值 M
        M = 1.0
        # 经过 _neg_byteorder 处理后的 a_s
        a_s = self._neg_byteorder(a)
        # 断言 a_s 的数据类型不是原生的
        assert_(not a_s.dtype.isnative)
        # 使用 a_s 的 clip 方法进行裁剪操作，传入 m 和 M
        act = a_s.clip(m, M)
        # 使用 self.fastclip 方法对 a_s 进行裁剪操作，传入 m 和 M
        ac = self.fastclip(a_s, m, M)
        # 断言数组 ac 和 act 相等
        assert_array_equal(ac, act)
    def test_type_cast_08(self):
        # 测试非本机类型与本机标量最小/最大值。
        # 生成数据
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 1.0
        # 将数据转换为非本机字节顺序
        a_s = self._neg_byteorder(a)
        # 断言数据类型不是本机类型
        assert_(not a_s.dtype.isnative)
        # 使用fastclip函数对数据进行裁剪
        ac = self.fastclip(a_s, m, M)
        # 使用clip函数对数据进行裁剪
        act = a_s.clip(m, M)
        # 断言两个结果数组相等
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo  # (reason="newbyteorder not supported")
    def test_type_cast_09(self):
        # 测试本机类型与非本机数组最小/最大值。
        # 生成数据
        a = self._generate_data(self.nr, self.nc)
        m = -0.5 * np.ones(a.shape)
        M = 1.0
        # 将数据转换为非本机字节顺序
        m_s = self._neg_byteorder(m)
        # 断言数据类型不是本机类型
        assert_(not m_s.dtype.isnative)
        # 使用fastclip函数对数据进行裁剪
        ac = self.fastclip(a, m_s, M)
        # 使用clip函数对数据进行裁剪
        act = self.clip(a, m_s, M)
        # 断言两个结果数组相等
        assert_array_equal(ac, act)

    def test_type_cast_10(self):
        # 测试本机int32类型与浮点数最小/最大值以及浮点数输出参数。
        # 生成数据
        a = self._generate_int_data(self.nr, self.nc)
        b = np.zeros(a.shape, dtype=np.float32)
        m = np.float32(-0.5)
        M = np.float32(1)
        # 使用clip函数对数据进行裁剪
        act = self.clip(a, m, M, out=b)
        # 使用fastclip函数对数据进行裁剪
        ac = self.fastclip(a, m, M, out=b)
        # 断言两个结果数组相等
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo  # (reason="newbyteorder not supported")
    def test_type_cast_11(self):
        # 测试非本机类型与本机标量最小/最大值以及非本机输出参数。
        # 生成数据
        a = self._generate_non_native_data(self.nr, self.nc)
        b = a.copy()
        b = b.astype(b.dtype.newbyteorder(">"))
        bt = b.copy()
        m = -0.5
        M = 1.0
        # 使用fastclip函数对数据进行裁剪
        self.fastclip(a, m, M, out=b)
        # 使用clip函数对数据进行裁剪
        self.clip(a, m, M, out=bt)
        # 断言两个结果数组相等
        assert_array_equal(b, bt)

    def test_type_cast_12(self):
        # 测试本机int32输入以及整数最小/最大值和浮点数输出。
        # 生成数据
        a = self._generate_int_data(self.nr, self.nc)
        b = np.zeros(a.shape, dtype=np.float32)
        m = np.int32(0)
        M = np.int32(1)
        # 使用clip函数对数据进行裁剪
        act = self.clip(a, m, M, out=b)
        # 使用fastclip函数对数据进行裁剪
        ac = self.fastclip(a, m, M, out=b)
        # 断言两个结果数组相等
        assert_array_equal(ac, act)

    def test_clip_with_out_simple(self):
        # 测试本机双精度输入与标量最小/最大值。
        # 生成数据
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = np.zeros(a.shape)
        act = np.zeros(a.shape)
        # 使用fastclip函数对数据进行裁剪
        self.fastclip(a, m, M, ac)
        # 使用clip函数对数据进行裁剪
        self.clip(a, m, M, act)
        # 断言两个结果数组相等
        assert_array_equal(ac, act)

    @xfail  # (reason="FIXME arrays not equal")
    def test_clip_with_out_simple2(self):
        # 测试本机int32输入与双精度最小/最大值以及int32输出。
        # 生成数据
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.float64(0)
        M = np.float64(2)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        # 使用clip函数对数据进行裁剪
        self.clip(a, m, M, act)
        # 断言两个结果数组相等
        assert_array_equal(ac, act)
    def test_clip_with_out_simple_int32(self):
        # 使用 int32 类型的输入进行测试，设置 int32 标量的最小值和最大值，并将输出类型设为 int64
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.int32(-1)
        M = np.int32(1)
        ac = np.zeros(a.shape, dtype=np.int64)
        act = ac.copy()
        self.fastclip(a, m, M, ac)  # 使用自定义函数 fastclip 进行裁剪操作
        self.clip(a, m, M, act)  # 使用 numpy 自带的 clip 函数进行裁剪操作
        assert_array_equal(ac, act)  # 断言两者裁剪结果是否相等

    @xfail  # (reason="FIXME arrays not equal")
    def test_clip_with_out_array_int32(self):
        # 使用 int32 类型的输入进行测试，设置数组形式的最小值和 int32 类型的最大值，并将输出类型设为 int32
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.zeros(a.shape, np.float64)
        M = np.float64(1)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.clip(a, m, M, act)  # 使用 numpy 自带的 clip 函数进行裁剪操作
        assert_array_equal(ac, act)  # 断言裁剪结果是否相等

    @xfail  # (reason="FIXME arrays not equal")
    def test_clip_with_out_array_outint32(self):
        # 使用 double 类型的输入进行测试，设置标量形式的最小值和最大值，并将输出类型设为 int32
        a = self._generate_data(self.nr, self.nc)
        m = -1.0
        M = 2.0
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.clip(a, m, M, act)  # 使用 numpy 自带的 clip 函数进行裁剪操作
        assert_array_equal(ac, act)  # 断言裁剪结果是否相等

    def test_clip_with_out_transposed(self):
        # 测试当转置时 out 参数的工作情况
        a = np.arange(16).reshape(4, 4)
        out = np.empty_like(a).T
        a.clip(4, 10, out=out)  # 使用 numpy 数组的 clip 方法进行裁剪，并指定输出到 out 参数
        expected = self.clip(a, 4, 10)  # 使用自定义的 clip 函数进行相同的裁剪操作
        assert_array_equal(out, expected)  # 断言裁剪结果是否相等

    def test_clip_with_out_memory_overlap(self):
        # 测试当 out 参数与原始数组有内存重叠时的工作情况
        a = np.arange(16).reshape(4, 4)
        ac = a.copy()
        a[:-1].clip(4, 10, out=a[1:])  # 使用 numpy 数组的 clip 方法进行裁剪，同时指定输出到有内存重叠的部分
        expected = self.clip(ac[:-1], 4, 10)  # 使用自定义的 clip 函数进行相同的裁剪操作
        assert_array_equal(a[1:], expected)  # 断言裁剪结果是否相等

    def test_clip_inplace_array(self):
        # 使用 double 类型的输入进行测试，设置数组形式的最小值和最大值，并使用原地操作
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = np.zeros(a.shape)
        M = 1.0
        self.fastclip(a, m, M, a)  # 使用自定义函数 fastclip 进行原地裁剪操作
        self.clip(a, m, M, ac)  # 使用 numpy 自带的 clip 函数进行裁剪操作
        assert_array_equal(a, ac)  # 断言裁剪结果是否相等

    def test_clip_inplace_simple(self):
        # 使用 double 类型的输入进行测试，设置标量形式的最小值和最大值，并使用原地操作
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = -0.5
        M = 0.6
        self.fastclip(a, m, M, a)  # 使用自定义函数 fastclip 进行原地裁剪操作
        self.clip(a, m, M, ac)  # 使用 numpy 自带的 clip 函数进行裁剪操作
        assert_array_equal(a, ac)  # 断言裁剪结果是否相等

    def test_clip_func_takes_out(self):
        # 确保 clip() 函数能够接受 out 参数
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = -0.5
        M = 0.6
        a2 = np.clip(a, m, M, out=a)  # 使用 numpy 自带的 clip 函数进行裁剪操作，并将输出直接赋给原始数组 a
        self.clip(a, m, M, ac)  # 使用自定义的 clip 函数进行相同的裁剪操作
        assert_array_equal(a2, ac)  # 断言裁剪结果是否相等
        assert_(a2 is a)  # 断言返回的数组与原始数组是同一个对象

    @skip(reason="Edge case; Wait until deprecation graduates")
    # 定义一个测试方法，用于测试处理包含 NaN 的情况下的数组剪切操作
    def test_clip_nan(self):
        # 创建一个包含 0 到 6 的浮点数数组
        d = np.arange(7.0)
        # 断言剪切操作会产生 DeprecationWarning 警告
        with assert_warns(DeprecationWarning):
            # 使用 clip 方法将数组中大于或等于 NaN 的值剪切为 NaN，然后断言结果与原始数组相等
            assert_equal(d.clip(min=np.nan), d)
        with assert_warns(DeprecationWarning):
            # 使用 clip 方法将数组中小于或等于 NaN 的值剪切为 NaN，然后断言结果与原始数组相等
            assert_equal(d.clip(max=np.nan), d)
        with assert_warns(DeprecationWarning):
            # 使用 clip 方法将数组中小于或等于 NaN 且大于或等于 NaN 的值剪切为 NaN，然后断言结果与原始数组相等
            assert_equal(d.clip(min=np.nan, max=np.nan), d)
        with assert_warns(DeprecationWarning):
            # 使用 clip 方法将数组中小于或等于 NaN 且大于或等于 -2 的值剪切为 NaN，然后断言结果与原始数组相等
            assert_equal(d.clip(min=-2, max=np.nan), d)
        with assert_warns(DeprecationWarning):
            # 使用 clip 方法将数组中大于或等于 10 且小于或等于 NaN 的值剪切为 NaN，然后断言结果与原始数组相等
            assert_equal(d.clip(min=np.nan, max=10), d)

    @parametrize(
        "amin, amax",
        [
            # 两个标量
            (1, 0),
            # 标量和数组混合
            (1, np.zeros(10)),
            # 两个数组
            (np.ones(10), np.zeros(10)),
        ],
    )
    # 定义一个参数化测试方法，用于测试不同参数条件下的数组剪切操作，标记有参数
    def test_clip_value_min_max_flip(self, amin, amax):
        # 创建一个包含 0 到 9 的整数数组
        a = np.arange(10, dtype=np.int64)
        # 根据 ufunc_docstrings.py 的要求，计算期望的剪切结果
        expected = np.minimum(np.maximum(a, amin), amax)
        # 执行 np.clip 方法进行剪切操作，得到实际结果
        actual = np.clip(a, amin, amax)
        # 断言实际结果与期望结果相等
        assert_equal(actual, expected)

    @parametrize(
        "arr, amin, amax",
        [
            # 来自 hypothesis 的具有问题的标量 NaN 情况案例
            (
                np.zeros(10, dtype=np.int64),
                np.array(np.nan),
                np.zeros(10, dtype=np.int32),
            ),
        ],
    )
    # 定义一个参数化测试方法，用于测试标量 NaN 在剪切操作中的传播行为，标记有参数
    def test_clip_scalar_nan_propagation(self, arr, amin, amax):
        # 强制标量 NaN 在比较中通过 clip() 方法进行传播
        # 计算期望的剪切结果
        expected = np.minimum(np.maximum(arr, amin), amax)
        # 执行 np.clip 方法进行剪切操作，得到实际结果
        actual = np.clip(arr, amin, amax)
        # 断言实际结果与期望结果相等
        assert_equal(actual, expected)

    @skip  # 在 CI 上由于版本问题，跳过使用 hypothesis 生成的测试用例
    @given(
        data=st.data(),
        arr=hynp.arrays(
            dtype=hynp.integer_dtypes() | hynp.floating_dtypes(),
            shape=hynp.array_shapes(),
        ),
    )
    def test_clip_property(self, data, arr):
        """A property-based test using Hypothesis.

        This aims for maximum generality: it could in principle generate *any*
        valid inputs to np.clip, and in practice generates much more varied
        inputs than human testers come up with.

        Because many of the inputs have tricky dependencies - compatible dtypes
        and mutually-broadcastable shapes - we use `st.data()` strategy draw
        values *inside* the test function, from strategies we construct based
        on previous values.  An alternative would be to define a custom strategy
        with `@st.composite`, but until we have duplicated code inline is fine.

        That accounts for most of the function; the actual test is just three
        lines to calculate and compare actual vs expected results!
        """
        # 定义可以用于数值类型的策略集合，包括整数和浮点数类型
        numeric_dtypes = hynp.integer_dtypes() | hynp.floating_dtypes()
        
        # 生成可以与基础数组形状广播兼容的边界形状
        # 和与基础形状广播兼容的结果形状。在下文中，可能会决定使用标量边界，
        # 但最好是无条件地提前生成这些形状。
        in_shapes, result_shape = data.draw(
            hynp.mutually_broadcastable_shapes(num_shapes=2, base_shape=arr.shape)
        )
        
        # 标量 `nan` 已经因其不同的行为而被弃用。
        s = numeric_dtypes.flatmap(lambda x: hynp.from_dtype(x, allow_nan=False))
        
        # 从数据生成器中绘制最小值 `amin`，可以是标量或数组形式
        amin = data.draw(
            s
            | hynp.arrays(
                dtype=numeric_dtypes, shape=in_shapes[0], elements={"allow_nan": False}
            )
        )
        
        # 从数据生成器中绘制最大值 `amax`，可以是标量或数组形式
        amax = data.draw(
            s
            | hynp.arrays(
                dtype=numeric_dtypes, shape=in_shapes[1], elements={"allow_nan": False}
            )
        )

        # 计算并比较实际结果 `result` 和预期结果 `expected`，确保它们相等。
        # 参见 gh-12519 和 gh-19457 中对该属性及 result_type 参数的讨论。
        result = np.clip(arr, amin, amax)
        t = np.result_type(arr, amin, amax)
        expected = np.minimum(amax, np.maximum(arr, amin, dtype=t), dtype=t)
        
        # 断言结果的数据类型应与 t 相等
        assert result.dtype == t
        
        # 断言数组的内容应与预期的结果相等
        assert_array_equal(result, expected)
# 定义一个测试类 TestAllclose，继承自 TestCase
class TestAllclose(TestCase):
    # 设置默认的相对误差 rtol 和绝对误差 atol
    rtol = 1e-5
    atol = 1e-8

    # 定义一个测试方法 tst_allclose，用于检查 x 和 y 是否全部接近
    def tst_allclose(self, x, y):
        # 断言 x 和 y 全部接近，否则输出错误信息
        assert_(np.allclose(x, y), f"{x} and {y} not close")

    # 定义一个测试方法 tst_not_allclose，用于检查 x 和 y 是否不接近
    def tst_not_allclose(self, x, y):
        # 断言 x 和 y 不接近，否则输出错误信息
        assert_(not np.allclose(x, y), f"{x} and {y} shouldn't be close")

    # 定义一个测试方法 test_ip_allclose，用于测试 np.allclose 的不同情况
    def test_ip_allclose(self):
        # Parametric test factory.
        # 创建测试数据集合 data
        arr = np.array([100, 1000])
        aran = np.arange(125).reshape((5, 5, 5))

        # 使用类属性的绝对误差和相对误差
        atol = self.atol
        rtol = self.rtol

        data = [
            ([1, 0], [1, 0]),
            ([atol], [0]),
            ([1], [1 + rtol + atol]),
            (arr, arr + arr * rtol),
            (arr, arr + arr * rtol + atol * 2),
            (aran, aran + aran * rtol),
            (np.inf, np.inf),
            (np.inf, [np.inf]),
        ]

        # 对于每对 x, y 在 data 中，调用 self.tst_allclose 方法进行测试
        for x, y in data:
            self.tst_allclose(x, y)

    # 定义一个测试方法 test_ip_not_allclose，用于测试 np.allclose 的不接近情况
    def test_ip_not_allclose(self):
        # Parametric test factory.
        aran = np.arange(125).reshape((5, 5, 5))

        # 使用类属性的绝对误差和相对误差
        atol = self.atol
        rtol = self.rtol

        data = [
            ([np.inf, 0], [1, np.inf]),
            ([np.inf, 0], [1, 0]),
            ([np.inf, np.inf], [1, np.inf]),
            ([np.inf, np.inf], [1, 0]),
            ([-np.inf, 0], [np.inf, 0]),
            ([np.nan, 0], [np.nan, 0]),
            ([atol * 2], [0]),
            ([1], [1 + rtol + atol * 2]),
            (aran, aran + aran * atol + atol * 2),
            (np.array([np.inf, 1]), np.array([0, np.inf])),
        ]

        # 对于每对 x, y 在 data 中，调用 self.tst_not_allclose 方法进行测试
        for x, y in data:
            self.tst_not_allclose(x, y)

    # 定义一个测试方法 test_no_parameter_modification，测试 np.allclose 的参数不被修改情况
    def test_no_parameter_modification(self):
        # 设置初始测试数据 x 和 y
        x = np.array([np.inf, 1])
        y = np.array([0, np.inf])

        # 调用 np.allclose 检查 x 和 y 是否接近
        np.allclose(x, y)

        # 断言调用 np.allclose 后 x 的值未被修改
        assert_array_equal(x, np.array([np.inf, 1]))

        # 断言调用 np.allclose 后 y 的值未被修改
        assert_array_equal(y, np.array([0, np.inf]))

    # 定义一个测试方法 test_min_int，测试 np.allclose 对于最小整数的情况
    def test_min_int(self):
        # 可能因为 abs(min_int) == min_int 而导致问题
        min_int = np.iinfo(np.int_).min
        a = np.array([min_int], dtype=np.int_)

        # 断言 np.allclose 检查 a 和 a 是否接近
        assert_(np.allclose(a, a))

    # 定义一个测试方法 test_equalnan，测试 np.allclose 对于包含 NaN 的情况
    def test_equalnan(self):
        # 设置初始测试数据 x
        x = np.array([1.0, np.nan])

        # 断言 np.allclose 检查 x 和 x 是否接近，包含 NaN 的情况下相等
        assert_(np.allclose(x, x, equal_nan=True))


# 定义一个测试类 TestIsclose，继承自 TestCase
class TestIsclose(TestCase):
    # 设置默认的相对误差 rtol 和绝对误差 atol
    rtol = 1e-5
    atol = 1e-8
    # 初始化函数，设置公共的测试参数
    def _setup(self):
        # 设置绝对误差
        atol = self.atol
        # 设置相对误差
        rtol = self.rtol
        # 创建包含两个元素的 NumPy 数组
        arr = np.array([100, 1000])
        # 创建形状为 (5, 5, 5) 的 NumPy 数组
        aran = np.arange(125).reshape((5, 5, 5))

        # 设置所有相等测试用例
        self.all_close_tests = [
            ([1, 0], [1, 0]),  # 正确相等
            ([atol], [0]),  # 绝对误差
            ([1], [1 + rtol + atol]),  # 相对和绝对误差
            (arr, arr + arr * rtol),  # 数组相等
            (arr, arr + arr * rtol + atol),  # 数组相等
            (aran, aran + aran * rtol),  # 数组相等
            (np.inf, np.inf),  # 正无穷相等
            (np.inf, [np.inf]),  # 正无穷相等
            ([np.inf, -np.inf], [np.inf, -np.inf]),  # 无穷相等
        ]
        
        # 设置全不相等测试用例
        self.none_close_tests = [
            ([np.inf, 0], [1, np.inf]),  # 全不相等
            ([np.inf, -np.inf], [1, 0]),  # 全不相等
            ([np.inf, np.inf], [1, -np.inf]),  # 全不相等
            ([np.inf, np.inf], [1, 0]),  # 全不相等
            ([np.nan, 0], [np.nan, -np.inf]),  # 全不相等
            ([atol * 2], [0]),  # 绝对误差
            ([1], [1 + rtol + atol * 2]),  # 相对和绝对误差
            (aran, aran + rtol * 1.1 * aran + atol * 1.1),  # 数组相等
            (np.array([np.inf, 1]), np.array([0, np.inf])),  # 数组相等
        ]
        
        # 设置部分相等测试用例
        self.some_close_tests = [
            ([np.inf, 0], [np.inf, atol * 2]),  # 部分相等
            ([atol, 1, 1e6 * (1 + 2 * rtol) + atol], [0, np.nan, 1e6]),  # 部分相等
            (np.arange(3), [0, 1, 2.1]),  # 数组部分相等
            (np.nan, [np.nan, np.nan, np.nan]),  # 数组部分相等
            ([0], [atol, np.inf, -np.inf, np.nan]),  # 部分相等
            (0, [atol, np.inf, -np.inf, np.nan]),  # 部分相等
        ]
        
        # 设置部分相等的预期结果
        self.some_close_results = [
            [True, False],  # 部分相等结果
            [True, False, False],  # 部分相等结果
            [True, True, False],  # 部分相等结果
            [False, False, False],  # 部分相等结果
            [True, False, False, False],  # 部分相等结果
            [True, False, False, False],  # 部分相等结果
        ]

    # 测试函数，验证 some_close_tests 中的每对 x, y 是否相等
    def test_ip_isclose(self):
        self._setup()
        tests = self.some_close_tests
        results = self.some_close_results
        for (x, y), result in zip(tests, results):
            assert_array_equal(np.isclose(x, y), result)

    # 测试函数，验证 all_close_tests 中的每对 x, y 是否完全相等
    def test_ip_all_isclose(self):
        self._setup()
        for x, y in self.all_close_tests:
            self.tst_all_isclose(x, y)

    # 测试函数，验证 none_close_tests 中的每对 x, y 是否完全不相等
    def test_ip_none_isclose(self):
        self._setup()
        for x, y in self.none_close_tests:
            self.tst_none_isclose(x, y)

    # 测试函数，验证 all_close_tests、none_close_tests 和 some_close_tests 中的每对 x, y 是否满足 np.isclose 和 np.allclose 的结果一致性
    def test_ip_isclose_allclose(self):
        self._setup()
        tests = self.all_close_tests + self.none_close_tests + self.some_close_tests
        for x, y in tests:
            self.tst_isclose_allclose(x, y)

    # 辅助函数，验证 x 和 y 是否完全相等
    def tst_all_isclose(self, x, y):
        assert_(np.all(np.isclose(x, y)), f"{x} and {y} not close")

    # 辅助函数，验证 x 和 y 是否完全不相等
    def tst_none_isclose(self, x, y):
        msg = "%s and %s shouldn't be close"
        assert_(not np.any(np.isclose(x, y)), msg % (x, y))

    # 辅助函数，验证 x 和 y 的 np.isclose 和 np.allclose 结果一致性
    def tst_isclose_allclose(self, x, y):
        msg = "isclose.all() and allclose aren't same for %s and %s"
        msg2 = "isclose and allclose aren't same for %s and %s"
        if np.isscalar(x) and np.isscalar(y):
            assert_(np.isclose(x, y) == np.allclose(x, y), msg=msg2 % (x, y))
        else:
            assert_array_equal(np.isclose(x, y).all(), np.allclose(x, y), msg % (x, y))
    # 测试 np.isclose 函数对两个 NaN 值的比较结果是否为 True
    def test_equal_nan(self):
        assert_array_equal(np.isclose(np.nan, np.nan, equal_nan=True), [True])
        # 创建包含 NaN 的数组，测试 np.isclose 对数组中的 NaN 的比较结果是否为 True
        arr = np.array([1.0, np.nan])
        assert_array_equal(np.isclose(arr, arr, equal_nan=True), [True, True])

    # 在 Torch Dynamo 下会失败的测试，测试 np.isclose 函数返回标量值时的情况
    @xfailIfTorchDynamo  # scalars vs 0D
    def test_scalar_return(self):
        # 断言 np.isclose(1, 1) 返回的结果是否是标量
        assert_(np.isscalar(np.isclose(1, 1)))

    # 测试 np.isclose 函数在不修改参数的情况下的行为
    def test_no_parameter_modification(self):
        x = np.array([np.inf, 1])
        y = np.array([0, np.inf])
        np.isclose(x, y)
        # 断言 x 数组的值是否未被修改
        assert_array_equal(x, np.array([np.inf, 1]))
        # 断言 y 数组的值是否未被修改
        assert_array_equal(y, np.array([0, np.inf]))

    # 测试 np.isclose 函数对非有限数值标量的比较结果
    def test_non_finite_scalar(self):
        # GH7014，当比较两个标量时，输出应该也是标量
        # 注意：由于存在数组标量的情况，测试已经修改
        # 断言 np.isclose(np.inf, -np.inf) 的结果是否为 False，并且转换为标量进行比较
        assert_(np.isclose(np.inf, -np.inf).item() is False)
        # 断言 np.isclose(0, np.inf) 的结果是否为 False，并且转换为标量进行比较
        assert_(np.isclose(0, np.inf).item() is False)
# 定义一个测试类 TestStdVar，用于测试标准方差和方差相关函数
class TestStdVar(TestCase):

    # 在每个测试方法执行前调用的方法，设置测试环境
    def setUp(self):
        super().setUp()
        # 创建一个 numpy 数组 A，包含整数和负数
        self.A = np.array([1, -1, 1, -1])
        # 设置真实的方差值为 1
        self.real_var = 1

    # 测试标准方差和方差的基本功能
    def test_basic(self):
        # 断言 np.var 计算的方差接近真实值
        assert_almost_equal(np.var(self.A), self.real_var)
        # 断言 np.std 计算的标准差的平方接近真实值
        assert_almost_equal(np.std(self.A) ** 2, self.real_var)

    # 测试对标量输入的方差和标准方差计算
    def test_scalars(self):
        # 断言对标量 1 的方差为 0
        assert_equal(np.var(1), 0)
        # 断言对标量 1 的标准方差为 0
        assert_equal(np.std(1), 0)

    # 测试使用 ddof=1 参数计算方差和标准方差
    def test_ddof1(self):
        # 断言使用 ddof=1 计算的方差接近真实值
        assert_almost_equal(
            np.var(self.A, ddof=1), self.real_var * len(self.A) / (len(self.A) - 1)
        )
        # 断言使用 ddof=1 计算的标准方差的平方接近真实值
        assert_almost_equal(
            np.std(self.A, ddof=1) ** 2, self.real_var * len(self.A) / (len(self.A) - 1)
        )

    # 测试使用 ddof=2 参数计算方差和标准方差
    def test_ddof2(self):
        # 断言使用 ddof=2 计算的方差接近真实值
        assert_almost_equal(
            np.var(self.A, ddof=2), self.real_var * len(self.A) / (len(self.A) - 2)
        )
        # 断言使用 ddof=2 计算的标准方差的平方接近真实值
        assert_almost_equal(
            np.std(self.A, ddof=2) ** 2, self.real_var * len(self.A) / (len(self.A) - 2)
        )

    # 测试将输出存储到标量的情况
    def test_out_scalar(self):
        # 创建一个长度为 10 的数组 d
        d = np.arange(10)
        # 创建一个初始值为 0.0 的数组 out
        out = np.array(0.0)
        # 使用 np.std 将数组 d 的标准方差存储到 out 中
        r = np.std(d, out=out)
        # 断言 r 是 out 的引用
        assert_(r is out)
        # 断言 r 与 out 相等
        assert_array_equal(r, out)
        # 使用 np.var 将数组 d 的方差存储到 out 中
        r = np.var(d, out=out)
        # 断言 r 是 out 的引用
        assert_(r is out)
        # 断言 r 与 out 相等
        assert_array_equal(r, out)
        # 使用 np.mean 将数组 d 的均值存储到 out 中
        r = np.mean(d, out=out)
        # 断言 r 是 out 的引用
        assert_(r is out)
        # 断言 r 与 out 相等
        assert_array_equal(r, out)


# 定义一个测试类 TestStdVarComplex，测试复数数组的标准方差和方差相关函数
class TestStdVarComplex(TestCase):

    # 测试复数数组的基本功能
    def test_basic(self):
        # 创建一个包含实数和虚数的 numpy 数组 A
        A = np.array([1, 1.0j, -1, -1.0j])
        # 设置真实的方差值为 1
        real_var = 1
        # 断言 np.var 计算的方差接近真实值
        assert_almost_equal(np.var(A), real_var)
        # 断言 np.std 计算的标准差的平方接近真实值
        assert_almost_equal(np.std(A) ** 2, real_var)

    # 测试对复数标量输入的方差和标准方差计算
    def test_scalars(self):
        # 断言对复数标量 1j 的方差为 0
        assert_equal(np.var(1j), 0)
        # 断言对复数标量 1j 的标准方差为 0
        assert_equal(np.std(1j), 0)


# 定义一个测试类 TestCreationFuncs，测试 numpy 的创建函数（ones, zeros, empty）
class TestCreationFuncs(TestCase):

    # 在每个测试方法执行前调用的方法，设置测试环境
    def setUp(self):
        super().setUp()
        # 创建一个包含所有 numpy 类型的集合 dtypes
        dtypes = {np.dtype(tp) for tp in "efdFDBbhil?"}
        self.dtypes = dtypes
        # 设置数组的存储顺序字典 orders，目前只设置了 'C'（行优先）
        self.orders = {
            "C": "c_contiguous"
        }  # XXX: reeenable when implemented, 'F': 'f_contiguous'}
        # 设置数组的维度 ndims
        self.ndims = 10

    # 检查 numpy 创建函数的通用测试方法
    def check_function(self, func, fill_value=None):
        # 参数 par 包含了测试的范围
        par = ((0, 1, 2), range(self.ndims), self.orders, self.dtypes)
        fill_kwarg = {}
        if fill_value is not None:
            fill_kwarg = {"fill_value": fill_value}

        # 使用 itertools.product 对所有参数进行组合
        for size, ndims, order, dtype in itertools.product(*par):
            shape = ndims * [size]

            # 调用函数 func 创建数组 arr
            arr = func(shape, order=order, dtype=dtype, **fill_kwarg)

            # 断言 arr 的数据类型与预期的 dtype 一致
            assert_equal(arr.dtype, dtype)
            # 断言 arr 的存储顺序符合 self.orders 中指定的顺序
            assert_(getattr(arr.flags, self.orders[order]))

            # 如果指定了 fill_value，则断言 arr 的所有元素都等于 fill_value
            if fill_value is not None:
                val = fill_value
                assert_equal(arr, dtype.type(val))

    # 测试 np.zeros 函数
    def test_zeros(self):
        self.check_function(np.zeros)

    # 测试 np.ones 函数
    def test_ones(self):
        self.check_function(np.ones)

    # 测试 np.empty 函数
    def test_empty(self):
        self.check_function(np.empty)
    # 定义一个名为 test_full 的测试方法，用于测试 np.full 函数的不同用例
    def test_full(self):
        # 调用 self.check_function 方法，检查 np.full 函数的行为是否符合预期（参数为0）
        self.check_function(np.full, 0)
        # 再次调用 self.check_function 方法，检查 np.full 函数的行为是否符合预期（参数为1）
        self.check_function(np.full, 1)

    # 根据条件装饰 test_for_reference_leak 方法，如果 TEST_WITH_TORCHDYNAMO 为真，则跳过该测试（原因是 dynamo 存在问题）
    # 如果 HAS_REFCOUNT 为假，则同样跳过该测试（原因是 Python 缺乏引用计数）
    @skipif(TEST_WITH_TORCHDYNAMO, reason="fails with dynamo")
    @skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_for_reference_leak(self):
        # 确保我们有一个用于引用的对象
        dim = 1
        # 获取初始时 dim 的引用计数
        beg = sys.getrefcount(dim)
        # 创建一个形状为 [dim]*10 的零矩阵，检查引用计数是否有变化
        np.zeros([dim] * 10)
        assert_(sys.getrefcount(dim) == beg)
        # 创建一个形状为 [dim]*10 的全一矩阵，检查引用计数是否有变化
        np.ones([dim] * 10)
        assert_(sys.getrefcount(dim) == beg)
        # 创建一个形状为 [dim]*10 的空矩阵，检查引用计数是否有变化
        np.empty([dim] * 10)
        assert_(sys.getrefcount(dim) == beg)
        # 创建一个形状为 [dim]*10，填充值为0的矩阵，检查引用计数是否有变化
        np.full([dim] * 10, 0)
        assert_(sys.getrefcount(dim) == beg)
@skip(reason="implement order etc")  # FIXME: make xfail
@instantiate_parametrized_tests
class TestLikeFuncs(TestCase):
    """Test ones_like, zeros_like, empty_like and full_like"""

    def setUp(self):
        super().setUp()
        self.data = [
            # Array scalars
            (np.array(3.0), None),
            (np.array(3), "f8"),
            # 1D arrays
            (np.arange(6, dtype="f4"), None),
            (np.arange(6), "c16"),
            # 2D C-layout arrays
            (np.arange(6).reshape(2, 3), None),
            (np.arange(6).reshape(3, 2), "i1"),
            # 2D F-layout arrays
            (np.arange(6).reshape((2, 3), order="F"), None),
            (np.arange(6).reshape((3, 2), order="F"), "i1"),
            # 3D C-layout arrays
            (np.arange(24).reshape(2, 3, 4), None),
            (np.arange(24).reshape(4, 3, 2), "f4"),
            # 3D F-layout arrays
            (np.arange(24).reshape((2, 3, 4), order="F"), None),
            (np.arange(24).reshape((4, 3, 2), order="F"), "f4"),
            # 3D non-C/F-layout arrays
            (np.arange(24).reshape(2, 3, 4).swapaxes(0, 1), None),
            (np.arange(24).reshape(4, 3, 2).swapaxes(0, 1), "?"),
        ]
        self.shapes = [
            (),
            (5,),
            (
                5,
                6,
            ),
            (
                5,
                6,
                7,
            ),
        ]

    def compare_array_value(self, dz, value, fill_value):
        if value is not None:
            if fill_value:
                # Conversion is close to what np.full_like uses
                # but we may want to convert directly in the future
                # which may result in errors (where this does not).
                z = np.array(value).astype(dz.dtype)
                assert_(np.all(dz == z))
            else:
                assert_(np.all(dz == value))

    def test_ones_like(self):
        # 测试 np.ones_like 函数，期望结果为全 1 的数组
        self.check_like_function(np.ones_like, 1)

    def test_zeros_like(self):
        # 测试 np.zeros_like 函数，期望结果为全 0 的数组
        self.check_like_function(np.zeros_like, 0)

    def test_empty_like(self):
        # 测试 np.empty_like 函数，期望结果为空数组（与输入数组形状相同）
        self.check_like_function(np.empty_like, None)

    def test_filled_like(self):
        # 测试 np.full_like 函数，期望结果为填充值的数组
        self.check_like_function(np.full_like, 0, True)
        self.check_like_function(np.full_like, 1, True)
        self.check_like_function(np.full_like, 1000, True)
        self.check_like_function(np.full_like, 123.456, True)
        # Inf 转换为整数可能导致无效值错误：忽略这些情况。
        self.check_like_function(np.full_like, np.inf, True)

    @parametrize("likefunc", [np.empty_like, np.full_like, np.zeros_like, np.ones_like])
    @parametrize("dtype", [str, bytes])
    # 定义一个测试函数，用于测试特定的数据类型操作函数，验证修复了 gh-19860 的问题
    def test_dtype_str_bytes(self, likefunc, dtype):
        # 创建一个二维数组 a，包含数字 0 到 15，形状为 (2, 8)
        a = np.arange(16).reshape(2, 8)
        # 从 a 中选择列，步长为 2，以确保 b 不是连续的
        b = a[:, ::2]
        # 根据 likefunc 的值选择适当的参数字典
        kwargs = {"fill_value": ""} if likefunc == np.full_like else {}
        # 使用 likefunc 函数处理 b，返回处理后的结果
        result = likefunc(b, dtype=dtype, **kwargs)
        # 如果 dtype 是 str 类型
        if dtype == str:
            # 断言结果的步幅为 (16, 4)
            assert result.strides == (16, 4)
        else:
            # 否则，dtype 是 bytes 类型
            # 断言结果的步幅为 (4, 1)
            assert result.strides == (4, 1)
# 定义一个测试类 TestCorrelate，继承自 TestCase，用于测试 np.correlate 函数的各种情况
class TestCorrelate(TestCase):

    # 定义一个私有方法 _setup，用于设置测试所需的各种数据
    def _setup(self, dt):
        # 创建一个 numpy 数组 self.x，包含整数 1 到 5，数据类型为 dt
        self.x = np.array([1, 2, 3, 4, 5], dtype=dt)
        # 创建一个 numpy 数组 self.xs，包含从 1 到 19 的数，步长为 3
        self.xs = np.arange(1, 20)[::3]
        # 创建一个 numpy 数组 self.y，包含整数 -1 到 -3，数据类型为 dt
        self.y = np.array([-1, -2, -3], dtype=dt)
        # 创建一个 numpy 数组 self.z1，包含一系列浮点数，数据类型为 dt
        self.z1 = np.array([-3.0, -8.0, -14.0, -20.0, -26.0, -14.0, -5.0], dtype=dt)
        # 创建一个 numpy 数组 self.z1_4，包含一系列浮点数，数据类型为 dt
        self.z1_4 = np.array([-2.0, -5.0, -8.0, -11.0, -14.0, -5.0], dtype=dt)
        # 创建一个 numpy 数组 self.z1r，包含一系列浮点数，数据类型为 dt
        self.z1r = np.array([-15.0, -22.0, -22.0, -16.0, -10.0, -4.0, -1.0], dtype=dt)
        # 创建一个 numpy 数组 self.z2，包含一系列浮点数，数据类型为 dt
        self.z2 = np.array([-5.0, -14.0, -26.0, -20.0, -14.0, -8.0, -3.0], dtype=dt)
        # 创建一个 numpy 数组 self.z2r，包含一系列浮点数，数据类型为 dt
        self.z2r = np.array([-1.0, -4.0, -10.0, -16.0, -22.0, -22.0, -15.0], dtype=dt)
        # 创建一个 numpy 数组 self.zs，包含一系列浮点数，数据类型为 dt
        self.zs = np.array(
            [-3.0, -14.0, -30.0, -48.0, -66.0, -84.0, -102.0, -54.0, -19.0], dtype=dt
        )

    # 定义测试方法 test_float，测试 np.correlate 函数在浮点数情况下的输出
    def test_float(self):
        # 使用 _setup 方法设置数据类型为 float 的测试数据
        self._setup(float)
        # 调用 np.correlate 计算 self.x 和 self.y 的相关性，模式为 "full"
        z = np.correlate(self.x, self.y, "full")
        # 断言 z 数组近似等于预期的 self.z1 数组
        assert_array_almost_equal(z, self.z1)
        # 再次调用 np.correlate 计算 self.x 和 self.y[:-1] 的相关性，模式为 "full"
        z = np.correlate(self.x, self.y[:-1], "full")
        # 断言 z 数组近似等于预期的 self.z1_4 数组
        assert_array_almost_equal(z, self.z1_4)
        # 再次调用 np.correlate 计算 self.y 和 self.x 的相关性，模式为 "full"
        z = np.correlate(self.y, self.x, "full")
        # 断言 z 数组近似等于预期的 self.z2 数组
        assert_array_almost_equal(z, self.z2)
        # 调用 np.correlate 计算 np.flip(self.x) 和 self.y 的相关性，模式为 "full"
        z = np.correlate(np.flip(self.x), self.y, "full")
        # 断言 z 数组近似等于预期的 self.z1r 数组
        assert_array_almost_equal(z, self.z1r)
        # 调用 np.correlate 计算 self.y 和 np.flip(self.x) 的相关性，模式为 "full"
        z = np.correlate(self.y, np.flip(self.x), "full")
        # 断言 z 数组近似等于预期的 self.z2r 数组
        assert_array_almost_equal(z, self.z2r)
        # 调用 np.correlate 计算 self.xs 和 self.y 的相关性，模式为 "full"
        z = np.correlate(self.xs, self.y, "full")
        # 断言 z 数组近似等于预期的 self.zs 数组
        assert_array_almost_equal(z, self.zs)

    # 定义测试方法 test_no_overwrite，测试 np.correlate 函数不会修改其输入数组
    def test_no_overwrite(self):
        # 创建一个长度为 100 的全 1 数组 d
        d = np.ones(100)
        # 创建一个长度为 3 的全 1 数组 k
        k = np.ones(3)
        # 调用 np.correlate 计算 d 和 k 的相关性
        np.correlate(d, k)
        # 断言数组 d 仍然等于全 1 数组
        assert_array_equal(d, np.ones(100))
        # 断言数组 k 仍然等于全 1 数组
        assert_array_equal(k, np.ones(3))

    # 定义测试方法 test_complex，测试 np.correlate 函数在复数情况下的输出
    def test_complex(self):
        # 创建一个复数类型的 numpy 数组 x
        x = np.array([1, 2, 3, 4 + 1j], dtype=complex)
        # 创建一个复数类型的 numpy 数组 y
        y = np.array([-1, -2j, 3 + 1j], dtype=complex)
        # 创建一个复数类型的 numpy 数组 r_z，并翻转和共轭处理
        r_z = np.array([3 - 1j, 6, 8 + 1j, 11 + 5j, -5 + 8j, -4 - 1j], dtype=complex)
        r_z = np.flip(r_z).conjugate()
        # 调用 np.correlate 计算 y 和 x 的相关性，模式为 "full"
        z = np.correlate(y, x, mode="full")
        # 断言 z 数组近似等于预期的 r_z 数组
        assert_array_almost_equal(z, r_z)

    # 定义测试方法 test_zero_size，测试 np.correlate 函数在输入数组长度为零时的行为
    def test_zero_size(self):
        # 使用 pytest.raises 检查调用 np.correlate 函数时抛出 ValueError 或 RuntimeError 异常
        with pytest.raises((ValueError, RuntimeError)):
            np.correlate(np.array([]), np.ones(1000), mode="full")
        with pytest.raises((ValueError, RuntimeError)):
            np.correlate(np.ones(1000), np.array([]), mode="full")

    # 使用 @skip 装饰器标记测试方法 test_mode，表示该测试方法不会被执行
    @skip(reason="do not implement deprecated behavior")
    # 定义测试方法 test_mode，测试 np.correlate 函数的不同模式参数
    def test_mode(self):
        # 创建一个长度为 100 的全 1 数组 d
        d = np.ones(100)
        # 创建一个长度为 3 的全 1 数组 k
        k = np.ones(3)
        # 调用 np.correlate 计算 d 和 k 的相关性，模式为 "valid"
        default_mode = np.correlate(d, k, mode="valid")
        # 使用 assert_warns 检查调用 np.correlate 函数时会发出 DeprecationWarning 警告
        with assert_warns(DeprecationWarning):
            valid_mode = np.correlate(d, k, mode="v")
        # 断言 valid_mode 数组等于 default_mode 数组
        assert_array_equal(valid_mode, default_mode)
        # 使用 assert_raises 检查调用 np.correlate 函数时会抛出 ValueError 异常
        with assert_raises(ValueError
    # 定义一个测试方法，用于测试 np.convolve 函数的功能
    def test_object(self):
        # 创建长度为 100 的浮点数列表 d
        d = [1.0] * 100
        # 创建长度为 3 的浮点数列表 k
        k = [1.0] * 3
        # 断言 np.convolve(d, k) 的结果与 np.full(98, 3)[2:-2] 几乎相等
        assert_array_almost_equal(np.convolve(d, k)[2:-2], np.full(98, 3))

    # 定义一个测试方法，用于测试 np.convolve 函数不改变原始数组的行为
    def test_no_overwrite(self):
        # 创建长度为 100 的全为 1 的数组 d
        d = np.ones(100)
        # 创建长度为 3 的全为 1 的数组 k
        k = np.ones(3)
        # 调用 np.convolve(d, k)，但不保存返回值
        np.convolve(d, k)
        # 断言数组 d 仍然全为 1
        assert_array_equal(d, np.ones(100))
        # 断言数组 k 仍然全为 1
        assert_array_equal(k, np.ones(3))

    # 定义一个被跳过的测试方法，因为其行为已被弃用
    @skip(reason="do not implement deprecated behavior")
    def test_mode(self):
        # 创建长度为 100 的全为 1 的数组 d
        d = np.ones(100)
        # 创建长度为 3 的全为 1 的数组 k
        k = np.ones(3)
        # 使用默认模式进行卷积操作
        default_mode = np.convolve(d, k, mode="full")
        # 使用被警告弃用的模式 "f" 进行卷积操作
        with assert_warns(DeprecationWarning):
            full_mode = np.convolve(d, k, mode="f")
        # 断言使用 "f" 模式的结果与默认模式的结果相等
        assert_array_equal(full_mode, default_mode)
        # 使用整数模式时应引发 ValueError 异常
        with assert_raises(ValueError):
            np.convolve(d, k, mode=-1)
        # 使用模式参数为 2 时，结果应与默认模式相等
        assert_array_equal(np.convolve(d, k, mode=2), full_mode)
        # 使用不支持的参数类型时应引发 TypeError 异常
        with assert_raises(TypeError):
            np.convolve(d, k, mode=None)

    # 定义一个测试方法，用于验证 np.convolve 函数在不同模式下的行为是否符合文档中的示例
    def test_numpy_doc_examples(self):
        # 测试标准卷积（"full" 模式）的结果是否与预期接近
        conv = np.convolve([1, 2, 3], [0, 1, 0.5])
        assert_allclose(conv, [0.0, 1.0, 2.5, 4.0, 1.5], atol=1e-15)

        # 测试 "same" 模式下的卷积结果是否与预期接近
        conv = np.convolve([1, 2, 3], [0, 1, 0.5], "same")
        assert_allclose(conv, [1.0, 2.5, 4.0], atol=1e-15)

        # 测试 "valid" 模式下的卷积结果是否与预期接近
        conv = np.convolve([1, 2, 3], [0, 1, 0.5], "valid")
        assert_allclose(conv, [2.5], atol=1e-15)
class TestDtypePositional(TestCase):
    def test_dtype_positional(self):
        # 创建一个包含两个元素的空数组，元素类型为布尔值
        np.empty((2,), bool)


@instantiate_parametrized_tests
class TestArgwhere(TestCase):
    @parametrize("nd", [0, 1, 2])
    def test_nd(self, nd):
        # 创建一个 nd 维度的空数组，每个维度有两个元素，元素类型为布尔值
        x = np.empty((2,) * nd, dtype=bool)

        # none
        # 将数组所有元素设置为 False
        x[...] = False
        # 验证 np.argwhere 的输出形状是否为 (0, nd)
        assert_equal(np.argwhere(x).shape, (0, nd))

        # only one
        # 将数组所有元素设置为 False
        x[...] = False
        # 将数组展平后的第一个元素设置为 True
        x.ravel()[0] = True
        # 验证 np.argwhere 的输出形状是否为 (1, nd)
        assert_equal(np.argwhere(x).shape, (1, nd))

        # all but one
        # 将数组所有元素设置为 True
        x[...] = True
        # 将数组展平后的第一个元素设置为 False
        x.ravel()[0] = False
        # 验证 np.argwhere 的输出形状是否为 (x.size - 1, nd)
        assert_equal(np.argwhere(x).shape, (x.size - 1, nd))

        # all
        # 将数组所有元素设置为 True
        x[...] = True
        # 验证 np.argwhere 的输出形状是否为 (x.size, nd)
        assert_equal(np.argwhere(x).shape, (x.size, nd))

    def test_2D(self):
        # 创建一个二维数组
        x = np.arange(6).reshape((2, 3))
        # 验证 np.argwhere(x > 1) 的输出是否符合预期
        assert_array_equal(np.argwhere(x > 1), [[0, 2], [1, 0], [1, 1], [1, 2]])

    def test_list(self):
        # 验证 np.argwhere([4, 0, 2, 1, 3]) 的输出是否符合预期
        assert_equal(np.argwhere([4, 0, 2, 1, 3]), [[0], [2], [3], [4]])


@xpassIfTorchDynamo  # (reason="TODO")
class TestStringFunction(TestCase):
    def test_set_string_function(self):
        a = np.array([1])
        # 设置 np.array 的字符串表示为 "FOO"
        np.set_string_function(lambda x: "FOO", repr=True)
        # 验证 repr(a) 的输出是否为 "FOO"
        assert_equal(repr(a), "FOO")
        # 恢复 np.array 的默认字符串表示
        np.set_string_function(None, repr=True)
        # 验证 repr(a) 的输出是否为 "array([1])"

        # 设置 np.array 的字符串表示为 "FOO"
        np.set_string_function(lambda x: "FOO", repr=False)
        # 验证 str(a) 的输出是否为 "FOO"
        assert_equal(str(a), "FOO")
        # 恢复 np.array 的默认字符串表示
        np.set_string_function(None, repr=False)
        # 验证 str(a) 的输出是否为 "[1]"


class TestRoll(TestCase):
    def test_roll1d(self):
        # 创建一个包含 0 到 9 的数组
        x = np.arange(10)
        # 将数组 x 向右循环移动 2 个位置
        xr = np.roll(x, 2)
        # 验证 xr 是否与预期的数组相等
        assert_equal(xr, np.array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7]))
    # 定义测试函数 test_roll2d，用于测试 numpy 中的 np.roll 函数对二维数组的操作
    def test_roll2d(self):
        # 创建一个二维数组 x2，reshape 后为 (2, 5)，包含元素 0 到 9
        x2 = np.reshape(np.arange(10), (2, 5))
        # 对 x2 进行向右滚动一步，生成新数组 x2r
        x2r = np.roll(x2, 1)
        # 断言 x2r 应该等于指定的数组
        assert_equal(x2r, np.array([[9, 0, 1, 2, 3], [4, 5, 6, 7, 8]]))

        # 对 x2 按 axis=0 方向进行向右滚动一步，生成新数组 x2r
        x2r = np.roll(x2, 1, axis=0)
        # 断言 x2r 应该等于指定的数组
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        # 对 x2 按 axis=1 方向进行向右滚动一步，生成新数组 x2r
        x2r = np.roll(x2, 1, axis=1)
        # 断言 x2r 应该等于指定的数组
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        # 同时按 axis=(0, 1) 方向进行向右滚动一步，生成新数组 x2r
        x2r = np.roll(x2, 1, axis=(0, 1))
        # 断言 x2r 应该等于指定的数组
        assert_equal(x2r, np.array([[9, 5, 6, 7, 8], [4, 0, 1, 2, 3]]))

        # 同时按 axis=(0, 1) 方向进行向右滚动一步，生成新数组 x2r
        x2r = np.roll(x2, (1, 0), axis=(0, 1))
        # 断言 x2r 应该等于指定的数组
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        # 同时按 axis=(0, 1) 方向进行向左滚动一步，生成新数组 x2r
        x2r = np.roll(x2, (-1, 0), axis=(0, 1))
        # 断言 x2r 应该等于指定的数组
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        # 同时按 axis=(0, 1) 方向进行向右滚动一步，生成新数组 x2r
        x2r = np.roll(x2, (0, 1), axis=(0, 1))
        # 断言 x2r 应该等于指定的数组
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        # 同时按 axis=(0, 1) 方向进行向左滚动一步，生成新数组 x2r
        x2r = np.roll(x2, (0, -1), axis=(0, 1))
        # 断言 x2r 应该等于指定的数组
        assert_equal(x2r, np.array([[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]]))

        # 同时按 axis=(0, 1) 方向进行向右滚动一步，生成新数组 x2r
        x2r = np.roll(x2, (1, 1), axis=(0, 1))
        # 断言 x2r 应该等于指定的数组
        assert_equal(x2r, np.array([[9, 5, 6, 7, 8], [4, 0, 1, 2, 3]]))

        # 同时按 axis=(0, 1) 方向进行向左滚动一步，生成新数组 x2r
        x2r = np.roll(x2, (-1, -1), axis=(0, 1))
        # 断言 x2r 应该等于指定的数组
        assert_equal(x2r, np.array([[6, 7, 8, 9, 5], [1, 2, 3, 4, 0]]))

        # 在同一轴上多次进行向右滚动一步，生成新数组 x2r
        x2r = np.roll(x2, 1, axis=(0, 0))
        # 断言 x2r 应该等于指定的数组
        assert_equal(x2r, np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]))

        # 在同一轴上多次进行向右滚动一步，生成新数组 x2r
        x2r = np.roll(x2, 1, axis=(1, 1))
        # 断言 x2r 应该等于指定的数组
        assert_equal(x2r, np.array([[3, 4, 0, 1, 2], [8, 9, 5, 6, 7]]))

        # 在同一轴上滚动超过一圈，向右滚动 6 步，生成新数组 x2r
        x2r = np.roll(x2, 6, axis=1)
        # 断言 x2r 应该等于指定的数组
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        # 在同一轴上滚动超过一圈，向左滚动 4 步，生成新数组 x2r
        x2r = np.roll(x2, -4, axis=1)
        # 断言 x2r 应该等于指定的数组
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

    # 定义测试函数 test_roll_empty，用于测试对空数组的 np.roll 操作
    def test_roll_empty(self):
        # 创建一个空的 numpy 数组 x
        x = np.array([])
        # 断言对空数组进行向右滚动一步后，应该还是空数组
        assert_equal(np.roll(x, 1), np.array([]))
class TestRollaxis(TestCase):
    # 期望的形状索引为 (轴, 起始位置) 对于形状为 (1, 2, 3, 4) 的数组
    tgtshape = {
        (0, 0): (1, 2, 3, 4),
        (0, 1): (1, 2, 3, 4),
        (0, 2): (2, 1, 3, 4),
        (0, 3): (2, 3, 1, 4),
        (0, 4): (2, 3, 4, 1),
        (1, 0): (2, 1, 3, 4),
        (1, 1): (1, 2, 3, 4),
        (1, 2): (1, 2, 3, 4),
        (1, 3): (1, 3, 2, 4),
        (1, 4): (1, 3, 4, 2),
        (2, 0): (3, 1, 2, 4),
        (2, 1): (1, 3, 2, 4),
        (2, 2): (1, 2, 3, 4),
        (2, 3): (1, 2, 3, 4),
        (2, 4): (1, 2, 4, 3),
        (3, 0): (4, 1, 2, 3),
        (3, 1): (1, 4, 2, 3),
        (3, 2): (1, 2, 4, 3),
        (3, 3): (1, 2, 3, 4),
        (3, 4): (1, 2, 3, 4),
    }

    # 测试异常情况
    def test_exceptions(self):
        # 创建形状为 (1, 2, 3, 4) 的数组 a
        a = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
        # 断言：np.rollaxis 在轴 -5 处引发 np.AxisError 异常
        assert_raises(np.AxisError, np.rollaxis, a, -5, 0)
        # 断言：np.rollaxis 在起始位置 -5 处引发 np.AxisError 异常
        assert_raises(np.AxisError, np.rollaxis, a, 0, -5)
        # 断言：np.rollaxis 在轴 4 处引发 np.AxisError 异常
        assert_raises(np.AxisError, np.rollaxis, a, 4, 0)
        # 断言：np.rollaxis 在起始位置 5 处引发 np.AxisError 异常
        assert_raises(np.AxisError, np.rollaxis, a, 0, 5)

    @xfail  # XXX: ndarray.attributes
    # 测试结果
    def test_results(self):
        # 创建形状为 (1, 2, 3, 4) 的数组 a 的副本
        a = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4).copy()
        # 获取数组 a 的索引
        aind = np.indices(a.shape)
        # 断言：a 的 OWNDATA 属性为 True
        assert_(a.flags["OWNDATA"])
        # 遍历 self.tgtshape 中的每对索引 (i, j)
        for i, j in self.tgtshape:
            # 正轴，正起始位置
            res = np.rollaxis(a, axis=i, start=j)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            # 断言：res[i0, i1, i2, i3] 的所有元素等于 a 的所有元素
            assert_(np.all(res[i0, i1, i2, i3] == a))
            # 断言：res 的形状与 self.tgtshape[(i, j)] 相等
            assert_(res.shape == self.tgtshape[(i, j)], str((i, j)))
            # 断言：res 的 OWNDATA 属性为 False
            assert_(not res.flags["OWNDATA"])

            # 负轴，正起始位置
            ip = i + 1
            res = np.rollaxis(a, axis=-ip, start=j)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            # 断言：res[i0, i1, i2, i3] 的所有元素等于 a 的所有元素
            assert_(np.all(res[i0, i1, i2, i3] == a))
            # 断言：res 的形状与 self.tgtshape[(4 - ip, j)] 相等
            assert_(res.shape == self.tgtshape[(4 - ip, j)])
            # 断言：res 的 OWNDATA 属性为 False
            assert_(not res.flags["OWNDATA"])

            # 正轴，负起始位置
            jp = j + 1 if j < 4 else j
            res = np.rollaxis(a, axis=i, start=-jp)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            # 断言：res[i0, i1, i2, i3] 的所有元素等于 a 的所有元素
            assert_(np.all(res[i0, i1, i2, i3] == a))
            # 断言：res 的形状与 self.tgtshape[(i, 4 - jp)] 相等
            assert_(res.shape == self.tgtshape[(i, 4 - jp)])
            # 断言：res 的 OWNDATA 属性为 False
            assert_(not res.flags["OWNDATA"])

            # 负轴，负起始位置
            ip = i + 1
            jp = j + 1 if j < 4 else j
            res = np.rollaxis(a, axis=-ip, start=-jp)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            # 断言：res[i0, i1, i2, i3] 的所有元素等于 a 的所有元素
            assert_(np.all(res[i0, i1, i2, i3] == a))
            # 断言：res 的形状与 self.tgtshape[(4 - ip, 4 - jp)] 相等
            assert_(res.shape == self.tgtshape[(4 - ip, 4 - jp)])
            # 断言：res 的 OWNDATA 属性为 False
            assert_(not res.flags["OWNDATA"])
    def test_move_to_end(self):
        # 创建一个随机的5x6x7的numpy数组
        x = np.random.randn(5, 6, 7)
        # 遍历不同的移动源和预期结果
        for source, expected in [
            (0, (6, 7, 5)),  # 当source为0时，将第一个维度移动到最后，期望得到形状(6, 7, 5)
            (1, (5, 7, 6)),  # 当source为1时，将第二个维度移动到最后，期望得到形状(5, 7, 6)
            (2, (5, 6, 7)),  # 当source为2时，不做移动，期望得到形状(5, 6, 7)
            (-1, (5, 6, 7)),  # 当source为-1时，不做移动，期望得到形状(5, 6, 7)
        ]:
            # 执行移动轴操作，并获取实际的形状
            actual = np.moveaxis(x, source, -1).shape
            # 使用自定义的断言函数检查实际形状是否与预期相符
            assert_(actual, expected)

    def test_move_new_position(self):
        # 创建一个随机的1x2x3x4的numpy数组
        x = np.random.randn(1, 2, 3, 4)
        # 遍历不同的移动源、目的地和预期结果
        for source, destination, expected in [
            (0, 1, (2, 1, 3, 4)),  # 将第一个维度移动到第二个维度位置，期望得到形状(2, 1, 3, 4)
            (1, 2, (1, 3, 2, 4)),  # 将第二个维度移动到第三个维度位置，期望得到形状(1, 3, 2, 4)
            (1, -1, (1, 3, 4, 2)),  # 将第二个维度移动到倒数第一个维度位置，期望得到形状(1, 3, 4, 2)
        ]:
            # 执行移动轴操作，并获取实际的形状
            actual = np.moveaxis(x, source, destination).shape
            # 使用自定义的断言函数检查实际形状是否与预期相符
            assert_(actual, expected)

    def test_preserve_order(self):
        # 创建一个全零的1x2x3x4的numpy数组
        x = np.zeros((1, 2, 3, 4))
        # 遍历不同的移动源和目的地
        for source, destination in [
            (0, 0),    # 不做移动，期望保持形状(1, 2, 3, 4)
            (3, -1),   # 将第四个维度移动到倒数第一个维度位置，期望得到形状(1, 2, 3, 4)
            (-1, 3),   # 将最后一个维度移动到第四个维度位置，期望得到形状(1, 2, 3, 4)
            ([0, -1], [0, -1]),  # 同时移动第一个和最后一个维度，期望保持形状(1, 2, 3, 4)
            ([2, 0], [2, 0]),    # 将第三个维度移动到第一个维度位置，第一个维度移动到第三个维度位置，期望保持形状(1, 2, 3, 4)
            (range(4), range(4)),  # 不做移动，期望保持形状(1, 2, 3, 4)
        ]:
            # 执行移动轴操作，并获取实际的形状
            actual = np.moveaxis(x, source, destination).shape
            # 使用自定义的断言函数检查实际形状是否为预期的全零数组形状
            assert_(actual, (1, 2, 3, 4))

    def test_move_multiples(self):
        # 创建一个全零的0x1x2x3的numpy数组
        x = np.zeros((0, 1, 2, 3))
        # 遍历不同的多轴移动源、目的地和预期结果
        for source, destination, expected in [
            ([0, 1], [2, 3], (2, 3, 0, 1)),   # 将第一个和第二个维度移动到第三个和第四个维度位置，期望得到形状(2, 3, 0, 1)
            ([2, 3], [0, 1], (2, 3, 0, 1)),   # 将第三个和第四个维度移动到第一个和第二个维度位置，期望得到形状(2, 3, 0, 1)
            ([0, 1, 2], [2, 3, 0], (2, 3, 0, 1)),  # 将前三个维度移动到后三个维度位置，期望得到形状(2, 3, 0, 1)
            ([3, 0], [1, 0], (0, 3, 1, 2)),   # 将第四个维度移动到第二个维度位置，第一个维度移动到第一个维度位置，期望得到形状(0, 3, 1, 2)
            ([0, 3], [0, 1], (0, 3, 1, 2)),   # 将第一个维度移动到第一个维度位置，第四个维度移动到第二个维度位置，期望得到形状(0, 3, 1, 2)
        ]:
            # 执行移动轴操作，并获取实际的形状
            actual = np.moveaxis(x, source, destination).shape
            # 使用自定义的断言函数检查实际形状是否与预期相符
            assert_(actual, expected)

    def test_errors(self):
        # 创建一个随机的1x2x3的numpy数组
        x = np.random.randn(1, 2, 3)
        # 检查各种错误情况的断言
        assert_raises(np.AxisError, np.moveaxis, x, 3, 0)  # 当source超出范围时，期望抛出AxisError异常
        assert_raises(np.AxisError, np.moveaxis, x, -4, 0)  # 当source超出范围时，期望抛出AxisError异常
        assert_raises(
            np.AxisError, np.moveaxis, x, 0, 5  # 当destination超出范围时，期望抛出AxisError异常
        )
        assert_raises(
            ValueError, np.moveaxis, x, [0, 0], [0, 1]  # 当source中有重复轴时，期望抛出ValueError异常
        )
        assert_raises(
            ValueError,  # 当destination中有重复轴时，期望抛出ValueError异常
            np.moveaxis,
            x,
            [0, 1],
            [1, 1],
        )
        assert_raises(
            (ValueError, RuntimeError),  # 当source和destination的轴数不匹配时，期望抛出ValueError或RuntimeError异常
            np.moveaxis,
            x,
            0,
            [0, 1],
        )
        assert_raises(
            (ValueError, RuntimeError),  # 当source和destination的轴数不匹配时，期望抛出ValueError或RuntimeError异常
            np.moveaxis,
            x,
            [0, 1],
            [0],
        )

        x = [1, 2, 3]
        # 测试对于非ndarray输入，返回的结果是否与原
# 定义一个测试类 TestCross，继承自 TestCase
class TestCross(TestCase):

    # 定义一个测试方法 test_2x2
    def test_2x2(self):
        # 初始化向量 u 和 v
        u = [1, 2]
        v = [3, 4]
        # 预期的叉乘结果
        z = -2
        # 计算 u 和 v 的叉乘，并将结果赋给 cp
        cp = np.cross(u, v)
        # 使用 assert_equal 断言 cp 的值与预期的 z 相等
        assert_equal(cp, z)
        # 计算 v 和 u 的叉乘，并将结果赋给 cp
        cp = np.cross(v, u)
        # 使用 assert_equal 断言 cp 的值与预期的 -z 相等
        assert_equal(cp, -z)

    # 定义一个测试方法 test_2x3
    def test_2x3(self):
        # 初始化向量 u 和 v
        u = [1, 2]
        v = [3, 4, 5]
        # 预期的叉乘结果
        z = np.array([10, -5, -2])
        # 计算 u 和 v 的叉乘，并将结果赋给 cp
        cp = np.cross(u, v)
        # 使用 assert_equal 断言 cp 的值与预期的 z 相等
        assert_equal(cp, z)
        # 计算 v 和 u 的叉乘，并将结果赋给 cp
        cp = np.cross(v, u)
        # 使用 assert_equal 断言 cp 的值与预期的 -z 相等
        assert_equal(cp, -z)

    # 定义一个测试方法 test_3x3
    def test_3x3(self):
        # 初始化向量 u 和 v
        u = [1, 2, 3]
        v = [4, 5, 6]
        # 预期的叉乘结果
        z = np.array([-3, 6, -3])
        # 计算 u 和 v 的叉乘，并将结果赋给 cp
        cp = np.cross(u, v)
        # 使用 assert_equal 断言 cp 的值与预期的 z 相等
        assert_equal(cp, z)
        # 计算 v 和 u 的叉乘，并将结果赋给 cp
        cp = np.cross(v, u)
        # 使用 assert_equal 断言 cp 的值与预期的 -z 相等
        assert_equal(cp, -z)

    # 定义一个测试方法 test_broadcasting
    def test_broadcasting(self):
        # Ticket #2624 (Trac #2032)
        # 使用 np.tile 创建 u 和 v 的广播版本
        u = np.tile([1, 2], (11, 1))
        v = np.tile([3, 4], (11, 1))
        # 预期的叉乘结果
        z = -2
        # 使用 assert_equal 断言 np.cross(u, v) 的值与预期的 z 相等
        assert_equal(np.cross(u, v), z)
        # 使用 assert_equal 断言 np.cross(v, u) 的值与预期的 -z 相等
        assert_equal(np.cross(v, u), -z)
        # 使用 assert_equal 断言 np.cross(u, u) 的值为 0
        assert_equal(np.cross(u, u), 0)

        # 使用 np.tile 创建 u 和 v 的广播版本，并转置其中一个
        u = np.tile([1, 2], (11, 1)).T
        v = np.tile([3, 4, 5], (11, 1))
        # 预期的叉乘结果
        z = np.tile([10, -5, -2], (11, 1))
        # 使用 assert_equal 断言 np.cross(u, v, axisa=0) 的值与预期的 z 相等
        assert_equal(np.cross(u, v, axisa=0), z)
        # 使用 assert_equal 断言 np.cross(v, u.T) 的值与预期的 -z 相等
        assert_equal(np.cross(v, u.T), -z)
        # 使用 assert_equal 断言 np.cross(v, v) 的值为 0
        assert_equal(np.cross(v, v), 0)

        # 使用 np.tile 创建 u 和 v 的广播版本，并转置两个数组
        u = np.tile([1, 2, 3], (11, 1)).T
        v = np.tile([3, 4], (11, 1)).T
        # 预期的叉乘结果
        z = np.tile([-12, 9, -2], (11, 1))
        # 使用 assert_equal 断言 np.cross(u, v, axisa=0, axisb=0) 的值与预期的 z 相等
        assert_equal(np.cross(u, v, axisa=0, axisb=0), z)
        # 使用 assert_equal 断言 np.cross(v.T, u.T) 的值与预期的 -z 相等
        assert_equal(np.cross(v.T, u.T), -z)
        # 使用 assert_equal 断言 np.cross(u.T, u.T) 的值为 0
        assert_equal(np.cross(u.T, u.T), 0)

        # 使用 np.tile 创建 u 和 v 的广播版本，并转置其中一个数组
        u = np.tile([1, 2, 3], (5, 1))
        v = np.tile([4, 5, 6], (5, 1)).T
        # 预期的叉乘结果
        z = np.tile([-3, 6, -3], (5, 1))
        # 使用 assert_equal 断言 np.cross(u, v, axisb=0) 的值与预期的 z 相等
        assert_equal(np.cross(u, v, axisb=0), z)
        # 使用 assert_equal 断言 np.cross(v.T, u) 的值与预期的 -z 相等
        assert_equal(np.cross(v.T, u), -z)
        # 使用 assert_equal 断言 np.cross(u, u) 的值为 0
        assert_equal(np.cross(u, u), 0)

    # 定义一个测试方法 test_broadcasting_shapes
    def test_broadcasting_shapes(self):
        # 初始化 u 和 v 的形状
        u = np.ones((2, 1, 3))
        v = np.ones((5, 3))
        # 使用 assert_equal 断言 np.cross(u, v).shape 的形状为 (2, 5, 3)
        assert_equal(np.cross(u, v).shape, (2, 5, 3))
        # 初始化 u 和 v 的形状
        u = np.ones((10, 3, 5))
        v = np.ones((2, 5))
        # 使用 assert_equal 断言 np.cross(u, v, axisa=1, axisb=0).shape 的形状为 (10, 5, 3)
        assert_equal(np.cross(u, v, axisa=1, axisb=0).shape, (10, 5, 3))
        # 使用 assert_raises 断言调用 np.cross(u, v, axisa=1, axisb=2) 会抛出 np.AxisError 异常
        assert_raises(np.AxisError, np.cross, u, v, axisa=1, axisb=2)
        # 使用 assert_raises 断言调用 np.cross(u, v, axisa=3, axisb=0) 会抛出 np.AxisError 异常
        assert_raises(np.AxisError, np.cross, u, v, axisa=3, axisb=0)
        # 初始化 u 和 v 的形状
        u = np.ones((10, 3, 5, 7))
        v = np.ones((5, 7, 2))
        # 使用 assert_equal 断言 np.cross(u, v, axisa=1, axisc=2).shape 的形状为 (10, 5, 3, 7)
        assert_equal(np.cross(u, v, axisa=1, axisc=2).shape, (10, 5, 3, 7))
        # 使用 assert_raises 断言调用 np.cross(u, v, axisa=-5, axisb=2) 会抛出 np.AxisError 异常
        assert_raises(np.AxisError, np.cross, u, v, axisa=-5, axisb=2)
        # 使用 assert_raises 断言调用 np.cross(u, v, axisa=1, axisb=-4) 会抛出 np.AxisError 异常
        assert_raises(np.AxisError, np.cross, u, v, axisa=1, axisb=-4)
        # 进行 gh-5885 的回
    # 定义一个测试函数，测试 np.outer() 方法中的 out 参数
    def test_outer_out_param(self):
        # 创建一个包含5个元素的全1数组
        arr1 = np.ones((5,))
        # 创建一个包含2个元素的全1数组
        arr2 = np.ones((2,))
        # 在-2到2之间生成5个等间距的数
        arr3 = np.linspace(-2, 2, 5)
        # 创建一个空的5x5数组
        out1 = np.empty(shape=(5, 5))
        # 创建一个空的2x5数组
        out2 = np.empty(shape=(2, 5))
        # 使用 np.outer() 方法计算 arr1 和 arr3 的外积，将结果存储在 out1 中
        res1 = np.outer(arr1, arr3, out1)
        # 断言计算结果与预期结果相等
        assert_equal(res1, out1)
        # 使用 np.outer() 方法计算 arr2 和 arr3 的外积，将结果存储在 out2 中
        assert_equal(np.outer(arr2, arr3, out2), out2)
@instantiate_parametrized_tests
class TestIndices(TestCase):
    # 参数化测试类装饰器，用于自动生成参数化的测试用例

    def test_simple(self):
        # 简单测试示例
        [x, y] = np.indices((4, 3))
        assert_array_equal(x, np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]))
        assert_array_equal(y, np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]))

    def test_single_input(self):
        # 测试单个输入情况
        [x] = np.indices((4,))
        assert_array_equal(x, np.array([0, 1, 2, 3]))

        [x] = np.indices((4,), sparse=True)
        assert_array_equal(x, np.array([0, 1, 2, 3]))

    def test_scalar_input(self):
        # 测试标量输入情况
        assert_array_equal([], np.indices(()))
        assert_array_equal([], np.indices((), sparse=True))
        assert_array_equal([[]], np.indices((0,)))
        assert_array_equal([[]], np.indices((0,), sparse=True))

    def test_sparse(self):
        # 测试稀疏矩阵情况
        [x, y] = np.indices((4, 3), sparse=True)
        assert_array_equal(x, np.array([[0], [1], [2], [3]]))
        assert_array_equal(y, np.array([[0, 1, 2]]))

    @parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
    @parametrize("dims", [(), (0,), (4, 3)])
    def test_return_type(self, dtype, dims):
        # 测试返回类型的情况
        inds = np.indices(dims, dtype=dtype)
        assert_(inds.dtype == dtype)

        for arr in np.indices(dims, dtype=dtype, sparse=True):
            assert_(arr.dtype == dtype)


@xpassIfTorchDynamo  # (reason="TODO")
class TestRequire(TestCase):
    # 装饰器，暂时跳过由 Torch Dynamo 执行

    flag_names = [
        "C",
        "C_CONTIGUOUS",
        "CONTIGUOUS",
        "F",
        "F_CONTIGUOUS",
        "FORTRAN",
        "A",
        "ALIGNED",
        "W",
        "WRITEABLE",
        "O",
        "OWNDATA",
    ]

    def generate_all_false(self, dtype):
        # 生成所有标志为假的数组
        arr = np.zeros((2, 2), [("junk", "i1"), ("a", dtype)])
        arr.setflags(write=False)
        a = arr["a"]
        assert_(not a.flags["C"])
        assert_(not a.flags["F"])
        assert_(not a.flags["O"])
        assert_(not a.flags["W"])
        assert_(not a.flags["A"])
        return a

    def set_and_check_flag(self, flag, dtype, arr):
        # 设置并检查特定标志
        if dtype is None:
            dtype = arr.dtype
        b = np.require(arr, dtype, [flag])
        assert_(b.flags[flag])
        assert_(b.dtype == dtype)

        # 进一步调用 np.require 应该返回相同的数组，除非指定 OWNDATA 标志
        c = np.require(b, None, [flag])
        if flag[0] != "O":
            assert_(c is b)
        else:
            assert_(c.flags[flag])

    def test_require_each(self):
        # 测试每个标志需求的情况
        id = ["f8", "i4"]
        fd = [None, "f8", "c16"]
        for idtype, fdtype, flag in itertools.product(id, fd, self.flag_names):
            a = self.generate_all_false(idtype)
            self.set_and_check_flag(flag, fdtype, a)

    def test_unknown_requirement(self):
        # 测试未知需求的情况
        a = self.generate_all_false("f8")
        assert_raises(KeyError, np.require, a, None, "Q")
    # 定义一个测试方法，用于测试非数组输入情况
    def test_non_array_input(self):
        # 使用 np.require() 函数创建一个数组 a，要求其数据类型为 'i4'，并指定标志为 ["C", "A", "O"]
        a = np.require([1, 2, 3, 4], "i4", ["C", "A", "O"])
        # 断言数组 a 的标志中包含 'O'
        assert_(a.flags["O"])
        # 断言数组 a 的标志中包含 'C'
        assert_(a.flags["C"])
        # 断言数组 a 的标志中包含 'A'
        assert_(a.flags["A"])
        # 断言数组 a 的数据类型为 'i4'
        assert_(a.dtype == "i4")
        # 断言数组 a 等于 [1, 2, 3, 4]
        assert_equal(a, [1, 2, 3, 4])

    # 定义一个测试方法，用于测试同时指定 'C' 和 'F' 标志的情况
    def test_C_and_F_simul(self):
        # 调用 self.generate_all_false("f8") 方法生成一个值全部为 False 的数组 a，数据类型为 'f8'
        a = self.generate_all_false("f8")
        # 断言调用 np.require(a, None, ["C", "F"]) 时会引发 ValueError 异常
        assert_raises(ValueError, np.require, a, None, ["C", "F"])
@xpassIfTorchDynamo  # (reason="TODO")
class TestBroadcast(TestCase):
    def test_broadcast_in_args(self):
        # 测试用例：gh-5881
        arrs = [
            np.empty((6, 7)),
            np.empty((5, 6, 1)),
            np.empty((7,)),
            np.empty((5, 1, 7)),
        ]
        # 创建多个广播对象
        mits = [
            np.broadcast(*arrs),
            np.broadcast(np.broadcast(*arrs[:0]), np.broadcast(*arrs[0:])),
            np.broadcast(np.broadcast(*arrs[:1]), np.broadcast(*arrs[1:])),
            np.broadcast(np.broadcast(*arrs[:2]), np.broadcast(*arrs[2:])),
            np.broadcast(arrs[0], np.broadcast(*arrs[1:-1]), arrs[-1]),
        ]
        for mit in mits:
            # 验证广播对象的形状和维度
            assert_equal(mit.shape, (5, 6, 7))
            assert_equal(mit.ndim, 3)
            assert_equal(mit.nd, 3)  # 这行存在错误，应该是 `mit.ndim`
            assert_equal(mit.numiter, 4)
            for a, ia in zip(arrs, mit.iters):
                # 验证每个数组是其对应迭代器的基础
                assert_(a is ia.base)

    def test_broadcast_single_arg(self):
        # 测试用例：gh-6899
        arrs = [np.empty((5, 6, 7))]
        mit = np.broadcast(*arrs)
        assert_equal(mit.shape, (5, 6, 7))
        assert_equal(mit.ndim, 3)
        assert_equal(mit.nd, 3)
        assert_equal(mit.numiter, 1)
        assert_(arrs[0] is mit.iters[0].base)

    def test_number_of_arguments(self):
        arr = np.empty((5,))
        for j in range(35):
            arrs = [arr] * j
            if j > 32:
                # 测试超过32个参数时是否会抛出 ValueError 异常
                assert_raises(ValueError, np.broadcast, *arrs)
            else:
                mit = np.broadcast(*arrs)
                assert_equal(mit.numiter, j)

    def test_broadcast_error_kwargs(self):
        # 测试用例：gh-13455
        arrs = [np.empty((5, 6, 7))]
        mit = np.broadcast(*arrs)
        mit2 = np.broadcast(*arrs, **{})  # noqa: PIE804
        assert_equal(mit.shape, mit2.shape)
        assert_equal(mit.ndim, mit2.ndim)
        assert_equal(mit.nd, mit2.nd)
        assert_equal(mit.numiter, mit2.numiter)
        assert_(mit.iters[0].base is mit2.iters[0].base)

        assert_raises(ValueError, np.broadcast, 1, **{"x": 1})  # noqa: PIE804

    @skip(reason="error messages do not match.")
    def test_shape_mismatch_error_message(self):
        with assert_raises(
            ValueError,
            match=r"arg 0 with shape \(1, 3\) and " r"arg 2 with shape \(2,\)",
        ):
            np.broadcast([[1, 2, 3]], [[4], [5]], [6, 7])


class TestTensordot(TestCase):
    def test_zero_dimension(self):
        # 测试解决问题 #5663
        a = np.zeros((3, 0))
        b = np.zeros((0, 4))
        td = np.tensordot(a, b, (1, 0))
        assert_array_equal(td, np.dot(a, b))

    def test_zero_dimension_einsum(self):
        # 测试解决问题 #5663
        a = np.zeros((3, 0))
        b = np.zeros((0, 4))
        td = np.tensordot(a, b, (1, 0))
        assert_array_equal(td, np.einsum("ij,jk", a, b))
    def test_zero_dimensional(self):
        # 定义测试方法，用于测试零维数组的情况
        # gh-12130 表示这是针对 GitHub issue 12130 的测试

        # 创建一个零维的 NumPy 数组，值为 1
        arr_0d = np.array(1)

        # 对零维数组进行张量点积计算，([] 表示没有要收缩的轴)
        # 这种情况下，不收缩任何轴是被明确定义的
        ret = np.tensordot(
            arr_0d, arr_0d, ([], [])
        )

        # 断言计算结果 ret 与数组 arr_0d 相等
        assert_array_equal(ret, arr_0d)
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```