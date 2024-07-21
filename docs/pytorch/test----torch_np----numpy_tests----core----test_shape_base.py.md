# `.\pytorch\test\torch_np\numpy_tests\core\test_shape_base.py`

```py
# Owner(s): ["module: dynamo"]

# 导入 functools 模块，用于创建偏函数
import functools

# 导入 unittest 模块中的 expectedFailure 和 skipIf 别名，用于标记预期失败的测试和条件跳过测试
from unittest import expectedFailure as xfail, skipIf as skipif

# 导入 numpy 库
import numpy

# 导入 pytest 库及其 raises 别名，用于断言异常
import pytest
from pytest import raises as assert_raises

# 导入 torch.testing._internal.common_utils 中的多个函数和常量
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo,
)

# 根据 TEST_WITH_TORCHDYNAMO 变量决定导入 numpy 还是 torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import (
        array,
        atleast_1d,
        atleast_2d,
        atleast_3d,
        AxisError,
        concatenate,
        hstack,
        newaxis,
        stack,
        vstack,
    )
    from numpy.testing import assert_, assert_array_equal, assert_equal
else:
    import torch._numpy as np
    from torch._numpy import (
        array,
        atleast_1d,
        atleast_2d,
        atleast_3d,
        AxisError,
        concatenate,
        hstack,
        newaxis,
        stack,
        vstack,
    )
    from torch._numpy.testing import assert_, assert_array_equal, assert_equal

# 创建 skip 函数作为 functools.partial(skipif, True) 的别名
skip = functools.partial(skipif, True)

# 定义一个常量 IS_PYPY，并赋值为 False
IS_PYPY = False

# 定义 TestCase 的子类 TestAtleast1d，用于测试 atleast_1d 函数
class TestAtleast1d(TestCase):

    # 测试 0 维数组的情况
    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [array([1]), array([2])]
        assert_array_equal(res, desired)

    # 测试 1 维数组的情况
    def test_1D_array(self):
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [array([1, 2]), array([2, 3])]
        assert_array_equal(res, desired)

    # 测试 2 维数组的情况
    def test_2D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    # 测试 3 维数组的情况
    def test_3D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    # 测试 r1array 函数的行为是否等效于 Travis O 的 r1array 函数
    def test_r1array(self):
        assert atleast_1d(3).shape == (1,)
        assert atleast_1d(3j).shape == (1,)
        assert atleast_1d(3.0).shape == (1,)
        assert atleast_1d([[2, 3], [4, 5]]).shape == (2, 2)


# 定义 TestCase 的子类 TestAtleast2d，用于测试 atleast_2d 函数
class TestAtleast2d(TestCase):

    # 测试 0 维数组的情况
    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [array([[1]]), array([[2]])]
        assert_array_equal(res, desired)

    # 测试 1 维数组的情况
    def test_1D_array(self):
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [array([[1, 2]]), array([[2, 3]])]
        assert_array_equal(res, desired)
    # 定义测试函数，用于测试处理二维数组的情况
    def test_2D_array(self):
        # 创建两个二维数组 a 和 b
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        # 对数组 a 和 b 应用 atleast_2d 函数，存储结果到 res 列表中
        res = [atleast_2d(a), atleast_2d(b)]
        # 定义预期结果列表，存储原始的数组 a 和 b
        desired = [a, b]
        # 断言 res 和 desired 是否相等
        assert_array_equal(res, desired)

    # 定义测试函数，用于测试处理三维数组的情况
    def test_3D_array(self):
        # 创建两个二维数组 a 和 b
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        # 将数组 a 和 b 分别封装为包含这两个数组的新数组，形成三维数组
        a = array([a, a])
        b = array([b, b])
        # 对数组 a 和 b 应用 atleast_2d 函数，存储结果到 res 列表中
        res = [atleast_2d(a), atleast_2d(b)]
        # 定义预期结果列表，存储原始的数组 a 和 b
        desired = [a, b]
        # 断言 res 和 desired 是否相等
        assert_array_equal(res, desired)

    # 定义测试函数，用于测试处理标量到数组转换的情况
    def test_r2array(self):
        """Test to make sure equivalent Travis O's r2array function"""
        # 断言 atleast_2d 函数应用于标量 3 后的形状为 (1, 1)
        assert atleast_2d(3).shape == (1, 1)
        # 断言 atleast_2d 函数应用于复数列表 [3j, 1] 后的形状为 (1, 2)
        assert atleast_2d([3j, 1]).shape == (1, 2)
        # 断言 atleast_2d 函数应用于三维列表 [[[3, 1], [4, 5]], [[3, 5], [1, 2]]] 后的形状为 (2, 2, 2)
        assert atleast_2d([[[3, 1], [4, 5]], [[3, 5], [1, 2]]]).shape == (2, 2, 2)
class TestAtleast3d(TestCase):
    # 定义测试类 TestAtleast3d，继承自 TestCase，用于测试 atleast_3d 函数的行为

    def test_0D_array(self):
        # 测试 0 维数组的情况
        a = array(1)
        b = array(2)
        res = [atleast_3d(a), atleast_3d(b)]
        # 使用 atleast_3d 函数处理数组 a 和 b，并将结果保存在 res 中
        desired = [array([[[1]]]), array([[[2]]])]
        # 预期的结果是将数组转换为至少 3 维的形式
        assert_array_equal(res, desired)
        # 断言 res 和 desired 应当相等

    def test_1D_array(self):
        # 测试 1 维数组的情况
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_3d(a), atleast_3d(b)]
        # 使用 atleast_3d 函数处理数组 a 和 b，并将结果保存在 res 中
        desired = [array([[[1], [2]]]), array([[[2], [3]]])]
        # 预期的结果是将数组转换为至少 3 维的形式
        assert_array_equal(res, desired)
        # 断言 res 和 desired 应当相等

    def test_2D_array(self):
        # 测试 2 维数组的情况
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_3d(a), atleast_3d(b)]
        # 使用 atleast_3d 函数处理数组 a 和 b，并将结果保存在 res 中
        desired = [a[:, :, newaxis], b[:, :, newaxis]]
        # 预期的结果是将数组在第三维上进行扩展
        assert_array_equal(res, desired)
        # 断言 res 和 desired 应当相等

    def test_3D_array(self):
        # 测试 3 维数组的情况
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_3d(a), atleast_3d(b)]
        # 使用 atleast_3d 函数处理数组 a 和 b，并将结果保存在 res 中
        desired = [a, b]
        # 预期的结果是不会改变已经是 3 维的数组结构
        assert_array_equal(res, desired)
        # 断言 res 和 desired 应当相等


class TestHstack(TestCase):
    # 定义测试类 TestHstack，继承自 TestCase，用于测试 hstack 函数的行为

    def test_non_iterable(self):
        # 测试输入参数非可迭代对象的情况
        assert_raises(TypeError, hstack, 1)
        # 断言调用 hstack 函数时会抛出 TypeError 异常

    def test_empty_input(self):
        # 测试输入为空的情况
        assert_raises(ValueError, hstack, ())
        # 断言调用 hstack 函数时会抛出 ValueError 异常

    def test_0D_array(self):
        # 测试 0 维数组的情况
        a = array(1)
        b = array(2)
        res = hstack([a, b])
        # 使用 hstack 函数处理数组 a 和 b，并将结果保存在 res 中
        desired = array([1, 2])
        # 预期的结果是将数组在水平方向堆叠
        assert_array_equal(res, desired)
        # 断言 res 和 desired 应当相等

    def test_1D_array(self):
        # 测试 1 维数组的情况
        a = array([1])
        b = array([2])
        res = hstack([a, b])
        # 使用 hstack 函数处理数组 a 和 b，并将结果保存在 res 中
        desired = array([1, 2])
        # 预期的结果是将数组在水平方向堆叠
        assert_array_equal(res, desired)
        # 断言 res 和 desired 应当相等

    def test_2D_array(self):
        # 测试 2 维数组的情况
        a = array([[1], [2]])
        b = array([[1], [2]])
        res = hstack([a, b])
        # 使用 hstack 函数处理数组 a 和 b，并将结果保存在 res 中
        desired = array([[1, 1], [2, 2]])
        # 预期的结果是将数组在水平方向堆叠
        assert_array_equal(res, desired)
        # 断言 res 和 desired 应当相等

    def test_generator(self):
        # 测试使用生成器作为输入的情况
        # numpy 1.24 emits warnings but we don't
        # with assert_warns(FutureWarning):
        hstack([np.arange(3) for _ in range(2)])
        # with assert_warns(FutureWarning):
        hstack([x for x in np.ones((3, 2))])  # noqa: C416

    @skipif(numpy.__version__ < "1.24", reason="NP_VER: fails on NumPy 1.23.x")
    def test_casting_and_dtype(self):
        # 测试类型转换和数据类型的情况
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        res = np.hstack(np.append(a, b), casting="unsafe", dtype=np.int64)
        # 使用 np.hstack 处理数组 a 和 b，并设置类型转换为 "unsafe"，数据类型为 np.int64
        expected_res = np.array([1, 2, 3, 2, 3, 4])
        # 预期的结果是将数组在水平方向堆叠，并进行类型转换和数据类型设定
        assert_array_equal(res, expected_res)
        # 断言 res 和 expected_res 应当相等

    def test_casting_and_dtype_type_error(self):
        # 测试类型转换和数据类型错误的情况
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        with pytest.raises(TypeError):
            hstack((a, b), casting="safe", dtype=np.int64)
        # 使用 hstack 处理数组 a 和 b，并设置类型转换为 "safe"，预期会抛出 TypeError 异常


class TestVstack(TestCase):
    # 定义测试类 TestVstack，继承自 TestCase，用于测试 vstack 函数的行为

    def test_non_iterable(self):
        # 测试输入参数非可迭代对象的情况
        assert_raises(TypeError, vstack, 1)
        # 断言调用 vstack 函数时会抛出 TypeError 异常

    def test_empty_input(self):
        # 测试输入为空的情况
        assert_raises(ValueError, vstack, ())
        # 断言调用 vstack 函数时会抛出 ValueError 异常

    def test_0D_array(self):
        # 测试 0 维数组的情况
        a = array(1)
        b = array(2)
        res = vstack([a, b])
        # 使用 vstack 函数处理数组 a 和 b，并将结果保存在 res 中
        desired = array([[1], [2]])
        # 预期的结果是将数组在垂直方向堆叠
        assert_array_equal(res, desired)
        # 断言 res 和 desired 应当相等
    # 定义测试函数，测试将两个一维数组堆叠成二维数组的情况
    def test_1D_array(self):
        # 创建第一个一维数组
        a = array([1])
        # 创建第二个一维数组
        b = array([2])
        # 使用 vstack 将两个一维数组堆叠成二维数组
        res = vstack([a, b])
        # 预期的结果二维数组
        desired = array([[1], [2]])
        # 断言堆叠后的结果与预期结果相等
        assert_array_equal(res, desired)

    # 定义测试函数，测试将两个二维数组堆叠成更大的二维数组的情况
    def test_2D_array(self):
        # 创建第一个二维数组
        a = array([[1], [2]])
        # 创建第二个二维数组
        b = array([[1], [2]])
        # 使用 vstack 将两个二维数组堆叠成更大的二维数组
        res = vstack([a, b])
        # 预期的结果二维数组
        desired = array([[1], [2], [1], [2]])
        # 断言堆叠后的结果与预期结果相等
        assert_array_equal(res, desired)

    # 定义测试函数，测试将两个一维数组堆叠成二维数组的情况
    def test_2D_array2(self):
        # 创建第一个一维数组
        a = array([1, 2])
        # 创建第二个一维数组
        b = array([1, 2])
        # 使用 vstack 将两个一维数组堆叠成二维数组
        res = vstack([a, b])
        # 预期的结果二维数组
        desired = array([[1, 2], [1, 2]])
        # 断言堆叠后的结果与预期结果相等
        assert_array_equal(res, desired)

    @xfail  # (reason="vstack w/generators")
    # 标记为预期失败的测试函数，测试在使用生成器堆叠时的情况
    def test_generator(self):
        # 使用 pytest 断言预期会抛出 TypeError 异常，提示需要堆叠的数组必须是
        # 数组对象而不是生成器
        with pytest.raises(TypeError, match="arrays to stack must be"):
            vstack(np.arange(3) for _ in range(2))

    @skipif(numpy.__version__ < "1.24", reason="casting kwarg is new in NumPy 1.24")
    # 在满足条件时跳过测试的装饰器，测试在指定 NumPy 版本中的类型转换功能
    def test_casting_and_dtype(self):
        # 创建第一个数组
        a = np.array([1, 2, 3])
        # 创建第二个数组
        b = np.array([2.5, 3.5, 4.5])
        # 使用 np.vstack 将两个数组堆叠成更大的数组，指定类型转换为 int64
        res = np.vstack((a, b), casting="unsafe", dtype=np.int64)
        # 预期的结果数组
        expected_res = np.array([[1, 2, 3], [2, 3, 4]])
        # 断言堆叠后的结果与预期结果相等
        assert_array_equal(res, expected_res)

    @skipif(numpy.__version__ < "1.24", reason="casting kwarg is new in NumPy 1.24")
    # 在满足条件时跳过测试的装饰器，测试在指定 NumPy 版本中的类型转换功能，并预期 TypeError
    def test_casting_and_dtype_type_error(self):
        # 创建第一个数组
        a = np.array([1, 2, 3])
        # 创建第二个数组
        b = np.array([2.5, 3.5, 4.5])
        # 使用 vstack 将两个数组堆叠成更大的数组，指定类型转换为 int64（预期出错）
        with pytest.raises(TypeError):
            vstack((a, b), casting="safe", dtype=np.int64)
# 使用装饰器实例化参数化测试，对应的测试类为 TestConcatenate
@instantiate_parametrized_tests
class TestConcatenate(TestCase):

    # 测试 np.concatenate() 函数在指定 out=... 和 dtype=... 参数时抛出 TypeError 异常
    def test_out_and_dtype_simple(self):
        # 创建长度为 3、4 和 7 的全一数组 a, b 和 out
        a, b, out = np.ones(3), np.ones(4), np.ones(3 + 4)

        # 使用 pytest 的上下文管理器检查是否抛出 TypeError 异常
        with pytest.raises(TypeError):
            np.concatenate((a, b), out=out, dtype=float)

    # 测试 np.concatenate() 函数返回复制的数组
    def test_returns_copy(self):
        # 创建一个 3x3 的单位矩阵 a
        a = np.eye(3)
        # 对 np.concatenate() 函数应用于 a 的结果进行修改
        b = np.concatenate([a])
        b[0, 0] = 2
        # 断言修改后的 b[0, 0] 不等于原始矩阵 a 的值
        assert b[0, 0] != a[0, 0]

    # 测试 np.concatenate() 函数中的异常情况
    def test_exceptions(self):
        # 测试轴参数 axis 必须在有效范围内的情况
        for ndim in [1, 2, 3]:
            # 创建维度为 (1,) * ndim 的全一数组 a
            a = np.ones((1,) * ndim)
            # 在 axis=0 的情况下，对两个数组 a 进行连接
            np.concatenate((a, a), axis=0)  # OK
            # 断言在指定 axis=ndim 时会抛出 IndexError 或者 np.AxisError 异常
            assert_raises((IndexError, np.AxisError), np.concatenate, (a, a), axis=ndim)
            # 断言在指定 axis=-(ndim + 1) 时会抛出 IndexError 或者 np.AxisError 异常
            assert_raises(
                (IndexError, np.AxisError), np.concatenate, (a, a), axis=-(ndim + 1)
            )

        # 标量无法进行连接，应该抛出 RuntimeError 或者 ValueError 异常
        assert_raises((RuntimeError, ValueError), concatenate, (0,))
        assert_raises((RuntimeError, ValueError), concatenate, (np.array(0),))

        # 测试输入数组的维度必须匹配，应该抛出 RuntimeError 或者 ValueError 异常
        assert_raises(
            (RuntimeError, ValueError),
            np.concatenate,
            (np.zeros(1), np.zeros((1, 1))),
        )

        # 测试除了连接轴以外，各个输入数组的形状必须完全匹配，应该抛出 RuntimeError 或者 ValueError 异常
        a = np.ones((1, 2, 3))
        b = np.ones((2, 2, 3))
        axis = list(range(3))
        for i in range(3):
            # 在指定 axis=axis[0] 的情况下连接数组 a 和 b
            np.concatenate((a, b), axis=axis[0])  # OK
            # 断言在指定 axis=axis[1] 的情况下会抛出 RuntimeError 或者 ValueError 异常
            assert_raises(
                (RuntimeError, ValueError),
                np.concatenate,
                (a, b),
                axis=axis[1],
            )
            # 断言在指定 axis=axis[2] 的情况下会抛出 RuntimeError 或者 ValueError 异常
            assert_raises(
                (RuntimeError, ValueError), np.concatenate, (a, b), axis=axis[2]
            )
            # 将数组 a 和 b 的轴进行移动以进行下一轮测试
            a = np.moveaxis(a, -1, 0)
            b = np.moveaxis(b, -1, 0)
            # 将 axis 列表的第一个元素移动到末尾，以便下一轮迭代使用
            axis.append(axis.pop(0))

        # 没有数组可供连接时，应该抛出 ValueError 异常
        assert_raises(ValueError, concatenate, ())
    # 定义一个测试函数，测试在 axis=None 的情况下的 np.concatenate 函数
    def test_concatenate_axis_None(self):
        # 创建一个 2x2 的浮点数 ndarray，范围是 [0, 4)
        a = np.arange(4, dtype=np.float64).reshape((2, 2))
        # 创建一个包含整数 [0, 1, 2] 的列表
        b = list(range(3))

        # 对两个 a 数组进行无轴向连接
        r = np.concatenate((a, a), axis=None)
        # 断言结果数组的数据类型与 a 的数据类型相同
        assert r.dtype == a.dtype
        # 断言结果数组的维度为 1
        assert r.ndim == 1

        # 对 a 数组和 b 列表进行无轴向连接
        r = np.concatenate((a, b), axis=None)
        # 断言结果数组的元素个数等于 a 和 b 元素个数之和
        assert r.size == a.size + len(b)
        # 断言结果数组的数据类型与 a 的数据类型相同
        assert r.dtype == a.dtype

        # 创建一个大小为 a.size + len(b) 的全零数组 out
        out = np.zeros(a.size + len(b))
        # 对 a 数组和 b 列表进行无轴向连接
        r = np.concatenate((a, b), axis=None)
        # 使用预先创建的 out 数组对 a 数组和 b 列表进行无轴向连接
        rout = np.concatenate((a, b), axis=None, out=out)
        # 断言 out 和 rout 是同一个对象
        assert out is rout
        # 断言 r 和 rout 的所有元素相等
        assert np.all(r == rout)

    # 标记为 @xpassIfTorchDynamo 的测试函数，注释中提到 "concatenate(x, axis=None) relies on x being a sequence"
    @xpassIfTorchDynamo
    def test_large_concatenate_axis_None(self):
        # 当没有指定轴向时，np.concatenate 使用扁平化版本
        # 这段代码测试了一个数组 x 的无轴向连接，断言 x 和结果数组 r 相等
        x = np.arange(1, 100)
        r = np.concatenate(x, None)
        assert np.all(x == r)

        # 这段代码可能已经不推荐使用：
        # 测试当 axis >= MAXDIMS 时，np.concatenate 的行为
        r = np.concatenate(x, 100)  # axis is >= MAXDIMS
        assert_array_equal(x, r)

    # 测试 concatenate 函数的各种用法
    def test_concatenate(self):
        # 测试 concatenate 函数
        # 当传入一个序列时，返回未修改的（但作为数组的）结果

        # XXX: 传入单个参数；依赖于 ndarray 是一个序列
        r4 = list(range(4))
        # assert_array_equal(concatenate((r4,)), r4)
        # # 任何序列
        # assert_array_equal(concatenate((tuple(r4),)), r4)
        # assert_array_equal(concatenate((array(r4),)), r4)
        # 默认进行一维连接
        r3 = list(range(3))
        assert_array_equal(concatenate((r4, r3)), r4 + r3)
        # 不同类型的序列混合
        assert_array_equal(concatenate((tuple(r4), r3)), r4 + r3)
        assert_array_equal(concatenate((array(r4), r3)), r4 + r3)
        # 明确指定轴向
        assert_array_equal(concatenate((r4, r3), 0), r4 + r3)
        # 包括负数轴向
        assert_array_equal(concatenate((r4, r3), -1), r4 + r3)
        
        # 二维数组连接
        a23 = array([[10, 11, 12], [13, 14, 15]])
        a13 = array([[0, 1, 2]])
        res = array([[10, 11, 12], [13, 14, 15], [0, 1, 2]])
        assert_array_equal(concatenate((a23, a13)), res)
        assert_array_equal(concatenate((a23, a13), 0), res)
        assert_array_equal(concatenate((a23.T, a13.T), 1), res.T)
        assert_array_equal(concatenate((a23.T, a13.T), -1), res.T)
        
        # 数组的形状必须匹配
        assert_raises((RuntimeError, ValueError), concatenate, (a23.T, a13.T), 0)
        
        # 三维数组连接
        res = np.arange(2 * 3 * 7).reshape((2, 3, 7))
        a0 = res[..., :4]
        a1 = res[..., 4:6]
        a2 = res[..., 6:]
        assert_array_equal(concatenate((a0, a1, a2), 2), res)
        assert_array_equal(concatenate((a0, a1, a2), -1), res)
        assert_array_equal(concatenate((a0.T, a1.T, a2.T), 0), res.T)

        # 使用 out 参数进行连接
        out = res.copy()
        rout = concatenate((a0, a1, a2), 2, out=out)
        assert_(out is rout)
        assert_equal(res, rout)
    @skip(reason="concat, arrays, sequence")
    # 装饰器：跳过当前测试用例，原因是涉及到连接、数组和序列操作
    @skipif(IS_PYPY, reason="PYPY handles sq_concat, nb_add differently than cpython")
    # 装饰器：如果在 PyPy 环境下，跳过当前测试用例，原因是 PyPy 处理 sq_concat 和 nb_add 与 CPython 不同

    def test_operator_concat(self):
        # 导入操作符模块
        import operator

        # 创建两个数组 a 和 b
        a = array([1, 2])
        b = array([3, 4])

        # 创建一个普通列表 n
        n = [1, 2]

        # 创建期望的结果数组 res
        res = array([1, 2, 3, 4])

        # 断言以下操作会引发 TypeError 异常，使用 operator.concat 连接数组 a 和 b
        assert_raises(TypeError, operator.concat, a, b)
        assert_raises(TypeError, operator.concat, a, n)
        assert_raises(TypeError, operator.concat, n, a)
        assert_raises(TypeError, operator.concat, a, 1)
        assert_raises(TypeError, operator.concat, 1, a)

    def test_bad_out_shape(self):
        # 创建两个数组 a 和 b
        a = array([1, 2])
        b = array([3, 4])

        # 断言以下操作会引发 ValueError 异常，使用 concatenate 函数连接 a 和 b，并使用不符合形状要求的输出数组
        assert_raises(ValueError, concatenate, (a, b), out=np.empty(5))
        assert_raises(ValueError, concatenate, (a, b), out=np.empty((4, 1)))
        assert_raises(ValueError, concatenate, (a, b), out=np.empty((1, 4)))

        # 使用 concatenate 函数连接 a 和 b，输出数组形状为 (4,)
        concatenate((a, b), out=np.empty(4))

    @parametrize("axis", [None, 0])
    @parametrize(
        "out_dtype", ["c8", "f4", "f8", "i8"]
    )  # torch does not have ">f8", "S4"
    @parametrize("casting", ["no", "equiv", "safe", "same_kind", "unsafe"])
    def test_out_and_dtype(self, axis, out_dtype, casting):
        # 比较使用 `out=out` 和 `dtype=out.dtype` 的用法
        out = np.empty(4, dtype=out_dtype)
        to_concat = (array([1.1, 2.2]), array([3.3, 4.4]))

        # 如果无法将 to_concat[0] 强制转换为 out_dtype，则断言会引发 TypeError 异常
        if not np.can_cast(to_concat[0], out_dtype, casting=casting):
            with assert_raises(TypeError):
                concatenate(to_concat, out=out, axis=axis, casting=casting)
            with assert_raises(TypeError):
                concatenate(to_concat, dtype=out.dtype, axis=axis, casting=casting)
        else:
            # 使用指定的 out 和 casting 参数连接数组，并断言返回的结果与 out 是同一对象
            res_out = concatenate(to_concat, out=out, axis=axis, casting=casting)
            # 使用指定的 out.dtype 和 casting 参数连接数组，并断言返回的结果与 out 相等
            res_dtype = concatenate(
                to_concat, dtype=out.dtype, axis=axis, casting=casting
            )
            assert res_out is out
            assert_array_equal(out, res_dtype)
            assert res_dtype.dtype == out_dtype

        # 断言以下操作会引发 TypeError 异常，使用指定的 out 和 dtype 参数连接数组
        with assert_raises(TypeError):
            concatenate(to_concat, out=out, dtype=out_dtype, axis=axis)
# 使用装饰器实例化带参数化测试的测试类
@instantiate_parametrized_tests
class TestStackMisc(TestCase):
    
    # 跳过条件：当 NumPy 版本低于 "1.24" 时，测试将被跳过
    @skipif(numpy.__version__ < "1.24", reason="NP_VER: fails on NumPy 1.23.x")
    @skipif(numpy.__version__ < "1.24", reason="NP_VER: fails on NumPy 1.23.x")
    # 参数化测试：axis 参数设定为 0
    @parametrize("axis", [0])
    # 参数化测试：out_dtype 参数设定为 ["c8", "f4", "f8", "i8"]
    @parametrize("out_dtype", ["c8", "f4", "f8", "i8"])  # torch does not have ">f8",
    # 参数化测试：casting 参数设定为 ["no", "equiv", "safe", "same_kind", "unsafe"]
    @parametrize("casting", ["no", "equiv", "safe", "same_kind", "unsafe"])
    def test_stack_out_and_dtype(self, axis, out_dtype, casting):
        # 准备数据用于堆叠操作
        to_concat = (array([1, 2]), array([3, 4]))
        # 预期的堆叠结果
        res = array([[1, 2], [3, 4]])
        # 创建一个与 res 相同形状的全零数组
        out = np.zeros_like(res)

        # 如果无法将 to_concat[0] 转换为指定的 out_dtype 类型，预期会抛出 TypeError 异常
        if not np.can_cast(to_concat[0], out_dtype, casting=casting):
            with assert_raises(TypeError):
                # 调用 stack 函数并传入参数，期望会触发类型错误异常
                stack(to_concat, dtype=out_dtype, axis=axis, casting=casting)
        else:
            # 调用 stack 函数并传入参数，指定输出数组 out 和其他参数
            res_out = stack(to_concat, out=out, axis=axis, casting=casting)
            # 调用 stack 函数并传入参数，指定 dtype 参数
            res_dtype = stack(to_concat, dtype=out_dtype, axis=axis, casting=casting)
            # 断言 res_out 与 out 是同一个对象
            assert res_out is out
            # 断言 out 与 res_dtype 在值上相等
            assert_array_equal(out, res_dtype)
            # 断言 res_dtype 的 dtype 等于指定的 out_dtype
            assert res_dtype.dtype == out_dtype

        # 预期使用不支持的参数类型调用 stack 函数时会抛出 TypeError 异常
        with assert_raises(TypeError):
            stack(to_concat, out=out, dtype=out_dtype, axis=axis)


# 标记为预期失败的测试类，注释指出需要实现 block(...) 功能
@xfail  # (reason="TODO: implement block(...)")
@instantiate_parametrized_tests
class TestBlock(TestCase):
    
    # pytest.fixture 装饰的方法，参数化为 ["block", "force_concatenate", "force_slicing"]
    @pytest.fixture(params=["block", "force_concatenate", "force_slicing"])
    def block(self, request):
        # 定义在小数组和大数组上使用不同路径的算法
        # 根据所需的元素复制数量触发算法选择
        # 定义一个测试 fixture，强制大多数测试通过两种代码路径
        # 最终，如果发现单一算法对小数组和大数组都更快，则应删除此测试 fixture
        def _block_force_concatenate(arrays):
            arrays, list_ndim, result_ndim, _ = _block_setup(arrays)
            return _block_concatenate(arrays, list_ndim, result_ndim)

        def _block_force_slicing(arrays):
            arrays, list_ndim, result_ndim, _ = _block_setup(arrays)
            return _block_slicing(arrays, list_ndim, result_ndim)

        # 根据 request.param 返回相应的函数
        if request.param == "force_concatenate":
            return _block_force_concatenate
        elif request.param == "force_slicing":
            return _block_force_slicing
        elif request.param == "block":
            return block
        else:
            raise ValueError("Unknown blocking request. There is a typo in the tests.")

    # 测试方法：验证 block 方法返回的数组是复制的副本
    def test_returns_copy(self, block):
        # 创建一个 3x3 的单位矩阵 a
        a = np.eye(3)
        # 调用 block 方法处理数组 a，返回结果保存在 b 中
        b = block(a)
        # 修改 b 的第一个元素为 2
        b[0, 0] = 2
        # 断言修改后的 b 的第一个元素不等于原始 a 的第一个元素
        assert b[0, 0] != a[0, 0]
    # 测试函数：计算块的总大小估计
    def test_block_total_size_estimate(self, block):
        # 设置一个块，返回元组中只需要第四个元素即总大小
        _, _, _, total_size = _block_setup([1])
        # 断言总大小为1
        assert total_size == 1

        # 设置一个块，返回元组中只需要第四个元素即总大小
        _, _, _, total_size = _block_setup([[1]])
        # 断言总大小为1
        assert total_size == 1

        # 设置一个块，返回元组中只需要第四个元素即总大小
        _, _, _, total_size = _block_setup([[1, 1]])
        # 断言总大小为2
        assert total_size == 2

        # 设置一个块，返回元组中只需要第四个元素即总大小
        _, _, _, total_size = _block_setup([[1], [1]])
        # 断言总大小为2
        assert total_size == 2

        # 设置一个块，返回元组中只需要第四个元素即总大小
        _, _, _, total_size = _block_setup([[1, 2], [3, 4]])
        # 断言总大小为4
        assert total_size == 4

    # 测试函数：简单的行级块操作
    def test_block_simple_row_wise(self, block):
        # 创建一个2x2的全1矩阵
        a_2d = np.ones((2, 2))
        # 创建一个2x2的全2矩阵
        b_2d = 2 * a_2d
        # 期望的结果矩阵，将a_2d和b_2d按行级块操作
        desired = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
        # 调用块函数进行操作
        result = block([a_2d, b_2d])
        # 断言结果与期望相同
        assert_equal(desired, result)

    # 测试函数：简单的列级块操作
    def test_block_simple_column_wise(self, block):
        # 创建一个2x2的全1矩阵
        a_2d = np.ones((2, 2))
        # 创建一个2x2的全2矩阵
        b_2d = 2 * a_2d
        # 期望的结果矩阵，将a_2d和b_2d按列级块操作
        expected = np.array([[1, 1], [1, 1], [2, 2], [2, 2]])
        # 调用块函数进行操作
        result = block([[a_2d], [b_2d]])
        # 断言结果与期望相同
        assert_equal(expected, result)

    # 测试函数：使用1维数组进行行级块操作
    def test_block_with_1d_arrays_row_wise(self, block):
        # 创建两个1维数组
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        # 期望的结果数组，将a和b按行级块操作
        expected = np.array([1, 2, 3, 2, 3, 4])
        # 调用块函数进行操作
        result = block([a, b])
        # 断言结果与期望相同
        assert_equal(expected, result)

    # 测试函数：使用1维数组进行多行级块操作
    def test_block_with_1d_arrays_multiple_rows(self, block):
        # 创建两个1维数组
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        # 期望的结果数组，将[a, b]和[a, b]按行级块操作
        expected = np.array([[1, 2, 3, 2, 3, 4], [1, 2, 3, 2, 3, 4]])
        # 调用块函数进行操作
        result = block([[a, b], [a, b]])
        # 断言结果与期望相同
        assert_equal(expected, result)

    # 测试函数：使用1维数组进行列级块操作
    def test_block_with_1d_arrays_column_wise(self, block):
        # 创建两个1维数组
        a_1d = np.array([1, 2, 3])
        b_1d = np.array([2, 3, 4])
        # 期望的结果数组，将[a_1d]和[b_1d]按列级块操作
        expected = np.array([[1, 2, 3], [2, 3, 4]])
        # 调用块函数进行操作
        result = block([[a_1d], [b_1d]])
        # 断言结果与期望相同
        assert_equal(expected, result)

    # 测试函数：混合1维和2维数组进行块操作
    def test_block_mixed_1d_and_2d(self, block):
        # 创建一个2x2的全1矩阵
        a_2d = np.ones((2, 2))
        # 创建一个1维数组
        b_1d = np.array([2, 2])
        # 期望的结果数组，将[a_2d]和[b_1d]进行块操作
        expected = np.array([[1, 1], [1, 1], [2, 2]])
        # 调用块函数进行操作
        result = block([[a_2d], [b_1d]])
        # 断言结果与期望相同
        assert_equal(expected, result)

    # 测试函数：更复杂的块操作
    def test_block_complicated(self, block):
        # 更复杂的操作
        one_2d = np.array([[1, 1, 1]])
        two_2d = np.array([[2, 2, 2]])
        three_2d = np.array([[3, 3, 3, 3, 3, 3]])
        four_1d = np.array([4, 4, 4, 4, 4, 4])
        five_0d = np.array(5)
        six_1d = np.array([6, 6, 6, 6, 6])
        zero_2d = np.zeros((2, 6))

        # 期望的结果矩阵
        expected = np.array(
            [
                [1, 1, 1, 2, 2, 2],
                [3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4],
                [5, 6, 6, 6, 6, 6],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        # 调用块函数进行更复杂的操作
        result = block(
            [[one_2d, two_2d], [three_2d], [four_1d], [five_0d, six_1d], [zero_2d]]
        )
        # 断言结果与期望相同
        assert_equal(result, expected)
    # 定义一个测试方法，用于测试嵌套的数组操作
    def test_nested(self, block):
        # 创建不同形状的 NumPy 数组
        one = np.array([1, 1, 1])
        two = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        three = np.array([3, 3, 3])
        four = np.array([4, 4, 4])
        five = np.array(5)
        six = np.array([6, 6, 6, 6, 6])
        zero = np.zeros((2, 6))

        # 调用给定的 block 函数，进行多层嵌套的数组操作
        result = block([[block([[one], [three], [four]]), two], [five, six], [zero]])
        
        # 预期的结果数组
        expected = np.array(
            [
                [1, 1, 1, 2, 2, 2],
                [3, 3, 3, 2, 2, 2],
                [4, 4, 4, 2, 2, 2],
                [5, 6, 6, 6, 6, 6],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        # 断言结果与预期相等
        assert_equal(result, expected)

    # 定义一个测试方法，用于测试三维数组的操作
    def test_3d(self, block):
        # 创建不同形状的三维 NumPy 数组
        a000 = np.ones((2, 2, 2), int) * 1
        a100 = np.ones((3, 2, 2), int) * 2
        a010 = np.ones((2, 3, 2), int) * 3
        a001 = np.ones((2, 2, 3), int) * 4
        a011 = np.ones((2, 3, 3), int) * 5
        a101 = np.ones((3, 2, 3), int) * 6
        a110 = np.ones((3, 3, 2), int) * 7
        a111 = np.ones((3, 3, 3), int) * 8

        # 调用给定的 block 函数，进行三维数组的操作
        result = block(
            [
                [
                    [a000, a001],
                    [a010, a011],
                ],
                [
                    [a100, a101],
                    [a110, a111],
                ],
            ]
        )

        # 预期的结果数组
        expected = np.array(
            [
                [
                    [1, 1, 4, 4, 4],
                    [1, 1, 4, 4, 4],
                    [3, 3, 5, 5, 5],
                    [3, 3, 5, 5, 5],
                    [3, 3, 5, 5, 5],
                ],
                [
                    [1, 1, 4, 4, 4],
                    [1, 1, 4, 4, 4],
                    [3, 3, 5, 5, 5],
                    [3, 3, 5, 5, 5],
                    [3, 3, 5, 5, 5],
                ],
                [
                    [2, 2, 6, 6, 6],
                    [2, 2, 6, 6, 6],
                    [7, 7, 8, 8, 8],
                    [7, 7, 8, 8, 8],
                    [7, 7, 8, 8, 8],
                ],
                [
                    [2, 2, 6, 6, 6],
                    [2, 2, 6, 6, 6],
                    [7, 7, 8, 8, 8],
                    [7, 7, 8, 8, 8],
                    [7, 7, 8, 8, 8],
                ],
                [
                    [2, 2, 6, 6, 6],
                    [2, 2, 6, 6, 6],
                    [7, 7, 8, 8, 8],
                    [7, 7, 8, 8, 8],
                    [7, 7, 8, 8, 8],
                ],
            ]
        )

        # 断言结果数组与预期相等
        assert_array_equal(result, expected)

    # 定义一个测试方法，用于测试不匹配形状的数组操作
    def test_block_with_mismatched_shape(self, block):
        # 创建两个不匹配形状的数组 a 和 b，预期抛出 ValueError 异常
        a = np.array([0, 0])
        b = np.eye(2)
        assert_raises(ValueError, block, [a, b])
        assert_raises(ValueError, block, [b, a])

        # 创建一个不匹配形状的数组列表 to_block，预期抛出 ValueError 异常
        to_block = [
            [np.ones((2, 3)), np.ones((2, 2))],
            [np.ones((2, 2)), np.ones((2, 2))],
        ]
        assert_raises(ValueError, block, to_block)
    # 测试函数，验证当输入参数不包含列表时的行为
    def test_no_lists(self, block):
        # 断言函数调用返回值与预期值相等
        assert_equal(block(1), np.array(1))
        # 断言函数调用返回值与预期值相等
        assert_equal(block(np.eye(3)), np.eye(3))

    # 测试函数，验证当输入参数的嵌套结构不合法时的行为
    def test_invalid_nesting(self, block):
        # 预期抛出 ValueError 异常，异常信息包含特定消息
        msg = "depths are mismatched"
        assert_raises_regex(ValueError, msg, block, [1, [2]])
        assert_raises_regex(ValueError, msg, block, [1, []])
        assert_raises_regex(ValueError, msg, block, [[1], 2])
        assert_raises_regex(ValueError, msg, block, [[], 2])
        assert_raises_regex(
            ValueError, msg, block, [[[1], [2]], [[3, 4]], [5]]  # 缺少括号
        )

    # 测试函数，验证当输入参数为空列表时的行为
    def test_empty_lists(self, block):
        # 预期抛出 ValueError 异常，异常信息包含特定消息
        assert_raises_regex(ValueError, "empty", block, [])
        assert_raises_regex(ValueError, "empty", block, [[]])
        assert_raises_regex(ValueError, "empty", block, [[1], []])

    # 测试函数，验证当输入参数包含元组时的行为
    def test_tuple(self, block):
        # 预期抛出 TypeError 异常，异常信息包含特定消息
        assert_raises_regex(TypeError, "tuple", block, ([1, 2], [3, 4]))
        assert_raises_regex(TypeError, "tuple", block, [(1, 2), (3, 4)])

    # 测试函数，验证当输入参数包含不同维度的数组时的行为
    def test_different_ndims(self, block):
        # 创建不同维度的 NumPy 数组
        a = 1.0
        b = 2 * np.ones((1, 2))
        c = 3 * np.ones((1, 1, 3))

        # 调用被测试的函数并获取结果
        result = block([a, b, c])
        # 预期的结果
        expected = np.array([[[1.0, 2.0, 2.0, 3.0, 3.0, 3.0]]])

        # 断言函数调用返回值与预期值相等
        assert_equal(result, expected)

    # 测试函数，验证当输入参数包含深度不同的多维数组时的行为
    def test_different_ndims_depths(self, block):
        # 创建不同维度和深度的 NumPy 数组
        a = 1.0
        b = 2 * np.ones((1, 2))
        c = 3 * np.ones((1, 2, 3))

        # 调用被测试的函数并获取结果
        result = block([[a, b], [c]])
        # 预期的结果
        expected = np.array([[[1.0, 2.0, 2.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]])

        # 断言函数调用返回值与预期值相等
        assert_equal(result, expected)

    # 测试函数，验证当输入参数为内存顺序不同的多维数组时的行为
    def test_block_memory_order(self, block):
        # 创建 C 和 Fortran 内存顺序的 3D 数组
        arr_c = np.zeros((3,) * 3, order="C")
        arr_f = np.zeros((3,) * 3, order="F")

        # 创建包含不同内存顺序数组的多维列表
        b_c = [[[arr_c, arr_c], [arr_c, arr_c]], [[arr_c, arr_c], [arr_c, arr_c]]]
        b_f = [[[arr_f, arr_f], [arr_f, arr_f]], [[arr_f, arr_f], [arr_f, arr_f]]]

        # 断言函数调用返回的数组是否为 C 连续的
        assert block(b_c).flags["C_CONTIGUOUS"]
        # 断言函数调用返回的数组是否为 Fortran 连续的
        assert block(b_f).flags["F_CONTIGUOUS"]

        # 创建 C 和 Fortran 内存顺序的 2D 数组
        arr_c = np.zeros((3, 3), order="C")
        arr_f = np.zeros((3, 3), order="F")

        # 创建包含不同内存顺序数组的二维列表
        b_c = [[arr_c, arr_c], [arr_c, arr_c]]
        b_f = [[arr_f, arr_f], [arr_f, arr_f]]

        # 断言函数调用返回的数组是否为 C 连续的
        assert block(b_c).flags["C_CONTIGUOUS"]
        # 断言函数调用返回的数组是否为 Fortran 连续的
        assert block(b_f).flags["F_CONTIGUOUS"]
# 如果当前脚本被直接执行（而非被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 调用运行测试函数
    run_tests()
```