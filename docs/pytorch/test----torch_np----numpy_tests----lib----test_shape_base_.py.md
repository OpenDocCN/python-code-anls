# `.\pytorch\test\torch_np\numpy_tests\lib\test_shape_base_.py`

```
# Owner(s): ["module: dynamo"]

# 导入必要的模块和函数
import functools  # 导入 functools 模块
import sys  # 导入 sys 模块

# 导入 pytest 中的特定函数并重命名为更简短的别名
from unittest import expectedFailure as xfail, skipIf as skipif
# 导入 pytest 中的异常断言函数并重命名
from pytest import raises as assert_raises

# 根据条件导入不同的模块和函数
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入测试相关的函数
    parametrize,  # 导入参数化测试相关函数
    run_tests,  # 导入运行测试的函数
    TEST_WITH_TORCHDYNAMO,  # 导入测试标志
    TestCase,  # 导入测试用例类
    xfailIfTorchDynamo,  # 导入与 Torch Dynamo 相关的标记函数
    xpassIfTorchDynamo,  # 导入与 Torch Dynamo 相关的标记函数
)

# 根据测试标志选择导入 NumPy 或 Torch NumPy 的不同部分
if TEST_WITH_TORCHDYNAMO:
    import numpy as np  # 导入 NumPy 库
    from numpy import (  # 导入 NumPy 的多个函数
        apply_along_axis,
        array_split,
        column_stack,
        dsplit,
        dstack,
        expand_dims,
        hsplit,
        kron,
        put_along_axis,
        split,
        take_along_axis,
        tile,
        vsplit,
    )
    from numpy.random import rand, randint  # 导入 NumPy 的随机数函数

    from numpy.testing import assert_, assert_array_equal, assert_equal  # 导入 NumPy 测试断言函数

else:
    import torch._numpy as np  # 导入 Torch NumPy 库
    from torch._numpy import (  # 导入 Torch NumPy 的多个函数
        array_split,
        column_stack,
        dsplit,
        dstack,
        expand_dims,
        hsplit,
        kron,
        put_along_axis,
        split,
        take_along_axis,
        tile,
        vsplit,
    )
    from torch._numpy.random import rand, randint  # 导入 Torch NumPy 的随机数函数
    from torch._numpy.testing import assert_, assert_array_equal, assert_equal  # 导入 Torch NumPy 测试断言函数


# 定义 skip 函数，用于在特定条件下跳过测试
skip = functools.partial(skipif, True)

# 检查系统是否为 64 位
IS_64BIT = sys.maxsize > 2**32


def _add_keepdims(func):
    """hack in keepdims behavior into a function taking an axis"""

    @functools.wraps(func)
    def wrapped(a, axis, **kwargs):
        # 调用 func 函数，并在返回结果上应用 keepdims 行为
        res = func(a, axis=axis, **kwargs)
        if axis is None:
            axis = 0  # 如果返回结果是标量，则在轴 0 上扩展维度
        return np.expand_dims(res, axis=axis)

    return wrapped


class TestTakeAlongAxis(TestCase):
    def test_argequivalent(self):
        """Test it translates from arg<func> to <func>"""
        a = rand(3, 4, 5)  # 创建一个随机数组

        funcs = [
            (np.sort, np.argsort, dict()),  # 对 sort 和 argsort 函数进行测试
            (_add_keepdims(np.min), _add_keepdims(np.argmin), dict()),  # 对 min 和 argmin 函数进行测试
            (_add_keepdims(np.max), _add_keepdims(np.argmax), dict()),  # 对 max 和 argmax 函数进行测试
            #  FIXME           (np.partition, np.argpartition, dict(kth=2)),  # 对 partition 和 argpartition 函数进行测试
        ]

        for func, argfunc, kwargs in funcs:
            for axis in list(range(a.ndim)) + [None]:
                # 调用 func 和 argfunc 函数，比较其结果是否相等
                a_func = func(a, axis=axis, **kwargs)
                ai_func = argfunc(a, axis=axis, **kwargs)
                assert_equal(a_func, take_along_axis(a, ai_func, axis=axis))  # 使用断言验证结果是否相等
    # 定义一个测试方法，用于验证当索引维度过少时是否会出错
    def test_invalid(self):
        """Test it errors when indices has too few dimensions"""
        # 创建一个形状为 (10, 10) 的全 1 数组
        a = np.ones((10, 10))
        # 创建一个形状为 (10, 2) 的全 1 数组，数据类型为 np.intp
        ai = np.ones((10, 2), dtype=np.intp)

        # 执行简单的检查以确保函数调用没有问题
        take_along_axis(a, ai, axis=1)

        # 测试索引维度过少时是否会抛出异常
        assert_raises(
            (ValueError, RuntimeError), take_along_axis, a, np.array(1), axis=1
        )
        # 测试布尔数组作为索引时是否会抛出异常
        assert_raises(
            (IndexError, RuntimeError), take_along_axis, a, ai.astype(bool), axis=1
        )
        # 测试浮点数组作为索引时是否会抛出异常
        assert_raises(
            (IndexError, RuntimeError), take_along_axis, a, ai.astype(float), axis=1
        )
        # 测试无效的轴索引是否会抛出 AxisError 异常
        assert_raises(np.AxisError, take_along_axis, a, ai, axis=10)

    # 定义一个测试方法，验证即使结果为空，也可以正常运行，即使插入了维度
    def test_empty(self):
        """Test everything is ok with empty results, even with inserted dims"""
        # 创建一个形状为 (3, 4, 5) 的全 1 数组
        a = np.ones((3, 4, 5))
        # 创建一个形状为 (3, 0, 5) 的全 1 数组，数据类型为 np.intp
        ai = np.ones((3, 0, 5), dtype=np.intp)

        # 调用 take_along_axis 函数，测试即使结果为空，也可以正常运行
        actual = take_along_axis(a, ai, axis=1)
        # 断言返回结果的形状与索引数组 ai 的形状相同
        assert_equal(actual.shape, ai.shape)

    # 定义一个测试方法，验证在非索引维度中进行广播时是否正常工作
    def test_broadcast(self):
        """Test that non-indexing dimensions are broadcast in both directions"""
        # 创建一个形状为 (3, 4, 1) 的全 1 数组
        a = np.ones((3, 4, 1))
        # 创建一个形状为 (1, 2, 5) 的全 1 数组，数据类型为 np.intp
        ai = np.ones((1, 2, 5), dtype=np.intp)
        
        # 调用 take_along_axis 函数，验证在非索引维度中进行广播是否正常工作
        actual = take_along_axis(a, ai, axis=1)
        # 断言返回结果的形状应为 (3, 2, 5)
        assert_equal(actual.shape, (3, 2, 5))
class TestPutAlongAxis(TestCase):
    def test_replace_max(self):
        # 创建基础数组 a_base，包含两个子数组
        a_base = np.array([[10, 30, 20], [60, 40, 50]])

        # 对于每个维度和 None（全局操作），循环执行以下操作
        for axis in list(range(a_base.ndim)) + [None]:
            # 复制 a_base 到 a，用于循环中的修改
            a = a_base.copy()

            # 找到最大值的索引并保持维度
            i_max = _add_keepdims(np.argmax)(a, axis=axis)
            # 在数组 a 中的 i_max 处放置新值 -99
            put_along_axis(a, i_max, -99, axis=axis)

            # 找到新的最小值索引，应当与之前找到的最大值索引相同
            i_min = _add_keepdims(np.argmin)(a, axis=axis)

            # 断言新的最小值索引与之前的最大值索引相同
            assert_equal(i_min, i_max)

    @xpassIfTorchDynamo  # (reason="RuntimeError: Expected index [1, 2, 5] to be smaller than self [3, 4, 1] apart from dimension 1")
    def test_broadcast(self):
        """测试非索引维度在两个方向上进行广播"""
        # 创建全为 1 的数组 a，形状为 (3, 4, 1)
        a = np.ones((3, 4, 1))
        # 创建索引数组 ai，形状为 (1, 2, 5)，取余数限制在 0 到 3 之间
        ai = np.arange(10, dtype=np.intp).reshape((1, 2, 5)) % 4
        # 在 axis=1 的维度上沿着 ai 放置值 20
        put_along_axis(a, ai, 20, axis=1)
        # 断言沿着 axis=1 的维度上取出的值与 20 相等
        assert_equal(take_along_axis(a, ai, axis=1), 20)


@xpassIfTorchDynamo  # (reason="apply_along_axis not implemented")
class TestApplyAlongAxis(TestCase):
    def test_simple(self):
        # 创建全为 1 的二维数组 a，形状为 (20, 10)
        a = np.ones((20, 10), "d")
        # 断言应用函数 len 沿着 axis=0 对 a 应用后的结果与预期一致
        assert_array_equal(apply_along_axis(len, 0, a), len(a) * np.ones(a.shape[1]))

    def test_simple101(self):
        # 创建全为 1 的二维数组 a，形状为 (10, 101)
        a = np.ones((10, 101), "d")
        # 断言应用函数 len 沿着 axis=0 对 a 应用后的结果与预期一致
        assert_array_equal(apply_along_axis(len, 0, a), len(a) * np.ones(a.shape[1]))

    def test_3d(self):
        # 创建一个三维数组 a，形状为 (3, 3, 3)，包含从 0 到 26 的整数
        a = np.arange(27).reshape((3, 3, 3))
        # 断言应用函数 np.sum 沿着 axis=0 对 a 应用后的结果与预期一致
        assert_array_equal(
            apply_along_axis(np.sum, 0, a), [[27, 30, 33], [36, 39, 42], [45, 48, 51]]
        )

    def test_scalar_array(self, cls=np.ndarray):
        # 创建一个二维数组 a，形状为 (6, 3)，元素为全为 1 的数组，视图类型为 cls
        a = np.ones((6, 3)).view(cls)
        # 应用函数 np.sum 沿着 axis=0 对 a 应用，结果类型为 cls
        res = apply_along_axis(np.sum, 0, a)
        # 断言结果 res 的类型为 cls
        assert_(isinstance(res, cls))
        # 断言结果 res 与预期结果 np.array([6, 6, 6]) 相等，视图类型为 cls
        assert_array_equal(res, np.array([6, 6, 6]).view(cls))

    def test_0d_array(self, cls=np.ndarray):
        def sum_to_0d(x):
            """对 x 求和，返回相同类型的零维数组"""
            assert_equal(x.ndim, 1)
            return np.squeeze(np.sum(x, keepdims=True))

        # 创建一个二维数组 a，形状为 (6, 3)，元素为全为 1 的数组，视图类型为 cls
        a = np.ones((6, 3)).view(cls)
        # 应用函数 sum_to_0d 沿着 axis=0 对 a 应用，结果类型为 cls
        res = apply_along_axis(sum_to_0d, 0, a)
        # 断言结果 res 的类型为 cls
        assert_(isinstance(res, cls))
        # 断言结果 res 与预期结果 np.array([6, 6, 6]) 相等，视图类型为 cls
        assert_array_equal(res, np.array([6, 6, 6]).view(cls))

        # 再次应用函数 sum_to_0d 沿着 axis=1 对 a 应用，结果类型为 cls
        res = apply_along_axis(sum_to_0d, 1, a)
        # 断言结果 res 的类型为 cls
        assert_(isinstance(res, cls))
        # 断言结果 res 与预期结果 np.array([3, 3, 3, 3, 3, 3]) 相等，视图类型为 cls
        assert_array_equal(res, np.array([3, 3, 3, 3, 3, 3]).view(cls))
    def test_axis_insertion(self, cls=np.ndarray):
        # 定义一个将一维数组转换为非对称非方阵的函数
        def f1to2(x):
            """produces an asymmetric non-square matrix from x"""
            # 断言输入数组是一维的
            assert_equal(x.ndim, 1)
            # 返回一个视图对象，该对象是输入数组的变形结果
            return (x[::-1] * x[1:, None]).view(cls)

        # 创建一个 6x3 的二维数组
        a2d = np.arange(6 * 3).reshape((6, 3))

        # 在第一个轴上进行二维插入
        actual = apply_along_axis(f1to2, 0, a2d)
        # 期望结果是沿着最后一个轴堆叠每一列处理后的结果
        expected = np.stack(
            [f1to2(a2d[:, i]) for i in range(a2d.shape[1])], axis=-1
        ).view(cls)
        # 断言实际结果和期望结果的类型相同
        assert_equal(type(actual), type(expected))
        # 断言实际结果和期望结果相等
        assert_equal(actual, expected)

        # 在最后一个轴上进行二维插入
        actual = apply_along_axis(f1to2, 1, a2d)
        # 期望结果是沿着第一个轴堆叠每一行处理后的结果
        expected = np.stack(
            [f1to2(a2d[i, :]) for i in range(a2d.shape[0])], axis=0
        ).view(cls)
        # 断言实际结果和期望结果的类型相同
        assert_equal(type(actual), type(expected))
        # 断言实际结果和期望结果相等
        assert_equal(actual, expected)

        # 在中间轴上进行三维插入
        a3d = np.arange(6 * 5 * 3).reshape((6, 5, 3))

        actual = apply_along_axis(f1to2, 1, a3d)
        # 期望结果是沿着最后一个轴堆叠每个深度层处理后的结果
        expected = np.stack(
            [
                np.stack([f1to2(a3d[i, :, j]) for i in range(a3d.shape[0])], axis=0)
                for j in range(a3d.shape[2])
            ],
            axis=-1,
        ).view(cls)
        # 断言实际结果和期望结果的类型相同
        assert_equal(type(actual), type(expected))
        # 断言实际结果和期望结果相等
        assert_equal(actual, expected)

    def test_axis_insertion_ma(self):
        # 定义一个将一维数组转换为非对称非方阵的函数，并使用掩码数组
        def f1to2(x):
            """produces an asymmetric non-square matrix from x"""
            # 断言输入数组是一维的
            assert_equal(x.ndim, 1)
            # 计算结果数组，然后根据条件创建掩码数组
            res = x[::-1] * x[1:, None]
            return np.ma.masked_where(res % 5 == 0, res)

        # 创建一个 6x3 的二维数组
        a = np.arange(6 * 3).reshape((6, 3))
        # 应用函数 f1to2 到数组 a 的每个轴上
        res = apply_along_axis(f1to2, 0, a)
        # 断言返回结果是掩码数组的实例
        assert_(isinstance(res, np.ma.masked_array))
        # 断言返回结果的维度为 3
        assert_equal(res.ndim, 3)
        # 断言每个深度层的掩码数组与对应列的处理结果的掩码数组相等
        assert_array_equal(res[:, :, 0].mask, f1to2(a[:, 0]).mask)
        assert_array_equal(res[:, :, 1].mask, f1to2(a[:, 1]).mask)
        assert_array_equal(res[:, :, 2].mask, f1to2(a[:, 2]).mask)

    def test_tuple_func1d(self):
        # 定义一个简单的一维数组转换函数
        def sample_1d(x):
            return x[1], x[0]

        # 对二维数组中的每一行应用函数 sample_1d
        res = np.apply_along_axis(sample_1d, 1, np.array([[1, 2], [3, 4]]))
        # 断言返回结果与期望结果相等
        assert_array_equal(res, np.array([[2, 1], [4, 3]]))

    def test_empty(self):
        # 当无法调用函数时，不能对空数组应用 apply_along_axis
        def never_call(x):
            assert_(False)  # should never be reached

        # 创建一个空的 0x0 数组
        a = np.empty((0, 0))
        # 断言对空数组在第 0 轴和第 1 轴应用 apply_along_axis 会抛出 ValueError
        assert_raises(ValueError, np.apply_along_axis, never_call, 0, a)
        assert_raises(ValueError, np.apply_along_axis, never_call, 1, a)

        # 当某些维度不为零时，有时对空数组应用 apply_along_axis 是可以接受的
        def empty_to_1(x):
            assert_(len(x) == 0)
            return 1

        # 创建一个 10x0 的空数组
        a = np.empty((10, 0))
        # 在第 1 轴上应用函数 empty_to_1
        actual = np.apply_along_axis(empty_to_1, 1, a)
        # 断言实际结果与预期结果相等
        assert_equal(actual, np.ones(10))
        # 断言对空数组在第 0 轴应用 apply_along_axis 会抛出 ValueError
        assert_raises(ValueError, np.apply_along_axis, empty_to_1, 0, a)
    @skip  # 被注释的装饰器，用于跳过测试，因为 TypeError: descriptor 'union' for 'set' objects doesn't apply to a 'numpy.int64' object

    # 创建一个 NumPy 数组，其中包含两个子数组，每个子数组包含三个集合
    d = np.array([[{1, 11}, {2, 22}, {3, 33}], [{4, 44}, {5, 55}, {6, 66}]])
    
    # 对 d 数组的第一维应用 lambda 函数，该函数对每列的集合执行 set.union 操作
    actual = np.apply_along_axis(lambda a: set.union(*a), 0, d)
    
    # 期望的结果，是一个 NumPy 数组，其中每个元素是合并相应列集合的结果
    expected = np.array([{1, 11, 4, 44}, {2, 22, 5, 55}, {3, 33, 6, 66}])

    # 使用 assert_equal 函数断言 actual 和 expected 数组相等
    assert_equal(actual, expected)

    # 针对每个元素的索引进行循环，检查 actual 和 expected 数组的元素类型是否一致
    for i in np.ndindex(actual.shape):
        assert_equal(type(actual[i]), type(expected[i]))
@xfail  # (reason="apply_over_axes not implemented")
class TestApplyOverAxes(TestCase):
    # 测试 apply_over_axes 函数的功能
    def test_simple(self):
        # 创建一个 3D 数组 a，形状为 (2, 3, 4)
        a = np.arange(24).reshape(2, 3, 4)
        # 对数组 a 应用 apply_over_axes 函数，对指定的轴进行求和操作
        aoa_a = apply_over_axes(np.sum, a, [0, 2])
        # 断言 aoa_a 的结果与给定的数组相等
        assert_array_equal(aoa_a, np.array([[[60], [92], [124]]]))


class TestExpandDims(TestCase):
    # 测试 expand_dims 函数的基本功能
    def test_functionality(self):
        # 定义一个形状为 (2, 3, 4, 5) 的空数组 a
        s = (2, 3, 4, 5)
        a = np.empty(s)
        # 对数组 a 的每个可能的轴进行扩展维度的测试
        for axis in range(-5, 4):
            b = expand_dims(a, axis)
            # 断言扩展后的数组 b 在指定轴的维度为 1
            assert_(b.shape[axis] == 1)
            # 断言压缩扩展后的数组 b 的形状与原始形状 s 相同
            assert_(np.squeeze(b).shape == s)

    # 测试在指定轴为元组时的 expand_dims 函数的功能
    def test_axis_tuple(self):
        a = np.empty((3, 3, 3))
        # 对于元组中的不同轴，测试 expand_dims 的结果形状是否符合预期
        assert np.expand_dims(a, axis=(0, 1, 2)).shape == (1, 1, 1, 3, 3, 3)
        assert np.expand_dims(a, axis=(0, -1, -2)).shape == (1, 3, 3, 3, 1, 1)
        assert np.expand_dims(a, axis=(0, 3, 5)).shape == (1, 3, 3, 1, 3, 1)
        assert np.expand_dims(a, axis=(0, -3, -5)).shape == (1, 1, 3, 1, 3, 3)

    # 测试超出范围的轴索引时 expand_dims 函数是否正确抛出异常
    def test_axis_out_of_range(self):
        s = (2, 3, 4, 5)
        a = np.empty(s)
        # 断言在超出范围的轴索引下，expand_dims 函数会抛出 AxisError 异常
        assert_raises(np.AxisError, expand_dims, a, -6)
        assert_raises(np.AxisError, expand_dims, a, 5)

        a = np.empty((3, 3, 3))
        # 断言在元组中包含超出范围的轴索引时，expand_dims 函数会抛出 AxisError 异常
        assert_raises(np.AxisError, expand_dims, a, (0, -6))
        assert_raises(np.AxisError, expand_dims, a, (0, 5))

    # 测试重复的轴索引时 expand_dims 函数是否正确抛出异常
    def test_repeated_axis(self):
        a = np.empty((3, 3, 3))
        # 断言在重复的轴索引下，expand_dims 函数会抛出 ValueError 异常
        assert_raises(ValueError, expand_dims, a, axis=(1, 1))


class TestArraySplit(TestCase):
    # 测试当 split 数为 0 时，array_split 函数是否会抛出异常
    def test_integer_0_split(self):
        a = np.arange(10)
        # 断言在 split 数为 0 时，array_split 函数会抛出 ValueError 异常
        assert_raises(ValueError, array_split, a, 0)
    # 定义一个测试函数，用于测试数组拆分的功能
    def test_integer_split(self):
        # 创建一个长度为 10 的 NumPy 数组 a，其中包含从 0 到 9 的整数
        a = np.arange(10)
        # 使用 array_split 函数将数组 a 拆分成 1 个子数组
        res = array_split(a, 1)
        # 期望的拆分结果，包含一个包含数组 a 所有元素的数组
        desired = [np.arange(10)]
        # 调用比较函数，比较实际结果 res 和期望结果 desired
        compare_results(res, desired)

        # 使用 array_split 函数将数组 a 拆分成 2 个子数组
        res = array_split(a, 2)
        # 期望的拆分结果，包含两个子数组，分别是 [0, 1, 2, 3, 4] 和 [5, 6, 7, 8, 9]
        desired = [np.arange(5), np.arange(5, 10)]
        compare_results(res, desired)

        # 使用 array_split 函数将数组 a 拆分成 3 个子数组
        res = array_split(a, 3)
        # 期望的拆分结果，包含三个子数组，分别是 [0, 1, 2, 3], [4, 5, 6], [7, 8, 9]
        desired = [np.arange(4), np.arange(4, 7), np.arange(7, 10)]
        compare_results(res, desired)

        # 使用 array_split 函数将数组 a 拆分成 4 个子数组
        res = array_split(a, 4)
        # 期望的拆分结果，包含四个子数组，分别是 [0, 1, 2], [3, 4, 5], [6, 7], [8, 9]
        desired = [np.arange(3), np.arange(3, 6), np.arange(6, 8), np.arange(8, 10)]
        compare_results(res, desired)

        # 使用 array_split 函数将数组 a 拆分成 5 个子数组
        res = array_split(a, 5)
        # 期望的拆分结果，包含五个子数组，分别是 [0, 1], [2, 3], [4, 5], [6, 7], [8, 9]
        desired = [
            np.arange(2),
            np.arange(2, 4),
            np.arange(4, 6),
            np.arange(6, 8),
            np.arange(8, 10),
        ]
        compare_results(res, desired)

        # 使用 array_split 函数将数组 a 拆分成 6 个子数组
        res = array_split(a, 6)
        # 期望的拆分结果，包含六个子数组，分别是 [0, 1], [2, 3], [4, 5], [6, 7], [8], [9]
        desired = [
            np.arange(2),
            np.arange(2, 4),
            np.arange(4, 6),
            np.arange(6, 8),
            np.arange(8, 9),
            np.arange(9, 10),
        ]
        compare_results(res, desired)

        # 使用 array_split 函数将数组 a 拆分成 7 个子数组
        res = array_split(a, 7)
        # 期望的拆分结果，包含七个子数组，分别是 [0, 1], [2, 3], [4, 5], [6], [7], [8], [9]
        desired = [
            np.arange(2),
            np.arange(2, 4),
            np.arange(4, 6),
            np.arange(6, 7),
            np.arange(7, 8),
            np.arange(8, 9),
            np.arange(9, 10),
        ]
        compare_results(res, desired)

        # 使用 array_split 函数将数组 a 拆分成 8 个子数组
        res = array_split(a, 8)
        # 期望的拆分结果，包含八个子数组，分别是 [0, 1], [2, 3], [4], [5], [6], [7], [8], [9]
        desired = [
            np.arange(2),
            np.arange(2, 4),
            np.arange(4, 5),
            np.arange(5, 6),
            np.arange(6, 7),
            np.arange(7, 8),
            np.arange(8, 9),
            np.arange(9, 10),
        ]
        compare_results(res, desired)

        # 使用 array_split 函数将数组 a 拆分成 9 个子数组
        res = array_split(a, 9)
        # 期望的拆分结果，包含九个子数组，分别是 [0, 1], [2], [3], [4], [5], [6], [7], [8], [9]
        desired = [
            np.arange(2),
            np.arange(2, 3),
            np.arange(3, 4),
            np.arange(4, 5),
            np.arange(5, 6),
            np.arange(6, 7),
            np.arange(7, 8),
            np.arange(8, 9),
            np.arange(9, 10),
        ]
        compare_results(res, desired)

        # 使用 array_split 函数将数组 a 拆分成 10 个子数组
        res = array_split(a, 10)
        # 期望的拆分结果，包含十个子数组，分别是 [0], [1], [2], [3], [4], [5], [6], [7], [8], [9]
        desired = [
            np.arange(1),
            np.arange(1, 2),
            np.arange(2, 3),
            np.arange(3, 4),
            np.arange(4, 5),
            np.arange(5, 6),
            np.arange(6, 7),
            np.arange(7, 8),
            np.arange(8, 9),
            np.arange(9, 10),
        ]
        compare_results(res, desired)

        # 使用 array_split 函数将数组 a 拆分成 11 个子数组
        res = array_split(a, 11)
        # 期望的拆分结果，包含十一个子数组，分别是 [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], []
        desired = [
            np.arange(1),
            np.arange(1, 2),
            np.arange(2, 3),
            np.arange(3, 4),
            np.arange(4, 5),
            np.arange(5, 6),
            np.arange(6, 7),
            np.arange(7, 8),
            np.arange(8, 9),
            np.arange(9, 10),
            np.array([]),
        ]
        compare_results(res, desired)
    # 测试函数：将二维数组按行分割成多个子数组
    def test_integer_split_2D_rows(self):
        # 创建一个二维数组 a，包含两个相同的行向量
        a = np.array([np.arange(10), np.arange(10)])
        # 将数组 a 按行分割成 3 个子数组，存储在 res 中
        res = array_split(a, 3, axis=0)
        # 目标结果 tgt，包含一个行向量和两个空行向量
        tgt = [np.array([np.arange(10)]), np.array([np.arange(10)]), np.zeros((0, 10))]
        # 比较 res 和 tgt 的结果
        compare_results(res, tgt)
        # 断言最后一个子数组的数据类型与原始数组 a 的数据类型相同
        assert_(a.dtype.type is res[-1].dtype.type)

        # 对手动指定分割点的情况进行相同操作：
        res = array_split(a, [0, 1], axis=0)
        tgt = [np.zeros((0, 10)), np.array([np.arange(10)]), np.array([np.arange(10)])]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)

    # 测试函数：将二维数组按列分割成多个子数组
    def test_integer_split_2D_cols(self):
        # 创建一个二维数组 a，包含两个相同的行向量
        a = np.array([np.arange(10), np.arange(10)])
        # 将数组 a 按列分割成 3 个子数组，存储在 res 中
        res = array_split(a, 3, axis=-1)
        # 期望的结果 desired，包含三个子数组，每个子数组包含不同的列向量
        desired = [
            np.array([np.arange(4), np.arange(4)]),
            np.array([np.arange(4, 7), np.arange(4, 7)]),
            np.array([np.arange(7, 10), np.arange(7, 10)]),
        ]
        # 比较 res 和 desired 的结果
        compare_results(res, desired)

    # 测试函数：将二维数组按默认轴分割成多个子数组
    def test_integer_split_2D_default(self):
        """如果改变了默认的轴，此测试将失败"""
        # 创建一个二维数组 a，包含两个相同的行向量
        a = np.array([np.arange(10), np.arange(10)])
        # 将数组 a 按默认轴分割成 3 个子数组，存储在 res 中
        res = array_split(a, 3)
        # 目标结果 tgt，包含一个行向量和两个空行向量
        tgt = [np.array([np.arange(10)]), np.array([np.arange(10)]), np.zeros((0, 10))]
        # 比较 res 和 tgt 的结果
        compare_results(res, tgt)
        # 断言最后一个子数组的数据类型与原始数组 a 的数据类型相同
        assert_(a.dtype.type is res[-1].dtype.type)
        # 可能还应该检查更高维度的情况

    # 跳过条件：如果不是 64 位平台，则跳过此测试
    @skipif(not IS_64BIT, reason="Needs 64bit platform")
    def test_integer_split_2D_rows_greater_max_int32(self):
        # 创建一个广播后的数组 a，其形状为 (1 << 32, 2)
        a = np.broadcast_to([0], (1 << 32, 2))
        # 将数组 a 按 4 等分，存储在 res 中
        res = array_split(a, 4)
        # 创建一个广播后的 chunk 数组，其形状为 (1 << 30, 2)
        chunk = np.broadcast_to([0], (1 << 30, 2))
        # 目标结果 tgt，包含 4 个 chunk 数组
        tgt = [chunk] * 4
        # 对每个结果 res[i] 和目标 tgt[i] 进行形状相等的断言检查
        for i in range(len(tgt)):
            assert_equal(res[i].shape, tgt[i].shape)

    # 测试函数：按指定索引将一维数组分割成多个子数组
    def test_index_split_simple(self):
        # 创建一个一维数组 a，包含从 0 到 9 的整数
        a = np.arange(10)
        # 指定索引分割点 indices
        indices = [1, 5, 7]
        # 将数组 a 按指定索引分割成多个子数组，存储在 res 中
        res = array_split(a, indices, axis=-1)
        # 期望的结果 desired，包含根据指定索引切分后的子数组
        desired = [np.arange(0, 1), np.arange(1, 5), np.arange(5, 7), np.arange(7, 10)]
        # 比较 res 和 desired 的结果
        compare_results(res, desired)

    # 测试函数：按指定索引将一维数组分割成多个子数组，包括边界值处理
    def test_index_split_low_bound(self):
        # 创建一个一维数组 a，包含从 0 到 9 的整数
        a = np.arange(10)
        # 指定索引分割点 indices
        indices = [0, 5, 7]
        # 将数组 a 按指定索引分割成多个子数组，存储在 res 中
        res = array_split(a, indices, axis=-1)
        # 期望的结果 desired，包含根据指定索引切分后的子数组
        desired = [np.array([]), np.arange(0, 5), np.arange(5, 7), np.arange(7, 10)]
        # 比较 res 和 desired 的结果
        compare_results(res, desired)

    # 测试函数：按指定索引将一维数组分割成多个子数组，包括边界值处理
    def test_index_split_high_bound(self):
        # 创建一个一维数组 a，包含从 0 到 9 的整数
        a = np.arange(10)
        # 指定索引分割点 indices
        indices = [0, 5, 7, 10, 12]
        # 将数组 a 按指定索引分割成多个子数组，存储在 res 中
        res = array_split(a, indices, axis=-1)
        # 期望的结果 desired，包含根据指定索引切分后的子数组
        desired = [
            np.array([]),
            np.arange(0, 5),
            np.arange(5, 7),
            np.arange(7, 10),
            np.array([]),
            np.array([]),
        ]
        # 比较 res 和 desired 的结果
        compare_results(res, desired)
class TestSplit(TestCase):
    # The split function is essentially the same as array_split,
    # except that it tests if splitting will result in an
    # equal split. Only test for this case.

    def test_equal_split(self):
        # Create an array of integers from 0 to 9
        a = np.arange(10)
        # Call the split function with array 'a' and split count 2
        res = split(a, 2)
        # Define the expected result as two sub-arrays split equally
        desired = [np.arange(5), np.arange(5, 10)]
        # Compare the result of split operation with the desired result
        compare_results(res, desired)

    def test_unequal_split(self):
        # Create an array of integers from 0 to 9
        a = np.arange(10)
        # Assert that calling split with array 'a' and split count 3 raises a ValueError
        assert_raises(ValueError, split, a, 3)


class TestColumnStack(TestCase):
    def test_non_iterable(self):
        # Assert that calling column_stack with a non-iterable argument like 1 raises a TypeError
        assert_raises(TypeError, column_stack, 1)

    def test_1D_arrays(self):
        # Example from the docstring: create two 1D arrays 'a' and 'b'
        a = np.array((1, 2, 3))
        b = np.array((2, 3, 4))
        # Define the expected result after column_stack operation
        expected = np.array([[1, 2], [2, 3], [3, 4]])
        # Perform column_stack operation on arrays 'a' and 'b'
        actual = np.column_stack((a, b))
        # Assert that the actual result matches the expected result
        assert_equal(actual, expected)

    def test_2D_arrays(self):
        # Same as hstack 2D docstring example: create two 2D arrays 'a' and 'b'
        a = np.array([[1], [2], [3]])
        b = np.array([[2], [3], [4]])
        # Define the expected result after column_stack operation
        expected = np.array([[1, 2], [2, 3], [3, 4]])
        # Perform column_stack operation on arrays 'a' and 'b'
        actual = np.column_stack((a, b))
        # Assert that the actual result matches the expected result
        assert_equal(actual, expected)

    def test_generator(self):
        # Test column_stack with a generator that yields arrays
        # Note: numpy 1.24 emits a warning here, but we don't handle warnings with assert_warns
        column_stack([np.arange(3) for _ in range(2)])


class TestDstack(TestCase):
    def test_non_iterable(self):
        # Assert that calling dstack with a non-iterable argument like 1 raises a TypeError
        assert_raises(TypeError, dstack, 1)

    def test_0D_array(self):
        # Create scalar arrays 'a' and 'b'
        a = np.array(1)
        b = np.array(2)
        # Perform dstack operation on arrays 'a' and 'b'
        res = dstack([a, b])
        # Define the expected result after dstack operation
        desired = np.array([[[1, 2]]])
        # Assert that the result of dstack matches the expected result
        assert_array_equal(res, desired)

    def test_1D_array(self):
        # Create 1D arrays 'a' and 'b'
        a = np.array([1])
        b = np.array([2])
        # Perform dstack operation on arrays 'a' and 'b'
        res = dstack([a, b])
        # Define the expected result after dstack operation
        desired = np.array([[[1, 2]]])
        # Assert that the result of dstack matches the expected result
        assert_array_equal(res, desired)

    def test_2D_array(self):
        # Create 2D arrays 'a' and 'b'
        a = np.array([[1], [2]])
        b = np.array([[1], [2]])
        # Perform dstack operation on arrays 'a' and 'b'
        res = dstack([a, b])
        # Define the expected result after dstack operation
        desired = np.array([[[1, 1]], [[2, 2]]])
        # Assert that the result of dstack matches the expected result
        assert_array_equal(res, desired)

    def test_2D_array2(self):
        # Create 1D arrays 'a' and 'b'
        a = np.array([1, 2])
        b = np.array([1, 2])
        # Perform dstack operation on arrays 'a' and 'b'
        res = dstack([a, b])
        # Define the expected result after dstack operation
        desired = np.array([[[1, 1], [2, 2]]])
        # Assert that the result of dstack matches the expected result
        assert_array_equal(res, desired)

    def test_generator(self):
        # Test dstack with a generator that yields arrays
        # Note: numpy 1.24 emits a warning here, but we don't handle warnings with assert_warns
        dstack([np.arange(3) for _ in range(2)])


class TestHsplit(TestCase):
    """Only testing for integer splits."""

    def test_non_iterable(self):
        # Assert that calling hsplit with non-iterable arguments like 1 and 1 raises a ValueError
        assert_raises(ValueError, hsplit, 1, 1)
    # 定义一个测试方法，用于测试零维数组的水平分割
    def test_0D_array(self):
        # 创建一个包含单个元素的 NumPy 数组
        a = np.array(1)
        # 尝试对单个元素的数组进行水平分割，预期会引发 ValueError 异常
        try:
            hsplit(a, 2)
            # 如果没有引发异常，则断言失败
            assert_(0)
        # 捕获预期的 ValueError 异常
        except ValueError:
            # 如果捕获到异常，则不做任何操作，测试通过
            pass
    
    # 定义一个测试方法，用于测试一维数组的水平分割
    def test_1D_array(self):
        # 创建一个一维 NumPy 数组
        a = np.array([1, 2, 3, 4])
        # 对一维数组进行水平分割，将数组分割成两部分
        res = hsplit(a, 2)
        # 期望得到的分割结果
        desired = [np.array([1, 2]), np.array([3, 4])]
        # 比较实际结果和期望结果
        compare_results(res, desired)
    
    # 定义一个测试方法，用于测试二维数组的水平分割
    def test_2D_array(self):
        # 创建一个二维 NumPy 数组
        a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        # 对二维数组进行水平分割，将数组的每一行分割成两部分
        res = hsplit(a, 2)
        # 期望得到的分割结果，每行分割后形成的子数组
        desired = [np.array([[1, 2], [1, 2]]), np.array([[3, 4], [3, 4]])]
        # 比较实际结果和期望结果
        compare_results(res, desired)
class TestVsplit(TestCase):
    """Only testing for integer splits."""

    # 测试非可迭代对象的情况
    def test_non_iterable(self):
        assert_raises(ValueError, vsplit, 1, 1)

    # 测试0维数组的情况
    def test_0D_array(self):
        a = np.array(1)
        assert_raises(ValueError, vsplit, a, 2)

    # 测试1维数组的情况
    def test_1D_array(self):
        a = np.array([1, 2, 3, 4])
        try:
            # 尝试进行垂直分割，预期会抛出值错误异常
            vsplit(a, 2)
            assert_(0)
        except ValueError:
            pass

    # 测试2维数组的情况
    def test_2D_array(self):
        a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        # 进行垂直分割
        res = vsplit(a, 2)
        desired = [np.array([[1, 2, 3, 4]]), np.array([[1, 2, 3, 4]])]
        # 比较分割结果和预期结果
        compare_results(res, desired)


class TestDsplit(TestCase):
    # Only testing for integer splits.
    # 只测试整数分割的情况
    def test_non_iterable(self):
        assert_raises(ValueError, dsplit, 1, 1)

    # 测试0维数组的情况
    def test_0D_array(self):
        a = np.array(1)
        assert_raises(ValueError, dsplit, a, 2)

    # 测试1维数组的情况
    def test_1D_array(self):
        a = np.array([1, 2, 3, 4])
        assert_raises(ValueError, dsplit, a, 2)

    # 测试2维数组的情况
    def test_2D_array(self):
        a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        try:
            # 尝试进行深度分割，预期会抛出值错误异常
            dsplit(a, 2)
            assert_(0)
        except ValueError:
            pass

    # 测试3维数组的情况
    def test_3D_array(self):
        a = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]])
        # 进行深度分割
        res = dsplit(a, 2)
        desired = [
            np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]]),
            np.array([[[3, 4], [3, 4]], [[3, 4], [3, 4]]]),
        ]
        # 比较分割结果和预期结果
        compare_results(res, desired)


class TestSqueeze(TestCase):
    # 测试基本情况
    def test_basic(self):
        # 创建不同形状的数组a, b, c
        a = rand(20, 10, 10, 1, 1)
        b = rand(20, 1, 10, 1, 20)
        c = rand(1, 1, 20, 10)
        # 压缩数组a，b，c，并与其重塑后的形状进行比较
        assert_array_equal(np.squeeze(a), np.reshape(a, (20, 10, 10)))
        assert_array_equal(np.squeeze(b), np.reshape(b, (20, 10, 20)))
        assert_array_equal(np.squeeze(c), np.reshape(c, (20, 10)))

        # 压缩为0维应该仍然返回一个ndarray
        a = [[[1.5]]]
        res = np.squeeze(a)
        assert_equal(res, 1.5)
        assert_equal(res.ndim, 0)
        assert type(res) is np.ndarray

    @xfailIfTorchDynamo
    # 测试基本情况2
    def test_basic_2(self):
        aa = np.ones((3, 1, 4, 1, 1))
        assert aa.squeeze().tensor._base is aa.tensor
    # 定义一个测试方法，用于测试 np.squeeze 函数在不同情况下的行为
    def test_squeeze_axis(self):
        # 创建一个三维数组 A
        A = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
        # 断言去除所有长度为 1 的维度后，形状应为 (3, 3)
        assert_equal(np.squeeze(A).shape, (3, 3))
        # 断言对于空轴列表，不会对数组进行任何变化
        assert_equal(np.squeeze(A, axis=()), A)

        # 断言对一个形状为 (1, 3, 1) 的零数组去除长度为 1 的维度后，形状应为 (3,)
        assert_equal(np.squeeze(np.zeros((1, 3, 1))).shape, (3,))
        # 断言对一个形状为 (1, 3, 1) 的零数组在轴 0 上去除长度为 1 的维度后，形状应为 (3, 1)
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=0).shape, (3, 1))
        # 断言对一个形状为 (1, 3, 1) 的零数组在轴 -1 上去除长度为 1 的维度后，形状应为 (1, 3)
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=-1).shape, (1, 3))
        # 断言对一个形状为 (1, 3, 1) 的零数组在轴 2 上去除长度为 1 的维度后，形状应为 (1, 3)
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=2).shape, (1, 3))
        # 断言对一个包含形状为 (3, 1) 的零数组的列表去除长度为 1 的维度后，形状应为 (3,)
        assert_equal(np.squeeze([np.zeros((3, 1))]).shape, (3,))
        # 断言对一个包含形状为 (3, 1) 的零数组的列表在轴 0 上去除长度为 1 的维度后，形状应为 (3, 1)
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=0).shape, (3, 1))
        # 断言对一个包含形状为 (3, 1) 的零数组的列表在轴 2 上去除长度为 1 的维度后，形状应为 (1, 3)
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=2).shape, (1, 3))
        # 断言对一个包含形状为 (3, 1) 的零数组的列表在轴 -1 上去除长度为 1 的维度后，形状应为 (1, 3)

    # 定义一个测试方法，用于验证 np.squeeze 方法在不同类型的数组上的行为
    def test_squeeze_type(self):
        # 创建一个包含单个元素的一维数组 a
        a = np.array([3])
        # 创建一个标量 b
        b = np.array(3)
        # 断言去除所有长度为 1 的维度后，a 和 b 的类型均为 np.ndarray
        assert type(a.squeeze()) is np.ndarray
        assert type(b.squeeze()) is np.ndarray

    # 标记为跳过的测试方法，原因是 'XXX: order='F' not implemented'
    @skip(reason="XXX: order='F' not implemented")
    def test_squeeze_contiguous(self):
        # 创建一个形状为 (1, 2) 的零数组 a，并对其进行去除长度为 1 的维度操作
        a = np.zeros((1, 2)).squeeze()
        # 创建一个按 Fortran（列优先）顺序存储的形状为 (2, 2, 2) 的零数组 b，并对其进行切片和去除长度为 1 的维度操作
        b = np.zeros((2, 2, 2), order="F")[:, :, ::2].squeeze()
        # 断言数组 a 是 C 连续的
        assert_(a.flags.c_contiguous)
        # 断言数组 a 是 Fortran 连续的
        assert_(a.flags.f_contiguous)
        # 断言数组 b 是 Fortran 连续的

    # 标记为跳过的测试方法，使用 @xpassIfTorchDynamo 注解
    @xpassIfTorchDynamo  # (reason="XXX: noop in torch, while numpy raises")
    def test_squeeze_axis_handling(self):
        # 使用 assert_raises 断言在给定的操作中会引发 ValueError 异常
        with assert_raises(ValueError):
            np.squeeze(np.array([[1], [2], [3]]), axis=0)
@instantiate_parametrized_tests
class TestKron(TestCase):
    def test_basic(self):
        # 使用0维的ndarray
        a = np.array(1)
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[1, 2], [3, 4]])
        # 断言np.kron(a, b)的结果与k相等
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array(1)
        # 断言np.kron(a, b)的结果与k相等
        assert_array_equal(np.kron(a, b), k)

        # 使用1维的ndarray
        a = np.array([3])
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[3, 6], [9, 12]])
        # 断言np.kron(a, b)的结果与k相等
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array([3])
        # 断言np.kron(a, b)的结果与k相等
        assert_array_equal(np.kron(a, b), k)

        # 使用3维的ndarray
        a = np.array([[[1]], [[2]]])
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        # 断言np.kron(a, b)的结果与k相等
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[[1]], [[2]]])
        k = np.array([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        # 断言np.kron(a, b)的结果与k相等
        assert_array_equal(np.kron(a, b), k)

    @skip(reason="NP_VER: fails on CI")
    @parametrize(
        "shape_a,shape_b",
        [
            ((1, 1), (1, 1)),
            ((1, 2, 3), (4, 5, 6)),
            ((2, 2), (2, 2, 2)),
            ((1, 0), (1, 1)),
            ((2, 0, 2), (2, 2)),
            ((2, 0, 0, 2), (2, 0, 2)),
        ],
    )
    def test_kron_shape(self, shape_a, shape_b):
        a = np.ones(shape_a)
        b = np.ones(shape_b)
        # 根据输入ndarray的形状，生成正规化后的形状
        normalised_shape_a = (1,) * max(0, len(shape_b) - len(shape_a)) + shape_a
        normalised_shape_b = (1,) * max(0, len(shape_a) - len(shape_b)) + shape_b
        expected_shape = np.multiply(normalised_shape_a, normalised_shape_b)

        k = np.kron(a, b)
        # 断言np.kron(a, b)的形状与expected_shape相等
        assert np.array_equal(k.shape, expected_shape), "Unexpected shape from kron"


class TestTile(TestCase):
    def test_basic(self):
        a = np.array([0, 1, 2])
        b = [[1, 2], [3, 4]]
        # 断言tile(a, 2)的结果与[0, 1, 2, 0, 1, 2]相等
        assert_equal(tile(a, 2), [0, 1, 2, 0, 1, 2])
        # 断言tile(a, (2, 2))的结果与[[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]]相等
        assert_equal(tile(a, (2, 2)), [[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]])
        # 断言tile(a, (1, 2))的结果与[[0, 1, 2, 0, 1, 2]]相等
        assert_equal(tile(a, (1, 2)), [[0, 1, 2, 0, 1, 2]])
        # 断言tile(b, 2)的结果与[[1, 2, 1, 2], [3, 4, 3, 4]]相等
        assert_equal(tile(b, 2), [[1, 2, 1, 2], [3, 4, 3, 4]])
        # 断言tile(b, (2, 1))的结果与[[1, 2], [3, 4], [1, 2], [3, 4]]相等
        assert_equal(tile(b, (2, 1)), [[1, 2], [3, 4], [1, 2], [3, 4]])
        # 断言tile(b, (2, 2))的结果与[[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]相等
        assert_equal(
            tile(b, (2, 2)), [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]
        )

    def test_tile_one_repetition_on_array_gh4679(self):
        a = np.arange(5)
        b = tile(a, 1)
        b += 2
        # 断言a与np.arange(5)相等
        assert_equal(a, np.arange(5))

    def test_empty(self):
        a = np.array([[[]]])
        b = np.array([[], []])
        c = tile(b, 2).shape
        d = tile(a, (3, 2, 5)).shape
        # 断言c与(2, 0)相等
        assert_equal(c, (2, 0))
        # 断言d与(3, 2, 0)相等
        assert_equal(d, (3, 2, 0))
    # 定义一个测试函数 `test_kroncompare`，用于测试 `kron` 函数的比较行为
    def test_kroncompare(self):
        # 定义重复次数列表 `reps`，每个元素是一个元组，表示重复的次数
        reps = [(2,), (1, 2), (2, 1), (2, 2), (2, 3, 2), (3, 2)]
        # 定义形状列表 `shape`，每个元素是一个元组，表示数组的形状
        shape = [(3,), (2, 3), (3, 4, 3), (3, 2, 3), (4, 3, 2, 4), (2, 2)]
        
        # 遍历形状列表 `shape` 中的每个形状 `s`
        for s in shape:
            # 生成一个随机数组 `b`，其形状为 `s`，元素值为 0 到 10 之间的随机整数
            b = randint(0, 10, size=s)
            
            # 遍历重复次数列表 `reps` 中的每个重复次数 `r`
            for r in reps:
                # 生成一个全为 1 的数组 `a`，形状为 `r`，数据类型与 `b` 相同
                a = np.ones(r, b.dtype)
                
                # 使用 `tile` 函数生成一个大数组 `large`，其形状为 `r * s`，元素由数组 `b` 填充
                large = tile(b, r)
                
                # 使用 `kron` 函数生成两个数组 `a` 和 `b` 的 Kronecker 乘积 `klarge`
                klarge = kron(a, b)
                
                # 断言 `large` 和 `klarge` 应相等
                assert_equal(large, klarge)
@xfail  # 可能有一天会实现，当前注释掉这个测试类
class TestMayShareMemory(TestCase):
    def test_basic(self):
        d = np.ones((50, 60))
        d2 = np.ones((30, 60, 6))
        # 断言两个数组是否可能共享内存
        assert_(np.may_share_memory(d, d))
        assert_(np.may_share_memory(d, d[::-1]))
        assert_(np.may_share_memory(d, d[::2]))
        assert_(np.may_share_memory(d, d[1:, ::-1]))

        # 断言两个数组是否不可能共享内存
        assert_(not np.may_share_memory(d[::-1], d2))
        assert_(not np.may_share_memory(d[::2], d2))
        assert_(not np.may_share_memory(d[1:, ::-1], d2))
        assert_(np.may_share_memory(d2[1:, ::-1], d2))


# Utility
def compare_results(res, desired):
    """比较数组列表的结果。"""
    if len(res) != len(desired):
        raise ValueError("Iterables have different lengths")
    # 使用 zip 函数逐一比较两个迭代器中的元素
    for x, y in zip(res, desired):
        assert_array_equal(x, y)


if __name__ == "__main__":
    run_tests()
```