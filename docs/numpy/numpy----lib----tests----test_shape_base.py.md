# `.\numpy\numpy\lib\tests\test_shape_base.py`

```py
# 导入所需的库和模块
import numpy as np  # 导入 NumPy 库，用于数值计算
import functools  # 导入 functools 模块，用于函数操作
import sys  # 导入 sys 模块，用于系统相关操作
import pytest  # 导入 pytest 模块，用于编写和运行测试

from numpy import (  # 导入 NumPy 中的多个函数和类
    apply_along_axis, apply_over_axes, array_split, split, hsplit, dsplit,
    vsplit, dstack, column_stack, kron, tile, expand_dims, take_along_axis,
    put_along_axis
    )
from numpy.exceptions import AxisError  # 导入 AxisError 异常类，用于处理轴异常
from numpy.testing import (  # 导入 NumPy 测试相关的函数和类
    assert_, assert_equal, assert_array_equal, assert_raises, assert_warns
    )


IS_64BIT = sys.maxsize > 2**32  # 判断系统位数是否为 64 位


def _add_keepdims(func):
    """ hack in keepdims behavior into a function taking an axis """
    @functools.wraps(func)
    def wrapped(a, axis, **kwargs):
        # 调用原始函数并添加 keepdims 行为
        res = func(a, axis=axis, **kwargs)
        if axis is None:
            axis = 0  # 如果结果是标量，可以在任意轴插入
        return np.expand_dims(res, axis=axis)  # 返回带有 keepdims 的结果数组
    return wrapped


class TestTakeAlongAxis:
    def test_argequivalent(self):
        """ Test it translates from arg<func> to <func> """
        from numpy.random import rand  # 从 NumPy 随机模块导入 rand 函数
        a = rand(3, 4, 5)  # 创建一个形状为 (3, 4, 5) 的随机数组 a

        funcs = [
            (np.sort, np.argsort, dict()),  # 元组列表，包含排序和排序索引函数
            (_add_keepdims(np.min), _add_keepdims(np.argmin), dict()),  # 使用带有 keepdims 的最小和最小索引函数
            (_add_keepdims(np.max), _add_keepdims(np.argmax), dict()),  # 使用带有 keepdims 的最大和最大索引函数
            #(np.partition, np.argpartition, dict(kth=2)),  # 部分排序和部分排序索引函数
        ]

        for func, argfunc, kwargs in funcs:  # 遍历函数列表
            for axis in list(range(a.ndim)) + [None]:  # 遍历数组的所有轴和 None
                a_func = func(a, axis=axis, **kwargs)  # 应用函数 func 到数组 a
                ai_func = argfunc(a, axis=axis, **kwargs)  # 应用函数 argfunc 到数组 a
                assert_equal(a_func, take_along_axis(a, ai_func, axis=axis))  # 断言函数的输出与 take_along_axis 的输出相等

    def test_invalid(self):
        """ Test it errors when indices has too few dimensions """
        a = np.ones((10, 10))  # 创建一个全为 1 的 10x10 数组 a
        ai = np.ones((10, 2), dtype=np.intp)  # 创建一个全为 1 的形状为 (10, 2) 的整数数组 ai

        # 确保正常工作
        take_along_axis(a, ai, axis=1)

        # 索引维度不足
        assert_raises(ValueError, take_along_axis, a, np.array(1), axis=1)
        # 不允许布尔数组
        assert_raises(IndexError, take_along_axis, a, ai.astype(bool), axis=1)
        # 不允许浮点数数组
        assert_raises(IndexError, take_along_axis, a, ai.astype(float), axis=1)
        # 无效的轴
        assert_raises(AxisError, take_along_axis, a, ai, axis=10)
        # 无效的索引
        assert_raises(ValueError, take_along_axis, a, ai, axis=None)

    def test_empty(self):
        """ Test everything is ok with empty results, even with inserted dims """
        a  = np.ones((3, 4, 5))  # 创建一个全为 1 的形状为 (3, 4, 5) 的数组 a
        ai = np.ones((3, 0, 5), dtype=np.intp)  # 创建一个形状为 (3, 0, 5) 的整数数组 ai

        actual = take_along_axis(a, ai, axis=1)  # 在轴 1 上执行 take_along_axis 操作
        assert_equal(actual.shape, ai.shape)  # 断言实际输出的形状与 ai 的形状相等

    def test_broadcast(self):
        """ Test that non-indexing dimensions are broadcast in both directions """
        a  = np.ones((3, 4, 1))  # 创建一个形状为 (3, 4, 1) 的全为 1 的数组 a
        ai = np.ones((1, 2, 5), dtype=np.intp)  # 创建一个形状为 (1, 2, 5) 的整数数组 ai
        actual = take_along_axis(a, ai, axis=1)  # 在轴 1 上执行 take_along_axis 操作
        assert_equal(actual.shape, (3, 2, 5))  # 断言实际输出的形状为 (3, 2, 5)

class TestPutAlongAxis:
    def test_replace_max(self):
        # 创建一个基础的二维 NumPy 数组
        a_base = np.array([[10, 30, 20], [60, 40, 50]])

        # 对数组的每个维度和整体进行迭代测试
        for axis in list(range(a_base.ndim)) + [None]:
            # 在循环中对数组进行深拷贝，以便进行修改
            a = a_base.copy()

            # 找到最大值的索引，并保持维度信息
            i_max = _add_keepdims(np.argmax)(a, axis=axis)
            # 使用指定的小值替换最大值
            put_along_axis(a, i_max, -99, axis=axis)

            # 寻找新的最小值索引，它应当等于之前找到的最大值索引
            i_min = _add_keepdims(np.argmin)(a, axis=axis)

            # 断言新的最小值索引等于最大值索引
            assert_equal(i_min, i_max)

    def test_broadcast(self):
        """ Test that non-indexing dimensions are broadcast in both directions """
        # 创建一个三维全为1的 NumPy 数组
        a = np.ones((3, 4, 1))
        # 创建一个用于索引的二维整数数组，确保在范围内
        ai = np.arange(10, dtype=np.intp).reshape((1, 2, 5)) % 4
        # 使用 put_along_axis 函数在指定轴向上设置值
        put_along_axis(a, ai, 20, axis=1)
        # 断言使用 take_along_axis 函数获取的值等于设定的值
        assert_equal(take_along_axis(a, ai, axis=1), 20)

    def test_invalid(self):
        """ Test invalid inputs """
        # 创建一个基础的二维 NumPy 数组
        a_base = np.array([[10, 30, 20], [60, 40, 50]])
        # 创建用于索引和值的数组
        indices = np.array([[0], [1]])
        values = np.array([[2], [1]])

        # 对数组进行深拷贝作为基础
        a = a_base.copy()
        # 使用 put_along_axis 函数在指定轴向上设置值
        put_along_axis(a, indices, values, axis=0)
        # 断言数组的所有元素等于预期的值
        assert np.all(a == [[2, 2, 2], [1, 1, 1]])

        # 测试无效的索引输入
        a = a_base.copy()
        # 使用 assert_raises 检测是否抛出 ValueError 异常
        with assert_raises(ValueError) as exc:
            put_along_axis(a, indices, values, axis=None)
        # 断言异常信息包含特定文本
        assert "single dimension" in str(exc.exception)
class TestApplyAlongAxis:
    # 测试类，用于测试 apply_along_axis 函数的不同用例

    def test_simple(self):
        # 简单测试用例：创建一个 20x10 的双精度浮点数数组 a，全为 1
        a = np.ones((20, 10), 'd')
        # 断言应用 len 函数在轴 0 上的结果与期望的数组相等
        assert_array_equal(
            apply_along_axis(len, 0, a), len(a)*np.ones(a.shape[1]))

    def test_simple101(self):
        # 另一个简单测试用例：创建一个 10x101 的双精度浮点数数组 a，全为 1
        a = np.ones((10, 101), 'd')
        # 断言应用 len 函数在轴 0 上的结果与期望的数组相等
        assert_array_equal(
            apply_along_axis(len, 0, a), len(a)*np.ones(a.shape[1]))

    def test_3d(self):
        # 3D 数组测试用例：创建一个形状为 (3, 3, 3) 的数组 a，其中元素为 0 到 26
        a = np.arange(27).reshape((3, 3, 3))
        # 断言应用 np.sum 函数在轴 0 上的结果与期望的数组相等
        assert_array_equal(apply_along_axis(np.sum, 0, a),
                           [[27, 30, 33], [36, 39, 42], [45, 48, 51]])

    def test_preserve_subclass(self):
        # 保留子类测试用例：定义一个函数 double，返回数组行的两倍
        def double(row):
            return row * 2
        
        # 定义一个继承自 np.ndarray 的子类 MyNDArray
        class MyNDArray(np.ndarray):
            pass
        
        # 创建一个 MyNDArray 类型的数组 m，形状为 (2, 2)，元素为 0 到 3
        m = np.array([[0, 1], [2, 3]]).view(MyNDArray)
        # 期望的数组 expected，元素为 [[0, 2], [4, 6]]
        expected = np.array([[0, 2], [4, 6]]).view(MyNDArray)

        # 对数组 m 应用 double 函数在轴 0 上，断言结果是 MyNDArray 类型，并且与期望数组相等
        result = apply_along_axis(double, 0, m)
        assert_(isinstance(result, MyNDArray))
        assert_array_equal(result, expected)

        # 对数组 m 应用 double 函数在轴 1 上，断言结果是 MyNDArray 类型，并且与期望数组相等
        result = apply_along_axis(double, 1, m)
        assert_(isinstance(result, MyNDArray))
        assert_array_equal(result, expected)

    def test_subclass(self):
        # 子类测试用例：定义一个最小的子类 MinimalSubclass，继承自 np.ndarray
        class MinimalSubclass(np.ndarray):
            data = 1
        
        # 定义一个函数 minimal_function，返回数组的 data 属性
        def minimal_function(array):
            return array.data
        
        # 创建一个 MinimalSubclass 类型的数组 a，形状为 (6, 3)，元素全为 0
        a = np.zeros((6, 3)).view(MinimalSubclass)

        # 断言应用 minimal_function 函数在轴 0 上的结果与期望的数组相等，期望数组为 [1, 1, 1]
        assert_array_equal(
            apply_along_axis(minimal_function, 0, a), np.array([1, 1, 1])
        )

    def test_scalar_array(self, cls=np.ndarray):
        # 标量数组测试用例：创建一个 6x3 的数组 a，元素全为 1，视图类型为 cls
        a = np.ones((6, 3)).view(cls)
        # 对数组 a 应用 np.sum 函数在轴 0 上，保存结果到 res
        res = apply_along_axis(np.sum, 0, a)
        # 断言 res 的类型是 cls，并且与期望的数组相等，期望数组为 [6, 6, 6]
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([6, 6, 6]).view(cls))

    def test_0d_array(self, cls=np.ndarray):
        # 零维数组测试用例：定义一个函数 sum_to_0d，对一维数组求和，并返回相同类的零维数组
        def sum_to_0d(x):
            """ Sum x, returning a 0d array of the same class """
            assert_equal(x.ndim, 1)
            return np.squeeze(np.sum(x, keepdims=True))
        
        # 创建一个 6x3 的数组 a，元素全为 1，视图类型为 cls
        a = np.ones((6, 3)).view(cls)
        
        # 对数组 a 应用 sum_to_0d 函数在轴 0 上，保存结果到 res
        res = apply_along_axis(sum_to_0d, 0, a)
        # 断言 res 的类型是 cls，并且与期望的数组相等，期望数组为 [6, 6, 6]
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([6, 6, 6]).view(cls))

        # 对数组 a 应用 sum_to_0d 函数在轴 1 上，保存结果到 res
        res = apply_along_axis(sum_to_0d, 1, a)
        # 断言 res 的类型是 cls，并且与期望的数组相等，期望数组为 [3, 3, 3, 3, 3, 3]
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([3, 3, 3, 3, 3, 3]).view(cls))
    def test_axis_insertion(self, cls=np.ndarray):
        # 定义一个函数 f1to2，从输入向量 x 生成一个非对称非方阵
        def f1to2(x):
            """produces an asymmetric non-square matrix from x"""
            # 断言 x 的维度为 1
            assert_equal(x.ndim, 1)
            # 返回 x 的逆序乘以 x 的转置，并将结果视图化为指定类别 cls
            return (x[::-1] * x[1:,None]).view(cls)

        # 创建一个 6x3 的二维数组 a2d
        a2d = np.arange(6*3).reshape((6, 3))

        # 对第一个轴进行二维插入
        actual = apply_along_axis(f1to2, 0, a2d)
        # 期望结果是对每列应用 f1to2 函数并堆叠，最终视图化为指定类别 cls
        expected = np.stack([
            f1to2(a2d[:,i]) for i in range(a2d.shape[1])
        ], axis=-1).view(cls)
        # 断言实际结果和期望结果的类型相同
        assert_equal(type(actual), type(expected))
        # 断言实际结果和期望结果相等
        assert_equal(actual, expected)

        # 对最后一个轴进行二维插入
        actual = apply_along_axis(f1to2, 1, a2d)
        # 期望结果是对每行应用 f1to2 函数并堆叠，最终视图化为指定类别 cls
        expected = np.stack([
            f1to2(a2d[i,:]) for i in range(a2d.shape[0])
        ], axis=0).view(cls)
        # 断言实际结果和期望结果的类型相同
        assert_equal(type(actual), type(expected))
        # 断言实际结果和期望结果相等
        assert_equal(actual, expected)

        # 对中间轴进行三维插入
        a3d = np.arange(6*5*3).reshape((6, 5, 3))

        actual = apply_along_axis(f1to2, 1, a3d)
        # 期望结果是对每个深度切片应用 f1to2 函数并堆叠，最终视图化为指定类别 cls
        expected = np.stack([
            np.stack([
                f1to2(a3d[i,:,j]) for i in range(a3d.shape[0])
            ], axis=0)
            for j in range(a3d.shape[2])
        ], axis=-1).view(cls)
        # 断言实际结果和期望结果的类型相同
        assert_equal(type(actual), type(expected))
        # 断言实际结果和期望结果相等
        assert_equal(actual, expected)

    def test_subclass_preservation(self):
        # 定义一个简单的子类 MinimalSubclass 继承自 np.ndarray
        class MinimalSubclass(np.ndarray):
            pass
        # 分别测试标量数组、0维数组和轴插入函数在 MinimalSubclass 下的行为
        self.test_scalar_array(MinimalSubclass)
        self.test_0d_array(MinimalSubclass)
        self.test_axis_insertion(MinimalSubclass)

    def test_axis_insertion_ma(self):
        # 定义一个函数 f1to2，从输入向量 x 生成一个非对称非方阵的掩码数组
        def f1to2(x):
            """produces an asymmetric non-square matrix from x"""
            # 断言 x 的维度为 1
            assert_equal(x.ndim, 1)
            # 计算 x 的逆序乘以 x 的转置，将结果进行掩码处理（余数为 0 的位置被掩盖）
            res = x[::-1] * x[1:,None]
            return np.ma.masked_where(res%5==0, res)
        # 创建一个 6x3 的二维数组 a
        a = np.arange(6*3).reshape((6, 3))
        # 应用 f1to2 函数到 a 的每列，期望结果是一个掩码数组
        res = apply_along_axis(f1to2, 0, a)
        # 断言 res 的类型为 np.ma.masked_array
        assert_(isinstance(res, np.ma.masked_array))
        # 断言 res 的维度为 3
        assert_equal(res.ndim, 3)
        # 检查每个深度切片的掩码与对应列的 f1to2 结果的掩码相匹配
        assert_array_equal(res[:,:,0].mask, f1to2(a[:,0]).mask)
        assert_array_equal(res[:,:,1].mask, f1to2(a[:,1]).mask)
        assert_array_equal(res[:,:,2].mask, f1to2(a[:,2]).mask)

    def test_tuple_func1d(self):
        # 定义一个简单的函数 sample_1d，交换输入向量 x 的第一个和第二个元素
        def sample_1d(x):
            return x[1], x[0]
        # 对输入数组 [[1, 2], [3, 4]] 应用 sample_1d 函数，期望结果是 [[2, 1], [4, 3]]
        res = np.apply_along_axis(sample_1d, 1, np.array([[1, 2], [3, 4]]))
        # 断言 res 等于期望的结果数组
        assert_array_equal(res, np.array([[2, 1], [4, 3]]))
    def test_empty(self):
        # 定义一个永远不会被调用的函数，用于测试 apply_along_axis
        def never_call(x):
            assert_(False) # 应该永远不会执行到这里

        # 创建一个空的 numpy 数组
        a = np.empty((0, 0))
        # 测试在维度 0 上调用 apply_along_axis 是否会引发 ValueError 异常
        assert_raises(ValueError, np.apply_along_axis, never_call, 0, a)
        # 测试在维度 1 上调用 apply_along_axis 是否会引发 ValueError 异常
        assert_raises(ValueError, np.apply_along_axis, never_call, 1, a)

        # 但是在某些非零维度情况下，也可以正常工作
        # 定义一个将空数组转换为 1 的函数
        def empty_to_1(x):
            assert_(len(x) == 0)
            return 1

        # 创建一个具有 10 行和 0 列的空 numpy 数组
        a = np.empty((10, 0))
        # 在每行上应用 empty_to_1 函数，预期结果应为全为 1 的数组
        actual = np.apply_along_axis(empty_to_1, 1, a)
        assert_equal(actual, np.ones(10))
        # 测试在维度 0 上调用 apply_along_axis 是否会引发 ValueError 异常
        assert_raises(ValueError, np.apply_along_axis, empty_to_1, 0, a)

    def test_with_iterable_object(self):
        # 来自问题 5248
        # 创建一个包含集合的多维 numpy 数组
        d = np.array([
            [{1, 11}, {2, 22}, {3, 33}],
            [{4, 44}, {5, 55}, {6, 66}]
        ])
        # 在维度 0 上应用 lambda 函数，该函数执行集合的并集操作
        actual = np.apply_along_axis(lambda a: set.union(*a), 0, d)
        # 预期的结果数组，每个元素为合并对应位置集合的结果
        expected = np.array([{1, 11, 4, 44}, {2, 22, 5, 55}, {3, 33, 6, 66}])

        # 断言实际输出与预期输出相等
        assert_equal(actual, expected)

        # 问题 8642 - assert_equal 无法检测到此问题！
        # 遍历实际输出的每个元素，检查其类型是否与预期输出的类型相同
        for i in np.ndindex(actual.shape):
            assert_equal(type(actual[i]), type(expected[i]))
class TestApplyOverAxes:
    # 测试 apply_over_axes 函数的功能
    def test_simple(self):
        # 创建一个 2x3x4 的数组
        a = np.arange(24).reshape(2, 3, 4)
        # 对数组 a 沿指定轴向应用 np.sum 函数
        aoa_a = apply_over_axes(np.sum, a, [0, 2])
        # 断言 aoa_a 的结果与给定的数组相等
        assert_array_equal(aoa_a, np.array([[[60], [92], [124]]]))


class TestExpandDims:
    # 测试 expand_dims 函数的功能
    def test_functionality(self):
        s = (2, 3, 4, 5)
        a = np.empty(s)
        # 遍历可能的轴向范围
        for axis in range(-5, 4):
            # 在数组 a 上扩展维度
            b = expand_dims(a, axis)
            # 断言扩展后的维度在指定轴上为 1
            assert_(b.shape[axis] == 1)
            # 断言压缩后的数组形状与原始形状相同
            assert_(np.squeeze(b).shape == s)

    # 测试在元组形式下的多轴扩展
    def test_axis_tuple(self):
        a = np.empty((3, 3, 3))
        # 断言使用元组指定轴向进行扩展后的形状
        assert np.expand_dims(a, axis=(0, 1, 2)).shape == (1, 1, 1, 3, 3, 3)
        assert np.expand_dims(a, axis=(0, -1, -2)).shape == (1, 3, 3, 3, 1, 1)
        assert np.expand_dims(a, axis=(0, 3, 5)).shape == (1, 3, 3, 1, 3, 1)
        assert np.expand_dims(a, axis=(0, -3, -5)).shape == (1, 1, 3, 1, 3, 3)

    # 测试超出范围的轴向
    def test_axis_out_of_range(self):
        s = (2, 3, 4, 5)
        a = np.empty(s)
        # 断言当轴向超出数组维度范围时引发 AxisError
        assert_raises(AxisError, expand_dims, a, -6)
        assert_raises(AxisError, expand_dims, a, 5)

        a = np.empty((3, 3, 3))
        # 断言当轴向以元组形式且包含超出范围的索引时引发 AxisError
        assert_raises(AxisError, expand_dims, a, (0, -6))
        assert_raises(AxisError, expand_dims, a, (0, 5))

    # 测试重复指定的轴向
    def test_repeated_axis(self):
        a = np.empty((3, 3, 3))
        # 断言当重复指定相同轴向时引发 ValueError
        assert_raises(ValueError, expand_dims, a, axis=(1, 1))

    # 测试子类化的情况
    def test_subclasses(self):
        a = np.arange(10).reshape((2, 5))
        a = np.ma.array(a, mask=a % 3 == 0)

        # 对子类化的数组进行维度扩展
        expanded = np.expand_dims(a, axis=1)
        # 断言扩展后的对象仍然是 MaskedArray 类的实例
        assert_(isinstance(expanded, np.ma.MaskedArray))
        # 断言扩展后的数组形状
        assert_equal(expanded.shape, (2, 1, 5))
        # 断言扩展后的 mask 属性形状与数组形状相同
        assert_equal(expanded.mask.shape, (2, 1, 5))


class TestArraySplit:
    # 测试在分割数组时指定零分割点的情况
    def test_integer_0_split(self):
        a = np.arange(10)
        # 断言当指定分割点为零时，引发 ValueError
        assert_raises(ValueError, array_split, a, 0)
    # 定义一个测试方法，用于测试数组分割函数的不同情况
    def test_integer_split(self):
        # 创建一个包含 0 到 9 的数组
        a = np.arange(10)
        # 调用数组分割函数，将数组 a 分割成1个子数组
        res = array_split(a, 1)
        # 期望的结果是包含数组 a 整体的列表
        desired = [np.arange(10)]
        # 比较函数输出和期望结果是否一致
        compare_results(res, desired)

        # 将数组 a 分割成2个子数组
        res = array_split(a, 2)
        # 期望的结果是分别包含数组 a 的前5个元素和后5个元素的列表
        desired = [np.arange(5), np.arange(5, 10)]
        # 比较函数输出和期望结果是否一致
        compare_results(res, desired)

        # 将数组 a 分割成3个子数组
        res = array_split(a, 3)
        # 期望的结果是分别包含数组 a 的前4个元素、中间3个元素和最后3个元素的列表
        desired = [np.arange(4), np.arange(4, 7), np.arange(7, 10)]
        # 比较函数输出和期望结果是否一致
        compare_results(res, desired)

        # 将数组 a 分割成4个子数组
        res = array_split(a, 4)
        # 期望的结果是分别包含数组 a 的前3个元素、中间3个元素、接着的2个元素和最后2个元素的列表
        desired = [np.arange(3), np.arange(3, 6), np.arange(6, 8),
                   np.arange(8, 10)]
        # 比较函数输出和期望结果是否一致
        compare_results(res, desired)

        # 将数组 a 分割成5个子数组
        res = array_split(a, 5)
        # 期望的结果是分别包含数组 a 的前2个元素、接下来的2个元素、接着的2个元素、再接着的2个元素和最后2个元素的列表
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6),
                   np.arange(6, 8), np.arange(8, 10)]
        # 比较函数输出和期望结果是否一致
        compare_results(res, desired)

        # 将数组 a 分割成6个子数组
        res = array_split(a, 6)
        # 期望的结果是分别包含数组 a 的前2个元素、接下来的2个元素、接着的2个元素、再接着的2个元素、再接着的1个元素和最后1个元素的列表
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6),
                   np.arange(6, 8), np.arange(8, 9), np.arange(9, 10)]
        # 比较函数输出和期望结果是否一致
        compare_results(res, desired)

        # 将数组 a 分割成7个子数组
        res = array_split(a, 7)
        # 期望的结果是分别包含数组 a 的前2个元素、接下来的2个元素、接着的2个元素、再接着的1个元素、再接着的1个元素、再接着的1个元素和最后1个元素的列表
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6),
                   np.arange(6, 7), np.arange(7, 8), np.arange(8, 9),
                   np.arange(9, 10)]
        # 比较函数输出和期望结果是否一致
        compare_results(res, desired)

        # 将数组 a 分割成8个子数组
        res = array_split(a, 8)
        # 期望的结果是分别包含数组 a 的前2个元素、接下来的2个元素、接着的1个元素、再接着的1个元素、再接着的1个元素、再接着的1个元素、最后1个元素和最后1个元素的列表
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 5),
                   np.arange(5, 6), np.arange(6, 7), np.arange(7, 8),
                   np.arange(8, 9), np.arange(9, 10)]
        # 比较函数输出和期望结果是否一致
        compare_results(res, desired)

        # 将数组 a 分割成9个子数组
        res = array_split(a, 9)
        # 期望的结果是分别包含数组 a 的前2个元素、接下来的1个元素、再接着的1个元素、再接着的1个元素、再接着的1个元素、再接着的1个元素、再接着的1个元素、最后1个元素和最后1个元素的列表
        desired = [np.arange(2), np.arange(2, 3), np.arange(3, 4),
                   np.arange(4, 5), np.arange(5, 6), np.arange(6, 7),
                   np.arange(7, 8), np.arange(8, 9), np.arange(9, 10)]
        # 比较函数输出和期望结果是否一致
        compare_results(res, desired)

        # 将数组 a 分割成10个子数组
        res = array_split(a, 10)
        # 期望的结果是分别包含数组 a 的前1个元素、再接着的1个元素、再接着的1个元素、再接着的1个元素、再接着的1个元素、再接着的1个元素、再接着的1个元素、再接着的1个元素、最后1个元素和最后1个元素的列表
        desired = [np.arange(1), np.arange(1, 2), np.arange(2, 3),
                   np.arange(3, 4), np.arange(4, 5), np.arange(5, 6),
                   np.arange(6, 7), np.arange(7, 8), np.arange(8, 9),
                   np.arange(9, 10)]
        # 比较函数输出和期望结果是否一致
        compare_results(res, desired)

        # 将数组 a 分割成11个子数组
        res = array_split(a, 11)
        # 期望的结果是分别包含数组 a 的前1个元素、再接着的1个元素、再接着的1个元素、再接着的1个元素、再接着的1个元素、再接着的1个元素、再接着的1个元素、再接着的1个元素、最后1个元素和空数组的列表
        desired = [np.arange(1), np.arange(1, 2), np.arange(2, 3),
                   np.arange(3, 4), np.arange(4, 5), np.arange(5, 6),
                   np.arange(6, 7), np.arange(7, 8), np.arange(8, 9),
                   np.arange(9, 10), np.array([])]
        # 比较函数输出和期望结果是否一致
        compare_results(res, desired)
    # 定义一个测试方法，用于测试在二维数组中按行进行整数切分
    def test_integer_split_2D_rows(self):
        # 创建一个包含两行每行有 0 到 9 的二维 NumPy 数组
        a = np.array([np.arange(10), np.arange(10)])
        # 在第一个轴上将数组 `a` 分成 3 部分
        res = array_split(a, 3, axis=0)
        # 目标结果是包含三个元素的列表，分别是包含 0 到 9 的数组、另一个包含 0 到 9 的数组、以及一个 0 行 10 列的零数组
        tgt = [np.array([np.arange(10)]), np.array([np.arange(10)]),
                   np.zeros((0, 10))]
        # 比较函数，用于比较结果 `res` 和目标 `tgt`
        compare_results(res, tgt)
        # 断言 `a` 数组的数据类型与 `res` 列表中最后一个元素的数据类型相同
        assert_(a.dtype.type is res[-1].dtype.type)

        # 对手动切分也做同样的操作:
        # 在第一个轴上根据索引 [0, 1] 对数组 `a` 进行切分
        res = array_split(a, [0, 1], axis=0)
        # 目标结果是包含四个元素的列表，分别是一个空的 0 行 10 列的数组、一个包含 0 到 9 的数组、另一个包含 0 到 9 的数组
        tgt = [np.zeros((0, 10)), np.array([np.arange(10)]),
               np.array([np.arange(10)])]
        # 再次使用比较函数，比较结果 `res` 和目标 `tgt`
        compare_results(res, tgt)
        # 断言 `a` 数组的数据类型与 `res` 列表中最后一个元素的数据类型相同
        assert_(a.dtype.type is res[-1].dtype.type)

    # 定义一个测试方法，用于测试在二维数组中按列进行整数切分
    def test_integer_split_2D_cols(self):
        # 创建一个包含两行每行有 0 到 9 的二维 NumPy 数组
        a = np.array([np.arange(10), np.arange(10)])
        # 在最后一个轴（列）上将数组 `a` 分成 3 部分
        res = array_split(a, 3, axis=-1)
        # 目标结果是包含三个元素的列表，分别是包含 0 到 3 的两行两列数组、包含 4 到 6 的两行两列数组、包含 7 到 9 的两行两列数组
        desired = [np.array([np.arange(4), np.arange(4)]),
                   np.array([np.arange(4, 7), np.arange(4, 7)]),
                   np.array([np.arange(7, 10), np.arange(7, 10)])]
        # 使用比较函数，比较结果 `res` 和目标 `desired`
        compare_results(res, desired)

    # 定义一个测试方法，用于测试在二维数组中按默认轴进行整数切分
    def test_integer_split_2D_default(self):
        """ This will fail if we change default axis
        """
        # 创建一个包含两行每行有 0 到 9 的二维 NumPy 数组
        a = np.array([np.arange(10), np.arange(10)])
        # 在默认的第一个轴（行）上将数组 `a` 分成 3 部分
        res = array_split(a, 3)
        # 目标结果是包含三个元素的列表，分别是包含 0 到 9 的数组、另一个包含 0 到 9 的数组、以及一个 0 行 10 列的零数组
        tgt = [np.array([np.arange(10)]), np.array([np.arange(10)]),
                   np.zeros((0, 10))]
        # 使用比较函数，比较结果 `res` 和目标 `tgt`
        compare_results(res, tgt)
        # 断言 `a` 数组的数据类型与 `res` 列表中最后一个元素的数据类型相同
        assert_(a.dtype.type is res[-1].dtype.type)
        # 或许应该检查更高的维度

    # 根据条件跳过测试，如果不在 64 位平台上则跳过
    @pytest.mark.skipif(not IS_64BIT, reason="Needs 64bit platform")
    def test_integer_split_2D_rows_greater_max_int32(self):
        # 创建一个形状为 (2^32, 2) 的广播数组，元素值均为 0
        a = np.broadcast_to([0], (1 << 32, 2))
        # 在第一个轴上将数组 `a` 分成 4 部分
        res = array_split(a, 4)
        # 创建一个形状为 (2^30, 2) 的广播数组，元素值均为 0，作为目标结果的一个分块
        chunk = np.broadcast_to([0], (1 << 30, 2))
        # 目标结果是包含四个元素的列表，每个元素都是形状为 (2^30, 2) 的广播数组
        tgt = [chunk] * 4
        # 遍历目标结果，断言结果 `res` 中每个分块的形状与目标结果中对应分块的形状相同
        for i in range(len(tgt)):
            assert_equal(res[i].shape, tgt[i].shape)

    # 定义一个测试方法，用于测试在一维数组中按索引进行切分
    def test_index_split_simple(self):
        # 创建一个包含 0 到 9 的一维 NumPy 数组
        a = np.arange(10)
        # 根据给定的索引 [1, 5, 7] 在最后一个轴上将数组 `a` 分成多个子数组
        res = array_split(a, [1, 5, 7], axis=-1)
        # 目标结果是包含四个元素的列表，分别是从 0 到 1、从 1 到 5、从 5 到 7、从 7 到 10 的子数组
        desired = [np.arange(0, 1), np.arange(1, 5), np.arange(5, 7),
                   np.arange(7, 10)]
        # 使用比较函数，比较结果 `res` 和目标 `desired`
        compare_results(res, desired)

    # 定义一个测试方法，用于测试在一维数组中按索引进行切分，其中低边界为 0
    def test_index_split_low_bound(self):
        # 创建一个包含 0 到 9 的一维 NumPy 数组
        a = np.arange(10)
        # 根据给定的索引 [0, 5, 7] 在最后一个轴上将数组 `a` 分成多个子数组
        res = array_split(a, [0, 5, 7], axis=-1)
        # 目标结果是包含四个元素的列表，分别是空数组、从 0 到 5、从 5 到 7、从 7 到 10 的子数组
        desired = [np.array([]), np.arange(0, 5), np.arange(5, 7),
                   np.arange(7, 10)]
        # 使用比较函数，比较结果 `res` 和目标 `desired`
        compare_results(res, desired)

    # 定义一个测试方法，用于测试在一维数组中按索引进行切分，其中高边界为数组长度
    def test_index_split_high_bound(self):
        # 创建一个包含 0 到 9 的一维 NumPy 数组
        a = np.arange(10)
        # 根据给定的索引 [0, 5, 7, 10, 12] 在最后一个轴上将数组 `a` 分成多个子数组
        res = array_split(a, [0, 5, 7, 10, 12], axis=-1)
        # 目标结果
class TestSplit:
    # The split function is essentially the same as array_split,
    # except that it test if splitting will result in an
    # equal split.  Only test for this case.

    def test_equal_split(self):
        # Create a NumPy array with values from 0 to 9
        a = np.arange(10)
        # Call the split function with array 'a' and split count 2
        res = split(a, 2)
        # Define the desired result as two arrays: [0, 1, 2, 3, 4] and [5, 6, 7, 8, 9]
        desired = [np.arange(5), np.arange(5, 10)]
        # Compare the result of split operation with the desired result
        compare_results(res, desired)

    def test_unequal_split(self):
        # Create a NumPy array with values from 0 to 9
        a = np.arange(10)
        # Assert that splitting array 'a' into 3 parts raises a ValueError
        assert_raises(ValueError, split, a, 3)


class TestColumnStack:
    def test_non_iterable(self):
        # Assert that passing a non-iterable argument (like '1') to column_stack raises a TypeError
        assert_raises(TypeError, column_stack, 1)

    def test_1D_arrays(self):
        # example from docstring
        # Create two 1-dimensional NumPy arrays 'a' and 'b'
        a = np.array((1, 2, 3))
        b = np.array((2, 3, 4))
        # Define the expected result of column_stack operation
        expected = np.array([[1, 2],
                             [2, 3],
                             [3, 4]])
        # Perform column_stack operation on arrays 'a' and 'b'
        actual = np.column_stack((a, b))
        # Assert that the actual result matches the expected result
        assert_equal(actual, expected)

    def test_2D_arrays(self):
        # same as hstack 2D docstring example
        # Create two 2-dimensional NumPy arrays 'a' and 'b'
        a = np.array([[1], [2], [3]])
        b = np.array([[2], [3], [4]])
        # Define the expected result of column_stack operation
        expected = np.array([[1, 2],
                             [2, 3],
                             [3, 4]])
        # Perform column_stack operation on arrays 'a' and 'b'
        actual = np.column_stack((a, b))
        # Assert that the actual result matches the expected result
        assert_equal(actual, expected)

    def test_generator(self):
        # Assert that passing a generator of arrays to column_stack raises a TypeError with a specific error message
        with pytest.raises(TypeError, match="arrays to stack must be"):
            column_stack((np.arange(3) for _ in range(2)))


class TestDstack:
    def test_non_iterable(self):
        # Assert that passing a non-iterable argument (like '1') to dstack raises a TypeError
        assert_raises(TypeError, dstack, 1)

    def test_0D_array(self):
        # Create a 0-dimensional NumPy array 'a'
        a = np.array(1)
        # Create another 0-dimensional NumPy array 'b'
        b = np.array(2)
        # Perform dstack operation on arrays 'a' and 'b'
        res = dstack([a, b])
        # Define the desired result of dstack operation
        desired = np.array([[[1, 2]]])
        # Assert that the result of dstack operation matches the desired result
        assert_array_equal(res, desired)

    def test_1D_array(self):
        # Create two 1-dimensional NumPy arrays 'a' and 'b'
        a = np.array([1])
        b = np.array([2])
        # Perform dstack operation on arrays 'a' and 'b'
        res = dstack([a, b])
        # Define the desired result of dstack operation
        desired = np.array([[[1, 2]]])
        # Assert that the result of dstack operation matches the desired result
        assert_array_equal(res, desired)

    def test_2D_array(self):
        # Create two 2-dimensional NumPy arrays 'a' and 'b'
        a = np.array([[1], [2]])
        b = np.array([[1], [2]])
        # Perform dstack operation on arrays 'a' and 'b'
        res = dstack([a, b])
        # Define the desired result of dstack operation
        desired = np.array([[[1, 1]], [[2, 2, ]]])
        # Assert that the result of dstack operation matches the desired result
        assert_array_equal(res, desired)

    def test_2D_array2(self):
        # Create two 1-dimensional NumPy arrays 'a' and 'b'
        a = np.array([1, 2])
        b = np.array([1, 2])
        # Perform dstack operation on arrays 'a' and 'b'
        res = dstack([a, b])
        # Define the desired result of dstack operation
        desired = np.array([[[1, 1], [2, 2]]])
        # Assert that the result of dstack operation matches the desired result
        assert_array_equal(res, desired)

    def test_generator(self):
        # Assert that passing a generator of arrays to dstack raises a TypeError with a specific error message
        with pytest.raises(TypeError, match="arrays to stack must be"):
            dstack((np.arange(3) for _ in range(2)))


# array_split has more comprehensive test of splitting.
# only do simple test on hsplit, vsplit, and dsplit
class TestHsplit:
    """Only testing for integer splits.

    """
    def test_non_iterable(self):
        # Assert that passing a non-iterable argument (like '1') to hsplit raises a ValueError
        assert_raises(ValueError, hsplit, 1, 1)

    def test_0D_array(self):
        # Create a 0-dimensional NumPy array 'a'
        a = np.array(1)
        try:
            # Attempt to perform hsplit operation on array 'a' with split count 2
            hsplit(a, 2)
            # If successful, assert an unexpected condition to fail the test
            assert_(0)
        except ValueError:
            # Catch the expected ValueError if hsplit operation fails
            pass
    # 定义一个测试函数，用于测试对一维数组进行水平分割的功能
    def test_1D_array(self):
        # 创建一个包含整数1到4的一维 NumPy 数组
        a = np.array([1, 2, 3, 4])
        # 调用 hsplit 函数对数组进行水平分割，分割成两部分
        res = hsplit(a, 2)
        # 预期的分割结果，分别是包含[1, 2]和[3, 4]的两个数组
        desired = [np.array([1, 2]), np.array([3, 4])]
        # 调用 compare_results 函数比较实际结果和预期结果
        compare_results(res, desired)
    
    # 定义一个测试函数，用于测试对二维数组进行水平分割的功能
    def test_2D_array(self):
        # 创建一个包含两行四列的二维 NumPy 数组
        a = np.array([[1, 2, 3, 4],
                      [1, 2, 3, 4]])
        # 调用 hsplit 函数对数组进行水平分割，分割成两部分
        res = hsplit(a, 2)
        # 预期的分割结果，分别是包含[[1, 2], [1, 2]]和[[3, 4], [3, 4]]的两个数组
        desired = [np.array([[1, 2], [1, 2]]), np.array([[3, 4], [3, 4]])]
        # 调用 compare_results 函数比较实际结果和预期结果
        compare_results(res, desired)
class TestVsplit:
    """Only testing for integer splits.
    
    Test class for verifying vsplit function behavior.
    """
    
    def test_non_iterable(self):
        # Assert that vsplit raises a ValueError when provided with non-iterable inputs
        assert_raises(ValueError, vsplit, 1, 1)
        
    def test_0D_array(self):
        # Create a 0-dimensional NumPy array
        a = np.array(1)
        # Assert that vsplit raises a ValueError when applied to a 0-dimensional array
        assert_raises(ValueError, vsplit, a, 2)
        
    def test_1D_array(self):
        # Create a 1-dimensional NumPy array
        a = np.array([1, 2, 3, 4])
        try:
            # Attempt to split the array into 2 parts using vsplit
            vsplit(a, 2)
            # Assert a failure because vsplit should raise a ValueError
            assert_(0)
        except ValueError:
            pass
    
    def test_2D_array(self):
        # Create a 2-dimensional NumPy array
        a = np.array([[1, 2, 3, 4],
                      [1, 2, 3, 4]])
        # Split the array into 2 vertical parts using vsplit
        res = vsplit(a, 2)
        # Define the expected result as a list of 2-dimensional arrays
        desired = [np.array([[1, 2, 3, 4]]), np.array([[1, 2, 3, 4]])]
        # Compare the actual result with the expected result
        compare_results(res, desired)


class TestDsplit:
    # Only testing for integer splits.
    """Test class for verifying dsplit function behavior."""

    def test_non_iterable(self):
        # Assert that dsplit raises a ValueError when provided with non-iterable inputs
        assert_raises(ValueError, dsplit, 1, 1)
        
    def test_0D_array(self):
        # Create a 0-dimensional NumPy array
        a = np.array(1)
        # Assert that dsplit raises a ValueError when applied to a 0-dimensional array
        assert_raises(ValueError, dsplit, a, 2)
        
    def test_1D_array(self):
        # Create a 1-dimensional NumPy array
        a = np.array([1, 2, 3, 4])
        # Assert that dsplit raises a ValueError when applied to a 1-dimensional array
        assert_raises(ValueError, dsplit, a, 2)
        
    def test_2D_array(self):
        # Create a 2-dimensional NumPy array
        a = np.array([[1, 2, 3, 4],
                      [1, 2, 3, 4]])
        try:
            # Attempt to split the array into 2 parts using dsplit
            dsplit(a, 2)
            # Assert a failure because dsplit should raise a ValueError
            assert_(0)
        except ValueError:
            pass
    
    def test_3D_array(self):
        # Create a 3-dimensional NumPy array
        a = np.array([[[1, 2, 3, 4],
                       [1, 2, 3, 4]],
                      [[1, 2, 3, 4],
                       [1, 2, 3, 4]]])
        # Split the array into 2 parts along the third dimension using dsplit
        res = dsplit(a, 2)
        # Define the expected result as a list of 3-dimensional arrays
        desired = [np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]]),
                   np.array([[[3, 4], [3, 4]], [[3, 4], [3, 4]]])]
        # Compare the actual result with the expected result
        compare_results(res, desired)


class TestSqueeze:
    """Test class for verifying squeeze function behavior."""
    
    def test_basic(self):
        # Importing necessary function from numpy.random
        from numpy.random import rand
        
        # Generate random arrays with varied dimensions
        a = rand(20, 10, 10, 1, 1)
        b = rand(20, 1, 10, 1, 20)
        c = rand(1, 1, 20, 10)
        
        # Assert that squeezing these arrays gives expected reshaped results
        assert_array_equal(np.squeeze(a), np.reshape(a, (20, 10, 10)))
        assert_array_equal(np.squeeze(b), np.reshape(b, (20, 10, 20)))
        assert_array_equal(np.squeeze(c), np.reshape(c, (20, 10)))
        
        # Test squeezing a 0-dimensional array to ensure it remains an ndarray
        a = [[[1.5]]]
        res = np.squeeze(a)
        assert_equal(res, 1.5)
        assert_equal(res.ndim, 0)
        assert_equal(type(res), np.ndarray)
    def test_basic(self):
        # 使用0维的ndarray
        a = np.array(1)  # 创建一个0维的ndarray，值为1
        b = np.array([[1, 2], [3, 4]])  # 创建一个2x2的二维ndarray
        k = np.array([[1, 2], [3, 4]])  # k被赋值为和b相同的二维ndarray
        assert_array_equal(np.kron(a, b), k)  # 断言np.kron(a, b)和k相等
        a = np.array([[1, 2], [3, 4]])  # 更新a为一个2x2的二维ndarray
        b = np.array(1)  # 更新b为一个0维的ndarray，值为1
        assert_array_equal(np.kron(a, b), k)  # 再次断言np.kron(a, b)和k相等

        # 使用1维的ndarray
        a = np.array([3])  # 创建一个包含单个元素3的1维ndarray
        b = np.array([[1, 2], [3, 4]])  # 创建一个2x2的二维ndarray
        k = np.array([[3, 6], [9, 12]])  # k被赋值为相应的2x2的二维ndarray
        assert_array_equal(np.kron(a, b), k)  # 断言np.kron(a, b)和k相等
        a = np.array([[1, 2], [3, 4]])  # 更新a为一个2x2的二维ndarray
        b = np.array([3])  # 更新b为一个包含单个元素3的1维ndarray
        assert_array_equal(np.kron(a, b), k)  # 再次断言np.kron(a, b)和k相等

        # 使用3维的ndarray
        a = np.array([[[1]], [[2]]])  # 创建一个包含两个元素的3维ndarray
        b = np.array([[1, 2], [3, 4]])  # 创建一个2x2的二维ndarray
        k = np.array([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])  # k被赋值为相应的3维ndarray
        assert_array_equal(np.kron(a, b), k)  # 断言np.kron(a, b)和k相等
        a = np.array([[1, 2], [3, 4]])  # 更新a为一个2x2的二维ndarray
        b = np.array([[[1]], [[2]]])  # 更新b为一个包含两个元素的3维ndarray
        k = np.array([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])  # k被赋值为相应的3维ndarray
        assert_array_equal(np.kron(a, b), k)  # 再次断言np.kron(a, b)和k相等

    def test_return_type(self):
        class myarray(np.ndarray):
            __array_priority__ = 1.0

        a = np.ones([2, 2])  # 创建一个2x2的全1数组
        ma = myarray(a.shape, a.dtype, a.data)  # 使用自定义类创建一个数组
        assert_equal(type(kron(a, a)), np.ndarray)  # 断言kron(a, a)的返回类型是np.ndarray
        assert_equal(type(kron(ma, ma)), myarray)  # 断言kron(ma, ma)的返回类型是myarray
        assert_equal(type(kron(a, ma)), myarray)  # 断言kron(a, ma)的返回类型是myarray
        assert_equal(type(kron(ma, a)), myarray)  # 断言kron(ma, a)的返回类型是myarray

    @pytest.mark.parametrize(
        "array_class", [np.asarray, np.asmatrix]
    )
    def test_kron_smoke(self, array_class):
        a = array_class(np.ones([3, 3]))  # 根据array_class创建一个3x3的全1数组
        b = array_class(np.ones([3, 3]))  # 根据array_class创建另一个3x3的全1数组
        k = array_class(np.ones([9, 9]))  # 根据array_class创建一个9x9的全1数组

        assert_array_equal(np.kron(a, b), k)  # 断言np.kron(a, b)和k相等

    def test_kron_ma(self):
        x = np.ma.array([[1, 2], [3, 4]], mask=[[0, 1], [1, 0]])  # 创建一个带有掩码的掩盖数组
        k = np.ma.array(np.diag([1, 4, 4, 16]), mask=~np.array(np.identity(4), dtype=bool))  # 创建一个带有掩码的掩盖数组

        assert_array_equal(k, np.kron(x, x))  # 断言np.kron(x, x)和k相等

    @pytest.mark.parametrize(
        "shape_a,shape_b", [
            ((1, 1), (1, 1)),
            ((1, 2, 3), (4, 5, 6)),
            ((2, 2), (2, 2, 2)),
            ((1, 0), (1, 1)),
            ((2, 0, 2), (2, 2)),
            ((2, 0, 0, 2), (2, 0, 2)),
        ])
    def test_kron_shape(self, shape_a, shape_b):
        a = np.ones(shape_a)  # 创建一个形状为shape_a的全1数组
        b = np.ones(shape_b)  # 创建一个形状为shape_b的全1数组
        normalised_shape_a = (1,) * max(0, len(shape_b)-len(shape_a)) + shape_a  # 规范化shape_a的形状
        normalised_shape_b = (1,) * max(0, len(shape_a)-len(shape_b)) + shape_b  # 规范化shape_b的形状
        expected_shape = np.multiply(normalised_shape_a, normalised_shape_b)  # 计算期望的形状

        k = np.kron(a, b)  # 计算np.kron(a, b)
        assert np.array_equal(
                k.shape, expected_shape), "Unexpected shape from kron"  # 断言k的形状与期望的形状相等
class TestTile:
    def test_basic(self):
        # 创建一个包含元素 [0, 1, 2] 的 NumPy 数组
        a = np.array([0, 1, 2])
        # 创建一个包含列表 [[1, 2], [3, 4]] 的 Python 列表
        b = [[1, 2], [3, 4]]
        # 测试 tile 函数对数组 a 的重复操作是否正确
        assert_equal(tile(a, 2), [0, 1, 2, 0, 1, 2])
        # 测试 tile 函数对数组 a 的 (2, 2) 形状的重复操作是否正确
        assert_equal(tile(a, (2, 2)), [[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]])
        # 测试 tile 函数对数组 a 的 (1, 2) 形状的重复操作是否正确
        assert_equal(tile(a, (1, 2)), [[0, 1, 2, 0, 1, 2]])
        # 测试 tile 函数对列表 b 的 2 次重复操作是否正确
        assert_equal(tile(b, 2), [[1, 2, 1, 2], [3, 4, 3, 4]])
        # 测试 tile 函数对列表 b 的 (2, 1) 形状的重复操作是否正确
        assert_equal(tile(b, (2, 1)), [[1, 2], [3, 4], [1, 2], [3, 4]])
        # 测试 tile 函数对列表 b 的 (2, 2) 形状的重复操作是否正确
        assert_equal(tile(b, (2, 2)), [[1, 2, 1, 2], [3, 4, 3, 4],
                                       [1, 2, 1, 2], [3, 4, 3, 4]])

    def test_tile_one_repetition_on_array_gh4679(self):
        # 创建一个从 0 到 4 的 NumPy 数组
        a = np.arange(5)
        # 对数组 a 进行 1 次重复操作，并将结果保存到数组 b
        b = tile(a, 1)
        # 将数组 b 中的所有元素加上 2
        b += 2
        # 断言数组 a 未被修改
        assert_equal(a, np.arange(5))

    def test_empty(self):
        # 创建一个包含空列表的 NumPy 数组
        a = np.array([[[]]])
        # 创建一个包含两个空列表的 NumPy 数组
        b = np.array([[], []])
        # 对数组 b 进行 2 次重复操作，并获取其形状
        c = tile(b, 2).shape
        # 对数组 a 进行 (3, 2, 5) 形状的重复操作，并获取其形状
        d = tile(a, (3, 2, 5)).shape
        # 断言 c 的形状为 (2, 0)
        assert_equal(c, (2, 0))
        # 断言 d 的形状为 (3, 2, 0)
        assert_equal(d, (3, 2, 0))

    def test_kroncompare(self):
        # 从 numpy.random 模块导入 randint 函数
        from numpy.random import randint

        # 定义重复操作的形状和数组的形状
        reps = [(2,), (1, 2), (2, 1), (2, 2), (2, 3, 2), (3, 2)]
        shape = [(3,), (2, 3), (3, 4, 3), (3, 2, 3), (4, 3, 2, 4), (2, 2)]
        # 遍历数组的形状
        for s in shape:
            # 生成指定形状的随机数组 b
            b = randint(0, 10, size=s)
            # 遍历重复操作的形状
            for r in reps:
                # 创建与数组 b 类型相同的全为 1 的数组 a
                a = np.ones(r, b.dtype)
                # 使用 tile 函数对数组 b 进行 r 形状的重复操作
                large = tile(b, r)
                # 使用 kron 函数对数组 a 和 b 进行 Kroncker 乘积
                klarge = kron(a, b)
                # 断言两者相等
                assert_equal(large, klarge)


class TestMayShareMemory:
    def test_basic(self):
        # 创建一个形状为 (50, 60) 的全为 1 的 NumPy 数组 d
        d = np.ones((50, 60))
        # 创建一个形状为 (30, 60, 6) 的全为 1 的 NumPy 数组 d2
        d2 = np.ones((30, 60, 6))
        # 断言数组 d 与自身共享内存
        assert_(np.may_share_memory(d, d))
        # 断言数组 d 与 d 的倒序视图不共享内存
        assert_(np.may_share_memory(d, d[::-1]))
        # 断言数组 d 与 d 的每隔一个元素视图不共享内存
        assert_(np.may_share_memory(d, d[::2]))
        # 断言数组 d 与 d 的行倒序视图不共享内存
        assert_(np.may_share_memory(d, d[1:, ::-1]))

        # 断言数组 d 的倒序视图与数组 d2 不共享内存
        assert_(not np.may_share_memory(d[::-1], d2))
        # 断言数组 d 的每隔一个元素视图与数组 d2 不共享内存
        assert_(not np.may_share_memory(d[::2], d2))
        # 断言数组 d 的行倒序视图与数组 d2 不共享内存
        assert_(not np.may_share_memory(d[1:, ::-1], d2))
        # 断言数组 d2 的行倒序视图与自身共享内存
        assert_(np.may_share_memory(d2[1:, ::-1], d2))


# Utility
def compare_results(res, desired):
    """Compare lists of arrays."""
    # 如果 res 和 desired 的长度不一致，则抛出 ValueError 异常
    if len(res) != len(desired):
        raise ValueError("Iterables have different lengths")
    # 遍历 res 和 desired 中的每对数组，并使用 assert_array_equal 断言它们相等
    for x, y in zip(res, desired):
        assert_array_equal(x, y)
```