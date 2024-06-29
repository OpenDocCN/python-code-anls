# `.\numpy\numpy\_core\tests\test_shape_base.py`

```py
import pytest  # 导入 pytest 库，用于单元测试
import numpy as np  # 导入 NumPy 库，用于数值计算和数组操作
from numpy._core import (  # 导入 NumPy 的核心功能模块
    array, arange, atleast_1d, atleast_2d, atleast_3d, block, vstack, hstack,
    newaxis, concatenate, stack
    )
from numpy.exceptions import AxisError  # 导入 NumPy 异常处理模块中的 AxisError 异常
from numpy._core.shape_base import (_block_dispatcher, _block_setup,  # 导入 NumPy 核心的形状基础功能模块
                                   _block_concatenate, _block_slicing)
from numpy.testing import (  # 导入 NumPy 测试模块的一些测试函数和常量
    assert_, assert_raises, assert_array_equal, assert_equal,
    assert_raises_regex, assert_warns, IS_PYPY
    )


class TestAtleast1d:
    def test_0D_array(self):
        a = array(1)  # 创建一个标量数组
        b = array(2)  # 创建一个标量数组
        res = [atleast_1d(a), atleast_1d(b)]  # 将输入至少转换为 1 维数组
        desired = [array([1]), array([2])]  # 预期的结果数组
        assert_array_equal(res, desired)  # 断言 res 和 desired 数组相等

    def test_1D_array(self):
        a = array([1, 2])  # 创建一个一维数组
        b = array([2, 3])  # 创建一个一维数组
        res = [atleast_1d(a), atleast_1d(b)]  # 将输入至少转换为 1 维数组
        desired = [array([1, 2]), array([2, 3])]  # 预期的结果数组
        assert_array_equal(res, desired)  # 断言 res 和 desired 数组相等

    def test_2D_array(self):
        a = array([[1, 2], [1, 2]])  # 创建一个二维数组
        b = array([[2, 3], [2, 3]])  # 创建一个二维数组
        res = [atleast_1d(a), atleast_1d(b)]  # 将输入至少转换为 1 维数组
        desired = [a, b]  # 预期的结果数组
        assert_array_equal(res, desired)  # 断言 res 和 desired 数组相等

    def test_3D_array(self):
        a = array([[1, 2], [1, 2]])  # 创建一个二维数组
        b = array([[2, 3], [2, 3]])  # 创建一个二维数组
        a = array([a, a])  # 创建一个三维数组
        b = array([b, b])  # 创建一个三维数组
        res = [atleast_1d(a), atleast_1d(b)]  # 将输入至少转换为 1 维数组
        desired = [a, b]  # 预期的结果数组
        assert_array_equal(res, desired)  # 断言 res 和 desired 数组相等

    def test_r1array(self):
        """ Test to make sure equivalent Travis O's r1array function
        """
        assert_(atleast_1d(3).shape == (1,))  # 断言至少将标量 3 转换为 1 维数组后的形状
        assert_(atleast_1d(3j).shape == (1,))  # 断言至少将复数标量 3j 转换为 1 维数组后的形状
        assert_(atleast_1d(3.0).shape == (1,))  # 断言至少将浮点数标量 3.0 转换为 1 维数组后的形状
        assert_(atleast_1d([[2, 3], [4, 5]]).shape == (2, 2))  # 断言至少将二维数组转换为 1 维数组后的形状


class TestAtleast2d:
    def test_0D_array(self):
        a = array(1)  # 创建一个标量数组
        b = array(2)  # 创建一个标量数组
        res = [atleast_2d(a), atleast_2d(b)]  # 将输入至少转换为 2 维数组
        desired = [array([[1]]), array([[2]])]  # 预期的结果数组
        assert_array_equal(res, desired)  # 断言 res 和 desired 数组相等

    def test_1D_array(self):
        a = array([1, 2])  # 创建一个一维数组
        b = array([2, 3])  # 创建一个一维数组
        res = [atleast_2d(a), atleast_2d(b)]  # 将输入至少转换为 2 维数组
        desired = [array([[1, 2]]), array([[2, 3]])]  # 预期的结果数组
        assert_array_equal(res, desired)  # 断言 res 和 desired 数组相等

    def test_2D_array(self):
        a = array([[1, 2], [1, 2]])  # 创建一个二维数组
        b = array([[2, 3], [2, 3]])  # 创建一个二维数组
        res = [atleast_2d(a), atleast_2d(b)]  # 将输入至少转换为 2 维数组
        desired = [a, b]  # 预期的结果数组
        assert_array_equal(res, desired)  # 断言 res 和 desired 数组相等

    def test_3D_array(self):
        a = array([[1, 2], [1, 2]])  # 创建一个二维数组
        b = array([[2, 3], [2, 3]])  # 创建一个二维数组
        a = array([a, a])  # 创建一个三维数组
        b = array([b, b])  # 创建一个三维数组
        res = [atleast_2d(a), atleast_2d(b)]  # 将输入至少转换为 2 维数组
        desired = [a, b]  # 预期的结果数组
        assert_array_equal(res, desired)  # 断言 res 和 desired 数组相等

    def test_r2array(self):
        """ Test to make sure equivalent Travis O's r2array function
        """
        assert_(atleast_2d(3).shape == (1, 1))  # 断言至少将标量 3 转换为 2 维数组后的形状
        assert_(atleast_2d([3j, 1]).shape == (1, 2))  # 断言至少将一维数组转换为 2 维数组后的形状
        assert_(atleast_2d([[[3, 1], [4, 5]], [[3, 5], [1, 2]]]).shape == (2, 2, 2))  # 断言至少将三维数组转换为 2 维数组后的形状


class TestAtleast3d:
    pass  # TestAtleast3d 类还未实现，因此暂不添加任何代码
    # 定义测试函数，用于测试处理零维数组的情况
    def test_0D_array(self):
        # 创建包含单个元素的一维数组 a
        a = array(1)
        # 创建包含单个元素的一维数组 b
        b = array(2)
        # 对数组 a 和 b 分别应用 atleast_3d 函数，返回结果列表 res
        res = [atleast_3d(a), atleast_3d(b)]
        # 预期结果列表 desired，包含将单个元素转换为三维数组的结果
        desired = [array([[[1]]]), array([[[2]]])]
        # 使用 assert_array_equal 函数比较 res 和 desired 是否相等
        assert_array_equal(res, desired)

    # 定义测试函数，用于测试处理一维数组的情况
    def test_1D_array(self):
        # 创建包含两个元素的一维数组 a 和 b
        a = array([1, 2])
        b = array([2, 3])
        # 对数组 a 和 b 分别应用 atleast_3d 函数，返回结果列表 res
        res = [atleast_3d(a), atleast_3d(b)]
        # 预期结果列表 desired，包含将一维数组转换为三维数组的结果
        desired = [array([[[1], [2]]]), array([[[2], [3]]])]
        # 使用 assert_array_equal 函数比较 res 和 desired 是否相等
        assert_array_equal(res, desired)

    # 定义测试函数，用于测试处理二维数组的情况
    def test_2D_array(self):
        # 创建一个二维数组 a 和 b
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        # 对数组 a 和 b 分别应用 atleast_3d 函数，返回结果列表 res
        res = [atleast_3d(a), atleast_3d(b)]
        # 预期结果列表 desired，包含将二维数组转换为三维数组的结果
        desired = [a[:,:, newaxis], b[:,:, newaxis]]
        # 使用 assert_array_equal 函数比较 res 和 desired 是否相等
        assert_array_equal(res, desired)

    # 定义测试函数，用于测试处理三维数组的情况
    def test_3D_array(self):
        # 创建一个二维数组 a 和 b
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        # 将二维数组 a 和 b 分别存储为三维数组的元素
        a = array([a, a])
        b = array([b, b])
        # 对数组 a 和 b 分别应用 atleast_3d 函数，返回结果列表 res
        res = [atleast_3d(a), atleast_3d(b)]
        # 预期结果列表 desired，直接使用数组 a 和 b 作为三维数组的结果
        desired = [a, b]
        # 使用 assert_array_equal 函数比较 res 和 desired 是否相等
        assert_array_equal(res, desired)
class TestHstack:
    # 测试非可迭代输入的情况，期望抛出 TypeError 异常
    def test_non_iterable(self):
        assert_raises(TypeError, hstack, 1)

    # 测试空输入的情况，期望抛出 ValueError 异常
    def test_empty_input(self):
        assert_raises(ValueError, hstack, ())

    # 测试 0 维数组的堆叠
    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = hstack([a, b])
        desired = array([1, 2])
        assert_array_equal(res, desired)

    # 测试 1 维数组的堆叠
    def test_1D_array(self):
        a = array([1])
        b = array([2])
        res = hstack([a, b])
        desired = array([1, 2])
        assert_array_equal(res, desired)

    # 测试 2 维数组的堆叠
    def test_2D_array(self):
        a = array([[1], [2]])
        b = array([[1], [2]])
        res = hstack([a, b])
        desired = array([[1, 1], [2, 2]])
        assert_array_equal(res, desired)

    # 测试生成器作为输入的情况，期望抛出 TypeError 异常
    def test_generator(self):
        with pytest.raises(TypeError, match="arrays to stack must be"):
            hstack((np.arange(3) for _ in range(2)))
        with pytest.raises(TypeError, match="arrays to stack must be"):
            hstack(map(lambda x: x, np.ones((3, 2))))

    # 测试指定类型转换和数据类型的情况
    def test_casting_and_dtype(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        res = np.hstack((a, b), casting="unsafe", dtype=np.int64)
        expected_res = np.array([1, 2, 3, 2, 3, 4])
        assert_array_equal(res, expected_res)
    
    # 测试类型转换和数据类型错误的情况，期望抛出 TypeError 异常
    def test_casting_and_dtype_type_error(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        with pytest.raises(TypeError):
            hstack((a, b), casting="safe", dtype=np.int64)


class TestVstack:
    # 测试非可迭代输入的情况，期望抛出 TypeError 异常
    def test_non_iterable(self):
        assert_raises(TypeError, vstack, 1)

    # 测试空输入的情况，期望抛出 ValueError 异常
    def test_empty_input(self):
        assert_raises(ValueError, vstack, ())

    # 测试 0 维数组的堆叠
    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = vstack([a, b])
        desired = array([[1], [2]])
        assert_array_equal(res, desired)

    # 测试 1 维数组的堆叠
    def test_1D_array(self):
        a = array([1])
        b = array([2])
        res = vstack([a, b])
        desired = array([[1], [2]])
        assert_array_equal(res, desired)

    # 测试 2 维数组的堆叠
    def test_2D_array(self):
        a = array([[1], [2]])
        b = array([[1], [2]])
        res = vstack([a, b])
        desired = array([[1], [2], [1], [2]])
        assert_array_equal(res, desired)

    # 测试另一种 2 维数组的堆叠方式
    def test_2D_array2(self):
        a = array([1, 2])
        b = array([1, 2])
        res = vstack([a, b])
        desired = array([[1, 2], [1, 2]])
        assert_array_equal(res, desired)

    # 测试生成器作为输入的情况，期望抛出 TypeError 异常
    def test_generator(self):
        with pytest.raises(TypeError, match="arrays to stack must be"):
            vstack((np.arange(3) for _ in range(2)))

    # 测试指定类型转换和数据类型的情况
    def test_casting_and_dtype(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        res = np.vstack((a, b), casting="unsafe", dtype=np.int64)
        expected_res = np.array([[1, 2, 3], [2, 3, 4]])
        assert_array_equal(res, expected_res)
    # 定义一个测试函数，用于测试类型转换和数据类型错误的情况
    def test_casting_and_dtype_type_error(self):
        # 创建一个包含整数的 NumPy 数组 a
        a = np.array([1, 2, 3])
        # 创建一个包含浮点数的 NumPy 数组 b
        b = np.array([2.5, 3.5, 4.5])
        # 使用 pytest 模块检测是否会引发 TypeError 异常
        with pytest.raises(TypeError):
            # 尝试在类型安全且数据类型为 np.int64 的情况下堆叠数组 a 和 b
            vstack((a, b), casting="safe", dtype=np.int64)
class TestConcatenate:
    # 定义测试方法，验证 np.concatenate 返回的是副本而非引用
    def test_returns_copy(self):
        # 创建一个 3x3 的单位矩阵 a
        a = np.eye(3)
        # 对 a 进行沿 axis=0 方向的连接，得到 b
        b = np.concatenate([a])
        # 修改 b 的第一个元素为 2
        b[0, 0] = 2
        # 断言 b 的修改不影响原始矩阵 a 的相同位置元素
        assert b[0, 0] != a[0, 0]

    # 定义测试方法，验证 np.concatenate 对各种异常情况的处理
    def test_exceptions(self):
        # 测试 axis 超出范围的情况
        for ndim in [1, 2, 3]:
            # 创建 ndim 维度为 1 的全 1 数组 a
            a = np.ones((1,)*ndim)
            # 沿 axis=0 连接 a 和 a，预期不会抛出异常
            np.concatenate((a, a), axis=0)  # OK
            # 断言沿指定 axis 超出范围时会抛出 AxisError 异常
            assert_raises(AxisError, np.concatenate, (a, a), axis=ndim)
            assert_raises(AxisError, np.concatenate, (a, a), axis=-(ndim + 1))

        # 测试无法连接标量的情况
        assert_raises(ValueError, concatenate, (0,))
        assert_raises(ValueError, concatenate, (np.array(0),))

        # 测试连接的数组维度不匹配的情况
        assert_raises_regex(
            ValueError,
            r"all the input arrays must have same number of dimensions, but "
            r"the array at index 0 has 1 dimension\(s\) and the array at "
            r"index 1 has 2 dimension\(s\)",
            np.concatenate, (np.zeros(1), np.zeros((1, 1))))

        # 测试除连接轴外其他维度不匹配的情况
        a = np.ones((1, 2, 3))
        b = np.ones((2, 2, 3))
        axis = list(range(3))
        for i in range(3):
            # 沿 axis=i 连接 a 和 b，预期不会抛出异常
            np.concatenate((a, b), axis=axis[0])  # OK
            # 断言在指定轴上维度不匹配时会抛出 ValueError 异常
            assert_raises_regex(
                ValueError,
                "all the input array dimensions except for the concatenation axis "
                "must match exactly, but along dimension {}, the array at "
                "index 0 has size 1 and the array at index 1 has size 2"
                .format(i),
                np.concatenate, (a, b), axis=axis[1])
            assert_raises(ValueError, np.concatenate, (a, b), axis=axis[2])
            a = np.moveaxis(a, -1, 0)
            b = np.moveaxis(b, -1, 0)
            axis.append(axis.pop(0))

        # 测试空输入时抛出 ValueError 异常
        assert_raises(ValueError, concatenate, ())

    # 定义测试方法，验证 np.concatenate 在 axis=None 时的行为
    def test_concatenate_axis_None(self):
        # 创建一个 2x2 的浮点数矩阵 a
        a = np.arange(4, dtype=np.float64).reshape((2, 2))
        # 创建一个包含 [0, 1, 2] 的列表 b
        b = list(range(3))
        # 创建一个包含单个字符串 'x' 的列表 c
        c = ['x']
        
        # 测试在 axis=None 时，返回的数组类型与输入矩阵 a 的类型相同
        r = np.concatenate((a, a), axis=None)
        assert_equal(r.dtype, a.dtype)
        # 断言返回的数组维度为 1
        assert_equal(r.ndim, 1)
        
        # 测试在 axis=None 时，返回的数组大小为 a 的大小加上 b 的长度
        r = np.concatenate((a, b), axis=None)
        assert_equal(r.size, a.size + len(b))
        assert_equal(r.dtype, a.dtype)
        
        # 测试在 axis=None 时，使用 dtype="U" 连接 a、b、c
        r = np.concatenate((a, b, c), axis=None, dtype="U")
        d = np.array(['0.0', '1.0', '2.0', '3.0',
                      '0', '1', '2', 'x'])
        assert_array_equal(r, d)
        
        # 测试在 axis=None 时，使用 out 参数将结果连接到预先分配的数组
        out = np.zeros(a.size + len(b))
        r = np.concatenate((a, b), axis=None)
        rout = np.concatenate((a, b), axis=None, out=out)
        assert_(out is rout)
        assert_equal(r, rout)
    def test_large_concatenate_axis_None(self):
        # 当没有指定 axis 参数时，concatenate 函数会使用扁平化的版本。
        # 这个函数曾经在处理多个数组时存在 bug（参见 gh-5979）。
        x = np.arange(1, 100)
        # 将 x 数组按照 axis=None 进行连接
        r = np.concatenate(x, None)
        # 断言连接后的结果与原始数组 x 相等
        assert_array_equal(x, r)

        # 曾经这种写法等同于 `axis=None`，但现在会失败（由于多个问题导致的未指定错误）。
        with pytest.raises(ValueError):
            # 尝试使用一个非法的 axis 参数值来调用 concatenate
            np.concatenate(x, 100)

    def test_concatenate(self):
        # 测试 concatenate 函数的各种用法
        # 单个序列返回不变的数组
        r4 = list(range(4))
        assert_array_equal(concatenate((r4,)), r4)
        # 任何序列类型
        assert_array_equal(concatenate((tuple(r4),)), r4)
        assert_array_equal(concatenate((array(r4),)), r4)
        # 1D 默认连接
        r3 = list(range(3))
        assert_array_equal(concatenate((r4, r3)), r4 + r3)
        # 不同类型序列的混合连接
        assert_array_equal(concatenate((tuple(r4), r3)), r4 + r3)
        assert_array_equal(concatenate((array(r4), r3)), r4 + r3)
        # 明确指定 axis 参数进行连接
        assert_array_equal(concatenate((r4, r3), 0), r4 + r3)
        # 包括负数作为 axis 参数
        assert_array_equal(concatenate((r4, r3), -1), r4 + r3)
        # 2D 连接
        a23 = array([[10, 11, 12], [13, 14, 15]])
        a13 = array([[0, 1, 2]])
        res = array([[10, 11, 12], [13, 14, 15], [0, 1, 2]])
        assert_array_equal(concatenate((a23, a13)), res)
        assert_array_equal(concatenate((a23, a13), 0), res)
        assert_array_equal(concatenate((a23.T, a13.T), 1), res.T)
        assert_array_equal(concatenate((a23.T, a13.T), -1), res.T)
        # 数组必须匹配形状
        assert_raises(ValueError, concatenate, (a23.T, a13.T), 0)
        # 3D 连接
        res = arange(2 * 3 * 7).reshape((2, 3, 7))
        a0 = res[..., :4]
        a1 = res[..., 4:6]
        a2 = res[..., 6:]
        assert_array_equal(concatenate((a0, a1, a2), 2), res)
        assert_array_equal(concatenate((a0, a1, a2), -1), res)
        assert_array_equal(concatenate((a0.T, a1.T, a2.T), 0), res.T)

        # 测试使用输出参数 out 进行连接
        out = res.copy()
        rout = concatenate((a0, a1, a2), 2, out=out)
        assert_(out is rout)
        assert_equal(res, rout)

    @pytest.mark.skipif(IS_PYPY, reason="PYPY handles sq_concat, nb_add differently than cpython")
    def test_operator_concat(self):
        import operator
        a = array([1, 2])
        b = array([3, 4])
        n = [1,2]
        res = array([1, 2, 3, 4])
        assert_raises(TypeError, operator.concat, a, b)
        assert_raises(TypeError, operator.concat, a, n)
        assert_raises(TypeError, operator.concat, n, a)
        assert_raises(TypeError, operator.concat, a, 1)
        assert_raises(TypeError, operator.concat, 1, a)
    # 定义一个测试函数，用于测试不良的输出形状
    def test_bad_out_shape(self):
        # 创建两个numpy数组a和b
        a = array([1, 2])
        b = array([3, 4])

        # 断言：使用np.empty创建输出为5的空数组时，应该引发值错误异常
        assert_raises(ValueError, concatenate, (a, b), out=np.empty(5))
        # 断言：使用np.empty创建输出形状为(4,1)的空数组时，应该引发值错误异常
        assert_raises(ValueError, concatenate, (a, b), out=np.empty((4,1)))
        # 断言：使用np.empty创建输出形状为(1,4)的空数组时，应该引发值错误异常
        assert_raises(ValueError, concatenate, (a, b), out=np.empty((1,4)))
        # 使用np.empty创建输出形状为(4,)的空数组进行连接
        concatenate((a, b), out=np.empty(4))

    # 使用pytest的参数化标记，测试不同情况下的输出和数据类型
    @pytest.mark.parametrize("axis", [None, 0])
    @pytest.mark.parametrize("out_dtype", ["c8", "f4", "f8", ">f8", "i8", "S4"])
    @pytest.mark.parametrize("casting",
            ['no', 'equiv', 'safe', 'same_kind', 'unsafe'])
    def test_out_and_dtype(self, axis, out_dtype, casting):
        # 创建一个指定数据类型的空数组out
        out = np.empty(4, dtype=out_dtype)
        # 创建两个要连接的数组
        to_concat = (array([1.1, 2.2]), array([3.3, 4.4]))

        # 如果无法将to_concat[0]转换为指定的out_dtype，则应该引发类型错误异常
        if not np.can_cast(to_concat[0], out_dtype, casting=casting):
            with assert_raises(TypeError):
                concatenate(to_concat, out=out, axis=axis, casting=casting)
            with assert_raises(TypeError):
                concatenate(to_concat, dtype=out.dtype,
                            axis=axis, casting=casting)
        else:
            # 否则，使用指定的out和数据类型连接数组，并进行断言检查
            res_out = concatenate(to_concat, out=out,
                                  axis=axis, casting=casting)
            res_dtype = concatenate(to_concat, dtype=out.dtype,
                                    axis=axis, casting=casting)
            assert res_out is out
            assert_array_equal(out, res_dtype)
            assert res_dtype.dtype == out_dtype

        # 最后，应该引发类型错误异常，因为不能同时指定out和dtype参数
        with assert_raises(TypeError):
            concatenate(to_concat, out=out, dtype=out_dtype, axis=axis)

    # 使用pytest的参数化标记，测试不同情况下的字符串数据类型
    @pytest.mark.parametrize("axis", [None, 0])
    @pytest.mark.parametrize("string_dt", ["S", "U", "S0", "U0"])
    @pytest.mark.parametrize("arrs",
            [([0.],), ([0.], [1]), ([0], ["string"], [1.])])
    def test_dtype_with_promotion(self, arrs, string_dt, axis):
        # 执行连接操作，使用给定的字符串数据类型和不安全的类型转换方式
        res = np.concatenate(arrs, axis=axis, dtype=string_dt, casting="unsafe")
        # 断言：实际的数据类型应该与转换后的双精度数组的数据类型相同
        assert res.dtype == np.array(1.).astype(string_dt).dtype

    # 使用pytest的参数化标记，测试字符串数据类型不会检查的情况
    @pytest.mark.parametrize("axis", [None, 0])
    def test_string_dtype_does_not_inspect(self, axis):
        # 应该引发类型错误异常，因为不能将None和整数1连接为字符串数据类型"S"
        with pytest.raises(TypeError):
            np.concatenate(([None], [1]), dtype="S", axis=axis)
        # 应该引发类型错误异常，因为不能将None和整数1连接为Unicode字符串数据类型"U"
        with pytest.raises(TypeError):
            np.concatenate(([None], [1]), dtype="U", axis=axis)

    # 使用pytest的参数化标记，测试子数组错误的情况
    @pytest.mark.parametrize("axis", [None, 0])
    def test_subarray_error(self, axis):
        # 应该引发类型错误异常，因为不能将(1,)整数数组连接为子数组dtype
        with pytest.raises(TypeError, match=".*subarray dtype"):
            np.concatenate(([1], [1]), dtype="(2,)i", axis=axis)
# 定义测试函数 `test_stack()`，用于测试 `stack` 函数的各种输入情况和异常情况
def test_stack():
    # 针对非可迭代输入，验证是否会抛出 TypeError 异常
    assert_raises(TypeError, stack, 1)

    # 针对0维输入的测试用例
    for input_ in [(1, 2, 3),               # 元组形式
                   [np.int32(1), np.int32(2), np.int32(3)],  # 使用 np.int32 数组
                   [np.array(1), np.array(2), np.array(3)]]:  # 使用 np.array 数组
        assert_array_equal(stack(input_), [1, 2, 3])

    # 1维输入的测试用例
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    r1 = array([[1, 2, 3], [4, 5, 6]])

    # 测试 np.stack 函数的不同参数组合和结果是否符合预期
    assert_array_equal(np.stack((a, b)), r1)
    assert_array_equal(np.stack((a, b), axis=1), r1.T)

    # 测试所有输入类型
    assert_array_equal(np.stack(list([a, b])), r1)
    assert_array_equal(np.stack(array([a, b])), r1)

    # 所有1维输入形状的测试用例
    arrays = [np.random.randn(3) for _ in range(10)]
    axes = [0, 1, -1, -2]
    expected_shapes = [(10, 3), (3, 10), (3, 10), (10, 3)]
    for axis, expected_shape in zip(axes, expected_shapes):
        assert_equal(np.stack(arrays, axis).shape, expected_shape)

    # 测试超出范围的轴异常
    assert_raises_regex(AxisError, 'out of bounds', stack, arrays, axis=2)
    assert_raises_regex(AxisError, 'out of bounds', stack, arrays, axis=-3)

    # 所有2维输入形状的测试用例
    arrays = [np.random.randn(3, 4) for _ in range(10)]
    axes = [0, 1, 2, -1, -2, -3]
    expected_shapes = [(10, 3, 4), (3, 10, 4), (3, 4, 10),
                       (3, 4, 10), (3, 10, 4), (10, 3, 4)]
    for axis, expected_shape in zip(axes, expected_shapes):
        assert_equal(np.stack(arrays, axis).shape, expected_shape)

    # 空数组的测试用例
    assert_(stack([[], [], []]).shape == (3, 0))
    assert_(stack([[], [], []], axis=1).shape == (0, 3))

    # 输出参数的测试用例
    out = np.zeros_like(r1)
    np.stack((a, b), out=out)
    assert_array_equal(out, r1)

    # 边界情况的异常测试
    assert_raises_regex(ValueError, 'need at least one array', stack, [])
    assert_raises_regex(ValueError, 'must have the same shape',
                        stack, [1, np.arange(3)])
    assert_raises_regex(ValueError, 'must have the same shape',
                        stack, [np.arange(3), 1])
    assert_raises_regex(ValueError, 'must have the same shape',
                        stack, [np.arange(3), 1], axis=1)
    assert_raises_regex(ValueError, 'must have the same shape',
                        stack, [np.zeros((3, 3)), np.zeros(3)], axis=1)
    assert_raises_regex(ValueError, 'must have the same shape',
                        stack, [np.arange(2), np.arange(3)])

    # 不接受生成器的测试用例
    with pytest.raises(TypeError, match="arrays to stack must be"):
        stack((x for x in range(3)))

    # 类型转换和数据类型测试
    a = np.array([1, 2, 3])
    b = np.array([2.5, 3.5, 4.5])
    res = np.stack((a, b), axis=1, casting="unsafe", dtype=np.int64)
    expected_res = np.array([[1, 2], [2, 3], [3, 4]])
    assert_array_equal(res, expected_res)

    # 类型转换和数据类型与 TypeError 的测试用例
    with assert_raises(TypeError):
        stack((a, b), dtype=np.int64, axis=1, casting="safe")
@pytest.mark.parametrize("out_dtype", ["c8", "f4", "f8", ">f8", "i8"])
# 参数化测试用例，测试不同的输出数据类型
@pytest.mark.parametrize("casting",
                         ['no', 'equiv', 'safe', 'same_kind', 'unsafe'])
# 参数化测试用例，测试不同的类型转换策略

def test_stack_out_and_dtype(axis, out_dtype, casting):
    # 准备要堆叠的数组
    to_concat = (array([1, 2]), array([3, 4]))
    # 预期的堆叠结果
    res = array([[1, 2], [3, 4]])
    # 创建一个与 res 相同形状的零数组
    out = np.zeros_like(res)

    # 如果无法将第一个数组转换为指定的输出数据类型，则抛出 TypeError 异常
    if not np.can_cast(to_concat[0], out_dtype, casting=casting):
        with assert_raises(TypeError):
            stack(to_concat, dtype=out_dtype,
                  axis=axis, casting=casting)
    else:
        # 测试指定输出数组的情况
        res_out = stack(to_concat, out=out,
                        axis=axis, casting=casting)
        # 测试指定输出数据类型的情况
        res_dtype = stack(to_concat, dtype=out_dtype,
                          axis=axis, casting=casting)
        # 确保使用指定输出数组时的返回结果是 out 数组本身
        assert res_out is out
        # 确保使用指定输出数据类型时的返回结果与 out 数组相等
        assert_array_equal(out, res_dtype)
        # 确保返回结果的数据类型与指定的输出数据类型相同
        assert res_dtype.dtype == out_dtype

    # 在指定输出数组的情况下，检查是否会抛出 TypeError 异常
    with assert_raises(TypeError):
        stack(to_concat, out=out, dtype=out_dtype, axis=axis)


class TestBlock:
    @pytest.fixture(params=['block', 'force_concatenate', 'force_slicing'])
    # 使用不同的参数化来设置测试块的行为
    def block(self, request):
        # 阻塞小数组和大数组会走不同的路径。
        # 算法的触发取决于所需元素的数量拷贝。
        # 定义一个测试夹具，强制大多数测试通过两个代码路径。
        # 如果找到单一算法对小数组和大数组都更快，应最终删除此部分。
        def _block_force_concatenate(arrays):
            arrays, list_ndim, result_ndim, _ = _block_setup(arrays)
            return _block_concatenate(arrays, list_ndim, result_ndim)

        def _block_force_slicing(arrays):
            arrays, list_ndim, result_ndim, _ = _block_setup(arrays)
            return _block_slicing(arrays, list_ndim, result_ndim)

        if request.param == 'force_concatenate':
            return _block_force_concatenate
        elif request.param == 'force_slicing':
            return _block_force_slicing
        elif request.param == 'block':
            return block
        else:
            raise ValueError('Unknown blocking request. There is a typo in the tests.')

    def test_returns_copy(self, block):
        # 创建一个单位矩阵
        a = np.eye(3)
        # 使用测试块对数组进行处理
        b = block(a)
        # 修改 b 的值
        b[0, 0] = 2
        # 确保修改后的值不等于原始数组 a 的值
        assert b[0, 0] != a[0, 0]

    def test_block_total_size_estimate(self, block):
        # 测试总大小估算
        _, _, _, total_size = _block_setup([1])
        assert total_size == 1

        _, _, _, total_size = _block_setup([[1]])
        assert total_size == 1

        _, _, _, total_size = _block_setup([[1, 1]])
        assert total_size == 2

        _, _, _, total_size = _block_setup([[1], [1]])
        assert total_size == 2

        _, _, _, total_size = _block_setup([[1, 2], [3, 4]])
        assert total_size == 4
    # 定义一个测试方法，测试简单的按行拼接的情况
    def test_block_simple_row_wise(self, block):
        # 创建一个2x2的全1数组
        a_2d = np.ones((2, 2))
        # 将a_2d中的每个元素乘以2，得到b_2d
        b_2d = 2 * a_2d
        # 预期结果，将a_2d和b_2d按行拼接
        desired = np.array([[1, 1, 2, 2],
                            [1, 1, 2, 2]])
        # 调用被测试的block方法，传入a_2d和b_2d，获取结果
        result = block([a_2d, b_2d])
        # 断言结果与预期相等
        assert_equal(desired, result)

    # 定义一个测试方法，测试简单的按列拼接的情况
    def test_block_simple_column_wise(self, block):
        # 创建一个2x2的全1数组
        a_2d = np.ones((2, 2))
        # 将a_2d中的每个元素乘以2，得到b_2d
        b_2d = 2 * a_2d
        # 预期结果，将a_2d和b_2d按列拼接
        expected = np.array([[1, 1],
                             [1, 1],
                             [2, 2],
                             [2, 2]])
        # 调用被测试的block方法，传入[a_2d]和[b_2d]的列表，获取结果
        result = block([[a_2d], [b_2d]])
        # 断言结果与预期相等
        assert_equal(expected, result)

    # 定义一个测试方法，测试包含1维数组按行拼接的情况
    def test_block_with_1d_arrays_row_wise(self, block):
        # 创建两个1维数组a和b
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        # 预期结果，将a和b按行拼接
        expected = np.array([1, 2, 3, 2, 3, 4])
        # 调用被测试的block方法，传入a和b，获取结果
        result = block([a, b])
        # 断言结果与预期相等
        assert_equal(expected, result)

    # 定义一个测试方法，测试包含1维数组按行拼接的多行情况
    def test_block_with_1d_arrays_multiple_rows(self, block):
        # 创建两个1维数组a和b
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        # 预期结果，将[a, b]和[a, b]按行拼接
        expected = np.array([[1, 2, 3, 2, 3, 4],
                             [1, 2, 3, 2, 3, 4]])
        # 调用被测试的block方法，传入[[a, b], [a, b]]，获取结果
        result = block([[a, b], [a, b]])
        # 断言结果与预期相等
        assert_equal(expected, result)

    # 定义一个测试方法，测试包含1维数组按列拼接的情况
    def test_block_with_1d_arrays_column_wise(self, block):
        # 创建两个1维数组a_1d和b_1d
        a_1d = np.array([1, 2, 3])
        b_1d = np.array([2, 3, 4])
        # 预期结果，将[a_1d]和[b_1d]按列拼接
        expected = np.array([[1, 2, 3],
                             [2, 3, 4]])
        # 调用被测试的block方法，传入[[a_1d], [b_1d]]，获取结果
        result = block([[a_1d], [b_1d]])
        # 断言结果与预期相等
        assert_equal(expected, result)

    # 定义一个测试方法，测试包含2维数组和1维数组混合拼接的情况
    def test_block_mixed_1d_and_2d(self, block):
        # 创建一个2x2的全1数组a_2d和一个1维数组b_1d
        a_2d = np.ones((2, 2))
        b_1d = np.array([2, 2])
        # 调用被测试的block方法，传入[[a_2d], [b_1d]]，获取结果
        result = block([[a_2d], [b_1d]])
        # 预期结果，将a_2d和b_1d按列拼接
        expected = np.array([[1, 1],
                             [1, 1],
                             [2, 2]])
        # 断言结果与预期相等
        assert_equal(expected, result)

    # 定义一个测试方法，测试更复杂的拼接情况
    def test_block_complicated(self, block):
        # 创建多个不同维度和形状的数组
        one_2d = np.array([[1, 1, 1]])
        two_2d = np.array([[2, 2, 2]])
        three_2d = np.array([[3, 3, 3, 3, 3, 3]])
        four_1d = np.array([4, 4, 4, 4, 4, 4])
        five_0d = np.array(5)
        six_1d = np.array([6, 6, 6, 6, 6])
        zero_2d = np.zeros((2, 6))

        # 预期结果，按照给定的组合拼接这些数组
        expected = np.array([[1, 1, 1, 2, 2, 2],
                             [3, 3, 3, 3, 3, 3],
                             [4, 4, 4, 4, 4, 4],
                             [5, 6, 6, 6, 6, 6],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])

        # 调用被测试的block方法，传入包含不同维度和形状数组的列表，获取结果
        result = block([[one_2d, two_2d],
                        [three_2d],
                        [four_1d],
                        [five_0d, six_1d],
                        [zero_2d]])
        # 断言结果与预期相等
        assert_equal(result, expected)
    # 定义一个测试方法，用于测试包含嵌套结构的块操作
    def test_nested(self, block):
        # 创建各种大小和形状的 NumPy 数组
        one = np.array([1, 1, 1])
        two = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        three = np.array([3, 3, 3])
        four = np.array([4, 4, 4])
        five = np.array(5)
        six = np.array([6, 6, 6, 6, 6])
        zero = np.zeros((2, 6))

        # 进行块操作，生成嵌套的数组结构
        result = block([
            [
                # 第一个子块，包含三个一维数组
                block([
                   [one],   # 第一个一维数组
                   [three], # 第二个一维数组
                   [four]   # 第三个一维数组
                ]),
                two  # 第二个子块，包含一个二维数组
            ],
            [five, six],  # 第三个子块，包含两个一维数组
            [zero]        # 第四个子块，包含一个二维数组
        ])
        # 预期的结果数组
        expected = np.array([[1, 1, 1, 2, 2, 2],
                             [3, 3, 3, 2, 2, 2],
                             [4, 4, 4, 2, 2, 2],
                             [5, 6, 6, 6, 6, 6],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])

        # 断言结果数组与预期数组相等
        assert_equal(result, expected)

    # 定义一个测试方法，用于测试包含三维数组的块操作
    def test_3d(self, block):
        # 创建各种大小和形状的三维 NumPy 数组
        a000 = np.ones((2, 2, 2), int) * 1
        a100 = np.ones((3, 2, 2), int) * 2
        a010 = np.ones((2, 3, 2), int) * 3
        a001 = np.ones((2, 2, 3), int) * 4
        a011 = np.ones((2, 3, 3), int) * 5
        a101 = np.ones((3, 2, 3), int) * 6
        a110 = np.ones((3, 3, 2), int) * 7
        a111 = np.ones((3, 3, 3), int) * 8

        # 进行块操作，生成嵌套的三维数组结构
        result = block([
            [
                [a000, a001],  # 第一个子块，包含两个二维数组
                [a010, a011],  # 第二个子块，包含两个二维数组
            ],
            [
                [a100, a101],  # 第三个子块，包含两个二维数组
                [a110, a111],  # 第四个子块，包含两个二维数组
            ]
        ])
        # 预期的结果数组
        expected = np.array([[[1, 1, 4, 4, 4],
                              [1, 1, 4, 4, 4],
                              [3, 3, 5, 5, 5],
                              [3, 3, 5, 5, 5],
                              [3, 3, 5, 5, 5]],

                             [[1, 1, 4, 4, 4],
                              [1, 1, 4, 4, 4],
                              [3, 3, 5, 5, 5],
                              [3, 3, 5, 5, 5],
                              [3, 3, 5, 5, 5]],

                             [[2, 2, 6, 6, 6],
                              [2, 2, 6, 6, 6],
                              [7, 7, 8, 8, 8],
                              [7, 7, 8, 8, 8],
                              [7, 7, 8, 8, 8]],

                             [[2, 2, 6, 6, 6],
                              [2, 2, 6, 6, 6],
                              [7, 7, 8, 8, 8],
                              [7, 7, 8, 8, 8],
                              [7, 7, 8, 8, 8]],

                             [[2, 2, 6, 6, 6],
                              [2, 2, 6, 6, 6],
                              [7, 7, 8, 8, 8],
                              [7, 7, 8, 8, 8],
                              [7, 7, 8, 8, 8]]])

        # 断言结果数组与预期数组相等
        assert_array_equal(result, expected)

    # 定义一个测试方法，用于测试形状不匹配的块操作
    def test_block_with_mismatched_shape(self, block):
        # 创建不同形状的 NumPy 数组
        a = np.array([0, 0])
        b = np.eye(2)

        # 断言块操作会抛出 ValueError 异常，因为形状不匹配
        assert_raises(ValueError, block, [a, b])
        assert_raises(ValueError, block, [b, a])

        # 创建不同形状的二维数组列表
        to_block = [[np.ones((2,3)), np.ones((2,2))],
                    [np.ones((2,2)), np.ones((2,2))]]

        # 断言块操作会抛出 ValueError 异常，因为形状不匹配
        assert_raises(ValueError, block, to_block)
    # 测试函数：验证对于不包含列表的情况，block 函数能正确处理
    def test_no_lists(self, block):
        # 断言：当输入参数为1时，返回一个形状为(1,)的 NumPy 数组
        assert_equal(block(1),         np.array(1))
        # 断言：当输入参数为3阶单位矩阵时，返回相同的3阶单位矩阵
        assert_equal(block(np.eye(3)), np.eye(3))

    # 测试函数：验证对于不合法的嵌套列表的情况，block 函数能引发 ValueError 异常并包含指定错误消息
    def test_invalid_nesting(self, block):
        msg = 'depths are mismatched'
        # 断言：当嵌套列表的深度不匹配时，会引发 ValueError 异常，并包含指定错误消息
        assert_raises_regex(ValueError, msg, block, [1, [2]])
        assert_raises_regex(ValueError, msg, block, [1, []])
        assert_raises_regex(ValueError, msg, block, [[1], 2])
        assert_raises_regex(ValueError, msg, block, [[], 2])
        assert_raises_regex(ValueError, msg, block, [
            [[1], [2]],
            [[3, 4]],
            [5]  # 缺少括号
        ])

    # 测试函数：验证对于空列表的情况，block 函数能引发 ValueError 异常并包含指定错误消息
    def test_empty_lists(self, block):
        assert_raises_regex(ValueError, 'empty', block, [])
        assert_raises_regex(ValueError, 'empty', block, [[]])
        assert_raises_regex(ValueError, 'empty', block, [[1], []])

    # 测试函数：验证对于元组作为输入的情况，block 函数能引发 TypeError 异常并包含指定错误消息
    def test_tuple(self, block):
        assert_raises_regex(TypeError, 'tuple', block, ([1, 2], [3, 4]))
        assert_raises_regex(TypeError, 'tuple', block, [(1, 2), (3, 4)])

    # 测试函数：验证对于不同维度的数组作为输入的情况，block 函数能正确处理
    def test_different_ndims(self, block):
        a = 1.
        b = 2 * np.ones((1, 2))
        c = 3 * np.ones((1, 1, 3))

        # 调用 block 函数，并断言返回的结果与预期的结果相等
        result = block([a, b, c])
        expected = np.array([[[1., 2., 2., 3., 3., 3.]]])
        assert_equal(result, expected)

    # 测试函数：验证对于深度不同的多维数组作为输入的情况，block 函数能正确处理
    def test_different_ndims_depths(self, block):
        a = 1.
        b = 2 * np.ones((1, 2))
        c = 3 * np.ones((1, 2, 3))

        # 调用 block 函数，并断言返回的结果与预期的结果相等
        result = block([[a, b], [c]])
        expected = np.array([[[1., 2., 2.],
                              [3., 3., 3.],
                              [3., 3., 3.]]])
        assert_equal(result, expected)

    # 测试函数：验证对于不同存储顺序的多维数组作为输入的情况，block 函数能正确处理
    def test_block_memory_order(self, block):
        # 创建 C 和 Fortran 存储顺序的三维数组
        arr_c = np.zeros((3,)*3, order='C')
        arr_f = np.zeros((3,)*3, order='F')

        # 创建对应存储顺序的多层嵌套数组 b_c 和 b_f
        b_c = [[[arr_c, arr_c],
                [arr_c, arr_c]],
               [[arr_c, arr_c],
                [arr_c, arr_c]]]

        b_f = [[[arr_f, arr_f],
                [arr_f, arr_f]],
               [[arr_f, arr_f],
                [arr_f, arr_f]]]

        # 断言：block 函数处理后的数组保持 C 和 Fortran 的存储顺序
        assert block(b_c).flags['C_CONTIGUOUS']
        assert block(b_f).flags['F_CONTIGUOUS']

        # 更新为二维数组的情况
        arr_c = np.zeros((3, 3), order='C')
        arr_f = np.zeros((3, 3), order='F')

        b_c = [[arr_c, arr_c],
               [arr_c, arr_c]]

        b_f = [[arr_f, arr_f],
               [arr_f, arr_f]]

        # 断言：block 函数处理后的数组保持 C 和 Fortran 的存储顺序
        assert block(b_c).flags['C_CONTIGUOUS']
        assert block(b_f).flags['F_CONTIGUOUS']
# 定义一个测试函数 test_block_dispatcher，用于测试 _block_dispatcher 函数的行为
def test_block_dispatcher():
    # 定义一个内部类 ArrayLike，用于模拟类似数组的对象
    class ArrayLike:
        pass
    
    # 创建三个 ArrayLike 的实例对象 a, b, c
    a = ArrayLike()
    b = ArrayLike()
    c = ArrayLike()
    
    # 测试 _block_dispatcher 函数处理单个 ArrayLike 对象的情况，期望返回包含 a 的列表
    assert_equal(list(_block_dispatcher(a)), [a])
    
    # 测试 _block_dispatcher 函数处理包含单个 ArrayLike 对象的列表的情况，期望返回包含 a 的列表
    assert_equal(list(_block_dispatcher([a])), [a])
    
    # 测试 _block_dispatcher 函数处理包含多个 ArrayLike 对象的列表的情况，期望返回包含 a 和 b 的列表
    assert_equal(list(_block_dispatcher([a, b])), [a, b])
    
    # 测试 _block_dispatcher 函数处理包含嵌套列表的情况，期望返回包含 a, b 和 c 的列表
    assert_equal(list(_block_dispatcher([[a], [b, [c]]])), [a, b, c])
    
    # 测试 _block_dispatcher 函数不会递归处理非列表类型的情况，期望返回包含 (a, b) 的列表
    assert_equal(list(_block_dispatcher((a, b))), [(a, b)])
```