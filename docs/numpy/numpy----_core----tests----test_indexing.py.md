# `.\numpy\numpy\_core\tests\test_indexing.py`

```
# 导入系统模块 sys
import sys
# 导入警告模块 warnings
import warnings
# 导入 functools 模块
import functools
# 导入 operator 模块
import operator

# 导入 pytest 测试框架
import pytest

# 导入 numpy 库，并使用 np 别名
import numpy as np
# 导入 numpy 内部的 array_indexing 测试模块
from numpy._core._multiarray_tests import array_indexing
# 导入 itertools 模块中的 product 函数
from itertools import product
# 导入 numpy 异常模块中的 ComplexWarning 和 VisibleDeprecationWarning 异常
from numpy.exceptions import ComplexWarning, VisibleDeprecationWarning
# 导入 numpy 测试模块中的一系列断言函数
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_raises_regex,
    assert_array_equal, assert_warns, HAS_REFCOUNT, IS_WASM
    )


class TestIndexing:
    # 定义测试类 TestIndexing
    def test_index_no_floats(self):
        # 创建一个三维 numpy 数组
        a = np.array([[[5]]])

        # 断言索引浮点数会引发 IndexError 异常
        assert_raises(IndexError, lambda: a[0.0])
        assert_raises(IndexError, lambda: a[0, 0.0])
        assert_raises(IndexError, lambda: a[0.0, 0])
        assert_raises(IndexError, lambda: a[0.0,:])
        assert_raises(IndexError, lambda: a[:, 0.0])
        assert_raises(IndexError, lambda: a[:, 0.0,:])
        assert_raises(IndexError, lambda: a[0.0,:,:])
        assert_raises(IndexError, lambda: a[0, 0, 0.0])
        assert_raises(IndexError, lambda: a[0.0, 0, 0])
        assert_raises(IndexError, lambda: a[0, 0.0, 0])
        assert_raises(IndexError, lambda: a[-1.4])
        assert_raises(IndexError, lambda: a[0, -1.4])
        assert_raises(IndexError, lambda: a[-1.4, 0])
        assert_raises(IndexError, lambda: a[-1.4,:])
        assert_raises(IndexError, lambda: a[:, -1.4])
        assert_raises(IndexError, lambda: a[:, -1.4,:])
        assert_raises(IndexError, lambda: a[-1.4,:,:])
        assert_raises(IndexError, lambda: a[0, 0, -1.4])
        assert_raises(IndexError, lambda: a[-1.4, 0, 0])
        assert_raises(IndexError, lambda: a[0, -1.4, 0])
        assert_raises(IndexError, lambda: a[0.0:, 0.0])
        assert_raises(IndexError, lambda: a[0.0:, 0.0,:])
    def test_slicing_no_floats(self):
        # 创建一个包含单个元素的二维 NumPy 数组
        a = np.array([[5]])

        # 测试从浮点数开始的切片操作，期望引发 TypeError 异常
        assert_raises(TypeError, lambda: a[0.0:])
        assert_raises(TypeError, lambda: a[0:, 0.0:2])
        assert_raises(TypeError, lambda: a[0.0::2, :0])
        assert_raises(TypeError, lambda: a[0.0:1:2, :])
        assert_raises(TypeError, lambda: a[:, 0.0:])

        # 测试以浮点数结束的切片操作，期望引发 TypeError 异常
        assert_raises(TypeError, lambda: a[:0.0])
        assert_raises(TypeError, lambda: a[:0, 1:2.0])
        assert_raises(TypeError, lambda: a[:0.0:2, :0])
        assert_raises(TypeError, lambda: a[:0.0, :])
        assert_raises(TypeError, lambda: a[:, 0:4.0:2])

        # 测试以浮点数步长的切片操作，期望引发 TypeError 异常
        assert_raises(TypeError, lambda: a[::1.0])
        assert_raises(TypeError, lambda: a[0:, :2:2.0])
        assert_raises(TypeError, lambda: a[1::4.0, :0])
        assert_raises(TypeError, lambda: a[::5.0, :])
        assert_raises(TypeError, lambda: a[:, 0:4:2.0])

        # 测试混合浮点数的切片操作，期望引发 TypeError 异常
        assert_raises(TypeError, lambda: a[1.0:2:2.0])
        assert_raises(TypeError, lambda: a[1.0::2.0])
        assert_raises(TypeError, lambda: a[0:, :2.0:2.0])
        assert_raises(TypeError, lambda: a[1.0:1:4.0, :0])
        assert_raises(TypeError, lambda: a[1.0:5.0:5.0, :])
        assert_raises(TypeError, lambda: a[:, 0.4:4.0:2.0])

        # 当步长为 0 时，依然应该得到 DeprecationWarning
        assert_raises(TypeError, lambda: a[::0.0])

    def test_index_no_array_to_index(self):
        # 创建一个包含单个元素的三维 NumPy 数组
        a = np.array([[[1]]])

        # 测试在索引中使用非标量数组，期望引发 TypeError 异常
        assert_raises(TypeError, lambda: a[a:a:a])

    def test_none_index(self):
        # 创建一个包含三个元素的一维 NumPy 数组
        a = np.array([1, 2, 3])

        # 测试使用 `None` 作为索引，期望返回一个新的轴
        assert_equal(a[None], a[np.newaxis])
        # 检查添加 `None` 索引后数组的维度增加了 1
        assert_equal(a[None].ndim, a.ndim + 1)

    def test_empty_tuple_index(self):
        # 创建一个包含三个元素的一维 NumPy 数组
        a = np.array([1, 2, 3])

        # 测试使用空元组作为索引，期望返回原数组的视图
        assert_equal(a[()], a)
        # 检查使用空元组索引后的数组与原数组共享数据
        assert_(a[()].base is a)

        # 创建一个标量值为 0 的 NumPy 数组
        a = np.array(0)
        # 检查使用空元组索引后仍然是一个整数类型
        assert_(isinstance(a[()], np.int_))

    def test_void_scalar_empty_tuple(self):
        # 创建一个空的复合类型标量数组，dtype 为 'V4'
        s = np.zeros((), dtype='V4')

        # 测试使用空元组索引，期望返回与原数组相同的标量
        assert_equal(s[()].dtype, s.dtype)
        assert_equal(s[()], s)
        # 检查使用 `...` 索引后返回的对象类型为 ndarray
        assert_equal(type(s[...]), np.ndarray)

    def test_same_kind_index_casting(self):
        # 创建一个包含 0 到 4 的整数的 NumPy 数组
        index = np.arange(5)
        # 将整数数组转换为无符号整数类型数组
        u_index = index.astype(np.uintp)
        # 创建一个包含 0 到 9 的整数的 NumPy 数组
        arr = np.arange(10)

        # 测试在索引操作中，使用不同类型但相同 kind 的索引，应当相等
        assert_array_equal(arr[index], arr[u_index])

        # 将数组 arr 中 u_index 对应的位置赋值为 [0, 1, 2, 3, 4]
        arr[u_index] = np.arange(5)
        assert_array_equal(arr, np.arange(10))

        # 创建一个 5x2 的二维数组
        arr = np.arange(10).reshape(5, 2)
        # 测试使用相同类型但不同类型的索引，应当相等
        assert_array_equal(arr[index], arr[u_index])

        # 将数组 arr 中 u_index 对应的行赋值为 [[0], [1], [2], [3], [4]]
        arr[u_index] = np.arange(5)[:, None]
        assert_array_equal(arr, np.arange(5)[:, None].repeat(2, axis=1))

        # 创建一个 5x5 的二维数组
        arr = np.arange(25).reshape(5, 5)
        # 测试使用相同类型但不同类型的索引，应当相等
        assert_array_equal(arr[u_index, u_index], arr[index, index])
    def test_empty_fancy_index(self):
        # 定义测试函数：测试空的高级索引

        # 创建一个包含元素为 [1, 2, 3] 的 NumPy 数组
        a = np.array([1, 2, 3])
        # 使用空列表索引返回一个空数组
        assert_equal(a[[]], [])
        # 确保返回的空数组与原数组具有相同的数据类型
        assert_equal(a[[]].dtype, a.dtype)

        # 创建一个空数组 b，指定数据类型为 np.intp
        b = np.array([], dtype=np.intp)
        # 使用空列表索引返回一个空数组
        assert_equal(a[[]], [])
        # 确保返回的空数组与原数组具有相同的数据类型
        assert_equal(a[[]].dtype, a.dtype)

        # 创建一个空数组 b
        b = np.array([])
        # 使用空数组 b 进行索引，预期会抛出 IndexError 异常
        assert_raises(IndexError, a.__getitem__, b)

    def test_ellipsis_index(self):
        # 定义测试函数：测试省略号索引

        # 创建一个二维数组 a
        a = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        # 确保 a[...] 不是原始数组 a 的引用
        assert_(a[...] is not a)
        # 使用省略号索引返回整个数组 a
        assert_equal(a[...], a)
        # 确保 a[...] 的底层数据是数组 a
        assert_(a[...].base is a)

        # 使用省略号进行切片可以跳过任意数量的维度
        assert_equal(a[0, ...], a[0])
        assert_equal(a[0, ...], a[0, :])
        assert_equal(a[..., 0], a[:, 0])

        # 使用省略号进行切片总是返回一个数组，而不是标量
        assert_equal(a[0, ..., 1], np.array(2))

        # 对于零维数组，可以使用 `(Ellipsis,)` 进行赋值
        b = np.array(1)
        b[(Ellipsis,)] = 2
        assert_equal(b, 2)

    def test_single_int_index(self):
        # 定义测试函数：测试单个整数索引

        # 创建一个二维数组 a
        a = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

        # 使用单个整数索引返回指定行
        assert_equal(a[0], [1, 2, 3])
        assert_equal(a[-1], [7, 8, 9])

        # 索引超出边界会引发 IndexError 异常
        assert_raises(IndexError, a.__getitem__, 1 << 30)
        assert_raises(IndexError, a.__getitem__, 1 << 64)

    def test_single_bool_index(self):
        # 定义测试函数：测试单个布尔索引

        # 创建一个二维数组 a
        a = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

        # 使用单个布尔数组索引返回整个数组 a
        assert_equal(a[np.array(True)], a[None])
        # 使用单个假布尔数组索引返回空数组
        assert_equal(a[np.array(False)], a[None][0:0])

    def test_boolean_shape_mismatch(self):
        # 定义测试函数：测试布尔索引形状不匹配的情况

        # 创建一个全为 1 的三维数组
        arr = np.ones((5, 4, 3))

        # 创建一个长度为 1 的布尔数组作为索引，预期抛出 IndexError 异常
        index = np.array([True])
        assert_raises(IndexError, arr.__getitem__, index)

        # 创建一个长度为 6 的布尔数组作为索引，预期抛出 IndexError 异常
        index = np.array([False] * 6)
        assert_raises(IndexError, arr.__getitem__, index)

        # 创建一个形状为 (4, 4) 的布尔数组作为索引，预期抛出 IndexError 异常
        index = np.zeros((4, 4), dtype=bool)
        assert_raises(IndexError, arr.__getitem__, index)

        # 在第一维上使用布尔数组索引，预期抛出 IndexError 异常
        assert_raises(IndexError, arr.__getitem__, (slice(None), index))

    def test_boolean_indexing_onedim(self):
        # 定义测试函数：测试在一维数组上使用布尔索引

        # 创建一个一维数组 a，元素全为 0.0
        a = np.array([[ 0.,  0.,  0.]])
        # 创建一个长度为 1 的布尔数组 b
        b = np.array([ True], dtype=bool)
        # 使用布尔数组 b 索引返回整个数组 a
        assert_equal(a[b], a)
        # 对数组 a 使用布尔数组 b 进行赋值
        a[b] = 1.
        assert_equal(a, [[1., 1., 1.]])
    def test_boolean_assignment_value_mismatch(self):
        # 测试布尔赋值在值的形状无法广播到订阅位置时应该失败。（参见 gh-3458）
        # 创建一个长度为4的数组 a
        a = np.arange(4)

        def f(a, v):
            # 将满足条件 a > -1 的位置赋值为 v
            a[a > -1] = v

        # 断言应该抛出 ValueError 异常，因为空列表不能广播到 a 的形状
        assert_raises(ValueError, f, a, [])
        # 断言应该抛出 ValueError 异常，因为长度为3的列表不能广播到 a 的形状
        assert_raises(ValueError, f, a, [1, 2, 3])
        # 断言应该抛出 ValueError 异常，因为长度为3的列表不能广播到 a[:1] 的形状
        assert_raises(ValueError, f, a[:1], [1, 2, 3])

    def test_boolean_assignment_needs_api(self):
        # 参见 gh-7666
        # 在 Python 2 中由于 GIL 未被持有，当迭代器不需要 GIL 但转移函数需要时，会导致段错误
        arr = np.zeros(1000)
        indx = np.zeros(1000, dtype=bool)
        indx[:100] = True
        # 使用对象数组进行赋值，将前100个位置设置为1
        arr[indx] = np.ones(100, dtype=object)

        expected = np.zeros(1000)
        expected[:100] = 1
        # 断言 arr 与 expected 数组相等
        assert_array_equal(arr, expected)

    def test_boolean_indexing_twodim(self):
        # 使用二维布尔数组对二维数组进行索引
        a = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        b = np.array([[ True, False,  True],
                      [False,  True, False],
                      [ True, False,  True]])
        # 断言使用布尔数组索引后的结果正确
        assert_equal(a[b], [1, 3, 5, 7, 9])
        # 断言使用 b[1] 的布尔数组索引后的结果正确
        assert_equal(a[b[1]], [[4, 5, 6]])
        # 断言使用 b[0] 的布尔数组索引后的结果等于使用 b[2] 的布尔数组索引后的结果
        assert_equal(a[b[0]], a[b[2]])

        # 布尔赋值操作
        a[b] = 0
        # 断言布尔赋值操作后的数组 a 符合预期
        assert_equal(a, [[0, 2, 0],
                         [4, 0, 6],
                         [0, 8, 0]])

    def test_boolean_indexing_list(self):
        # 修复 #13715 的回归测试。这是一个使用后释放的 bug，尽管此测试不能直接捕获，但会在 valgrind 中显示出来。
        a = np.array([1, 2, 3])
        b = [True, False, True]
        # 测试两种情况，因为第一种会有快速路径
        assert_equal(a[b], [1, 3])
        assert_equal(a[None, b], [[1, 3]])

    def test_reverse_strides_and_subspace_bufferinit(self):
        # 测试简单和子空间花式索引时是否翻转了步长。
        a = np.ones(5)
        # 创建一个逆序的整数数组
        b = np.zeros(5, dtype=np.intp)[::-1]
        # 创建一个逆序的整数数组
        c = np.arange(5)[::-1]

        # 使用逆序的索引 b 将数组 a 的部分位置赋值为 c 的值
        a[b] = c
        # 如果步长未被翻转，那么 arange 中的 0 会被放在最后。
        assert_equal(a[0], 0)

        # 同时测试子空间缓冲是否被初始化：
        a = np.ones((5, 2))
        c = np.arange(10).reshape(5, 2)[::-1]
        # 使用逆序的索引 b 对数组 a 的子空间赋值为数组 c
        a[b, :] = c
        # 断言数组 a 的第一行是否等于 [0, 1]
        assert_equal(a[0], [0, 1])

    def test_reversed_strides_result_allocation(self):
        # 测试计算结果数组的输出步长时的一个 bug，当子空间大小为1时（同时测试其他情况）。
        a = np.arange(10)[:, None]
        i = np.arange(10)[::-1]
        # 断言 a[i] 与 a[i.copy('C')] 是否相等
        assert_array_equal(a[i], a[i.copy('C')])

        a = np.arange(20).reshape(-1, 2)
    def test_uncontiguous_subspace_assignment(self):
        # 在开发过程中曾出现一个 bug，根据 ndim 而不是 size 激活了跳过逻辑。
        a = np.full((3, 4, 2), -1)
        b = np.full((3, 4, 2), -1)

        # 使用不连续的索引分配数据
        a[[0, 1]] = np.arange(2 * 4 * 2).reshape(2, 4, 2).T
        # 使用复制的方式进行相同操作
        b[[0, 1]] = np.arange(2 * 4 * 2).reshape(2, 4, 2).T.copy()

        assert_equal(a, b)

    def test_too_many_fancy_indices_special_case(self):
        # 文档化行为，这是一个小的限制。
        a = np.ones((1,) * 64)  # 64 是 NPY_MAXDIMS
        # 预期引发 IndexError 异常，因为使用了太多的花式索引
        assert_raises(IndexError, a.__getitem__, (np.array([0]),) * 64)

    def test_scalar_array_bool(self):
        # NumPy 布尔类型可以作为布尔索引使用（Python 原生布尔类型目前不行）
        a = np.array(1)
        assert_equal(a[np.bool(True)], a[np.array(True)])
        assert_equal(a[np.bool(False)], a[np.array(False)])

        # 在废弃布尔类型作为整数之后：
        # a = np.array([0,1,2])
        # assert_equal(a[True, :], a[None, :])
        # assert_equal(a[:, True], a[:, None])
        #
        # assert_(not np.may_share_memory(a, a[True, :]))

    def test_everything_returns_views(self):
        # 在 `...` 之前会返回数组本身。
        a = np.arange(5)

        assert_(a is not a[()])
        assert_(a is not a[...])
        assert_(a is not a[:])

    def test_broaderrors_indexing(self):
        a = np.zeros((5, 5))
        # 预期引发 IndexError 异常，因为索引超出了数组维度
        assert_raises(IndexError, a.__getitem__, ([0, 1], [0, 1, 2]))
        assert_raises(IndexError, a.__setitem__, ([0, 1], [0, 1, 2]), 0)

    def test_trivial_fancy_out_of_bounds(self):
        a = np.zeros(5)
        ind = np.ones(20, dtype=np.intp)
        ind[-1] = 10
        # 预期引发 IndexError 异常，因为索引超出了数组的边界
        assert_raises(IndexError, a.__getitem__, ind)
        assert_raises(IndexError, a.__setitem__, ind, 0)
        ind = np.ones(20, dtype=np.intp)
        ind[0] = 11
        # 预期引发 IndexError 异常，因为索引超出了数组的边界
        assert_raises(IndexError, a.__getitem__, ind)
        assert_raises(IndexError, a.__setitem__, ind, 0)

    def test_trivial_fancy_not_possible(self):
        # 测试在索引非连续或非一维时，不应误用简化的快速路径，参见 gh-11467。
        a = np.arange(6)
        idx = np.arange(6, dtype=np.intp).reshape(2, 1, 3)[:, :, 0]
        assert_array_equal(a[idx], idx)

        # 这种情况不应该进入快速路径，注意 idx 在这里是一个非连续的非一维数组。
        a[idx] = -1
        res = np.arange(6)
        res[0] = -1
        res[3] = -1
        assert_array_equal(a, res)
    def test_nonbaseclass_values(self):
        # 定义一个继承自 np.ndarray 的子类 SubClass
        class SubClass(np.ndarray):
            # 定义特殊方法 __array_finalize__，用于数组初始化后的处理
            def __array_finalize__(self, old):
                # 在初始化过程中执行特定操作，这里将数组填充为 99
                self.fill(99)

        # 创建一个 5x5 的全零数组 a
        a = np.zeros((5, 5))
        # 通过复制和视图创建一个 SubClass 类型的数组 s
        s = a.copy().view(type=SubClass)
        # 将 s 数组填充为 1
        s.fill(1)

        # 将 s 数组的值分别赋给 a 数组的第 0 到第 4 行
        a[[0, 1, 2, 3, 4], :] = s
        # 断言 a 数组中所有元素是否都为 1
        assert_((a == 1).all())

        # 将 s 数组的值分别赋给 a 数组的第 0 到第 4 列
        a[:, [0, 1, 2, 3, 4]] = s
        # 断言 a 数组中所有元素是否都为 1
        assert_((a == 1).all())

        # 将 s 数组的值赋给 a 数组的所有元素
        a.fill(0)
        a[...] = s
        # 断言 a 数组中所有元素是否都为 1
        assert_((a == 1).all())

    def test_array_like_values(self):
        # 类似于上面的测试，但使用 memoryview 替代数组
        a = np.zeros((5, 5))
        s = np.arange(25, dtype=np.float64).reshape(5, 5)

        # 将 memoryview(s) 赋给 a 数组的第 0 到第 4 行
        a[[0, 1, 2, 3, 4], :] = memoryview(s)
        # 断言 a 数组是否与 s 数组相等
        assert_array_equal(a, s)

        # 将 memoryview(s) 赋给 a 数组的第 0 到第 4 列
        a[:, [0, 1, 2, 3, 4]] = memoryview(s)
        # 断言 a 数组是否与 s 数组相等
        assert_array_equal(a, s)

        # 将 memoryview(s) 赋给 a 数组的所有元素
        a[...] = memoryview(s)
        # 断言 a 数组是否与 s 数组相等
        assert_array_equal(a, s)

    def test_subclass_writeable(self):
        # 创建一个结构化数组 d，包含字段 'target' 和 'V_mag'
        d = np.rec.array([('NGC1001', 11), ('NGC1002', 1.), ('NGC1003', 1.)],
                         dtype=[('target', 'S20'), ('V_mag', '>f4')])
        # 创建布尔类型的索引数组 ind
        ind = np.array([False,  True,  True], dtype=bool)
        # 断言通过布尔索引访问的 d 的子数组是否可写
        assert_(d[ind].flags.writeable)
        
        # 创建整数类型的索引数组 ind
        ind = np.array([0, 1])
        # 断言通过整数索引访问的 d 的子数组是否可写
        assert_(d[ind].flags.writeable)
        
        # 断言通过省略号索引访问的 d 的子数组是否可写
        assert_(d[...].flags.writeable)
        
        # 断言通过单个整数索引访问的 d 的子数组是否可写
        assert_(d[0].flags.writeable)

    def test_memory_order(self):
        # 这里不必保留。复杂索引的内存布局不那么简单。
        a = np.arange(10)
        b = np.arange(10).reshape(5,2).T
        # 断言通过复杂索引 b 访问的 a 数组是否是列优先的
        assert_(a[b].flags.f_contiguous)

        # 使用不同的实现分支：
        a = a.reshape(-1, 1)
        # 断言通过复杂索引 b 和第一列访问的 a 数组是否是列优先的
        assert_(a[b, 0].flags.f_contiguous)

    def test_scalar_return_type(self):
        # 完整的标量索引应该返回标量，对象数组不应在其项上调用 PyArray_Return
        class Zero:
            # 最基本的有效索引
            def __index__(self):
                return 0

        z = Zero()

        class ArrayLike:
            # 简单的数组，应该表现得像数组一样
            def __array__(self, dtype=None, copy=None):
                return np.array(0)

        a = np.zeros(())
        # 断言空元组 () 索引访问 a 数组是否返回 np.float64 标量
        assert_(isinstance(a[()], np.float64))
        a = np.zeros(1)
        # 断言使用 Zero 对象 z 索引访问 a 数组是否返回 np.float64 标量
        assert_(isinstance(a[z], np.float64))
        a = np.zeros((1, 1))
        # 断言使用 Zero 对象 z 和 np.array(0) 索引访问 a 数组是否返回 np.float64 标量
        assert_(isinstance(a[z, np.array(0)], np.float64))
        # 断言使用 Zero 对象 z 和 ArrayLike() 索引访问 a 数组是否返回 np.float64 标量
        assert_(isinstance(a[z, ArrayLike()], np.float64))

        # 对象数组也不会频繁调用它：
        b = np.array(0)
        a = np.array(0, dtype=object)
        # 断言空元组 () 索引访问 a 数组是否返回 np.ndarray 对象
        assert_(isinstance(a[()], np.ndarray))
        a = np.array([b, None])
        # 断言使用 Zero 对象 z 索引访问 a 数组是否返回 np.ndarray 对象
        assert_(isinstance(a[z], np.ndarray))
        a = np.array([[b, None]])
        # 断言使用 Zero 对象 z 和 ArrayLike() 索引访问 a 数组是否返回 np.ndarray 对象
        assert_(isinstance(a[z, ArrayLike()], np.ndarray))
    def test_small_regressions(self):
        # Reference count of intp for index checks
        # 创建一个包含单个元素的 NumPy 数组
        a = np.array([0])
        # 如果支持引用计数功能，则获取 intp 类型的引用计数
        if HAS_REFCOUNT:
            refcount = sys.getrefcount(np.dtype(np.intp))
        # 使用整数索引进行元素设置，该过程会在单独的函数中检查索引
        a[np.array([0], dtype=np.intp)] = 1
        # 使用无符号 8 位整数类型进行索引元素设置
        a[np.array([0], dtype=np.uint8)] = 1
        # 断言对超出范围的索引设置会引发 IndexError 异常
        assert_raises(IndexError, a.__setitem__,
                      np.array([1], dtype=np.intp), 1)
        assert_raises(IndexError, a.__setitem__,
                      np.array([1], dtype=np.uint8), 1)

        # 如果支持引用计数功能，则断言 intp 类型的引用计数不变
        if HAS_REFCOUNT:
            assert_equal(sys.getrefcount(np.dtype(np.intp)), refcount)

    def test_unaligned(self):
        # 创建一个由 'a' 字符填充的 int8 类型的零数组，然后进行切片
        v = (np.zeros(64, dtype=np.int8) + ord('a'))[1:-7]
        # 将切片后的数组视图转换为 S8 类型
        d = v.view(np.dtype("S8"))
        # 创建一个未对齐的源数组
        x = (np.zeros(16, dtype=np.int8) + ord('a'))[1:-7]
        # 将切片后的数组视图转换为 S8 类型
        x = x.view(np.dtype("S8"))
        # 使用数组 'x' 中的数据填充数组 'd' 中对应位置的元素
        x[...] = np.array("b" * 8, dtype="S")
        # 创建一个范围数组 'b'
        b = np.arange(d.size)
        # 简单情况下的断言：断言数组 'd' 与数组 'b' 的元素相等
        assert_equal(d[b], d)
        # 使用数组 'x' 中的数据填充数组 'd' 中对应索引 'b' 的元素
        d[b] = x
        # 非简单情况下的断言：使用未对齐的索引数组 'b' 对数组 'd' 进行索引
        # 创建一个未对齐的索引数组
        b = np.zeros(d.size + 1).view(np.int8)[1:-(np.intp(0).itemsize - 1)]
        b = b.view(np.intp)[:d.size]
        # 使用数组 'x' 中的数据填充数组 'd' 中对应索引 'b' 的元素
        d[b.astype(np.int16)] = x
        # 对布尔数组 'b % 2 == 0' 进行索引，并不返回结果
        d[b % 2 == 0]
        # 使用数组 'x' 中的数据填充布尔数组 'b % 2 == 0' 对应的元素
        d[b % 2 == 0] = x[::2]

    def test_tuple_subclass(self):
        # 创建一个全为 1 的 5x5 数组
        arr = np.ones((5, 5))

        # 检查元组子类是否也可以作为索引
        class TupleSubclass(tuple):
            pass
        # 创建一个元组子类实例 'index'
        index = ([1], [1])
        index = TupleSubclass(index)
        # 断言数组 'arr' 使用 'index' 索引后的形状为 (1,)
        assert_(arr[index].shape == (1,))
        # 与非 nd-索引（tuple）不同的是：
        assert_(arr[index,].shape != (1,))

    def test_broken_sequence_not_nd_index(self):
        # 见 gh-5063:
        # 如果我们有一个对象声称自己是一个序列，但在获取项目时失败，
        # 这不应该被转换为 nd-索引（tuple）
        # 如果此对象在其他方面是一个有效的索引，它应该工作
        # 这个对象非常可疑，可能是不好的：
        class SequenceLike:
            def __index__(self):
                return 0

            def __len__(self):
                return 1

            def __getitem__(self, item):
                raise IndexError('Not possible')

        arr = np.arange(10)
        # 断言使用 SequenceLike 实例作为索引与直接使用该实例索引的结果相等
        assert_array_equal(arr[SequenceLike()], arr[SequenceLike(),])

        # 还要测试字段索引不会因此而段错误
        # 通过对结构化数组进行索引，以类似的原因
        arr = np.zeros((1,), dtype=[('f1', 'i8'), ('f2', 'i8')])
        assert_array_equal(arr[SequenceLike()], arr[SequenceLike(),])
    def test_indexing_array_weird_strides(self):
        # 这个测试函数涉及到数组的奇怪步幅情况
        # 以下的形状来自于一个问题，并创建了正确的迭代器缓冲区大小
        x = np.ones(10)  # 创建一个包含10个元素的全为1的数组
        x2 = np.ones((10, 2))  # 创建一个形状为(10, 2)的全为1的数组
        ind = np.arange(10)[:, None, None, None]  # 创建一个形状为(10, 1, 1, 1)的索引数组
        ind = np.broadcast_to(ind, (10, 55, 4, 4))  # 将索引数组广播到形状为(10, 55, 4, 4)

        # 单个高级索引的情况
        assert_array_equal(x[ind], x[ind.copy()])  # 断言数组通过相同的高级索引和其副本得到的结果相等
        # 更高维度的高级索引
        zind = np.zeros(4, dtype=np.intp)  # 创建一个具有4个零元素的整数索引数组
        assert_array_equal(x2[ind, zind], x2[ind.copy(), zind])  # 断言数组通过相同的高级索引和其副本得到的结果相等

    def test_indexing_array_negative_strides(self):
        # 来自于gh-8264
        # 如果在迭代中使用负步长，核心可能会崩溃
        arro = np.zeros((4, 4))  # 创建一个形状为(4, 4)的全零数组
        arr = arro[::-1, ::-1]  # 对数组进行逆向切片操作

        slices = (slice(None), [0, 1, 2, 3])  # 创建一个切片元组
        arr[slices] = 10  # 将切片应用到数组上，并赋值为10
        assert_array_equal(arr, 10.)  # 断言数组的所有元素都等于10.0

    def test_character_assignment(self):
        # 这是一个通过CopyObject进行的函数示例
        # 它曾经有一个未经测试的特殊路径用于标量（字符特殊dtype情况，可能应该被弃用）
        arr = np.zeros((1, 5), dtype="c")  # 创建一个形状为(1, 5)的全零数组，数据类型为字符
        arr[0] = np.str_("asdfg")  # 必须作为一个序列进行赋值
        assert_array_equal(arr[0], np.array("asdfg", dtype="c"))  # 断言数组的第一行等于指定的字符数组
        assert arr[0, 1] == b"s"  # 确保不是所有元素都被设置为"a"

    @pytest.mark.parametrize("index",
            [True, False, np.array([0])])
    @pytest.mark.parametrize("num", [64, 80])
    @pytest.mark.parametrize("original_ndim", [1, 64])
    def test_too_many_advanced_indices(self, index, num, original_ndim):
        # 这些是基于我们能够处理的参数数量的限制
        # 对于`num=32`（以及所有布尔值情况），实际上结果是定义好的
        # 但由于技术原因，使用NpyIter（NPY_MAXARGS）限制了它
        arr = np.ones((1,) * original_ndim)  # 创建一个全为1的数组，形状为(1,)*original_ndim
        with pytest.raises(IndexError):
            arr[(index,) * num]  # 断言使用过多的高级索引会抛出IndexError异常
        with pytest.raises(IndexError):
            arr[(index,) * num] = 1.  # 断言使用过多的高级索引会抛出IndexError异常

    @pytest.mark.skipif(IS_WASM, reason="no threading")
    # 定义一个测试方法，用于验证结构化数据类型中整数数组索引时是否安全，这涉及到 `copyswap(n)` 的使用，详见 gh-15387。
    # 这个测试可能会表现出随机性。
    
    from concurrent.futures import ThreadPoolExecutor
    # 导入线程池执行器，用于并行执行任务
    
    # 创建一个深度嵌套的数据类型，增加出错的可能性：
    dt = np.dtype([("", "f8")])
    dt = np.dtype([("", dt)] * 2)
    dt = np.dtype([("", dt)] * 2)
    # 创建一个足够大的数组，以增加遇到线程问题的概率
    arr = np.random.uniform(size=(6000, 8)).view(dt)[:, 0]
    
    rng = np.random.default_rng()
    # 定义一个函数，接受数组作为参数
    def func(arr):
        indx = rng.integers(0, len(arr), size=6000, dtype=np.intp)
        arr[indx]
    
    tpe = ThreadPoolExecutor(max_workers=8)
    # 创建一个最大工作线程为8的线程池执行器
    
    futures = [tpe.submit(func, arr) for _ in range(10)]
    # 提交10个任务给线程池执行器并获取每个任务的future对象
    
    for f in futures:
        f.result()
        # 等待每个任务完成
    
    # 断言数组的数据类型是否为预期的数据类型 `dt`
    assert arr.dtype is dt
    # 使用断言验证条件，如果条件不成立，则会引发异常
    
    # 下面开始另一个测试方法的定义
    def test_nontuple_ndindex(self):
        a = np.arange(25).reshape((5, 5))
        assert_equal(a[[0, 1]], np.array([a[0], a[1]]))
        assert_equal(a[[0, 1], [0, 1]], np.array([0, 6]))
        assert_raises(IndexError, a.__getitem__, [slice(None)])
class TestFieldIndexing:
    def test_scalar_return_type(self):
        # Field access on an array should return an array, even if it
        # is 0-d.
        a = np.zeros((), [('a','f8')])
        assert_(isinstance(a['a'], np.ndarray))  # 检查字段访问返回的类型是否为 ndarray
        assert_(isinstance(a[['a']], np.ndarray))  # 检查字段访问返回的类型是否为 ndarray


class TestBroadcastedAssignments:
    def assign(self, a, ind, val):
        a[ind] = val  # 对数组 a 执行索引操作并赋值为 val
        return a  # 返回更新后的数组 a

    def test_prepending_ones(self):
        a = np.zeros((3, 2))  # 创建一个形状为 (3, 2) 的零数组

        a[...] = np.ones((1, 3, 2))  # 使用全为 1 的数组填充整个数组 a
        # Fancy with subspace with and without transpose
        a[[0, 1, 2], :] = np.ones((1, 3, 2))  # 在指定的子空间内使用全为 1 的数组进行赋值
        a[:, [0, 1]] = np.ones((1, 3, 2))  # 在指定的子空间内使用全为 1 的数组进行赋值
        # Fancy without subspace (with broadcasting)
        a[[[0], [1], [2]], [0, 1]] = np.ones((1, 3, 2))  # 在指定的子空间内使用全为 1 的数组进行赋值

    def test_prepend_not_one(self):
        assign = self.assign
        s_ = np.s_
        a = np.zeros(5)  # 创建一个长度为 5 的零数组

        # Too large and not only ones.
        assert_raises(ValueError, assign, a, s_[...],  np.ones((2, 1)))  # 检查赋值操作是否会引发 ValueError
        assert_raises(ValueError, assign, a, s_[[1, 2, 3],], np.ones((2, 1)))  # 检查赋值操作是否会引发 ValueError
        assert_raises(ValueError, assign, a, s_[[[1], [2]],], np.ones((2,2,1)))  # 检查赋值操作是否会引发 ValueError

    def test_simple_broadcasting_errors(self):
        assign = self.assign
        s_ = np.s_
        a = np.zeros((5, 1))  # 创建一个形状为 (5, 1) 的零数组

        assert_raises(ValueError, assign, a, s_[...], np.zeros((5, 2)))  # 检查赋值操作是否会引发 ValueError
        assert_raises(ValueError, assign, a, s_[...], np.zeros((5, 0)))  # 检查赋值操作是否会引发 ValueError
        assert_raises(ValueError, assign, a, s_[:, [0]], np.zeros((5, 2)))  # 检查赋值操作是否会引发 ValueError
        assert_raises(ValueError, assign, a, s_[:, [0]], np.zeros((5, 0)))  # 检查赋值操作是否会引发 ValueError
        assert_raises(ValueError, assign, a, s_[[0], :], np.zeros((2, 1)))  # 检查赋值操作是否会引发 ValueError

    @pytest.mark.parametrize("index", [
            (..., [1, 2], slice(None)),
            ([0, 1], ..., 0),
            (..., [1, 2], [1, 2])])
    def test_broadcast_error_reports_correct_shape(self, index):
        values = np.zeros((100, 100))  # 创建一个形状为 (100, 100) 的零数组

        arr = np.zeros((3, 4, 5, 6, 7))  # 创建一个形状为 (3, 4, 5, 6, 7) 的零数组
        # We currently report without any spaces (could be changed)
        shape_str = str(arr[index].shape).replace(" ", "")  # 获取索引 arr[index] 的形状并去除空格
        
        with pytest.raises(ValueError) as e:
            arr[index] = values  # 尝试对 arr 的索引位置赋值为 values

        assert str(e.value).endswith(shape_str)  # 检查错误消息的结尾是否与形状字符串匹配

    def test_index_is_larger(self):
        # Simple case of fancy index broadcasting of the index.
        a = np.zeros((5, 5))  # 创建一个形状为 (5, 5) 的零数组
        a[[[0], [1], [2]], [0, 1, 2]] = [2, 3, 4]  # 在指定的索引位置进行赋值操作

        assert_((a[:3, :3] == [2, 3, 4]).all())  # 检查赋值是否成功

    def test_broadcast_subspace(self):
        a = np.zeros((100, 100))  # 创建一个形状为 (100, 100) 的零数组
        v = np.arange(100)[:,None]  # 创建一个形状为 (100, 1) 的数组
        b = np.arange(100)[::-1]  # 创建一个倒序的长度为 100 的数组
        a[b] = v  # 在指定的索引位置进行赋值操作
        assert_((a[::-1] == v).all())  # 检查赋值是否成功
    def test_basic(self):
        # 定义一个子类 SubClass，继承自 np.ndarray
        class SubClass(np.ndarray):
            pass

        # 创建一个从 0 到 4 的 ndarray
        a = np.arange(5)
        
        # 使用 SubClass 视图对 a 进行切片操作
        s = a.view(SubClass)
        
        # 对视图 s 进行切片操作，得到 s 的一个子切片 s_slice
        s_slice = s[:3]
        
        # 断言 s_slice 的类型是 SubClass
        assert_(type(s_slice) is SubClass)
        
        # 断言 s_slice 的 base 是 s
        assert_(s_slice.base is s)
        
        # 断言 s_slice 的值与 a 的前三个元素相等
        assert_array_equal(s_slice, a[:3])

        # 使用 fancy indexing 从 s 中选择特定索引，得到 s 的一个新子类实例 s_fancy
        s_fancy = s[[0, 1, 2]]
        
        # 断言 s_fancy 的类型是 SubClass
        assert_(type(s_fancy) is SubClass)
        
        # 断言 s_fancy 的 base 不是 s
        assert_(s_fancy.base is not s)
        
        # 断言 s_fancy 的 base 类型是 np.ndarray
        assert_(type(s_fancy.base) is np.ndarray)
        
        # 断言 s_fancy 的值与 a 中索引为 [0, 1, 2] 的元素相等
        assert_array_equal(s_fancy, a[[0, 1, 2]])
        
        # 断言 s_fancy 的 base 值与 a 中索引为 [0, 1, 2] 的元素相等
        assert_array_equal(s_fancy.base, a[[0, 1, 2]])

        # 使用布尔索引从 s 中选择大于 0 的元素，得到 s 的一个新子类实例 s_bool
        s_bool = s[s > 0]
        
        # 断言 s_bool 的类型是 SubClass
        assert_(type(s_bool) is SubClass)
        
        # 断言 s_bool 的 base 不是 s
        assert_(s_bool.base is not s)
        
        # 断言 s_bool 的 base 类型是 np.ndarray
        assert_(type(s_bool.base) is np.ndarray)
        
        # 断言 s_bool 的值与 a 中大于 0 的元素相等
        assert_array_equal(s_bool, a[a > 0])
        
        # 断言 s_bool 的 base 值与 a 中大于 0 的元素相等
        assert_array_equal(s_bool.base, a[a > 0])

    def test_fancy_on_read_only(self):
        # 测试只读 SubClass 的 fancy indexing 不会创建只读副本（gh-14132）
        class SubClass(np.ndarray):
            pass

        # 创建一个从 0 到 4 的 ndarray
        a = np.arange(5)
        
        # 使用 SubClass 视图对 a 进行设置为只读
        s = a.view(SubClass)
        s.flags.writeable = False
        
        # 对只读视图 s 进行 fancy indexing 操作
        s_fancy = s[[0, 1, 2]]
        
        # 断言 s_fancy 的可写标志为 True
        assert_(s_fancy.flags.writeable)


    def test_finalize_gets_full_info(self):
        # Array finalize 应该在填充数组时被调用
        class SubClass(np.ndarray):
            def __array_finalize__(self, old):
                self.finalize_status = np.array(self)
                self.old = old

        # 创建一个从 0 到 9 的 ndarray，视图为 SubClass
        s = np.arange(10).view(SubClass)
        
        # 对 s 进行切片操作，得到一个新子类实例 new_s
        new_s = s[:3]
        
        # 断言 new_s 的 finalize_status 等于 new_s 自身
        assert_array_equal(new_s.finalize_status, new_s)
        
        # 断言 new_s 的 old 属性等于 s
        assert_array_equal(new_s.old, s)

        # 对 s 进行 fancy indexing 操作，得到一个新子类实例 new_s
        new_s = s[[0,1,2,3]]
        
        # 断言 new_s 的 finalize_status 等于 new_s 自身
        assert_array_equal(new_s.finalize_status, new_s)
        
        # 断言 new_s 的 old 属性等于 s
        assert_array_equal(new_s.old, s)

        # 对 s 进行布尔索引操作，得到一个新子类实例 new_s
        new_s = s[s > 0]
        
        # 断言 new_s 的 finalize_status 等于 new_s 自身
        assert_array_equal(new_s.finalize_status, new_s)
        
        # 断言 new_s 的 old 属性等于 s
        assert_array_equal(new_s.old, s)
class TestFancyIndexingCast:
    def test_boolean_index_cast_assign(self):
        # 设置布尔索引和浮点数组。
        shape = (8, 63)
        bool_index = np.zeros(shape).astype(bool)
        bool_index[0, 1] = True
        zero_array = np.zeros(shape)

        # 将值赋给布尔索引是允许的。
        zero_array[bool_index] = np.array([1])
        assert_equal(zero_array[0, 1], 1)

        # 使用复数警告检查，进行高级索引赋值，尽管会得到一个转换警告。
        assert_warns(ComplexWarning,
                     zero_array.__setitem__, ([0], [1]), np.array([2 + 1j]))
        assert_equal(zero_array[0, 1], 2)  # 没有复数部分

        # 将复数转换为浮点数，舍弃虚部。
        assert_warns(ComplexWarning,
                     zero_array.__setitem__, bool_index, np.array([1j]))
        assert_equal(zero_array[0, 1], 0)

class TestFancyIndexingEquivalence:
    def test_object_assign(self):
        # 检查字段和对象特殊情况是否使用了copyto。
        # 右侧的值在此处无法转换为数组。
        a = np.arange(5, dtype=object)
        b = a.copy()
        a[:3] = [1, (1,2), 3]
        b[[0, 1, 2]] = [1, (1,2), 3]
        assert_array_equal(a, b)

        # 对子空间高级索引进行相同的测试
        b = np.arange(5, dtype=object)[None, :]
        b[[0], :3] = [[1, (1,2), 3]]
        assert_array_equal(a, b[0])

        # 检查轴的交换是否起作用。
        # 之前有一个错误，使得后续的赋值引发ValueError，
        # 因为一个不正确转置的临时右侧值 (gh-5714)。
        b = b.T
        b[:3, [0]] = [[1], [(1,2)], [3]]
        assert_array_equal(a, b[:, 0])

        # 子空间的内存顺序的另一个测试
        arr = np.ones((3, 4, 5), dtype=object)
        # 用于比较的等效切片赋值
        cmp_arr = arr.copy()
        cmp_arr[:1, ...] = [[[1], [2], [3], [4]]]
        arr[[0], ...] = [[[1], [2], [3], [4]]]
        assert_array_equal(arr, cmp_arr)
        arr = arr.copy('F')
        arr[[0], ...] = [[[1], [2], [3], [4]]]
        assert_array_equal(arr, cmp_arr)

    def test_cast_equivalence(self):
        # 是的，普通切片使用不安全的转换。
        a = np.arange(5)
        b = a.copy()

        a[:3] = np.array(['2', '-3', '-1'])
        b[[0, 2, 1]] = np.array(['2', '-1', '-3'])
        assert_array_equal(a, b)

        # 对子空间高级索引进行相同的测试
        b = np.arange(5)[None, :]
        b[[0], :3] = np.array([['2', '-3', '-1']])
        assert_array_equal(a, b[0])


class TestMultiIndexingAutomated:
    """
    These tests use code to mimic the C-Code indexing for selection.
    """
    NOTE:

        * This still lacks tests for complex item setting.
        * If you change behavior of indexing, you might want to modify
          these tests to try more combinations.
        * Behavior was written to match numpy version 1.8. (though a
          first version matched 1.7.)
        * Only tuple indices are supported by the mimicking code.
          (and tested as of writing this)
        * Error types should match most of the time as long as there
          is only one error. For multiple errors, what gets raised
          will usually not be the same one. They are *not* tested.

    Update 2016-11-30: It is probably not worth maintaining this test
    indefinitely and it can be dropped if maintenance becomes a burden.

    """

    # 设置测试环境的方法
    def setup_method(self):
        # 创建一个四维的 numpy 数组 a，其形状为 (3, 1, 5, 6)，并填充 arange 生成的数据
        self.a = np.arange(np.prod([3, 1, 5, 6])).reshape(3, 1, 5, 6)
        # 创建一个空的 numpy 数组 b，形状为 (3, 0, 5, 6)
        self.b = np.empty((3, 0, 5, 6))
        # 复杂索引的测试集合，包含多种不同类型的索引对象
        self.complex_indices = ['skip', Ellipsis,
            0,
            # 布尔索引，特殊情况下可以吃掉维度，需要测试所有 False 的情况
            np.array([True, False, False]),
            np.array([[True, False], [False, True]]),
            np.array([[[False, False], [False, False]]]),
            # 一些切片:
            slice(-5, 5, 2),
            slice(1, 1, 100),
            slice(4, -1, -2),
            slice(None, None, -3),
            # 一些高级索引:
            np.empty((0, 1, 1), dtype=np.intp),  # 空数组，可以广播
            np.array([0, 1, -2]),
            np.array([[2], [0], [1]]),
            np.array([[0, -1], [0, 1]], dtype=np.dtype('intp').newbyteorder()),
            np.array([2, -1], dtype=np.int8),
            np.zeros([1]*31, dtype=int),  # 触发太大的数组
            np.array([0., 1.])]  # 无效的数据类型
        # 一些简单的索引，涵盖更多情况
        self.simple_indices = [Ellipsis, None, -1, [1], np.array([True]),
                               'skip']
        # 简单的索引以填充其余情况
        self.fill_indices = [slice(None, None), 0]

    def _check_multi_index(self, arr, index):
        """检查多索引项的获取和简单设置。

        Parameters
        ----------
        arr : ndarray
            要索引的数组，必须是重塑后的 arange。
        index : tuple of indexing objects
            正在测试的索引。
        """
        # 测试获取索引项
        try:
            mimic_get, no_copy = self._get_multi_index(arr, index)
        except Exception as e:
            if HAS_REFCOUNT:
                prev_refcount = sys.getrefcount(arr)
            # 断言引发的异常类型与期望一致
            assert_raises(type(e), arr.__getitem__, index)
            assert_raises(type(e), arr.__setitem__, index, 0)
            if HAS_REFCOUNT:
                assert_equal(prev_refcount, sys.getrefcount(arr))
            return

        # 比较索引结果
        self._compare_index_result(arr, index, mimic_get, no_copy)
    def _check_single_index(self, arr, index):
        """Check a single index item getting and simple setting.

        Parameters
        ----------
        arr : ndarray
            要进行索引的数组，必须是一个 arange。
        index : indexing object
            被测试的索引。必须是单个索引对象，不能是索引对象的元组
            （参见 `_check_multi_index`）。
        """
        try:
            # 调用 `_get_multi_index` 方法获取模仿的索引结果和是否复制的标志
            mimic_get, no_copy = self._get_multi_index(arr, (index,))
        except Exception as e:
            if HAS_REFCOUNT:
                # 获取数组的引用计数
                prev_refcount = sys.getrefcount(arr)
            # 断言索引操作抛出的异常类型
            assert_raises(type(e), arr.__getitem__, index)
            assert_raises(type(e), arr.__setitem__, index, 0)
            if HAS_REFCOUNT:
                # 断言操作后数组的引用计数不变
                assert_equal(prev_refcount, sys.getrefcount(arr))
            return

        # 比较索引结果和模仿的索引结果
        self._compare_index_result(arr, index, mimic_get, no_copy)

    def _compare_index_result(self, arr, index, mimic_get, no_copy):
        """Compare mimicked result to indexing result.
        """
        # 复制数组以防止影响原始数据
        arr = arr.copy()
        # 执行索引操作
        indexed_arr = arr[index]
        # 断言索引操作的结果与模仿的索引结果相等
        assert_array_equal(indexed_arr, mimic_get)
        # 检查是否得到视图，除非是大小为0或者0维的数组（这时候不是视图，也不重要）
        if indexed_arr.size != 0 and indexed_arr.ndim != 0:
            # 检查是否共享内存
            assert_(np.may_share_memory(indexed_arr, arr) == no_copy)
            # 检查原始数组的引用计数
            if HAS_REFCOUNT:
                if no_copy:
                    # 如果没有复制，引用计数增加1
                    assert_equal(sys.getrefcount(arr), 3)
                else:
                    assert_equal(sys.getrefcount(arr), 2)

        # 测试非广播的赋值操作
        b = arr.copy()
        b[index] = mimic_get + 1000
        if b.size == 0:
            return  # 没有可比较的内容...
        if no_copy and indexed_arr.ndim != 0:
            # 如果没有复制且索引数组不是0维，则原地修改索引数组以操纵原始数据
            indexed_arr += 1000
            assert_array_equal(arr, b)
            return
        # 利用数组原本是 arange 的特性进行赋值操作
        arr.flat[indexed_arr.ravel()] += 1000
        assert_array_equal(arr, b)

    def test_boolean(self):
        a = np.array(5)
        # 断言布尔索引返回正确的值
        assert_equal(a[np.array(True)], 5)
        # 布尔索引赋值
        a[np.array(True)] = 1
        assert_equal(a, 1)
        # 注意：这与正常的广播操作不同，因为 arr[boolean_array] 的工作方式类似于多索引。
        # 这意味着它被对齐到左边。这对于与 arr[boolean_array,] 保持一致可能是正确的，而且根本不进行广播操作。
        self._check_multi_index(
            self.a, (np.zeros_like(self.a, dtype=bool),))
        self._check_multi_index(
            self.a, (np.zeros_like(self.a, dtype=bool)[..., 0],))
        self._check_multi_index(
            self.a, (np.zeros_like(self.a, dtype=bool)[None, ...],))
    def test_multidim(self):
        # 测试多维索引的功能

        # 捕获警告，防止 np.array(True) 在完整整数索引中被接受，当单独运行文件时
        with warnings.catch_warnings():
            # 忽略 DeprecationWarning 类型的警告
            warnings.filterwarnings('error', '', DeprecationWarning)
            # 忽略 VisibleDeprecationWarning 类型的警告
            warnings.filterwarnings('error', '', VisibleDeprecationWarning)

            # 定义一个函数，用来判断是否需要跳过某个索引
            def isskip(idx):
                return isinstance(idx, str) and idx == "skip"

            # 针对简单的位置（0、2、3）循环进行测试
            for simple_pos in [0, 2, 3]:
                # 准备索引的组合进行检查
                tocheck = [self.fill_indices, self.complex_indices,
                           self.fill_indices, self.fill_indices]
                # 将简单位置替换为简单索引
                tocheck[simple_pos] = self.simple_indices
                # 遍历所有可能的索引组合
                for index in product(*tocheck):
                    # 过滤掉需要跳过的索引，然后执行多维索引检查
                    index = tuple(i for i in index if not isskip(i))
                    self._check_multi_index(self.a, index)
                    self._check_multi_index(self.b, index)

        # 对非常简单的获取单个元素进行检查
        self._check_multi_index(self.a, (0, 0, 0, 0))
        self._check_multi_index(self.b, (0, 0, 0, 0))

        # 同时也检查（简单情况下的）索引过多的情况
        assert_raises(IndexError, self.a.__getitem__, (0, 0, 0, 0, 0))
        assert_raises(IndexError, self.a.__setitem__, (0, 0, 0, 0, 0), 0)
        assert_raises(IndexError, self.a.__getitem__, (0, 0, [1], 0, 0))
        assert_raises(IndexError, self.a.__setitem__, (0, 0, [1], 0, 0), 0)

    def test_1d(self):
        # 测试一维数组的功能
        a = np.arange(10)
        for index in self.complex_indices:
            # 对单个索引进行检查
            self._check_single_index(a, index)
class TestFloatNonIntegerArgument:
    """
    These test that ``TypeError`` is raised when you try to use
    non-integers as arguments to for indexing and slicing e.g. ``a[0.0:5]``
    and ``a[0.5]``, or other functions like ``array.reshape(1., -1)``.

    """

    def test_valid_indexing(self):
        # These should raise no errors.
        a = np.array([[[5]]])

        # 索引操作，使用整数数组
        a[np.array([0])]
        # 索引操作，使用整数列表
        a[[0, 0]]
        # 切片操作，使用整数列表
        a[:, [0, 0]]
        # 切片操作，使用整数
        a[:, 0, :]
        # 完整切片操作
        a[:, :, :]

    def test_valid_slicing(self):
        # These should raise no errors.
        a = np.array([[[5]]])

        # 完整切片操作
        a[::]
        # 从索引0开始切片
        a[0:]
        # 切片到索引1（不包含）
        a[:2]
        # 切片从索引0到2（不包含）
        a[0:2]
        # 步长为2的切片
        a[::2]
        # 从索引1开始，步长为2的切片
        a[1::2]
        # 切片从索引0到2（不包含），步长为2
        a[:2:2]
        # 切片从索引1到2（不包含），步长为2
        a[1:2:2]

    def test_non_integer_argument_errors(self):
        a = np.array([[5]])

        # 测试reshape函数使用非整数参数抛出TypeError异常
        assert_raises(TypeError, np.reshape, a, (1., 1., -1))
        assert_raises(TypeError, np.reshape, a, (np.array(1.), -1))
        # 测试take函数使用非整数参数抛出TypeError异常
        assert_raises(TypeError, np.take, a, [0], 1.)
        assert_raises(TypeError, np.take, a, [0], np.float64(1.))

    def test_non_integer_sequence_multiplication(self):
        # NumPy标量序列乘法不应使用非整数
        def mult(a, b):
            return a * b

        # 测试使用非整数参数抛出TypeError异常
        assert_raises(TypeError, mult, [1], np.float64(3))
        # 以下应该正常运行
        mult([1], np.int_(3))

    def test_reduce_axis_float_index(self):
        d = np.zeros((3, 3, 3))
        # 测试使用浮点数作为reduce操作的轴索引抛出TypeError异常
        assert_raises(TypeError, np.min, d, 0.5)
        assert_raises(TypeError, np.min, d, (0.5, 1))
        assert_raises(TypeError, np.min, d, (1, 2.2))
        assert_raises(TypeError, np.min, d, (.2, 1.2))


class TestBooleanIndexing:
    # Using a boolean as integer argument/indexing is an error.
    def test_bool_as_int_argument_errors(self):
        a = np.array([[[1]]])

        # 测试reshape函数使用布尔值参数抛出TypeError异常
        assert_raises(TypeError, np.reshape, a, (True, -1))
        assert_raises(TypeError, np.reshape, a, (np.bool(True), -1))
        # 注意，operator.index(np.array(True))不能工作，布尔数组因此也被弃用，但错误信息不同：
        assert_raises(TypeError, operator.index, np.array(True))
        assert_warns(DeprecationWarning, operator.index, np.True_)
        # 测试take函数使用布尔值参数抛出TypeError异常
        assert_raises(TypeError, np.take, args=(a, [0], False))

    def test_boolean_indexing_weirdness(self):
        # Weird boolean indexing things
        a = np.ones((2, 3, 4))
        # 使用False作为索引时返回的形状应为空
        assert a[False, True, ...].shape == (0, 2, 3, 4)
        # 使用True和其他混合索引方式返回的形状应为(1, 2)
        assert a[True, [0, 1], True, True, [1], [[2]]].shape == (1, 2)
        # 使用False和其他索引方式应该抛出IndexError异常
        assert_raises(IndexError, lambda: a[False, [0, 1], ...])
    # 定义一个测试函数，测试快速路径下的布尔索引情况
    def test_boolean_indexing_fast_path(self):
        # 创建一个3x3的全1数组
        a = np.ones((3, 3))

        # 使用错误的布尔索引，预期引发IndexError异常，错误信息为：
        # "boolean index did not match indexed array along axis 0; size of axis is 3 but size of corresponding boolean axis is 1"
        idx1 = np.array([[False]*9])
        assert_raises_regex(IndexError,
            "boolean index did not match indexed array along axis 0; "
            "size of axis is 3 but size of corresponding boolean axis is 1",
            lambda: a[idx1])

        # 使用错误的布尔索引，预期引发IndexError异常，错误信息同上
        idx2 = np.array([[False]*8 + [True]])
        assert_raises_regex(IndexError,
            "boolean index did not match indexed array along axis 0; "
            "size of axis is 3 but size of corresponding boolean axis is 1",
            lambda: a[idx2])

        # 使用正确长度的布尔索引，预期引发IndexError异常，错误信息同上
        idx3 = np.array([[False]*10])
        assert_raises_regex(IndexError,
            "boolean index did not match indexed array along axis 0; "
            "size of axis is 3 but size of corresponding boolean axis is 1",
            lambda: a[idx3])

        # 使用错误的布尔索引，预期引发IndexError异常，错误信息为：
        # "boolean index did not match indexed array along axis 1; size of axis is 1 but size of corresponding boolean axis is 2"
        a = np.ones((1, 1, 2))
        idx = np.array([[[True], [False]]])
        assert_raises_regex(IndexError,
            "boolean index did not match indexed array along axis 1; "
            "size of axis is 1 but size of corresponding boolean axis is 2",
            lambda: a[idx])
class TestArrayToIndexDeprecation:
    """Creating an index from array not 0-D is an error.

    This class tests scenarios where creating an index from arrays that are not 0-dimensional raises errors.
    """

    def test_array_to_index_error(self):
        # Define a 3-dimensional numpy array
        a = np.array([[[1]]])

        # Assert that attempting to convert a non-0-D array to index raises a TypeError
        assert_raises(TypeError, operator.index, np.array([1]))

        # Assert that reshaping 'a' with an invalid shape (-1) raises a TypeError
        assert_raises(TypeError, np.reshape, a, (a, -1))

        # Assert that using 'np.take' with invalid arguments raises a TypeError
        assert_raises(TypeError, np.take, a, [0], a)


class TestNonIntegerArrayLike:
    """Tests that array_likes are only valid if they can be safely cast to integers.

    This class tests scenarios where array-like objects should only allow safe casting to integers,
    otherwise IndexError should be raised.
    """

    def test_basic(self):
        # Create a numpy array with range 0 to 9
        a = np.arange(10)

        # Assert that attempting to access non-integer indices raises an IndexError
        assert_raises(IndexError, a.__getitem__, [0.5, 1.5])
        assert_raises(IndexError, a.__getitem__, (['1', '2'],))

        # Assert that accessing with an empty list is valid
        a.__getitem__([])


class TestMultipleEllipsisError:
    """An index can only have a single ellipsis.

    This class tests scenarios where using multiple ellipses (ellipsis objects) in numpy indexing raises an IndexError.
    """

    def test_basic(self):
        # Create a numpy array with range 0 to 9
        a = np.arange(10)

        # Assert that using multiple ellipses raises an IndexError
        assert_raises(IndexError, lambda: a[..., ...])
        assert_raises(IndexError, a.__getitem__, ((Ellipsis,) * 2,))
        assert_raises(IndexError, a.__getitem__, ((Ellipsis,) * 3,))


class TestCApiAccess:
    """Tests for C-API access functionalities.

    This class tests various functionalities related to C-API access in numpy arrays.
    """

    def test_getitem(self):
        # Define a partial function 'subscript' using 'array_indexing' function with fixed first argument as 0
        subscript = functools.partial(array_indexing, 0)

        # Test cases for '__getitem__' method

        # Assert that accessing element of a 0-dimensional array raises IndexError
        assert_raises(IndexError, subscript, np.ones(()), 0)

        # Assert that accessing out-of-bounds values raises IndexError
        assert_raises(IndexError, subscript, np.ones(10), 11)
        assert_raises(IndexError, subscript, np.ones(10), -11)
        assert_raises(IndexError, subscript, np.ones((10, 10)), 11)
        assert_raises(IndexError, subscript, np.ones((10, 10)), -11)

        # Create a numpy array with range 0 to 9
        a = np.arange(10)

        # Assert that array indexing behaves correctly for a 1-dimensional array
        assert_array_equal(a[4], subscript(a, 4))

        # Reshape 'a' to a 2-dimensional array and test indexing
        a = a.reshape(5, 2)
        assert_array_equal(a[-4], subscript(a, -4))

    def test_setitem(self):
        # Define a partial function 'assign' using 'array_indexing' function with fixed first argument as 1
        assign = functools.partial(array_indexing, 1)

        # Test cases for '__setitem__' method

        # Assert that deletion operation raises ValueError
        assert_raises(ValueError, assign, np.ones(10), 0)

        # Assert that assigning values to elements of a 0-dimensional array raises IndexError
        assert_raises(IndexError, assign, np.ones(()), 0, 0)

        # Assert that assigning to out-of-bounds indices raises IndexError
        assert_raises(IndexError, assign, np.ones(10), 11, 0)
        assert_raises(IndexError, assign, np.ones(10), -11, 0)
        assert_raises(IndexError, assign, np.ones((10, 10)), 11, 0)
        assert_raises(IndexError, assign, np.ones((10, 10)), -11, 0)

        # Create a numpy array with range 0 to 9
        a = np.arange(10)

        # Assign a new value to index 4 and assert the change
        assign(a, 4, 10)
        assert_(a[4] == 10)

        # Reshape 'a' to a 2-dimensional array and test assignment
        a = a.reshape(5, 2)
        assign(a, 4, 10)
        assert_array_equal(a[-1], [10, 10])
```