# `.\pytorch\test\torch_np\numpy_tests\core\test_indexing.py`

```py
# Owner(s): ["module: dynamo"]
# 导入必要的模块和函数
import functools  # 导入 functools 模块，用于创建偏函数
import operator   # 导入 operator 模块，用于操作符相关的函数
import re         # 导入 re 模块，用于正则表达式操作
import sys        # 导入 sys 模块，用于系统相关的操作
import warnings   # 导入 warnings 模块，用于处理警告信息

# 导入 itertools 中的 product 函数
from itertools import product  

# 导入 unittest 中的装饰器 expectedFailure 和 skipIf，以及异常类 SkipTest
from unittest import expectedFailure as xfail, skipIf as skipif, SkipTest  

# 导入 pytest 测试框架
import pytest  

# 导入 pytest 中的 raises 函数，起别名为 assert_raises
from pytest import raises as assert_raises  

# 导入 torch.testing._internal.common_utils 中的一系列函数和类
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo,
)

# 根据 TEST_WITH_TORCHDYNAMO 的值选择性导入 numpy 或 torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np  # 导入 numpy 库，并起别名 np
    from numpy.testing import (  # 导入 numpy.testing 中的断言函数
        assert_,
        assert_array_equal,
        assert_equal,
        assert_warns,
        HAS_REFCOUNT,
    )
else:
    import torch._numpy as np  # 导入 torch._numpy 库，并起别名 np
    from torch._numpy.testing import (  # 导入 torch._numpy.testing 中的断言函数
        assert_,
        assert_array_equal,
        assert_equal,
        assert_warns,
        HAS_REFCOUNT,
    )

# 使用 functools.partial 创建 skip 函数，其行为类似于 skipif(True, ...)
skip = functools.partial(skipif, True)


# 使用 instantiate_parametrized_tests 装饰器声明一个参数化测试类 TestIndexing
@instantiate_parametrized_tests
class TestIndexing(TestCase):
    # 定义测试方法 test_index_no_floats
    def test_index_no_floats(self):
        a = np.array([[[5]]])  # 创建一个 numpy 数组 a，包含一个值为 5 的元素

        # 断言各种使用浮点数作为索引的情况会引发 IndexError 异常
        assert_raises(IndexError, lambda: a[0.0])
        assert_raises(IndexError, lambda: a[0, 0.0])
        assert_raises(IndexError, lambda: a[0.0, 0])
        assert_raises(IndexError, lambda: a[0.0, :])
        assert_raises(IndexError, lambda: a[:, 0.0])
        assert_raises(IndexError, lambda: a[:, 0.0, :])
        assert_raises(IndexError, lambda: a[0.0, :, :])
        assert_raises(IndexError, lambda: a[0, 0, 0.0])
        assert_raises(IndexError, lambda: a[0.0, 0, 0])
        assert_raises(IndexError, lambda: a[0, 0.0, 0])
        assert_raises(IndexError, lambda: a[-1.4])
        assert_raises(IndexError, lambda: a[0, -1.4])
        assert_raises(IndexError, lambda: a[-1.4, 0])
        assert_raises(IndexError, lambda: a[-1.4, :])
        assert_raises(IndexError, lambda: a[:, -1.4])
        assert_raises(IndexError, lambda: a[:, -1.4, :])
        assert_raises(IndexError, lambda: a[-1.4, :, :])
        assert_raises(IndexError, lambda: a[0, 0, -1.4])
        assert_raises(IndexError, lambda: a[-1.4, 0, 0])
        assert_raises(IndexError, lambda: a[0, -1.4, 0])

        # 注意：torch 的索引参数会按深度优先进行验证，因此会优先引发 TypeError 而不是 IndexError
        # 例如：
        # >>> a = np.array([[[5]]])
        # >>> a[0.0:, 0.0]
        # IndexError: only integers, slices (`:`), ellipsis (`...`),
        # numpy.newaxis (`None`) and integer or boolean arrays are valid indices
        # >>> t = torch.as_tensor([[[5]]])  # 与 a 相同
        # >>> t[0.0:, 0.0]
        # TypeError: slice indices must be integers or None or have an __index__ method
        #
        assert_raises((IndexError, TypeError), lambda: a[0.0:, 0.0])
        assert_raises((IndexError, TypeError), lambda: a[0.0:, 0.0, :])
    # 定义一个测试方法，用于测试 NumPy 数组的切片操作不接受浮点数参数的情况
    def test_slicing_no_floats(self):
        # 创建一个包含单个元素的二维 NumPy 数组
        a = np.array([[5]])

        # 测试切片起始位置为浮点数的情况，预期会引发 TypeError 异常
        assert_raises(TypeError, lambda: a[0.0:])
        assert_raises(TypeError, lambda: a[0:, 0.0:2])
        assert_raises(TypeError, lambda: a[0.0::2, :0])
        assert_raises(TypeError, lambda: a[0.0:1:2, :])
        assert_raises(TypeError, lambda: a[:, 0.0:])

        # 测试切片结束位置为浮点数的情况，预期会引发 TypeError 异常
        assert_raises(TypeError, lambda: a[:0.0])
        assert_raises(TypeError, lambda: a[:0, 1:2.0])
        assert_raises(TypeError, lambda: a[:0.0:2, :0])
        assert_raises(TypeError, lambda: a[:0.0, :])
        assert_raises(TypeError, lambda: a[:, 0:4.0:2])

        # 测试切片步长为浮点数的情况，预期会引发 TypeError 异常
        assert_raises(TypeError, lambda: a[::1.0])
        assert_raises(TypeError, lambda: a[0:, :2:2.0])
        assert_raises(TypeError, lambda: a[1::4.0, :0])
        assert_raises(TypeError, lambda: a[::5.0, :])
        assert_raises(TypeError, lambda: a[:, 0:4:2.0])

        # 测试混合使用浮点数的情况，预期会引发 TypeError 异常
        assert_raises(TypeError, lambda: a[1.0:2:2.0])
        assert_raises(TypeError, lambda: a[1.0::2.0])
        assert_raises(TypeError, lambda: a[0:, :2.0:2.0])
        assert_raises(TypeError, lambda: a[1.0:1:4.0, :0])
        assert_raises(TypeError, lambda: a[1.0:5.0:5.0, :])
        assert_raises(TypeError, lambda: a[:, 0.4:4.0:2.0])

        # 测试步长为零时是否仍会触发 DeprecationWarning
        assert_raises(TypeError, lambda: a[::0.0])

    @skip(reason="torch allows slicing with non-0d array components")
    # 定义一个测试方法，用于测试在 NumPy 中使用非零维度数组作为索引是否会引发异常（但 PyTorch 允许）
    def test_index_no_array_to_index(self):
        # 创建一个包含单个元素的三维 NumPy 数组
        a = np.array([[[1]]])

        # 测试使用数组作为索引时是否会引发 TypeError 异常
        assert_raises(TypeError, lambda: a[a:a:a])

        # NumPy 中使用标量作为索引不会引发异常，例如：
        #
        #     >>> i = np.int64(1)
        #     >>> a[i:i:i]
        #     array([], shape=(0, 1, 1), dtype=int64)
        #

    # 定义一个测试方法，用于测试在 NumPy 中使用 None 作为索引时的行为
    def test_none_index(self):
        # 创建一个包含三个元素的一维 NumPy 数组
        a = np.array([1, 2, 3])

        # 使用 None 作为索引会添加一个新的轴（即增加数组的维度）
        assert_equal(a[None], a[np.newaxis])
        assert_equal(a[None].ndim, a.ndim + 1)

    @skip
    # 定义一个测试方法，但由于装饰器 @skip，不会被运行
    def test_empty_tuple_index(self):
        # 创建一个包含三个元素的一维 NumPy 数组
        a = np.array([1, 2, 3])

        # 使用空元组作为索引会创建一个视图
        assert_equal(a[()], a)
        assert_(a[()].tensor._base is a.tensor)

        # 创建一个标量的 NumPy 数组
        a = np.array(0)

        # 确保空元组索引适用于标量，返回标量自身
        assert_(isinstance(a[()], np.int_))
    def test_same_kind_index_casting(self):
        # Indexes should be cast with same-kind and not safe, even if that
        # is somewhat unsafe. So test various different code paths.
        # 创建一个从 0 到 4 的整数数组作为索引
        index = np.arange(5)
        # 将整数数组转换为无符号 8 位整数数组作为索引
        u_index = index.astype(np.uint8)  # i.e. cast to default uint indexing dtype
        # 创建一个从 0 到 9 的整数数组
        arr = np.arange(10)

        # 验证使用整数数组索引和无符号整数数组索引得到的值是相等的
        assert_array_equal(arr[index], arr[u_index])
        # 使用无符号整数数组索引对数组进行赋值
        arr[u_index] = np.arange(5)
        # 验证数组的内容与预期相符
        assert_array_equal(arr, np.arange(10))

        # 创建一个 0 到 9 的整数数组，reshape 成 5 行 2 列的二维数组
        arr = np.arange(10).reshape(5, 2)
        # 验证使用整数数组索引和无符号整数数组索引得到的值是相等的
        assert_array_equal(arr[index], arr[u_index])

        # 使用无符号整数数组索引对二维数组进行赋值，每个元素都扩展成一列
        arr[u_index] = np.arange(5)[:, None]
        # 验证数组的内容与预期相符
        assert_array_equal(arr, np.arange(5)[:, None].repeat(2, axis=1))

        # 创建一个从 0 到 24 的整数数组，reshape 成 5 行 5 列的二维数组
        arr = np.arange(25).reshape(5, 5)
        # 验证使用整数数组索引和无符号整数数组索引得到的值是相等的
        assert_array_equal(arr[u_index, u_index], arr[index, index])

    def test_empty_fancy_index(self):
        # Empty list index creates an empty array
        # with the same dtype (but with weird shape)
        # 创建一个包含整数 1、2、3 的数组
        a = np.array([1, 2, 3])
        # 验证空列表索引得到一个空数组
        assert_equal(a[[]], [])
        # 验证空列表索引得到的数组的数据类型与原数组相同
        assert_equal(a[[]].dtype, a.dtype)

        # 创建一个空数组 b，数据类型为 intp（平台相关的整数）
        b = np.array([], dtype=np.intp)
        # 验证空列表索引得到一个空数组
        assert_equal(a[[]], [])
        # 验证空列表索引得到的数组的数据类型与原数组相同
        assert_equal(a[[]].dtype, a.dtype)

        # 创建一个空数组 b
        b = np.array([])
        # 验证尝试用空数组作为索引会抛出 IndexError 异常
        assert_raises(IndexError, a.__getitem__, b)

    def test_ellipsis_index(self):
        # 创建一个 3x3 的二维数组
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # 验证使用省略号索引返回的对象不是原数组对象
        assert_(a[...] is not a)
        # 验证使用省略号索引得到的数组与原数组相等
        assert_equal(a[...], a)
        # 在 numpy <1.9 版本中，`a[...]` 会直接返回原数组 `a`

        # 使用省略号在多维数组上进行切片，可以跳过任意数量的维度
        assert_equal(a[0, ...], a[0])
        assert_equal(a[0, ...], a[0, :])
        assert_equal(a[..., 0], a[:, 0])

        # 使用省略号在多维数组上进行切片，得到的始终是数组，而不是标量
        assert_equal(a[0, ..., 1], np.array(2))

        # 对于零维数组，使用 `(Ellipsis,)` 进行赋值
        b = np.array(1)
        b[(Ellipsis,)] = 2
        assert_equal(b, 2)

    @xpassIfTorchDynamo  # 'torch_.np.array() does not have base attribute.
    def test_ellipsis_index_2(self):
        # 创建一个 3x3 的二维数组
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # 验证使用省略号索引返回的对象不是原数组对象
        assert_(a[...] is not a)
        # 验证使用省略号索引得到的数组与原数组相等
        assert_equal(a[...], a)
        # 在 numpy <1.9 版本中，`a[...]` 会直接返回原数组 `a`
        # 验证使用省略号索引得到的数组的基础属性指向原数组
        assert_(a[...].base is a)

    def test_single_int_index(self):
        # Single integer index selects one row
        # 创建一个 3x3 的二维数组
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # 验证使用单个整数索引可以选择一行
        assert_equal(a[0], [1, 2, 3])
        assert_equal(a[-1], [7, 8, 9])

        # 验证超出索引范围会产生 IndexError 异常
        assert_raises(IndexError, a.__getitem__, 1 << 30)
        # 验证索引溢出会产生 IndexError 异常
        # 注意，torch 在此处会引发 RuntimeError
        assert_raises((IndexError, RuntimeError), a.__getitem__, 1 << 64)

    def test_single_bool_index(self):
        # Single boolean index
        # 创建一个 3x3 的二维数组
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # 验证使用单个布尔值索引返回的数组与原数组的子集相等
        assert_equal(a[np.array(True)], a[None])
        assert_equal(a[np.array(False)], a[None][0:0])
    def test_boolean_shape_mismatch(self):
        # 创建一个形状为 (5, 4, 3) 的全一数组
        arr = np.ones((5, 4, 3))

        # 创建一个长度为 1 的布尔数组作为索引，预期引发 IndexError 异常
        index = np.array([True])
        assert_raises(IndexError, arr.__getitem__, index)

        # 创建一个长度为 6 的全 False 布尔数组作为索引，预期引发 IndexError 异常
        index = np.array([False] * 6)
        assert_raises(IndexError, arr.__getitem__, index)

        # 创建一个形状为 (4, 4) 的全零布尔数组作为索引，预期引发 IndexError 异常
        index = np.zeros((4, 4), dtype=bool)
        assert_raises(IndexError, arr.__getitem__, index)

        # 使用切片 None 和上述布尔数组作为索引，预期引发 IndexError 异常
        assert_raises(IndexError, arr.__getitem__, (slice(None), index))

    def test_boolean_indexing_onedim(self):
        # 使用长度为 1 的布尔数组对二维数组进行索引
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([True], dtype=bool)
        assert_equal(a[b], a)
        # 对布尔数组的真值位置赋值为 1.0
        a[b] = 1.0
        assert_equal(a, [[1.0, 1.0, 1.0]])

    @skip(reason="NP_VER: fails on CI")
    def test_boolean_assignment_value_mismatch(self):
        # 当值的形状无法广播到订阅时，布尔赋值应该失败
        a = np.arange(4)

        def f(a, v):
            a[a > -1] = v

        # 预期引发 RuntimeError、ValueError 或 TypeError 异常
        assert_raises((RuntimeError, ValueError, TypeError), f, a, [])
        assert_raises((RuntimeError, ValueError, TypeError), f, a, [1, 2, 3])
        assert_raises((RuntimeError, ValueError, TypeError), f, a[:1], [1, 2, 3])

    def test_boolean_indexing_twodim(self):
        # 使用二维布尔数组对二维数组进行索引
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = np.array([[True, False, True], [False, True, False], [True, False, True]])
        assert_equal(a[b], [1, 3, 5, 7, 9])
        assert_equal(a[b[1]], [[4, 5, 6]])
        assert_equal(a[b[0]], a[b[2]])

        # 对布尔数组的真值位置赋值为 0
        a[b] = 0
        assert_equal(a, [[0, 2, 0], [4, 0, 6], [0, 8, 0]])

    def test_boolean_indexing_list(self):
        # 回归测试 #13715。这是一个潜在的 use-after-free bug，虽然这个测试不能直接捕获到该 bug，但可以在 valgrind 中显示出来。
        a = np.array([1, 2, 3])
        b = [True, False, True]
        # 测试两个变体，因为第一个采用快速路径
        assert_equal(a[b], [1, 3])
        assert_equal(a[None, b], [[1, 3]])

    def test_reverse_strides_and_subspace_bufferinit(self):
        # 测试在简单和子空间花式索引中不反转步幅。
        a = np.ones(5)
        b = np.zeros(5, dtype=np.intp)[::-1]
        c = np.arange(5)[::-1]

        # 使用 c 的值在 a 中以 b 为索引进行赋值
        a[b] = c
        # 如果步幅没有反转，arange 中的 0 将会处于最后。
        assert_equal(a[0], 0)

        # 同时测试子空间缓冲区初始化：
        a = np.ones((5, 2))
        c = np.arange(10).reshape(5, 2)[::-1]
        # 使用 c 的值在 a 中以 b 为索引进行赋值
        a[b, :] = c
        assert_equal(a[0], [0, 1])
    def test_reversed_strides_result_allocation(self):
        # 测试修复了一个计算结果数组输出步长的 bug
        # 当子空间大小为 1 时（同时也测试其他情况）
        
        # 创建一个列向量
        a = np.arange(10)[:, None]
        # 创建一个逆序的索引数组
        i = np.arange(10)[::-1]
        # 验证通过逆序索引访问数组与通过逆序拷贝后的索引访问结果是否一致
        assert_array_equal(a[i], a[i.copy("C")])

        # 创建一个二维数组
        a = np.arange(20).reshape(-1, 2)

    def test_uncontiguous_subspace_assignment(self):
        # 在开发过程中曾经有一个 bug，基于 ndim 而不是 size 激活了跳过逻辑
        
        # 创建一个三维全为 -1 的数组
        a = np.full((3, 4, 2), -1)
        b = np.full((3, 4, 2), -1)

        # 使用非连续索引为子空间赋值
        a[[0, 1]] = np.arange(2 * 4 * 2).reshape(2, 4, 2).T
        b[[0, 1]] = np.arange(2 * 4 * 2).reshape(2, 4, 2).T.copy()

        # 验证数组 a 和 b 是否相等
        assert_equal(a, b)

    @skip(reason="torch does not limit dims to 32")
    def test_too_many_fancy_indices_special_case(self):
        # 只是记录行为，这是一个小限制
        
        # 创建一个 32 维全为 1 的数组，32 是 NPY_MAXDIMS
        a = np.ones((1,) * 32)
        # 验证当数组维度超过限制时是否抛出 IndexError
        assert_raises(IndexError, a.__getitem__, (np.array([0]),) * 32)

    def test_scalar_array_bool(self):
        # NumPy 布尔值可以用作布尔索引（Python 中的布尔值目前不行）
        
        # 创建一个标量数组
        a = np.array(1)
        # 验证使用 np.bool_(True) 和 np.array(True) 作为索引是否相等
        assert_equal(a[np.bool_(True)], a[np.array(True)])
        assert_equal(a[np.bool_(False)], a[np.array(False)])

        # 在废弃布尔值作为整数之后：
        # a = np.array([0,1,2])
        # assert_equal(a[True, :], a[None, :])
        # assert_equal(a[:, True], a[:, None])
        #
        # assert_(not np.may_share_memory(a, a[True, :]))

    def test_everything_returns_views(self):
        # 在之前 `...` 会返回数组本身
        
        # 创建一个一维数组
        a = np.arange(5)

        # 验证通过不同的索引方式得到的是否是数组的视图
        assert_(a is not a[()])
        assert_(a is not a[...])
        assert_(a is not a[:])

    def test_broaderrors_indexing(self):
        # 验证索引时的广播错误
        
        # 创建一个 5x5 的全零数组
        a = np.zeros((5, 5))
        # 验证使用不合法的索引是否抛出 IndexError
        assert_raises(IndexError, a.__getitem__, ([0, 1], [0, 1, 2]))
        assert_raises(IndexError, a.__setitem__, ([0, 1], [0, 1, 2]), 0)

    def test_trivial_fancy_out_of_bounds(self):
        # 测试当索引超出边界时的情况
        
        # 创建一个长度为 5 的全零数组
        a = np.zeros(5)
        # 创建一个长度为 20 的全一整数索引数组
        ind = np.ones(20, dtype=np.intp)
        ind[-1] = 10
        # 验证当索引超出边界时是否抛出 IndexError
        assert_raises(IndexError, a.__getitem__, ind)
        assert_raises((IndexError, RuntimeError), a.__setitem__, ind, 0)
        ind = np.ones(20, dtype=np.intp)
        ind[0] = 11
        # 验证当索引超出边界时是否抛出 IndexError
        assert_raises(IndexError, a.__getitem__, ind)
        assert_raises((IndexError, RuntimeError), a.__setitem__, ind, 0)

    def test_trivial_fancy_not_possible(self):
        # 测试在索引非连续或非一维情况下，是否正确选择非快速路径
        
        # 创建一个长度为 6 的数组
        a = np.arange(6)
        # 创建一个非连续非一维索引数组
        idx = np.arange(6, dtype=np.intp).reshape(2, 1, 3)[:, :, 0]
        # 验证通过这种索引获取的结果是否与索引数组相等
        assert_array_equal(a[idx], idx)

        # 这种情况不应该进入快速路径，注意 idx 是一个非连续非一维数组
        a[idx] = -1
        # 预期结果是将索引为 0 和 3 的元素赋值为 -1
        res = np.arange(6)
        res[0] = -1
        res[3] = -1
        assert_array_equal(a, res)
    def test_memory_order(self):
        # This is not necessary to preserve. Memory layouts for
        # more complex indices are not as simple.
        # 创建一个包含 0 到 9 的一维数组 a
        a = np.arange(10)
        # 创建一个二维数组 b，形状为 (2, 5)，转置后为 (5, 2)
        b = np.arange(10).reshape(5, 2).T
        # 检查通过复杂索引 a[b] 是否按列连续存储
        assert_(a[b].flags.f_contiguous)

        # Takes a different implementation branch:
        # 将数组 a 重塑为列数为 1 的二维数组
        a = a.reshape(-1, 1)
        # 检查通过复杂索引 a[b, 0] 是否按列连续存储
        assert_(a[b, 0].flags.f_contiguous)

    @skipIfTorchDynamo()  # XXX: flaky, depends on implementation details
    def test_small_regressions(self):
        # Reference count of intp for index checks
        # 创建一个包含单个元素 0 的数组 a
        a = np.array([0])
        if HAS_REFCOUNT:
            # 获取 np.intp 类型的引用计数
            refcount = sys.getrefcount(np.dtype(np.intp))
        # 使用 np.array([0], dtype=np.intp) 对 a 进行赋值操作
        a[np.array([0], dtype=np.intp)] = 1
        # 使用 np.array([0], dtype=np.uint8) 对 a 进行赋值操作
        a[np.array([0], dtype=np.uint8)] = 1
        # 断言尝试使用索引 np.array([1], dtype=np.intp) 设置 a 的元素会引发 IndexError
        assert_raises(IndexError, a.__setitem__, np.array([1], dtype=np.intp), 1)
        # 断言尝试使用索引 np.array([1], dtype=np.uint8) 设置 a 的元素会引发 IndexError
        assert_raises(IndexError, a.__setitem__, np.array([1], dtype=np.uint8), 1)

        if HAS_REFCOUNT:
            # 断言 np.intp 类型的引用计数与之前相同
            assert_equal(sys.getrefcount(np.dtype(np.intp)), refcount)

    def test_tuple_subclass(self):
        # 创建一个全为 1 的 5x5 数组 arr
        arr = np.ones((5, 5))

        # A tuple subclass should also be an nd-index
        # 创建一个元组的子类 TupleSubclass
        class TupleSubclass(tuple):
            pass

        index = ([1], [1])
        index = TupleSubclass(index)
        # 断言 arr[index] 的形状为 (1,)
        assert_(arr[index].shape == (1,))
        # 与非 nd-index 不同，断言 arr[index,] 的形状不为 (1,)
        assert_(arr[index,].shape != (1,))

    @xpassIfTorchDynamo  # (reason="XXX: low-prio behaviour to support")
    def test_broken_sequence_not_nd_index(self):
        # See https://github.com/numpy/numpy/issues/5063
        # If we have an object which claims to be a sequence, but fails
        # on item getting, this should not be converted to an nd-index (tuple)
        # If this object happens to be a valid index otherwise, it should work
        # This object here is very dubious and probably bad though:
        # 创建一个类 SequenceLike，模拟一个序列对象，但在获取项时会抛出 IndexError
        class SequenceLike:
            def __index__(self):
                return 0

            def __len__(self):
                return 1

            def __getitem__(self, item):
                raise IndexError("Not possible")

        arr = np.arange(10)
        # 断言 arr[SequenceLike()] 与 arr[SequenceLike(),] 的结果相等
        assert_array_equal(arr[SequenceLike()], arr[SequenceLike(),])

        # also test that field indexing does not segfault
        # for a similar reason, by indexing a structured array
        # 创建一个结构化数组 arr，包含一个字段为 "f1" 和 "f2" 的元组，值均为 0
        arr = np.zeros((1,), dtype=[("f1", "i8"), ("f2", "i8")])
        # 断言 arr[SequenceLike()] 与 arr[SequenceLike(),] 的结果相等
        assert_array_equal(arr[SequenceLike()], arr[SequenceLike(),])
    def test_indexing_array_weird_strides(self):
        # See also gh-6221
        # the shapes used here come from the issue and create the correct
        # size for the iterator buffering size.
        x = np.ones(10)
        x2 = np.ones((10, 2))
        ind = np.arange(10)[:, None, None, None]
        ind = np.broadcast_to(ind, (10, 55, 4, 4))

        # single advanced index case
        assert_array_equal(x[ind], x[ind.copy()])
        # higher dimensional advanced index
        zind = np.zeros(4, dtype=np.intp)
        assert_array_equal(x2[ind, zind], x2[ind.copy(), zind])

    def test_indexing_array_negative_strides(self):
        # From gh-8264,
        # core dumps if negative strides are used in iteration
        arro = np.zeros((4, 4))
        arr = arro[::-1, ::-1]

        slices = (slice(None), [0, 1, 2, 3])
        arr[slices] = 10
        assert_array_equal(arr, 10.0)

    @parametrize("index", [True, False, np.array([0])])
    @parametrize("num", [32, 40])
    @parametrize("original_ndim", [1, 32])
    def test_too_many_advanced_indices(self, index, num, original_ndim):
        # These are limitations based on the number of arguments we can process.
        # For `num=32` (and all boolean cases), the result is actually defined;
        # but the use of NpyIter (NPY_MAXARGS) limits it for technical reasons.
        if not (isinstance(index, np.ndarray) and original_ndim < num):
            # unskipped cases fail because of assigning too many indices
            raise SkipTest("torch does not limit dims to 32")

        arr = np.ones((1,) * original_ndim)
        with pytest.raises(IndexError):
            arr[(index,) * num]
        with pytest.raises(IndexError):
            arr[(index,) * num] = 1.0

    def test_nontuple_ndindex(self):
        a = np.arange(25).reshape((5, 5))
        assert_equal(a[[0, 1]], np.array([a[0], a[1]]))
        assert_equal(a[[0, 1], [0, 1]], np.array([0, 6]))
        raise SkipTest(
            "torch happily consumes non-tuple sequences with multi-axis "
            "indices (i.e. slices) as an index, whereas NumPy invalidates "
            "them, assumedly to keep things simple. This invalidation "
            "behaviour is just too niche to bother emulating."
        )
        assert_raises(IndexError, a.__getitem__, [slice(None)])



        # Test case for handling arrays with weird strides during indexing
        def test_indexing_array_weird_strides(self):
            # See also GitHub issue #6221
            # The shapes used here resolve the issue regarding iterator buffering size.
            x = np.ones(10)
            x2 = np.ones((10, 2))
            ind = np.arange(10)[:, None, None, None]
            ind = np.broadcast_to(ind, (10, 55, 4, 4))

            # Testing single advanced index case
            assert_array_equal(x[ind], x[ind.copy()])
            # Testing higher dimensional advanced index
            zind = np.zeros(4, dtype=np.intp)
            assert_array_equal(x2[ind, zind], x2[ind.copy(), zind])

        # Test case for handling arrays with negative strides during indexing
        def test_indexing_array_negative_strides(self):
            # Derived from GitHub issue #8264,
            # Core dumps if negative strides are used in iteration
            arro = np.zeros((4, 4))
            arr = arro[::-1, ::-1]

            slices = (slice(None), [0, 1, 2, 3])
            arr[slices] = 10
            assert_array_equal(arr, 10.0)

        @parametrize("index", [True, False, np.array([0])])
        @parametrize("num", [32, 40])
        @parametrize("original_ndim", [1, 32])
        def test_too_many_advanced_indices(self, index, num, original_ndim):
            # These tests check limitations based on the number of arguments that can be processed.
            # For `num=32` (and all boolean cases), the result is actually defined;
            # however, NpyIter (NPY_MAXARGS) imposes limits for technical reasons.
            if not (isinstance(index, np.ndarray) and original_ndim < num):
                # Cases that are not skipped fail due to attempting to assign too many indices
                raise SkipTest("torch does not limit dims to 32")

            arr = np.ones((1,) * original_ndim)
            with pytest.raises(IndexError):
                arr[(index,) * num]
            with pytest.raises(IndexError):
                arr[(index,) * num] = 1.0

        # Test case for handling non-tuple ndindex behavior
        def test_nontuple_ndindex(self):
            a = np.arange(25).reshape((5, 5))
            assert_equal(a[[0, 1]], np.array([a[0], a[1]]))
            assert_equal(a[[0, 1], [0, 1]], np.array([0, 6]))
            raise SkipTest(
                "torch happily consumes non-tuple sequences with multi-axis "
                "indices (i.e. slices) as an index, whereas NumPy invalidates "
                "them, assumedly to keep things simple. This invalidation "
                "behavior is just too niche to bother emulating."
            )
            assert_raises(IndexError, a.__getitem__, [slice(None)])
@instantiate_parametrized_tests
class TestBroadcastedAssignments(TestCase):
    # 测试类，用于测试广播赋值的各种情况

    def assign(self, a, ind, val):
        # 将数组 a 中指定索引 ind 处的值赋为 val，并返回更新后的数组 a
        a[ind] = val
        return a

    def test_prepending_ones(self):
        # 测试将各种形式的 ones 数组赋值给数组 a 的不同子空间

        a = np.zeros((3, 2))

        # 将形状为 (1, 3, 2) 的 ones 数组赋给数组 a 的全局视图
        a[...] = np.ones((1, 3, 2))
        
        # 在指定的行索引 [0, 1, 2] 处，将形状为 (1, 3, 2) 的 ones 数组赋给数组 a
        # 这里涉及了 fancy indexing 和广播
        a[[0, 1, 2], :] = np.ones((1, 3, 2))
        
        # 在列索引 [0, 1] 处，将形状为 (1, 3, 2) 的 ones 数组赋给数组 a
        # 同样涉及了 fancy indexing 和广播
        a[:, [0, 1]] = np.ones((1, 3, 2))
        
        # 在特定索引 [[0], [1], [2]] 处，将形状为 (1, 3, 2) 的 ones 数组赋给数组 a
        # 这里涉及了复杂的 fancy indexing 和广播
        a[[[0], [1], [2]], [0, 1]] = np.ones((1, 3, 2))

    def test_prepend_not_one(self):
        # 测试赋值非 ones 数组给数组 a 的情况

        assign = self.assign
        s_ = np.s_
        a = np.zeros(5)

        # 尝试将形状为 (2, 1) 的 ones 数组赋给数组 a 的全局视图
        # 预期会抛出 ValueError 或 RuntimeError 异常
        try:
            assign(a, s_[...], np.ones((2, 1)))
        except Exception as e:
            self.assertTrue(isinstance(e, (ValueError, RuntimeError)))
        
        # 使用 assert_raises 验证将形状为 (2, 1) 的 ones 数组赋给数组 a 的 fancy 索引 [1, 2, 3] 的情况
        assert_raises(
            (ValueError, RuntimeError), assign, a, s_[[1, 2, 3],], np.ones((2, 1))
        )
        
        # 使用 assert_raises 验证将形状为 (2, 2, 1) 的 ones 数组赋给数组 a 的 fancy 索引 [[1], [2]] 的情况
        assert_raises(
            (ValueError, RuntimeError), assign, a, s_[[[1], [2]],], np.ones((2, 2, 1))
        )

    def test_simple_broadcasting_errors(self):
        # 测试简单的广播错误情况

        assign = self.assign
        s_ = np.s_
        a = np.zeros((5, 1))

        # 尝试将形状为 (5, 2) 的 zeros 数组赋给数组 a 的全局视图
        # 预期会抛出 ValueError 或 RuntimeError 异常
        try:
            assign(a, s_[...], np.zeros((5, 2)))
        except Exception as e:
            self.assertTrue(isinstance(e, (ValueError, RuntimeError)))
        
        # 尝试将形状为 (5, 0) 的 zeros 数组赋给数组 a 的全局视图
        # 预期会抛出 ValueError 或 RuntimeError 异常
        try:
            assign(a, s_[...], np.zeros((5, 0)))
        except Exception as e:
            self.assertTrue(isinstance(e, (ValueError, RuntimeError)))
        
        # 使用 assert_raises 验证将形状为 (5, 2) 的 zeros 数组赋给数组 a 的列索引 [0] 的情况
        assert_raises(
            (ValueError, RuntimeError), assign, a, s_[:, [0]], np.zeros((5, 2))
        )
        
        # 使用 assert_raises 验证将形状为 (5, 0) 的 zeros 数组赋给数组 a 的列索引 [0] 的情况
        assert_raises(
            (ValueError, RuntimeError), assign, a, s_[:, [0]], np.zeros((5, 0))
        )
        
        # 使用 assert_raises 验证将形状为 (2, 1) 的 zeros 数组赋给数组 a 的行索引 [0] 的情况
        assert_raises(
            (ValueError, RuntimeError), assign, a, s_[[0], :], np.zeros((2, 1))
        )

    @parametrize(
        "index", [(..., [1, 2], slice(None)), ([0, 1], ..., 0), (..., [1, 2], [1, 2])]
    )
    def test_broadcast_error_reports_correct_shape(self, index):
        # 测试广播错误报告正确的形状信息

        values = np.zeros((100, 100))  # will never broadcast below

        arr = np.zeros((3, 4, 5, 6, 7))

        # 使用 pytest.raises 验证在给定索引 index 处赋值 values 时是否会抛出 ValueError 或 RuntimeError 异常
        with pytest.raises((ValueError, RuntimeError)) as e:
            arr[index] = values

        shape = arr[index].shape
        r_inner_shape = "".join(f"{side}, ?" for side in shape[:-1]) + str(shape[-1])
        # 使用正则表达式验证异常信息是否包含正确的形状信息
        assert re.search(rf"[\(\[]{r_inner_shape}[\]\)]$", str(e.value))

    def test_index_is_larger(self):
        # 测试当索引比数组的维度更大时的情况

        # 简单的 fancy 索引广播赋值示例，将 [2, 3, 4] 赋给数组 a 的特定位置
        a = np.zeros((5, 5))
        a[[[0], [1], [2]], [0, 1, 2]] = [2, 3, 4]

        # 使用 assert_ 验证赋值是否成功
        assert_((a[:3, :3] == [2, 3, 4]).all())

    def test_broadcast_subspace(self):
        # 测试使用 fancy 索引给数组的子空间赋值的情况

        a = np.zeros((100, 100))
        v = np.arange(100)[:, None]
        b = np.arange(100)[::-1]
        # 将数组 v 赋给数组 a 的特定位置，涉及 fancy 索引和广播
        a[b] = v
        # 使用 assert_ 验证赋值是否成功
        assert_((a[::-1] == v).all())


class TestFancyIndexingCast(TestCase):
    @xpassIfTorchDynamo  # (
    # 定义一个测试方法，用于测试布尔索引和浮点数组的赋值行为
    def test_boolean_index_cast_assign(self):
        # 设置布尔索引和浮点数组
        shape = (8, 63)
        bool_index = np.zeros(shape).astype(bool)  # 创建一个布尔类型的全零数组
        bool_index[0, 1] = True  # 将指定位置设置为 True
        zero_array = np.zeros(shape)  # 创建一个全零数组

        # 对浮点数组进行赋值操作
        zero_array[bool_index] = np.array([1])
        assert_equal(zero_array[0, 1], 1)  # 验证赋值结果是否符合预期

        # 使用复杂索引进行赋值，尽管会出现类型转换警告
        assert_warns(
            np.ComplexWarning, zero_array.__setitem__, ([0], [1]), np.array([2 + 1j])
        )
        assert_equal(zero_array[0, 1], 2)  # 验证赋值后的结果，确保没有复数部分

        # 将复数转换为浮点数进行赋值，并丢弃虚部
        assert_warns(
            np.ComplexWarning, zero_array.__setitem__, bool_index, np.array([1j])
        )
        assert_equal(zero_array[0, 1], 0)  # 验证赋值后的结果是否符合预期
@xfail  # (reason="XXX: requires broadcast() and broadcast_to()")
class TestMultiIndexingAutomated(TestCase):
    """
    这些测试用例使用代码来模拟 C 代码中的索引选择。

    NOTE:

        * 这些测试仍然缺少复杂项设置的测试。
        * 如果你改变了索引操作的行为，可能需要修改这些测试以尝试更多的组合。
        * 此行为编写时与 numpy 版本 1.8 匹配（尽管最初版本匹配 1.7）。
        * 仅支持元组索引的模拟代码。（在编写时已测试）
        * 错误类型通常应该匹配，只要存在一个错误。对于多个错误，引发的错误通常不同。它们 *不* 被测试。

    Update 2016-11-30: 如果维护此测试变得困难，则可能不值得持续维护，可以删除。

    """

    def setupUp(self):
        self.a = np.arange(np.prod([3, 1, 5, 6])).reshape(3, 1, 5, 6)
        self.b = np.empty((3, 0, 5, 6))
        self.complex_indices = [
            "skip",
            Ellipsis,
            0,
            # 布尔索引，针对某些特殊情况的多维度需求，还需要测试全 False 的情况
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
            np.array([[0, -1], [0, 1]], dtype=np.dtype("intp")),
            np.array([2, -1], dtype=np.int8),
            np.zeros([1] * 31, dtype=int),  # 触发太大的数组
            np.array([0.0, 1.0]),
        ]  # 无效的数据类型
        # 一些更简单但仍覆盖更多的索引
        self.simple_indices = [Ellipsis, None, -1, [1], np.array([True]), "skip"]
        # 非常简单的索引，用来填充其余的部分
        self.fill_indices = [slice(None, None), 0]
    def _check_multi_index(self, arr, index):
        """Check a multi index item getting and simple setting.

        Parameters
        ----------
        arr : ndarray
            Array to be indexed, must be a reshaped arange.
        index : tuple of indexing objects
            Index being tested.
        """
        # 尝试获取索引项
        try:
            # 调用 _get_multi_index 方法获取模拟的获取结果和非复制引用
            mimic_get, no_copy = self._get_multi_index(arr, index)
        except Exception as e:
            # 如果出现异常
            if HAS_REFCOUNT:
                # 如果支持引用计数，获取之前的引用计数
                prev_refcount = sys.getrefcount(arr)
            # 断言捕获到的异常类型，调用 arr.__getitem__ 方法获取 index 处的元素
            assert_raises(type(e), arr.__getitem__, index)
            # 断言捕获到的异常类型，调用 arr.__setitem__ 方法设置 index 处的元素为 0
            assert_raises(type(e), arr.__setitem__, index, 0)
            # 如果支持引用计数，断言之前和当前的引用计数相等
            if HAS_REFCOUNT:
                assert_equal(prev_refcount, sys.getrefcount(arr))
            # 返回，结束函数调用
            return

        # 调用 _compare_index_result 方法，比较索引结果
        self._compare_index_result(arr, index, mimic_get, no_copy)

    def _check_single_index(self, arr, index):
        """Check a single index item getting and simple setting.

        Parameters
        ----------
        arr : ndarray
            Array to be indexed, must be an arange.
        index : indexing object
            Index being tested. Must be a single index and not a tuple
            of indexing objects (see also `_check_multi_index`).
        """
        # 尝试获取单个索引项
        try:
            # 调用 _get_multi_index 方法获取模拟的获取结果和非复制引用，将 index 包装成元组传递
            mimic_get, no_copy = self._get_multi_index(arr, (index,))
        except Exception as e:
            # 如果出现异常
            if HAS_REFCOUNT:
                # 如果支持引用计数，获取之前的引用计数
                prev_refcount = sys.getrefcount(arr)
            # 断言捕获到的异常类型，调用 arr.__getitem__ 方法获取 index 处的元素
            assert_raises(type(e), arr.__getitem__, index)
            # 断言捕获到的异常类型，调用 arr.__setitem__ 方法设置 index 处的元素为 0
            assert_raises(type(e), arr.__setitem__, index, 0)
            # 如果支持引用计数，断言之前和当前的引用计数相等
            if HAS_REFCOUNT:
                assert_equal(prev_refcount, sys.getrefcount(arr))
            # 返回，结束函数调用
            return

        # 调用 _compare_index_result 方法，比较索引结果
        self._compare_index_result(arr, index, mimic_get, no_copy)
    # 定义一个方法，用于比较模拟索引结果和实际索引结果
    def _compare_index_result(self, arr, index, mimic_get, no_copy):
        """Compare mimicked result to indexing result."""
        # 抛出跳过测试的异常，因为 torch 不支持子类化
        raise SkipTest("torch does not support subclassing")
        
        # 复制数组 arr
        arr = arr.copy()
        
        # 使用索引对数组 arr 进行索引操作，得到 indexed_arr
        indexed_arr = arr[index]
        
        # 断言 indexed_arr 与 mimic_get 相等
        assert_array_equal(indexed_arr, mimic_get)
        
        # 检查是否得到视图，除非数组大小为 0 或维度为 0（此时不是视图，不重要）
        if indexed_arr.size != 0 and indexed_arr.ndim != 0:
            # 断言 indexed_arr 和 arr 是否共享内存
            assert_(np.may_share_memory(indexed_arr, arr) == no_copy)
            
            # 检查原始数组 arr 的引用计数
            if HAS_REFCOUNT:
                if no_copy:
                    # 引用计数增加了一：
                    assert_equal(sys.getrefcount(arr), 3)
                else:
                    assert_equal(sys.getrefcount(arr), 2)

        # 测试非广播的设置项目：
        # 复制数组 arr，赋值给 b
        b = arr.copy()
        
        # 使用索引对数组 b 进行设置操作，赋值为 mimic_get 加 1000
        b[index] = mimic_get + 1000
        
        # 如果 b 的大小为 0，则没有比较的内容...
        if b.size == 0:
            return
        
        # 如果 no_copy 为真且 indexed_arr 的维度不为 0：
        if no_copy and indexed_arr.ndim != 0:
            # 原地修改 indexed_arr，以操纵原始数组：
            indexed_arr += 1000
            # 断言数组 arr 等于数组 b
            assert_array_equal(arr, b)
            return
        
        # 利用数组最初是 arange 的事实：
        arr.flat[indexed_arr.ravel()] += 1000
        
        # 断言数组 arr 等于数组 b
        assert_array_equal(arr, b)
    
    # 测试布尔索引
    def test_boolean(self):
        # 创建一个包含数字 5 的数组 a
        a = np.array(5)
        
        # 断言使用布尔数组索引 a 得到的值为 5
        assert_equal(a[np.array(True)], 5)
        
        # 使用布尔数组索引对 a 进行设置操作，赋值为 1
        a[np.array(True)] = 1
        
        # 断言数组 a 的值等于 1
        assert_equal(a, 1)
        
        # 注意：这与普通的广播操作不同，因为 arr[boolean_array] 的工作方式类似于多索引。
        # 这意味着它被对齐到左侧。这对于与 arr[boolean_array,] 一致性很重要，也不进行任何广播。
        
        # 使用 _check_multi_index 方法检查多索引操作
        self._check_multi_index(self.a, (np.zeros_like(self.a, dtype=bool),))
        
        # 使用 _check_multi_index 方法检查多索引操作
        self._check_multi_index(self.a, (np.zeros_like(self.a, dtype=bool)[..., 0],))
        
        # 使用 _check_multi_index 方法检查多索引操作
        self._check_multi_index(self.a, (np.zeros_like(self.a, dtype=bool)[None, ...],))
    # 定义一个测试多维索引的方法
    def test_multidim(self):
        # 自动测试第二个（或第一个）位置的复杂索引组合，以及在另一个位置的简单索引。
        with warnings.catch_warnings():
            # 忽略警告，避免 np.array(True) 在完整整数索引中被接受，在单独运行文件时。
            warnings.filterwarnings("error", "", DeprecationWarning)
            warnings.filterwarnings("error", "", np.VisibleDeprecationWarning)

            # 定义一个函数，用于判断是否需要跳过某个索引
            def isskip(idx):
                return isinstance(idx, str) and idx == "skip"

            # 遍历简单索引可能出现的位置：0, 2, 3
            for simple_pos in [0, 2, 3]:
                # 定义一个待检查的索引列表
                tocheck = [
                    self.fill_indices,
                    self.complex_indices,
                    self.fill_indices,
                    self.fill_indices,
                ]
                # 将简单索引放置在指定位置
                tocheck[simple_pos] = self.simple_indices
                # 遍历所有可能的索引组合
                for index in product(*tocheck):
                    # 过滤掉需要跳过的索引，并调用 _check_multi_index 方法检查多维索引
                    index = tuple(i for i in index if not isskip(i))
                    self._check_multi_index(self.a, index)
                    self._check_multi_index(self.b, index)

        # 检查非常简单的获取单个元素的情况：
        self._check_multi_index(self.a, (0, 0, 0, 0))
        self._check_multi_index(self.b, (0, 0, 0, 0))

        # 同时检查（简单情况下的）索引过多的情况：
        assert_raises(IndexError, self.a.__getitem__, (0, 0, 0, 0, 0))
        assert_raises(IndexError, self.a.__setitem__, (0, 0, 0, 0, 0), 0)
        assert_raises(IndexError, self.a.__getitem__, (0, 0, [1], 0, 0))
        assert_raises(IndexError, self.a.__setitem__, (0, 0, [1], 0, 0), 0)

    # 定义一个测试一维数组的方法
    def test_1d(self):
        # 创建一个包含 0 到 9 的一维数组 a
        a = np.arange(10)
        # 遍历复杂索引列表中的索引，并调用 _check_single_index 方法检查单个索引
        for index in self.complex_indices:
            self._check_single_index(a, index)
class TestFloatNonIntegerArgument(TestCase):
    """
    These test that ``TypeError`` is raised when you try to use
    non-integers as arguments to for indexing and slicing e.g. ``a[0.0:5]``
    and ``a[0.5]``, or other functions like ``array.reshape(1., -1)``.

    """

    def test_valid_indexing(self):
        # 创建一个三维 NumPy 数组 a
        a = np.array([[[5]]])

        # 正确的索引操作不应该引发错误
        a[np.array([0])]
        a[[0, 0]]
        a[:, [0, 0]]
        a[:, 0, :]
        a[:, :, :]

    def test_valid_slicing(self):
        # 创建一个三维 NumPy 数组 a
        a = np.array([[[5]]])

        # 正确的切片操作不应该引发错误
        a[::]
        a[0:]
        a[:2]
        a[0:2]
        a[::2]
        a[1::2]
        a[:2:2]
        a[1:2:2]

    def test_non_integer_argument_errors(self):
        # 创建一个二维 NumPy 数组 a
        a = np.array([[5]])

        # 断言调用 np.reshape 时使用非整数参数会引发 TypeError
        assert_raises(TypeError, np.reshape, a, (1.0, 1.0, -1))
        assert_raises(TypeError, np.reshape, a, (np.array(1.0), -1))
        # 断言调用 np.take 时使用非整数参数会引发 TypeError
        assert_raises(TypeError, np.take, a, [0], 1.0)
        assert_raises((TypeError, RuntimeError), np.take, a, [0], np.float64(1.0))

    @skip(
        reason=("torch doesn't have scalar types with distinct element-wise behaviours")
    )
    def test_non_integer_sequence_multiplication(self):
        # 定义一个函数 mult，用于执行两个数的乘法操作
        def mult(a, b):
            return a * b

        # 断言调用 mult 函数时使用非整数参数会引发 TypeError
        assert_raises(TypeError, mult, [1], np.float_(3))
        # 以下操作应该不会引发错误
        mult([1], np.int_(3))

    def test_reduce_axis_float_index(self):
        # 创建一个全零的三维 NumPy 数组 d
        d = np.zeros((3, 3, 3))
        # 断言调用 np.min 时使用非整数参数会引发 TypeError
        assert_raises(TypeError, np.min, d, 0.5)
        assert_raises(TypeError, np.min, d, (0.5, 1))
        assert_raises(TypeError, np.min, d, (1, 2.2))
        assert_raises(TypeError, np.min, d, (0.2, 1.2))


class TestBooleanIndexing(TestCase):
    # 使用布尔值作为整数参数/索引是错误的
    def test_bool_as_int_argument_errors(self):
        # 创建一个三维 NumPy 数组 a
        a = np.array([[[1]]])

        # 断言调用 np.reshape 时使用布尔值参数会引发 TypeError
        assert_raises(TypeError, np.reshape, a, (True, -1))
        # 断言调用 np.take 时使用布尔值参数会引发 TypeError
        assert_raises(TypeError, np.take, args=(a, [0], False))

        # 这里抛出 SkipTest 异常，因为 torch 在处理布尔张量时会将其视为整数，不会引发 TypeError
        raise SkipTest("torch consumes boolean tensors as ints, no bother raising here")
        assert_raises(TypeError, np.reshape, a, (np.bool_(True), -1))
        assert_raises(TypeError, operator.index, np.array(True))

    def test_boolean_indexing_weirdness(self):
        # 布尔值索引的奇怪行为
        # 创建一个全为 1 的三维 NumPy 数组 a
        a = np.ones((2, 3, 4))
        # 使用 False 作为索引，预期结果的形状应为 (0, 2, 3, 4)
        assert a[False, True, ...].shape == (0, 2, 3, 4)
        # 使用一些复杂的布尔值索引，预期结果的形状应为 (1, 2)
        assert a[True, [0, 1], True, True, [1], [[2]]].shape == (1, 2)
        # 断言使用布尔值索引时会引发 IndexError
        assert_raises(IndexError, lambda: a[False, [0, 1], ...])
    # 定义一个测试方法，测试布尔索引的快速路径处理
    def test_boolean_indexing_fast_path(self):
        # 创建一个全为1的3x3的NumPy数组
        a = np.ones((3, 3))

        # 对于这种情况，之前可能会错误地没有抛出异常，或者抛出错误的异常。
        # 使用布尔数组索引，预期会抛出IndexError异常
        idx1 = np.array([[False] * 9])
        with pytest.raises(IndexError):
            a[idx1]

        # 对于这种情况，之前可能会错误地允许操作，但预期应该抛出IndexError异常。
        # 使用布尔数组索引，预期会抛出IndexError异常
        idx2 = np.array([[False] * 8 + [True]])
        with pytest.raises(IndexError):
            a[idx2]

        # 这种情况下，与之前保持一致。上述两种情况应该像这样工作。
        # 使用布尔数组索引，预期会抛出IndexError异常
        idx3 = np.array([[False] * 10])
        with pytest.raises(IndexError):
            a[idx3]

        # 对于这种情况，之前可能会抛出ValueError异常，指示操作数无法广播在一起。
        # 创建一个形状为(1, 1, 2)的全为1的NumPy数组
        a = np.ones((1, 1, 2))
        # 创建一个布尔数组索引，形状为(1, 2, 1)
        idx = np.array([[[True], [False]]])
        with pytest.raises(IndexError):
            a[idx]
class TestArrayToIndexDeprecation(TestCase):
    """Creating an index from array not 0-D is an error."""

    def test_array_to_index_error(self):
        # 创建一个三维数组 a，用于测试
        a = np.array([[[1]]])

        # 断言调用 np.take(a, [0], a) 时会引发 TypeError 或 RuntimeError 异常
        assert_raises((TypeError, RuntimeError), np.take, a, [0], a)

        # 抛出 SkipTest 异常，跳过以下测试代码块的执行
        raise SkipTest(
            "Multi-dimensional tensors are indexable just as long as they only "
            "contain a single element, no bother raising here"
        )

        # 不会执行到这里的断言，因为前面已经抛出了 SkipTest 异常
        assert_raises(TypeError, operator.index, np.array([1]))

        # 抛出 SkipTest 异常，跳过以下测试代码块的执行
        raise SkipTest("torch consumes tensors as ints, no bother raising here")

        # 不会执行到这里的断言，因为前面已经抛出了 SkipTest 异常
        assert_raises(TypeError, np.reshape, a, (a, -1))


class TestNonIntegerArrayLike(TestCase):
    """Tests that array_likes only valid if can safely cast to integer.

    For instance, lists give IndexError when they cannot be safely cast to
    an integer.

    """

    @skip(
        reason=(
            "torch consumes floats by way of falling back on its deprecated "
            "__index__ behaviour, no bother raising here"
        )
    )
    def test_basic(self):
        # 创建一个包含 0 到 9 的数组 a
        a = np.arange(10)

        # 断言调用 a.__getitem__([0.5, 1.5]) 会引发 IndexError 异常
        assert_raises(IndexError, a.__getitem__, [0.5, 1.5])
        
        # 断言调用 a.__getitem__((["1", "2"],)) 会引发 IndexError 异常
        assert_raises(IndexError, a.__getitem__, (["1", "2"],))

        # 调用 a.__getitem__([])，表示获取空索引，有效操作
        a.__getitem__([])


class TestMultipleEllipsisError(TestCase):
    """An index can only have a single ellipsis."""

    @xfail  # (
    #    reason=(
    #        "torch currently consumes multiple ellipsis, no bother raising "
    #        "here. See https://github.com/pytorch/pytorch/issues/59787#issue-917252204"
    #    )
    # )
    def test_basic(self):
        # 创建一个包含 0 到 9 的数组 a
        a = np.arange(10)

        # 断言调用 a[..., ...] 会引发 IndexError 异常
        assert_raises(IndexError, lambda: a[..., ...])

        # 断言调用 a.__getitem__((Ellipsis,) * 2) 会引发 IndexError 异常
        assert_raises(IndexError, a.__getitem__, ((Ellipsis,) * 2,))

        # 断言调用 a.__getitem__((Ellipsis,) * 3) 会引发 IndexError 异常
        assert_raises(IndexError, a.__getitem__, ((Ellipsis,) * 3,))


if __name__ == "__main__":
    run_tests()
```