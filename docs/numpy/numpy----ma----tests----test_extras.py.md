# `.\numpy\numpy\ma\tests\test_extras.py`

```
# pylint: disable-msg=W0611, W0612, W0511
"""
Tests suite for MaskedArray.
Adapted from the original test_ma by Pierre Gerard-Marchant

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: test_extras.py 3473 2007-10-29 15:18:13Z jarrod.millman $

"""
# 导入警告模块
import warnings
# 导入 itertools 模块
import itertools
# 导入 pytest 测试框架
import pytest

# 导入 NumPy 库，并从核心部分导入数值相关函数和类
import numpy as np
from numpy._core.numeric import normalize_axis_tuple
# 从 NumPy 测试模块中导入函数和类
from numpy.testing import (
    assert_warns, suppress_warnings
    )
# 从 NumPy 掩码数组测试工具模块中导入函数和类
from numpy.ma.testutils import (
    assert_, assert_array_equal, assert_equal, assert_almost_equal
    )
# 从 NumPy 掩码核心模块中导入函数和类
from numpy.ma.core import (
    array, arange, masked, MaskedArray, masked_array, getmaskarray, shape,
    nomask, ones, zeros, count
    )
# 从 NumPy 掩码额外工具模块中导入函数和类
from numpy.ma.extras import (
    atleast_1d, atleast_2d, atleast_3d, mr_, dot, polyfit, cov, corrcoef,
    median, average, unique, setxor1d, setdiff1d, union1d, intersect1d, in1d,
    ediff1d, apply_over_axes, apply_along_axis, compress_nd, compress_rowcols,
    mask_rowcols, clump_masked, clump_unmasked, flatnotmasked_contiguous,
    notmasked_contiguous, notmasked_edges, masked_all, masked_all_like, isin,
    diagflat, ndenumerate, stack, vstack, _covhelper
    )


class TestGeneric:
    # 测试 masked_all 函数
    def test_masked_all(self):
        # Tests masked_all
        # Standard dtype
        # 测试标准数据类型
        test = masked_all((2,), dtype=float)
        control = array([1, 1], mask=[1, 1], dtype=float)
        assert_equal(test, control)
        
        # Flexible dtype
        # 测试灵活数据类型
        dt = np.dtype({'names': ['a', 'b'], 'formats': ['f', 'f']})
        test = masked_all((2,), dtype=dt)
        control = array([(0, 0), (0, 0)], mask=[(1, 1), (1, 1)], dtype=dt)
        assert_equal(test, control)
        
        test = masked_all((2, 2), dtype=dt)
        control = array([[(0, 0), (0, 0)], [(0, 0), (0, 0)]],
                        mask=[[(1, 1), (1, 1)], [(1, 1), (1, 1)]],
                        dtype=dt)
        assert_equal(test, control)
        
        # Nested dtype
        # 测试嵌套数据类型
        dt = np.dtype([('a', 'f'), ('b', [('ba', 'f'), ('bb', 'f')])])
        test = masked_all((2,), dtype=dt)
        control = array([(1, (1, 1)), (1, (1, 1))],
                        mask=[(1, (1, 1)), (1, (1, 1))], dtype=dt)
        assert_equal(test, control)
        
        test = masked_all((1, 1), dtype=dt)
        control = array([[(1, (1, 1))]], mask=[[(1, (1, 1))]], dtype=dt)
        assert_equal(test, control)
    def test_masked_all_with_object_nested(self):
        # Test masked_all works with nested array with dtype of an 'object'
        # refers to issue #15895
        
        # 定义一个自定义的数据类型，包含一个嵌套数组 'c'，整体形状为 (1,)
        my_dtype = np.dtype([('b', ([('c', object)], (1,)))])
        
        # 创建一个形状为 (1,) 的 masked array，使用上面定义的数据类型
        masked_arr = np.ma.masked_all((1,), my_dtype)

        # 断言 'masked_arr['b']' 的类型为 MaskedArray 对象
        assert_equal(type(masked_arr['b']), np.ma.core.MaskedArray)
        
        # 断言 'masked_arr['b']['c']' 的类型为 MaskedArray 对象
        assert_equal(type(masked_arr['b']['c']), np.ma.core.MaskedArray)
        
        # 断言 'masked_arr['b']['c']' 的长度为 1
        assert_equal(len(masked_arr['b']['c']), 1)
        
        # 断言 'masked_arr['b']['c']' 的形状为 (1, 1)
        assert_equal(masked_arr['b']['c'].shape, (1, 1))
        
        # 断言 'masked_arr['b']['c']._fill_value' 的形状为空元组 ()
        assert_equal(masked_arr['b']['c']._fill_value.shape, ())

    def test_masked_all_with_object(self):
        # same as above except that the array is not nested
        # 与上面相同，不过这次数组不是嵌套的
        
        # 定义一个自定义的数据类型，包含一个形状为 (1,) 的 object 类型数组 'b'
        my_dtype = np.dtype([('b', (object, (1,)))])
        
        # 创建一个形状为 (1,) 的 masked array，使用上面定义的数据类型
        masked_arr = np.ma.masked_all((1,), my_dtype)

        # 断言 'masked_arr['b']' 的类型为 MaskedArray 对象
        assert_equal(type(masked_arr['b']), np.ma.core.MaskedArray)
        
        # 断言 'masked_arr['b']' 的长度为 1
        assert_equal(len(masked_arr['b']), 1)
        
        # 断言 'masked_arr['b']' 的形状为 (1, 1)
        assert_equal(masked_arr['b'].shape, (1, 1))
        
        # 断言 'masked_arr['b']._fill_value' 的形状为空元组 ()
        assert_equal(masked_arr['b']._fill_value.shape, ())

    def test_masked_all_like(self):
        # Tests masked_all
        # Standard dtype
        
        # 创建一个标准的 float 类型数组 base
        base = array([1, 2], dtype=float)
        
        # 使用 masked_all_like 创建一个与 base 相同形状的 masked array test
        test = masked_all_like(base)
        
        # 创建一个标准的 float 类型数组 control，其中元素为 1，且所有元素都被 mask 掉
        control = array([1, 1], mask=[1, 1], dtype=float)
        
        # 断言 test 和 control 相等
        assert_equal(test, control)
        
        # Flexible dtype
        
        # 定义一个灵活的数据类型 dt，包含两个字段 'a' 和 'b'，类型都是 float
        dt = np.dtype({'names': ['a', 'b'], 'formats': ['f', 'f']})
        
        # 创建一个与 control 形状相同的 masked array，使用上面定义的数据类型 dt
        test = masked_all_like(control, dtype=dt)
        
        # 断言 test 和 control 相等
        assert_equal(test, control)
        
        # Nested dtype
        
        # 定义一个嵌套的数据类型 dt，包含一个字段 'a' 类型为 float，和一个嵌套的数组 'b'，包含两个字段 'ba' 和 'bb'，类型都是 float
        dt = np.dtype([('a', 'f'), ('b', [('ba', 'f'), ('bb', 'f')])])
        
        # 创建一个与 control 形状相同的 masked array，使用上面定义的数据类型 dt
        test = masked_all_like(control, dtype=dt)
        
        # 断言 test 和 control 相等
        assert_equal(test, control)

    def check_clump(self, f):
        # 检查 clump 函数的功能
        
        # 对于范围从 1 到 6 的每一个整数 i
        for i in range(1, 7):
            # 对于范围从 0 到 2^i - 1 的每一个整数 j
            for j in range(2**i):
                # 创建一个长度为 i 的整数数组 k
                k = np.arange(i, dtype=int)
                
                # 创建一个长度为 i，填充为 j 的整数数组 ja
                ja = np.full(i, j, dtype=int)
                
                # 创建一个长度为 2^i 的 masked array a，其中元素为 2 的 k 次方
                a = masked_array(2**k)
                
                # 设置 a 的 mask，使得只保留 ja & (2**k) != 0 的元素
                a.mask = (ja & (2**k)) != 0
                
                # 初始化变量 s 为 0
                s = 0
                
                # 对于 f(a) 返回的每一个 slice sl
                for sl in f(a):
                    # 将 a 中 sl 对应的数据元素求和并加到 s 上
                    s += a.data[sl].sum()
                
                # 如果 f 是 clump_unmasked 函数，则断言压缩后的 a 的总和等于 s
                if f == clump_unmasked:
                    assert_equal(a.compressed().sum(), s)
                else:
                    # 否则，反转 a 的 mask，并断言压缩后的 a 的总和等于 s
                    a.mask = ~a.mask
                    assert_equal(a.compressed().sum(), s)

    def test_clump_masked(self):
        # Test clump_masked
        
        # 创建一个包含 0 到 9 的整数的 masked array a
        a = masked_array(np.arange(10))
        
        # 将索引为 [0, 1, 2, 6, 8, 9] 的元素设为 masked
        a[[0, 1, 2, 6, 8, 9]] = masked
        
        # 调用 clump_masked 函数，返回的结果 test 应与 control 相等
        test = clump_masked(a)
        control = [slice(0, 3), slice(6, 7), slice(8, 10)]
        assert_equal(test, control)
        
        # 调用 self.check_clump 函数，检查 clump_masked 函数的功能
        self.check_clump(clump_masked)
    # 定义一个测试方法 test_clump_unmasked，用于测试 clump_unmasked 函数
    def test_clump_unmasked(self):
        # 创建一个包含遮盖数组的 masked_array，数组范围为0到9
        a = masked_array(np.arange(10))
        # 将指定索引处的元素标记为 masked（遮盖）
        a[[0, 1, 2, 6, 8, 9]] = masked
        # 调用 clump_unmasked 函数，获取结果
        test = clump_unmasked(a)
        # 期望的结果
        control = [slice(3, 6), slice(7, 8), ]
        # 断言测试结果与期望结果相等
        assert_equal(test, control)

        # 调用 self.check_clump 方法，验证 clump_unmasked 函数的其他方面

    # 定义一个测试方法 test_flatnotmasked_contiguous，用于测试 flatnotmasked_contiguous 函数
    def test_flatnotmasked_contiguous(self):
        # 创建一个包含整数0到9的数组 a
        a = arange(10)
        # 对没有遮盖的情况进行测试，预期得到包含一个完整范围的切片
        test = flatnotmasked_contiguous(a)
        assert_equal(test, [slice(0, a.size)])

        # 将数组 a 的遮盖设置为全为 False 的情况进行测试
        a.mask = np.zeros(10, dtype=bool)
        assert_equal(test, [slice(0, a.size)])

        # 对包含部分遮盖的情况进行测试
        a[(a < 3) | (a > 8) | (a == 5)] = masked
        test = flatnotmasked_contiguous(a)
        assert_equal(test, [slice(3, 5), slice(6, 9)])

        # 将数组 a 的所有元素标记为遮盖，并进行测试
        a[:] = masked
        test = flatnotmasked_contiguous(a)
        # 预期返回空列表，因为没有非遮盖的连续区域
        assert_equal(test, [])
class TestAverage:
    # Several tests of average. Why so many ? Good point...
    # 定义一个测试类 TestAverage，用于测试 average 函数，包含多个测试用例

    def test_testAverage1(self):
        # Test of average.
        # 测试 average 函数的基本用例
        ott = array([0., 1., 2., 3.], mask=[True, False, False, False])
        # 创建一个带遮罩的数组 ott，其中第一个元素被遮罩
        assert_equal(2.0, average(ott, axis=0))
        # 断言计算 ott 数组沿着 axis=0 的平均值为 2.0
        assert_equal(2.0, average(ott, weights=[1., 1., 2., 1.]))
        # 断言计算 ott 数组使用权重 [1., 1., 2., 1.] 的加权平均值为 2.0
        result, wts = average(ott, weights=[1., 1., 2., 1.], returned=True)
        # 计算 ott 数组使用权重 [1., 1., 2., 1.] 的加权平均值，并返回结果和权重
        assert_equal(2.0, result)
        # 断言计算结果为 2.0
        assert_(wts == 4.0)
        # 断言权重为 4.0
        ott[:] = masked
        # 将 ott 数组全部遮罩
        assert_equal(average(ott, axis=0).mask, [True])
        # 断言计算遮罩后 ott 数组沿着 axis=0 的平均值的遮罩为 [True]
        ott = array([0., 1., 2., 3.], mask=[True, False, False, False])
        # 重新初始化 ott 数组
        ott = ott.reshape(2, 2)
        # 将 ott 数组重塑为 2x2 的形状
        ott[:, 1] = masked
        # 将 ott 数组第二列遮罩
        assert_equal(average(ott, axis=0), [2.0, 0.0])
        # 断言计算 ott 数组沿着 axis=0 的平均值为 [2.0, 0.0]
        assert_equal(average(ott, axis=1).mask[0], [True])
        # 断言计算 ott 数组沿着 axis=1 的平均值的遮罩第一个元素为 [True]
        assert_equal([2., 0.], average(ott, axis=0))
        # 断言计算 ott 数组沿着 axis=0 的平均值为 [2., 0.]
        result, wts = average(ott, axis=0, returned=True)
        # 计算 ott 数组沿着 axis=0 的平均值，并返回结果和权重
        assert_equal(wts, [1., 0.])
        # 断言权重为 [1., 0.]

    def test_testAverage2(self):
        # More tests of average.
        # 更多 average 函数的测试用例
        w1 = [0, 1, 1, 1, 1, 0]
        # 定义权重数组 w1
        w2 = [[0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1]]
        # 定义权重数组 w2
        x = arange(6, dtype=np.float64)
        # 创建一个浮点型数组 x，包含 0 到 5 的值
        assert_equal(average(x, axis=0), 2.5)
        # 断言计算 x 数组沿着 axis=0 的平均值为 2.5
        assert_equal(average(x, axis=0, weights=w1), 2.5)
        # 断言计算 x 数组使用权重 w1 的加权平均值为 2.5
        y = array([arange(6, dtype=np.float64), 2.0 * arange(6)])
        # 创建一个包含两行的浮点型数组 y，第一行是 0 到 5 的值，第二行是第一行值的两倍
        assert_equal(average(y, None), np.add.reduce(np.arange(6)) * 3. / 12.)
        # 断言计算 y 数组的平均值，未指定 axis，计算全局平均值
        assert_equal(average(y, axis=0), np.arange(6) * 3. / 2.)
        # 断言计算 y 数组沿着 axis=0 的平均值
        assert_equal(average(y, axis=1),
                     [average(x, axis=0), average(x, axis=0) * 2.0])
        # 断言计算 y 数组沿着 axis=1 的平均值
        assert_equal(average(y, None, weights=w2), 20. / 6.)
        # 断言计算 y 数组的平均值，未指定 axis，使用权重 w2
        assert_equal(average(y, axis=0, weights=w2),
                     [0., 1., 2., 3., 4., 10.])
        # 断言计算 y 数组沿着 axis=0 的平均值，使用权重 w2
        assert_equal(average(y, axis=1),
                     [average(x, axis=0), average(x, axis=0) * 2.0])
        # 断言计算 y 数组沿着 axis=1 的平均值
        m1 = zeros(6)
        # 创建一个长度为 6 的零数组 m1
        m2 = [0, 0, 1, 1, 0, 0]
        # 定义一个长度为 6 的数组 m2，部分元素被置为 1
        m3 = [[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0]]
        # 定义一个包含两行的数组 m3，每行有一些元素被置为 1
        m4 = ones(6)
        # 创建一个长度为 6 的元素全为 1 的数组 m4
        m5 = [0, 1, 1, 1, 1, 1]
        # 定义一个长度为 6 的数组 m5，除了第一个元素外，所有元素都被置为 1
        assert_equal(average(masked_array(x, m1), axis=0), 2.5)
        # 断言计算被遮罩的 x 数组沿着 axis=0 的平均值为 2.5
        assert_equal(average(masked_array(x, m2), axis=0), 2.5)
        # 断言计算被遮罩的 x 数组（使用 m2 遮罩）沿着 axis=0 的平均值为 2.5
        assert_equal(average(masked_array(x, m4), axis=0).mask, [True])
        # 断言计算被遮罩的 x 数组（使用 m4 遮罩）沿着 axis=0 的平均值的遮罩为 [True]
        assert_equal(average(masked_array(x, m5), axis=0), 0.0)
        # 断言计算被遮罩的 x 数组（使用 m5 遮罩）沿着 axis=0 的平均值为 0.0
        assert_equal(count(average(masked_array(x, m4), axis=0)), 0)
        # 断言计算被遮罩的 x 数组（使用 m4 遮罩）沿着 axis=0 的平均值的元素数量为 0
        z = masked_array(y, m3)
        # 创建一个使用 m3 遮罩的 y 数组 z
        assert_equal(average(z, None), 20. / 6.)
        # 断言计算被遮罩的 y 数组 z 的平均值，未指定 axis
        assert_equal(average(z, axis=0), [0., 1., 99., 99., 4.0, 7.5])
        # 断言计算被遮罩的 y 数组 z 沿
    def test_testAverage3(self):
        # 测试 average 函数的更多用例
        # 创建数组 a 和 b，分别为 [0, 1, 2, 3, 4, 5] 和 [0, 3, 6, 9, 12, 15]
        a = arange(6)
        b = arange(6) * 3
        # 计算在 axis=1 上的加权平均，返回结果 r1 和权重 w1
        r1, w1 = average([[a, b], [b, a]], axis=1, returned=True)
        # 断言 r1 和 w1 的形状相等
        assert_equal(shape(r1), shape(w1))
        # 再次断言 r1 和 w1 的形状相等
        assert_equal(r1.shape, w1.shape)
        # 计算在 axis=0 上的加权平均，weights=[3, 1]，返回结果 r2 和权重 w2
        r2, w2 = average(ones((2, 2, 3)), axis=0, weights=[3, 1], returned=True)
        # 断言 w2 和 r2 的形状相等
        assert_equal(shape(w2), shape(r2))
        # 计算在 axis=None 上的平均值，返回结果 r2 和权重 w2
        r2, w2 = average(ones((2, 2, 3)), returned=True)
        # 断言 w2 和 r2 的形状相等
        assert_equal(shape(w2), shape(r2))
        # 计算在 axis=None 上的加权平均，weights 全为 1，返回结果 r2 和权重 w2
        r2, w2 = average(ones((2, 2, 3)), weights=ones((2, 2, 3)), returned=True)
        # 断言 w2 和 r2 的形状相等
        assert_equal(shape(w2), shape(r2))
        # 创建二维数组 a2d，[[1.0, 2.0], [0.0, 4.0]]
        a2d = array([[1, 2], [0, 4]], float)
        # 创建掩码数组 a2dm，[[False, False], [True, False]]
        a2dm = masked_array(a2d, [[False, False], [True, False]])
        # 计算在 axis=0 上的平均值，返回结果 a2da，应为 [0.5, 3.0]
        a2da = average(a2d, axis=0)
        # 断言 a2da 的结果符合预期 [0.5, 3.0]
        assert_equal(a2da, [0.5, 3.0])
        # 计算在 axis=0 上的掩码数组平均值，返回结果 a2dma，应为 [1.0, 3.0]
        a2dma = average(a2dm, axis=0)
        # 断言 a2dma 的结果符合预期 [1.0, 3.0]
        assert_equal(a2dma, [1.0, 3.0])
        # 计算在 axis=None 上的掩码数组平均值，返回结果 a2dma，应为 7. / 3.
        a2dma = average(a2dm, axis=None)
        # 断言 a2dma 的结果符合预期 7. / 3.
        assert_equal(a2dma, 7. / 3.)
        # 计算在 axis=1 上的掩码数组平均值，返回结果 a2dma，应为 [1.5, 4.0]
        a2dma = average(a2dm, axis=1)
        # 断言 a2dma 的结果符合预期 [1.5, 4.0]

    def test_testAverage4(self):
        # 测试 average 函数中 keepdims 参数的工作情况
        # 创建数组 x，形状为 (3, 1)，[[2], [3], [4]]
        x = np.array([2, 3, 4]).reshape(3, 1)
        # 创建掩码数组 b，形状同 x，[[2], [3], [--]]
        b = np.ma.array(x, mask=[[False], [False], [True]])
        # 创建权重数组 w，形状为 (3, 1)，[[4], [5], [6]]
        w = np.array([4, 5, 6]).reshape(3, 1)
        # 计算在 axis=1 上的加权平均，返回结果 actual，应为 [[2.], [3.], [4.]]
        actual = average(b, weights=w, axis=1, keepdims=True)
        # 创建掩码数组 desired，形状同 actual，[[2.], [3.], [--]]
        desired = masked_array([[2.], [3.], [4.]], [[False], [False], [True]])
        # 断言 actual 和 desired 的结果相等
        assert_equal(actual, desired)
    # 定义一个测试方法，用于测试权重和输入维度不同的情况
    def test_weight_and_input_dims_different(self):
        # 模仿 np.average() 函数在 lib/test/test_function_base.py 中的测试
        # 创建一个形状为 (2, 2, 3) 的三维数组 y，包含从 0 到 11 的连续整数
        y = np.arange(12).reshape(2, 2, 3)
        # 创建一个形状为 (2, 2, 3) 的三维数组 w，包含特定的权重值
        w = np.array([0., 0., 1., .5, .5, 0., 0., .5, .5, 1., 0., 0.])\
            .reshape(2, 2, 3)

        # 创建一个形状为 (2, 2, 3) 的布尔数组 m，全为 False，用于创建掩码
        m = np.full((2, 2, 3), False)
        # 将 y 封装成一个带掩码的 masked array 对象 yma
        yma = np.ma.array(y, mask=m)
        # 获取 w 的第一个轴上的切片，形状为 (2, 2)
        subw0 = w[:, :, 0]

        # 计算使用 subw0 权重的 yma 的平均值，沿着第 0 和第 1 轴
        actual = average(yma, axis=(0, 1), weights=subw0)
        # 期望的结果，带有掩码
        desired = masked_array([7., 8., 9.], mask=[False, False, False])
        # 断言计算结果与期望结果的近似性
        assert_almost_equal(actual, desired)

        # 更新 m 的掩码设置
        m = np.full((2, 2, 3), False)
        m[:, :, 0] = True
        m[0, 0, 1] = True
        yma = np.ma.array(y, mask=m)
        # 重新计算带有更新掩码的 yma 的平均值
        actual = average(yma, axis=(0, 1), weights=subw0)
        # 更新的期望结果，带有更新的掩码
        desired = masked_array(
            [np.nan, 8., 9.],
            mask=[True, False, False])
        # 断言计算结果与期望结果的近似性
        assert_almost_equal(actual, desired)

        # 重置 m 的掩码设置
        m = np.full((2, 2, 3), False)
        yma = np.ma.array(y, mask=m)

        # 获取 w 的第二个轴上的切片，形状为 (2, 3)
        subw1 = w[1, :, :]
        # 计算使用 subw1 权重的 yma 的平均值，沿着第 1 和第 2 轴
        actual = average(yma, axis=(1, 2), weights=subw1)
        # 期望的结果，带有掩码
        desired = masked_array([2.25, 8.25], mask=[False, False])
        # 断言计算结果与期望结果的近似性
        assert_almost_equal(actual, desired)

        # 当权重的形状与指定的轴不匹配时，抛出 ValueError 异常
        with pytest.raises(
                ValueError,
                match="Shape of weights must be consistent with "
                      "shape of a along specified axis"):
            average(yma, axis=(0, 1, 2), weights=subw0)

        with pytest.raises(
                ValueError,
                match="Shape of weights must be consistent with "
                      "shape of a along specified axis"):
            average(yma, axis=(0, 1), weights=subw1)

        # 当交换轴时，使用 (1, 0) 轴的平均值应该等同于使用转置后的权重 subw0.T
        actual = average(yma, axis=(1, 0), weights=subw0)
        desired = average(yma, axis=(0, 1), weights=subw0.T)
        # 断言计算结果与期望结果的近似性
        assert_almost_equal(actual, desired)

    def test_onintegers_with_mask(self):
        # 测试带有掩码的整数数组的平均值计算
        a = average(array([1, 2]))
        # 断言计算结果与期望结果相等
        assert_equal(a, 1.5)
        a = average(array([1, 2, 3, 4], mask=[False, False, True, True]))
        # 断言计算结果与期望结果相等
        assert_equal(a, 1.5)
    def test_complex(self):
        # 测试复杂数据情况。
        # （针对 https://github.com/numpy/numpy/issues/2684 的回归测试）

        # 创建一个布尔掩码数组
        mask = np.array([[0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0]], dtype=bool)
        
        # 创建一个带掩码的复数数组
        a = masked_array([[0, 1+2j, 3+4j, 5+6j, 7+8j],
                          [9j, 0+1j, 2+3j, 4+5j, 7+7j]],
                         mask=mask)

        # 计算数组 a 的平均值
        av = average(a)
        # 计算压缩后数组 a 的平均值，并进行近似相等性断言
        expected = np.average(a.compressed())
        assert_almost_equal(av.real, expected.real)
        assert_almost_equal(av.imag, expected.imag)

        # 沿着 axis=0 计算数组 a 的平均值
        av0 = average(a, axis=0)
        # 计算实部和虚部的加权平均值，并进行近似相等性断言
        expected0 = average(a.real, axis=0) + average(a.imag, axis=0)*1j
        assert_almost_equal(av0.real, expected0.real)
        assert_almost_equal(av0.imag, expected0.imag)

        # 沿着 axis=1 计算数组 a 的平均值
        av1 = average(a, axis=1)
        # 计算实部和虚部的加权平均值，并进行近似相等性断言
        expected1 = average(a.real, axis=1) + average(a.imag, axis=1)*1j
        assert_almost_equal(av1.real, expected1.real)
        assert_almost_equal(av1.imag, expected1.imag)

        # 测试带有 'weights' 参数的情况
        wts = np.array([[0.5, 1.0, 2.0, 1.0, 0.5],
                        [1.0, 1.0, 1.0, 1.0, 1.0]])
        
        # 使用权重计算数组 a 的加权平均值
        wav = average(a, weights=wts)
        # 计算压缩后数组 a 的加权平均值，并进行近似相等性断言
        expected = np.average(a.compressed(), weights=wts[~mask])
        assert_almost_equal(wav.real, expected.real)
        assert_almost_equal(wav.imag, expected.imag)

        # 沿着 axis=0 使用权重计算数组 a 的加权平均值
        wav0 = average(a, weights=wts, axis=0)
        # 计算实部和虚部的加权平均值，并进行近似相等性断言
        expected0 = (average(a.real, weights=wts, axis=0) +
                     average(a.imag, weights=wts, axis=0)*1j)
        assert_almost_equal(wav0.real, expected0.real)
        assert_almost_equal(wav0.imag, expected0.imag)

        # 沿着 axis=1 使用权重计算数组 a 的加权平均值
        wav1 = average(a, weights=wts, axis=1)
        # 计算实部和虚部的加权平均值，并进行近似相等性断言
        expected1 = (average(a.real, weights=wts, axis=1) +
                     average(a.imag, weights=wts, axis=1)*1j)
        assert_almost_equal(wav1.real, expected1.real)
        assert_almost_equal(wav1.imag, expected1.imag)

    @pytest.mark.parametrize(
        'x, axis, expected_avg, weights, expected_wavg, expected_wsum',
        [([1, 2, 3], None, [2.0], [3, 4, 1], [1.75], [8.0]),
         ([[1, 2, 5], [1, 6, 11]], 0, [[1.0, 4.0, 8.0]],
          [1, 3], [[1.0, 5.0, 9.5]], [[4, 4, 4]])],
    )
    def test_basic_keepdims(self, x, axis, expected_avg,
                            weights, expected_wavg, expected_wsum):
        # 测试带有 keepdims=True 的基本情况

        # 计算数组 x 的平均值，并进行形状相等性断言
        avg = np.ma.average(x, axis=axis, keepdims=True)
        assert avg.shape == np.shape(expected_avg)
        assert_array_equal(avg, expected_avg)

        # 使用权重计算数组 x 的加权平均值，并进行形状相等性断言
        wavg = np.ma.average(x, axis=axis, weights=weights, keepdims=True)
        assert wavg.shape == np.shape(expected_wavg)
        assert_array_equal(wavg, expected_wavg)

        # 使用权重计算数组 x 的加权平均值和加权和，并进行形状相等性断言
        wavg, wsum = np.ma.average(x, axis=axis, weights=weights,
                                   returned=True, keepdims=True)
        assert wavg.shape == np.shape(expected_wavg)
        assert_array_equal(wavg, expected_wavg)
        assert wsum.shape == np.shape(expected_wsum)
        assert_array_equal(wsum, expected_wsum)
    # 定义一个测试方法，用于测试带有掩码权重的情况
    def test_masked_weights(self):
        # 使用掩码数组创建一个掩码数组对象 a，形状为 3x3
        a = np.ma.array(np.arange(9).reshape(3, 3),
                        mask=[[1, 0, 0], [1, 0, 0], [0, 0, 0]])
        
        # 创建未掩码的权重数组对象 weights_unmasked
        weights_unmasked = masked_array([5, 28, 31], mask=False)
        
        # 创建带有掩码的权重数组对象 weights_masked
        weights_masked = masked_array([5, 28, 31], mask=[1, 0, 0])

        # 计算未掩码情况下的平均值，沿着 axis=0 方向，使用权重 weights_unmasked，不返回权重
        avg_unmasked = average(a, axis=0,
                               weights=weights_unmasked, returned=False)
        # 预期的未掩码情况下的平均值
        expected_unmasked = np.array([6.0, 5.21875, 6.21875])
        # 断言计算结果与预期结果的接近程度
        assert_almost_equal(avg_unmasked, expected_unmasked)

        # 计算带有掩码情况下的平均值，沿着 axis=0 方向，使用权重 weights_masked，不返回权重
        avg_masked = average(a, axis=0, weights=weights_masked, returned=False)
        # 预期的带有掩码情况下的平均值
        expected_masked = np.array([6.0, 5.576271186440678, 6.576271186440678])
        # 断言计算结果与预期结果的接近程度
        assert_almost_equal(avg_masked, expected_masked)

        # weights 如果需要的话应该被掩码，这取决于数组的掩码情况。
        # 这是为了避免对被掩码的 NaN 或其他值求和，这些值不会被零取消
        a = np.ma.array([1.0,   2.0,   3.0,  4.0],
                   mask=[False, False, True, True])
        
        # 计算未掩码情况下的平均值，使用权重 [1, 1, 1, np.nan]
        avg_unmasked = average(a, weights=[1, 1, 1, np.nan])
        # 断言计算结果与预期的 1.5 的接近程度
        assert_almost_equal(avg_unmasked, 1.5)

        # 创建一个 3x4 的掩码数组对象 a，其中包含了掩码
        a = np.ma.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 1.0, 2.0, 3.0],
        ], mask=[
            [False, True, True, False],
            [True, False, True, True],
            [True, False, True, False],
        ])
        
        # 沿着 axis=0 方向计算带有掩码的平均值，使用权重 [1, np.nan, 1]
        avg_masked = np.ma.average(a, weights=[1, np.nan, 1], axis=0)
        # 预期的带有掩码的平均值数组
        avg_expected = np.ma.array([1.0, np.nan, np.nan, 3.5],
                              mask=[False, True, True, False])
        
        # 断言计算结果与预期结果的接近程度
        assert_almost_equal(avg_masked, avg_expected)
        # 断言掩码数组的掩码情况是否与预期一致
        assert_equal(avg_masked.mask, avg_expected.mask)
class TestConcatenator:
    # Tests for mr_, the equivalent of r_ for masked arrays.

    def test_1d(self):
        # Tests mr_ on 1D arrays.
        assert_array_equal(mr_[1, 2, 3, 4, 5, 6], array([1, 2, 3, 4, 5, 6]))
        b = ones(5)
        m = [1, 0, 0, 0, 0]
        d = masked_array(b, mask=m)
        # Concatenate masked arrays and plain values using mr_
        c = mr_[d, 0, 0, d]
        assert_(isinstance(c, MaskedArray))
        assert_array_equal(c, [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1])
        assert_array_equal(c.mask, mr_[m, 0, 0, m])

    def test_2d(self):
        # Tests mr_ on 2D arrays.
        a_1 = np.random.rand(5, 5)
        a_2 = np.random.rand(5, 5)
        m_1 = np.round(np.random.rand(5, 5), 0)
        m_2 = np.round(np.random.rand(5, 5), 0)
        b_1 = masked_array(a_1, mask=m_1)
        b_2 = masked_array(a_2, mask=m_2)
        # Append columns using mr_
        d = mr_['1', b_1, b_2]
        assert_(d.shape == (5, 10))
        assert_array_equal(d[:, :5], b_1)
        assert_array_equal(d[:, 5:], b_2)
        assert_array_equal(d.mask, np.r_['1', m_1, m_2])
        d = mr_[b_1, b_2]
        assert_(d.shape == (10, 5))
        assert_array_equal(d[:5,:], b_1)
        assert_array_equal(d[5:,:], b_2)
        assert_array_equal(d.mask, np.r_[m_1, m_2])

    def test_masked_constant(self):
        # Tests mr_ with masked constants.
        actual = mr_[np.ma.masked, 1]
        assert_equal(actual.mask, [True, False])
        assert_equal(actual.data[1], 1)

        actual = mr_[[1, 2], np.ma.masked]
        assert_equal(actual.mask, [False, False, True])
        assert_equal(actual.data[:2], [1, 2])


class TestNotMasked:
    # Tests notmasked_edges and notmasked_contiguous.
    def test_edges(self):
        # Tests unmasked_edges
        # 创建一个 5x5 的 NumPy 掩码数组
        data = masked_array(np.arange(25).reshape(5, 5),
                            mask=[[0, 0, 1, 0, 0],
                                  [0, 0, 0, 1, 1],
                                  [1, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0],
                                  [1, 1, 1, 0, 0]],)
        # 调用 notmasked_edges 函数，测试不同的参数
        test = notmasked_edges(data, None)
        # 断言测试结果与预期结果一致
        assert_equal(test, [0, 24])
        # 再次调用 notmasked_edges 函数，使用参数 0
        test = notmasked_edges(data, 0)
        # 断言第一个返回值与预期的列表一致
        assert_equal(test[0], [(0, 0, 1, 0, 0), (0, 1, 2, 3, 4)])
        # 断言第二个返回值与预期的列表一致
        assert_equal(test[1], [(3, 3, 3, 4, 4), (0, 1, 2, 3, 4)])
        # 再次调用 notmasked_edges 函数，使用参数 1
        test = notmasked_edges(data, 1)
        # 断言第一个返回值与预期的列表一致
        assert_equal(test[0], [(0, 1, 2, 3, 4), (0, 0, 2, 0, 3)])
        # 断言第二个返回值与预期的列表一致
        assert_equal(test[1], [(0, 1, 2, 3, 4), (4, 2, 4, 4, 4)])
        #
        # 将 data 的数据部分传递给 notmasked_edges 函数，测试不同的参数
        test = notmasked_edges(data.data, None)
        # 断言测试结果与预期结果一致
        assert_equal(test, [0, 24])
        # 再次调用 notmasked_edges 函数，使用参数 0
        test = notmasked_edges(data.data, 0)
        # 断言第一个返回值与预期的列表一致
        assert_equal(test[0], [(0, 0, 0, 0, 0), (0, 1, 2, 3, 4)])
        # 断言第二个返回值与预期的列表一致
        assert_equal(test[1], [(4, 4, 4, 4, 4), (0, 1, 2, 3, 4)])
        # 再次调用 notmasked_edges 函数，使用参数 -1
        test = notmasked_edges(data.data, -1)
        # 断言第一个返回值与预期的列表一致
        assert_equal(test[0], [(0, 1, 2, 3, 4), (0, 0, 0, 0, 0)])
        # 断言第二个返回值与预期的列表一致
        assert_equal(test[1], [(0, 1, 2, 3, 4), (4, 4, 4, 4, 4)])
        #
        # 修改 data 中的倒数第二行为 masked
        data[-2] = masked
        # 再次调用 notmasked_edges 函数，使用参数 0
        test = notmasked_edges(data, 0)
        # 断言第一个返回值与预期的列表一致
        assert_equal(test[0], [(0, 0, 1, 0, 0), (0, 1, 2, 3, 4)])
        # 断言第二个返回值与预期的列表一致
        assert_equal(test[1], [(1, 1, 2, 4, 4), (0, 1, 2, 3, 4)])
        # 再次调用 notmasked_edges 函数，使用参数 -1
        test = notmasked_edges(data, -1)
        # 断言第一个返回值与预期的列表一致
        assert_equal(test[0], [(0, 1, 2, 4), (0, 0, 2, 3)])
        # 断言第二个返回值与预期的列表一致
        assert_equal(test[1], [(0, 1, 2, 4), (4, 2, 4, 4)])

    def test_contiguous(self):
        # Tests notmasked_contiguous
        # 创建一个 3x8 的 NumPy 掩码数组
        a = masked_array(np.arange(24).reshape(3, 8),
                         mask=[[0, 0, 0, 0, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [0, 0, 0, 0, 0, 0, 1, 0]])
        # 调用 notmasked_contiguous 函数，测试不同的参数
        tmp = notmasked_contiguous(a, None)
        # 断言测试结果与预期结果一致
        assert_equal(tmp, [
            slice(0, 4, None),
            slice(16, 22, None),
            slice(23, 24, None)
        ])

        # 再次调用 notmasked_contiguous 函数，使用参数 0
        tmp = notmasked_contiguous(a, 0)
        # 断言测试结果与预期结果一致
        assert_equal(tmp, [
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(2, 3, None)],
            [slice(2, 3, None)],
            [],
            [slice(2, 3, None)]
        ])
        #
        # 再次调用 notmasked_contiguous 函数，使用参数 1
        tmp = notmasked_contiguous(a, 1)
        # 断言测试结果与预期结果一致
        assert_equal(tmp, [
            [slice(0, 4, None)],
            [],
            [slice(0, 6, None), slice(7, 8, None)]
        ])
# 定义一个测试类 TestCompressFunctions，用于测试压缩函数的功能
class TestCompressFunctions:

    # 定义测试方法 test_compress_rowcols，测试 compress_rowcols 函数
    def test_compress_rowcols(self):
        # 创建一个 3x3 的数组 x，内容为 0 到 8，其中部分元素被遮盖（masked）
        x = array(np.arange(9).reshape(3, 3),
                  mask=[[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        # 断言调用 compress_rowcols(x) 后的结果与预期结果相等，验证行列压缩的功能
        assert_equal(compress_rowcols(x), [[4, 5], [7, 8]])
        # 断言调用 compress_rowcols(x, 0) 后的结果与预期结果相等，验证按行压缩的功能
        assert_equal(compress_rowcols(x, 0), [[3, 4, 5], [6, 7, 8]])
        # 断言调用 compress_rowcols(x, 1) 后的结果与预期结果相等，验证按列压缩的功能
        assert_equal(compress_rowcols(x, 1), [[1, 2], [4, 5], [7, 8]])
        
        # 修改 x 的数据内容和遮盖情况，进行下一组测试
        x = array(x._data, mask=[[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_equal(compress_rowcols(x), [[0, 2], [6, 8]])
        assert_equal(compress_rowcols(x, 0), [[0, 1, 2], [6, 7, 8]])
        assert_equal(compress_rowcols(x, 1), [[0, 2], [3, 5], [6, 8]])
        
        # 再次修改 x 的数据内容和遮盖情况，进行下一组测试
        x = array(x._data, mask=[[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_equal(compress_rowcols(x), [[8]])
        assert_equal(compress_rowcols(x, 0), [[6, 7, 8]])
        assert_equal(compress_rowcols(x, 1,), [[2], [5], [8]])
        
        # 最后修改 x 的数据内容和遮盖情况，进行最后一组测试
        x = array(x._data, mask=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # 断言调用 compress_rowcols(x).size 后的结果为 0，验证空结果情况
        assert_equal(compress_rowcols(x).size, 0)
        assert_equal(compress_rowcols(x, 0).size, 0)
        assert_equal(compress_rowcols(x, 1).size, 0)

    # 定义测试方法 test_mask_rowcols，测试 mask_rowcols 函数
    def test_mask_rowcols(self):
        # 创建一个 3x3 的数组 x，内容为 0 到 8，其中部分元素被遮盖（masked）
        x = array(np.arange(9).reshape(3, 3),
                  mask=[[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        # 断言调用 mask_rowcols(x) 后的 mask 结果与预期结果相等，验证行列遮盖的功能
        assert_equal(mask_rowcols(x).mask,
                     [[1, 1, 1], [1, 0, 0], [1, 0, 0]])
        # 断言调用 mask_rowcols(x, 0) 后的 mask 结果与预期结果相等，验证按行遮盖的功能
        assert_equal(mask_rowcols(x, 0).mask,
                     [[1, 1, 1], [0, 0, 0], [0, 0, 0]])
        # 断言调用 mask_rowcols(x, 1) 后的 mask 结果与预期结果相等，验证按列遮盖的功能
        assert_equal(mask_rowcols(x, 1).mask,
                     [[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        
        # 修改 x 的数据内容和遮盖情况，进行下一组测试
        x = array(x._data, mask=[[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_equal(mask_rowcols(x).mask,
                     [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        assert_equal(mask_rowcols(x, 0).mask,
                     [[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        assert_equal(mask_rowcols(x, 1).mask,
                     [[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        
        # 再次修改 x 的数据内容和遮盖情况，进行下一组测试
        x = array(x._data, mask=[[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_equal(mask_rowcols(x).mask,
                     [[1, 1, 1], [1, 1, 1], [1, 1, 0]])
        assert_equal(mask_rowcols(x, 0).mask,
                     [[1, 1, 1], [1, 1, 1], [0, 0, 0]])
        assert_equal(mask_rowcols(x, 1,).mask,
                     [[1, 1, 0], [1, 1, 0], [1, 1, 0]])
        
        # 最后修改 x 的数据内容和遮盖情况，进行最后一组测试
        x = array(x._data, mask=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # 断言调用 mask_rowcols(x).all() 后的结果为 masked，验证所有元素遮盖的情况
        assert_(mask_rowcols(x).all() is masked)
        assert_(mask_rowcols(x, 0).all() is masked)
        assert_(mask_rowcols(x, 1).all() is masked)
        # 断言调用 mask_rowcols(x).mask.all() 后的结果为 True，验证所有元素都被遮盖
        assert_(mask_rowcols(x).mask.all())
        assert_(mask_rowcols(x, 0).mask.all())
        assert_(mask_rowcols(x, 1).mask.all())

    # 使用 pytest 的参数化标记，对 func 和 rowcols_axis 参数进行参数化
    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize(["func", "rowcols_axis"],
                             [(np.ma.mask_rows, 0), (np.ma.mask_cols, 1)])
    # 测试关于 `mask_rows` 和 `mask_cols` 函数中 axis 参数的弃用警告
    def test_mask_row_cols_axis_deprecation(self, axis, func, rowcols_axis):
        # 创建一个 3x3 的数组 x，并指定部分元素被遮盖（masked）
        x = array(np.arange(9).reshape(3, 3),
                  mask=[[1, 0, 0], [0, 0, 0], [0, 0, 0]])

        # 使用 assert_warns 确保在调用 func 函数时会出现 DeprecationWarning 警告
        with assert_warns(DeprecationWarning):
            # 调用 func 函数，传入指定的 axis 参数
            res = func(x, axis=axis)
            # 断言结果与 mask_rowcols 函数的返回值相等
            assert_equal(res, mask_rowcols(x, rowcols_axis))

    # 测试 dot 函数返回的结果是否为 MaskedArray 类型，参见 GitHub issue gh-6611
    def test_dot_returns_maskedarray(self):
        # 创建一个单位矩阵 a，并将其转换为 MaskedArray 类型的数组 b
        a = np.eye(3)
        b = array(a)
        # 断言 dot(a, a) 返回的类型是 MaskedArray
        assert_(type(dot(a, a)) is MaskedArray)
        # 断言 dot(a, b) 返回的类型是 MaskedArray
        assert_(type(dot(a, b)) is MaskedArray)
        # 断言 dot(b, a) 返回的类型是 MaskedArray
        assert_(type(dot(b, a)) is MaskedArray)
        # 断言 dot(b, b) 返回的类型是 MaskedArray
        assert_(type(dot(b, b)) is MaskedArray)

    # 测试 dot 函数的 out 参数功能
    def test_dot_out(self):
        # 创建一个单位矩阵 a，并将其转换为 MaskedArray 类型的数组
        a = array(np.eye(3))
        # 创建一个全零矩阵 out
        out = array(np.zeros((3, 3)))
        # 调用 dot 函数，将结果存储到预先创建的 out 矩阵中
        res = dot(a, a, out=out)
        # 断言返回的结果 res 是之前创建的 out 矩阵
        assert_(res is out)
        # 断言 a 和 res 的内容相等
        assert_equal(a, res)
class TestApplyAlongAxis:
    # Tests 2D functions
    def test_3d(self):
        # 创建一个 2x2x3 的三维数组
        a = arange(12.).reshape(2, 2, 3)

        # 定义一个操作函数，返回第二维度的数据
        def myfunc(b):
            return b[1]

        # 对数组 `a` 沿着第三个维度应用 myfunc 函数
        xa = apply_along_axis(myfunc, 2, a)
        # 断言结果与预期的相等
        assert_equal(xa, [[1, 4], [7, 10]])

    # Tests kwargs functions
    def test_3d_kwargs(self):
        # 创建一个 2x2x3 的三维数组
        a = arange(12).reshape(2, 2, 3)

        # 定义一个操作函数，带有一个偏移量参数，默认为 0
        def myfunc(b, offset=0):
            return b[1+offset]

        # 对数组 `a` 沿着第三个维度应用 myfunc 函数，并传入偏移量为 1
        xa = apply_along_axis(myfunc, 2, a, offset=1)
        # 断言结果与预期的相等
        assert_equal(xa, [[2, 5], [8, 11]])


class TestApplyOverAxes:
    # Tests apply_over_axes
    def test_basic(self):
        # 创建一个 2x3x4 的三维数组
        a = arange(24).reshape(2, 3, 4)
        
        # 在指定轴上应用 np.sum 函数，轴列表为 [0, 2]
        test = apply_over_axes(np.sum, a, [0, 2])
        # 创建控制数组，用于对比结果
        ctrl = np.array([[[60], [92], [124]]])
        # 断言结果与预期的相等
        assert_equal(test, ctrl)

        # 将数组中的奇数位置的元素设为 masked
        a[(a % 2).astype(bool)] = masked
        # 再次在指定轴上应用 np.sum 函数，轴列表为 [0, 2]
        test = apply_over_axes(np.sum, a, [0, 2])
        # 更新控制数组
        ctrl = np.array([[[28], [44], [60]]])
        # 断言结果与预期的相等
        assert_equal(test, ctrl)


class TestMedian:
    def test_pytype(self):
        # 计算包含无穷大的数组的中位数，沿着最后一个轴（-1）
        r = np.ma.median([[np.inf, np.inf], [np.inf, np.inf]], axis=-1)
        # 断言结果与预期的相等
        assert_equal(r, np.inf)

    def test_inf(self):
        # 测试处理含有 masked 的情况，计算包含无穷大的 masked 数组的中位数
        r = np.ma.median(np.ma.masked_array([[np.inf, np.inf],
                                             [np.inf, np.inf]]), axis=-1)
        # 断言结果与预期的相等
        assert_equal(r, np.inf)
        r = np.ma.median(np.ma.masked_array([[np.inf, np.inf],
                                             [np.inf, np.inf]]), axis=None)
        # 断言结果与预期的相等
        assert_equal(r, np.inf)
        # 所有元素都被 masked
        r = np.ma.median(np.ma.masked_array([[np.inf, np.inf],
                                             [np.inf, np.inf]], mask=True),
                         axis=-1)
        # 断言结果的 mask 全为 True
        assert_equal(r.mask, True)
        r = np.ma.median(np.ma.masked_array([[np.inf, np.inf],
                                             [np.inf, np.inf]], mask=True),
                         axis=None)
        # 断言结果的 mask 全为 True
        assert_equal(r.mask, True)

    def test_non_masked(self):
        # 测试处理未 masked 的情况
        x = np.arange(9)
        # 断言未 masked 情况下的中位数计算正确
        assert_equal(np.ma.median(x), 4.)
        # 断言返回的对象类型不是 MaskedArray
        assert_(type(np.ma.median(x)) is not MaskedArray)
        x = range(8)
        # 断言未 masked 情况下的中位数计算正确
        assert_equal(np.ma.median(x), 3.5)
        # 断言返回的对象类型不是 MaskedArray
        assert_(type(np.ma.median(x)) is not MaskedArray)
        x = 5
        # 断言未 masked 情况下的中位数计算正确
        assert_equal(np.ma.median(x), 5.)
        # 断言返回的对象类型不是 MaskedArray
        assert_(type(np.ma.median(x)) is not MaskedArray)
        # 整数类型的数组
        x = np.arange(9 * 8).reshape(9, 8)
        # 断言未 masked 情况下，沿着每列计算的中位数正确
        assert_equal(np.ma.median(x, axis=0), np.median(x, axis=0))
        # 断言未 masked 情况下，沿着每行计算的中位数正确
        assert_equal(np.ma.median(x, axis=1), np.median(x, axis=1))
        # 断言返回的对象类型不是 MaskedArray
        assert_(np.ma.median(x, axis=1) is not MaskedArray)
        # 浮点数类型的数组
        x = np.arange(9 * 8.).reshape(9, 8)
        # 断言未 masked 情况下，沿着每列计算的中位数正确
        assert_equal(np.ma.median(x, axis=0), np.median(x, axis=0))
        # 断言未 masked 情况下，沿着每行计算的中位数正确
        assert_equal(np.ma.median(x, axis=1), np.median(x, axis=1))
        # 断言返回的对象类型不是 MaskedArray
        assert_(np.ma.median(x, axis=1) is not MaskedArray)
    def test_docstring_examples(self):
        "test the examples given in the docstring of ma.median"
        # 创建一个长度为8的数组，其中前4个未屏蔽，后4个被屏蔽
        x = array(np.arange(8), mask=[0]*4 + [1]*4)
        # 断言计算数组的中位数为1.5
        assert_equal(np.ma.median(x), 1.5)
        # 断言中位数的形状为空元组
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        # 断言中位数不是 MaskedArray 类型
        assert_(type(np.ma.median(x)) is not MaskedArray)
        
        # 创建一个形状为(2, 5)的数组，其中前6个未屏蔽，后4个被屏蔽
        x = array(np.arange(10).reshape(2, 5), mask=[0]*6 + [1]*4)
        # 断言计算数组的中位数为2.5
        assert_equal(np.ma.median(x), 2.5)
        # 断言中位数的形状为空元组
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        # 断言中位数不是 MaskedArray 类型
        assert_(type(np.ma.median(x)) is not MaskedArray)
        
        # 在最后一个轴上计算中位数，覆盖输入数组
        ma_x = np.ma.median(x, axis=-1, overwrite_input=True)
        # 断言计算后的结果为[2., 5.]
        assert_equal(ma_x, [2., 5.])
        # 断言结果的形状为(2,)
        assert_equal(ma_x.shape, (2,), "shape mismatch")
        # 断言结果是 MaskedArray 类型
        assert_(type(ma_x) is MaskedArray)

    def test_axis_argument_errors(self):
        # 错误消息模板
        msg = "mask = %s, ndim = %s, axis = %s, overwrite_input = %s"
        # 遍历不同的数组维度
        for ndmin in range(5):
            # 遍历不同的掩码情况
            for mask in [False, True]:
                # 创建指定维度和掩码的数组
                x = array(1, ndmin=ndmin, mask=mask)

                # 对于每个轴和覆盖输入的组合，验证是否引发异常
                args = itertools.product(range(-ndmin, ndmin), [False, True])
                for axis, over in args:
                    try:
                        np.ma.median(x, axis=axis, overwrite_input=over)
                    except Exception:
                        raise AssertionError(msg % (mask, ndmin, axis, over))

                # 对于无效的轴值，验证是否引发 AxisError 异常
                args = itertools.product([-(ndmin + 1), ndmin], [False, True])
                for axis, over in args:
                    try:
                        np.ma.median(x, axis=axis, overwrite_input=over)
                    except np.exceptions.AxisError:
                        pass
                    else:
                        raise AssertionError(msg % (mask, ndmin, axis, over))

    def test_masked_0d(self):
        # 检查未屏蔽值的中位数
        x = array(1, mask=False)
        assert_equal(np.ma.median(x), 1)
        # 检查屏蔽值的中位数为 Masked
        x = array(1, mask=True)
        assert_equal(np.ma.median(x), np.ma.masked)
    def test_masked_1d(self):
        # 创建一个带掩码的一维数组 x
        x = array(np.arange(5), mask=True)
        # 断言计算 x 的中位数应该是掩码值
        assert_equal(np.ma.median(x), np.ma.masked)
        # 断言计算 x 的中位数的形状应该是标量
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        # 断言 np.ma.median(x) 的类型应该是 MaskedConstant
        assert_(type(np.ma.median(x)) is np.ma.core.MaskedConstant)
        
        # 创建一个没有掩码的一维数组 x
        x = array(np.arange(5), mask=False)
        # 断言计算 x 的中位数应该是 2.0
        assert_equal(np.ma.median(x), 2.)
        # 断言计算 x 的中位数的形状应该是标量
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        # 断言 np.ma.median(x) 的类型不应该是 MaskedArray
        assert_(type(np.ma.median(x)) is not MaskedArray)
        
        # 创建一个带有特定掩码的一维数组 x
        x = array(np.arange(5), mask=[0,1,0,0,0])
        # 断言计算 x 的中位数应该是 2.5
        assert_equal(np.ma.median(x), 2.5)
        # 断言计算 x 的中位数的形状应该是标量
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        # 断言 np.ma.median(x) 的类型不应该是 MaskedArray
        assert_(type(np.ma.median(x)) is not MaskedArray)
        
        # 创建一个完全被掩码的一维数组 x
        x = array(np.arange(5), mask=[0,1,1,1,1])
        # 断言计算 x 的中位数应该是 0.0
        assert_equal(np.ma.median(x), 0.)
        # 断言计算 x 的中位数的形状应该是标量
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        # 断言 np.ma.median(x) 的类型不应该是 MaskedArray
        assert_(type(np.ma.median(x)) is not MaskedArray)
        
        # 创建一个带有整数值的特定掩码的一维数组 x
        x = array(np.arange(5), mask=[0,1,1,0,0])
        # 断言计算 x 的中位数应该是 3.0
        assert_equal(np.ma.median(x), 3.)
        # 断言计算 x 的中位数的形状应该是标量
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        # 断言 np.ma.median(x) 的类型不应该是 MaskedArray
        assert_(type(np.ma.median(x)) is not MaskedArray)
        
        # 创建一个带有浮点数值的特定掩码的一维数组 x
        x = array(np.arange(5.), mask=[0,1,1,0,0])
        # 断言计算 x 的中位数应该是 3.0
        assert_equal(np.ma.median(x), 3.)
        # 断言计算 x 的中位数的形状应该是标量
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        # 断言 np.ma.median(x) 的类型不应该是 MaskedArray
        assert_(type(np.ma.median(x)) is not MaskedArray)
        
        # 创建一个带有整数值的复杂掩码的一维数组 x
        x = array(np.arange(6), mask=[0,1,1,1,1,0])
        # 断言计算 x 的中位数应该是 2.5
        assert_equal(np.ma.median(x), 2.5)
        # 断言计算 x 的中位数的形状应该是标量
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        # 断言 np.ma.median(x) 的类型不应该是 MaskedArray
        assert_(type(np.ma.median(x)) is not MaskedArray)
        
        # 创建一个带有浮点数值的复杂掩码的一维数组 x
        x = array(np.arange(6.), mask=[0,1,1,1,1,0])
        # 断言计算 x 的中位数应该是 2.5
        assert_equal(np.ma.median(x), 2.5)
        # 断言计算 x 的中位数的形状应该是标量
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        # 断言 np.ma.median(x) 的类型不应该是 MaskedArray
        assert_(type(np.ma.median(x)) is not MaskedArray)

    def test_1d_shape_consistency(self):
        # 断言两个具有不同掩码的一维数组的中位数的形状应该相同
        assert_equal(np.ma.median(array([1,2,3],mask=[0,0,0])).shape,
                     np.ma.median(array([1,2,3],mask=[0,1,0])).shape )

    def test_2d(self):
        # 测试带有二维数组的中位数计算
        # 设置数组的行数和列数
        (n, p) = (101, 30)
        # 创建带掩码的一维数组 x
        x = masked_array(np.linspace(-1., 1., n),)
        # 在数组的开头和结尾添加掩码
        x[:10] = x[-10:] = masked
        # 创建一个空的二维数组 z
        z = masked_array(np.empty((n, p), dtype=float))
        # 将 x 复制到 z 的第一列
        z[:, 0] = x[:]
        # 创建一个索引数组
        idx = np.arange(len(x))
        # 随机打乱索引，并将打乱后的 x 复制到 z 的其余列中
        for i in range(1, p):
            np.random.shuffle(idx)
            z[:, i] = x[idx]
        # 断言 z 第一列的中位数应该是 0
        assert_equal(median(z[:, 0]), 0)
        # 断言 z 整体的中位数应该是 0
        assert_equal(median(z), 0)
        # 断言 z 每列的中位数应该是一个全为零的数组
        assert_equal(median(z, axis=0), np.zeros(p))
        # 断言 z 转置后每行的中位数应该是一个全为零的数组
        assert_equal(median(z.T, axis=1), np.zeros(p))
    def test_2d_waxis(self):
        # Tests median with 2D arrays and different axis.
        x = masked_array(np.arange(30).reshape(10, 3))
        x[:3] = x[-3:] = masked
        assert_equal(median(x), 14.5)
        assert_(type(np.ma.median(x)) is not MaskedArray)
        assert_equal(median(x, axis=0), [13.5, 14.5, 15.5])
        assert_(type(np.ma.median(x, axis=0)) is MaskedArray)
        assert_equal(median(x, axis=1), [0, 0, 0, 10, 13, 16, 19, 0, 0, 0])
        assert_(type(np.ma.median(x, axis=1)) is MaskedArray)
        assert_equal(median(x, axis=1).mask, [1, 1, 1, 0, 0, 0, 0, 1, 1, 1])

    def test_3d(self):
        # Tests median with 3D arrays.
        x = np.ma.arange(24).reshape(3, 4, 2)
        x[x % 3 == 0] = masked
        assert_equal(median(x, 0), [[12, 9], [6, 15], [12, 9], [18, 15]])
        x.shape = (4, 3, 2)
        assert_equal(median(x, 0), [[99, 10], [11, 99], [13, 14]])
        x = np.ma.arange(24).reshape(4, 3, 2)
        x[x % 5 == 0] = masked
        assert_equal(median(x, 0), [[12, 10], [8, 9], [16, 17]])

    def test_neg_axis(self):
        # Tests median with negative axis values.
        x = masked_array(np.arange(30).reshape(10, 3))
        x[:3] = x[-3:] = masked
        assert_equal(median(x, axis=-1), median(x, axis=1))

    def test_out_1d(self):
        # Tests median with output to 1D arrays.
        for v in (30, 30., 31, 31.):
            x = masked_array(np.arange(v))
            x[:3] = x[-3:] = masked
            out = masked_array(np.ones(()))
            r = median(x, out=out)
            if v == 30:
                assert_equal(out, 14.5)
            else:
                assert_equal(out, 15.)
            assert_(r is out)
            assert_(type(r) is MaskedArray)

    def test_out(self):
        # Tests median with output arrays.
        for v in (40, 40., 30, 30.):
            x = masked_array(np.arange(v).reshape(10, -1))
            x[:3] = x[-3:] = masked
            out = masked_array(np.ones(10))
            r = median(x, axis=1, out=out)
            if v == 30:
                e = masked_array([0.]*3 + [10, 13, 16, 19] + [0.]*3,
                                 mask=[True] * 3 + [False] * 4 + [True] * 3)
            else:
                e = masked_array([0.]*3 + [13.5, 17.5, 21.5, 25.5] + [0.]*3,
                                 mask=[True]*3 + [False]*4 + [True]*3)
            assert_equal(r, e)
            assert_(r is out)
            assert_(type(r) is MaskedArray)

    @pytest.mark.parametrize(
        argnames='axis',
        argvalues=[
            None,
            1,
            (1, ),
            (0, 1),
            (-3, -1),
        ]
    )
    def test_keepdims_out(self, axis):
        # 创建一个布尔类型的零矩阵作为遮罩，形状为 (3, 5, 7, 11)
        mask = np.zeros((3, 5, 7, 11), dtype=bool)
        # 随机设置部分元素为 True：
        w = np.random.random((4, 200)) * np.array(mask.shape)[:, None]
        w = w.astype(np.intp)
        # 将 mask 中指定位置设置为 NaN
        mask[tuple(w)] = np.nan
        # 使用 mask 创建一个带有遮罩的 masked_array 对象 d，初始值为全 1
        d = masked_array(np.ones(mask.shape), mask=mask)
        if axis is None:
            # 如果 axis 为 None，则输出形状为 d 的维度全为 1 的元组
            shape_out = (1,) * d.ndim
        else:
            # 根据传入的 axis 规范化后的元组
            axis_norm = normalize_axis_tuple(axis, d.ndim)
            # 根据 axis_norm 创建输出形状 shape_out
            shape_out = tuple(
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim))
        # 创建一个空的 masked_array 对象 out，形状为 shape_out
        out = masked_array(np.empty(shape_out))
        # 计算 d 沿指定 axis 的中位数，保持维度为 True，并将结果存入 out
        result = median(d, axis=axis, keepdims=True, out=out)
        # 检查 result 是否为 out
        assert result is out
        # 检查 result 的形状是否与 shape_out 相同
        assert_equal(result.shape, shape_out)

    def test_single_non_masked_value_on_axis(self):
        # 创建一个包含非遮罩值的数据列表
        data = [[1., 0.],
                [0., 3.],
                [0., 0.]]
        # 创建一个带有遮罩值的 masked_array 对象 masked_arr
        masked_arr = np.ma.masked_equal(data, 0)
        # 预期的中位数值列表
        expected = [1., 3.]
        # 检查沿 axis=0 的中位数是否与预期值相等
        assert_array_equal(np.ma.median(masked_arr, axis=0),
                           expected)

    def test_nan(self):
        # 遍历两种 mask 情况
        for mask in (False, np.zeros(6, dtype=bool)):
            # 创建一个带有 NaN 值的 masked_array 对象 dm
            dm = np.ma.array([[1, np.nan, 3], [1, 2, 3]])
            dm.mask = mask

            # 标量结果
            r = np.ma.median(dm, axis=None)
            assert_(np.isscalar(r))
            assert_array_equal(r, np.nan)
            r = np.ma.median(dm.ravel(), axis=0)
            assert_(np.isscalar(r))
            assert_array_equal(r, np.nan)

            # 沿 axis=0 计算中位数，预期结果为 MaskedArray 类型
            r = np.ma.median(dm, axis=0)
            assert_equal(type(r), MaskedArray)
            assert_array_equal(r, [1, np.nan, 3])
            # 沿 axis=1 计算中位数，预期结果为 MaskedArray 类型
            r = np.ma.median(dm, axis=1)
            assert_equal(type(r), MaskedArray)
            assert_array_equal(r, [np.nan, 2])
            # 沿 axis=-1 计算中位数，预期结果为 MaskedArray 类型
            r = np.ma.median(dm, axis=-1)
            assert_equal(type(r), MaskedArray)
            assert_array_equal(r, [np.nan, 2])

        # 创建一个带有遮罩值的 masked_array 对象 dm
        dm = np.ma.array([[1, np.nan, 3], [1, 2, 3]])
        dm[:, 2] = np.ma.masked
        # 检查沿 axis=None 的中位数是否为 NaN
        assert_array_equal(np.ma.median(dm, axis=None), np.nan)
        # 沿 axis=0 计算中位数，预期结果为 [1, NaN, 3]
        assert_array_equal(np.ma.median(dm, axis=0), [1, np.nan, 3])
        # 沿 axis=1 计算中位数，预期结果为 [NaN, 1.5]
        assert_array_equal(np.ma.median(dm, axis=1), [np.nan, 1.5])

    def test_out_nan(self):
        # 创建一个全为零的 masked_array 对象 o
        o = np.ma.masked_array(np.zeros((4,)))
        # 创建一个全为一的 masked_array 对象 d，并设置部分值为 NaN 和遮罩
        d = np.ma.masked_array(np.ones((3, 4)))
        d[2, 1] = np.nan
        d[2, 2] = np.ma.masked
        # 沿 axis=0 计算中位数，并将结果存入 o，预期结果为 o 全为遮罩
        assert_equal(np.ma.median(d, 0, out=o), o)
        # 创建一个全为零的形状为 (3,) 的 masked_array 对象 o
        o = np.ma.masked_array(np.zeros((3,)))
        # 沿 axis=1 计算中位数，并将结果存入 o，预期结果为 o 全为遮罩
        assert_equal(np.ma.median(d, 1, out=o), o)
        # 创建一个全为零的标量 masked_array 对象 o
        o = np.ma.masked_array(np.zeros(()))
        # 计算整体中位数，并将结果存入 o，预期结果为 o 为遮罩
        assert_equal(np.ma.median(d, out=o), o)
    def test_nan_behavior(self):
        # 创建一个带有掩码数组的 MaskedArray 对象，包含从 0 到 23 的浮点数
        a = np.ma.masked_array(np.arange(24, dtype=float))
        # 设置每隔三个元素为掩码
        a[::3] = np.ma.masked
        # 将索引为 2 的元素设为 NaN
        a[2] = np.nan
        # 断言计算数组的中位数结果为 NaN
        assert_array_equal(np.ma.median(a), np.nan)
        # 断言沿着 axis=0 方向计算数组的中位数结果为 NaN
        assert_array_equal(np.ma.median(a, axis=0), np.nan)

        # 创建一个形状为 (2, 3, 4) 的带有掩码数组的 MaskedArray 对象
        a = np.ma.masked_array(np.arange(24, dtype=float).reshape(2, 3, 4))
        # 根据条件创建掩码
        a.mask = np.arange(a.size) % 2 == 1
        # 复制数组
        aorig = a.copy()
        # 将索引为 (1, 2, 3) 的元素设为 NaN
        a[1, 2, 3] = np.nan
        # 将索引为 (1, 1, 2) 的元素设为 NaN
        a[1, 1, 2] = np.nan

        # 没有指定 axis，断言计算数组的中位数结果为 NaN
        assert_array_equal(np.ma.median(a), np.nan)
        # 断言计算数组的中位数结果是标量
        assert_(np.isscalar(np.ma.median(a)))

        # 沿着 axis=0 方向计算数组的中位数，生成参考结果 b
        b = np.ma.median(aorig, axis=0)
        b[2, 3] = np.nan
        b[1, 2] = np.nan
        # 断言沿着 axis=0 方向计算数组的中位数与参考结果 b 相等
        assert_equal(np.ma.median(a, 0), b)

        # 沿着 axis=1 方向计算数组的中位数，生成参考结果 b
        b = np.ma.median(aorig, axis=1)
        b[1, 3] = np.nan
        b[1, 2] = np.nan
        # 断言沿着 axis=1 方向计算数组的中位数与参考结果 b 相等
        assert_equal(np.ma.median(a, 1), b)

        # 沿着 axis=(0, 2) 方向计算数组的中位数，生成参考结果 b
        b = np.ma.median(aorig, axis=(0, 2))
        b[1] = np.nan
        b[2] = np.nan
        # 断言沿着 axis=(0, 2) 方向计算数组的中位数与参考结果 b 相等
        assert_equal(np.ma.median(a, (0, 2)), b)

    def test_ambigous_fill(self):
        # 创建一个二维数组，使用 255 作为填充值
        # 255 被用作排序时的填充值
        a = np.array([[3, 3, 255], [3, 3, 255]], dtype=np.uint8)
        # 创建一个带有掩码的 MaskedArray 对象，将值为 3 的元素设为掩码
        a = np.ma.masked_array(a, mask=a == 3)
        # 断言沿着 axis=1 方向计算数组的中位数结果为 255
        assert_array_equal(np.ma.median(a, axis=1), 255)
        # 断言沿着 axis=1 方向计算数组的中位数的掩码为 False
        assert_array_equal(np.ma.median(a, axis=1).mask, False)
        # 断言沿着 axis=0 方向计算数组的中位数结果与 a 的第一行相等
        assert_array_equal(np.ma.median(a, axis=0), a[0])
        # 断言计算整个数组的中位数结果为 255
        assert_array_equal(np.ma.median(a), 255)
    # 定义一个名为 test_special 的测试方法，使用了 numpy 和 numpy.ma 模块
    def test_special(self):
        # 循环遍历列表中的 np.inf 和 -np.inf
        for inf in [np.inf, -np.inf]:
            # 创建一个包含 inf 和 np.nan 的二维数组 a
            a = np.array([[inf,  np.nan], [np.nan, np.nan]])
            # 使用 np.isnan(a) 创建一个布尔掩码，将其中的 NaN 设置为掩码
            a = np.ma.masked_array(a, mask=np.isnan(a))
            # 断言计算沿轴0和轴1的 a 的中位数，应为 [inf, np.nan]
            assert_equal(np.ma.median(a, axis=0), [inf,  np.nan])
            assert_equal(np.ma.median(a, axis=1), [inf,  np.nan])
            # 断言计算 a 的全局中位数，应为 inf
            assert_equal(np.ma.median(a), inf)

            # 创建另一个二维数组 a，包含 np.nan 和 inf
            a = np.array([[np.nan, np.nan, inf], [np.nan, np.nan, inf]])
            # 使用 np.isnan(a) 创建掩码，将其中的 NaN 设置为掩码
            a = np.ma.masked_array(a, mask=np.isnan(a))
            # 断言计算沿轴1的 a 的中位数，应为 inf
            assert_array_equal(np.ma.median(a, axis=1), inf)
            # 断言检查沿轴1的掩码，应为 False（即无掩码）
            assert_array_equal(np.ma.median(a, axis=1).mask, False)
            # 断言计算沿轴0的 a 的中位数，应为 a[0]
            assert_array_equal(np.ma.median(a, axis=0), a[0])
            # 断言计算 a 的全局中位数，应为 inf
            assert_array_equal(np.ma.median(a), inf)

            # 对于没有掩码的情况
            # 创建一个二维数组 a，包含 inf
            a = np.array([[inf, inf], [inf, inf]])
            # 断言计算 a 的全局中位数，应为 inf
            assert_equal(np.ma.median(a), inf)
            # 断言计算沿轴0的 a 的中位数，应为 inf
            assert_equal(np.ma.median(a, axis=0), inf)
            # 断言计算沿轴1的 a 的中位数，应为 inf
            assert_equal(np.ma.median(a, axis=1), inf)

            # 创建一个带有掩码的浮点数二维数组 a
            a = np.array([[inf, 7, -inf, -9],
                          [-10, np.nan, np.nan, 5],
                          [4, np.nan, np.nan, inf]],
                          dtype=np.float32)
            # 使用 np.isnan(a) 创建掩码，将其中的 NaN 设置为掩码
            a = np.ma.masked_array(a, mask=np.isnan(a))
            # 根据 inf 的正负值选择不同的断言条件
            if inf > 0:
                # 断言计算沿轴0的 a 的中位数，应为 [4., 7., -inf, 5.]
                assert_equal(np.ma.median(a, axis=0), [4., 7., -inf, 5.])
                # 断言计算 a 的全局中位数，应为 4.5
                assert_equal(np.ma.median(a), 4.5)
            else:
                # 断言计算沿轴0的 a 的中位数，应为 [-10., 7., -inf, -9.]
                assert_equal(np.ma.median(a, axis=0), [-10., 7., -inf, -9.])
                # 断言计算 a 的全局中位数，应为 -2.5
                assert_equal(np.ma.median(a), -2.5)
            # 断言计算沿轴1的 a 的中位数，应为 [-1., -2.5, inf]
            assert_equal(np.ma.median(a, axis=1), [-1., -2.5, inf])

            # 嵌套循环，生成多组具有不同数量 inf 的二维数组 a
            for i in range(0, 10):
                for j in range(1, 10):
                    # 创建一个二维数组 a，包含 i 个 np.nan 和 j 个 inf
                    a = np.array([([np.nan] * i) + ([inf] * j)] * 2)
                    # 使用 np.isnan(a) 创建掩码，将其中的 NaN 设置为掩码
                    a = np.ma.masked_array(a, mask=np.isnan(a))
                    # 断言计算 a 的全局中位数，应为 inf
                    assert_equal(np.ma.median(a), inf)
                    # 断言计算沿轴1的 a 的中位数，应为 inf
                    assert_equal(np.ma.median(a, axis=1), inf)
                    # 断言计算沿轴0的 a 的中位数，应为 ([np.nan] * i) + [inf] * j
                    assert_equal(np.ma.median(a, axis=0),
                                 ([np.nan] * i) + [inf] * j)
    # 定义测试方法，用于测试空数组情况
    def test_empty(self):
        # 创建一个空的掩码数组 a
        a = np.ma.masked_array(np.array([], dtype=float))
        # 在上下文中抑制特定警告
        with suppress_warnings() as w:
            # 记录 RuntimeWarning 警告
            w.record(RuntimeWarning)
            # 断言计算 a 的中位数结果为 NaN
            assert_array_equal(np.ma.median(a), np.nan)
            # 断言第一个记录的警告类别为 RuntimeWarning
            assert_(w.log[0].category is RuntimeWarning)

        # 创建一个多维空数组 a
        a = np.ma.masked_array(np.array([], dtype=float, ndmin=3))
        # 在上下文中抑制特定警告
        with suppress_warnings() as w:
            # 记录 RuntimeWarning 警告
            w.record(RuntimeWarning)
            # 设置警告过滤器，始终记录 RuntimeWarning
            warnings.filterwarnings('always', '', RuntimeWarning)
            # 断言计算 a 的中位数结果为 NaN
            assert_array_equal(np.ma.median(a), np.nan)
            # 断言第一个记录的警告类别为 RuntimeWarning
            assert_(w.log[0].category is RuntimeWarning)

        # 创建一个二维空数组 b
        b = np.ma.masked_array(np.array([], dtype=float, ndmin=2))
        # 断言沿轴 0 计算 a 的中位数结果与 b 相等
        assert_equal(np.ma.median(a, axis=0), b)
        # 断言沿轴 1 计算 a 的中位数结果与 b 相等
        assert_equal(np.ma.median(a, axis=1), b)

        # 创建一个二维包含 NaN 的数组 b
        b = np.ma.masked_array(np.array(np.nan, dtype=float, ndmin=2))
        # 在上下文中捕获警告
        with warnings.catch_warnings(record=True) as w:
            # 设置警告过滤器，始终记录 RuntimeWarning
            warnings.filterwarnings('always', '', RuntimeWarning)
            # 断言沿轴 2 计算 a 的中位数结果与 b 相等
            assert_equal(np.ma.median(a, axis=2), b)
            # 断言第一个记录的警告类别为 RuntimeWarning
            assert_(w[0].category is RuntimeWarning)

    # 定义测试方法，用于测试对象数组情况
    def test_object(self):
        # 创建一个对象数组 o，包含连续的数字
        o = np.ma.masked_array(np.arange(7.))
        # 断言 np.ma.median(o.astype(object)) 的类型为 float
        assert_(type(np.ma.median(o.astype(object))), float)
        # 将第二个元素设置为 NaN
        o[2] = np.nan
        # 断言 np.ma.median(o.astype(object)) 的类型为 float
        assert_(type(np.ma.median(o.astype(object))), float)
class TestCov:

    def setup_method(self):
        # 设置测试方法的初始化，创建一个包含随机数据的一维数组
        self.data = array(np.random.rand(12))

    def test_covhelper(self):
        x = self.data
        # 测试_covhelper函数的输出类型为float32
        assert_(_covhelper(x, rowvar=True)[1].dtype, np.float32)
        assert_(_covhelper(x, y=x, rowvar=False)[1].dtype, np.float32)
        # 测试在转换为float后，_covhelper函数的输出是否相等
        mask = x > 0.5
        assert_array_equal(
            _covhelper(
                np.ma.masked_array(x, mask), rowvar=True
            )[1].astype(bool),
            ~mask.reshape(1, -1),
        )
        assert_array_equal(
            _covhelper(
                np.ma.masked_array(x, mask), y=x, rowvar=False
            )[1].astype(bool),
            np.vstack((~mask, ~mask)),
        )

    def test_1d_without_missing(self):
        # 测试在没有缺失值的一维变量上计算协方差
        x = self.data
        assert_almost_equal(np.cov(x), cov(x))
        assert_almost_equal(np.cov(x, rowvar=False), cov(x, rowvar=False))
        assert_almost_equal(np.cov(x, rowvar=False, bias=True),
                            cov(x, rowvar=False, bias=True))

    def test_2d_without_missing(self):
        # 测试在没有缺失值的二维变量上计算协方差
        x = self.data.reshape(3, 4)
        assert_almost_equal(np.cov(x), cov(x))
        assert_almost_equal(np.cov(x, rowvar=False), cov(x, rowvar=False))
        assert_almost_equal(np.cov(x, rowvar=False, bias=True),
                            cov(x, rowvar=False, bias=True))

    def test_1d_with_missing(self):
        # 测试带有缺失值的一维变量上计算协方差
        x = self.data
        x[-1] = masked
        x -= x.mean()
        nx = x.compressed()
        assert_almost_equal(np.cov(nx), cov(x))
        assert_almost_equal(np.cov(nx, rowvar=False), cov(x, rowvar=False))
        assert_almost_equal(np.cov(nx, rowvar=False, bias=True),
                            cov(x, rowvar=False, bias=True))
        #
        try:
            # 测试在不允许使用遮罩的情况下调用cov函数是否会抛出ValueError异常
            cov(x, allow_masked=False)
        except ValueError:
            pass
        #
        # 测试带有缺失值的两个一维变量上计算协方差
        nx = x[1:-1]
        assert_almost_equal(np.cov(nx, nx[::-1]), cov(x, x[::-1]))
        assert_almost_equal(np.cov(nx, nx[::-1], rowvar=False),
                            cov(x, x[::-1], rowvar=False))
        assert_almost_equal(np.cov(nx, nx[::-1], rowvar=False, bias=True),
                            cov(x, x[::-1], rowvar=False, bias=True))
    # 定义一个测试方法，用于测试带有缺失值的二维变量的协方差函数
    def test_2d_with_missing(self):
        # 获取数据集并赋值给变量 x
        x = self.data
        # 将最后一个元素设置为缺失值 masked
        x[-1] = masked
        # 将 x 重新整形为 3 行 4 列的二维数组
        x = x.reshape(3, 4)
        # 生成一个逻辑非掩码数组，用于表示有效值
        valid = np.logical_not(getmaskarray(x)).astype(int)
        # 计算有效值矩阵的乘积，得到有效值的分数 frac
        frac = np.dot(valid, valid.T)
        # 对 x 减去每行均值并填充缺失值为 0，得到 xf
        xf = (x - x.mean(1)[:, None]).filled(0)
        # 使用自定义的协方差函数计算 xf 的协方差，并进行断言比较
        assert_almost_equal(cov(x),
                            np.cov(xf) * (x.shape[1] - 1) / (frac - 1.))
        # 使用带偏差的 np.cov 函数计算 xf 的协方差，并进行断言比较
        assert_almost_equal(cov(x, bias=True),
                            np.cov(xf, bias=True) * x.shape[1] / frac)
        # 重新计算有效值矩阵的乘积，得到有效值的分数 frac
        frac = np.dot(valid.T, valid)
        # 对 x 减去每列均值并填充缺失值为 0，得到 xf
        xf = (x - x.mean(0)).filled(0)
        # 使用自定义的协方差函数计算 xf 的协方差，并进行断言比较
        assert_almost_equal(cov(x, rowvar=False),
                            (np.cov(xf, rowvar=False) *
                             (x.shape[0] - 1) / (frac - 1.)))
        # 使用带偏差的 np.cov 函数计算 xf 的协方差，并进行断言比较
        assert_almost_equal(cov(x, rowvar=False, bias=True),
                            (np.cov(xf, rowvar=False, bias=True) *
                             x.shape[0] / frac))
class TestCorrcoef:

    def setup_method(self):
        # 设置测试方法的初始化，创建包含12个随机数的数组self.data和self.data2
        self.data = array(np.random.rand(12))
        self.data2 = array(np.random.rand(12))

    def test_ddof(self):
        # 测试ddof参数，预计会引发DeprecationWarning警告
        x, y = self.data, self.data2
        expected = np.corrcoef(x)
        expected2 = np.corrcoef(x, y)
        with suppress_warnings() as sup:
            warnings.simplefilter("always")
            assert_warns(DeprecationWarning, corrcoef, x, ddof=-1)
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            # ddof对函数几乎没有影响
            assert_almost_equal(np.corrcoef(x, ddof=0), corrcoef(x, ddof=0))
            assert_almost_equal(corrcoef(x, ddof=-1), expected)
            assert_almost_equal(corrcoef(x, y, ddof=-1), expected2)
            assert_almost_equal(corrcoef(x, ddof=3), expected)
            assert_almost_equal(corrcoef(x, y, ddof=3), expected2)

    def test_bias(self):
        x, y = self.data, self.data2
        expected = np.corrcoef(x)
        # 测试bias参数，预计会引发DeprecationWarning警告
        with suppress_warnings() as sup:
            warnings.simplefilter("always")
            assert_warns(DeprecationWarning, corrcoef, x, y, True, False)
            assert_warns(DeprecationWarning, corrcoef, x, y, True, True)
            assert_warns(DeprecationWarning, corrcoef, x, bias=False)
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            # bias对函数几乎没有影响
            assert_almost_equal(corrcoef(x, bias=1), expected)

    def test_1d_without_missing(self):
        # 测试没有缺失值的一维变量上的corrcoef函数
        x = self.data
        assert_almost_equal(np.corrcoef(x), corrcoef(x))
        assert_almost_equal(np.corrcoef(x, rowvar=False),
                            corrcoef(x, rowvar=False))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            assert_almost_equal(np.corrcoef(x, rowvar=False, bias=True),
                                corrcoef(x, rowvar=False, bias=True))

    def test_2d_without_missing(self):
        # 测试没有缺失值的二维变量上的corrcoef函数
        x = self.data.reshape(3, 4)
        assert_almost_equal(np.corrcoef(x), corrcoef(x))
        assert_almost_equal(np.corrcoef(x, rowvar=False),
                            corrcoef(x, rowvar=False))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            assert_almost_equal(np.corrcoef(x, rowvar=False, bias=True),
                                corrcoef(x, rowvar=False, bias=True))
    def test_1d_with_missing(self):
        # Test corrcoef 1 1D variable w/missing values
        # 使用带有缺失值的一维变量进行相关系数测试
        x = self.data
        x[-1] = masked
        # 将最后一个元素设为缺失值
        x -= x.mean()
        # 减去均值，标准化数据
        nx = x.compressed()
        # 去除缺失值后的数据
        assert_almost_equal(np.corrcoef(nx), corrcoef(x))
        # 检查相关系数函数对比，不考虑行变量
        assert_almost_equal(np.corrcoef(nx, rowvar=False),
                            corrcoef(x, rowvar=False))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            # 忽略特定警告，确保相关系数的偏置参数和自由度调整参数不影响结果
            assert_almost_equal(np.corrcoef(nx, rowvar=False, bias=True),
                                corrcoef(x, rowvar=False, bias=True))
        try:
            corrcoef(x, allow_masked=False)
        except ValueError:
            pass
        # 2个带有缺失值的一维变量
        nx = x[1:-1]
        assert_almost_equal(np.corrcoef(nx, nx[::-1]), corrcoef(x, x[::-1]))
        # 检查两个一维变量之间的相关系数，包括反向
        assert_almost_equal(np.corrcoef(nx, nx[::-1], rowvar=False),
                            corrcoef(x, x[::-1], rowvar=False))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            # ddof 和 bias 对函数几乎没有影响
            assert_almost_equal(np.corrcoef(nx, nx[::-1]),
                                corrcoef(x, x[::-1], bias=1))
            assert_almost_equal(np.corrcoef(nx, nx[::-1]),
                                corrcoef(x, x[::-1], ddof=2))

    def test_2d_with_missing(self):
        # Test corrcoef on 2D variable w/ missing value
        # 使用带有缺失值的二维变量进行相关系数测试
        x = self.data
        x[-1] = masked
        # 将最后一个元素设为缺失值
        x = x.reshape(3, 4)
        # 将数据重塑为3行4列的二维数组

        test = corrcoef(x)
        control = np.corrcoef(x)
        assert_almost_equal(test[:-1, :-1], control[:-1, :-1])
        # 检查二维数组的相关系数结果

        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            # ddof 和 bias 对函数几乎没有影响
            assert_almost_equal(corrcoef(x, ddof=-2)[:-1, :-1],
                                control[:-1, :-1])
            assert_almost_equal(corrcoef(x, ddof=3)[:-1, :-1],
                                control[:-1, :-1])
            assert_almost_equal(corrcoef(x, bias=1)[:-1, :-1],
                                control[:-1, :-1])
class TestPolynomial:
    # 多项式测试类

    def test_polyfit(self):
        # 测试 polyfit 函数
        # 对于 ndarrays
        x = np.random.rand(10)
        y = np.random.rand(20).reshape(-1, 2)
        assert_almost_equal(polyfit(x, y, 3), np.polyfit(x, y, 3))
        
        # 对于 1D maskedarrays
        x = x.view(MaskedArray)
        x[0] = masked
        y = y.view(MaskedArray)
        y[0, 0] = y[-1, -1] = masked
        
        # 测试 polyfit 函数的返回值，并进行断言比较
        (C, R, K, S, D) = polyfit(x, y[:, 0], 3, full=True)
        (c, r, k, s, d) = np.polyfit(x[1:], y[1:, 0].compressed(), 3,
                                     full=True)
        for (a, a_) in zip((C, R, K, S, D), (c, r, k, s, d)):
            assert_almost_equal(a, a_)
        
        # 同样的测试方式，但是针对另一个列
        (C, R, K, S, D) = polyfit(x, y[:, -1], 3, full=True)
        (c, r, k, s, d) = np.polyfit(x[1:-1], y[1:-1, -1], 3, full=True)
        for (a, a_) in zip((C, R, K, S, D), (c, r, k, s, d)):
            assert_almost_equal(a, a_)
        
        # 测试对整个 y 数组进行 polyfit 的情况
        (C, R, K, S, D) = polyfit(x, y, 3, full=True)
        (c, r, k, s, d) = np.polyfit(x[1:-1], y[1:-1,:], 3, full=True)
        for (a, a_) in zip((C, R, K, S, D), (c, r, k, s, d)):
            assert_almost_equal(a, a_)
        
        # 测试带权重 w 的 polyfit
        w = np.random.rand(10) + 1
        wo = w.copy()
        xs = x[1:-1]
        ys = y[1:-1]
        ws = w[1:-1]
        (C, R, K, S, D) = polyfit(x, y, 3, full=True, w=w)
        (c, r, k, s, d) = np.polyfit(xs, ys, 3, full=True, w=ws)
        assert_equal(w, wo)
        for (a, a_) in zip((C, R, K, S, D), (c, r, k, s, d)):
            assert_almost_equal(a, a_)

    def test_polyfit_with_masked_NaNs(self):
        # 测试带有 NaN 和 masked 值的 polyfit
        x = np.random.rand(10)
        y = np.random.rand(20).reshape(-1, 2)

        x[0] = np.nan
        y[-1,-1] = np.nan
        x = x.view(MaskedArray)
        y = y.view(MaskedArray)
        x[0] = masked
        y[-1,-1] = masked

        (C, R, K, S, D) = polyfit(x, y, 3, full=True)
        (c, r, k, s, d) = np.polyfit(x[1:-1], y[1:-1,:], 3, full=True)
        for (a, a_) in zip((C, R, K, S, D), (c, r, k, s, d)):
            assert_almost_equal(a, a_)


class TestArraySetOps:

    def test_unique_onlist(self):
        # 测试在列表上的 unique 函数
        data = [1, 1, 1, 2, 2, 3]
        test = unique(data, return_index=True, return_inverse=True)
        assert_(isinstance(test[0], MaskedArray))
        assert_equal(test[0], masked_array([1, 2, 3], mask=[0, 0, 0]))
        assert_equal(test[1], [0, 3, 5])
        assert_equal(test[2], [0, 0, 0, 1, 1, 2])
    def test_unique_onmaskedarray(self):
        # 测试在具有掩码数据的情况下使用 unique 函数，使用 use_mask=True 参数
        data = masked_array([1, 1, 1, 2, 2, 3], mask=[0, 0, 1, 0, 1, 0])
        test = unique(data, return_index=True, return_inverse=True)
        assert_equal(test[0], masked_array([1, 2, 3, -1], mask=[0, 0, 0, 1]))
        assert_equal(test[1], [0, 3, 5, 2])
        assert_equal(test[2], [0, 0, 3, 1, 3, 2])
        #
        # 设置数据的填充值为 3
        data.fill_value = 3
        # 使用给定的掩码和填充值创建掩码数组
        data = masked_array(data=[1, 1, 1, 2, 2, 3],
                            mask=[0, 0, 1, 0, 1, 0], fill_value=3)
        test = unique(data, return_index=True, return_inverse=True)
        assert_equal(test[0], masked_array([1, 2, 3, -1], mask=[0, 0, 0, 1]))
        assert_equal(test[1], [0, 3, 5, 2])
        assert_equal(test[2], [0, 0, 3, 1, 3, 2])

    def test_unique_allmasked(self):
        # 测试全为掩码数据的情况
        data = masked_array([1, 1, 1], mask=True)
        test = unique(data, return_index=True, return_inverse=True)
        assert_equal(test[0], masked_array([1, ], mask=[True]))
        assert_equal(test[1], [0])
        assert_equal(test[2], [0, 0, 0])
        #
        # 测试掩码数据的情况
        data = masked
        test = unique(data, return_index=True, return_inverse=True)
        assert_equal(test[0], masked_array(masked))
        assert_equal(test[1], [0])
        assert_equal(test[2], [0])

    def test_ediff1d(self):
        # 测试 ediff1d 函数
        x = masked_array(np.arange(5), mask=[1, 0, 0, 0, 1])
        control = array([1, 1, 1, 4], mask=[1, 0, 0, 1])
        test = ediff1d(x)
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_ediff1d_tobegin(self):
        # 测试带有 to_begin 参数的 ediff1d 函数
        x = masked_array(np.arange(5), mask=[1, 0, 0, 0, 1])
        test = ediff1d(x, to_begin=masked)
        control = array([0, 1, 1, 1, 4], mask=[1, 1, 0, 0, 1])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)
        #
        test = ediff1d(x, to_begin=[1, 2, 3])
        control = array([1, 2, 3, 1, 1, 1, 4], mask=[0, 0, 0, 1, 0, 0, 1])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_ediff1d_toend(self):
        # 测试带有 to_end 参数的 ediff1d 函数
        x = masked_array(np.arange(5), mask=[1, 0, 0, 0, 1])
        test = ediff1d(x, to_end=masked)
        control = array([1, 1, 1, 4, 0], mask=[1, 0, 0, 1, 1])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)
        #
        test = ediff1d(x, to_end=[1, 2, 3])
        control = array([1, 1, 1, 4, 1, 2, 3], mask=[1, 0, 0, 1, 0, 0, 0])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)
    def test_ediff1d_tobegin_toend(self):
        # Test ediff1d w/ to_begin and to_end
        x = masked_array(np.arange(5), mask=[1, 0, 0, 0, 1])
        # 调用 ediff1d 函数，设置 to_begin 和 to_end 参数，并对结果进行测试
        test = ediff1d(x, to_end=masked, to_begin=masked)
        control = array([0, 1, 1, 1, 4, 0], mask=[1, 1, 0, 0, 1, 1])
        # 断言测试结果与预期结果相等
        assert_equal(test, control)
        # 断言测试结果经过填充后与预期结果经过填充后相等
        assert_equal(test.filled(0), control.filled(0))
        # 断言测试结果的屏蔽（mask）与预期结果的屏蔽相等
        assert_equal(test.mask, control.mask)
        #
        test = ediff1d(x, to_end=[1, 2, 3], to_begin=masked)
        control = array([0, 1, 1, 1, 4, 1, 2, 3],
                        mask=[1, 1, 0, 0, 1, 0, 0, 0])
        # 断言测试结果与预期结果相等
        assert_equal(test, control)
        # 断言测试结果经过填充后与预期结果经过填充后相等
        assert_equal(test.filled(0), control.filled(0))
        # 断言测试结果的屏蔽（mask）与预期结果的屏蔽相等
        assert_equal(test.mask, control.mask)

    def test_ediff1d_ndarray(self):
        # Test ediff1d w/ a ndarray
        x = np.arange(5)
        # 调用 ediff1d 函数，对 ndarray 进行处理，并进行测试
        test = ediff1d(x)
        control = array([1, 1, 1, 1], mask=[0, 0, 0, 0])
        # 断言测试结果与预期结果相等
        assert_equal(test, control)
        # 断言测试结果是 MaskedArray 类型
        assert_(isinstance(test, MaskedArray))
        # 断言测试结果经过填充后与预期结果经过填充后相等
        assert_equal(test.filled(0), control.filled(0))
        # 断言测试结果的屏蔽（mask）与预期结果的屏蔽相等
        assert_equal(test.mask, control.mask)
        #
        test = ediff1d(x, to_end=masked, to_begin=masked)
        control = array([0, 1, 1, 1, 1, 0], mask=[1, 0, 0, 0, 0, 1])
        # 断言测试结果是 MaskedArray 类型
        assert_(isinstance(test, MaskedArray))
        # 断言测试结果经过填充后与预期结果经过填充后相等
        assert_equal(test.filled(0), control.filled(0))
        # 断言测试结果的屏蔽（mask）与预期结果的屏蔽相等
        assert_equal(test.mask, control.mask)

    def test_intersect1d(self):
        # Test intersect1d
        x = array([1, 3, 3, 3], mask=[0, 0, 0, 1])
        y = array([3, 1, 1, 1], mask=[0, 0, 0, 1])
        # 调用 intersect1d 函数，测试两个数组的交集
        test = intersect1d(x, y)
        control = array([1, 3, -1], mask=[0, 0, 1])
        # 断言测试结果与预期结果相等
        assert_equal(test, control)

    def test_setxor1d(self):
        # Test setxor1d
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 2, 3, 4, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        # 调用 setxor1d 函数，测试两个数组的对称差集
        test = setxor1d(a, b)
        assert_equal(test, array([3, 4, 7]))
        #
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = [1, 2, 3, 4, 5]
        # 调用 setxor1d 函数，测试数组和普通序列的对称差集
        test = setxor1d(a, b)
        assert_equal(test, array([3, 4, 7, -1], mask=[0, 0, 0, 1]))
        #
        a = array([1, 2, 3])
        b = array([6, 5, 4])
        # 调用 setxor1d 函数，测试两个普通数组的对称差集
        test = setxor1d(a, b)
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, [1, 2, 3, 4, 5, 6])
        #
        a = array([1, 8, 2, 3], mask=[0, 1, 0, 0])
        b = array([6, 5, 4, 8], mask=[0, 0, 0, 1])
        # 调用 setxor1d 函数，测试带屏蔽的数组的对称差集
        test = setxor1d(a, b)
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, [1, 2, 3, 4, 5, 6])
        #
        # 断言空数组的对称差集为空
        assert_array_equal([], setxor1d([], []))
    def test_setxor1d_unique(self):
        # 测试 setxor1d 函数，使用 assume_unique=True 参数
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = [1, 2, 3, 4, 5]
        # 调用 setxor1d 函数进行计算
        test = setxor1d(a, b, assume_unique=True)
        # 断言结果是否与预期相等
        assert_equal(test, array([3, 4, 7, -1], mask=[0, 0, 0, 1]))
        #
        a = array([1, 8, 2, 3], mask=[0, 1, 0, 0])
        b = array([6, 5, 4, 8], mask=[0, 0, 0, 1])
        # 再次调用 setxor1d 函数进行计算
        test = setxor1d(a, b, assume_unique=True)
        # 断言结果是否为 MaskedArray 类型
        assert_(isinstance(test, MaskedArray))
        # 断言结果是否与预期相等
        assert_equal(test, [1, 2, 3, 4, 5, 6])
        #
        a = array([[1], [8], [2], [3]])
        b = array([[6, 5], [4, 8]])
        # 第三次调用 setxor1d 函数进行计算
        test = setxor1d(a, b, assume_unique=True)
        # 断言结果是否为 MaskedArray 类型
        assert_(isinstance(test, MaskedArray))
        # 断言结果是否与预期相等
        assert_equal(test, [1, 2, 3, 4, 5, 6])

    def test_isin(self):
        # 大部分 isin 的行为已经由 in1d 的测试覆盖
        # 如果移除 in1d 函数，需要修改这些测试来测试 isin 函数。
        a = np.arange(24).reshape([2, 3, 4])
        mask = np.zeros([2, 3, 4])
        mask[1, 2, 0] = 1
        a = array(a, mask=mask)
        b = array(data=[0, 10, 20, 30,  1,  3, 11, 22, 33],
                  mask=[0,  1,  0,  1,  0,  1,  0,  1,  0])
        ec = zeros((2, 3, 4), dtype=bool)
        ec[0, 0, 0] = True
        ec[0, 0, 1] = True
        ec[0, 2, 3] = True
        # 调用 isin 函数进行计算
        c = isin(a, b)
        # 断言结果是否为 MaskedArray 类型
        assert_(isinstance(c, MaskedArray))
        # 断言结果数组是否与预期数组 ec 相等
        assert_array_equal(c, ec)
        # 将 np.isin 的结果与 ma.isin 进行比较
        d = np.isin(a, b[~b.mask]) & ~a.mask
        # 断言两者结果是否相等
        assert_array_equal(c, d)

    def test_in1d(self):
        # 测试 in1d 函数
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 2, 3, 4, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        # 调用 in1d 函数进行计算
        test = in1d(a, b)
        # 断言结果是否与预期相等
        assert_equal(test, [True, True, True, False, True])
        #
        a = array([5, 5, 2, 1, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 5, -1], mask=[0, 0, 1])
        # 再次调用 in1d 函数进行计算
        test = in1d(a, b)
        # 断言结果是否与预期相等
        assert_equal(test, [True, True, False, True, True])
        #
        assert_array_equal([], in1d([], []))

    def test_in1d_invert(self):
        # 测试 in1d 函数的 invert 参数
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 2, 3, 4, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        # 断言 np.invert(in1d(a, b)) 的结果与 in1d(a, b, invert=True) 的结果是否相等
        assert_equal(np.invert(in1d(a, b)), in1d(a, b, invert=True))

        a = array([5, 5, 2, 1, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 5, -1], mask=[0, 0, 1])
        # 断言 np.invert(in1d(a, b)) 的结果与 in1d(a, b, invert=True) 的结果是否相等
        assert_equal(np.invert(in1d(a, b)), in1d(a, b, invert=True))

        assert_array_equal([], in1d([], [], invert=True))
    # 定义测试函数test_union1d
    def test_union1d(self):
        # 测试union1d函数
        # 创建数组a，包含数据和掩码
        a = array([1, 2, 5, 7, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        # 创建数组b，包含数据和掩码
        b = array([1, 2, 3, 4, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        # 使用union1d函数对数组a和b进行合并
        test = union1d(a, b)
        # 创建参照数组control
        control = array([1, 2, 3, 4, 5, 7, -1], mask=[0, 0, 0, 0, 0, 0, 1])
        # 断言test和control是否相等
        assert_equal(test, control)

        # 测试gh-10340，union1d的参数如果不是1D，则应将其展开
        # 创建数组x，包含数据和掩码
        x = array([[0, 1, 2], [3, 4, 5]], mask=[[0, 0, 0], [0, 0, 1]])
        # 创建数组y，包含数据和掩码
        y = array([0, 1, 2, 3, 4], mask=[0, 0, 0, 0, 1])
        # 创建参照数组ez
        ez = array([0, 1, 2, 3, 4, 5], mask=[0, 0, 0, 0, 0, 1])
        # 使用union1d函数对数组x和y进行合并
        z = union1d(x, y)
        # 断言z和ez是否相等
        assert_equal(z, ez)
        # 断言空数组的情况
        assert_array_equal([], union1d([], []))

    # 定义测试函数test_setdiff1d
    def test_setdiff1d(self):
        # 测试setdiff1d函数
        # 创建数组a，包含数据和掩码
        a = array([6, 5, 4, 7, 7, 1, 2, 1], mask=[0, 0, 0, 0, 0, 0, 0, 1])
        # 创建数组b
        b = array([2, 4, 3, 3, 2, 1, 5])
        # 使用setdiff1d函数对数组a和b进行差集运算
        test = setdiff1d(a, b)
        # 断言test和期望的数组是否相等
        assert_equal(test, array([6, 7, -1], mask=[0, 0, 1]))
        # 测试arange的情况
        a = arange(10)
        b = arange(8)
        # 断言两个数组的差集是否与期望数组相等
        assert_equal(setdiff1d(a, b), array([8, 9]))
        a = array([], np.uint32, mask=[])
        # 断言setdiff1d返回的数据类型是否为uint32
        assert_equal(setdiff1d(a, []).dtype, np.uint32)

    # 定义测试函数test_setdiff1d_char_array
    def test_setdiff1d_char_array(self):
        # 测试setdiff1d_char_array函数
        # 创建包含字符的数组a和b
        a = np.array(['a', 'b', 'c'])
        b = np.array(['a', 'b', 's'])
        # 断言两个数组的差集是否与期望数组相等
        assert_array_equal(setdiff1d(a, b), np.array(['c']))
class TestShapeBase:

    def test_atleast_2d(self):
        # Test atleast_2d
        # 创建一个带有掩码的数组 `a`
        a = masked_array([0, 1, 2], mask=[0, 1, 0])
        # 对 `a` 运行 atleast_2d 函数，返回 `b`
        b = atleast_2d(a)
        # 断言 `b` 的形状为 (1, 3)
        assert_equal(b.shape, (1, 3))
        # 断言 `b` 的掩码形状与数据形状相同
        assert_equal(b.mask.shape, b.data.shape)
        # 断言 `a` 的形状为 (3,)
        assert_equal(a.shape, (3,))
        # 断言 `a` 的掩码形状与数据形状相同
        assert_equal(a.mask.shape, a.data.shape)
        # 再次断言 `b` 的掩码形状与数据形状相同
        assert_equal(b.mask.shape, b.data.shape)

    def test_shape_scalar(self):
        # the atleast and diagflat function should work with scalars
        # GitHub issue #3367
        # Additionally, the atleast functions should accept multiple scalars
        # correctly

        # 对标量值 1.0 运行 atleast_1d 函数，返回 `b`
        b = atleast_1d(1.0)
        # 断言 `b` 的形状为 (1,)
        assert_equal(b.shape, (1,))
        # 断言 `b` 的掩码形状与形状相同
        assert_equal(b.mask.shape, b.shape)
        # 断言 `b` 的数据形状与形状相同
        assert_equal(b.data.shape, b.shape)

        # 对多个标量值 1.0, 2.0 运行 atleast_1d 函数，返回 `b`
        b = atleast_1d(1.0, 2.0)
        # 对于 `b` 中的每个元素 `a`
        for a in b:
            # 断言 `a` 的形状为 (1,)
            assert_equal(a.shape, (1,))
            # 断言 `a` 的掩码形状与形状相同
            assert_equal(a.mask.shape, a.shape)
            # 断言 `a` 的数据形状与形状相同
            assert_equal(a.data.shape, a.shape)

        # 对标量值 1.0 运行 atleast_2d 函数，返回 `b`
        b = atleast_2d(1.0)
        # 断言 `b` 的形状为 (1, 1)
        assert_equal(b.shape, (1, 1))
        # 断言 `b` 的掩码形状与形状相同
        assert_equal(b.mask.shape, b.shape)
        # 断言 `b` 的数据形状与形状相同
        assert_equal(b.data.shape, b.shape)

        # 对多个标量值 1.0, 2.0 运行 atleast_2d 函数，返回 `b`
        b = atleast_2d(1.0, 2.0)
        # 对于 `b` 中的每个元素 `a`
        for a in b:
            # 断言 `a` 的形状为 (1, 1)
            assert_equal(a.shape, (1, 1))
            # 断言 `a` 的掩码形状与形状相同
            assert_equal(a.mask.shape, a.shape)
            # 断言 `a` 的数据形状与形状相同
            assert_equal(a.data.shape, a.shape)

        # 对标量值 1.0 运行 atleast_3d 函数，返回 `b`
        b = atleast_3d(1.0)
        # 断言 `b` 的形状为 (1, 1, 1)
        assert_equal(b.shape, (1, 1, 1))
        # 断言 `b` 的掩码形状与形状相同
        assert_equal(b.mask.shape, b.shape)
        # 断言 `b` 的数据形状与形状相同
        assert_equal(b.data.shape, b.shape)

        # 对多个标量值 1.0, 2.0 运行 atleast_3d 函数，返回 `b`
        b = atleast_3d(1.0, 2.0)
        # 对于 `b` 中的每个元素 `a`
        for a in b:
            # 断言 `a` 的形状为 (1, 1, 1)
            assert_equal(a.shape, (1, 1, 1))
            # 断言 `a` 的掩码形状与形状相同
            assert_equal(a.mask.shape, a.shape)
            # 断言 `a` 的数据形状与形状相同
            assert_equal(a.data.shape, a.shape)

        # 对标量值 1.0 运行 diagflat 函数，返回 `b`
        b = diagflat(1.0)
        # 断言 `b` 的形状为 (1, 1)
        assert_equal(b.shape, (1, 1))
        # 断言 `b` 的掩码形状与数据形状相同
        assert_equal(b.mask.shape, b.data.shape)


class TestNDEnumerate:

    def test_ndenumerate_nomasked(self):
        # 创建普通数组 `ordinary`
        ordinary = np.arange(6.).reshape((1, 3, 2))
        # 创建与 `ordinary` 相同形状的全零掩码数组 `empty_mask`
        empty_mask = np.zeros_like(ordinary, dtype=bool)
        # 使用 `empty_mask` 创建带有掩码的数组 `with_mask`
        with_mask = masked_array(ordinary, mask=empty_mask)
        # 断言 `np.ndenumerate(ordinary)` 与 `ndenumerate(ordinary)` 的结果列表相同
        assert_equal(list(np.ndenumerate(ordinary)),
                     list(ndenumerate(ordinary)))
        # 断言 `ndenumerate(ordinary)` 与 `ndenumerate(with_mask)` 的结果列表相同
        assert_equal(list(ndenumerate(ordinary)),
                     list(ndenumerate(with_mask)))
        # 断言 `ndenumerate(with_mask)` 与 `ndenumerate(with_mask, compressed=False)` 的结果列表相同
        assert_equal(list(ndenumerate(with_mask)),
                     list(ndenumerate(with_mask, compressed=False)))

    def test_ndenumerate_allmasked(self):
        # 创建全掩码数组 `a`
        a = masked_all(())
        # 创建形状为 (100,) 的全掩码数组 `b`
        b = masked_all((100,))
        # 创建形状为 (2, 3, 4) 的全掩码数组 `c`
        c = masked_all((2, 3, 4))
        # 断言 `ndenumerate(a)` 的结果列表为空
        assert_equal(list(ndenumerate(a)), [])
        # 断言 `ndenumerate(b)` 的结果列表为空
        assert_equal(list(ndenumerate(b)), [])
        # 断言 `ndenumerate(b, compressed=False)` 的结果列表与预期列表相同
        assert_equal(list(ndenumerate(b, compressed=False)),
                     list(zip(np.ndindex((100,)), 100 * [masked])))
        # 断言 `ndenumerate(c)` 的结果列表为空
        assert_equal(list(ndenumerate(c)), [])
        # 断言 `ndenumerate(c, compressed=False)` 的结果列表与预期列表相同
        assert_equal(list(ndenumerate(c, compressed=False)),
                     list(zip(np.ndindex((2, 3, 4)), 2 * 3 * 4 * [masked])))
    # 定义一个测试函数，用于测试混合掩码的ndenumerate功能
    def test_ndenumerate_mixedmasked(self):
        # 创建一个包含掩码的数组a，形状为(3, 4)，范围从0到11
        a = masked_array(np.arange(12).reshape((3, 4)),
                         mask=[[1, 1, 1, 1],
                               [1, 1, 0, 1],
                               [0, 0, 0, 0]])
        # 预期的迭代结果，包含一系列((行索引, 列索引), 值)的元组
        items = [((1, 2), 6),
                 ((2, 0), 8), ((2, 1), 9), ((2, 2), 10), ((2, 3), 11)]
        # 断言ndenumerate函数返回的结果与预期的items相等
        assert_equal(list(ndenumerate(a)), items)
        # 断言使用compressed=False参数时，ndenumerate函数返回的元素个数等于数组a的大小
        assert_equal(len(list(ndenumerate(a, compressed=False))), a.size)
        # 遍历使用compressed=False参数的ndenumerate函数的结果
        for coordinate, value in ndenumerate(a, compressed=False):
            # 断言遍历的每个坐标对应的值与数组a中对应位置的值相等
            assert_equal(a[coordinate], value)
class TestStack:

    def test_stack_1d(self):
        # 创建两个带掩码的一维数组
        a = masked_array([0, 1, 2], mask=[0, 1, 0])
        b = masked_array([9, 8, 7], mask=[1, 0, 0])

        # 在 axis=0 上堆叠数组 a 和 b
        c = stack([a, b], axis=0)
        # 断言堆叠后的数组形状为 (2, 3)
        assert_equal(c.shape, (2, 3))
        # 断言 a 的掩码与堆叠后的第一个元素的掩码相同
        assert_array_equal(a.mask, c[0].mask)
        # 断言 b 的掩码与堆叠后的第二个元素的掩码相同

        assert_array_equal(b.mask, c[1].mask)

        # 在垂直方向堆叠数组 a 和 b
        d = vstack([a, b])
        # 断言堆叠后的数据部分相同
        assert_array_equal(c.data, d.data)
        # 断言堆叠后的掩码部分相同
        assert_array_equal(c.mask, d.mask)

        # 在 axis=1 上堆叠数组 a 和 b
        c = stack([a, b], axis=1)
        # 断言堆叠后的数组形状为 (3, 2)
        assert_equal(c.shape, (3, 2))
        # 断言 a 的掩码与堆叠后第一列的掩码相同
        assert_array_equal(a.mask, c[:, 0].mask)
        # 断言 b 的掩码与堆叠后第二列的掩码相同

        assert_array_equal(b.mask, c[:, 1].mask)

    def test_stack_masks(self):
        # 创建两个带掩码的一维数组，其中 a 的所有元素都被掩码，b 没有掩码
        a = masked_array([0, 1, 2], mask=True)
        b = masked_array([9, 8, 7], mask=False)

        # 在 axis=0 上堆叠数组 a 和 b
        c = stack([a, b], axis=0)
        # 断言堆叠后的数组形状为 (2, 3)
        assert_equal(c.shape, (2, 3))
        # 断言 a 的掩码与堆叠后的第一个元素的掩码相同
        assert_array_equal(a.mask, c[0].mask)
        # 断言 b 的掩码与堆叠后的第二个元素的掩码相同

        assert_array_equal(b.mask, c[1].mask)

        # 在垂直方向堆叠数组 a 和 b
        d = vstack([a, b])
        # 断言堆叠后的数据部分相同
        assert_array_equal(c.data, d.data)
        # 断言堆叠后的掩码部分相同
        assert_array_equal(c.mask, d.mask)

        # 在 axis=1 上堆叠数组 a 和 b
        c = stack([a, b], axis=1)
        # 断言堆叠后的数组形状为 (3, 2)
        assert_equal(c.shape, (3, 2))
        # 断言 a 的掩码与堆叠后第一列的掩码相同
        assert_array_equal(a.mask, c[:, 0].mask)
        # 断言 b 的掩码与堆叠后第二列的掩码相同

        assert_array_equal(b.mask, c[:, 1].mask)

    def test_stack_nd(self):
        # 创建两个多维数组 a1 和 a2，每个元素都带有随机掩码
        shp = (3, 2)
        d1 = np.random.randint(0, 10, shp)
        d2 = np.random.randint(0, 10, shp)
        m1 = np.random.randint(0, 2, shp).astype(bool)
        m2 = np.random.randint(0, 2, shp).astype(bool)
        a1 = masked_array(d1, mask=m1)
        a2 = masked_array(d2, mask=m2)

        # 在 axis=0 上堆叠数组 a1 和 a2
        c = stack([a1, a2], axis=0)
        c_shp = (2,) + shp
        # 断言堆叠后的数组形状符合预期
        assert_equal(c.shape, c_shp)
        # 断言 a1 的掩码与堆叠后的第一个元素的掩码相同
        assert_array_equal(a1.mask, c[0].mask)
        # 断言 a2 的掩码与堆叠后的第二个元素的掩码相同

        assert_array_equal(a2.mask, c[1].mask)

        # 在 axis=-1 上堆叠数组 a1 和 a2
        c = stack([a1, a2], axis=-1)
        c_shp = shp + (2,)
        # 断言堆叠后的数组形状符合预期
        assert_equal(c.shape, c_shp)
        # 断言 a1 的掩码与堆叠后的第一个元素的掩码相同
        assert_array_equal(a1.mask, c[..., 0].mask)
        # 断言 a2 的掩码与堆叠后的第二个元素的掩码相同

        assert_array_equal(a2.mask, c[..., 1].mask)

        # 创建两个四维数组 a1 和 a2，每个元素都带有随机掩码
        shp = (3, 2, 4, 5,)
        d1 = np.random.randint(0, 10, shp)
        d2 = np.random.randint(0, 10, shp)
        m1 = np.random.randint(0, 2, shp).astype(bool)
        m2 = np.random.randint(0, 2, shp).astype(bool)
        a1 = masked_array(d1, mask=m1)
        a2 = masked_array(d2, mask=m2)

        # 在 axis=0 上堆叠数组 a1 和 a2
        c = stack([a1, a2], axis=0)
        c_shp = (2,) + shp
        # 断言堆叠后的数组形状符合预期
        assert_equal(c.shape, c_shp)
        # 断言 a1 的掩码与堆叠后的第一个元素的掩码相同
        assert_array_equal(a1.mask, c[0].mask)
        # 断言 a2 的掩码与堆叠后的第二个元素的掩码相同

        assert_array_equal(a2.mask, c[1].mask)

        # 在 axis=-1 上堆叠数组 a1 和 a2
        c = stack([a1, a2], axis=-1)
        c_shp = shp + (2,)
        # 断言堆叠后的数组形状符合预期
        assert_equal(c.shape, c_shp)
        # 断言 a1 的掩码与堆叠后的第一个元素的掩码相同
        assert_array_equal(a1.mask, c[..., 0].mask)
        # 断言 a2 的掩码与堆叠后的第二个元素的掩码相同

        assert_array_equal(a2.mask, c[..., 1].mask)
```