# `.\numpy\numpy\lib\tests\test_index_tricks.py`

```py
import pytest  # 导入 pytest 模块

import numpy as np  # 导入 numpy 库，并使用 np 别名
from numpy.testing import (  # 导入 numpy.testing 模块下的多个函数
    assert_, assert_equal, assert_array_equal, assert_almost_equal,
    assert_array_almost_equal, assert_raises, assert_raises_regex,
    )
from numpy.lib._index_tricks_impl import (  # 导入 numpy.lib._index_tricks_impl 模块下的多个函数
    mgrid, ogrid, ndenumerate, fill_diagonal, diag_indices, diag_indices_from,
    index_exp, ndindex, c_, r_, s_, ix_
    )


class TestRavelUnravelIndex:
    def test_basic(self):
        assert_equal(np.unravel_index(2, (2, 2)), (1, 0))  # 测试 np.unravel_index 函数的基本用法

        # 测试新的 shape 参数是否正常工作
        assert_equal(np.unravel_index(indices=2,
                                      shape=(2, 2)),
                                      (1, 0))

        # 测试处理无效的第二个关键字参数，包括旧名称 `dims`。
        with assert_raises(TypeError):  # 检查是否会抛出 TypeError 异常
            np.unravel_index(indices=2, hape=(2, 2))

        with assert_raises(TypeError):
            np.unravel_index(2, hape=(2, 2))

        with assert_raises(TypeError):
            np.unravel_index(254, ims=(17, 94))

        with assert_raises(TypeError):
            np.unravel_index(254, dims=(17, 94))

        assert_equal(np.ravel_multi_index((1, 0), (2, 2)), 2)  # 测试 np.ravel_multi_index 函数的基本用法
        assert_equal(np.unravel_index(254, (17, 94)), (2, 66))
        assert_equal(np.ravel_multi_index((2, 66), (17, 94)), 254)
        assert_raises(ValueError, np.unravel_index, -1, (2, 2))  # 检查是否会抛出 ValueError 异常
        assert_raises(TypeError, np.unravel_index, 0.5, (2, 2))
        assert_raises(ValueError, np.unravel_index, 4, (2, 2))
        assert_raises(ValueError, np.ravel_multi_index, (-3, 1), (2, 2))
        assert_raises(ValueError, np.ravel_multi_index, (2, 1), (2, 2))
        assert_raises(ValueError, np.ravel_multi_index, (0, -3), (2, 2))
        assert_raises(ValueError, np.ravel_multi_index, (0, 2), (2, 2))
        assert_raises(TypeError, np.ravel_multi_index, (0.1, 0.), (2, 2))

        assert_equal(np.unravel_index((2*3 + 1)*6 + 4, (4, 3, 6)), [2, 1, 4])  # 测试给定索引的解索引操作
        assert_equal(
            np.ravel_multi_index([2, 1, 4], (4, 3, 6)), (2*3 + 1)*6 + 4)

        arr = np.array([[3, 6, 6], [4, 5, 1]])
        assert_equal(np.ravel_multi_index(arr, (7, 6)), [22, 41, 37])  # 测试用数组作为索引的多维解索引操作
        assert_equal(
            np.ravel_multi_index(arr, (7, 6), order='F'), [31, 41, 13])
        assert_equal(
            np.ravel_multi_index(arr, (4, 6), mode='clip'), [22, 23, 19])
        assert_equal(np.ravel_multi_index(arr, (4, 4), mode=('clip', 'wrap')),
                     [12, 13, 13])
        assert_equal(np.ravel_multi_index((3, 1, 4, 1), (6, 7, 8, 9)), 1621)

        assert_equal(np.unravel_index(np.array([22, 41, 37]), (7, 6)),
                     [[3, 6, 6], [4, 5, 1]])  # 测试用数组解索引操作
        assert_equal(
            np.unravel_index(np.array([31, 41, 13]), (7, 6), order='F'),
            [[3, 6, 6], [4, 5, 1]])
        assert_equal(np.unravel_index(1621, (6, 7, 8, 9)), [3, 1, 4, 1])
    # 定义测试函数，用于测试空索引相关的异常情况
    def test_empty_indices(self):
        # 错误消息1，指示索引必须是整数，因为提供的空序列不是
        msg1 = 'indices must be integral: the provided empty sequence was'
        # 错误消息2，指示只有整数索引是允许的
        msg2 = 'only int indices permitted'
        
        # 断言：测试 np.unravel_index 函数对空列表索引的 TypeError 异常抛出
        assert_raises_regex(TypeError, msg1, np.unravel_index, [], (10, 3, 5))
        # 断言：测试 np.unravel_index 函数对空元组索引的 TypeError 异常抛出
        assert_raises_regex(TypeError, msg1, np.unravel_index, (), (10, 3, 5))
        # 断言：测试 np.unravel_index 函数对空 np.array 索引的 TypeError 异常抛出
        assert_raises_regex(TypeError, msg2, np.unravel_index, np.array([]),
                            (10, 3, 5))
        # 断言：测试 np.unravel_index 函数对空 np.array(dtype=int) 索引的返回值为空列表
        assert_equal(np.unravel_index(np.array([],dtype=int), (10, 3, 5)),
                     [[], [], []])
        
        # 断言：测试 np.ravel_multi_index 函数对空列表索引的 TypeError 异常抛出
        assert_raises_regex(TypeError, msg1, np.ravel_multi_index, ([], []),
                            (10, 3))
        # 断言：测试 np.ravel_multi_index 函数对包含非整数元素的列表索引的 TypeError 异常抛出
        assert_raises_regex(TypeError, msg1, np.ravel_multi_index, ([], ['abc']),
                            (10, 3))
        # 断言：测试 np.ravel_multi_index 函数对空 np.array 索引的 TypeError 异常抛出
        assert_raises_regex(TypeError, msg2, np.ravel_multi_index,
                    (np.array([]), np.array([])), (5, 3))
        # 断言：测试 np.ravel_multi_index 函数对空 np.array(dtype=int) 索引的返回值为空列表
        assert_equal(np.ravel_multi_index(
                (np.array([], dtype=int), np.array([], dtype=int)), (5, 3)), [])
        # 断言：测试 np.ravel_multi_index 函数对空 np.array([[], []], dtype=int) 索引的返回值为空列表
        assert_equal(np.ravel_multi_index(np.array([[], []], dtype=int),
                     (5, 3)), [])

    # 定义测试函数，用于测试大索引情况下的功能
    def test_big_indices(self):
        # 如果 np.intp 类型为 np.int64，则执行以下测试（用于处理 issue #7546）
        if np.intp == np.int64:
            # 定义包含大索引的数组 arr
            arr = ([1, 29], [3, 5], [3, 117], [19, 2],
                   [2379, 1284], [2, 2], [0, 1])
            # 断言：测试 np.ravel_multi_index 函数对大索引的返回值是否正确
            assert_equal(
                np.ravel_multi_index(arr, (41, 7, 120, 36, 2706, 8, 6)),
                [5627771580, 117259570957])

        # 测试 np.unravel_index 函数对大索引的 ValueError 异常抛出（用于处理 issue #9538）
        assert_raises(ValueError, np.unravel_index, 1, (2**32-1, 2**31+1))

        # 定义一个虚拟的数组 dummy_arr 用于测试数组大小检查
        dummy_arr = ([0],[0])
        # 计算 np.intp 类型的最大值的一半
        half_max = np.iinfo(np.intp).max // 2
        
        # 断言：测试 np.ravel_multi_index 函数对较大数组的正确处理
        assert_equal(
            np.ravel_multi_index(dummy_arr, (half_max, 2)), [0])
        # 断言：测试 np.ravel_multi_index 函数对超出较大数组范围的 ValueError 异常抛出
        assert_raises(ValueError,
            np.ravel_multi_index, dummy_arr, (half_max+1, 2))
        # 断言：测试 np.ravel_multi_index 函数在指定 'F'（Fortran）顺序时的处理
        assert_equal(
            np.ravel_multi_index(dummy_arr, (half_max, 2), order='F'), [0])
        # 断言：测试 np.ravel_multi_index 函数在指定 'F'（Fortran）顺序时的超出范围处理
        assert_raises(ValueError,
            np.ravel_multi_index, dummy_arr, (half_max+1, 2), order='F')
    def test_dtypes(self):
        # 测试不同的数据类型
        for dtype in [np.int16, np.uint16, np.int32,
                      np.uint32, np.int64, np.uint64]:
            # 创建包含不同数据类型的坐标数组
            coords = np.array(
                [[1, 0, 1, 2, 3, 4], [1, 6, 1, 3, 2, 0]], dtype=dtype)
            # 定义形状
            shape = (5, 8)
            # 计算坐标的线性索引
            uncoords = 8*coords[0]+coords[1]
            # 断言多维索引和线性索引的转换
            assert_equal(np.ravel_multi_index(coords, shape), uncoords)
            assert_equal(coords, np.unravel_index(uncoords, shape))
            # 使用列序优先（Fortran风格）计算线性索引
            uncoords = coords[0]+5*coords[1]
            assert_equal(
                np.ravel_multi_index(coords, shape, order='F'), uncoords)
            assert_equal(coords, np.unravel_index(uncoords, shape, order='F'))

            # 创建包含三个维度的坐标数组
            coords = np.array(
                [[1, 0, 1, 2, 3, 4], [1, 6, 1, 3, 2, 0], [1, 3, 1, 0, 9, 5]],
                dtype=dtype)
            # 定义形状
            shape = (5, 8, 10)
            # 计算坐标的线性索引
            uncoords = 10*(8*coords[0]+coords[1])+coords[2]
            # 断言多维索引和线性索引的转换
            assert_equal(np.ravel_multi_index(coords, shape), uncoords)
            assert_equal(coords, np.unravel_index(uncoords, shape))
            # 使用列序优先（Fortran风格）计算线性索引
            uncoords = coords[0]+5*(coords[1]+8*coords[2])
            assert_equal(
                np.ravel_multi_index(coords, shape, order='F'), uncoords)
            assert_equal(coords, np.unravel_index(uncoords, shape, order='F'))

    def test_clipmodes(self):
        # 测试裁剪模式
        assert_equal(
            np.ravel_multi_index([5, 1, -1, 2], (4, 3, 7, 12), mode='wrap'),
            np.ravel_multi_index([1, 1, 6, 2], (4, 3, 7, 12)))
        assert_equal(np.ravel_multi_index([5, 1, -1, 2], (4, 3, 7, 12),
                                          mode=(
                                              'wrap', 'raise', 'clip', 'raise')),
                     np.ravel_multi_index([1, 1, 0, 2], (4, 3, 7, 12)))
        assert_raises(
            ValueError, np.ravel_multi_index, [5, 1, -1, 2], (4, 3, 7, 12))

    def test_writeability(self):
        # 查看gh-7269问题
        x, y = np.unravel_index([1, 2, 3], (4, 5))
        assert_(x.flags.writeable)
        assert_(y.flags.writeable)

    def test_0d(self):
        # gh-580问题
        x = np.unravel_index(0, ())
        assert_equal(x, ())

        assert_raises_regex(ValueError, "0d array", np.unravel_index, [0], ())
        assert_raises_regex(
            ValueError, "out of bounds", np.unravel_index, [1], ())

    @pytest.mark.parametrize("mode", ["clip", "wrap", "raise"])
    def test_empty_array_ravel(self, mode):
        # 测试空数组的ravel操作
        res = np.ravel_multi_index(
                    np.zeros((3, 0), dtype=np.intp), (2, 1, 0), mode=mode)
        assert(res.shape == (0,))

        with assert_raises(ValueError):
            np.ravel_multi_index(
                    np.zeros((3, 1), dtype=np.intp), (2, 1, 0), mode=mode)
    def test_empty_array_unravel(self):
        # 使用 numpy 的 unravel_index 函数对一个空数组进行多维索引解析
        res = np.unravel_index(np.zeros(0, dtype=np.intp), (2, 1, 0))
        # 确保 res 是一个包含三个空数组的元组
        assert(len(res) == 3)
        # 确保 res 中每个数组的形状都是 (0,)
        assert(all(a.shape == (0,) for a in res))

        # 使用 assert_raises 检测 ValueError 异常
        with assert_raises(ValueError):
            # 尝试在维度为 (2, 1, 0) 的空数组中索引 [1]，应触发 ValueError 异常
            np.unravel_index([1], (2, 1, 0))
# 定义一个名为 TestGrid 的测试类
class TestGrid:
    
    # 测试基本用法
    def test_basic(self):
        # 创建等距网格 a，包含 10 个点，范围从 -1 到 1
        a = mgrid[-1:1:10j]
        # 创建等距网格 b，步长为 0.1，范围从 -1 到 1
        b = mgrid[-1:1:0.1]
        # 断言 a 的形状为 (10,)
        assert_(a.shape == (10,))
        # 断言 b 的形状为 (20,)
        assert_(b.shape == (20,))
        # 断言 a 的第一个元素为 -1
        assert_(a[0] == -1)
        # 断言 a 的最后一个元素接近于 1
        assert_almost_equal(a[-1], 1)
        # 断言 b 的第一个元素为 -1
        assert_(b[0] == -1)
        # 断言 b 第二个元素与第一个元素之间的差接近于 0.1，精确度为 11 位小数
        assert_almost_equal(b[1]-b[0], 0.1, 11)
        # 断言 b 的最后一个元素接近于 b 的第一个元素加上 19 乘以 0.1，精确度为 11 位小数
        assert_almost_equal(b[-1], b[0]+19*0.1, 11)
        # 断言 a 的第二个元素与第一个元素之间的差接近于 2.0/9.0，精确度为 11 位小数
        assert_almost_equal(a[1]-a[0], 2.0/9.0, 11)

    # 测试与 linspace 等效性
    def test_linspace_equivalence(self):
        # 使用 linspace 创建数组 y，并返回步长 st
        y, st = np.linspace(2, 10, retstep=True)
        # 断言 st 接近于 8/49.0，精确度为 13 位小数
        assert_almost_equal(st, 8/49.0)
        # 断言 y 与 mgrid[2:10:50j] 几乎相等，精确度为 13
        assert_array_almost_equal(y, mgrid[2:10:50j], 13)

    # 测试多维网格
    def test_nd(self):
        # 创建二维网格 c，包含 10x10 个点，范围分别为 (-1,1) 和 (-2,2)
        c = mgrid[-1:1:10j, -2:2:10j]
        # 创建二维网格 d，步长分别为 0.1 和 0.2，范围分别为 (-1,1) 和 (-2,2)
        d = mgrid[-1:1:0.1, -2:2:0.2]
        # 断言 c 的形状为 (2, 10, 10)
        assert_(c.shape == (2, 10, 10))
        # 断言 d 的形状为 (2, 20, 20)
        assert_(d.shape == (2, 20, 20))
        # 断言 c 的第一个子数组的第一行元素全为 -1
        assert_array_equal(c[0][0, :], -np.ones(10, 'd'))
        # 断言 c 的第二个子数组的第一列元素全为 -2
        assert_array_equal(c[1][:, 0], -2*np.ones(10, 'd'))
        # 断言 c 的第一个子数组的最后一行元素全为 1，精确度为 11 位小数
        assert_array_almost_equal(c[0][-1, :], np.ones(10, 'd'), 11)
        # 断言 c 的第二个子数组的最后一列元素全为 2，精确度为 11 位小数
        assert_array_almost_equal(c[1][:, -1], 2*np.ones(10, 'd'), 11)
        # 断言 d 的第一个子数组的第二行元素与第一行元素之间的差全为 0.1，精确度为 11 位小数
        assert_array_almost_equal(d[0, 1, :] - d[0, 0, :], 0.1*np.ones(20, 'd'), 11)
        # 断言 d 的第二个子数组的第二列元素与第一列元素之间的差全为 0.2，精确度为 11 位小数
        assert_array_almost_equal(d[1, :, 1] - d[1, :, 0], 0.2*np.ones(20, 'd'), 11)

    # 测试稀疏网格
    def test_sparse(self):
        # 创建完整网格 grid_full，包含 10x10 个点，范围分别为 (-1,1) 和 (-2,2)
        grid_full   = mgrid[-1:1:10j, -2:2:10j]
        # 创建稀疏网格 grid_sparse，包含 10x10 个点，范围分别为 (-1,1) 和 (-2,2)
        grid_sparse = ogrid[-1:1:10j, -2:2:10j]

        # 稀疏网格可以通过广播变得密集
        # 广播 grid_sparse 成为 grid_broadcast
        grid_broadcast = np.broadcast_arrays(*grid_sparse)
        # 遍历 grid_full 和 grid_broadcast，断言它们的元素相等
        for f, b in zip(grid_full, grid_broadcast):
            assert_equal(f, b)

    # 使用 pytest 参数化测试不同输入情况
    @pytest.mark.parametrize("start, stop, step, expected", [
        # 第一组参数化测试
        (None, 10, 10j, (200, 10)),
        # 第二组参数化测试
        (-10, 20, None, (1800, 30)),
        ])
    def test_mgrid_size_none_handling(self, start, stop, step, expected):
        # 回归测试 mgrid 处理 None 值的情况
        # 对 start 和 step 值的内部处理，目的是覆盖先前未测试过的代码路径
        # 创建网格 grid
        grid = mgrid[start:stop:step, start:stop:step]
        # 创建一个较小的网格 grid_small，以探索未测试过的代码路径之一
        grid_small = mgrid[start:stop:step]
        # 断言 grid 的大小等于预期的第一个元素
        assert_equal(grid.size, expected[0])
        # 断言 grid_small 的大小等于预期的第二个元素
        assert_equal(grid_small.size, expected[1])

    # 测试接受 np.float64 类型输入的情况
    def test_accepts_npfloating(self):
        # 回归测试 #16466
        # 使用 mgrid 创建浮点网格 grid64，范围从 0.1 到 0.33，步长为 0.1
        grid64 = mgrid[0.1:0.33:0.1, ]
        # 使用 mgrid 创建浮点网格 grid32，范围从 np.float32(0.1) 到 np.float32(0.33)，步长为 np.float32(0.1)
        grid32 = mgrid[np.float32(0.1):np.float32(0.33):np.float32(0.1), ]
        # 断言 grid64 与 grid32 几乎相等
        assert_array_almost_equal(grid64, grid32)
        # 断言 grid32 的数据类型为 np.float32
        assert grid32.dtype == np.float32
        # 断言 grid64 的数据类型为 np.float64

        # 对于单个切片，采用不同的代码路径
        grid64 = mgrid[0.1:0.33:0.1]
        grid32 = mgrid[np.float32(0.1):np.float32(0.33):np.float32(0.1)]
        # 断言 grid32 的数据类型为 np.float64
        assert_(grid32.dtype == np.float64)
        # 断言 grid64 与 grid32 几乎相等
        assert_array_almost_equal(grid64, grid32
    def test_accepts_longdouble(self):
        # 用于测试 np.longdouble 数据类型的接受性，是回归测试 #16945
        grid64 = mgrid[0.1:0.33:0.1, ]
        grid128 = mgrid[
            np.longdouble(0.1):np.longdouble(0.33):np.longdouble(0.1),
        ]
        # 断言 grid128 的数据类型为 np.longdouble
        assert_(grid128.dtype == np.longdouble)
        # 检查 grid64 和 grid128 数组几乎相等
        assert_array_almost_equal(grid64, grid128)

        grid128c_a = mgrid[0:np.longdouble(1):3.4j]
        grid128c_b = mgrid[0:np.longdouble(1):3.4j, ]
        # 断言 grid128c_a 和 grid128c_b 的数据类型都为 np.longdouble
        assert_(grid128c_a.dtype == grid128c_b.dtype == np.longdouble)
        # 断言 grid128c_a 和 grid128c_b[0] 的数组内容相等
        assert_array_equal(grid128c_a, grid128c_b[0])

        # 用于单个切片路径的不同代码路径
        grid64 = mgrid[0.1:0.33:0.1]
        grid128 = mgrid[
            np.longdouble(0.1):np.longdouble(0.33):np.longdouble(0.1)
        ]
        # 断言 grid128 的数据类型为 np.longdouble
        assert_(grid128.dtype == np.longdouble)
        # 检查 grid64 和 grid128 数组几乎相等
        assert_array_almost_equal(grid64, grid128)

    def test_accepts_npcomplexfloating(self):
        # 相关于 #16466 的测试
        assert_array_almost_equal(
            mgrid[0.1:0.3:3j, ], mgrid[0.1:0.3:np.complex64(3j), ]
        )

        # 用于单个切片路径的不同代码路径
        assert_array_almost_equal(
            mgrid[0.1:0.3:3j], mgrid[0.1:0.3:np.complex64(3j)]
        )

        # 相关于 #16945 的测试
        grid64_a = mgrid[0.1:0.3:3.3j]
        grid64_b = mgrid[0.1:0.3:3.3j, ][0]
        # 断言 grid64_a 和 grid64_b 的数据类型都为 np.float64
        assert_(grid64_a.dtype == grid64_b.dtype == np.float64)
        # 断言 grid64_a 和 grid64_b 的数组内容相等
        assert_array_equal(grid64_a, grid64_b)

        grid128_a = mgrid[0.1:0.3:np.clongdouble(3.3j)]
        grid128_b = mgrid[0.1:0.3:np.clongdouble(3.3j), ][0]
        # 断言 grid128_a 和 grid128_b 的数据类型都为 np.longdouble
        assert_(grid128_a.dtype == grid128_b.dtype == np.longdouble)
        # 断言 grid64_a 和 grid64_b 的数组内容相等（这里应该是 grid64_a 和 grid128_b，而不是 grid64_a 和 grid64_b）
        assert_array_equal(grid64_a, grid128_b)
class TestConcatenator:
    def test_1d(self):
        # 测试 r_[] 函数对 1 维数组的拼接
        assert_array_equal(r_[1, 2, 3, 4, 5, 6], np.array([1, 2, 3, 4, 5, 6]))
        b = np.ones(5)
        # 使用 r_[] 函数拼接包含数组 b、0、0、数组 b 的新数组 c
        c = r_[b, 0, 0, b]
        assert_array_equal(c, [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1])

    def test_mixed_type(self):
        # 测试 r_[] 函数对混合类型数组的拼接
        g = r_[10.1, 1:10]
        assert_(g.dtype == 'f8')

    def test_more_mixed_type(self):
        # 测试 r_[] 函数对更复杂的混合类型数组的拼接
        g = r_[-10.1, np.array([1]), np.array([2, 3, 4]), 10.0]
        assert_(g.dtype == 'f8')

    def test_complex_step(self):
        # 对 #12262 的回归测试
        g = r_[0:36:100j]
        assert_(g.shape == (100,))

        # 相关于 #16466 的测试
        g = r_[0:36:np.complex64(100j)]
        assert_(g.shape == (100,))

    def test_2d(self):
        b = np.random.rand(5, 5)
        c = np.random.rand(5, 5)
        d = r_['1', b, c]  # 在列方向上追加
        assert_(d.shape == (5, 10))
        assert_array_equal(d[:, :5], b)
        assert_array_equal(d[:, 5:], c)
        d = r_[b, c]
        assert_(d.shape == (10, 5))
        assert_array_equal(d[:5, :], b)
        assert_array_equal(d[5:, :], c)

    def test_0d(self):
        # 测试 r_[] 函数对 0 维数组的拼接
        assert_equal(r_[0, np.array(1), 2], [0, 1, 2])
        assert_equal(r_[[0, 1, 2], np.array(3)], [0, 1, 2, 3])
        assert_equal(r_[np.array(0), [1, 2, 3]], [0, 1, 2, 3])


class TestNdenumerate:
    def test_basic(self):
        a = np.array([[1, 2], [3, 4]])
        # 测试 ndenumerate 函数的基本功能
        assert_equal(list(ndenumerate(a)),
                     [((0, 0), 1), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)])


class TestIndexExpression:
    def test_regression_1(self):
        # 对 #1196 的回归测试
        a = np.arange(2)
        assert_equal(a[:-1], a[s_[:-1]])
        assert_equal(a[:-1], a[index_exp[:-1]])

    def test_simple_1(self):
        a = np.random.rand(4, 5, 6)

        assert_equal(a[:, :3, [1, 2]], a[index_exp[:, :3, [1, 2]]])
        assert_equal(a[:, :3, [1, 2]], a[s_[:, :3, [1, 2]]])


class TestIx_:
    def test_regression_1(self):
        # 测试空的未类型化输入是否创建了索引类型的输出，gh-5804
        a, = np.ix_(range(0))
        assert_equal(a.dtype, np.intp)

        a, = np.ix_([])
        assert_equal(a.dtype, np.intp)

        # 如果指定了类型，则不要改变它
        a, = np.ix_(np.array([], dtype=np.float32))
        assert_equal(a.dtype, np.float32)

    def test_shape_and_dtype(self):
        sizes = (4, 5, 3, 2)
        # 测试列表和数组的情况
        for func in (range, np.arange):
            arrays = np.ix_(*[func(sz) for sz in sizes])
            for k, (a, sz) in enumerate(zip(arrays, sizes)):
                assert_equal(a.shape[k], sz)
                assert_(all(sh == 1 for j, sh in enumerate(a.shape) if j != k))
                assert_(np.issubdtype(a.dtype, np.integer))

    def test_bool(self):
        bool_a = [True, False, True, True]
        int_a, = np.nonzero(bool_a)
        assert_equal(np.ix_(bool_a)[0], int_a)
    # 定义一个测试函数，用于测试处理仅限于一维索引的情况
    def test_1d_only(self):
        # 创建一个二维列表作为输入索引
        idx2d = [[1, 2, 3], [4, 5, 6]]
        # 断言预期会引发 ValueError 异常，调用 np.ix_ 函数时传入二维索引将引发异常
        assert_raises(ValueError, np.ix_, idx2d)

    # 定义另一个测试函数，用于测试重复输入的情况
    def test_repeated_input(self):
        # 定义一个向量的长度
        length_of_vector = 5
        # 使用 np.arange 创建一个数组 x，其元素为 0 到 length_of_vector-1
        x = np.arange(length_of_vector)
        # 调用 np.ix_ 函数，对向量 x 进行索引操作，并将结果存储在 out 变量中
        out = ix_(x, x)
        # 断言 out 变量的第一个元素的形状为 (length_of_vector, 1)
        assert_equal(out[0].shape, (length_of_vector, 1))
        # 断言 out 变量的第二个元素的形状为 (1, length_of_vector)
        assert_equal(out[1].shape, (1, length_of_vector))
        # 检查输入向量 x 的形状未被修改
        assert_equal(x.shape, (length_of_vector,))
def test_c_():
    # 使用 c_ 函数创建一个数组，该数组包含多个输入数组的串联
    a = c_[np.array([[1, 2, 3]]), 0, 0, np.array([[4, 5, 6]])]
    # 断言数组 a 应该等于预期的串联结果
    assert_equal(a, [[1, 2, 3, 0, 0, 4, 5, 6]])


class TestFillDiagonal:
    def test_basic(self):
        # 创建一个 3x3 的整数数组，全部元素初始化为 0
        a = np.zeros((3, 3), int)
        # 调用 fill_diagonal 函数，在数组 a 中以值 5 填充对角线元素
        fill_diagonal(a, 5)
        # 断言数组 a 应该等于预期的对角线填充结果
        assert_array_equal(
            a, np.array([[5, 0, 0],
                         [0, 5, 0],
                         [0, 0, 5]])
            )

    def test_tall_matrix(self):
        # 创建一个 10x3 的整数数组，全部元素初始化为 0
        a = np.zeros((10, 3), int)
        # 调用 fill_diagonal 函数，在数组 a 中以值 5 填充对角线元素
        fill_diagonal(a, 5)
        # 断言数组 a 应该等于预期的对角线填充结果
        assert_array_equal(
            a, np.array([[5, 0, 0],
                         [0, 5, 0],
                         [0, 0, 5],
                         [0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]])
            )

    def test_tall_matrix_wrap(self):
        # 创建一个 10x3 的整数数组，全部元素初始化为 0
        a = np.zeros((10, 3), int)
        # 调用 fill_diagonal 函数，在数组 a 中以值 5 填充对角线元素，同时进行循环填充
        fill_diagonal(a, 5, True)
        # 断言数组 a 应该等于预期的对角线填充结果
        assert_array_equal(
            a, np.array([[5, 0, 0],
                         [0, 5, 0],
                         [0, 0, 5],
                         [0, 0, 0],
                         [5, 0, 0],
                         [0, 5, 0],
                         [0, 0, 5],
                         [0, 0, 0],
                         [5, 0, 0],
                         [0, 5, 0]])
            )

    def test_wide_matrix(self):
        # 创建一个 3x10 的整数数组，全部元素初始化为 0
        a = np.zeros((3, 10), int)
        # 调用 fill_diagonal 函数，在数组 a 中以值 5 填充对角线元素
        fill_diagonal(a, 5)
        # 断言数组 a 应该等于预期的对角线填充结果
        assert_array_equal(
            a, np.array([[5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 5, 0, 0, 0, 0, 0, 0, 0]])
            )

    def test_operate_4d_array(self):
        # 创建一个 3x3x3x3 的整数数组，全部元素初始化为 0
        a = np.zeros((3, 3, 3, 3), int)
        # 调用 fill_diagonal 函数，在数组 a 中以值 4 填充对角线元素
        fill_diagonal(a, 4)
        # 创建一个数组 i，包含 [0, 1, 2]
        i = np.array([0, 1, 2])
        # 断言非零元素的索引应该等于预期的数组索引
        assert_equal(np.where(a != 0), (i, i, i, i))

    def test_low_dim_handling(self):
        # 创建一个长度为 3 的整数数组，全部元素初始化为 0
        # 期望填充对角线操作引发 ValueError 异常
        a = np.zeros(3, int)
        with assert_raises_regex(ValueError, "at least 2-d"):
            # 尝试使用 fill_diagonal 函数填充对角线，预期引发异常
            fill_diagonal(a, 5)

    def test_hetero_shape_handling(self):
        # 创建一个 3x3x7x3 的整数数组，全部元素初始化为 0
        # 期望填充对角线操作引发 ValueError 异常
        a = np.zeros((3,3,7,3), int)
        with assert_raises_regex(ValueError, "equal length"):
            # 尝试使用 fill_diagonal 函数填充对角线，预期引发异常
            fill_diagonal(a, 2)


def test_diag_indices():
    # 调用 diag_indices 函数生成一个包含对角线索引的元组
    di = diag_indices(4)
    # 创建一个 4x4 的整数数组
    a = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    # 使用 diag_indices 返回的索引 di 来设置数组 a 的对角线元素为 100
    a[di] = 100
    # 断言数组 a 应该等于预期的对角线填充结果
    assert_array_equal(
        a, np.array([[100, 2, 3, 4],
                     [5, 100, 7, 8],
                     [9, 10, 100, 12],
                     [13, 14, 15, 100]])
        )

    # 使用 diag_indices 函数生成一个包含对角线索引的元组
    d3 = diag_indices(2, 3)

    # 创建一个 2x2x2 的整数数组，全部元素初始化为 0
    # 使用 diag_indices 返回的索引 d3 来设置数组对角线元素为 1
    a = np.zeros((2, 2, 2), int)
    a[d3] = 1
    # 使用 assert_array_equal 函数对比两个数组是否相等
    assert_array_equal(
        # 第一个数组 a
        a, np.array([
            # 第一个元素是三维数组
            [[1, 0],  # 第一层是二维数组，包含 [1, 0]
             [0, 0]],  # 第二层是二维数组，包含 [0, 0]
            [[0, 0],  # 第一层是二维数组，包含 [0, 0]
             [0, 1]]   # 第二层是二维数组，包含 [0, 1]
        ])
    )
class TestDiagIndicesFrom:

    # 定义测试函数，验证 diag_indices_from 函数的正确性
    def test_diag_indices_from(self):
        # 创建一个 4x4 的随机数组
        x = np.random.random((4, 4))
        # 调用 diag_indices_from 函数获取主对角线索引
        r, c = diag_indices_from(x)
        # 断言主对角线的行索引与 np.arange(4) 相等
        assert_array_equal(r, np.arange(4))
        # 断言主对角线的列索引与 np.arange(4) 相等
        assert_array_equal(c, np.arange(4))

    # 测试当输入数组维度过小时是否会抛出 ValueError 异常
    def test_error_small_input(self):
        # 创建一个长度为 7 的全 1 数组
        x = np.ones(7)
        # 使用 assert_raises_regex 断言捕获到 ValueError 异常，并验证异常信息包含 "at least 2-d"
        with assert_raises_regex(ValueError, "at least 2-d"):
            diag_indices_from(x)

    # 测试当输入数组形状不匹配时是否会抛出 ValueError 异常
    def test_error_shape_mismatch(self):
        # 创建一个形状为 (3, 3, 2, 3) 的全 0 整数数组
        x = np.zeros((3, 3, 2, 3), int)
        # 使用 assert_raises_regex 断言捕获到 ValueError 异常，并验证异常信息包含 "equal length"
        with assert_raises_regex(ValueError, "equal length"):
            diag_indices_from(x)


# 定义测试 ndindex 函数的各种用例
def test_ndindex():
    # 获取 (1, 2, 3) 维度的所有索引组合，并转换为列表
    x = list(ndindex(1, 2, 3))
    # 通过 ndenumerate(np.zeros((1, 2, 3))) 获取期望的索引列表
    expected = [ix for ix, e in ndenumerate(np.zeros((1, 2, 3)))]
    # 断言 ndindex 的输出与期望的索引列表相等
    assert_array_equal(x, expected)

    # 以元组 (1, 2, 3) 的形式调用 ndindex，验证输出是否与之前相同
    x = list(ndindex((1, 2, 3)))
    assert_array_equal(x, expected)

    # 测试使用标量和元组调用 ndindex
    x = list(ndindex((3,)))
    assert_array_equal(x, list(ndindex(3)))

    # 测试不带任何参数调用 ndindex，确保返回值为单个空元组
    x = list(ndindex())
    assert_equal(x, [()])

    # 再次测试带有空元组作为参数的情况
    x = list(ndindex(()))
    assert_equal(x, [()])

    # 确保大小为 0 的 ndindex 正常工作
    x = list(ndindex(*[0]))
    assert_equal(x, [])
```