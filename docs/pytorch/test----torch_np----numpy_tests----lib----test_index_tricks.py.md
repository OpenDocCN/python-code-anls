# `.\pytorch\test\torch_np\numpy_tests\lib\test_index_tricks.py`

```
# Owner(s): ["module: dynamo"]  # 标记代码所有者为模块 dynamo

import functools  # 导入 functools 模块

from unittest import expectedFailure as xfail, skipIf  # 从 unittest 模块导入 expectedFailure 别名为 xfail，以及 skipIf

from pytest import raises as assert_raises  # 从 pytest 模块导入 raises 别名为 assert_raises
# from pytest import assert_raises_regex,  # 省略了 assert_raises_regex 的导入

from torch.testing._internal.common_utils import (  # 从 torch.testing._internal.common_utils 导入以下函数和变量
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo,
)

skip = functools.partial(skipIf, True)  # 使用 functools.partial 创建 skip 函数，用于条件为 True 时跳过测试


# 如果需要跟踪这些代码，应该使用 NumPy
# 如果在 eager 模式下测试，则使用 torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np  # 导入 numpy 库并使用 np 别名
    from numpy import (  # 从 numpy 中导入特定函数和对象
        diag_indices,
        diag_indices_from,
        fill_diagonal,
        index_exp,
        s_,
    )
    from numpy.testing import (  # 从 numpy.testing 导入断言函数
        assert_,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
        assert_raises_regex,
    )
else:
    import torch._numpy as np  # 导入 torch._numpy 库并使用 np 别名
    from torch._numpy import (  # 从 torch._numpy 导入特定函数和对象
        diag_indices,
        diag_indices_from,
        fill_diagonal,
        index_exp,
        s_,
    )
    from torch._numpy.testing import (  # 从 torch._numpy.testing 导入断言函数
        assert_,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
    )


@xpassIfTorchDynamo  # 标记为在 Torch Dynamo 下通过测试
@instantiate_parametrized_tests  # 实例化参数化测试
class TestRavelUnravelIndex(TestCase):  # 定义测试类 TestRavelUnravelIndex，继承自 TestCase 类
    # 定义测试方法 `test_basic`，用于测试 `np.unravel_index` 和 `np.ravel_multi_index` 函数
    def test_basic(self):
        # 断言 np.unravel_index(2, (2, 2)) 返回结果为 (1, 0)
        assert_equal(np.unravel_index(2, (2, 2)), (1, 0))

        # 测试带有新形状参数的 np.unravel_index(indices=2, shape=(2, 2))
        assert_equal(np.unravel_index(indices=2, shape=(2, 2)), (1, 0))

        # 测试处理无效的第二关键字参数，包括旧名称 `dims`
        with assert_raises(TypeError):
            np.unravel_index(indices=2, hape=(2, 2))

        with assert_raises(TypeError):
            np.unravel_index(2, hape=(2, 2))

        with assert_raises(TypeError):
            np.unravel_index(254, ims=(17, 94))

        with assert_raises(TypeError):
            np.unravel_index(254, dims=(17, 94))

        # 断言 np.ravel_multi_index((1, 0), (2, 2)) 返回结果为 2
        assert_equal(np.ravel_multi_index((1, 0), (2, 2)), 2)

        # 断言 np.unravel_index(254, (17, 94)) 返回结果为 (2, 66)
        assert_equal(np.unravel_index(254, (17, 94)), (2, 66))

        # 断言 np.ravel_multi_index((2, 66), (17, 94)) 返回结果为 254
        assert_equal(np.ravel_multi_index((2, 66), (17, 94)), 254)

        # 断言 np.unravel_index(-1, (2, 2)) 引发 ValueError 异常
        assert_raises(ValueError, np.unravel_index, -1, (2, 2))

        # 断言 np.unravel_index(0.5, (2, 2)) 引发 TypeError 异常
        assert_raises(TypeError, np.unravel_index, 0.5, (2, 2))

        # 断言 np.unravel_index(4, (2, 2)) 引发 ValueError 异常
        assert_raises(ValueError, np.unravel_index, 4, (2, 2))

        # 断言 np.ravel_multi_index((-3, 1), (2, 2)) 引发 ValueError 异常
        assert_raises(ValueError, np.ravel_multi_index, (-3, 1), (2, 2))

        # 断言 np.ravel_multi_index((2, 1), (2, 2)) 引发 ValueError 异常
        assert_raises(ValueError, np.ravel_multi_index, (2, 1), (2, 2))

        # 断言 np.ravel_multi_index((0, -3), (2, 2)) 引发 ValueError 异常
        assert_raises(ValueError, np.ravel_multi_index, (0, -3), (2, 2))

        # 断言 np.ravel_multi_index((0, 2), (2, 2)) 引发 ValueError 异常
        assert_raises(ValueError, np.ravel_multi_index, (0, 2), (2, 2))

        # 断言 np.ravel_multi_index((0.1, 0.0), (2, 2)) 引发 TypeError 异常
        assert_raises(TypeError, np.ravel_multi_index, (0.1, 0.0), (2, 2))

        # 断言 np.unravel_index((2 * 3 + 1) * 6 + 4, (4, 3, 6)) 返回结果为 [2, 1, 4]
        assert_equal(np.unravel_index((2 * 3 + 1) * 6 + 4, (4, 3, 6)), [2, 1, 4])

        # 断言 np.ravel_multi_index([2, 1, 4], (4, 3, 6)) 返回结果为 (2 * 3 + 1) * 6 + 4
        assert_equal(np.ravel_multi_index([2, 1, 4], (4, 3, 6)), (2 * 3 + 1) * 6 + 4)

        arr = np.array([[3, 6, 6], [4, 5, 1]])
        # 断言 np.ravel_multi_index(arr, (7, 6)) 返回结果为 [22, 41, 37]
        assert_equal(np.ravel_multi_index(arr, (7, 6)), [22, 41, 37])

        # 断言 np.ravel_multi_index(arr, (7, 6), order="F") 返回结果为 [31, 41, 13]
        assert_equal(np.ravel_multi_index(arr, (7, 6), order="F"), [31, 41, 13])

        # 断言 np.ravel_multi_index(arr, (4, 6), mode="clip") 返回结果为 [22, 23, 19]
        assert_equal(np.ravel_multi_index(arr, (4, 6), mode="clip"), [22, 23, 19])

        # 断言 np.ravel_multi_index(arr, (4, 4), mode=("clip", "wrap")) 返回结果为 [12, 13, 13]
        assert_equal(np.ravel_multi_index(arr, (4, 4), mode=("clip", "wrap")), [12, 13, 13])

        # 断言 np.ravel_multi_index((3, 1, 4, 1), (6, 7, 8, 9)) 返回结果为 1621
        assert_equal(np.ravel_multi_index((3, 1, 4, 1), (6, 7, 8, 9)), 1621)

        # 断言 np.unravel_index(np.array([22, 41, 37]), (7, 6)) 返回结果为 [[3, 6, 6], [4, 5, 1]]
        assert_equal(np.unravel_index(np.array([22, 41, 37]), (7, 6)), [[3, 6, 6], [4, 5, 1]])

        # 断言 np.unravel_index(np.array([31, 41, 13]), (7, 6), order="F") 返回结果为 [[3, 6, 6], [4, 5, 1]]
        assert_equal(np.unravel_index(np.array([31, 41, 13]), (7, 6), order="F"), [[3, 6, 6], [4, 5, 1]])

        # 断言 np.unravel_index(1621, (6, 7, 8, 9)) 返回结果为 [3, 1, 4, 1]
        assert_equal(np.unravel_index(1621, (6, 7, 8, 9)), [3, 1, 4, 1])
    # 定义测试方法，验证空索引输入时的异常情况
    def test_empty_indices(self):
        # 错误消息定义：索引必须是整数，提供了空序列
        msg1 = "indices must be integral: the provided empty sequence was"
        # 错误消息定义：仅允许整数索引
        msg2 = "only int indices permitted"
        # 验证对空列表调用 np.unravel_index 时是否引发 TypeError 异常，并检查错误消息
        assert_raises_regex(TypeError, msg1, np.unravel_index, [], (10, 3, 5))
        # 验证对空元组调用 np.unravel_index 时是否引发 TypeError 异常，并检查错误消息
        assert_raises_regex(TypeError, msg1, np.unravel_index, (), (10, 3, 5))
        # 验证对空 NumPy 数组调用 np.unravel_index 时是否引发 TypeError 异常，并检查错误消息
        assert_raises_regex(TypeError, msg2, np.unravel_index, np.array([]), (10, 3, 5))
        # 验证对空的整数 NumPy 数组调用 np.unravel_index 是否返回正确的空数组
        assert_equal(
            np.unravel_index(np.array([], dtype=int), (10, 3, 5)), [[], [], []]
        )
        # 验证对空列表调用 np.ravel_multi_index 时是否引发 TypeError 异常，并检查错误消息
        assert_raises_regex(TypeError, msg1, np.ravel_multi_index, ([], []), (10, 3))
        # 验证对包含非整数元素的列表调用 np.ravel_multi_index 是否引发 TypeError 异常，并检查错误消息
        assert_raises_regex(
            TypeError, msg1, np.ravel_multi_index, ([], ["abc"]), (10, 3)
        )
        # 验证对包含空的整数 NumPy 数组调用 np.ravel_multi_index 是否返回正确的空数组
        assert_raises_regex(
            TypeError, msg2, np.ravel_multi_index, (np.array([]), np.array([])), (5, 3)
        )
        # 验证对包含空的整数 NumPy 数组调用 np.ravel_multi_index 是否返回正确的空数组
        assert_equal(
            np.ravel_multi_index(
                (np.array([], dtype=int), np.array([], dtype=int)), (5, 3)
            ),
            [],
        )
        # 验证对包含空的整数 NumPy 二维数组调用 np.ravel_multi_index 是否返回正确的空数组
        assert_equal(np.ravel_multi_index(np.array([[], []], dtype=int), (5, 3)), [])

    # 定义测试方法，验证大索引情况下的功能
    def test_big_indices(self):
        # 如果整数类型是 int64，则执行下面的测试
        if np.intp == np.int64:
            # 定义一个大索引的数组，并验证 np.ravel_multi_index 返回的结果是否正确
            arr = ([1, 29], [3, 5], [3, 117], [19, 2], [2379, 1284], [2, 2], [0, 1])
            assert_equal(
                np.ravel_multi_index(arr, (41, 7, 120, 36, 2706, 8, 6)),
                [5627771580, 117259570957],
            )

        # 验证对大索引情况下 np.unravel_index 是否引发 ValueError 异常
        assert_raises(ValueError, np.unravel_index, 1, (2**32 - 1, 2**31 + 1))

        # 验证对超大数组进行溢出检查是否能正常处理，并验证错误消息
        dummy_arr = ([0], [0])
        half_max = np.iinfo(np.intp).max // 2
        assert_equal(np.ravel_multi_index(dummy_arr, (half_max, 2)), [0])
        assert_raises(ValueError, np.ravel_multi_index, dummy_arr, (half_max + 1, 2))
        assert_equal(np.ravel_multi_index(dummy_arr, (half_max, 2), order="F"), [0])
        assert_raises(
            ValueError, np.ravel_multi_index, dummy_arr, (half_max + 1, 2), order="F"
        )
    # 测试不同的数据类型
    def test_dtypes(self):
        # 使用不同的数据类型进行测试
        for dtype in [np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
            # 创建一个二维数组，指定数据类型为当前循环的 dtype
            coords = np.array([[1, 0, 1, 2, 3, 4], [1, 6, 1, 3, 2, 0]], dtype=dtype)
            # 定义数组的形状
            shape = (5, 8)
            # 根据公式计算一维坐标
            uncoords = 8 * coords[0] + coords[1]
            # 断言多维索引与计算的一维坐标相等
            assert_equal(np.ravel_multi_index(coords, shape), uncoords)
            # 断言多维索引反向操作与原始坐标相等
            assert_equal(coords, np.unravel_index(uncoords, shape))
            # 使用另一种公式计算一维坐标
            uncoords = coords[0] + 5 * coords[1]
            # 断言按指定顺序（Fortran）的多维索引与计算的一维坐标相等
            assert_equal(np.ravel_multi_index(coords, shape, order="F"), uncoords)
            # 断言按指定顺序（Fortran）的多维索引反向操作与原始坐标相等
            assert_equal(coords, np.unravel_index(uncoords, shape, order="F"))

            # 使用三维数组进行测试，指定数据类型为当前循环的 dtype
            coords = np.array(
                [[1, 0, 1, 2, 3, 4], [1, 6, 1, 3, 2, 0], [1, 3, 1, 0, 9, 5]],
                dtype=dtype,
            )
            # 定义数组的形状
            shape = (5, 8, 10)
            # 根据公式计算一维坐标
            uncoords = 10 * (8 * coords[0] + coords[1]) + coords[2]
            # 断言多维索引与计算的一维坐标相等
            assert_equal(np.ravel_multi_index(coords, shape), uncoords)
            # 断言多维索引反向操作与原始坐标相等
            assert_equal(coords, np.unravel_index(uncoords, shape))
            # 使用另一种公式计算一维坐标
            uncoords = coords[0] + 5 * (coords[1] + 8 * coords[2])
            # 断言按指定顺序（Fortran）的多维索引与计算的一维坐标相等
            assert_equal(np.ravel_multi_index(coords, shape, order="F"), uncoords)
            # 断言按指定顺序（Fortran）的多维索引反向操作与原始坐标相等
            assert_equal(coords, np.unravel_index(uncoords, shape, order="F"))

    # 测试多维索引的裁剪模式
    def test_clipmodes(self):
        # 使用裁剪模式进行测试
        assert_equal(
            np.ravel_multi_index([5, 1, -1, 2], (4, 3, 7, 12), mode="wrap"),
            np.ravel_multi_index([1, 1, 6, 2], (4, 3, 7, 12)),
        )
        assert_equal(
            np.ravel_multi_index(
                [5, 1, -1, 2], (4, 3, 7, 12), mode=("wrap", "raise", "clip", "raise")
            ),
            np.ravel_multi_index([1, 1, 0, 2], (4, 3, 7, 12)),
        )
        assert_raises(ValueError, np.ravel_multi_index, [5, 1, -1, 2], (4, 3, 7, 12))

    # 测试可写性（见 GitHub issue #7269）
    def test_writeability(self):
        # 检查 np.unravel_index 返回的数组是否可写
        x, y = np.unravel_index([1, 2, 3], (4, 5))
        assert_(x.flags.writeable)
        assert_(y.flags.writeable)

    # 测试零维数组的处理（见 GitHub issue #580）
    def test_0d(self):
        # 使用零维数组进行测试
        x = np.unravel_index(0, ())
        assert_equal(x, ())

        # 断言对于零维数组，np.unravel_index 抛出特定异常
        assert_raises_regex(ValueError, "0d array", np.unravel_index, [0], ())
        # 断言对于超出边界的索引，np.unravel_index 抛出特定异常
        assert_raises_regex(ValueError, "out of bounds", np.unravel_index, [1], ())

    # 使用不同的裁剪模式参数化测试空数组的多维索引（见参数化部分）
    @parametrize("mode", ["clip", "wrap", "raise"])
    def test_empty_array_ravel(self, mode):
        # 对空数组进行多维索引，使用指定的裁剪模式
        res = np.ravel_multi_index(
            np.zeros((3, 0), dtype=np.intp), (2, 1, 0), mode=mode
        )
        # 断言结果数组的形状为空
        assert res.shape == (0,)

        # 断言对于非空数组，使用指定的裁剪模式，np.ravel_multi_index 抛出特定异常
        with assert_raises(ValueError):
            np.ravel_multi_index(np.zeros((3, 1), dtype=np.intp), (2, 1, 0), mode=mode)

    # 测试空数组的多维索引反向操作
    def test_empty_array_unravel(self):
        # 对空数组进行多维索引反向操作
        res = np.unravel_index(np.zeros(0, dtype=np.intp), (2, 1, 0))
        # 断言结果是包含三个空数组的元组
        assert len(res) == 3
        assert all(a.shape == (0,) for a in res)

        # 断言对于非空数组，np.unravel_index 抛出特定异常
        with assert_raises(ValueError):
            np.unravel_index([1], (2, 1, 0))
@xfail  # (reason="mgrid not implemented")
@instantiate_parametrized_tests
class TestGrid(TestCase):
    # 测试基本的 mgrid 功能
    def test_basic(self):
        # 创建一个等间隔网格 a
        a = mgrid[-1:1:10j]
        # 创建一个等间隔网格 b
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
        # 断言 b 的第二个元素减去第一个元素接近于 0.1，精度为 11 位小数
        assert_almost_equal(b[1] - b[0], 0.1, 11)
        # 断言 b 的最后一个元素接近于 b 的第一个元素加上 19*0.1，精度为 11 位小数
        assert_almost_equal(b[-1], b[0] + 19 * 0.1, 11)
        # 断言 a 的第二个元素减去第一个元素接近于 2.0/9.0，精度为 11 位小数
        assert_almost_equal(a[1] - a[0], 2.0 / 9.0, 11)

    @xfail  # (reason="retstep not implemented")
    # 测试与 linspace 等价性的 mgrid 功能，目前预期不通过，因为 retstep 功能未实现
    def test_linspace_equivalence(self):
        # 调用 linspace 函数，获取返回的数值和步长
        y, st = np.linspace(2, 10, retstep=True)
        # 断言步长 st 接近于 8 / 49.0
        assert_almost_equal(st, 8 / 49.0)
        # 断言 y 数组与 mgrid[2:10:50j] 数组近似相等，精度为 13 位小数
        assert_array_almost_equal(y, mgrid[2:10:50j], 13)

    # 测试多维 mgrid 功能
    def test_nd(self):
        # 创建一个二维等间隔网格 c
        c = mgrid[-1:1:10j, -2:2:10j]
        # 创建一个二维等间隔网格 d
        d = mgrid[-1:1:0.1, -2:2:0.2]
        # 断言 c 的形状为 (2, 10, 10)
        assert_(c.shape == (2, 10, 10))
        # 断言 d 的形状为 (2, 20, 20)
        assert_(d.shape == (2, 20, 20))
        # 断言 c 的第一个维度的第一行为 -1 的数组
        assert_array_equal(c[0][0, :], -np.ones(10, "d"))
        # 断言 c 的第二个维度的第一列为 -2 的数组
        assert_array_equal(c[1][:, 0], -2 * np.ones(10, "d"))
        # 断言 c 的第一个维度的最后一行接近于全为 1 的数组，精度为 11 位小数
        assert_array_almost_equal(c[0][-1, :], np.ones(10, "d"), 11)
        # 断言 c 的第二个维度的最后一列接近于全为 2 的数组，精度为 11 位小数
        assert_array_almost_equal(c[1][:, -1], 2 * np.ones(10, "d"), 11)
        # 断言 d 的第一个维度的第二行减去第一行接近于全为 0.1 的数组，精度为 11 位小数
        assert_array_almost_equal(d[0, 1, :] - d[0, 0, :], 0.1 * np.ones(20, "d"), 11)
        # 断言 d 的第二个维度的第二列减去第一列接近于全为 0.2 的数组，精度为 11 位小数
        assert_array_almost_equal(d[1, :, 1] - d[1, :, 0], 0.2 * np.ones(20, "d"), 11)

    # 测试稀疏网格功能
    def test_sparse(self):
        # 创建一个完全网格 grid_full 和一个稀疏网格 grid_sparse
        grid_full = mgrid[-1:1:10j, -2:2:10j]
        grid_sparse = ogrid[-1:1:10j, -2:2:10j]

        # 稀疏网格可以通过广播变成密集网格
        grid_broadcast = np.broadcast_arrays(*grid_sparse)
        # 使用 assert_equal 逐一断言完全网格和广播网格的每个元素相等
        for f, b in zip(grid_full, grid_broadcast):
            assert_equal(f, b)

    @parametrize(
        "start, stop, step, expected",
        [
            # 测试 mgrid 处理 None 值的情况
            (None, 10, 10j, (200, 10)),
            (-10, 20, None, (1800, 30)),
        ],
    )
    # 测试 mgrid 处理 None 值的情况
    def test_mgrid_size_none_handling(self, start, stop, step, expected):
        # 使用 mgrid 创建一个网格 grid
        grid = mgrid[start:stop:step, start:stop:step]
        # 使用 mgrid 创建一个小一点的网格 grid_small
        grid_small = mgrid[start:stop:step]
        # 断言 grid 的大小等于期望的第一个值
        assert_equal(grid.size, expected[0])
        # 断言 grid_small 的大小等于期望的第二个值
        assert_equal(grid_small.size, expected[1])

    @xfail  # (reason="mgrid not implementd")
    # 测试 mgrid 接受 np.float64 类型
    def test_accepts_npfloating(self):
        # regression test for #16466
        # 使用 mgrid 创建一个 np.float64 类型的网格 grid64
        grid64 = mgrid[0.1:0.33:0.1,]
        # 使用 mgrid 创建一个 np.float64 类型的网格 grid32
        grid32 = mgrid[np.float32(0.1) : np.float32(0.33) : np.float32(0.1),]
        # 断言 grid32 的数据类型为 np.float64
        assert_(grid32.dtype == np.float64)
        # 断言 grid64 和 grid32 数组近似相等
        assert_array_almost_equal(grid64, grid32)

        # different code path for single slice
        # 使用 mgrid 创建一个 np.float64 类型的网格 grid64
        grid64 = mgrid[0.1:0.33:0.1]
        # 使用 mgrid 创建一个 np.float64 类型的网格 grid32
        grid32 = mgrid[np.float32(0.1) : np.float32(0.33) : np.float32(0.1)]
        # 断言 grid32 的数据类型为 np.float64
        assert_(grid32.dtype == np.float64)
        # 断言 grid64 和 grid32 数组近似相等
        assert_array_almost_equal(grid64, grid32)
    @skip(reason="longdouble")
    # 装饰器，标记该测试函数不执行，原因是对 longdouble 的支持
    def test_accepts_longdouble(self):
        # 回归测试，针对 issue #16945
        grid64 = mgrid[0.1:0.33:0.1,]
        grid128 = mgrid[np.longdouble(0.1) : np.longdouble(0.33) : np.longdouble(0.1),]
        # 断言 grid128 的数据类型是 np.longdouble
        assert_(grid128.dtype == np.longdouble)
        # 检查 grid64 和 grid128 是否几乎相等
        assert_array_almost_equal(grid64, grid128)

        grid128c_a = mgrid[0 : np.longdouble(1) : 3.4j]
        grid128c_b = mgrid[0 : np.longdouble(1) : 3.4j,]
        # 断言 grid128c_a 和 grid128c_b 的数据类型都是 np.longdouble
        assert_(grid128c_a.dtype == grid128c_b.dtype == np.longdouble)
        # 断言 grid128c_a 和 grid128c_b[0] 数组是否完全相等
        assert_array_equal(grid128c_a, grid128c_b[0])

        # 单独切片的不同代码路径
        grid64 = mgrid[0.1:0.33:0.1]
        grid128 = mgrid[np.longdouble(0.1) : np.longdouble(0.33) : np.longdouble(0.1)]
        # 断言 grid128 的数据类型是 np.longdouble
        assert_(grid128.dtype == np.longdouble)
        # 检查 grid64 和 grid128 是否几乎相等
        assert_array_almost_equal(grid64, grid128)

    @skip(reason="longdouble")
    # 装饰器，标记该测试函数不执行，原因是对 longdouble 的支持
    def test_accepts_npcomplexfloating(self):
        # 相关于 issue #16466
        assert_array_almost_equal(
            mgrid[0.1:0.3:3j,], mgrid[0.1 : 0.3 : np.complex64(3j),]
        )

        # 单独切片的不同代码路径
        assert_array_almost_equal(
            mgrid[0.1:0.3:3j], mgrid[0.1 : 0.3 : np.complex64(3j)]
        )

        # 相关于 issue #16945
        grid64_a = mgrid[0.1:0.3:3.3j]
        grid64_b = mgrid[0.1:0.3:3.3j,][0]
        # 断言 grid64_a 和 grid64_b 的数据类型都是 np.float64
        assert_(grid64_a.dtype == grid64_b.dtype == np.float64)
        # 断言 grid64_a 和 grid64_b 数组是否完全相等
        assert_array_equal(grid64_a, grid64_b)

        grid128_a = mgrid[0.1 : 0.3 : np.clongdouble(3.3j)]
        grid128_b = mgrid[0.1 : 0.3 : np.clongdouble(3.3j),][0]
        # 断言 grid128_a 和 grid128_b 的数据类型都是 np.longdouble
        assert_(grid128_a.dtype == grid128_b.dtype == np.longdouble)
        # 断言 grid64_a 和 grid64_b 数组是否完全相等
        assert_array_equal(grid64_a, grid64_b)
@xfail  # (reason="r_ not implemented")
# 定义一个测试类 TestConcatenator，用于测试 r_ 函数
class TestConcatenator(TestCase):
    # 定义测试函数 test_1d，测试 r_ 函数在一维数组上的行为
    def test_1d(self):
        # 断言 r_[1, 2, 3, 4, 5, 6] 的结果与 np.array([1, 2, 3, 4, 5, 6]) 相等
        assert_array_equal(r_[1, 2, 3, 4, 5, 6], np.array([1, 2, 3, 4, 5, 6]))
        
        # 创建一个全为1的数组 b
        b = np.ones(5)
        # 使用 r_ 在 b 后面添加 0, 0 和 b，得到数组 c
        c = r_[b, 0, 0, b]
        # 断言数组 c 与 [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1] 相等
        assert_array_equal(c, [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1])

    # 定义测试函数 test_mixed_type，测试 r_ 函数在混合类型数组上的行为
    def test_mixed_type(self):
        # 使用 r_ 创建一个混合类型数组 g
        g = r_[10.1, 1:10]
        # 断言数组 g 的数据类型为 "f8"
        assert_(g.dtype == "f8")

    # 定义测试函数 test_more_mixed_type，测试 r_ 函数在更复杂的混合类型数组上的行为
    def test_more_mixed_type(self):
        # 使用 r_ 创建一个更复杂的混合类型数组 g
        g = r_[-10.1, np.array([1]), np.array([2, 3, 4]), 10.0]
        # 断言数组 g 的数据类型为 "f8"
        assert_(g.dtype == "f8")

    # 定义测试函数 test_complex_step，测试 r_ 函数在复数步长上的行为
    def test_complex_step(self):
        # Regression test for #12262
        # 使用 r_ 创建一个复数步长的数组 g
        g = r_[0:36:100j]
        # 断言数组 g 的形状为 (100,)
        assert_(g.shape == (100,))

        # Related to #16466
        # 使用 r_ 创建一个复数步长的数组 g
        g = r_[0 : 36 : np.complex64(100j)]
        # 断言数组 g 的形状为 (100,)
        assert_(g.shape == (100,))

    # 定义测试函数 test_2d，测试 r_ 函数在二维数组上的行为
    def test_2d(self):
        # 创建两个随机的 5x5 数组 b 和 c
        b = np.random.rand(5, 5)
        c = np.random.rand(5, 5)
        
        # 使用 r_ 在 "1" 轴上连接数组 b 和 c，得到数组 d
        d = r_["1", b, c]  # append columns
        # 断言数组 d 的形状为 (5, 10)
        assert_(d.shape == (5, 10))
        # 断言数组 d 的前半部分列与数组 b 相等
        assert_array_equal(d[:, :5], b)
        # 断言数组 d 的后半部分列与数组 c 相等
        assert_array_equal(d[:, 5:], c)
        
        # 使用 r_ 连接数组 b 和 c
        d = r_[b, c]
        # 断言数组 d 的形状为 (10, 5)
        assert_(d.shape == (10, 5))
        # 断言数组 d 的前半部分行与数组 b 相等
        assert_array_equal(d[:5, :], b)
        # 断言数组 d 的后半部分行与数组 c 相等
        assert_array_equal(d[5:, :], c)

    # 定义测试函数 test_0d，测试 r_ 函数在零维数组上的行为
    def test_0d(self):
        # 断言 r_[0, np.array(1), 2] 的结果与 [0, 1, 2] 相等
        assert_equal(r_[0, np.array(1), 2], [0, 1, 2])
        # 断言 r_[[0, 1, 2], np.array(3)] 的结果与 [0, 1, 2, 3] 相等
        assert_equal(r_[[0, 1, 2], np.array(3)], [0, 1, 2, 3])
        # 断言 r_[np.array(0), [1, 2, 3]] 的结果与 [0, 1, 2, 3] 相等
        assert_equal(r_[np.array(0), [1, 2, 3]], [0, 1, 2, 3])


@xfail  # (reason="ndenumerate not implemented")
# 定义一个测试类 TestNdenumerate，用于测试 ndenumerate 函数
class TestNdenumerate(TestCase):
    # 定义测试函数 test_basic，测试 ndenumerate 函数的基本行为
    def test_basic(self):
        # 创建一个二维数组 a
        a = np.array([[1, 2], [3, 4]])
        # 断言使用 ndenumerate 对数组 a 进行枚举的结果与预期相等
        assert_equal(
            list(ndenumerate(a)), [((0, 0), 1), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)]
        )


# 定义一个测试类 TestIndexExpression，用于测试索引表达式的行为
class TestIndexExpression(TestCase):
    # 定义测试函数 test_regression_1，测试索引表达式的回归行为
    def test_regression_1(self):
        # ticket #1196
        # 创建一个数组 a
        a = np.arange(2)
        # 断言 a[:-1] 与 a[s_[:-1]] 的结果相等
        assert_equal(a[:-1], a[s_[:-1]])
        # 断言 a[:-1] 与 a[index_exp[:-1]] 的结果相等
        assert_equal(a[:-1], a[index_exp[:-1]])

    # 定义测试函数 test_simple_1，测试索引表达式的简单行为
    def test_simple_1(self):
        # 创建一个随机的 4x5x6 数组 a
        a = np.random.rand(4, 5, 6)

        # 断言使用索引表达式对 a 进行切片的结果与预期相等
        assert_equal(a[:, :3, [1, 2]], a[index_exp[:, :3, [1, 2]]])
        # 断言使用索引表达式对 a 进行切片的结果与预期相等
        assert_equal(a[:, :3, [1, 2]], a[s_[:, :3, [1, 2]]])


@xfail  # (reason="ix_ not implemented")
# 定义一个测试类 TestIx_，用于测试 ix_ 函数
class TestIx_(TestCase):
    # 定义测试函数 test_regression_1，测试 ix_ 函数的回归行为
    def test_regression_1(self):
        # Test empty untyped inputs create outputs of indexing type, gh-5804
        # 使用 ix_ 创建空输入的输出索引类型，验证 gh-5804
        (a,) = ix_(range(0))
        # 断言输出数组 a 的数据类型为 np.intp
        assert_equal(a.dtype, np.intp)

        (a,) = ix_([])
        # 断言输出数组 a 的数据类型为 np.intp
        assert_equal(a.dtype, np.intp)

        # but if the type is specified, don't change it
        # 但如果指定了类型，则不要更改它
        (a,) = ix_(np.array([], dtype=np.float32))
        # 断言输出数组 a 的数据类型为 np.float32
        assert_equal(a.dtype, np.float32)

    # 定义测试函数 test_shape_and_dtype，测试 ix_ 函数的形状和数据类型
    def test_shape_and_dtype(self):
        sizes = (4, 5,
    # 定义一个测试方法，用于测试布尔数组的索引功能
    def test_bool(self):
        # 创建一个包含布尔值的列表
        bool_a = [True, False, True, True]
        # 使用 numpy 的 nonzero 函数找出布尔值为 True 的索引位置
        (int_a,) = np.nonzero(bool_a)
        # 断言调用 ix_ 函数后返回的第一个元素与非零索引数组相同
        assert_equal(ix_(bool_a)[0], int_a)

    # 定义一个测试方法，用于测试只能处理一维数组的情况
    def test_1d_only(self):
        # 创建一个二维索引列表
        idx2d = [[1, 2, 3], [4, 5, 6]]
        # 断言调用 ix_ 函数时会引发 ValueError 异常，因为输入不是一维数组
        assert_raises(ValueError, ix_, idx2d)

    # 定义一个测试方法，用于测试重复输入的情况
    def test_repeated_input(self):
        # 定义向量的长度
        length_of_vector = 5
        # 使用 numpy 创建一个长度为 length_of_vector 的数组 x
        x = np.arange(length_of_vector)
        # 调用 ix_ 函数，同时传入相同的数组 x 作为输入
        out = ix_(x, x)
        # 断言 ix_ 函数返回的第一个元素的形状是 (length_of_vector, 1)
        assert_equal(out[0].shape, (length_of_vector, 1))
        # 断言 ix_ 函数返回的第二个元素的形状是 (1, length_of_vector)
        assert_equal(out[1].shape, (1, length_of_vector))
        # 检查原始输入数组 x 的形状是否未被修改
        assert_equal(x.shape, (length_of_vector,))
class TestC(TestCase):
    @xpassIfTorchDynamo  # 使用装饰器 xpassIfTorchDynamo，标记为需要跳过测试的原因为 "c_ not implemented"
    def test_c_(self):
        # 创建一个包含两个行向量的数组，并在它们之间插入两个零向量
        a = np.c_[np.array([[1, 2, 3]]), 0, 0, np.array([[4, 5, 6]])]
        # 断言数组 a 是否等于预期的值 [[1, 2, 3, 0, 0, 4, 5, 6]]
        assert_equal(a, [[1, 2, 3, 0, 0, 4, 5, 6]])


class TestFillDiagonal(TestCase):
    def test_basic(self):
        # 创建一个3x3的零矩阵，数据类型为整数
        a = np.zeros((3, 3), dtype=int)
        # 调用函数 fill_diagonal，将矩阵对角线元素填充为 5
        fill_diagonal(a, 5)
        # 断言数组 a 是否等于预期的值 np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]])
        assert_array_equal(a, np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]]))

    def test_tall_matrix(self):
        # 创建一个10x3的零矩阵，数据类型为整数
        a = np.zeros((10, 3), dtype=int)
        # 调用函数 fill_diagonal，将矩阵对角线元素填充为 5
        fill_diagonal(a, 5)
        # 断言数组 a 是否等于预期的值，一个10x3的矩阵，对角线元素填充为 5，其余为零
        assert_array_equal(
            a,
            np.array(
                [
                    [5, 0, 0],
                    [0, 5, 0],
                    [0, 0, 5],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ),
        )

    def test_tall_matrix_wrap(self):
        # 创建一个10x3的零矩阵，数据类型为整数
        a = np.zeros((10, 3), dtype=int)
        # 调用函数 fill_diagonal，将矩阵对角线元素填充为 5，wrap 参数设为 True
        fill_diagonal(a, 5, True)
        # 断言数组 a 是否等于预期的值，一个10x3的矩阵，对角线元素填充为 5，wrap 后的对角线也填充为 5
        assert_array_equal(
            a,
            np.array(
                [
                    [5, 0, 0],
                    [0, 5, 0],
                    [0, 0, 5],
                    [0, 0, 0],
                    [5, 0, 0],
                    [0, 5, 0],
                    [0, 0, 5],
                    [0, 0, 0],
                    [5, 0, 0],
                    [0, 5, 0],
                ]
            ),
        )

    def test_wide_matrix(self):
        # 创建一个3x10的零矩阵，数据类型为整数
        a = np.zeros((3, 10), dtype=int)
        # 调用函数 fill_diagonal，将矩阵对角线元素填充为 5
        fill_diagonal(a, 5)
        # 断言数组 a 是否等于预期的值，一个3x10的矩阵，对角线元素填充为 5，其余为零
        assert_array_equal(
            a,
            np.array(
                [
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 5, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        )

    def test_operate_4d_array(self):
        # 创建一个3x3x3x3的零矩阵，数据类型为整数
        a = np.zeros((3, 3, 3, 3), dtype=int)
        # 调用函数 fill_diagonal，将矩阵对角线元素填充为 4
        fill_diagonal(a, 4)
        # 创建一个索引数组 i，包含 [0, 1, 2]
        i = np.array([0, 1, 2])
        # 断言 a 中非零元素的位置是否与索引 i 的四维重复
        assert_equal(np.where(a != 0), (i, i, i, i))

    def test_low_dim_handling(self):
        # 测试处理低维度数组时是否会引发 ValueError
        a = np.zeros(3, dtype=int)
        with assert_raises(ValueError):
            fill_diagonal(a, 5)

    def test_hetero_shape_handling(self):
        # 测试处理高维度和形状不匹配数组时是否会引发 ValueError
        a = np.zeros((3, 3, 7, 3), dtype=int)
        with assert_raises(ValueError):
            fill_diagonal(a, 2)


class TestDiagIndices(TestCase):
    # 定义一个测试函数 test_diag_indices，用于测试 diag_indices 函数的返回结果
    def test_diag_indices(self):
        # 调用 diag_indices 函数获取一个数组的对角线索引
        di = diag_indices(4)
        # 创建一个 4x4 的 NumPy 数组
        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        # 使用 diag_indices 返回的索引在数组 a 上设置值为 100
        a[di] = 100
        # 断言数组 a 是否与预期数组相等
        assert_array_equal(
            a,
            np.array(
                [[100, 2, 3, 4], [5, 100, 7, 8], [9, 10, 100, 12], [13, 14, 15, 100]]
            ),
        )
    
        # 现在，我们创建用于操作 3 维数组的索引：
        d3 = diag_indices(2, 3)
    
        # 并使用它来将一个全零数组的对角线设置为 1：
        a = np.zeros((2, 2, 2), dtype=int)
        a[d3] = 1
        # 断言数组 a 是否与预期数组相等
        assert_array_equal(a, np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]]))
# 定义测试用例类 TestDiagIndicesFrom，继承自 unittest 的 TestCase 类
class TestDiagIndicesFrom(TestCase):

    # 测试 diag_indices_from 函数
    def test_diag_indices_from(self):
        # 创建一个 4x4 的随机数组
        x = np.random.random((4, 4))
        # 调用 diag_indices_from 函数，获取主对角线的行列索引
        r, c = diag_indices_from(x)
        # 断言 r 应该等于 [0, 1, 2, 3]
        assert_array_equal(r, np.arange(4))
        # 断言 c 应该等于 [0, 1, 2, 3]
        assert_array_equal(c, np.arange(4))

    # 测试当输入过小时是否会抛出 ValueError 异常
    def test_error_small_input(self):
        # 创建一个长度为 7 的全为 1 的数组
        x = np.ones(7)
        # 使用 assert_raises 确保调用 diag_indices_from 函数会抛出 ValueError 异常
        with assert_raises(ValueError):
            diag_indices_from(x)

    # 测试当输入形状不匹配时是否会抛出 ValueError 异常
    def test_error_shape_mismatch(self):
        # 创建一个形状为 (3, 3, 2, 3) 的全为 0 的整数数组
        x = np.zeros((3, 3, 2, 3), dtype=int)
        # 使用 assert_raises 确保调用 diag_indices_from 函数会抛出 ValueError 异常
        with assert_raises(ValueError):
            diag_indices_from(x)


# 定义测试用例类 TestNdIndex，继承自 unittest 的 TestCase 类
class TestNdIndex(TestCase):

    # 标记该测试用例为预期失败（xfail），原因是 ndindex 函数尚未实现
    @xfail  # (reason="ndindex not implemented")
    # 测试 ndindex 函数
    def test_ndindex(self):
        # 调用 ndindex(1, 2, 3)，将结果转为列表 x
        x = list(ndindex(1, 2, 3))
        # 生成预期结果，使用 ndenumerate(np.zeros((1, 2, 3))) 来获取所有索引
        expected = [ix for ix, e in ndenumerate(np.zeros((1, 2, 3)))]
        # 断言 x 应该等于预期结果 expected
        assert_array_equal(x, expected)

        # 再次调用 ndindex((1, 2, 3))，期望结果与上一次一致
        x = list(ndindex((1, 2, 3)))
        assert_array_equal(x, expected)

        # 测试使用标量和元组作为输入
        x = list(ndindex((3,)))
        assert_array_equal(x, list(ndindex(3)))

        # 确保不传入任何参数时，ndindex 返回单一空元组的列表
        x = list(ndindex())
        assert_equal(x, [()])

        # 传入空元组作为参数，也应当返回单一空元组的列表
        x = list(ndindex(()))
        assert_equal(x, [()])

        # 确保传入长度为 0 的参数列表时，ndindex 正确返回空列表
        x = list(ndindex(*[0]))
        assert_equal(x, [])


# 当脚本直接运行时，执行所有测试
if __name__ == "__main__":
    run_tests()
```