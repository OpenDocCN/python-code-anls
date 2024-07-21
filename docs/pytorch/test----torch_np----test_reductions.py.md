# `.\pytorch\test\torch_np\test_reductions.py`

```py
# Owner(s): ["module: dynamo"]
# 引入必要的库和模块
from unittest import skipIf, SkipTest  # 引入跳过测试相关的异常类和装饰器

import numpy  # 引入 NumPy 库

import pytest  # 引入 pytest 测试框架
from pytest import raises as assert_raises  # 将 pytest 中的 raises 方法重命名为 assert_raises

from torch.testing._internal.common_utils import (  # 从 Torch 内部测试工具中导入以下方法和对象
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo,
)

# 如果使用 TorchDynamo 进行测试，则使用 NumPy 库
# 如果在 eager 模式下进行测试，则使用 torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np  # 引入 NumPy 库，并重命名为 np
    import numpy.core.numeric as _util  # 导入 numpy.core.numeric 中的 _util 模块（用于 normalize_axis_tuple）
    from numpy.testing import (  # 从 NumPy 测试模块中导入以下断言函数
        assert_allclose,
        assert_almost_equal,
        assert_array_equal,
        assert_equal,
    )
else:
    import torch._numpy as np  # 引入 Torch 的 NumPy 接口，重命名为 np
    from torch._numpy import _util  # 从 Torch 的 NumPy 接口中导入 _util 模块
    from torch._numpy.testing import (  # 从 Torch 的 NumPy 测试模块中导入以下断言函数
        assert_allclose,
        assert_almost_equal,
        assert_array_equal,
        assert_equal,
    )


class TestFlatnonzero(TestCase):
    def test_basic(self):
        x = np.arange(-2, 3)  # 创建一个 NumPy 数组 x，其中包含从 -2 到 2 的整数
        assert_equal(np.flatnonzero(x), [0, 1, 3, 4])  # 断言 np.flatnonzero(x) 的结果与 [0, 1, 3, 4] 相等


class TestAny(TestCase):
    def test_basic(self):
        y1 = [0, 0, 1, 0]  # 创建列表 y1
        y2 = [0, 0, 0, 0]  # 创建列表 y2
        y3 = [1, 0, 1, 0]  # 创建列表 y3
        assert np.any(y1)  # 断言 y1 中存在非零元素
        assert np.any(y3)  # 断言 y3 中存在非零元素
        assert not np.any(y2)  # 断言 y2 中不存在非零元素

    def test_nd(self):
        y1 = [[0, 0, 0], [0, 1, 0], [1, 1, 0]]  # 创建二维列表 y1
        assert np.any(y1)  # 断言 y1 中存在非零元素
        assert_equal(np.any(y1, axis=0), [1, 1, 0])  # 断言沿着 axis=0 轴的 np.any(y1) 结果为 [1, 1, 0]
        assert_equal(np.any(y1, axis=1), [0, 1, 1])  # 断言沿着 axis=1 轴的 np.any(y1) 结果为 [0, 1, 1]
        assert_equal(np.any(y1), True)  # 断言 np.any(y1) 的结果为 True
        assert isinstance(np.any(y1, axis=1), np.ndarray)  # 断言 np.any(y1, axis=1) 的结果是 NumPy 数组

    # YYY: deduplicate
    def test_method_vs_function(self):
        y = np.array([[0, 1, 0, 3], [1, 0, 2, 0]])  # 创建二维 NumPy 数组 y
        assert_equal(np.any(y), y.any())  # 断言 np.any(y) 的结果与 y.any() 方法的结果相等


class TestAll(TestCase):
    def test_basic(self):
        y1 = [0, 1, 1, 0]  # 创建列表 y1
        y2 = [0, 0, 0, 0]  # 创建列表 y2
        y3 = [1, 1, 1, 1]  # 创建列表 y3
        assert not np.all(y1)  # 断言 y1 中存在非全为 True 的元素
        assert np.all(y3)  # 断言 y3 中所有元素全为 True
        assert not np.all(y2)  # 断言 y2 中所有元素全为 False
        assert np.all(~np.array(y2))  # 断言 np.array(y2) 的结果取反后所有元素全为 True

    def test_nd(self):
        y1 = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]  # 创建二维列表 y1
        assert not np.all(y1)  # 断言 y1 中存在非全为 True 的元素
        assert_equal(np.all(y1, axis=0), [0, 0, 1])  # 断言沿着 axis=0 轴的 np.all(y1) 结果为 [0, 0, 1]
        assert_equal(np.all(y1, axis=1), [0, 0, 1])  # 断言沿着 axis=1 轴的 np.all(y1) 结果为 [0, 0, 1]
        assert_equal(np.all(y1), False)  # 断言 np.all(y1) 的结果为 False

    def test_method_vs_function(self):
        y = np.array([[0, 1, 0, 3], [1, 0, 2, 0]])  # 创建二维 NumPy 数组 y
        assert_equal(np.all(y), y.all())  # 断言 np.all(y) 的结果与 y.all() 方法的结果相等


class TestMean(TestCase):
    def test_mean(self):
        A = [[1, 2, 3], [4, 5, 6]]  # 创建二维列表 A
        assert np.mean(A) == 3.5  # 断言 np.mean(A) 的结果为 3.5
        assert np.all(np.mean(A, 0) == np.array([2.5, 3.5, 4.5]))  # 断言沿着 axis=0 轴的 np.mean(A) 结果与指定数组相等
        assert np.all(np.mean(A, 1) == np.array([2.0, 5.0]))  # 断言沿着 axis=1 轴的 np.mean(A) 结果与指定数组相等

        # XXX: numpy emits a warning on empty slice
        assert np.isnan(np.mean([]))  # 断言对空切片进行 np.mean() 操作时结果为 NaN

        m = np.asarray(A)  # 将 A 转换为 NumPy 数组 m
        assert np.mean(A) == m.mean()  # 断言 np.mean(A) 的结果与 m.mean() 方法的结果相等
    def test_mean_values(self):
        # 创建一个 4x5 的浮点数数组 rmat，元素从 0 到 19
        rmat = np.arange(20, dtype=float).reshape((4, 5))
        # 创建一个复数数组 cmat，每个元素为 rmat 中的对应元素加上虚数单位乘以自身
        cmat = rmat + 1j * rmat

        # 导入警告模块
        import warnings

        # 捕获警告并设置警告处理方式为抛出异常
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # 对 rmat 和 cmat 进行迭代计算
            for mat in [rmat, cmat]:
                # 对每个轴（0 和 1）分别计算和的目标值 tgt 和均值乘以长度的结果 res
                for axis in [0, 1]:
                    tgt = mat.sum(axis=axis)
                    res = np.mean(mat, axis=axis) * mat.shape[axis]
                    # 断言 res 和 tgt 接近（所有元素）
                    assert_allclose(res, tgt)

                # 对未指定轴的情况下计算和的目标值 tgt 和均值乘以数组大小的结果 res
                for axis in [None]:
                    tgt = mat.sum(axis=axis)
                    res = np.mean(mat, axis=axis) * mat.size
                    # 断言 res 和 tgt 接近（所有元素）
                    assert_allclose(res, tgt)

    def test_mean_float16(self):
        # 如果使用 float16 而不是 float32 计算均值，将会失败
        assert np.mean(np.ones(100000, dtype="float16")) == 1

    @xpassIfTorchDynamo  # (reason="XXX: mean(..., where=...) not implemented")
    def test_mean_where(self):
        # 创建一个 4x4 的数组 a
        a = np.arange(16).reshape((4, 4))
        # 创建布尔数组 wh_full 和 wh_partial
        wh_full = np.array(
            [
                [False, True, False, True],
                [True, False, True, False],
                [True, True, False, False],
                [False, False, True, True],
            ]
        )
        wh_partial = np.array([[False], [True], [True], [False]])
        
        # 创建多种测试案例
        _cases = [
            (1, True, [1.5, 5.5, 9.5, 13.5]),
            (0, wh_full, [6.0, 5.0, 10.0, 9.0]),
            (1, wh_full, [2.0, 5.0, 8.5, 14.5]),
            (0, wh_partial, [6.0, 7.0, 8.0, 9.0]),
        ]
        # 对每个测试案例进行断言，检验均值函数在指定条件下的计算结果
        for _ax, _wh, _res in _cases:
            assert_allclose(a.mean(axis=_ax, where=_wh), np.array(_res))
            assert_allclose(np.mean(a, axis=_ax, where=_wh), np.array(_res))

        # 创建一个 2x2x4 的数组 a3d
        a3d = np.arange(16).reshape((2, 2, 4))
        # 创建布尔数组 _wh_partial
        _wh_partial = np.array([False, True, True, False])
        _res = [[1.5, 5.5], [9.5, 13.5]]
        # 对三维数组 a3d 进行断言，检验均值函数在指定条件下的计算结果
        assert_allclose(a3d.mean(axis=2, where=_wh_partial), np.array(_res))
        assert_allclose(np.mean(a3d, axis=2, where=_wh_partial), np.array(_res))

        # 检验在警告下均值函数的行为，预期会触发 RuntimeWarning 警告
        with pytest.warns(RuntimeWarning) as w:
            assert_allclose(
                a.mean(axis=1, where=wh_partial), np.array([np.nan, 5.5, 9.5, np.nan])
            )
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(a.mean(where=False), np.nan)
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(np.mean(a, where=False), np.nan)
# 使用装饰器实例化参数化测试，这里用于测试对应的测试类
@instantiate_parametrized_tests
class TestSum(TestCase):
    
    # 测试 np.sum 函数的基本功能
    def test_sum(self):
        # 定义一个二维数组 m
        m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        # 预期的结果 tgt，表示每行元素的和，保持二维结构
        tgt = [[6], [15], [24]]
        # 使用 np.sum 计算 m 按行求和，保持维度，结果保存在 out 中
        out = np.sum(m, axis=1, keepdims=True)
        # 断言预期结果与实际结果相等
        assert_equal(tgt, out)

        # 将 m 转换为 numpy 数组 am
        am = np.asarray(m)
        # 断言 np.sum(m) 与 am.sum() 的结果相等
        assert_equal(np.sum(m), am.sum())

    # 测试 np.sum 在浮点数情况下的稳定性
    def test_sum_stability(self):
        # 创建一个包含 500 个值为 1 的浮点数组 a
        a = np.ones(500, dtype=np.float32)
        # 创建一个浮点数值为 0 的 zero
        zero = np.zeros(1, dtype="float32")[0]
        # 使用 assert_allclose 检查 (a / 10.0).sum() - a.size / 10.0 是否接近于 zero，允许的绝对误差为 1.5e-4
        assert_allclose((a / 10.0).sum() - a.size / 10.0, zero, atol=1.5e-4)

        # 将数据类型改为 np.float64
        a = np.ones(500, dtype=np.float64)
        # 使用 assert_allclose 检查 (a / 10.0).sum() - a.size / 10.0 是否接近于 0.0，允许的绝对误差为 1.5e-13
        assert_allclose((a / 10.0).sum() - a.size / 10.0, 0.0, atol=1.5e-13)

    # 测试 np.sum 在布尔数组情况下的功能
    def test_sum_boolean(self):
        # 创建一个长度为 7 的布尔数组 a，元素为 0 或 1
        a = np.arange(7) % 2 == 0
        # 计算布尔数组 a 的和
        res = a.sum()
        # 断言计算结果与预期的 4 相等
        assert_equal(res, 4)

        # 以 np.float64 类型计算布尔数组 a 的和
        res_float = a.sum(dtype=np.float64)
        # 使用 assert_allclose 检查计算结果与预期值 4.0 是否接近，允许的绝对误差为 1e-15
        assert_allclose(res_float, 4.0, atol=1e-15)
        # 断言计算结果的数据类型为 float64
        assert res_float.dtype == "float64"

    # 在特定条件下测试 np.sum 函数对不同数据类型的行为
    @skipIf(numpy.__version__ < "1.24", reason="NP_VER: fails on NumPy 1.23.x")
    @xpassIfTorchDynamo  # (reason="sum: does not warn on overflow")
    def test_sum_dtypes_warnings(self):
        # 遍历不同的数据类型和值范围
        for dt in (int, np.float16, np.float32, np.float64):
            for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127, 128, 1024, 1235):
                # 如果求和导致溢出，将触发 RuntimeWarning 警告
                import warnings
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", RuntimeWarning)

                    # 计算等差数列的和 tgt
                    tgt = dt(v * (v + 1) / 2)
                    overflow = not np.isfinite(tgt)
                    # 断言触发的警告数量是否符合预期
                    assert_equal(len(w), 1 * overflow)

                    # 创建一个从 1 到 v 的等差数列 d，数据类型为 dt
                    d = np.arange(1, v + 1, dtype=dt)

                    # 使用 assert_almost_equal 检查 np.sum(d) 是否接近于 tgt，允许的绝对误差自动确定
                    assert_almost_equal(np.sum(d), tgt)
                    # 再次检查触发的警告数量是否符合预期
                    assert_equal(len(w), 2 * overflow)

                    # 使用 assert_almost_equal 检查 np.sum(np.flip(d)) 是否接近于 tgt，允许的绝对误差自动确定
                    assert_almost_equal(np.sum(np.flip(d)), tgt)
                    # 最后检查触发的警告数量是否符合预期
                    assert_equal(len(w), 3 * overflow)

    # 测试 np.sum 在不同数据类型下的使用情况
    def test_sum_dtypes_2(self):
        # 遍历不同的数据类型
        for dt in (int, np.float16, np.float32, np.float64):
            # 创建一个包含 500 个值为 1 的数组 d，数据类型为 dt
            d = np.ones(500, dtype=dt)
            # 使用 assert_almost_equal 检查 np.sum(d[::2]) 是否接近于 250.0，允许的绝对误差自动确定
            assert_almost_equal(np.sum(d[::2]), 250.0)
            # 使用 assert_almost_equal 检查 np.sum(d[1::2]) 是否接近于 250.0，允许的绝对误差自动确定
            assert_almost_equal(np.sum(d[1::2]), 250.0)
            # 使用 assert_almost_equal 检查 np.sum(d[::3]) 是否接近于 167.0，允许的绝对误差自动确定
            assert_almost_equal(np.sum(d[::3]), 167.0)
            # 使用 assert_almost_equal 检查 np.sum(d[1::3]) 是否接近于 167.0，允许的绝对误差自动确定
            assert_almost_equal(np.sum(d[1::3]), 167.0)
            # 使用 assert_almost_equal 检查 np.sum(np.flip(d)[::2]) 是否接近于 250.0，允许的绝对误差自动确定
            assert_almost_equal(np.sum(np.flip(d)[::2]), 250.0)

            # 使用 assert_almost_equal 检查 np.sum(np.flip(d)[1::2]) 是否接近于 250.0，允许的绝对误差自动确定
            assert_almost_equal(np.sum(np.flip(d)[1::2]), 250.0)

            # 使用 assert_almost_equal 检查 np.sum(np.flip(d)[::3]) 是否接近于 167.0，允许的绝对误差自动确定
            assert_almost_equal(np.sum(np.flip(d)[::3]), 167.0)
            # 使用 assert_almost_equal 检查 np.sum(np.flip(d)[1::3]) 是否接近于 167.0，允许的绝对误差自动确定
            assert_almost_equal(np.sum(np.flip(d)[1::3]), 167.0)

            # 对于第一个元素不为 0 的情况进行求和测试
            d = np.ones((1,), dtype=dt)
            d += d
            # 使用 assert_almost_equal 检查 d 的值是否接近于 2.0，允许的绝对误差自动确定
            assert_almost_equal(d, 2.0)

    # 参数化测试，测试复数类型的 np.sum 行为
    @parametrize("dt", [np.complex64, np.complex128])
    # 定义测试函数，测试复杂数据类型的求和操作
    def test_sum_complex_1(self, dt):
        # 对于给定的值集合，计算目标复数
        for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127, 128, 1024, 1235):
            tgt = dt(v * (v + 1) / 2) - dt((v * (v + 1) / 2) * 1j)
            # 创建长度为v的空数组d，数据类型为dt
            d = np.empty(v, dtype=dt)
            # 设置d数组的实部为1到v的范围
            d.real = np.arange(1, v + 1)
            # 设置d数组的虚部为负数的1到v的范围
            d.imag = -np.arange(1, v + 1)
            # 断言所有元素的总和等于目标值tgt，允许误差为1.5e-5
            assert_allclose(np.sum(d), tgt, atol=1.5e-5)
            # 断言翻转后数组的总和等于目标值tgt，允许误差为1.5e-7
            assert_allclose(np.sum(np.flip(d)), tgt, atol=1.5e-7)

    @parametrize("dt", [np.complex64, np.complex128])
    # 参数化测试函数，测试不同的复数数据类型
    def test_sum_complex_2(self, dt):
        # 创建长度为500的全1数组，数据类型为dt，加上虚数单位1j
        d = np.ones(500, dtype=dt) + 1j
        # 断言偶数索引位置的元素总和等于250.0 + 250j，允许误差为1.5e-7
        assert_allclose(np.sum(d[::2]), 250.0 + 250j, atol=1.5e-7)
        # 断言奇数索引位置的元素总和等于250.0 + 250j，允许误差为1.5e-7
        assert_allclose(np.sum(d[1::2]), 250.0 + 250j, atol=1.5e-7)
        # 断言每3个元素的总和等于167.0 + 167j，允许误差为1.5e-7
        assert_allclose(np.sum(d[::3]), 167.0 + 167j, atol=1.5e-7)
        # 断言每3个元素的总和等于167.0 + 167j，允许误差为1.5e-7
        assert_allclose(np.sum(d[1::3]), 167.0 + 167j, atol=1.5e-7)
        # 断言翻转数组中偶数索引位置的元素总和等于250.0 + 250j，允许误差为1.5e-7
        assert_allclose(np.sum(np.flip(d)[::2]), 250.0 + 250j, atol=1.5e-7)
        # 断言翻转数组中奇数索引位置的元素总和等于250.0 + 250j，允许误差为1.5e-7
        assert_allclose(np.sum(np.flip(d)[1::2]), 250.0 + 250j, atol=1.5e-7)
        # 断言翻转数组中每3个元素的总和等于167.0 + 167j，允许误差为1.5e-7
        assert_allclose(np.sum(np.flip(d)[::3]), 167.0 + 167j, atol=1.5e-7)
        # 断言翻转数组中每3个元素的总和等于167.0 + 167j，允许误差为1.5e-7
        assert_allclose(np.sum(np.flip(d)[1::3]), 167.0 + 167j, atol=1.5e-7)
        # 对于仅包含一个元素的数组d，加倍并断言结果等于2.0 + 2j，允许误差为1.5e-7
        d = np.ones((1,), dtype=dt) + 1j
        d += d
        assert_allclose(d, 2.0 + 2j, atol=1.5e-7)

    @xpassIfTorchDynamo  # (reason="initial=... need implementing")
    # 标记测试函数，测试初始值参数的使用
    def test_sum_initial(self):
        # 整数，单轴
        # 断言对单个元素3求和，初始值为2，结果应为5
        assert_equal(np.sum([3], initial=2), 5)

        # 浮点数
        # 断言对单个元素0.2求和，初始值为0.1，结果应接近0.3
        assert_almost_equal(np.sum([0.2], initial=0.1), 0.3)

        # 多个不相邻轴
        # 断言对全1数组(2, 3, 5)按轴(0, 2)求和，初始值为2，结果应为[12, 12, 12]
        assert_equal(
            np.sum(np.ones((2, 3, 5), dtype=np.int64), axis=(0, 2), initial=2),
            [12, 12, 12],
        )

    @xpassIfTorchDynamo  # (reason="where=... need implementing")
    # 标记测试函数，测试条件参数的使用
    def test_sum_where(self):
        # 在test_reduction_with_where中进行更广泛的测试。
        # 断言对矩阵[[1.0, 2.0], [3.0, 4.0]]按条件[True, False]求和，结果应为4.0
        assert_equal(np.sum([[1.0, 2.0], [3.0, 4.0]], where=[True, False]), 4.0)
        # 断言对矩阵[[1.0, 2.0], [3.0, 4.0]]按轴0，初始值为5.0，条件[True, False]求和，结果应为[9.0, 5.0]
        assert_equal(
            np.sum([[1.0, 2.0], [3.0, 4.0]], axis=0, initial=5.0, where=[True, False]),
            [9.0, 5.0],
        )
# 使用 parametrize 装饰器创建 parametrize_axis 参数化测试数据，测试不同的轴参数
parametrize_axis = parametrize(
    "axis", [0, 1, 2, -1, -2, (0, 1), (1, 0), (0, 1, 2), (1, -1, 0)]
)
# 使用 parametrize 装饰器创建 parametrize_func 参数化测试数据，测试不同的函数
parametrize_func = parametrize(
    "func",
    [
        np.any,
        np.all,
        np.argmin,
        np.argmax,
        np.min,
        np.max,
        np.mean,
        np.sum,
        np.prod,
        np.std,
        np.var,
        np.count_nonzero,
    ],
)

# 定义失败的轴参数集合
fails_axes_tuples = {
    np.any,
    np.all,
    np.argmin,
    np.argmax,
    np.prod,
}

# 定义失败的 out 参数集合
fails_out_arg = {
    np.count_nonzero,
}

# 定义限制 dtype 转换的函数集合
restricts_dtype_casts = {np.var, np.std}

# 定义失败的空元组参数集合
fails_empty_tuple = {np.argmin, np.argmax}


@instantiate_parametrized_tests
class TestGenericReductions(TestCase):
    """Run a set of generic tests to verify that self.func acts like a
    reduction operation.

    Specifically, this class checks axis=... and keepdims=... parameters.
    To check the out=... parameter, see the _GenericHasOutTestMixin class below.

    To use: subclass, define self.func and self.allowed_axes.
    """

    @parametrize_func
    def test_bad_axis(self, func):
        # 基本功能检查
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        # 检查是否抛出 TypeError 异常，当 axis 参数为 "foo" 时
        assert_raises(TypeError, func, m, axis="foo")
        # 检查是否抛出 AxisError 异常，当 axis 参数为 3 时
        assert_raises(np.AxisError, func, m, axis=3)
        # 检查是否抛出 TypeError 异常，当 axis 参数为二维数组时
        assert_raises(TypeError, func, m, axis=np.array([[1], [2]]))
        # 检查是否抛出 TypeError 异常，当 axis 参数为浮点数时
        assert_raises(TypeError, func, m, axis=1.5)

        # TODO: 添加对 np.int32(3) 等情况的测试，待实现

    @parametrize_func
    def test_array_axis(self, func):
        # 创建数组 a
        a = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])
        # 断言函数 func 对于轴参数为 np.array(-1) 与轴参数为 -1 时的结果相同
        assert_equal(func(a, axis=np.array(-1)), func(a, axis=-1))

        # 断言是否抛出 TypeError 异常，当 axis 参数为 np.array([1, 2]) 时
        with assert_raises(TypeError):
            func(a, axis=np.array([1, 2]))

    @parametrize_func
    def test_axis_empty_generic(self, func):
        # 如果 func 在 fails_empty_tuple 集合中，跳过测试
        if func in fails_empty_tuple:
            raise SkipTest("func(..., axis=()) is not valid")

        # 创建数组 a
        a = np.array([[0, 0, 1], [1, 0, 1]])
        # 断言函数 func 对于空轴参数的处理结果与扩展后的结果相同
        assert_array_equal(func(a, axis=()), func(np.expand_dims(a, axis=0), axis=0))

    @parametrize_func
    def test_axis_bad_tuple(self, func):
        # 基本功能检查
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        # 如果 func 在 fails_axes_tuples 集合中，跳过测试
        if func in fails_axes_tuples:
            raise SkipTest(f"{func.__name__} does not allow tuple axis.")

        # 断言是否抛出 ValueError 异常，当 axis 参数为 (1, 1) 时
        with assert_raises(ValueError):
            func(m, axis=(1, 1))

    @parametrize_axis
    @parametrize_func
    def test_keepdims_generic(self, axis, func):
        # 如果 func 在 fails_axes_tuples 集合中，跳过测试
        if func in fails_axes_tuples:
            raise SkipTest(f"{func.__name__} does not allow tuple axis.")

        # 创建数组 a
        a = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        # 断言带有 keepdims 参数的结果与扩展后的结果相同
        with_keepdims = func(a, axis, keepdims=True)
        expanded = np.expand_dims(func(a, axis=axis), axis=axis)
        assert_array_equal(with_keepdims, expanded)

    @skipIf(numpy.__version__ < "1.24", reason="NP_VER: fails on CI w/old numpy")
    @parametrize_func
    # 定义一个测试函数，用于测试在指定函数下，对于 axis=None 和 keepdims=True 的情况
    def test_keepdims_generic_axis_none(self, func):
        # 创建一个形状为 (2, 3, 4) 的 numpy 数组 a，其中元素为 0 到 23
        a = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        # 使用指定函数 func 处理数组 a，axis=None，keepdims=True，返回结果保持维度
        with_keepdims = func(a, axis=None, keepdims=True)
        # 使用指定函数 func 处理数组 a，axis=None，得到标量结果
        scalar = func(a, axis=None)
        # 创建一个形状与 a 相同，但填充值为 scalar 的数组
        expanded = np.full((1,) * a.ndim, fill_value=scalar)
        # 断言 with_keepdims 和 expanded 数组相等
        assert_array_equal(with_keepdims, expanded)

    @parametrize_func
    # 定义一个带参数化的测试函数，用于测试带有 out 参数的函数调用，处理标量情况
    def test_out_scalar(self, func):
        # 若 func 不支持 out 参数，则跳过测试
        if func in fails_out_arg:
            raise SkipTest(f"{func.__name__} does not have out= arg.")
        
        # 创建一个形状为 (2, 3, 4) 的 numpy 数组 a，其中元素为 0 到 23
        a = np.arange(2 * 3 * 4).reshape((2, 3, 4))

        # 对数组 a 进行 func 函数处理，不使用 out 参数
        result = func(a)
        # 创建一个与 result 具有相同形状和类型的空数组 out
        out = np.empty_like(result)
        # 使用 out 参数对数组 a 进行 func 函数处理
        result_with_out = func(a, out=out)

        # 断言 result_with_out 和 out 是同一个对象
        assert result_with_out is out
        # 断言 result 和 result_with_out 数组相等
        assert_array_equal(result, result_with_out)

    # 定义一个私有方法，用于检查带有 axis 和 keepdims 参数的 out 参数使用情况
    def _check_out_axis(self, axis, dtype, keepdims):
        # 创建一个形状为 (2, 3, 4) 的 numpy 数组 a，其中元素为 0 到 23
        a = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        # 使用指定函数 func 处理数组 a，指定 axis 和 keepdims 参数，并转换结果为指定的 dtype 类型
        result = self.func(a, axis=axis, keepdims=keepdims).astype(dtype)

        # 创建一个形状与 result 相同，dtype 为指定 dtype 的空数组 out
        out = np.empty_like(result, dtype=dtype)
        # 使用 out 参数对数组 a 进行 func 函数处理，指定 axis 和 keepdims 参数
        result_with_out = self.func(a, axis=axis, keepdims=keepdims, out=out)

        # 断言 result_with_out 和 out 是同一个对象
        assert result_with_out is out
        # 断言 result_with_out 的 dtype 与指定的 dtype 相同
        assert result_with_out.dtype == dtype
        # 断言 result 和 result_with_out 数组相等
        assert_array_equal(result, result_with_out)

        # TODO: 如果 result 的 dtype 不等于 out 的 dtype，out 是否会对结果进行类型转换？

        # 如果 out 的形状不正确（任何情况下 out 不会广播）
        # np.any(m, out=np.empty_like(m)) 会引发 ValueError（维度不正确）
        # pytorch.any 会发出警告并调整 out 数组的大小。
        # 这里我们遵循 pytorch 的方式，因为结果是 numpy 功能的超集
        # out 参数的用法会超出 numpy 的功能范围
    # 使用 @parametrize_func 和 @parametrize_axis 装饰器对测试函数进行参数化
    @parametrize_func
    @parametrize_axis
    # 定义测试函数 test_keepdims_out，接受 func 和 axis 两个参数
    def test_keepdims_out(self, func, axis):
        # 如果 func 在 fails_out_arg 列表中，抛出跳过测试的异常，说明 func 不支持 out= 参数
        if func in fails_out_arg:
            raise SkipTest(f"{func.__name__} does not have out= arg.")
        # 如果 func 在 fails_axes_tuples 列表中，抛出跳过测试的异常，说明 func 不支持元组形式的 axis 参数
        if func in fails_axes_tuples:
            raise SkipTest(f"{func.__name__} does not hangle tuple axis.")

        # 创建一个全为 1 的多维数组 d，形状为 (3, 5, 7, 11)
        d = np.ones((3, 5, 7, 11))
        
        # 如果 axis 是 None，则将 shape_out 设置为 d 的维度个数的元组，每个维度大小为 1
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            # 将 axis 标准化为元组形式，并根据标准化后的 axis 计算 shape_out
            axis_norm = _util.normalize_axis_tuple(axis, d.ndim)
            shape_out = tuple(
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim)
            )
        
        # 创建一个空数组 out，其形状为 shape_out
        out = np.empty(shape_out)

        # 调用 func 函数，传入 d、axis、keepdims=True 和 out 参数，将结果赋给 result
        result = func(d, axis=axis, keepdims=True, out=out)
        
        # 断言 result 和 out 是同一个对象
        assert result is out
        # 断言 result 的形状与 shape_out 相同
        assert_equal(result.shape, shape_out)
@instantiate_parametrized_tests
class TestGenericCumSumProd(TestCase):
    """Run a set of generic tests to verify that cumsum/cumprod are sane."""
    
    @parametrize("func", [np.cumsum, np.cumprod])
    def test_bad_axis(self, func):
        # Basic check of functionality
        # 创建一个二维 NumPy 数组作为测试数据
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        # 断言：当 axis 参数为字符串时，应该引发 TypeError
        assert_raises(TypeError, func, m, axis="foo")
        # 断言：当 axis 超出数组维度范围时，应该引发 np.AxisError
        assert_raises(np.AxisError, func, m, axis=3)
        # 断言：当 axis 参数为非法类型时，应该引发 TypeError
        assert_raises(TypeError, func, m, axis=np.array([[1], [2]]))
        # 断言：当 axis 参数为浮点数时，应该引发 TypeError
        assert_raises(TypeError, func, m, axis=1.5)

        # TODO: add tests with np.int32(3) etc, when implemented

    @parametrize("func", [np.cumsum, np.cumprod])
    def test_array_axis(self, func):
        # 创建一个二维 NumPy 数组作为测试数据
        a = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])
        # 断言：当 axis 参数为负整数数组时，应该和对应的负整数效果相同
        assert_equal(func(a, axis=np.array(-1)), func(a, axis=-1))

        # 使用 assert_raises 上下文管理器，断言：当 axis 参数为整数数组时，应该引发 TypeError
        with assert_raises(TypeError):
            func(a, axis=np.array([1, 2]))

    @parametrize("func", [np.cumsum, np.cumprod])
    def test_axis_empty_generic(self, func):
        # 创建一个二维 NumPy 数组作为测试数据
        a = np.array([[0, 0, 1], [1, 0, 1]])
        # 断言：当 axis 参数为 None 时，应该得到和 axis=0 时相同的结果
        assert_array_equal(func(a, axis=None), func(a.ravel(), axis=0))

    @parametrize("func", [np.cumsum, np.cumprod])
    def test_axis_bad_tuple(self, func):
        # Basic check of functionality
        # 创建一个二维 NumPy 数组作为测试数据
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])
        # 使用 assert_raises 上下文管理器，断言：当 axis 参数为元组时，应该引发 TypeError
        with assert_raises(TypeError):
            func(m, axis=(1, 1))


if __name__ == "__main__":
    run_tests()
```