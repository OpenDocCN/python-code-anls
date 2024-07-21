# `.\pytorch\test\torch_np\numpy_tests\lib\test_twodim_base.py`

```py
# Owner(s): ["module: dynamo"]

"""Test functions for matrix module

"""
import functools  # 导入 functools 模块，用于创建偏函数

from unittest import expectedFailure as xfail, skipIf as skipif  # 从 unittest 模块中导入 expectedFailure 和 skipIf 装饰器

import pytest  # 导入 pytest 测试框架
from pytest import raises as assert_raises  # 从 pytest 模块中导入 raises 函数并重命名为 assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入测试工具函数 instantiate_parametrized_tests
    parametrize,  # 导入测试参数化装饰器 parametrize
    run_tests,  # 导入运行测试函数 run_tests
    TEST_WITH_TORCHDYNAMO,  # 导入测试标志 TEST_WITH_TORCHDYNAMO
    TestCase,  # 导入测试用例基类 TestCase
    xpassIfTorchDynamo,  # 导入条件装饰器 xpassIfTorchDynamo
)


# If we are going to trace through these, we should use NumPy
# If testing on eager mode, we use torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np  # 如果 TEST_WITH_TORCHDYNAMO 为 True，则导入 numpy
    from numpy import (
        arange,  # 导入 numpy 中的 arange 函数
        array,  # 导入 numpy 中的 array 函数
        diag,  # 导入 numpy 中的 diag 函数
        eye,  # 导入 numpy 中的 eye 函数
        fliplr,  # 导入 numpy 中的 fliplr 函数
        flipud,  # 导入 numpy 中的 flipud 函数
        histogram2d,  # 导入 numpy 中的 histogram2d 函数
        ones,  # 导入 numpy 中的 ones 函数
        tri,  # 导入 numpy 中的 tri 函数
        tril_indices,  # 导入 numpy 中的 tril_indices 函数
        tril_indices_from,  # 导入 numpy 中的 tril_indices_from 函数
        triu_indices,  # 导入 numpy 中的 triu_indices 函数
        triu_indices_from,  # 导入 numpy 中的 triu_indices_from 函数
        vander,  # 导入 numpy 中的 vander 函数
        zeros,  # 导入 numpy 中的 zeros 函数
    )
    from numpy.testing import (
        assert_allclose,  # 导入 numpy 测试模块中的 assert_allclose 函数
        assert_array_almost_equal,  # 导入 numpy 测试模块中的 assert_array_almost_equal 函数
        assert_array_equal,  # 导入 numpy 测试模块中的 assert_array_equal 函数
        assert_equal,  # 导入 numpy 测试模块中的 assert_equal 函数
    )
else:
    import torch._numpy as np  # 如果 TEST_WITH_TORCHDYNAMO 为 False，则导入 torch._numpy 作为 np
    from torch._numpy import (
        arange,  # 导入 torch._numpy 中的 arange 函数
        array,  # 导入 torch._numpy 中的 array 函数
        diag,  # 导入 torch._numpy 中的 diag 函数
        eye,  # 导入 torch._numpy 中的 eye 函数
        fliplr,  # 导入 torch._numpy 中的 fliplr 函数
        flipud,  # 导入 torch._numpy 中的 flipud 函数
        histogram2d,  # 导入 torch._numpy 中的 histogram2d 函数
        ones,  # 导入 torch._numpy 中的 ones 函数
        tri,  # 导入 torch._numpy 中的 tri 函数
        tril_indices,  # 导入 torch._numpy 中的 tril_indices 函数
        tril_indices_from,  # 导入 torch._numpy 中的 tril_indices_from 函数
        triu_indices,  # 导入 torch._numpy 中的 triu_indices 函数
        triu_indices_from,  # 导入 torch._numpy 中的 triu_indices_from 函数
        vander,  # 导入 torch._numpy 中的 vander 函数
        zeros,  # 导入 torch._numpy 中的 zeros 函数
    )
    from torch._numpy.testing import (
        assert_allclose,  # 导入 torch._numpy 测试模块中的 assert_allclose 函数
        assert_array_almost_equal,  # 导入 torch._numpy 测试模块中的 assert_array_almost_equal 函数
        assert_array_equal,  # 导入 torch._numpy 测试模块中的 assert_array_equal 函数
        assert_equal,  # 导入 torch._numpy 测试模块中的 assert_equal 函数
    )


skip = functools.partial(skipif, True)  # 创建 skip 变量作为 functools.partial 函数的偏函数，用于跳过测试


def get_mat(n):
    data = np.arange(n)  # 创建一个长度为 n 的 numpy 数组
    # data = np.add.outer(data, data)
    data = data[:, None] + data[None, :]  # 计算外积并更新 data
    return data


class TestEye(TestCase):
    def test_basic(self):
        assert_equal(  # 断言函数，比较生成的单位矩阵与预期结果是否相等
            eye(4), array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        )

        assert_equal(  # 断言函数，比较生成的单位矩阵与预期结果是否相等
            eye(4, dtype="f"),
            array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "f"),
        )

        assert_equal(  # 断言函数，比较生成的单位矩阵与预期结果是否相等
            eye(3) == 1, eye(3, dtype=bool)
        )

    def test_diag(self):
        assert_equal(  # 断言函数，比较生成的带有偏移的单位矩阵与预期结果是否相等
            eye(4, k=1), array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        )

        assert_equal(  # 断言函数，比较生成的带有偏移的单位矩阵与预期结果是否相等
            eye(4, k=-1),
            array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
        )

    def test_2d(self):
        assert_equal(  # 断言函数，比较生成的带有不同形状的单位矩阵与预期结果是否相等
            eye(4, 3), array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        )

        assert_equal(  # 断言函数，比较生成的带有不同形状的单位矩阵与预期结果是否相等
            eye(3, 4), array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        )

    def test_diag2d(self):
        assert_equal(  # 断言函数，比较生成的带有偏移的二维单位矩阵与预期结果是否相等
            eye(3, 4, k=2), array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        )

        assert_equal(  # 断言函数，比较生成的带有偏移的二维单位矩阵与预期结果是否相等
            eye(4, 3, k=-2), array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0
    # 测试函数，验证 eye 函数在不同参数下生成的单位矩阵是否符合预期
    def test_eye_bounds(self):
        assert_equal(eye(2, 2, 1), [[0, 1], [0, 0]])  # 验证单位矩阵的右上角元素为 1
        assert_equal(eye(2, 2, -1), [[0, 0], [1, 0]])  # 验证单位矩阵的左下角元素为 1
        assert_equal(eye(2, 2, 2), [[0, 0], [0, 0]])  # 验证超出矩阵大小的 k 值返回全零矩阵
        assert_equal(eye(2, 2, -2), [[0, 0], [0, 0]])  # 验证超出矩阵大小的 k 值返回全零矩阵
        assert_equal(eye(3, 2, 2), [[0, 0], [0, 0], [0, 0]])  # 验证超出矩阵大小的 k 值返回全零矩阵
        assert_equal(eye(3, 2, 1), [[0, 1], [0, 0], [0, 0]])  # 验证单位矩阵的右上角元素为 1
        assert_equal(eye(3, 2, -1), [[0, 0], [1, 0], [0, 1]])  # 验证单位矩阵的左下角和右上角元素为 1
        assert_equal(eye(3, 2, -2), [[0, 0], [0, 0], [1, 0]])  # 验证单位矩阵的左下角元素为 1
        assert_equal(eye(3, 2, -3), [[0, 0], [0, 0], [0, 0]])  # 验证超出矩阵大小的 k 值返回全零矩阵

    # 测试函数，验证 eye 函数在 dtype=bool 参数下生成的单位矩阵是否符合预期
    def test_bool(self):
        assert_equal(eye(2, 2, dtype=bool), [[True, False], [False, True]])  # 验证 bool 类型下单位矩阵的正确性

    @xpassIfTorchDynamo  # 标记为需要跳过的测试，原因是待实现的非默认顺序功能
    def test_order(self):
        # 创建一个 C 顺序的单位矩阵
        mat_c = eye(4, 3, k=-1)
        # 创建一个 Fortran (F) 顺序的单位矩阵
        mat_f = eye(4, 3, k=-1, order="F")
        assert_equal(mat_c, mat_f)  # 验证两种顺序下生成的单位矩阵是否相等
        assert mat_c.flags.c_contiguous  # 验证 C 顺序下矩阵的连续性
        assert not mat_c.flags.f_contiguous  # 验证 C 顺序下矩阵不是 Fortran 顺序的连续性
        assert not mat_f.flags.c_contiguous  # 验证 Fortran 顺序下矩阵不是 C 顺序的连续性
        assert mat_f.flags.f_contiguous  # 验证 Fortran 顺序下矩阵的连续性
# 定义 TestCase 的子类 TestDiag，用于测试 diag 函数的各种情况
class TestDiag(TestCase):
    
    # 测试 diag 函数对向量的应用
    def test_vector(self):
        # 创建一个整数类型的向量 vals，内容为 [0, 100, 200, 300, 400]
        vals = (100 * arange(5)).astype("l")
        # 创建一个全零的 5x5 矩阵 b
        b = zeros((5, 5))
        # 将 vals 中的每个元素设置为矩阵 b 的主对角线元素
        for k in range(5):
            b[k, k] = vals[k]
        # 断言 diag(vals) 和矩阵 b 相等
        assert_equal(diag(vals), b)
        
        # 创建一个全零的 7x7 矩阵 b 和其复制 c
        b = zeros((7, 7))
        c = b.copy()
        # 将 vals 中的每个元素分别设置为矩阵 b 和 c 中指定位置的对角线元素
        for k in range(5):
            b[k, k + 2] = vals[k]
            c[k + 2, k] = vals[k]
        # 断言 diag(vals, k=2) 和矩阵 b 相等
        assert_equal(diag(vals, k=2), b)
        # 断言 diag(vals, k=-2) 和矩阵 c 相等
        assert_equal(diag(vals, k=-2), c)

    # 测试 diag 函数对矩阵的应用
    def test_matrix(self):
        # 调用 check_matrix 方法，传入一个由 get_mat(5) 生成的矩阵乘以 100 再加 1，并转换为整数类型的数组 vals
        self.check_matrix(vals=(100 * get_mat(5) + 1).astype("l"))

    # 检查 diag 函数对矩阵的应用
    def check_matrix(self, vals):
        # 创建一个全零的 5x1 数组 b
        b = zeros((5,))
        # 将 vals 中的每个对角线元素分别赋给数组 b
        for k in range(5):
            b[k] = vals[k, k]
        # 断言 diag(vals) 和数组 b 相等
        assert_equal(diag(vals), b)
        
        # 将数组 b 的元素全部置为 0
        b = b * 0
        # 将 vals 中每个指定位置的对角线元素赋给数组 b 的前三个元素
        for k in range(3):
            b[k] = vals[k, k + 2]
        # 断言 diag(vals, 2) 和数组 b 的前三个元素相等
        assert_equal(diag(vals, 2), b[:3])
        
        # 将数组 b 的元素全部置为 0
        b = b * 0
        # 将 vals 中每个指定位置的对角线元素赋给数组 b 的前三个元素
        for k in range(3):
            b[k] = vals[k + 2, k]
        # 断言 diag(vals, -2) 和数组 b 的前三个元素相等
        assert_equal(diag(vals, -2), b[:3])

    # 标记为跳过，需实现 orders
    @xpassIfTorchDynamo  # (reason="TODO implement orders")
    def test_fortran_order(self):
        # 创建一个 Fortran 顺序的整数类型数组 vals，内容为 get_mat(5) 乘以 100 再加 1
        vals = array((100 * get_mat(5) + 1), order="F", dtype="l")
        # 调用 check_matrix 方法，传入数组 vals
        self.check_matrix(vals)

    # 测试 diag 函数对边界情况的应用
    def test_diag_bounds(self):
        # 创建一个二维列表 A
        A = [[1, 2], [3, 4], [5, 6]]
        # 断言 diag(A, k=2) 返回空列表
        assert_equal(diag(A, k=2), [])
        # 断言 diag(A, k=1) 返回 [2]
        assert_equal(diag(A, k=1), [2])
        # 断言 diag(A, k=0) 返回 [1, 4]
        assert_equal(diag(A, k=0), [1, 4])
        # 断言 diag(A, k=-1) 返回 [3, 6]
        assert_equal(diag(A, k=-1), [3, 6])
        # 断言 diag(A, k=-2) 返回 [5]
        assert_equal(diag(A, k=-2), [5])
        # 断言 diag(A, k=-3) 返回空列表
        assert_equal(diag(A, k=-3), [])

    # 测试 diag 函数对异常情况的处理
    def test_failure(self):
        # 断言 diag([[[1]]]) 抛出 ValueError 或 RuntimeError 异常
        assert_raises((ValueError, RuntimeError), diag, [[[1]]])


# 定义 TestCase 的子类 TestFliplr，用于测试 fliplr 函数的各种情况
class TestFliplr(TestCase):
    
    # 测试 fliplr 函数的基本用法
    def test_basic(self):
        # 断言 fliplr(ones(4)) 抛出 ValueError 或 RuntimeError 异常
        assert_raises((ValueError, RuntimeError), fliplr, ones(4))
        # 创建一个矩阵 a，内容为 get_mat(4) 的结果
        a = get_mat(4)
        # 使用 np.flip 函数对矩阵 a 进行水平翻转，赋给 b
        b = np.flip(a, 1)
        # 断言 fliplr(a) 和矩阵 b 相等
        assert_equal(fliplr(a), b)
        # 创建一个二维列表 a
        a = [[0, 1, 2], [3, 4, 5]]
        # 创建一个预期的结果列表 b
        b = [[2, 1, 0], [5, 4, 3]]
        # 断言 fliplr(a) 和列表 b 相等
        assert_equal(fliplr(a), b)


# 定义 TestCase 的子类 TestFlipud，用于测试 flipud 函数的各种情况
class TestFlipud(TestCase):
    
    # 测试 flipud 函数的基本用法
    def test_basic(self):
        # 创建一个矩阵 a，内容为 get_mat(4) 的结果
        a = get_mat(4)
        # 使用 np.flip 函数对矩阵 a 进行垂直翻转，赋给 b
        b = np.flip(a, 0)
        # 断言 flipud(a) 和矩阵 b 相等
        assert_equal(flipud(a), b)
        # 创建一个二维列表 a
        a = [[0, 1, 2], [3, 4, 5]]
        # 创建一个预期的结果列表 b
        b = [[3, 4, 5], [0, 1, 2]]
        # 断言 flipud(a) 和列表 b 相等
        assert_equal(flipud(a), b)


# 实例化参数化测试
@instantiate_parametrized_tests
class TestHistogram2d(TestCase):
    pass  # 在这里没有添加任何测试，只是简单地传递了一个空的测试类声明
    # 定义一个测试方法，用于测试简单的二维直方图生成
    def test_simple(self):
        # 创建输入数据的数组 x 和 y
        x = array([0.41702200, 0.72032449, 1.1437481e-4, 0.302332573, 0.146755891])
        y = array([0.09233859, 0.18626021, 0.34556073, 0.39676747, 0.53881673])
        
        # 定义 x 和 y 的边界
        xedges = np.linspace(0, 1, 10)
        yedges = np.linspace(0, 1, 10)
        
        # 生成二维直方图 H，并取其第一个元素（频次或密度）
        H = histogram2d(x, y, (xedges, yedges))[0]
        
        # 预期的结果数组 answer
        answer = array(
            [
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        
        # 断言生成的直方图 H 的转置与预期结果 answer 相等
        assert_array_equal(H.T, answer)
        
        # 再次生成二维直方图 H（这次指定 xedges 参数），并断言其转置与预期结果 answer 相等
        H = histogram2d(x, y, xedges)[0]
        assert_array_equal(H.T, answer)
        
        # 生成包含边界信息的二维直方图 H、xedges 和 yedges，并断言 H 与单位矩阵 eye(10, 10) 相等
        H, xedges, yedges = histogram2d(list(range(10)), list(range(10)))
        assert_array_equal(H, eye(10, 10))
        
        # 断言 xedges 与 np.linspace(0, 9, 11) 相等，断言 yedges 与 np.linspace(0, 9, 11) 相等
        assert_array_equal(xedges, np.linspace(0, 9, 11))
        assert_array_equal(yedges, np.linspace(0, 9, 11))

    # 定义一个测试方法，用于测试非对称数据的二维直方图生成（包含密度）
    def test_asym(self):
        # 创建输入数据的数组 x 和 y
        x = array([1, 1, 2, 3, 4, 4, 4, 5])
        y = array([1, 3, 2, 0, 1, 2, 3, 4])
        
        # 生成非对称数据的二维直方图 H、xed 和 yed，指定参数 density=True
        H, xed, yed = histogram2d(x, y, (6, 5), range=[[0, 6], [0, 5]], density=True)
        
        # 预期的结果数组 answer
        answer = array(
            [
                [0.0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )
        
        # 断言生成的直方图 H 与预期结果 answer 除以 8.0（密度因子）相等，精度为 3 位小数
        assert_array_almost_equal(H, answer / 8.0, 3)
        
        # 断言 xed 与 np.linspace(0, 6, 7) 相等，断言 yed 与 np.linspace(0, 5, 6) 相等
        assert_array_equal(xed, np.linspace(0, 6, 7))
        assert_array_equal(yed, np.linspace(0, 5, 6))

    # 定义一个测试方法，用于测试密度信息的二维直方图生成
    def test_density(self):
        # 创建输入数据的数组 x 和 y
        x = array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        y = array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        
        # 生成包含密度信息的二维直方图 H、xed 和 yed，并指定边界参数
        H, xed, yed = histogram2d(x, y, [[1, 2, 3, 5], [1, 2, 3, 5]], density=True)
        
        # 预期的结果数组 answer，根据输入数据和边界计算得到
        answer = array([[1, 1, 0.5], [1, 1, 0.5], [0.5, 0.5, 0.25]]) / 9.0
        
        # 断言生成的直方图 H 与预期结果 answer 相等，精度为 3 位小数
        assert_array_almost_equal(H, answer, 3)

    # 定义一个测试方法，用于测试全部为离群值的情况下的二维直方图生成
    def test_all_outliers(self):
        # 创建包含全部为离群值的随机数组 r
        r = np.random.rand(100) + 1.0 + 1e6  # histogramdd rounds by decimal=6
        
        # 生成包含离群值的二维直方图 H、xed 和 yed，指定边界范围
        H, xed, yed = histogram2d(r, r, (4, 5), range=([0, 1], [0, 1]))
        
        # 断言生成的直方图 H 全部为零
        assert_array_equal(H, 0)

    # 定义一个测试方法，用于测试空输入数据的二维直方图生成
    def test_empty(self):
        # 生成空输入数据的二维直方图 a、edge1 和 edge2，指定 bins 参数
        a, edge1, edge2 = histogram2d([], [], bins=([0, 1], [0, 1]))
        
        # 断言生成的直方图 a 与全零数组 np.array([[0.0]]) 相等，允许误差为 1e-15
        assert_allclose(a, np.array([[0.0]]), atol=1e-15)
        
        # 生成空输入数据的二维直方图 a、edge1 和 edge2，指定 bins 参数为整数
        a, edge1, edge2 = histogram2d([], [], bins=4)
        
        # 断言生成的直方图 a 与全零的 4x4 数组相等，允许误差为 1e-15
        assert_allclose(a, np.zeros((4, 4)), atol=1e-15)
    # 定义一个测试方法，用于测试二维直方图函数的不同二进制参数组合
    def test_binparameter_combination(self):
        # 定义输入数据数组 x 和 y
        x = array([0, 0.09207008, 0.64575234, 0.12875982, 0.47390599, 0.59944483, 1])
        y = array([0, 0.14344267, 0.48988575, 0.30558665, 0.44700682, 0.15886423, 1])
        # 定义直方图的边界
        edges = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
        # 使用给定的边界计算二维直方图 H 和对应的边界 xe, ye
        H, xe, ye = histogram2d(x, y, (edges, 4))
        # 预期的二维数组结果
        answer = array(
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # 断言计算得到的直方图 H 与预期结果相等
        assert_array_equal(H, answer)
        # 断言计算得到的 y 轴边界 ye 与预期结果相等
        assert_array_equal(ye, array([0.0, 0.25, 0.5, 0.75, 1]))
        
        # 使用另一种边界计算二维直方图 H 和对应的边界 xe, ye
        H, xe, ye = histogram2d(x, y, (4, edges))
        # 预期的二维数组结果
        answer = array(
            [
                [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        # 断言计算得到的直方图 H 与预期结果相等
        assert_array_equal(H, answer)
        # 断言计算得到的 x 轴边界 xe 与预期结果相等
        assert_array_equal(xe, array([0.0, 0.25, 0.5, 0.75, 1]))

    # 使用 skip 装饰器标记测试用例，给出测试失败的原因
    @skip(reason="NP_VER: fails on CI with older NumPy")
    # 使用 parametrize 装饰器标记测试用例，传入不同的参数组合进行测试
    @parametrize("x_len, y_len", [(10, 11), (20, 19)])
    # 定义测试方法，测试当 x 和 y 长度不相同时的情况
    def test_bad_length(self, x_len, y_len):
        # 创建长度分别为 x_len 和 y_len 的全为 1 的数组 x 和 y
        x, y = np.ones(x_len), np.ones(y_len)
        # 使用 pytest 断言，检查是否抛出 ValueError 异常，并匹配特定的错误信息
        with pytest.raises(ValueError, match="x and y must have the same length."):
            histogram2d(x, y)
class TestTri(TestCase):
    # 定义测试用例类 TestTri，继承自 TestCase

    def test_dtype(self):
        # 定义测试方法 test_dtype

        out = array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        # 预期输出的数组 out

        assert_array_equal(tri(3), out)
        # 断言 tri(3) 函数输出与预期数组 out 相等

        assert_array_equal(tri(3, dtype=bool), out.astype(bool))
        # 断言 tri(3, dtype=bool) 函数输出与预期数组 out 转换为布尔型后的数组相等

    def test_tril_triu_ndim2(self):
        # 定义测试方法 test_tril_triu_ndim2

        for dtype in np.typecodes["AllFloat"] + np.typecodes["AllInteger"]:
            # 遍历所有浮点数和整数类型

            a = np.ones((2, 2), dtype=dtype)
            # 创建全为 1 的数组 a，指定数据类型为当前循环的 dtype

            b = np.tril(a)
            # 计算数组 a 的下三角部分，赋值给 b

            c = np.triu(a)
            # 计算数组 a 的上三角部分，赋值给 c

            assert_array_equal(b, [[1, 0], [1, 1]])
            # 断言数组 b 与预期的下三角数组 [[1, 0], [1, 1]] 相等

            assert_array_equal(c, b.T)
            # 断言数组 c 与 b 的转置相等，即 c 应为 b 的上三角部分

            # 应返回与原始数组相同的数据类型
            assert_equal(b.dtype, a.dtype)
            # 断言 b 的数据类型与 a 的数据类型相同

            assert_equal(c.dtype, a.dtype)
            # 断言 c 的数据类型与 a 的数据类型相同

    def test_tril_triu_ndim3(self):
        # 定义测试方法 test_tril_triu_ndim3

        for dtype in np.typecodes["AllFloat"] + np.typecodes["AllInteger"]:
            # 遍历所有浮点数和整数类型

            a = np.array(
                [
                    [[1, 1], [1, 1]],
                    [[1, 1], [1, 0]],
                    [[1, 1], [0, 0]],
                ],
                dtype=dtype,
            )
            # 创建具有三维结构的数组 a，指定数据类型为当前循环的 dtype

            a_tril_desired = np.array(
                [
                    [[1, 0], [1, 1]],
                    [[1, 0], [1, 0]],
                    [[1, 0], [0, 0]],
                ],
                dtype=dtype,
            )
            # 创建预期的下三角数组 a_tril_desired，数据类型与 a 相同

            a_triu_desired = np.array(
                [
                    [[1, 1], [0, 1]],
                    [[1, 1], [0, 0]],
                    [[1, 1], [0, 0]],
                ],
                dtype=dtype,
            )
            # 创建预期的上三角数组 a_triu_desired，数据类型与 a 相同

            a_triu_observed = np.triu(a)
            # 计算数组 a 的上三角部分，观察结果赋值给 a_triu_observed

            a_tril_observed = np.tril(a)
            # 计算数组 a 的下三角部分，观察结果赋值给 a_tril_observed

            assert_array_equal(a_triu_observed, a_triu_desired)
            # 断言观察到的上三角部分与预期的相等

            assert_array_equal(a_tril_observed, a_tril_desired)
            # 断言观察到的下三角部分与预期的相等

            assert_equal(a_triu_observed.dtype, a.dtype)
            # 断言上三角部分的数据类型与数组 a 的数据类型相同

            assert_equal(a_tril_observed.dtype, a.dtype)
            # 断言下三角部分的数据类型与数组 a 的数据类型相同

    def test_tril_triu_with_inf(self):
        # 定义测试方法 test_tril_triu_with_inf

        arr = np.array([[1, 1, np.inf], [1, 1, 1], [np.inf, 1, 1]])
        # 创建包含无穷大值的数组 arr

        out_tril = np.array([[1, 0, 0], [1, 1, 0], [np.inf, 1, 1]])
        # 创建预期的下三角数组 out_tril

        out_triu = out_tril.T
        # 计算 out_tril 的转置，赋值给 out_triu

        assert_array_equal(np.triu(arr), out_triu)
        # 断言数组 arr 的上三角部分与预期的上三角数组 out_triu 相等

        assert_array_equal(np.tril(arr), out_tril)
        # 断言数组 arr 的下三角部分与预期的下三角数组 out_tril 相等

    def test_tril_triu_dtype(self):
        # 定义测试方法 test_tril_triu_dtype

        for c in "efdFDBbhil?":  # np.typecodes["All"]:
            # 遍历所有数据类型字符

            arr = np.zeros((3, 3), dtype=c)
            # 创建全为 0 的数组 arr，指定数据类型为当前循环的 c

            assert_equal(np.triu(arr).dtype, arr.dtype)
            # 断言 arr 的上三角部分的数据类型与 arr 的数据类型相同

            assert_equal(np.tril(arr).dtype, arr.dtype)
            # 断言 arr 的下三角部分的数据类型与 arr 的数据类型相同

    @xfail  # (reason="TODO: implement mask_indices")
    def test_mask_indices(self):
        # 定义测试方法 test_mask_indices，标记为待实现

        # simple test without offset
        iu = mask_indices(3, np.triu)
        # 调用 mask_indices 函数生成上三角矩阵的索引 iu

        a = np.arange(9).reshape(3, 3)
        # 创建 0 到 8 的数组，reshape 成 3x3 的矩阵，赋值给 a

        assert_array_equal(a[iu], array([0, 1, 2, 4, 5, 8]))
        # 断言从数组 a 中使用索引 iu 提取的元素与指定的数组相等

        # Now with an offset
        iu1 = mask_indices(3, np.triu, 1)
        # 调用 mask_indices 函数生成上三角矩阵的索引 iu1，带有偏移值

        assert_array_equal(a[iu1], array([1, 2, 5]))
        # 断言从数组 a 中使用索引 iu1 提取的元素与指定的数组相等

    @xfail  # (reason="np.tril_indices == our tuple(tril_indices)")
    # 标记为待修复，注明原因
    # 定义一个测试函数，用于测试 tril_indices 函数的行为
    def test_tril_indices(self):
        # 生成一个未偏移和带偏移的索引数组
        il1 = tril_indices(4)
        il2 = tril_indices(4, k=2)
        il3 = tril_indices(4, m=5)
        il4 = tril_indices(4, k=2, m=5)

        # 创建一个 4x4 的二维数组 a 和一个 4x5 的二维数组 b
        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        b = np.arange(1, 21).reshape(4, 5)

        # 对 a 和 b 数组进行索引操作，并使用 assert_array_equal 进行断言比较
        assert_array_equal(a[il1], array([1, 5, 6, 9, 10, 11, 13, 14, 15, 16]))
        assert_array_equal(b[il3], array([1, 6, 7, 11, 12, 13, 16, 17, 18, 19]))

        # 对 a 和 b 数组进行赋值操作，并使用 assert_array_equal 进行断言比较
        a[il1] = -1
        assert_array_equal(
            a,
            array([[-1, 2, 3, 4], [-1, -1, 7, 8], [-1, -1, -1, 12], [-1, -1, -1, -1]]),
        )
        b[il3] = -1
        assert_array_equal(
            b,
            array(
                [
                    [-1, 2, 3, 4, 5],
                    [-1, -1, 8, 9, 10],
                    [-1, -1, -1, 14, 15],
                    [-1, -1, -1, -1, 20],
                ]
            ),
        )

        # 对 a 和 b 数组的两个主对角线右侧几乎覆盖整个数组进行赋值操作
        a[il2] = -10
        assert_array_equal(
            a,
            array(
                [
                    [-10, -10, -10, 4],
                    [-10, -10, -10, -10],
                    [-10, -10, -10, -10],
                    [-10, -10, -10, -10],
                ]
            ),
        )
        b[il4] = -10
        assert_array_equal(
            b,
            array(
                [
                    [-10, -10, -10, 4, 5],
                    [-10, -10, -10, -10, 10],
                    [-10, -10, -10, -10, -10],
                    [-10, -10, -10, -10, -10],
                ]
            ),
        )
@xfail  # 标记为预期失败的测试用例，原因是"np.triu_indices == our tuple(triu_indices)"
class TestTriuIndices(TestCase):
    def test_triu_indices(self):
        # 获取4x4矩阵的上三角部分的索引
        iu1 = triu_indices(4)
        # 获取4x4矩阵的上三角部分的索引，偏移2个位置
        iu2 = triu_indices(4, k=2)
        # 获取4x5矩阵的上三角部分的索引，矩阵实际大小为4x5
        iu3 = triu_indices(4, m=5)
        # 获取4x5矩阵的上三角部分的索引，偏移2个位置，矩阵实际大小为4x5
        iu4 = triu_indices(4, k=2, m=5)

        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        b = np.arange(1, 21).reshape(4, 5)

        # 验证索引用于获取数组的值
        assert_array_equal(a[iu1], array([1, 2, 3, 4, 6, 7, 8, 11, 12, 16]))
        assert_array_equal(
            b[iu3], array([1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 19, 20])
        )

        # 验证索引用于赋值操作
        a[iu1] = -1
        assert_array_equal(
            a,
            array(
                [[-1, -1, -1, -1], [5, -1, -1, -1], [9, 10, -1, -1], [13, 14, 15, -1]]
            ),
        )
        b[iu3] = -1
        assert_array_equal(
            b,
            array(
                [
                    [-1, -1, -1, -1, -1],
                    [6, -1, -1, -1, -1],
                    [11, 12, -1, -1, -1],
                    [16, 17, 18, -1, -1],
                ]
            ),
        )

        # 验证索引覆盖了几乎整个数组（主对角线右侧的两个对角线）
        a[iu2] = -10
        assert_array_equal(
            a,
            array(
                [
                    [-1, -1, -10, -10],
                    [5, -1, -1, -10],
                    [9, 10, -1, -1],
                    [13, 14, 15, -1],
                ]
            ),
        )
        b[iu4] = -10
        assert_array_equal(
            b,
            array(
                [
                    [-1, -1, -10, -10, -10],
                    [6, -1, -1, -10, -10],
                    [11, 12, -1, -1, -10],
                    [16, 17, 18, -1, -1],
                ]
            ),
        )


class TestTrilIndicesFrom(TestCase):
    def test_exceptions(self):
        # 断言抛出 ValueError 异常，因为输入数组维度不正确
        assert_raises(ValueError, tril_indices_from, np.ones((2,)))
        # 断言抛出 ValueError 异常，因为输入数组维度不正确
        assert_raises(ValueError, tril_indices_from, np.ones((2, 2, 2)))
        # assert_raises(ValueError, tril_indices_from, np.ones((2, 3)))


class TestTriuIndicesFrom(TestCase):
    def test_exceptions(self):
        # 断言抛出 ValueError 异常，因为输入数组维度不正确
        assert_raises(ValueError, triu_indices_from, np.ones((2,)))
        # 断言抛出 ValueError 异常，因为输入数组维度不正确
        assert_raises(ValueError, triu_indices_from, np.ones((2, 2, 2)))
        # assert_raises(ValueError, triu_indices_from, np.ones((2, 3)))


class TestVander(TestCase):
    def test_basic(self):
        c = np.array([0, 1, -2, 3])
        v = vander(c)
        powers = np.array(
            [[0, 0, 0, 0, 1], [1, 1, 1, 1, 1], [16, -8, 4, -2, 1], [81, 27, 9, 3, 1]]
        )
        # 验证默认情况下的 N 值
        assert_array_equal(v, powers[:, 1:])
        # 验证一系列的 N 值，包括 0 和 5（大于默认值）
        m = powers.shape[1]
        for n in range(6):
            v = vander(c, N=n)
            assert_array_equal(v, powers[:, m - n : m])
    # 定义测试函数 test_dtypes(self)
    def test_dtypes(self):
        # 创建包含整数的 numpy 数组 c，数据类型为 np.int8
        c = array([11, -12, 13], dtype=np.int8)
        # 对数组 c 进行 Vandermonde 矩阵变换，生成 v
        v = vander(c)
        # 预期的 Vandermonde 矩阵
        expected = np.array([[121, 11, 1], [144, -12, 1], [169, 13, 1]])
        # 使用 assert_array_equal 断言 v 与 expected 数组相等
        assert_array_equal(v, expected)

        # 创建包含复数的 numpy 数组 c
        c = array([1.0 + 1j, 1.0 - 1j])
        # 对数组 c 进行 Vandermonde 矩阵变换，指定 N=3，生成 v
        v = vander(c, N=3)
        # 预期的 Vandermonde 矩阵，包含复数
        expected = np.array([[2j, 1 + 1j, 1], [-2j, 1 - 1j, 1]])
        
        # 由于数据是浮点数，但值较小且可以精确匹配，使用 assert_array_equal 断言 v 与 expected 数组相等是安全的
        # 这比 assert_array_almost_equal 更合适，因为数据可以精确匹配
        assert_array_equal(v, expected)
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```