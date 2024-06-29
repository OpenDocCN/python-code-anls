# `.\numpy\numpy\lib\tests\test_twodim_base.py`

```
"""Test functions for matrix module

"""
# 导入必要的模块和函数
from numpy.testing import (
    assert_equal, assert_array_equal, assert_array_max_ulp,
    assert_array_almost_equal, assert_raises, assert_
)
from numpy import (
    arange, add, fliplr, flipud, zeros, ones, eye, array, diag, histogram2d,
    tri, mask_indices, triu_indices, triu_indices_from, tril_indices,
    tril_indices_from, vander,
)
import numpy as np  # 导入 NumPy 库

import pytest  # 导入 pytest 模块


def get_mat(n):
    # 创建一个 n x n 的矩阵，其中每个元素是其行和列索引的和
    data = arange(n)
    data = add.outer(data, data)
    return data


class TestEye:
    def test_basic(self):
        # 测试生成单位矩阵
        assert_equal(eye(4),
                     array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]))

        # 测试生成指定数据类型的单位矩阵
        assert_equal(eye(4, dtype='f'),
                     array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], 'f'))

        # 测试生成布尔类型的单位矩阵
        assert_equal(eye(3) == 1,
                     eye(3, dtype=bool))

    def test_uint64(self):
        # 测试对于 uint64 类型的单位矩阵生成
        # gh-9982 的回归测试
        assert_equal(eye(np.uint64(2), dtype=int), array([[1, 0], [0, 1]]))
        assert_equal(eye(np.uint64(2), M=np.uint64(4), k=np.uint64(1)),
                     array([[0, 1, 0, 0], [0, 0, 1, 0]]))

    def test_diag(self):
        # 测试生成具有偏移量 k 的单位对角矩阵
        assert_equal(eye(4, k=1),
                     array([[0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0]]))

        assert_equal(eye(4, k=-1),
                     array([[0, 0, 0, 0],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0]]))

    def test_2d(self):
        # 测试生成二维单位矩阵
        assert_equal(eye(4, 3),
                     array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [0, 0, 0]]))

        assert_equal(eye(3, 4),
                     array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0]]))

    def test_diag2d(self):
        # 测试生成具有偏移量 k 的二维单位对角矩阵
        assert_equal(eye(3, 4, k=2),
                     array([[0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0]]))

        assert_equal(eye(4, 3, k=-2),
                     array([[0, 0, 0],
                            [0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0]]))
    # 测试函数，验证 `eye` 函数的边界条件
    def test_eye_bounds(self):
        # 断言：生成 2x2 的单位矩阵，主对角线向上偏移1个位置
        assert_equal(eye(2, 2, 1), [[0, 1], [0, 0]])
        # 断言：生成 2x2 的单位矩阵，主对角线向下偏移1个位置
        assert_equal(eye(2, 2, -1), [[0, 0], [1, 0]])
        # 断言：生成 2x2 的单位矩阵，但超过矩阵维度，返回全0矩阵
        assert_equal(eye(2, 2, 2), [[0, 0], [0, 0]])
        # 断言：生成 2x2 的单位矩阵，主对角线向下偏移超过矩阵维度，返回全0矩阵
        assert_equal(eye(2, 2, -2), [[0, 0], [0, 0]])
        # 断言：生成 3x2 的单位矩阵，但超过矩阵维度，返回全0矩阵
        assert_equal(eye(3, 2, 2), [[0, 0], [0, 0], [0, 0]])
        # 断言：生成 3x2 的单位矩阵，主对角线向上偏移1个位置
        assert_equal(eye(3, 2, 1), [[0, 1], [0, 0], [0, 0]])
        # 断言：生成 3x2 的单位矩阵，主对角线向下偏移1个位置，且超过矩阵维度，返回全0矩阵
        assert_equal(eye(3, 2, -1), [[0, 0], [1, 0], [0, 1]])
        # 断言：生成 3x2 的单位矩阵，主对角线向下偏移2个位置，且超过矩阵维度，返回全0矩阵
        assert_equal(eye(3, 2, -2), [[0, 0], [0, 0], [1, 0]])
        # 断言：生成 3x2 的单位矩阵，主对角线向下偏移3个位置，超过矩阵维度，返回全0矩阵
        assert_equal(eye(3, 2, -3), [[0, 0], [0, 0], [0, 0]])

    # 测试函数，验证 `eye` 函数对字符串类型的处理
    def test_strings(self):
        # 断言：生成 2x2 的单位矩阵，数据类型为字节串，字符串长度为3
        assert_equal(eye(2, 2, dtype='S3'),
                     [[b'1', b''], [b'', b'1']])

    # 测试函数，验证 `eye` 函数对布尔类型的处理
    def test_bool(self):
        # 断言：生成 2x2 的单位矩阵，数据类型为布尔型
        assert_equal(eye(2, 2, dtype=bool), [[True, False], [False, True]])

    # 测试函数，验证 `eye` 函数对矩阵顺序的处理
    def test_order(self):
        # 生成 4x3 的单位矩阵，主对角线向上偏移1个位置，以 C 顺序存储
        mat_c = eye(4, 3, k=-1)
        # 生成 4x3 的单位矩阵，主对角线向上偏移1个位置，以 Fortran（F）顺序存储
        mat_f = eye(4, 3, k=-1, order='F')
        # 断言：验证两种顺序下生成的矩阵内容相同
        assert_equal(mat_c, mat_f)
        # 断言：验证以 C 顺序存储的矩阵标志
        assert mat_c.flags.c_contiguous
        # 断言：验证以 C 顺序存储的矩阵不是以 Fortran 顺序存储
        assert not mat_c.flags.f_contiguous
        # 断言：验证以 Fortran 顺序存储的矩阵不是以 C 顺序存储
        assert not mat_f.flags.c_contiguous
        # 断言：验证以 Fortran 顺序存储的矩阵标志
        assert mat_f.flags.f_contiguous
class TestDiag:
    # 测试对角向量情况
    def test_vector(self):
        # 创建一个整型的数组，包含 0 到 400 之间的数，步长为 100
        vals = (100 * arange(5)).astype('l')
        # 创建一个5x5的全零数组
        b = zeros((5, 5))
        # 遍历范围为5的循环
        for k in range(5):
            # 在对角线上分别赋值
            b[k, k] = vals[k]
        # 断言对角线函数对vals和b的值相等
        assert_equal(diag(vals), b)
        # 创建一个7x7的全零数组
        b = zeros((7, 7))
        # 复制b数组
        c = b.copy()
        # 遍历范围为5的循环
        for k in range(5):
            # 在对角线上的不同位置赋值
            b[k, k + 2] = vals[k]
            c[k + 2, k] = vals[k]
        # 断言对角线函数对vals和b的值相等
        assert_equal(diag(vals, k=2), b)
        # 断言对角线函数对vals和c的值相等
        assert_equal(diag(vals, k=-2), c)

    # 测试对角矩阵情况
    def test_matrix(self, vals=None):
        # 如果vals为空，则获取一个5x5的矩阵并转换为整型数组
        if vals is None:
            vals = (100 * get_mat(5) + 1).astype('l')
        # 创建一个包含5个元素的全零数组
        b = zeros((5,))
        # 遍历范围为5的循环
        for k in range(5):
            # 在对角线上分别赋值
            b[k] = vals[k, k]
        # 断言对角线函数对vals和b的值相等
        assert_equal(diag(vals), b)
        # 将b数组元素全部置零
        b = b * 0
        # 遍历范围为3的循环
        for k in range(3):
            # 在对角线上的不同位置赋值
            b[k] = vals[k, k + 2]
        # 断言对角线函数对vals和b的前三个值相等
        assert_equal(diag(vals, 2), b[:3])
        # 遍历范围为3的循环
        for k in range(3):
            # 在对角线上的不同位置赋值
            b[k] = vals[k + 2, k]
        # 断言对角线函数对vals和b的前三个值相等
        assert_equal(diag(vals, -2), b[:3])

    # 测试Fortran顺序的情况
    def test_fortran_order(self):
        # 创建一个Fortran顺序的5x5矩阵，并转换为整型数组
        vals = array((100 * get_mat(5) + 1), order='F', dtype='l')
        # 调用test_matrix函数进行测试
        self.test_matrix(vals)

    # 测试对角线边界情况
    def test_diag_bounds(self):
        # 创建一个列表A
        A = [[1, 2], [3, 4], [5, 6]]
        # 断言对角线函数应返回空列表
        assert_equal(diag(A, k=2), [])
        # 断言对角线函数应返回包含值2的列表
        assert_equal(diag(A, k=1), [2])
        # 断言对角线函数应返回包含值1和4的列表
        assert_equal(diag(A, k=0), [1, 4])
        # 断言对角线函数应返回包含值3和6的列表
        assert_equal(diag(A, k=-1), [3, 6])
        # 断言对角线函数应返回包含值5的列表
        assert_equal(diag(A, k=-2), [5])
        # 断言对角线函数应返回空列表
        assert_equal(diag(A, k=-3), [])

    # 测试失败的情况
    def test_failure(self):
        # 断言当传入一个三重嵌套的列表时，应引发ValueError异常
        assert_raises(ValueError, diag, [[[1]]])


class TestFliplr:
    # 测试基本情况
    def test_basic(self):
        # 断言当传入一个全1的4x4数组时，应引发ValueError异常
        assert_raises(ValueError, fliplr, ones(4))
        # 获取一个4x4矩阵
        a = get_mat(4)
        # 创建一个在水平方向翻转后的矩阵b
        b = a[:, ::-1]
        # 断言翻转函数对a和b的值相等
        assert_equal(fliplr(a), b)
        # 创建一个列表a
        a = [[0, 1, 2],
             [3, 4, 5]]
        # 创建一个在水平方向翻转后的列表b
        b = [[2, 1, 0],
             [5, 4, 3]]
        # 断言翻转函数对a和b的值相等
        assert_equal(fliplr(a), b)


class TestFlipud:
    # 测试基本情况
    def test_basic(self):
        # 获取一个4x4矩阵
        a = get_mat(4)
        # 创建一个在垂直方向翻转后的矩阵b
        b = a[::-1, :]
        # 断言翻转函数对a和b的值相等
        assert_equal(flipud(a), b)
        # 创建一个列表a
        a = [[0, 1, 2],
             [3, 4, 5]]
        # 创建一个在垂直方向翻转后的列表b
        b = [[3, 4, 5],
             [0, 1, 2]]
        # 断言翻转函数对a和b的值相等
        assert_equal(flipud(a), b)


class TestHistogram2d:
    pass
    # 测试简单的二维直方图计算，用于测试基本功能
    def test_simple(self):
        x = array(
            [0.41702200, 0.72032449, 1.1437481e-4, 0.302332573, 0.146755891])
        y = array(
            [0.09233859, 0.18626021, 0.34556073, 0.39676747, 0.53881673])
        # 在指定区间内生成 x 和 y 的边界
        xedges = np.linspace(0, 1, 10)
        yedges = np.linspace(0, 1, 10)
        # 计算二维直方图，并获取直方图数据
        H = histogram2d(x, y, (xedges, yedges))[0]
        # 预期的正确结果
        answer = array(
            [[0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # 断言直方图数据与预期结果的相等性
        assert_array_equal(H.T, answer)
        # 重新计算二维直方图，使用不同的参数形式
        H = histogram2d(x, y, xedges)[0]
        # 再次断言直方图数据与预期结果的相等性
        assert_array_equal(H.T, answer)
        # 执行直方图计算，并获取直方图数据、以及新的边界
        H, xedges, yedges = histogram2d(list(range(10)), list(range(10)))
        # 断言直方图数据与单位矩阵的相等性
        assert_array_equal(H, eye(10, 10))
        # 断言 x 边界与预期结果的相等性
        assert_array_equal(xedges, np.linspace(0, 9, 11))
        # 断言 y 边界与预期结果的相等性
        assert_array_equal(yedges, np.linspace(0, 9, 11))

    # 测试不对称数据的二维直方图计算，验证不对称情况下的功能
    def test_asym(self):
        x = array([1, 1, 2, 3, 4, 4, 4, 5])
        y = array([1, 3, 2, 0, 1, 2, 3, 4])
        # 计算带有额外参数的二维直方图，包括范围、密度等设置
        H, xed, yed = histogram2d(
            x, y, (6, 5), range=[[0, 6], [0, 5]], density=True)
        # 预期的正确结果
        answer = array(
            [[0., 0, 0, 0, 0],
             [0, 1, 0, 1, 0],
             [0, 0, 1, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 1]])
        # 断言直方图数据与预期结果的近似相等性
        assert_array_almost_equal(H, answer/8., 3)
        # 断言 x 边界与预期结果的相等性
        assert_array_equal(xed, np.linspace(0, 6, 7))
        # 断言 y 边界与预期结果的相等性
        assert_array_equal(yed, np.linspace(0, 5, 6))

    # 测试密度计算的二维直方图功能
    def test_density(self):
        x = array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        y = array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        # 计算密度归一化的二维直方图，使用自定义的边界
        H, xed, yed = histogram2d(
            x, y, [[1, 2, 3, 5], [1, 2, 3, 5]], density=True)
        # 预期的正确结果
        answer = array([[1, 1, .5],
                        [1, 1, .5],
                        [.5, .5, .25]])/9.
        # 断言直方图数据与预期结果的近似相等性
        assert_array_almost_equal(H, answer, 3)

    # 测试全部数据为异常值的情况
    def test_all_outliers(self):
        r = np.random.rand(100) + 1. + 1e6  # histogramdd rounds by decimal=6
        # 计算异常值情况下的二维直方图，期望所有元素为零
        H, xed, yed = histogram2d(r, r, (4, 5), range=([0, 1], [0, 1]))
        # 断言直方图数据全为零
        assert_array_equal(H, 0)

    # 测试空数据集的情况
    def test_empty(self):
        # 测试空数据的二维直方图计算，使用自定义的边界
        a, edge1, edge2 = histogram2d([], [], bins=([0, 1], [0, 1]))
        # 断言直方图数据与预期结果的最大误差不超过单位最小浮点数
        assert_array_max_ulp(a, array([[0.]]))

        # 再次测试空数据的二维直方图计算，使用相同的边界数量
        a, edge1, edge2 = histogram2d([], [], bins=4)
        # 断言直方图数据与全零矩阵的最大误差不超过单位最小浮点数
        assert_array_max_ulp(a, np.zeros((4, 4)))
    # 定义测试方法，用于验证不同二进制参数组合的直方图计算
    def test_binparameter_combination(self):
        # 定义输入的两个一维数组 x 和 y
        x = array(
            [0, 0.09207008, 0.64575234, 0.12875982, 0.47390599,
             0.59944483, 1])
        y = array(
            [0, 0.14344267, 0.48988575, 0.30558665, 0.44700682,
             0.15886423, 1])
        # 定义直方图的边界
        edges = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
        # 调用 histogram2d 函数计算二维直方图，返回直方图 H 和边界 xe, ye
        H, xe, ye = histogram2d(x, y, (edges, 4))
        # 预期的二维数组结果
        answer = array(
            [[2., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 1., 0., 0.],
             [1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 1.]])
        # 使用 assert_array_equal 断言函数检查 H 是否与预期结果 answer 相等
        assert_array_equal(H, answer)
        # 使用 assert_array_equal 断言函数检查 ye 是否与预期结果相等
        assert_array_equal(ye, array([0., 0.25, 0.5, 0.75, 1]))
        
        # 重新计算直方图，但是交换了 bins 的顺序
        H, xe, ye = histogram2d(x, y, (4, edges))
        # 更新预期的二维数组结果
        answer = array(
            [[1., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
             [0., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
        # 使用 assert_array_equal 断言函数检查 H 是否与更新后的预期结果 answer 相等
        assert_array_equal(H, answer)
        # 使用 assert_array_equal 断言函数检查 xe 是否与预期结果相等
        assert_array_equal(xe, array([0., 0.25, 0.5, 0.75, 1]))

    # 定义测试方法，用于验证直方图函数的分发机制
    def test_dispatch(self):
        # 定义一个类 ShouldDispatch，实现了 __array_function__ 方法
        class ShouldDispatch:
            def __array_function__(self, function, types, args, kwargs):
                # 返回传入的 types, args 和 kwargs
                return types, args, kwargs

        # 初始化一个列表 xy
        xy = [1, 2]
        # 创建 ShouldDispatch 的实例 s_d
        s_d = ShouldDispatch()
        
        # 调用 histogram2d 函数，验证对 s_d 和 xy 的直方图计算调用是否被分发
        r = histogram2d(s_d, xy)
        # 使用 assert_ 函数断言 r 是否等于预期的结果元组
        assert_(r == ((ShouldDispatch,), (s_d, xy), {}))
        
        # 再次调用 histogram2d 函数，验证对 xy 和 s_d 的直方图计算调用是否被分发
        r = histogram2d(xy, s_d)
        # 使用 assert_ 函数断言 r 是否等于预期的结果元组
        assert_(r == ((ShouldDispatch,), (xy, s_d), {}))
        
        # 再次调用 histogram2d 函数，验证对 xy 和 xy（使用 bins=s_d 参数）的直方图计算调用是否被分发
        r = histogram2d(xy, xy, bins=s_d)
        # 使用 assert_ 函数断言 r 是否等于预期的结果元组
        assert_(r, ((ShouldDispatch,), (xy, xy), dict(bins=s_d)))
        
        # 再次调用 histogram2d 函数，验证对 xy 和 xy（使用 bins=[s_d, 5] 参数）的直方图计算调用是否被分发
        r = histogram2d(xy, xy, bins=[s_d, 5])
        # 使用 assert_ 函数断言 r 是否等于预期的结果元组
        assert_(r, ((ShouldDispatch,), (xy, xy), dict(bins=[s_d, 5])))
        
        # 使用 assert_raises 函数验证调用 histogram2d 函数时，传入 bins=[s_d] 参数是否会引发异常
        assert_raises(Exception, histogram2d, xy, xy, bins=[s_d])
        
        # 再次调用 histogram2d 函数，验证对 xy 和 xy（使用 weights=s_d 参数）的直方图计算调用是否被分发
        r = histogram2d(xy, xy, weights=s_d)
        # 使用 assert_ 函数断言 r 是否等于预期的结果元组
        assert_(r, ((ShouldDispatch,), (xy, xy), dict(weights=s_d)))

    # 使用 pytest 的参数化装饰器，定义测试方法，用于验证不同长度的输入数组引发的 ValueError 异常
    @pytest.mark.parametrize(("x_len", "y_len"), [(10, 11), (20, 19)])
    def test_bad_length(self, x_len, y_len):
        # 创建两个长度不同的全 1 数组 x 和 y
        x, y = np.ones(x_len), np.ones(y_len)
        # 使用 assertRaises 函数验证调用 histogram2d 函数时，传入不同长度的 x 和 y 是否会引发 ValueError 异常
        with pytest.raises(ValueError,
                           match='x and y must have the same length.'):
            histogram2d(x, y)
# 定义一个测试类 TestTri
class TestTri:
    # 测试函数，测试 np.tri 函数的返回值是否正确
    def test_dtype(self):
        # 创建一个三阶单位矩阵，用于与 np.tri(3) 的结果比较
        out = array([[1, 0, 0],
                     [1, 1, 0],
                     [1, 1, 1]])
        # 断言 np.tri(3) 返回的结果与 out 相等
        assert_array_equal(tri(3), out)
        # 断言使用 bool 类型调用 np.tri(3) 的结果与 out.astype(bool) 相等
        assert_array_equal(tri(3, dtype=bool), out.astype(bool))


# 测试函数，测试 np.tril 和 np.triu 在二维数组上的行为
def test_tril_triu_ndim2():
    # 遍历所有浮点数和整数类型
    for dtype in np.typecodes['AllFloat'] + np.typecodes['AllInteger']:
        # 创建一个全为 1 的二维数组，数据类型为当前循环的类型 dtype
        a = np.ones((2, 2), dtype=dtype)
        # 使用 np.tril 函数生成 a 的下三角矩阵
        b = np.tril(a)
        # 使用 np.triu 函数生成 a 的上三角矩阵
        c = np.triu(a)
        # 断言 np.tril(a) 的结果与预期的下三角矩阵 [[1, 0], [1, 1]] 相等
        assert_array_equal(b, [[1, 0], [1, 1]])
        # 断言 np.triu(a) 的结果与 np.tril(a) 的转置相等
        assert_array_equal(c, b.T)
        # 断言 np.tril(a) 和 np.triu(a) 的数据类型与 a 相同
        assert_equal(b.dtype, a.dtype)
        assert_equal(c.dtype, a.dtype)


# 测试函数，测试 np.tril 和 np.triu 在三维数组上的行为
def test_tril_triu_ndim3():
    # 遍历所有浮点数和整数类型
    for dtype in np.typecodes['AllFloat'] + np.typecodes['AllInteger']:
        # 创建一个三维数组 a，数据类型为当前循环的类型 dtype
        a = np.array([
            [[1, 1], [1, 1]],
            [[1, 1], [1, 0]],
            [[1, 1], [0, 0]],
            ], dtype=dtype)
        # 预期的 a 的下三角矩阵
        a_tril_desired = np.array([
            [[1, 0], [1, 1]],
            [[1, 0], [1, 0]],
            [[1, 0], [0, 0]],
            ], dtype=dtype)
        # 预期的 a 的上三角矩阵
        a_triu_desired = np.array([
            [[1, 1], [0, 1]],
            [[1, 1], [0, 0]],
            [[1, 1], [0, 0]],
            ], dtype=dtype)
        # 使用 np.triu 函数生成 a 的上三角矩阵
        a_triu_observed = np.triu(a)
        # 使用 np.tril 函数生成 a 的下三角矩阵
        a_tril_observed = np.tril(a)
        # 断言 np.triu(a) 的结果与预期的上三角矩阵相等
        assert_array_equal(a_triu_observed, a_triu_desired)
        # 断言 np.tril(a) 的结果与预期的下三角矩阵相等
        assert_array_equal(a_tril_observed, a_tril_desired)
        # 断言 np.triu(a) 和 np.tril(a) 的数据类型与 a 相同
        assert_equal(a_triu_observed.dtype, a.dtype)
        assert_equal(a_tril_observed.dtype, a.dtype)


# 测试函数，测试 np.tril 和 np.triu 处理含有无穷大值的数组时的行为
def test_tril_triu_with_inf():
    # 创建一个包含无穷大值的数组 arr
    arr = np.array([[1, 1, np.inf],
                    [1, 1, 1],
                    [np.inf, 1, 1]])
    # 预期的 arr 的下三角矩阵
    out_tril = np.array([[1, 0, 0],
                         [1, 1, 0],
                         [np.inf, 1, 1]])
    # 预期的 arr 的上三角矩阵
    out_triu = out_tril.T
    # 断言 np.triu(arr) 的结果与预期的上三角矩阵相等
    assert_array_equal(np.triu(arr), out_triu)
    # 断言 np.tril(arr) 的结果与预期的下三角矩阵相等
    assert_array_equal(np.tril(arr), out_tril)


# 测试函数，测试 np.tril 和 np.triu 返回值的数据类型与输入数组相同
def test_tril_triu_dtype():
    # 遍历所有数据类型
    for c in np.typecodes['All']:
        # 跳过 'V' 类型
        if c == 'V':
            continue
        # 创建一个全为 0 的 3x3 数组 arr，数据类型为当前循环的类型 c
        arr = np.zeros((3, 3), dtype=c)
        # 断言 np.triu(arr) 的数据类型与 arr 相同
        assert_equal(np.triu(arr).dtype, arr.dtype)
        # 断言 np.tril(arr) 的数据类型与 arr 相同
        assert_equal(np.tril(arr).dtype, arr.dtype)

    # 检查特殊情况
    # 创建一个 datetime64 类型的数组 arr
    arr = np.array([['2001-01-01T12:00', '2002-02-03T13:56'],
                    ['2004-01-01T12:00', '2003-01-03T13:45']],
                   dtype='datetime64')
    # 断言 np.triu(arr) 的数据类型与 arr 相同
    assert_equal(np.triu(arr).dtype, arr.dtype)
    # 断言 np.tril(arr) 的数据类型与 arr 相同
    assert_equal(np.tril(arr).dtype, arr.dtype)

    # 创建一个结构化数据类型为 'f4,f4' 的全为 0 的 3x3 数组 arr
    arr = np.zeros((3, 3), dtype='f4,f4')
    # 断言 np.triu(arr) 的数据类型与 arr 相同
    assert_equal(np.triu(arr).dtype, arr.dtype)
    # 断言 np.tril(arr) 的数据类型与 arr 相同
    assert_equal(np.tril(arr).dtype, arr.dtype)


# 测试函数，测试 mask_indices 函数的行为
def test_mask_indices():
    # 简单测试，无偏移量
    # 调用 mask_indices(3, np.triu) 函数，返回上三角矩阵的非零元素索引
    iu = mask_indices(3, np.triu)
    # 创建一个 3x3 的数组 a
    a = np.arange(9).reshape(3, 3)
    # 断言 a[iu] 的结果与预期的非零元素索引数组相等
    assert_array_equal(a[iu], array([0, 1, 2, 4, 5, 8]))
    
    # 带偏移量的测试
    # 调用 mask_indices(3, np.triu, 1) 函数，返回上三角矩阵的非零元素索引（带偏移量）
    iu1 = mask_indices(3, np.triu, 1)
    # 断言 a[iu1] 的结果与预期的非零元素索引数组相等
    assert_array_equal(a[iu1], array([1, 2, 5]))


# 测试函数，测试 tril_indices 函数
def test_tril_indices():
    # 创建一个表示不带偏移的下三角矩阵的索引
    il1 = tril_indices(4)
    # 创建一个表示带有偏移的下三角矩阵的索引
    il2 = tril_indices(4, k=2)
    # 创建一个表示带有更大行数限制的下三角矩阵的索引
    il3 = tril_indices(4, m=5)
    # 创建一个表示带有偏移和更大行数限制的下三角矩阵的索引
    il4 = tril_indices(4, k=2, m=5)

    # 创建一个4x4的NumPy数组
    a = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    # 创建一个4x5的NumPy数组，元素为1到20
    b = np.arange(1, 21).reshape(4, 5)

    # 对数组a进行索引操作，并使用assert_array_equal进行断言
    assert_array_equal(a[il1],
                       np.array([1, 5, 6, 9, 10, 11, 13, 14, 15, 16]))
    # 对数组b进行索引操作，并使用assert_array_equal进行断言
    assert_array_equal(b[il3],
                       np.array([1, 6, 7, 11, 12, 13, 16, 17, 18, 19]))

    # 对数组a进行赋值操作，并使用assert_array_equal进行断言
    a[il1] = -1
    assert_array_equal(a,
                       np.array([[-1, 2, 3, 4],
                                 [-1, -1, 7, 8],
                                 [-1, -1, -1, 12],
                                 [-1, -1, -1, -1]]))
    # 对数组b进行赋值操作，并使用assert_array_equal进行断言
    b[il3] = -1
    assert_array_equal(b,
                       np.array([[-1, 2, 3, 4, 5],
                                 [-1, -1, 8, 9, 10],
                                 [-1, -1, -1, 14, 15],
                                 [-1, -1, -1, -1, 20]]))
    # 对数组a进行赋值操作，覆盖几乎整个数组（主对角线右侧的两个对角线）
    a[il2] = -10
    assert_array_equal(a,
                       np.array([[-10, -10, -10, 4],
                                 [-10, -10, -10, -10],
                                 [-10, -10, -10, -10],
                                 [-10, -10, -10, -10]]))
    # 对数组b进行赋值操作，覆盖几乎整个数组（带有偏移和更大行数限制的两个对角线）
    b[il4] = -10
    assert_array_equal(b,
                       np.array([[-10, -10, -10, 4, 5],
                                 [-10, -10, -10, -10, 10],
                                 [-10, -10, -10, -10, -10],
                                 [-10, -10, -10, -10, -10]]))
class TestTriuIndices:
    def test_triu_indices(self):
        # 生成一个包含主对角线及其以上部分的上三角索引数组
        iu1 = triu_indices(4)
        # 生成一个包含主对角线及其以上部分、从主对角线向右偏移2的上三角索引数组
        iu2 = triu_indices(4, k=2)
        # 生成一个包含主对角线及其以上部分、矩阵行数为5的上三角索引数组
        iu3 = triu_indices(4, m=5)
        # 生成一个包含主对角线及其以上部分、从主对角线向右偏移2、矩阵行数为5的上三角索引数组

        a = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12],
                      [13, 14, 15, 16]])
        b = np.arange(1, 21).reshape(4, 5)

        # 用iu1索引数组获取a的元素，与指定数组比较并断言相等
        assert_array_equal(a[iu1],
                           np.array([1, 2, 3, 4, 6, 7, 8, 11, 12, 16]))
        # 用iu3索引数组获取b的元素，与指定数组比较并断言相等
        assert_array_equal(b[iu3],
                           np.array([1, 2, 3, 4, 5, 7, 8, 9,
                                     10, 13, 14, 15, 19, 20]))

        # 使用iu1索引数组修改a的元素为-1，并与指定数组比较并断言相等
        a[iu1] = -1
        assert_array_equal(a,
                           np.array([[-1, -1, -1, -1],
                                     [5, -1, -1, -1],
                                     [9, 10, -1, -1],
                                     [13, 14, 15, -1]]))
        # 使用iu3索引数组修改b的元素为-1，并与指定数组比较并断言相等
        b[iu3] = -1
        assert_array_equal(b,
                           np.array([[-1, -1, -1, -1, -1],
                                     [6, -1, -1, -1, -1],
                                     [11, 12, -1, -1, -1],
                                     [16, 17, 18, -1, -1]]))

        # 使用iu2索引数组修改a的元素为-10，并与指定数组比较并断言相等
        a[iu2] = -10
        assert_array_equal(a,
                           np.array([[-1, -1, -10, -10],
                                     [5, -1, -1, -10],
                                     [9, 10, -1, -1],
                                     [13, 14, 15, -1]]))
        # 使用iu4索引数组修改b的元素为-10，并与指定数组比较并断言相等
        b[iu4] = -10
        assert_array_equal(b,
                           np.array([[-1, -1, -10, -10, -10],
                                     [6, -1, -1, -10, -10],
                                     [11, 12, -1, -1, -10],
                                     [16, 17, 18, -1, -1]]))


class TestTrilIndicesFrom:
    def test_exceptions(self):
        # 对于维度不为2的矩阵，引发值错误异常
        assert_raises(ValueError, tril_indices_from, np.ones((2,)))
        # 对于维度不为2的3维矩阵，引发值错误异常
        assert_raises(ValueError, tril_indices_from, np.ones((2, 2, 2)))
        # 对于维度为(2, 3)的矩阵，可能引发值错误异常，但已注释掉


class TestTriuIndicesFrom:
    def test_exceptions(self):
        # 对于维度不为2的矩阵，引发值错误异常
        assert_raises(ValueError, triu_indices_from, np.ones((2,)))
        # 对于维度不为2的3维矩阵，引发值错误异常
        assert_raises(ValueError, triu_indices_from, np.ones((2, 2, 2)))
        # 对于维度为(2, 3)的矩阵，可能引发值错误异常，但已注释掉
    # 定义一个测试方法，用于测试 `vander` 函数的基本功能
    def test_basic(self):
        # 创建一个包含整数的 NumPy 数组
        c = np.array([0, 1, -2, 3])
        # 调用 `vander` 函数生成一个 Vandermonde 矩阵
        v = vander(c)
        # 创建一个预期的幂次矩阵，用于与 `vander` 函数生成的结果进行比较
        powers = np.array([[0, 0, 0, 0, 1],
                           [1, 1, 1, 1, 1],
                           [16, -8, 4, -2, 1],
                           [81, 27, 9, 3, 1]])
        # 检查默认的 N 值是否符合预期
        assert_array_equal(v, powers[:, 1:])
        # 检查一系列不同的 N 值，包括 0 和 5（大于默认值）
        m = powers.shape[1]
        for n in range(6):
            # 调用 `vander` 函数，指定不同的 N 值
            v = vander(c, N=n)
            # 检查生成的 Vandermonde 矩阵是否符合预期
            assert_array_equal(v, powers[:, m-n:m])

    # 定义一个测试方法，用于测试 `vander` 函数处理不同数据类型的情况
    def test_dtypes(self):
        # 创建一个包含 int8 类型数据的 NumPy 数组
        c = np.array([11, -12, 13], dtype=np.int8)
        # 调用 `vander` 函数生成一个 Vandermonde 矩阵
        v = vander(c)
        # 创建一个预期的 Vandermonde 矩阵，用于与 `vander` 函数生成的结果进行比较
        expected = np.array([[121, 11, 1],
                             [144, -12, 1],
                             [169, 13, 1]])
        # 检查生成的 Vandermonde 矩阵是否符合预期
        assert_array_equal(v, expected)

        # 创建一个包含复数的 NumPy 数组
        c = np.array([1.0+1j, 1.0-1j])
        # 调用 `vander` 函数生成一个 Vandermonde 矩阵，指定 N 值为 3
        v = vander(c, N=3)
        # 创建一个预期的 Vandermonde 矩阵，用于与 `vander` 函数生成的结果进行比较
        expected = np.array([[2j, 1+1j, 1],
                             [-2j, 1-1j, 1]])
        # 由于数据是浮点型，但值是小整数，使用 `assert_array_equal` 进行比较应该是安全的
        # （而不是使用 `assert_array_almost_equal`）
        assert_array_equal(v, expected)
```