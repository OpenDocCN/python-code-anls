# `D:\src\scipysrc\scipy\scipy\interpolate\tests\test_ndgriddata.py`

```
import numpy as np  # 导入 NumPy 库，并简称为 np
from numpy.testing import assert_equal, assert_array_equal, assert_allclose  # 导入 NumPy 测试相关的断言函数
import pytest  # 导入 pytest 测试框架
from pytest import raises as assert_raises  # 导入 pytest 的 raises 函数并简称为 assert_raises

from scipy.interpolate import (griddata, NearestNDInterpolator,  # 从 scipy.interpolate 导入插值相关函数和类
                               LinearNDInterpolator,
                               CloughTocher2DInterpolator)

parametrize_interpolators = pytest.mark.parametrize(  # 使用 pytest 的 parametrize 标记，用于参数化测试
    "interpolator", [NearestNDInterpolator, LinearNDInterpolator,
                     CloughTocher2DInterpolator]
)

class TestGriddata:  # 定义测试类 TestGriddata
    def test_fill_value(self):  # 定义测试方法 test_fill_value
        x = [(0,0), (0,1), (1,0)]  # 定义 x 数组
        y = [1, 2, 3]  # 定义 y 数组

        yi = griddata(x, y, [(1,1), (1,2), (0,0)], fill_value=-1)  # 使用 griddata 进行插值计算，指定填充值为 -1
        assert_array_equal(yi, [-1., -1, 1])  # 断言插值结果与预期结果相等

        yi = griddata(x, y, [(1,1), (1,2), (0,0)])  # 使用 griddata 进行插值计算，未指定填充值
        assert_array_equal(yi, [np.nan, np.nan, 1])  # 断言插值结果与预期结果相等，预期包含 NaN 值

    def test_alternative_call(self):  # 定义测试方法 test_alternative_call
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],  # 定义二维数组 x
                     dtype=np.float64)
        y = (np.arange(x.shape[0], dtype=np.float64)[:,None]  # 创建一个列向量 y
             + np.array([0,1])[None,:])

        for method in ('nearest', 'linear', 'cubic'):  # 遍历插值方法
            for rescale in (True, False):  # 遍历是否重新缩放的选项
                msg = repr((method, rescale))  # 创建描述消息
                yi = griddata((x[:,0], x[:,1]), y, (x[:,0], x[:,1]), method=method,  # 使用 griddata 进行插值计算
                              rescale=rescale)
                assert_allclose(y, yi, atol=1e-14, err_msg=msg)  # 断言插值结果与预期结果相近

    def test_multivalue_2d(self):  # 定义测试方法 test_multivalue_2d
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],  # 定义二维数组 x
                     dtype=np.float64)
        y = (np.arange(x.shape[0], dtype=np.float64)[:,None]  # 创建一个列向量 y
             + np.array([0,1])[None,:])

        for method in ('nearest', 'linear', 'cubic'):  # 遍历插值方法
            for rescale in (True, False):  # 遍历是否重新缩放的选项
                msg = repr((method, rescale))  # 创建描述消息
                yi = griddata(x, y, x, method=method, rescale=rescale)  # 使用 griddata 进行插值计算
                assert_allclose(y, yi, atol=1e-14, err_msg=msg)  # 断言插值结果与预期结果相近

    def test_multipoint_2d(self):  # 定义测试方法 test_multipoint_2d
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],  # 定义二维数组 x
                     dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)  # 创建一个一维数组 y

        xi = x[:,None,:] + np.array([0,0,0])[None,:,None]  # 创建三维数组 xi

        for method in ('nearest', 'linear', 'cubic'):  # 遍历插值方法
            for rescale in (True, False):  # 遍历是否重新缩放的选项
                msg = repr((method, rescale))  # 创建描述消息
                yi = griddata(x, y, xi, method=method, rescale=rescale)  # 使用 griddata 进行插值计算

                assert_equal(yi.shape, (5, 3), err_msg=msg)  # 断言插值结果的形状符合预期
                assert_allclose(yi, np.tile(y[:,None], (1, 3)),  # 断言插值结果与预期结果相近
                                atol=1e-14, err_msg=msg)
    # 定义一个名为 test_complex_2d 的测试方法，用于测试复杂的二维情况
    def test_complex_2d(self):
        # 创建一个包含五个二维点的 NumPy 数组 x，数据类型为 np.float64
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.float64)
        # 创建一个与 x 大小相同的一维数组 y，数据类型为 np.float64
        y = np.arange(x.shape[0], dtype=np.float64)
        # 将 y 转换为复数形式，并做翻转后减去 2j*y[::-1]
        y = y - 2j*y[::-1]

        # 创建 xi 数组，其形状是 x[:,None,:] 加上一个形状为 (1,3,1) 的数组
        xi = x[:,None,:] + np.array([0,0,0])[None,:,None]

        # 嵌套循环，对每种插值方法 ('nearest', 'linear', 'cubic') 和每种重缩放选项 (True, False) 进行测试
        for method in ('nearest', 'linear', 'cubic'):
            for rescale in (True, False):
                # 生成描述消息，显示当前测试的方法和重缩放选项
                msg = repr((method, rescale))
                # 使用 griddata 函数对 x, y, xi 进行插值计算得到 yi
                yi = griddata(x, y, xi, method=method, rescale=rescale)

                # 断言 yi 的形状应为 (5, 3)，否则输出错误消息 msg
                assert_equal(yi.shape, (5, 3), err_msg=msg)
                # 断言 yi 应接近于 np.tile(y[:,None], (1, 3))，允许的误差为 1e-14，否则输出错误消息 msg
                assert_allclose(yi, np.tile(y[:,None], (1, 3)),
                                atol=1e-14, err_msg=msg)

    # 定义一个名为 test_1d 的测试方法，用于测试一维情况
    def test_1d(self):
        # 创建包含六个一维点的 NumPy 数组 x 和对应的数组 y
        x = np.array([1, 2.5, 3, 4.5, 5, 6])
        y = np.array([1, 2, 0, 3.9, 2, 1])

        # 嵌套循环，对每种插值方法 ('nearest', 'linear', 'cubic') 进行测试
        for method in ('nearest', 'linear', 'cubic'):
            # 断言 griddata 函数对 x, y, x 进行插值计算后结果应接近于 y，允许的误差为 1e-14
            assert_allclose(griddata(x, y, x, method=method), y,
                            err_msg=method, atol=1e-14)
            # 断言 griddata 函数对 x.reshape(6, 1), y, x 进行插值计算后结果应接近于 y，允许的误差为 1e-14
            assert_allclose(griddata(x.reshape(6, 1), y, x, method=method), y,
                            err_msg=method, atol=1e-14)
            # 断言 griddata 函数对 (x,), y, (x,) 进行插值计算后结果应接近于 y，允许的误差为 1e-14
            assert_allclose(griddata((x,), y, (x,), method=method), y,
                            err_msg=method, atol=1e-14)

    # 定义一个名为 test_1d_borders 的测试方法，用于测试一维情况中的边界情况
    def test_1d_borders(self):
        # 创建包含六个一维点的 NumPy 数组 x 和对应的数组 y，以及两个超出范围的插值点 xi 和其应有的 yi_should
        x = np.array([1, 2.5, 3, 4.5, 5, 6])
        y = np.array([1, 2, 0, 3.9, 2, 1])
        xi = np.array([0.9, 6.5])
        yi_should = np.array([1.0, 1.0])

        method = 'nearest'
        # 断言 griddata 函数对 x, y, xi 进行插值计算后结果应接近于 yi_should，允许的误差为 1e-14
        assert_allclose(griddata(x, y, xi,
                                 method=method), yi_should,
                        err_msg=method,
                        atol=1e-14)
        # 断言 griddata 函数对 x.reshape(6, 1), y, xi 进行插值计算后结果应接近于 yi_should，允许的误差为 1e-14
        assert_allclose(griddata(x.reshape(6, 1), y, xi,
                                 method=method), yi_should,
                        err_msg=method,
                        atol=1e-14)
        # 断言 griddata 函数对 (x,), y, (xi,) 进行插值计算后结果应接近于 yi_should，允许的误差为 1e-14
        assert_allclose(griddata((x,), y, (xi,),
                                 method=method), yi_should,
                        err_msg=method,
                        atol=1e-14)

    # 定义一个名为 test_1d_unsorted 的测试方法，用于测试未排序的一维情况
    def test_1d_unsorted(self):
        # 创建包含六个未排序一维点的 NumPy 数组 x 和对应的数组 y
        x = np.array([2.5, 1, 4.5, 5, 6, 3])
        y = np.array([1, 2, 0, 3.9, 2, 1])

        # 嵌套循环，对每种插值方法 ('nearest', 'linear', 'cubic') 进行测试
        for method in ('nearest', 'linear', 'cubic'):
            # 断言 griddata 函数对 x, y, x 进行插值计算后结果应接近于 y，允许的误差为 1e-10
            assert_allclose(griddata(x, y, x, method=method), y,
                            err_msg=method, atol=1e-10)
            # 断言 griddata 函数对 x.reshape(6, 1), y, x 进行插值计算后结果应接近于 y，允许的误差为 1e-10
            assert_allclose(griddata(x.reshape(6, 1), y, x, method=method), y,
                            err_msg=method, atol=1e-10)
            # 断言 griddata 函数对 (x,), y, (x,) 进行插值计算后结果应接近于 y，允许的误差为 1e-10
            assert_allclose(griddata((x,), y, (x,), method=method), y,
                            err_msg=method, atol=1e-10)
    def test_square_rescale_manual(self):
        # 创建包含五个点的二维数组，表示原始坐标点
        points = np.array([(0,0), (0,100), (10,100), (10,0), (1, 5)], dtype=np.float64)
        # 创建包含五个点的二维数组，表示重新缩放后的坐标点
        points_rescaled = np.array([(0,0), (0,1), (1,1), (1,0), (0.1, 0.05)],
                                   dtype=np.float64)
        # 创建包含五个浮点数的一维数组，表示每个点对应的值
        values = np.array([1., 2., -3., 5., 9.], dtype=np.float64)

        # 创建两个广播后的数组 xx 和 yy，分别在 x 和 y 方向均匀分布
        xx, yy = np.broadcast_arrays(np.linspace(0, 10, 14)[:,None],
                                     np.linspace(0, 100, 14)[None,:])
        # 将 xx 和 yy 拉平成一维数组
        xx = xx.ravel()
        yy = yy.ravel()
        # 创建 xi 数组，包含 xx 和 yy 的转置，复制一份副本
        xi = np.array([xx, yy]).T.copy()

        # 针对三种插值方法分别进行测试
        for method in ('nearest', 'linear', 'cubic'):
            # 设置错误消息为当前方法名
            msg = method
            # 使用 griddata 对 points_rescaled 进行插值计算，得到 zi 数组
            zi = griddata(points_rescaled, values, xi/np.array([10, 100.]),
                          method=method)
            # 使用 griddata 对 points 进行插值计算，并开启 rescale 参数，得到 zi_rescaled 数组
            zi_rescaled = griddata(points, values, xi, method=method,
                                   rescale=True)
            # 断言 zi 和 zi_rescaled 在给定的误差范围内相等
            assert_allclose(zi, zi_rescaled, err_msg=msg,
                            atol=1e-12)

    def test_xi_1d(self):
        # 检查 1-D xi 被解释为一个坐标
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.float64)
        # 创建与 x 形状相同的一维浮点数数组 y
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 2j*y[::-1]

        # 创建一维数组 xi 包含两个坐标点
        xi = np.array([0.5, 0.5])

        # 针对三种插值方法分别进行测试
        for method in ('nearest', 'linear', 'cubic'):
            # 使用 griddata 对 x, y 和 xi 进行插值计算，得到 p1 和 p2
            p1 = griddata(x, y, xi, method=method)
            p2 = griddata(x, y, xi[None,:], method=method)
            # 断言 p1 和 p2 在误差范围内相等
            assert_allclose(p1, p2, err_msg=method)

            # 创建包含一个和三个坐标点的一维数组 xi1 和 xi3
            xi1 = np.array([0.5])
            xi3 = np.array([0.5, 0.5, 0.5])
            # 断言对 xi1 和 xi3 使用 griddata 会引发 ValueError
            assert_raises(ValueError, griddata, x, y, xi1,
                          method=method)
            assert_raises(ValueError, griddata, x, y, xi3,
                          method=method)
class TestNearestNDInterpolator:
    def test_nearest_options(self):
        # 测试 NearestNDInterpolator 是否接受 cKDTree 选项
        npts, nd = 4, 3
        x = np.arange(npts*nd).reshape((npts, nd))  # 创建一个二维数组 x
        y = np.arange(npts)  # 创建一个一维数组 y
        nndi = NearestNDInterpolator(x, y)  # 创建 NearestNDInterpolator 对象

        opts = {'balanced_tree': False, 'compact_nodes': False}  # 设置选项字典 opts
        nndi_o = NearestNDInterpolator(x, y, tree_options=opts)  # 使用选项创建另一个 NearestNDInterpolator 对象
        assert_allclose(nndi(x), nndi_o(x), atol=1e-14)  # 断言两个对象的输出在误差容限内相等

    def test_nearest_list_argument(self):
        nd = np.array([[0, 0, 0, 0, 1, 0, 1],
                       [0, 0, 0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1, 1, 2]])
        d = nd[:, 3:]  # 从 nd 中选择部分列形成新数组 d

        # z 是 np.array
        NI = NearestNDInterpolator((d[0], d[1]), d[2])  # 创建 NearestNDInterpolator 对象 NI
        assert_array_equal(NI([0.1, 0.9], [0.1, 0.9]), [0, 2])  # 断言 NI 对 query 点的输出与期望值相等

        # z 是列表
        NI = NearestNDInterpolator((d[0], d[1]), list(d[2]))  # 创建另一个 NearestNDInterpolator 对象 NI
        assert_array_equal(NI([0.1, 0.9], [0.1, 0.9]), [0, 2])  # 断言 NI 对 query 点的输出与期望值相等

    def test_nearest_query_options(self):
        nd = np.array([[0, 0.5, 0, 1],
                       [0, 0, 0.5, 1],
                       [0, 1, 1, 2]])
        delta = 0.1
        query_points = [0 + delta, 1 + delta], [0 + delta, 1 + delta]

        # case 1 - query max_dist 小于 query 点到 nd 的最近距离
        NI = NearestNDInterpolator((nd[0], nd[1]), nd[2])  # 创建 NearestNDInterpolator 对象 NI
        distance_upper_bound = np.sqrt(delta ** 2 + delta ** 2) - 1e-7  # 设置最大距离上限
        assert_array_equal(NI(query_points, distance_upper_bound=distance_upper_bound),
                           [np.nan, np.nan])  # 断言 NI 对 query 点的输出与期望值相等，预期为 NaN

        # case 2 - query p 是 inf，预期返回 [0, 2]
        distance_upper_bound = np.sqrt(delta ** 2 + delta ** 2) - 1e-7  # 设置最大距离上限
        p = np.inf  # 设置 p 为无穷大
        assert_array_equal(
            NI(query_points, distance_upper_bound=distance_upper_bound, p=p),
            [0, 2]
        )  # 断言 NI 对 query 点的输出与期望值相等

        # case 3 - query max_dist 大于最大距离，预期返回非 np.nan
        distance_upper_bound = np.sqrt(delta ** 2 + delta ** 2) + 1e-7  # 设置较大的最大距离上限
        assert_array_equal(
            NI(query_points, distance_upper_bound=distance_upper_bound),
            [0, 2]
        )  # 断言 NI 对 query 点的输出与期望值相等

    def test_nearest_query_valid_inputs(self):
        nd = np.array([[0, 1, 0, 1],
                       [0, 0, 1, 1],
                       [0, 1, 1, 2]])
        NI = NearestNDInterpolator((nd[0], nd[1]), nd[2])  # 创建 NearestNDInterpolator 对象 NI
        with assert_raises(TypeError):  # 断言捕获 TypeError 异常
            NI([0.5, 0.5], query_options="not a dictionary")  # 调用 NI，传递错误的 query_options 参数


class TestNDInterpolators:
    @parametrize_interpolators
    # 定义一个测试方法，用于测试插值器的广播输入功能，使用插值器参数interpolator
    def test_broadcastable_input(self, interpolator):
        # input data
        np.random.seed(0)
        # 创建长度为10的随机数组x和y
        x = np.random.random(10)
        y = np.random.random(10)
        # 计算x和y的欧几里得范数，即sqrt(x^2 + y^2)，存储在z中
        z = np.hypot(x, y)

        # x-y grid for interpolation
        # 生成在[min(x), max(x)]范围内的均匀分布的数组X
        X = np.linspace(min(x), max(x))
        # 生成在[min(y), max(y)]范围内的均匀分布的数组Y
        Y = np.linspace(min(y), max(y))
        # 使用X和Y创建网格点坐标矩阵
        X, Y = np.meshgrid(X, Y)
        # 将X和Y展平并转置，得到XY数组，每行代表一个二维坐标点
        XY = np.vstack((X.ravel(), Y.ravel())).T
        # 使用插值器interpolator和(x, y)对初始化插值器interp
        interp = interpolator(list(zip(x, y)), z)
        
        # single array input
        # 对XY中的每个点进行插值计算，返回插值结果interp_points0
        interp_points0 = interp(XY)
        
        # tuple input
        # 将(X, Y)作为元组输入插值器，返回插值结果interp_points1
        interp_points1 = interp((X, Y))
        
        # 使用(X, 0.0)作为元组输入插值器，返回插值结果interp_points2
        interp_points2 = interp((X, 0.0))
        
        # broadcastable input
        # 使用X, Y作为广播输入插值器，返回插值结果interp_points3
        interp_points3 = interp(X, Y)
        
        # 使用X, 0.0作为广播输入插值器，返回插值结果interp_points4
        interp_points4 = interp(X, 0.0)

        # 断言确保所有插值结果的大小相等
        assert_equal(interp_points0.size ==
                     interp_points1.size ==
                     interp_points2.size ==
                     interp_points3.size ==
                     interp_points4.size, True)

    @parametrize_interpolators
    # 用于测试插值器的只读性质，使用插值器参数interpolator
    def test_read_only(self, interpolator):
        # input data
        np.random.seed(0)
        # 创建一个形状为(10, 2)的随机二维数组xy
        xy = np.random.random((10, 2))
        # 分别从xy中提取x和y坐标
        x, y = xy[:, 0], xy[:, 1]
        # 计算x和y的欧几里得范数，存储在z中
        z = np.hypot(x, y)

        # interpolation points
        # 创建一个形状为(50, 2)的随机二维数组XY，表示插值点
        XY = np.random.random((50, 2))

        # 将xy, z, XY数组设置为只读模式，防止在插值过程中被修改
        xy.setflags(write=False)
        z.setflags(write=False)
        XY.setflags(write=False)

        # 使用插值器interpolator和xy, z初始化插值器interp
        interp = interpolator(xy, z)
        # 对XY中的点进行插值计算，由于xy, z为只读，确保不会在计算过程中被修改
        interp(XY)
```