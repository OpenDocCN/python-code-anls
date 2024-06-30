# `D:\src\scipysrc\scipy\scipy\interpolate\tests\test_interpnd.py`

```
import os  # 导入操作系统接口模块
import sys  # 导入系统相关模块

import numpy as np  # 导入NumPy库，并简化命名为np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
                           suppress_warnings)  # 从NumPy测试模块导入多个断言函数
from pytest import raises as assert_raises  # 导入pytest中的raises函数，并简化命名为assert_raises
import pytest  # 导入pytest测试框架

from scipy._lib._testutils import check_free_memory  # 从SciPy测试工具模块导入函数
import scipy.interpolate.interpnd as interpnd  # 导入SciPy中的插值函数模块
import scipy.spatial._qhull as qhull  # 导入SciPy中的Qhull算法模块

import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
import threading  # 导入线程模块

_IS_32BIT = (sys.maxsize < 2**32)  # 判断是否为32位系统


def data_file(basename):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        'data', basename)  # 返回基于当前文件位置的数据文件路径


class TestLinearNDInterpolation:
    def test_smoketest(self):
        # Test at single points
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.float64)  # 创建包含坐标点的NumPy数组
        y = np.arange(x.shape[0], dtype=np.float64)  # 创建对应的测试值数组

        yi = interpnd.LinearNDInterpolator(x, y)(x)  # 使用LinearNDInterpolator进行插值计算
        assert_almost_equal(y, yi)  # 断言插值结果与预期值y几乎相等

    def test_smoketest_alternate(self):
        # Test at single points, alternate calling convention
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.float64)  # 创建包含坐标点的NumPy数组
        y = np.arange(x.shape[0], dtype=np.float64)  # 创建对应的测试值数组

        yi = interpnd.LinearNDInterpolator((x[:,0], x[:,1]), y)(x[:,0], x[:,1])  # 使用LinearNDInterpolator进行插值计算，使用不同的调用约定
        assert_almost_equal(y, yi)  # 断言插值结果与预期值y几乎相等

    def test_complex_smoketest(self):
        # Test at single points
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.float64)  # 创建包含坐标点的NumPy数组
        y = np.arange(x.shape[0], dtype=np.float64)  # 创建对应的测试值数组
        y = y - 3j*y  # 将测试值数组转换为复数形式

        yi = interpnd.LinearNDInterpolator(x, y)(x)  # 使用LinearNDInterpolator进行插值计算
        assert_almost_equal(y, yi)  # 断言插值结果与预期值y几乎相等

    def test_tri_input(self):
        # Test at single points
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.float64)  # 创建包含坐标点的NumPy数组
        y = np.arange(x.shape[0], dtype=np.float64)  # 创建对应的测试值数组
        y = y - 3j*y  # 将测试值数组转换为复数形式

        tri = qhull.Delaunay(x)  # 使用Qhull算法创建Delaunay三角剖分
        interpolator = interpnd.LinearNDInterpolator(tri, y)  # 创建LinearNDInterpolator插值对象
        yi = interpolator(x)  # 对x进行插值计算
        assert_almost_equal(y, yi)  # 断言插值结果与预期值y几乎相等
        assert interpolator.tri is tri  # 断言插值对象的三角剖分对象与创建的三角剖分对象相同
    def test_square(self):
        # 测试在正方形上使用重心插值，与手动实现进行比较

        # 定义正方形的顶点坐标和对应的值
        points = np.array([(0,0), (0,1), (1,1), (1,0)], dtype=np.float64)
        values = np.array([1., 2., -3., 5.], dtype=np.float64)

        # 注意事项：假设三角形是 (0, 1, 3) 和 (1, 2, 3)
        #
        #  1----2
        #  | \  |
        #  |  \ |
        #  0----3

        def ip(x, y):
            # 判断点 (x, y) 是否位于三角形 (0, 1, 3) 中
            t1 = (x + y <= 1)
            t2 = ~t1

            # 根据 t1 和 t2 分别提取 x, y 的值
            x1 = x[t1]
            y1 = y[t1]

            x2 = x[t2]
            y2 = y[t2]

            # 初始化结果数组 z
            z = 0*x

            # 计算三角形 (0, 1, 3) 中的插值结果
            z[t1] = (values[0]*(1 - x1 - y1)
                     + values[1]*y1
                     + values[3]*x1)

            # 计算三角形 (1, 2, 3) 中的插值结果
            z[t2] = (values[2]*(x2 + y2 - 1)
                     + values[1]*(1 - x2)
                     + values[3]*(1 - y2))
            return z

        # 生成网格点 (xx, yy)，确保广播到相同的形状
        xx, yy = np.broadcast_arrays(np.linspace(0, 1, 14)[:,None],
                                     np.linspace(0, 1, 14)[None,:])
        xx = xx.ravel()
        yy = yy.ravel()

        # 将 xx, yy 组成二维数组 xi，并复制一份
        xi = np.array([xx, yy]).T.copy()

        # 使用线性插值器 interpnd.LinearNDInterpolator 对 points, values 进行插值
        zi = interpnd.LinearNDInterpolator(points, values)(xi)

        # 断言插值结果 zi 与手动插值 ip(xx, yy) 相近
        assert_almost_equal(zi, ip(xx, yy))

    def test_smoketest_rescale(self):
        # 在单点上进行测试

        # 定义一组点 x 和对应的序号 y
        x = np.array([(0, 0), (-5, -5), (-5, 5), (5, 5), (2.5, 3)],
                     dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)

        # 使用 rescale=True 的线性插值器 interpnd.LinearNDInterpolator 进行插值
        yi = interpnd.LinearNDInterpolator(x, y, rescale=True)(x)

        # 断言插值结果 yi 与 y 相近
        assert_almost_equal(y, yi)

    def test_square_rescale(self):
        # 在带有重新缩放的矩形上测试重心插值，与没有重新缩放的实现进行比较

        # 定义矩形的顶点坐标和对应的值
        points = np.array([(0,0), (0,100), (10,100), (10,0)], dtype=np.float64)
        values = np.array([1., 2., -3., 5.], dtype=np.float64)

        # 生成网格点 (xx, yy)，确保广播到相同的形状
        xx, yy = np.broadcast_arrays(np.linspace(0, 10, 14)[:,None],
                                     np.linspace(0, 100, 14)[None,:])
        xx = xx.ravel()
        yy = yy.ravel()

        # 将 xx, yy 组成二维数组 xi，并复制一份
        xi = np.array([xx, yy]).T.copy()

        # 使用线性插值器 interpnd.LinearNDInterpolator 对 points, values 进行插值
        zi = interpnd.LinearNDInterpolator(points, values)(xi)

        # 使用 rescale=True 的线性插值器 interpnd.LinearNDInterpolator 对 points, values 进行插值
        zi_rescaled = interpnd.LinearNDInterpolator(points, values,
                rescale=True)(xi)

        # 断言插值结果 zi 和 zi_rescaled 相近
        assert_almost_equal(zi, zi_rescaled)

    def test_tripoints_input_rescale(self):
        # 在带有重新缩放的三角形上测试重心插值，与没有重新缩放的实现进行比较

        # 定义一组点 x 和对应的复数序号 y
        x = np.array([(0,0), (-5,-5), (-5,5), (5, 5), (2.5, 3)],
                     dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 3j*y  # 将 y 转换为复数

        # 使用 qhull.Delaunay 对 x 进行三角剖分
        tri = qhull.Delaunay(x)

        # 使用线性插值器 interpnd.LinearNDInterpolator 对 tri.points, y 进行插值
        yi = interpnd.LinearNDInterpolator(tri.points, y)(x)

        # 使用 rescale=True 的线性插值器 interpnd.LinearNDInterpolator 对 tri.points, y 进行插值
        yi_rescale = interpnd.LinearNDInterpolator(tri.points, y,
                rescale=True)(x)

        # 断言插值结果 yi 和 yi_rescale 相近
        assert_almost_equal(yi, yi_rescale)
    def test_tri_input_rescale(self):
        # Test at single points
        # 定义输入点集 x，包含五个二维坐标
        x = np.array([(0,0), (-5,-5), (-5,5), (5, 5), (2.5, 3)],
                     dtype=np.float64)
        # 创建对应的输出值 y，为每个点的复数值
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 3j*y

        # 使用输入点集 x 构建 Delaunay 三角网格
        tri = qhull.Delaunay(x)
        # 定义错误匹配信息
        match = ("Rescaling is not supported when passing a "
                 "Delaunay triangulation as ``points``.")
        # 断言在调用 LinearNDInterpolator 时会抛出 ValueError 异常并匹配错误信息
        with pytest.raises(ValueError, match=match):
            interpnd.LinearNDInterpolator(tri, y, rescale=True)(x)

    def test_pickle(self):
        # Test at single points
        # 设置随机种子
        np.random.seed(1234)
        # 生成 30 个随机二维坐标点集 x
        x = np.random.rand(30, 2)
        # 生成对应的复数随机值集 y
        y = np.random.rand(30) + 1j*np.random.rand(30)

        # 创建 LinearNDInterpolator 对象 ip，并通过 pickle 序列化和反序列化得到 ip2
        ip = interpnd.LinearNDInterpolator(x, y)
        ip2 = pickle.loads(pickle.dumps(ip))

        # 断言 ip 和 ip2 在点 (0.5, 0.5) 处的近似相等性
        assert_almost_equal(ip(0.5, 0.5), ip2(0.5, 0.5))

    @pytest.mark.slow
    @pytest.mark.skipif(_IS_32BIT, reason='it fails on 32-bit')
    def test_threading(self):
        # This test was taken from issue 8856
        # https://github.com/scipy/scipy/issues/8856
        # 检查可用内存是否足够
        check_free_memory(10000)

        # 创建径向和角向的刻度数组
        r_ticks = np.arange(0, 4200, 10)
        phi_ticks = np.arange(0, 4200, 10)
        r_grid, phi_grid = np.meshgrid(r_ticks, phi_ticks)

        # 定义线性插值函数 do_interp
        def do_interp(interpolator, slice_rows, slice_cols):
            # 创建网格坐标
            grid_x, grid_y = np.mgrid[slice_rows, slice_cols]
            # 执行插值计算
            res = interpolator((grid_x, grid_y))
            return res

        # 创建点集和对应的值集
        points = np.vstack((r_grid.ravel(), phi_grid.ravel())).T
        values = (r_grid * phi_grid).ravel()
        # 创建 LinearNDInterpolator 对象 interpolator
        interpolator = interpnd.LinearNDInterpolator(points, values)

        # 创建四个工作线程，每个线程执行不同的插值任务
        worker_thread_1 = threading.Thread(
            target=do_interp,
            args=(interpolator, slice(0, 2100), slice(0, 2100)))
        worker_thread_2 = threading.Thread(
            target=do_interp,
            args=(interpolator, slice(2100, 4200), slice(0, 2100)))
        worker_thread_3 = threading.Thread(
            target=do_interp,
            args=(interpolator, slice(0, 2100), slice(2100, 4200)))
        worker_thread_4 = threading.Thread(
            target=do_interp,
            args=(interpolator, slice(2100, 4200), slice(2100, 4200)))

        # 启动四个工作线程
        worker_thread_1.start()
        worker_thread_2.start()
        worker_thread_3.start()
        worker_thread_4.start()

        # 等待四个工作线程执行完毕
        worker_thread_1.join()
        worker_thread_2.join()
        worker_thread_3.join()
        worker_thread_4.join()
class TestEstimateGradients2DGlobal:
    def test_smoketest(self):
        x = np.array([(0, 0), (0, 2),
                      (1, 0), (1, 2), (0.25, 0.75), (0.6, 0.8)], dtype=float)
        tri = qhull.Delaunay(x)

        # Should be exact for linear functions, independent of triangulation

        # 定义四个测试函数及其梯度
        funcs = [
            (lambda x, y: 0*x + 1, (0, 0)),
            (lambda x, y: 0 + x, (1, 0)),
            (lambda x, y: -2 + y, (0, 1)),
            (lambda x, y: 3 + 3*x + 14.15*y, (3, 14.15))
        ]

        # 遍历每个测试函数及其期望梯度
        for j, (func, grad) in enumerate(funcs):
            z = func(x[:,0], x[:,1])  # 计算函数值
            dz = interpnd.estimate_gradients_2d_global(tri, z, tol=1e-6)  # 估计全局二维梯度

            # 断言梯度数组形状为 (6, 2)
            assert_equal(dz.shape, (6, 2))
            # 断言计算得到的梯度与期望梯度在一定容差范围内接近
            assert_allclose(dz, np.array(grad)[None,:] + 0*dz,
                            rtol=1e-5, atol=1e-5, err_msg="item %d" % j)

    def test_regression_2359(self):
        # 检查回归 --- 对于某些点集，梯度估计可能会进入无限循环
        points = np.load(data_file('estimate_gradients_hang.npy'))  # 加载数据点
        values = np.random.rand(points.shape[0])  # 随机生成值
        tri = qhull.Delaunay(points)  # 创建 Delaunay 三角网

        # 这应该不会陷入死循环
        with suppress_warnings() as sup:
            sup.filter(interpnd.GradientEstimationWarning,
                       "Gradient estimation did not converge")
            interpnd.estimate_gradients_2d_global(tri, values, maxiter=1)


class TestCloughTocher2DInterpolator:

    def _check_accuracy(self, func, x=None, tol=1e-6, alternate=False,
                        rescale=False, **kw):
        np.random.seed(1234)
        if x is None:
            x = np.array([(0, 0), (0, 1),
                          (1, 0), (1, 1), (0.25, 0.75), (0.6, 0.8),
                          (0.5, 0.2)],
                         dtype=float)

        if not alternate:
            ip = interpnd.CloughTocher2DInterpolator(x, func(x[:,0], x[:,1]),
                                                     tol=1e-6, rescale=rescale)
        else:
            ip = interpnd.CloughTocher2DInterpolator((x[:,0], x[:,1]),
                                                     func(x[:,0], x[:,1]),
                                                     tol=1e-6, rescale=rescale)

        p = np.random.rand(50, 2)

        if not alternate:
            a = ip(p)  # 插值器对点集 p 进行插值
        else:
            a = ip(p[:,0], p[:,1])  # 使用另一种调用方式进行插值

        b = func(p[:,0], p[:,1])  # 计算函数在点集 p 上的真实值

        try:
            assert_allclose(a, b, **kw)  # 断言插值结果与真实值在容差范围内接近
        except AssertionError:
            print("_check_accuracy: abs(a-b):", abs(a - b))
            print("ip.grad:", ip.grad)
            raise
    # 定义测试线性函数的简单测试用例
    def test_linear_smoketest(self):
        # 定义四个线性函数并存储在列表中
        funcs = [
            lambda x, y: 0*x + 1,            # f(x, y) = 1
            lambda x, y: 0 + x,              # f(x, y) = x
            lambda x, y: -2 + y,             # f(x, y) = y - 2
            lambda x, y: 3 + 3*x + 14.15*y,  # f(x, y) = 3 + 3*x + 14.15*y
        ]

        # 对每个函数进行测试
        for j, func in enumerate(funcs):
            # 调用函数检查精度，不同的误差衡量标准和错误消息
            self._check_accuracy(func, tol=1e-13, atol=1e-7, rtol=1e-7,
                                 err_msg="Function %d" % j)
            # 使用另一种配置再次检查精度
            self._check_accuracy(func, tol=1e-13, atol=1e-7, rtol=1e-7,
                                 alternate=True,
                                 err_msg="Function (alternate) %d" % j)
            # 检查是否需要重新缩放
            self._check_accuracy(func, tol=1e-13, atol=1e-7, rtol=1e-7,
                                 err_msg="Function (rescaled) %d" % j, rescale=True)
            # 使用另一种配置再次检查精度和重新缩放
            self._check_accuracy(func, tol=1e-13, atol=1e-7, rtol=1e-7,
                                 alternate=True, rescale=True,
                                 err_msg="Function (alternate, rescaled) %d" % j)

    # 定义测试二次函数的简单测试用例
    def test_quadratic_smoketest(self):
        # 定义四个二次函数并存储在列表中
        funcs = [
            lambda x, y: x**2,          # f(x, y) = x^2
            lambda x, y: y**2,          # f(x, y) = y^2
            lambda x, y: x**2 - y**2,   # f(x, y) = x^2 - y^2
            lambda x, y: x*y,           # f(x, y) = x * y
        ]

        # 对每个函数进行测试
        for j, func in enumerate(funcs):
            # 调用函数检查精度，使用不同的误差衡量标准和错误消息
            self._check_accuracy(func, tol=1e-9, atol=0.22, rtol=0,
                                 err_msg="Function %d" % j)
            # 如果需要，重新缩放后再次检查精度
            self._check_accuracy(func, tol=1e-9, atol=0.22, rtol=0,
                                 err_msg="Function %d" % j, rescale=True)

    # 定义测试三角输入的简单测试用例
    def test_tri_input(self):
        # 在单个点进行测试
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 3j*y

        # 使用点集 x 创建 Delaunay 三角网
        tri = qhull.Delaunay(x)
        # 使用 CloughTocher2DInterpolator 进行插值
        yi = interpnd.CloughTocher2DInterpolator(tri, y)(x)
        # 断言 yi 应该与 y 几乎相等
        assert_almost_equal(y, yi)

    # 定义测试三角输入并重新缩放的简单测试用例
    def test_tri_input_rescale(self):
        # 在单个点进行测试
        x = np.array([(0,0), (-5,-5), (-5,5), (5, 5), (2.5, 3)],
                     dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 3j*y

        # 使用点集 x 创建 Delaunay 三角网
        tri = qhull.Delaunay(x)
        # 试图使用重新缩放参数创建 CloughTocher2DInterpolator，预计引发异常
        match = ("Rescaling is not supported when passing a "
                 "Delaunay triangulation as ``points``.")
        with pytest.raises(ValueError, match=match):
            interpnd.CloughTocher2DInterpolator(tri, y, rescale=True)(x)
    def test_tripoints_input_rescale(self):
        # 测试单个点
        x = np.array([(0,0), (-5,-5), (-5,5), (5, 5), (2.5, 3)],
                     dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 3j*y

        # 使用点集 x 构建 Delaunay 三角网格
        tri = qhull.Delaunay(x)
        # 使用 CloughTocher2DInterpolator 插值器计算插值
        yi = interpnd.CloughTocher2DInterpolator(tri.points, y)(x)
        # 使用启用了 rescale 选项的插值器计算插值
        yi_rescale = interpnd.CloughTocher2DInterpolator(tri.points, y, rescale=True)(x)
        # 断言两种插值结果几乎相等
        assert_almost_equal(yi, yi_rescale)

    @pytest.mark.fail_slow(5)
    def test_dense(self):
        # 对于密集网格应更加准确
        funcs = [
            lambda x, y: x**2,
            lambda x, y: y**2,
            lambda x, y: x**2 - y**2,
            lambda x, y: x*y,
            lambda x, y: np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
        ]

        np.random.seed(4321)  # 使用不同于检查的种子！
        # 创建包含随机点的网格
        grid = np.r_[np.array([(0,0), (0,1), (1,0), (1,1)], dtype=float),
                     np.random.rand(30*30, 2)]

        # 对每个函数进行迭代测试
        for j, func in enumerate(funcs):
            # 使用 _check_accuracy 方法检查精度，不启用 rescale
            self._check_accuracy(func, x=grid, tol=1e-9, atol=5e-3, rtol=1e-2,
                                 err_msg="Function %d" % j)
            # 使用 _check_accuracy 方法检查精度，启用 rescale
            self._check_accuracy(func, x=grid, tol=1e-9, atol=5e-3, rtol=1e-2,
                                 err_msg="Function %d" % j, rescale=True)

    def test_wrong_ndim(self):
        # 创建错误维度的输入数据
        x = np.random.randn(30, 3)
        y = np.random.randn(30)
        # 断言应引发 ValueError 异常
        assert_raises(ValueError, interpnd.CloughTocher2DInterpolator, x, y)

    def test_pickle(self):
        # 测试单个点
        np.random.seed(1234)
        x = np.random.rand(30, 2)
        y = np.random.rand(30) + 1j*np.random.rand(30)

        # 创建 CloughTocher2DInterpolator 对象并进行序列化和反序列化
        ip = interpnd.CloughTocher2DInterpolator(x, y)
        ip2 = pickle.loads(pickle.dumps(ip))

        # 断言序列化前后对象在给定点上的函数值几乎相等
        assert_almost_equal(ip(0.5, 0.5), ip2(0.5, 0.5))
    def test_boundary_tri_symmetry(self):
        # 测试函数：test_boundary_tri_symmetry，用于验证边界三角形的对称性保持

        # 创建一个等边三角形的点集和值集
        points = np.array([(0, 0), (1, 0), (0.5, np.sqrt(3)/2)])
        values = np.array([1, 0, 0])

        # 使用 CloughTocher2DInterpolator 类创建插值对象 ip
        ip = interpnd.CloughTocher2DInterpolator(points, values)

        # 将插值对象 ip 的梯度设为零
        ip.grad[...] = 0

        # 验证插值在三角形的两个点 p1 和 p2 处的对称性
        alpha = 0.3
        p1 = np.array([0.5 * np.cos(alpha), 0.5 * np.sin(alpha)])
        p2 = np.array([0.5 * np.cos(np.pi/3 - alpha), 0.5 * np.sin(np.pi/3 - alpha)])

        v1 = ip(p1)  # 计算插值在点 p1 处的值
        v2 = ip(p2)  # 计算插值在点 p2 处的值
        assert_allclose(v1, v2)  # 验证 v1 和 v2 的值接近（对称性）

        # 验证插值对仿射变换不变
        np.random.seed(1)
        A = np.random.randn(2, 2)
        b = np.random.randn(2)

        # 对原始点集进行仿射变换
        points = A.dot(points.T).T + b[None, :]
        p1 = A.dot(p1) + b
        p2 = A.dot(p2) + b

        # 使用变换后的点集创建新的插值对象 ip
        ip = interpnd.CloughTocher2DInterpolator(points, values)
        ip.grad[...] = 0

        # 计算变换后的点 p1 和 p2 处的插值值，并与之前的 v1, v2 进行比较
        w1 = ip(p1)
        w2 = ip(p2)
        assert_allclose(w1, v1)  # 验证 w1 与 v1 的值接近
        assert_allclose(w2, v2)  # 验证 w2 与 v2 的值接近
```