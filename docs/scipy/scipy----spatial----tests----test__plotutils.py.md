# `D:\src\scipysrc\scipy\scipy\spatial\tests\test__plotutils.py`

```
import pytest  # 导入 pytest 库，用于编写和运行测试
import numpy as np  # 导入 NumPy 库，并使用 np 别名
from numpy.testing import assert_, assert_array_equal, assert_allclose  # 从 NumPy 测试模块导入断言函数

try:
    import matplotlib  # 尝试导入 matplotlib 库
    matplotlib.rcParams['backend'] = 'Agg'  # 设置 matplotlib 的后端为 'Agg'
    import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，并使用 plt 别名
    has_matplotlib = True  # 标记 matplotlib 是否可用
except Exception:
    has_matplotlib = False  # 如果导入出错，则标记 matplotlib 不可用

from scipy.spatial import \  # 导入 scipy.spatial 库的相关模块
     delaunay_plot_2d, voronoi_plot_2d, convex_hull_plot_2d, \
     Delaunay, Voronoi, ConvexHull  # 导入具体的函数和类


@pytest.mark.skipif(not has_matplotlib, reason="Matplotlib not available")
class TestPlotting:  # 定义测试类 TestPlotting，用于测试绘图相关功能
    points = [(0,0), (0,1), (1,0), (1,1)]  # 定义测试点集合

    def test_delaunay(self):
        # Smoke test
        fig = plt.figure()  # 创建一个新的 matplotlib 图形对象
        obj = Delaunay(self.points)  # 创建 Delaunay 三角网格对象
        s_before = obj.simplices.copy()  # 备份原始的三角形面片数据
        r = delaunay_plot_2d(obj, ax=fig.gca())  # 绘制二维 Delaunay 图，并获取返回的图形对象
        assert_array_equal(obj.simplices, s_before)  # 断言三角形面片数据未被修改
        assert_(r is fig)  # 断言返回的图形对象是之前创建的图形对象
        delaunay_plot_2d(obj, ax=fig.gca())  # 再次绘制二维 Delaunay 图

    def test_voronoi(self):
        # Smoke test
        fig = plt.figure()  # 创建一个新的 matplotlib 图形对象
        obj = Voronoi(self.points)  # 创建 Voronoi 图对象
        r = voronoi_plot_2d(obj, ax=fig.gca())  # 绘制 Voronoi 图，并获取返回的图形对象
        assert_(r is fig)  # 断言返回的图形对象是之前创建的图形对象
        voronoi_plot_2d(obj)  # 再次绘制 Voronoi 图
        voronoi_plot_2d(obj, show_vertices=False)  # 再次绘制 Voronoi 图，但不显示顶点

    def test_convex_hull(self):
        # Smoke test
        fig = plt.figure()  # 创建一个新的 matplotlib 图形对象
        tri = ConvexHull(self.points)  # 创建凸包对象
        r = convex_hull_plot_2d(tri, ax=fig.gca())  # 绘制二维凸包，并获取返回的图形对象
        assert_(r is fig)  # 断言返回的图形对象是之前创建的图形对象
        convex_hull_plot_2d(tri)  # 再次绘制二维凸包

    def test_gh_19653(self):
        # aspect ratio sensitivity of voronoi_plot_2d
        # infinite Voronoi edges
        points = np.array([[245.059986986012, 10.971011721360075],
                           [320.49044143557785, 10.970258360366753],
                           [239.79023081978914, 13.108487516946218],
                           [263.38325791238833, 12.93241352743668],
                           [219.53334398353175, 13.346107628161008]])
        vor = Voronoi(points)  # 根据给定点集创建 Voronoi 图对象
        fig = voronoi_plot_2d(vor)  # 绘制二维 Voronoi 图，并获取返回的图形对象
        ax = fig.gca()  # 获取当前的坐标轴对象
        infinite_segments = ax.collections[1].get_segments()  # 获取无限远 Voronoi 边的段落集合
        expected_segments = np.array([[[282.77256, -254.76904],
                                       [282.729714, -4544.744698]],
                                      [[282.77256014, -254.76904029],
                                       [430.08561382, 4032.67658742]],
                                      [[229.26733285,  -20.39957514],
                                       [-168.17167404, -4291.92545966]],
                                      [[289.93433364, 5151.40412217],
                                       [330.40553385, 9441.18887532]]])
        assert_allclose(infinite_segments, expected_segments)  # 断言获取的无限远边段落集合与预期的段落集合近似
    def test_gh_19653_smaller_aspect(self):
        # 定义一个测试方法，用于测试更小的宽高比情况下的合理行为
        points = np.array([[24.059986986012, 10.971011721360075],
                           [32.49044143557785, 10.970258360366753],
                           [23.79023081978914, 13.108487516946218],
                           [26.38325791238833, 12.93241352743668],
                           [21.53334398353175, 13.346107628161008]])
        # 创建一个包含五个点坐标的 NumPy 数组
        vor = Voronoi(points)
        # 使用这些点生成 Voronoi 图
        fig = voronoi_plot_2d(vor)
        # 绘制 2D Voronoi 图，并将结果存储在 fig 中
        ax = fig.gca()
        # 获取当前图形的轴对象
        infinite_segments = ax.collections[1].get_segments()
        # 获取绘制的 Voronoi 图中无限边的线段集合
        expected_segments = np.array([[[28.274979, 8.335027],
                                       [28.270463, -42.19763338]],
                                      [[28.27497869, 8.33502697],
                                       [43.73223829, 56.44555501]],
                                      [[22.51805823, 11.8621754],
                                       [-12.09266506, -24.95694485]],
                                      [[29.53092448, 78.46952378],
                                       [33.82572726, 128.81934455]]])
        # 预期的无限边线段集合，用于断言比较
        assert_allclose(infinite_segments, expected_segments)
        # 使用 NumPy 的 assert_allclose 函数验证实际结果与预期结果的近似性
```