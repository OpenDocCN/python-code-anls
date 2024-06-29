# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_triangulation.py`

```py
import numpy as np  # 导入 NumPy 库，并使用 np 别名
from numpy.testing import (  # 导入 NumPy 测试工具模块的多个函数
    assert_array_equal, assert_array_almost_equal, assert_array_less)
import numpy.ma.testutils as matest  # 导入 NumPy 掩码测试工具

import pytest  # 导入 pytest 测试框架

import matplotlib as mpl  # 导入 matplotlib 库并使用 mpl 别名
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块并使用 plt 别名
import matplotlib.tri as mtri  # 导入 matplotlib 的三角网格处理模块
from matplotlib.path import Path  # 从 matplotlib.path 模块导入 Path 类
from matplotlib.testing.decorators import image_comparison, check_figures_equal  # 导入测试相关的装饰器

class TestTriangulationParams:  # 定义测试类 TestTriangulationParams

    x = [-1, 0, 1, 0]  # 定义 x 坐标数组
    y = [0, -1, 0, 1]  # 定义 y 坐标数组
    triangles = [[0, 1, 2], [0, 2, 3]]  # 定义三角形索引数组
    mask = [False, True]  # 定义掩码数组，用于标记哪些点被掩盖

    @pytest.mark.parametrize('args, kwargs, expected', [  # 使用 pytest 的参数化装饰器定义测试参数
        ([x, y], {}, [x, y, None, None]),  # 参数组合：无三角形、无掩码
        ([x, y, triangles], {}, [x, y, triangles, None]),  # 参数组合：指定三角形、无掩码
        ([x, y], dict(triangles=triangles), [x, y, triangles, None]),  # 参数组合：无三角形、指定掩码
        ([x, y], dict(mask=mask), [x, y, None, mask]),  # 参数组合：无三角形、指定掩码
        ([x, y, triangles], dict(mask=mask), [x, y, triangles, mask]),  # 参数组合：指定三角形、指定掩码
        ([x, y], dict(triangles=triangles, mask=mask), [x, y, triangles, mask])  # 参数组合：指定三角形、指定掩码
    ])
    def test_extract_triangulation_params(self, args, kwargs, expected):  # 定义测试函数 test_extract_triangulation_params
        other_args = [1, 2]  # 其他位置参数
        other_kwargs = {'a': 3, 'b': '4'}  # 其他关键字参数
        x_, y_, triangles_, mask_, args_, kwargs_ = \  # 调用 Triangulation._extract_triangulation_params 方法解析参数
            mtri.Triangulation._extract_triangulation_params(
                args + other_args, {**kwargs, **other_kwargs})
        x, y, triangles, mask = expected  # 期望的结果
        assert x_ is x  # 断言 x 坐标与期望一致
        assert y_ is y  # 断言 y 坐标与期望一致
        assert_array_equal(triangles_, triangles)  # 断言三角形索引数组与期望一致
        assert mask_ is mask  # 断言掩码数组与期望一致
        assert args_ == other_args  # 断言位置参数与期望一致
        assert kwargs_ == other_kwargs  # 断言关键字参数与期望一致


def test_extract_triangulation_positional_mask():  # 定义测试函数 test_extract_triangulation_positional_mask
    # mask cannot be passed positionally
    mask = [True]  # 创建掩码数组
    args = [[0, 2, 1], [0, 0, 1], [[0, 1, 2]], mask]  # 创建参数列表
    x_, y_, triangles_, mask_, args_, kwargs_ = \  # 调用 Triangulation._extract_triangulation_params 方法解析参数
        mtri.Triangulation._extract_triangulation_params(args, {})
    assert mask_ is None  # 断言掩码为 None
    assert args_ == [mask]  # 断言位置参数与期望一致
    # the positional mask must be caught downstream because this must pass
    # unknown args through


def test_triangulation_init():  # 定义测试函数 test_triangulation_init
    x = [-1, 0, 1, 0]  # 定义 x 坐标数组
    y = [0, -1, 0, 1]  # 定义 y 坐标数组
    with pytest.raises(ValueError, match="x and y must be equal-length"):  # 使用 pytest 断言捕获 ValueError 异常
        mtri.Triangulation(x, [1, 2])  # 初始化 Triangulation 对象
    with pytest.raises(  # 使用 pytest 断言捕获 ValueError 异常
            ValueError,
            match=r"triangles must be a \(N, 3\) int array, but found shape "
                  r"\(3,\)"):
        mtri.Triangulation(x, y, [0, 1, 2])  # 初始化 Triangulation 对象
    with pytest.raises(  # 使用 pytest 断言捕获 ValueError 异常
            ValueError,
            match=r"triangles must be a \(N, 3\) int array, not 'other'"):
        mtri.Triangulation(x, y, 'other')  # 初始化 Triangulation 对象
    with pytest.raises(ValueError, match="found value 99"):  # 使用 pytest 断言捕获 ValueError 异常
        mtri.Triangulation(x, y, [[0, 1, 99]])  # 初始化 Triangulation 对象
    with pytest.raises(ValueError, match="found value -1"):  # 使用 pytest 断言捕获 ValueError 异常
        mtri.Triangulation(x, y, [[0, 1, -1]])  # 初始化 Triangulation 对象


def test_triangulation_set_mask():  # 定义测试函数 test_triangulation_set_mask
    x = [-1, 0, 1, 0]  # 定义 x 坐标数组
    y = [0, -1, 0, 1]  # 定义 y 坐标数组
    triangles = [[0, 1, 2], [2, 3, 0]]  # 定义三角形索引数组
    triang = mtri.Triangulation(x, y, triangles)  # 初始化 Triangulation 对象

    # Check neighbors, which forces creation of C++ triangulation
    # 验证 triang 对象的 neighbors 属性是否与预期值相等
    assert_array_equal(triang.neighbors, [[-1, -1, 1], [-1, -1, 0]])

    # 设置 triang 对象的 mask 属性为指定的值，并验证设置成功
    triang.set_mask([False, True])
    assert_array_equal(triang.mask, [False, True])

    # 重置 triang 对象的 mask 属性为 None，并验证成功重置
    triang.set_mask(None)
    assert triang.mask is None

    # 准备错误消息，用于在设置 mask 属性时捕获预期的 ValueError 异常
    msg = r"mask array must have same length as triangles array"
    # 遍历不同的 mask 设置，验证是否会引发 ValueError 异常，并匹配预期的错误消息
    for mask in ([False, True, False], [False], [True], False, True):
        with pytest.raises(ValueError, match=msg):
            triang.set_mask(mask)
def test_delaunay():
    # No duplicate points, regular grid.
    nx = 5
    ny = 4
    # 生成网格点坐标
    x, y = np.meshgrid(np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, ny))
    # 将网格展平为一维数组
    x = x.ravel()
    y = y.ravel()
    npoints = nx * ny
    ntriangles = 2 * (nx - 1) * (ny - 1)
    nedges = 3 * nx * ny - 2 * nx - 2 * ny + 1

    # 创建 Delaunay 三角剖分
    triang = mtri.Triangulation(x, y)

    # 以下测试要求任何不包含重复点的三角剖分都能通过
    # 测试点的浮点数值
    assert_array_almost_equal(triang.x, x)
    assert_array_almost_equal(triang.y, y)

    # 测试三角形的整数索引
    assert len(triang.triangles) == ntriangles
    assert np.min(triang.triangles) == 0
    assert np.max(triang.triangles) == npoints - 1

    # 测试边的整数索引
    assert len(triang.edges) == nedges
    assert np.min(triang.edges) == 0
    assert np.max(triang.edges) == npoints - 1

    # 测试邻居的整数索引
    # 检查由 C++ 三角剖分类计算的邻居是否与 Delaunay 程序返回的相同
    neighbors = triang.neighbors
    triang._neighbors = None
    assert_array_equal(triang.neighbors, neighbors)

    # 每个点是否至少用于一个三角形？
    assert_array_equal(np.unique(triang.triangles), np.arange(npoints))


def test_delaunay_duplicate_points():
    npoints = 10
    duplicate = 7
    duplicate_of = 3

    np.random.seed(23)
    x = np.random.random(npoints)
    y = np.random.random(npoints)
    x[duplicate] = x[duplicate_of]
    y[duplicate] = y[duplicate_of]

    # 创建 Delaunay 三角剖分
    triang = mtri.Triangulation(x, y)

    # 重复点应被忽略，因此重复点的索引不应出现在任何三角形中
    assert_array_equal(np.unique(triang.triangles),
                       np.delete(np.arange(npoints), duplicate))


def test_delaunay_points_in_line():
    # 无法对全部在一条直线上的点进行三角剖分，但要检查 Delaunay 代码是否优雅地失败
    x = np.linspace(0.0, 10.0, 11)
    y = np.linspace(0.0, 10.0, 11)
    with pytest.raises(RuntimeError):
        mtri.Triangulation(x, y)

    # 添加一点不在直线上，三角剖分就可以进行
    x = np.append(x, 2.0)
    y = np.append(y, 8.0)
    mtri.Triangulation(x, y)


@pytest.mark.parametrize('x, y', [
    # 如果传入少于 3 个点，三角剖分应该引发 ValueError
    ([], []),
    ([1], [5]),
    ([1, 2], [5, 6]),
    # 如果传入重复点，导致少于 3 个唯一点，也应该引发 ValueError
    ([1, 2, 1], [5, 6, 5]),
    ([1, 2, 2], [5, 6, 6]),
    ([1, 1, 1, 2, 1, 2], [5, 5, 5, 6, 5, 6]),
])
def test_delaunay_insufficient_points(x, y):
    with pytest.raises(ValueError):
        mtri.Triangulation(x, y)


def test_delaunay_robust():
    # 当 mtri.Triangulation 使用 matplotlib.delaunay 时会失败，但使用 qhull 时可以成功
    tri_points = np.array([
        [0.8660254037844384, -0.5000000000000004],  # 定义一个包含7个三角形顶点的numpy数组
        [0.7577722283113836, -0.5000000000000004],
        [0.6495190528383288, -0.5000000000000003],
        [0.5412658773652739, -0.5000000000000003],
        [0.811898816047911, -0.40625000000000044],
        [0.7036456405748561, -0.4062500000000004],
        [0.5953924651018013, -0.40625000000000033]])
    test_points = np.asarray([
        [0.58, -0.46],  # 定义一个包含7个测试点的numpy数组
        [0.65, -0.46],
        [0.65, -0.42],
        [0.7, -0.48],
        [0.7, -0.44],
        [0.75, -0.44],
        [0.8, -0.48]])

    # 判断一个由三个顶点定义的三角形是否包含指定的测试点(xy)的实用函数。
    # 避免用在位于三角形边上或非常接近边缘的点进行调用。
    def tri_contains_point(xtri, ytri, xy):
        tri_points = np.vstack((xtri, ytri)).T  # 将三角形顶点堆叠为一个坐标数组
        return Path(tri_points).contains_point(xy)  # 使用matplotlib的Path类判断点是否在三角形内部

    # 返回指定三角剖分中包含测试点(xy)的三角形数量的实用函数。
    # 避免用在位于任何三角形边上或非常接近边缘的点进行调用。
    def tris_contain_point(triang, xy):
        return sum(tri_contains_point(triang.x[tri], triang.y[tri], xy)  # 对每个三角形调用tri_contains_point函数并求和
                   for tri in triang.triangles)

    # 使用matplotlib的delaunay创建一个无效的三角剖分，其中包含重叠的三角形；qhull正常工作。
    triang = mtri.Triangulation(tri_points[:, 0], tri_points[:, 1])  # 使用tri_points的坐标创建一个三角剖分对象

    for test_point in test_points:
        assert tris_contain_point(triang, test_point) == 1  # 断言每个测试点(test_points)都只被包含在一个三角形中

    # 如果忽略tri_points的第一个点，在计算凸包时，matplotlib的delaunay会抛出KeyError；qhull正常工作。
    triang = mtri.Triangulation(tri_points[1:, 0], tri_points[1:, 1])  # 使用tri_points去掉第一个点的坐标重新创建三角剖分对象
@image_comparison(['tripcolor1.png'])
# 定义测试函数，比较生成的图像与预期图像是否一致
def test_tripcolor():
    # 定义点的 x 和 y 坐标数组
    x = np.asarray([0, 0.5, 1, 0,   0.5, 1,   0, 0.5, 1, 0.75])
    y = np.asarray([0, 0,   0, 0.5, 0.5, 0.5, 1, 1,   1, 0.75])
    # 定义三角形的顶点索引数组
    triangles = np.asarray([
        [0, 1, 3], [1, 4, 3],
        [1, 2, 4], [2, 5, 4],
        [3, 4, 6], [4, 7, 6],
        [4, 5, 9], [7, 4, 9], [8, 7, 9], [5, 8, 9]])

    # 使用 x, y 和 triangles 创建三角剖分对象
    triang = mtri.Triangulation(x, y, triangles)

    # 计算每个点的颜色
    Cpoints = x + 0.5 * y

    # 计算每个面的颜色
    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    Cfaces = 0.5 * xmid + ymid

    # 在子图1中绘制点的颜色
    plt.subplot(121)
    plt.tripcolor(triang, Cpoints, edgecolors='k')
    plt.title('point colors')

    # 在子图2中绘制面的颜色
    plt.subplot(122)
    plt.tripcolor(triang, facecolors=Cfaces, edgecolors='k')
    plt.title('facecolors')


# 定义测试函数，用于测试不同的 tripcolor 参数组合
def test_tripcolor_color():
    # 定义不同的 x 和 y 坐标数组
    x = [-1, 0, 1, 0]
    y = [0, -1, 0, 1]
    fig, ax = plt.subplots()

    # 测试缺少参数引发的 TypeError
    with pytest.raises(TypeError, match=r"tripcolor\(\) missing 1 required "):
        ax.tripcolor(x, y)

    # 测试 c 长度不匹配引发的 ValueError
    with pytest.raises(ValueError, match="The length of c must match either"):
        ax.tripcolor(x, y, [1, 2, 3])

    # 测试 facecolors 长度不匹配引发的 ValueError
    with pytest.raises(ValueError,
                       match="length of facecolors must match .* triangles"):
        ax.tripcolor(x, y, facecolors=[1, 2, 3, 4])

    # 测试 shading 参数设置不正确引发的 ValueError
    with pytest.raises(ValueError,
                       match="'gouraud' .* at the points.* not at the faces"):
        ax.tripcolor(x, y, facecolors=[1, 2], shading='gouraud')

    # 测试不正确的位置参数引发的 TypeError
    with pytest.raises(TypeError,
                       match="positional.*'c'.*keyword-only.*'facecolors'"):
        ax.tripcolor(x, y, C=[1, 2, 3, 4])

    # 测试不正确的位置参数引发的 TypeError
    with pytest.raises(TypeError, match="Unexpected positional parameter"):
        ax.tripcolor(x, y, [1, 2], 'unused_positional')

    # 测试不同有效的颜色规范
    ax.tripcolor(x, y, [1, 2, 3, 4])  # 绘制边缘
    ax.tripcolor(x, y, [1, 2, 3, 4], shading='gouraud')  # 绘制边缘
    ax.tripcolor(x, y, [1, 2])  # 绘制面
    ax.tripcolor(x, y, facecolors=[1, 2])  # 绘制面


# 定义测试函数，用于测试 clim 参数设置
def test_tripcolor_clim():
    # 设置随机种子
    np.random.seed(19680801)
    # 生成随机数据
    a, b, c = np.random.rand(10), np.random.rand(10), np.random.rand(10)

    # 创建图形和坐标轴对象
    ax = plt.figure().add_subplot()
    # 设置颜色限制范围
    clim = (0.25, 0.75)
    # 获取 tripcolor 绘制后的归一化对象
    norm = ax.tripcolor(a, b, c, clim=clim).norm
    # 验证归一化对象的 vmin 和 vmax 是否符合预期
    assert (norm.vmin, norm.vmax) == clim


# 定义测试函数，用于测试警告信息
def test_tripcolor_warnings():
    # 定义不同的 x 和 y 坐标数组以及颜色数组
    x = [-1, 0, 1, 0]
    y = [0, -1, 0, 1]
    c = [0.4, 0.5]
    fig, ax = plt.subplots()

    # 测试 facecolors 参数优先级高于 c 参数时的警告信息
    with pytest.warns(UserWarning, match="Positional parameter c .*no effect"):
        ax.tripcolor(x, y, c, facecolors=c)

    # 测试 facecolors 参数优先级高于 c 参数时的警告信息
    with pytest.warns(UserWarning, match="Positional parameter c .*no effect"):
        ax.tripcolor(x, y, 'interpreted as c', facecolors=c)
    # 测试 Triangulation 方法不会修改传递给它的 triangles 数组。
    # 创建一个包含三角形索引的 NumPy 数组，每个三角形由三个顶点索引组成。
    triangles = np.array([[3, 2, 0], [3, 1, 0]], dtype=np.int32)
    # 创建一个包含点坐标的 NumPy 数组，每个点由 (x, y) 坐标对组成。
    points = np.array([(0, 0), (0, 1.1), (1, 0), (1, 1)])
    
    # 复制 triangles 数组，以便稍后进行断言比较。
    old_triangles = triangles.copy()
    # 使用 mtri.Triangulation 方法创建一个三角网格对象，传递点的 x 和 y 坐标以及三角形索引。
    # 并访问其 edges 属性，这是测试 Triangulation 方法的一部分。
    mtri.Triangulation(points[:, 0], points[:, 1], triangles).edges
    # 断言原始 triangles 数组与经过 Triangulation 方法后的 triangles 数组相等。
    assert_array_equal(old_triangles, triangles)
def test_trifinder():
    # Test points within triangles of masked triangulation.
    # 创建一个4x4的网格
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    x = x.ravel()  # 将网格展平为一维数组
    y = y.ravel()  # 将网格展平为一维数组
    triangles = [[0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5], [2, 3, 6],
                 [3, 7, 6], [4, 5, 8], [5, 9, 8], [5, 6, 9], [6, 10, 9],
                 [6, 7, 10], [7, 11, 10], [8, 9, 12], [9, 13, 12], [9, 10, 13],
                 [10, 14, 13], [10, 11, 14], [11, 15, 14]]
    mask = np.zeros(len(triangles))  # 创建一个与三角形数量相同的零数组
    mask[8:10] = 1  # 将指定范围内的元素设置为1，用作蒙版（mask）
    triang = mtri.Triangulation(x, y, triangles, mask)  # 创建带有蒙版的三角剖分对象
    trifinder = triang.get_trifinder()  # 获取用于查找三角形索引的函数

    xs = [0.25, 1.25, 2.25, 3.25]
    ys = [0.25, 1.25, 2.25, 3.25]
    xs, ys = np.meshgrid(xs, ys)  # 创建另一个4x4的网格
    xs = xs.ravel()  # 将网格展平为一维数组
    ys = ys.ravel()  # 将网格展平为一维数组
    tris = trifinder(xs, ys)  # 使用 trifinder 函数查找网格点所在的三角形索引
    assert_array_equal(tris, [0, 2, 4, -1, 6, -1, 10, -1,
                              12, 14, 16, -1, -1, -1, -1, -1])  # 断言检查三角形索引是否符合预期

    tris = trifinder(xs-0.5, ys-0.5)  # 使用 trifinder 函数查找偏移后的网格点所在的三角形索引
    assert_array_equal(tris, [-1, -1, -1, -1, -1, 1, 3, 5,
                              -1, 7, -1, 11, -1, 13, 15, 17])  # 断言检查三角形索引是否符合预期

    # Test points exactly on boundary edges of masked triangulation.
    # 在蒙版三角剖分的边界上测试点
    xs = [0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 1.5, 1.5, 0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 1.0, 2.0, 1.5, 1.5, 1.5, 1.5]
    tris = trifinder(xs, ys)  # 使用 trifinder 函数查找边界上点所在的三角形索引
    assert_array_equal(tris, [0, 2, 4, 13, 15, 17, 3, 14, 6, 7, 10, 11])  # 断言检查三角形索引是否符合预期

    # Test points exactly on boundary corners of masked triangulation.
    # 在蒙版三角剖分的边界角上测试点
    xs = [0.0, 3.0]
    ys = [0.0, 3.0]
    tris = trifinder(xs, ys)  # 使用 trifinder 函数查找角上点所在的三角形索引
    assert_array_equal(tris, [0, 17])  # 断言检查三角形索引是否符合预期

    #
    # Test triangles with horizontal colinear points.  These are not valid
    # triangulations, but we try to deal with the simplest violations.
    #

    # If +ve, triangulation is OK, if -ve triangulation invalid,
    # if zero have colinear points but should pass tests anyway.
    delta = 0.0

    x = [1.5, 0,  1,  2, 3, 1.5,   1.5]
    y = [-1,  0,  0,  0, 0, delta, 1]
    triangles = [[0, 2, 1], [0, 3, 2], [0, 4, 3], [1, 2, 5], [2, 3, 5],
                 [3, 4, 5], [1, 5, 6], [4, 6, 5]]
    triang = mtri.Triangulation(x, y, triangles)  # 创建包含水平共线点的三角剖分对象
    trifinder = triang.get_trifinder()  # 获取用于查找三角形索引的函数

    xs = [-0.1, 0.4, 0.9, 1.4, 1.9, 2.4, 2.9]
    ys = [-0.1, 0.1]
    xs, ys = np.meshgrid(xs, ys)  # 创建新的网格
    tris = trifinder(xs, ys)  # 使用 trifinder 函数查找网格点所在的三角形索引
    assert_array_equal(tris, [[-1, 0, 0, 1, 1, 2, -1],
                              [-1, 6, 6, 6, 7, 7, -1]])  # 断言检查三角形索引是否符合预期

    #
    # Test triangles with vertical colinear points.  These are not valid
    # triangulations, but we try to deal with the simplest violations.
    #

    # If +ve, triangulation is OK, if -ve triangulation invalid,
    # if zero have colinear points but should pass tests anyway.
    delta = 0.0

    x = [-1, -delta, 0,  0,  0, 0, 1]
    y = [1.5, 1.5,   0,  1,  2, 3, 1.5]
    triangles = [[0, 1, 2], [0, 1, 5], [1, 2, 3], [1, 3, 4], [1, 4, 5],
                 [2, 6, 3], [3, 6, 4], [4, 6, 5]]
    triang = mtri.Triangulation(x, y, triangles)  # 创建包含垂直共线点的三角剖分对象
    trifinder = triang.get_trifinder()  # 获取用于查找三角形索引的函数

    xs = [-0.1, 0.1]
    # 定义一个包含特定 y 值的列表
    ys = [-0.1, 0.4, 0.9, 1.4, 1.9, 2.4, 2.9]
    # 创建一个网格，xs 和 ys 是坐标网格的结果
    xs, ys = np.meshgrid(xs, ys)
    # 使用 trifinder 函数计算网格点所在的三角形索引
    tris = trifinder(xs, ys)
    # 断言三角形索引数组与预期值相等
    assert_array_equal(tris, [[-1, -1], [0, 5], [0, 5], [0, 6], [1, 6], [1, 7], [-1, -1]])

    # 测试通过设置掩码来改变三角化是否导致 trifinder 重新初始化
    x = [0, 1, 0, 1]
    y = [0, 0, 1, 1]
    triangles = [[0, 1, 2], [1, 3, 2]]
    # 使用给定的顶点和三角形列表创建 Triangulation 对象
    triang = mtri.Triangulation(x, y, triangles)
    # 获取新的 trifinder 函数
    trifinder = triang.get_trifinder()

    # 定义一个包含特定 x 值的列表
    xs = [-0.2, 0.2, 0.8, 1.2]
    # 定义一个包含特定 y 值的列表
    ys = [0.5, 0.5, 0.5, 0.5]
    # 使用 trifinder 函数计算新的网格点所在的三角形索引
    tris = trifinder(xs, ys)
    # 断言三角形索引数组与预期值相等
    assert_array_equal(tris, [-1, 0, 1, -1])

    # 设置新的掩码以改变三角化
    triang.set_mask([1, 0])
    # 断言 trifinder 函数是否与更新后的 triang 对象中的 trifinder 相等
    assert trifinder == triang.get_trifinder()
    # 使用 trifinder 函数计算再次更新后的网格点所在的三角形索引
    tris = trifinder(xs, ys)
    # 断言三角形索引数组与预期值相等
    assert_array_equal(tris, [-1, -1, 1, -1])
def test_triinterp():
    # 在带有遮罩三角剖分的三角形内部测试点。
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    x = x.ravel()  # 将二维网格展平为一维数组
    y = y.ravel()  # 将二维网格展平为一维数组
    z = 1.23*x - 4.79*y  # 计算 z 值
    triangles = [[0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5], [2, 3, 6],
                 [3, 7, 6], [4, 5, 8], [5, 9, 8], [5, 6, 9], [6, 10, 9],
                 [6, 7, 10], [7, 11, 10], [8, 9, 12], [9, 13, 12], [9, 10, 13],
                 [10, 14, 13], [10, 11, 14], [11, 15, 14]]
    mask = np.zeros(len(triangles))  # 创建一个全零数组，长度与三角形数目相同
    mask[8:10] = 1  # 设置部分三角形的遮罩为1
    triang = mtri.Triangulation(x, y, triangles, mask)  # 创建带有遮罩的三角剖分对象

    # 创建线性三角插值对象和三次样条插值对象
    linear_interp = mtri.LinearTriInterpolator(triang, z)
    cubic_min_E = mtri.CubicTriInterpolator(triang, z)
    cubic_geom = mtri.CubicTriInterpolator(triang, z, kind='geom')

    xs = np.linspace(0.25, 2.75, 6)
    ys = [0.25, 0.75, 2.25, 2.75]
    xs, ys = np.meshgrid(xs, ys)  # 使用网格点创建测试数组，array.ndim = 2
    for interp in (linear_interp, cubic_min_E, cubic_geom):
        zs = interp(xs, ys)  # 对测试数组进行插值计算
        assert_array_almost_equal(zs, (1.23*xs - 4.79*ys))  # 断言插值结果与预期相近

    # 测试超出三角剖分的点
    xs = [-0.25, 1.25, 1.75, 3.25]
    ys = xs
    xs, ys = np.meshgrid(xs, ys)
    for interp in (linear_interp, cubic_min_E, cubic_geom):
        zs = linear_interp(xs, ys)  # 对超出三角剖分的点进行线性插值
        assert_array_equal(zs.mask, [[True]*4]*4)  # 断言结果数组的遮罩全部为True

    # 测试混合配置（内部和外部点）
    xs = np.linspace(0.25, 1.75, 6)
    ys = [0.25, 0.75, 1.25, 1.75]
    xs, ys = np.meshgrid(xs, ys)
    for interp in (linear_interp, cubic_min_E, cubic_geom):
        zs = interp(xs, ys)  # 对混合点进行插值计算
        matest.assert_array_almost_equal(zs, (1.23*xs - 4.79*ys))  # 断言插值结果与预期相近
        mask = (xs >= 1) * (xs <= 2) * (ys >= 1) * (ys <= 2)
        assert_array_equal(zs.mask, mask)  # 断言结果数组的遮罩与预期相符合

    # 二阶补丁测试：在具有“任意形状”三角形的网格上，
    # 对于二次函数和如果 kind=user 则对于立方插值器进行精确的补丁测试
    (a, b, c) = (1.23, -4.79, 0.6)

    def quad(x, y):
        return a*(x-0.5)**2 + b*(y-0.5)**2 + c*x*y  # 定义二次函数

    def gradient_quad(x, y):
        return (2*a*(x-0.5) + c*y, 2*b*(y-0.5) + c*x)  # 定义二次函数的梯度

    x = np.array([0.2, 0.33367, 0.669, 0., 1., 1., 0.])
    y = np.array([0.3, 0.80755, 0.4335, 0., 0., 1., 1.])
    triangles = np.array([[0, 1, 2], [3, 0, 4], [4, 0, 2], [4, 2, 5],
                          [1, 5, 2], [6, 5, 1], [6, 1, 0], [6, 0, 3]])
    triang = mtri.Triangulation(x, y, triangles)  # 创建新的三角剖分对象
    z = quad(x, y)  # 计算二次函数在给定点的值
    dz = gradient_quad(x, y)  # 计算二次函数在给定点的梯度

    # 对于二阶补丁测试的点进行插值计算
    xs = np.linspace(0., 1., 5)
    ys = np.linspace(0., 1., 5)
    xs, ys = np.meshgrid(xs, ys)
    cubic_user = mtri.CubicTriInterpolator(triang, z, kind='user', dz=dz)
    interp_zs = cubic_user(xs, ys)
    assert_array_almost_equal(interp_zs, quad(xs, ys))  # 断言插值结果与二次函数的预期值相近
    (interp_dzsdx, interp_dzsdy) = cubic_user.gradient(x, y)
    (dzsdx, dzsdy) = gradient_quad(x, y)
    assert_array_almost_equal(interp_dzsdx, dzsdx)  # 断言插值梯度与二次函数的梯度在 x 方向上相近
    assert_array_almost_equal(interp_dzsdy, dzsdy)  # 断言插值梯度与二次函数的梯度在 y 方向上相近
    # 定义使用的点数
    n = 11
    # 生成网格点的 x 和 y 坐标，分别在 [0, 1] 区间内均匀分布
    x, y = np.meshgrid(np.linspace(0., 1., n+1), np.linspace(0., 1., n+1))
    # 将网格点坐标展平为一维数组
    x = x.ravel()
    y = y.ravel()
    # 对每个网格点计算二次函数的值，返回一个与 x, y 对应的数组 z
    z = quad(x, y)
    # 使用网格点生成三角剖分对象，使用预定义的三角形分割
    triang = mtri.Triangulation(x, y, triangles=meshgrid_triangles(n+1))
    # 在新的网格上定义新的 x, y 坐标，分别在 [0.1, 0.9] 区间内均匀分布
    xs, ys = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
    # 将新的网格点坐标展平为一维数组
    xs = xs.ravel()
    ys = ys.ravel()
    # 创建线性三角形插值器对象，基于前面生成的三角剖分和 z 数组
    linear_interp = mtri.LinearTriInterpolator(triang, z)
    # 创建使用最小能量三次插值方法的三角形插值器对象
    cubic_min_E = mtri.CubicTriInterpolator(triang, z)
    # 创建使用几何方法的三次插值器对象
    cubic_geom = mtri.CubicTriInterpolator(triang, z, kind='geom')
    # 计算在新网格上的准确值 zs，即二次函数在新点上的值
    zs = quad(xs, ys)
    # 计算线性插值与准确值的差异
    diff_lin = np.abs(linear_interp(xs, ys) - zs)
    # 对于每种三次插值方法，计算其插值值与准确值的差异
    for interp in (cubic_min_E, cubic_geom):
        diff_cubic = np.abs(interp(xs, ys) - zs)
        # 断言线性插值的最大误差至少是三次插值最大误差的十倍
        assert np.max(diff_lin) >= 10 * np.max(diff_cubic)
        # 断言线性插值误差的平方范数至少是三次插值误差平方范数的一百倍
        assert (np.dot(diff_lin, diff_lin) >=
                100 * np.dot(diff_cubic, diff_cubic))
# 定义一个测试函数，用于验证 TriCubicInterpolator 的 C1 连续性
def test_triinterpcubic_C1_continuity():
    # 下面是四个测试，演示 TriCubicInterpolator 在任意三角形上的 C1 连续性：
    #
    # 1) 测试在所有 9 个形状函数的角点处的函数及其导数的连续性，同时也测试在同一位置的函数值。
    # 2) 测试沿每条边的 C1 连续性（由于梯度是二阶多项式，只需在中点测试即可）。
    # 3) 测试在三角形重心处的 C1 连续性（三个子三角形的交汇点）。
    # 4) 测试在中位数的 1/3 点处的 C1 连续性（两个子三角形之间的中点）。

    # 辅助测试函数 check_continuity
    def check_continuity(interpolator, loc, values=None):
        """
        检查插值器（及其导数）在位置 loc 附近的连续性。如果提供了 *values*，
        还可以检查 loc 处的值。

        *interpolator* TriInterpolator
        *loc* 要测试的位置 (x0, y0)
        *values*（可选）数组 [z0, dzx0, dzy0]，用于检查 *loc* 处的值
        """
        n_star = 24       # loc 边界上连续性点的数量
        epsilon = 1.e-10  # loc 边界的距离
        k = 100.          # 连续性系数
        (loc_x, loc_y) = loc
        # 在 loc 附近创建 n_star 个点
        star_x = loc_x + epsilon*np.cos(np.linspace(0., 2*np.pi, n_star))
        star_y = loc_y + epsilon*np.sin(np.linspace(0., 2*np.pi, n_star))
        # 计算 loc 处的插值值 z 及其梯度 (dzx, dzy)
        z = interpolator([loc_x], [loc_y])[0]
        (dzx, dzy) = interpolator.gradient([loc_x], [loc_y])
        # 如果提供了 values，则断言 z、dzx、dzy 与给定值几乎相等
        if values is not None:
            assert_array_almost_equal(z, values[0])
            assert_array_almost_equal(dzx[0], values[1])
            assert_array_almost_equal(dzy[0], values[2])
        # 计算边界点 star_x, star_y 处的插值值和梯度，与 loc 处的值比较
        diff_z = interpolator(star_x, star_y) - z
        (tab_dzx, tab_dzy) = interpolator.gradient(star_x, star_y)
        diff_dzx = tab_dzx - dzx
        diff_dzy = tab_dzy - dzy
        # 断言插值值和梯度与 loc 处的值之差小于 epsilon*k
        assert_array_less(diff_z, epsilon*k)
        assert_array_less(diff_dzx, epsilon*k)
        assert_array_less(diff_dzy, epsilon*k)

    # 在单位正方形内绘制任意三角形 (a, b, c)
    (ax, ay) = (0.2, 0.3)
    (bx, by) = (0.33367, 0.80755)
    (cx, cy) = (0.669, 0.4335)
    x = np.array([ax, bx, cx, 0., 1., 1., 0.])
    y = np.array([ay, by, cy, 0., 0., 1., 1.])
    triangles = np.array([[0, 1, 2], [3, 0, 4], [4, 0, 2], [4, 2, 5],
                          [1, 5, 2], [6, 5, 1], [6, 1, 0], [6, 0, 3]])
    # 创建三角剖分对象
    triang = mtri.Triangulation(x, y, triangles)
    # 对于每个自由度 ID，创建一些必要的数组并初始化为零
    for idof in range(9):
        z = np.zeros(7, dtype=np.float64)  # 初始化长度为 7 的零数组 z
        dzx = np.zeros(7, dtype=np.float64)  # 初始化长度为 7 的零数组 dzx
        dzy = np.zeros(7, dtype=np.float64)  # 初始化长度为 7 的零数组 dzy
        values = np.zeros([3, 3], dtype=np.float64)  # 初始化大小为 3x3 的零数组 values
        case = idof // 3  # 计算自由度的类型（0, 1, 2）
        values[case, idof % 3] = 1.0  # 根据自由度 ID 设置 values 数组的相应位置为 1.0

        # 根据自由度的类型设置 z, dzx, dzy 数组的值
        if case == 0:
            z[idof] = 1.0  # 对于类型 0 的自由度，设置 z 数组对应位置为 1.0
        elif case == 1:
            dzx[idof % 3] = 1.0  # 对于类型 1 的自由度，设置 dzx 数组对应位置为 1.0
        elif case == 2:
            dzy[idof % 3] = 1.0  # 对于类型 2 的自由度，设置 dzy 数组对应位置为 1.0

        # 使用 CubicTriInterpolator 创建插值对象 interp
        interp = mtri.CubicTriInterpolator(triang, z, kind='user', dz=(dzx, dzy))

        # Test 1) 检查节点处的值和连续性
        check_continuity(interp, (ax, ay), values[:, 0])  # 检查在 (ax, ay) 处的连续性
        check_continuity(interp, (bx, by), values[:, 1])  # 检查在 (bx, by) 处的连续性
        check_continuity(interp, (cx, cy), values[:, 2])  # 检查在 (cx, cy) 处的连续性

        # Test 2) 检查中间节点处的连续性
        check_continuity(interp, ((ax+bx)*0.5, (ay+by)*0.5))  # 检查在中点 (ax+bx)/2, (ay+by)/2 处的连续性
        check_continuity(interp, ((ax+cx)*0.5, (ay+cy)*0.5))  # 检查在中点 (ax+cx)/2, (ay+cy)/2 处的连续性
        check_continuity(interp, ((cx+bx)*0.5, (cy+by)*0.5))  # 检查在中点 (cx+bx)/2, (cy+by)/2 处的连续性

        # Test 3) 检查重心处的连续性
        check_continuity(interp, ((ax+bx+cx)/3., (ay+by+cy)/3.))  # 检查在重心处的连续性

        # Test 4) 检查中位三分点处的连续性
        check_continuity(interp, ((4.*ax+bx+cx)/6., (4.*ay+by+cy)/6.))  # 检查在中位三分点处的连续性
        check_continuity(interp, ((ax+4.*bx+cx)/6., (ay+4.*by+cy)/6.))  # 检查在中位三分点处的连续性
        check_continuity(interp, ((ax+bx+4.*cx)/6., (ay+by+4.*cy)/6.))  # 检查在中位三分点处的连续性
def test_triinterpcubic_cg_solver():
    # 现在测试稀疏共轭梯度求解器的三次立方插值器，使用 *kind* = 'min_E'
    # 1) 一个常用的测试案例涉及二维泊松矩阵。
    def poisson_sparse_matrix(n, m):
        """
        根据有限差分数值方案，在均匀 (n, m) 网格上离散化的二维泊松方程，
        返回稀疏的 (n*m, n*m) COO 格式的矩阵。
        """
        l = m*n
        rows = np.concatenate([
            np.arange(l, dtype=np.int32),
            np.arange(l-1, dtype=np.int32), np.arange(1, l, dtype=np.int32),
            np.arange(l-n, dtype=np.int32), np.arange(n, l, dtype=np.int32)])
        cols = np.concatenate([
            np.arange(l, dtype=np.int32),
            np.arange(1, l, dtype=np.int32), np.arange(l-1, dtype=np.int32),
            np.arange(n, l, dtype=np.int32), np.arange(l-n, dtype=np.int32)])
        vals = np.concatenate([
            4*np.ones(l, dtype=np.float64),
            -np.ones(l-1, dtype=np.float64), -np.ones(l-1, dtype=np.float64),
            -np.ones(l-n, dtype=np.float64), -np.ones(l-n, dtype=np.float64)])
        # 实际上，+1 和 -1 对角线有一些零元素
        vals[l:2*l-1][m-1::m] = 0.
        vals[2*l-1:3*l-2][m-1::m] = 0.
        return vals, rows, cols, (n*m, n*m)

    # 实例化一个大小为 48 x 48 的稀疏泊松矩阵：
    (n, m) = (12, 4)
    mat = mtri._triinterpolate._Sparse_Matrix_coo(*poisson_sparse_matrix(n, m))
    mat.compress_csc()
    mat_dense = mat.to_dense()

    # 对所有 48 个基向量进行稀疏求解测试
    for itest in range(n*m):
        b = np.zeros(n*m, dtype=np.float64)
        b[itest] = 1.
        x, _ = mtri._triinterpolate._cg(A=mat, b=b, x0=np.zeros(n*m),
                                        tol=1.e-10)
        assert_array_almost_equal(np.dot(mat_dense, x), b)

    # 2) 在同一个矩阵中插入 2 行 - 列，对角线上无元素
    # （但仍通过额外对角线元素与矩阵的其余部分相连）
    (i_zero, j_zero) = (12, 49)
    vals, rows, cols, _ = poisson_sparse_matrix(n, m)
    rows = rows + 1*(rows >= i_zero) + 1*(rows >= j_zero)
    cols = cols + 1*(cols >= i_zero) + 1*(cols >= j_zero)
    # 添加额外的对角线元素
    rows = np.concatenate([rows, [i_zero, i_zero-1, j_zero, j_zero-1]])
    cols = np.concatenate([cols, [i_zero-1, i_zero, j_zero-1, j_zero]])
    vals = np.concatenate([vals, [1., 1., 1., 1.]])
    mat = mtri._triinterpolate._Sparse_Matrix_coo(vals, rows, cols,
                                                  (n*m + 2, n*m + 2))
    mat.compress_csc()
    mat_dense = mat.to_dense()
    # 对所有 50 个基向量进行稀疏求解测试
    # 对于范围在 0 到 n*m+1 的每一个索引进行循环测试
    for itest in range(n*m + 2):
        # 创建一个长度为 n*m+2 的全零数组，数据类型为 np.float64
        b = np.zeros(n*m + 2, dtype=np.float64)
        # 将当前索引 itest 处设置为 1，其余位置保持为 0
        b[itest] = 1.
        # 使用共轭梯度法求解线性方程组 Ax = b，返回解 x 和迭代次数 _
        x, _ = mtri._triinterpolate._cg(A=mat, b=b, x0=np.ones(n * m + 2),
                                        tol=1.e-10)
        # 断言：稠密矩阵 mat_dense 与矩阵乘积 np.dot(mat_dense, x) 的结果近似等于向量 b
        assert_array_almost_equal(np.dot(mat_dense, x), b)

    # 3) 现在测试一个简单的情况，即当压缩时发生重复项的求和（即具有相同行和列的项）。
    # 创建值全为 1 的长度为 17 的数组，数据类型为 np.float64
    vals = np.ones(17, dtype=np.float64)
    # 创建行索引数组，指定每个元素所在的行
    rows = np.array([0, 1, 2, 0, 0, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
                    dtype=np.int32)
    # 创建列索引数组，指定每个元素所在的列
    cols = np.array([0, 1, 2, 1, 1, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                    dtype=np.int32)
    # 指定稀疏矩阵的维度
    dim = (3, 3)
    # 使用给定的值、行索引、列索引和维度创建稀疏 COO 格式矩阵对象 mat
    mat = mtri._triinterpolate._Sparse_Matrix_coo(vals, rows, cols, dim)
    # 将稀疏矩阵 mat 压缩为 CSC 格式
    mat.compress_csc()
    # 将压缩后的稀疏矩阵 mat 转换为稠密矩阵 mat_dense
    mat_dense = mat.to_dense()
    # 断言：稠密矩阵 mat_dense 与预期的稠密矩阵的近似相等
    assert_array_almost_equal(mat_dense, np.array([
        [1., 2., 0.],
        [2., 1., 5.],
        [0., 5., 1.]], dtype=np.float64))
def test_triinterpcubic_geom_weights():
    # Tests to check computation of weights for _DOF_estimator_geom:
    # The weight sum per triangle can be 1. (in case all angles < 90 degrees)
    # or (2*w_i) where w_i = 1-alpha_i/np.pi is the weight of apex i; alpha_i
    # is the apex angle > 90 degrees.

    # 初始化三角形的顶点坐标
    (ax, ay) = (0., 1.687)
    x = np.array([ax, 0.5*ax, 0., 1.])
    y = np.array([ay, -ay, 0., 0.])

    # 初始化顶点处的高度值为零
    z = np.zeros(4, dtype=np.float64)

    # 定义两个三角形的顶点索引
    triangles = [[0, 2, 3], [1, 3, 2]]

    # 初始化用于存储权重和的数组，有4种可能性，每种有2个三角形
    sum_w = np.zeros([4, 2])

    # 循环旋转图形，生成不同角度的三角剖分
    for theta in np.linspace(0., 2*np.pi, 14):  # rotating the figure...
        x_rot = np.cos(theta)*x + np.sin(theta)*y
        y_rot = -np.sin(theta)*x + np.cos(theta)*y

        # 创建三角剖分对象
        triang = mtri.Triangulation(x_rot, y_rot, triangles)

        # 创建三次三角插值对象
        cubic_geom = mtri.CubicTriInterpolator(triang, z, kind='geom')

        # 计算几何加权的自由度估计器
        dof_estimator = mtri._triinterpolate._DOF_estimator_geom(cubic_geom)

        # 计算几何权重
        weights = dof_estimator.compute_geom_weights()

        # 对4种可能性进行测试
        sum_w[0, :] = np.sum(weights, 1) - 1
        for itri in range(3):
            sum_w[itri+1, :] = np.sum(weights, 1) - 2*weights[:, itri]

        # 使用断言检查结果
        assert_array_almost_equal(np.min(np.abs(sum_w), axis=0),
                                  np.array([0., 0.], dtype=np.float64))


def test_triinterp_colinear():
    # Tests interpolating inside a triangulation with horizontal colinear
    # points (refer also to the tests :func:`test_trifinder` ).

    # 这些不是有效的三角剖分，但我们尝试处理最简单的违规情况（例如默认的TriFinder处理的情况）。

    # 注意，使用kind='min_E'或'geom'的LinearTriInterpolator和CubicTriInterpolator仍然通过线性路径测试。
    # 我们还通过在调用:meth:`_interpolate_multikeys`时强制*tri_index*来测试插值在平面三角形内部的情况。

    # 如果是正数，则三角剖分是正常的；如果是负数，则三角剖分无效；
    # 如果是零，则存在共线点，但应该通过测试。
    delta = 0.

    # 初始化初始图形的顶点坐标
    x0 = np.array([1.5, 0,  1,  2, 3, 1.5,   1.5])
    y0 = np.array([-1,  0,  0,  0, 0, delta, 1])

    # 测试不同的仿射变换
    transformations = [[1, 0], [0, 1], [1, 1], [1, 2], [-2, -1], [-2, 1]]
    # 对于每个变换操作中的旋转变换，计算旋转后的坐标 (x_rot, y_rot)
    for transformation in transformations:
        x_rot = transformation[0]*x0 + transformation[1]*y0
        y_rot = -transformation[1]*x0 + transformation[0]*y0
        # 使用旋转后的坐标 (x_rot, y_rot) 作为新的坐标 (x, y)
        (x, y) = (x_rot, y_rot)
        # 根据给定的线性变换参数计算 z 值
        z = 1.23*x - 4.79*y
        # 定义一个包含六个三角形的列表，每个三角形由其顶点索引组成
        triangles = [[0, 2, 1], [0, 3, 2], [0, 4, 3], [1, 2, 5], [2, 3, 5],
                     [3, 4, 5], [1, 5, 6], [4, 6, 5]]
        # 创建 Triangulation 对象 triang，基于旋转后的坐标 (x, y) 和定义的三角形列表
        triang = mtri.Triangulation(x, y, triangles)
        # 在 x 和 y 的最小到最大范围内创建均匀分布的网格点
        xs = np.linspace(np.min(triang.x), np.max(triang.x), 20)
        ys = np.linspace(np.min(triang.y), np.max(triang.y), 20)
        # 创建 xs 和 ys 的网格
        xs, ys = np.meshgrid(xs, ys)
        xs = xs.ravel()  # 将 xs 展平为一维数组
        ys = ys.ravel()  # 将 ys 展平为一维数组
        # 根据 triang 对象的三角形查找器，确定位于三角形外部的点，并创建遮罩数组
        mask_out = (triang.get_trifinder()(xs, ys) == -1)
        # 创建一个带遮罩的 masked array zs_target，使用线性插值计算目标值
        zs_target = np.ma.array(1.23*xs - 4.79*ys, mask=mask_out)

        # 创建三种不同的插值方法：线性插值、最小能量三次插值、几何三次插值
        linear_interp = mtri.LinearTriInterpolator(triang, z)
        cubic_min_E = mtri.CubicTriInterpolator(triang, z)
        cubic_geom = mtri.CubicTriInterpolator(triang, z, kind='geom')

        # 对每种插值方法进行测试，确保计算结果与目标值 zs_target 几乎相等
        for interp in (linear_interp, cubic_min_E, cubic_geom):
            zs = interp(xs, ys)
            assert_array_almost_equal(zs_target, zs)

        # 在平坦的三角形编号为 4 的区域进行插值测试，即三角形 [2, 3, 5]
        # 通过在调用 _interpolate_multikeys 方法时传入 tri_index 来实现
        itri = 4
        pt1 = triang.triangles[itri, 0]
        pt2 = triang.triangles[itri, 1]
        # 在三角形的边界上创建均匀分布的点 xs 和 ys
        xs = np.linspace(triang.x[pt1], triang.x[pt2], 10)
        ys = np.linspace(triang.y[pt1], triang.y[pt2], 10)
        # 计算在三角形内部的目标值 zs_target
        zs_target = 1.23*xs - 4.79*ys
        # 对每种插值方法进行测试，确保计算结果与目标值 zs_target 几乎相等
        for interp in (linear_interp, cubic_min_E, cubic_geom):
            zs, = interp._interpolate_multikeys(
                xs, ys, tri_index=itri*np.ones(10, dtype=np.int32))
            assert_array_almost_equal(zs_target, zs)
def test_triinterp_transformations():
    # 1) 测试插值方案在整个图形旋转时是否不变。
    # 注意：对于具有 kind='min_E' 的 CubicTriInterpolator，这个测试对于非各向同性刚度矩阵 E 的 _ReducedHCT_Element 是非平凡的（使用 E=np.diag([1., 1., 1.]）），并且是 :meth:`get_Kff_and_Ff` 的同一类的良好测试。
    #
    # 2) 同样测试插值方案在整个图形沿着一个轴扩展时是否不变。
    n_angles = 20
    n_radii = 10
    min_radius = 0.15

    def z(x, y):
        r1 = np.hypot(0.5 - x, 0.5 - y)
        theta1 = np.arctan2(0.5 - x, 0.5 - y)
        r2 = np.hypot(-x - 0.2, -y - 0.2)
        theta2 = np.arctan2(-x - 0.2, -y - 0.2)
        z = -(2*(np.exp((r1/10)**2)-1)*30. * np.cos(7.*theta1) +
              (np.exp((r2/10)**2)-1)*30. * np.cos(11.*theta2) +
              0.7*(x**2 + y**2))
        return (np.max(z)-z)/(np.max(z)-np.min(z))

    # 首先创建点的 x 和 y 坐标。
    radii = np.linspace(min_radius, 0.95, n_radii)
    angles = np.linspace(0 + n_angles, 2*np.pi + n_angles,
                         n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi/n_angles
    x0 = (radii*np.cos(angles)).flatten()
    y0 = (radii*np.sin(angles)).flatten()
    triang0 = mtri.Triangulation(x0, y0)  # Delaunay 三角剖分
    z0 = z(x0, y0)

    # 然后创建测试点
    xs0 = np.linspace(-1., 1., 23)
    ys0 = np.linspace(-1., 1., 23)
    xs0, ys0 = np.meshgrid(xs0, ys0)
    xs0 = xs0.ravel()
    ys0 = ys0.ravel()

    interp_z0 = {}
    for i_angle in range(2):
        # 旋转所有点
        theta = 2*np.pi / n_angles * i_angle
        x = np.cos(theta)*x0 + np.sin(theta)*y0
        y = -np.sin(theta)*x0 + np.cos(theta)*y0
        xs = np.cos(theta)*xs0 + np.sin(theta)*ys0
        ys = -np.sin(theta)*xs0 + np.cos(theta)*ys0
        triang = mtri.Triangulation(x, y, triang0.triangles)
        linear_interp = mtri.LinearTriInterpolator(triang, z0)
        cubic_min_E = mtri.CubicTriInterpolator(triang, z0)
        cubic_geom = mtri.CubicTriInterpolator(triang, z0, kind='geom')
        dic_interp = {'lin': linear_interp,
                      'min_E': cubic_min_E,
                      'geom': cubic_geom}
        # 测试插值是否旋转不变...
        for interp_key in ['lin', 'min_E', 'geom']:
            interp = dic_interp[interp_key]
            if i_angle == 0:
                interp_z0[interp_key] = interp(xs0, ys0)  # 存储
            else:
                interpz = interp(xs, ys)
                matest.assert_array_almost_equal(interpz,
                                                 interp_z0[interp_key])

    scale_factor = 987654.3210
    for scaled_axis in ('x', 'y'):
        # 对于每个缩放轴进行操作（沿缩放轴的扩展）

        # 如果当前是在x轴上进行缩放
        if scaled_axis == 'x':
            # 计算x坐标的缩放后的值
            x = scale_factor * x0
            # y坐标保持不变
            y = y0
            # x方向上的网格点坐标进行缩放
            xs = scale_factor * xs0
            # y方向上的网格点坐标保持不变
            ys = ys0
        else:
            # x坐标保持不变
            x = x0
            # 计算y坐标的缩放后的值
            y = scale_factor * y0
            # x方向上的网格点坐标保持不变
            xs = xs0
            # y方向上的网格点坐标进行缩放
            ys = scale_factor * ys0

        # 根据给定的坐标创建三角剖分对象
        triang = mtri.Triangulation(x, y, triang0.triangles)
        # 创建线性三角插值器
        linear_interp = mtri.LinearTriInterpolator(triang, z0)
        # 创建立方体三角插值器（最小能量方法）
        cubic_min_E = mtri.CubicTriInterpolator(triang, z0)
        # 创建立方体三角插值器（几何方法）
        cubic_geom = mtri.CubicTriInterpolator(triang, z0, kind='geom')
        # 将插值器对象存入字典
        dic_interp = {'lin': linear_interp,
                      'min_E': cubic_min_E,
                      'geom': cubic_geom}

        # 测试插值是否在沿着一个轴的扩展下不变
        for interp_key in ['lin', 'min_E', 'geom']:
            # 使用对应的插值器计算插值结果
            interpz = dic_interp[interp_key](xs, ys)
            # 断言插值结果与预期值近似相等
            matest.assert_array_almost_equal(interpz, interp_z0[interp_key])
@image_comparison(['tri_smooth_contouring.png'], remove_text=True, tol=0.072)
# 定义一个测试函数，用于生成三角平滑轮廓图像，并与指定的参考图像进行比较
def test_tri_smooth_contouring():
    # 定义所需参数
    n_angles = 20  # 角度数量
    n_radii = 10   # 半径数量
    min_radius = 0.15  # 最小半径

    # 定义一个复杂的函数 z(x, y)，用于计算高度数据
    def z(x, y):
        r1 = np.hypot(0.5 - x, 0.5 - y)  # 到中心点(0.5, 0.5)的距离
        theta1 = np.arctan2(0.5 - x, 0.5 - y)  # 到中心点的角度
        r2 = np.hypot(-x - 0.2, -y - 0.2)  # 到(-0.2, -0.2)的距离
        theta2 = np.arctan2(-x - 0.2, -y - 0.2)  # 到(-0.2, -0.2)的角度
        # 计算高度数据 z，复杂的叠加计算
        z = -(2*(np.exp((r1/10)**2)-1)*30. * np.cos(7.*theta1) +
              (np.exp((r2/10)**2)-1)*30. * np.cos(11.*theta2) +
              0.7*(x**2 + y**2))
        return (np.max(z)-z)/(np.max(z)-np.min(z))  # 标准化处理后的高度数据

    # 创建 x 和 y 坐标点
    radii = np.linspace(min_radius, 0.95, n_radii)  # 在最小半径到0.95之间线性分布的半径值
    angles = np.linspace(0 + n_angles, 2*np.pi + n_angles, n_angles, endpoint=False)  # 角度分布
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)  # 扩展角度数组
    angles[:, 1::2] += np.pi/n_angles  # 调整奇数列的角度
    x0 = (radii*np.cos(angles)).flatten()  # x 坐标
    y0 = (radii*np.sin(angles)).flatten()  # y 坐标
    triang0 = mtri.Triangulation(x0, y0)  # 创建 Delaunay 三角化对象
    z0 = z(x0, y0)  # 计算 z 值
    triang0.set_mask(np.hypot(x0[triang0.triangles].mean(axis=1),
                              y0[triang0.triangles].mean(axis=1))
                     < min_radius)  # 根据条件设置掩码

    # 生成精细化后的三角网格和对应的 z 值
    refiner = mtri.UniformTriRefiner(triang0)
    tri_refi, z_test_refi = refiner.refine_field(z0, subdiv=4)

    # 设置等高线绘制的水平线
    levels = np.arange(0., 1., 0.025)
    plt.triplot(triang0, lw=0.5, color='0.5')  # 绘制三角网格
    plt.tricontour(tri_refi, z_test_refi, levels=levels, colors="black")  # 绘制三角等高线图


@image_comparison(['tri_smooth_gradient.png'], remove_text=True, tol=0.092)
# 定义一个测试函数，用于生成三角平滑梯度图像，并与指定的参考图像进行比较
def test_tri_smooth_gradient():
    # 定义一个电偶极子电势函数 V(x, y)
    def dipole_potential(x, y):
        """An electric dipole potential V."""  # 电偶极子电势 V 的定义
        r_sq = x**2 + y**2  # 到原点的距离平方
        theta = np.arctan2(y, x)  # 到原点的极角
        z = np.cos(theta)/r_sq  # 根据公式计算电势
        return (np.max(z)-z) / (np.max(z)-np.min(z))  # 标准化处理后的电势值

    # 创建一个三角化对象
    n_angles = 30  # 角度数量
    n_radii = 10   # 半径数量
    min_radius = 0.2  # 最小半径
    radii = np.linspace(min_radius, 0.95, n_radii)  # 在最小半径到0.95之间线性分布的半径值
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)  # 角度分布
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)  # 扩展角度数组
    angles[:, 1::2] += np.pi/n_angles  # 调整奇数列的角度
    x = (radii*np.cos(angles)).flatten()  # x 坐标
    y = (radii*np.sin(angles)).flatten()  # y 坐标
    V = dipole_potential(x, y)  # 计算电势值
    triang = mtri.Triangulation(x, y)  # 创建三角化对象
    triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
                             y[triang.triangles].mean(axis=1))
                    < min_radius)  # 根据条件设置掩码

    # 对数据进行精细化处理，插值电势 V
    refiner = mtri.UniformTriRefiner(triang)
    tri_refi, z_test_refi = refiner.refine_field(V, subdiv=3)

    # 计算电场 (Ex, Ey) 作为 -V 的梯度
    tci = mtri.CubicTriInterpolator(triang, -V)
    Ex, Ey = tci.gradient(triang.x, triang.y)
    E_norm = np.hypot(Ex, Ey)  # 计算电场强度
    # 创建一个新的图形窗口
    plt.figure()
    # 设置图形窗口的纵横比为相等，使得三角剖分绘制不会出现形变
    plt.gca().set_aspect('equal')
    # 绘制三角剖分的边界线，颜色为浅灰色
    plt.triplot(triang, color='0.8')

    # 定义等值线的数值范围
    levels = np.arange(0., 1., 0.01)
    # 选择热度图作为等值线的颜色映射
    cmap = mpl.colormaps['hot']
    # 绘制三角形上的等值线，使用指定的等值线数值范围和颜色映射，设置线宽
    plt.tricontour(tri_refi, z_test_refi, levels=levels, cmap=cmap,
                   linewidths=[2.0, 1.0, 1.0, 1.0])
    
    # 绘制电场向量场的方向图
    plt.quiver(triang.x, triang.y, Ex/E_norm, Ey/E_norm,
               units='xy', scale=10., zorder=3, color='blue',
               width=0.007, headwidth=3., headlength=4.)
    
    # 留下ax.use_sticky_margins作为True，因此视图限制是等值线数据的限制。
def test_tritools():
    # Tests TriAnalyzer.scale_factors on masked triangulation
    # Tests circle_ratios on equilateral and right-angled triangle.
    # 定义 x 坐标数组，包括一些测试点的坐标
    x = np.array([0., 1., 0.5, 0., 2.])
    # 定义 y 坐标数组，对应于 x 的测试点的纵坐标
    y = np.array([0., 0., 0.5*np.sqrt(3.), -1., 1.])
    # 定义三角形顶点索引数组
    triangles = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 4]], dtype=np.int32)
    # 定义一个布尔掩码数组，用于标记三角形是否被遮罩
    mask = np.array([False, False, True], dtype=bool)
    # 创建 Triangulation 对象
    triang = mtri.Triangulation(x, y, triangles, mask=mask)
    # 创建 TriAnalyzer 对象
    analyser = mtri.TriAnalyzer(triang)
    # 断言检查 TriAnalyzer 的 scale_factors 属性是否符合预期
    assert_array_almost_equal(analyser.scale_factors, [1, 1/(1+3**.5/2)])
    # 断言检查 circle_ratios 方法的结果是否符合预期，包括使用 rescale=False 的情况
    assert_array_almost_equal(
        analyser.circle_ratios(rescale=False),
        np.ma.masked_array([0.5, 1./(1.+np.sqrt(2.)), np.nan], mask))

    # Tests circle ratio of a flat triangle
    # 定义另一组测试点的 x 坐标数组
    x = np.array([0., 1., 2.])
    # 定义另一组测试点的 y 坐标数组
    y = np.array([1., 1.+3., 1.+6.])
    # 定义一个三角形顶点索引数组，构成一个三角形
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    # 创建 Triangulation 对象
    triang = mtri.Triangulation(x, y, triangles)
    # 创建 TriAnalyzer 对象
    analyser = mtri.TriAnalyzer(triang)
    # 断言检查 circle_ratios 方法的结果是否符合预期，预期结果为 [0.]
    assert_array_almost_equal(analyser.circle_ratios(), np.array([0.]))

    # Tests TriAnalyzer.get_flat_tri_mask
    # 创建一个 [-1, 1] x [-1, 1] 的网格三角剖分，包含四个角落和中心的“平坦”三角形
    n = 9

    def power(x, a):
        return np.abs(x)**a*np.sign(x)

    # 生成 x 坐标网格
    x = np.linspace(-1., 1., n+1)
    # 生成 y 坐标网格
    x, y = np.meshgrid(power(x, 2.), power(x, 0.25))
    x = x.ravel()
    y = y.ravel()

    # 创建 Triangulation 对象
    triang = mtri.Triangulation(x, y, triangles=meshgrid_triangles(n+1))
    # 创建 TriAnalyzer 对象
    analyser = mtri.TriAnalyzer(triang)
    # 获取平坦三角形的掩码
    mask_flat = analyser.get_flat_tri_mask(0.2)
    # 创建用于验证的掩码数组
    verif_mask = np.zeros(162, dtype=bool)
    # 指定应该为 True 的角落索引
    corners_index = [0, 1, 2, 3, 14, 15, 16, 17, 18, 19, 34, 35, 126, 127,
                     142, 143, 144, 145, 146, 147, 158, 159, 160, 161]
    verif_mask[corners_index] = True
    # 断言检查计算得到的 mask_flat 是否与 verif_mask 相等
    assert_array_equal(mask_flat, verif_mask)

    # 现在包括一个中心的洞（遮罩三角形）。中心也应该被 get_flat_tri_mask 消除。
    # 创建一个新的遮罩数组，表示中心的遮罩三角形
    mask = np.zeros(162, dtype=bool)
    mask[80] = True
    triang.set_mask(mask)
    # 获取更新后的平坦三角形的掩码
    mask_flat = analyser.get_flat_tri_mask(0.2)
    # 指定应该为 True 的中心索引
    center_index = [44, 45, 62, 63, 78, 79, 80, 81, 82, 83, 98, 99, 116, 117]
    verif_mask[center_index] = True
    # 断言检查更新后的 mask_flat 是否与 verif_mask 相等
    assert_array_equal(mask_flat, verif_mask)
    # 将 x_verif 和 y_verif 展平为一维数组
    x_verif = x_verif.ravel()
    y_verif = y_verif.ravel()

    # 创建布尔数组，指示 x_verif * (2.5 + y_verif) 是否在 x_refi * (2.5 + y_refi) 中
    ind1d = np.isin(np.around(x_verif*(2.5+y_verif), 8),
                    np.around(x_refi*(2.5+y_refi), 8))

    # 断言 ind1d 全部为 True
    assert_array_equal(ind1d, True)

    # 测试精细化三角网格的掩码
    refi_mask = refi_triang.mask

    # 计算精细化三角形重心的 x 和 y 坐标
    refi_tri_barycenter_x = np.sum(refi_triang.x[refi_triang.triangles],
                                   axis=1) / 3.
    refi_tri_barycenter_y = np.sum(refi_triang.y[refi_triang.triangles],
                                   axis=1) / 3.

    # 获取三角查找器对象
    tri_finder = triang.get_trifinder()

    # 在精细化三角形的重心位置找到对应的三角形索引
    refi_tri_indices = tri_finder(refi_tri_barycenter_x,
                                  refi_tri_barycenter_y)

    # 获取相应的三角形掩码
    refi_tri_mask = triang.mask[refi_tri_indices]

    # 断言 refi_mask 与 refi_tri_mask 数组相等
    assert_array_equal(refi_mask, refi_tri_mask)

    # 测试三角形编号不影响插值结果
    x = np.asarray([0.0, 1.0, 0.0, 1.0])
    y = np.asarray([0.0, 0.0, 1.0, 1.0])

    # 创建两个三角剖分对象
    triang = [mtri.Triangulation(x, y, [[0, 1, 3], [3, 2, 0]]),
              mtri.Triangulation(x, y, [[0, 1, 3], [2, 0, 3]])]

    # 计算点到原点 (0.3, 0.4) 的距离
    z = np.hypot(x - 0.3, y - 0.4)

    # 精细化两个三角剖分并重新排序点
    xyz_data = []
    for i in range(2):
        refiner = mtri.UniformTriRefiner(triang[i])
        refined_triang, refined_z = refiner.refine_field(z, subdiv=1)
        xyz = np.dstack((refined_triang.x, refined_triang.y, refined_z))[0]
        xyz = xyz[np.lexsort((xyz[:, 1], xyz[:, 0]))]
        xyz_data += [xyz]

    # 断言精细化后的数据点 xyz_data[0] 和 xyz_data[1] 几乎相等
    assert_array_almost_equal(xyz_data[0], xyz_data[1])
@pytest.mark.parametrize('interpolator',  # 使用 pytest 的 parametrize 装饰器，定义测试参数化，interpolator 参数可取 LinearTriInterpolator 或 CubicTriInterpolator
                         [mtri.LinearTriInterpolator,  # 参数化测试的第一个参数为 LinearTriInterpolator 类
                          mtri.CubicTriInterpolator],  # 参数化测试的第二个参数为 CubicTriInterpolator 类
                         ids=['linear', 'cubic'])  # 对应的测试标识分别为 'linear' 和 'cubic'
def test_trirefine_masked(interpolator):
    # Repeated points means we will have fewer triangles than points, and thus
    # get masking.
    x, y = np.mgrid[:2, :2]  # 创建一个 2x2 的网格
    x = np.repeat(x.flatten(), 2)  # 将 x 坐标展开并重复每个值两次
    y = np.repeat(y.flatten(), 2)  # 将 y 坐标展开并重复每个值两次

    z = np.zeros_like(x)  # 创建一个与 x 同样形状的全零数组 z
    tri = mtri.Triangulation(x, y)  # 使用 x, y 创建三角剖分对象
    refiner = mtri.UniformTriRefiner(tri)  # 使用三角剖分对象创建均匀三角剖分器
    interp = interpolator(tri, z)  # 使用给定的插值器对三角剖分进行插值
    refiner.refine_field(z, triinterpolator=interp, subdiv=2)  # 对 z 进行细化处理


def meshgrid_triangles(n):
    """
    Return (2*(N-1)**2, 3) array of triangles to mesh (N, N)-point np.meshgrid.
    """
    tri = []  # 创建一个空列表用于存储三角形
    for i in range(n-1):
        for j in range(n-1):
            a = i + j*n  # 计算三角形的第一个顶点索引
            b = (i+1) + j*n  # 计算三角形的第二个顶点索引
            c = i + (j+1)*n  # 计算三角形的第三个顶点索引
            d = (i+1) + (j+1)*n  # 计算三角形的第四个顶点索引
            tri += [[a, b, d], [a, d, c]]  # 将两个三角形的顶点索引添加到三角形列表中
    return np.array(tri, dtype=np.int32)  # 将三角形列表转换为 NumPy 数组并返回


def test_triplot_return():
    # Check that triplot returns the artists it adds
    ax = plt.figure().add_subplot()  # 创建一个子图对象 ax
    triang = mtri.Triangulation(
        [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0],  # 使用指定的坐标和三角形索引创建 Triangulation 对象 triang
        triangles=[[0, 1, 3], [3, 2, 0]])
    assert ax.triplot(triang, "b-") is not None, \
        'triplot should return the artist it adds'  # 断言 triplot 方法返回的对象不为空，即添加的图形对象


def test_trirefiner_fortran_contiguous_triangles():
    # github issue 4180.  Test requires two arrays of triangles that are
    # identical except that one is C-contiguous and one is fortran-contiguous.
    triangles1 = np.array([[2, 0, 3], [2, 1, 0]])  # 创建一个 C 风格的三角形数组 triangles1
    assert not np.isfortran(triangles1)  # 断言 triangles1 不是 Fortran 风格存储的数组

    triangles2 = np.array(triangles1, copy=True, order='F')  # 创建一个 Fortran 风格的三角形数组 triangles2
    assert np.isfortran(triangles2)  # 断言 triangles2 是 Fortran 风格存储的数组

    x = np.array([0.39, 0.59, 0.43, 0.32])  # 创建 x 坐标数组
    y = np.array([33.99, 34.01, 34.19, 34.18])  # 创建 y 坐标数组
    triang1 = mtri.Triangulation(x, y, triangles1)  # 使用给定的坐标和三角形索引创建 Triangulation 对象 triang1
    triang2 = mtri.Triangulation(x, y, triangles2)  # 使用给定的坐标和三角形索引创建 Triangulation 对象 triang2

    refiner1 = mtri.UniformTriRefiner(triang1)  # 使用 triang1 创建均匀三角剖分器 refiner1
    refiner2 = mtri.UniformTriRefiner(triang2)  # 使用 triang2 创建均匀三角剖分器 refiner2

    fine_triang1 = refiner1.refine_triangulation(subdiv=1)  # 对 triang1 进行细化处理，返回细化后的 Triangulation 对象 fine_triang1
    fine_triang2 = refiner2.refine_triangulation(subdiv=1)  # 对 triang2 进行细化处理，返回细化后的 Triangulation 对象 fine_triang2

    assert_array_equal(fine_triang1.triangles, fine_triang2.triangles)  # 断言两个细化后的三角剖分对象的三角形索引数组相等


def test_qhull_triangle_orientation():
    # github issue 4437.
    xi = np.linspace(-2, 2, 100)  # 创建一个从 -2 到 2 的包含 100 个点的等间距数组 xi
    x, y = map(np.ravel, np.meshgrid(xi, xi))  # 创建 xi 的网格，并将其展开为一维数组 x, y
    w = (x > y - 1) & (x < -1.95) & (y > -1.2)  # 创建一个布尔掩码 w
    x, y = x[w], y[w]  # 根据掩码 w 过滤出符合条件的 x, y 坐标值
    theta = np.radians(25)  # 将角度 25 转换为弧度
    x1 = x*np.cos(theta) - y*np.sin(theta)  # 应用旋转变换到 x 坐标
    y1 = x*np.sin(theta) + y*np.cos(theta)  # 应用旋转变换到 y 坐标

    # Calculate Delaunay triangulation using Qhull.
    triang = mtri.Triangulation(x1, y1)  # 使用旋转后的坐标创建三角剖分对象 triang

    # Neighbors returned by Qhull.
    qhull_neighbors = triang.neighbors  # 获取由 Qhull 计算得到的邻居数组 qhull_neighbors

    # Obtain neighbors using own C++ calculation.
    triang._neighbors = None  # 清除 Triangulation 对象的 _neighbors 属性
    own_neighbors = triang.neighbors  # 使用自己的方法获取邻居数组 own_neighbors

    assert_array_equal(qhull_neighbors, own_neighbors)  # 断言 Qhull 计算的邻居数组与自己的计算结果相等


def test_trianalyzer_mismatched_indices():
    # github issue 4999.
    # 创建一个包含浮点数的 NumPy 数组，表示 x 坐标
    x = np.array([0., 1., 0.5, 0., 2.])
    
    # 创建一个包含浮点数的 NumPy 数组，表示 y 坐标
    y = np.array([0., 0., 0.5*np.sqrt(3.), -1., 1.])
    
    # 创建一个包含整数的 NumPy 数组，表示三角形的顶点索引
    triangles = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 4]], dtype=np.int32)
    
    # 创建一个布尔类型的 NumPy 数组，用于指示哪些三角形是有效的
    mask = np.array([False, False, True], dtype=bool)
    
    # 使用上述定义的 x、y、triangles 和 mask 创建一个三角剖分对象
    triang = mtri.Triangulation(x, y, triangles, mask=mask)
    
    # 使用三角剖分对象创建 TriAnalyzer 分析器对象
    analyser = mtri.TriAnalyzer(triang)
    
    # 调用 TriAnalyzer 对象的 _get_compressed_triangulation 方法，
    # 该方法在较早的 numpy 版本中会引发 VisibleDeprecationWarning
    # 在修复之前会发生此警告。
    analyser._get_compressed_triangulation()
def test_tricontourf_decreasing_levels():
    # github issue 5477.
    # 定义三角网格的节点坐标和对应的值
    x = [0.0, 1.0, 1.0]
    y = [0.0, 0.0, 1.0]
    z = [0.2, 0.4, 0.6]
    # 创建新的图形窗口
    plt.figure()
    # 测试：使用 tricontourf 绘制填充等高线图，并期望引发 ValueError 异常
    with pytest.raises(ValueError):
        plt.tricontourf(x, y, z, [1.0, 0.0])


def test_internal_cpp_api():
    # Following github issue 8197.
    # 导入 matplotlib 的 _tri 模块，确保懒加载模块已加载
    from matplotlib import _tri  # noqa: ensure lazy-loaded module *is* loaded.

    # 创建三角剖分对象，期望引发 TypeError 异常
    with pytest.raises(
            TypeError,
            match=r'__init__\(\): incompatible constructor arguments.'):
        mpl._tri.Triangulation()

    # 创建三角剖分对象，期望引发 ValueError 异常
    with pytest.raises(
            ValueError, match=r'x and y must be 1D arrays of the same length'):
        mpl._tri.Triangulation([], [1], [[]], (), (), (), False)

    x = [0, 1, 1]
    y = [0, 0, 1]
    # 创建三角剖分对象，期望引发 ValueError 异常
    with pytest.raises(
            ValueError,
            match=r'triangles must be a 2D array of shape \(\?,3\)'):
        mpl._tri.Triangulation(x, y, [[0, 1]], (), (), (), False)

    tris = [[0, 1, 2]]
    # 创建三角剖分对象，期望引发 ValueError 异常
    with pytest.raises(
            ValueError,
            match=r'mask must be a 1D array with the same length as the '
                  r'triangles array'):
        mpl._tri.Triangulation(x, y, tris, [0, 1], (), (), False)

    # 创建三角剖分对象，期望引发 ValueError 异常
    with pytest.raises(
            ValueError, match=r'edges must be a 2D array with shape \(\?,2\)'):
        mpl._tri.Triangulation(x, y, tris, (), [[1]], (), False)

    # 创建三角剖分对象，期望引发 ValueError 异常
    with pytest.raises(
            ValueError,
            match=r'neighbors must be a 2D array with the same shape as the '
                  r'triangles array'):
        mpl._tri.Triangulation(x, y, tris, (), (), [[-1]], False)

    # 创建三角剖分对象
    triang = mpl._tri.Triangulation(x, y, tris, (), (), (), False)

    # 计算平面系数，期望引发 ValueError 异常
    with pytest.raises(
            ValueError,
            match=r'z must be a 1D array with the same length as the '
                  r'triangulation x and y arrays'):
        triang.calculate_plane_coefficients([])

    # 设置 mask 属性为 [0, 1]，期望引发 ValueError 异常
    for mask in ([0, 1], None):
        with pytest.raises(
                ValueError,
                match=r'mask must be a 1D array with the same length as the '
                      r'triangles array'):
            triang.set_mask(mask)

    # 设置 mask 属性为 ()，相当于 Python 的 Triangulation 中 mask=None
    triang.set_mask(())
    # 断言获取的边缘数组为空数组
    assert_array_equal(triang.get_edges(), np.empty((0, 2)))

    # 设置 mask 属性为 ()，相当于 Python 的 Triangulation 中 mask=None
    triang.set_mask(())
    # 断言获取的边缘数组为特定值
    assert_array_equal(triang.get_edges(), [[1, 0], [2, 0], [2, 1]])

    # 创建 TriContourGenerator 对象，期望引发 TypeError 异常
    with pytest.raises(
            TypeError,
            match=r'__init__\(\): incompatible constructor arguments.'):
        mpl._tri.TriContourGenerator()

    # 创建 TriContourGenerator 对象，期望引发 ValueError 异常
    with pytest.raises(
            ValueError,
            match=r'z must be a 1D array with the same length as the x and y '
                  r'arrays'):
        mpl._tri.TriContourGenerator(triang, [1])

    z = [0, 1, 2]
    # 创建 TriContourGenerator 对象，期望引发 ValueError 异常
    tcg = mpl._tri.TriContourGenerator(triang, z)

    # 创建填充等高线图，期望引发 ValueError 异常
    with pytest.raises(
            ValueError, match=r'filled contour levels must be increasing'):
        tcg.create_filled_contour(1, 0)
    # 使用 pytest 的断言来测试异常情况：期望 TypeError 异常，且错误消息匹配特定的正则表达式
    with pytest.raises(
            TypeError,
            match=r'__init__\(\): incompatible constructor arguments.'):
        # 尝试实例化 TrapezoidMapTriFinder 类，期望抛出 TypeError 异常
        mpl._tri.TrapezoidMapTriFinder()
    
    # 创建 TrapezoidMapTriFinder 的实例 trifinder，使用给定的 triang 对象进行初始化
    trifinder = mpl._tri.TrapezoidMapTriFinder(triang)
    
    # 使用 pytest 的断言来测试异常情况：期望 ValueError 异常，且错误消息匹配特定的正则表达式
    with pytest.raises(
            ValueError, match=r'x and y must be array-like with same shape'):
        # 调用 trifinder 对象的 find_many 方法，传入参数 [0] 和 [0, 1]
        trifinder.find_many([0], [0, 1])
def test_qhull_large_offset():
    # github issue 8682.
    # 创建包含点坐标的 NumPy 数组 x 和 y
    x = np.asarray([0, 1, 0, 1, 0.5])
    y = np.asarray([0, 0, 1, 1, 0.5])

    # 设置偏移量
    offset = 1e10
    # 创建两个 Triangulation 对象，分别包含原始坐标和偏移后的坐标
    triang = mtri.Triangulation(x, y)
    triang_offset = mtri.Triangulation(x + offset, y + offset)
    # 断言两个 Triangulation 对象的三角形数量相同
    assert len(triang.triangles) == len(triang_offset.triangles)


def test_tricontour_non_finite_z():
    # github issue 10167.
    # 创建包含点坐标的列表 x 和 y
    x = [0, 1, 0, 1]
    y = [0, 0, 1, 1]
    # 创建 Triangulation 对象
    triang = mtri.Triangulation(x, y)
    # 创建新的图形对象
    plt.figure()

    # 使用 pytest 检查 tricontourf 函数对非有限 z 值的处理
    with pytest.raises(ValueError, match='z array must not contain non-finite '
                                         'values within the triangulation'):
        plt.tricontourf(triang, [0, 1, 2, np.inf])

    with pytest.raises(ValueError, match='z array must not contain non-finite '
                                         'values within the triangulation'):
        plt.tricontourf(triang, [0, 1, 2, -np.inf])

    with pytest.raises(ValueError, match='z array must not contain non-finite '
                                         'values within the triangulation'):
        plt.tricontourf(triang, [0, 1, 2, np.nan])

    with pytest.raises(ValueError, match='z must not contain masked points '
                                         'within the triangulation'):
        plt.tricontourf(triang, np.ma.array([0, 1, 2, 3], mask=[1, 0, 0, 0]))


def test_tricontourset_reuse():
    # 如果从一个 tricontour(f) 调用返回的 TriContourSet 对象作为另一个调用的第一个参数，
    # 则底层的 C++ 等高线生成器将被重用。
    # 创建包含点坐标的列表 x, y 和 z
    x = [0.0, 0.5, 1.0]
    y = [0.0, 1.0, 0.0]
    z = [1.0, 2.0, 3.0]
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 创建第一个 TriContourSet 对象
    tcs1 = ax.tricontourf(x, y, z)
    # 创建第二个 TriContourSet 对象
    tcs2 = ax.tricontour(x, y, z)
    # 断言两个 TriContourSet 对象使用了不同的 contour generator
    assert tcs2._contour_generator != tcs1._contour_generator
    # 将第一个 TriContourSet 对象作为参数创建第三个 TriContourSet 对象
    tcs3 = ax.tricontour(tcs1, z)
    # 断言第三个 TriContourSet 对象与第一个 TriContourSet 对象使用相同的 contour generator
    assert tcs3._contour_generator == tcs1._contour_generator


@check_figures_equal()
def test_triplot_with_ls(fig_test, fig_ref):
    # 创建包含点坐标的列表 x 和 y，以及包含数据的二维列表 data
    x = [0, 2, 1]
    y = [0, 0, 1]
    data = [[0, 1, 2]]
    # 在测试图和参考图上创建子图，并使用 ls 参数绘制三角形边界
    fig_test.subplots().triplot(x, y, data, ls='--')
    fig_ref.subplots().triplot(x, y, data, linestyle='--')


def test_triplot_label():
    # 创建包含点坐标的列表 x 和 y，以及包含数据的二维列表 data
    x = [0, 2, 1]
    y = [0, 0, 1]
    data = [[0, 1, 2]]
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制三角形边界并添加标签
    lines, markers = ax.triplot(x, y, data, label='label')
    # 获取图例的处理对象和标签
    handles, labels = ax.get_legend_handles_labels()
    # 断言标签与预期的一致
    assert labels == ['label']
    # 断言处理对象的数量为1
    assert len(handles) == 1
    # 断言处理对象与 lines 对象一致
    assert handles[0] is lines


def test_tricontour_path():
    # 创建包含点坐标的列表 x 和 y
    x = [0, 4, 4, 0, 2]
    y = [0, 0, 4, 4, 2]
    # 创建 Triangulation 对象
    triang = mtri.Triangulation(x, y)
    # 创建图形和坐标轴对象
    _, ax = plt.subplots()

    # 使用 tricontour 绘制等高线，并获取路径
    cs = ax.tricontour(triang, [1, 0, 0, 0, 0], levels=[0.5])
    paths = cs.get_paths()
    # 断言路径的数量为1
    assert len(paths) == 1
    # 断言路径的顶点与预期值近似相等
    expected_vertices = [[2, 0], [1, 1], [0, 2]]
    assert_array_almost_equal(paths[0].vertices, expected_vertices)
    # 断言路径的代码数组与预期值相等
    assert_array_equal(paths[0].codes, [1, 2, 2])
    # 断言路径的多边形近似与预期值相等
    assert_array_almost_equal(
        paths[0].to_polygons(closed_only=False), [expected_vertices])
    # 使用三角剖分对象和指定的数值列表在轴上创建三角等值线图
    cs = ax.tricontour(triang, [0, 0, 0, 0, 1], levels=[0.5])
    # 获取等值线图中的路径
    paths = cs.get_paths()
    # 确保路径列表长度为1
    assert len(paths) == 1
    # 预期的路径顶点列表
    expected_vertices = [[3, 1], [3, 3], [1, 3], [1, 1], [3, 1]]
    # 检查路径对象的顶点数组是否与预期相等
    assert_array_almost_equal(paths[0].vertices, expected_vertices)
    # 检查路径对象的编码数组是否与预期相等
    assert_array_equal(paths[0].codes, [1, 2, 2, 2, 79])
    # 将路径对象转换为多边形并检查其顶点数组是否与预期相等
    assert_array_almost_equal(paths[0].to_polygons(), [expected_vertices])
def test_tricontourf_path():
    x = [0, 4, 4, 0, 2]  # 定义 x 坐标序列
    y = [0, 0, 4, 4, 2]  # 定义 y 坐标序列
    triang = mtri.Triangulation(x, y)  # 使用 x, y 创建三角剖分对象 triang
    _, ax = plt.subplots()  # 创建一个图形和一个轴对象 ax

    # Polygon inside domain
    cs = ax.tricontourf(triang, [0, 0, 0, 0, 1], levels=[0.5, 1.5])  # 在轴对象 ax 上绘制三角形区域，并填充不同区域
    paths = cs.get_paths()  # 获取填充区域的路径
    assert len(paths) == 1  # 断言填充区域路径数量为 1
    expected_vertices = [[3, 1], [3, 3], [1, 3], [1, 1], [3, 1]]  # 预期的顶点坐标列表
    assert_array_almost_equal(paths[0].vertices, expected_vertices)  # 断言填充区域的顶点坐标与预期相近
    assert_array_equal(paths[0].codes, [1, 2, 2, 2, 79])  # 断言填充区域的路径代码
    assert_array_almost_equal(paths[0].to_polygons(), [expected_vertices])  # 断言填充区域可以转换为多边形，与预期的顶点列表相近

    # Polygon following boundary and inside domain
    cs = ax.tricontourf(triang, [1, 0, 0, 0, 0], levels=[0.5, 1.5])  # 在轴对象 ax 上绘制三角形区域，并填充不同区域
    paths = cs.get_paths()  # 获取填充区域的路径
    assert len(paths) == 1  # 断言填充区域路径数量为 1
    expected_vertices = [[2, 0], [1, 1], [0, 2], [0, 0], [2, 0]]  # 预期的顶点坐标列表
    assert_array_almost_equal(paths[0].vertices, expected_vertices)  # 断言填充区域的顶点坐标与预期相近
    assert_array_equal(paths[0].codes, [1, 2, 2, 2, 79])  # 断言填充区域的路径代码
    assert_array_almost_equal(paths[0].to_polygons(), [expected_vertices])  # 断言填充区域可以转换为多边形，与预期的顶点列表相近

    # Polygon is outer boundary with hole
    cs = ax.tricontourf(triang, [0, 0, 0, 0, 1], levels=[-0.5, 0.5])  # 在轴对象 ax 上绘制三角形区域，并填充不同区域
    paths = cs.get_paths()  # 获取填充区域的路径
    assert len(paths) == 1  # 断言填充区域路径数量为 1
    expected_vertices = [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0],  # 预期的顶点坐标列表
                         [1, 1], [1, 3], [3, 3], [3, 1], [1, 1]]
    assert_array_almost_equal(paths[0].vertices, expected_vertices)  # 断言填充区域的顶点坐标与预期相近
    assert_array_equal(paths[0].codes, [1, 2, 2, 2, 79, 1, 2, 2, 2, 79])  # 断言填充区域的路径代码
    assert_array_almost_equal(paths[0].to_polygons(), np.split(expected_vertices, [5]))  # 断言填充区域可以转换为多边形，与预期的顶点列表相近
```