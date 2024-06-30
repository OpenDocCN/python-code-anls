# `D:\src\scipysrc\scipy\scipy\spatial\tests\test_qhull.py`

```
# 导入操作系统相关的模块
import os
# 导入用于复制对象的模块
import copy

# 导入NumPy库，并导入一些断言函数用于测试
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_, assert_allclose, assert_array_equal)
# 导入pytest库，并导入raises别名用于断言异常
import pytest
from pytest import raises as assert_raises

# 导入SciPy中的几个模块
import scipy.spatial._qhull as qhull
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi

# 导入itertools模块
import itertools

# 定义一个函数，用于返回输入列表的排序后的元组
def sorted_tuple(x):
    return tuple(sorted(x))

# 定义一个断言函数，用于比较两个无序元组列表是否相等
def assert_unordered_tuple_list_equal(a, b, tpl=tuple):
    # 将NumPy数组转换为列表
    if isinstance(a, np.ndarray):
        a = a.tolist()
    if isinstance(b, np.ndarray):
        b = b.tolist()
    # 对列表中的每个元素应用给定的转换函数，并进行排序
    a = list(map(tpl, a))
    a.sort()
    b = list(map(tpl, b))
    b.sort()
    # 断言排序后的列表是否相等
    assert_equal(a, b)

# 设置NumPy的随机种子
np.random.seed(1234)

# 定义一个包含多个二维点的列表
points = [(0,0), (0,1), (1,0), (1,1), (0.5, 0.5), (0.5, 1.5)]

# 定义一个具有特定数值的NumPy数组，用于测试
pathological_data_1 = np.array([
    [-3.14,-3.14], [-3.14,-2.36], [-3.14,-1.57], [-3.14,-0.79],
    [-3.14,0.0], [-3.14,0.79], [-3.14,1.57], [-3.14,2.36],
    [-3.14,3.14], [-2.36,-3.14], [-2.36,-2.36], [-2.36,-1.57],
    [-2.36,-0.79], [-2.36,0.0], [-2.36,0.79], [-2.36,1.57],
    [-2.36,2.36], [-2.36,3.14], [-1.57,-0.79], [-1.57,0.79],
    [-1.57,-1.57], [-1.57,0.0], [-1.57,1.57], [-1.57,-3.14],
    [-1.57,-2.36], [-1.57,2.36], [-1.57,3.14], [-0.79,-1.57],
    [-0.79,1.57], [-0.79,-3.14], [-0.79,-2.36], [-0.79,-0.79],
    [-0.79,0.0], [-0.79,0.79], [-0.79,2.36], [-0.79,3.14],
    [0.0,-3.14], [0.0,-2.36], [0.0,-1.57], [0.0,-0.79], [0.0,0.0],
    [0.0,0.79], [0.0,1.57], [0.0,2.36], [0.0,3.14], [0.79,-3.14],
    [0.79,-2.36], [0.79,-0.79], [0.79,0.0], [0.79,0.79],
    [0.79,2.36], [0.79,3.14], [0.79,-1.57], [0.79,1.57],
    [1.57,-3.14], [1.57,-2.36], [1.57,2.36], [1.57,3.14],
    [1.57,-1.57], [1.57,0.0], [1.57,1.57], [1.57,-0.79],
    [1.57,0.79], [2.36,-3.14], [2.36,-2.36], [2.36,-1.57],
    [2.36,-0.79], [2.36,0.0], [2.36,0.79], [2.36,1.57],
    [2.36,2.36], [2.36,3.14], [3.14,-3.14], [3.14,-2.36],
    [3.14,-1.57], [3.14,-0.79], [3.14,0.0], [3.14,0.79],
    [3.14,1.57], [3.14,2.36], [3.14,3.14],
])

# 定义另一个特定的NumPy数组，用于测试
pathological_data_2 = np.array([
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1], [0, 0], [0, 1],
    [1, -1 - np.finfo(np.float64).eps], [1, 0], [1, 1],
])

# 包含随机数值的列表，用于测试bug #2850
bug_2850_chunks = [np.random.rand(10, 2),
                   np.array([[0,0], [0,1], [1,0], [1,1]])  # add corners
                   ]

# 同上，并增加额外的块用于测试
bug_2850_chunks_2 = (bug_2850_chunks +
                     [np.random.rand(10, 2),
                      0.25 + np.array([[0,0], [0,1], [1,0], [1,1]])])

# 不同数据集的字典，每个数据集包含特定形状的NumPy数组
DATASETS = {
    'some-points': np.asarray(points),
    'random-2d': np.random.rand(30, 2),
    'random-3d': np.random.rand(30, 3),
    'random-4d': np.random.rand(30, 4),
    'random-5d': np.random.rand(30, 5),
    'random-6d': np.random.rand(10, 6),
    'random-7d': np.random.rand(10, 7),
    'random-8d': np.random.rand(10, 8),
    'pathological-1': pathological_data_1,
    'pathological-2': pathological_data_2
}

# 增量数据集的字典，包含用于bug #2850的数据块和空值
INCREMENTAL_DATASETS = {
    'bug-2850': (bug_2850_chunks, None),
}
    'bug-2850-2': (bug_2850_chunks_2, None),


注释：


# 创建一个字典条目，键为 'bug-2850-2'，值为一个元组
# 元组的第一个元素是变量 bug_2850_chunks_2 的值，第二个元素为 None
}

# 定义一个函数用于生成增量数据集，基于给定的基础数据集
def _add_inc_data(name, chunksize):
    """
    从基础数据集生成增量数据集
    """
    # 获取指定名称的数据集
    points = DATASETS[name]
    # 获取数据集的维度
    ndim = points.shape[1]

    opts = None
    nmin = ndim + 2

    # 根据数据集名称选择选项
    if name == 'some-points':
        # 如果是 'some-points' 数据集，则使用 'QJ Pp' 选项
        opts = 'QJ Pp'
    elif name == 'pathological-1':
        # 如果是 'pathological-1' 数据集，则设置足够的点以获取不同的 x 坐标
        nmin = 12

    # 将数据集分块处理
    chunks = [points[:nmin]]
    for j in range(nmin, len(points), chunksize):
        chunks.append(points[j:j+chunksize])

    # 生成新的数据集名称
    new_name = "%s-chunk-%d" % (name, chunksize)
    # 确保新的数据集名称不在 INCREMENTAL_DATASETS 中
    assert new_name not in INCREMENTAL_DATASETS
    # 将新的数据集加入 INCREMENTAL_DATASETS 中
    INCREMENTAL_DATASETS[new_name] = (chunks, opts)

# 遍历所有的基础数据集，并为每个数据集生成不同大小的增量数据集
for name in DATASETS:
    for chunksize in 1, 4, 16:
        _add_inc_data(name, chunksize)


class Test_Qhull:
    def test_swapping(self):
        # 检查 Qhull 状态切换是否正常工作

        # 创建 Qhull 对象 x
        x = qhull._Qhull(b'v',
                         np.array([[0,0],[0,1],[1,0],[1,1.],[0.5,0.5]]),
                         b'Qz')
        # 深拷贝 x 的 Voronoi 图
        xd = copy.deepcopy(x.get_voronoi_diagram())

        # 创建 Qhull 对象 y
        y = qhull._Qhull(b'v',
                         np.array([[0,0],[0,1],[1,0],[1,2.]]),
                         b'Qz')
        # 深拷贝 y 的 Voronoi 图
        yd = copy.deepcopy(y.get_voronoi_diagram())

        # 再次深拷贝 x 的 Voronoi 图
        xd2 = copy.deepcopy(x.get_voronoi_diagram())
        # 关闭 x 对象
        x.close()
        # 再次深拷贝 y 的 Voronoi 图
        yd2 = copy.deepcopy(y.get_voronoi_diagram())
        # 关闭 y 对象
        y.close()

        # 断言 x 和 y 对象的 get_voronoi_diagram 方法会抛出 RuntimeError
        assert_raises(RuntimeError, x.get_voronoi_diagram)
        assert_raises(RuntimeError, y.get_voronoi_diagram)

        # 断言 xd 和 xd2 的各个部分相等
        assert_allclose(xd[0], xd2[0])
        assert_unordered_tuple_list_equal(xd[1], xd2[1], tpl=sorted_tuple)
        assert_unordered_tuple_list_equal(xd[2], xd2[2], tpl=sorted_tuple)
        assert_unordered_tuple_list_equal(xd[3], xd2[3], tpl=sorted_tuple)
        assert_array_equal(xd[4], xd2[4])

        # 断言 yd 和 yd2 的各个部分相等
        assert_allclose(yd[0], yd2[0])
        assert_unordered_tuple_list_equal(yd[1], yd2[1], tpl=sorted_tuple)
        assert_unordered_tuple_list_equal(yd[2], yd2[2], tpl=sorted_tuple)
        assert_unordered_tuple_list_equal(yd[3], yd2[3], tpl=sorted_tuple)
        assert_array_equal(yd[4], yd2[4])

        # 再次关闭 x 和 y 对象
        x.close()
        assert_raises(RuntimeError, x.get_voronoi_diagram)
        y.close()
        assert_raises(RuntimeError, y.get_voronoi_diagram)

    def test_issue_8051(self):
        # 检查问题 8051 是否得到解决

        # 定义一个包含特定点的 numpy 数组
        points = np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],[2, 0], [2, 1], [2, 2]]
        )
        # 创建 Voronoi 对象并传入点集
        Voronoi(points)


class TestUtilities:
    """
    检查实用函数是否正常工作
    """
    def test_find_simplex(self):
        # Simple check that simplex finding works
        
        # 定义一个包含四个点的二维数组，数据类型为浮点数
        points = np.array([(0,0), (0,1), (1,1), (1,0)], dtype=np.float64)
        
        # 使用 Qhull 库的 Delaunay 函数构建三角剖分对象
        tri = qhull.Delaunay(points)

        # +---+
        # |\ 0|
        # | \ |
        # |1 \|
        # +---+

        # 断言三角剖分对象的简单xes属性等于预期值
        assert_equal(tri.simplices, [[1, 3, 2], [3, 1, 0]])

        # 针对预定义的点列表进行迭代
        for p in [(0.25, 0.25, 1),
                  (0.75, 0.75, 0),
                  (0.3, 0.2, 1)]:
            # 使用三角剖分对象的find_simplex方法查找点的简单x索引
            i = tri.find_simplex(p[:2])
            # 断言简单x索引等于预期值，如果不等则输出详细错误信息
            assert_equal(i, p[2], err_msg=f'{p!r}')
            # 使用 Qhull 库的 tsearch 函数也对点进行简单搜索，并与find_simplex的结果进行比较
            j = qhull.tsearch(tri, p[:2])
            assert_equal(i, j)

    def test_plane_distance(self):
        # Compare plane distance from hyperplane equations obtained from Qhull
        # to manually computed plane equations
        
        # 定义包含五个点的二维数组，数据类型为浮点数
        x = np.array([(0,0), (1, 1), (1, 0), (0.99189033, 0.37674127),
                      (0.99440079, 0.45182168)], dtype=np.float64)
        
        # 定义一个点的二维数组，数据类型为浮点数
        p = np.array([0.99966555, 0.15685619], dtype=np.float64)

        # 使用 Qhull 库的 Delaunay 函数构建三角剖分对象
        tri = qhull.Delaunay(x)

        # 将所有点提升到高维空间
        z = tri.lift_points(x)
        pz = tri.lift_points(p)

        # 计算指定点到各平面的距离
        dist = tri.plane_distance(p)

        # 遍历三角剖分对象的简单xes属性中的每个简单x
        for j, v in enumerate(tri.simplices):
            x1 = z[v[0]]
            x2 = z[v[1]]
            x3 = z[v[2]]

            # 计算简单x的法向量
            n = np.cross(x1 - x3, x2 - x3)
            n /= np.sqrt(np.dot(n, n))
            n *= -np.sign(n[2])

            # 计算指定点到当前简单x的距离
            d = np.dot(n, pz - x3)

            # 断言计算出的距离等于预期距离
            assert_almost_equal(dist[j], d)

    def test_convex_hull(self):
        # Simple check that the convex hull seems to works
        
        # 定义一个包含四个点的二维数组，数据类型为浮点数
        points = np.array([(0,0), (0,1), (1,1), (1,0)], dtype=np.float64)
        
        # 使用 Qhull 库的 Delaunay 函数构建三角剖分对象
        tri = qhull.Delaunay(points)

        # +---+
        # |\ 0|
        # | \ |
        # |1 \|
        # +---+

        # 断言三角剖分对象的凸包属性等于预期值
        assert_equal(tri.convex_hull, [[3, 2], [1, 2], [1, 0], [3, 0]])

    def test_volume_area(self):
        #Basic check that we get back the correct volume and area for a cube
        
        # 定义一个包含八个点的三维数组，表示立方体的顶点，数据类型为浮点数
        points = np.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0),
                           (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)])
        
        # 使用 Qhull 库的 ConvexHull 函数构建凸包对象
        hull = qhull.ConvexHull(points)

        # 断言立方体的体积等于预期值，相对误差为1e-14
        assert_allclose(hull.volume, 1., rtol=1e-14,
                        err_msg="Volume of cube is incorrect")
        
        # 断言立方体的表面积等于预期值，相对误差为1e-14
        assert_allclose(hull.area, 6., rtol=1e-14,
                        err_msg="Area of cube is incorrect")
    def test_random_volume_area(self):
        """Test that the results for a random 10-point convex are
        coherent with the output of qconvex Qt s FA"""
        # 定义一个包含 10 个随机点的数组
        points = np.array([(0.362568364506, 0.472712355305, 0.347003084477),
                           (0.733731893414, 0.634480295684, 0.950513180209),
                           (0.511239955611, 0.876839441267, 0.418047827863),
                           (0.0765906233393, 0.527373281342, 0.6509863541),
                           (0.146694972056, 0.596725793348, 0.894860986685),
                           (0.513808585741, 0.069576205858, 0.530890338876),
                           (0.512343805118, 0.663537132612, 0.037689295973),
                           (0.47282965018, 0.462176697655, 0.14061843691),
                           (0.240584597123, 0.778660020591, 0.722913476339),
                           (0.951271745935, 0.967000673944, 0.890661319684)])
        # 使用 qhull 库计算凸包
        hull = qhull.ConvexHull(points)
        # 检查凸包的体积是否与预期值相符
        assert_allclose(hull.volume, 0.14562013, rtol=1e-07,
                        err_msg="Volume of random polyhedron is incorrect")
        # 检查凸包的表面积是否与预期值相符
        assert_allclose(hull.area, 1.6670425, rtol=1e-07,
                        err_msg="Area of random polyhedron is incorrect")

    def test_incremental_volume_area_random_input(self):
        """Test that incremental mode gives the same volume/area as
        non-incremental mode and incremental mode with restart"""
        # 定义测试用的点的数量和维度
        nr_points = 20
        dim = 3
        # 创建一个随机生成的点的数组
        points = np.random.random((nr_points, dim))
        # 创建增量模式下的凸包对象和带重启的增量模式下的凸包对象
        inc_hull = qhull.ConvexHull(points[:dim+1, :], incremental=True)
        inc_restart_hull = qhull.ConvexHull(points[:dim+1, :], incremental=True)
        # 遍历剩余的点，依次添加到凸包对象中，并进行断言检查
        for i in range(dim+1, nr_points):
            hull = qhull.ConvexHull(points[:i+1, :])
            inc_hull.add_points(points[i:i+1, :])
            inc_restart_hull.add_points(points[i:i+1, :], restart=True)
            # 检查凸包的体积在增量模式和标准模式下是否一致
            assert_allclose(hull.volume, inc_hull.volume, rtol=1e-7)
            # 检查凸包的体积在增量模式（带重启）和标准模式下是否一致
            assert_allclose(hull.volume, inc_restart_hull.volume, rtol=1e-7)
            # 检查凸包的表面积在增量模式和标准模式下是否一致
            assert_allclose(hull.area, inc_hull.area, rtol=1e-7)
            # 检查凸包的表面积在增量模式（带重启）和标准模式下是否一致
            assert_allclose(hull.area, inc_restart_hull.area, rtol=1e-7)
    def _check_barycentric_transforms(self, tri, err_msg="",
                                      unit_cube=False,
                                      unit_cube_tol=0):
        """Check that a triangulation has reasonable barycentric transforms"""
        vertices = tri.points[tri.simplices]  # 提取三角剖分的顶点坐标
        sc = 1/(tri.ndim + 1.0)  # 计算标量值 sc，用于后续计算
        centroids = vertices.sum(axis=1) * sc  # 计算每个简单形状的质心坐标

        # Either: (i) the simplex has a `nan` barycentric transform,
        # or, (ii) the centroid is in the simplex

        def barycentric_transform(tr, x):
            r = tr[:,-1,:]  # 获取简单形状的参考点
            Tinv = tr[:,:-1,:]  # 获取简单形状的逆转换矩阵
            return np.einsum('ijk,ik->ij', Tinv, x - r)  # 计算重心坐标变换

        eps = np.finfo(float).eps  # 获取浮点数类型的机器精度

        c = barycentric_transform(tri.transform, centroids)  # 计算质心坐标的变换值
        with np.errstate(invalid="ignore"):
            ok = np.isnan(c).all(axis=1) | (abs(c - sc)/sc < 0.1).all(axis=1)  # 检查变换值是否合理

        assert_(ok.all(), f"{err_msg} {np.nonzero(~ok)}")  # 断言所有变换值都合理，否则输出错误消息

        # Invalid simplices must be (nearly) zero volume
        q = vertices[:,:-1,:] - vertices[:,-1,None,:]  # 计算简单形状的边缘向量
        volume = np.array([np.linalg.det(q[k,:,:])
                           for k in range(tri.nsimplex)])  # 计算简单形状的体积
        ok = np.isfinite(tri.transform[:,0,0]) | (volume < np.sqrt(eps))  # 检查简单形状的体积是否合理
        assert_(ok.all(), f"{err_msg} {np.nonzero(~ok)}")  # 断言所有简单形状的体积都合理，否则输出错误消息

        # Also, find_simplex for the centroid should end up in some
        # simplex for the non-degenerate cases
        j = tri.find_simplex(centroids)  # 找到质心坐标所在的简单形状索引
        ok = (j != -1) | np.isnan(tri.transform[:,0,0])  # 检查质心坐标是否在有效的简单形状内
        assert_(ok.all(), f"{err_msg} {np.nonzero(~ok)}")  # 断言所有质心坐标都在有效的简单形状内，否则输出错误消息

        if unit_cube:
            # If in unit cube, no interior point should be marked out of hull
            at_boundary = (centroids <= unit_cube_tol).any(axis=1)  # 检查质心坐标是否在单位立方体边界内
            at_boundary |= (centroids >= 1 - unit_cube_tol).any(axis=1)
            ok = (j != -1) | at_boundary  # 检查质心坐标是否在有效的简单形状内或者在单位立方体边界内
            assert_(ok.all(), f"{err_msg} {np.nonzero(~ok)}")  # 断言所有质心坐标都在有效的简单形状内或者在单位立方体边界内，否则输出错误消息

    @pytest.mark.fail_slow(10)
    def test_degenerate_barycentric_transforms(self):
        # The triangulation should not produce invalid barycentric
        # transforms that stump the simplex finding
        data = np.load(os.path.join(os.path.dirname(__file__), 'data',
                                    'degenerate_pointset.npz'))  # 加载退化点集数据
        points = data['c']  # 获取点集的坐标数据
        data.close()  # 关闭数据文件

        tri = qhull.Delaunay(points)  # 进行三角剖分

        # Check that there are not too many invalid simplices
        bad_count = np.isnan(tri.transform[:,0,0]).sum()  # 统计无效简单形状的数量
        assert_(bad_count < 23, bad_count)  # 断言无效简单形状的数量小于 23，否则输出数量

        # Check the transforms
        self._check_barycentric_transforms(tri)  # 调用函数检查三角剖分的重心坐标变换

    @pytest.mark.slow
    @pytest.mark.fail_slow(20)
    # OK per https://github.com/scipy/scipy/pull/20487#discussion_r1572684869
    def test_more_barycentric_transforms(self):
        # Triangulate some "nasty" grids
        # 对一些复杂的网格进行三角化

        eps = np.finfo(float).eps
        # 计算浮点数的最小精度

        npoints = {2: 70, 3: 11, 4: 5, 5: 3}
        # 不同维度下的网格点数量

        for ndim in range(2, 6):
            # Generate an uniform grid in n-d unit cube
            # 在 n 维单位立方体中生成均匀网格
            x = np.linspace(0, 1, npoints[ndim])
            grid = np.c_[
                list(map(np.ravel, np.broadcast_arrays(*np.ix_(*([x]*ndim)))))
            ].T

            err_msg = "ndim=%d" % ndim
            # 错误消息，指示当前维度

            # Check using regular grid
            # 使用常规网格进行检查
            tri = qhull.Delaunay(grid)
            self._check_barycentric_transforms(tri, err_msg=err_msg,
                                               unit_cube=True)

            # Check with eps-perturbations
            # 使用 eps 扰动进行检查
            np.random.seed(1234)
            m = (np.random.rand(grid.shape[0]) < 0.2)
            grid[m,:] += 2*eps*(np.random.rand(*grid[m,:].shape) - 0.5)

            tri = qhull.Delaunay(grid)
            self._check_barycentric_transforms(tri, err_msg=err_msg,
                                               unit_cube=True,
                                               unit_cube_tol=2*eps)

            # Check with duplicated data
            # 使用重复数据进行检查
            tri = qhull.Delaunay(np.r_[grid, grid])
            self._check_barycentric_transforms(tri, err_msg=err_msg,
                                               unit_cube=True,
                                               unit_cube_tol=2*eps)
class TestVertexNeighborVertices:
    # 测试顶点及其相邻顶点的功能
    def _check(self, tri):
        # 为每个顶点创建一个空集合，以期望的格式存储顶点的相邻顶点
        expected = [set() for j in range(tri.points.shape[0])]
        # 遍历每个三角形（简单xes）
        for s in tri.simplices:
            # 对于每对顶点a和b，如果它们不相等，则将b添加到a的期望相邻顶点集合中
            for a in s:
                for b in s:
                    if a != b:
                        expected[a].add(b)

        # 获取三角网格对象tri的顶点相邻顶点的索引指针和索引数组
        indptr, indices = tri.vertex_neighbor_vertices

        # 获取实际得到的顶点相邻顶点集合列表
        got = [set(map(int, indices[indptr[j]:indptr[j+1]]))
               for j in range(tri.points.shape[0])]

        # 断言实际得到的相邻顶点集合与期望的相同，如果不同则抛出错误信息
        assert_equal(got, expected, err_msg=f"{got!r} != {expected!r}")

    def test_triangle(self):
        # 创建三角形的测试点集
        points = np.array([(0,0), (0,1), (1,0)], dtype=np.float64)
        # 创建Delaney三角化对象tri
        tri = qhull.Delaunay(points)
        # 执行检查函数_check来验证三角化tri
        self._check(tri)

    def test_rectangle(self):
        # 创建矩形的测试点集
        points = np.array([(0,0), (0,1), (1,1), (1,0)], dtype=np.float64)
        # 创建Delaney三角化对象tri
        tri = qhull.Delaunay(points)
        # 执行检查函数_check来验证三角化tri
        self._check(tri)

    def test_complicated(self):
        # 创建复杂形状的测试点集
        points = np.array([(0,0), (0,1), (1,1), (1,0),
                           (0.5, 0.5), (0.9, 0.5)], dtype=np.float64)
        # 创建Delaney三角化对象tri
        tri = qhull.Delaunay(points)
        # 执行检查函数_check来验证三角化tri
        self._check(tri)


class TestDelaunay:
    """
    Check that triangulation works.

    """
    def test_masked_array_fails(self):
        # 创建带掩码数组的测试输入
        masked_array = np.ma.masked_all(1)
        # 断言使用掩码数组初始化Delaunay对象会引发值错误
        assert_raises(ValueError, qhull.Delaunay, masked_array)

    def test_array_with_nans_fails(self):
        # 创建包含NaN值的测试输入点集
        points_with_nan = np.array([(0,0), (0,1), (1,1), (1,np.nan)], dtype=np.float64)
        # 断言使用包含NaN值的数组初始化Delaunay对象会引发值错误
        assert_raises(ValueError, qhull.Delaunay, points_with_nan)

    def test_nd_simplex(self):
        # 对n维简单xes进行简单的烟雾测试：三角化n维简单xes
        for nd in range(2, 8):
            # 创建n维简单xes的测试点集
            points = np.zeros((nd+1, nd))
            for j in range(nd):
                points[j,j] = 1.0
            points[-1,:] = 1.0

            # 创建n维简单xes的Delaney三角化对象tri
            tri = qhull.Delaunay(points)

            # 对三角面元进行排序
            tri.simplices.sort()

            # 断言三角面元的顺序与预期的顺序相同
            assert_equal(tri.simplices, np.arange(nd+1, dtype=int)[None, :])
            # 断言三角面元的相邻面元为空
            assert_equal(tri.neighbors, -1 + np.zeros((nd+1), dtype=int)[None,:])

    def test_2d_square(self):
        # 对2维正方形进行简单的烟雾测试
        points = np.array([(0,0), (0,1), (1,1), (1,0)], dtype=np.float64)
        # 创建2维正方形的Delaney三角化对象tri
        tri = qhull.Delaunay(points)

        # 断言三角面元的顺序与预期的顺序相同
        assert_equal(tri.simplices, [[1, 3, 2], [3, 1, 0]])
        # 断言三角面元的相邻面元与预期相同
        assert_equal(tri.neighbors, [[-1, -1, 1], [-1, -1, 0]])

    def test_duplicate_points(self):
        # 创建包含重复点的测试输入
        x = np.array([0, 1, 0, 1], dtype=np.float64)
        y = np.array([0, 0, 1, 1], dtype=np.float64)

        # 创建带有重复点的扩展数组
        xp = np.r_[x, x]
        yp = np.r_[y, y]

        # 不应该在包含重复点的输入上失败
        qhull.Delaunay(np.c_[x, y])
        qhull.Delaunay(np.c_[xp, yp])
    def test_pathological(self):
        # 测试函数，用于检验边界情况

        # 使用预定义的数据集 'pathological-1' 进行测试
        points = DATASETS['pathological-1']
        # 使用 qhull.Delaunay 构建 Delaunay 三角网
        tri = qhull.Delaunay(points)
        # 断言：三角网中所有三角形顶点的最大值应与输入点集的最大值相等
        assert_equal(tri.points[tri.simplices].max(), points.max())
        # 断言：三角网中所有三角形顶点的最小值应与输入点集的最小值相等
        assert_equal(tri.points[tri.simplices].min(), points.min())

        # 使用预定义的数据集 'pathological-2' 进行测试
        points = DATASETS['pathological-2']
        # 使用 qhull.Delaunay 构建 Delaunay 三角网
        tri = qhull.Delaunay(points)
        # 断言：三角网中所有三角形顶点的最大值应与输入点集的最大值相等
        assert_equal(tri.points[tri.simplices].max(), points.max())
        # 断言：三角网中所有三角形顶点的最小值应与输入点集的最小值相等
        assert_equal(tri.points[tri.simplices].min(), points.min())

    def test_joggle(self):
        # 测试函数，用于检验选项 QJ 确保所有输入点都作为三角网的顶点

        # 生成一个随机的 10x2 的点集
        points = np.random.rand(10, 2)
        # 将点集数据复制一份，以测试重复输入数据的情况
        points = np.r_[points, points]

        # 使用 qhull.Delaunay 构建 Delaunay 三角网，指定选项为 "QJ Qbb Pp"
        tri = qhull.Delaunay(points, qhull_options="QJ Qbb Pp")
        # 断言：三角网中所有简单形的顶点应该是从 0 到点集长度的连续整数
        assert_array_equal(np.unique(tri.simplices.ravel()),
                           np.arange(len(points)))

    def test_coplanar(self):
        # 测试函数，用于检验共面点输出选项的有效性

        # 生成一个随机的 10x2 的点集
        points = np.random.rand(10, 2)
        # 将点集数据复制一份，以测试重复输入数据的情况
        points = np.r_[points, points]

        # 使用 qhull.Delaunay 构建 Delaunay 三角网
        tri = qhull.Delaunay(points)

        # 断言：三角网中所有简单形的唯一顶点数量应该是点集长度的一半
        assert_(len(np.unique(tri.simplices.ravel())) == len(points)//2)
        # 断言：三角网的共面点数量应该是点集长度的一半
        assert_(len(tri.coplanar) == len(points)//2)
        # 断言：三角网的共面点的 z 坐标唯一值的数量应该是点集长度的一半
        assert_(len(np.unique(tri.coplanar[:,2])) == len(points)//2)
        # 断言：所有顶点到简单形索引的映射应该大于等于 0
        assert_(np.all(tri.vertex_to_simplex >= 0))

    def test_furthest_site(self):
        # 测试函数，用于检验最远点选项的功能性

        # 给定一个固定的点集
        points = [(0, 0), (0, 1), (1, 0), (0.5, 0.5), (1.1, 1.1)]
        # 使用 qhull.Delaunay 构建 Delaunay 三角网，启用最远点选项
        tri = qhull.Delaunay(points, furthest_site=True)

        # 从 Qhull 的预期结果中获得的简单形列表
        expected = np.array([(1, 4, 0), (4, 2, 0)])  # from Qhull
        # 断言：三角网的简单形应与预期结果相等
        assert_array_equal(tri.simplices, expected)

    @pytest.mark.parametrize("name", sorted(INCREMENTAL_DATASETS))
    def test_incremental(self, name):
        # 测试三角化的增量构建

        # 从预定义的数据集中获取当前名称对应的数据块和选项
        chunks, opts = INCREMENTAL_DATASETS[name]
        
        # 将所有数据块中的点连接起来形成一个点集
        points = np.concatenate(chunks, axis=0)

        # 使用第一个数据块创建 Delaunay 三角化对象，并开启增量构建模式
        obj = qhull.Delaunay(chunks[0], incremental=True,
                             qhull_options=opts)
        
        # 逐步添加每个数据块中的点到三角化对象中
        for chunk in chunks[1:]:
            obj.add_points(chunk)

        # 使用连接后的所有点创建一个新的 Delaunay 三角化对象
        obj2 = qhull.Delaunay(points)

        # 重新使用第一个数据块创建一个新的 Delaunay 三角化对象，再次开启增量构建模式
        obj3 = qhull.Delaunay(chunks[0], incremental=True,
                              qhull_options=opts)
        
        # 如果数据块数量大于1，则将除第一个外的所有数据块中的点添加到对象中，并进行重启
        if len(chunks) > 1:
            obj3.add_points(np.concatenate(chunks[1:], axis=0),
                            restart=True)

        # 检查增量模式得到的三角化结果与一次性构建模式的结果是否一致
        if name.startswith('pathological'):
            # 对于某些特定情况，增量模式和一次性模式可能会得到不同的但都有效的三角化结果
            # 在绘制时看起来可能正常，但如何进行准确的检查？
            assert_array_equal(np.unique(obj.simplices.ravel()),
                               np.arange(points.shape[0]))
            assert_array_equal(np.unique(obj2.simplices.ravel()),
                               np.arange(points.shape[0]))
        else:
            # 对于一般情况，验证增量模式和一次性模式得到的三角化结果是否相等
            assert_unordered_tuple_list_equal(obj.simplices, obj2.simplices,
                                              tpl=sorted_tuple)

        # 验证第二个三角化对象和第三个增量模式构建得到的对象的结果是否相等
        assert_unordered_tuple_list_equal(obj2.simplices, obj3.simplices,
                                          tpl=sorted_tuple)
# 检查两个从相同点集构造的凸包是否相等
def assert_hulls_equal(points, facets_1, facets_2):
    # 将facets_1和facets_2中的每个面元都转换为排序后的元组，然后放入集合中
    facets_1 = set(map(sorted_tuple, facets_1))
    facets_2 = set(map(sorted_tuple, facets_2))

    # 如果facets_1和facets_2不相等，并且点集的形状是二维的
    if facets_1 != facets_2 and points.shape[1] == 2:
        # 对于特殊情况，直接比较可能失败
        # --- 那么来自Delaunay的凸包与通过其他方法计算的凸包（由于舍入误差等原因）可能不同
        # 根据问题，几乎完全位于凸包上的（三角形内的）点是否作为凸包的顶点包含在内
        #
        # 因此我们检查结果，并在Delaunay的凸包线段是通常凸包的子集时接受它。

        eps = 1000 * np.finfo(float).eps  # 定义一个小的数，用于比较浮点数误差

        # 遍历facets_1中的每个面元
        for a, b in facets_1:
            # 遍历facets_2中的每个面元
            for ap, bp in facets_2:
                t = points[bp] - points[ap]
                t /= np.linalg.norm(t)       # 计算切线
                n = np.array([-t[1], t[0]])  # 计算法线

                # 检查两个线段是否平行于同一条直线
                c1 = np.dot(n, points[b] - points[ap])
                c2 = np.dot(n, points[a] - points[ap])
                if not np.allclose(np.dot(c1, n), 0):
                    continue
                if not np.allclose(np.dot(c2, n), 0):
                    continue

                # 检查线段(a, b)是否包含在(ap, bp)中
                c1 = np.dot(t, points[a] - points[ap])
                c2 = np.dot(t, points[b] - points[ap])
                c3 = np.dot(t, points[bp] - points[ap])
                if c1 < -eps or c1 > c3 + eps:
                    continue
                if c2 < -eps or c2 > c3 + eps:
                    continue

                # 如果通过上述检查，则表明两个凸包相等
                break
            else:
                raise AssertionError("comparison fails")

        # 如果通过所有检查，则认为是相等的
        return

    # 如果不满足上述条件，直接比较facets_1和facets_2是否相等
    assert_equal(facets_1, facets_2)
    def test_hull_consistency_tri(self, name):
        # 检查由 qhull 在 ndim 中返回的凸壳
        # 和从 ndim Delaunay 构造的凸壳是否一致
        points = DATASETS[name]  # 从数据集中获取指定名称的点集合

        tri = qhull.Delaunay(points)  # 使用 qhull 创建 Delaunay 三角剖分
        hull = qhull.ConvexHull(points)  # 使用 qhull 创建凸壳

        assert_hulls_equal(points, tri.convex_hull, hull.simplices)  # 断言凸壳的一致性

        # 检查凸壳极值是否符合预期
        if points.shape[1] == 2:
            assert_equal(np.unique(hull.simplices), np.sort(hull.vertices))  # 对于二维情况，检查顶点顺序
        else:
            assert_equal(np.unique(hull.simplices), hull.vertices)  # 对于其他维度，检查顶点

    @pytest.mark.parametrize("name", sorted(INCREMENTAL_DATASETS))
    def test_incremental(self, name):
        # 测试增量构建凸壳
        chunks, _ = INCREMENTAL_DATASETS[name]  # 获取增量数据集的数据块和元数据
        points = np.concatenate(chunks, axis=0)  # 将数据块连接成一个点集合

        obj = qhull.ConvexHull(chunks[0], incremental=True)  # 使用增量方式创建凸壳对象
        for chunk in chunks[1:]:
            obj.add_points(chunk)  # 逐步添加数据块中的点

        obj2 = qhull.ConvexHull(points)  # 直接使用整体点集创建凸壳对象

        obj3 = qhull.ConvexHull(chunks[0], incremental=True)  # 使用增量方式再次创建凸壳对象
        if len(chunks) > 1:
            obj3.add_points(np.concatenate(chunks[1:], axis=0), restart=True)  # 在增量模式下添加剩余数据

        # 检查增量模式和整体模式下的凸壳是否一致
        assert_hulls_equal(points, obj.simplices, obj2.simplices)
        assert_hulls_equal(points, obj.simplices, obj3.simplices)

    def test_vertices_2d(self):
        # 二维情况下，顶点应按逆时针顺序排列
        np.random.seed(1234)
        points = np.random.rand(30, 2)  # 生成一个随机二维点集

        hull = qhull.ConvexHull(points)  # 使用 qhull 创建凸壳
        assert_equal(np.unique(hull.simplices), np.sort(hull.vertices))  # 断言顶点的顺序

        # 检查顶点是否按逆时针排列
        x, y = hull.points[hull.vertices].T
        angle = np.arctan2(y - y.mean(), x - x.mean())
        assert_(np.all(np.diff(np.unwrap(angle)) > 0))

    def test_volume_area(self):
        # 基本检查立方体的体积和表面积是否正确
        points = np.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0),
                           (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)])  # 定义一个立方体的点集

        tri = qhull.ConvexHull(points)  # 使用 qhull 创建凸壳

        assert_allclose(tri.volume, 1., rtol=1e-14)  # 断言体积是否接近1
        assert_allclose(tri.area, 6., rtol=1e-14)  # 断言表面积是否接近6

    @pytest.mark.parametrize("incremental", [False, True])
    def test_good2d(self, incremental):
        # 确保 QGn 选项给出正确的 "good" 值
        points = np.array([[0.2, 0.2],
                           [0.2, 0.4],
                           [0.4, 0.4],
                           [0.4, 0.2],
                           [0.3, 0.6]])  # 定义一个二维点集

        hull = qhull.ConvexHull(points=points,
                                incremental=incremental,
                                qhull_options='QG4')  # 使用指定选项创建凸壳对象

        expected = np.array([False, True, False, False], dtype=bool)  # 预期的 "good" 值
        actual = hull.good  # 实际的 "good" 值
        assert_equal(actual, expected)  # 断言实际值与预期值是否相等
    @pytest.mark.parametrize("visibility", [
                              "QG4",  # visible=True
                              "QG-4",  # visible=False
                              ])
    @pytest.mark.parametrize("new_gen, expected", [
        # 在凸包内放置 QG4 生成器，使所有的面都不可见
        (np.array([[0.3, 0.7]]),
         np.array([False, False, False, False, False], dtype=bool)),
        # 在正方形的对角相反位置添加一个生成器，
        # 应保留单个可见面并添加一个不可见面
        (np.array([[0.3, -0.7]]),
         np.array([False, True, False, False, False], dtype=bool)),
        # 将正方形顶部可见面分割成两个可见面，
        # visibility 在数组末尾因为 add_points 连接
        (np.array([[0.3, 0.41]]),
         np.array([False, False, False, True, True], dtype=bool)),
        # 根据当前的 Qhull 选项，共面性不影响可见性；
        # 这种情况下，移动一个可见和一个不可见面，并添加一个共面面
        # 简单形式在索引位置 2 是移动的可见面，最后一个是共面面
        (np.array([[0.5, 0.6], [0.6, 0.6]]),
         np.array([False, False, True, False, False], dtype=bool)),
        # 放置新生成器，使其刚好包裹凸包内的查询点，
        # 但只是在双精度限制内
        # 注意：测试精确退化性不如这种情况可预测，可能是因为我们默认的 Qhull 选项处理精度问题
        (np.array([[0.3, 0.6 + 1e-16]]),
         np.array([False, False, False, False, False], dtype=bool)),
        ])
    def test_good2d_incremental_changes(self, new_gen, expected,
                                        visibility):
        # 使用通常的正方形凸包
        # 从 test_good2d 中的生成器
        points = np.array([[0.2, 0.2],
                           [0.2, 0.4],
                           [0.4, 0.4],
                           [0.4, 0.2],
                           [0.3, 0.6]])
        hull = qhull.ConvexHull(points=points,
                                incremental=True,
                                qhull_options=visibility)
        hull.add_points(new_gen)
        actual = hull.good
        if '-' in visibility:
            expected = np.invert(expected)
        assert_equal(actual, expected)

    @pytest.mark.parametrize("incremental", [False, True])


这段代码是一个用于测试凸包处理器的测试函数，使用不同的生成器和可见性选项来测试凸包的行为。
    def test_good2d_no_option(self, incremental):
        # 处理当不存在 "good" 属性的情况
        # 因为未指定 Qgn 或 Qg-n
        points = np.array([[0.2, 0.2],
                           [0.2, 0.4],
                           [0.4, 0.4],
                           [0.4, 0.2],
                           [0.3, 0.6]])
        # 创建二维点集的凸包对象
        hull = qhull.ConvexHull(points=points,
                                incremental=incremental)
        # 获取凸包对象的 "good" 属性值
        actual = hull.good
        # 断言 "good" 属性为 None
        assert actual is None
        # 在增量添加后保持 None 的值
        if incremental:
            hull.add_points(np.zeros((1, 2)))
            actual = hull.good
            assert actual is None

    @pytest.mark.parametrize("incremental", [False, True])
    def test_good2d_inside(self, incremental):
        # 确保 QGn 选项给出正确的 "good" 值
        # 当点 n 在其余点的凸包内时，"good" 全为 False
        points = np.array([[0.2, 0.2],
                           [0.2, 0.4],
                           [0.4, 0.4],
                           [0.4, 0.2],
                           [0.3, 0.3]])
        # 创建二维点集的凸包对象，使用 QG4 选项
        hull = qhull.ConvexHull(points=points,
                                incremental=incremental,
                                qhull_options='QG4')
        # 期望的 "good" 值数组
        expected = np.array([False, False, False, False], dtype=bool)
        # 获取凸包对象的 "good" 属性值，并断言其与期望值相等
        actual = hull.good
        assert_equal(actual, expected)

    @pytest.mark.parametrize("incremental", [False, True])
    def test_good3d(self, incremental):
        # 确保 QGn 选项给出三维图形的正确 "good" 值
        points = np.array([[0.0, 0.0, 0.0],
                           [0.90029516, -0.39187448, 0.18948093],
                           [0.48676420, -0.72627633, 0.48536925],
                           [0.57651530, -0.81179274, -0.09285832],
                           [0.67846893, -0.71119562, 0.18406710]])
        # 创建三维点集的凸包对象，使用 QG0 选项
        hull = qhull.ConvexHull(points=points,
                                incremental=incremental,
                                qhull_options='QG0')
        # 期望的 "good" 值数组
        expected = np.array([True, False, False, False], dtype=bool)
        # 断言凸包对象的 "good" 属性与期望值相等
        assert_equal(hull.good, expected)
class TestVoronoi:

    @pytest.mark.parametrize("qhull_opts, extra_pts", [
        # 定义参数化测试参数：qhull_opts 是 Qhull 的选项，extra_pts 是额外点的数量
        ("Qbb Qc Qz", 1),  # 使用 Qbb Qc Qz 选项，添加一个额外的无穷远点
        ("Qbb Qc", 0),     # 使用 Qbb Qc 选项，没有额外的点
    ])
    @pytest.mark.parametrize("n_pts", [50, 100])
    @pytest.mark.parametrize("ndim", [2, 3])
    def test_point_region_structure(self,
                                    qhull_opts,
                                    n_pts,
                                    extra_pts,
                                    ndim):
        # 测试点区域结构，参见 gh-16773
        rng = np.random.default_rng(7790)
        points = rng.random((n_pts, ndim))  # 生成 n_pts 个随机点，每个点有 ndim 维度
        vor = Voronoi(points, qhull_options=qhull_opts)  # 使用给定的点和选项创建 Voronoi 图
        pt_region = vor.point_region  # 获取每个点所在的区域索引
        assert pt_region.max() == n_pts - 1 + extra_pts  # 断言最大的区域索引应为 n_pts - 1 + extra_pts
        assert pt_region.size == len(vor.regions) - extra_pts  # 断言区域索引数组的大小应为 Voronoi 图的区域数减去额外点数
        assert len(vor.regions) == n_pts + extra_pts  # 断言 Voronoi 图的区域数应为 n_pts 加上额外点数
        assert vor.points.shape[0] == n_pts  # 断言 Voronoi 图中的点数应为 n_pts
        # 如果 Voronoi 图的区域数据结构中有空子列表，它不应被索引，因为它对应于内部添加的无穷远点，而不是生成点
        if extra_pts:
            sublens = [len(x) for x in vor.regions]
            # 只允许一个无穷远点（空区域）
            assert sublens.count(0) == 1
            assert sublens.index(0) not in pt_region

    def test_masked_array_fails(self):
        masked_array = np.ma.masked_all(1)
        assert_raises(ValueError, qhull.Voronoi, masked_array)

    def test_simple(self):
        # 简单的已知 Voronoi 图案例
        points = [(0, 0), (0, 1), (0, 2),
                  (1, 0), (1, 1), (1, 2),
                  (2, 0), (2, 1), (2, 2)]

        # 期望的输出结果字符串
        output = """
        2
        5 10 1
        -10.101 -10.101
           0.5    0.5
           0.5    1.5
           1.5    0.5
           1.5    1.5
        2 0 1
        3 2 0 1
        2 0 2
        3 3 0 1
        4 1 2 4 3
        3 4 0 2
        2 0 3
        3 4 0 3
        2 0 4
        0
        12
        4 0 3 0 1
        4 0 1 0 1
        4 1 4 1 2
        4 1 2 0 2
        4 2 5 0 2
        4 3 4 1 3
        4 3 6 0 3
        4 4 5 2 4
        4 4 7 3 4
        4 5 8 0 4
        4 6 7 0 3
        4 7 8 0 4
        """
        self._compare_qvoronoi(points, output)

    def _compare_qvoronoi(self, points, expected_output):
        # 比较实际计算的 Voronoi 图和预期的输出结果
        pass  # 这个函数用于比较计算的 Voronoi 图和预期的输出结果，但在示例中没有具体实现
    def _compare_qvoronoi(self, points, output, **kw):
        """Compare to output from 'qvoronoi o Fv < data' to Voronoi()"""

        # Parse output
        # 将输出数据解析为二维列表，每行转换为浮点数列表
        output = [list(map(float, x.split())) for x in output.strip().splitlines()]
        nvertex = int(output[1][0])  # 从输出中获取顶点数
        vertices = list(map(tuple, output[3:2+nvertex]))  # 提取顶点坐标并转换为元组列表（不包括无穷远的顶点）
        nregion = int(output[1][1])  # 从输出中获取区域数
        regions = [[int(y)-1 for y in x[1:]]  # 提取每个区域的顶点索引列表（从0开始）
                   for x in output[2+nvertex:2+nvertex+nregion]]
        ridge_points = [[int(y) for y in x[1:3]]  # 提取每条边的端点索引列表
                        for x in output[3+nvertex+nregion:]]
        ridge_vertices = [[int(y)-1 for y in x[3:]]  # 提取每条边连接的顶点索引列表（从0开始）
                          for x in output[3+nvertex+nregion:]]

        # Compare results
        # 使用 qhull 库计算 Voronoi 图
        vor = qhull.Voronoi(points, **kw)

        def sorttuple(x):
            return tuple(sorted(x))

        # 比较计算结果与预期顶点列表是否一致
        assert_allclose(vor.vertices, vertices)
        # 比较计算结果与预期区域列表（以集合形式存储）是否一致
        assert_equal(set(map(tuple, vor.regions)),
                     set(map(tuple, regions)))

        # 将 Voronoi 图中的边信息整理成元组列表，并排序
        p1 = list(zip(list(map(sorttuple, ridge_points)),
                      list(map(sorttuple, ridge_vertices))))
        p2 = list(zip(list(map(sorttuple, vor.ridge_points.tolist())),
                      list(map(sorttuple, vor.ridge_vertices))))
        p1.sort()
        p2.sort()

        # 比较计算结果与预期边信息是否一致
        assert_equal(p1, p2)

    @pytest.mark.parametrize("name", sorted(DATASETS))
    def test_ridges(self, name):
        # Check that the ridges computed by Voronoi indeed separate
        # the regions of nearest neighborhood, by comparing the result
        # to KDTree.

        points = DATASETS[name]

        # 使用 KDTree 构建点集的搜索树
        tree = KDTree(points)
        # 使用 qhull 库计算 Voronoi 图
        vor = qhull.Voronoi(points)

        for p, v in vor.ridge_dict.items():
            # 只考虑有限的边界线段
            if not np.all(np.asarray(v) >= 0):
                continue

            # 计算边界线段的中点
            ridge_midpoint = vor.vertices[v].mean(axis=0)
            d = 1e-6 * (points[p[0]] - ridge_midpoint)

            # 使用 KDTree 查询最近点，验证边界线段的正确性
            dist, k = tree.query(ridge_midpoint + d, k=1)
            assert_equal(k, p[0])

            dist, k = tree.query(ridge_midpoint - d, k=1)
            assert_equal(k, p[1])

    def test_furthest_site(self):
        points = [(0, 0), (0, 1), (1, 0), (0.5, 0.5), (1.1, 1.1)]

        # 用于 furthest_site=True 时的 qhull 命令行输出
        output = """
        2
        3 5 1
        -10.101 -10.101
        0.6000000000000001    0.5
           0.5 0.6000000000000001
        3 0 2 1
        2 0 1
        2 0 2
        0
        3 0 2 1
        5
        4 0 2 0 2
        4 0 4 1 2
        4 0 1 0 1
        4 1 4 0 1
        4 2 4 0 2
        """
        # 调用 _compare_qvoronoi 函数，验证 furthest_site=True 时的计算结果
        self._compare_qvoronoi(points, output, furthest_site=True)

    def test_furthest_site_flag(self):
        points = [(0, 0), (0, 1), (1, 0), (0.5, 0.5), (1.1, 1.1)]

        # 验证 Voronoi() 的 furthest_site 参数设置
        vor = Voronoi(points)
        assert_equal(vor.furthest_site,False)
        vor = Voronoi(points,furthest_site=True)
        assert_equal(vor.furthest_site,True)

    @pytest.mark.fail_slow(10)
    @pytest.mark.parametrize("name", sorted(INCREMENTAL_DATASETS))
    # 使用 pytest 的 parametrize 功能，对测试用例进行参数化，name 参数从 INCREMENTAL_DATASETS 中按字母顺序取值
    def test_incremental(self, name):
        # 测试增量构建三角剖分

        if INCREMENTAL_DATASETS[name][0][0].shape[1] > 3:
            # 如果数据集的维度大于3，测试速度太慢，因此跳过测试（对结果的测试 --- qhull 仍然快速）
            return

        chunks, opts = INCREMENTAL_DATASETS[name]
        # 从 INCREMENTAL_DATASETS 中获取数据块和选项

        points = np.concatenate(chunks, axis=0)
        # 将数据块按行连接，形成一个连续的点集

        obj = qhull.Voronoi(chunks[0], incremental=True,
                             qhull_options=opts)
        # 使用 qhull 库创建 Voronoi 对象，使用首个数据块进行增量构建，应用给定的选项

        for chunk in chunks[1:]:
            obj.add_points(chunk)
        # 对剩余的数据块逐个添加到 Voronoi 对象中

        obj2 = qhull.Voronoi(points)
        # 使用所有点集创建一个新的 Voronoi 对象

        obj3 = qhull.Voronoi(chunks[0], incremental=True,
                             qhull_options=opts)
        # 使用同样的数据块和选项再次创建 Voronoi 对象

        if len(chunks) > 1:
            obj3.add_points(np.concatenate(chunks[1:], axis=0),
                            restart=True)
            # 如果数据块数量大于1，则将除了首个数据块外的所有数据块作为一个整体添加到 obj3 中，重新启动构建

        # 检查增量模式的结果与一次性模式的结果是否一致
        assert_equal(len(obj.point_region), len(obj2.point_region))
        assert_equal(len(obj.point_region), len(obj3.point_region))

        # 增量模式可能导致顶点的顺序不同或重复，需要进行映射
        for objx in obj, obj3:
            vertex_map = {-1: -1}
            # 创建顶点映射字典，初始化为将不存在的顶点映射为自身

            for i, v in enumerate(objx.vertices):
                for j, v2 in enumerate(obj2.vertices):
                    if np.allclose(v, v2):
                        vertex_map[i] = j
                        # 如果找到两个顶点非常接近，则将它们映射到同一个索引

            def remap(x):
                if hasattr(x, '__len__'):
                    return tuple({remap(y) for y in x})
                    # 递归地将对象中的所有元素重新映射
                try:
                    return vertex_map[x]
                    # 返回根据 vertex_map 映射的索引
                except KeyError as e:
                    message = (f"incremental result has spurious vertex "
                               f"at {objx.vertices[x]!r}")
                    raise AssertionError(message) from e
                    # 如果出现未预期的顶点，则抛出错误

            def simplified(x):
                items = set(map(sorted_tuple, x))
                if () in items:
                    items.remove(())
                    # 移除空元组
                items = [x for x in items if len(x) > 1]
                items.sort()
                return items
                # 对列表中的元素进行排序和去重

            assert_equal(
                simplified(remap(objx.regions)),
                simplified(obj2.regions)
                )
            # 检查映射后的区域是否与 obj2 中的区域一致

            assert_equal(
                simplified(remap(objx.ridge_vertices)),
                simplified(obj2.ridge_vertices)
                )
            # 检查映射后的边界顶点是否与 obj2 中的边界顶点一致

            # XXX: 比较 ridge_points --- 不清楚如何准确比较
    # 定义一个名为 Test_HalfspaceIntersection 的测试类
    class Test_HalfspaceIntersection:
        
        # 定义一个用于检查两个数组是否近似相等的方法，要求每行在 arr2 中只出现一次
        def assert_unordered_allclose(self, arr1, arr2, rtol=1e-7):
            """Check that every line in arr1 is only once in arr2"""
            # 检查 arr1 和 arr2 的形状是否相同
            assert_equal(arr1.shape, arr2.shape)

            # 创建一个布尔数组，用于记录 arr2 中是否包含与 arr1 中每行近似相等的行
            truths = np.zeros((arr1.shape[0],), dtype=bool)
            for l1 in arr1:
                # 找到 arr2 中与 l1 近似相等的行的索引
                indexes = np.nonzero((abs(arr2 - l1) < rtol).all(axis=1))[0]
                # 确保只有一个索引符合条件
                assert_equal(indexes.shape, (1,))
                truths[indexes[0]] = True
            # 确保 truths 数组中所有元素都为 True
            assert_(truths.all())

        # 使用 pytest 的参数化装饰器，测试 cube_halfspace_intersection 方法
        @pytest.mark.parametrize("dt", [np.float64, int])
        def test_cube_halfspace_intersection(self, dt):
            # 定义半空间数组
            halfspaces = np.array([[-1, 0, 0],
                                   [0, -1, 0],
                                   [1, 0, -2],
                                   [0, 1, -2]], dtype=dt)
            # 定义可行点
            feasible_point = np.array([1, 1], dtype=dt)

            # 定义预期的交点数组
            points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])

            # 创建 HalfspaceIntersection 对象 hull
            hull = qhull.HalfspaceIntersection(halfspaces, feasible_point)

            # 检查 hull 对象的 intersections 属性是否与预期的 points 数组近似相等
            assert_allclose(hull.intersections, points)

        # 测试 self_dual_polytope_intersection 方法
        def test_self_dual_polytope_intersection(self):
            # 获取 selfdual-4d-polytope.txt 文件的完整路径
            fname = os.path.join(os.path.dirname(__file__), 'data',
                                 'selfdual-4d-polytope.txt')
            # 从文件中读取不等式数组 ineqs
            ineqs = np.genfromtxt(fname)
            # 构造半空间数组 halfspaces
            halfspaces = -np.hstack((ineqs[:, 1:], ineqs[:, :1]))

            # 定义可行点 feas_point
            feas_point = np.array([0., 0., 0., 0.])
            # 创建 HalfspaceIntersection 对象 hs
            hs = qhull.HalfspaceIntersection(halfspaces, feas_point)

            # 检查 hs 对象的 intersections 属性的形状是否为 (24, 4)
            assert_equal(hs.intersections.shape, (24, 4))

            # 检查 hs 对象的 dual_volume 属性是否接近于 32.0
            assert_almost_equal(hs.dual_volume, 32.0)
            # 检查 hs 对象的 dual_facets 列表长度是否为 24，每个 facet 都应包含 6 个元素
            for facet in hs.dual_facets:
                assert_equal(len(facet), 6)

            # 计算半空间与可行点之间的距离，用于生成 dual_points
            dists = halfspaces[:, -1] + halfspaces[:, :-1].dot(feas_point)
            # 检查生成的 dual_points 是否与计算结果 self.assert_unordered_allclose((halfspaces[:, :-1].T/dists).T, hs.dual_points)
            points 中的每个点进行排列组合，检查是否在 intersections 中只出现一次
            points = itertools.permutations([0., 0., 0.5, -0.5])
            for point in points:
                assert_equal(np.sum((hs.intersections == point).all(axis=1)), 1)
    def test_incremental(self):
        # 定义一个立方体的半空间列表
        halfspaces = np.array([[0., 0., -1., -0.5],
                               [0., -1., 0., -0.5],
                               [-1., 0., 0., -0.5],
                               [1., 0., 0., -0.5],
                               [0., 1., 0., -0.5],
                               [0., 0., 1., -0.5]])
        
        # 定义额外的法向量和偏移量，用于切割每个顶点
        extra_normals = np.array([[1., 1., 1.],
                                  [1., 1., -1.],
                                  [1., -1., 1.],
                                  [1, -1., -1.]])
        offsets = np.array([[-1.]] * 8)
        
        # 组合额外的半空间：每个法向量及其相反方向，每个法向量的偏移量为-1
        extra_halfspaces = np.hstack((np.vstack((extra_normals, -extra_normals)),
                                      offsets))

        # 定义可行点（这里是立方体的中心点）
        feas_point = np.array([0., 0., 0.])

        # 创建一个增量式半空间交集对象，使用初始的半空间和可行点
        inc_hs = qhull.HalfspaceIntersection(halfspaces, feas_point, incremental=True)

        # 创建另一个增量式半空间交集对象，使用相同的初始半空间和可行点，用于重启方式
        inc_res_hs = qhull.HalfspaceIntersection(halfspaces, feas_point,
                                                 incremental=True)

        # 遍历额外的半空间
        for i, ehs in enumerate(extra_halfspaces):
            # 向增量式半空间交集对象中添加额外的半空间
            inc_hs.add_halfspaces(ehs[np.newaxis, :])

            # 使用重启方式向另一个增量式半空间交集对象中添加额外的半空间
            inc_res_hs.add_halfspaces(ehs[np.newaxis, :], restart=True)

            # 创建包含所有半空间的新半空间交集对象
            total = np.vstack((halfspaces, extra_halfspaces[:i+1, :]))

            # 创建一个直接计算的半空间交集对象，使用新的半空间和可行点
            hs = qhull.HalfspaceIntersection(total, feas_point)

            # 断言增量式半空间交集对象和使用重启方式的对象的半空间列表相同
            assert_allclose(inc_hs.halfspaces, inc_res_hs.halfspaces)
            # 断言增量式半空间交集对象的半空间列表与直接计算的对象的半空间列表相同
            assert_allclose(inc_hs.halfspaces, hs.halfspaces)

            # 断言直接计算的半空间交集对象的交点与使用重启方式的对象的交点相同
            assert_allclose(hs.intersections, inc_res_hs.intersections)
            # 增量式计算会导致交点的顺序不同于直接计算
            self.assert_unordered_allclose(inc_hs.intersections, hs.intersections)

        # 关闭增量式半空间交集对象
        inc_hs.close()
    def`
    def test_cube(self):
        # 定义立方体的半空间方程组:
        halfspaces = np.array([[-1., 0., 0., 0.],  # x >= 0
                               [1., 0., 0., -1.],  # x <= 1
                               [0., -1., 0., 0.],  # y >= 0
                               [0., 1., 0., -1.],  # y <= 1
                               [0., 0., -1., 0.],  # z >= 0
                               [0., 0., 1., -1.]])  # z <= 1
        # 定义一个点在空间中的位置
        point = np.array([0.5, 0.5, 0.5])

        # 创建半空间交集对象
        hs = qhull.HalfspaceIntersection(halfspaces, point)

        # 预期的 Qhull 输出点集合
        qhalf_points = np.array([
            [-2, 0, 0],
            [2, 0, 0],
            [0, -2, 0],
            [0, 2, 0],
            [0, 0, -2],
            [0, 0, 2]])
        # 预期的 Qhull 输出面集合
        qhalf_facets = [
            [2, 4, 0],
            [4, 2, 1],
            [5, 2, 0],
            [2, 5, 1],
            [3, 4, 1],
            [4, 3, 0],
            [5, 3, 1],
            [3, 5, 0]]

        # 断言：Qhull 输出的面的数量应与半空间交集对象的对偶面数量相等
        assert len(qhalf_facets) == len(hs.dual_facets)
        # 逐一比较每个面的点集合，考虑到面的方向可能不同
        for a, b in zip(qhalf_facets, hs.dual_facets):
            assert set(a) == set(b)  # 面的顶点集合可能顺序不同

        # 断言：Qhull 输出的点集合应与半空间交集对象的对偶点集合近似相等
        assert_allclose(hs.dual_points, qhalf_points)
# 使用 pytest 的 parametrize 装饰器，为 test_gh_20623 函数参数化，测试不同的图表类型
@pytest.mark.parametrize("diagram_type", [Voronoi, qhull.Delaunay])
def test_gh_20623(diagram_type):
    # 创建一个指定种子的随机数生成器
    rng = np.random.default_rng(123)
    # 生成一个形状为 (4, 10, 3) 的随机数组，代表无效的数据格式
    invalid_data = rng.random((4, 10, 3))
    # 使用 pytest 提供的断言，检查是否会引发 ValueError 异常，并且异常信息包含 "dimensions"
    with pytest.raises(ValueError, match="dimensions"):
        # 调用传入的图表类型构造函数，传入无效数据进行测试
        diagram_type(invalid_data)
```