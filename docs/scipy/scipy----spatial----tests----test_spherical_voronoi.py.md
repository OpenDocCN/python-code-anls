# `D:\src\scipysrc\scipy\scipy\spatial\tests\test_spherical_voronoi.py`

```
import numpy as np
import itertools
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           assert_array_almost_equal)
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma

# 设置容差值
TOL = 1E-10

# 生成正四面体的顶点坐标数组
def _generate_tetrahedron():
    return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])

# 生成立方体的顶点坐标数组
def _generate_cube():
    return np.array(list(itertools.product([-1, 1.], repeat=3)))

# 生成八面体的顶点坐标数组
def _generate_octahedron():
    return np.array([[-1, 0, 0], [+1, 0, 0], [0, -1, 0],
                     [0, +1, 0], [0, 0, -1], [0, 0, +1]])

# 生成十二面体的顶点坐标数组
def _generate_dodecahedron():
    x1 = _generate_cube()
    x2 = np.array([[0, -phi, -1 / phi],
                   [0, -phi, +1 / phi],
                   [0, +phi, -1 / phi],
                   [0, +phi, +1 / phi]])
    x3 = np.array([[-1 / phi, 0, -phi],
                   [+1 / phi, 0, -phi],
                   [-1 / phi, 0, +phi],
                   [+1 / phi, 0, +phi]])
    x4 = np.array([[-phi, -1 / phi, 0],
                   [-phi, +1 / phi, 0],
                   [+phi, -1 / phi, 0],
                   [+phi, +1 / phi, 0]])
    return np.concatenate((x1, x2, x3, x4))

# 生成二十面体的顶点坐标数组
def _generate_icosahedron():
    x = np.array([[0, -1, -phi],
                  [0, -1, +phi],
                  [0, +1, -phi],
                  [0, +1, +phi]])
    return np.concatenate([np.roll(x, i, axis=1) for i in range(3)])

# 根据名称生成多面体的顶点坐标数组
def _generate_polytope(name):
    polygons = ["triangle", "square", "pentagon", "hexagon", "heptagon",
                "octagon", "nonagon", "decagon", "undecagon", "dodecagon"]
    polyhedra = ["tetrahedron", "cube", "octahedron", "dodecahedron",
                 "icosahedron"]
    if name not in polygons and name not in polyhedra:
        raise ValueError("unrecognized polytope")

    if name in polygons:
        n = polygons.index(name) + 3
        thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
        p = np.vstack([np.cos(thetas), np.sin(thetas)]).T
    elif name == "tetrahedron":
        p = _generate_tetrahedron()
    elif name == "cube":
        p = _generate_cube()
    elif name == "octahedron":
        p = _generate_octahedron()
    elif name == "dodecahedron":
        p = _generate_dodecahedron()
    elif name == "icosahedron":
        p = _generate_icosahedron()

    return p / np.linalg.norm(p, axis=1, keepdims=True)

# 计算超球面的表面积
def _hypersphere_area(dim, radius):
    # https://en.wikipedia.org/wiki/N-sphere#Closed_forms
    return 2 * np.pi**(dim / 2) / gamma(dim / 2) * radius**(dim - 1)

# 从超球面均匀随机采样点
def _sample_sphere(n, dim, seed=None):
    # Sample points uniformly at random from the hypersphere
    rng = np.random.RandomState(seed=seed)
    points = rng.randn(n, dim)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    # 返回变量 points，这是函数的输出结果
    return points
class TestSphericalVoronoi:
    
    # 初始化方法，设置测试所需的点集合
    def setup_method(self):
        self.points = np.array([
            [-0.78928481, -0.16341094, 0.59188373],
            [-0.66839141, 0.73309634, 0.12578818],
            [0.32535778, -0.92476944, -0.19734181],
            [-0.90177102, -0.03785291, -0.43055335],
            [0.71781344, 0.68428936, 0.12842096],
            [-0.96064876, 0.23492353, -0.14820556],
            [0.73181537, -0.22025898, -0.6449281],
            [0.79979205, 0.54555747, 0.25039913]]
        )

    # 测试构造函数
    def test_constructor(self):
        center = np.array([1, 2, 3])
        radius = 2
        
        # 创建 SphericalVoronoi 对象 s1，使用默认中心和半径
        s1 = SphericalVoronoi(self.points)
        # 创建 SphericalVoronoi 对象 s2，将点缩放并设置半径
        s2 = SphericalVoronoi(self.points * radius, radius)
        # 创建 SphericalVoronoi 对象 s3，将点平移并设置中心
        s3 = SphericalVoronoi(self.points + center, center=center)
        # 创建 SphericalVoronoi 对象 s4，将点同时缩放和平移并设置中心和半径
        s4 = SphericalVoronoi(self.points * radius + center, radius, center)
        
        # 断言检查 s1 的中心和半径是否符合预期
        assert_array_equal(s1.center, np.array([0, 0, 0]))
        assert_equal(s1.radius, 1)
        
        # 断言检查 s2 的中心和半径是否符合预期
        assert_array_equal(s2.center, np.array([0, 0, 0]))
        assert_equal(s2.radius, 2)
        
        # 断言检查 s3 的中心和半径是否符合预期
        assert_array_equal(s3.center, center)
        assert_equal(s3.radius, 1)
        
        # 断言检查 s4 的中心和半径是否符合预期
        assert_array_equal(s4.center, center)
        assert_equal(s4.radius, radius)
        
        # 测试使用 memoryview 的非序列/ndarray 的数组类型
        s5 = SphericalVoronoi(memoryview(self.points))  # type: ignore[arg-type]
        assert_array_equal(s5.center, np.array([0, 0, 0]))
        assert_equal(s5.radius, 1)

    # 测试顶点和区域的平移不变性
    def test_vertices_regions_translation_invariance(self):
        sv_origin = SphericalVoronoi(self.points)
        center = np.array([1, 1, 1])
        sv_translated = SphericalVoronoi(self.points + center, center=center)
        
        # 断言检查原始对象和平移后对象的区域是否相同
        assert_equal(sv_origin.regions, sv_translated.regions)
        # 断言检查顶点是否经过正确平移
        assert_array_almost_equal(sv_origin.vertices + center,
                                  sv_translated.vertices)

    # 测试顶点和区域的缩放不变性
    def test_vertices_regions_scaling_invariance(self):
        sv_unit = SphericalVoronoi(self.points)
        sv_scaled = SphericalVoronoi(self.points * 2, 2)
        
        # 断言检查单位对象和缩放后对象的区域是否相同
        assert_equal(sv_unit.regions, sv_scaled.regions)
        # 断言检查顶点是否经过正确缩放
        assert_array_almost_equal(sv_unit.vertices * 2,
                                  sv_scaled.vertices)

    # 测试旧的半径 API 错误
    def test_old_radius_api_error(self):
        # 使用 pytest 检查是否会引发 ValueError，且匹配指定错误信息
        with pytest.raises(ValueError, match='`radius` is `None`. *'):
            SphericalVoronoi(self.points, radius=None)

    # 测试对区域顶点进行排序
    def test_sort_vertices_of_regions(self):
        sv = SphericalVoronoi(self.points)
        unsorted_regions = sv.regions
        sv.sort_vertices_of_regions()
        
        # 断言检查区域顶点是否已经排序
        assert_equal(sorted(sv.regions), sorted(unsorted_regions))
    def test_sort_vertices_of_regions_flattened(self):
        # 期望结果是将各区域顶点排序后展平的列表
        expected = sorted([[0, 6, 5, 2, 3], [2, 3, 10, 11, 8, 7], [0, 6, 4, 1],
                           [4, 8, 7, 5, 6], [9, 11, 10], [2, 7, 5],
                           [1, 4, 8, 11, 9], [0, 3, 10, 9, 1]])
        # 将期望结果展平为一维列表
        expected = list(itertools.chain(*sorted(expected)))  # type: ignore
        # 创建 SphericalVoronoi 对象，传入点集合 self.points
        sv = SphericalVoronoi(self.points)
        # 对每个区域的顶点进行排序
        sv.sort_vertices_of_regions()
        # 获取实际排序后的顶点列表，并展平为一维列表
        actual = list(itertools.chain(*sorted(sv.regions)))
        # 断言实际结果与期望结果一致
        assert_array_equal(actual, expected)

    def test_sort_vertices_of_regions_dimensionality(self):
        # 定义一个三维的点集合用于测试
        points = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0.5, 0.5, 0.5, 0.5]])
        # 使用 pytest 断言预期会抛出 TypeError 异常，异常信息匹配 "three-dimensional"
        with pytest.raises(TypeError, match="three-dimensional"):
            # 创建 SphericalVoronoi 对象，传入点集合 points
            sv = SphericalVoronoi(points)
            # 对每个区域的顶点进行排序
            sv.sort_vertices_of_regions()

    def test_num_vertices(self):
        # 创建 SphericalVoronoi 对象，传入点集合 self.points
        sv = SphericalVoronoi(self.points)
        # 根据 Euler 公式计算预期的顶点数量
        expected = self.points.shape[0] * 2 - 4
        # 获取实际的顶点数量
        actual = sv.vertices.shape[0]
        # 断言实际结果与期望结果一致
        assert_equal(actual, expected)

    def test_voronoi_circles(self):
        # 创建 SphericalVoronoi 对象，传入点集合 self.points
        sv = SphericalVoronoi(self.points)
        # 对每个顶点进行测试
        for vertex in sv.vertices:
            # 计算当前顶点到所有点的距离
            distances = distance.cdist(sv.points, np.array([vertex]))
            # 取最近的三个距离并排序
            closest = np.array(sorted(distances)[0:3])
            # 断言最近的两个距离非常接近（精度到小数点后第七位），并使用顶点的字符串表示作为描述
            assert_almost_equal(closest[0], closest[1], 7, str(vertex))
            assert_almost_equal(closest[0], closest[2], 7, str(vertex))

    def test_duplicate_point_handling(self):
        # 当输入点集合中有重复点时，应抛出 ValueError 异常
        # 相关于 Issue# 7046
        self.degenerate = np.concatenate((self.points, self.points))
        with assert_raises(ValueError):
            # 创建 SphericalVoronoi 对象，传入含有重复点的点集合 self.degenerate
            SphericalVoronoi(self.degenerate)

    def test_incorrect_radius_handling(self):
        # 当输入的半径值不可能匹配输入的生成器点时，应抛出 ValueError 异常
        with assert_raises(ValueError):
            # 创建 SphericalVoronoi 对象，传入点集合 self.points 和不合适的半径值
            SphericalVoronoi(self.points, radius=0.98)

    def test_incorrect_center_handling(self):
        # 当输入的中心点不可能匹配输入的生成器点时，应抛出 ValueError 异常
        with assert_raises(ValueError):
            # 创建 SphericalVoronoi 对象，传入点集合 self.points 和不合适的中心点
            SphericalVoronoi(self.points, center=[0.1, 0, 0])

    @pytest.mark.parametrize("dim", range(2, 6))
    @pytest.mark.parametrize("shift", [False, True])
    @pytest.mark.parametrize("dim", range(2, 6))
    def test_single_hemisphere_handling(self, dim, shift):
        # 设置球面采样点的数量
        n = 10
        # 生成球面上的采样点
        points = _sample_sphere(n, dim, seed=0)
        # 将球面上的点限制在半球内
        points[:, 0] = np.abs(points[:, 0])
        # 计算球心位置
        center = (np.arange(dim) + 1) * shift
        # 创建球面 Voronoi 图对象
        sv = SphericalVoronoi(points + center, center=center)
        # 计算每个 Voronoi 区域的外接圆半径
        dots = np.einsum('ij,ij->i', sv.vertices - center,
                                     sv.points[sv._simplices[:, 0]] - center)
        circumradii = np.arccos(np.clip(dots, -1, 1))
        # 断言最大外接圆半径大于 pi / 2
        assert np.max(circumradii) > np.pi / 2

    @pytest.mark.parametrize("n", [1, 2, 10])
    @pytest.mark.parametrize("dim", range(2, 6))
    @pytest.mark.parametrize("shift", [False, True])
    def test_rank_deficient(self, n, dim, shift):
        # 计算球心位置
        center = (np.arange(dim) + 1) * shift
        # 生成低维球面上的采样点
        points = _sample_sphere(n, dim - 1, seed=0)
        # 在低维球面上增加一个额外的维度
        points = np.hstack([points, np.zeros((n, 1))])
        # 断言在球心为中心的球面 Voronoi 图中会引发 ValueError 异常
        with pytest.raises(ValueError, match="Rank of input points"):
            SphericalVoronoi(points + center, center=center)

    @pytest.mark.parametrize("dim", range(2, 6))
    def test_higher_dimensions(self, dim):
        # 设置球面采样点的数量
        n = 100
        # 生成高维球面上的采样点
        points = _sample_sphere(n, dim, seed=0)
        # 创建球面 Voronoi 图对象
        sv = SphericalVoronoi(points)
        # 断言 Voronoi 图的顶点数等于维度数
        assert sv.vertices.shape[1] == dim
        # 断言 Voronoi 图的区域数等于采样点数
        assert len(sv.regions) == n

        # 验证欧拉特征
        cell_counts = []
        simplices = np.sort(sv._simplices)
        for i in range(1, dim + 1):
            cells = []
            for indices in itertools.combinations(range(dim), i):
                cells.append(simplices[:, list(indices)])
            cells = np.unique(np.concatenate(cells), axis=0)
            cell_counts.append(len(cells))
        expected_euler = 1 + (-1)**(dim-1)
        actual_euler = sum([(-1)**i * e for i, e in enumerate(cell_counts)])
        # 断言计算得到的欧拉特征与预期值相等
        assert expected_euler == actual_euler

    @pytest.mark.parametrize("dim", range(2, 6))
    def test_cross_polytope_regions(self, dim):
        # The hypercube is the dual of the cross-polytope, so the voronoi
        # vertices of the cross-polytope lie on the points of the hypercube.

        # 生成十字多面体的点
        points = np.concatenate((-np.eye(dim), np.eye(dim)))
        # 创建球面 Voronoi 图对象
        sv = SphericalVoronoi(points)
        # 断言每个 Voronoi 区域的顶点数等于 2^(dim - 1)
        assert all([len(e) == 2**(dim - 1) for e in sv.regions])

        # 生成超立方体的点
        expected = np.vstack(list(itertools.product([-1, 1], repeat=dim)))
        expected = expected.astype(np.float64) / np.sqrt(dim)

        # 验证 Voronoi 图的顶点是否正确放置
        dist = distance.cdist(sv.vertices, expected)
        res = linear_sum_assignment(dist)
        # 断言顶点位置的总距离小于容差值 TOL
        assert dist[res].sum() < TOL

    @pytest.mark.parametrize("dim", range(2, 6))
    # 定义一个测试方法，用于验证超立方体的区域
    def test_hypercube_regions(self, dim):
        # 十字多胞体是超立方体的对偶体，因此超立方体的Voronoi顶点位于十字多胞体的点上。

        # 生成超立方体的点
        points = np.vstack(list(itertools.product([-1, 1], repeat=dim)))
        points = points.astype(np.float64) / np.sqrt(dim)
        # 创建球形Voronoi图对象
        sv = SphericalVoronoi(points)

        # 生成十字多胞体的点
        expected = np.concatenate((-np.eye(dim), np.eye(dim)))

        # 测试Voronoi顶点是否正确放置
        dist = distance.cdist(sv.vertices, expected)
        # 使用匈牙利算法解决分配问题，确保距离之和小于TOL
        res = linear_sum_assignment(dist)
        assert dist[res].sum() < TOL

    # 使用参数化测试，测试区域重构
    @pytest.mark.parametrize("n", [10, 500])
    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("radius", [0.5, 1, 2])
    @pytest.mark.parametrize("shift", [False, True])
    @pytest.mark.parametrize("single_hemisphere", [False, True])
    def test_area_reconstitution(self, n, dim, radius, shift,
                                 single_hemisphere):
        # 从球面上采样点
        points = _sample_sphere(n, dim, seed=0)

        # 如果是单半球测试，将所有点移动到球的一侧
        if single_hemisphere:
            points[:, 0] = np.abs(points[:, 0])

        # 计算中心点
        center = (np.arange(dim) + 1) * shift
        points = radius * points + center

        # 创建球形Voronoi图对象
        sv = SphericalVoronoi(points, radius=radius, center=center)
        # 计算区域面积
        areas = sv.calculate_areas()
        # 断言区域面积之和接近于超球体的面积
        assert_almost_equal(areas.sum(), _hypersphere_area(dim, radius))

    # 使用参数化测试，测试等面积重构
    @pytest.mark.parametrize("poly", ["triangle", "dodecagon",
                                      "tetrahedron", "cube", "octahedron",
                                      "dodecahedron", "icosahedron"])
    def test_equal_area_reconstitution(self, poly):
        # 生成多面体的点
        points = _generate_polytope(poly)
        n, dim = points.shape
        # 创建球形Voronoi图对象
        sv = SphericalVoronoi(points)
        # 计算区域面积
        areas = sv.calculate_areas()
        # 断言区域面积与超球体面积的比值接近于1/n
        assert_almost_equal(areas, _hypersphere_area(dim, 1) / n)

    # 测试不支持的维度下的区域计算
    def test_area_unsupported_dimension(self):
        dim = 4
        # 生成十字多胞体的点
        points = np.concatenate((-np.eye(dim), np.eye(dim)))
        # 创建球形Voronoi图对象
        sv = SphericalVoronoi(points)
        # 断言调用区域计算时会抛出TypeError，并匹配给定的错误信息
        with pytest.raises(TypeError, match="Only supported"):
            sv.calculate_areas()

    # 使用参数化测试，测试属性的数据类型
    @pytest.mark.parametrize("radius", [1, 1.])
    @pytest.mark.parametrize("center", [None, (1, 2, 3), (1., 2., 3.)])
    def test_attribute_types(self, radius, center):
        # 将点乘以半径
        points = radius * self.points
        if center is not None:
            points += center

        # 创建球形Voronoi图对象
        sv = SphericalVoronoi(points, radius=radius, center=center)
        # 断言点的数据类型为np.float64
        assert sv.points.dtype is np.dtype(np.float64)
        # 断言中心点的数据类型为np.float64
        assert sv.center.dtype is np.dtype(np.float64)
        # 断言半径的类型为float
        assert isinstance(sv.radius, float)
    # 定义测试方法，用于测试区域整数类型是否保持不变
    # 查看问题号 #13412
    def test_region_types(self):
        # 创建 SphericalVoronoi 对象，使用给定的点集 self.points
        sv = SphericalVoronoi(self.points)
        # 获取第一个区域的第一个元素的数据类型
        dtype = type(sv.regions[0][0])
        # 同时确保每个区域都是嵌套列表的类型，参见问题 gh-19177
        for region in sv.regions:
            assert isinstance(region, list)
        # 对区域的顶点进行排序
        sv.sort_vertices_of_regions()
        # 断言第一个区域的第一个元素的数据类型仍然与最初获取的相同
        assert type(sv.regions[0][0]) == dtype
        # 再次对区域的顶点进行排序
        sv.sort_vertices_of_regions()
        # 再次断言第一个区域的第一个元素的数据类型仍然与最初获取的相同
        assert type(sv.regions[0][0]) == dtype
```