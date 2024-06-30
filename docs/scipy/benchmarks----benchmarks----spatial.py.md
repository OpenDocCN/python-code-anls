# `D:\src\scipysrc\scipy\benchmarks\benchmarks\spatial.py`

```
import numpy as np  # 导入NumPy库，用于数值计算

from .common import Benchmark, LimitedParamBenchmark, safe_import  # 导入自定义模块和函数

with safe_import():  # 使用安全导入上下文管理器
    from scipy.spatial import cKDTree, KDTree  # 导入SciPy中的KD树实现
with safe_import():  # 继续安全导入其他模块
    from scipy.spatial import distance  # 导入距离计算模块
with safe_import():  # 继续安全导入其他模块
    from scipy.spatial import ConvexHull, Voronoi  # 导入凸包和Voronoi图模块
with safe_import():  # 继续安全导入其他模块
    from scipy.spatial import SphericalVoronoi  # 导入球面Voronoi图模块
with safe_import():  # 继续安全导入其他模块
    from scipy.spatial import geometric_slerp  # 导入几何球面插值模块
with safe_import():  # 继续安全导入其他模块
    from scipy.spatial.transform import Rotation  # 导入旋转变换模块


class Build(Benchmark):  # 定义Benchmark类的子类Build，用于性能评估
    params = [  # 参数化测试的参数列表
        [(3,10000,1000), (8,10000,1000), (16,10000,1000)],  # 不同的(m, n, r)参数组合
        ['KDTree', 'cKDTree'],  # 使用的类名称
    ]
    param_names = ['(m, n, r)', 'class']  # 参数名称列表

    def setup(self, mnr, cls_name):  # 设置方法，准备测试所需的数据和对象
        self.cls = KDTree if cls_name == 'KDTree' else cKDTree  # 根据cls_name选择KDTree或cKDTree类
        m, n, r = mnr  # 解包参数元组

        rng = np.random.default_rng(1234)  # 创建随机数生成器对象
        self.data = np.concatenate((rng.standard_normal((n//2,m)),  # 创建数据集
                                    rng.standard_normal((n-n//2,m))+np.ones(m)))

        self.queries = np.concatenate((rng.standard_normal((r//2,m)),  # 创建查询集
                                       rng.standard_normal((r-r//2,m))+np.ones(m)))

    def time_build(self, mnr, cls_name):  # 构建KD树的性能评估方法
        """
        Constructing kd-tree
        =======================
        dim | # points |  time
        """
        m, n, r = mnr  # 解包参数元组
        if cls_name == 'cKDTree_flat':  # 如果cls_name是'cKDTree_flat'
            self.T = self.cls(self.data, leafsize=n)  # 使用leafsize=n构建KDTree或cKDTree对象
        else:
            self.cls(self.data)  # 否则，仅构建KDTree或cKDTree对象


class PresortedDataSetup(Benchmark):  # 定义Benchmark类的子类PresortedDataSetup，用于性能评估和数据准备
    params = [  # 参数化测试的参数列表
        [(3, 10 ** 4, 1000), (8, 10 ** 4, 1000), (16, 10 ** 4, 1000)],  # 不同的(m, n, r)参数组合
        [True, False],  # 是否使用平衡树
        ['random', 'sorted'],  # 数据排序方式
        [0.5]  # 查询半径
    ]
    param_names = ['(m, n, r)', 'balanced', 'order', 'radius']  # 参数名称列表

    def setup(self, mnr, balanced, order, radius):  # 设置方法，准备测试所需的数据和对象
        m, n, r = mnr  # 解包参数元组

        rng = np.random.default_rng(1234)  # 创建随机数生成器对象
        self.data = {
            'random': rng.uniform(size=(n, m)),  # 创建随机数据集
            'sorted': np.repeat(np.arange(n, 0, -1)[:, np.newaxis],  # 创建排序数据集
                                m,
                                axis=1) / n
        }

        self.queries = rng.uniform(size=(r, m))  # 创建查询集
        self.T = cKDTree(self.data.get(order), balanced_tree=balanced)  # 使用cKDTree构建树对象


class BuildUnbalanced(PresortedDataSetup):  # 定义PresortedDataSetup类的子类BuildUnbalanced，用于非平衡数据构建性能评估
    params = PresortedDataSetup.params[:-1]  # 继承PresortedDataSetup类的参数列表，去掉最后一个元素
    param_names = PresortedDataSetup.param_names[:-1]  # 继承PresortedDataSetup类的参数名称列表，去掉最后一个元素

    def setup(self, *args):  # 设置方法，准备测试所需的数据和对象
        super().setup(*args, None)  # 调用父类的setup方法，并传入参数

    def time_build(self, mnr, balanced, order):  # 构建KD树的性能评估方法
        cKDTree(self.data.get(order), balanced_tree=balanced)  # 使用cKDTree构建树对象


class QueryUnbalanced(PresortedDataSetup):  # 定义PresortedDataSetup类的子类QueryUnbalanced，用于非平衡数据查询性能评估
    params = PresortedDataSetup.params[:-1]  # 继承PresortedDataSetup类的参数列表，去掉最后一个元素
    param_names = PresortedDataSetup.param_names[:-1]  # 继承PresortedDataSetup类的参数名称列表，去掉最后一个元素

    def setup(self, *args):  # 设置方法，准备测试所需的数据和对象
        super().setup(*args, None)  # 调用父类的setup方法，并传入参数

    def time_query(self, mnr, balanced, order):  # 查询性能评估方法
        self.T.query(self.queries)  # 查询查询集中的数据点


class RadiusUnbalanced(PresortedDataSetup):  # 定义PresortedDataSetup类的子类RadiusUnbalanced，用于非平衡数据半径查询性能评估
    params = PresortedDataSetup.params[:]  # 继承PresortedDataSetup类的参数列表
    params[0] = [(3, 1000, 30), (8, 1000, 30), (16, 1000, 30)]  # 修改第一个参数组的取值范围
    # 定义一个方法 `time_query_pairs`，接收参数 `self`, `mnr`, `balanced`, `order`, `radius`
    def time_query_pairs(self, mnr, balanced, order, radius):
        # 调用 self.T 对象的 query_pairs 方法，传入参数 radius，用于查询给定半径范围内的成对数据
        self.T.query_pairs(radius)
    
    # 定义一个方法 `time_query_ball_point`，接收参数 `self`, `mnr`, `balanced`, `order`, `radius`
    def time_query_ball_point(self, mnr, balanced, order, radius):
        # 调用 self.T 对象的 query_ball_point 方法，传入参数 self.queries 和 radius，用于查询给定半径范围内的球形数据
        self.T.query_ball_point(self.queries, radius)
LEAF_SIZES = [8, 128]
BOX_SIZES = [None, 0.0, 1.0]

class Query(LimitedParamBenchmark):
    params = [
        # 参数组合列表：不同的(m, n, r)组合
        [(3,10000,1000), (8,10000,1000), (16,10000,1000)],
        # 参数p的取值：1, 2, 无穷大
        [1, 2, np.inf],
        # 参数boxsize的取值：None, 0.0, 1.0
        BOX_SIZES, LEAF_SIZES,
    ]
    # 参数的名称
    param_names = ['(m, n, r)', 'p', 'boxsize', 'leafsize']
    # 参数组合的总数
    num_param_combinations = 21

    @staticmethod
    def do_setup(self, mnr, p, boxsize, leafsize):
        # 解包参数(m, n, r)
        m, n, r = mnr

        # 使用默认随机数生成器创建随机数种子1234
        rng = np.random.default_rng(1234)

        # 生成随机数据矩阵，大小为(n, m)
        self.data = rng.uniform(size=(n, m))
        # 生成查询数据矩阵，大小为(r, m)
        self.queries = rng.uniform(size=(r, m))

        # 使用给定参数创建cKDTree对象T
        self.T = cKDTree(self.data, leafsize=leafsize, boxsize=boxsize)

    def setup(self, mnr, p, boxsize, leafsize):
        # 调用父类LimitedParamBenchmark的setup方法
        LimitedParamBenchmark.setup(self, mnr, p, boxsize, leafsize)
        # 调用自身的do_setup方法
        Query.do_setup(self, mnr, p, boxsize, leafsize)

    def time_query(self, mnr, p, boxsize, leafsize):
        """
        查询kd树
        dim | # points | # queries |  KDTree  | cKDTree | flat cKDTree
        """
        # 对cKDTree对象T执行查询操作，使用参数p
        self.T.query(self.queries, p=p)

    # 保留旧的基准测试结果（如果更改基准测试，请移除此注释）
    time_query.version = (
        "327bc0627d5387347e9cdcf4c52a550c813bb80a859eeb0f3e5bfe6650a8a1db"
    )


class Radius(LimitedParamBenchmark):
    params = [
        # 参数组合列表：仅包含一个(m, n, r)组合
        [(3,10000,1000)],
        # 参数p的取值：1, 2, 无穷大
        [1, 2, np.inf],
        # 参数probe radius的取值：0.2, 0.5
        [0.2, 0.5],
        BOX_SIZES, LEAF_SIZES,
    ]
    # 参数的名称
    param_names = ['(m, n, r)', 'p', 'probe radius', 'boxsize', 'leafsize']
    # 参数组合的总数
    num_param_combinations = 7

    def __init__(self):
        # 设置time_query_pairs方法的参数
        self.time_query_pairs.__func__.params = list(self.params)
        self.time_query_pairs.__func__.params[0] = [(3,1000,30),
                                                    (8,1000,30),
                                                    (16,1000,30)]
        # 设置time_query_ball_point方法和time_query_ball_point_nosort方法的setup方法
        self.time_query_ball_point.__func__.setup = self.setup_query_ball_point
        self.time_query_ball_point_nosort.__func__.setup = self.setup_query_ball_point
        # 设置time_query_pairs方法的setup方法
        self.time_query_pairs.__func__.setup = self.setup_query_pairs

    def setup(self, *args):
        # 空的setup方法，无实际操作
        pass

    def setup_query_ball_point(self, mnr, p, probe_radius, boxsize, leafsize):
        # 调用父类LimitedParamBenchmark的setup方法，指定参数种子为3
        LimitedParamBenchmark.setup(self, mnr, p, probe_radius, boxsize, leafsize,
                                    param_seed=3)
        # 调用Query类的do_setup方法
        Query.do_setup(self, mnr, p, boxsize, leafsize)

    def setup_query_pairs(self, mnr, p, probe_radius, boxsize, leafsize):
        # 设置query_pairs方法的setup方法
        # query_pairs足够快，因此可以运行所有参数组合
        Query.do_setup(self, mnr, p, boxsize, leafsize)

    def time_query_ball_point(self, mnr, p, probe_radius, boxsize, leafsize):
        # 对cKDTree对象T执行ball point查询操作，使用参数probe_radius和p
        self.T.query_ball_point(self.queries, probe_radius, p=p)

    def time_query_ball_point_nosort(self, mnr, p, probe_radius, boxsize, leafsize):
        # 对cKDTree对象T执行不排序的ball point查询操作，使用参数probe_radius和p
        self.T.query_ball_point(self.queries, probe_radius, p=p,
                                return_sorted=False)

    def time_query_pairs(self, mnr, p, probe_radius, boxsize, leafsize):
        # 对cKDTree对象T执行pairs查询操作，使用参数probe_radius和p
        self.T.query_pairs(probe_radius, p=p)
    # 保留旧的基准测试结果（如果更改基准测试，请删除此行）
    time_query_ball_point.version = (
        "e0c2074b35db7e5fca01a43b0fba8ab33a15ed73d8573871ea6feb57b3df4168"
    )
    time_query_pairs.version = (
        "cf669f7d619e81e4a09b28bb3fceaefbdd316d30faf01524ab33d41661a53f56"
    )
class Neighbors(LimitedParamBenchmark):
    # 参数组合列表
    params = [
        [(3,1000,1000),
         (8,1000,1000),
         (16,1000,1000)],  # 不同的 (m, n1, n2) 组合
        [1, 2, np.inf],  # 不同的 p 值
        [0.2, 0.5],  # 不同的 probe radius 值
        BOX_SIZES, LEAF_SIZES,  # 外部定义的 boxsize 和 leafsize 变量
        ['cKDTree', 'cKDTree_weighted'],  # 不同的类别 cls
    ]
    # 参数名称
    param_names = ['(m, n1, n2)', 'p', 'probe radius', 'boxsize', 'leafsize', 'cls']
    # 参数组合总数
    num_param_combinations = 17

    def setup(self, mn1n2, p, probe_radius, boxsize, leafsize, cls):
        # 调用父类的 setup 方法，设置参数
        LimitedParamBenchmark.setup(self, mn1n2, p, probe_radius,
                                    boxsize, leafsize, cls)

        m, n1, n2 = mn1n2

        # 随机生成数据集 data1 和 data2
        self.data1 = np.random.uniform(size=(n1, m))
        self.data2 = np.random.uniform(size=(n2, m))

        # 初始化权重向量 w1 和 w2
        self.w1 = np.ones(n1)
        self.w2 = np.ones(n2)

        # 创建两个 cKDTree 对象 T1 和 T2
        self.T1 = cKDTree(self.data1, boxsize=boxsize, leafsize=leafsize)
        self.T2 = cKDTree(self.data2, boxsize=boxsize, leafsize=leafsize)

    def time_sparse_distance_matrix(self, mn1n2, p, probe_radius,
                                    boxsize, leafsize, cls):
        # 计算稀疏距离矩阵的运行时间
        self.T1.sparse_distance_matrix(self.T2, probe_radius, p=p)

    def time_count_neighbors(self, mn1n2, p, probe_radius, boxsize, leafsize, cls):
        """
        Count neighbors kd-tree
        dim | # points T1 | # points T2 | p | probe radius |  BoxSize | LeafSize | cls
        """
        # 计算邻居数量的运行时间，基于不同的条件
        if cls != 'cKDTree_weighted':
            self.T1.count_neighbors(self.T2, probe_radius, p=p)
        else:
            self.T1.count_neighbors(self.T2, probe_radius,
                                    weights=(self.w1, self.w2), p=p)

    # 保留旧的基准结果 (如果更改基准，请删除此部分)
    time_sparse_distance_matrix.version = (
        "9aa921dce6da78394ab29d949be27953484613dcf9c9632c01ae3973d4b29596"
    )
    time_count_neighbors.version = (
        "830287f1cf51fa6ba21854a60b03b2a6c70b2f2485c3cdcfb19a360e0a7e2ca2"
    )


class CNeighbors(Benchmark):
    params = [
        [
          (2,1000,1000),
          (8,1000,1000),
          (16,1000,1000)
        ],  # 不同的 (m, n1, n2) 组合
        [2, 10, 100, 400, 1000],  # 不同的 Nr 值
    ]
    param_names = ['(m, n1, n2)', 'Nr']

    def setup(self, mn1n2, Nr):
        m, n1, n2 = mn1n2

        # 随机生成数据集 data1 和 data2
        data1 = np.random.uniform(size=(n1, m))
        data2 = np.random.uniform(size=(n2, m))

        # 初始化权重向量 w1 和 w2
        self.w1 = np.ones(len(data1))
        self.w2 = np.ones(len(data2))

        # 创建四个不同叶子大小的 cKDTree 对象 T1d, T2d, T1s, T2s
        self.T1d = cKDTree(data1, leafsize=1)
        self.T2d = cKDTree(data2, leafsize=1)
        self.T1s = cKDTree(data1, leafsize=8)
        self.T2s = cKDTree(data2, leafsize=8)

        # 创建半径范围为 0 到 0.5 的 Nr 个值的数组 r
        self.r = np.linspace(0, 0.5, Nr)

    def time_count_neighbors_deep(self, mn1n2, Nr):
        """
        Count neighbors for a very deep kd-tree
        dim | # points T1 | # points T2 | Nr
        """
        # 计算深度 kd 树邻居数量的运行时间
        self.T1d.count_neighbors(self.T2d, self.r)
    def time_count_neighbors_shallow(self, mn1n2, Nr):
        """
        Count neighbors for a shallow kd-tree
        dim | # points T1 | # points T2 | Nr
        """
        # 调用 self 对象的 T1s 属性的 count_neighbors 方法，
        # 传入 self.T2s 和 self.r 作为参数
        self.T1s.count_neighbors(self.T2s, self.r)
def generate_spherical_points(num_points):
    # 生成球面上的均匀分布点
    # 参考：https://stackoverflow.com/a/23785326
    rng = np.random.default_rng(123)
    # 生成一个形状为 (num_points, 3) 的正态分布随机数数组
    points = rng.normal(size=(num_points, 3))
    # 对每个点进行归一化，使其位于单位球面上
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    return points


def generate_circle_points(num_points):
    # 尝试避免完整圆形的退化问题
    # 到达 2 * pi 时
    angles = np.linspace(0, 1.9999 * np.pi, num_points)
    # 生成一个形状为 (num_points, 2) 的空数组
    points = np.empty(shape=(num_points, 2))
    # 计算每个点的 x 和 y 坐标
    points[..., 0] = np.cos(angles)
    points[..., 1] = np.sin(angles)
    return points


class SphericalVor(Benchmark):
    params = [10, 100, 1000, 5000, 10000]
    param_names = ['num_points']

    def setup(self, num_points):
        # 在设置阶段生成球面上的点
        self.points = generate_spherical_points(num_points)

    def time_spherical_voronoi_calculation(self, num_points):
        """执行球面 Voronoi 计算，但不包括 Voronoi 多边形中顶点的排序。"""
        # 创建 SphericalVoronoi 对象并计时
        SphericalVoronoi(self.points, radius=1, center=np.zeros(3))


class SphericalVorSort(Benchmark):
    params = [10, 100, 1000, 5000, 10000]
    param_names = ['num_points']

    def setup(self, num_points):
        # 在设置阶段生成球面上的点，并创建 SphericalVoronoi 对象
        self.points = generate_spherical_points(num_points)
        self.sv = SphericalVoronoi(self.points, radius=1,
                                   center=np.zeros(3))

    def time_spherical_polygon_vertex_sorting(self, num_points):
        """计时 Spherical Voronoi 代码中的顶点排序操作。"""
        # 计时排序 Voronoi 多边形中顶点的操作
        self.sv.sort_vertices_of_regions()


class SphericalVorAreas(Benchmark):
    params = ([10, 100, 1000, 5000, 10000],
              [2, 3])
    param_names = ['num_points', 'ndim']

    def setup(self, num_points, ndim):
        if ndim == 2:
            center = np.zeros(2)
            # 生成圆形上的点（二维情况）
            self.points = generate_circle_points(num_points)
        else:
            center = np.zeros(3)
            # 生成球面上的点（三维情况）
            self.points = generate_spherical_points(num_points)
        # 创建 SphericalVoronoi 对象并计时球面 Voronoi 计算
        self.sv = SphericalVoronoi(self.points, radius=1,
                                   center=center)

    def time_spherical_polygon_area_calculation(self, num_points, ndim):
        """计时 Spherical Voronoi 代码中的面积计算操作。"""
        # 计时计算 Voronoi 多边形面积的操作
        self.sv.calculate_areas()


class Xdist(Benchmark):
    params = ([10, 100, 1000],
              ['euclidean', 'minkowski', 'cityblock',
               'seuclidean', 'sqeuclidean', 'cosine', 'correlation',
               'hamming', 'jaccard', 'jensenshannon', 'chebyshev', 'canberra',
               'braycurtis', 'mahalanobis', 'yule', 'dice', 'kulczynski1',
               'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath',
               'minkowski-P3'])
    param_names = ['num_points', 'metric']
    def setup(self, num_points, metric):
        # 使用种子值123初始化随机数生成器
        rng = np.random.default_rng(123)
        # 生成一个形状为(num_points, 3)的随机数数组作为点集
        self.points = rng.random((num_points, 3))
        # 将距离度量标准保存在实例变量中
        self.metric = metric
        # 如果距离度量为'minkowski-P3'，则设置额外的参数p=3.0，并使用'minkowski'替换原距离度量
        if metric == 'minkowski-P3':
            self.kwargs = {'p': 3.0}
            self.metric = 'minkowski'
        else:
            # 否则不需要额外的参数
            self.kwargs = {}

    def time_cdist(self, num_points, metric):
        """Time scipy.spatial.distance.cdist over a range of input data
        sizes and metrics.
        """
        # 计时 scipy.spatial.distance.cdist 函数在点集 self.points 上的运行时间，
        # 使用指定的距离度量 self.metric 和额外参数 self.kwargs
        distance.cdist(self.points, self.points, self.metric, **self.kwargs)

    def time_pdist(self, num_points, metric):
        """Time scipy.spatial.distance.pdist over a range of input data
        sizes and metrics.
        """
        # 计时 scipy.spatial.distance.pdist 函数在点集 self.points 上的运行时间，
        # 使用指定的距离度量 self.metric 和额外参数 self.kwargs
        distance.pdist(self.points, self.metric, **self.kwargs)
class SingleDist(Benchmark):
    params = (['euclidean', 'minkowski', 'cityblock',
               'seuclidean', 'sqeuclidean', 'cosine', 'correlation',
               'hamming', 'jaccard', 'jensenshannon', 'chebyshev', 'canberra',
               'braycurtis', 'mahalanobis', 'yule', 'dice', 'kulczynski1',
               'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath',
               'minkowski-P3'])
    param_names = ['metric']

    def setup(self, metric):
        # 使用固定种子生成随机数发生器
        rng = np.random.default_rng(123)
        # 生成一个 2x3 的随机数组作为测试点
        self.points = rng.random((2, 3))
        # 将当前的距离度量标识存储在实例中
        self.metric = metric
        if metric == 'minkowski-P3':
            # 对于 minkowski-P3，使用不同的 p 值（此处为 3.0）
            self.kwargs = {'p': 3.0}
            # 实际使用的距离度量标识为 minkowski
            self.metric = 'minkowski'
        elif metric == 'mahalanobis':
            # 对于 mahalanobis，使用指定的协方差矩阵 VI
            self.kwargs = {'VI': [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]}
        elif metric == 'seuclidean':
            # 对于 seuclidean，使用指定的缩放向量 V
            self.kwargs = {'V': [1, 0.1, 0.1]}
        else:
            # 其他情况下不需要额外的参数
            self.kwargs = {}

    def time_dist(self, metric):
        """Time distance metrics individually (without batching with
        cdist or pdist).
        """
        # 动态调用 distance 模块中对应的距离度量函数，传入相应的参数和关键字参数
        getattr(distance, self.metric)(self.points[0], self.points[1],
                                       **self.kwargs)


class XdistWeighted(Benchmark):
    params = (
        [10, 20, 100],
        ['euclidean', 'minkowski', 'cityblock', 'sqeuclidean', 'cosine',
         'correlation', 'hamming', 'jaccard', 'chebyshev', 'canberra',
         'braycurtis', 'yule', 'dice', 'kulczynski1', 'rogerstanimoto',
         'russellrao', 'sokalmichener', 'sokalsneath', 'minkowski-P3'])
    param_names = ['num_points', 'metric']

    def setup(self, num_points, metric):
        # 使用固定种子生成随机数发生器
        rng = np.random.default_rng(123)
        # 生成一个 num_points x 3 的随机数组作为测试点
        self.points = rng.random((num_points, 3))
        # 将当前的距离度量标识存储在实例中
        self.metric = metric
        if metric == 'minkowski-P3':
            # 对于 minkowski-P3，使用不同的 p 值（此处为 3.0）
            self.kwargs = {'p': 3.0}
            # 实际使用的距离度量标识为 minkowski
            self.metric = 'minkowski'
        else:
            # 其他情况下不需要额外的参数
            self.kwargs = {}
        # 初始化权重数组
        self.weights = np.ones(3)

    def time_cdist(self, num_points, metric):
        """Time scipy.spatial.distance.cdist for weighted distance metrics."""
        # 调用 scipy.spatial.distance.cdist 计算加权距离矩阵
        distance.cdist(self.points, self.points, self.metric, w=self.weights,
                       **self.kwargs)

    def time_pdist(self, num_points, metric):
        """Time scipy.spatial.distance.pdist for weighted distance metrics."""
        # 调用 scipy.spatial.distance.pdist 计算加权距离向量
        distance.pdist(self.points, self.metric, w=self.weights, **self.kwargs)


class SingleDistWeighted(Benchmark):
    params = (['euclidean', 'minkowski', 'cityblock', 'sqeuclidean', 'cosine',
               'correlation', 'hamming', 'jaccard', 'chebyshev', 'canberra',
               'braycurtis', 'yule', 'dice', 'kulczynski1', 'rogerstanimoto',
               'russellrao', 'sokalmichener', 'sokalsneath', 'minkowski-P3'])
    param_names = ['metric']
    # 在类中定义设置方法，初始化对象的属性和参数
    def setup(self, metric):
        # 使用随机数生成器创建一个新的随机数生成器实例
        rng = np.random.default_rng(123)
        # 生成一个 2x3 的随机数组，作为对象的一个属性
        self.points = rng.random((2, 3))
        # 将传入的 metric 参数作为对象的一个属性
        self.metric = metric
        # 如果 metric 是 'minkowski-P3'，则设置特定的参数
        if metric == 'minkowski-P3':
            # p=2 是欧氏度量，这里设置 p=3 的明可夫斯基度量
            self.kwargs = {'p': 3.0, 'w': np.ones(3)}
            # 将 metric 更新为 'minkowski'，以匹配参数
            self.metric = 'minkowski'
        else:
            # 对于其他 metric，设置默认的参数
            self.kwargs = {'w': np.ones(3)}

    # 在类中定义一个方法，用于计算加权距离的时间度量
    def time_dist_weighted(self, metric):
        """Time weighted distance metrics individually (without batching
        with cdist or pdist).
        """
        # 根据 metric 的值调用 distance 模块中的相应函数，计算点之间的加权距离
        getattr(distance, self.metric)(self.points[0], self.points[1],
                                       **self.kwargs)
class ConvexHullBench(Benchmark):
    # 定义 ConvexHullBench 类，继承自 Benchmark 基类
    params = ([10, 100, 1000, 5000], [True, False])
    # 定义参数列表，分别为点数和增量标志
    param_names = ['num_points', 'incremental']
    # 定义参数名称列表，分别为 num_points 和 incremental

    def setup(self, num_points, incremental):
        # 设置测试的前置条件，初始化随机数生成器和随机点集
        rng = np.random.default_rng(123)
        self.points = rng.random((num_points, 3))

    def time_convex_hull(self, num_points, incremental):
        """Time scipy.spatial.ConvexHull over a range of input data sizes
        and settings.
        """
        # 计算 ConvexHull 的执行时间，传入随机点集和增量标志
        ConvexHull(self.points, incremental)


class VoronoiBench(Benchmark):
    # 定义 VoronoiBench 类，继承自 Benchmark 基类
    params = ([10, 100, 1000, 5000, 10000], [False, True])
    # 定义参数列表，分别为点数和 furthest_site 标志
    param_names = ['num_points', 'furthest_site']
    # 定义参数名称列表，分别为 num_points 和 furthest_site

    def setup(self, num_points, furthest_site):
        # 设置测试的前置条件，初始化随机数生成器和随机点集
        rng = np.random.default_rng(123)
        self.points = rng.random((num_points, 3))

    def time_voronoi_calculation(self, num_points, furthest_site):
        """Time conventional Voronoi diagram calculation."""
        # 计算传统 Voronoi 图的计算时间，传入随机点集和 furthest_site 标志
        Voronoi(self.points, furthest_site=furthest_site)


class Hausdorff(Benchmark):
    # 定义 Hausdorff 类，继承自 Benchmark 基类
    params = [10, 100, 1000]
    # 定义参数列表，表示点数
    param_names = ['num_points']
    # 定义参数名称列表，表示 num_points

    def setup(self, num_points):
        # 设置测试的前置条件，初始化随机数生成器和两组随机点集
        rng = np.random.default_rng(123)
        self.points1 = rng.random((num_points, 3))
        self.points2 = rng.random((num_points, 3))

    def time_directed_hausdorff(self, num_points):
        # 计算 directed_hausdorff 的执行时间，传入两组随机点集
        distance.directed_hausdorff(self.points1, self.points2)


class GeometricSlerpBench(Benchmark):
    # 定义 GeometricSlerpBench 类，继承自 Benchmark 基类
    params = [10, 1000, 10000]
    # 定义参数列表，表示点数
    param_names = ['num_points']
    # 定义参数名称列表，表示 num_points

    def setup(self, num_points):
        # 设置测试的前置条件，生成球面上的随机点，并选择两个点作为插值的起始和终止点
        points = generate_spherical_points(50)
        self.start = points[0]
        self.end = points[-1]
        self.t = np.linspace(0, 1, num_points)

    def time_geometric_slerp_3d(self, num_points):
        # 计算 geometric_slerp() 的执行时间，进行 3D 插值计算
        geometric_slerp(start=self.start,
                        end=self.end,
                        t=self.t)


class RotationBench(Benchmark):
    # 定义 RotationBench 类，继承自 Benchmark 基类
    params = [1, 10, 1000, 10000]
    # 定义参数列表，表示旋转数量
    param_names = ['num_rotations']
    # 定义参数名称列表，表示 num_rotations

    def setup(self, num_rotations):
        # 设置测试的前置条件，初始化随机数生成器和随机旋转矩阵
        rng = np.random.default_rng(1234)
        self.rotations = Rotation.random(num_rotations, random_state=rng)

    def time_matrix_conversion(self, num_rotations):
        '''Time converting rotation from and to matrices'''
        # 计算从旋转矩阵到旋转对象的转换时间
        Rotation.from_matrix(self.rotations.as_matrix())

    def time_euler_conversion(self, num_rotations):
        '''Time converting rotation from and to euler angles'''
        # 计算从欧拉角到旋转对象的转换时间
        Rotation.from_euler("XYZ", self.rotations.as_euler("XYZ"))

    def time_rotvec_conversion(self, num_rotations):
        '''Time converting rotation from and to rotation vectors'''
        # 计算从旋转向量到旋转对象的转换时间
        Rotation.from_rotvec(self.rotations.as_rotvec())

    def time_mrp_conversion(self, num_rotations):
        '''Time converting rotation from and to Modified Rodrigues Parameters'''
        # 计算从修改的 Rodrigues 参数到旋转对象的转换时间
        Rotation.from_mrp(self.rotations.as_mrp())
    def time_mul_inv(self, num_rotations):
        '''Time multiplication and inverse of rotations'''
        # 对 rotations 属性进行乘法操作，并将结果赋值给 rotations 属性
        self.rotations * self.rotations.inv()
```