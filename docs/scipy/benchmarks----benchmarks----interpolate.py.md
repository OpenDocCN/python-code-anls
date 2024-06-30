# `D:\src\scipysrc\scipy\benchmarks\benchmarks\interpolate.py`

```
# 导入必要的库
import numpy as np

# 从本地导入模块和函数
from .common import run_monitored, set_mem_rlimit, Benchmark, safe_import

# 安全导入 scipy.stats 模块中的 spearmanr 函数
with safe_import():
    from scipy.stats import spearmanr

# 安全导入 scipy.interpolate 模块中的 griddata 函数
with safe_import():
    import scipy.interpolate as interpolate

# 安全导入 scipy.sparse 模块中的 csr_matrix 类
with safe_import():
    from scipy.sparse import csr_matrix

# 定义一个 Benchmark 的子类 Leaks
class Leaks(Benchmark):
    # 定义类属性 unit，表示性能测量单位
    unit = "relative increase with repeats"

    # 定义 track_leaks 方法，用于检测内存泄漏
    def track_leaks(self):
        # 设置内存限制
        set_mem_rlimit()

        # 设置重复次数列表
        repeats = [2, 5, 10, 50, 200]
        # 存储每次运行的内存峰值
        peak_mems = []

        # 遍历重复次数列表
        for repeat in repeats:
            # 定义包含测试代码的字符串
            code = """
            import numpy as np
            from scipy.interpolate import griddata

            def func(x, y):
                return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

            grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
            points = np.random.rand(1000, 2)
            values = func(points[:,0], points[:,1])

            for t in range(%(repeat)d):
                for method in ['nearest', 'linear', 'cubic']:
                    griddata(points, values, (grid_x, grid_y), method=method)

            """ % dict(repeat=repeat)

            # 运行带监控的测试代码，获取返回结果中的内存峰值
            _, peak_mem = run_monitored(code)
            peak_mems.append(peak_mem)

        # 计算重复次数与内存峰值之间的 Spearman 相关系数和 p 值
        corr, p = spearmanr(repeats, peak_mems)
        # 如果 p 值小于 0.05，打印可能存在内存泄漏的警告信息
        if p < 0.05:
            print("*"*79)
            print("PROBABLE MEMORY LEAK")
            print("*"*79)
        else:
            print("PROBABLY NO MEMORY LEAK")

        # 返回内存峰值的最大值与最小值之比
        return max(peak_mems) / min(peak_mems)

# 定义 BenchPPoly 类，继承自 Benchmark
class BenchPPoly(Benchmark):

    # 设置测试环境
    def setup(self):
        # 创建随机数生成器
        rng = np.random.default_rng(1234)
        m, k = 55, 3
        # 生成排序后的随机数数组
        x = np.sort(rng.random(m+1))
        # 创建随机系数数组
        c = rng.random((k, m))
        # 使用随机系数和排序后的随机数数组创建 PPoly 对象
        self.pp = interpolate.PPoly(c, x)

        # 设置评估点的数量
        npts = 100
        # 在指定范围内生成均匀间隔的数值
        self.xp = np.linspace(0, 1, npts)

    # 定义 time_evaluation 方法，用于评估 PPoly 对象的性能
    def time_evaluation(self):
        self.pp(self.xp)

# 定义 GridData 类，继承自 Benchmark
class GridData(Benchmark):
    # 定义参数名称列表
    param_names = ['n_grids', 'method']
    # 定义参数组合列表
    params = [
        [10j, 100j, 1000j],    # 不同网格数量
        ['nearest', 'linear', 'cubic']    # 不同插值方法
    ]

    # 设置测试环境
    def setup(self, n_grids, method):
        # 定义函数 func，用于生成测试数据
        self.func = lambda x, y: x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
        # 生成网格坐标
        self.grid_x, self.grid_y = np.mgrid[0:1:n_grids, 0:1:n_grids]
        # 生成随机采样点
        self.points = np.random.rand(1000, 2)
        # 计算采样点对应的函数值
        self.values = self.func(self.points[:, 0], self.points[:, 1])

    # 定义 time_evaluation 方法，用于评估插值函数的性能
    def time_evaluation(self, n_grids, method):
        interpolate.griddata(self.points, self.values, (self.grid_x, self.grid_y),
                             method=method)
    # 设置函数，初始化稀疏矩阵的形状和非零元素数量
    def setup(self):
        shape = (7395, 6408)
        num_nonzero = 488686

        # 使用默认随机数生成器创建随机数发生器对象
        rng = np.random.default_rng(1234)

        # 生成随机的行和列索引
        random_rows = rng.integers(0, shape[0], num_nonzero)
        random_cols = rng.integers(0, shape[1], num_nonzero)

        # 生成指定数量的随机浮点数值
        random_values = rng.random(num_nonzero, dtype=np.float32)

        # 创建稀疏矩阵对象，使用随机行列索引和随机数值填充
        sparse_matrix = csr_matrix((random_values, (random_rows, random_cols)), 
                                   shape=shape, dtype=np.float32)
        
        # 将稀疏矩阵转换为稠密数组
        sparse_matrix = sparse_matrix.toarray()

        # 记录稠密数组中非零元素的坐标
        self.coords = np.column_stack(np.nonzero(sparse_matrix))
        
        # 记录稠密数组中非零元素的值
        self.values = sparse_matrix[self.coords[:, 0], self.coords[:, 1]]
        
        # 创建网格的 x 和 y 坐标
        self.grid_x, self.grid_y = np.mgrid[0:sparse_matrix.shape[0], 
                                            0:sparse_matrix.shape[1]]

    # 计算使用三次样条插值的网格数据
    def peakmem_griddata(self):
        interpolate.griddata(self.coords, self.values, (self.grid_x, self.grid_y), 
                             method='cubic')
class Interpolate1d(Benchmark):
    # 参数名称列表，包括样本数和插值方法
    param_names = ['n_samples', 'method']
    # 参数取值，分别是样本数和插值方法的不同组合
    params = [
        [10, 50, 100, 1000, 10000],  # 不同的样本数
        ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'],  # 不同的插值方法
    ]

    def setup(self, n_samples, method):
        # 创建等间隔的样本点 x
        self.x = np.arange(n_samples)
        # 根据指数函数生成 y 值，用于插值
        self.y = np.exp(-self.x/3.0)
        # 创建插值函数对象，根据不同的插值方法
        self.interpolator = interpolate.interp1d(self.x, self.y, kind=method)
        # 在 x 范围内创建更密集的评估点 xp
        self.xp = np.linspace(self.x[0], self.x[-1], 4*n_samples)

    def time_interpolate(self, n_samples, method):
        """Time the construction overhead."""
        # 计时插值函数构造的开销
        interpolate.interp1d(self.x, self.y, kind=method)

    def time_interpolate_eval(self, n_samples, method):
        """Time the evaluation."""
        # 计时插值函数在 xp 点上的评估时间
        self.interpolator(self.xp)


class Interpolate2d(Benchmark):
    # 参数名称列表，包括样本数和插值方法
    param_names = ['n_samples', 'method']
    # 参数取值，分别是样本数和插值方法的不同组合
    params = [
        [10, 50, 100],  # 不同的样本数
        ['linear', 'cubic', 'quintic'],  # 不同的插值方法
    ]

    def setup(self, n_samples, method):
        r_samples = n_samples / 2.
        # 创建二维网格 x 和 y
        self.x = np.arange(-r_samples, r_samples, 0.25)
        self.y = np.arange(-r_samples, r_samples, 0.25)
        # 创建网格坐标 xx 和 yy
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        # 根据二维坐标生成 z 值，用于二维插值
        self.z = np.sin(self.xx**2+self.yy**2)


class Rbf(Benchmark):
    # 参数名称列表，包括样本数和函数类型
    param_names = ['n_samples', 'function']
    # 参数取值，分别是样本数和函数类型的不同组合
    params = [
        [10, 50, 100],  # 不同的样本数
        ['multiquadric', 'inverse', 'gaussian', 'linear',
         'cubic', 'quintic', 'thin_plate']  # 不同的函数类型
    ]

    def setup(self, n_samples, function):
        # 创建一维样本点 x
        self.x = np.arange(n_samples)
        # 根据正弦函数生成对应的 y 值，用于径向基函数插值
        self.y = np.sin(self.x)
        r_samples = n_samples / 2.
        # 创建二维样本点 X 和 Y
        self.X = np.arange(-r_samples, r_samples, 0.25)
        self.Y = np.arange(-r_samples, r_samples, 0.25)
        # 根据二维坐标生成 z 值，用于径向基函数二维插值
        self.z = np.exp(-self.X**2-self.Y**2)

    def time_rbf_1d(self, n_samples, function):
        # 计时一维径向基函数插值的时间
        interpolate.Rbf(self.x, self.y, function=function)

    def time_rbf_2d(self, n_samples, function):
        # 计时二维径向基函数插值的时间
        interpolate.Rbf(self.X, self.Y, self.z, function=function)


class RBFInterpolator(Benchmark):
    # 参数名称列表，包括邻居数、样本数和核函数类型
    param_names = ['neighbors', 'n_samples', 'kernel']
    # 参数取值，分别是邻居数、样本数和核函数类型的不同组合
    params = [
        [None, 50],  # 不同的邻居数
        [10, 100, 1000],  # 不同的样本数
        ['linear', 'thin_plate_spline', 'cubic', 'quintic', 'multiquadric',
         'inverse_multiquadric', 'inverse_quadratic', 'gaussian']  # 不同的核函数类型
    ]

    def setup(self, neighbors, n_samples, kernel):
        rng = np.random.RandomState(0)
        # 创建二维坐标 y 和 x
        self.y = rng.uniform(-1, 1, (n_samples, 2))
        self.x = rng.uniform(-1, 1, (n_samples, 2))
        # 根据坐标计算 d 值，用于径向基函数插值
        self.d = np.sum(self.y, axis=1)*np.exp(-6*np.sum(self.y**2, axis=1))

    def time_rbf_interpolator(self, neighbors, n_samples, kernel):
        # 计时径向基函数插值器的执行时间
        interp = interpolate.RBFInterpolator(
            self.y,
            self.d,
            neighbors=neighbors,
            epsilon=5.0,
            kernel=kernel
            )
        interp(self.x)


class UnivariateSpline(Benchmark):
    # 参数名称列表，包括样本数和插值的阶数
    param_names = ['n_samples', 'degree']
    # 参数取值，分别是样本数和插值的阶数的不同组合
    params = [
        [10, 50, 100],  # 不同的样本数
        [3, 4, 5]  # 不同的插值的阶数
    ]
    # 定义一个方法 `setup`，用于设置数据集的大小和多项式的阶数
    def setup(self, n_samples, degree):
        # 计算采样点的范围
        r_samples = n_samples / 2.
        # 生成一维数组 `self.x`，范围从 `-r_samples` 到 `r_samples`，步长为 `0.25`
        self.x = np.arange(-r_samples, r_samples, 0.25)
        # 生成对应于 `self.x` 的 `y` 值，以高斯函数为基础加上小量随机扰动
        self.y = np.exp(-self.x**2) + 0.1 * np.random.randn(*self.x.shape)

    # 定义一个方法 `time_univariate_spline`，用于计算一元样条插值的时间
    def time_univariate_spline(self, n_samples, degree):
        # 使用 `UnivariateSpline` 类进行一元样条插值，传入 `self.x` 和 `self.y`，以及阶数 `degree`
        interpolate.UnivariateSpline(self.x, self.y, k=degree)
class BivariateSpline(Benchmark):
    """
    Author: josef-pktd and scipy mailinglist example
    'http://scipy-user.10969.n7.nabble.com/BivariateSpline-examples\
    -and-my-crashing-python-td14801.html'
    """
    # 定义参数名为 'n_samples'，可选参数为 [10, 20, 30]
    param_names = ['n_samples']
    # 参数组合为 n_samples = [10, 20, 30]
    params = [
        [10, 20, 30]
    ]

    def setup(self, n_samples):
        # 创建从 0 到 n_samples-1 步长为 0.5 的数组 x 和 y
        x = np.arange(0, n_samples, 0.5)
        y = np.arange(0, n_samples, 0.5)
        # 创建 x 和 y 的网格
        x, y = np.meshgrid(x, y)
        # 将网格展平为一维数组
        x = x.ravel()
        y = y.ravel()
        # 计算 x 和 y 的最小值和最大值，并扩展范围
        xmin = x.min()-1
        xmax = x.max()+1
        ymin = y.min()-1
        ymax = y.max()+1
        # 设置参数 s 为 1.1
        s = 1.1
        # 在 ymin+s 到 ymax-s 之间均匀分布生成 10 个点，作为 y 的节点
        self.yknots = np.linspace(ymin+s, ymax-s, 10)
        # 在 xmin+s 到 xmax-s 之间均匀分布生成 10 个点，作为 x 的节点
        self.xknots = np.linspace(xmin+s, xmax-s, 10)
        # 生成一个与 x 相同形状的数组，值为 sin(x) 加上小量噪声
        self.z = np.sin(x) + 0.1*np.random.normal(size=x.shape)
        # 保存 x 和 y 的值
        self.x = x
        self.y = y

    def time_smooth_bivariate_spline(self, n_samples):
        # 通过 x, y 和 z 创建一个 SmoothBivariateSpline 对象，用于插值
        interpolate.SmoothBivariateSpline(self.x, self.y, self.z)

    def time_lsq_bivariate_spline(self, n_samples):
        # 通过 x, y, z, xknots 和 yknots 创建一个 LSQBivariateSpline 对象，用于最小二乘插值
        interpolate.LSQBivariateSpline(self.x, self.y, self.z,
                                       self.xknots.flat, self.yknots.flat)


class Interpolate(Benchmark):
    """
    Linear Interpolate in scipy and numpy
    """
    # 定义参数名为 'n_samples' 和 'module'，可选参数分别为 [10, 50, 100] 和 ['numpy', 'scipy']
    param_names = ['n_samples', 'module']
    # 参数组合为 n_samples = [10, 50, 100]，module = ['numpy', 'scipy']
    params = [
        [10, 50, 100],
        ['numpy', 'scipy']
    ]

    def setup(self, n_samples, module):
        # 创建一个长度为 n_samples 的数组 x，值为指数函数衰减
        self.x = np.arange(n_samples)
        self.y = np.exp(-self.x/3.0)
        # 创建一个与 x 形状相同的数组，值为标准正态分布随机数
        self.z = np.random.normal(size=self.x.shape)

    def time_interpolate(self, n_samples, module):
        # 如果 module 为 'scipy'，使用 scipy 的 interp1d 进行线性插值
        if module == 'scipy':
            interpolate.interp1d(self.x, self.y, kind="linear")
        else:
            # 否则，使用 numpy 的 interp 进行线性插值
            np.interp(self.z, self.x, self.y)


class RegularGridInterpolator(Benchmark):
    """
    Benchmark RegularGridInterpolator with method="linear".
    """
    # 定义参数名为 'ndim', 'max_coord_size', 'n_samples', 'flipped'，可选参数分别为 [2, 3, 4], [10, 40, 200], [10, 100, 1000, 10000], [1, -1]
    param_names = ['ndim', 'max_coord_size', 'n_samples', 'flipped']
    # 参数组合为 ndim = [2, 3, 4], max_coord_size = [10, 40, 200], n_samples = [10, 100, 1000, 10000], flipped = [1, -1]
    params = [
        [2, 3, 4],
        [10, 40, 200],
        [10, 100, 1000, 10000],
        [1, -1]
    ]

    def setup(self, ndim, max_coord_size, n_samples, flipped):
        # 使用随机数生成器创建一个随机数发生器
        rng = np.random.default_rng(314159)

        # 计算每个维度上的坐标大小，每个维度的坐标大小为前一个维度的一半
        coord_sizes = [max_coord_size // 2**i for i in range(ndim)]
        # 生成每个维度上的坐标点，按照 flipped 的值确定方向
        self.points = [np.sort(rng.random(size=s))[::flipped]
                       for s in coord_sizes]
        # 生成与坐标大小相同形状的随机数数组
        self.values = rng.random(size=coord_sizes)

        # 计算每个维度上的边界，并在这些边界内生成 n_samples 个样本点 xi
        bounds = [(p.min(), p.max()) for p in self.points]
        xi = [rng.uniform(low, high, size=n_samples)
              for low, high in bounds]
        self.xi = np.array(xi).T

        # 创建 RegularGridInterpolator 对象，使用 self.points 和 self.values 进行插值
        self.interp = interpolate.RegularGridInterpolator(
            self.points,
            self.values,
        )

    def time_rgi_setup_interpolator(self, ndim, max_coord_size,
                                    n_samples, flipped):
        # 在 benchmark 中重新创建 RegularGridInterpolator 对象
        self.interp = interpolate.RegularGridInterpolator(
            self.points,
            self.values,
        )
    # 定义一个方法 `time_rgi`，接受四个参数：ndim（维度）、max_coord_size（最大坐标尺寸）、n_samples（样本数）、flipped（是否翻转）
    def time_rgi(self, ndim, max_coord_size, n_samples, flipped):
        # 在当前对象上调用 `interp` 方法，以 `self.xi` 作为参数
        self.interp(self.xi)
class RGI_Cubic(Benchmark):
    """
    Benchmark RegularGridInterpolator with method="cubic".
    """
    # 参数名称列表
    param_names = ['ndim', 'n_samples', 'method']
    # 参数组合列表
    params = [
        [2],  # ndim 取值为 2
        [10, 40, 100, 200, 400],  # n_samples 取值为这些
        ['cubic', 'cubic_legacy']  # method 取值为 'cubic' 或 'cubic_legacy'
    ]

    def setup(self, ndim, n_samples, method):
        # 使用种子 314159 创建随机数生成器
        rng = np.random.default_rng(314159)

        # 为每个维度创建排序后的随机采样点
        self.points = [np.sort(rng.random(size=n_samples))
                       for _ in range(ndim)]
        # 创建随机值数组，维度为 ndim
        self.values = rng.random(size=[n_samples]*ndim)

        # 确定采样点的边界，并从每个维度的边界中选择在边界内的采样点 xi
        bounds = [(p.min(), p.max()) for p in self.points]
        xi = [rng.uniform(low, high, size=n_samples)
              for low, high in bounds]
        self.xi = np.array(xi).T

        # 创建 RegularGridInterpolator 对象
        self.interp = interpolate.RegularGridInterpolator(
            self.points,
            self.values,
            method=method
        )

    def time_rgi_setup_interpolator(self, ndim, n_samples, method):
        # 重新创建 RegularGridInterpolator 对象以测试设置时间
        self.interp = interpolate.RegularGridInterpolator(
            self.points,
            self.values,
            method=method
        )

    def time_rgi(self, ndim, n_samples, method):
        # 在已设置的 RegularGridInterpolator 对象上进行插值计算
        self.interp(self.xi)


class RegularGridInterpolatorValues(interpolate.RegularGridInterpolator):
    def __init__(self, points, xi, **kwargs):
        # 创建用于初始化的虚拟值数组
        values = np.zeros(tuple([len(pt) for pt in points]))
        super().__init__(points, values, **kwargs)
        self._is_initialized = False
        # 预计算 xi 的相关值
        (self.xi, self.xi_shape, self.ndim,
         self.nans, self.out_of_bounds) = self._prepare_xi(xi)
        # 查找 xi 的索引和归一化距离
        self.indices, self.norm_distances = self._find_indices(xi.T)
        self._is_initialized = True

    def _prepare_xi(self, xi):
        if not self._is_initialized:
            return super()._prepare_xi(xi)
        else:
            # 返回预计算的值
            return (self.xi, self.xi_shape, self.ndim,
                    self.nans, self.out_of_bounds)

    def _find_indices(self, xi):
        if not self._is_initialized:
            return super()._find_indices(xi)
        else:
            # 返回预计算的索引和归一化距离
            return self.indices, self.norm_distances

    def __call__(self, values, method=None):
        # 检查输入的 values
        values = self._check_values(values)
        # 检查填充值 fillvalue
        self._check_fill_value(values, self.fill_value)
        # 检查维度匹配性
        self._check_dimensionality(self.grid, values)
        # 如果需要，翻转数据
        self.values = np.flip(values, axis=self._descending_dimensions)
        # 调用父类的 __call__ 方法进行插值计算
        return super().__call__(self.xi, method=method)


class RegularGridInterpolatorSubclass(Benchmark):
    """
    Benchmark RegularGridInterpolator with method="linear".
    """
    # 参数名称列表
    param_names = ['ndim', 'max_coord_size', 'n_samples', 'flipped']
    # 参数组合列表
    params = [
        [2, 3, 4],  # ndim 取值为 2, 3, 4
        [10, 40, 200],  # max_coord_size 取值为这些
        [10, 100, 1000, 10000],  # n_samples 取值为这些
        [1, -1]  # flipped 取值为 1 或 -1
    ]
    # 设置函数，用于初始化对象的属性
    def setup(self, ndim, max_coord_size, n_samples, flipped):
        # 创建一个基于固定种子的随机数生成器对象
        rng = np.random.default_rng(314159)

        # 计算每个维度上的坐标大小，按指数级递减
        coord_sizes = [max_coord_size // 2**i for i in range(ndim)]

        # 为每个维度生成排序后的随机坐标点，根据 flipped 参数决定正序还是逆序
        self.points = [np.sort(rng.random(size=s))[::flipped]
                       for s in coord_sizes]

        # 为每个坐标点生成随机值
        self.values = rng.random(size=coord_sizes)

        # 确定每个维度上的取值范围，并生成 n_samples 个样本点 xi
        bounds = [(p.min(), p.max()) for p in self.points]
        xi = [rng.uniform(low, high, size=n_samples)
              for low, high in bounds]
        self.xi = np.array(xi).T

        # 使用 RegularGridInterpolatorValues 类创建插值器对象 interp
        self.interp = RegularGridInterpolatorValues(
            self.points,
            self.xi,
        )

    # 测试函数，用于测试设置插值器对象的性能
    def time_rgi_setup_interpolator(self, ndim, max_coord_size,
                                    n_samples, flipped):
        # 使用相同的参数重新设置插值器对象 interp
        self.interp = RegularGridInterpolatorValues(
            self.points,
            self.xi,
        )

    # 测试函数，用于测试插值器对象 interp 的性能
    def time_rgi(self, ndim, max_coord_size, n_samples, flipped):
        # 调用插值器对象 interp 来进行插值计算，使用预先设定的随机值 self.values
        self.interp(self.values)
class CloughTocherInterpolatorValues(interpolate.CloughTocher2DInterpolator):
    """Subclass of the CT2DInterpolator with optional `values`.

    This is mainly a demo of the functionality. See
    https://github.com/scipy/scipy/pull/18376 for discussion
    """

    def __init__(self, points, xi, tol=1e-6, maxiter=400, **kwargs):
        # 调用父类的构造函数来初始化 CloughTocher2DInterpolator
        interpolate.CloughTocher2DInterpolator.__init__(self, points, None,
                                                        tol=tol, maxiter=maxiter)
        self.xi = None
        # 对 xi 进行预处理
        self._preprocess_xi(*xi)

    def _preprocess_xi(self, *args):
        # 如果 xi 尚未初始化，则调用父类的预处理函数，并返回处理后的 xi 和插值点的形状
        if self.xi is None:
            self.xi, self.interpolation_points_shape = (
                interpolate.CloughTocher2DInterpolator._preprocess_xi(self, *args)
            )
        return self.xi, self.interpolation_points_shape
    
    def __call__(self, values):
        # 设置对象的值
        self._set_values(values)
        # 调用父类的 __call__ 方法进行插值计算，并返回结果
        return super().__call__(self.xi)


class CloughTocherInterpolatorSubclass(Benchmark):
    """
    Benchmark CloughTocherInterpolatorValues.

    Derived from the docstring example,
    https://docs.scipy.org/doc/scipy-1.11.2/reference/generated/scipy.interpolate.CloughTocher2DInterpolator.html
    """

    param_names = ['n_samples']
    params = [10, 50, 100]

    def setup(self, n_samples):
        # 使用种子 314159 初始化随机数生成器
        rng = np.random.default_rng(314159)

        # 生成 n_samples 个在 [-0.5, 0.5] 范围内的随机数作为 x 和 y
        x = rng.random(n_samples) - 0.5
        y = rng.random(n_samples) - 0.5

        # 计算 z 值作为 hypot(x, y) 的结果
        self.z = np.hypot(x, y)

        # 生成 X 和 Y，分别作为 x 和 y 的网格化坐标
        X = np.linspace(min(x), max(x))
        Y = np.linspace(min(y), max(y))
        self.X, self.Y = np.meshgrid(X, Y)

        # 使用 CloughTocherInterpolatorValues 初始化 self.interp 对象
        self.interp = CloughTocherInterpolatorValues(
            list(zip(x, y)), (self.X, self.Y)
        )

    def time_clough_tocher(self, n_samples):
        # 调用 self.interp 对象进行插值计算，使用 self.z 作为值
        self.interp(self.z)
```