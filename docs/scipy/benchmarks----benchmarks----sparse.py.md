# `D:\src\scipysrc\scipy\benchmarks\benchmarks\sparse.py`

```
"""
Simple benchmarks for the sparse module
"""
# 导入警告模块
import warnings
# 导入时间模块
import time
import timeit
# 导入pickle模块
import pickle

# 导入numpy库及其部分子模块和函数
import numpy as np
from numpy import ones, array, asarray, empty

# 导入Benchmark类和safe_import函数
from .common import Benchmark, safe_import

# 使用safe_import导入scipy.sparse模块及其部分子模块和SparseEfficiencyWarning警告
with safe_import():
    from scipy import sparse
    from scipy.sparse import (coo_matrix, dia_matrix, lil_matrix,
                              dok_matrix, rand, SparseEfficiencyWarning)


# 定义一个生成随机稀疏矩阵的函数
def random_sparse(m, n, nnz_per_row):
    # 创建行索引数组
    rows = np.arange(m).repeat(nnz_per_row)
    # 创建列索引数组
    cols = np.random.randint(0, n, size=nnz_per_row*m)
    # 创建值数组
    vals = np.random.random_sample(m*nnz_per_row)
    # 使用COO格式创建稀疏矩阵，并转换为CSR格式
    return coo_matrix((vals, (rows, cols)), (m, n)).tocsr()


# 定义一个生成二维泊松问题的稀疏矩阵的函数
def poisson2d(N, dtype='d', format=None):
    """
    Return a sparse matrix for the 2D Poisson problem
    with standard 5-point finite difference stencil on a
    square N-by-N grid.
    """
    # 处理N为1的情况
    if N == 1:
        # 创建包含中心对角线元素的对角矩阵
        diags = asarray([[4]], dtype=dtype)
        return dia_matrix((diags, [0]), shape=(1, 1)).asformat(format)

    # 定义偏移量数组
    offsets = array([0, -N, N, -1, 1])

    # 创建空的对角线数组
    diags = empty((5, N**2), dtype=dtype)

    # 设置主对角线元素为4
    diags[0] = 4  # main diagonal
    # 设置所有非主对角线元素为-1
    diags[1:] = -1  # all offdiagonals

    # 设置第一个下对角线的特定元素为0
    diags[3, N-1::N] = 0  # first lower diagonal
    # 设置第一个上对角线的特定元素为0
    diags[4, N::N] = 0  # first upper diagonal

    # 使用对角矩阵格式创建稀疏矩阵并按指定格式返回
    return dia_matrix((diags, offsets), shape=(N**2, N**2)).asformat(format)


# 定义一个继承自Benchmark类的Arithmetic类
class Arithmetic(Benchmark):
    # 参数名列表
    param_names = ['format', 'XY', 'op']
    # 参数组合列表
    params = [
        ['csr', 'csc', 'coo', 'dia'],
        ['AA', 'AB', 'BA', 'BB'],
        ['__add__', '__sub__', 'multiply', '__mul__']
    ]

    # 初始化设置方法
    def setup(self, format, XY, op):
        # 创建稀疏矩阵字典
        matrices = dict(A=poisson2d(250, format=format),
                        B=poisson2d(250, format=format)**2)

        # 获取XY对应的矩阵
        x = matrices[XY[0]]
        self.y = matrices[XY[1]]
        # 获取操作函数
        self.fn = getattr(x, op)
        self.fn(self.y)  # warmup

    # 定义性能测试方法
    def time_arithmetic(self, format, XY, op):
        self.fn(self.y)


# 定义一个继承自Benchmark类的Sort类
class Sort(Benchmark):
    # 参数值列表
    params = ['Rand10', 'Rand25', 'Rand50', 'Rand100', 'Rand200']
    # 参数名列表
    param_names = ['matrix']

    # 初始化设置方法
    def setup(self, matrix):
        n = 10000
        # 如果matrix以'Rand'开头
        if matrix.startswith('Rand'):
            k = int(matrix[4:])
            # 创建随机稀疏矩阵，并设置has_sorted_indices为False
            self.A = random_sparse(n, n, k)
            self.A.has_sorted_indices = False
            # 修改前两个元素的列索引
            self.A.indices[:2] = 2, 1
        else:
            raise NotImplementedError()

    # 定义性能测试方法
    def time_sort(self, matrix):
        """sort CSR column indices"""
        self.A.sort_indices()


# 定义一个继承自Benchmark类的Matvec类
class Matvec(Benchmark):
    # 参数值列表
    params = [
        ['Identity', 'Poisson5pt', 'Block2x2', 'Block3x3'],
        ['dia', 'csr', 'csc', 'dok', 'lil', 'coo', 'bsr']
    ]
    # 参数名列表
    param_names = ['matrix', 'format']
    # 设置函数，用于初始化稀疏矩阵和格式
    def setup(self, matrix, format):
        # 如果矩阵为单位矩阵
        if matrix == 'Identity':
            # 如果格式是 'lil' 或者 'dok'，则抛出未实现错误
            if format in ('lil', 'dok'):
                raise NotImplementedError()
            # 创建一个 10000x10000 的单位矩阵，并按指定格式存储到 self.A 中
            self.A = sparse.eye(10000, 10000, format=format)
        # 如果矩阵是 5 点有限差分泊松方程矩阵
        elif matrix == 'Poisson5pt':
            # 使用给定格式创建一个 300x300 的二维泊松方程矩阵，并存储到 self.A 中
            self.A = poisson2d(300, format=format)
        # 如果矩阵是 2x2 块矩阵
        elif matrix == 'Block2x2':
            # 如果格式不是 'csr' 或 'bsr'，则抛出未实现错误
            if format not in ('csr', 'bsr'):
                raise NotImplementedError()
            # 定义块的大小为 (2, 2)
            b = (2, 2)
            # 创建一个 150x150 的二维泊松方程矩阵，然后使用 sparse.kron 方法和 ones 函数生成块矩阵
            # 将其转换为 BSR 格式，并存储到 self.A 中
            self.A = sparse.kron(poisson2d(150),
                                 ones(b)).tobsr(blocksize=b).asformat(format)
        # 如果矩阵是 3x3 块矩阵
        elif matrix == 'Block3x3':
            # 如果格式不是 'csr' 或 'bsr'，则抛出未实现错误
            if format not in ('csr', 'bsr'):
                raise NotImplementedError()
            # 定义块的大小为 (3, 3)
            b = (3, 3)
            # 创建一个 100x100 的二维泊松方程矩阵，然后使用 sparse.kron 方法和 ones 函数生成块矩阵
            # 将其转换为 BSR 格式，并存储到 self.A 中
            self.A = sparse.kron(poisson2d(100),
                                 ones(b)).tobsr(blocksize=b).asformat(format)
        else:
            # 如果矩阵名称不在已知列表中，则抛出未实现错误
            raise NotImplementedError()

        # 初始化 self.x，其长度为 self.A 的列数，数据类型为 float
        self.x = ones(self.A.shape[1], dtype=float)

    # 计算矩阵向量乘法的函数
    def time_matvec(self, matrix, format):
        # 计算 self.A 与 self.x 的乘积
        self.A * self.x
# 创建一个继承自Benchmark的Matvecs类，用于矩阵向量乘法的性能基准测试
class Matvecs(Benchmark):
    # 定义参数列表，包括不同的矩阵格式
    params = ['dia', 'coo', 'csr', 'csc', 'bsr']
    # 定义参数名，即格式的名称
    param_names = ["format"]

    # 设置方法，在每次测试前初始化数据
    def setup(self, format):
        # 创建一个二维泊松方程的稀疏矩阵A，根据指定格式
        self.A = poisson2d(300, format=format)
        # 创建一个与A的列数相同，行数为10的全1数组，数据类型与A相同
        self.x = ones((self.A.shape[1], 10), dtype=self.A.dtype)

    # 定义矩阵向量乘法的性能测试方法
    def time_matvecs(self, format):
        # 执行矩阵向量乘法操作
        self.A * self.x


# 创建一个继承自Benchmark的Matmul类，用于大规模矩阵乘法的性能基准测试
class Matmul(Benchmark):
    # 设置方法，在每次测试前初始化数据
    def setup(self):
        # 定义第一个矩阵的高度和宽度
        H1, W1 = 1, 100000
        # 定义第二个矩阵的高度和宽度，其中第二个矩阵的宽度与第一个矩阵的宽度相同
        H2, W2 = W1, 1000
        # 定义第一个矩阵的非零元素数量
        C1 = 10
        # 定义第二个矩阵的非零元素数量
        C2 = 1000000

        # 使用默认随机数生成器创建rng对象
        rng = np.random.default_rng(0)

        # 生成第一个稀疏矩阵的行索引、列索引和数据
        i = rng.integers(H1, size=C1)
        j = rng.integers(W1, size=C1)
        data = rng.random(C1)
        # 将数据按照坐标格式创建稀疏矩阵，并将其转换为压缩行格式（csr）
        self.matrix1 = coo_matrix((data, (i, j)), shape=(H1, W1)).tocsr()

        # 生成第二个稀疏矩阵的行索引、列索引和数据
        i = rng.integers(H2, size=C2)
        j = rng.integers(W2, size=C2)
        data = rng.random(C2)
        # 将数据按照坐标格式创建稀疏矩阵，并将其转换为压缩行格式（csr）
        self.matrix2 = coo_matrix((data, (i, j)), shape=(H2, W2)).tocsr()

    # 定义大规模矩阵乘法的性能测试方法
    def time_large(self):
        # 进行100次矩阵乘法操作
        for i in range(100):
            self.matrix1 * self.matrix2

    # 保留旧的基准测试结果版本信息（如果更改基准测试，请删除此注释）
    time_large.version = (
        "33aee08539377a7cb0fabaf0d9ff9d6d80079a428873f451b378c39f6ead48cb"
    )


# 创建一个继承自Benchmark的Construction类，用于矩阵构建操作的性能基准测试
class Construction(Benchmark):
    # 定义参数列表，包括不同的矩阵类型和格式
    params = [
        ['Empty', 'Identity', 'Poisson5pt'],
        ['lil', 'dok']
    ]
    # 定义参数名，分别表示矩阵名称和格式名称
    param_names = ['matrix', 'format']

    # 设置方法，在每次测试前初始化数据
    def setup(self, name, format):
        # 根据不同的矩阵名称选择不同的初始化方式
        if name == 'Empty':
            # 创建一个10000x10000的空稀疏矩阵
            self.A = coo_matrix((10000, 10000))
        elif name == 'Identity':
            # 创建一个10000x10000的单位稀疏矩阵，使用坐标格式（coo）
            self.A = sparse.eye(10000, format='coo')
        else:
            # 创建一个二维泊松方程的100x100稀疏矩阵，使用坐标格式（coo）
            self.A = poisson2d(100, format='coo')

        # 定义格式字典，将格式名称映射到对应的矩阵类型构造器
        formats = {'lil': lil_matrix, 'dok': dok_matrix}
        # 根据给定格式名称选择对应的构造器
        self.cls = formats[format]

    # 定义矩阵构建操作的性能测试方法
    def time_construction(self, name, format):
        # 根据选定的构造器类型创建一个同形状的稀疏矩阵T
        T = self.cls(self.A.shape)
        # 遍历原始稀疏矩阵A的非零元素，复制到新矩阵T中
        for i, j, v in zip(self.A.row, self.A.col, self.A.data):
            T[i, j] = v


# 创建一个继承自Benchmark的BlockDiagDenseConstruction类，用于稠密块对角矩阵构建操作的性能基准测试
class BlockDiagDenseConstruction(Benchmark):
    # 定义参数名，表示矩阵数量
    param_names = ['num_matrices']
    # 定义参数列表，包括不同的矩阵数量
    params = [1000, 5000, 10000, 15000, 20000]

    # 设置方法，在每次测试前初始化数据
    def setup(self, num_matrices):
        # 初始化一个空列表，用于存储随机生成的稠密矩阵
        self.matrices = []
        # 循环生成指定数量的随机矩阵
        for i in range(num_matrices):
            # 随机生成当前矩阵的行数和列数（均为1到3之间）
            rows = np.random.randint(1, 4)
            columns = np.random.randint(1, 4)
            # 使用0到9之间的随机整数填充当前矩阵
            mat = np.random.randint(0, 10, (rows, columns))
            # 将当前矩阵添加到列表中
            self.matrices.append(mat)

    # 定义稠密块对角矩阵构建操作的性能测试方法
    def time_block_diag(self, num_matrices):
        # 调用sparse库中的block_diag函数构建稠密块对角矩阵
        sparse.block_diag(self.matrices)


# 创建一个继承自Benchmark的BlockDiagSparseConstruction类，用于稀疏块对角矩阵构建操作的性能基准测试
class BlockDiagSparseConstruction(Benchmark):
    # 定义参数名，表示矩阵数量
    param_names = ['num_matrices']
    # 定义参数列表，包括不同的矩阵数量
    params = [100, 500, 1000, 1500, 2000]

    # 设置方法，在每次测试前初始化数据
    def setup(self, num_matrices):
        # 初始化一个空列表，用于存储随机生成的稀疏矩阵
        self.matrices = []
        # 循环生成指定数量的随机矩阵
        for i in range(num_matrices):
            # 随机生成当前矩阵的行数和列数（均为1到19之间）
            rows =
    # 设置函数，用于初始化稀疏矩阵
    def setup(self, num_rows):
        # 定义矩阵的列数为10万
        num_cols = int(1e5)
        # 定义稀疏性，即非零元素占比
        density = 2e-3
        # 计算每行的非零元素个数
        nnz_per_row = int(density * num_cols)
        # 使用随机稀疏矩阵生成函数创建稀疏矩阵
        self.mat = random_sparse(num_rows, num_cols, nnz_per_row)

    # 计算函数，用于测试稀疏矩阵的水平堆叠操作的执行时间
    def time_csr_hstack(self, num_rows):
        # 对稀疏矩阵进行水平堆叠操作，即将矩阵和自身水平拼接
        sparse.hstack([self.mat, self.mat])
class Conversion(Benchmark):
    params = [
        ['csr', 'csc', 'coo', 'dia', 'lil', 'dok', 'bsr'],  # 参数化测试的来源格式列表
        ['csr', 'csc', 'coo', 'dia', 'lil', 'dok', 'bsr'],  # 参数化测试的目标格式列表
    ]
    param_names = ['from_format', 'to_format']  # 参数名称，用于在测试报告中标识来源和目标格式

    def setup(self, fromfmt, tofmt):
        base = poisson2d(100, format=fromfmt)  # 创建一个二维泊松分布矩阵，指定格式为来源格式

        try:
            self.fn = getattr(base, 'to' + tofmt)  # 尝试获取对应目标格式的转换方法
        except Exception:
            def fn():  # 如果获取失败，定义一个函数抛出运行时错误
                raise RuntimeError()
            self.fn = fn  # 将定义的函数赋给self.fn

    def time_conversion(self, fromfmt, tofmt):
        self.fn()  # 执行转换函数


class Getset(Benchmark):
    params = [
        [1, 10, 100, 1000, 10000],  # 参数化测试的N值列表，表示要设置或获取的元素数量
        ['different', 'same'],  # 参数化测试的稀疏模式，用于测试不同或相同的索引设置方式
        ['csr', 'csc', 'lil', 'dok']  # 参数化测试的格式列表，表示要测试的稀疏矩阵存储格式
    ]
    param_names = ['N', 'sparsity pattern', 'format']  # 参数名称，用于在测试报告中标识N值、稀疏模式和格式

    unit = "seconds"  # 测试时间单位为秒

    def setup(self, N, sparsity_pattern, format):
        if format == 'dok' and N > 500:
            raise NotImplementedError()  # 如果选择格式为'dok'且N大于500，则抛出未实现错误

        self.A = rand(1000, 1000, density=1e-5)  # 创建一个稀疏随机矩阵A，密度为1e-5

        A = self.A
        N = int(N)

        # indices to assign to
        i, j = [], []
        while len(i) < N:
            n = N - len(i)
            ip = np.random.randint(0, A.shape[0], size=n)  # 随机生成n个行索引
            jp = np.random.randint(0, A.shape[1], size=n)  # 随机生成n个列索引
            i = np.r_[i, ip]  # 将生成的行索引追加到i列表中
            j = np.r_[j, jp]  # 将生成的列索引追加到j列表中
        v = np.random.rand(n)  # 生成n个随机数作为值

        if N == 1:
            i = int(i)  # 如果N为1，将i转换为整数
            j = int(j)  # 将j转换为整数
            v = float(v)  # 将v转换为浮点数

        base = A.asformat(format)  # 将矩阵A转换为指定格式的稀疏矩阵base

        self.m = base.copy()  # 复制base矩阵并赋给self.m
        self.i = i  # 将行索引i赋给self.i
        self.j = j  # 将列索引j赋给self.j
        self.v = v  # 将值v赋给self.v

    def _timeit(self, kernel, recopy):
        min_time = 1e99

        if not recopy:
            kernel(self.m, self.i, self.j, self.v)  # 如果不需要复制，直接调用kernel函数

        number = 1
        start = time.time()

        while time.time() - start < 0.1:
            if recopy:
                m = self.m.copy()  # 如果需要复制，复制self.m并赋给m
            else:
                m = self.m

            while True:
                duration = timeit.timeit(
                    lambda: kernel(m, self.i, self.j, self.v), number=number)  # 计算执行kernel函数的时间
                if duration > 1e-5:
                    break
                else:
                    number *= 10

            min_time = min(min_time, duration / number)  # 计算最小时间

        return min_time  # 返回最小时间

    def track_fancy_setitem(self, N, sparsity_pattern, format):
        def kernel(A, i, j, v):
            A[i, j] = v  # 在稀疏矩阵A中设置索引i、j处的值为v

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SparseEfficiencyWarning)  # 忽略稀疏矩阵效率警告
            return self._timeit(kernel, sparsity_pattern == 'different')  # 执行时间测量，根据稀疏模式选择是否复制矩阵

    def time_fancy_getitem(self, N, sparsity_pattern, format):
        self.m[self.i, self.j]  # 获取稀疏矩阵self.m中索引为self.i、self.j的值


class NullSlice(Benchmark):
    params = [[0.05, 0.01], ['csr', 'csc', 'lil']]  # 参数化测试的密度和格式列表
    param_names = ['density', 'format']  # 参数名称，用于在测试报告中标识密度和格式
    def _setup(self, density, format):
        # 设置随机生成矩阵的大小
        n = 100000
        k = 1000

        # 使用给定的密度和格式生成一个稀疏矩阵的非精确版本
        nz = int(n*k * density)  # 计算非零元素的数量
        row = np.random.randint(0, n, size=nz)  # 在0到n之间生成随机行索引
        col = np.random.randint(0, k, size=nz)  # 在0到k之间生成随机列索引
        data = np.ones(nz, dtype=np.float64)  # 创建一个包含1的数据数组
        X = coo_matrix((data, (row, col)), shape=(n, k))  # 使用COO格式创建稀疏矩阵
        X.sum_duplicates()  # 合并相同元素
        X = X.asformat(format)  # 将矩阵格式化为指定的格式
        with open(f'{density}-{format}.pck', 'wb') as f:
            pickle.dump(X, f, protocol=pickle.HIGHEST_PROTOCOL)  # 将矩阵X保存到文件中

    def setup_cache(self):
        # 遍历给定的密度和格式参数列表，调用_setup方法进行矩阵生成和保存
        for density in self.params[0]:
            for fmt in self.params[1]:
                self._setup(density, fmt)

    setup_cache.timeout = 120  # 设置setup_cache方法的超时时间为120秒

    def setup(self, density, format):
        # 从文件中加载预生成的矩阵数据
        with open(f'{density}-{format}.pck', 'rb') as f:
            self.X = pickle.load(f)

    def time_getrow(self, density, format):
        # 评估获取矩阵指定行的性能
        self.X.getrow(100)

    def time_getcol(self, density, format):
        # 评估获取矩阵指定列的性能
        self.X.getcol(100)

    def time_3_rows(self, density, format):
        # 评估同时获取矩阵的三行的性能
        self.X[[0, 100, 105], :]

    def time_10000_rows(self, density, format):
        # 评估同时获取矩阵的一万行的性能
        self.X[np.arange(10000), :]

    def time_3_cols(self, density, format):
        # 评估同时获取矩阵的三列的性能
        self.X[:, [0, 100, 105]]

    def time_100_cols(self, density, format):
        # 评估同时获取矩阵的一百列的性能
        self.X[:, np.arange(100)]

    # 保留旧的基准测试结果（如果更改基准测试，请删除此行）
    time_10000_rows.version = (
        "dc19210b894d5fd41d4563f85b7459ef5836cddaf77154b539df3ea91c5d5c1c"
    )
    time_100_cols.version = (
        "8d43ed52084cdab150018eedb289a749a39f35d4dfa31f53280f1ef286a23046"
    )
    time_3_cols.version = (
        "93e5123910772d62b3f72abff56c2732f83d217221bce409b70e77b89c311d26"
    )
    time_3_rows.version = (
        "a9eac80863a0b2f4b510269955041930e5fdd15607238257eb78244f891ebfe6"
    )
    time_getcol.version = (
        "291388763b355f0f3935db9272a29965d14fa3f305d3306059381e15300e638b"
    )
    time_getrow.version = (
        "edb9e4291560d6ba8dd58ef371b3a343a333bc10744496adb3ff964762d33c68"
    )
class Diagonal(Benchmark):
    # 参数列表，包含两个参数：密度和格式
    params = [[0.01, 0.1, 0.5], ['csr', 'csc', 'coo', 'lil', 'dok', 'dia']]
    # 参数名列表，分别对应密度和格式
    param_names = ['density', 'format']

    def setup(self, density, format):
        # 设置矩阵维度
        n = 1000
        # 如果格式为'dok'并且计算出的密度大于等于500，则抛出未实现错误
        if format == 'dok' and n * density >= 500:
            raise NotImplementedError()

        # 忽略稀疏矩阵效率警告
        warnings.simplefilter('ignore', SparseEfficiencyWarning)

        # 生成稀疏随机矩阵X，根据给定的格式和密度
        self.X = sparse.rand(n, n, format=format, density=density)

    def time_diagonal(self, density, format):
        # 计算矩阵X的对角线元素
        self.X.diagonal()

    # 保留旧的基准测试结果版本（如果更改基准测试，则删除此项）
    time_diagonal.version = (
        "d84f53fdc6abc208136c8ce48ca156370f6803562f6908eb6bd1424f50310cf1"
    )


class Sum(Benchmark):
    # 参数列表，包含两个参数：密度和格式
    params = [[0.01, 0.1, 0.5], ['csr', 'csc', 'coo', 'lil', 'dok', 'dia']]
    # 参数名列表，分别对应密度和格式
    param_names = ['density', 'format']

    def setup(self, density, format):
        # 设置矩阵维度
        n = 1000
        # 如果格式为'dok'并且计算出的密度大于等于500，则抛出未实现错误
        if format == 'dok' and n * density >= 500:
            raise NotImplementedError()

        # 忽略稀疏矩阵效率警告
        warnings.simplefilter('ignore', SparseEfficiencyWarning)
        # 生成稀疏随机矩阵X，根据给定的格式和密度
        self.X = sparse.rand(n, n, format=format, density=density)

    def time_sum(self, density, format):
        # 计算矩阵X所有元素的和
        self.X.sum()

    def time_sum_axis0(self, density, format):
        # 计算矩阵X每列元素的和
        self.X.sum(axis=0)

    def time_sum_axis1(self, density, format):
        # 计算矩阵X每行元素的和
        self.X.sum(axis=1)

    # 保留旧的基准测试结果版本（如果更改基准测试，则删除此项）
    time_sum.version = (
        "05c305857e771024535e546360203b17f5aca2b39b023a49ab296bd746d6cdd3"
    )
    time_sum_axis0.version = (
        "8aca682fd69aa140c69c028679826bdf43c717589b1961b4702d744ed72effc6"
    )
    time_sum_axis1.version = (
        "1a6e05244b77f857c61f8ee09ca3abd006a10ba07eff10b1c5f9e0ac20f331b2"
    )


class Iteration(Benchmark):
    # 参数列表，包含两个参数：密度和格式
    params = [[0.05, 0.01], ['csr', 'csc', 'lil']]
    # 参数名列表，分别对应密度和格式
    param_names = ['density', 'format']

    def setup(self, density, format):
        # 设置矩阵行数和列数
        n = 500
        k = 1000
        # 生成稀疏随机矩阵X，根据给定的格式和密度
        self.X = sparse.rand(n, k, format=format, density=density)

    def time_iteration(self, density, format):
        # 迭代矩阵X的每一行
        for row in self.X:
            pass


class Densify(Benchmark):
    # 参数列表，包含两个参数：格式和顺序
    params = [
        ['dia', 'csr', 'csc', 'dok', 'lil', 'coo', 'bsr'],
        ['C', 'F'],
    ]
    # 参数名列表，分别对应格式和顺序
    param_names = ['format', 'order']

    def setup(self, format, order):
        # 忽略稀疏矩阵效率警告
        warnings.simplefilter('ignore', SparseEfficiencyWarning)
        # 生成稀疏随机矩阵X，根据给定的格式和密度（密度为0.01）
        self.X = sparse.rand(1000, 1000, format=format, density=0.01)

    def time_toarray(self, format, order):
        # 将稀疏矩阵X转换为密集数组，根据给定的顺序
        self.X.toarray(order=order)

    # 保留旧的基准测试结果版本（如果更改基准测试，则删除此项）
    time_toarray.version = (
        "2fbf492ec800b982946a62785beda803460b913cc80080043a5d407025893b2b"
    )


class Random(Benchmark):
    # 参数列表，包含一个参数：密度
    params = [
        np.arange(0, 1.1, 0.1).tolist()
    ]
    # 参数名列表，只有密度一个参数
    param_names = ['density']

    def setup(self, density):
        # 忽略稀疏矩阵效率警告
        warnings.simplefilter('ignore', SparseEfficiencyWarning)
        # 设置矩阵行数和列数
        self.nrows = 1000
        self.ncols = 1000
        self.format = 'csr'
    # 定义一个方法 `time_rand`，接受 `density` 参数，用于生成稀疏矩阵
    def time_rand(self, density):
        # 使用 sparse 模块中的 rand 函数生成一个稀疏矩阵
        sparse.rand(self.nrows, self.ncols,
                    format=self.format, density=density)
```