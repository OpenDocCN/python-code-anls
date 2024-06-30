# `D:\src\scipysrc\scipy\benchmarks\benchmarks\sparse_linalg_lobpcg.py`

```
import numpy as np
import warnings
from .common import Benchmark, safe_import

# 安全导入相关模块和函数，避免导入错误导致程序异常
with safe_import():
    # 从 scipy.linalg 中导入特定函数和类
    from scipy.linalg import (eigh,
                              cholesky_banded, cho_solve_banded, eig_banded)
    # 从 scipy.sparse.linalg 中导入特定函数和类
    from scipy.sparse.linalg import lobpcg, eigsh, LinearOperator
    # 从 scipy.sparse.linalg._special_sparse_arrays 中导入特定类
    from scipy.sparse.linalg._special_sparse_arrays import (Sakurai,
                                                            MikotaPair)

# 确保我们在进行基准测试时得到一致的结果；
# 如果代码无法正常收敛，基准测试的时间结果将无效。
msg = ("the benchmark code did not converge as expected, "
       "the timing is therefore useless")

# 定义 Benchmark 类的子类 Bench
class Bench(Benchmark):
    # 参数空列表和包含特定求解器名称的列表
    params = [
        [],
        ['lobpcg', 'eigsh', 'lapack']
    ]
    # 参数名称分别为 'n' 和 'solver'
    param_names = ['n', 'solver']

    # Bench 类的初始化方法
    def __init__(self):
        # 设置 mikota 方法的参数和初始化方法
        self.time_mikota.__func__.params = list(self.params)
        self.time_mikota.__func__.params[0] = [128, 256, 512, 1024, 2048]
        self.time_mikota.__func__.setup = self.setup_mikota

        # 设置 sakurai 方法的参数和初始化方法
        self.time_sakurai.__func__.params = list(self.params)
        self.time_sakurai.__func__.params[0] = [50]
        self.time_sakurai.__func__.setup = self.setup_sakurai

        # 设置 sakurai_inverse 方法的参数和初始化方法
        self.time_sakurai_inverse.__func__.params = list(self.params)
        self.time_sakurai_inverse.__func__.params[0] = [500, 1000]
        self.time_sakurai_inverse.__func__.setup = self.setup_sakurai_inverse

    # 设置 mikota 方法的初始化方法
    def setup_mikota(self, n, solver):
        # 设置矩阵的形状
        self.shape = (n, n)
        # 创建 MikotaPair 对象
        mik = MikotaPair(n)
        # 获取对象的 k 和 m
        mik_k = mik.k
        mik_m = mik.m
        # 分别设置 Ac, Aa, Bc, Ba 和 Ab 属性
        self.Ac = mik_k
        self.Aa = mik_k.toarray()
        self.Bc = mik_m
        self.Ba = mik_m.toarray()
        self.Ab = mik_k.tobanded()
        # 获取对象的特征值
        self.eigenvalues = mik.eigenvalues

        # 如果求解器为 'lapack' 且 n 大于 512，则抛出未实现错误
        if solver == 'lapack' and n > 512:
            raise NotImplementedError()

    # 设置 sakurai 方法的初始化方法
    def setup_sakurai(self, n, solver):
        # 设置矩阵的形状
        self.shape = (n, n)
        # 创建 Sakurai 对象
        sakurai_obj = Sakurai(n, dtype='int')
        # 设置 A 和 Aa 属性
        self.A = sakurai_obj
        self.Aa = sakurai_obj.toarray()
        # 获取对象的特征值
        self.eigenvalues = sakurai_obj.eigenvalues

    # 设置 sakurai_inverse 方法的初始化方法
    def setup_sakurai_inverse(self, n, solver):
        # 设置矩阵的形状
        self.shape = (n, n)
        # 创建 Sakurai 对象
        sakurai_obj = Sakurai(n)
        # 将对象转换为 banded 形式，并转换为 np.float64 类型
        self.A = sakurai_obj.tobanded().astype(np.float64)
        # 获取对象的特征值
        self.eigenvalues = sakurai_obj.eigenvalues
    def time_mikota(self, n, solver):
        # 定义内部函数 `a`，用于解决带状线性系统 `Ax = lambda Bx`
        def a(x):
            return cho_solve_banded((c, False), x)

        # 设置特征值个数 `m`
        m = 10
        # 计算 `m` 个特征值
        ee = self.eigenvalues(m)
        # 计算精度容差 `tol`
        tol = m * n * n * n * np.finfo(float).eps
        # 初始化随机数生成器
        rng = np.random.default_rng(0)
        # 生成 `n x m` 的正态分布随机矩阵 `X`
        X = rng.normal(size=(n, m))

        if solver == 'lobpcg':
            # 若使用 `lobpcg` 求解器
            # `lobpcg` 允许直接传入可调用参数 `Ac` 和 `Bc`
            # `lobpcg` 解决 ``Ax = lambda Bx``，并在此应用一个预处理器
            # 由 `Ab` 的 `np.float32` 矩阵逆组成，它本身是 `np.float64`
            c = cholesky_banded(self.Ab.astype(np.float32))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 调用 `lobpcg` 进行求解
                el, _ = lobpcg(self.Ac, X, self.Bc, M=a, tol=1e-4,
                               maxiter=40, largest=False)
            # 计算精度
            accuracy = max(abs(ee - el) / ee)
            # 断言精度符合要求
            assert accuracy < tol, msg
        elif solver == 'eigsh':
            # 若使用 `eigsh` 求解器
            # `eigsh` ARPACK 在 ``Bx = 1/lambda Ax`` 上调用
            # 以获得类似 `lobpcg` 的快速收敛速度
            # 需要以完整的 `np.float64` 精度提供由 `Ab` 的 Cholesky 给出的矩阵 ``A`` 的逆
            # `eigsh` ARPACK 不允许直接传入可调用参数 `Bc`，而需要以 `LinearOperator` 格式输入
            B = LinearOperator((n, n), matvec=self.Bc, matmat=self.Bc, dtype='float64')
            A = LinearOperator((n, n), matvec=self.Ac, matmat=self.Ac, dtype='float64')
            c = cholesky_banded(self.Ab)
            a_l = LinearOperator((n, n), matvec=a, matmat=a, dtype='float64')
            # 调用 `eigsh` 进行求解
            ea, _ = eigsh(B, k=m, M=A, Minv=a_l, which='LA', tol=1e-4, maxiter=50,
                          v0=rng.normal(size=(n, 1)))
            # 计算精度
            accuracy = max(abs(ee - np.sort(1. / ea)) / ee)
            # 断言精度符合要求
            assert accuracy < tol, msg
        else:
            # 若使用 `eigh` 求解器
            # `eigh` 是广义特征值问题的唯一密集求解器
            # ``Ax = lambda Bx`` 需要两个矩阵作为密集数组输入
            # 对于大矩阵尺寸非常缓慢
            ed, _ = eigh(self.Aa, self.Ba, subset_by_index=(0, m - 1))
            # 计算精度
            accuracy = max(abs(ee - ed) / ee)
            # 断言精度符合要求
            assert accuracy < tol, msg
    def time_sakurai(self, n, solver):
        # Sakurai 矩阵 ``A`` 条件数很差，导致 `lobpcg` 和 `eigsh` ARPACK 在矩阵 ``A`` 上的收敛非常缓慢
        # 即使对中等大小的最小特征值也需要大量迭代次数
        # 计算其最小特征值甚至从中等大小开始
        # 需要大量迭代次数
        m = 3
        # 计算前 m 个特征值
        ee = self.eigenvalues(m)
        # 设置容差为 100 * n^3 * np.finfo(float).eps
        tol = 100 * n * n * n * np.finfo(float).eps
        # 使用默认随机数生成器创建大小为 (n, m) 的正态分布矩阵 X
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, m))
        # 根据 solver 的选择进行不同的处理
        if solver == 'lobpcg':
            # 忽略警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 使用 lobpcg 方法计算 A 的最小 m 个特征值
                el, _ = lobpcg(self.A, X, tol=1e-9, maxiter=5000, largest=False)
            # 计算精度
            accuracy = max(abs(ee - el) / ee)
            # 断言精度小于容差，否则输出消息 msg
            assert accuracy < tol, msg
        elif solver == 'eigsh':
            # 构造线性操作器 a_l，使用 a 函数进行矩阵向量乘法和矩阵乘法
            a_l = LinearOperator((n, n), matvec=self.A, matmat=self.A, dtype='float64')
            # 使用 eigsh 方法计算 A 的最小 m 个特征值
            ea, _ = eigsh(a_l, k=m, which='SA', tol=1e-9, maxiter=15000,
                          v0=rng.normal(size=(n, 1)))
            # 计算精度
            accuracy = max(abs(ee - ea) / ee)
            # 断言精度小于容差，否则输出消息 msg
            assert accuracy < tol, msg
        else:
            # 使用 eigh 方法计算 Aa 的前 m 个特征值
            ed, _ = eigh(self.Aa, subset_by_index=(0, m - 1))
            # 计算精度
            accuracy = max(abs(ee - ed) / ee)
            # 断言精度小于容差，否则输出消息 msg
            assert accuracy < tol, msg

    def time_sakurai_inverse(self, n, solver):
        # 在 `lobpcg` 和 `eigsh` ARPACK 中应用逆迭代
        # 使用 Cholesky 分解处理带状形式的完整 `np.float64` 精度
        # 以便快速收敛，并与密集带状特征值求解器 `eig_banded` 进行比较
        def a(x):
            return cho_solve_banded((c, False), x)
        m = 3
        # 计算前 m 个特征值
        ee = self.eigenvalues(m)
        # 设置容差为 10 * n^3 * np.finfo(float).eps
        tol = 10 * n * n * n * np.finfo(float).eps
        # 使用默认随机数生成器创建大小为 (n, m) 的正态分布矩阵 X
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, m))
        # 根据 solver 的选择进行不同的处理
        if solver == 'lobpcg':
            # 计算带状 Cholesky 分解
            c = cholesky_banded(self.A)
            # 忽略警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 使用 lobpcg 方法计算 a 的最小 m 个特征值
                el, _ = lobpcg(a, X, tol=1e-9, maxiter=8)
            # 计算精度
            accuracy = max(abs(ee - 1. / el) / ee)
            # 断言精度小于容差，否则输出消息 msg
            assert accuracy < tol, msg
        elif solver == 'eigsh':
            # 计算带状 Cholesky 分解
            c = cholesky_banded(self.A)
            # 构造线性操作器 a_l，使用 a 函数进行矩阵向量乘法和矩阵乘法
            a_l = LinearOperator((n, n), matvec=a, matmat=a, dtype='float64')
            # 使用 eigsh 方法计算 a_l 的最小 m 个特征值
            ea, _ = eigsh(a_l, k=m, which='LA', tol=1e-9, maxiter=8,
                          v0=rng.normal(size=(n, 1)))
            # 计算精度
            accuracy = max(abs(ee - np.sort(1. / ea)) / ee)
            # 断言精度小于容差，否则输出消息 msg
            assert accuracy < tol, msg
        else:
            # 使用 eig_banded 方法计算 A 的前 m 个特征值
            ed, _ = eig_banded(self.A, select='i', select_range=[0, m-1])
            # 计算精度
            accuracy = max(abs(ee - ed) / ee)
            # 断言精度小于容差，否则输出消息 msg
            assert accuracy < tol, msg
```