# `D:\src\scipysrc\scipy\benchmarks\benchmarks\linalg.py`

```
import math  # 导入 math 模块

import numpy.linalg as nl  # 导入 numpy.linalg 模块并使用 nl 别名

import numpy as np  # 导入 numpy 库
from numpy.testing import assert_  # 从 numpy.testing 导入 assert_ 函数
from numpy.random import rand  # 从 numpy.random 导入 rand 函数

from .common import Benchmark, safe_import  # 从当前包的 common 模块中导入 Benchmark 和 safe_import 函数

with safe_import():  # 使用 safe_import 函数确保在导入 scipy.linalg 时不会出现错误
    import scipy.linalg as sl  # 导入 scipy.linalg 模块并使用 sl 别名


def random(size):
    return rand(*size)  # 返回一个指定 size 的随机数组


class Bench(Benchmark):
    params = [
        [20, 100, 500, 1000],  # 参数列表包含不同的尺寸大小
        ['contig', 'nocont'],  # 参数列表包含连续和非连续数组
        ['numpy', 'scipy']  # 参数列表包含 numpy 和 scipy 两个模块
    ]
    param_names = ['size', 'contiguous', 'module']  # 参数名列表定义了各参数的名称

    def __init__(self):
        # 对于大尺寸大小，不太适合对 svd 进行基准测试
        self.time_svd.__func__.params = [[20, 100, 500]] + self.params[1:]

    def setup(self, size, contig, module):
        if module == 'numpy' and size >= 200:
            # 跳过：慢且不适合对 numpy 进行基准测试
            raise NotImplementedError()

        a = random([size, size])  # 创建一个随机矩阵 a，大小为 size x size
        # 确保矩阵 a 是非奇异的：
        for i in range(size):
            a[i, i] = 10 * (.1 + a[i, i])
        b = random([size])  # 创建一个随机向量 b，大小为 size

        if contig != 'contig':
            a = a[-1::-1, -1::-1]  # 将矩阵 a 转换为非连续数组
            assert_(not a.flags['CONTIGUOUS'])  # 断言矩阵 a 不是连续的

        self.a = a  # 将矩阵 a 存储在实例变量中
        self.b = b  # 将向量 b 存储在实例变量中

    def time_solve(self, size, contig, module):
        if module == 'numpy':
            nl.solve(self.a, self.b)  # 使用 numpy.linalg.solve 求解线性方程组
        else:
            sl.solve(self.a, self.b)  # 使用 scipy.linalg.solve 求解线性方程组

    def time_solve_triangular(self, size, contig, module):
        # 将 self.a 视为一个下三角矩阵，忽略严格上三角部分
        if module == 'numpy':
            pass  # numpy 不需要专门处理下三角解
        else:
            sl.solve_triangular(self.a, self.b, lower=True)  # 使用 scipy.linalg.solve_triangular 求解下三角线性方程组

    def time_inv(self, size, contig, module):
        if module == 'numpy':
            nl.inv(self.a)  # 使用 numpy.linalg.inv 计算矩阵的逆
        else:
            sl.inv(self.a)  # 使用 scipy.linalg.inv 计算矩阵的逆

    def time_det(self, size, contig, module):
        if module == 'numpy':
            nl.det(self.a)  # 使用 numpy.linalg.det 计算矩阵的行列式
        else:
            sl.det(self.a)  # 使用 scipy.linalg.det 计算矩阵的行列式

    def time_eigvals(self, size, contig, module):
        if module == 'numpy':
            nl.eigvals(self.a)  # 使用 numpy.linalg.eigvals 计算矩阵的特征值
        else:
            sl.eigvals(self.a)  # 使用 scipy.linalg.eigvals 计算矩阵的特征值

    def time_svd(self, size, contig, module):
        if module == 'numpy':
            nl.svd(self.a)  # 使用 numpy.linalg.svd 计算矩阵的奇异值分解
        else:
            sl.svd(self.a)  # 使用 scipy.linalg.svd 计算矩阵的奇异值分解

    # 保留旧的基准测试结果（如果更改基准测试，请删除此注释）
    time_det.version = (
        "87e530ee50eb6b6c06c7a8abe51c2168e133d5cbd486f4c1c2b9cedc5a078325"
    )
    time_eigvals.version = (
        "9d68d3a6b473df9bdda3d3fd25c7f9aeea7d5cee869eec730fb2a2bcd1dfb907"
    )
    time_inv.version = (
        "20beee193c84a5713da9749246a7c40ef21590186c35ed00a4fe854cce9e153b"
    )
    time_solve.version = (
        "1fe788070f1c9132cbe78a47fdb4cce58266427fc636d2aa9450e3c7d92c644c"
    )
    time_svd.version = (
        "0ccbda456d096e459d4a6eefc6c674a815179e215f83931a81cfa8c18e39d6e3"
    )


class Norm(Benchmark):
    params = [
        [(20, 20), (100, 100), (1000, 1000), (20, 1000), (1000, 20)],  # 不同尺寸的矩阵对
        ['contig', 'nocont'],  # 连续和非连续数组
        ['numpy', 'scipy']  # numpy 和 scipy 两个模块
    ]
    # 定义参数名列表，包括 shape、contiguous、module
    param_names = ['shape', 'contiguous', 'module']
    
    # 定义设置方法，用于初始化和设置数组 a
    def setup(self, shape, contig, module):
        # 使用 np.random.randn 生成一个形状为 shape 的随机数组 a
        a = np.random.randn(*shape)
        # 如果 contiguous 不等于 'contig'，则将数组 a 转换为非连续存储的数组
        if contig != 'contig':
            a = a[-1::-1, -1::-1]  # 将数组转换为非连续数组
            # 断言数组 a 不是连续存储的
            assert_(not a.flags['CONTIGUOUS'])
        # 将数组 a 赋值给实例变量 self.a
        self.a = a
    
    # 定义计算 1-范数的方法
    def time_1_norm(self, size, contig, module):
        # 如果 module 是 'numpy'，使用 numpy 库计算数组 a 的 1-范数
        if module == 'numpy':
            nl.norm(self.a, ord=1)
        else:
            # 否则使用 scipy 库计算数组 a 的 1-范数
            sl.norm(self.a, ord=1)
    
    # 定义计算无穷范数的方法
    def time_inf_norm(self, size, contig, module):
        # 如果 module 是 'numpy'，使用 numpy 库计算数组 a 的无穷范数
        if module == 'numpy':
            nl.norm(self.a, ord=np.inf)
        else:
            # 否则使用 scipy 库计算数组 a 的无穷范数
            sl.norm(self.a, ord=np.inf)
    
    # 定义计算Frobenius范数的方法
    def time_frobenius_norm(self, size, contig, module):
        # 如果 module 是 'numpy'，使用 numpy 库计算数组 a 的 Frobenius 范数
        if module == 'numpy':
            nl.norm(self.a)
        else:
            # 否则使用 scipy 库计算数组 a 的 Frobenius 范数
            sl.norm(self.a)
class Lstsq(Benchmark):
    """
    Test the speed of four least-squares solvers on not full rank matrices.
    Also check the difference in the solutions.

    The matrix has the size ``(m, 2/3*m)``; the rank is ``1/2 * m``.
    Matrix values are random in the range (-5, 5), the same is for the right
    hand side.  The complex matrix is the sum of real and imaginary matrices.
    """

    param_names = ['dtype', 'size', 'driver']
    params = [
        [np.float64, np.complex128],  # 数据类型可以是 np.float64 或 np.complex128
        [10, 100, 1000],  # 尺寸可以是 10, 100, 或 1000
        ['gelss', 'gelsy', 'gelsd', 'numpy'],  # 使用的 LAPACK 驱动程序可以是 'gelss', 'gelsy', 'gelsd', 或 'numpy'
    ]

    def setup(self, dtype, size, lapack_driver):
        if lapack_driver == 'numpy' and size >= 200:
            # 如果使用 numpy 驱动并且尺寸大于等于 200，则跳过：因为速度慢且不适合作为基准测试
            raise NotImplementedError()

        rng = np.random.default_rng(1234)
        n = math.ceil(2./3. * size)  # 计算矩阵列数
        k = math.ceil(1./2. * size)   # 计算矩阵秩
        m = size                      # 矩阵行数

        if dtype is np.complex128:
            A = ((10 * rng.random((m,k)) - 5) +  # 创建复数类型的随机矩阵 A
                 1j*(10 * rng.random((m,k)) - 5))
            temp = ((10 * rng.random((k,n)) - 5) +  # 创建复数类型的随机矩阵 temp
                    1j*(10 * rng.random((k,n)) - 5))
            b = ((10 * rng.random((m,1)) - 5) +  # 创建复数类型的随机向量 b
                 1j*(10 * rng.random((m,1)) - 5))
        else:
            A = (10 * rng.random((m,k)) - 5)   # 创建实数类型的随机矩阵 A
            temp = 10 * rng.random((k,n)) - 5  # 创建实数类型的随机矩阵 temp
            b = 10 * rng.random((m,1)) - 5     # 创建实数类型的随机向量 b

        self.A = A.dot(temp)  # 计算 A * temp 并存储在 self.A 中
        self.b = b            # 存储向量 b

    def time_lstsq(self, dtype, size, lapack_driver):
        if lapack_driver == 'numpy':
            np.linalg.lstsq(self.A, self.b,
                            rcond=np.finfo(self.A.dtype).eps * 100)  # 使用 numpy 的最小二乘法求解器
        else:
            sl.lstsq(self.A, self.b, cond=None, overwrite_a=False,
                     overwrite_b=False, check_finite=False,
                     lapack_driver=lapack_driver)  # 使用其他 LAPACK 驱动程序的最小二乘法求解器

    # Retain old benchmark results (remove this if changing the benchmark)
    time_lstsq.version = (
        "15ee0be14a0a597c7d1c9a3dab2c39e15c8ac623484410ffefa406bf6b596ebe"
    )


class SpecialMatrices(Benchmark):
    param_names = ['size']
    params = [[4, 128]]  # 尺寸可以是 4 或 128

    def setup(self, size):
        self.x = np.arange(1, size + 1).astype(float)  # 创建一个从 1 到 size 的浮点数数组
        self.small_blocks = [np.ones([2, 2])] * (size//2)  # 创建大小为 2x2 的块状矩阵数组
        self.big_blocks = [np.ones([size//2, size//2]),    # 创建大小为 size//2 x size//2 的块状矩阵数组
                           np.ones([size//2, size//2])]

    def time_block_diag_small(self, size):
        sl.block_diag(*self.small_blocks)  # 测试 sl.block_diag 函数在小块矩阵上的执行时间

    def time_block_diag_big(self, size):
        sl.block_diag(*self.big_blocks)    # 测试 sl.block_diag 函数在大块矩阵上的执行时间

    def time_circulant(self, size):
        sl.circulant(self.x)  # 测试 sl.circulant 函数在给定向量上的执行时间

    def time_companion(self, size):
        sl.companion(self.x)  # 测试 sl.companion 函数在给定向量上的执行时间

    def time_dft(self, size):
        sl.dft(size)  # 测试 sl.dft 函数在给定尺寸上的执行时间

    def time_hadamard(self, size):
        sl.hadamard(size)  # 测试 sl.hadamard 函数在给定尺寸上的执行时间

    def time_hankel(self, size):
        sl.hankel(self.x)  # 测试 sl.hankel 函数在给定向量上的执行时间

    def time_helmert(self, size):
        sl.helmert(size)  # 测试 sl.helmert 函数在给定尺寸上的执行时间

    def time_hilbert(self, size):
        sl.hilbert(size)  # 测试 sl.hilbert 函数在给定尺寸上的执行时间
    # 调用sl模块中的invhilbert函数，并传入size参数
    def time_invhilbert(self, size):
        sl.invhilbert(size)
    
    # 调用sl模块中的leslie函数，使用self.x作为第一个参数，self.x的切片作为第二个参数
    def time_leslie(self, size):
        sl.leslie(self.x, self.x[1:])
    
    # 调用sl模块中的pascal函数，并传入size参数
    def time_pascal(self, size):
        sl.pascal(size)
    
    # 调用sl模块中的invpascal函数，并传入size参数
    def time_invpascal(self, size):
        sl.invpascal(size)
    
    # 调用sl模块中的toeplitz函数，使用self.x作为参数
    def time_toeplitz(self, size):
        sl.toeplitz(self.x)
# 定义一个名为 GetFuncs 的类，继承自 Benchmark 类
class GetFuncs(Benchmark):
    
    # 定义 setup 方法，用于初始化实例属性 self.x
    def setup(self):
        self.x = np.eye(1)

    # 定义 time_get_blas_funcs 方法，用于测试获取 BLAS 函数 'gemm' 的性能
    def time_get_blas_funcs(self):
        sl.blas.get_blas_funcs('gemm', dtype=float)

    # 定义 time_get_blas_funcs_2 方法，用于测试获取多个 BLAS 函数（'gemm', 'axpy'）的性能
    # 参数为两个 numpy 数组 self.x 和 self.x
    def time_get_blas_funcs_2(self):
        sl.blas.get_blas_funcs(('gemm', 'axpy'), (self.x, self.x))

    # 定义 time_small_cholesky 方法，用于测试对小矩阵 self.x 执行 Cholesky 分解的性能
    def time_small_cholesky(self):
        sl.cholesky(self.x)
```