# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_isolve\tests\test_lgmres.py`

```
"""Tests for the linalg._isolve.lgmres module
"""

# 导入必要的测试工具函数
from numpy.testing import (assert_, assert_allclose, assert_equal,
                           suppress_warnings)

# 导入 pytest 和 Python 版本信息
import pytest
from platform import python_implementation

# 导入 NumPy 和 SciPy 所需的库
import numpy as np
from numpy import zeros, array, allclose
from scipy.linalg import norm
from scipy.sparse import csr_matrix, eye, rand

# 导入线性操作相关的类和函数
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import splu
from scipy.sparse.linalg._isolve import lgmres, gmres

# 定义稀疏矩阵 Am 和向量 b
Am = csr_matrix(array([[-2, 1, 0, 0, 0, 9],
                       [1, -2, 1, 0, 5, 0],
                       [0, 1, -2, 1, 0, 0],
                       [0, 0, 1, -2, 1, 0],
                       [0, 3, 0, 1, -2, 1],
                       [1, 0, 0, 0, 1, -2]]))
b = array([1, 2, 3, 4, 5, 6])

# 定义计数器 count
count = [0]

# 定义矩阵向量乘法函数 matvec
def matvec(v):
    count[0] += 1
    return Am @ v

# 创建线性操作对象 A
A = LinearOperator(matvec=matvec, shape=Am.shape, dtype=Am.dtype)

# 定义解线性方程组函数 do_solve
def do_solve(**kw):
    count[0] = 0
    # 使用 lgmres 求解线性方程组，设置初始解 x0 和其他参数
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning, ".*called without specifying.*")
        x0, flag = lgmres(A, b, x0=zeros(A.shape[0]),
                          inner_m=6, rtol=1e-14, **kw)
    count_0 = count[0]
    # 断言解的精度符合要求
    assert_(allclose(A @ x0, b, rtol=1e-12, atol=1e-12), norm(A @ x0 - b))
    return x0, count_0

# 定义测试类 TestLGMRES
class TestLGMRES:
    # 测试预条件的功能
    def test_preconditioner(self):
        # 检查预条件的效果
        pc = splu(Am.tocsc())
        M = LinearOperator(matvec=pc.solve, shape=A.shape, dtype=A.dtype)

        x0, count_0 = do_solve()
        x1, count_1 = do_solve(M=M)

        assert_(count_1 == 3)
        assert_(count_1 < count_0 / 2)
        assert_(allclose(x1, x0, rtol=1e-14))

    # 测试外部向量的行为
    def test_outer_v(self):
        # 检查增广向量的行为是否符合预期

        outer_v = []
        x0, count_0 = do_solve(outer_k=6, outer_v=outer_v)
        assert_(len(outer_v) > 0)
        assert_(len(outer_v) <= 6)

        x1, count_1 = do_solve(outer_k=6, outer_v=outer_v,
                               prepend_outer_v=True)
        assert_(count_1 == 2, count_1)
        assert_(count_1 < count_0 / 2)
        assert_(allclose(x1, x0, rtol=1e-14))

        # ---

        outer_v = []
        x0, count_0 = do_solve(outer_k=6, outer_v=outer_v,
                               store_outer_Av=False)
        assert_(array([v[1] is None for v in outer_v]).all())
        assert_(len(outer_v) > 0)
        assert_(len(outer_v) <= 6)

        x1, count_1 = do_solve(outer_k=6, outer_v=outer_v,
                               prepend_outer_v=True)
        assert_(count_1 == 3, count_1)
        assert_(count_1 < count_0 / 2)
        assert_(allclose(x1, x0, rtol=1e-14))

    @pytest.mark.skipif(python_implementation() == 'PyPy',
                        reason="Fails on PyPy CI runs. See #9507")
    def test_arnoldi(self):
        # 设置随机数种子，以便结果可重复
        np.random.seed(1234)

        # 创建一个稀疏的2000x2000单位矩阵，再加上一个密度为5e-4的随机矩阵，构成系数矩阵 A
        A = eye(2000) + rand(2000, 2000, density=5e-4)
        
        # 创建一个长度为2000的随机向量 b
        b = np.random.rand(2000)

        # 使用 lgmres 进行迭代求解线性方程组 Ax = b，设置初值 x0 为全零向量，内部迭代次数为15，外部最大迭代次数为1
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            
            # 使用 lgmres 求解线性方程组
            x0, flag0 = lgmres(A, b, x0=zeros(A.shape[0]), inner_m=15, maxiter=1)
            
            # 使用 gmres 求解线性方程组
            x1, flag1 = gmres(A, b, x0=zeros(A.shape[0]), restart=15, maxiter=1)

        # 断言内部迭代的返回标志为1，表示迭代收敛
        assert_equal(flag0, 1)
        # 断言外部迭代的返回标志为1，表示迭代收敛
        assert_equal(flag1, 1)
        
        # 计算解的残差的二范数，并断言其大于1e-4，表示解的精度达到预期
        norm = np.linalg.norm(A.dot(x0) - b)
        assert_(norm > 1e-4)
        
        # 断言两种方法求解得到的解 x0 和 x1 在给定的公差下相等
        assert_allclose(x0, x1)

    def test_cornercase(self):
        # 设置随机数种子，以便结果可重复
        np.random.seed(1234)

        # 对于不同的 n 值进行循环测试
        for n in [3, 5, 10, 100]:
            # 创建一个大小为 n 的2倍的单位矩阵 A
            A = 2*eye(n)

            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning, ".*called without specifying.*")
                
                # 创建一个全1的向量作为右端向量 b
                b = np.ones(n)
                
                # 使用 lgmres 求解线性方程组，最大迭代次数为10
                x, info = lgmres(A, b, maxiter=10)
                # 断言迭代结束信息为0，表示迭代收敛
                assert_equal(info, 0)
                # 断言解的精度在公差1e-14内，表示解的精度达到预期
                assert_allclose(A.dot(x) - b, 0, atol=1e-14)

                # 使用 lgmres 求解线性方程组，公差为0，最大迭代次数为10
                x, info = lgmres(A, b, rtol=0, maxiter=10)
                if info == 0:
                    # 如果迭代收敛，断言解的精度在公差1e-14内，表示解的精度达到预期
                    assert_allclose(A.dot(x) - b, 0, atol=1e-14)

                # 创建一个随机向量作为右端向量 b
                b = np.random.rand(n)
                
                # 使用 lgmres 求解线性方程组，最大迭代次数为10
                x, info = lgmres(A, b, maxiter=10)
                # 断言迭代结束信息为0，表示迭代收敛
                assert_equal(info, 0)
                # 断言解的精度在公差1e-14内，表示解的精度达到预期
                assert_allclose(A.dot(x) - b, 0, atol=1e-14)

                # 使用 lgmres 求解线性方程组，公差为0，最大迭代次数为10
                x, info = lgmres(A, b, rtol=0, maxiter=10)
                if info == 0:
                    # 如果迭代收敛，断言解的精度在公差1e-14内，表示解的精度达到预期
                    assert_allclose(A.dot(x) - b, 0, atol=1e-14)

    def test_nans(self):
        # 创建一个3x3的单位下三角矩阵 A，其中A[1,1]元素为 NaN
        A = eye(3, format='lil')
        A[1, 1] = np.nan
        # 创建一个长度为3的全1向量 b
        b = np.ones(3)

        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            # 使用 lgmres 求解线性方程组，公差为0，最大迭代次数为10
            x, info = lgmres(A, b, rtol=0, maxiter=10)
            # 断言迭代结束信息为1，表示因为NaN元素而导致迭代中断
            assert_equal(info, 1)

    def test_breakdown_with_outer_v(self):
        # 创建一个2x2的浮点型矩阵 A
        A = np.array([[1, 2], [3, 4]], dtype=float)
        # 创建一个长度为2的浮点型向量 b
        b = np.array([1, 2])

        # 使用 np.linalg.solve 求解线性方程组 Ax = b，得到精确解 x
        x = np.linalg.solve(A, b)
        # 创建一个长度为2的浮点型向量 v0
        v0 = np.array([1, 0])

        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            
            # 使用 lgmres 求解线性方程组，外部向量列表为 [(v0, None), (x, None)], 最大迭代次数为1
            xp, info = lgmres(A, b, outer_v=[(v0, None), (x, None)], maxiter=1)

        # 断言得到的解 xp 和 精确解 x 在公差1e-12内相等，表示解的精度达到预期
        assert_allclose(xp, x, atol=1e-12)
    def test_breakdown_underdetermined(self):
        # 测试处理欠定系统情况下的情况
        # 尽管由于幂零矩阵 A 导致求解器出现故障，应在一个内部迭代中找到 Krylov 空间中的 LSQ 解决方案。
        
        # 定义一个幂零矩阵 A
        A = np.array([[0, 1, 1, 1],
                      [0, 0, 1, 1],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]], dtype=float)

        # 定义多个右端向量 b 的列表
        bs = [
            np.array([1, 1, 1, 1]),
            np.array([1, 1, 1, 0]),
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 0, 0]),
        ]

        # 对每个右端向量 b 进行迭代
        for b in bs:
            # 在忽略特定警告的上下文中执行
            with suppress_warnings() as sup:
                # 过滤掉特定的 DeprecationWarning
                sup.filter(DeprecationWarning, ".*called without specifying.*")
                # 调用 LGMRES 方法求解方程 Ax = b，最大迭代次数为 1
                xp, info = lgmres(A, b, maxiter=1)
            
            # 计算残差 ||Ax - b||
            resp = np.linalg.norm(A.dot(xp) - b)

            # 构造矩阵 K = [b, Ab, A^2b, A^3b]，并求解 A(Ky) = b 的最小二乘解 y
            K = np.c_[b, A.dot(b), A.dot(A.dot(b)), A.dot(A.dot(A.dot(b)))]
            y, _, _, _ = np.linalg.lstsq(A.dot(K), b, rcond=-1)
            # 计算最小二乘解 x = Ky 的残差 ||Ax - b||
            x = K.dot(y)
            res = np.linalg.norm(A.dot(x) - b)

            # 断言残差 resp 和 res 很接近，用于验证求解结果的准确性
            assert_allclose(resp, res, err_msg=repr(b))

    def test_denormals(self):
        # 测试矩阵包含无法精确表示倒数的数字时，求解器的行为
        # 确保矩阵 A 包含这些数字时不会发出警告，并且求解器能正常工作。
        
        # 定义一个矩阵 A，包含无法精确表示倒数的数字
        A = np.array([[1, 2],
                      [3, 4]], dtype=float)
        A *= 100 * np.nextafter(0, 1)

        # 定义右端向量 b
        b = np.array([1, 1])

        # 在忽略特定警告的上下文中执行
        with suppress_warnings() as sup:
            # 过滤掉特定的 DeprecationWarning
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            # 调用 LGMRES 方法求解方程 Ax = b
            xp, info = lgmres(A, b)

        # 如果求解成功（info == 0），则断言 Ax 与 b 很接近，用于验证求解结果的准确性
        if info == 0:
            assert_allclose(A.dot(xp), b)
```