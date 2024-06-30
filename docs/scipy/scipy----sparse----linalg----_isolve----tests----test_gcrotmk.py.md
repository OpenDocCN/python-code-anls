# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_isolve\tests\test_gcrotmk.py`

```
#!/usr/bin/env python
"""Tests for the linalg._isolve.gcrotmk module
"""

from numpy.testing import (assert_, assert_allclose, assert_equal,
                           suppress_warnings)

import numpy as np
from numpy import zeros, array, allclose
from scipy.linalg import norm
from scipy.sparse import csr_matrix, eye, rand

from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import splu
from scipy.sparse.linalg._isolve import gcrotmk, gmres


# 创建稀疏矩阵 Am，表示为 CSR 格式
Am = csr_matrix(array([[-2,1,0,0,0,9],
                       [1,-2,1,0,5,0],
                       [0,1,-2,1,0,0],
                       [0,0,1,-2,1,0],
                       [0,3,0,1,-2,1],
                       [1,0,0,0,1,-2]]))
# 定义向量 b
b = array([1,2,3,4,5,6])
# 定义计数器列表 count，初始值为 0
count = [0]


# 定义线性运算函数 matvec，用于矩阵向量乘法
def matvec(v):
    count[0] += 1
    return Am @ v


# 创建 LinearOperator 对象 A，用于封装 matvec 函数
A = LinearOperator(matvec=matvec, shape=Am.shape, dtype=Am.dtype)


# 定义求解函数 do_solve
def do_solve(**kw):
    count[0] = 0
    # 使用 gcrotmk 函数求解线性方程 A x = b，设置初始解 x0 为零向量
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning, ".*called without specifying.*")
        x0, flag = gcrotmk(A, b, x0=zeros(A.shape[0]), rtol=1e-14, **kw)
    count_0 = count[0]
    # 断言 A x0 接近于 b，使用 rtol 和 atol 指定误差容限
    assert_(allclose(A @ x0, b, rtol=1e-12, atol=1e-12), norm(A @ x0 - b))
    return x0, count_0


# 定义测试类 TestGCROTMK
class TestGCROTMK:
    # 定义测试方法 test_preconditioner
    def test_preconditioner(self):
        # 检查预条件处理是否有效
        pc = splu(Am.tocsc())
        M = LinearOperator(matvec=pc.solve, shape=A.shape, dtype=A.dtype)

        # 调用 do_solve 函数，比较有无预条件处理的情况
        x0, count_0 = do_solve()
        x1, count_1 = do_solve(M=M)

        # 断言调用次数减少到原来的一半以下
        assert_equal(count_1, 3)
        assert_(count_1 < count_0 / 2)
        # 断言解 x1 接近于解 x0
        assert_(allclose(x1, x0, rtol=1e-14))

    # 定义测试方法 test_arnoldi
    def test_arnoldi(self):
        np.random.seed(1)

        # 创建随机稀疏矩阵 A 和向量 b
        A = eye(2000) + rand(2000, 2000, density=5e-4)
        b = np.random.rand(2000)

        # 使用 gcrotmk 函数和 gmres 函数求解线性方程，进行比较
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            x0, flag0 = gcrotmk(A, b, x0=zeros(A.shape[0]), m=15, k=0, maxiter=1)
            x1, flag1 = gmres(A, b, x0=zeros(A.shape[0]), restart=15, maxiter=1)

        # 断言求解是否成功
        assert_equal(flag0, 1)
        assert_equal(flag1, 1)
        # 断言解的误差小于指定阈值
        assert np.linalg.norm(A.dot(x0) - b) > 1e-3

        # 断言两种方法得到的解接近
        assert_allclose(x0, x1)
    def test_cornercase(self):
        # 使用种子1234设置随机数生成器，以确保可重复性
        np.random.seed(1234)

        # 对于 tol=0，可能由于舍入误差导致无法收敛 --- 确保在此情况下返回值正确，且不会引发异常

        # 针对不同的 n 值进行测试
        for n in [3, 5, 10, 100]:
            # 创建一个大小为 n x n 的单位矩阵的两倍，作为矩阵 A
            A = 2 * eye(n)

            # 使用抑制警告上下文，过滤特定的 DeprecationWarning
            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning, ".*called without specifying.*")
                
                # 创建一个大小为 n 的全为1的向量 b，使用 gcrotmk 函数求解线性方程 A * x = b，最大迭代次数为10
                x, info = gcrotmk(A, b=np.ones(n), maxiter=10)
                # 断言 info 的值为 0，表明成功收敛
                assert_equal(info, 0)
                # 断言 A * x - b 的结果在给定的公差 1e-14 范围内接近于0
                assert_allclose(A.dot(x) - b, 0, atol=1e-14)

                # 使用 rtol=0 参数再次调用 gcrotmk 函数
                x, info = gcrotmk(A, b=np.ones(n), rtol=0, maxiter=10)
                # 如果 info 为 0，再次断言 A * x - b 的结果在公差 1e-14 范围内接近于0
                if info == 0:
                    assert_allclose(A.dot(x) - b, 0, atol=1e-14)

                # 创建一个大小为 n 的随机浮点数向量 b
                b = np.random.rand(n)
                # 使用 gcrotmk 函数求解线性方程 A * x = b，最大迭代次数为10
                x, info = gcrotmk(A, b, maxiter=10)
                # 断言 info 的值为 0，表明成功收敛
                assert_equal(info, 0)
                # 断言 A * x - b 的结果在给定的公差 1e-14 范围内接近于0
                assert_allclose(A.dot(x) - b, 0, atol=1e-14)

                # 使用 rtol=0 参数再次调用 gcrotmk 函数
                x, info = gcrotmk(A, b, rtol=0, maxiter=10)
                # 如果 info 为 0，再次断言 A * x - b 的结果在公差 1e-14 范围内接近于0
                if info == 0:
                    assert_allclose(A.dot(x) - b, 0, atol=1e-14)

    def test_nans(self):
        # 创建一个大小为 3x3 的单位对角线稀疏矩阵 A，其中 A[1,1] 设置为 NaN
        A = eye(3, format='lil')
        A[1,1] = np.nan
        # 创建一个大小为 3 的全为1的向量 b
        b = np.ones(3)

        # 使用抑制警告上下文，过滤特定的 DeprecationWarning
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            # 使用 gcrotmk 函数求解线性方程 A * x = b，rtol=0，最大迭代次数为10
            x, info = gcrotmk(A, b, rtol=0, maxiter=10)
            # 断言 info 的值为 1，表明未能成功收敛
            assert_equal(info, 1)

    def test_truncate(self):
        # 使用种子1234设置随机数生成器，以确保可重复性
        np.random.seed(1234)
        # 创建一个大小为 30x30 的随机矩阵 A，同时加上单位矩阵
        A = np.random.rand(30, 30) + np.eye(30)
        # 创建一个大小为 30 的随机浮点数向量 b
        b = np.random.rand(30)

        # 遍历 'oldest' 和 'smallest' 两种截断方式进行测试
        for truncate in ['oldest', 'smallest']:
            # 使用抑制警告上下文，过滤特定的 DeprecationWarning
            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning, ".*called without specifying.*")
                # 使用 gcrotmk 函数求解线性方程 A * x = b，m=10, k=10，指定截断方式为 truncate，
                # 公差为 1e-4，最大迭代次数为 200
                x, info = gcrotmk(A, b, m=10, k=10, truncate=truncate,
                                  rtol=1e-4, maxiter=200)
            # 断言 info 的值为 0，表明成功收敛
            assert_equal(info, 0)
            # 断言 A * x - b 的结果在给定的公差 1e-3 范围内接近于0
            assert_allclose(A.dot(x) - b, 0, atol=1e-3)

    def test_CU(self):
        # 遍历 discard_C 参数为 True 和 False 两种情况进行测试
        for discard_C in (True, False):
            # 检查 C,U 的行为是否符合预期
            CU = []
            # 调用 do_solve 函数，返回解 x0 和计数 count_0
            x0, count_0 = do_solve(CU=CU, discard_C=discard_C)
            # 断言 CU 非空
            assert_(len(CU) > 0)
            # 断言 CU 的长度不超过 6
            assert_(len(CU) <= 6)

            # 如果 discard_C 为 True，则断言所有的 c 均为 None
            if discard_C:
                for c, u in CU:
                    assert_(c is None)

            # 再次调用 do_solve 函数，返回解 x1 和计数 count_1
            x1, count_1 = do_solve(CU=CU, discard_C=discard_C)
            # 如果 discard_C 为 True，断言 count_1 等于 2 + len(CU)
            # 如果 discard_C 为 False，断言 count_1 等于 3
            if discard_C:
                assert_equal(count_1, 2 + len(CU))
            else:
                assert_equal(count_1, 3)
            # 断言 count_1 不超过 count_0/2
            assert_(count_1 <= count_0 / 2)
            # 断言 x1 和 x0 在公差 1e-14 范围内接近
            assert_allclose(x1, x0, atol=1e-14)
    def test_denormals(self):
        # 定义测试函数 test_denormals
        # 检查当矩阵包含无法用浮点数表示的数值时，即使没有警告被发出，求解器也能正常工作。
        
        # 创建一个2x2的浮点数数组 A
        A = np.array([[1, 2], [3, 4]], dtype=float)
        # 将数组 A 中的每个元素乘以接近零的最大正数，并乘以 100
        A *= 100 * np.nextafter(0, 1)

        # 创建一个包含两个元素的数组 b
        b = np.array([1, 1])

        # 使用 suppress_warnings 上下文管理器，以捕获并抑制警告
        with suppress_warnings() as sup:
            # 过滤特定的 DeprecationWarning，以避免特定信息被输出
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            # 调用 gcrotmk 函数，传入数组 A 和 b，接收返回的 xp 和 info
            xp, info = gcrotmk(A, b)

        # 如果求解信息 info 等于 0，则断言矩阵 A 乘以 xp 等于 b
        if info == 0:
            assert_allclose(A.dot(xp), b)
```