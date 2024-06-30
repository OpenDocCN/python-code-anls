# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_isolve\tests\test_lsqr.py`

```
# 导入必要的库和模块
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_allclose, assert_array_equal, assert_equal  # 导入 NumPy 测试工具
import pytest  # 导入 pytest，用于单元测试
import scipy.sparse  # 导入 SciPy 稀疏矩阵模块
import scipy.sparse.linalg  # 导入 SciPy 稀疏矩阵的线性代数运算模块
from scipy.sparse.linalg import lsqr  # 从 SciPy 稀疏矩阵的线性代数模块导入 lsqr 函数

# 设置一个测试问题
n = 35  # 设置问题维度为 35
G = np.eye(n)  # 创建一个 n x n 的单位矩阵 G
normal = np.random.normal  # 定义正态分布随机数生成函数别名
norm = np.linalg.norm  # 定义向量或矩阵的范数计算函数别名

# 循环生成随机数用于更新 G 矩阵
for jj in range(5):
    gg = normal(size=n)  # 生成长度为 n 的随机正态分布数组 gg
    hh = gg * gg.T  # 计算 gg 的外积
    G += (hh + hh.T) * 0.5  # 更新 G 矩阵
    G += normal(size=n) * normal(size=n)  # 更新 G 矩阵的另一部分随机数值

b = normal(size=n)  # 生成长度为 n 的随机正态分布数组 b

# lsqr() 函数的 atol 和 btol 关键字参数的公差设置
tol = 2e-10  # 设置公差为 2e-10
atol_test = 4e-10  # 设置测试用 assert_allclose 的绝对公差为 4e-10
rtol_test = 2e-8  # 设置测试用 assert_allclose 的相对公差为 2e-8
show = False  # 设置是否显示详细信息为 False
maxit = None  # 设置迭代次数上限为 None


def test_lsqr_basic():
    b_copy = b.copy()  # 复制数组 b
    xo, *_ = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit)  # 调用 lsqr() 解线性方程
    assert_array_equal(b_copy, b)  # 检查 b 的复制是否与原始 b 相等

    svx = np.linalg.solve(G, b)  # 使用 NumPy 求解线性方程 Gx = b
    assert_allclose(xo, svx, atol=atol_test, rtol=rtol_test)  # 使用 assert_allclose 检查 lsqr() 的结果与 solve() 的结果是否接近

    # 添加阻尼项 damp > 0 后的 lsqr() 求解
    damp = 1.5  # 设置阻尼因子为 1.5
    xo, *_ = lsqr(
        G, b, damp=damp, show=show, atol=tol, btol=tol, iter_lim=maxit)  # 调用 lsqr() 求解阻尼问题

    Gext = np.r_[G, damp * np.eye(G.shape[1])]  # 扩展 G 矩阵
    bext = np.r_[b, np.zeros(G.shape[1])]  # 扩展 b 数组
    svx, *_ = np.linalg.lstsq(Gext, bext, rcond=None)  # 使用最小二乘法求解扩展问题
    assert_allclose(xo, svx, atol=atol_test, rtol=rtol_test)  # 检查 lsqr() 的阻尼解与最小二乘法的解是否接近


def test_gh_2466():
    row = np.array([0, 0])  # 定义行索引数组
    col = np.array([0, 1])  # 定义列索引数组
    val = np.array([1, -1])  # 定义数值数组
    A = scipy.sparse.coo_matrix((val, (row, col)), shape=(1, 2))  # 创建稀疏 COO 矩阵 A
    b = np.asarray([4])  # 创建稀疏矩阵的右侧向量 b
    lsqr(A, b)  # 调用 lsqr() 求解稀疏矩阵问题


def test_well_conditioned_problems():
    # 测试稀疏 lsqr 求解器在不同随机种子下返回正确的解
    # 这是对潜在的 ZeroDivisionError 的非回归测试
    n = 10  # 设置问题维度为 10
    A_sparse = scipy.sparse.eye(n, n)  # 创建一个稀疏单位矩阵 A_sparse
    A_dense = A_sparse.toarray()  # 将稀疏矩阵转换为密集矩阵

    with np.errstate(invalid='raise'):  # 设置错误状态处理
        for seed in range(30):
            rng = np.random.RandomState(seed + 10)  # 使用不同的随机种子生成随机数发生器
            beta = rng.rand(n)  # 生成长度为 n 的随机数数组 beta
            beta[beta == 0] = 0.00001  # 确保所有的 beta 元素不为零
            b = A_sparse @ beta[:, np.newaxis]  # 计算稀疏矩阵与 beta 的乘积
            output = lsqr(A_sparse, b, show=show)  # 调用 lsqr() 求解稀疏矩阵问题

            # 检查终止条件是否对应于 Ax = b 的近似解
            assert_equal(output[1], 1)  # 检查 lsqr() 的输出是否满足终止条件
            solution = output[0]  # 获取 lsqr() 的解

            # 检查是否恢复了真实解
            assert_allclose(solution, beta)  # 检查 lsqr() 的解是否与真实解 beta 接近

            # 确认检查: 与密集矩阵求解器的比较
            reference_solution = np.linalg.solve(A_dense, b).ravel()  # 使用密集矩阵求解真实解
            assert_allclose(solution, reference_solution)  # 检查 lsqr() 的解是否与密集求解器的解接近


def test_b_shapes():
    # 测试 b 为标量的情况
    A = np.array([[1.0, 2.0]])  # 创建一个 1x2 的数组 A
    b = 3.0  # 设置标量 b
    x = lsqr(A, b)[0]  # 调用 lsqr() 求解线性方程
    assert norm(A.dot(x) - b) == pytest.approx(0)  # 检查解 x 是否满足方程 Ax = b
    # 创建一个10x10的单位矩阵 A
    A = np.eye(10)
    # 创建一个10x1的列向量 b，其中元素全为1
    b = np.ones((10, 1))
    # 使用 lsqr 方法求解线性方程组 A*x = b，返回解向量 x
    x = lsqr(A, b)[0]
    # 断言：验证 A*x 与 b 的差的范数近似为0，使用 pytest 的近似匹配
    assert norm(A.dot(x) - b.ravel()) == pytest.approx(0)
# 定义一个测试函数，用于验证初始化设置的正确性

def test_initialization():
    # 复制 b 数组的内容到 b_copy 中，用于后续比较
    b_copy = b.copy()
    
    # 使用 lsqr 函数求解线性方程 Gx = b，并返回结果 x_ref
    x_ref = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit)
    
    # 创建一个与 x_ref[0] 相同形状的全零数组作为初始值 x0
    x0 = np.zeros(x_ref[0].shape)
    
    # 使用 lsqr 函数再次求解线性方程 Gx = b，指定初始值为 x0，并返回结果 x
    x = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit, x0=x0)
    
    # 断言 b_copy 与原始 b 数组相等
    assert_array_equal(b_copy, b)
    
    # 断言 x_ref[0] 与 x[0] 的值在容差范围内接近
    assert_allclose(x_ref[0], x[0])

    # 测试使用单次迭代的热启动功能
    # 将 lsqr 函数用于 Gx = b，只进行一次迭代，并返回结果的第一个元素作为 x0
    x0 = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=1)[0]
    
    # 使用 lsqr 函数再次求解线性方程 Gx = b，指定初始值为 x0，并返回结果 x
    x = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit, x0=x0)
    
    # 断言 x_ref[0] 与 x[0] 的值在容差范围内接近
    assert_allclose(x_ref[0], x[0])
    
    # 断言 b_copy 与原始 b 数组相等
    assert_array_equal(b_copy, b)
```