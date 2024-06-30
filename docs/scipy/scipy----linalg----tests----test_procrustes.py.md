# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_procrustes.py`

```
# 导入必要的模块和函数
from itertools import product, permutations

# 导入 numpy 库，并使用别名 np
import numpy as np

# 导入 numpy 测试工具中的特定函数
from numpy.testing import assert_array_less, assert_allclose

# 导入 pytest 库中的 raises 函数，并使用别名 assert_raises
from pytest import raises as assert_raises

# 导入 scipy.linalg 中的特定函数
from scipy.linalg import inv, eigh, norm
from scipy.linalg import orthogonal_procrustes

# 导入 scipy.sparse._sputils 模块中的 matrix 函数
from scipy.sparse._sputils import matrix


# 测试函数：测试当输入的数组维度过大时是否会引发 ValueError 异常
def test_orthogonal_procrustes_ndim_too_large():
    # 设置随机种子以便复现结果
    np.random.seed(1234)
    
    # 生成随机数组 A 和 B，维度为 (3, 4, 5)
    A = np.random.randn(3, 4, 5)
    B = np.random.randn(3, 4, 5)
    
    # 断言调用 orthogonal_procrustes 函数时会引发 ValueError 异常
    assert_raises(ValueError, orthogonal_procrustes, A, B)


# 测试函数：测试当输入的数组维度过小时是否会引发 ValueError 异常
def test_orthogonal_procrustes_ndim_too_small():
    np.random.seed(1234)
    
    # 生成随机数组 A 和 B，维度为 (3,)
    A = np.random.randn(3)
    B = np.random.randn(3)
    
    # 断言调用 orthogonal_procrustes 函数时会引发 ValueError 异常
    assert_raises(ValueError, orthogonal_procrustes, A, B)


# 测试函数：测试当输入的数组形状不匹配时是否会引发 ValueError 异常
def test_orthogonal_procrustes_shape_mismatch():
    np.random.seed(1234)
    
    # 定义不同的数组形状组合
    shapes = ((3, 3), (3, 4), (4, 3), (4, 4))
    
    # 对形状的排列组合进行循环遍历
    for a, b in permutations(shapes, 2):
        # 根据形状生成随机数组 A 和 B
        A = np.random.randn(*a)
        B = np.random.randn(*b)
        
        # 断言调用 orthogonal_procrustes 函数时会引发 ValueError 异常
        assert_raises(ValueError, orthogonal_procrustes, A, B)


# 测试函数：测试当输入的数组包含无限或 NaN 值时是否会引发 ValueError 异常
def test_orthogonal_procrustes_checkfinite_exception():
    np.random.seed(1234)
    m, n = 2, 3
    
    # 生成不同的正常数组 A_good 和 B_good
    A_good = np.random.randn(m, n)
    B_good = np.random.randn(m, n)
    
    # 对每个包含无限或 NaN 值的情况进行循环遍历
    for bad_value in np.inf, -np.inf, np.nan:
        # 复制正常数组 A_good 和 B_good，并替换特定位置的值为 bad_value
        A_bad = A_good.copy()
        A_bad[1, 2] = bad_value
        B_bad = B_good.copy()
        B_bad[1, 2] = bad_value
        
        # 对每一对包含无限或 NaN 值的数组进行测试
        for A, B in ((A_good, B_bad), (A_bad, B_good), (A_bad, B_bad)):
            # 断言调用 orthogonal_procrustes 函数时会引发 ValueError 异常
            assert_raises(ValueError, orthogonal_procrustes, A, B)


# 测试函数：测试正交 Procrustes 问题在尺度不变性下的表现
def test_orthogonal_procrustes_scale_invariance():
    np.random.seed(1234)
    m, n = 4, 3
    
    # 对于随机生成的原始数组 A_orig 和 B_orig，进行三次循环测试
    for i in range(3):
        A_orig = np.random.randn(m, n)
        B_orig = np.random.randn(m, n)
        
        # 调用 orthogonal_procrustes 函数求解 R_orig 和 s
        R_orig, s = orthogonal_procrustes(A_orig, B_orig)
        
        # 对于不同的缩放因子进行循环测试
        for A_scale in np.square(np.random.randn(3)):
            for B_scale in np.square(np.random.randn(3)):
                # 对缩放后的数组调用 orthogonal_procrustes 函数
                R, s = orthogonal_procrustes(A_orig * A_scale, B_orig * B_scale)
                
                # 断言调用结果 R 与 R_orig 具有近似相等的值
                assert_allclose(R, R_orig)


# 测试函数：测试正交 Procrustes 问题在不同数组类型下的转换性能
def test_orthogonal_procrustes_array_conversion():
    np.random.seed(1234)
    
    # 对不同的数组维度组合进行循环测试
    for m, n in ((6, 4), (4, 4), (4, 6)):
        A_arr = np.random.randn(m, n)
        B_arr = np.random.randn(m, n)
        
        # 将数组 A_arr 和 B_arr 转换为不同的类型，并分别进行测试
        As = (A_arr, A_arr.tolist(), matrix(A_arr))
        Bs = (B_arr, B_arr.tolist(), matrix(B_arr))
        
        # 求解正交 Procrustes 问题的旋转矩阵 R_arr
        R_arr, s = orthogonal_procrustes(A_arr, B_arr)
        
        # 计算 A_arr 经过 R_arr 变换后的结果 AR_arr
        AR_arr = A_arr.dot(R_arr)
        
        # 对每一对数组类型组合进行测试
        for A, B in product(As, Bs):
            # 调用 orthogonal_procrustes 函数求解 R 和 s
            R, s = orthogonal_procrustes(A, B)
            
            # 计算 A_arr 经过 R 变换后的结果 AR
            AR = A_arr.dot(R)
            
            # 断言调用结果 AR 与 AR_arr 具有近似相等的值
            assert_allclose(AR, AR_arr)


# 测试函数：测试正交 Procrustes 问题的基本功能
def test_orthogonal_procrustes():
    np.random.seed(1234)
    # 针对每个元组 (6, 4), (4, 4), (4, 6)，依次执行以下操作：
    # 从标准正态分布中随机采样一个 m × n 的矩阵 B
    B = np.random.randn(m, n)
    
    # 从标准正态分布中随机采样一个 n × n 的矩阵 X
    X = np.random.randn(n, n)
    
    # 计算 X 的转置与 X 的和的特征值和特征向量
    w, V = eigh(X.T + X)
    
    # 检查 V 的逆是否与其转置相等，即 V 是否为正交矩阵
    assert_allclose(inv(V), V.T)
    
    # 使用 V 的转置构造一个矩阵 A，使得 A = B · V^T
    A = np.dot(B, V.T)
    
    # 使用正交 Procrustes 分析找到从 A 到 B 的正交变换 R
    R, s = orthogonal_procrustes(A, B)
    
    # 检查 R 的逆是否与其转置相等，即 R 是否为正交矩阵
    assert_allclose(inv(R), R.T)
    
    # 检查 A · R 是否接近于 B
    assert_allclose(A.dot(R), B)
    
    # 创建一个扰动后的输入矩阵 A_perturbed，即 A 的每个元素都加上一个较小的随机扰动
    A_perturbed = A + 1e-2 * np.random.randn(m, n)
    
    # 使用正交 Procrustes 分析找到从 A_perturbed 到 B 的正交变换 R_prime
    R_prime, s = orthogonal_procrustes(A_perturbed, B)
    
    # 检查 R_prime 的逆是否与其转置相等，即 R_prime 是否为正交矩阵
    assert_allclose(inv(R_prime), R_prime.T)
    
    # 计算扰动后的输入矩阵 A_perturbed 分别经过 R 和 R_prime 的近似值
    naive_approx = A_perturbed.dot(R)
    optim_approx = A_perturbed.dot(R_prime)
    
    # 计算使用 Frobenius 范数的矩阵近似误差
    naive_approx_error = norm(naive_approx - B, ord='fro')
    optim_approx_error = norm(optim_approx - B, ord='fro')
    
    # 检查优化后的正交 Procrustes 近似误差是否小于未优化的
    assert_array_less(optim_approx_error, naive_approx_error)
def _centered(A):
    # 计算矩阵 A 沿着列的均值
    mu = A.mean(axis=0)
    # 返回去中心化后的矩阵和均值向量
    return A - mu, mu


def test_orthogonal_procrustes_exact_example():
    # 检查一个小的应用示例。
    # 它涉及平移、缩放、反射和旋转。
    #
    #         |
    #   a  b  |
    #         |
    #   d  c  |        w
    #         |
    # --------+--- x ----- z ---
    #         |
    #         |        y
    #         |
    #
    A_orig = np.array([[-3, 3], [-2, 3], [-2, 2], [-3, 2]], dtype=float)
    B_orig = np.array([[3, 2], [1, 0], [3, -2], [5, 0]], dtype=float)
    A, A_mu = _centered(A_orig)
    B, B_mu = _centered(B_orig)
    # 计算正交 Procrustes 分析，得到旋转矩阵 R 和缩放因子 s
    R, s = orthogonal_procrustes(A, B)
    # 计算缩放因子
    scale = s / np.square(norm(A))
    # 计算近似的 B 值
    B_approx = scale * np.dot(A, R) + B_mu
    # 断言近似的 B 值与原始 B 值相近
    assert_allclose(B_approx, B_orig, atol=1e-8)


def test_orthogonal_procrustes_stretched_example():
    # 使用一个在 y 轴上拉伸的目标再次尝试。
    A_orig = np.array([[-3, 3], [-2, 3], [-2, 2], [-3, 2]], dtype=float)
    B_orig = np.array([[3, 40], [1, 0], [3, -40], [5, 0]], dtype=float)
    A, A_mu = _centered(A_orig)
    B, B_mu = _centered(B_orig)
    R, s = orthogonal_procrustes(A, B)
    scale = s / np.square(norm(A))
    B_approx = scale * np.dot(A, R) + B_mu
    expected = np.array([[3, 21], [-18, 0], [3, -21], [24, 0]], dtype=float)
    assert_allclose(B_approx, expected, atol=1e-8)
    # 检查不匹配的对称性。
    expected_disparity = 0.4501246882793018
    AB_disparity = np.square(norm(B_approx - B_orig) / norm(B))
    assert_allclose(AB_disparity, expected_disparity)
    # 对 B 和 A 进行正交 Procrustes 分析，计算 A 的近似值
    R, s = orthogonal_procrustes(B, A)
    scale = s / np.square(norm(B))
    A_approx = scale * np.dot(B, R) + A_mu
    BA_disparity = np.square(norm(A_approx - A_orig) / norm(A))
    assert_allclose(BA_disparity, expected_disparity)


def test_orthogonal_procrustes_skbio_example():
    # 这个变换也是精确的。
    # 它使用平移、缩放和反射。
    #
    #   |
    #   | a
    #   | b
    #   | c d
    # --+---------
    #   |
    #   |       w
    #   |
    #   |       x
    #   |
    #   |   z   y
    #   |
    #
    A_orig = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], dtype=float)
    B_orig = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], dtype=float)
    B_standardized = np.array([
        [-0.13363062, 0.6681531],
        [-0.13363062, 0.13363062],
        [-0.13363062, -0.40089186],
        [0.40089186, -0.40089186]])
    A, A_mu = _centered(A_orig)
    B, B_mu = _centered(B_orig)
    R, s = orthogonal_procrustes(A, B)
    scale = s / np.square(norm(A))
    B_approx = scale * np.dot(A, R) + B_mu
    # 断言近似的 B 值与原始 B 值相近
    assert_allclose(B_approx, B_orig)
    # 断言标准化后的 B 与预期标准化值相近
    assert_allclose(B / norm(B), B_standardized)


def test_empty():
    # 创建一个空的矩阵
    a = np.empty((0, 0))
    # 对空矩阵进行正交 Procrustes 分析
    r, s = orthogonal_procrustes(a, a)
    # 断言分析结果与空矩阵相似
    assert_allclose(r, np.empty((0, 0)))

    # 创建一个空的矩阵（行数为 0，列数为 3）
    a = np.empty((0, 3))
    # 对空矩阵进行正交 Procrustes 分析
    r, s = orthogonal_procrustes(a, a)
    # 断言分析结果与单位矩阵相似
    assert_allclose(r, np.identity(3))
```