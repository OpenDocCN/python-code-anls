# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_decomp_polar.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.linalg import norm  # 导入 norm 函数，用于计算向量的范数
from numpy.testing import (assert_, assert_allclose, assert_equal)  # 导入测试函数，用于进行数值比较的断言
from scipy.linalg import polar, eigh  # 导入极分解和特征值计算函数

diag2 = np.array([[2, 0], [0, 3]])  # 创建一个2x2的对角矩阵
a13 = np.array([[1, 2, 2]])  # 创建一个1x3的数组

precomputed_cases = [  # 预先计算好的测试用例列表
    [[[0]], 'right', [[1]], [[0]]],  # 第一个预先计算的情况
    [[[0]], 'left', [[1]], [[0]]],   # 第二个预先计算的情况
    [[[9]], 'right', [[1]], [[9]]],  # 第三个预先计算的情况
    [[[9]], 'left', [[1]], [[9]]],   # 第四个预先计算的情况
    [diag2, 'right', np.eye(2), diag2],  # 第五个预先计算的情况
    [diag2, 'left', np.eye(2), diag2],   # 第六个预先计算的情况
    [a13, 'right', a13/norm(a13[0]), a13.T.dot(a13)/norm(a13[0])],  # 第七个预先计算的情况
]

verify_cases = [  # 验证测试用例列表
    [[1, 2], [3, 4]],                   # 第一个验证情况
    [[1, 2, 3]],                        # 第二个验证情况
    [[1], [2], [3]],                    # 第三个验证情况
    [[1, 2, 3], [3, 4, 0]],             # 第四个验证情况
    [[1, 2], [3, 4], [5, 5]],           # 第五个验证情况
    [[1, 2], [3, 4+5j]],                # 第六个验证情况
    [[1, 2, 3j]],                       # 第七个验证情况
    [[1], [2], [3j]],                   # 第八个验证情况
    [[1, 2, 3+2j], [3, 4-1j, -4j]],     # 第九个验证情况
    [[1, 2], [3-2j, 4+0.5j], [5, 5]],   # 第十个验证情况
    [[10000, 10, 1], [-1, 2, 3j], [0, 1, 2]],  # 第十一个验证情况
    np.empty((0, 0)),                   # 第十二个验证情况
    np.empty((0, 2)),                   # 第十三个验证情况
    np.empty((2, 0)),                   # 第十四个验证情况
]

def check_precomputed_polar(a, side, expected_u, expected_p):
    # 检查预先计算的极分解结果是否与给定的参数一致
    u, p = polar(a, side=side)
    assert_allclose(u, expected_u, atol=1e-15)  # 检查 u 是否与预期的 expected_u 接近
    assert_allclose(p, expected_p, atol=1e-15)  # 检查 p 是否与预期的 expected_p 接近

def verify_polar(a):
    # 计算极分解，然后验证结果是否符合预期的属性
    product_atol = np.sqrt(np.finfo(float).eps)  # 设置数值比较的绝对误差限

    aa = np.asarray(a)
    m, n = aa.shape

    u, p = polar(a, side='right')  # 计算右极分解
    assert_equal(u.shape, (m, n))  # 断言 u 的形状与预期一致
    assert_equal(p.shape, (n, n))  # 断言 p 的形状与预期一致
    assert_allclose(u.dot(p), a, atol=product_atol)  # 检查是否满足 a = up
    if m >= n:
        assert_allclose(u.conj().T.dot(u), np.eye(n), atol=1e-15)  # 如果 m >= n，检查 u 是否为单位正交矩阵
    else:
        assert_allclose(u.dot(u.conj().T), np.eye(m), atol=1e-15)  # 如果 m < n，检查 u 是否为单位正交矩阵
    assert_allclose(p.conj().T, p)  # 检查 p 是否为共轭转置等于自身的 Hermitian 矩阵
    evals = eigh(p, eigvals_only=True)  # 计算 p 的特征值
    nonzero_evals = evals[abs(evals) > 1e-14]  # 获取非零特征值
    assert_((nonzero_evals >= 0).all())  # 断言所有非零特征值都大于等于零

    u, p = polar(a, side='left')  # 计算左极分解
    assert_equal(u.shape, (m, n))  # 断言 u 的形状与预期一致
    assert_equal(p.shape, (m, m))  # 断言 p 的形状与预期一致
    assert_allclose(p.dot(u), a, atol=product_atol)  # 检查是否满足 a = pu
    if m >= n:
        assert_allclose(u.conj().T.dot(u), np.eye(n), atol=1e-15)  # 如果 m >= n，检查 u 是否为单位正交矩阵
    else:
        assert_allclose(u.dot(u.conj().T), np.eye(m), atol=1e-15)  # 如果 m < n，检查 u 是否为单位正交矩阵
    assert_allclose(p.conj().T, p)  # 检查 p 是否为共轭转置等于自身的 Hermitian 矩阵
    evals = eigh(p, eigvals_only=True)  # 计算 p 的特征值
    nonzero_evals = evals[abs(evals) > 1e-14]  # 获取非零特征值
    assert_((nonzero_evals >= 0).all())  # 断言所有非零特征值都大于等于零

def test_precomputed_cases():
    # 对预先计算的每个情况进行测试
    for a, side, expected_u, expected_p in precomputed_cases:
        check_precomputed_polar(a, side, expected_u, expected_p)

def test_verify_cases():
    # 对验证用例列表中的每个情况进行测试
    for a in verify_cases:
        verify_polar(a)
```