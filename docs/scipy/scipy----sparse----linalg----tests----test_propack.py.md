# `D:\src\scipysrc\scipy\scipy\sparse\linalg\tests\test_propack.py`

```
import os  # 导入操作系统接口模块
import pytest  # 导入 pytest 测试框架

import numpy as np  # 导入 NumPy 数学库，并用 np 别名引用
from numpy.testing import assert_allclose  # 从 NumPy 的测试模块中导入断言函数 assert_allclose
from pytest import raises as assert_raises  # 从 pytest 中导入 raises 函数，并用 assert_raises 别名引用
from scipy.sparse.linalg._svdp import _svdp  # 导入 SciPy 稀疏矩阵的 SVD 函数
from scipy.sparse import csr_matrix, csc_matrix  # 从 SciPy 稀疏矩阵模块中导入 csr_matrix 和 csc_matrix 类型


# dtype_flavour 到 tolerance 的映射
TOLS = {
    np.float32: 1e-4,
    np.float64: 1e-8,
    np.complex64: 1e-4,
    np.complex128: 1e-8,
}


def is_complex_type(dtype):
    """检查数据类型是否为复数类型"""
    return np.dtype(dtype).kind == "c"


_dtypes = []
for dtype_flavour in TOLS.keys():
    marks = []
    if is_complex_type(dtype_flavour):
        marks = [pytest.mark.slow]  # 如果是复数类型，则标记为慢速测试
    _dtypes.append(pytest.param(dtype_flavour, marks=marks,
                                id=dtype_flavour.__name__))  # 将每个 dtype 添加到 pytest 参数化列表中，并附带标记
_dtypes = tuple(_dtypes)  # 将参数化列表转换为元组，用于测试


def generate_matrix(constructor, n, m, f,
                    dtype=float, rseed=0, **kwargs):
    """生成一个随机稀疏矩阵"""
    rng = np.random.RandomState(rseed)
    if is_complex_type(dtype):
        M = (- 5 + 10 * rng.rand(n, m)
             - 5j + 10j * rng.rand(n, m)).astype(dtype)  # 如果是复数类型，生成复数矩阵
    else:
        M = (-5 + 10 * rng.rand(n, m)).astype(dtype)  # 否则，生成实数矩阵
    M[M.real > 10 * f - 5] = 0  # 将矩阵中大于阈值的实部置零
    return constructor(M, **kwargs)  # 返回使用指定构造函数构建的矩阵


def assert_orthogonal(u1, u2, rtol, atol):
    """检查前 k 行的 u1 和 u2 是否正交"""
    A = abs(np.dot(u1.conj().T, u2))  # 计算共轭转置 u1 与 u2 的点积的绝对值
    assert_allclose(A, np.eye(u1.shape[1], u2.shape[1]), rtol=rtol, atol=atol)  # 使用断言检查是否接近单位矩阵


def check_svdp(n, m, constructor, dtype, k, irl_mode, which, f=0.8):
    tol = TOLS[dtype]  # 获取指定数据类型的容差值

    M = generate_matrix(np.asarray, n, m, f, dtype)  # 生成随机稀疏矩阵 M
    Msp = constructor(M)  # 使用给定的构造函数构建稀疏矩阵 Msp

    u1, sigma1, vt1 = np.linalg.svd(M, full_matrices=False)  # 对 M 进行全奇异值分解
    u2, sigma2, vt2, _ = _svdp(Msp, k=k, which=which, irl_mode=irl_mode,
                               tol=tol)  # 对 Msp 进行稀疏 SVD

    # 检查 which 参数
    if which.upper() == 'SM':
        u1 = np.roll(u1, k, 1)  # 如果 which 是 'SM'，对 u1 进行行滚动操作
        vt1 = np.roll(vt1, k, 0)  # 对 vt1 进行列滚动操作
        sigma1 = np.roll(sigma1, k)  # 对 sigma1 进行滚动操作

    # 检查奇异值是否一致
    assert_allclose(sigma1[:k], sigma2, rtol=tol, atol=tol)

    # 检查奇异向量是否正交
    assert_orthogonal(u1, u2, rtol=tol, atol=tol)
    assert_orthogonal(vt1.T, vt2.T, rtol=tol, atol=tol)


@pytest.mark.parametrize('ctor', (np.array, csr_matrix, csc_matrix))
@pytest.mark.parametrize('dtype', _dtypes)
@pytest.mark.parametrize('irl', (True, False))
@pytest.mark.parametrize('which', ('LM', 'SM'))
def test_svdp(ctor, dtype, irl, which):
    """测试稀疏 SVD 函数"""
    np.random.seed(0)
    n, m, k = 10, 20, 3
    if which == 'SM' and not irl:
        message = "`which`='SM' requires irl_mode=True"
        with assert_raises(ValueError, match=message):
            check_svdp(n, m, ctor, dtype, k, irl, which)
    else:
        check_svdp(n, m, ctor, dtype, k, irl, which)


@pytest.mark.xslow
@pytest.mark.parametrize('dtype', _dtypes)
@pytest.mark.parametrize('irl', (False, True))
@pytest.mark.timeout(120)  # 对于复数类型 complex64 超过 60 秒：提前依赖覆盖 64 位 BLAS
def test_examples(dtype, irl):
    """测试示例"""
    # 定义不同数据类型的绝对误差阈值字典，用于不同类型数据的比较
    # complex64 的绝对误差阈值从 1e-4 调整为 1e-3，因为在 BLIS、Netlib 和 MKL+AVX512 上出现测试失败
    # 参见 https://github.com/conda-forge/scipy-feedstock/pull/198#issuecomment-999180432
    atol = {
        np.float32: 1.3e-4,
        np.float64: 1e-9,
        np.complex64: 1e-3,
        np.complex128: 1e-9,
    }[dtype]

    # 获取当前脚本文件的路径前缀
    path_prefix = os.path.dirname(__file__)
    
    # 定义相对路径，指向 PROPACK 2.1 分发的测试矩阵数据文件
    relative_path = "propack_test_data.npz"
    
    # 将路径前缀和相对路径合并，得到完整的文件路径
    filename = os.path.join(path_prefix, relative_path)
    
    # 使用 numpy 加载 .npz 格式的数据文件，允许使用 pickle 序列化
    with np.load(filename, allow_pickle=True) as data:
        # 根据数据类型是否为复数类型，选择加载不同的矩阵数据并转换为指定数据类型
        if is_complex_type(dtype):
            A = data['A_complex'].item().astype(dtype)
        else:
            A = data['A_real'].item().astype(dtype)

    # 设定奇异值分解的截断数目 k，并调用 _svdp 函数进行奇异值分解
    k = 200
    u, s, vh, _ = _svdp(A, k, irl_mode=irl, random_state=0)

    # 对于复数类型的例子，由于存在重复的奇异值，仅检查前部非重复的奇异向量以避免排列问题
    sv_check = 27 if is_complex_type(dtype) else k
    u = u[:, :sv_check]
    vh = vh[:sv_check, :]
    s = s[:sv_check]

    # 检查奇异向量的正交性
    assert_allclose(np.eye(u.shape[1]), u.conj().T @ u, atol=atol)
    assert_allclose(np.eye(vh.shape[0]), vh @ vh.conj().T, atol=atol)

    # 确保使用 np.linalg.svd 和 PROPACK 重构的矩阵之间的差异的范数很小
    u3, s3, vh3 = np.linalg.svd(A.todense())
    u3 = u3[:, :sv_check]
    s3 = s3[:sv_check]
    vh3 = vh3[:sv_check, :]
    A3 = u3 @ np.diag(s3) @ vh3
    recon = u @ np.diag(s) @ vh
    assert_allclose(np.linalg.norm(A3 - recon), 0, atol=atol)
# 使用 pytest.mark.parametrize 装饰器为 test_shifts 函数参数 shifts 和 dtype 进行参数化测试
@pytest.mark.parametrize('shifts', (None, -10, 0, 1, 10, 70))
@pytest.mark.parametrize('dtype', _dtypes[:2])
def test_shifts(shifts, dtype):
    # 设定随机种子以确保可重复性
    np.random.seed(0)
    n, k = 70, 10
    # 生成一个 n x n 的随机数组 A
    A = np.random.random((n, n))
    # 如果 shifts 不为 None 且不满足特定条件，则期望抛出 ValueError 异常
    if shifts is not None and ((shifts < 0) or (k > min(n-1-shifts, n))):
        with pytest.raises(ValueError):
            # 调用 _svdp 函数进行奇异值分解，使用给定的参数
            _svdp(A, k, shifts=shifts, kmax=5*k, irl_mode=True)
    else:
        # 否则，调用 _svdp 函数进行奇异值分解，使用给定的参数
        _svdp(A, k, shifts=shifts, kmax=5*k, irl_mode=True)


# 使用 pytest.mark.slow 标记测试为较慢运行，使用 pytest.mark.xfail 标记为预期失败测试
@pytest.mark.slow
@pytest.mark.xfail()
def test_shifts_accuracy():
    # 设定随机种子以确保可重复性
    np.random.seed(0)
    n, k = 70, 10
    # 生成一个 n x n 的随机数组 A，并将其转换为 float64 类型
    A = np.random.random((n, n)).astype(np.float64)
    # 对 A 进行奇异值分解，分别设置 shifts=None 和 shifts=32，并指定 which 参数为 'SM'，irl_mode 参数为 True
    u1, s1, vt1, _ = _svdp(A, k, shifts=None, which='SM', irl_mode=True)
    u2, s2, vt2, _ = _svdp(A, k, shifts=32, which='SM', irl_mode=True)
    # 断言 s1 和 s2 接近（即数值相等），若不符合预期则标记为测试失败
    assert_allclose(s1, s2)
```