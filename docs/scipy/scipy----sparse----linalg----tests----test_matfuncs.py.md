# `D:\src\scipysrc\scipy\scipy\sparse\linalg\tests\test_matfuncs.py`

```
# 导入模块和包
""" Test functions for scipy.linalg._matfuncs module

"""
import math  # 导入数学模块

import numpy as np  # 导入NumPy库
from numpy import array, eye, exp, random  # 从NumPy库中导入特定函数
from numpy.testing import (  # 从NumPy测试模块中导入多个断言函数
        assert_allclose, assert_, assert_array_almost_equal, assert_equal,
        assert_array_almost_equal_nulp, suppress_warnings)

from scipy.sparse import csc_matrix, csc_array, SparseEfficiencyWarning  # 从SciPy稀疏矩阵模块导入特定类和警告
from scipy.sparse._construct import eye as speye  # 从SciPy稀疏矩阵构造模块导入eye函数并重命名为speye
from scipy.sparse.linalg._matfuncs import (expm, _expm,  # 从SciPy稀疏矩阵线性代数函数模块导入多个函数和类
        ProductOperator, MatrixPowerOperator,
        _onenorm_matrix_power_nnm, matrix_power)
from scipy.sparse._sputils import matrix  # 从SciPy稀疏矩阵工具模块导入matrix函数
from scipy.linalg import logm  # 从SciPy线性代数模块导入logm函数
from scipy.special import factorial, binom  # 从SciPy特殊函数模块导入factorial和binom函数
import scipy.sparse  # 导入SciPy稀疏矩阵模块
import scipy.sparse.linalg  # 导入SciPy稀疏矩阵线性代数模块


def _burkardt_13_power(n, p):
    """
    A helper function for testing matrix functions.

    Parameters
    ----------
    n : integer greater than 1
        Order of the square matrix to be returned.
    p : non-negative integer
        Power of the matrix.

    Returns
    -------
    out : ndarray representing a square matrix
        A Forsythe matrix of order n, raised to the power p.

    """
    # 输入验证。
    if n != int(n) or n < 2:
        raise ValueError('n must be an integer greater than 1')  # 如果n不是大于1的整数，则引发值错误异常
    n = int(n)
    if p != int(p) or p < 0:
        raise ValueError('p must be a non-negative integer')  # 如果p不是非负整数，则引发值错误异常
    p = int(p)

    # 显式构造矩阵。
    a, b = divmod(p, n)
    large = np.power(10.0, -n*a)
    small = large * np.power(10.0, -n)
    return np.diag([large]*(n-b), b) + np.diag([small]*b, b-n)


def test_onenorm_matrix_power_nnm():
    np.random.seed(1234)  # 设置随机种子以确保结果可重现
    for n in range(1, 5):
        for p in range(5):
            M = np.random.random((n, n))  # 创建随机的n x n矩阵M
            Mp = np.linalg.matrix_power(M, p)  # 计算M的p次幂
            observed = _onenorm_matrix_power_nnm(M, p)  # 使用自定义函数计算M的p次幂的1-范数
            expected = np.linalg.norm(Mp, 1)  # 计算M的p次幂的1-范数作为期望值
            assert_allclose(observed, expected)  # 断言自定义函数计算的1-范数与NumPy计算的1-范数接近


def test_matrix_power():
    np.random.seed(1234)  # 设置随机种子以确保结果可重现
    row, col = np.random.randint(0, 4, size=(2, 6))  # 生成随机行和列索引
    data = np.random.random(size=(6,))  # 生成随机数据数组
    Amat = csc_matrix((data, (row, col)), shape=(4, 4))  # 使用随机数据创建稀疏CSC矩阵Amat
    A = csc_array((data, (row, col)), shape=(4, 4))  # 使用随机数据创建稀疏CSC数组A
    Adense = A.toarray()  # 将稀疏矩阵A转换为稠密数组Adense
    for power in (2, 5, 6):
        Apow = matrix_power(A, power).toarray()  # 计算稀疏矩阵A的power次幂并转换为稠密数组
        Amat_pow = (Amat**power).toarray()  # 使用Amat的**操作符计算power次幂并转换为稠密数组
        Adense_pow = np.linalg.matrix_power(Adense, power)  # 计算Adense的power次幂
        assert_allclose(Apow, Adense_pow)  # 断言Apow与Adense_pow接近
        assert_allclose(Apow, Amat_pow)  # 断言Apow与Amat_pow接近


class TestExpM:
    def test_zero_ndarray(self):
        a = array([[0.,0],[0,0]])  # 创建包含零元素的NumPy数组a
        assert_array_almost_equal(expm(a),[[1,0],[0,1]])  # 断言计算expm(a)结果与期望值[[1,0],[0,1]]接近

    def test_zero_sparse(self):
        a = csc_matrix([[0.,0],[0,0]])  # 创建包含零元素的稀疏CSC矩阵a
        assert_array_almost_equal(expm(a).toarray(),[[1,0],[0,1]])  # 断言计算expm(a)并转换为稠密数组的结果与期望值[[1,0],[0,1]]接近

    def test_zero_matrix(self):
        a = matrix([[0.,0],[0,0]])  # 创建包含零元素的matrix对象a
        assert_array_almost_equal(expm(a),[[1,0],[0,1]])  # 断言计算expm(a)结果与期望值[[1,0],[0,1]]接近
    # 测试不同类型和格式的输入矩阵在指数矩阵函数上的表现

    def test_misc_types(self):
        # 使用 expm 函数计算包含单个元素的 numpy 数组的指数矩阵
        A = expm(np.array([[1]]))
        # 检查不同输入格式的数组是否产生相同的结果
        assert_allclose(expm(((1,),)), A)
        assert_allclose(expm([[1]]), A)
        assert_allclose(expm(matrix([[1]])), A)
        assert_allclose(expm(np.array([[1]])), A)
        # 检查使用稀疏格式输入的矩阵是否产生相同的结果
        assert_allclose(expm(csc_matrix([[1]])).toarray(), A)

        # 使用 expm 函数计算包含单个复数元素的 numpy 数组的指数矩阵
        B = expm(np.array([[1j]]))
        # 检查不同输入格式的数组是否产生相同的结果
        assert_allclose(expm(((1j,),)), B)
        assert_allclose(expm([[1j]]), B)
        assert_allclose(expm(matrix([[1j]])), B)
        # 检查使用稀疏格式输入的矩阵是否产生相同的结果
        assert_allclose(expm(csc_matrix([[1j]])).toarray(), B)

    # 测试稀疏双对角矩阵在指数矩阵函数上的表现
    def test_bidiagonal_sparse(self):
        # 创建一个稀疏的复数压缩列格式矩阵
        A = csc_matrix([
            [1, 3, 0],
            [0, 1, 5],
            [0, 0, 2]], dtype=float)
        # 计算指数矩阵的预期结果
        e1 = math.exp(1)
        e2 = math.exp(2)
        expected = np.array([
            [e1, 3*e1, 15*(e2 - 2*e1)],
            [0, e1, 5*(e2 - e1)],
            [0, 0, e2]], dtype=float)
        # 计算稀疏矩阵的指数矩阵并转换为数组形式
        observed = expm(A).toarray()
        # 断言计算结果与预期结果的接近程度
        assert_array_almost_equal(observed, expected)

    # 测试在不同浮点数数据类型下的 Padé 近似情况
    def test_padecases_dtype_float(self):
        # 遍历不同的数据类型和缩放比例
        for dtype in [np.float32, np.float64]:
            for scale in [1e-2, 1e-1, 5e-1, 1, 10]:
                # 创建缩放后的单位矩阵
                A = scale * eye(3, dtype=dtype)
                # 计算观察到的指数矩阵
                observed = expm(A)
                # 计算预期的指数矩阵
                expected = exp(scale, dtype=dtype) * eye(3, dtype=dtype)
                # 断言观察到的结果与预期结果的接近程度
                assert_array_almost_equal_nulp(observed, expected, nulp=100)

    # 测试在不同复数数据类型下的 Padé 近似情况
    def test_padecases_dtype_complex(self):
        # 遍历不同的数据类型和缩放比例
        for dtype in [np.complex64, np.complex128]:
            for scale in [1e-2, 1e-1, 5e-1, 1, 10]:
                # 创建缩放后的单位矩阵
                A = scale * eye(3, dtype=dtype)
                # 计算观察到的指数矩阵
                observed = expm(A)
                # 计算预期的指数矩阵
                expected = exp(scale, dtype=dtype) * eye(3, dtype=dtype)
                # 断言观察到的结果与预期结果的接近程度
                assert_array_almost_equal_nulp(observed, expected, nulp=100)

    # 测试在不同稀疏浮点数数据类型下的 Padé 近似情况
    def test_padecases_dtype_sparse_float(self):
        # 由于 float32 和 complex64 导致 spsolve/UMFpack 出错
        dtype = np.float64
        # 遍历不同的缩放比例
        for scale in [1e-2, 1e-1, 5e-1, 1, 10]:
            # 创建缩放后的稀疏单位矩阵
            a = scale * speye(3, 3, dtype=dtype, format='csc')
            # 计算预期的指数矩阵
            e = exp(scale, dtype=dtype) * eye(3, dtype=dtype)
            # 使用特定警告过滤器，计算精确和非精确单范数的指数矩阵
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
                exact_onenorm = _expm(a, use_exact_onenorm=True).toarray()
                inexact_onenorm = _expm(a, use_exact_onenorm=False).toarray()
            # 断言精确单范数和非精确单范数的计算结果与预期结果的接近程度
            assert_array_almost_equal_nulp(exact_onenorm, e, nulp=100)
            assert_array_almost_equal_nulp(inexact_onenorm, e, nulp=100)

    # 测试在不同稀疏复数数据类型下的 Padé 近似情况
    def test_padecases_dtype_sparse_complex(self):
        # 由于 float32 和 complex64 导致 spsolve/UMFpack 出错
        dtype = np.complex128
        # 遍历不同的缩放比例
        for scale in [1e-2, 1e-1, 5e-1, 1, 10]:
            # 创建缩放后的稀疏单位矩阵
            a = scale * speye(3, 3, dtype=dtype, format='csc')
            # 计算预期的指数矩阵
            e = exp(scale) * eye(3, dtype=dtype)
            # 使用特定警告过滤器，计算指数矩阵并转换为数组形式
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, "Changing the sparsity structure")
                assert_array_almost_equal_nulp(expm(a).toarray(), e, nulp=100)
    def test_logm_consistency(self):
        # 设置随机种子以保证可重复性
        random.seed(1234)
        # 循环不同的数据类型
        for dtype in [np.float64, np.complex128]:
            # 循环不同的矩阵大小
            for n in range(1, 10):
                # 循环不同的尺度因子
                for scale in [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]:
                    # 生成具有给定尺度的矩阵 A
                    A = (eye(n) + random.rand(n, n) * scale).astype(dtype)
                    # 如果矩阵 A 是复数类型，加入虚数部分
                    if np.iscomplexobj(A):
                        A = A + 1j * random.rand(n, n) * scale
                    # 断言 expm(logm(A)) 得到的矩阵与 A 很接近
                    assert_array_almost_equal(expm(logm(A)), A)

    def test_integer_matrix(self):
        # 定义一个整数矩阵 Q
        Q = np.array([
            [-3, 1, 1, 1],
            [1, -3, 1, 1],
            [1, 1, -3, 1],
            [1, 1, 1, -3]])
        # 断言 expm(Q) 与 expm(1.0 * Q) 得到的结果非常接近
        assert_allclose(expm(Q), expm(1.0 * Q))

    def test_integer_matrix_2(self):
        # 检查整数溢出的情况
        Q = np.array([[-500, 500, 0, 0],
                      [0, -550, 360, 190],
                      [0, 630, -630, 0],
                      [0, 0, 0, 0]], dtype=np.int16)
        # 断言 expm(Q) 与 expm(1.0 * Q) 得到的结果非常接近
        assert_allclose(expm(Q), expm(1.0 * Q))

        # 将 Q 转换为稀疏矩阵，并且断言转换后的结果与原始矩阵的 expm 值非常接近
        Q = csc_matrix(Q)
        assert_allclose(expm(Q).toarray(), expm(1.0 * Q).toarray())

    def test_triangularity_perturbation(self):
        # 实验 (1) 来自 Awad H. Al-Mohy 和 Nicholas J. Higham (2012) 的论文
        # 改进的矩阵对数的逆比例缩放和平方算法
        A = np.array([
            [3.2346e-1, 3e4, 3e4, 3e4],
            [0, 3.0089e-1, 3e4, 3e4],
            [0, 0, 3.221e-1, 3e4],
            [0, 0, 0, 3.0744e-1]],
            dtype=float)
        # A_logm 是 A 的对数的预计算结果
        A_logm = np.array([
            [-1.12867982029050462e+00, 9.61418377142025565e+04,
             -4.52485573953179264e+09, 2.92496941103871812e+14],
            [0.00000000000000000e+00, -1.20101052953082288e+00,
             9.63469687211303099e+04, -4.68104828911105442e+09],
            [0.00000000000000000e+00, 0.00000000000000000e+00,
             -1.13289322264498393e+00, 9.53249183094775653e+04],
            [0.00000000000000000e+00, 0.00000000000000000e+00,
             0.00000000000000000e+00, -1.17947533272554850e+00]],
            dtype=float)
        # 断言 expm(A_logm) 得到的矩阵与 A 很接近，指定相对和绝对误差容忍度
        assert_allclose(expm(A_logm), A, rtol=1e-4)

        # 对上三角矩阵 A_logm_perturbed 进行微小扰动
        random.seed(1234)
        tiny = 1e-17
        A_logm_perturbed = A_logm.copy()
        A_logm_perturbed[1, 0] = tiny
        # 忽略警告，计算扰动后的 expm(A_logm_perturbed)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "Ill-conditioned.*")
            A_expm_logm_perturbed = expm(A_logm_perturbed)
        rtol = 1e-4
        atol = 100 * tiny
        # 断言 A_expm_logm_perturbed 与 A 不是非常接近，给定相对和绝对误差容忍度
        assert_(not np.allclose(A_expm_logm_perturbed, A, rtol=rtol, atol=atol))
    def test_burkardt_1(self):
        # This matrix is diagonal.
        # The calculation of the matrix exponential is simple.
        #
        # This is the first of a series of matrix exponential tests
        # collected by John Burkardt from the following sources.
        #
        # Alan Laub,
        # Review of "Linear System Theory" by Joao Hespanha,
        # SIAM Review,
        # Volume 52, Number 4, December 2010, pages 779--781.
        #
        # Cleve Moler and Charles Van Loan,
        # Nineteen Dubious Ways to Compute the Exponential of a Matrix,
        # Twenty-Five Years Later,
        # SIAM Review,
        # Volume 45, Number 1, March 2003, pages 3--49.
        #
        # Cleve Moler,
        # Cleve's Corner: A Balancing Act for the Matrix Exponential,
        # 23 July 2012.
        #
        # Robert Ward,
        # Numerical computation of the matrix exponential
        # with accuracy estimate,
        # SIAM Journal on Numerical Analysis,
        # Volume 14, Number 4, September 1977, pages 600--610.
        
        # Calculate the exponential of 1 and 2
        exp1 = np.exp(1)
        exp2 = np.exp(2)
        
        # Define a 2x2 diagonal matrix A
        A = np.array([
            [1, 0],
            [0, 2],
            ], dtype=float)
        
        # Desired exponential values for the matrix A
        desired = np.array([
            [exp1, 0],
            [0, exp2],
            ], dtype=float)
        
        # Compute the matrix exponential of A
        actual = expm(A)
        
        # Assert that the computed matrix exponential matches the desired values
        assert_allclose(actual, desired)

    def test_burkardt_2(self):
        # This matrix is symmetric.
        # The calculation of the matrix exponential is straightforward.
        
        # Define a 2x2 symmetric matrix A
        A = np.array([
            [1, 3],
            [3, 2],
            ], dtype=float)
        
        # Desired exponential values for the matrix A
        desired = np.array([
            [39.322809708033859, 46.166301438885753],
            [46.166301438885768, 54.711576854329110],
            ], dtype=float)
        
        # Compute the matrix exponential of A
        actual = expm(A)
        
        # Assert that the computed matrix exponential matches the desired values
        assert_allclose(actual, desired)

    def test_burkardt_3(self):
        # This example is due to Laub.
        # This matrix is ill-suited for the Taylor series approach.
        # As powers of A are computed, the entries blow up too quickly.
        
        # Calculate the exponentials of 1 and 39
        exp1 = np.exp(1)
        exp39 = np.exp(39)
        
        # Define a 2x2 matrix A
        A = np.array([
            [0, 1],
            [-39, -40],
            ], dtype=float)
        
        # Desired exponential values for the matrix A
        desired = np.array([
            [
                39/(38*exp1) - 1/(38*exp39),
                -np.expm1(-38) / (38*exp1)],
            [
                39*np.expm1(-38) / (38*exp1),
                -1/(38*exp1) + 39/(38*exp39)],
            ], dtype=float)
        
        # Compute the matrix exponential of A
        actual = expm(A)
        
        # Assert that the computed matrix exponential matches the desired values
        assert_allclose(actual, desired)
    def test_burkardt_4(self):
        # This example is due to Moler and Van Loan.
        # The example will cause problems for the series summation approach,
        # as well as for diagonal Pade approximations.
        # 定义矩阵 A，数据类型为浮点数
        A = np.array([
            [-49, 24],
            [-64, 31],
            ], dtype=float)
        # 定义矩阵 U
        U = np.array([[3, 1], [4, 2]], dtype=float)
        # 定义矩阵 V
        V = np.array([[1, -1/2], [-2, 3/2]], dtype=float)
        # 定义向量 w
        w = np.array([-17, -1], dtype=float)
        # 计算期望结果，对 U 中每个元素乘以 w 中对应位置的指数值，再与 V 相乘
        desired = np.dot(U * np.exp(w), V)
        # 计算实际结果，调用 expm 函数计算 A 的指数
        actual = expm(A)
        # 使用 assert_allclose 检查实际结果和期望结果的近似性
        assert_allclose(actual, desired)

    def test_burkardt_5(self):
        # This example is due to Moler and Van Loan.
        # This matrix is strictly upper triangular
        # All powers of A are zero beyond some (low) limit.
        # This example will cause problems for Pade approximations.
        # 定义严格上三角的矩阵 A
        A = np.array([
            [0, 6, 0, 0],
            [0, 0, 6, 0],
            [0, 0, 0, 6],
            [0, 0, 0, 0],
            ], dtype=float)
        # 定义期望的结果
        desired = np.array([
            [1, 6, 18, 36],
            [0, 1, 6, 18],
            [0, 0, 1, 6],
            [0, 0, 0, 1],
            ], dtype=float)
        # 计算实际结果，调用 expm 函数计算 A 的指数
        actual = expm(A)
        # 使用 assert_allclose 检查实际结果和期望结果的近似性
        assert_allclose(actual, desired)

    def test_burkardt_6(self):
        # This example is due to Moler and Van Loan.
        # This matrix does not have a complete set of eigenvectors.
        # That means the eigenvector approach will fail.
        # 计算 e 的指数值
        exp1 = np.exp(1)
        # 定义矩阵 A
        A = np.array([
            [1, 1],
            [0, 1],
            ], dtype=float)
        # 定义期望的结果
        desired = np.array([
            [exp1, exp1],
            [0, exp1],
            ], dtype=float)
        # 计算实际结果，调用 expm 函数计算 A 的指数
        actual = expm(A)
        # 使用 assert_allclose 检查实际结果和期望结果的近似性
        assert_allclose(actual, desired)

    def test_burkardt_7(self):
        # This example is due to Moler and Van Loan.
        # This matrix is very close to example 5.
        # Mathematically, it has a complete set of eigenvectors.
        # Numerically, however, the calculation will be suspect.
        # 计算 e 的指数值
        exp1 = np.exp(1)
        # 计算机器精度
        eps = np.spacing(1)
        # 定义矩阵 A
        A = np.array([
            [1 + eps, 1],
            [0, 1 - eps],
            ], dtype=float)
        # 定义期望的结果
        desired = np.array([
            [exp1, exp1],
            [0, exp1],
            ], dtype=float)
        # 计算实际结果，调用 expm 函数计算 A 的指数
        actual = expm(A)
        # 使用 assert_allclose 检查实际结果和期望结果的近似性
        assert_allclose(actual, desired)

    def test_burkardt_8(self):
        # This matrix was an example in Wikipedia.
        # 计算 e 的指数值
        exp4 = np.exp(4)
        exp16 = np.exp(16)
        # 定义矩阵 A
        A = np.array([
            [21, 17, 6],
            [-5, -1, -6],
            [4, 4, 16],
            ], dtype=float)
        # 定义期望的结果
        desired = np.array([
            [13*exp16 - exp4, 13*exp16 - 5*exp4, 2*exp16 - 2*exp4],
            [-9*exp16 + exp4, -9*exp16 + 5*exp4, -2*exp16 + 2*exp4],
            [16*exp16, 16*exp16, 4*exp16],
            ], dtype=float) * 0.25
        # 计算实际结果，调用 expm 函数计算 A 的指数
        actual = expm(A)
        # 使用 assert_allclose 检查实际结果和期望结果的近似性
        assert_allclose(actual, desired)
    def test_burkardt_9(self):
        # 这个测试用例展示了对函数 F01ECF 的一个示例矩阵，由 NAG 库提供。
        # 矩阵 A 是一个 4x4 的浮点数数组
        A = np.array([
            [1, 2, 2, 2],
            [3, 1, 1, 2],
            [3, 2, 1, 2],
            [3, 3, 3, 1],
            ], dtype=float)
        # 期望的结果，也是一个 4x4 的浮点数数组
        desired = np.array([
            [740.7038, 610.8500, 542.2743, 549.1753],
            [731.2510, 603.5524, 535.0884, 542.2743],
            [823.7630, 679.4257, 603.5524, 610.8500],
            [998.4355, 823.7630, 731.2510, 740.7038],
            ], dtype=float)
        # 计算实际结果
        actual = expm(A)
        # 使用 assert_allclose 进行实际结果和期望结果的比较
        assert_allclose(actual, desired)

    def test_burkardt_10(self):
        # 这是 Ward 的第一个示例。
        # A 是一个 3x3 的浮点数数组，代表一个有缺陷但非贬义的矩阵
        A = np.array([
            [4, 2, 0],
            [1, 4, 1],
            [1, 1, 4],
            ], dtype=float)
        # 使用 scipy.linalg.eigvals 计算 A 的特征值并排序，期望结果为 (3, 3, 6)
        assert_allclose(sorted(scipy.linalg.eigvals(A)), (3, 3, 6))
        # 期望的结果，也是一个 3x3 的浮点数数组
        desired = np.array([
            [147.8666224463699, 183.7651386463682, 71.79703239999647],
            [127.7810855231823, 183.7651386463682, 91.88256932318415],
            [127.7810855231824, 163.6796017231806, 111.9681062463718],
            ], dtype=float)
        # 计算实际结果
        actual = expm(A)
        # 使用 assert_allclose 进行实际结果和期望结果的比较
        assert_allclose(actual, desired)

    def test_burkardt_11(self):
        # 这是 Ward 的第二个示例。
        # A 是一个对称矩阵，3x3 的浮点数数组
        A = np.array([
            [29.87942128909879, 0.7815750847907159, -2.289519314033932],
            [0.7815750847907159, 25.72656945571064, 8.680737820540137],
            [-2.289519314033932, 8.680737820540137, 34.39400925519054],
            ], dtype=float)
        # 使用 scipy.linalg.eigvalsh 计算 A 的特征值，期望结果为 (20, 30, 40)
        assert_allclose(scipy.linalg.eigvalsh(A), (20, 30, 40))
        # 期望的结果，一个 3x3 的浮点数数组
        desired = np.array([
             [
                 5.496313853692378E+15,
                 -1.823188097200898E+16,
                 -3.047577080858001E+16],
             [
                -1.823188097200899E+16,
                6.060522870222108E+16,
                1.012918429302482E+17],
             [
                -3.047577080858001E+16,
                1.012918429302482E+17,
                1.692944112408493E+17],
            ], dtype=float)
        # 计算实际结果
        actual = expm(A)
        # 使用 assert_allclose 进行实际结果和期望结果的比较
        assert_allclose(actual, desired)

    def test_burkardt_12(self):
        # 这是 Ward 的第三个示例。
        # A 是一个 3x3 的浮点数数组
        A = np.array([
            [-131, 19, 18],
            [-390, 56, 54],
            [-387, 57, 52],
            ], dtype=float)
        # 使用 scipy.linalg.eigvals 计算 A 的特征值并排序，期望结果为 (-20, -2, -1)
        assert_allclose(sorted(scipy.linalg.eigvals(A)), (-20, -2, -1))
        # 期望的结果，一个 3x3 的浮点数数组
        desired = np.array([
            [-1.509644158793135, 0.3678794391096522, 0.1353352811751005],
            [-5.632570799891469, 1.471517758499875, 0.4060058435250609],
            [-4.934938326088363, 1.103638317328798, 0.5413411267617766],
            ], dtype=float)
        # 计算实际结果
        actual = expm(A)
        # 使用 assert_allclose 进行实际结果和期望结果的比较
        assert_allclose(actual, desired)
    def test_burkardt_13(self):
        # Ward的第4个示例。
        # 这是Forsythe矩阵的一个版本。
        # 特征向量问题条件很差。
        # Ward算法在此问题上难以估计其结果的精度。
        #
        # 检查构造这类矩阵的一个实例。
        A4_actual = _burkardt_13_power(4, 1)
        A4_desired = [[0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [1e-4, 0, 0, 0]]
        assert_allclose(A4_actual, A4_desired)
        # 检查几个实例的expm。
        for n in (2, 3, 4, 10):
            # 使用Taylor级数近似expm。
            # 这在这类矩阵族中效果很好，
            # 因为每个矩阵在求和之前，
            # 即使在除以阶乘之前，
            # 其每个条目都是非负的，最大条目为10**(-floor(p/n)*n)。
            k = max(1, int(np.ceil(16/n)))
            desired = np.zeros((n, n), dtype=float)
            for p in range(n*k):
                Ap = _burkardt_13_power(n, p)
                assert_equal(np.min(Ap), 0)
                assert_allclose(np.max(Ap), np.power(10, -np.floor(p/n)*n))
                desired += Ap / factorial(p)
            actual = expm(_burkardt_13_power(n, 1))
            assert_allclose(actual, desired)

    def test_burkardt_14(self):
        # 这是Moler的示例。
        # 这个缩放不良的矩阵导致MATLAB的expm()函数出现问题。
        A = np.array([
            [0, 1e-8, 0],
            [-(2e10 + 4e8/6.), -3, 2e10],
            [200./3., 0, -200./3.],
            ], dtype=float)
        desired = np.array([
            [0.446849468283175, 1.54044157383952e-09, 0.462811453558774],
            [-5743067.77947947, -0.0152830038686819, -4526542.71278401],
            [0.447722977849494, 1.54270484519591e-09, 0.463480648837651],
            ], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_pascal(self):
        # 测试Pascal三角形。
        # 零幂指数，用于触发失败 (gh-8029)

        for scale in [1.0, 1e-3, 1e-6]:
            for n in range(0, 80, 3):
                sc = scale ** np.arange(n, -1, -1)
                if np.any(sc < 1e-300):
                    break

                A = np.diag(np.arange(1, n + 1), -1) * scale
                B = expm(A)

                got = B
                expected = binom(np.arange(n + 1)[:,None],
                                 np.arange(n + 1)[None,:]) * sc[None,:] / sc[:,None]
                atol = 1e-13 * abs(expected).max()
                assert_allclose(got, expected, atol=atol)
    def test_matrix_input(self):
        # 大型 np.matrix 输入应该可以正常工作，gh-5546
        A = np.zeros((200, 200))  # 创建一个 200x200 的零矩阵 A
        A[-1, 0] = 1  # 在矩阵 A 的最后一行第一列位置设置值为 1
        B0 = expm(A)  # 计算矩阵 A 的指数值 B0
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "the matrix subclass.*")  # 忽略 DeprecationWarning 类型的警告
            sup.filter(PendingDeprecationWarning, "the matrix subclass.*")  # 忽略 PendingDeprecationWarning 类型的警告
            B = expm(np.matrix(A))  # 使用 np.matrix 封装 A 后计算其指数值 B
        assert_allclose(B, B0)  # 断言 B 与 B0 接近（即相等）

    def test_exp_sinch_overflow(self):
        # 检查中间步骤的溢出问题是否已修复，gh-11839
        L = np.array([[1.0, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, -0.5, -0.5, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, -0.5, -0.5],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        E0 = expm(-L)  # 计算矩阵 -L 的指数值 E0
        E1 = expm(-2**11 * L)  # 计算矩阵 -2^11 * L 的指数值 E1
        E2 = E0  # 将 E2 初始化为 E0
        for j in range(11):
            E2 = E2 @ E2  # 迭代计算 E2 的平方 11 次

        assert_allclose(E1, E2)  # 断言 E1 与 E2 接近（即相等）
# 定义一个测试类 TestOperators
class TestOperators:

    # 定义测试方法 test_product_operator，用于测试乘积运算符
    def test_product_operator(self):
        # 设定随机数种子为 1234，确保每次生成的随机数一致
        random.seed(1234)
        # 设置矩阵维度 n 和乘积操作的列数 k
        n = 5
        k = 2
        # 设置样本数 nsamples
        nsamples = 10
        # 循环进行多次样本测试
        for i in range(nsamples):
            # 生成随机的 n x n 矩阵 A、B、C 和 n x k 矩阵 D
            A = np.random.randn(n, n)
            B = np.random.randn(n, n)
            C = np.random.randn(n, n)
            D = np.random.randn(n, k)
            # 创建乘积操作对象 op，使用 A、B、C 作为参数
            op = ProductOperator(A, B, C)
            # 断言 op 对象进行矩阵乘积 matmat(D) 的结果与 A.dot(B).dot(C).dot(D) 相近
            assert_allclose(op.matmat(D), A.dot(B).dot(C).dot(D))
            # 断言 op 对象的转置操作 op.T 对矩阵乘积 matmat(D) 的结果与 (A.dot(B).dot(C)).T.dot(D) 相近
            assert_allclose(op.T.matmat(D), (A.dot(B).dot(C)).T.dot(D))

    # 定义测试方法 test_matrix_power_operator，用于测试矩阵幂运算符
    def test_matrix_power_operator(self):
        # 设定随机数种子为 1234，确保每次生成的随机数一致
        random.seed(1234)
        # 设置矩阵维度 n 和乘积操作的列数 k，以及矩阵幂的幂次 p
        n = 5
        k = 2
        p = 3
        # 设置样本数 nsamples
        nsamples = 10
        # 循环进行多次样本测试
        for i in range(nsamples):
            # 生成随机的 n x n 矩阵 A 和 n x k 矩阵 B
            A = np.random.randn(n, n)
            B = np.random.randn(n, k)
            # 创建矩阵幂操作对象 op，使用 A 和幂次 p 作为参数
            op = MatrixPowerOperator(A, p)
            # 断言 op 对象进行矩阵乘积 matmat(B) 的结果与 np.linalg.matrix_power(A, p).dot(B) 相近
            assert_allclose(op.matmat(B), np.linalg.matrix_power(A, p).dot(B))
            # 断言 op 对象的转置操作 op.T 对矩阵乘积 matmat(B) 的结果与 np.linalg.matrix_power(A, p).T.dot(B) 相近
            assert_allclose(op.T.matmat(B), np.linalg.matrix_power(A, p).T.dot(B))
```