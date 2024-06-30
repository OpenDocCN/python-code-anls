# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_matfuncs.py`

```
# 导入必要的模块和库
import random  # 导入随机数模块
import functools  # 导入函数工具模块

import numpy as np  # 导入 NumPy 库，并使用 np 别名
from numpy import array, identity, dot, sqrt  # 从 NumPy 导入特定函数
from numpy.testing import (assert_array_almost_equal, assert_allclose, assert_,  # 导入测试函数
                           assert_array_less, assert_array_equal, assert_warns)
import pytest  # 导入 pytest 测试框架

import scipy.linalg  # 导入 SciPy 线性代数模块
from scipy.linalg import (funm, signm, logm, sqrtm, fractional_matrix_power,  # 从 SciPy 线性代数模块导入特定函数
                          expm, expm_frechet, expm_cond, norm, khatri_rao)
from scipy.linalg import _matfuncs_inv_ssq  # 导入 SciPy 线性代数模块的私有函数
from scipy.linalg._matfuncs import pick_pade_structure  # 导入 SciPy 线性代数模块的私有函数
import scipy.linalg._expm_frechet  # 导入 SciPy 线性代数模块的私有模块

from scipy.optimize import minimize  # 导入 minimize 函数


def _get_al_mohy_higham_2012_experiment_1():
    """
    Return the test matrix from Experiment (1) of [1]_.

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)
           "Improved Inverse Scaling and Squaring Algorithms
           for the Matrix Logarithm."
           SIAM Journal on Scientific Computing, 34 (4). C152-C169.
           ISSN 1095-7197

    """
    # 定义并返回实验(1)中的测试矩阵
    A = np.array([
        [3.2346e-1, 3e4, 3e4, 3e4],
        [0, 3.0089e-1, 3e4, 3e4],
        [0, 0, 3.2210e-1, 3e4],
        [0, 0, 0, 3.0744e-1]], dtype=float)
    return A


class TestSignM:

    def test_nils(self):
        # 定义一个测试用例，验证 signm 函数对给定数组的运算结果
        a = array([[29.2, -24.2, 69.5, 49.8, 7.],
                   [-9.2, 5.2, -18., -16.8, -2.],
                   [-10., 6., -20., -18., -2.],
                   [-9.6, 9.6, -25.5, -15.4, -2.],
                   [9.8, -4.8, 18., 18.2, 2.]])
        cr = array([[11.94933333,-2.24533333,15.31733333,21.65333333,-2.24533333],
                    [-3.84266667,0.49866667,-4.59066667,-7.18666667,0.49866667],
                    [-4.08,0.56,-4.92,-7.6,0.56],
                    [-4.03466667,1.04266667,-5.59866667,-7.02666667,1.04266667],
                    [4.15733333,-0.50133333,4.90933333,7.81333333,-0.50133333]])
        r = signm(a)
        assert_array_almost_equal(r, cr)  # 断言计算结果与预期结果的接近程度

    def test_defective1(self):
        # 定义一个测试用例，验证对于一个特定的瑞利矩阵，signm 函数的行为
        a = array([[0.0,1,0,0],[1,0,1,0],[0,0,0,1],[0,0,1,0]])
        signm(a, disp=False)
        # XXX: 什么是正确的结果？

    def test_defective2(self):
        # 定义一个测试用例，验证对于一个特定的矩阵，signm 函数的行为
        a = array((
            [29.2,-24.2,69.5,49.8,7.0],
            [-9.2,5.2,-18.0,-16.8,-2.0],
            [-10.0,6.0,-20.0,-18.0,-2.0],
            [-9.6,9.6,-25.5,-15.4,-2.0],
            [9.8,-4.8,18.0,18.2,2.0]))
        signm(a, disp=False)
        # XXX: 什么是正确的结果？

    def test_defective3(self):
        # 定义一个测试用例，验证对于一个特定的矩阵，signm 函数的行为
        a = array([[-2., 25., 0., 0., 0., 0., 0.],
                   [0., -3., 10., 3., 3., 3., 0.],
                   [0., 0., 2., 15., 3., 3., 0.],
                   [0., 0., 0., 0., 15., 3., 0.],
                   [0., 0., 0., 0., 3., 10., 0.],
                   [0., 0., 0., 0., 0., -2., 25.],
                   [0., 0., 0., 0., 0., 0., -3.]])
        signm(a, disp=False)
        # XXX: 什么是正确的结果？
class TestLogM:

    def test_nils(self):
        # 创建一个7x7的数组，表示一个特定的矩阵
        a = array([[-2., 25., 0., 0., 0., 0., 0.],
                   [0., -3., 10., 3., 3., 3., 0.],
                   [0., 0., 2., 15., 3., 3., 0.],
                   [0., 0., 0., 0., 15., 3., 0.],
                   [0., 0., 0., 0., 3., 10., 0.],
                   [0., 0., 0., 0., 0., -2., 25.],
                   [0., 0., 0., 0., 0., 0., -3.]])
        # 根据矩阵a计算m
        m = (identity(7)*3.1+0j)-a
        # 计算m的对数矩阵，不显示输出
        logm(m, disp=False)
        #XXX: 正确的结果是什么？

    def test_al_mohy_higham_2012_experiment_1_logm(self):
        # logm成功完成了回路旅行。
        # 注意，回路旅行的expm部分条件较差。
        # 获取实验1的特定矩阵
        A = _get_al_mohy_higham_2012_experiment_1()
        # 计算A的对数矩阵和信息，不显示输出
        A_logm, info = logm(A, disp=False)
        # 计算对数矩阵的指数矩阵，与A进行比较
        A_round_trip = expm(A_logm)
        # 检查A的回路旅行是否满足特定的数值容差
        assert_allclose(A_round_trip, A, rtol=5e-5, atol=1e-14)

    def test_al_mohy_higham_2012_experiment_1_funm_log(self):
        # 使用np.log的原始funm未能完成回路旅行。
        # 注意，回路旅行的expm部分条件较差。
        # 获取实验1的特定矩阵
        A = _get_al_mohy_higham_2012_experiment_1()
        # 计算A的funm(np.log)和信息，不显示输出
        A_funm_log, info = funm(A, np.log, disp=False)
        # 计算funm(np.log)的指数矩阵，与A进行比较
        A_round_trip = expm(A_funm_log)
        # 检查A的回路旅行是否不满足特定的数值容差
        assert_(not np.allclose(A_round_trip, A, rtol=1e-5, atol=1e-14))

    def test_round_trip_random_float(self):
        np.random.seed(1234)
        for n in range(1, 6):
            M_unscaled = np.random.randn(n, n)
            for scale in np.logspace(-4, 4, 9):
                M = M_unscaled * scale

                # 特征值与分支切割相关。
                W = np.linalg.eigvals(M)
                err_msg = f'M:{M} eivals:{W}'

                # 检查sqrtm的回路旅行，因为它在logm中使用。
                M_sqrtm, info = sqrtm(M, disp=False)
                M_sqrtm_round_trip = M_sqrtm.dot(M_sqrtm)
                assert_allclose(M_sqrtm_round_trip, M)

                # 检查logm的回路旅行。
                M_logm, info = logm(M, disp=False)
                M_logm_round_trip = expm(M_logm)
                assert_allclose(M_logm_round_trip, M, err_msg=err_msg)

    def test_round_trip_random_complex(self):
        np.random.seed(1234)
        for n in range(1, 6):
            M_unscaled = np.random.randn(n, n) + 1j * np.random.randn(n, n)
            for scale in np.logspace(-4, 4, 9):
                M = M_unscaled * scale
                # 计算M的对数矩阵和信息，不显示输出
                M_logm, info = logm(M, disp=False)
                # 计算对数矩阵的指数矩阵，与M进行比较
                M_round_trip = expm(M_logm)
                # 检查M的回路旅行是否满足特定的数值容差
                assert_allclose(M_round_trip, M)
    def test_logm_type_preservation_and_conversion(self):
        # The logm matrix function should preserve the type of a matrix
        # whose eigenvalues are positive with zero imaginary part.
        # Test this preservation for variously structured matrices.
        complex_dtype_chars = ('F', 'D', 'G')
        for matrix_as_list in (
                [[1, 0], [0, 1]],           # Matrix with all real eigenvalues
                [[1, 0], [1, 1]],           # Matrix with real eigenvalues, not all distinct
                [[2, 1], [1, 1]],           # Matrix with all real eigenvalues, not all distinct
                [[2, 3], [1, 2]]):          # Matrix with all real eigenvalues, not all distinct

            # check that the spectrum has the expected properties
            W = scipy.linalg.eigvals(matrix_as_list)
            assert_(not any(w.imag or w.real < 0 for w in W))

            # check float type preservation
            A = np.array(matrix_as_list, dtype=float)
            A_logm, info = logm(A, disp=False)
            assert_(A_logm.dtype.char not in complex_dtype_chars)

            # check complex type preservation
            A = np.array(matrix_as_list, dtype=complex)
            A_logm, info = logm(A, disp=False)
            assert_(A_logm.dtype.char in complex_dtype_chars)

            # check float->complex type conversion for the matrix negation
            A = -np.array(matrix_as_list, dtype=float)
            A_logm, info = logm(A, disp=False)
            assert_(A_logm.dtype.char in complex_dtype_chars)

    def test_complex_spectrum_real_logm(self):
        # This matrix has complex eigenvalues and real logm.
        # Its output dtype depends on its input dtype.
        M = [[1, 1, 2], [2, 1, 1], [1, 2, 1]]
        for dt in float, complex:
            X = np.array(M, dtype=dt)
            w = scipy.linalg.eigvals(X)
            assert_(1e-2 < np.absolute(w.imag).sum())
            Y, info = logm(X, disp=False)
            assert_(np.issubdtype(Y.dtype, np.inexact))
            assert_allclose(expm(Y), X)

    def test_real_mixed_sign_spectrum(self):
        # These matrices have real eigenvalues with mixed signs.
        # The output logm dtype is complex, regardless of input dtype.
        for M in (
                [[1, 0], [0, -1]],          # Matrix with real eigenvalues of mixed signs
                [[0, 1], [1, 0]]):          # Matrix with all real eigenvalues, not all distinct
            for dt in float, complex:
                A = np.array(M, dtype=dt)
                A_logm, info = logm(A, disp=False)
                assert_(np.issubdtype(A_logm.dtype, np.complexfloating))

    def test_exactly_singular(self):
        A = np.array([[0, 0], [1j, 1j]])
        B = np.asarray([[1, 1], [0, 0]])
        for M in A, A.T, B, B.T:
            expected_warning = _matfuncs_inv_ssq.LogmExactlySingularWarning
            L, info = assert_warns(expected_warning, logm, M, disp=False)
            E = expm(L)
            assert_allclose(E, M, atol=1e-14)

    def test_nearly_singular(self):
        M = np.array([[1e-100]])
        expected_warning = _matfuncs_inv_ssq.LogmNearlySingularWarning
        L, info = assert_warns(expected_warning, logm, M, disp=False)
        E = expm(L)
        assert_allclose(E, M, atol=1e-14)
    def test_opposite_sign_complex_eigenvalues(self):
        # 用于测试具有相反符号复特征值的情况
        E = [[0, 1], [-1, 0]]  # 定义一个包含特定复特征值的矩阵 E
        L = [[0, np.pi*0.5], [-np.pi*0.5, 0]]  # 对应的对数矩阵 L
        assert_allclose(expm(L), E, atol=1e-14)  # 使用 expm 函数计算 L 的指数，并断言其结果与 E 相近
        assert_allclose(logm(E), L, atol=1e-14)  # 使用 logm 函数计算 E 的对数，并断言其结果与 L 相近

        E = [[1j, 4], [0, -1j]]  # 另一个复特征值矩阵 E
        L = [[1j*np.pi*0.5, 2*np.pi], [0, -1j*np.pi*0.5]]  # 对应的对数矩阵 L
        assert_allclose(expm(L), E, atol=1e-14)  # 同样使用 expm 函数进行断言
        assert_allclose(logm(E), L, atol=1e-14)  # 同样使用 logm 函数进行断言

        E = [[1j, 0], [0, -1j]]  # 第三个复特征值矩阵 E
        L = [[1j*np.pi*0.5, 0], [0, -1j*np.pi*0.5]]  # 对应的对数矩阵 L
        assert_allclose(expm(L), E, atol=1e-14)  # 同样使用 expm 函数进行断言
        assert_allclose(logm(E), L, atol=1e-14)  # 同样使用 logm 函数进行断言

    def test_readonly(self):
        n = 5  # 设置矩阵维度为 5
        a = np.ones((n, n)) + np.identity(n)  # 创建一个 5x5 的全 1 矩阵，并加上单位矩阵，构成 a
        a.flags.writeable = False  # 将矩阵 a 设为不可写
        logm(a)  # 对不可写矩阵计算对数

    @pytest.mark.xfail(reason="ValueError: attempt to get argmax of an empty sequence")
    @pytest.mark.parametrize('dt', [int, float, np.float32, complex, np.complex64])
    def test_empty(self, dt):
        a = np.empty((0, 0), dtype=dt)  # 创建一个空的 dtype 类型为 dt 的数组 a
        log_a = logm(a)  # 对空数组计算对数
        a0 = np.eye(2, dtype=dt)  # 创建一个 2x2 的单位数组 a0，dtype 为 dt
        log_a0 = logm(a0)  # 对单位数组计算对数

        assert log_a.shape == (0, 0)  # 断言空数组的对数形状为 (0, 0)
        assert log_a.dtype == log_a0.dtype  # 断言空数组的对数的数据类型与单位数组对数的数据类型相同
class TestSqrtM:
    # 测试随机浮点数的平方根运算是否正确
    def test_round_trip_random_float(self):
        np.random.seed(1234)
        # 对不同大小的随机浮点数矩阵进行测试
        for n in range(1, 6):
            M_unscaled = np.random.randn(n, n)
            # 对不同的比例因子进行测试
            for scale in np.logspace(-4, 4, 9):
                M = M_unscaled * scale
                # 计算矩阵的平方根和相关信息
                M_sqrtm, info = sqrtm(M, disp=False)
                # 计算平方根后的矩阵再次相乘，应当接近原始矩阵
                M_sqrtm_round_trip = M_sqrtm.dot(M_sqrtm)
                # 断言计算结果与原始矩阵的接近程度
                assert_allclose(M_sqrtm_round_trip, M)

    # 测试随机复数矩阵的平方根运算是否正确
    def test_round_trip_random_complex(self):
        np.random.seed(1234)
        # 对不同大小的随机复数矩阵进行测试
        for n in range(1, 6):
            M_unscaled = np.random.randn(n, n) + 1j * np.random.randn(n, n)
            # 对不同的比例因子进行测试
            for scale in np.logspace(-4, 4, 9):
                M = M_unscaled * scale
                # 计算矩阵的平方根和相关信息
                M_sqrtm, info = sqrtm(M, disp=False)
                # 计算平方根后的矩阵再次相乘，应当接近原始矩阵
                M_sqrtm_round_trip = M_sqrtm.dot(M_sqrtm)
                # 断言计算结果与原始矩阵的接近程度
                assert_allclose(M_sqrtm_round_trip, M)

    # 测试一个特殊的矩阵
    def test_bad(self):
        # 参考文献中给出的矩阵数据
        e = 2**-5
        se = sqrt(e)
        a = array([[1.0,0,0,1],
                   [0,e,0,0],
                   [0,0,e,0],
                   [0,0,0,1]])
        sa = array([[1,0,0,0.5],
                    [0,se,0,0],
                    [0,0,se,0],
                    [0,0,0,1]])
        n = a.shape[0]
        # 断言计算后的矩阵是否接近原始矩阵
        assert_array_almost_equal(dot(sa,sa),a)
        # 使用默认的平方根计算方法进行验证
        esa = sqrtm(a, disp=False, blocksize=n)[0]
        # 断言计算后的矩阵是否接近原始矩阵
        assert_array_almost_equal(dot(esa,esa),a)
        # 使用2x2块的平方根计算方法进行验证
        esa = sqrtm(a, disp=False, blocksize=2)[0]
        # 断言计算后的矩阵是否接近原始矩阵
        assert_array_almost_equal(dot(esa,esa),a)

    # 测试平方根运算对矩阵类型的保留和转换
    def test_sqrtm_type_preservation_and_conversion(self):
        # 对于特定结构的矩阵，平方根运算应该保留其类型
        complex_dtype_chars = ('F', 'D', 'G')
        for matrix_as_list in (
                [[1, 0], [0, 1]],
                [[1, 0], [1, 1]],
                [[2, 1], [1, 1]],
                [[2, 3], [1, 2]],
                [[1, 1], [1, 1]]):

            # 检查矩阵的特征值是否符合预期的属性
            W = scipy.linalg.eigvals(matrix_as_list)
            assert_(not any(w.imag or w.real < 0 for w in W))

            # 检查对浮点类型的保留
            A = np.array(matrix_as_list, dtype=float)
            A_sqrtm, info = sqrtm(A, disp=False)
            assert_(A_sqrtm.dtype.char not in complex_dtype_chars)

            # 检查对复数类型的保留
            A = np.array(matrix_as_list, dtype=complex)
            A_sqrtm, info = sqrtm(A, disp=False)
            assert_(A_sqrtm.dtype.char in complex_dtype_chars)

            # 检查对浮点数转换为复数的类型转换
            A = -np.array(matrix_as_list, dtype=float)
            A_sqrtm, info = sqrtm(A, disp=False)
            assert_(A_sqrtm.dtype.char in complex_dtype_chars)
    def test_sqrtm_type_conversion_mixed_sign_or_complex_spectrum(self):
        # 定义复数数据类型的字符表示
        complex_dtype_chars = ('F', 'D', 'G')
        # 对每个测试矩阵执行以下操作
        for matrix_as_list in (
                [[1, 0], [0, -1]],          # 第一个测试矩阵
                [[0, 1], [1, 0]],           # 第二个测试矩阵
                [[0, 1, 0], [0, 0, 1], [1, 0, 0]]):  # 第三个测试矩阵

            # 检查矩阵的谱具有预期的属性
            W = scipy.linalg.eigvals(matrix_as_list)
            assert_(any(w.imag or w.real < 0 for w in W))

            # 检查复数到复数的类型转换
            A = np.array(matrix_as_list, dtype=complex)
            A_sqrtm, info = sqrtm(A, disp=False)
            assert_(A_sqrtm.dtype.char in complex_dtype_chars)

            # 检查浮点数到复数的类型转换
            A = np.array(matrix_as_list, dtype=float)
            A_sqrtm, info = sqrtm(A, disp=False)
            assert_(A_sqrtm.dtype.char in complex_dtype_chars)

    def test_blocksizes(self):
        # 确保在块大小不能整除 n 时不出错
        np.random.seed(1234)
        for n in range(1, 8):
            A = np.random.rand(n, n) + 1j*np.random.randn(n, n)
            A_sqrtm_default, info = sqrtm(A, disp=False, blocksize=n)
            assert_allclose(A, np.linalg.matrix_power(A_sqrtm_default, 2))
            for blocksize in range(1, 10):
                A_sqrtm_new, info = sqrtm(A, disp=False, blocksize=blocksize)
                assert_allclose(A_sqrtm_default, A_sqrtm_new)

    def test_al_mohy_higham_2012_experiment_1(self):
        # 对一个棘手的上三角矩阵求其矩阵平方根
        A = _get_al_mohy_higham_2012_experiment_1()
        A_sqrtm, info = sqrtm(A, disp=False)
        A_round_trip = A_sqrtm.dot(A_sqrtm)
        assert_allclose(A_round_trip, A, rtol=1e-5)
        assert_allclose(np.tril(A_round_trip), np.tril(A))

    def test_strict_upper_triangular(self):
        # 这个矩阵没有平方根
        for dt in int, float:
            A = np.array([
                [0, 3, 0, 0],
                [0, 0, 3, 0],
                [0, 0, 0, 3],
                [0, 0, 0, 0]], dtype=dt)
            A_sqrtm, info = sqrtm(A, disp=False)
            assert_(np.isnan(A_sqrtm).all())

    def test_weird_matrix(self):
        # 矩阵 B 存在平方根
        for dt in int, float:
            A = np.array([
                [0, 0, 1],
                [0, 0, 0],
                [0, 1, 0]], dtype=dt)
            B = np.array([
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0]], dtype=dt)
            assert_array_equal(B, A.dot(A))

            # 但 scipy 的 sqrtm 函数找不到它
            B_sqrtm, info = sqrtm(B, disp=False)
            assert_(np.isnan(B_sqrtm).all())

    def test_disp(self):
        np.random.seed(1234)

        # 随机生成一个 3x3 的矩阵 A，并求其平方根
        A = np.random.rand(3, 3)
        B = sqrtm(A, disp=True)
        assert_allclose(B.dot(B), A)
    # 测试函数，验证对具有复数特征值的矩阵的操作
    def test_opposite_sign_complex_eigenvalues(self):
        # 定义一个具有复数特征值的2x2矩阵M
        M = [[2j, 4], [0, -2j]]
        # 定义一个对应的旋转矩阵R
        R = [[1+1j, 2], [0, 1-1j]]
        # 验证矩阵乘法R * R是否接近于M，允许的误差为1e-14
        assert_allclose(np.dot(R, R), M, atol=1e-14)
        # 验证M的平方根矩阵是否接近于R，允许的误差为1e-14
        assert_allclose(sqrtm(M), R, atol=1e-14)

    # 测试函数，验证对称矩阵的平方根计算
    def test_gh4866(self):
        # 定义一个对称矩阵M
        M = np.array([[1, 0, 0, 1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [1, 0, 0, 1]])
        # 定义其平方根矩阵R
        R = np.array([[sqrt(0.5), 0, 0, sqrt(0.5)],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [sqrt(0.5), 0, 0, sqrt(0.5)]])
        # 验证矩阵乘法R * R是否接近于M，允许的误差为1e-14
        assert_allclose(np.dot(R, R), M, atol=1e-14)
        # 验证M的平方根矩阵是否接近于R，允许的误差为1e-14
        assert_allclose(sqrtm(M), R, atol=1e-14)

    # 测试函数，验证对角矩阵的平方根计算
    def test_gh5336(self):
        # 定义一个对角矩阵M
        M = np.diag([2, 1, 0])
        # 定义其平方根矩阵R
        R = np.diag([sqrt(2), 1, 0])
        # 验证矩阵乘法R * R是否接近于M，允许的误差为1e-14
        assert_allclose(np.dot(R, R), M, atol=1e-14)
        # 验证M的平方根矩阵是否接近于R，允许的误差为1e-14
        assert_allclose(sqrtm(M), R, atol=1e-14)

    # 测试函数，验证全零矩阵的平方根计算
    def test_gh7839(self):
        # 定义一个全零矩阵M
        M = np.zeros((2, 2))
        # 定义其平方根矩阵R
        R = np.zeros((2, 2))
        # 验证矩阵乘法R * R是否接近于M，允许的误差为1e-14
        assert_allclose(np.dot(R, R), M, atol=1e-14)
        # 验证M的平方根矩阵是否接近于R，允许的误差为1e-14
        assert_allclose(sqrtm(M), R, atol=1e-14)

    # 标记为预期失败的测试函数，验证填充矩阵的平方根计算
    @pytest.mark.xfail(reason="failing on macOS after gh-20212")
    def test_gh17918(self):
        # 定义一个填充元素为0.94的19x19空矩阵M
        M = np.empty((19, 19))
        M.fill(0.94)
        np.fill_diagonal(M, 1)
        # 验证M的平方根矩阵是否为实数类型对象
        assert np.isrealobj(sqrtm(M))

    # 测试函数，验证不同数据类型的输入对平方根计算后的数据类型保留
    def test_data_size_preservation_uint_in_float_out(self):
        # 定义不同数据类型的全零矩阵M，并验证其平方根矩阵数据类型为期望的浮点数类型
        M = np.zeros((10, 10), dtype=np.uint8)
        assert sqrtm(M).dtype == np.float16
        M = np.zeros((10, 10), dtype=np.uint16)
        assert sqrtm(M).dtype == np.float16
        M = np.zeros((10, 10), dtype=np.uint32)
        assert sqrtm(M).dtype == np.float32
        M = np.zeros((10, 10), dtype=np.uint64)
        assert sqrtm(M).dtype == np.float64

    # 测试函数，验证不同数据类型的输入对平方根计算后的数据类型保留
    def test_data_size_preservation_int_in_float_out(self):
        # 定义不同数据类型的全零矩阵M，并验证其平方根矩阵数据类型为期望的浮点数类型
        M = np.zeros((10, 10), dtype=np.int8)
        assert sqrtm(M).dtype == np.float16
        M = np.zeros((10, 10), dtype=np.int16)
        assert sqrtm(M).dtype == np.float16
        M = np.zeros((10, 10), dtype=np.int32)
        assert sqrtm(M).dtype == np.float32
        M = np.zeros((10, 10), dtype=np.int64)
        assert sqrtm(M).dtype == np.float64

    # 测试函数，验证不同数据类型的输入对平方根计算后的数据类型保留
    def test_data_size_preservation_int_in_comp_out(self):
        # 定义不同数据类型的矩阵M，并验证其平方根矩阵数据类型为期望的复数类型
        M = np.array([[2, 4], [0, -2]], dtype=np.int8)
        assert sqrtm(M).dtype == np.complex64
        M = np.array([[2, 4], [0, -2]], dtype=np.int16)
        assert sqrtm(M).dtype == np.complex64
        M = np.array([[2, 4], [0, -2]], dtype=np.int32)
        assert sqrtm(M).dtype == np.complex64
        M = np.array([[2, 4], [0, -2]], dtype=np.int64)
        assert sqrtm(M).dtype == np.complex128
    # 测试数据大小的保留：输入为浮点数，输出也为浮点数
    def test_data_size_preservation_float_in_float_out(self):
        # 创建一个尺寸为 10x10 的浮点16位零矩阵
        M = np.zeros((10, 10), dtype=np.float16)
        # 断言求解 M 的平方根的数据类型为浮点16位
        assert sqrtm(M).dtype == np.float16

        # 创建一个尺寸为 10x10 的浮点32位零矩阵
        M = np.zeros((10, 10), dtype=np.float32)
        # 断言求解 M 的平方根的数据类型为浮点32位
        assert sqrtm(M).dtype == np.float32

        # 创建一个尺寸为 10x10 的浮点64位零矩阵
        M = np.zeros((10, 10), dtype=np.float64)
        # 断言求解 M 的平方根的数据类型为浮点64位
        assert sqrtm(M).dtype == np.float64

        # 如果 numpy 支持浮点128位
        if hasattr(np, 'float128'):
            # 创建一个尺寸为 10x10 的浮点128位零矩阵
            M = np.zeros((10, 10), dtype=np.float128)
            # 断言求解 M 的平方根的数据类型为浮点128位
            assert sqrtm(M).dtype == np.float128

    # 测试数据大小的保留：输入为浮点数，输出为复数
    def test_data_size_preservation_float_in_comp_out(self):
        # 创建一个 2x2 的浮点16位矩阵
        M = np.array([[2, 4], [0, -2]], dtype=np.float16)
        # 断言求解 M 的平方根的数据类型为复数64位
        assert sqrtm(M).dtype == np.complex64

        # 创建一个 2x2 的浮点32位矩阵
        M = np.array([[2, 4], [0, -2]], dtype=np.float32)
        # 断言求解 M 的平方根的数据类型为复数64位
        assert sqrtm(M).dtype == np.complex64

        # 创建一个 2x2 的浮点64位矩阵
        M = np.array([[2, 4], [0, -2]], dtype=np.float64)
        # 断言求解 M 的平方根的数据类型为复数128位
        assert sqrtm(M).dtype == np.complex128

        # 如果 numpy 支持浮点128位和复数256位
        if hasattr(np, 'float128') and hasattr(np, 'complex256'):
            # 创建一个 2x2 的浮点128位矩阵
            M = np.array([[2, 4], [0, -2]], dtype=np.float128)
            # 断言求解 M 的平方根的数据类型为复数256位
            assert sqrtm(M).dtype == np.complex256

    # 测试数据大小的保留：输入为复数，输出为复数
    def test_data_size_preservation_comp_in_comp_out(self):
        # 创建一个 2x2 的复数64位矩阵
        M = np.array([[2j, 4], [0, -2j]], dtype=np.complex64)
        # 断言求解 M 的平方根的数据类型为复数128位
        assert sqrtm(M).dtype == np.complex128

        # 如果 numpy 支持复数256位
        if hasattr(np, 'complex256'):
            # 创建一个 2x2 的复数128位矩阵
            M = np.array([[2j, 4], [0, -2j]], dtype=np.complex128)
            # 断言求解 M 的平方根的数据类型为复数256位
            assert sqrtm(M).dtype == np.complex256

            # 创建一个 2x2 的复数256位矩阵
            M = np.array([[2j, 4], [0, -2j]], dtype=np.complex256)
            # 断言求解 M 的平方根的数据类型为复数256位
            assert sqrtm(M).dtype == np.complex256

    # 测试空矩阵的情况
    @pytest.mark.parametrize('dt', [int, float, np.float32, complex, np.complex64])
    def test_empty(self, dt):
        # 创建一个空的 dtype 类型的矩阵
        a = np.empty((0, 0), dtype=dt)
        # 计算其平方根
        s = sqrtm(a)
        # 创建一个单位矩阵，与 a 具有相同的 dtype 类型
        a0 = np.eye(2, dtype=dt)
        # 计算其平方根
        s0 = sqrtm(a0)

        # 断言 s 的形状为 (0, 0)
        assert s.shape == (0, 0)
        # 断言 s 的数据类型与 s0 相同
        assert s.dtype == s0.dtype
class TestFractionalMatrixPower:
    # 定义一个测试类 TestFractionalMatrixPower

    def test_round_trip_random_complex(self):
        # 测试复数矩阵的往返计算
        np.random.seed(1234)
        # 设置随机种子以确保可重复性
        for p in range(1, 5):
            # 对于指数 p 从 1 到 4
            for n in range(1, 5):
                # 对于矩阵大小 n 从 1 到 4
                M_unscaled = np.random.randn(n, n) + 1j * np.random.randn(n, n)
                # 生成一个随机的复数矩阵 M_unscaled
                for scale in np.logspace(-4, 4, 9):
                    # 对于缩放因子 scale 在 10^-4 到 10^4 之间的对数空间内
                    M = M_unscaled * scale
                    # 缩放矩阵 M_unscaled 到 M
                    M_root = fractional_matrix_power(M, 1/p)
                    # 计算 M 的 p 次分数幂 M_root
                    M_round_trip = np.linalg.matrix_power(M_root, p)
                    # 计算 M_root 的 p 次幂 M_round_trip
                    assert_allclose(M_round_trip, M)
                    # 断言 M_round_trip 等于原始矩阵 M

    def test_round_trip_random_float(self):
        # 测试浮点数矩阵的往返计算
        # 这个测试比较麻烦，因为它可能触发分支切割;
        # 当矩阵具有一个没有虚部且带有一个实数负部分的特征值时，就会发生这种情况，
        # 这意味着主分支不存在。
        np.random.seed(1234)
        # 设置随机种子以确保可重复性
        for p in range(1, 5):
            # 对于指数 p 从 1 到 4
            for n in range(1, 5):
                # 对于矩阵大小 n 从 1 到 4
                M_unscaled = np.random.randn(n, n)
                # 生成一个随机的浮点数矩阵 M_unscaled
                for scale in np.logspace(-4, 4, 9):
                    # 对于缩放因子 scale 在 10^-4 到 10^4 之间的对数空间内
                    M = M_unscaled * scale
                    # 缩放矩阵 M_unscaled 到 M
                    M_root = fractional_matrix_power(M, 1/p)
                    # 计算 M 的 p 次分数幂 M_root
                    M_round_trip = np.linalg.matrix_power(M_root, p)
                    # 计算 M_root 的 p 次幂 M_round_trip
                    assert_allclose(M_round_trip, M)
                    # 断言 M_round_trip 等于原始矩阵 M

    def test_larger_abs_fractional_matrix_powers(self):
        # 测试更大的绝对值分数幂矩阵
        np.random.seed(1234)
        # 设置随机种子以确保可重复性
        for n in (2, 3, 5):
            # 对于矩阵大小 n 分别为 2, 3, 5
            for i in range(10):
                # 迭代 10 次
                M = np.random.randn(n, n) + 1j * np.random.randn(n, n)
                # 生成一个随机的复数矩阵 M
                M_one_fifth = fractional_matrix_power(M, 0.2)
                # 计算 M 的 0.2 次分数幂 M_one_fifth
                # 测试往返计算
                M_round_trip = np.linalg.matrix_power(M_one_fifth, 5)
                # 计算 M_one_fifth 的 5 次幂 M_round_trip
                assert_allclose(M, M_round_trip)
                # 断言 M_round_trip 等于原始矩阵 M
                # 测试一个较大的绝对值分数幂
                X = fractional_matrix_power(M, -5.4)
                # 计算 M 的 -5.4 次分数幂 X
                Y = np.linalg.matrix_power(M_one_fifth, -27)
                # 计算 M_one_fifth 的 -27 次幂 Y
                assert_allclose(X, Y)
                # 断言 X 等于 Y
                # 测试另一个较大的绝对值分数幂
                X = fractional_matrix_power(M, 3.8)
                # 计算 M 的 3.8 次分数幂 X
                Y = np.linalg.matrix_power(M_one_fifth, 19)
                # 计算 M_one_fifth 的 19 次幂 Y
                assert_allclose(X, Y)
                # 断言 X 等于 Y
    def test_random_matrices_and_powers(self):
        # 每次独立迭代此模糊测试都会选择随机参数。
        # 它试图覆盖一些边界情况。
        np.random.seed(1234)
        nsamples = 20
        for i in range(nsamples):
            # 随机选择一个矩阵大小和一个随机实数幂。
            n = random.randrange(1, 5)
            p = np.random.randn()

            # 生成一个随机的实数或复数矩阵。
            matrix_scale = np.exp(random.randrange(-4, 5))
            A = np.random.randn(n, n)
            if random.choice((True, False)):
                A = A + 1j * np.random.randn(n, n)
            A = A * matrix_scale

            # 检查几种等效的计算分数矩阵幂的方法。
            # 这些可以比较，因为它们都使用主分支。
            A_power = fractional_matrix_power(A, p)
            A_logm, info = logm(A, disp=False)
            A_power_expm_logm = expm(A_logm * p)
            assert_allclose(A_power, A_power_expm_logm)

    def test_al_mohy_higham_2012_experiment_1(self):
        # 对一个复杂的上三角矩阵进行分数幂测试。
        A = _get_al_mohy_higham_2012_experiment_1()

        # 测试余项矩阵幂。
        A_funm_sqrt, info = funm(A, np.sqrt, disp=False)
        A_sqrtm, info = sqrtm(A, disp=False)
        A_rem_power = _matfuncs_inv_ssq._remainder_matrix_power(A, 0.5)
        A_power = fractional_matrix_power(A, 0.5)
        assert_allclose(A_rem_power, A_power, rtol=1e-11)
        assert_allclose(A_sqrtm, A_power)
        assert_allclose(A_sqrtm, A_funm_sqrt)

        # 测试更多的分数幂。
        for p in (1/2, 5/3):
            A_power = fractional_matrix_power(A, p)
            A_round_trip = fractional_matrix_power(A_power, 1/p)
            assert_allclose(A_round_trip, A, rtol=1e-2)
            assert_allclose(np.tril(A_round_trip, 1), np.tril(A, 1))

    def test_briggs_helper_function(self):
        np.random.seed(1234)
        for a in np.random.randn(10) + 1j * np.random.randn(10):
            for k in range(5):
                x_observed = _matfuncs_inv_ssq._briggs_helper_function(a, k)
                x_expected = a ** np.exp2(-k) - 1
                assert_allclose(x_observed, x_expected)
    def test_type_preservation_and_conversion(self):
        # 测试类型保留和转换
        # fractional_matrix_power 矩阵函数应当保留具有正实部且零虚部特征值的矩阵的类型。
        # 对各种结构的矩阵进行此保留测试。

        # 复数数据类型的字符表示
        complex_dtype_chars = ('F', 'D', 'G')

        for matrix_as_list in (
                [[1, 0], [0, 1]],
                [[1, 0], [1, 1]],
                [[2, 1], [1, 1]],
                [[2, 3], [1, 2]]):

            # 检查特征值具有期望的特性
            W = scipy.linalg.eigvals(matrix_as_list)
            assert_(not any(w.imag or w.real < 0 for w in W))

            # 检查各种正负幂次的类型保留
            # 其绝对值大于和小于1。
            for p in (-2.4, -0.9, 0.2, 3.3):

                # 检查浮点类型的保留
                A = np.array(matrix_as_list, dtype=float)
                A_power = fractional_matrix_power(A, p)
                assert_(A_power.dtype.char not in complex_dtype_chars)

                # 检查复数类型的保留
                A = np.array(matrix_as_list, dtype=complex)
                A_power = fractional_matrix_power(A, p)
                assert_(A_power.dtype.char in complex_dtype_chars)

                # 检查浮点到复数的类型转换（对矩阵取负）
                A = -np.array(matrix_as_list, dtype=float)
                A_power = fractional_matrix_power(A, p)
                assert_(A_power.dtype.char in complex_dtype_chars)

    def test_type_conversion_mixed_sign_or_complex_spectrum(self):
        # 混合符号或复数谱的类型转换测试
        complex_dtype_chars = ('F', 'D', 'G')

        for matrix_as_list in (
                [[1, 0], [0, -1]],
                [[0, 1], [1, 0]],
                [[0, 1, 0], [0, 0, 1], [1, 0, 0]]):

            # 检查特征值具有期望的特性
            W = scipy.linalg.eigvals(matrix_as_list)
            assert_(any(w.imag or w.real < 0 for w in W))

            # 检查各种正负幂次的类型转换
            # 其绝对值大于和小于1。
            for p in (-2.4, -0.9, 0.2, 3.3):

                # 检查复数到复数的类型转换
                A = np.array(matrix_as_list, dtype=complex)
                A_power = fractional_matrix_power(A, p)
                assert_(A_power.dtype.char in complex_dtype_chars)

                # 检查浮点到复数的类型转换
                A = np.array(matrix_as_list, dtype=float)
                A_power = fractional_matrix_power(A, p)
                assert_(A_power.dtype.char in complex_dtype_chars)

    @pytest.mark.xfail(reason='Too unstable across LAPACKs.')
    def test_singular(self):
        # Negative fractional powers do not work with singular matrices.
        # 循环遍历不同的矩阵作为列表，测试负分数幂在奇异矩阵中的表现
        for matrix_as_list in (
                [[0, 0], [0, 0]],             # 2x2零矩阵
                [[1, 1], [1, 1]],             # 2x2全一矩阵
                [[1, 2], [3, 6]],             # 2x2非奇异矩阵
                [[0, 0, 0], [0, 1, 1], [0, -1, 1]]):  # 3x3矩阵

            # Check fractional powers both for float and for complex types.
            # 针对浮点数和复数类型，检查分数幂的表现
            for newtype in (float, complex):
                A = np.array(matrix_as_list, dtype=newtype)
                # 对于负的分数幂，期望结果为NaN
                for p in (-0.7, -0.9, -2.4, -1.3):
                    A_power = fractional_matrix_power(A, p)
                    assert_(np.isnan(A_power).all())
                # 对于正的分数幂，进行幂次计算和逆操作，并期望结果接近原矩阵
                for p in (0.2, 1.43):
                    A_power = fractional_matrix_power(A, p)
                    A_round_trip = fractional_matrix_power(A_power, 1/p)
                    assert_allclose(A_round_trip, A)

    def test_opposite_sign_complex_eigenvalues(self):
        # 定义一个具有相反符号复特征值的矩阵M和其预期的结果R
        M = [[2j, 4], [0, -2j]]
        R = [[1+1j, 2], [0, 1-1j]]
        # 断言矩阵乘积RR与M在给定的tolerance下接近
        assert_allclose(np.dot(R, R), M, atol=1e-14)
        # 对M进行0.5次幂的计算，并断言结果接近R
        assert_allclose(fractional_matrix_power(M, 0.5), R, atol=1e-14)
class TestExpM:
    # 定义测试类 TestExpM

    def test_zero(self):
        # 测试零矩阵的指数函数
        a = array([[0.,0],[0,0]])
        # 创建一个二维数组表示零矩阵
        assert_array_almost_equal(expm(a),[[1,0],[0,1]])
        # 断言计算的指数函数结果与预期的单位矩阵接近

    def test_single_elt(self):
        # 测试单个元素的指数函数
        elt = expm(1)
        # 计算数字 1 的指数函数
        assert_allclose(elt, np.array([[np.e]]))
        # 断言计算的结果接近自然常数 e 的二维数组形式

    @pytest.mark.parametrize('dt', [int, float, np.float32, complex, np.complex64])
    def test_empty_matrix_input(self, dt):
        # 测试空矩阵输入的指数函数
        # 处理 GitHub issue #11082
        A = np.zeros((0, 0), dtype=dt)
        # 创建一个空的零矩阵，指定数据类型
        result = expm(A)
        # 计算指数函数
        assert result.size == 0
        # 断言结果的大小为零
        assert result.dtype == A.dtype
        # 断言结果的数据类型与输入矩阵的数据类型相同

    def test_2x2_input(self):
        # 测试2x2矩阵输入的指数函数
        E = np.e
        # 定义自然常数 e
        a = array([[1, 4], [1, 1]])
        # 创建一个二维数组表示2x2矩阵
        aa = (E**4 + 1)/(2*E)
        bb = (E**4 - 1)/E
        # 计算预期的指数函数结果中的值
        assert_allclose(expm(a), array([[aa, bb], [bb/4, aa]]))
        # 断言计算的指数函数结果与预期结果接近
        assert expm(a.astype(np.complex64)).dtype.char == 'F'
        # 断言转换为复数类型后的结果数据类型为 'F'
        assert expm(a.astype(np.float32)).dtype.char == 'f'
        # 断言转换为单精度浮点数类型后的结果数据类型为 'f'

    def test_nx2x2_input(self):
        # 测试多个2x2矩阵输入的指数函数
        E = np.e
        # 定义自然常数 e
        # 这些是具有整数特征值的整数矩阵
        a = np.array([[[1, 4], [1, 1]],
                      [[1, 3], [1, -1]],
                      [[1, 3], [4, 5]],
                      [[1, 3], [5, 3]],
                      [[4, 5], [-3, -4]]], order='F')
        # 创建一个三维数组表示多个2x2矩阵，按 Fortran 风格存储
        # 精确结果通过符号计算得到
        a_res = np.array([
                          [[(E**4+1)/(2*E), (E**4-1)/E],
                           [(E**4-1)/4/E, (E**4+1)/(2*E)]],
                          [[1/(4*E**2)+(3*E**2)/4, (3*E**2)/4-3/(4*E**2)],
                           [E**2/4-1/(4*E**2), 3/(4*E**2)+E**2/4]],
                          [[3/(4*E)+E**7/4, -3/(8*E)+(3*E**7)/8],
                           [-1/(2*E)+E**7/2, 1/(4*E)+(3*E**7)/4]],
                          [[5/(8*E**2)+(3*E**6)/8, -3/(8*E**2)+(3*E**6)/8],
                           [-5/(8*E**2)+(5*E**6)/8, 3/(8*E**2)+(5*E**6)/8]],
                          [[-3/(2*E)+(5*E)/2, -5/(2*E)+(5*E)/2],
                           [3/(2*E)-(3*E)/2, 5/(2*E)-(3*E)/2]]
                         ])
        # 创建一个三维数组表示预期的指数函数结果
        assert_allclose(expm(a), a_res)
        # 断言计算的指数函数结果与预期结果接近

    def test_readonly(self):
        # 测试只读矩阵输入的指数函数
        n = 7
        a = np.ones((n, n))
        # 创建一个全为 1 的 n x n 矩阵
        a.flags.writeable = False
        # 将矩阵设置为只读
        expm(a)
        # 计算指数函数，预期会引发 ValueError 异常

    @pytest.mark.fail_slow(5)
    def test_gh18086(self):
        # 测试 GitHub issue #18086
        A = np.zeros((400, 400), dtype=float)
        # 创建一个全零的400x400浮点数矩阵
        rng = np.random.default_rng(100)
        # 使用种子值为100创建随机数生成器
        i = rng.integers(0, 399, 500)
        j = rng.integers(0, 399, 500)
        # 生成两个数组，包含500个介于0到399之间的随机整数
        A[i, j] = rng.random(500)
        # 在 A 的随机位置赋值随机浮点数
        # 当 m = 9 时出现问题
        Am = np.empty((5, 400, 400), dtype=float)
        # 创建一个空的5x400x400浮点数数组
        Am[0] = A.copy()
        # 将 A 的副本存储在 Am 的第一个元素中
        m, s = pick_pade_structure(Am)
        # 调用 pick_pade_structure 函数获取结果
        assert m == 9
        # 断言返回的 m 值为 9
        # 检查结果的准确性
        first_res = expm(A)
        # 计算 A 的指数函数
        np.testing.assert_array_almost_equal(logm(first_res), A)
        # 断言计算的对数函数结果与 A 接近
        # 检查结果的一致性
        for i in range(5):
            next_res = expm(A)
            # 计算 A 的指数函数
            np.testing.assert_array_almost_equal(first_res, next_res)
            # 断言首次计算的指数函数结果与后续计算的结果接近
    def test_expm_frechet(self):
        # a test of the basic functionality
        
        # 定义一个4x4的浮点数数组M
        M = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [0, 0, 1, 2],
            [0, 0, 5, 6],
            ], dtype=float)
        
        # 定义一个2x2的浮点数数组A
        A = np.array([
            [1, 2],
            [5, 6],
            ], dtype=float)
        
        # 定义一个2x2的浮点数数组E
        E = np.array([
            [3, 4],
            [7, 8],
            ], dtype=float)
        
        # 计算A的矩阵指数
        expected_expm = scipy.linalg.expm(A)
        
        # 计算M的矩阵指数的左上角2x2子矩阵
        expected_frechet = scipy.linalg.expm(M)[:2, 2:]
        
        # 针对不同的kwargs参数进行循环测试
        for kwargs in ({}, {'method':'SPS'}, {'method':'blockEnlarge'}):
            # 调用expm_frechet函数，获取观察到的矩阵指数和Frechet导数
            observed_expm, observed_frechet = expm_frechet(A, E, **kwargs)
            
            # 断言观察到的矩阵指数与期望的矩阵指数接近
            assert_allclose(expected_expm, observed_expm)
            
            # 断言观察到的Frechet导数与期望的Frechet导数接近
            assert_allclose(expected_frechet, observed_frechet)

    def test_small_norm_expm_frechet(self):
        # methodically test matrices with a range of norms, for better coverage
        
        # 定义一个4x4的浮点数数组M_original
        M_original = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [0, 0, 1, 2],
            [0, 0, 5, 6],
            ], dtype=float)
        
        # 定义一个2x2的浮点数数组A_original
        A_original = np.array([
            [1, 2],
            [5, 6],
            ], dtype=float)
        
        # 定义一个2x2的浮点数数组E_original
        E_original = np.array([
            [3, 4],
            [7, 8],
            ], dtype=float)
        
        # 计算A_original的L1范数
        A_original_norm_1 = scipy.linalg.norm(A_original, 1)
        
        # 选定一组m值列表
        selected_m_list = [1, 3, 5, 7, 9, 11, 13, 15]
        
        # 对选定的m值进行成对迭代
        m_neighbor_pairs = zip(selected_m_list[:-1], selected_m_list[1:])
        
        # 遍历m_neighbor_pairs中的每对ma, mb
        for ma, mb in m_neighbor_pairs:
            # 访问ell_table_61中的值，计算ell_a和ell_b
            ell_a = scipy.linalg._expm_frechet.ell_table_61[ma]
            ell_b = scipy.linalg._expm_frechet.ell_table_61[mb]
            
            # 计算目标的L1范数
            target_norm_1 = 0.5 * (ell_a + ell_b)
            
            # 计算缩放比例
            scale = target_norm_1 / A_original_norm_1
            
            # 根据缩放比例缩放M_original, A_original, E_original
            M = scale * M_original
            A = scale * A_original
            E = scale * E_original
            
            # 计算A的矩阵指数
            expected_expm = scipy.linalg.expm(A)
            
            # 计算M的矩阵指数的左上角2x2子矩阵
            expected_frechet = scipy.linalg.expm(M)[:2, 2:]
            
            # 调用expm_frechet函数，获取观察到的矩阵指数和Frechet导数
            observed_expm, observed_frechet = expm_frechet(A, E)
            
            # 断言观察到的矩阵指数与期望的矩阵指数接近
            assert_allclose(expected_expm, observed_expm)
            
            # 断言观察到的Frechet导数与期望的Frechet导数接近
            assert_allclose(expected_frechet, observed_frechet)
    # 定义一个测试函数，用于模糊测试
    def test_fuzz(self):
        # 尝试一系列不同的随机函数作为输入
        rfuncs = (
                np.random.uniform,
                np.random.normal,
                np.random.standard_cauchy,
                np.random.exponential)
        # 测试次数
        ntests = 100
        # 循环进行测试
        for i in range(ntests):
            # 从随机函数列表中随机选择一个函数
            rfunc = random.choice(rfuncs)
            # 生成目标L1范数为1的随机数
            target_norm_1 = random.expovariate(1.0)
            # 随机生成矩阵的大小
            n = random.randrange(2, 16)
            # 生成原始矩阵A
            A_original = rfunc(size=(n,n))
            # 生成原始矩阵E
            E_original = rfunc(size=(n,n))
            # 计算原始矩阵A的L1范数
            A_original_norm_1 = scipy.linalg.norm(A_original, 1)
            # 计算缩放比例，使得A的L1范数为目标值target_norm_1
            scale = target_norm_1 / A_original_norm_1
            # 缩放矩阵A和E
            A = scale * A_original
            E = scale * E_original
            # 构造增广矩阵M
            M = np.vstack([
                np.hstack([A, E]),
                np.hstack([np.zeros_like(A), A])])
            # 计算预期的矩阵指数函数expm(A)
            expected_expm = scipy.linalg.expm(A)
            # 计算预期的Frechet导数
            expected_frechet = scipy.linalg.expm(M)[:n, n:]
            # 调用被测函数expm_frechet计算观察到的结果
            observed_expm, observed_frechet = expm_frechet(A, E)
            # 断言预期结果与观察结果的接近程度
            assert_allclose(expected_expm, observed_expm, atol=5e-8)
            assert_allclose(expected_frechet, observed_frechet, atol=1e-7)

    # 定义一个测试函数，用于检测特定矩阵情况
    def test_problematic_matrix(self):
        # 这个测试用例揭示了一个已修复的bug
        A = np.array([
                [1.50591997, 1.93537998],
                [0.41203263, 0.23443516],
                ], dtype=float)
        E = np.array([
                [1.87864034, 2.07055038],
                [1.34102727, 0.67341123],
                ], dtype=float)
        # 计算矩阵A的L1范数
        scipy.linalg.norm(A, 1)
        # 调用被测函数expm_frechet，使用不同的方法计算结果
        sps_expm, sps_frechet = expm_frechet(
                A, E, method='SPS')
        blockEnlarge_expm, blockEnlarge_frechet = expm_frechet(
                A, E, method='blockEnlarge')
        # 断言使用不同方法计算的结果接近
        assert_allclose(sps_expm, blockEnlarge_expm)
        assert_allclose(sps_frechet, blockEnlarge_frechet)

    # 标记为慢速测试，并跳过执行，因为该测试有意设计为较慢
    @pytest.mark.slow
    @pytest.mark.skip(reason='this test is deliberately slow')
    def test_medium_matrix(self):
        # 分析这个测试以查看速度差异
        n = 1000
        # 随机生成指数分布的大型矩阵A和E
        A = np.random.exponential(size=(n, n))
        E = np.random.exponential(size=(n, n))
        # 使用不同方法计算expm_frechet函数的结果
        sps_expm, sps_frechet = expm_frechet(
                A, E, method='SPS')
        blockEnlarge_expm, blockEnlarge_frechet = expm_frechet(
                A, E, method='blockEnlarge')
        # 断言使用不同方法计算的结果接近
        assert_allclose(sps_expm, blockEnlarge_expm)
        assert_allclose(sps_frechet, blockEnlarge_frechet)
# 定义一个函数 `_help_expm_cond_search`，用于计算扩展指数函数条件数的负缩放相对误差
def _help_expm_cond_search(A, A_norm, X, X_norm, eps, p):
    # 将参数 p 重塑为与 A 相同的形状
    p = np.reshape(p, A.shape)
    # 计算向量 p 的范数
    p_norm = norm(p)
    # 计算扰动向量，用于构造扩展指数函数的输入矩阵
    perturbation = eps * p * (A_norm / p_norm)
    # 计算扰动后的扩展指数函数结果
    X_prime = expm(A + perturbation)
    # 计算缩放相对误差
    scaled_relative_error = norm(X_prime - X) / (X_norm * eps)
    # 返回负的缩放相对误差作为结果
    return -scaled_relative_error


# 定义一个函数 `_normalized_like`，用于计算两个矩阵 A 和 B 的归一化乘积
def _normalized_like(A, B):
    return A * (scipy.linalg.norm(B) / scipy.linalg.norm(A))


# 定义一个函数 `_relative_error`，用于计算函数 f 在矩阵 A 上加上扰动 perturbation 后的相对误差
def _relative_error(f, A, perturbation):
    # 计算原始输入矩阵 A 的函数值
    X = f(A)
    # 计算扰动后输入矩阵 A + perturbation 的函数值
    X_prime = f(A + perturbation)
    # 计算两者之间的相对误差
    return norm(X_prime - X) / norm(X)


# 定义一个测试类 `TestExpmConditionNumber`
class TestExpmConditionNumber:
    # 定义测试方法 `test_expm_cond_smoke`，用于验证扩展指数函数条件数的基本功能
    def test_expm_cond_smoke(self):
        # 设置随机种子以保证可重复性
        np.random.seed(1234)
        # 循环测试不同维度的随机矩阵 A
        for n in range(1, 4):
            A = np.random.randn(n, n)
            # 计算并检查扩展指数函数条件数 kappa 是否大于 0
            kappa = expm_cond(A)
            assert_array_less(0, kappa)

    # 定义测试方法 `test_expm_bad_condition_number`，用于验证特定矩阵情况下的扩展指数函数条件数
    def test_expm_bad_condition_number(self):
        # 给定一个特定的矩阵 A，其元素以科学计数法表示
        A = np.array([
            [-1.128679820, 9.614183771e4, -4.524855739e9, 2.924969411e14],
            [0, -1.201010529, 9.634696872e4, -4.681048289e9],
            [0, 0, -1.132893222, 9.532491830e4],
            [0, 0, 0, -1.179475332],
            ])
        # 计算并检查扩展指数函数条件数 kappa 是否大于 1e36
        kappa = expm_cond(A)
        assert_array_less(1e36, kappa)

    # 定义测试方法 `test_univariate`，用于验证单变量矩阵的扩展指数函数条件数
    def test_univariate(self):
        # 设置随机种子以保证可重复性
        np.random.seed(12345)
        # 循环测试均匀分布的一维矩阵 A
        for x in np.linspace(-5, 5, num=11):
            A = np.array([[x]])
            # 检查扩展指数函数条件数是否接近于绝对值 abs(x)
            assert_allclose(expm_cond(A), abs(x))
        # 循环测试对数分布的一维矩阵 A
        for x in np.logspace(-2, 2, num=11):
            A = np.array([[x]])
            # 检查扩展指数函数条件数是否接近于绝对值 abs(x)
            assert_allclose(expm_cond(A), abs(x))
        # 循环测试随机正态分布的一维矩阵 A
        for i in range(10):
            A = np.random.randn(1, 1)
            # 检查扩展指数函数条件数是否接近于 A 的绝对值
            assert_allclose(expm_cond(A), np.absolute(A)[0, 0])

    # 标记测试方法为慢速测试
    @pytest.mark.slow
    # 定义一个测试方法，用于检验 expm_cond 函数在条件模糊情况下的表现
    def test_expm_cond_fuzz(self):
        # 设定随机种子以便结果可重复
        np.random.seed(12345)
        # 定义一个小的误差界限
        eps = 1e-5
        # 设定样本数目
        nsamples = 10
        # 对每个样本进行迭代
        for i in range(nsamples):
            # 随机生成一个介于2到4之间的整数，作为矩阵的维度
            n = np.random.randint(2, 5)
            # 随机生成一个 n x n 的矩阵 A
            A = np.random.randn(n, n)
            # 计算矩阵 A 的 Frobenius 范数
            A_norm = scipy.linalg.norm(A)
            # 计算矩阵指数函数 expm(A)
            X = expm(A)
            # 计算矩阵 expm(A) 的 Frobenius 范数
            X_norm = scipy.linalg.norm(X)
            # 计算矩阵 A 的条件数 kappa
            kappa = expm_cond(A)

            # 寻找导致相对误差最大的小扰动
            f = functools.partial(_help_expm_cond_search,
                    A, A_norm, X, X_norm, eps)
            # 初始化猜测值为全1的数组
            guess = np.ones(n*n)
            # 使用 L-BFGS-B 方法最小化函数 f，寻找最优解
            out = minimize(f, guess, method='L-BFGS-B')
            xopt = out.x
            yopt = f(xopt)
            # 计算最优解对应的扰动 p_best
            p_best = eps * _normalized_like(np.reshape(xopt, A.shape), A)
            # 计算相对误差 p_best_relerr
            p_best_relerr = _relative_error(expm, A, p_best)
            # 断言最大相对误差应接近 -yopt * eps
            assert_allclose(p_best_relerr, -yopt * eps)

            # 检查识别出的扰动确实比具有类似范数的随机扰动具有更大的相对误差
            for j in range(5):
                # 生成一个具有范数为 p_best 范数的随机扰动
                p_rand = eps * _normalized_like(np.random.randn(*A.shape), A)
                # 断言 p_best 范数与 p_rand 范数接近
                assert_allclose(norm(p_best), norm(p_rand))
                # 计算随机扰动 p_rand 的相对误差
                p_rand_relerr = _relative_error(expm, A, p_rand)
                # 断言随机扰动的相对误差小于 p_best 的相对误差
                assert_array_less(p_rand_relerr, p_best_relerr)

            # 最大的相对误差不应远远大于 eps 乘以条件数 kappa
            # 当 eps 趋近于零时，它不应大于 kappa
            assert_array_less(p_best_relerr, (1 + 2*eps) * eps * kappa)
    # 定义一个测试类 TestKhatriRao，用于测试 khatri_rao 函数的各种情况
class TestKhatriRao:

    # 测试基本情况：两个已知矩阵的 Khatri-Rao 乘积
    def test_basic(self):
        a = khatri_rao(array([[1, 2], [3, 4]]),
                       array([[5, 6], [7, 8]]))

        # 断言 a 的结果与预期的矩阵相等
        assert_array_equal(a, array([[5, 12],
                                     [7, 16],
                                     [15, 24],
                                     [21, 32]]))

        # 测试空矩阵的情况：Khatri-Rao 乘积后的形状是否符合预期
        b = khatri_rao(np.empty([2, 2]), np.empty([2, 2]))
        assert_array_equal(b.shape, (4, 2))

    # 测试列数不相等的情况：应该抛出 ValueError 异常
    def test_number_of_columns_equality(self):
        with pytest.raises(ValueError):
            a = array([[1, 2, 3],
                       [4, 5, 6]])
            b = array([[1, 2],
                       [3, 4]])
            khatri_rao(a, b)

    # 测试输入数组不是二维数组的情况：应该抛出 ValueError 异常
    def test_to_assure_2d_array(self):
        with pytest.raises(ValueError):
            # 第一种情况：两个数组都是一维数组
            a = array([1, 2, 3])
            b = array([4, 5, 6])
            khatri_rao(a, b)

        with pytest.raises(ValueError):
            # 第二种情况：第一个数组是一维数组，第二个数组是二维数组
            a = array([1, 2, 3])
            b = array([
                [1, 2, 3],
                [4, 5, 6]
            ])
            khatri_rao(a, b)

        with pytest.raises(ValueError):
            # 第三种情况：第一个数组是二维数组，第二个数组是一维数组
            a = array([
                [1, 2, 3],
                [7, 8, 9]
            ])
            b = array([4, 5, 6])
            khatri_rao(a, b)

    # 测试两种计算 Khatri-Rao 乘积的方法得到的结果是否相等
    def test_equality_of_two_equations(self):
        a = array([[1, 2], [3, 4]])
        b = array([[5, 6], [7, 8]])

        # 计算第一种方法得到的结果 res1
        res1 = khatri_rao(a, b)
        # 计算第二种方法得到的结果 res2
        res2 = np.vstack([np.kron(a[:, k], b[:, k])
                          for k in range(b.shape[1])]).T

        # 断言两种方法得到的结果相等
        assert_array_equal(res1, res2)

    # 测试空矩阵的 Khatri-Rao 乘积
    def test_empty(self):
        # 测试第一种情况：a 是空矩阵，b 不是空矩阵
        a = np.empty((0, 2))
        b = np.empty((3, 2))
        res = khatri_rao(a, b)
        # 断言结果 res 应该与预期的空矩阵形状相等
        assert_allclose(res, np.empty((0, 2)))

        # 测试第二种情况：a 不是空矩阵，b 是空矩阵
        a = np.empty((3, 0))
        b = np.empty((5, 0))
        res = khatri_rao(a, b)
        # 断言结果 res 应该与预期的空矩阵形状相等
        assert_allclose(res, np.empty((15, 0)))
```