# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_trustregion_exact.py`

```
"""
Unit tests for trust-region iterative subproblem.

To run it in its simplest form::
  nosetests test_optimize.py

"""
import pytest
import numpy as np
from scipy.optimize._trustregion_exact import (
    estimate_smallest_singular_value,
    singular_leading_submatrix,
    IterativeSubproblem)
from scipy.linalg import (svd, get_lapack_funcs, det, qr, norm)
from numpy.testing import (assert_array_equal,
                           assert_equal, assert_array_almost_equal)


def random_entry(n, min_eig, max_eig, case):
    """
    Generate a random matrix with specified eigenvalues and gradient vector.

    Parameters:
    - n: int, size of the square matrix
    - min_eig: float, minimum value for eigenvalues
    - max_eig: float, maximum value for eigenvalues
    - case: str, specifies the type of gradient vector ('hard', 'jac_equal_zero', or other)

    Returns:
    - A: ndarray, generated random matrix
    - g: ndarray, gradient vector based on the specified case
    """

    # Generate random matrix
    rand = np.random.uniform(-1, 1, (n, n))

    # QR decomposition
    Q, _, _ = qr(rand, pivoting='True')

    # Generate random eigenvalues
    eigvalues = np.random.uniform(min_eig, max_eig, n)
    eigvalues = np.sort(eigvalues)[::-1]

    # Generate matrix
    Qaux = np.multiply(eigvalues, Q)
    A = np.dot(Qaux, Q.T)

    # Generate gradient vector accordingly
    # to the case is being tested.
    if case == 'hard':
        g = np.zeros(n)
        g[:-1] = np.random.uniform(-1, 1, n-1)
        g = np.dot(Q, g)
    elif case == 'jac_equal_zero':
        g = np.zeros(n)
    else:
        g = np.random.uniform(-1, 1, n)

    return A, g


class TestEstimateSmallestSingularValue:
    """
    Test case for estimating the smallest singular value of a matrix.
    """

    def test_for_ill_condiotioned_matrix(self):
        """
        Test estimation of smallest singular value for an ill-conditioned matrix.
        """

        # Ill-conditioned triangular matrix
        C = np.array([[1, 2, 3, 4],
                      [0, 0.05, 60, 7],
                      [0, 0, 0.8, 9],
                      [0, 0, 0, 10]])

        # Get svd decomposition
        U, s, Vt = svd(C)

        # Get smallest singular value and correspondent right singular vector.
        smin_svd = s[-1]
        zmin_svd = Vt[-1, :]

        # Estimate smallest singular value
        smin, zmin = estimate_smallest_singular_value(C)

        # Check the estimation
        assert_array_almost_equal(smin, smin_svd, decimal=8)
        assert_array_almost_equal(abs(zmin), abs(zmin_svd), decimal=8)


class TestSingularLeadingSubmatrix:
    """
    Test case for singular leading submatrix functionality.
    """

    def test_for_already_singular_leading_submatrix(self):
        """
        Test function for handling a matrix with a singular leading submatrix.
        """

        # Define test matrix A.
        # Note that the leading 2x2 submatrix is singular.
        A = np.array([[1, 2, 3],
                      [2, 4, 5],
                      [3, 5, 6]])

        # Get Cholesky from lapack functions
        cholesky, = get_lapack_funcs(('potrf',), (A,))

        # Compute Cholesky Decomposition
        c, k = cholesky(A, lower=False, overwrite_a=False, clean=True)

        delta, v = singular_leading_submatrix(A, c, k)

        A[k-1, k-1] += delta

        # Check if the leading submatrix is singular.
        assert_array_almost_equal(det(A[:k, :k]), 0)

        # Check if `v` fulfil the specified properties
        quadratic_term = np.dot(v, np.dot(A, v))
        assert_array_almost_equal(quadratic_term, 0)
    def test_for_simetric_indefinite_matrix(self):
        # 定义测试矩阵 A。
        # 注意前 5x5 的子矩阵是不定的。
        A = np.asarray([[1, 2, 3, 7, 8],
                        [2, 5, 5, 9, 0],
                        [3, 5, 11, 1, 2],
                        [7, 9, 1, 7, 5],
                        [8, 0, 2, 5, 8]])

        # 从 lapack 函数中获取 Cholesky 分解
        cholesky, = get_lapack_funcs(('potrf',), (A,))

        # 计算 Cholesky 分解
        c, k = cholesky(A, lower=False, overwrite_a=False, clean=True)

        # 计算增量 delta 和向量 v
        delta, v = singular_leading_submatrix(A, c, k)

        # 将 A 的第 k-1 行、列元素增加 delta
        A[k-1, k-1] += delta

        # 检查前 kxk 子矩阵是否奇异
        assert_array_almost_equal(det(A[:k, :k]), 0)

        # 检查向量 v 是否满足指定的属性
        quadratic_term = np.dot(v, np.dot(A, v))
        assert_array_almost_equal(quadratic_term, 0)

    def test_for_first_element_equal_to_zero(self):
        # 定义测试矩阵 A。
        # 注意前 2x2 的子矩阵是奇异的。
        A = np.array([[0, 3, 11],
                      [3, 12, 5],
                      [11, 5, 6]])

        # 从 lapack 函数中获取 Cholesky 分解
        cholesky, = get_lapack_funcs(('potrf',), (A,))

        # 计算 Cholesky 分解
        c, k = cholesky(A, lower=False, overwrite_a=False, clean=True)

        # 计算增量 delta 和向量 v
        delta, v = singular_leading_submatrix(A, c, k)

        # 将 A 的第 k-1 行、列元素增加 delta
        A[k-1, k-1] += delta

        # 检查前 kxk 子矩阵是否奇异
        assert_array_almost_equal(det(A[:k, :k]), 0)

        # 检查向量 v 是否满足指定的属性
        quadratic_term = np.dot(v, np.dot(A, v))
        assert_array_almost_equal(quadratic_term, 0)


这段代码是两个测试方法，用于测试对称不定矩阵和首个元素为零的特定矩阵的 Cholesky 分解，并检验其性质是否符合预期。
class TestIterativeSubproblem:

    def test_for_the_easy_case(self):

        # `H` is chosen such that `g` is not orthogonal to the
        # eigenvector associated with the smallest eigenvalue `s`.
        # 定义对称矩阵 `H`，确保向量 `g` 不与最小特征值 `s` 对应的特征向量正交
        H = [[10, 2, 3, 4],
             [2, 1, 7, 1],
             [3, 7, 1, 7],
             [4, 1, 7, 2]]
        g = [1, 1, 1, 1]

        # Trust Radius
        # 定义信赖域半径
        trust_radius = 1

        # Solve Subproblem
        # 解决子问题，使用 IterativeSubproblem 类
        subprob = IterativeSubproblem(x=0,
                                      fun=lambda x: 0,
                                      jac=lambda x: np.array(g),
                                      hess=lambda x: np.array(H),
                                      k_easy=1e-10,
                                      k_hard=1e-10)
        # 调用 solve 方法解决子问题，并返回解 `p` 和是否触碰边界 `hits_boundary`
        p, hits_boundary = subprob.solve(trust_radius)

        assert_array_almost_equal(p, [0.00393332, -0.55260862,
                                      0.67065477, -0.49480341])
        assert_array_almost_equal(hits_boundary, True)

    def test_for_the_hard_case(self):

        # `H` is chosen such that `g` is orthogonal to the
        # eigenvector associated with the smallest eigenvalue `s`.
        # 定义对称矩阵 `H`，确保向量 `g` 与最小特征值 `s` 对应的特征向量正交
        H = [[10, 2, 3, 4],
             [2, 1, 7, 1],
             [3, 7, 1, 7],
             [4, 1, 7, 2]]
        g = [6.4852641521327437, 1, 1, 1]
        s = -8.2151519874416614

        # Trust Radius
        # 定义信赖域半径
        trust_radius = 1

        # Solve Subproblem
        # 解决子问题，使用 IterativeSubproblem 类
        subprob = IterativeSubproblem(x=0,
                                      fun=lambda x: 0,
                                      jac=lambda x: np.array(g),
                                      hess=lambda x: np.array(H),
                                      k_easy=1e-10,
                                      k_hard=1e-10)
        # 调用 solve 方法解决子问题，并返回解 `p` 和是否触碰边界 `hits_boundary`
        p, hits_boundary = subprob.solve(trust_radius)

        # 断言最小特征值 `-s` 与子问题对象的当前特征值 `lambda_current` 几乎相等
        assert_array_almost_equal(-s, subprob.lambda_current)

    def test_for_interior_convergence(self):

        # 定义对称矩阵 `H` 和向量 `g`
        H = [[1.812159, 0.82687265, 0.21838879, -0.52487006, 0.25436988],
             [0.82687265, 2.66380283, 0.31508988, -0.40144163, 0.08811588],
             [0.21838879, 0.31508988, 2.38020726, -0.3166346, 0.27363867],
             [-0.52487006, -0.40144163, -0.3166346, 1.61927182, -0.42140166],
             [0.25436988, 0.08811588, 0.27363867, -0.42140166, 1.33243101]]

        g = [0.75798952, 0.01421945, 0.33847612, 0.83725004, -0.47909534]

        # Solve Subproblem
        # 解决子问题，使用 IterativeSubproblem 类
        subprob = IterativeSubproblem(x=0,
                                      fun=lambda x: 0,
                                      jac=lambda x: np.array(g),
                                      hess=lambda x: np.array(H))
        # 调用 solve 方法解决子问题，并返回解 `p` 和是否触碰边界 `hits_boundary`
        p, hits_boundary = subprob.solve(1.1)

        # 断言解 `p` 与预期解几乎相等
        assert_array_almost_equal(p, [-0.68585435, 0.1222621, -0.22090999,
                                      -0.67005053, 0.31586769])
        # 断言未触碰边界
        assert_array_almost_equal(hits_boundary, False)
        # 断言当前特征值 `lambda_current` 为 0
        assert_array_almost_equal(subprob.lambda_current, 0)
        # 断言迭代次数 `niter` 为 1
        assert_array_almost_equal(subprob.niter, 1)
    # 定义一个测试方法，用于测试雅可比矩阵接近零的情况

    H = [[0.88547534, 2.90692271, 0.98440885, -0.78911503, -0.28035809],
         [2.90692271, -0.04618819, 0.32867263, -0.83737945, 0.17116396],
         [0.98440885, 0.32867263, -0.87355957, -0.06521957, -1.43030957],
         [-0.78911503, -0.83737945, -0.06521957, -1.645709, -0.33887298],
         [-0.28035809, 0.17116396, -1.43030957, -0.33887298, -1.68586978]]

    g = [0, 0, 0, 0, 0]

    # 解决子问题
    subprob = IterativeSubproblem(x=0,
                                  fun=lambda x: 0,
                                  jac=lambda x: np.array(g),
                                  hess=lambda x: np.array(H),
                                  k_easy=1e-10,
                                  k_hard=1e-10)
    # 调用 solve 方法求解子问题，参数为 1.1
    p, hits_boundary = subprob.solve(1.1)

    # 断言 p 数组近似等于指定的值数组
    assert_array_almost_equal(p, [0.06910534, -0.01432721,
                                  -0.65311947, -0.23815972,
                                  -0.84954934])
    # 断言 hits_boundary 为 True
    assert_array_almost_equal(hits_boundary, True)



    # 定义另一个测试方法，用于测试雅可比矩阵中某些元素接近零的情况

    H = [[0.88547534, 2.90692271, 0.98440885, -0.78911503, -0.28035809],
         [2.90692271, -0.04618819, 0.32867263, -0.83737945, 0.17116396],
         [0.98440885, 0.32867263, -0.87355957, -0.06521957, -1.43030957],
         [-0.78911503, -0.83737945, -0.06521957, -1.645709, -0.33887298],
         [-0.28035809, 0.17116396, -1.43030957, -0.33887298, -1.68586978]]

    g = [0, 0, 0, 0, 1e-15]

    # 解决子问题
    subprob = IterativeSubproblem(x=0,
                                  fun=lambda x: 0,
                                  jac=lambda x: np.array(g),
                                  hess=lambda x: np.array(H),
                                  k_easy=1e-10,
                                  k_hard=1e-10)
    # 调用 solve 方法求解子问题，参数为 1.1
    p, hits_boundary = subprob.solve(1.1)

    # 断言 p 数组近似等于指定的值数组
    assert_array_almost_equal(p, [0.06910534, -0.01432721,
                                  -0.65311947, -0.23815972,
                                  -0.84954934])
    # 断言 hits_boundary 为 True
    assert_array_almost_equal(hits_boundary, True)



    # 使用 pytest 的标记，指定此测试方法为 "fail_slow" 类型，并设置超时时间为 10 秒
    @pytest.mark.fail_slow(10)
```