# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_trustregion_krylov.py`

```
"""
Unit tests for Krylov space trust-region subproblem solver.

To run it in its simplest form::
  nosetests test_optimize.py

"""
import numpy as np
from scipy.optimize._trlib import (get_trlib_quadratic_subproblem)
from numpy.testing import (assert_,
                           assert_almost_equal,
                           assert_equal, assert_array_almost_equal)

# 使用 get_trlib_quadratic_subproblem 函数创建 KrylovQP 对象，设置相对误差容限
KrylovQP = get_trlib_quadratic_subproblem(tol_rel_i=1e-8, tol_rel_b=1e-6)

# 使用 get_trlib_quadratic_subproblem 函数创建 KrylovQP_disp 对象，设置相对误差容限和显示参数
KrylovQP_disp = get_trlib_quadratic_subproblem(tol_rel_i=1e-8, tol_rel_b=1e-6,
                                               disp=True)

class TestKrylovQuadraticSubproblem:

    def test_for_the_easy_case(self):

        # `H` is chosen such that `g` is not orthogonal to the
        # eigenvector associated with the smallest eigenvalue.
        H = np.array([[1.0, 0.0, 4.0],
                      [0.0, 2.0, 0.0],
                      [4.0, 0.0, 3.0]])
        g = np.array([5.0, 0.0, 4.0])

        # 信任域半径
        trust_radius = 1.0

        # 解决子问题
        subprob = KrylovQP(x=0,
                           fun=lambda x: 0,
                           jac=lambda x: g,
                           hess=lambda x: None,
                           hessp=lambda x, y: H.dot(y))
        p, hits_boundary = subprob.solve(trust_radius)

        assert_array_almost_equal(p, np.array([-1.0, 0.0, 0.0]))
        assert_equal(hits_boundary, True)
        # 检查 KKT 条件是否满足
        assert_almost_equal(
                np.linalg.norm(H.dot(p) + subprob.lam * p + g),
                0.0)
        # 检查信任域约束
        assert_almost_equal(np.linalg.norm(p), trust_radius)

        trust_radius = 0.5
        p, hits_boundary = subprob.solve(trust_radius)

        assert_array_almost_equal(p,
                np.array([-0.46125446, 0., -0.19298788]))
        assert_equal(hits_boundary, True)
        # 检查 KKT 条件是否满足
        assert_almost_equal(
                np.linalg.norm(H.dot(p) + subprob.lam * p + g),
                0.0)
        # 检查信任域约束
        assert_almost_equal(np.linalg.norm(p), trust_radius)
    def test_for_the_hard_case(self):
        # `H` is chosen such that `g` is orthogonal to the
        # eigenvector associated with the smallest eigenvalue.
        H = np.array([[1.0, 0.0, 4.0],
                      [0.0, 2.0, 0.0],
                      [4.0, 0.0, 3.0]])
        g = np.array([0.0, 2.0, 0.0])

        # Trust Radius
        trust_radius = 1.0

        # Solve Subproblem
        subprob = KrylovQP(x=0,
                           fun=lambda x: 0,
                           jac=lambda x: g,
                           hess=lambda x: None,
                           hessp=lambda x, y: H.dot(y))
        p, hits_boundary = subprob.solve(trust_radius)

        assert_array_almost_equal(p, np.array([0.0, -1.0, 0.0]))
        # check kkt satisfaction
        assert_almost_equal(
                np.linalg.norm(H.dot(p) + subprob.lam * p + g),
                0.0)
        # check trust region constraint
        assert_almost_equal(np.linalg.norm(p), trust_radius)

        trust_radius = 0.5
        p, hits_boundary = subprob.solve(trust_radius)

        assert_array_almost_equal(p, np.array([0.0, -0.5, 0.0]))
        # check kkt satisfaction
        assert_almost_equal(
                np.linalg.norm(H.dot(p) + subprob.lam * p + g),
                0.0)
        # check trust region constraint
        assert_almost_equal(np.linalg.norm(p), trust_radius)

    def test_for_interior_convergence(self):
        # Define a 5x5 matrix H with specific numeric values
        H = np.array([[1.812159, 0.82687265, 0.21838879, -0.52487006, 0.25436988],
                      [0.82687265, 2.66380283, 0.31508988, -0.40144163, 0.08811588],
                      [0.21838879, 0.31508988, 2.38020726, -0.3166346, 0.27363867],
                      [-0.52487006, -0.40144163, -0.3166346, 1.61927182, -0.42140166],
                      [0.25436988, 0.08811588, 0.27363867, -0.42140166, 1.33243101]])
        # Define a 1x5 vector g with specific numeric values
        g = np.array([0.75798952, 0.01421945, 0.33847612, 0.83725004, -0.47909534])
        trust_radius = 1.1

        # Solve Subproblem using KrylovQP class
        subprob = KrylovQP(x=0,
                           fun=lambda x: 0,
                           jac=lambda x: g,
                           hess=lambda x: None,
                           hessp=lambda x, y: H.dot(y))
        # Solve the subproblem with the given trust radius
        p, hits_boundary = subprob.solve(trust_radius)

        # check KKT satisfaction condition
        assert_almost_equal(
                np.linalg.norm(H.dot(p) + subprob.lam * p + g),
                0.0)

        # Assert the computed `p` matches the expected values
        assert_array_almost_equal(p, [-0.68585435, 0.1222621, -0.22090999,
                                      -0.67005053, 0.31586769])
        # Assert `hits_boundary` is False
        assert_array_almost_equal(hits_boundary, False)
    def test_for_very_close_to_zero(self):
        # 定义一个5x5的numpy数组H，表示Hessian矩阵
        H = np.array([[0.88547534, 2.90692271, 0.98440885, -0.78911503, -0.28035809],
                      [2.90692271, -0.04618819, 0.32867263, -0.83737945, 0.17116396],
                      [0.98440885, 0.32867263, -0.87355957, -0.06521957, -1.43030957],
                      [-0.78911503, -0.83737945, -0.06521957, -1.645709, -0.33887298],
                      [-0.28035809, 0.17116396, -1.43030957, -0.33887298, -1.68586978]])
        # 定义一个长度为5的numpy数组g，表示梯度向量
        g = np.array([0, 0, 0, 0, 1e-6])
        # 定义信赖区域半径
        trust_radius = 1.1

        # 解决子问题
        subprob = KrylovQP(x=0,
                           fun=lambda x: 0,
                           jac=lambda x: g,
                           hess=lambda x: None,
                           hessp=lambda x, y: H.dot(y))
        # 解子问题，获得最优解p和是否触及边界hits_boundary
        p, hits_boundary = subprob.solve(trust_radius)

        # 检查KKT条件满足度
        assert_almost_equal(
                np.linalg.norm(H.dot(p) + subprob.lam * p + g),
                0.0)
        # 检查信赖区域约束
        assert_almost_equal(np.linalg.norm(p), trust_radius)

        # 检查最优解p是否接近于指定值
        assert_array_almost_equal(p, [0.06910534, -0.01432721,
                                      -0.65311947, -0.23815972,
                                      -0.84954934])
        # 检查hits_boundary是否为True
        assert_array_almost_equal(hits_boundary, True)

    def test_disp(self, capsys):
        # 定义一个5x5的负单位矩阵H，表示Hessian矩阵
        H = -np.eye(5)
        # 定义一个长度为5的numpy数组g，表示梯度向量
        g = np.array([0, 0, 0, 0, 1e-6])
        # 定义信赖区域半径
        trust_radius = 1.1

        # 解决子问题
        subprob = KrylovQP_disp(x=0,
                                fun=lambda x: 0,
                                jac=lambda x: g,
                                hess=lambda x: None,
                                hessp=lambda x, y: H.dot(y))
        # 解子问题，获得最优解p和是否触及边界hits_boundary
        p, hits_boundary = subprob.solve(trust_radius)
        # 读取和捕获标准输出和错误输出
        out, err = capsys.readouterr()
        # 断言标准输出以特定字符串开头
        assert_(out.startswith(' TR Solving trust region problem'), repr(out))
```