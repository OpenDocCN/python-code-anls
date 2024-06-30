# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_trustregion.py`

```
"""
Unit tests for trust-region optimization routines.

To run it in its simplest form::
  nosetests test_optimize.py

"""
# 导入所需的库和模块
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
from scipy.optimize import (minimize, rosen, rosen_der, rosen_hess,
                            rosen_hess_prod)


class Accumulator:
    """ This is for testing callbacks."""
    # 累加器类，用于测试回调函数
    def __init__(self):
        self.count = 0
        self.accum = None

    def __call__(self, x):
        # 调用对象实例时，更新计数并累加参数 x 的值
        self.count += 1
        if self.accum is None:
            self.accum = np.array(x)
        else:
            self.accum += x


class TestTrustRegionSolvers:

    def setup_method(self):
        # 初始化测试方法，设置优化的目标值和不同的初始猜测
        self.x_opt = [1.0, 1.0]
        self.easy_guess = [2.0, 2.0]
        self.hard_guess = [-1.2, 1.0]

    def test_dogleg_accuracy(self):
        # 测试 dogleg 方法的精度和 return_all 选项
        x0 = self.hard_guess
        r = minimize(rosen, x0, jac=rosen_der, hess=rosen_hess, tol=1e-8,
                     method='dogleg', options={'return_all': True},)
        assert_allclose(x0, r['allvecs'][0])
        assert_allclose(r['x'], r['allvecs'][-1])
        assert_allclose(r['x'], self.x_opt)

    def test_dogleg_callback(self):
        # 测试回调机制和 maxiter、return_all 选项
        accumulator = Accumulator()
        maxiter = 5
        r = minimize(rosen, self.hard_guess, jac=rosen_der, hess=rosen_hess,
                     callback=accumulator, method='dogleg',
                     options={'return_all': True, 'maxiter': maxiter},)
        assert_equal(accumulator.count, maxiter)
        assert_equal(len(r['allvecs']), maxiter+1)
        assert_allclose(r['x'], r['allvecs'][-1])
        assert_allclose(sum(r['allvecs'][1:]), accumulator.accum)

    def test_dogleg_user_warning(self):
        # 测试用户警告和 disp、maxiter 选项
        with pytest.warns(RuntimeWarning,
                          match=r'Maximum number of iterations'):
            minimize(rosen, self.hard_guess, jac=rosen_der,
                     hess=rosen_hess, method='dogleg',
                     options={'disp': True, 'maxiter': 1}, )
    def test_solver_concordance(self):
        # 断言：dogleg 方法在 Rosenbrock 测试函数上使用的迭代次数比 ncg 方法少，
        # 尽管这并不一定意味着 dogleg 对于该函数更快或更好，
        # 特别是对于其他测试函数而言可能并非如此。
        
        # 设置测试函数及其导数和黑塞矩阵函数
        f = rosen
        g = rosen_der
        h = rosen_hess
        
        # 对于两个初始点 self.easy_guess 和 self.hard_guess 分别进行以下测试
        for x0 in (self.easy_guess, self.hard_guess):
            # 使用 dogleg 方法进行优化，返回所有步骤的结果
            r_dogleg = minimize(f, x0, jac=g, hess=h, tol=1e-8,
                                method='dogleg', options={'return_all': True})
            # 使用 trust-ncg 方法进行优化，返回所有步骤的结果
            r_trust_ncg = minimize(f, x0, jac=g, hess=h, tol=1e-8,
                                   method='trust-ncg',
                                   options={'return_all': True})
            # 使用 trust-krylov 方法进行优化，返回所有步骤的结果
            r_trust_krylov = minimize(f, x0, jac=g, hess=h, tol=1e-8,
                                      method='trust-krylov',
                                      options={'return_all': True})
            # 使用 newton-cg 方法进行优化，返回所有步骤的结果
            r_ncg = minimize(f, x0, jac=g, hess=h, tol=1e-8,
                             method='newton-cg', options={'return_all': True})
            # 使用 trust-exact 方法进行优化，返回所有步骤的结果
            r_iterative = minimize(f, x0, jac=g, hess=h, tol=1e-8,
                                   method='trust-exact',
                                   options={'return_all': True})
            
            # 断言：dogleg 方法得到的最优点 r_dogleg['x'] 应与预期的最优点 self.x_opt 接近
            assert_allclose(self.x_opt, r_dogleg['x'])
            # 断言：trust-ncg 方法得到的最优点 r_trust_ncg['x'] 应与预期的最优点 self.x_opt 接近
            assert_allclose(self.x_opt, r_trust_ncg['x'])
            # 断言：trust-krylov 方法得到的最优点 r_trust_krylov['x'] 应与预期的最优点 self.x_opt 接近
            assert_allclose(self.x_opt, r_trust_krylov['x'])
            # 断言：newton-cg 方法得到的最优点 r_ncg['x'] 应与预期的最优点 self.x_opt 接近
            assert_allclose(self.x_opt, r_ncg['x'])
            # 断言：trust-exact 方法得到的最优点 r_iterative['x'] 应与预期的最优点 self.x_opt 接近
            assert_allclose(self.x_opt, r_iterative['x'])
            
            # 断言：dogleg 方法的迭代步骤数 len(r_dogleg['allvecs']) 应小于 ncg 方法的迭代步骤数 len(r_ncg['allvecs'])
            assert_(len(r_dogleg['allvecs']) < len(r_ncg['allvecs']))

    def test_trust_ncg_hessp(self):
        # 对于三个初始点 self.easy_guess, self.hard_guess, self.x_opt 分别进行 trust-ncg 方法的优化测试
        for x0 in (self.easy_guess, self.hard_guess, self.x_opt):
            # 使用 trust-ncg 方法进行优化，返回所有步骤的结果
            r = minimize(rosen, x0, jac=rosen_der, hessp=rosen_hess_prod,
                         tol=1e-8, method='trust-ncg')
            # 断言：最优点 r['x'] 应与预期的最优点 self.x_opt 接近
            assert_allclose(self.x_opt, r['x'])

    def test_trust_ncg_start_in_optimum(self):
        # 使用 trust-ncg 方法从最优点 self.x_opt 开始进行优化
        r = minimize(rosen, x0=self.x_opt, jac=rosen_der, hess=rosen_hess,
                     tol=1e-8, method='trust-ncg')
        # 断言：最优点 r['x'] 应与预期的最优点 self.x_opt 接近
        assert_allclose(self.x_opt, r['x'])

    def test_trust_krylov_start_in_optimum(self):
        # 使用 trust-krylov 方法从最优点 self.x_opt 开始进行优化
        r = minimize(rosen, x0=self.x_opt, jac=rosen_der, hess=rosen_hess,
                     tol=1e-8, method='trust-krylov')
        # 断言：最优点 r['x'] 应与预期的最优点 self.x_opt 接近
        assert_allclose(self.x_opt, r['x'])

    def test_trust_exact_start_in_optimum(self):
        # 使用 trust-exact 方法从最优点 self.x_opt 开始进行优化
        r = minimize(rosen, x0=self.x_opt, jac=rosen_der, hess=rosen_hess,
                     tol=1e-8, method='trust-exact')
        # 断言：最优点 r['x'] 应与预期的最优点 self.x_opt 接近
        assert_allclose(self.x_opt, r['x'])
```