# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_nonlin.py`

```
# 导入必要的模块和函数
from numpy.testing import assert_
import pytest

# 导入非线性优化相关的函数和类
from scipy.optimize import _nonlin as nonlin, root

# 导入稀疏矩阵相关的类
from scipy.sparse import csr_array

# 导入数组操作相关的函数和类
from numpy import diag, dot

# 导入线性代数相关的函数
from numpy.linalg import inv

# 导入NumPy库
import numpy as np

# 导入SciPy库
import scipy

# 从test_minpack模块中导入pressure_network函数
from .test_minpack import pressure_network

# 定义一个包含各种非线性求解器的字典
SOLVERS = {'anderson': nonlin.anderson,
           'diagbroyden': nonlin.diagbroyden,
           'linearmixing': nonlin.linearmixing,
           'excitingmixing': nonlin.excitingmixing,
           'broyden1': nonlin.broyden1,
           'broyden2': nonlin.broyden2,
           'krylov': nonlin.newton_krylov}

# 必须正常工作的非线性求解器字典
MUST_WORK = {'anderson': nonlin.anderson, 'broyden1': nonlin.broyden1,
             'broyden2': nonlin.broyden2, 'krylov': nonlin.newton_krylov}

# ----------------------------------------------------------------------------
# 测试问题
# ----------------------------------------------------------------------------


# 定义函数F，接受参数x，返回非线性方程组的函数值
def F(x):
    # 将x转换为NumPy数组，并转置
    x = np.asarray(x).T
    # 定义对角矩阵d
    d = diag([3, 2, 1.5, 1, 0.5])
    # 定义常数c
    c = 0.01
    # 计算函数值f
    f = -d @ x - c * float(x.T @ x) * x
    return f

# 设置F函数的初始猜测值xin
F.xin = [1, 1, 1, 1, 1]
# 定义已知失败的情况字典
F.KNOWN_BAD = {}
F.JAC_KSP_BAD = {}
F.ROOT_JAC_KSP_BAD = {}

# 定义函数F2，接受参数x，返回x本身
def F2(x):
    return x

# 设置F2函数的初始猜测值xin
F2.xin = [1, 2, 3, 4, 5, 6]
# 定义已知失败的情况字典
F2.KNOWN_BAD = {'linearmixing': nonlin.linearmixing,
                'excitingmixing': nonlin.excitingmixing}
F2.JAC_KSP_BAD = {}
F2.ROOT_JAC_KSP_BAD = {}

# 定义函数F2_lucky，接受参数x，返回x本身
def F2_lucky(x):
    return x

# 设置F2_lucky函数的初始猜测值xin
F2_lucky.xin = [0, 0, 0, 0, 0, 0]
# 定义已知失败的情况字典
F2_lucky.KNOWN_BAD = {}
F2_lucky.JAC_KSP_BAD = {}
F2_lucky.ROOT_JAC_KSP_BAD = {}

# 定义函数F3，接受参数x，返回矩阵A乘以x减去向量b的结果
def F3(x):
    A = np.array([[-2, 1, 0.], [1, -2, 1], [0, 1, -2]])
    b = np.array([1, 2, 3.])
    return A @ x - b

# 设置F3函数的初始猜测值xin
F3.xin = [1, 2, 3]
# 定义已知失败的情况字典
F3.KNOWN_BAD = {}
F3.JAC_KSP_BAD = {}
F3.ROOT_JAC_KSP_BAD = {}

# 定义函数F4_powell，接受参数x，返回包含两个方程的列表
def F4_powell(x):
    A = 1e4
    return [A*x[0]*x[1] - 1, np.exp(-x[0]) + np.exp(-x[1]) - (1 + 1/A)]

# 设置F4_powell函数的初始猜测值xin
F4_powell.xin = [-1, -2]
# 定义已知失败的情况字典，包含在极端情况下无法收敛的非线性问题和使用Krylov方法近似雅可比矩阵时无法收敛的根问题
F4_powell.KNOWN_BAD = {'linearmixing': nonlin.linearmixing,
                       'excitingmixing': nonlin.excitingmixing,
                       'diagbroyden': nonlin.diagbroyden}
# 以下注释提供了详细的描述，说明在极端情况下的问题
# 对于非线性问题使用MINRES无法收敛，对于使用GMRES/BiCGStab/CGS/MINRES/TFQMR方法近似雅可比矩阵的根问题也无法收敛
F4_powell.JAC_KSP_BAD = {'minres'}
F4_powell.ROOT_JAC_KSP_BAD = {'gmres', 'bicgstab', 'cgs', 'minres', 'tfqmr'}

# 定义函数F5，接受参数x，返回pressure_network函数对应的结果
def F5(x):
    return pressure_network(x, 4, np.array([.5, .5, .5, .5]))

# 设置F5函数的初始猜测值xin
F5.xin = [2., 0, 2, 0]
# 定义已知失败的情况字典，包含在极端情况下无法收敛的非线性问题和使用Krylov方法近似雅可比矩阵时无法收敛的根问题
F5.KNOWN_BAD = {'excitingmixing': nonlin.excitingmixing,
                'linearmixing': nonlin.linearmixing,
                'diagbroyden': nonlin.diagbroyden}
# 以下注释提供了详细的描述，说明在极端情况下的问题
# 对于非线性问题使用CGS/MINRES无法收敛，对于根问题使用MINRES方法近似雅可比矩阵时也无法收敛
F5.JAC_KSP_BAD = {'cgs', 'minres'}
F5.ROOT_JAC_KSP_BAD = {'minres'}

# 定义函数F6，接受参数x，返回x中的前两个元素
def F6(x):
    x1, x2 = x
    # 创建一个 2x2 的 NumPy 数组 J0，其中包含特定的数值
    J0 = np.array([[-4.256, 14.7],
                   [0.8394989, 0.59964207]])
    
    # 创建一个包含两个元素的 NumPy 数组 v，每个元素是一个数学表达式的结果
    # 第一个元素：(x1 + 3) * (x2**5 - 7) + 18
    # 第二个元素：sin(x2 * exp(x1) - 1)
    v = np.array([(x1 + 3) * (x2**5 - 7) + 18,
                  np.sin(x2 * np.exp(x1) - 1)])
    
    # 使用 np.linalg.solve 求解线性方程组 J0 * x = v，返回解 x
    return -np.linalg.solve(J0, v)
# 定义 F6.xin，包含两个元素的列表
F6.xin = [-0.5, 1.4]
# 定义 F6.KNOWN_BAD 字典，将字符串键映射到非线性方法函数对象
F6.KNOWN_BAD = {'excitingmixing': nonlin.excitingmixing,
                'linearmixing': nonlin.linearmixing,
                'diagbroyden': nonlin.diagbroyden}
# 初始化空字典 F6.JAC_KSP_BAD
F6.JAC_KSP_BAD = {}
# 初始化空字典 F6.ROOT_JAC_KSP_BAD
F6.ROOT_JAC_KSP_BAD = {}


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


class TestNonlin:
    """
    Check the Broyden methods for a few test problems.

    broyden1, broyden2, and newton_krylov must succeed for
    all functions. Some of the others don't -- tests in KNOWN_BAD are skipped.

    """

    def _check_nonlin_func(self, f, func, f_tol=1e-2):
        # 检查所有在 KrylovJacobian 类中提到的方法
        if func == SOLVERS['krylov']:
            # 对于 ['gmres', 'bicgstab', 'cgs', 'minres', 'tfqmr'] 中的每个方法
            for method in ['gmres', 'bicgstab', 'cgs', 'minres', 'tfqmr']:
                # 如果 method 在 f.JAC_KSP_BAD 中，则跳过当前循环
                if method in f.JAC_KSP_BAD:
                    continue

                # 调用 func(f, f.xin, method=method, line_search=None, f_tol=f_tol, maxiter=200, verbose=0)
                x = func(f, f.xin, method=method, line_search=None,
                         f_tol=f_tol, maxiter=200, verbose=0)
                # 断言 np.absolute(f(x)).max() 小于 f_tol
                assert_(np.absolute(f(x)).max() < f_tol)

        # 否则，调用 func(f, f.xin, f_tol=f_tol, maxiter=200, verbose=0)
        x = func(f, f.xin, f_tol=f_tol, maxiter=200, verbose=0)
        # 断言 np.absolute(f(x)).max() 小于 f_tol
        assert_(np.absolute(f(x)).max() < f_tol)

    def _check_root(self, f, method, f_tol=1e-2):
        # 测试 Krylov 方法
        if method == 'krylov':
            # 对于 ['gmres', 'bicgstab', 'cgs', 'minres', 'tfqmr'] 中的每个 jac_method
            for jac_method in ['gmres', 'bicgstab', 'cgs', 'minres', 'tfqmr']:
                # 如果 jac_method 在 f.ROOT_JAC_KSP_BAD 中，则跳过当前循环
                if jac_method in f.ROOT_JAC_KSP_BAD:
                    continue

                # 调用 root(f, f.xin, method=method, options={'ftol': f_tol, 'maxiter': 200, 'disp': 0, 'jac_options': {'method': jac_method}})
                res = root(f, f.xin, method=method,
                           options={'ftol': f_tol, 'maxiter': 200,
                                    'disp': 0,
                                    'jac_options': {'method': jac_method}})
                # 断言 np.absolute(res.fun).max() 小于 f_tol
                assert_(np.absolute(res.fun).max() < f_tol)

        # 否则，调用 root(f, f.xin, method=method, options={'ftol': f_tol, 'maxiter': 200, 'disp': 0})
        res = root(f, f.xin, method=method,
                   options={'ftol': f_tol, 'maxiter': 200, 'disp': 0})
        # 断言 np.absolute(res.fun).max() 小于 f_tol
        assert_(np.absolute(res.fun).max() < f_tol)

    @pytest.mark.xfail
    def _check_func_fail(self, *a, **kw):
        pass

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_problem_nonlin(self):
        # 对于列表中的每个函数 f
        for f in [F, F2, F2_lucky, F3, F4_powell, F5, F6]:
            # 对于 SOLVERS.values() 中的每个函数 func
            for func in SOLVERS.values():
                # 如果 func 在 f.KNOWN_BAD.values() 中
                if func in f.KNOWN_BAD.values():
                    # 如果 func 在 MUST_WORK.values() 中，则调用 self._check_func_fail(f, func)
                    if func in MUST_WORK.values():
                        self._check_func_fail(f, func)
                    continue
                # 否则，调用 self._check_nonlin_func(f, func)
                self._check_nonlin_func(f, func)

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    @pytest.mark.parametrize("method", ['lgmres', 'gmres', 'bicgstab', 'cgs',
                                        'minres', 'tfqmr'])
    # 测试函数，用于验证 tol_norm 关键字在 nonlin_solve 中的使用
    def test_tol_norm_called(self, method):
        # 设置一个标志，用来检查是否使用了 tol_norm
        self._tol_norm_used = False

        # 定义一个局部的规范化函数，将最大绝对值作为规范化值
        def local_norm_func(x):
            # 标记已经使用了 tol_norm
            self._tol_norm_used = True
            return np.absolute(x).max()

        # 调用 newton_krylov 方法进行非线性求解
        nonlin.newton_krylov(F, F.xin, method=method, f_tol=1e-2,
                             maxiter=200, verbose=0,
                             tol_norm=local_norm_func)
        # 断言确保 _tol_norm_used 被设置为 True
        assert_(self._tol_norm_used)

    # 测试函数，验证不同函数和求解器的根的情况
    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_problem_root(self):
        # 遍历函数列表和求解器列表
        for f in [F, F2, F2_lucky, F3, F4_powell, F5, F6]:
            for meth in SOLVERS:
                # 如果求解器在已知有问题的列表中，则跳过
                if meth in f.KNOWN_BAD:
                    # 如果求解器在必须工作的列表中，则检查函数是否失败
                    if meth in MUST_WORK:
                        self._check_func_fail(f, meth)
                    continue
                # 否则，检查根是否正确
                self._check_root(f, meth)

    # 测试函数，验证无法收敛的情况
    def test_no_convergence(self):
        # 定义一个永远不会收敛的函数
        def wont_converge(x):
            return 1e3 + x
        
        # 使用 pytest 的断言来检查是否抛出 NoConvergence 异常
        with pytest.raises(scipy.optimize.NoConvergence):
            nonlin.newton_krylov(wont_converge, xin=[0], maxiter=1)
class TestSecant:
    """Check that some Jacobian approximations satisfy the secant condition"""

    # 初始化一组测试向量，每个向量都是包含5个元素的numpy数组
    xs = [np.array([1., 2., 3., 4., 5.]),
          np.array([2., 3., 4., 5., 1.]),
          np.array([3., 4., 5., 1., 2.]),
          np.array([4., 5., 1., 2., 3.]),
          np.array([9., 1., 9., 1., 3.]),
          np.array([0., 1., 9., 1., 3.]),
          np.array([5., 5., 7., 1., 1.]),
          np.array([1., 2., 7., 5., 1.]),]
    
    # 计算每个测试向量对应的函数值，并存储在列表中
    fs = [x**2 - 1 for x in xs]

    def _check_secant(self, jac_cls, npoints=1, **kw):
        """
        Check that the given Jacobian approximation satisfies secant
        conditions for last `npoints` points.
        """
        # 创建一个指定类型的Jacobian对象，并根据关键字参数进行初始化设置
        jac = jac_cls(**kw)
        jac.setup(self.xs[0], self.fs[0], None)
        
        # 遍历测试向量和对应的函数值，进行Jacobian更新和secant条件检查
        for j, (x, f) in enumerate(zip(self.xs[1:], self.fs[1:])):
            jac.update(x, f)

            # 检查最后 `npoints` 个点的secant条件是否满足
            for k in range(min(npoints, j+1)):
                dx = self.xs[j-k+1] - self.xs[j-k]
                df = self.fs[j-k+1] - self.fs[j-k]
                assert_(np.allclose(dx, jac.solve(df)))

            # 检查 `npoints` 个点的secant条件是否是严格的
            if j >= npoints:
                dx = self.xs[j-npoints+1] - self.xs[j-npoints]
                df = self.fs[j-npoints+1] - self.fs[j-npoints]
                assert_(not np.allclose(dx, jac.solve(df)))

    def test_broyden1(self):
        # 测试BroydenFirst类的Jacobian近似是否满足secant条件
        self._check_secant(nonlin.BroydenFirst)

    def test_broyden2(self):
        # 测试BroydenSecond类的Jacobian近似是否满足secant条件
        self._check_secant(nonlin.BroydenSecond)

    def test_broyden1_update(self):
        # 检查BroydenFirst类的Jacobian更新是否按照密集矩阵的方式工作
        jac = nonlin.BroydenFirst(alpha=0.1)
        jac.setup(self.xs[0], self.fs[0], None)

        B = np.identity(5) * (-1/0.1)

        for last_j, (x, f) in enumerate(zip(self.xs[1:], self.fs[1:])):
            df = f - self.fs[last_j]
            dx = x - self.xs[last_j]
            B += (df - dot(B, dx))[:, None] * dx[None, :] / dot(dx, dx)
            jac.update(x, f)
            # 断言当前的Jacobian近似结果是否与预期的B矩阵密集形式一致
            assert_(np.allclose(jac.todense(), B, rtol=1e-10, atol=1e-13))

    def test_broyden2_update(self):
        # 检查BroydenSecond类的Jacobian更新是否按照密集矩阵的方式工作
        jac = nonlin.BroydenSecond(alpha=0.1)
        jac.setup(self.xs[0], self.fs[0], None)

        H = np.identity(5) * (-0.1)

        for last_j, (x, f) in enumerate(zip(self.xs[1:], self.fs[1:])):
            df = f - self.fs[last_j]
            dx = x - self.xs[last_j]
            H += (dx - dot(H, df))[:, None] * df[None, :] / dot(df, df)
            jac.update(x, f)
            # 断言当前的Jacobian近似结果是否与预期的H矩阵的逆矩阵形式一致
            assert_(np.allclose(jac.todense(), inv(H), rtol=1e-10, atol=1e-13))

    def test_anderson(self):
        # Anderson混合（w0=0）满足最近M次迭代的secant条件
        # 参见文献 [Ey] V. Eyert, J. Comp. Phys., 124, 271 (1996).
        self._check_secant(nonlin.Anderson, M=3, w0=0, npoints=3)
    some methods find the exact solution in a finite number of steps"""

    # 定义一个私有方法，用于检查非线性求解器的功能
    def _check(self, jac, N, maxiter, complex=False, **kw):
        # 设置随机种子以确保结果可复现
        np.random.seed(123)

        # 生成一个随机的 N × N 矩阵 A
        A = np.random.randn(N, N)
        # 如果需要复数运算，添加复数部分
        if complex:
            A = A + 1j*np.random.randn(N, N)
        
        # 生成一个随机的长度为 N 的向量 b
        b = np.random.randn(N)
        # 如果需要复数运算，添加复数部分
        if complex:
            b = b + 1j*np.random.randn(N)

        # 定义函数 func(x)，用于求解线性方程组 A*x = b
        def func(x):
            return np.dot(A, x) - b

        # 调用非线性求解器进行求解
        sol = nonlin.nonlin_solve(func, np.zeros(N), jac, maxiter=maxiter,
                                  f_tol=1e-6, line_search=None, verbose=0)
        # 断言求解结果是否满足精度要求
        assert_(np.allclose(np.dot(A, sol), b, atol=1e-6))

    # 测试 BroydenFirst 非线性求解器
    def test_broyden1(self):
        # Broyden 方法在 2*N 步内确切地求解线性系统
        self._check(nonlin.BroydenFirst(alpha=1.0), 20, 41, False)
        self._check(nonlin.BroydenFirst(alpha=1.0), 20, 41, True)

    # 测试 BroydenSecond 非线性求解器
    def test_broyden2(self):
        # Broyden 方法在 2*N 步内确切地求解线性系统
        self._check(nonlin.BroydenSecond(alpha=1.0), 20, 41, False)
        self._check(nonlin.BroydenSecond(alpha=1.0), 20, 41, True)

    # 测试 Anderson 非线性求解器
    def test_anderson(self):
        # Anderson 方法与 Broyden 方法相似，如果给定足够的存储空间
        self._check(nonlin.Anderson(M=50, alpha=1.0), 20, 29, False)
        self._check(nonlin.Anderson(M=50, alpha=1.0), 20, 29, True)

    # 测试 KrylovJacobian 非线性求解器
    def test_krylov(self):
        # Krylov 方法在 N 个内部步骤中确切地求解线性系统
        self._check(nonlin.KrylovJacobian, 20, 2, False, inner_m=10)
        self._check(nonlin.KrylovJacobian, 20, 2, True, inner_m=10)

    # 检查自动计算 Jacobian 矩阵的功能
    def _check_autojac(self, A, b):
        # 定义函数 func(x)，用于求解线性方程组 A*x = b
        def func(x):
            return np.dot(A, x) - b

        # 定义 Jacobian 函数 jac(v)，直接返回 A
        def jac(v):
            return A

        # 调用非线性求解器进行求解
        sol = nonlin.nonlin_solve(func, np.zeros(b.shape[0]), jac, maxiter=2,
                                  f_tol=1e-6, line_search=None, verbose=0)
        # 断言求解结果是否满足精度要求
        np.testing.assert_allclose(np.dot(A, sol), b, atol=1e-6)

        # 测试输入的 Jacobian 矩阵作为数组，而不是函数的情况
        sol = nonlin.nonlin_solve(func, np.zeros(b.shape[0]), A, maxiter=2,
                                  f_tol=1e-6, line_search=None, verbose=0)
        # 断言求解结果是否满足精度要求
        np.testing.assert_allclose(np.dot(A, sol), b, atol=1e-6)

    # 测试稀疏矩阵输入的 Jacobian 计算
    def test_jac_sparse(self):
        # 创建一个稀疏矩阵 A 和向量 b 进行自动计算 Jacobian
        A = csr_array([[1, 2], [2, 1]])
        b = np.array([1, -1])
        self._check_autojac(A, b)
        self._check_autojac((1 + 2j) * A, (2 + 2j) * b)

    # 测试 ndarray 输入的 Jacobian 计算
    def test_jac_ndarray(self):
        # 创建一个 ndarray 矩阵 A 和向量 b 进行自动计算 Jacobian
        A = np.array([[1, 2], [2, 1]])
        b = np.array([1, -1])
        self._check_autojac(A, b)
        self._check_autojac((1 + 2j) * A, (2 + 2j) * b)
class TestJacobianDotSolve:
    """
    Check that solve/dot methods in Jacobian approximations are consistent
    """

    def _func(self, x):
        return x**2 - 1 + np.dot(self.A, x)
        # 定义一个函数，计算 x^2 - 1 加上 self.A 与 x 的点乘结果

    def _check_dot(self, jac_cls, complex=False, tol=1e-6, **kw):
        np.random.seed(123)
        # 设置随机种子为 123

        N = 7
        # 设置维度 N 为 7

        def rand(*a):
            q = np.random.rand(*a)
            if complex:
                q = q + 1j*np.random.rand(*a)
            return q
            # 定义一个函数 rand，生成形状为 a 的随机数组，如果 complex 为真，则生成复数数组

        def assert_close(a, b, msg):
            d = abs(a - b).max()
            f = tol + abs(b).max()*tol
            if d > f:
                raise AssertionError(f'{msg}: err {d:g}')
            # 定义一个断言函数 assert_close，比较两个数组 a 和 b 是否在容许误差 tol 内相等，如果不相等则抛出异常

        self.A = rand(N, N)
        # 生成 N × N 的随机矩阵赋值给 self.A

        # initialize
        x0 = np.random.rand(N)
        # 生成长度为 N 的随机数组 x0
        jac = jac_cls(**kw)
        # 根据参数 kw 实例化一个 jac_cls 对象
        jac.setup(x0, self._func(x0), self._func)
        # 调用 jac 对象的 setup 方法，传入 x0、self._func(x0) 和 self._func 作为参数

        # check consistency
        for k in range(2*N):
            v = rand(N)
            # 生成长度为 N 的随机数组 v

            if hasattr(jac, '__array__'):
                Jd = np.array(jac)
                # 如果 jac 对象有 '__array__' 属性，则将 jac 转换为 numpy 数组 Jd

                if hasattr(jac, 'solve'):
                    Gv = jac.solve(v)
                    Gv2 = np.linalg.solve(Jd, v)
                    assert_close(Gv, Gv2, 'solve vs array')
                    # 如果 jac 对象有 'solve' 属性，则计算 jac.solve(v) 和 np.linalg.solve(Jd, v) 的差距

                if hasattr(jac, 'rsolve'):
                    Gv = jac.rsolve(v)
                    Gv2 = np.linalg.solve(Jd.T.conj(), v)
                    assert_close(Gv, Gv2, 'rsolve vs array')
                    # 如果 jac 对象有 'rsolve' 属性，则计算 jac.rsolve(v) 和 np.linalg.solve(Jd.T.conj(), v) 的差距

                if hasattr(jac, 'matvec'):
                    Jv = jac.matvec(v)
                    Jv2 = np.dot(Jd, v)
                    assert_close(Jv, Jv2, 'dot vs array')
                    # 如果 jac 对象有 'matvec' 属性，则计算 jac.matvec(v) 和 np.dot(Jd, v) 的差距

                if hasattr(jac, 'rmatvec'):
                    Jv = jac.rmatvec(v)
                    Jv2 = np.dot(Jd.T.conj(), v)
                    assert_close(Jv, Jv2, 'rmatvec vs array')
                    # 如果 jac 对象有 'rmatvec' 属性，则计算 jac.rmatvec(v) 和 np.dot(Jd.T.conj(), v) 的差距

            if hasattr(jac, 'matvec') and hasattr(jac, 'solve'):
                Jv = jac.matvec(v)
                Jv2 = jac.solve(jac.matvec(Jv))
                assert_close(Jv, Jv2, 'dot vs solve')
                # 如果 jac 对象同时有 'matvec' 和 'solve' 属性，则计算 jac.solve(jac.matvec(Jv)) 和 Jv 的差距

            if hasattr(jac, 'rmatvec') and hasattr(jac, 'rsolve'):
                Jv = jac.rmatvec(v)
                Jv2 = jac.rmatvec(jac.rsolve(Jv))
                assert_close(Jv, Jv2, 'rmatvec vs rsolve')
                # 如果 jac 对象同时有 'rmatvec' 和 'rsolve' 属性，则计算 jac.rmatvec(jac.rsolve(Jv)) 和 Jv 的差距

            x = rand(N)
            jac.update(x, self._func(x))
            # 生成长度为 N 的随机数组 x，然后调用 jac 对象的 update 方法，传入 x 和 self._func(x) 作为参数

    def test_broyden1(self):
        self._check_dot(nonlin.BroydenFirst, complex=False)
        self._check_dot(nonlin.BroydenFirst, complex=True)
        # 调用 _check_dot 方法，传入 nonlin.BroydenFirst 类和 complex=False 或 complex=True 作为参数进行测试

    def test_broyden2(self):
        self._check_dot(nonlin.BroydenSecond, complex=False)
        self._check_dot(nonlin.BroydenSecond, complex=True)
        # 调用 _check_dot 方法，传入 nonlin.BroydenSecond 类和 complex=False 或 complex=True 作为参数进行测试

    def test_anderson(self):
        self._check_dot(nonlin.Anderson, complex=False)
        self._check_dot(nonlin.Anderson, complex=True)
        # 调用 _check_dot 方法，传入 nonlin.Anderson 类和 complex=False 或 complex=True 作为参数进行测试

    def test_diagbroyden(self):
        self._check_dot(nonlin.DiagBroyden, complex=False)
        self._check_dot(nonlin.DiagBroyden, complex=True)
        # 调用 _check_dot 方法，传入 nonlin.DiagBroyden 类和 complex=False 或 complex=True 作为参数进行测试

    def test_linearmixing(self):
        self._check_dot(nonlin.LinearMixing, complex=False)
        self._check_dot(nonlin.LinearMixing, complex=True)
        # 调用 _check_dot 方法，传入 nonlin.LinearMixing 类和 complex=False 或 complex=True 作为参数进行测试
    # 定义一个测试方法，测试非线性系统中的 ExcitingMixing 类的功能
    def test_excitingmixing(self):
        # 调用 _check_dot 方法，检查 ExcitingMixing 类的功能，传入 complex=False 参数
        self._check_dot(nonlin.ExcitingMixing, complex=False)
        # 再次调用 _check_dot 方法，检查 ExcitingMixing 类的功能，传入 complex=True 参数
        self._check_dot(nonlin.ExcitingMixing, complex=True)
    
    # 定义另一个测试方法，测试非线性系统中的 KrylovJacobian 类的功能
    def test_krylov(self):
        # 调用 _check_dot 方法，检查 KrylovJacobian 类的功能，传入 complex=False 和 tol=1e-3 参数
        self._check_dot(nonlin.KrylovJacobian, complex=False, tol=1e-3)
        # 再次调用 _check_dot 方法，检查 KrylovJacobian 类的功能，传入 complex=True 和 tol=1e-3 参数
        self._check_dot(nonlin.KrylovJacobian, complex=True, tol=1e-3)
# 定义一个测试类 TestNonlinOldTests，用于测试非线性求解器的功能
class TestNonlinOldTests:
    """ Test case for a simple constrained entropy maximization problem
    (the machine translation example of Berger et al in
    Computational Linguistics, vol 22, num 1, pp 39--72, 1996.)
    """

    # 测试 Broyden 方法（类型1），验证求解精度
    def test_broyden1(self):
        x = nonlin.broyden1(F, F.xin, iter=12, alpha=1)
        assert_(nonlin.norm(x) < 1e-9)
        assert_(nonlin.norm(F(x)) < 1e-9)

    # 测试 Broyden 方法（类型2），验证求解精度
    def test_broyden2(self):
        x = nonlin.broyden2(F, F.xin, iter=12, alpha=1)
        assert_(nonlin.norm(x) < 1e-9)
        assert_(nonlin.norm(F(x)) < 1e-9)

    # 测试 Anderson 加速方法，验证求解精度
    def test_anderson(self):
        x = nonlin.anderson(F, F.xin, iter=12, alpha=0.03, M=5)
        assert_(nonlin.norm(x) < 0.33)

    # 测试线性混合方法，验证求解精度
    def test_linearmixing(self):
        x = nonlin.linearmixing(F, F.xin, iter=60, alpha=0.5)
        assert_(nonlin.norm(x) < 1e-7)
        assert_(nonlin.norm(F(x)) < 1e-7)

    # 测试激动人心的混合方法，验证求解精度
    def test_exciting(self):
        x = nonlin.excitingmixing(F, F.xin, iter=20, alpha=0.5)
        assert_(nonlin.norm(x) < 1e-5)
        assert_(nonlin.norm(F(x)) < 1e-5)

    # 测试对角 Broyden 方法，验证求解精度
    def test_diagbroyden(self):
        x = nonlin.diagbroyden(F, F.xin, iter=11, alpha=1)
        assert_(nonlin.norm(x) < 1e-8)
        assert_(nonlin.norm(F(x)) < 1e-8)

    # 测试根求解器 Broyden 方法（类型1），验证求解精度
    def test_root_broyden1(self):
        res = root(F, F.xin, method='broyden1',
                   options={'nit': 12, 'jac_options': {'alpha': 1}})
        assert_(nonlin.norm(res.x) < 1e-9)
        assert_(nonlin.norm(res.fun) < 1e-9)

    # 测试根求解器 Broyden 方法（类型2），验证求解精度
    def test_root_broyden2(self):
        res = root(F, F.xin, method='broyden2',
                   options={'nit': 12, 'jac_options': {'alpha': 1}})
        assert_(nonlin.norm(res.x) < 1e-9)
        assert_(nonlin.norm(res.fun) < 1e-9)

    # 测试根求解器 Anderson 加速方法，验证求解精度
    def test_root_anderson(self):
        res = root(F, F.xin, method='anderson',
                   options={'nit': 12,
                            'jac_options': {'alpha': 0.03, 'M': 5}})
        assert_(nonlin.norm(res.x) < 0.33)

    # 测试根求解器线性混合方法，验证求解精度
    def test_root_linearmixing(self):
        res = root(F, F.xin, method='linearmixing',
                   options={'nit': 60,
                            'jac_options': {'alpha': 0.5}})
        assert_(nonlin.norm(res.x) < 1e-7)
        assert_(nonlin.norm(res.fun) < 1e-7)

    # 测试根求解器激动人心的混合方法，验证求解精度
    def test_root_excitingmixing(self):
        res = root(F, F.xin, method='excitingmixing',
                   options={'nit': 20,
                            'jac_options': {'alpha': 0.5}})
        assert_(nonlin.norm(res.x) < 1e-5)
        assert_(nonlin.norm(res.fun) < 1e-5)

    # 测试根求解器对角 Broyden 方法，验证求解精度
    def test_root_diagbroyden(self):
        res = root(F, F.xin, method='diagbroyden',
                   options={'nit': 11,
                            'jac_options': {'alpha': 1}})
        assert_(nonlin.norm(res.x) < 1e-8)
        assert_(nonlin.norm(res.fun) < 1e-8)
```