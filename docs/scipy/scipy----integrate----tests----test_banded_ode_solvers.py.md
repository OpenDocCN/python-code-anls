# `D:\src\scipysrc\scipy\scipy\integrate\tests\test_banded_ode_solvers.py`

```
import itertools
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import ode

# 定义函数 _band_count，用于计算矩阵 a 的下三角和上三角带宽
def _band_count(a):
    """Returns ml and mu, the lower and upper band sizes of a."""
    nrows, ncols = a.shape
    ml = 0
    # 计算下三角带宽 ml
    for k in range(-nrows+1, 0):
        if np.diag(a, k).any():
            ml = -k
            break
    mu = 0
    # 计算上三角带宽 mu
    for k in range(nrows-1, 0, -1):
        if np.diag(a, k).any():
            mu = k
            break
    return ml, mu

# 定义函数 _linear_func，表示线性系统 dy/dt = a * y 的右端项
def _linear_func(t, y, a):
    """Linear system dy/dt = a * y"""
    return a.dot(y)

# 定义函数 _linear_jac，表示线性系统 dy/dt = a * y 的雅可比矩阵
def _linear_jac(t, y, a):
    """Jacobian of a * y is a."""
    return a

# 定义函数 _linear_banded_jac，用于生成带状雅可比矩阵
def _linear_banded_jac(t, y, a):
    """Banded Jacobian."""
    ml, mu = _band_count(a)
    # 构建带状雅可比矩阵的列表
    bjac = [np.r_[[0] * k, np.diag(a, k)] for k in range(mu, 0, -1)]
    bjac.append(np.diag(a))
    for k in range(-1, -ml-1, -1):
        bjac.append(np.r_[np.diag(a, k), [0] * (-k)])
    return bjac

# 定义函数 _solve_linear_sys，用于求解线性常微分方程组
def _solve_linear_sys(a, y0, tend=1, dt=0.1,
                      solver=None, method='bdf', use_jac=True,
                      with_jacobian=False, banded=False):
    """Use scipy.integrate.ode to solve a linear system of ODEs.

    a : square ndarray
        Matrix of the linear system to be solved.
    y0 : ndarray
        Initial condition
    tend : float
        Stop time.
    dt : float
        Step size of the output.
    solver : str
        If not None, this must be "vode", "lsoda" or "zvode".
    method : str
        Either "bdf" or "adams".
    use_jac : bool
        Determines if the jacobian function is passed to ode().
    with_jacobian : bool
        Passed to ode.set_integrator().
    banded : bool
        Determines whether a banded or full jacobian is used.
        If `banded` is True, `lband` and `uband` are determined by the
        values in `a`.
    """
    # 根据参数 banded 确定带宽
    if banded:
        lband, uband = _band_count(a)
    else:
        lband = None
        uband = None

    # 根据参数 use_jac 和 banded 设置 ode 求解器的函数
    if use_jac:
        if banded:
            r = ode(_linear_func, _linear_banded_jac)
        else:
            r = ode(_linear_func, _linear_jac)
    else:
        r = ode(_linear_func)

    # 根据矩阵 a 的类型选择求解器
    if solver is None:
        if np.iscomplexobj(a):
            solver = "zvode"
        else:
            solver = "vode"

    # 设置 ode 求解器的参数
    r.set_integrator(solver,
                     with_jacobian=with_jacobian,
                     method=method,
                     lband=lband, uband=uband,
                     rtol=1e-9, atol=1e-10,
                     )
    t0 = 0
    r.set_initial_value(y0, t0)
    r.set_f_params(a)
    r.set_jac_params(a)

    # 执行求解过程
    t = [t0]
    y = [y0]
    while r.successful() and r.t < tend:
        r.integrate(r.t + dt)
        t.append(r.t)
        y.append(r.y)

    t = np.array(t)
    y = np.array(y)
    return t, y

# 定义函数 _analytical_solution，提供线性微分方程组 dy/dt = a*y 的解析解
def _analytical_solution(a, y0, t):
    """
    Analytical solution to the linear differential equations dy/dt = a*y.

    The solution is only valid if `a` is diagonalizable.

    Returns a 2-D array with shape (len(t), len(y0)).
    """

# 此处省略了解析解的实现部分，因此无需进一步注释
    # 使用 numpy 的线性代数模块计算矩阵 a 的特征值 lam 和特征向量 v
    lam, v = np.linalg.eig(a)
    # 使用 numpy 的线性代数模块解线性方程 v * c = y0，求解出系数向量 c
    c = np.linalg.solve(v, y0)
    # 根据特征值 lam、时间向量 t 和系数向量 c 计算每个时间点上的解向量 e
    e = c * np.exp(lam * t.reshape(-1, 1))
    # 将计算得到的解向量 e 与特征向量 v 的转置相乘，得到最终的解 sol
    sol = e.dot(v.T)
    # 返回计算结果 sol
    return sol
# 定义一个测试函数，用于测试带有带状雅可比矩阵的 ODE 求解器 "lsoda"、"vode" 和 "zvode"
def test_banded_ode_solvers():
    # 创建一个精确时间数组，从0到1之间均匀分布成5个点
    t_exact = np.linspace(0, 1.0, 5)

    # --- 用于测试 "lsoda" 和 "vode" 求解器的实数数组 ---

    # lband = 2, uband = 1 的实数数组:
    a_real = np.array([[-0.6, 0.1, 0.0, 0.0, 0.0],
                       [0.2, -0.5, 0.9, 0.0, 0.0],
                       [0.1, 0.1, -0.4, 0.1, 0.0],
                       [0.0, 0.3, -0.1, -0.9, -0.3],
                       [0.0, 0.0, 0.1, 0.1, -0.7]])

    # lband = 0, uband = 1 的实数数组:
    a_real_upper = np.triu(a_real)

    # lband = 2, uband = 0 的实数数组:
    a_real_lower = np.tril(a_real)

    # lband = 0, uband = 0 的实数数组:
    a_real_diag = np.triu(a_real_lower)

    # 将所有实数矩阵放入列表中
    real_matrices = [a_real, a_real_upper, a_real_lower, a_real_diag]
    real_solutions = []

    # 对每个实数矩阵计算其精确解
    for a in real_matrices:
        y0 = np.arange(1, a.shape[0] + 1)
        y_exact = _analytical_solution(a, y0, t_exact)
        real_solutions.append((y0, t_exact, y_exact))

    # 定义一个函数，用于检查每个实数矩阵对应的解是否符合预期
    def check_real(idx, solver, meth, use_jac, with_jac, banded):
        a = real_matrices[idx]
        y0, t_exact, y_exact = real_solutions[idx]
        # 解线性系统，返回时间和解
        t, y = _solve_linear_sys(a, y0,
                                 tend=t_exact[-1],
                                 dt=t_exact[1] - t_exact[0],
                                 solver=solver,
                                 method=meth,
                                 use_jac=use_jac,
                                 with_jacobian=with_jac,
                                 banded=banded)
        # 断言时间和解与精确解的接近程度
        assert_allclose(t, t_exact)
        assert_allclose(y, y_exact)

    # 使用 itertools.product 对不同参数组合进行迭代，以测试每种组合下的解算器
    for idx in range(len(real_matrices)):
        p = [['vode', 'lsoda'],   # solver
             ['bdf', 'adams'],    # method
             [False, True],       # use_jac
             [False, True],       # with_jacobian
             [False, True]]       # banded
        for solver, meth, use_jac, with_jac, banded in itertools.product(*p):
            check_real(idx, solver, meth, use_jac, with_jac, banded)

    # --- 用于测试 "zvode" 求解器的复数数组 ---

    # 复数，lband = 2, uband = 1 的复数数组:
    a_complex = a_real - 0.5j * a_real

    # 复数，lband = 0, uband = 0 的复数数组:
    a_complex_diag = np.diag(np.diag(a_complex))

    # 将所有复数矩阵放入列表中
    complex_matrices = [a_complex, a_complex_diag]
    complex_solutions = []

    # 对每个复数矩阵计算其精确解
    for a in complex_matrices:
        y0 = np.arange(1, a.shape[0] + 1) + 1j
        y_exact = _analytical_solution(a, y0, t_exact)
        complex_solutions.append((y0, t_exact, y_exact))
    # 定义一个函数，用于检查复杂情况下的线性系统求解结果是否正确
    def check_complex(idx, solver, meth, use_jac, with_jac, banded):
        # 从复杂矩阵列表中获取第 idx 个复杂矩阵
        a = complex_matrices[idx]
        # 从复杂解决方案列表中获取第 idx 个复杂解决方案的初始值、精确时间和精确解
        y0, t_exact, y_exact = complex_solutions[idx]
        # 调用内部函数 _solve_linear_sys 进行线性系统求解
        t, y = _solve_linear_sys(a, y0,
                                 tend=t_exact[-1],  # 设置求解结束时间为精确时间的最后一个值
                                 dt=t_exact[1] - t_exact[0],  # 设置时间步长为精确时间的差值
                                 solver=solver,  # 指定求解器名称
                                 method=meth,  # 指定求解方法
                                 use_jac=use_jac,  # 指定是否使用雅可比矩阵
                                 with_jacobian=with_jac,  # 指定是否提供雅可比矩阵
                                 banded=banded)  # 指定是否使用带状格式
        # 断言求解得到的时间数组 t 与精确时间 t_exact 接近
        assert_allclose(t, t_exact)
        # 断言求解得到的解数组 y 与精确解 y_exact 接近
        assert_allclose(y, y_exact)

    # 遍历复杂矩阵列表中的每个索引 idx
    for idx in range(len(complex_matrices)):
        # 定义参数组合列表 p，包含四个参数列表
        p = [['bdf', 'adams'],   # method 参数可能取值
             [False, True],      # use_jac 参数可能取值
             [False, True],      # with_jacobian 参数可能取值
             [False, True]]      # banded 参数可能取值
        # 使用 itertools.product 生成 p 中各参数的所有组合
        for meth, use_jac, with_jac, banded in itertools.product(*p):
            # 对每个参数组合调用 check_complex 函数进行检查
            check_complex(idx, "zvode", meth, use_jac, with_jac, banded)
```