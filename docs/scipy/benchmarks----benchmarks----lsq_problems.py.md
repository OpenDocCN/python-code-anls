# `D:\src\scipysrc\scipy\benchmarks\benchmarks\lsq_problems.py`

```
"""Benchmark problems for nonlinear least squares."""

# 导入用于分析模块信息和系统参数的库
import inspect
import sys
# 导入数值计算库和多项式库
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
# 导入数值积分函数
from scipy.integrate import odeint

# 定义一个类，用于表示非线性最小二乘优化问题的基准测试
class LSQBenchmarkProblem:
    """Template class for nonlinear least squares benchmark problems.

    The optimized variable is n-dimensional vector x and the objective function 
    has the form

    F(x) = ||f(x)||^2 = sum(f_i(x)^2, i = 1, ..., m)

    Where f is a vector function f = (f_1, ..., f_m), we call f_i as residuals.

    Jacobian of f is an m by n matrix, its (i, j) element is the partial
    derivative of f_i with respect to x_j.

    Parameters
    ----------
    n : int
        Number of optimized variables.
    m : int
        Number of residuals.
    x0 : ndarray, shape(n, )
        Initial guess for optimized variable.
    fopt : float
        The sum of squared residuals at the optimum point. It must be provided
        with the relative accuracy orders of magnitude higher than expected
        `ftol` parameter of benchmarked optimization method.
    lb : None or ndarray, shape(n, ), optional
        Lower bounds for each optimized variable, -np.inf specifies no bound.
        None means no bound for all variables.
    ub : None or ndarray, shape(n ), optional
        Upper bound for each optimized variable, np.inf specified no bound.
        None means no bound for all variables.

    Attributes
    ----------
    INITIAL_GUESSES : list of ndarray
        List containing initial guesses to try. Fill this list in a derived
        class with at least one item.
    """

    # 类属性，用于存储初始猜测值，可以在派生类中填充至少一个初始值
    INITIAL_GUESSES = None

    def __init__(self, n, m, fopt, x0, lb=None, ub=None):
        # 初始化函数，设置优化变量的维度、残差的个数、最优点处残差平方和、初始猜测值、边界条件
        self.n = n
        self.m = m
        self.fopt = fopt
        self.x0 = x0
        self.lb = lb
        self.ub = ub

    def fun(self, x):
        """Evaluate residuals at point `x`.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Point of evaluation.

        Returns
        -------
        ndarray, shape (m,)
            Vector of residuals at point `x`.
        """
        # 抽象方法，计算给定点 x 处的残差向量 f(x)，子类需实现具体功能
        raise NotImplementedError

    def jac(self, x):
        """Evaluate jacobian at point x.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Vector of residuals f(x).

        Returns
        -------
        ndarray, shape (m, n)
            Jacobian matrix of `self.fun` at point `x`.
        """
        # 抽象方法，计算给定点 x 处的雅可比矩阵，子类需实现具体功能
        raise NotImplementedError
    def check_answer(self, x, ftol):
        """Check if `x` yields the objective value close enough to
        the optimal value.
        
        Parameters
        ----------
        x : ndarray, shape (n,)
            The point to test.
        ftol : float
            Maximum allowed relative error in the objective function value.
        
        Returns
        -------
        bool
            Whether `x` is optimal enough. If `x` violates bounds constraints
            then False is returned.
        """
        # 检查 `x` 是否违反了下界约束或上界约束，如果违反则返回 False
        if (self.lb is not None and np.any(x < self.lb) or
                self.ub is not None and np.any(x > self.ub)):
            return False
        
        # 计算目标函数值的平方和
        f = np.sum(self.fun(x) ** 2)
        
        # 返回是否达到了允许的目标函数值的相对误差范围内的最优解
        return f < (1 + ftol) * self.fopt
class AlphaPineneDirect(LSQBenchmarkProblem):
    """Isomerization of alpha-pinene problem, direct formulation [1]_.

    Number of variables --- 5, number of residuals --- 40, no bounds.

    .. [1] Brett M. Averick et al. "The MINPACK-2 Test Problem Collection",
           p. 20
    """

    # 初始猜测的变量值，作为类变量
    INITIAL_GUESSES = [
        np.array([5.84, 2.65, 1.63, 27.77, 4.61]) * 1e-5
    ]

    def __init__(self, x0):
        # 调用父类 LSQBenchmarkProblem 的构造函数初始化对象
        super().__init__(5, 40, 2.064572e1, x0)
        
        # 时间点数组，单位为秒
        self.t = np.array([0, 1230, 3060, 4920, 7800, 10680, 15030, 22620,
                           36420], dtype=float)
        
        # 初始浓度数组，百分比
        self.y0 = np.array([100, 0, 0, 0, 0], dtype=float)
        
        # 观测到的浓度变化，每行表示一个时间点对应的浓度
        self.y = np.array([
            [100, 0, 0, 0, 0],
            [88.35, 7.3, 2.3, 0.4, 1.75],
            [76.4, 15.6, 4.5, 0.7, 2.8],
            [65.1, 23.1, 5.3, 1.1, 5.8],
            [50.4, 32.9, 6, 1.5, 9.3],
            [37.5, 42.7, 6.0, 1.9, 12],
            [25.9, 49.1, 5.9, 2.2, 17],
            [14, 57.4, 5.1, 2.6, 21],
            [4.5, 63.1, 3.8, 2.9, 25.7]
        ])

    def fun_ode_rhs(self, y, t, x):
        # 定义常微分方程的右侧函数
        return np.array(
            [-(x[0] + x[1]) * y[0],
             x[0] * y[0],
             x[1] * y[0] - (x[2] + x[3]) * y[2] + x[4] * y[4],
             x[2] * y[2],
             x[3] * y[2] - x[4] * y[4]]
        )

    def jac_ode_rhs(self, y, t, x):
        # 计算常微分方程右侧函数的雅可比矩阵
        jac_part = np.array(
            [-y[0], -y[0], 0, 0, 0,
             y[0], 0, 0, 0, 0,
             0, y[0], -y[2], -y[2], y[4],
             0, 0, y[2], 0, 0,
             0, 0, 0, y[2], -y[4]]
        )
        return np.hstack((self.fun_ode_rhs(y, t, x), jac_part))

    def fun(self, x):
        # 定义目标函数，通过数值积分求解常微分方程得到预测值，返回残差
        y_hat = odeint(self.fun_ode_rhs, self.y0, self.t, args=(x,))
        return y_hat[1:].ravel() - self.y[1:].ravel()

    def jac(self, x):
        # 定义目标函数的雅可比矩阵，通过数值积分求解雅可比矩阵
        result = odeint(self.jac_ode_rhs, np.hstack((self.y0, np.zeros(25))),
                        self.t, args=(x,))
        return result[1:, 5:].reshape((40, 5))


class CoatingThickness(LSQBenchmarkProblem):
    """Coating thickness standardization problem, [1]_.

    Number of variables --- 134, number of residuals --- 252, no bounds.

    .. [1] Brett M. Averick et al. "The MINPACK-2 Test Problem Collection",
           p. 25
    """

    # 初始猜测的变量值，作为类变量
    INITIAL_GUESSES = [
        np.hstack(([-8.0, 13.0, 1.2, 0.2, 0.1, 6.0, 5.5, -5.2],
                   np.zeros(126)))
    ]
    # 定义初始化方法，接受一个参数 x0
    def __init__(self, x0):
        # 调用父类的初始化方法，传入参数 134, 252, 0.5054986 和 x0
        super().__init__(134, 252, 0.5054986, x0)
        # 计算 self.m 的四分之一并赋值给 self.n0
        self.n0 = self.m // 4
        # 创建一个 2x80 的 NumPy 数组，并赋给 self.xi
        self.xi = np.array([
            [0.7140, 0.7169, 0.7232, 0.7151, 0.6848, 0.7070, 0.7177, 0.7073,
             0.6734, 0.7174, 0.7125, 0.6947, 0.7121, 0.7166, 0.6894, 0.6897,
             0.7024, 0.7026, 0.6800, 0.6957, 0.6987, 0.7111, 0.7097, 0.6809,
             0.7139, 0.7046, 0.6950, 0.7032, 0.7019, 0.6975, 0.6955, 0.7056,
             0.6965, 0.6848, 0.6995, 0.6105, 0.6027, 0.6084, 0.6081, 0.6057,
             0.6116, 0.6052, 0.6136, 0.6032, 0.6081, 0.6092, 0.6122, 0.6157,
             0.6191, 0.6169, 0.5483, 0.5371, 0.5576, 0.5521, 0.5495, 0.5499,
             0.4937, 0.5092, 0.5433, 0.5018, 0.5363, 0.4977, 0.5296],
            [5.145, 5.241, 5.389, 5.211, 5.154, 5.105, 5.191, 5.013, 5.582,
             5.208, 5.142, 5.284, 5.262, 6.838, 6.215, 6.817, 6.889, 6.732,
             6.717, 6.468, 6.776, 6.574, 6.465, 6.090, 6.350, 4.255, 4.154,
             4.211, 4.287, 4.104, 4.007, 4.261, 4.150, 4.040, 4.155, 5.086,
             5.021, 5.040, 5.247, 5.125, 5.136, 4.949, 5.253, 5.154, 5.227,
             5.120, 5.291, 5.294, 5.304, 5.209, 5.384, 5.490, 5.563, 5.532,
             5.372, 5.423, 7.237, 6.944, 6.957, 7.138, 7.009, 7.074, 7.046]
        ])
        # 创建一个包含 120 个元素的 NumPy 数组，并赋给 self.y
        self.y = np.array(
            [9.3636, 9.3512, 9.4891, 9.1888, 9.3161, 9.2585, 9.2913, 9.3914,
             9.4524, 9.4995, 9.4179, 9.468, 9.4799, 11.2917, 11.5062, 11.4579,
             11.3977, 11.3688, 11.3897, 11.3104, 11.3882, 11.3629, 11.3149,
             11.2474, 11.2507, 8.1678, 8.1017, 8.3506, 8.3651, 8.2994, 8.1514,
             8.2229, 8.1027, 8.3785, 8.4118, 8.0955, 8.0613, 8.0979, 8.1364,
             8.1700, 8.1684, 8.0885, 8.1839, 8.1478, 8.1827, 8.029, 8.1000,
             8.2579, 8.2248, 8.2540, 6.8518, 6.8547, 6.8831, 6.9137, 6.8984,
             6.8888, 8.5189, 8.5308, 8.5184, 8.5222, 8.5705, 8.5353, 8.5213,
             8.3158, 8.1995, 8.2283, 8.1857, 8.2738, 8.2131, 8.2613, 8.2315,
             8.2078, 8.2996, 8.3026, 8.0995, 8.2990, 9.6753, 9.6687, 9.5704,
             9.5435, 9.6780, 9.7668, 9.7827, 9.7844, 9.7011, 9.8006, 9.7610,
             9.7813, 7.3073, 7.2572, 7.4686, 7.3659, 7.3587, 7.3132, 7.3542,
             7.2339, 7.4375, 7.4022, 10.7914, 10.6554, 10.7359, 10.7583,
             10.7735, 10.7907, 10.6465, 10.6994, 10.7756, 10.7402, 10.6800,
             10.7000, 10.8160, 10.6921, 10.8677, 12.3495, 12.4424, 12.4303,
             12.5086, 12.4513, 12.4625, 16.2290, 16.2781, 16.2082, 16.2715,
             16.2464, 16.1626, 16.1568]
        )
        # 设置缩放因子 self.scale1 和 self.scale2
        self.scale1 = 4.08
        self.scale2 = 0.417
    # 定义一个方法 `fun`，接受参数 `self` 和 `x`
    def fun(self, x):
        # 构建一个列向量 xi，由两部分组成：
        # 第一部分：self.xi[0] 加上 x[8:8+self.n0] 的部分
        # 第二部分：self.xi[1] 加上 x[8+self.n0:] 的部分
        xi = np.vstack(
            (self.xi[0] + x[8:8 + self.n0],
             self.xi[1] + x[8 + self.n0:])
        )
        # 计算 z1，使用以下公式：
        # z1 = x[0] + x[1] * xi[0] + x[2] * xi[1] + x[3] * xi[0] * xi[1]
        z1 = x[0] + x[1] * xi[0] + x[2] * xi[1] + x[3] * xi[0] * xi[1]
        # 计算 z2，使用以下公式：
        # z2 = x[4] + x[5] * xi[0] + x[6] * xi[1] + x[7] * xi[0] * xi[1]
        z2 = x[4] + x[5] * xi[0] + x[6] * xi[1] + x[7] * xi[0] * xi[1]
        # 返回一个行向量，包含以下部分：
        # z1 减去 self.y 的前 self.n0 个元素
        # z2 减去 self.y 的后 self.n0 个元素
        # x[8:8+self.n0] 乘以 self.scale1
        # x[8+self.n0:] 乘以 self.scale2
        return np.hstack(
            (z1 - self.y[:self.n0],
             z2 - self.y[self.n0:],
             self.scale1 * x[8:8 + self.n0],
             self.scale2 * x[8 + self.n0:])
        )

    # 定义一个方法 `jac`，接受参数 `self` 和 `x`
    def jac(self, x):
        # 创建一个零矩阵 J，大小为 self.m 行，self.n 列
        J = np.zeros((self.m, self.n))
        # 创建一个包含从 0 到 self.n0-1 的整数数组
        ind = np.arange(self.n0)
        # 构建一个列向量 xi，由两部分组成：
        # 第一部分：self.xi[0] 加上 x[8:8+self.n0] 的部分
        # 第二部分：self.xi[1] 加上 x[8+self.n0:] 的部分
        xi = np.vstack(
            (self.xi[0] + x[8:8 + self.n0],
             self.xi[1] + x[8 + self.n0:])
        )
        # 设置 J 的部分元素值：
        # 第 0 至 self.n0-1 行，第 0 列设为 1
        J[:self.n0, 0] = 1
        # 第 0 至 self.n0-1 行，第 1 列设为 xi[0]
        J[:self.n0, 1] = xi[0]
        # 第 0 至 self.n0-1 行，第 2 列设为 xi[1]
        J[:self.n0, 2] = xi[1]
        # 第 0 至 self.n0-1 行，第 3 列设为 xi[0] * xi[1]
        J[:self.n0, 3] = xi[0] * xi[1]
        # 在 J 中，使用 ind 作为行索引和 ind+8 作为列索引，设定元素值：
        # 第 0 至 self.n0-1 行，第 8+ind 列设为 x[1] + x[3] * xi[1]
        J[ind, ind + 8] = x[1] + x[3] * xi[1]
        # 在 J 中，使用 ind 作为行索引和 ind+8+self.n0 作为列索引，设定元素值：
        # 第 0 至 self.n0-1 行，第 8+self.n0+ind 列设为 x[2] + x[3] * xi[0]
        J[ind, ind + 8 + self.n0] = x[2] + x[3] * xi[0]

        # 在 J 中，使用 self.n0 到 2*self.n0-1 行，第 4 列设为 1
        J[self.n0:2 * self.n0, 4] = 1
        # 在 J 中，使用 self.n0 到 2*self.n0-1 行，第 5 列设为 xi[0]
        J[self.n0:2 * self.n0, 5] = xi[0]
        # 在 J 中，使用 self.n0 到 2*self.n0-1 行，第 6 列设为 xi[1]
        J[self.n0:2 * self.n0, 6] = xi[1]
        # 在 J 中，使用 self.n0 到 2*self.n0-1 行，第 7 列设为 xi[0] * xi[1]
        J[self.n0:2 * self.n0, 7] = xi[0] * xi[1]
        # 在 J 中，使用 self.n0+ind 作为行索引和 ind+8 作为列索引，设定元素值：
        # 第 self.n0 至 2*self.n0-1 行，第 8+ind 列设为 x[5] + x[7] * xi[1]
        J[ind + self.n0, ind + 8] = x[5] + x[7] * xi[1]
        # 在 J 中，使用 self.n0+ind 作为行索引和 ind+8+self.n0 作为列索引，设定元素值：
        # 第 self.n0 至 2*self.n0-1 行，第 8+self.n0+ind 列设为 x[6] + x[7] * xi[0]
        J[ind + self.n0, ind + 8 + self.n0] = x[6] + x[7] * xi[0]

        # 在 J 中，使用 self.n0+2*ind 作为行索引和 ind+8 作为列索引，设定元素值：
        # 第 2*self.n0 至 3*self.n0-1 行，第 8+ind 列设为 self.scale1
        J[ind + 2 * self.n0, ind + 8] = self.scale1
        # 在 J 中，使用 self.n0+3*ind 作为行索引和 ind+8+self.n0 作为列索引，设定元素值：
        # 第 3*self.n0 至 4*self.n0-1 行，第 8+self.n0+ind 列设为 self.scale2
        J[ind + 3 * self.n0, ind + 8 + self.n0] = self.scale2

        # 返回计算好的 Jacobian 矩阵 J
        return J
class ExponentialFitting(LSQBenchmarkProblem):
    """The problem of fitting the sum of exponentials with linear degrees
    to data, [1]_.

    Number of variables --- 5, number of residuals --- 33, no bounds.

    .. [1] Brett M. Averick et al. "The MINPACK-2 Test Problem Collection",
           p. 26
    """

    # 初始猜测值列表，包含一个 NumPy 数组作为元素
    INITIAL_GUESSES = [
        np.array([0.5, 1.5, -1, 1e-2, 2e-2])
    ]

    def __init__(self, x0):
        # 调用父类 LSQBenchmarkProblem 的构造函数，传递参数
        super().__init__(5, 33, 5.464895e-5, x0)
        # 创建时间序列 t，用于计算函数值
        self.t = np.arange(self.m, dtype=float) * 10
        # 创建观测值数组 y，用于函数拟合
        self.y = 1e-1 * np.array(
            [8.44, 9.08, 9.32, 9.36, 9.25, 9.08, 8.81, 8.5, 8.18,
             7.84, 7.51, 7.18, 6.85, 6.58, 6.28, 6.03, 5.8, 5.58,
             5.38, 5.22, 5.06, 4.9, 4.78, 4.67, 4.57, 4.48, 4.38,
             4.31, 4.24, 4.2, 4.14, 4.11, 4.06]
        )

    def fun(self, x):
        # 定义拟合函数的误差计算方法
        return (x[0] + x[1] * np.exp(-x[3] * self.t) +
                x[2] * np.exp(-x[4] * self.t) - self.y)

    def jac(self, x):
        # 创建一个空的雅可比矩阵 J，大小为 m x n
        J = np.empty((self.m, self.n))
        # 填充雅可比矩阵 J 的各列
        J[:, 0] = 1
        J[:, 1] = np.exp(-x[3] * self.t)
        J[:, 2] = np.exp(-x[4] * self.t)
        J[:, 3] = -x[1] * self.t * np.exp(-x[3] * self.t)
        J[:, 4] = -x[2] * self.t * np.exp(-x[4] * self.t)
        return J


class GaussianFitting(LSQBenchmarkProblem):
    """The problem of fitting the sum of exponentials with linear and
    quadratic degrees to data, [1]_.

    Number of variables --- 11, number of residuals --- 65, no bounds.

    .. [1] Brett M. Averick et al. "The MINPACK-2 Test Problem Collection",
           p. 27
    """
    
    # 初始猜测值列表，包含一个 NumPy 数组作为元素
    INITIAL_GUESSES = [
        np.array([1.3, 6.5e-1, 6.5e-1, 7.0e-1, 6.0e-1,
                  3.0, 5.0, 7.0, 2.0, 4.5, 5.5])
    ]

    def __init__(self, x0):
        # 调用父类 LSQBenchmarkProblem 的构造函数，传递参数
        super().__init__(11, 65, 4.013772e-02, x0)
        # 创建时间序列 t，用于计算函数值
        self.t = np.arange(self.m, dtype=float) * 1e-1
        # 创建观测值数组 y，用于函数拟合
        self.y = np.array(
            [1.366, 1.191, 1.112, 1.013, 9.91e-1, 8.85e-1, 8.31e-1, 8.47e-1,
             7.86e-1, 7.25e-1, 7.46e-1, 6.79e-1, 6.08e-1, 6.55e-1, 6.16e-1,
             6.06e-1, 6.02e-1, 6.26e-1, 6.51e-1, 7.24e-1, 6.49e-1, 6.49e-1,
             6.94e-1, 6.44e-1, 6.24e-1, 6.61e-1, 6.12e-1, 5.58e-1, 5.33e-1,
             4.95e-1, 5.0e-1, 4.23e-1, 3.95e-1, 3.75e-1, 3.72e-1, 3.91e-1,
             3.96e-1, 4.05e-1, 4.28e-1, 4.29e-1, 5.23e-1, 5.62e-1, 6.07e-1,
             6.53e-1, 6.72e-1, 7.08e-1, 6.33e-1, 6.68e-1, 6.45e-1, 6.32e-1,
             5.91e-1, 5.59e-1, 5.97e-1, 6.25e-1, 7.39e-1, 7.1e-1, 7.29e-1,
             7.2e-1, 6.36e-1, 5.81e-1, 4.28e-1, 2.92e-1, 1.62e-1, 9.8e-2,
             5.4e-2]
        )

    def fun(self, x):
        # 定义拟合函数的误差计算方法
        return (x[0] * np.exp(-x[4] * self.t) +
                x[1] * np.exp(-x[5] * (self.t - x[8]) ** 2) +
                x[2] * np.exp(-x[6] * (self.t - x[9]) ** 2) +
                x[3] * np.exp(-x[7] * (self.t - x[10]) ** 2) - self.y)
    # 定义 Jacobian 矩阵计算函数，接受参数 self 和 x
    def jac(self, x):
        # 创建一个空的 numpy 数组 J，形状为 (self.m, self.n)
        J = np.empty((self.m, self.n))
        # 计算指数函数的结果，分别为 e0 到 e3
        e0 = np.exp(-x[4] * self.t)
        e1 = np.exp(-x[5] * (self.t - x[8]) ** 2)
        e2 = np.exp(-x[6] * (self.t - x[9]) ** 2)
        e3 = np.exp(-x[7] * (self.t - x[10]) ** 2)
        # 将计算结果赋值给 Jacobian 矩阵 J 的不同列
        J[:, 0] = e0
        J[:, 1] = e1
        J[:, 2] = e2
        J[:, 3] = e3
        # 计算 J 的剩余列，根据特定公式和参数 x 的不同元素
        J[:, 4] = -x[0] * self.t * e0
        J[:, 5] = -x[1] * (self.t - x[8]) ** 2 * e1
        J[:, 6] = -x[2] * (self.t - x[9]) ** 2 * e2
        J[:, 7] = -x[3] * (self.t - x[10]) ** 2 * e3
        J[:, 8] = 2 * x[1] * x[5] * (self.t - x[8]) * e1
        J[:, 9] = 2 * x[2] * x[6] * (self.t - x[9]) * e2
        J[:, 10] = 2 * x[3] * x[7] * (self.t - x[10]) * e3
        # 返回计算得到的 Jacobian 矩阵 J
        return J
class ThermistorResistance(LSQBenchmarkProblem):
    """The problem of fitting thermistor parameters to data, [1]_.

    Number of variables --- 3, number of residuals --- 16, no bounds.

    .. [1] Brett M. Averick et al. "The MINPACK-2 Test Problem Collection",
           p. 28
    """

    INITIAL_GUESSES = [
        np.array([2e-2, 4e3, 2.5e2])
    ]
    
    # 初始化函数，接收参数 x0_ind
    def __init__(self, x0_ind):
        # 调用父类 LSQBenchmarkProblem 的初始化方法，传递参数 3, 16, 87.94585, x0_ind
        super().__init__(3, 16, 87.94585, x0_ind)
        # 设置温度数组 t，使用 numpy 创建等差数列，长度为 self.m
        self.t = 5 + 45 * (1 + np.arange(self.m, dtype=float))
        # 设置测量值数组 y，使用 numpy 创建浮点数数组
        self.y = np.array(
            [3.478e4, 2.861e4, 2.365e4, 1.963e4, 1.637e4, 1.372e4, 1.154e4,
             9.744e3, 8.261e3, 7.03e3, 6.005e3, 5.147e3, 4.427e3, 3.82e3,
             3.307e3, 2.872e3]
        )

    # 定义函数 fun，接收参数 x，计算并返回拟合函数与测量值的残差
    def fun(self, x):
        return x[0] * np.exp(x[1] / (self.t + x[2])) - self.y

    # 定义函数 jac，接收参数 x，计算并返回雅可比矩阵
    def jac(self, x):
        J = np.empty((self.m, self.n))
        e = np.exp(x[1] / (self.t + x[2]))
        J[:, 0] = e
        J[:, 1] = x[0] / (self.t + x[2]) * e
        J[:, 2] = -x[0] * x[1] * (self.t + x[2]) ** -2 * e
        return J


class EnzymeReaction(LSQBenchmarkProblem):
    """The problem of fitting kinetic parameters for an enzyme reaction, [1]_.

    Number of variables --- 4, number of residuals --- 11, no bounds.

    .. [1] Brett M. Averick et al. "The MINPACK-2 Test Problem Collection",
           p. 29
    """

    INITIAL_GUESSES = [
        np.array([2.5, 3.9, 4.15, 3.9]) * 1e-1
    ]
    
    # 初始化函数，接收参数 x0_ind
    def __init__(self, x0_ind):
        # 调用父类 LSQBenchmarkProblem 的初始化方法，传递参数 4, 11, 3.075057e-04, x0_ind
        super().__init__(4, 11, 3.075057e-04, x0_ind)
        # 设置浓度数组 u，使用 numpy 创建浮点数数组
        self.u = np.array([4.0, 2.0, 1.0, 5.0e-1, 2.5e-1, 1.67e-1,
                           1.25e-1, 1.0e-1, 8.33e-2, 7.14e-2, 6.25e-2])
        # 设置反应速率数组 y，使用 numpy 创建浮点数数组
        self.y = np.array([1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2,
                           6.27e-2, 4.56e-2, 3.42e-2, 3.23e-2, 2.35e-2,
                           2.46e-2])

    # 定义函数 fun，接收参数 x，计算并返回拟合函数与测量值的残差
    def fun(self, x):
        return (x[0] * (self.u ** 2 + x[1] * self.u) /
                (self.u ** 2 + x[2] * self.u + x[3]) - self.y)

    # 定义函数 jac，接收参数 x，计算并返回雅可比矩阵
    def jac(self, x):
        J = np.empty((self.m, self.n))
        den = self.u ** 2 + x[2] * self.u + x[3]
        num = self.u ** 2 + x[1] * self.u
        J[:, 0] = num / den
        J[:, 1] = x[0] * self.u / den
        J[:, 2] = -x[0] * num * self.u / den ** 2
        J[:, 3] = -x[0] * num / den ** 2
        return J


class ChebyshevQuadrature(LSQBenchmarkProblem):
    """The problem of determining the optimal nodes of a quadrature formula
     with equal weights, [1]_.

    Number of variables --- 11, number of residuals --- 11, no bounds.

    .. [1] Brett M. Averick et al. "The MINPACK-2 Test Problem Collection",
           p. 30
    """

    INITIAL_GUESSES = [
        (1 + np.arange(11, dtype=float)) / 12
    ]
    
    # 初始化函数，接收参数 x0
    def __init__(self, x0):
        # 调用父类 LSQBenchmarkProblem 的初始化方法，传递参数 11, 11, 2.799761e-03, x0
        super().__init__(11, 11, 2.799761e-03, x0)
        # 创建 Chebyshev 类的实例 cp，domain 参数为 [0.0, 1.0]
        cp = Chebyshev(1)
        # 设置 Chebyshev 多项式基函数列表 T_all，包含从 cp 中获取的各个基函数
        self.T_all = [cp.basis(i, domain=[0.0, 1.0]) for i in range(11)]
    # 定义一个方法 `fun`，接受参数 `x`
    def fun(self, x):
        # 创建一个长度为 `self.n` 的空数组 `f`
        f = np.empty(self.n)
        # 遍历范围为 `self.m` 的索引
        for i in range(self.m):
            # 获取 `self.T_all` 列表中的第 `i` 个元素，并赋值给 `T`
            T = self.T_all[i]
            # 计算 `T(x)` 的平均值，并减去 `T` 对 `[0.0, 1.0]` 区间的积分结果，将结果保存到 `f[i]`
            f[i] = np.mean(T(x)) - T.integ(lbnd=0.0)(1.0)
        # 返回数组 `f`
        return f

    # 定义一个方法 `jac`，接受参数 `x`
    def jac(self, x):
        # 创建一个形状为 `(self.m, self.n)` 的空数组 `J`
        J = np.empty((self.m, self.n))
        # 遍历范围为 `self.m` 的索引
        for i in range(self.m):
            # 获取 `self.T_all` 列表中的第 `i` 个元素，并赋值给 `T`
            T = self.T_all[i]
            # 计算 `T(x)` 的导数，并将结果保存到 `J[i]` 中
            J[i] = T.deriv()(x)
        # 对数组 `J` 中的每个元素除以 `self.n`
        J /= self.n
        # 返回数组 `J`
        return J
# 定义一个函数，用于从当前文件中提取所有用于基准测试的最小二乘问题。
def extract_lsq_problems():
    """Extract all least squares problems in this file for benchmarking.

    Returns
    -------
    dict, str -> LSQBenchmarkProblem
        The key is a problem name.
        The value is an instance of LSQBenchmarkProblem.
    """
    # 初始化一个空字典，用于存储问题名和相应的 LSQBenchmarkProblem 实例
    problems = {}
    
    # 使用 inspect 模块的 getmembers 函数遍历当前模块中的所有类
    for name, problem_class in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        # 检查条件：
        # 1. 类名不是 "LSQBenchmarkProblem"
        # 2. 类是 LSQBenchmarkProblem 的子类
        # 3. 类具有名为 'INITIAL_GUESSES' 的属性
        if (name != "LSQBenchmarkProblem" and
            issubclass(problem_class, LSQBenchmarkProblem) and
            hasattr(problem_class, 'INITIAL_GUESSES')):
            
            # 遍历该类的 INITIAL_GUESSES 属性
            for i, x0 in enumerate(problem_class.INITIAL_GUESSES):
                # 如果 INITIAL_GUESSES 中有多个初始猜测，则在问题名中加上索引号
                if len(problem_class.INITIAL_GUESSES) > 1:
                    key_name = f"{name}_{i}"
                else:
                    key_name = name
                
                # 创建该类的实例并存储到 problems 字典中
                problems[key_name] = problem_class(x0)
    
    # 返回存储了问题名和对应 LSQBenchmarkProblem 实例的字典
    return problems
```