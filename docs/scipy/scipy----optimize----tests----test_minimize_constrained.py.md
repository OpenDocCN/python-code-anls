# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_minimize_constrained.py`

```
import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (assert_array_almost_equal,
                           assert_array_less, assert_, assert_allclose,
                           suppress_warnings)
from scipy.optimize import (NonlinearConstraint,
                            LinearConstraint,
                            Bounds,
                            minimize,
                            BFGS,
                            SR1,
                            rosen)

class Maratos:
    """Problem 15.4 from Nocedal and Wright

    The following optimization problem:
        minimize 2*(x[0]**2 + x[1]**2 - 1) - x[0]
        Subject to: x[0]**2 + x[1]**2 - 1 = 0
    """

    def __init__(self, degrees=60, constr_jac=None, constr_hess=None):
        # 将角度转换为弧度
        rads = degrees/180*np.pi
        # 初始化起始点和最优解
        self.x0 = [np.cos(rads), np.sin(rads)]
        self.x_opt = np.array([1.0, 0.0])
        self.constr_jac = constr_jac
        self.constr_hess = constr_hess
        self.bounds = None

    def fun(self, x):
        # 目标函数定义
        return 2*(x[0]**2 + x[1]**2 - 1) - x[0]

    def grad(self, x):
        # 目标函数的梯度向量
        return np.array([4*x[0]-1, 4*x[1]])

    def hess(self, x):
        # 目标函数的黑塞矩阵
        return 4*np.eye(2)

    @property
    def constr(self):
        # 约束条件的定义
        def fun(x):
            return x[0]**2 + x[1]**2

        if self.constr_jac is None:
            # 如果未提供雅可比矩阵，则默认定义它
            def jac(x):
                return [[2*x[0], 2*x[1]]]
        else:
            jac = self.constr_jac

        if self.constr_hess is None:
            # 如果未提供黑塞矩阵，则默认定义它
            def hess(x, v):
                return 2*v[0]*np.eye(2)
        else:
            hess = self.constr_hess

        return NonlinearConstraint(fun, 1, 1, jac, hess)


class MaratosTestArgs:
    """Problem 15.4 from Nocedal and Wright

    The following optimization problem:
        minimize 2*(x[0]**2 + x[1]**2 - 1) - x[0]
        Subject to: x[0]**2 + x[1]**2 - 1 = 0
    """

    def __init__(self, a, b, degrees=60, constr_jac=None, constr_hess=None):
        # 将角度转换为弧度
        rads = degrees/180*np.pi
        # 初始化起始点、最优解和其他参数
        self.x0 = [np.cos(rads), np.sin(rads)]
        self.x_opt = np.array([1.0, 0.0])
        self.constr_jac = constr_jac
        self.constr_hess = constr_hess
        self.a = a
        self.b = b
        self.bounds = None

    def _test_args(self, a, b):
        # 检查参数 a 和 b 是否匹配实例中的值，不匹配则抛出异常
        if self.a != a or self.b != b:
            raise ValueError()

    def fun(self, x, a, b):
        # 检查参数 a 和 b，然后计算目标函数值
        self._test_args(a, b)
        return 2*(x[0]**2 + x[1]**2 - 1) - x[0]

    def grad(self, x, a, b):
        # 检查参数 a 和 b，然后计算目标函数的梯度向量
        self._test_args(a, b)
        return np.array([4*x[0]-1, 4*x[1]])

    def hess(self, x, a, b):
        # 检查参数 a 和 b，然后计算目标函数的黑塞矩阵
        self._test_args(a, b)
        return 4*np.eye(2)

    @property
    def constr(self):
        # 约束条件的定义，与 Maratos 类中的 constr 方法类似，略有不同
        def fun(x):
            return x[0]**2 + x[1]**2

        if self.constr_jac is None:
            def jac(x):
                return [[2*x[0], 2*x[1]]]
        else:
            jac = self.constr_jac

        if self.constr_hess is None:
            def hess(x, v):
                return 2*v[0]*np.eye(2)
        else:
            hess = self.constr_hess

        return NonlinearConstraint(fun, 1, 1, jac, hess)
    # 定义一个方法 `constr`，用于构造非线性约束条件对象
    def constr(self):
        # 定义内部函数 `fun`，计算输入向量 x 的平方和
        def fun(x):
            return x[0]**2 + x[1]**2

        # 如果未提供约束的雅可比矩阵 `constr_jac`
        if self.constr_jac is None:
            # 定义内部函数 `jac`，计算给定点 x 的雅可比矩阵
            def jac(x):
                return [[4*x[0], 4*x[1]]]
        else:
            # 使用外部提供的 `constr_jac` 函数作为雅可比矩阵函数
            jac = self.constr_jac

        # 如果未提供约束的黑塞矩阵 `constr_hess`
        if self.constr_hess is None:
            # 定义内部函数 `hess`，计算给定点 x 和向量 v 的黑塞矩阵
            def hess(x, v):
                return 2*v[0]*np.eye(2)
        else:
            # 使用外部提供的 `constr_hess` 函数作为黑塞矩阵函数
            hess = self.constr_hess

        # 返回一个 NonlinearConstraint 对象，约束条件为 fun(x) = 1，Jacobi 矩阵为 jac，黑塞矩阵为 hess
        return NonlinearConstraint(fun, 1, 1, jac, hess)
class MaratosGradInFunc:
    """Problem 15.4 from Nocedal and Wright

    The following optimization problem:
        minimize 2*(x[0]**2 + x[1]**2 - 1) - x[0]
        Subject to: x[0]**2 + x[1]**2 - 1 = 0
    """

    def __init__(self, degrees=60, constr_jac=None, constr_hess=None):
        # 将角度转换为弧度
        rads = degrees/180*np.pi
        # 初始点 x0 是单位圆上的一个点，角度为 degrees
        self.x0 = [np.cos(rads), np.sin(rads)]
        # 最优解 x_opt 是 (1.0, 0.0)
        self.x_opt = np.array([1.0, 0.0])
        # 约束的雅可比矩阵和黑塞矩阵
        self.constr_jac = constr_jac
        self.constr_hess = constr_hess
        # 不设定变量的边界
        self.bounds = None

    def fun(self, x):
        # 目标函数和其梯度
        return (2*(x[0]**2 + x[1]**2 - 1) - x[0],
                np.array([4*x[0]-1, 4*x[1]]))

    @property
    def grad(self):
        # 声明梯度是可用的
        return True

    def hess(self, x):
        # 黑塞矩阵恒为4乘以单位矩阵
        return 4*np.eye(2)

    @property
    def constr(self):
        # 约束条件函数及其雅可比矩阵和黑塞矩阵
        def fun(x):
            return x[0]**2 + x[1]**2

        if self.constr_jac is None:
            def jac(x):
                return [[4*x[0], 4*x[1]]]
        else:
            jac = self.constr_jac

        if self.constr_hess is None:
            def hess(x, v):
                return 2*v[0]*np.eye(2)
        else:
            hess = self.constr_hess

        return NonlinearConstraint(fun, 1, 1, jac, hess)


class HyperbolicIneq:
    """Problem 15.1 from Nocedal and Wright

    The following optimization problem:
        minimize 1/2*(x[0] - 2)**2 + 1/2*(x[1] - 1/2)**2
        Subject to: 1/(x[0] + 1) - x[1] >= 1/4
                                   x[0] >= 0
                                   x[1] >= 0
    """
    def __init__(self, constr_jac=None, constr_hess=None):
        # 初始点 x0 是 (0, 0)
        self.x0 = [0, 0]
        # 最优解 x_opt 是 (1.952823, 0.088659)
        self.x_opt = [1.952823, 0.088659]
        # 约束的雅可比矩阵和黑塞矩阵
        self.constr_jac = constr_jac
        self.constr_hess = constr_hess
        # 变量 x[0] 和 x[1] 的边界为大于等于0
        self.bounds = Bounds(0, np.inf)

    def fun(self, x):
        # 目标函数
        return 1/2*(x[0] - 2)**2 + 1/2*(x[1] - 1/2)**2

    def grad(self, x):
        # 目标函数的梯度
        return [x[0] - 2, x[1] - 1/2]

    def hess(self, x):
        # 黑塞矩阵是单位矩阵
        return np.eye(2)

    @property
    def constr(self):
        # 约束条件函数及其雅可比矩阵和黑塞矩阵
        def fun(x):
            return 1/(x[0] + 1) - x[1]

        if self.constr_jac is None:
            def jac(x):
                return [[-1/(x[0] + 1)**2, -1]]
        else:
            jac = self.constr_jac

        if self.constr_hess is None:
            def hess(x, v):
                return 2*v[0]*np.array([[1/(x[0] + 1)**3, 0],
                                        [0, 0]])
        else:
            hess = self.constr_hess

        return NonlinearConstraint(fun, 0.25, np.inf, jac, hess)


class Rosenbrock:
    """Rosenbrock function.

    The following optimization problem:
        minimize sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    """

    def __init__(self, n=2, random_state=0):
        # 初始点 x0 是从均匀分布中随机生成的
        rng = np.random.RandomState(random_state)
        self.x0 = rng.uniform(-1, 1, n)
        # 最优解 x_opt 是全为1的向量
        self.x_opt = np.ones(n)
        # 不设定变量的边界
        self.bounds = None
    # 定义一个函数 `fun`，计算给定向量 x 的特定函数值
    def fun(self, x):
        # 将输入 x 转换为 NumPy 数组
        x = np.asarray(x)
        # 计算特定函数 100 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0 的总和
        r = np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0,
                   axis=0)
        return r

    # 定义一个函数 `grad`，计算给定向量 x 的梯度
    def grad(self, x):
        # 将输入 x 转换为 NumPy 数组
        x = np.asarray(x)
        # 取中间部分的子数组 xm
        xm = x[1:-1]
        # 取中间部分的子数组 xm_m1 和 xm_p1
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        # 初始化梯度数组
        der = np.zeros_like(x)
        # 计算梯度值，赋给相应的索引位置
        der[1:-1] = (200 * (xm - xm_m1**2) -
                     400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm))
        # 计算第一个元素的梯度
        der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        # 计算最后一个元素的梯度
        der[-1] = 200 * (x[-1] - x[-2]**2)
        return der

    # 定义一个函数 `hess`，计算给定向量 x 的 Hessian 矩阵
    def hess(self, x):
        # 将输入 x 至少转换为一维 NumPy 数组
        x = np.atleast_1d(x)
        # 初始化 Hessian 矩阵 H
        H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
        # 初始化对角线数组 diagonal
        diagonal = np.zeros(len(x), dtype=x.dtype)
        # 计算对角线元素的值
        diagonal[0] = 1200 * x[0]**2 - 400 * x[1] + 2
        diagonal[-1] = 200
        diagonal[1:-1] = 202 + 1200 * x[1:-1]**2 - 400 * x[2:]
        # 更新 Hessian 矩阵 H
        H = H + np.diag(diagonal)
        return H

    # 定义一个只读属性 `constr`，返回空元组
    @property
    def constr(self):
        return ()
class IneqRosenbrock(Rosenbrock):
    """Rosenbrock subject to inequality constraints.

    The following optimization problem:
        minimize sum(100.0*(x[1] - x[0]**2)**2.0 + (1 - x[0])**2)
        subject to: x[0] + 2 x[1] <= 1

    Taken from matlab ``fmincon`` documentation.
    """
    def __init__(self, random_state=0):
        # 调用父类构造函数初始化 Rosenbrock 函数，传入维度和随机种子
        Rosenbrock.__init__(self, 2, random_state)
        # 设置初始点
        self.x0 = [-1, -0.5]
        # 最优解（用于比较优化结果）
        self.x_opt = [0.5022, 0.2489]
        # 约束条件设为 None，即没有显式的边界设定
        self.bounds = None

    @property
    def constr(self):
        # 设置不等式约束的系数矩阵 A 和右侧向量 b
        A = [[1, 2]]
        b = 1
        # 返回线性约束对象，表示 x[0] + 2*x[1] <= 1
        return LinearConstraint(A, -np.inf, b)



class BoundedRosenbrock(Rosenbrock):
    """Rosenbrock subject to inequality constraints.

    The following optimization problem:
        minimize sum(100.0*(x[1] - x[0]**2)**2.0 + (1 - x[0])**2)
        subject to:  -2 <= x[0] <= 0
                      0 <= x[1] <= 2

    Taken from matlab ``fmincon`` documentation.
    """
    def __init__(self, random_state=0):
        # 调用父类构造函数初始化 Rosenbrock 函数，传入维度和随机种子
        Rosenbrock.__init__(self, 2, random_state)
        # 设置初始点
        self.x0 = [-0.2, 0.2]
        # 最优解设为 None，即未指定
        self.x_opt = None
        # 设置边界条件，x[0] 在 [-2, 0] 之间，x[1] 在 [0, 2] 之间
        self.bounds = Bounds([-2, 0], [0, 2])



class EqIneqRosenbrock(Rosenbrock):
    """Rosenbrock subject to equality and inequality constraints.

    The following optimization problem:
        minimize sum(100.0*(x[1] - x[0]**2)**2.0 + (1 - x[0])**2)
        subject to: x[0] + 2 x[1] <= 1
                    2 x[0] + x[1] = 1

    Taken from matlab ``fimincon`` documentation.
    """
    def __init__(self, random_state=0):
        # 调用父类构造函数初始化 Rosenbrock 函数，传入维度和随机种子
        Rosenbrock.__init__(self, 2, random_state)
        # 设置初始点
        self.x0 = [-1, -0.5]
        # 最优解
        self.x_opt = [0.41494, 0.17011]
        # 约束条件设为 None，即没有显式的边界设定
        self.bounds = None

    @property
    def constr(self):
        # 设置不等式约束的系数矩阵 A_ineq 和右侧向量 b_ineq
        A_ineq = [[1, 2]]
        b_ineq = 1
        # 设置等式约束的系数矩阵 A_eq 和右侧向量 b_eq
        A_eq = [[2, 1]]
        b_eq = 1
        # 返回两个线性约束对象，表示 x[0] + 2*x[1] <= 1 和 2*x[0] + x[1] = 1
        return (LinearConstraint(A_ineq, -np.inf, b_ineq),
                LinearConstraint(A_eq, b_eq, b_eq))



class Elec:
    """Distribution of electrons on a sphere.

    Problem no 2 from COPS collection [2]_. Find
    the equilibrium state distribution (of minimal
    potential) of the electrons positioned on a
    conducting sphere.

    References
    ----------
    .. [1] E. D. Dolan, J. J. Mor\'{e}, and T. S. Munson,
           "Benchmarking optimization software with COPS 3.0.",
            Argonne National Lab., Argonne, IL (US), 2004.
    """
    def __init__(self, n_electrons=200, random_state=0,
                 constr_jac=None, constr_hess=None):
        # 设置电子数量和随机种子
        self.n_electrons = n_electrons
        self.rng = np.random.RandomState(random_state)
        # 初始化电子在球面上的初始位置
        phi = self.rng.uniform(0, 2 * np.pi, self.n_electrons)
        theta = self.rng.uniform(-np.pi, np.pi, self.n_electrons)
        x = np.cos(theta) * np.cos(phi)
        y = np.cos(theta) * np.sin(phi)
        z = np.sin(theta)
        self.x0 = np.hstack((x, y, z))  # 将坐标合并为初始点向量
        self.x_opt = None  # 最优解设为 None，即未指定
        self.constr_jac = constr_jac  # 约束条件的雅可比矩阵，默认为 None
        self.constr_hess = constr_hess  # 约束条件的黑塞矩阵，默认为 None
        self.bounds = None  # 边界条件设为 None，即没有显式的边界设定
    # 定义一个方法用于从输入向量 x 中提取坐标信息，并返回 x 坐标、y 坐标和 z 坐标
    def _get_cordinates(self, x):
        x_coord = x[:self.n_electrons]  # 提取 x 坐标，取前 self.n_electrons 个元素
        y_coord = x[self.n_electrons:2 * self.n_electrons]  # 提取 y 坐标，取中间 self.n_electrons 个元素
        z_coord = x[2 * self.n_electrons:]  # 提取 z 坐标，取剩余的元素
        return x_coord, y_coord, z_coord

    # 定义一个方法用于计算坐标差值，并返回 dx、dy、dz
    def _compute_coordinate_deltas(self, x):
        x_coord, y_coord, z_coord = self._get_cordinates(x)
        dx = x_coord[:, None] - x_coord  # 计算 x 坐标的差值矩阵
        dy = y_coord[:, None] - y_coord  # 计算 y 坐标的差值矩阵
        dz = z_coord[:, None] - z_coord  # 计算 z 坐标的差值矩阵
        return dx, dy, dz

    # 定义一个方法用于计算某个函数 fun 在输入向量 x 处的值，并返回结果
    def fun(self, x):
        dx, dy, dz = self._compute_coordinate_deltas(x)
        with np.errstate(divide='ignore'):
            dm1 = (dx**2 + dy**2 + dz**2) ** -0.5  # 计算坐标差值的模的倒数，避免除零错误
        dm1[np.diag_indices_from(dm1)] = 0  # 对角线元素置为零，确保不影响结果
        return 0.5 * np.sum(dm1)  # 返回计算得到的函数值的一半

    # 定义一个方法用于计算某个函数 grad 在输入向量 x 处的梯度，并返回结果
    def grad(self, x):
        dx, dy, dz = self._compute_coordinate_deltas(x)

        with np.errstate(divide='ignore'):
            dm3 = (dx**2 + dy**2 + dz**2) ** -1.5  # 计算坐标差值的模的三次方的倒数，避免除零错误
        dm3[np.diag_indices_from(dm3)] = 0  # 对角线元素置为零，确保不影响结果

        # 计算梯度的各个分量
        grad_x = -np.sum(dx * dm3, axis=1)
        grad_y = -np.sum(dy * dm3, axis=1)
        grad_z = -np.sum(dz * dm3, axis=1)

        return np.hstack((grad_x, grad_y, grad_z))  # 返回计算得到的梯度向量

    # 定义一个方法用于计算某个函数 hess 在输入向量 x 处的黑塞矩阵，并返回结果
    def hess(self, x):
        dx, dy, dz = self._compute_coordinate_deltas(x)
        d = (dx**2 + dy**2 + dz**2) ** 0.5  # 计算坐标差值的模

        with np.errstate(divide='ignore'):
            dm3 = d ** -3  # 计算模的三次方的倒数，避免除零错误
            dm5 = d ** -5  # 计算模的五次方的倒数，避免除零错误

        i = np.arange(self.n_electrons)
        dm3[i, i] = 0  # 对角线元素置为零，确保不影响结果
        dm5[i, i] = 0  # 对角线元素置为零，确保不影响结果

        # 计算黑塞矩阵的各个分量
        Hxx = dm3 - 3 * dx**2 * dm5
        Hxx[i, i] = -np.sum(Hxx, axis=1)

        Hxy = -3 * dx * dy * dm5
        Hxy[i, i] = -np.sum(Hxy, axis=1)

        Hxz = -3 * dx * dz * dm5
        Hxz[i, i] = -np.sum(Hxz, axis=1)

        Hyy = dm3 - 3 * dy**2 * dm5
        Hyy[i, i] = -np.sum(Hyy, axis=1)

        Hyz = -3 * dy * dz * dm5
        Hyz[i, i] = -np.sum(Hyz, axis=1)

        Hzz = dm3 - 3 * dz**2 * dm5
        Hzz[i, i] = -np.sum(Hzz, axis=1)

        # 构建整体的黑塞矩阵
        H = np.vstack((
            np.hstack((Hxx, Hxy, Hxz)),
            np.hstack((Hxy, Hyy, Hyz)),
            np.hstack((Hxz, Hyz, Hzz))
        ))

        return H  # 返回计算得到的黑塞矩阵

    @property
    def constr(self):
        # 定义一个方法用于计算约束函数 fun 在输入向量 x 处的值，并返回结果
        def fun(x):
            x_coord, y_coord, z_coord = self._get_cordinates(x)
            return x_coord**2 + y_coord**2 + z_coord**2 - 1

        # 如果约束的雅可比矩阵未提供，则定义一个计算雅可比矩阵的方法
        if self.constr_jac is None:
            def jac(x):
                x_coord, y_coord, z_coord = self._get_cordinates(x)
                Jx = 2 * np.diag(x_coord)
                Jy = 2 * np.diag(y_coord)
                Jz = 2 * np.diag(z_coord)
                return csc_matrix(np.hstack((Jx, Jy, Jz)))
        else:
            jac = self.constr_jac

        # 如果约束的黑塞矩阵未提供，则定义一个计算黑塞矩阵的方法
        if self.constr_hess is None:
            def hess(x, v):
                D = 2 * np.diag(v)
                return block_diag(D, D, D)
        else:
            hess = self.constr_hess

        return NonlinearConstraint(fun, -np.inf, 0, jac, hess)
# 定义一个测试类 TestTrustRegionConstr
class TestTrustRegionConstr:
    # 定义一个类变量 list_of_problems，包含了多个 Maratos、HyperbolicIneq、Rosenbrock 和 Elec 对象
    list_of_problems = [Maratos(),  # 创建一个 Maratos 对象
                        Maratos(constr_hess='2-point'),  # 创建一个使用 '2-point' 约束 Hessian 的 Maratos 对象
                        Maratos(constr_hess=SR1()),  # 创建一个使用 SR1 约束 Hessian 的 Maratos 对象
                        Maratos(constr_jac='2-point', constr_hess=SR1()),  # 创建一个使用 '2-point' 约束 Jacobian 和 SR1 约束 Hessian 的 Maratos 对象
                        MaratosGradInFunc(),  # 创建一个 MaratosGradInFunc 对象
                        HyperbolicIneq(),  # 创建一个 HyperbolicIneq 对象
                        HyperbolicIneq(constr_hess='3-point'),  # 创建一个使用 '3-point' 约束 Hessian 的 HyperbolicIneq 对象
                        HyperbolicIneq(constr_hess=BFGS()),  # 创建一个使用 BFGS 约束 Hessian 的 HyperbolicIneq 对象
                        HyperbolicIneq(constr_jac='3-point',  # 创建一个使用 '3-point' 约束 Jacobian 和 BFGS 约束 Hessian 的 HyperbolicIneq 对象
                                       constr_hess=BFGS()),
                        Rosenbrock(),  # 创建一个 Rosenbrock 对象
                        IneqRosenbrock(),  # 创建一个 IneqRosenbrock 对象
                        EqIneqRosenbrock(),  # 创建一个 EqIneqRosenbrock 对象
                        BoundedRosenbrock(),  # 创建一个 BoundedRosenbrock 对象
                        Elec(n_electrons=2),  # 创建一个 n_electrons 参数为 2 的 Elec 对象
                        Elec(n_electrons=2, constr_hess='2-point'),  # 创建一个 n_electrons 参数为 2，并使用 '2-point' 约束 Hessian 的 Elec 对象
                        Elec(n_electrons=2, constr_hess=SR1()),  # 创建一个 n_electrons 参数为 2，并使用 SR1 约束 Hessian 的 Elec 对象
                        Elec(n_electrons=2, constr_jac='3-point',  # 创建一个 n_electrons 参数为 2，并使用 '3-point' 约束 Jacobian 和 SR1 约束 Hessian 的 Elec 对象
                             constr_hess=SR1())]

    # 使用 pytest.mark.parametrize 装饰器，为 prob 参数提供多个测试参数，从 list_of_problems 中选择
    # grad 参数有三种可能取值：'prob.grad'、'3-point' 和 False
    @pytest.mark.parametrize('prob', list_of_problems)
    @pytest.mark.parametrize('grad', ('prob.grad', '3-point', False))
    # hess 参数有五种可能取值：'prob.hess'、'3-point'、SR1()、两种 BFGS 不同的 exception_strategy 参数设置
    @pytest.mark.parametrize('hess', ("prob.hess", '3-point', SR1(),
                                      BFGS(exception_strategy='damp_update'),
                                      BFGS(exception_strategy='skip_update')))
    def test_list_of_problems(self, prob, grad, hess):
        # 根据参数设置梯度和黑塞矩阵的值
        grad = prob.grad if grad == "prob.grad" else grad
        hess = prob.hess if hess == "prob.hess" else hess
        
        # 检查是否需要跳过测试
        if (grad in {'2-point', '3-point', 'cs', False} and
                hess in {'2-point', '3-point', 'cs'}):
            pytest.skip("Numerical Hessian needs analytical gradient")
        
        # 检查是否存在不兼容的梯度设置
        if prob.grad is True and grad in {'3-point', False}:
            pytest.skip("prob.grad incompatible with grad in {'3-point', False}")
        
        # 检查是否属于敏感情况，需要标记为测试失败
        sensitive = (isinstance(prob, BoundedRosenbrock) and grad == '3-point'
                     and isinstance(hess, BFGS))
        if sensitive:
            pytest.xfail("Seems sensitive to initial conditions w/ Accelerate")
        
        # 忽略特定警告
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "delta_grad == 0.0")
            # 调用优化函数进行最小化
            result = minimize(prob.fun, prob.x0,
                              method='trust-constr',
                              jac=grad, hess=hess,
                              bounds=prob.bounds,
                              constraints=prob.constr)
        
        # 断言优化结果是否正确
        if prob.x_opt is not None:
            assert_array_almost_equal(result.x, prob.x_opt,
                                      decimal=5)
            # 检查优化条件是否符合要求（gtol）
            if result.status == 1:
                assert_array_less(result.optimality, 1e-8)
        
        # 检查优化条件是否符合要求（xtol）
        if result.status == 2:
            assert_array_less(result.tr_radius, 1e-8)
            
            # 如果使用的方法是 "tr_interior_point"，检查 barrier 参数是否符合要求
            if result.method == "tr_interior_point":
                assert_array_less(result.barrier_parameter, 1e-8)
        
        # 检查是否达到最大迭代次数
        message = f"Invalid termination condition: {result.status}."
        assert result.status not in {0, 3}, message
    # 定义测试函数 test_hessp，用于测试 Hessian 乘以向量 p 的计算
    def test_hessp(self):
        # 创建 Maratos 类的实例 prob
        prob = Maratos()

        # 定义 hessp 函数，计算 Hessian 矩阵 H 与向量 p 的乘积
        def hessp(x, p):
            # 获取 Hessian 矩阵 H
            H = prob.hess(x)
            # 计算 H 与 p 的乘积
            return H.dot(p)

        # 使用 minimize 函数最小化 prob.fun，使用 'trust-constr' 方法
        result = minimize(prob.fun, prob.x0,
                          method='trust-constr',
                          jac=prob.grad, hessp=hessp,
                          bounds=prob.bounds,
                          constraints=prob.constr)

        # 如果 prob.x_opt 不为 None，则断言 result.x 等于 prob.x_opt，精度为 2
        if prob.x_opt is not None:
            assert_array_almost_equal(result.x, prob.x_opt, decimal=2)

        # 当 result.status 为 1 时，检查优化结果的最优性，要求 result.optimality 小于 1e-8
        if result.status == 1:
            assert_array_less(result.optimality, 1e-8)
        
        # 当 result.status 为 2 时，检查优化结果的步长半径，要求 result.tr_radius 小于 1e-8
        if result.status == 2:
            assert_array_less(result.tr_radius, 1e-8)
            
            # 如果 result.method 为 "tr_interior_point"，检查 barrier 参数，要求小于 1e-8
            if result.method == "tr_interior_point":
                assert_array_less(result.barrier_parameter, 1e-8)
        
        # 当 result.status 为 0 或 3 时，抛出运行时错误，指示终止条件无效
        if result.status in (0, 3):
            raise RuntimeError("Invalid termination condition.")

    # 定义测试函数 test_args，测试 minimize 函数接受参数的情况
    def test_args(self):
        # 创建 MaratosTestArgs 类的实例 prob，传入参数 "a" 和 234
        prob = MaratosTestArgs("a", 234)

        # 使用 minimize 函数最小化 prob.fun，传入参数 ("a", 234)，使用 'trust-constr' 方法
        result = minimize(prob.fun, prob.x0, ("a", 234),
                          method='trust-constr',
                          jac=prob.grad, hess=prob.hess,
                          bounds=prob.bounds,
                          constraints=prob.constr)

        # 如果 prob.x_opt 不为 None，则断言 result.x 等于 prob.x_opt，精度为 2
        if prob.x_opt is not None:
            assert_array_almost_equal(result.x, prob.x_opt, decimal=2)

        # 当 result.status 为 1 时，检查优化结果的最优性，要求 result.optimality 小于 1e-8
        if result.status == 1:
            assert_array_less(result.optimality, 1e-8)
        
        # 当 result.status 为 2 时，检查优化结果的步长半径，要求 result.tr_radius 小于 1e-8
        if result.status == 2:
            assert_array_less(result.tr_radius, 1e-8)
            
            # 如果 result.method 为 "tr_interior_point"，检查 barrier 参数，要求小于 1e-8
            if result.method == "tr_interior_point":
                assert_array_less(result.barrier_parameter, 1e-8)
        
        # 当 result.status 为 0 或 3 时，抛出运行时错误，指示终止条件无效
        if result.status in (0, 3):
            raise RuntimeError("Invalid termination condition.")

    # 定义测试函数 test_raise_exception，测试 minimize 函数在参数错误时是否抛出异常
    def test_raise_exception(self):
        # 创建 Maratos 类的实例 prob
        prob = Maratos()
        # 定义期望的错误消息
        message = "Whenever the gradient is estimated via finite-differences"
        
        # 使用 pytest.raises 检查 minimize 函数在使用无效参数时是否抛出 ValueError，并匹配期望的错误消息
        with pytest.raises(ValueError, match=message):
            minimize(prob.fun, prob.x0, method='trust-constr', jac='2-point',
                     hess='2-point', constraints=prob.constr)

    # 定义测试函数 test_issue_9044，测试 GitHub 问题 #9044 的解决情况
    def test_issue_9044(self):
        # 定义回调函数 callback，用于检查返回的 OptimizeResult 是否包含预期的键
        def callback(x, info):
            assert_('nit' in info)
            assert_('niter' in info)

        # 使用 minimize 函数最小化 lambda 函数 x**2，初始值为 [0]，梯度为 2*x，Hessian 为 2
        # 使用 'trust-constr' 方法，并传入回调函数 callback
        result = minimize(lambda x: x**2, [0], jac=lambda x: 2*x,
                          hess=lambda x: 2, callback=callback,
                          method='trust-constr')

        # 断言 result.get('success') 为 True，表示优化成功
        assert_(result.get('success'))
        
        # 检查 result.get('nit', -1) 是否等于 1，表示迭代次数
        assert_(result.get('nit', -1) == 1)

        # 同时检查 'niter' 属性是否存在，以确保向后兼容性
        assert_(result.get('niter', -1) == 1)
    def test_issue_15093(self):
        # 定义测试函数，用于验证问题编号15093
        # scipy文档将边界定义为包含关系，因此即使keep_feasible为True，
        # 在边界上设置x0也不应该成为问题。
        # 之前，trust-constr会将边界视为不包含关系。

        # 设置初始点x0为数组[0., 0.5]
        x0 = np.array([0., 0.5])

        # 定义目标函数obj，接受参数x，计算并返回x1和x2的平方和
        def obj(x):
            x1 = x[0]
            x2 = x[1]
            return x1 ** 2 + x2 ** 2

        # 创建边界对象Bounds，定义范围为[0., 0.]到[1., 1.]，并设置keep_feasible为True
        bounds = Bounds(np.array([0., 0.]), np.array([1., 1.]),
                        keep_feasible=True)

        # 使用suppress_warnings上下文管理器，过滤掉特定的UserWarning消息"delta_grad == 0.0"
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "delta_grad == 0.0")
            # 调用minimize函数，使用trust-constr方法，传入obj作为目标函数，x0作为初始点，bounds作为边界
            result = minimize(
                method='trust-constr',
                fun=obj,
                x0=x0,
                bounds=bounds)

        # 断言结果字典result中的'success'键为真
        assert result['success']
class TestEmptyConstraint:
    """
    Here we minimize x^2+y^2 subject to x^2-y^2>1.
    The actual minimum is at (0, 0) which fails the constraint.
    Therefore we will find a minimum on the boundary at (+/-1, 0).

    When minimizing on the boundary, optimize uses a set of
    constraints that removes the constraint that sets that
    boundary.  In our case, there's only one constraint, so
    the result is an empty constraint.

    This tests that the empty constraint works.
    """

    def test_empty_constraint(self):

        # 定义目标函数，即要最小化的函数
        def function(x):
            return x[0]**2 + x[1]**2

        # 定义目标函数的雅可比矩阵（Jacobian matrix）
        def functionjacobian(x):
            return np.array([2.*x[0], 2.*x[1]])

        # 定义目标函数的 Hessian 矢量积（Hessian-vector product）
        def functionhvp(x, v):
            return 2.*v

        # 定义约束条件函数
        def constraint(x):
            return np.array([x[0]**2 - x[1]**2])

        # 定义约束条件函数的雅可比矩阵
        def constraintjacobian(x):
            return np.array([[2*x[0], -2*x[1]]])

        # 定义约束条件函数的 Lagrange 乘子的 Hessian 乘积
        def constraintlcoh(x, v):
            return np.array([[2., 0.], [0., -2.]]) * v[0]

        # 创建 NonlinearConstraint 对象，设置约束条件
        constraint = NonlinearConstraint(constraint, 1., np.inf,
                                         constraintjacobian, constraintlcoh)

        # 设置优化的起始点
        startpoint = [1., 2.]

        # 设置变量的边界条件
        bounds = Bounds([-np.inf, -np.inf], [np.inf, np.inf])

        # 调用 minimize 函数进行优化
        result = minimize(
          function,
          startpoint,
          method='trust-constr',
          jac=functionjacobian,
          hessp=functionhvp,
          constraints=[constraint],
          bounds=bounds,
        )

        # 断言优化结果的精度符合预期
        assert_array_almost_equal(abs(result.x), np.array([1, 0]), decimal=4)


def test_bug_11886():
    # 定义优化的目标函数
    def opt(x):
        return x[0]**2+x[1]**2

    # 设置线性约束条件
    with np.testing.suppress_warnings() as sup:
        sup.filter(PendingDeprecationWarning)
        A = np.matrix(np.diag([1, 1]))
    lin_cons = LinearConstraint(A, -1, np.inf)
    
    # 调用 minimize 函数进行优化，验证是否出现错误
    minimize(opt, 2*[1], constraints=lin_cons)


# Remove xfail when gh-11649 is resolved
@pytest.mark.xfail(reason="Known bug in trust-constr; see gh-11649.",
                   strict=True)
def test_gh11649():
    # 设置变量的边界条件
    bnds = Bounds(lb=[-1, -1], ub=[1, 1], keep_feasible=True)

    # 定义断言函数，用于验证结果是否在边界内
    def assert_inbounds(x):
        assert np.all(x >= bnds.lb)
        assert np.all(x <= bnds.ub)

    # 定义优化的目标函数
    def obj(x):
        assert_inbounds(x)
        return np.exp(x[0])*(4*x[0]**2 + 2*x[1]**2 + 4*x[0]*x[1] + 2*x[1] + 1)

    # 定义非线性约束条件函数
    def nce(x):
        assert_inbounds(x)
        return x[0]**2 + x[1]

    def nci(x):
        assert_inbounds(x)
        return x[0]*x[1]

    x0 = np.array((0.99, -0.99))
    # 创建 NonlinearConstraint 对象，设置非线性约束条件
    nlcs = [NonlinearConstraint(nci, -10, np.inf),
            NonlinearConstraint(nce, 1, 1)]

    # 使用 trust-constr 方法进行优化
    res = minimize(fun=obj, x0=x0, method='trust-constr',
                   bounds=bnds, constraints=nlcs)
    assert res.success
    assert_inbounds(res.x)
    assert nlcs[0].lb < nlcs[0].fun(res.x) < nlcs[0].ub
    assert_allclose(nce(res.x), nlcs[1].ub)

    # 对比使用 slsqp 方法进行的优化结果
    ref = minimize(fun=obj, x0=x0, method='slsqp',
                   bounds=bnds, constraints=nlcs)
    # 断言检查：验证 res.fun 是否与 ref.fun 的值“几乎相等”
    assert_allclose(res.fun, ref.fun)
# 定义一个测试函数，用于验证 GitHub 问题 #20665 中当等式约束多于变量时报告的混乱错误信息是否得到改进。
def test_gh20665_too_many_constraints():
    # 设定期望的错误信息内容，用于匹配异常断言
    message = "...more equality constraints than independent variables..."
    # 使用 pytest 的异常断言，验证是否抛出 ValueError 异常且错误信息与期望相匹配
    with pytest.raises(ValueError, match=message):
        # 初始化起始点为全为1的长度为2的向量
        x0 = np.ones((2,))
        # 设定等式约束 A_eq 为 3x2 的矩阵，b_eq 为全为1的长度为3的向量
        A_eq, b_eq = np.arange(6).reshape((3, 2)), np.ones((3,))
        # 创建 NonlinearConstraint 对象 g，使用 lambda 表达式计算 A_eq @ x
        g = NonlinearConstraint(lambda x:  A_eq @ x, lb=b_eq, ub=b_eq)
        # 调用 minimize 函数进行优化，使用 trust-constr 方法，并添加约束 g
        minimize(rosen, x0, method='trust-constr', constraints=[g])
    
    # 使用 'SVDFactorization' 方法调用 minimize 函数，确保不会出现错误
    with np.testing.suppress_warnings() as sup:
        sup.filter(UserWarning)
        minimize(rosen, x0, method='trust-constr', constraints=[g],
                 options={'factorization_method': 'SVDFactorization'})


class TestBoundedNelderMead:

    @pytest.mark.parametrize('bounds, x_opt',
                             [(Bounds(-np.inf, np.inf), Rosenbrock().x_opt),
                              (Bounds(-np.inf, -0.8), [-0.8, -0.8]),
                              (Bounds(3.0, np.inf), [3.0, 9.0]),
                              (Bounds([3.0, 1.0], [4.0, 5.0]), [3., 5.]),
                              ])
    # 测试 Rosenbrock 函数在给定边界下的行为
    def test_rosen_brock_with_bounds(self, bounds, x_opt):
        # 创建 Rosenbrock 函数实例
        prob = Rosenbrock()
        # 使用 suppress_warnings 上下文管理器，过滤 "Initial guess is not within the specified bounds" 的 UserWarning
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "Initial guess is not within "
                                    "the specified bounds")
            # 调用 minimize 函数，使用 Nelder-Mead 方法，给定边界 bounds
            result = minimize(prob.fun, [-10, -10],
                              method='Nelder-Mead',
                              bounds=bounds)
            # 断言优化结果的解在设定的下界 bounds.lb 之上
            assert np.less_equal(bounds.lb, result.x).all()
            # 断言优化结果的解在设定的上界 bounds.ub 之下
            assert np.less_equal(result.x, bounds.ub).all()
            # 断言优化结果的目标函数值与真实 Rosenbrock 函数值相近
            assert np.allclose(prob.fun(result.x), result.fun)
            # 断言优化结果的解与期望解 x_opt 相近，允许的误差为 1.e-3
            assert np.allclose(result.x, x_opt, atol=1.e-3)

    # 测试 Rosenbrock 函数在所有边界相等时的行为
    def test_equal_all_bounds(self):
        # 创建 Rosenbrock 函数实例
        prob = Rosenbrock()
        # 设定边界 bounds，使得上下界相等
        bounds = Bounds([4.0, 5.0], [4.0, 5.0])
        # 使用 suppress_warnings 上下文管理器，过滤 "Initial guess is not within the specified bounds" 的 UserWarning
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "Initial guess is not within "
                                    "the specified bounds")
            # 调用 minimize 函数，使用 Nelder-Mead 方法，给定边界 bounds
            result = minimize(prob.fun, [-10, 8],
                              method='Nelder-Mead',
                              bounds=bounds)
            # 断言优化结果的解与设定的边界上界 bounds.ub 相等
            assert np.allclose(result.x, [4.0, 5.0])

    # 测试 Rosenbrock 函数在部分边界相等时的行为
    def test_equal_one_bounds(self):
        # 创建 Rosenbrock 函数实例
        prob = Rosenbrock()
        # 设定边界 bounds，其中一个维度的上下界相等，另一个不相等
        bounds = Bounds([4.0, 5.0], [4.0, 20.0])
        # 使用 suppress_warnings 上下文管理器，过滤 "Initial guess is not within the specified bounds" 的 UserWarning
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "Initial guess is not within "
                                    "the specified bounds")
            # 调用 minimize 函数，使用 Nelder-Mead 方法，给定边界 bounds
            result = minimize(prob.fun, [-10, 8],
                              method='Nelder-Mead',
                              bounds=bounds)
            # 断言优化结果的解与期望解 [4.0, 16.0] 相近
            assert np.allclose(result.x, [4.0, 16.0])
    # 定义一个测试方法，用于测试参数边界设置不合法的情况
    def test_invalid_bounds(self):
        # 创建 Rosenbrock 优化问题实例
        prob = Rosenbrock()
        # 设置错误信息文本
        message = 'An upper bound is less than the corresponding lower bound.'
        # 使用 pytest 检测是否会抛出 ValueError 异常，并匹配特定错误信息
        with pytest.raises(ValueError, match=message):
            # 创建边界对象，其中第一个维度的上界小于下界，预期会引发异常
            bounds = Bounds([-np.inf, 1.0], [4.0, -5.0])
            # 使用 Nelder-Mead 方法进行优化，但会因边界设置错误而失败
            minimize(prob.fun, [-10, 3],
                     method='Nelder-Mead',
                     bounds=bounds)

    # 标记为预期失败的测试方法，原因是在 Azure Linux 和 macOS 上构建时失败，参见 gh-13846
    def test_outside_bounds_warning(self):
        # 创建 Rosenbrock 优化问题实例
        prob = Rosenbrock()
        # 设置警告信息文本
        message = "Initial guess is not within the specified bounds"
        # 使用 pytest 检测是否会发出 UserWarning 警告，并匹配特定警告信息
        with pytest.warns(UserWarning, match=message):
            # 创建边界对象，其中初始猜测的第二个维度超出了设定的上界
            bounds = Bounds([-np.inf, 1.0], [4.0, 5.0])
            # 使用 Nelder-Mead 方法进行优化，但会因初始猜测超出边界而发出警告
            minimize(prob.fun, [-10, 8],
                     method='Nelder-Mead',
                     bounds=bounds)
```