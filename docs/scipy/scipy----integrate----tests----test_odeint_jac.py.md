# `D:\src\scipysrc\scipy\scipy\integrate\tests\test_odeint_jac.py`

```
# 导入 NumPy 库，并使用 np 作为别名
import numpy as np
# 从 numpy.testing 中导入 assert_equal 和 assert_allclose 函数
from numpy.testing import assert_equal, assert_allclose
# 从 scipy.integrate 中导入 odeint 函数
from scipy.integrate import odeint
# 导入 scipy.integrate._test_odeint_banded 模块，并使用 banded5x5 作为别名
import scipy.integrate._test_odeint_banded as banded5x5


# 定义右手边函数 rhs，接受 y 和 t 两个参数
def rhs(y, t):
    # 创建一个与 y 形状相同的全零数组 dydt
    dydt = np.zeros_like(y)
    # 调用 banded5x5 模块的 banded5x5 函数，计算 dydt
    banded5x5.banded5x5(t, y, dydt)
    # 返回 dydt
    return dydt


# 定义雅可比矩阵函数 jac，接受 y 和 t 两个参数
def jac(y, t):
    # 计算 y 的长度并保存在变量 n 中
    n = len(y)
    # 创建一个形状为 (n, n) 的零矩阵 jac，使用 Fortran 风格存储顺序
    jac = np.zeros((n, n), order='F')
    # 调用 banded5x5 模块的 banded5x5_jac 函数，计算雅可比矩阵并存储在 jac 中
    banded5x5.banded5x5_jac(t, y, 1, 1, jac)
    # 返回雅可比矩阵 jac
    return jac


# 定义带状雅可比矩阵函数 bjac，接受 y 和 t 两个参数
def bjac(y, t):
    # 计算 y 的长度并保存在变量 n 中
    n = len(y)
    # 创建一个形状为 (4, n) 的零矩阵 bjac，使用 Fortran 风格存储顺序
    bjac = np.zeros((4, n), order='F')
    # 调用 banded5x5 模块的 banded5x5_bjac 函数，计算带状雅可比矩阵并存储在 bjac 中
    banded5x5.banded5x5_bjac(t, y, 1, 1, bjac)
    # 返回带状雅可比矩阵 bjac
    return bjac


# 定义常量 JACTYPE_FULL，并赋值为 1，表示完全雅可比矩阵
JACTYPE_FULL = 1
# 定义常量 JACTYPE_BANDED，并赋值为 4，表示带状雅可比矩阵
JACTYPE_BANDED = 4


# 定义函数 check_odeint，接受 jactype 参数
def check_odeint(jactype):
    # 根据 jactype 的值选择不同的 ml、mu 和 jacobian 函数
    if jactype == JACTYPE_FULL:
        ml = None
        mu = None
        jacobian = jac
    elif jactype == JACTYPE_BANDED:
        ml = 2
        mu = 1
        jacobian = bjac
    else:
        # 如果 jactype 不是合法值，则抛出 ValueError 异常
        raise ValueError(f"invalid jactype: {jactype!r}")

    # 创建初始状态 y0，包含值从 1.0 到 6.0
    y0 = np.arange(1.0, 6.0)
    # 设置相对误差和绝对误差的容忍度
    rtol = 1e-11
    atol = 1e-13
    # 设置时间步长 dt 和步数 nsteps
    dt = 0.125
    nsteps = 64
    # 创建时间数组 t，以 dt 为步长，共 nsteps+1 个时间点
    t = dt * np.arange(nsteps+1)

    # 调用 odeint 函数求解微分方程组，返回结果 sol 和信息 info
    sol, info = odeint(rhs, y0, t,
                       Dfun=jacobian, ml=ml, mu=mu,
                       atol=atol, rtol=rtol, full_output=True)
    # 获取最终时间点的状态 yfinal
    yfinal = sol[-1]
    # 获取 odeint 求解过程中的统计信息
    odeint_nst = info['nst'][-1]
    odeint_nfe = info['nfe'][-1]
    odeint_nje = info['nje'][-1]

    # 创建 y1，复制 y0 的值
    y1 = y0.copy()
    # 调用 banded5x5 模块的 banded5x5_solve 函数求解微分方程，修改 y1 的值
    nst, nfe, nje = banded5x5.banded5x5_solve(y1, nsteps, dt, jactype)

    # 使用 assert_allclose 检查 yfinal 和 y1 是否非常接近
    assert_allclose(yfinal, y1, rtol=1e-12)
    # 使用 assert_equal 检查 odeint 和 banded5x5_solve 的统计信息是否一致
    assert_equal((odeint_nst, odeint_nfe, odeint_nje), (nst, nfe, nje))


# 定义函数 test_odeint_full_jac，调用 check_odeint 函数，传入 JACTYPE_FULL
def test_odeint_full_jac():
    check_odeint(JACTYPE_FULL)


# 定义函数 test_odeint_banded_jac，调用 check_odeint 函数，传入 JACTYPE_BANDED
def test_odeint_banded_jac():
    check_odeint(JACTYPE_BANDED)
```