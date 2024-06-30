# `D:\src\scipysrc\scipy\scipy\integrate\_ivp\tests\test_rk.py`

```
# 导入 pytest 模块，用于测试
import pytest
# 从 numpy.testing 模块导入 assert_allclose 和 assert_，用于测试数值是否接近及其他断言
from numpy.testing import assert_allclose, assert_
# 导入 numpy 库，并用 np 别名表示
import numpy as np
# 从 scipy.integrate 模块导入 RK23, RK45, DOP853 等积分器
from scipy.integrate import RK23, RK45, DOP853
# 导入 scipy.integrate._ivp 中的 dop853_coefficients，用于 DOP853 积分器的系数
from scipy.integrate._ivp import dop853_coefficients


# 使用 pytest.mark.parametrize 装饰器，测试不同的积分器（RK23, RK45, DOP853）的系数属性
@pytest.mark.parametrize("solver", [RK23, RK45, DOP853])
def test_coefficient_properties(solver):
    # 断言积分器 B 系数之和为 1，相对误差限制为 1e-15
    assert_allclose(np.sum(solver.B), 1, rtol=1e-15)
    # 断言积分器 A 系数每行之和与积分器 C 系数相等，相对误差限制为 1e-14
    assert_allclose(np.sum(solver.A, axis=1), solver.C, rtol=1e-14)


# 测试 DOP853 积分器的系数属性
def test_coefficient_properties_dop853():
    # 断言 DOP853 积分器的 B 系数之和为 1，相对误差限制为 1e-15
    assert_allclose(np.sum(dop853_coefficients.B), 1, rtol=1e-15)
    # 断言 DOP853 积分器的 A 系数每行之和与 C 系数相等，相对误差限制为 1e-14
    assert_allclose(np.sum(dop853_coefficients.A, axis=1),
                    dop853_coefficients.C,
                    rtol=1e-14)


# 使用 pytest.mark.parametrize 装饰器，测试不同积分器类（RK23, RK45, DOP853）的误差估计
@pytest.mark.parametrize("solver_class", [RK23, RK45, DOP853])
def test_error_estimation(solver_class):
    # 设定步长为 0.2
    step = 0.2
    # 使用给定的积分器类创建 solver 对象，用 lambda 函数表示微分方程
    solver = solver_class(lambda t, y: y, 0, [1], 1, first_step=step)
    # 进行一步积分计算
    solver.step()
    # 计算估计误差
    error_estimate = solver._estimate_error(solver.K, step)
    # 计算实际误差
    error = solver.y - np.exp([step])
    # 断言实际误差的绝对值小于估计误差的绝对值
    assert_(np.abs(error) < np.abs(error_estimate))


# 使用 pytest.mark.parametrize 装饰器，测试复杂微分方程的误差估计
@pytest.mark.parametrize("solver_class", [RK23, RK45, DOP853])
def test_error_estimation_complex(solver_class):
    # 设定步长为 0.2
    h = 0.2
    # 使用给定的积分器类创建 solver 对象，用 lambda 函数表示复杂微分方程
    solver = solver_class(lambda t, y: 1j * y, 0, [1j], 1, first_step=h)
    # 进行一步积分计算
    solver.step()
    # 计算误差的范数估计
    err_norm = solver._estimate_error_norm(solver.K, h, scale=[1])
    # 断言误差范数是实数对象
    assert np.isrealobj(err_norm)
```