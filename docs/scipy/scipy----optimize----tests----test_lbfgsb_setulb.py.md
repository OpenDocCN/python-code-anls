# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_lbfgsb_setulb.py`

```
# 导入NumPy库，命名为np
import numpy as np
# 从scipy.optimize模块中导入_lbfgsb子模块和minimize函数
from scipy.optimize import _lbfgsb, minimize


def objfun(x):
    """定义简化的目标函数用于测试lbfgsb方法是否违反边界约束

    x: 输入的变量向量
    返回元组(f, g)，其中f是目标函数值，g是梯度向量
    """
    # 预设的几个测试点
    x0 = [0.8750000000000278,
          0.7500000000000153,
          0.9499999999999722,
          0.8214285714285992,
          0.6363636363636085]
    x1 = [1.0, 0.0, 1.0, 0.0, 0.0]
    x2 = [1.0,
          0.0,
          0.9889733043149325,
          0.0,
          0.026353554421041155]
    x3 = [1.0,
          0.0,
          0.9889917442915558,
          0.0,
          0.020341986743231205]

    # 预设的目标函数值
    f0 = 5163.647901211178
    f1 = 5149.8181642072905
    f2 = 5149.379332309634
    f3 = 5149.374490771297

    # 预设的梯度向量
    g0 = np.array([-0.5934820547965749,
                   1.6251549718258351,
                   -71.99168459202559,
                   5.346636965797545,
                   37.10732723092604])
    g1 = np.array([-0.43295349282641515,
                   1.008607936794592,
                   18.223666726602975,
                   31.927010036981997,
                   -19.667512518739386])
    g2 = np.array([-0.4699874455100256,
                   0.9466285353668347,
                   -0.016874360242016825,
                   48.44999161133457,
                   5.819631620590712])
    g3 = np.array([-0.46970678696829116,
                   0.9612719312174818,
                   0.006129809488833699,
                   48.43557729419473,
                   6.005481418498221])

    # 根据输入的x值选择相应的预设目标函数值和梯度向量
    if np.allclose(x, x0):
        f = f0
        g = g0
    elif np.allclose(x, x1):
        f = f1
        g = g1
    elif np.allclose(x, x2):
        f = f2
        g = g2
    elif np.allclose(x, x3):
        f = f3
        g = g3
    else:
        raise ValueError(
            '请求点处的简化目标函数未定义')
    return (np.copy(f), np.copy(g))


def test_setulb_floatround():
    """测试setulb()函数是否由于浮点舍入误差违反边界约束

    检查由于浮点舍入误差而导致的约束违反
    """

    # 初始化问题的维度和约束
    n = 5
    m = 10
    factr = 1e7
    pgtol = 1e-5
    maxls = 20
    iprint = -1
    nbd = np.full((n,), 2)  # 初始化边界类型，2表示上下界都有
    low_bnd = np.zeros(n, np.float64)  # 下界初始化为0
    upper_bnd = np.ones(n, np.float64)  # 上界初始化为1

    # 初始化优化变量x0和工作变量x
    x0 = np.array(
        [0.8750000000000278,
         0.7500000000000153,
         0.9499999999999722,
         0.8214285714285992,
         0.6363636363636085])
    x = np.copy(x0)

    # 初始化目标函数值f和梯度向量g
    f = np.array(0.0, np.float64)
    g = np.zeros(n, np.float64)

    # 定义Fortran中的整数类型
    fortran_int = _lbfgsb.types.intvar.dtype

    # 初始化工作数组和整数数组
    wa = np.zeros(2*m*n + 5*n + 11*m*m + 8*m, np.float64)
    iwa = np.zeros(3*n, fortran_int)
    task = np.zeros(1, 'S60')
    csave = np.zeros(1, 'S60')
    lsave = np.zeros(4, fortran_int)
    isave = np.zeros(44, fortran_int)
    dsave = np.zeros(29, np.float64)

    # 初始化任务状态为START
    task[:] = b'START'
    # 循环执行7次以复现错误
    for n_iter in range(7):  # 7 steps required to reproduce error
        # 调用 objfun 函数计算目标函数值和梯度
        f, g = objfun(x)

        # 调用 _lbfgsb.setulb 函数设置优化问题的边界和参数
        _lbfgsb.setulb(m, x, low_bnd, upper_bnd, nbd, f, g, factr,
                       pgtol, wa, iwa, task, iprint, csave, lsave,
                       isave, dsave, maxls)

        # 使用断言确保优化后的解 x 在定义的边界内
        assert (x <= upper_bnd).all() and (x >= low_bnd).all(), (
            "_lbfgsb.setulb() stepped to a point outside of the bounds")
# 定义一个测试函数，用于验证问题18730报告的问题：l-bfgs-b 方法不能处理返回单精度梯度数组的目标函数
def test_gh_issue18730():
    # 定义一个返回单精度梯度数组的目标函数
    def fun_single_precision(x):
        # 将输入向量 x 转换为单精度浮点数类型
        x = x.astype(np.float32)
        # 计算目标函数值为 x 的平方和，梯度为 2*x
        return np.sum(x**2), (2*x)

    # 使用 l-bfgs-b 方法最小化 fun_single_precision 函数
    res = minimize(fun_single_precision, x0=np.array([1., 1.]), jac=True,
                   method="l-bfgs-b")
    # 使用 np.testing.assert_allclose 来验证最小化结果的 fun 值接近于 0，允许误差为 1e-15
    np.testing.assert_allclose(res.fun, 0., atol=1e-15)
```