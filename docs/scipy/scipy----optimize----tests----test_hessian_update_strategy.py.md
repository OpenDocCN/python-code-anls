# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_hessian_update_strategy.py`

```
# 导入正则表达式模块
import re
# 导入深拷贝函数
from copy import deepcopy

# 导入NumPy库，并将其重命名为np，用于科学计算
import numpy as np
# 导入pytest测试框架
import pytest
# 导入NumPy的线性代数模块中的norm函数
from numpy.linalg import norm
# 导入NumPy的测试模块中的测试用例和几个断言函数
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less)
# 导入SciPy优化模块中的两个优化算法：BFGS和SR1
from scipy.optimize import (BFGS, SR1)


class Rosenbrock:
    """Rosenbrock function.

    The following optimization problem:
        minimize sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    """

    def __init__(self, n=2, random_state=0):
        # 使用随机数生成器创建初始点x0
        rng = np.random.RandomState(random_state)
        self.x0 = rng.uniform(-1, 1, n)
        # 设置全局最优解x_opt为全1向量
        self.x_opt = np.ones(n)

    def fun(self, x):
        # 将输入参数x转换为NumPy数组
        x = np.asarray(x)
        # 计算Rosenbrock函数的值
        r = np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0,
                   axis=0)
        return r

    def grad(self, x):
        # 将输入参数x转换为NumPy数组
        x = np.asarray(x)
        # 分别计算Rosenbrock函数对每个变量的梯度
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = (200 * (xm - xm_m1**2) -
                     400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm))
        der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        der[-1] = 200 * (x[-1] - x[-2]**2)
        return der

    def hess(self, x):
        # 将输入参数x转换为至少一维的NumPy数组
        x = np.atleast_1d(x)
        # 计算Rosenbrock函数的Hessian矩阵
        H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
        diagonal = np.zeros(len(x), dtype=x.dtype)
        diagonal[0] = 1200 * x[0]**2 - 400 * x[1] + 2
        diagonal[-1] = 200
        diagonal[1:-1] = 202 + 1200 * x[1:-1]**2 - 400 * x[2:]
        H = H + np.diag(diagonal)
        return H


class TestHessianUpdateStrategy(TestCase):
    # 定义测试函数 test_hessian_initialization，用于测试 Hessian 矩阵的初始化
    def test_hessian_initialization(self):

        # 定义矩阵维度为 5
        ndims = 5
        # 创建对称矩阵作为测试数据
        symmetric_matrix = np.array([[43, 24, 33, 34, 49],
                                     [24, 36, 44, 15, 44],
                                     [33, 44, 37, 1, 30],
                                     [34, 15, 1, 5, 46],
                                     [49, 44, 30, 46, 22]])
        # 初始化比例尺度和真实矩阵的元组列表
        init_scales = (
            ('auto', np.eye(ndims)),
            (2, np.eye(ndims) * 2),
            (np.arange(1, ndims + 1) * np.eye(ndims),
             np.arange(1, ndims + 1) * np.eye(ndims)),
            (symmetric_matrix, symmetric_matrix),)

        # 遍历近似类型 'hess' 和 'inv_hess'
        for approx_type in ['hess', 'inv_hess']:
            # 遍历不同的初始化比例尺度和对应的真实矩阵
            for init_scale, true_matrix in init_scales:
                # 创建 BFGS 和 SR1 类的 quasi_newton 对象
                # 当 min_curvature 或 min_denominator 较大时，跳过更新，使用初始矩阵
                quasi_newton = (BFGS(init_scale=init_scale,
                                     min_curvature=1e50,
                                     exception_strategy='skip_update'),
                                SR1(init_scale=init_scale,
                                    min_denominator=1e50))

                # 遍历 quasi_newton 中的每个对象
                for qn in quasi_newton:
                    # 初始化 quasi_newton 对象的矩阵为 ndims 维度的 approx_type 类型
                    qn.initialize(ndims, approx_type)
                    # 获取初始化后的矩阵 B
                    B = qn.get_matrix()

                    # 断言 B 等于单位矩阵 np.eye(ndims)
                    assert_array_equal(B, np.eye(ndims))
                    
                    # 对于自动初始化尺度 'auto'，不进行进一步测试
                    if isinstance(init_scale, str) and init_scale == 'auto':
                        continue

                    # 更新 quasi_newton 对象
                    qn.update(np.ones(ndims) * 1e-5, np.arange(ndims) + 0.2)
                    # 获取更新后的矩阵 B
                    B = qn.get_matrix()
                    # 断言 B 等于真实矩阵 true_matrix
                    assert_array_equal(B, true_matrix)

    # 对于这组点，已知在 Hessian 更新过程中没有异常发生
    # 因此没有跳过或抑制任何更新。
    def test_initialize_catch_illegal(self):
        ndims = 3
        # 禁止使用复数
        inits_msg_errtype = ((complex(3.14),
                              r"float\(\) argument must be a string or a "
                              r"(real )?number, not 'complex'",
                              TypeError),

                             (np.array([3.2, 2.3, 1.2]).astype(np.complex128),
                              "init_scale contains complex elements, "
                              "must be real.",
                              TypeError),

                             (np.array([[43, 24, 33],
                                        [24, 36, 44, ],
                                        [33, 44, 37, ]]).astype(np.complex128),
                              "init_scale contains complex elements, "
                              "must be real.",
                              TypeError),

                             # 非方阵
                             (np.array([[43, 55, 66]]),
                              re.escape(
                                  "If init_scale is an array, it must have the "
                                  "dimensions of the hess/inv_hess: (3, 3)."
                                  " Got (1, 3)."),
                              ValueError),

                             # 非对称
                             (np.array([[43, 24, 33],
                                        [24.1, 36, 44, ],
                                        [33, 44, 37, ]]),
                              re.escape("If init_scale is an array, it must be"
                                        " symmetric (passing scipy.linalg.issymmetric)"
                                        " to be an approximation of a hess/inv_hess."),
                              ValueError),
                             )
        for approx_type in ['hess', 'inv_hess']:
            for init_scale, message, errortype in inits_msg_errtype:
                # 根据初始化类型选择相应的拟牛顿方法
                quasi_newton = (BFGS(init_scale=init_scale),
                                SR1(init_scale=init_scale))

                for qn in quasi_newton:
                    # 初始化拟牛顿方法
                    qn.initialize(ndims, approx_type)
                    # 使用 pytest 断言捕获特定类型的错误，并匹配相应的消息
                    with pytest.raises(errortype, match=message):
                        # 更新拟牛顿方法的参数
                        qn.update(np.ones(ndims), np.arange(ndims))
    def test_SR1_skip_update(self):
        # 定义辅助问题对象为Rosenbrock函数，维度为5
        prob = Rosenbrock(n=5)
        # 定义迭代点列表x_list
        x_list = [[0.0976270, 0.4303787, 0.2055267, 0.0897663, -0.15269040],
                  [0.1847239, 0.0505757, 0.2123832, 0.0255081, 0.00083286],
                  [0.2142498, -0.0188480, 0.0503822, 0.0347033, 0.03323606],
                  [0.2071680, -0.0185071, 0.0341337, -0.0139298, 0.02881750],
                  [0.1533055, -0.0322935, 0.0280418, -0.0083592, 0.01503699],
                  [0.1382378, -0.0276671, 0.0266161, -0.0074060, 0.02801610],
                  [0.1651957, -0.0049124, 0.0269665, -0.0040025, 0.02138184],
                  [0.2354930, 0.0443711, 0.0173959, 0.0041872, 0.00794563],
                  [0.4168118, 0.1433867, 0.0111714, 0.0126265, -0.00658537],
                  [0.4681972, 0.2153273, 0.0225249, 0.0152704, -0.00463809],
                  [0.6023068, 0.3346815, 0.0731108, 0.0186618, -0.00371541],
                  [0.6415743, 0.3985468, 0.1324422, 0.0214160, -0.00062401],
                  [0.7503690, 0.5447616, 0.2804541, 0.0539851, 0.00242230],
                  [0.7452626, 0.5644594, 0.3324679, 0.0865153, 0.00454960],
                  [0.8059782, 0.6586838, 0.4229577, 0.1452990, 0.00976702],
                  [0.8549542, 0.7226562, 0.4991309, 0.2420093, 0.02772661],
                  [0.8571332, 0.7285741, 0.5279076, 0.2824549, 0.06030276],
                  [0.8835633, 0.7727077, 0.5957984, 0.3411303, 0.09652185],
                  [0.9071558, 0.8299587, 0.6771400, 0.4402896, 0.17469338]]
        # 计算梯度列表grad_list，分别对x_list中的每个点计算梯度
        grad_list = [prob.grad(x) for x in x_list]
        # 计算变化量delta_x，即相邻点之间的差值
        delta_x = [np.array(x_list[i+1])-np.array(x_list[i])
                   for i in range(len(x_list)-1)]
        # 计算梯度变化量delta_grad，即相邻梯度之间的差值
        delta_grad = [grad_list[i+1]-grad_list[i]
                      for i in range(len(grad_list)-1)]
        # 初始化SR1对象hess，初始缩放因子为1，最小分母为0.01
        hess = SR1(init_scale=1, min_denominator=1e-2)
        # 初始化hess对象，设置矩阵维度为x_list中第一个点的维度，并命名为'hess'
        hess.initialize(len(x_list[0]), 'hess')
        # 比较Hessian矩阵及其逆矩阵的情况
        for i in range(len(delta_x)-1):
            # 取出当前的s值，即delta_x中的第i个元素
            s = delta_x[i]
            # 取出当前的y值，即delta_grad中的第i个元素
            y = delta_grad[i]
            # 使用当前的s和y更新hess对象
            hess.update(s, y)
        # 测试跳过更新的情况
        # 复制当前的Hessian矩阵到B
        B = np.copy(hess.get_matrix())
        # 取出delta_x中的第17个元素作为s
        s = delta_x[17]
        # 取出delta_grad中的第17个元素作为y
        y = delta_grad[17]
        # 使用s和y更新hess对象
        hess.update(s, y)
        # 将更新后的Hessian矩阵复制到B_updated
        B_updated = np.copy(hess.get_matrix())
        # 断言B与B_updated是否相等
        assert_array_equal(B, B_updated)
    def test_BFGS_skip_update(self):
        # 定义辅助问题 Rosenbrock 函数，参数为维度 n=5
        prob = Rosenbrock(n=5)
        
        # 定义迭代点列表 x_list
        x_list = [[0.0976270, 0.4303787, 0.2055267, 0.0897663, -0.15269040],
                  [0.1847239, 0.0505757, 0.2123832, 0.0255081, 0.00083286],
                  [0.2142498, -0.0188480, 0.0503822, 0.0347033, 0.03323606],
                  [0.2071680, -0.0185071, 0.0341337, -0.0139298, 0.02881750],
                  [0.1533055, -0.0322935, 0.0280418, -0.0083592, 0.01503699],
                  [0.1382378, -0.0276671, 0.0266161, -0.0074060, 0.02801610],
                  [0.1651957, -0.0049124, 0.0269665, -0.0040025, 0.02138184]]
        
        # 计算每个迭代点的梯度，存储在 grad_list 中
        grad_list = [prob.grad(x) for x in x_list]
        
        # 计算每个相邻迭代点的差值 delta_x
        delta_x = [np.array(x_list[i+1])-np.array(x_list[i])
                   for i in range(len(x_list)-1)]
        
        # 计算每个相邻迭代点梯度的差值 delta_grad
        delta_grad = [grad_list[i+1]-grad_list[i]
                      for i in range(len(grad_list)-1)]
        
        # 初始化 BFGS 类，设置初始缩放因子为 1，最小曲率为 10
        hess = BFGS(init_scale=1, min_curvature=10)
        
        # 初始化 Hessian 矩阵，维度与迭代点相同
        hess.initialize(len(x_list[0]), 'hess')
        
        # 比较 Hessian 矩阵及其逆矩阵
        for i in range(len(delta_x)-1):
            s = delta_x[i]
            y = delta_grad[i]
            hess.update(s, y)
        
        # 测试跳过更新操作
        B = np.copy(hess.get_matrix())
        s = delta_x[5]
        y = delta_grad[5]
        hess.update(s, y)
        B_updated = np.copy(hess.get_matrix())
        
        # 断言两个矩阵 B 和 B_updated 相等
        assert_array_equal(B, B_updated)
```