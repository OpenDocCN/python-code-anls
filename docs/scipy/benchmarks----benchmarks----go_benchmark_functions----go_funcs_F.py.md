# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_F.py`

```
from .go_benchmark import Benchmark

class FreudensteinRoth(Benchmark):
    r"""
    FreudensteinRoth objective function.

    This class defines the Freudenstein & Roth [1]_ global optimization problem.
    This is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{FreudensteinRoth}}(x) =  \left\{x_1 - 13 + \left[(5 - x_2) x_2
        - 2 \right] x_2 \right\}^2 + \left \{x_1 - 29 
        + \left[(x_2 + 1) x_2 - 14 \right] x_2 \right\}^2


    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [5, 4]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设定维度
        Benchmark.__init__(self, dimensions)

        # 定义变量边界，每个维度的取值范围为 [-10.0, 10.0]
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        
        # 自定义边界，特定维度的取值范围为 [-3, 3] 和 [-5, 5]
        self.custom_bounds = [(-3, 3), (-5, 5)]

        # 全局最优解，期望值为 [5.0, 4.0]
        self.global_optimum = [[5.0, 4.0]]
        
        # 全局最优函数值为 0.0
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估次数
        self.nfev += 1

        # 计算第一个目标函数 f1
        f1 = (-13.0 + x[0] + ((5.0 - x[1]) * x[1] - 2.0) * x[1]) ** 2
        
        # 计算第二个目标函数 f2
        f2 = (-29.0 + x[0] + ((x[1] + 1.0) * x[1] - 14.0) * x[1]) ** 2

        # 返回目标函数 f 的值
        return f1 + f2
```