# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_I.py`

```
# 从 numpy 库中导入 sin 和 sum 函数
from numpy import sin, sum
# 从当前目录下的 go_benchmark 模块中导入 Benchmark 类
from .go_benchmark import Benchmark

# 定义 Infinity 类，继承自 Benchmark 类
class Infinity(Benchmark):

    r"""
    Infinity objective function.

    This class defines the Infinity [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Infinity}}(x) = \sum_{i=1}^{n} x_i^{6} 
        \left [ \sin\left ( \frac{1}{x_i} \right ) + 2 \right ]


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-1, 1]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    # 初始化方法，设置维度，默认为 2
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置变量 _bounds，表示每个维度的取值范围为 [-1.0, 1.0]
        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))

        # 设置全局最优解为所有维度为 0 的情况
        self.global_optimum = [[1e-16 for _ in range(self.N)]]
        
        # 设置全局最优值为 0.0
        self.fglob = 0.0
        
        # 设置维度变化标志为 True
        self.change_dimensionality = True

    # 定义目标函数 fun，接受参数 x 和可选参数 *args
    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 计算并返回目标函数值，按照公式 sum(x ** 6.0 * (sin(1.0 / x) + 2.0)) 计算
        return sum(x ** 6.0 * (sin(1.0 / x) + 2.0))
```