# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_O.py`

```
# 从 numpy 中导入所需的函数和模块：sum, cos, exp, pi, asarray
# 还从当前目录的 go_benchmark 模块导入 Benchmark 类
from numpy import sum, cos, exp, pi, asarray
from .go_benchmark import Benchmark

# 定义一个名为 OddSquare 的类，它继承自 Benchmark 类
class OddSquare(Benchmark):

    r"""
    Odd Square objective function.

    This class defines the Odd Square [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{OddSquare}}(x) = -e^{-\frac{d}{2\pi}} \cos(\pi d)
       \left( 1 + \frac{0.02h}{d + 0.01} \right )


    Where, in this exercise:

    .. math::

        \begin{cases}
        d = n \cdot \smash{\displaystyle\max_{1 \leq i \leq n}} 
            \left[ (x_i - b_i)^2 \right ] \\
        h = \sum_{i=1}^{n} (x_i - b_i)^2
        \end{cases}

    And :math:`b = [1, 1.3, 0.8, -0.4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4, 1, 1.3,
                    0.8, -0.4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4]`

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-5 \pi, 5 \pi]` for :math:`i = 1, ..., n` and
    :math:`n \leq 20`.

    *Global optimum*: :math:`f(x_i) = -1.0084` for :math:`x \approx b`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO The best solution changes on dimensionality
    """

    # 类的初始化方法，接受一个 dimensions 参数，默认为 2
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，传入 dimensions
        Benchmark.__init__(self, dimensions)

        # 设置变量 _bounds，表示每个维度的取值范围，初始化为 [-5π, 5π] 的列表
        self._bounds = list(zip([-5.0 * pi] * self.N,
                           [5.0 * pi] * self.N))
        
        # 设置自定义的边界 custom_bounds，为一个固定的范围列表
        self.custom_bounds = ([-2.0, 4.0], [-2.0, 4.0])
        
        # 设置常量数组 a，包含 20 个预定义的浮点数
        self.a = asarray([1, 1.3, 0.8, -0.4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4]
                         * 2)
        
        # 设置全局最优解的坐标，为一个包含单个点的列表
        self.global_optimum = [[1.0873320463871847, 1.3873320456818079]]
        
        # 设置全局最优值 fglob，为一个特定的浮点数
        self.fglob = -1.00846728102

    # 定义函数 fun，用于计算目标函数的值，接受参数 x 和可选参数 args
    def fun(self, x, *args):
        # 增加函数评估的计数器 nfev
        self.nfev += 1
        
        # 从数组 a 中获取前 N 个元素作为向量 b
        b = self.a[0: self.N]
        
        # 计算变量 d，代表每个维度的最大差的平方与维度数的乘积
        d = self.N * max((x - b) ** 2.0)
        
        # 计算变量 h，代表所有维度上差的平方和
        h = sum((x - b) ** 2.0)
        
        # 计算并返回目标函数 f(x) 的值
        return (-exp(-d / (2.0 * pi)) * cos(pi * d)
                * (1.0 + 0.02 * h / (d + 0.01)))
```