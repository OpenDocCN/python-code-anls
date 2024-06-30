# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_S.py`

```
from numpy import (abs, asarray, cos, floor, arange, pi, prod, roll, sin,
                   sqrt, sum, repeat, atleast_2d, tril)
from numpy.random import uniform
from .go_benchmark import Benchmark

class Salomon(Benchmark):
    """
    Salomon objective function.

    This class defines the Salomon [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Salomon}}(x) = 1 - \cos \left (2 \pi
        \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in
    [-100, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的构造函数，设置问题的维度
        Benchmark.__init__(self, dimensions)
        # 设置搜索空间的边界
        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))
        # 设置自定义边界
        self.custom_bounds = [(-50, 50), (-50, 50)]
        # 设置全局最优解
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        # 设置全局最优值
        self.fglob = 0.0
        # 设置维度变换标志
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 计算向量的欧几里得范数
        u = sqrt(sum(x ** 2))
        # 计算 Salomon 函数的值
        return 1 - cos(2 * pi * u) + 0.1 * u


class Sargan(Benchmark):
    """
    Sargan objective function.

    This class defines the Sargan [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Sargan}}(x) = \sum_{i=1}^{n} n \left (x_i^2
        + 0.4 \sum_{i \neq j}^{n} x_ix_j \right)

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-100, 100]` for
    :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的构造函数，设置问题的维度
        Benchmark.__init__(self, dimensions)
        # 设置搜索空间的边界
        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))
        # 设置自定义边界
        self.custom_bounds = [(-5, 5), (-5, 5)]
        # 设置全局最优解
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        # 设置全局最优值
        self.fglob = 0.0
        # 设置维度变换标志
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 将输入向量分解成 x0 和 x1
        x0 = x[:-1]
        x1 = roll(x, -1)[:-1]

        # 计算 Sargan 函数的值
        return sum(self.N * (x ** 2 + 0.4 * sum(x0 * x1)))


class Schaffer01(Benchmark):
    """
    Schaffer 1 objective function.

    This class defines the Schaffer 1 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:
    """
    # 定义一个多维优化基准类的子类，用于评估Schaffer01函数
    def __init__(self, dimensions=2):
        # 调用父类Benchmark的初始化方法，设定维度
        Benchmark.__init__(self, dimensions)

        # 设定变量的取值范围为[-100, 100]
        self._bounds = list(zip([-100.0] * self.N,
                           [100.0] * self.N))
        
        # 设置自定义的边界条件为[-10, 10]
        self.custom_bounds = [(-10, 10), (-10, 10)]

        # 设置全局最优解为[0, 0]，对应的函数值为0
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 0.0

    # 定义Schaffer01函数，计算给定点x处的函数值
    def fun(self, x, *args):
        # 增加函数评估次数的计数器
        self.nfev += 1

        # 计算Schaffer01函数中的u值
        u = (x[0] ** 2 + x[1] ** 2)
        
        # 计算Schaffer01函数的分子部分
        num = sin(u) ** 2 - 0.5
        
        # 计算Schaffer01函数的分母部分
        den = (1 + 0.001 * u) ** 2
        
        # 计算Schaffer01函数的值并返回
        return 0.5 + num / den
class Schaffer04(Benchmark):
    """
    Schaffer 4 objective function.

    This class defines the Schaffer 4 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Schaffer04}}(x) = 0.5 + \frac{\cos^2 \left( \sin(x_1^2 - x_2^2)
        \right ) - 0.5}{1 + 0.001(x_1^2 + x_2^2)^2}^2

    with :math:`x_i \in [-100, 100]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0.292579` for :math:`x = [0, 1.253115]`

    .. [1] Mishra, S. Some new test functions for global optimization and
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)  # 调用父类 Benchmark 的初始化方法，设定维度

        self._bounds = list(zip([-100.0] * self.N,  # 设置变量 x 的范围上下限
                           [100.0] * self.N))
        self.custom_bounds = [(-10, 10), (-10, 10)]  # 设置自定义的变量 x 的范围

        self.global_optimum = [[0.0, 1.253115]]  # 设置全局最优解的坐标
        self.fglob = 0.292579  # 设置全局最优解的函数值

    def fun(self, x, *args):
        self.nfev += 1  # 记录函数调用次数

        num = cos(sin(x[0] ** 2 - x[1] ** 2)) ** 2 - 0.5  # 计算分子部分
        den = (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2  # 计算分母部分
        return 0.5 + num / den  # 计算目标函数值并返回
    performance of repulsive particle swarm method.
    Munich Personal RePEc Archive, 2006, 2718
    """

    # 初始化类，设定默认维度为2
    def __init__(self, dimensions=2):
        # 调用父类Benchmark的初始化方法，设置维度
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间的边界，这里默认为每个维度 [-100.0, 100.0]
        self._bounds = list(zip([-100.0] * self.N,
                           [100.0] * self.N))
        
        # 设置自定义的边界，这里是特定维度的边界限制 (-10, 10)
        self.custom_bounds = [(-10, 10), (-10, 10)]

        # 设定全局最优解的位置
        self.global_optimum = [[0.0, 1.253115]]
        
        # 设定全局最优解的适应度值
        self.fglob = 0.292579

    # 定义优化问题的目标函数
    def fun(self, x, *args):
        # 每次调用目标函数，增加计算的次数
        self.nfev += 1

        # 计算目标函数的表达式
        num = cos(sin(abs(x[0] ** 2 - x[1] ** 2))) ** 2 - 0.5
        den = (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
        return 0.5 + num / den
# class Schwefel02(Benchmark):
#
#     r"""
#     Schwefel 2 objective function.
#
#     This class defines the Schwefel 2 [1]_ global optimization problem. This
#     is a unimodal minimization problem defined as follows:
#
#     .. math::
#
#         f_{\text{Schwefel02}}(x) = \sum_{i=1}^n \left(\sum_{j=1}^i 
#         x_i \right)^2
#
#
#     Here, :math:`n` represents the number of dimensions and
#     :math:`x_i \in [-100, 100]` for :math:`i = 1, ..., n`.
#
#     *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
#     :math:`i = 1, ..., n`
#     """
#
#     def __init__(self, dimensions=2):
#         Benchmark.__init__(self, dimensions)
#         self._bounds = list(zip([-100.0] * self.N,
#                            [100.0] * self.N))
#         self.custom_bounds = ([-4.0, 4.0], [-4.0, 4.0])
#
#         self.global_optimum = [[0.0 for _ in range(self.N)]]
#         self.fglob = 0.0
#         self.change_dimensionality = True
#
#     def fun(self, x, *args):
#         self.nfev += 1
#
#         # Calculate the Schwefel 2 objective function for the given point x
#         return sum([(sum(x[:i+1])) ** 2 for i in range(self.N)])
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    # 继承Benchmark类的构造函数，初始化Benchmark2类
    def __init__(self, dimensions=2):
        # 调用父类Benchmark的初始化函数，设置维度
        Benchmark.__init__(self, dimensions)
        # 定义默认边界为 [-100.0, 100.0] 并存储到_bounds变量中
        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))
        # 定义自定义边界为 [-4.0, 4.0] 并存储到custom_bounds变量中
        self.custom_bounds = ([-4.0, 4.0], [-4.0, 4.0])
        # 设置全局最优解为 [0.0, 0.0, ..., 0.0]，存储到global_optimum变量中
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        # 初始化全局最优解的函数值为0.0，存储到fglob变量中
        self.fglob = 0.0
        # 标记改变维度的标志为True，用于表明函数可以处理不同维度的输入
        self.change_dimensionality = True

    # 定义函数fun，计算特定输入x的函数值，并增加评估计数
    def fun(self, x, *args):
        # 增加函数评估计数
        self.nfev += 1

        # 生成一个重复x的矩阵，并取其下三角部分的和
        mat = repeat(atleast_2d(x), self.N, axis=0)
        inner = sum(tril(mat), axis=1)
        # 返回内部和的平方和作为函数值
        return sum(inner ** 2)
class Schwefel04(Benchmark):
    
    r"""
    Schwefel 4 objective function.

    This class defines the Schwefel 4 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Schwefel04}}(x) = \sum_{i=1}^n \left[(x_i - 1)^2
        + (x_1 - x_i^2)^2 \right]


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [0, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for:math:`x_i = 1` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的初始化方法，并设定维度
        Benchmark.__init__(self, dimensions)
        # 设置搜索空间边界
        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))
        # 设置自定义搜索空间边界
        self.custom_bounds = ([0.0, 2.0], [0.0, 2.0])

        # 设置全局最优解
        self.global_optimum = [[1.0 for _ in range(self.N)]]
        # 设置全局最优解的函数值
        self.fglob = 0.0
        # 改变维度的标志
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估次数的计数器
        self.nfev += 1

        # 计算 Schwefel 4 函数的值
        return sum((x - 1.0) ** 2.0 + (x[0] - x ** 2.0) ** 2.0)


class Schwefel06(Benchmark):

    r"""
    Schwefel 6 objective function.

    This class defines the Schwefel 6 [1]_ global optimization problem. This
    is a unimodal minimization problem defined as follows:

    .. math::

       f_{\text{Schwefel06}}(x) = \max(\lvert x_1 + 2x_2 - 7 \rvert,
                                   \lvert 2x_1 + x_2 - 5 \rvert)


    with :math:`x_i \in [-100, 100]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 3]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的初始化方法，并设定维度
        Benchmark.__init__(self, dimensions)
        # 设置搜索空间边界
        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))
        # 设置自定义搜索空间边界
        self.custom_bounds = ([-10.0, 10.0], [-10.0, 10.0])

        # 设置全局最优解
        self.global_optimum = [[1.0, 3.0]]
        # 设置全局最优解的函数值
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估次数的计数器
        self.nfev += 1

        # 计算 Schwefel 6 函数的值
        return max(abs(x[0] + 2 * x[1] - 7), abs(2 * x[0] + x[1] - 5))


class Schwefel20(Benchmark):

    r"""
    Schwefel 20 objective function.

    This class defines the Schwefel 20 [1]_ global optimization problem. This
    is a unimodal minimization problem defined as follows:

    .. math::

       f_{\text{Schwefel20}}(x) = \sum_{i=1}^n \lvert x_i \rvert


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-100, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    """

    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的初始化方法，并设定维度
        Benchmark.__init__(self, dimensions)
        # 设置搜索空间边界
        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))

        # 设置全局最优解
        self.global_optimum = [[0.0] * self.N]
        # 设置全局最优解的函数值
        self.fglob = 0.0
    # 定义一个新的类 `Schwefel`, 继承自 `Benchmark` 类
    class Schwefel(Benchmark):
    
        # 初始化方法，接受一个参数 `dimensions` 默认为 2
        def __init__(self, dimensions=2):
            # 调用父类 `Benchmark` 的初始化方法，传入 `dimensions` 参数
            Benchmark.__init__(self, dimensions)
            
            # 设置 `_bounds` 属性为一个列表，元素是元组，每个元组包含两个值 -100.0 和 100.0，共 `self.N` 个元组
            self._bounds = list(zip([-100.0] * self.N,
                               [100.0] * self.N))
    
            # 初始化 `global_optimum` 属性为一个列表，包含一个长度为 `self.N` 的列表，每个元素为 0.0
            self.global_optimum = [[0.0 for _ in range(self.N)]]
    
            # 初始化 `fglob` 属性为 0.0
            self.fglob = 0.0
    
            # 设置 `change_dimensionality` 属性为 True
            self.change_dimensionality = True
    
        # 定义一个方法 `fun`，接受参数 `x` 和 `*args`
        def fun(self, x, *args):
            # 增加 `nfev` 属性的值
            self.nfev += 1
    
            # 返回向量 `x` 中所有元素绝对值的和作为函数值
            return sum(abs(x))
class Schwefel21(Benchmark):
    r"""
    Schwefel 21 objective function.

    This class defines the Schwefel 21 [1]_ global optimization problem. This
    is a unimodal minimization problem defined as follows:

    .. math::

        f_{\text{Schwefel21}}(x) = \smash{\displaystyle\max_{1 \leq i \leq n}}
                                   \lvert x_i \rvert

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-100, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设置维度
        Benchmark.__init__(self, dimensions)
        # 设置变量边界为 [-100, 100] 的列表
        self._bounds = list(zip([-100.0] * self.N,
                           [100.0] * self.N))

        # 设置全局最优解为每维度为 0 的列表
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        # 设置全局最优解的函数值为 0.0
        self.fglob = 0.0
        # 标记是否改变维度
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 每调用一次 fun 方法，增加一次函数评估次数计数器
        self.nfev += 1

        # 返回向量 x 中绝对值的最大值
        return max(abs(x))


class Schwefel22(Benchmark):
    r"""
    Schwefel 22 objective function.

    This class defines the Schwefel 22 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Schwefel22}}(x) = \sum_{i=1}^n \lvert x_i \rvert
                                  + \prod_{i=1}^n \lvert x_i \rvert

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-100, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设置维度
        Benchmark.__init__(self, dimensions)
        # 设置变量边界为 [-100, 100] 的列表
        self._bounds = list(zip([-100.0] * self.N,
                           [100.0] * self.N))
        # 设置自定义边界为 [-10, 10] 的元组对
        self.custom_bounds = ([-10.0, 10.0], [-10.0, 10.0])

        # 设置全局最优解为每维度为 0 的列表
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        # 设置全局最优解的函数值为 0.0
        self.fglob = 0.0
        # 标记是否改变维度
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 每调用一次 fun 方法，增加一次函数评估次数计数器
        self.nfev += 1

        # 返回向量 x 中绝对值的和加上绝对值的乘积
        return sum(abs(x)) + prod(abs(x))


class Schwefel26(Benchmark):
    r"""
    Schwefel 26 objective function.

    This class defines the Schwefel 26 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Schwefel26}}(x) = 418.9829n - \sum_{i=1}^n x_i
                                  \sin(\sqrt{|x_i|})

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-500, 500]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 420.968746` for

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设置维度
        Benchmark.__init__(self, dimensions)
        # 设置变量边界为 [-500, 500] 的列表
        self._bounds = list(zip([-500.0] * self.N,
                           [500.0] * self.N))

        # 设置全局最优解为每维度为 420.968746 的列表
        self.global_optimum = [[420.968746 for _ in range(self.N)]]
        # 设置全局最优解的函数值为 0.0
        self.fglob = 0.0
        # 标记是否改变维度
        self.change_dimensionality = True
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    # 定义一个新的类，继承自Benchmark类，初始化方法接收一个维度参数，默认为2
    def __init__(self, dimensions=2):
        # 调用父类Benchmark的初始化方法，传入维度参数
        Benchmark.__init__(self, dimensions)
        # 设置变量_bounds，是一个列表，包含N个(-500.0)到(500.0)的元组，每个元组表示一对边界
        self._bounds = list(zip([-500.0] * self.N,
                           [500.0] * self.N))

        # 设置全局最优值global_optimum，是一个列表，包含一个长度为N的列表，每个元素为420.968746
        self.global_optimum = [[420.968746 for _ in range(self.N)]]
        
        # 设置fglob为0.0，表示全局最优解的目标函数值
        self.fglob = 0.0
        
        # 设置change_dimensionality为True，表示可以改变维度
        self.change_dimensionality = True

    # 定义方法fun，接收参数x和可变参数args
    def fun(self, x, *args):
        # 每调用一次fun方法，增加nfev计数
        self.nfev += 1
        
        # 返回目标函数的计算结果
        return 418.982887 * self.N - sum(x * sin(sqrt(abs(x))))
class Schwefel36(Benchmark):
    
    r"""
    Schwefel 36 objective function.
    
    This class defines the Schwefel 36 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:
    
    .. math::
    
        f_{\text{Schwefel36}}(x) = -x_1x_2(72 - 2x_1 - 2x_2)
    
    
    with :math:`x_i \in [0, 500]` for :math:`i = 1, 2`.
    
    *Global optimum*: :math:`f(x) = -3456` for :math:`x = [12, 12]`
    
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """
    
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设置维度数
        Benchmark.__init__(self, dimensions)
        # 设置变量边界为 [0.0, 500.0] 的列表
        self._bounds = list(zip([0.0] * self.N, [500.0] * self.N))
        # 设置自定义边界为 ([0.0, 20.0], [0.0, 20.0])
        self.custom_bounds = ([0.0, 20.0], [0.0, 20.0])
        
        # 设置全局最优解为 [[12.0, 12.0]]
        self.global_optimum = [[12.0, 12.0]]
        # 设置全局最优函数值为 -3456.0
        self.fglob = -3456.0
    
    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1
        
        # 计算 Schwefel 36 函数的值
        return -x[0] * x[1] * (72 - 2 * x[0] - 2 * x[1])


class Shekel05(Benchmark):
    
    r"""
    Shekel 5 objective function.
    
    This class defines the Shekel 5 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:
    
    .. math::
    
        f_{\text{Shekel05}}(x) = \sum_{i=1}^{m} \frac{1}{c_{i}
        + \sum_{j=1}^{n} (x_{j} - a_{ij})^2 }
    
    Where, in this exercise:
    
    .. math::
    
        a = 
        \begin{bmatrix}
        4.0 & 4.0 & 4.0 & 4.0 \\ 1.0 & 1.0 & 1.0 & 1.0 \\
        8.0 & 8.0 & 8.0 & 8.0 \\ 6.0 & 6.0 & 6.0 & 6.0 \\
        3.0 & 7.0 & 3.0 & 7.0 
        \end{bmatrix}
    .. math::
    
        c = \begin{bmatrix} 0.1 \\ 0.2 \\ 0.2 \\ 0.4 \\ 0.4 \end{bmatrix}
    
    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [0, 10]` for :math:`i = 1, ..., 4`.
    
    *Global optimum*: :math:`f(x) = -10.15319585` for :math:`x_i = 4` for
    :math:`i = 1, ..., 4`
    
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    
    TODO: this is a different global minimum compared to Jamil#130.  The
    minimum is found by doing lots of optimisations. The solution is supposed
    to be at [4] * N, is there any numerical overflow?
    """
    # 初始化函数，设置默认维度为4
    def __init__(self, dimensions=4):
        # 调用 Benchmark 类的初始化方法，传入维度参数
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间的边界，每个维度的范围为 [0.0, 10.0]
        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))

        # 设置全局最优解，是一个包含一个列表的列表
        self.global_optimum = [[4.00003715092,
                                4.00013327435,
                                4.00003714871,
                                4.0001332742]]
        
        # 设置全局最优解对应的函数值
        self.fglob = -10.1531996791
        
        # 设置矩阵 A，包含多行，每行有4个元素
        self.A = asarray([[4.0, 4.0, 4.0, 4.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [8.0, 8.0, 8.0, 8.0],
                          [6.0, 6.0, 6.0, 6.0],
                          [3.0, 7.0, 3.0, 7.0]])
        
        # 设置向量 C，包含5个元素
        self.C = asarray([0.1, 0.2, 0.2, 0.4, 0.4])

    # 定义评估函数 fun，计算目标函数的值
    def fun(self, x, *args):
        # 增加评估函数调用次数
        self.nfev += 1

        # 计算目标函数值，返回负的总和
        return -sum(1 / (sum((x - self.A) ** 2, axis=1) + self.C))
class Shekel07(Benchmark):
    r"""
    Shekel 7 objective function.

    This class defines the Shekel 7 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Shekel07}}(x) = \sum_{i=1}^{m} \frac{1}{c_{i}
                                 + \sum_{j=1}^{n} (x_{j} - a_{ij})^2 }

    Where, in this exercise:

    .. math::

        a =
        \begin{bmatrix}
        4.0 & 4.0 & 4.0 & 4.0 \\
        1.0 & 1.0 & 1.0 & 1.0 \\
        8.0 & 8.0 & 8.0 & 8.0 \\
        6.0 & 6.0 & 6.0 & 6.0 \\
        3.0 & 7.0 & 3.0 & 7.0 \\
        2.0 & 9.0 & 2.0 & 9.0 \\
        5.0 & 5.0 & 3.0 & 3.0
        \end{bmatrix}

    .. math::

        c =
        \begin{bmatrix}
        0.1 \\ 0.2 \\ 0.2 \\ 0.4 \\ 0.4 \\ 0.6 \\ 0.3 
        \end{bmatrix}

    with :math:`x_i \in [0, 10]` for :math:`i = 1, ..., 4`.

    *Global optimum*: :math:`f(x) = -10.4028188` for :math:`x_i = 4` for
    :math:`i = 1, ..., 4`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: this is a different global minimum compared to Jamil#131. This
    minimum is obtained after running lots of minimisations!  Is there any
    numerical overflow that causes the minimum solution to not be [4] * N?
    """

    def __init__(self, dimensions=4):
        # 调用父类构造函数初始化基准类
        Benchmark.__init__(self, dimensions)

        # 定义变量上下界
        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))

        # 设置全局最优解
        self.global_optimum = [[4.00057291078,
                                4.0006893679,
                                3.99948971076,
                                3.99960615785]]
        # 设置全局最优值
        self.fglob = -10.4029405668

        # 设置矩阵 A
        self.A = asarray([[4.0, 4.0, 4.0, 4.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [8.0, 8.0, 8.0, 8.0],
                          [6.0, 6.0, 6.0, 6.0],
                          [3.0, 7.0, 3.0, 7.0],
                          [2.0, 9.0, 2.0, 9.0],
                          [5.0, 5.0, 3.0, 3.0]])

        # 设置向量 C
        self.C = asarray([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3])

    def fun(self, x, *args):
        # 增加函数评估次数
        self.nfev += 1

        # 计算 Shekel 7 函数值并返回
        return -sum(1 / (sum((x - self.A) ** 2, axis=1) + self.C))


class Shekel10(Benchmark):
    r"""
    Shekel 10 objective function.

    This class defines the Shekel 10 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Shekel10}}(x) = \sum_{i=1}^{m} \frac{1}{c_{i} 
                                + \sum_{j=1}^{n} (x_{j} - a_{ij})^2 }`

    Where, in this exercise:
    # 定义一个类，继承自Benchmark类，用于处理dimensions维度的基准测试问题
    def __init__(self, dimensions=4):
        # 调用父类Benchmark的初始化方法，设定问题的维度
        Benchmark.__init__(self, dimensions)

        # 设置问题的变量边界，每个变量的取值范围为[0.0, 10.0]
        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))

        # 设置全局最优解，这里给出了一个二维数组，表示问题的全局最优解
        self.global_optimum = [[4.0007465377266271,
                                4.0005929234621407,
                                3.9996633941680968,
                                3.9995098017834123]]
        # 设置全局最优解对应的目标函数值
        self.fglob = -10.536409816692023

        # 设置矩阵A，作为问题公式中的系数矩阵，存储为NumPy数组
        self.A = asarray([[4.0, 4.0, 4.0, 4.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [8.0, 8.0, 8.0, 8.0],
                          [6.0, 6.0, 6.0, 6.0],
                          [3.0, 7.0, 3.0, 7.0],
                          [2.0, 9.0, 2.0, 9.0],
                          [5.0, 5.0, 3.0, 3.0],
                          [8.0, 1.0, 8.0, 1.0],
                          [6.0, 2.0, 6.0, 2.0],
                          [7.0, 3.6, 7.0, 3.6]])

        # 设置向量C，作为问题公式中的常数项，存储为NumPy数组
        self.C = asarray([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

    # 定义问题的目标函数，计算给定变量x的函数值
    def fun(self, x, *args):
        # 增加函数评估次数的计数器
        self.nfev += 1

        # 计算目标函数值，这里是一个负的总和表达式，表示最小化问题
        return -sum(1 / (sum((x - self.A) ** 2, axis=1) + self.C))
class Shubert01(Benchmark):
    """
    Shubert 1 objective function.

    This class defines the Shubert 1 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Shubert01}}(x) = \prod_{i=1}^{n}\left(\sum_{j=1}^{5}
                                  cos(j+1)x_i+j \right )

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -186.7309` for
    :math:`x = [-7.0835, 4.8580]` (and many others).

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO: Jamil#133 is missing a prefactor of j before the cos function.
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的构造函数，并初始化维度数
        Benchmark.__init__(self, dimensions)

        # 定义搜索空间边界
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))

        # 设置全局最优解和对应的函数值
        self.global_optimum = [[-7.0835, 4.8580]]
        self.fglob = -186.7309

        # 设置是否改变维度标志
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估计数
        self.nfev += 1

        # 生成 5x1 的列向量 j，包含从1到5的整数
        j = atleast_2d(arange(1, 6)).T

        # 计算 Shubert 1 函数的每个分量
        y = j * cos((j + 1) * x + j)

        # 返回所有分量的乘积作为最终的函数值
        return prod(sum(y, axis=0))


class Shubert03(Benchmark):
    """
    Shubert 3 objective function.

    This class defines the Shubert 3 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Shubert03}}(x) = \sum_{i=1}^n \sum_{j=1}^5 -j 
                                  \sin((j+1)x_i + j)

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -24.062499` for
    :math:`x = [5.791794, 5.791794]` (and many others).

     .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO: Jamil#134 has wrong global minimum value, and is missing a minus sign
    before the whole thing.
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的构造函数，并初始化维度数
        Benchmark.__init__(self, dimensions)

        # 定义搜索空间边界
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))

        # 设置全局最优解和对应的函数值
        self.global_optimum = [[5.791794, 5.791794]]
        self.fglob = -24.062499

        # 设置是否改变维度标志
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估计数
        self.nfev += 1

        # 生成 5x1 的列向量 j，包含从1到5的整数
        j = atleast_2d(arange(1, 6)).T

        # 计算 Shubert 3 函数的每个分量
        y = -j * sin((j + 1) * x + j)

        # 返回所有分量的和作为最终的函数值
        return sum(sum(y))


class Shubert04(Benchmark):
    """
    Shubert 4 objective function.

    This class defines the Shubert 4 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Shubert04}}(x) = \left(\sum_{i=1}^n \sum_{j=1}^5 -j
                                  \cos ((j+1)x_i + j)\right)

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -29.016015` for
    :math:`x = [-0.80032121, -7.08350592]` (and many others).
    """
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    
    TODO: Jamil#135 has wrong global minimum value, and is missing a minus sign
    before the whole thing.
    """
    
    # 定义一个名为 Benchmark 的基类的子类 Jamil 类
    class Jamil(Benchmark):
        
        # 初始化方法，设置维度，默认为 2 维
        def __init__(self, dimensions=2):
            # 调用父类 Benchmark 的初始化方法，传入维度参数
            Benchmark.__init__(self, dimensions)
            
            # 设置变量 _bounds 为元组列表，表示每个维度的取值范围 [-10.0, 10.0]
            self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
            
            # 设置全局最优解为特定值的列表
            self.global_optimum = [[-0.80032121, -7.08350592]]
            
            # 设置全局最优解的函数值
            self.fglob = -29.016015
            
            # 设置变量 change_dimensionality 为 True
            self.change_dimensionality = True
    
        # 定义函数 fun，计算给定输入 x 的函数值
        def fun(self, x, *args):
            # 增加计算函数调用次数的计数器
            self.nfev += 1
            
            # 生成一个列向量 j，包含从 1 到 5 的整数，然后转置为列向量
            j = atleast_2d(arange(1, 6)).T
            
            # 计算函数的具体形式，返回一个标量值
            y = -j * cos((j + 1) * x + j)
            
            # 返回所有元素的和作为函数值
            return sum(sum(y))
# 定义一个继承自Benchmark的类SineEnvelope，表示SineEnvelope目标函数
class SineEnvelope(Benchmark):

    r"""
    SineEnvelope objective function.

    This class defines the SineEnvelope [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{SineEnvelope}}(x) = -\sum_{i=1}^{n-1}\left[\frac{\sin^2(
                                       \sqrt{x_{i+1}^2+x_{i}^2}-0.5)}
                                       {(0.001(x_{i+1}^2+x_{i}^2)+1)^2}
                                       + 0.5\right]

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-100, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO: Jamil #136
    """

    # 初始化方法，设置维度为2（默认值）
    def __init__(self, dimensions=2):
        # 调用父类Benchmark的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置变量_bounds为一个元组列表，每个维度的范围为[-100.0, 100.0]
        self._bounds = list(zip([-100.0] * self.N,
                           [100.0] * self.N))
        
        # 设置自定义边界custom_bounds为特定范围
        self.custom_bounds = [(-20, 20), (-20, 20)]

        # 设置全局最优解global_optimum为所有维度为0的列表
        self.global_optimum = [[0 for _ in range(self.N)]]

        # 设置全局最优值fglob为0.0
        self.fglob = 0.0

        # 设置维度变换标志为True
        self.change_dimensionality = True

    # 目标函数方法fun，计算SineEnvelope函数的值
    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 分别提取x的前n-1维和后n-1维
        X0 = x[:-1]
        X1 = x[1:]

        # 计算平方和的开方
        X02X12 = X0 ** 2 + X1 ** 2

        # 计算目标函数值并返回
        return sum((sin(sqrt(X02X12)) ** 2 - 0.5) / (1 + 0.001 * X02X12) ** 2
                   + 0.5)


# 定义一个继承自Benchmark的类SixHumpCamel，表示SixHumpCamel目标函数
class SixHumpCamel(Benchmark):

    r"""
    Six Hump Camel objective function.

    This class defines the Six Hump Camel [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{SixHumpCamel}}(x) = 4x_1^2+x_1x_2-4x_2^2-2.1x_1^4+
                                    4x_2^4+\frac{1}{3}x_1^6

    with :math:`x_i \in [-5, 5]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -1.031628453489877` for
    :math:`x = [0.08984201368301331 , -0.7126564032704135]` or 
    :math:`x = [-0.08984201368301331, 0.7126564032704135]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    # 初始化方法，设置维度为2（默认值）
    def __init__(self, dimensions=2):
        # 调用父类Benchmark的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置变量_bounds为一个元组列表，每个维度的范围为[-5.0, 5.0]
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        
        # 设置自定义边界custom_bounds为特定范围
        self.custom_bounds = [(-2, 2), (-1.5, 1.5)]

        # 设置全局最优解global_optimum为两个特定的点
        self.global_optimum = [(0.08984201368301331, -0.7126564032704135),
                               (-0.08984201368301331, 0.7126564032704135)]
        
        # 设置全局最优值fglob为特定值
        self.fglob = -1.031628

    # 目标函数方法fun，计算SixHumpCamel函数的值
    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 计算SixHumpCamel函数的值并返回
        return ((4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2 + x[0] * x[1]
                + (4 * x[1] ** 2 - 4) * x[1] ** 2)


class Sodp(Benchmark):

    r"""
    Sodp objective function.

    This class defines the Sum Of Different Powers [1]_ global optimization
    """
    problem. This is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Sodp}}(x) = \sum_{i=1}^{n} \lvert{x_{i}}\rvert^{i + 1}

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-1, 1]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """


    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))
        # 设置问题的搜索空间边界，每个维度的取值范围为 [-1.0, 1.0]

        self.global_optimum = [[0 for _ in range(self.N)]]
        # 全局最优解，当所有维度的值均为 0 时达到最优解

        self.fglob = 0.0
        # 全局最优值，对应于全局最优解为零

        self.change_dimensionality = True
        # 标志位，指示是否可以改变问题的维度


    def fun(self, x, *args):
        self.nfev += 1
        # 计算函数调用次数

        i = arange(1, self.N + 1)
        # 创建一个从 1 到 self.N 的数组，表示维度的序列

        return sum(abs(x) ** (i + 1))
        # 计算多模式问题的目标函数值，根据给定的公式求和
class Sphere(Benchmark):
    
    r"""
    Sphere objective function.
    
    This class defines the Sphere [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:
    
    .. math::
    
        f_{\text{Sphere}}(x) = \sum_{i=1}^{n} x_i^2
    
    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-1, 1]` for :math:`i = 1, ..., n`.
    
    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`
    
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    
    TODO Jamil has stupid limits
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的构造函数，设定维度
        Benchmark.__init__(self, dimensions)
        # 设置问题的搜索范围（边界）
        self._bounds = list(zip([-5.12] * self.N, [5.12] * self.N))

        # 设置全局最优解
        self.global_optimum = [[0 for _ in range(self.N)]]
        # 设置全局最优值
        self.fglob = 0.0
        # 标记是否需要改变维度
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加评估次数计数器
        self.nfev += 1
        
        # 计算 Sphere 函数的值
        return sum(x ** 2)


class Step(Benchmark):

    r"""
    Step objective function.

    This class defines the Step [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Step}}(x) = \sum_{i=1}^{n} \left ( \lfloor x_i
                             + 0.5 \rfloor \right )^2

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-100, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0.5` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的构造函数，设定维度
        Benchmark.__init__(self, dimensions)
        # 设置问题的搜索范围（边界）
        self._bounds = list(zip([-100.0] * self.N,
                           [100.0] * self.N))
        # 设置自定义搜索范围
        self.custom_bounds = ([-5, 5], [-5, 5])

        # 设置全局最优解
        self.global_optimum = [[0. for _ in range(self.N)]]
        # 设置全局最优值
        self.fglob = 0.0
        # 标记是否需要改变维度
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加评估次数计数器
        self.nfev += 1
        
        # 计算 Step 函数的值
        return sum(floor(abs(x)))


class Step2(Benchmark):

    r"""
    Step objective function.

    This class defines the Step 2 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Step}}(x) = \sum_{i=1}^{n} \left ( \lfloor x_i
                             + 0.5 \rfloor \right )^2

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-100, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0.5` for
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """
    # 初始化函数，设置对象的维度参数
    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的初始化方法，传入维度参数
        Benchmark.__init__(self, dimensions)
        # 根据维度生成边界列表，每个维度的边界为 [-100.0, 100.0]
        self._bounds = list(zip([-100.0] * self.N,
                           [100.0] * self.N))
        # 设置自定义边界
        self.custom_bounds = ([-5, 5], [-5, 5])

        # 设置全局最优解，每个维度的值为 0.5
        self.global_optimum = [[0.5 for _ in range(self.N)]]
        # 设置全局最优值
        self.fglob = 0.5
        # 标记需改变维度性质的状态为真
        self.change_dimensionality = True

    # 定义函数 fun，计算目标函数值，并增加计数器 nfev
    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 计算目标函数值，返回结果
        return sum((floor(x) + 0.5) ** 2.0)
# 定义一个继承自Benchmark类的Stochastic类，表示随机目标函数优化问题
class Stochastic(Benchmark):

    r"""
    Stochastic objective function.

    This class defines the Stochastic [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Stochastic}}(x) = \sum_{i=1}^{n} \epsilon_i 
                                    \left | {x_i - \frac{1}{i}} \right |

    The variable :math:`\epsilon_i, (i=1,...,n)` is a random variable uniformly
    distributed in :math:`[0, 1]`.

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-5, 5]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = [1/n]` for
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=2):
        # 调用父类Benchmark的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置变量边界为 [-5.0, 5.0] 的列表，表示每个维度的取值范围
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))

        # 全局最优解，对于每个维度i，设置为 [1/i] 的列表
        self.global_optimum = [[1.0 / _ for _ in range(1, self.N + 1)]]
        
        # 全局最优解函数值为 0.0
        self.fglob = 0.0
        
        # 表示是否改变维度的标志，设置为True
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 生成一个长度为self.N的均匀分布的随机数数组rnd
        rnd = uniform(0.0, 1.0, size=(self.N, ))
        
        # 生成一个从1到self.N的整数数组i
        i = arange(1, self.N + 1)

        # 计算函数值，返回rnd和|x - 1/i|的乘积之和
        return sum(rnd * abs(x - 1.0 / i))


# 定义一个继承自Benchmark类的StretchedV类，表示Stretched V目标函数优化问题
class StretchedV(Benchmark):

    r"""
    StretchedV objective function.

    This class defines the Stretched V [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{StretchedV}}(x) = \sum_{i=1}^{n-1} t^{1/4}
                                   [\sin (50t^{0.1}) + 1]^2

    Where, in this exercise:

    .. math::

       t = x_{i+1}^2 + x_i^2


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0., 0.]` when
    :math:`n = 2`.

    .. [1] Adorio, E. MVF - "Multivariate Test Functions Library in C for
    Unconstrained Global Optimization", 2005

    TODO All the sources disagree on the equation, in some the 1 is in the
    brackets, in others it is outside. In Jamil#142 it's not even 1. Here
    we go with the Adorio option.
    """

    def __init__(self, dimensions=2):
        # 调用父类Benchmark的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置变量边界为 [-10, 10] 的列表，表示每个维度的取值范围
        self._bounds = list(zip([-10] * self.N, [10] * self.N))

        # 全局最优解，设置为 [0, 0]
        self.global_optimum = [[0, 0]]
        
        # 全局最优解函数值为 0.0
        self.fglob = 0.0
        
        # 表示是否改变维度的标志，设置为True
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 计算t的值，t = x_{i+1}^2 + x_i^2，其中i从0到self.N-2
        t = x[1:] ** 2 + x[: -1] ** 2
        
        # 计算函数值，返回t^{1/4} * [\sin (50t^{0.1}) + 1]^2 的和
        return sum(t ** 0.25 * (sin(50.0 * t ** 0.1 + 1) ** 2))


# 定义一个继承自Benchmark类的StyblinskiTang类，表示Styblinski-Tang目标函数优化问题
class StyblinskiTang(Benchmark):

    r"""
    StyblinskiTang objective function.

    This class defines the Styblinski-Tang [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{StyblinskiTang}}(x) = \sum_{i=1}^{n} \left(x_i^4
                                       - 16x_i^2 + 5x_i \right)
    """
    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-5, 5]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -39.16616570377142n` for
    :math:`x_i = -2.903534018185960` for :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    # 定义一个 Benchmark 类的子类，初始化时设置维度，默认为 2
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，传入维度参数
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间的边界，每个维度的范围均为 [-5.0, 5.0]
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))

        # 设置全局最优解，对每个维度都设置为 [-2.903534018185960]
        self.global_optimum = [[-2.903534018185960 for _ in range(self.N)]]

        # 设置全局最优解对应的函数值
        self.fglob = -39.16616570377142 * self.N

        # 标记可以改变维度
        self.change_dimensionality = True

    # 定义评估函数 fun，计算给定点 x 的函数值
    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 计算函数值，这里使用了经典的 benchmark 函数形式
        return sum(x ** 4 - 16 * x ** 2 + 5 * x) / 2
```