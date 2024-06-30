# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_M.py`

```
# 从 numpy 库中导入多个函数，用于数学运算
from numpy import (abs, asarray, cos, exp, log, arange, pi, prod, sin, sqrt,
                   sum, tan)
# 从当前目录下的 go_benchmark 模块中导入 Benchmark 类和 safe_import 函数
from .go_benchmark import Benchmark, safe_import

# 使用 safe_import 上下文管理器导入 scipy 库中的 factorial 函数
with safe_import():
    from scipy.special import factorial


class Matyas(Benchmark):

    r"""
    Matyas objective function.

    This class defines the Matyas [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Matyas}}(x) = 0.26(x_1^2 + x_2^2) - 0.48 x_1 x_2


    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的构造函数初始化对象
        Benchmark.__init__(self, dimensions)

        # 设置问题的变量边界
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))

        # 设置全局最优解
        self.global_optimum = [[0 for _ in range(self.N)]]
        # 设置全局最优值
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 计算 Matyas 函数的值
        return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


class McCormick(Benchmark):

    r"""
    McCormick objective function.

    This class defines the McCormick [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

       f_{\text{McCormick}}(x) = - x_{1} + 2 x_{2} + \left(x_{1}
       - x_{2}\right)^{2} + \sin\left(x_{1} + x_{2}\right) + 1

    with :math:`x_1 \in [-1.5, 4]`, :math:`x_2 \in [-3, 4]`.

    *Global optimum*: :math:`f(x) = -1.913222954981037` for
    :math:`x = [-0.5471975602214493, -1.547197559268372]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的构造函数初始化对象
        Benchmark.__init__(self, dimensions)

        # 设置问题的变量边界
        self._bounds = [(-1.5, 4.0), (-3.0, 3.0)]

        # 设置全局最优解
        self.global_optimum = [[-0.5471975602214493, -1.547197559268372]]
        # 设置全局最优值
        self.fglob = -1.913222954981037

    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 计算 McCormick 函数的值
        return (sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0]
                + 2.5 * x[1] + 1)


class Meyer(Benchmark):

    r"""
    Meyer [1]_ objective function.

    ..[1] https://www.itl.nist.gov/div898/strd/nls/data/mgh10.shtml

    TODO NIST regression standard
    """
    # 初始化函数，用于设置对象的初始状态
    def __init__(self, dimensions=3):
        # 调用 Benchmark 类的初始化函数，继承其属性和方法
        Benchmark.__init__(self, dimensions)

        # 设置变量 _bounds，它是一个包含元组的列表，每个元组表示一个区间的上下界
        self._bounds = list(zip([0., 100., 100.],
                           [1, 1000., 500.]))

        # 设置全局最优解 global_optimum，它是一个包含一个列表的列表
        self.global_optimum = [[5.6096364710e-3, 6.1813463463e3,
                                3.4522363462e2]]

        # 设置 fglob，表示函数的全局最优值
        self.fglob = 8.7945855171e1

        # 设置系数 a，它是一个包含浮点数的 NumPy 数组
        self.a = asarray([3.478E+04, 2.861E+04, 2.365E+04, 1.963E+04, 1.637E+04,
                          1.372E+04, 1.154E+04, 9.744E+03, 8.261E+03, 7.030E+03,
                          6.005E+03, 5.147E+03, 4.427E+03, 3.820E+03, 3.307E+03,
                          2.872E+03])

        # 设置系数 b，它是一个包含浮点数的 NumPy 数组
        self.b = asarray([5.000E+01, 5.500E+01, 6.000E+01, 6.500E+01, 7.000E+01,
                          7.500E+01, 8.000E+01, 8.500E+01, 9.000E+01, 9.500E+01,
                          1.000E+02, 1.050E+02, 1.100E+02, 1.150E+02, 1.200E+02,
                          1.250E+02])

    # 函数 fun，用于计算给定参数 x 的函数值
    def fun(self, x, *args):
        # 每调用一次 fun 函数，增加 nfev（函数评估次数）计数器的值
        self.nfev += 1

        # 计算向量 vec 的值，根据公式 x[0] * exp(x[1] / (self.b + x[2]))
        vec = x[0] * exp(x[1] / (self.b + x[2]))

        # 计算并返回残差平方和，即 (self.a - vec) 的平方和
        return sum((self.a - vec) ** 2)
class Michalewicz(Benchmark):
    """
    Michalewicz objective function.

    This class defines the Michalewicz global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Michalewicz}}(x) = - \sum_{i=1}^{2} \sin\left(x_i\right)
       \sin^{2 m}\left(\frac{i x_i^{2}}{\pi}\right)


    Where, in this exercise, :math:`m = 10`.

    with :math:`x_i \in [0, \pi]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x_i) = -1.8013` for :math:`x = [0, 0]`

    .. [1] Adorio, E. MVF - "Multivariate Test Functions Library in C for
    Unconstrained Global Optimization", 2005

    TODO: could change dimensionality, but global minimum might change.
    """

    def __init__(self, dimensions=2):
        # 调用父类构造函数，设置维度
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间边界
        self._bounds = list(zip([0.0] * self.N, [pi] * self.N))

        # 设置全局最优解和对应的函数值
        self.global_optimum = [[2.20290555, 1.570796]]
        self.fglob = -1.8013

    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        m = 10.0
        i = arange(1, self.N + 1)
        # 计算 Michalewicz 函数的值
        return -sum(sin(x) * sin(i * x ** 2 / pi) ** (2 * m))


class MieleCantrell(Benchmark):
    """
    Miele-Cantrell objective function.

    This class defines the Miele-Cantrell global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{MieleCantrell}}({x}) = (e^{-x_1} - x_2)^4 + 100(x_2 - x_3)^6
       + \tan^4(x_3 - x_4) + x_1^8


    with :math:`x_i \in [-1, 1]` for :math:`i = 1, ..., 4`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 1, 1, 1]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=4):
        # 调用父类构造函数，设置维度
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间边界
        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))

        # 设置全局最优解和对应的函数值
        self.global_optimum = [[0.0, 1.0, 1.0, 1.0]]
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 计算 Miele-Cantrell 函数的值
        return ((exp(-x[0]) - x[1]) ** 4 + 100 * (x[1] - x[2]) ** 6
                + tan(x[2] - x[3]) ** 4 + x[0] ** 8)


class Mishra01(Benchmark):
    """
    Mishra 1 objective function.

    This class defines the Mishra 1 global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Mishra01}}(x) = (1 + x_n)^{x_n}


    where

    .. math::

        x_n = n - \sum_{i=1}^{n-1} x_i


    with :math:`x_i \in [0, 1]` for :math:`i =1, ..., n`.

    *Global optimum*: :math:`f(x) = 2` for :math:`x_i = 1` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """
    # 初始化函数，用于设置类的初始状态
    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的初始化函数，继承其属性和方法
        Benchmark.__init__(self, dimensions)
    
        # 设置问题空间的边界，构建一个由元组组成的列表，每个元组包含一个上下边界
        self._bounds = list(zip([0.0] * self.N, [1.0 + 1e-9] * self.N))
    
        # 设置全局最优解，这里是一个包含一个列表的列表，每个元素为维度数个 1.0
        self.global_optimum = [[1.0 for _ in range(self.N)]]
    
        # 设置全局最优解的函数值
        self.fglob = 2.0
    
        # 标记可以改变维度的特性
        self.change_dimensionality = True
    
    # 定义函数 fun，用于计算特定输入 x 的函数值
    def fun(self, x, *args):
        # 增加函数评估次数的计数器
        self.nfev += 1
    
        # 计算 xn，即输入 x 中除最后一个元素外的其他元素之和与 self.N 的差
        xn = self.N - sum(x[0:-1])
    
        # 返回函数值，根据给定公式计算
        return (1 + xn) ** xn
# 定义 Mishra02 类，继承自 Benchmark 类
class Mishra02(Benchmark):

    r"""
    Mishra 2 objective function.

    This class defines the Mishra 2 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Mishra02}}({x}) = (1 + x_n)^{x_n}


     with

     .. math::

         x_n = n - \sum_{i=1}^{n-1} \frac{(x_i + x_{i+1})}{2}


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [0, 1]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 2` for :math:`x_i = 1`
    for :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    # 初始化 Mishra02 类的实例
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设定问题的搜索空间边界，x_i 的取值范围为 [0.0, 1.0 + 1e-9]
        self._bounds = list(zip([0.0] * self.N,
                           [1.0 + 1e-9] * self.N))

        # 设置全局最优解
        self.global_optimum = [[1.0 for _ in range(self.N)]]
        # 设置全局最优解的函数值
        self.fglob = 2.0
        # 标志维度是否可变
        self.change_dimensionality = True

    # 定义 Mishra02 函数的计算方法
    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 计算 x_n
        xn = self.N - sum((x[:-1] + x[1:]) / 2.0)
        # 计算并返回目标函数值
        return (1 + xn) ** xn


# 定义 Mishra03 类，继承自 Benchmark 类
class Mishra03(Benchmark):

    r"""
    Mishra 3 objective function.

    This class defines the Mishra 3 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Mishra03}}(x) = \sqrt{\lvert \cos{\sqrt{\lvert x_1^2
       + x_2^2 \rvert}} \rvert} + 0.01(x_1 + x_2)


    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -0.1999` for
    :math:`x = [-9.99378322, -9.99918927]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: I think that Jamil#76 has the wrong global minimum, a smaller one
    is possible
    """

    # 初始化 Mishra03 类的实例
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设定问题的搜索空间边界，x_i 的取值范围为 [-10.0, 10.0]
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))

        # 设置全局最优解
        self.global_optimum = [[-9.99378322, -9.99918927]]
        # 设置全局最优解的函数值
        self.fglob = -0.19990562

    # 定义 Mishra03 函数的计算方法
    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 计算并返回目标函数值
        return (0.01 * (x[0] + x[1])
                + sqrt(abs(cos(sqrt(abs(x[0] ** 2 + x[1] ** 2))))))


# 定义 Mishra04 类，继承自 Benchmark 类
class Mishra04(Benchmark):

    r"""
    Mishra 4 objective function.

    This class defines the Mishra 4 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Mishra04}}({x}) = \sqrt{\lvert \sin{\sqrt{\lvert
       x_1^2 + x_2^2 \rvert}} \rvert} + 0.01(x_1 + x_2)

    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -0.17767` for
    :math:`x = [-8.71499636, -9.0533148]`
    """
    该部分代码定义了一个名为 Benchmark 的类，用于实现某个优化问题的基准函数。

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设置维度属性
        Benchmark.__init__(self, dimensions)

        # 设置问题的搜索空间边界
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))

        # 设置全局最优解
        self.global_optimum = [[-8.88055269734, -8.89097599857]]

        # 设置全局最优解对应的函数值
        self.fglob = -0.177715264826

    def fun(self, x, *args):
        # 每次调用函数计算时增加函数评估计数
        self.nfev += 1

        # 定义并返回优化问题的目标函数
        return (0.01 * (x[0] + x[1])
                + sqrt(abs(sin(sqrt(abs(x[0] ** 2 + x[1] ** 2))))))

    """
class Mishra07(Benchmark):

    r"""
    Mishra 7 objective function.

    This class defines the Mishra 7 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Mishra07}}(x) = \left [\prod_{i=1}^{n} x_i - n! \right]^2


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = \sqrt{n}`
    for :math:`i = 1, ..., n`

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间边界
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))

        # 全局最优解
        self.global_optimum = [[0.0] * self.N]

        # 全局最优值
        self.fglob = 0.0

    def fun(self, x, *args):
        self.nfev += 1

        # 计算目标函数值
        product_term = numpy.prod(x)  # 计算所有维度的乘积
        factorial_term = math.factorial(self.N)  # 计算阶乘 n!
        return (product_term - factorial_term) ** 2
    """
        .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
        For Global Optimization Problems Int. Journal of Mathematical Modelling
        and Numerical Optimisation, 2013, 4, 150-194.
        """
    
        # 初始化 Benchmark 类的子类，设定维度，默认为 2 维
        def __init__(self, dimensions=2):
            # 调用父类 Benchmark 的初始化方法
            Benchmark.__init__(self, dimensions)
    
            # 设定变量的取值范围为 [-10.0, 10.0]，维度由 self.N 决定
            self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
            
            # 自定义的变量取值范围，覆盖默认范围
            self.custom_bounds = [(-2, 2), (-2, 2)]
            
            # 全局最优解，每个维度的值为 sqrt(self.N)
            self.global_optimum = [[sqrt(self.N) for i in range(self.N)]]
            
            # 目标函数的全局最小值
            self.fglob = 0.0
            
            # 表示该函数可以改变维度（不是固定维度）
            self.change_dimensionality = True
    
        # 定义函数 fun，计算给定输入 x 的目标函数值
        def fun(self, x, *args):
            # 增加函数评估的次数计数器
            self.nfev += 1
    
            # 返回目标函数值，表达式为 (prod(x) - factorial(self.N)) 的平方
            return (prod(x) - factorial(self.N)) ** 2.0
class Mishra08(Benchmark):
    r"""
    Mishra 8 objective function.

    This class defines the Mishra 8 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Mishra08}}(x) = 0.001 \left[\lvert x_1^{10} - 20x_1^9
       + 180x_1^8 - 960 x_1^7 + 3360x_1^6 - 8064x_1^5 + 13340x_1^4 - 15360x_1^3
       + 11520x_1^2 - 5120x_1 + 2624 \rvert \lvert x_2^4 + 12x_2^3 + 54x_2^2
       + 108x_2 + 81 \rvert \right]^2

    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [2, -3]`

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005

    TODO Line 1065
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间边界为每个维度都是 [-10, 10]
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        
        # 自定义的特定边界，覆盖默认的边界设置
        self.custom_bounds = [(1.0, 2.0), (-4.0, 1.0)]
        
        # 全局最优解，这个函数的最小值在 x = [2.0, -3.0] 处
        self.global_optimum = [[2.0, -3.0]]
        
        # 设置全局最小值为 0
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估的计数
        self.nfev += 1

        # 计算 Mishra 8 函数的值
        val = abs(x[0] ** 10 - 20 * x[0] ** 9 + 180 * x[0] ** 8
                  - 960 * x[0] ** 7 + 3360 * x[0] ** 6 - 8064 * x[0] ** 5
                  + 13340 * x[0] ** 4 - 15360 * x[0] ** 3 + 11520 * x[0] ** 2
                  - 5120 * x[0] + 2624)
        val += abs(x[1] ** 4 + 12 * x[1] ** 3 +
                   54 * x[1] ** 2 + 108 * x[1] + 81)
        
        # 返回函数值乘以 0.001 的平方
        return 0.001 * val ** 2


class Mishra09(Benchmark):
    r"""
    Mishra 9 objective function.

    This class defines the Mishra 9 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Mishra09}}({x}) = \left[ ab^2c + abc^2 + b^2
       + (x_1 + x_2 - x_3)^2 \right]^2

    Where, in this exercise:

    .. math::

        \begin{cases} a = 2x_1^3 + 5x_1x_2 + 4x_3 - 2x_1^2x_3 - 18 \\
        b = x_1 + x_2^3 + x_1x_2^2 + x_1x_3^2 - 22 \\
        c = 8x_1^2 + 2x_2x_3 + 2x_2^2 + 3x_2^3 - 52 \end{cases}

    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2, 3`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 2, 3]`

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005

    TODO Line 1103
    """

    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间边界为每个维度都是 [-10, 10]
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        
        # 全局最优解，这个函数的最小值在 x = [1.0, 2.0, 3.0] 处
        self.global_optimum = [[1.0, 2.0, 3.0]]
        
        # 设置全局最小值为 0
        self.fglob = 0.0
    # 定义一个方法 `fun`，接受参数 `self`, `x` 和任意数量的额外参数 `args`
    def fun(self, x, *args):
        # 增加函数调用次数计数器 `nfev` 的值
        self.nfev += 1

        # 计算表达式 a 的值
        a = (2 * x[0] ** 3 + 5 * x[0] * x[1]
             + 4 * x[2] - 2 * x[0] ** 2 * x[2] - 18)
        
        # 计算表达式 b 的值
        b = x[0] + x[1] ** 3 + x[0] * x[1] ** 2 + x[0] * x[2] ** 2 - 22.0
        
        # 计算表达式 c 的值
        c = (8 * x[0] ** 2 + 2 * x[1] * x[2]
            + 2 * x[1] ** 2 + 3 * x[1] ** 3 - 52)

        # 返回计算出的复杂函数值
        return (a * c * b ** 2 + a * b * c ** 2 + b ** 2
                + (x[0] + x[1] - x[2]) ** 2) ** 2
class Mishra10(Benchmark):
    
    r"""
    Mishra 10 objective function.

    This class defines the Mishra 10 global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::
       f_{\text{Mishra10}}({x}) = \left[ \lfloor x_1 \perp x_2 \rfloor -
       \lfloor x_1 \rfloor - \lfloor x_2 \rfloor \right]^2

    with :math:`x_i \in [-10, 10]` for :math:`i =1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [2, 2]`

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005

    TODO line 1115
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        # 定义变量的取值范围
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        # 设置全局最优解
        self.global_optimum = [[2.0, 2.0]]
        # 设置全局最优解的函数值
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 取变量 x 的整数部分
        x1, x2 = int(x[0]), int(x[1])
        # 计算两个函数值
        f1 = x1 + x2
        f2 = x1 * x2
        # 返回函数值的平方
        return (f1 - f2) ** 2.0


class Mishra11(Benchmark):
    
    r"""
    Mishra 11 objective function.

    This class defines the Mishra 11 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::
       f_{\text{Mishra11}}(x) = \left [ \frac{1}{n} \sum_{i=1}^{n} \lvert x_i
       \rvert - \left(\prod_{i=1}^{n} \lvert x_i \rvert \right )^{\frac{1}{n}}
       \right]^2

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        # 定义变量的取值范围
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        # 自定义额外的变量范围
        self.custom_bounds = [(-3, 3), (-3, 3)]
        # 设置全局最优解
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        # 设置全局最优解的函数值
        self.fglob = 0.0
        # 设置是否改变维度的标志
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        N = self.N
        # 计算函数值并返回
        return ((1.0 / N) * sum(abs(x)) - (prod(abs(x))) ** 1.0 / N) ** 2.0
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """


    # 此部分是文档字符串，用于描述数学公式和参考文献引用

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设置维度
        Benchmark.__init__(self, dimensions)

        # 设置变量边界为 [-10.0, 10.0] 的列表
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        
        # 设置自定义边界为 [(-5, 5), (-5, 5)] 的列表
        self.custom_bounds = [(-5, 5), (-5, 5)]
        
        # 设置全局最优解为所有元素为 0.0 的列表
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        
        # 设置全局最优解的函数值为 0.0
        self.fglob = 0.0
        
        # 设置维度改变标志为 True
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 每次调用函数计数器 nfev 自增 1
        self.nfev += 1
        
        # 返回 x 各元素绝对值之和乘积的结果
        return sum(abs(x)) * prod(abs(x))
```