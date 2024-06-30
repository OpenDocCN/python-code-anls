# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_P.py`

```
from numpy import (abs, sum, sin, cos, sqrt, log, prod, where, pi, exp, arange,
                   floor, log10, atleast_2d, zeros)
from .go_benchmark import Benchmark

class Parsopoulos(Benchmark):
    """
    Parsopoulos objective function.

    This class defines the Parsopoulos [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Parsopoulos}}(x) = \cos(x_1)^2 + \sin(x_2)^2

    with :math:`x_i \in [-5, 5]` for :math:`i = 1, 2`.

    *Global optimum*: This function has infinite number of global minima in R2,
    at points :math:`\left(k\frac{\pi}{2}, \lambda \pi \right)`,
    where :math:`k = \pm1, \pm3, ...` and :math:`\lambda = 0, \pm1, \pm2, ...`

    In the given domain problem, function has 12 global minima all equal to
    zero.

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        # 设置变量边界为 [-5.0, 5.0] 的列表
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))

        # 全局最优解为 [[pi/2.0, pi]]
        self.global_optimum = [[pi / 2.0, pi]]

        # 全局最优值为 0
        self.fglob = 0

    def fun(self, x, *args):
        # 增加函数评估计数
        self.nfev += 1

        # 返回 Parsopoulos 函数的值
        return cos(x[0]) ** 2.0 + sin(x[1]) ** 2.0


class Pathological(Benchmark):
    """
    Pathological objective function.

    This class defines the Pathological [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Pathological}}(x) = \sum_{i=1}^{n -1} \frac{\sin^{2}\left(
        \sqrt{100 x_{i+1}^{2} + x_{i}^{2}}\right) -0.5}{0.001 \left(x_{i}^{2}
        - 2x_{i}x_{i+1} + x_{i+1}^{2}\right)^{2} + 0.50}

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-100, 100]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0.` for :math:`x = [0, 0]` for
    :math:`i = 1, 2`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        # 设置变量边界为 [-100.0, 100.0] 的列表
        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))

        # 全局最优解为 [[0, 0]]
        self.global_optimum = [[0 for _ in range(self.N)]]

        # 全局最优值为 0.0
        self.fglob = 0.

    def fun(self, x, *args):
        # 增加函数评估计数
        self.nfev += 1

        # 计算 Pathological 函数的值并返回
        vec = (0.5 + (sin(sqrt(100 * x[: -1] ** 2 + x[1:] ** 2)) ** 2 - 0.5) /
               (1. + 0.001 * (x[: -1] ** 2 - 2 * x[: -1] * x[1:]
                              + x[1:] ** 2) ** 2))
        return sum(vec)
    """
        .. math::
    
            f_{\text{Paviani}}(x) = \sum_{i=1}^{10} \left[\log^{2}\left(10
            - x_i\right) + \log^{2}\left(x_i -2\right)\right]
            - \left(\prod_{i=1}^{10} x_i^{10} \right)^{0.2}
    
        定义了 Paviani 函数，用于全局优化问题的基准函数之一，描述了函数形式和参数范围。
    
        with :math:`x_i \in [2.001, 9.999]` for :math:`i = 1, ... , 10`.
    
        指定了变量 :math:`x_i` 的取值范围为 [2.001, 9.999]，其中 :math:`i` 从 1 到 10。
    
        *Global optimum*: :math:`f(x_i) = -45.7784684040686` for
        :math:`x_i = 9.350266` for :math:`i = 1, ..., 10`
    
        给出了全局最优解的数值和对应的变量取值。
    
        .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
        For Global Optimization Problems Int. Journal of Mathematical Modelling
        and Numerical Optimisation, 2013, 4, 150-194.
    
        引用了 Benchmark 函数的文献来源，说明了函数的背景和用途。
    
        TODO: think Gavana web/code definition is wrong because final product term
        shouldn't raise x to power 10.
    
        标记一个待办事项，指出 Gavana 网站或代码定义中的问题，表示最终乘积项不应该将 x 的每个分量都乘以 10。
        """
    
        class Paviani(Benchmark):
            """
            Paviani 函数类，继承自 Benchmark 类。
    
            初始化函数，设定函数维度。
    
            Args:
                dimensions (int): 函数的维度，默认为 10。
    
            Attributes:
                _bounds (list): 函数变量 x 的取值范围列表。
                global_optimum (list): 全局最优解的列表。
                fglob (float): 全局最优解的函数值。
            """
    
            def __init__(self, dimensions=10):
                # 调用父类 Benchmark 的初始化方法，设定维度
                Benchmark.__init__(self, dimensions)
    
                # 初始化函数变量 x 的取值范围
                self._bounds = list(zip([2.001] * self.N, [9.999] * self.N))
    
                # 设置全局最优解的变量取值
                self.global_optimum = [[9.350266 for _ in range(self.N)]]
    
                # 设置全局最优解的函数值
                self.fglob = -45.7784684040686
    
            def fun(self, x, *args):
                """
                Paviani 函数的计算方法，计算给定变量 x 的函数值。
    
                Args:
                    x (list): 输入的变量列表。
                    *args: 其他参数（可选）。
    
                Returns:
                    float: Paviani 函数在给定变量 x 下的函数值。
                """
                # 增加计数器，记录函数调用次数
                self.nfev += 1
    
                # 计算 Paviani 函数的表达式
                return sum(log(x - 2) ** 2.0 + log(10.0 - x) ** 2.0) - prod(x) ** 0.2
class Penalty01(Benchmark):
    r"""
    Penalty 1 objective function.

    This class defines the Penalty 1 [1]_ global optimization problem. This is a
    imultimodal minimization problem defined as follows:

    .. math::

        f_{\text{Penalty01}}(x) = \frac{\pi}{30} \left\{10 \sin^2(\pi y_1)
        + \sum_{i=1}^{n-1} (y_i - 1)^2 \left[1 + 10 \sin^2(\pi y_{i+1}) \right]
        + (y_n - 1)^2 \right \} + \sum_{i=1}^n u(x_i, 10, 100, 4)


    Where, in this exercise:

    .. math::

        y_i = 1 + \frac{1}{4}(x_i + 1)


    And:

    .. math::

        u(x_i, a, k, m) =
        \begin{cases}
        k(x_i - a)^m & \textrm{if} \hspace{5pt} x_i > a \\
        0 & \textrm{if} \hspace{5pt} -a \leq x_i \leq a \\
        k(-x_i - a)^m & \textrm{if} \hspace{5pt} x_i < -a 
        \end{cases}


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-50, 50]` for :math:`i= 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = -1` for
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设置维度属性
        Benchmark.__init__(self, dimensions)

        # 设置变量 _bounds 为每个维度的范围 [-50.0, 50.0]
        self._bounds = list(zip([-50.0] * self.N, [50.0] * self.N))
        
        # 设置自定义边界 custom_bounds 为 [-5.0, 5.0]，同样应用于每个维度
        self.custom_bounds = ([-5.0, 5.0], [-5.0, 5.0])
        
        # 设置全局最优解 global_optimum 为所有维度都是 [-1.0]
        self.global_optimum = [[-1.0 for _ in range(self.N)]]
        
        # 设置全局最优解值 fglob 为 0.0
        self.fglob = 0.0
        
        # 设置维度改变标志 change_dimensionality 为 True
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 每次调用 fun 方法，增加计数器 nfev
        self.nfev += 1

        # 初始化参数 a, b, c
        a, b, c = 10.0, 100.0, 4.0

        # 计算绝对值后的 x
        xx = abs(x)
        
        # 根据条件计算 u 函数的值
        u = where(xx > a, b * (xx - a) ** c, 0.0)

        # 计算 y
        y = 1.0 + (x + 1.0) / 4.0

        # 计算并返回目标函数值
        return (sum(u) + (pi / 30.0) * (10.0 * sin(pi * y[0]) ** 2.0
                + sum((y[: -1] - 1.0) ** 2.0
                      * (1.0 + 10.0 * sin(pi * y[1:]) ** 2.0))
                + (y[-1] - 1) ** 2.0))


class Penalty02(Benchmark):
    r"""
    Penalty 2 objective function.

    This class defines the Penalty 2 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Penalty02}}(x) = 0.1 \left\{\sin^2(3\pi x_1) + \sum_{i=1}^{n-1}
        (x_i - 1)^2 \left[1 + \sin^2(3\pi x_{i+1}) \right ]
        + (x_n - 1)^2 \left [1 + \sin^2(2 \pi x_n) \right ]\right \}
        + \sum_{i=1}^n u(x_i, 5, 100, 4)

    Where, in this exercise:

    .. math::

        u(x_i, a, k, m) = 
        \begin{cases}
        k(x_i - a)^m & \textrm{if} \hspace{5pt} x_i > a \\
        0 & \textrm{if} \hspace{5pt} -a \leq x_i \leq a \\
        k(-x_i - a)^m & \textrm{if} \hspace{5pt} x_i < -a \\
        \end{cases}


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-50, 50]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 1` for
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """
   `
    # 定义一个初始化方法，用于设置基准测试的维度，默认为二维
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设定维度
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间的边界，默认为 [-50.0, 50.0] 的范围
        self._bounds = list(zip([-50.0] * self.N, [50.0] * self.N))
        
        # 设置自定义的搜索空间边界，用于特定需求的优化
        self.custom_bounds = ([-4.0, 4.0], [-4.0, 4.0])

        # 设置全局最优解，默认为每个维度都为 1.0
        self.global_optimum = [[1.0 for _ in range(self.N)]]
        
        # 初始化全局最优值为 0.0
        self.fglob = 0.0
        
        # 设置维度变化标志，用于标识是否改变了优化问题的维度
        self.change_dimensionality = True

    # 定义目标函数 fun，用于评估给定的解 x
    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 初始化常数参数 a, b, c
        a, b, c = 5.0, 100.0, 4.0

        # 计算绝对值向量 xx
        xx = abs(x)
        
        # 根据 xx 计算非线性约束 u
        u = where(xx > a, b * (xx - a) ** c, 0.0)

        # 计算目标函数值，包括罚函数和约束项
        return (sum(u) + 0.1 * (10 * sin(3.0 * pi * x[0]) ** 2.0
                + sum((x[:-1] - 1.0) ** 2.0
                      * (1.0 + sin(3 * pi * x[1:]) ** 2.0))
                + (x[-1] - 1) ** 2.0 * (1 + sin(2 * pi * x[-1]) ** 2.0)))


(A) Need any further clarification on specific parts of the code?  
(B) Interested in exploring different optimization algorithms or techniques?  
(C) Curious about how to integrate this code into a larger project or workflow?
class PenHolder(Benchmark):
    r"""
    PenHolder objective function.

    This class defines the PenHolder [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{PenHolder}}(x) = -e^{\left|{e^{-\left|{- \frac{\sqrt{x_{1}^{2}
        + x_{2}^{2}}}{\pi} + 1}\right|} \cos\left(x_{1}\right)
        \cos\left(x_{2}\right)}\right|^{-1}}

    with :math:`x_i \in [-11, 11]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x_i) = -0.9635348327265058` for
    :math:`x_i = \pm 9.646167671043401` for :math:`i = 1, 2`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用父类的构造函数，初始化维度
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间的边界
        self._bounds = list(zip([-11.0] * self.N, [11.0] * self.N))

        # 设置全局最优解和其对应的函数值
        self.global_optimum = [[-9.646167708023526, 9.646167671043401]]
        self.fglob = -0.9635348327265058

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 计算公式中的中间变量
        a = abs(1. - (sqrt(x[0] ** 2 + x[1] ** 2) / pi))
        b = cos(x[0]) * cos(x[1]) * exp(a)

        # 返回目标函数值
        return -exp(-abs(b) ** -1)


``` 
class PermFunction01(Benchmark):
    r"""
    PermFunction 1 objective function.

    This class defines the PermFunction1 [1]_ global optimization problem. This is
    a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{PermFunction01}}(x) = \sum_{k=1}^n \left\{ \sum_{j=1}^n (j^k
        + \beta) \left[ \left(\frac{x_j}{j}\right)^k - 1 \right] \right\}^2

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-n, n + 1]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = i` for
    :math:`i = 1, ..., n`

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005

    TODO: line 560
    """

    def __init__(self, dimensions=2):
        # 调用父类的构造函数，初始化维度
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间的边界
        self._bounds = list(zip([-self.N] * self.N, [self.N + 1] * self.N))

        # 设置全局最优解和其对应的函数值
        self.global_optimum = [list(range(1, self.N + 1))]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 设置公式中的常数参数
        b = 0.5
        k = atleast_2d(arange(self.N) + 1).T
        j = atleast_2d(arange(self.N) + 1)

        # 计算目标函数的值
        s = (j ** k + b) * ((x / j) ** k - 1)
        return sum(sum(s, axis=1) ** 2)
    """
    TODO: line 582
    """

    # 初始化函数，设置默认维度为2
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化函数
        Benchmark.__init__(self, dimensions)

        # 设置边界范围，每个维度的取值范围为 [-N, N+1]
        self._bounds = list(zip([-self.N] * self.N,
                           [self.N + 1] * self.N))

        # 设置自定义边界范围
        self.custom_bounds = ([0, 1.5], [0, 1.0])

        # 设置全局最优解，对应每个维度的最优取值为 1/i，i 从 1 到 N
        self.global_optimum = [1. / arange(1, self.N + 1)]

        # 设置全局最优值
        self.fglob = 0.0

        # 标记是否改变维度
        self.change_dimensionality = True

    # 定义求解函数的方法 fun
    def fun(self, x, *args):
        # 记录函数评估次数
        self.nfev += 1

        # 设置常数 b
        b = 10

        # 生成维度序列 k，并转置为列向量
        k = atleast_2d(arange(self.N) + 1).T

        # 生成维度序列 j，并保持为行向量
        j = atleast_2d(arange(self.N) + 1)

        # 计算函数表达式中的子项 s
        s = (j + b) * (x ** k - (1. / j) ** k)

        # 计算并返回函数值
        return sum(sum(s, axis=1) ** 2)
class Pinter(Benchmark):
    """
    Pinter objective function.

    This class defines the Pinter [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Pinter}}(x) = \sum_{i=1}^n ix_i^2 + \sum_{i=1}^n 20i
       \sin^2 A + \sum_{i=1}^n i \log_{10} (1 + iB^2)

    Where, in this exercise:

    .. math::

        \begin{cases}
        A = x_{i-1} \sin x_i + \sin x_{i+1} \\
        B = x_{i-1}^2 - 2x_i + 3x_{i + 1} - \cos x_i + 1\\
        \end{cases}

    Where :math:`x_0 = x_n` and :math:`x_{n + 1} = x_1`.

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 初始化 Benchmark 类
        Benchmark.__init__(self, dimensions)

        # 定义变量边界
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))

        # 设置全局最优解和最优值
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估计数
        self.nfev += 1

        # 定义索引 i 为 1 到 n 的数组
        i = arange(self.N) + 1

        # 创建长度为 N+2 的数组 xx，并填充数据
        xx = zeros(self.N + 2)
        xx[1: - 1] = x
        xx[0] = x[-1]
        xx[-1] = x[0]

        # 计算 A 和 B 的值
        A = xx[0: -2] * sin(xx[1: - 1]) + sin(xx[2:])
        B = xx[0: -2] ** 2 - 2 * xx[1: - 1] + 3 * xx[2:] - cos(xx[1: - 1]) + 1

        # 计算目标函数值
        return (sum(i * x ** 2)
                + sum(20 * i * sin(A) ** 2)
                + sum(i * log10(1 + i * B ** 2)))


class Plateau(Benchmark):
    """
    Plateau objective function.

    This class defines the Plateau [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Plateau}}(x) = 30 + \sum_{i=1}^n \lfloor \lvert x_i
        \rvert\rfloor

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-5.12, 5.12]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 30` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=2):
        # 初始化 Benchmark 类
        Benchmark.__init__(self, dimensions)

        # 定义变量边界
        self._bounds = list(zip([-5.12] * self.N, [5.12] * self.N))

        # 设置全局最优解和最优值
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 30.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估计数
        self.nfev += 1

        # 计算目标函数值
        return 30.0 + sum(floor(abs(x)))
    # 定义一个基于 Powell 函数的 Benchmark 类
    def __init__(self, dimensions=4):
        # 调用父类 Benchmark 的初始化方法，设置维度数
        Benchmark.__init__(self, dimensions)

        # 设置变量边界为 [-4.0, 5.0] 的四维空间
        self._bounds = list(zip([-4.0] * self.N, [5.0] * self.N))
        
        # 设置全局最优解为 [0, 0, 0, 0]
        self.global_optimum = [[0, 0, 0, 0]]
        
        # 初始化全局最优解的函数值为 0
        self.fglob = 0

    # 定义 Powell 函数的计算方法
    def fun(self, x, *args):
        # 增加函数评估的计数器
        self.nfev += 1

        # 计算 Powell 函数的值
        return ((x[0] + 10 * x[1]) ** 2 + 5 * (x[2] - x[3]) ** 2
                + (x[1] - 2 * x[2]) ** 4 + 10 * (x[0] - x[3]) ** 4)
class PowerSum(Benchmark):
    """
    Power sum objective function.

    This class defines the Power Sum global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{PowerSum}}(x) = \sum_{k=1}^n\left[\left(\sum_{i=1}^n x_i^k
        \right) - b_k \right]^2

    Where, in this exercise, :math:`b = [8, 18, 44, 114]`

    Here, :math:`x_i \in [0, 4]` for :math:`i = 1, ..., 4`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 2, 2, 3]`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=4):
        # 调用父类 Benchmark 的构造函数
        Benchmark.__init__(self, dimensions)

        # 定义变量的取值范围
        self._bounds = list(zip([0.0] * self.N, [4.0] * self.N))

        # 设定全局最优解和对应的函数值
        self.global_optimum = [[1.0, 2.0, 2.0, 3.0]]
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 定义目标函数的常数项 b
        b = [8.0, 18.0, 44.0, 114.0]

        # 生成 k 矩阵，用于计算目标函数
        k = atleast_2d(arange(self.N) + 1).T

        # 计算目标函数值
        return sum((sum(x ** k, axis=1) - b) ** 2)


class Price01(Benchmark):
    """
    Price 1 objective function.

    This class defines the Price 1 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Price01}}(x) = (\lvert x_1 \rvert - 5)^2
        + (\lvert x_2 \rvert - 5)^2

    with :math:`x_i \in [-500, 500]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x_i) = 0.0` for :math:`x = [5, 5]` or
    :math:`x = [5, -5]` or :math:`x = [-5, 5]` or :math:`x = [-5, -5]`.

    .. [1] Price, W. A controlled random search procedure for global
    optimisation Computer Journal, 1977, 20, 367-370
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的构造函数
        Benchmark.__init__(self, dimensions)

        # 定义变量的取值范围
        self._bounds = list(zip([-500.0] * self.N, [500.0] * self.N))
        
        # 定义自定义的变量取值范围
        self.custom_bounds = ([-10.0, 10.0], [-10.0, 10.0])

        # 设定全局最优解和对应的函数值
        self.global_optimum = [[5.0, 5.0]]
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 计算 Price 1 函数的值
        return (abs(x[0]) - 5.0) ** 2.0 + (abs(x[1]) - 5.0) ** 2.0


class Price02(Benchmark):
    """
    Price 2 objective function.

    This class defines the Price 2 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Price02}}(x) = 1 + \sin^2(x_1) + \sin^2(x_2)
       - 0.1e^{(-x_1^2 - x_2^2)}

    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0.9` for :math:`x_i = [0, 0]`

    .. [1] Price, W. A controlled random search procedure for global
    optimisation Computer Journal, 1977, 20, 367-370
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的构造函数
        Benchmark.__init__(self, dimensions)

        # 定义变量的取值范围
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))

        # 设定全局最优解和对应的函数值
        self.global_optimum = [[0.0, 0.0]]
        self.fglob = 0.9
    # 定义一个类方法 `fun`，接受 `self` 和位置参数 `x` 以及可变位置参数 `args`
    def fun(self, x, *args):
        # 将对象属性 `nfev` 的值加一，用于计数函数调用次数
        self.nfev += 1

        # 返回一个计算结果，包括 1.0 和 `sin(x) ** 2` 的总和，减去 `0.1 * exp(-x[0] ** 2.0 - x[1] ** 2.0)`
        return 1.0 + sum(sin(x) ** 2) - 0.1 * exp(-x[0] ** 2.0 - x[1] ** 2.0)
class Price03(Benchmark):
    r"""
    Price 3 objective function.

    This class defines the Price 3 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Price03}}(x) = 100(x_2 - x_1^2)^2 + \left[6.4(x_2 - 0.5)^2
       - x_1 - 0.6 \right]^2

    with :math:`x_i \in [-50, 50]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [-5, -5]`,
    :math:`x = [-5, 5]`, :math:`x = [5, -5]`, :math:`x = [5, 5]`.

    .. [1] Price, W. A controlled random search procedure for global
    optimisation Computer Journal, 1977, 20, 367-370

    TODO Jamil #96 has an erroneous factor of 6 in front of the square brackets
    """

    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的初始化方法，设置维度
        Benchmark.__init__(self, dimensions)

        # 定义问题的搜索范围
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        # 自定义的特定范围，仅适用于部分维度
        self.custom_bounds = ([0, 2], [0, 2])

        # 全局最优解
        self.global_optimum = [[1.0, 1.0]]
        # 全局最优值
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 返回 Price 3 函数的值
        return (100 * (x[1] - x[0] ** 2) ** 2
                + (6.4 * (x[1] - 0.5) ** 2 - x[0] - 0.6) ** 2)


class Price04(Benchmark):
    r"""
    Price 4 objective function.

    This class defines the Price 4 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Price04}}(x) = (2 x_1^3 x_2 - x_2^3)^2
        + (6 x_1 - x_2^2 + x_2)^2

    with :math:`x_i \in [-50, 50]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0]`,
    :math:`x = [2, 4]` and :math:`x = [1.464, -2.506]`

    .. [1] Price, W. A controlled random search procedure for global
    optimisation Computer Journal, 1977, 20, 367-370
    """

    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的初始化方法，设置维度
        Benchmark.__init__(self, dimensions)

        # 定义问题的搜索范围
        self._bounds = list(zip([-50.0] * self.N, [50.0] * self.N))
        # 自定义的特定范围，仅适用于部分维度
        self.custom_bounds = ([0, 2], [0, 2])

        # 全局最优解
        self.global_optimum = [[2.0, 4.0]]
        # 全局最优值
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 返回 Price 4 函数的值
        return ((2.0 * x[1] * x[0] ** 3.0 - x[1] ** 3.0) ** 2.0
                + (6.0 * x[0] - x[1] ** 2.0 + x[1]) ** 2.0)
```