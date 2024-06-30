# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_univariate.py`

```
# 导入需要的数学函数和Benchmark类
from numpy import cos, exp, log, pi, sin, sqrt
from .go_benchmark import Benchmark

#-----------------------------------------------------------------------
#                 UNIVARIATE SINGLE-OBJECTIVE PROBLEMS
#-----------------------------------------------------------------------

class Problem02(Benchmark):
    """
    Univariate Problem02 objective function.

    This class defines the Univariate Problem02 global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem02}}(x) = \\sin(x) + \\sin \\left(\\frac{10}{3}x \\right)

    Bound constraints: :math:`x \\in [2.7, 7.5]`

    .. figure:: figures/Problem02.png
        :alt: Univariate Problem02 function
        :align: center

        **Univariate Problem02 function**

    *Global optimum*: :math:`f(x)=-1.899599` for :math:`x = 5.145735`
    """

    def __init__(self, dimensions=1):
        # 调用Benchmark类的构造函数初始化
        Benchmark.__init__(self, dimensions)

        # 定义变量的边界条件
        self._bounds = [(2.7, 7.5)]

        # 设置全局最优解和对应的函数值
        self.global_optimum = 5.145735
        self.fglob = -1.899599

    def fun(self, x, *args):
        # 增加函数评估次数
        self.nfev += 1

        # 获取变量值
        x = x[0]
        
        # 计算目标函数值
        return sin(x) + sin(10.0 / 3.0 * x)


class Problem03(Benchmark):
    """
    Univariate Problem03 objective function.

    This class defines the Univariate Problem03 global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem03}}(x) = - \\sum_{k=1}^6 k \\sin[(k+1)x+k]

    Bound constraints: :math:`x \\in [-10, 10]`

    .. figure:: figures/Problem03.png
        :alt: Univariate Problem03 function
        :align: center

        **Univariate Problem03 function**

    *Global optimum*: :math:`f(x)=-12.03124` for :math:`x = -6.7745761`
    """

    def __init__(self, dimensions=1):
        # 调用Benchmark类的构造函数初始化
        Benchmark.__init__(self, dimensions)

        # 定义变量的边界条件
        self._bounds = [(-10, 10)]

        # 设置全局最优解和对应的函数值
        self.global_optimum = -6.7745761
        self.fglob = -12.03124

    def fun(self, x, *args):
        # 增加函数评估次数
        self.nfev += 1

        # 获取变量值
        x = x[0]
        y = 0.0
        for k in range(1, 6):
            # 计算目标函数的累加项
            y += k * sin((k + 1) * x + k)

        # 返回目标函数值的负数
        return -y


class Problem04(Benchmark):
    """
    Univariate Problem04 objective function.

    This class defines the Univariate Problem04 global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem04}}(x) = - \\left(16x^2 - 24x + 5 \\right) e^{-x}

    Bound constraints: :math:`x \\in [1.9, 3.9]`

    .. figure:: figures/Problem04.png
        :alt: Univariate Problem04 function
        :align: center

        **Univariate Problem04 function**

    *Global optimum*: :math:`f(x)=-3.85045` for :math:`x = 2.868034`
    """

    def __init__(self, dimensions=1):
        # 调用Benchmark类的构造函数初始化
        Benchmark.__init__(self, dimensions)

        # 定义变量的边界条件
        self._bounds = [(1.9, 3.9)]

        # 设置全局最优解和对应的函数值
        self.global_optimum = 2.868034
        self.fglob = -3.85045
    # 定义一个方法 `fun`，接受 `self` 和参数 `x`、`*args`
    def fun(self, x, *args):
        # 增加 `nfev` 实例变量的值，表示函数调用次数
        self.nfev += 1

        # 获取 `x` 参数的第一个元素，因为 `x` 是一个元组或列表
        x = x[0]
        # 计算并返回函数 `- (16 * x ** 2 - 24 * x + 5) * exp(-x)` 的值
        return -(16 * x ** 2 - 24 * x + 5) * exp(-x)
class Problem05(Benchmark):
    """
    Univariate Problem05 objective function.

    This class defines the Univariate Problem05 global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem05}}(x) = - \\left(1.4 - 3x \\right) \\sin(18x)

    Bound constraints: :math:`x \\in [0, 1.2]`

    .. figure:: figures/Problem05.png
        :alt: Univariate Problem05 function
        :align: center

        **Univariate Problem05 function**

    *Global optimum*: :math:`f(x)=-1.48907` for :math:`x = 0.96609`

    """

    def __init__(self, dimensions=1):
        Benchmark.__init__(self, dimensions)

        # Define the bounds for the optimization variable 'x'
        self._bounds = [(0.0, 1.2)]

        # Define the global optimum value of 'x'
        self.global_optimum = 0.96609
        # Define the global minimum value of the objective function 'f(x)'
        self.fglob = -1.48907

    def fun(self, x, *args):
        # Increment the number of function evaluations
        self.nfev += 1

        # Extract the value of 'x' from the input array
        x = x[0]
        # Compute the objective function value using Problem05 definition
        return -(1.4 - 3 * x) * sin(18.0 * x)


class Problem06(Benchmark):
    """
    Univariate Problem06 objective function.

    This class defines the Univariate Problem06 global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem06}}(x) = - \\left[x + \\sin(x) \\right] e^{-x^2}

    Bound constraints: :math:`x \\in [-10, 10]`

    .. figure:: figures/Problem06.png
        :alt: Univariate Problem06 function
        :align: center

        **Univariate Problem06 function**

    *Global optimum*: :math:`f(x)=-0.824239` for :math:`x = 0.67956`

    """

    def __init__(self, dimensions=1):
        Benchmark.__init__(self, dimensions)

        # Define the bounds for the optimization variable 'x'
        self._bounds = [(-10.0, 10.0)]

        # Define the global optimum value of 'x'
        self.global_optimum = 0.67956
        # Define the global minimum value of the objective function 'f(x)'
        self.fglob = -0.824239

    def fun(self, x, *args):
        # Increment the number of function evaluations
        self.nfev += 1

        # Extract the value of 'x' from the input array
        x = x[0]
        # Compute the objective function value using Problem06 definition
        return -(x + sin(x)) * exp(-x ** 2.0)


class Problem07(Benchmark):
    """
    Univariate Problem07 objective function.

    This class defines the Univariate Problem07 global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem07}}(x) = \\sin(x) + \\sin \\left(\\frac{10}{3}x
                                  \\right) + \\log(x) - 0.84x + 3

    Bound constraints: :math:`x \\in [2.7, 7.5]`

    .. figure:: figures/Problem07.png
        :alt: Univariate Problem07 function
        :align: center

        **Univariate Problem07 function**

    *Global optimum*: :math:`f(x)=-1.6013` for :math:`x = 5.19978`

    """

    def __init__(self, dimensions=1):
        Benchmark.__init__(self, dimensions)

        # Define the bounds for the optimization variable 'x'
        self._bounds = [(2.7, 7.5)]

        # Define the global optimum value of 'x'
        self.global_optimum = 5.19978
        # Define the global minimum value of the objective function 'f(x)'
        self.fglob = -1.6013

    def fun(self, x, *args):
        # Increment the number of function evaluations
        self.nfev += 1

        # Extract the value of 'x' from the input array
        x = x[0]
        # Compute the objective function value using Problem07 definition
        return sin(x) + sin(10.0 / 3.0 * x) + log(x) - 0.84 * x + 3


class Problem08(Benchmark):
    """
    Univariate Problem08 objective function.

    This class defines the Univariate Problem08 global optimization problem. This
    """
    """
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem08}}(x) = - \\sum_{k=1}^6 k \\cos[(k+1)x+k]

    Bound constraints: :math:`x \\in [-10, 10]`

    .. figure:: figures/Problem08.png
        :alt: Univariate Problem08 function
        :align: center

        **Univariate Problem08 function**

    *Global optimum*: :math:`f(x)=-14.508` for :math:`x = -7.083506`

    """

    # 定义一个继承自Benchmark类的Problem08类，用于单变量的最小化问题
    def __init__(self, dimensions=1):
        # 调用父类Benchmark的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置问题的变量范围约束为 [-10, 10]
        self._bounds = [(-10, 10)]

        # 设置全局最优解的目标函数值和对应的变量值
        self.global_optimum = -7.083506
        self.fglob = -14.508

    # 定义问题的目标函数 fun，计算给定变量 x 下的函数值
    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 获取变量 x 的第一个元素（单变量问题）
        x = x[0]

        # 初始化函数值 y
        y = 0.0
        # 计算多项式函数的值
        for k in range(1, 6):
            y += k * cos((k + 1) * x + k)

        # 返回函数值的负值作为目标函数值（因为是最小化问题）
        return -y
class Problem09(Benchmark):
    """
    Univariate Problem09 objective function.

    This class defines the Univariate Problem09 global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem09}}(x) = \\sin(x) + \\sin \\left(\\frac{2}{3} x \\right)

    Bound constraints: :math:`x \\in [3.1, 20.4]`

    .. figure:: figures/Problem09.png
        :alt: Univariate Problem09 function
        :align: center

        **Univariate Problem09 function**

    *Global optimum*: :math:`f(x)=-1.90596` for :math:`x = 17.039`

    """

    def __init__(self, dimensions=1):
        # 调用父类的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置变量范围
        self._bounds = [(3.1, 20.4)]

        # 设置全局最优解和全局最优值
        self.global_optimum = 17.039
        self.fglob = -1.90596

    def fun(self, x, *args):
        # 增加函数评估计数
        self.nfev += 1

        # 获取输入变量x的值
        x = x[0]
        
        # 计算目标函数值
        return sin(x) + sin(2.0 / 3.0 * x)


class Problem10(Benchmark):
    """
    Univariate Problem10 objective function.

    This class defines the Univariate Problem10 global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem10}}(x) = -x\\sin(x)

    Bound constraints: :math:`x \\in [0, 10]`

    .. figure:: figures/Problem10.png
        :alt: Univariate Problem10 function
        :align: center

        **Univariate Problem10 function**

    *Global optimum*: :math:`f(x)=-7.916727` for :math:`x = 7.9787`

    """

    def __init__(self, dimensions=1):
        # 调用父类的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置变量范围
        self._bounds = [(0, 10)]

        # 设置全局最优解和全局最优值
        self.global_optimum = 7.9787
        self.fglob = -7.916727

    def fun(self, x, *args):
        # 增加函数评估计数
        self.nfev += 1

        # 获取输入变量x的值
        x = x[0]
        
        # 计算目标函数值
        return -x * sin(x)


class Problem11(Benchmark):
    """
    Univariate Problem11 objective function.

    This class defines the Univariate Problem11 global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem11}}(x) = 2\\cos(x) + \\cos(2x)

    Bound constraints: :math:`x \\in [-\\pi/2, 2\\pi]`

    .. figure:: figures/Problem11.png
        :alt: Univariate Problem11 function
        :align: center

        **Univariate Problem11 function**

    *Global optimum*: :math:`f(x)=-1.5` for :math:`x = 2.09439`

    """

    def __init__(self, dimensions=1):
        # 调用父类的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置变量范围
        self._bounds = [(-pi / 2, 2 * pi)]

        # 设置全局最优解和全局最优值
        self.global_optimum = 2.09439
        self.fglob = -1.5

    def fun(self, x, *args):
        # 增加函数评估计数
        self.nfev += 1

        # 获取输入变量x的值
        x = x[0]
        
        # 计算目标函数值
        return 2 * cos(x) + cos(2 * x)


class Problem12(Benchmark):
    """
    Univariate Problem12 objective function.

    This class defines the Univariate Problem12 global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem12}}(x) = \\sin^3(x) + \\cos^3(x)

    Bound constraints: :math:`x \\in [0, 2\\pi]`
    """

    def __init__(self, dimensions=1):
        # 调用父类的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置变量范围
        self._bounds = [(0, 2 * pi)]
    """
    .. figure:: figures/Problem12.png
        :alt: Univariate Problem12 function
        :align: center

        **Univariate Problem12 function**

    *Global optimum*: :math:`f(x)=-1` for :math:`x = \\pi`

    """

    # 定义一个 Benchmark12 类，继承自 Benchmark 类
    def __init__(self, dimensions=1):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置变量的取值范围，这里是一个包含元组的列表，元组指定了边界 (0, 2 * pi)
        self._bounds = [(0, 2 * pi)]

        # 设置全局最优解，这里是 π
        self.global_optimum = pi
        # 设置全局最优解的函数值，这里是 -1
        self.fglob = -1

    # 定义函数 fun，接受参数 x 和任意额外参数 *args
    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 将 x 转换为标量，假设 x 是一个包含一个元素的列表或元组
        x = x[0]
        # 返回函数 sin(x)^3 + cos(x)^3 的值
        return (sin(x)) ** 3.0 + (cos(x)) ** 3.0
class Problem13(Benchmark):

    """
    Univariate Problem13 objective function.

    This class defines the Univariate Problem13 global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem13}}(x) = -x^{2/3} - (1 - x^2)^{1/3}

    Bound constraints: :math:`x \\in [0.001, 0.99]`

    .. figure:: figures/Problem13.png
        :alt: Univariate Problem13 function
        :align: center

        **Univariate Problem13 function**

    *Global optimum*: :math:`f(x)=-1.5874` for :math:`x = 1/\\sqrt(2)`

    """

    def __init__(self, dimensions=1):
        # 调用父类构造函数初始化维度
        Benchmark.__init__(self, dimensions)

        # 定义问题的变量边界
        self._bounds = [(0.001, 0.99)]

        # 设置全局最优解和全局最优函数值
        self.global_optimum = 1.0 / sqrt(2)
        self.fglob = -1.5874

    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 提取变量 x 的值
        x = x[0]
        # 返回目标函数值
        return -x ** (2.0 / 3.0) - (1.0 - x ** 2) ** (1.0 / 3.0)


class Problem14(Benchmark):

    """
    Univariate Problem14 objective function.

    This class defines the Univariate Problem14 global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem14}}(x) = -e^{-x} \\sin(2\\pi x)

    Bound constraints: :math:`x \\in [0, 4]`

    .. figure:: figures/Problem14.png
        :alt: Univariate Problem14 function
        :align: center

        **Univariate Problem14 function**

    *Global optimum*: :math:`f(x)=-0.788685` for :math:`x = 0.224885`

    """

    def __init__(self, dimensions=1):
        # 调用父类构造函数初始化维度
        Benchmark.__init__(self, dimensions)

        # 定义问题的变量边界
        self._bounds = [(0.0, 4.0)]

        # 设置全局最优解和全局最优函数值
        self.global_optimum = 0.224885
        self.fglob = -0.788685

    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 提取变量 x 的值
        x = x[0]
        # 返回目标函数值
        return -exp(-x) * sin(2.0 * pi * x)


class Problem15(Benchmark):

    """
    Univariate Problem15 objective function.

    This class defines the Univariate Problem15 global optimization problem.
    This is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem15}}(x) = \\frac{x^{2} - 5 x + 6}{x^{2} + 1}

    Bound constraints: :math:`x \\in [-5, 5]`

    .. figure:: figures/Problem15.png
        :alt: Univariate Problem15 function
        :align: center

        **Univariate Problem15 function**

    *Global optimum*: :math:`f(x)=-0.03553` for :math:`x = 2.41422`

    """

    def __init__(self, dimensions=1):
        # 调用父类构造函数初始化维度
        Benchmark.__init__(self, dimensions)

        # 定义问题的变量边界
        self._bounds = [(-5.0, 5.0)]

        # 设置全局最优解和全局最优函数值
        self.global_optimum = 2.41422
        self.fglob = -0.03553

    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 提取变量 x 的值
        x = x[0]
        # 返回目标函数值
        return -(-x ** 2.0 + 5 * x - 6) / (x ** 2 + 1)


class Problem18(Benchmark):

    """
    Univariate Problem18 objective function.

    This class defines the Univariate Problem18 global optimization problem.
    This is a multimodal minimization problem defined as follows:
    """
    .. math::

         f_{\\text{Problem18}}(x)
         = \\begin{cases}(x-2)^2 & \\textrm{if} \\hspace{5pt} x
           \\leq 3 \\\\ 2\\log(x-2)+1&\\textrm{otherwise}\\end{cases}

    Bound constraints: :math:`x \\in [0, 6]`

    .. figure:: figures/Problem18.png
        :alt: Univariate Problem18 function
        :align: center

        **Univariate Problem18 function**

    *Global optimum*: :math:`f(x)=0` for :math:`x = 2`

    """

    # 定义一个继承自Benchmark的类，表示问题18的单变量函数
    def __init__(self, dimensions=1):
        Benchmark.__init__(self, dimensions)

        # 设置变量的取值范围边界
        self._bounds = [(0.0, 6.0)]

        # 设置全局最优解的位置和函数值
        self.global_optimum = 2
        self.fglob = 0

    # 定义问题18的单变量函数，参数x是一个长度为1的列表
    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 获取x的实际值
        x = x[0]

        # 根据不同的x取值返回对应的函数值
        if x <= 3:
            return (x - 2.0) ** 2.0
        else:
            return 2 * log(x - 2.0) + 1
class Problem20(Benchmark):
    """
    Univariate Problem20 objective function.

    This class defines the Univariate Problem20 global optimization problem.
    This is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem20}}(x) = -[x-\\sin(x)]e^{-x^2}

    Bound constraints: :math:`x \\in [-10, 10]`

    .. figure:: figures/Problem20.png
        :alt: Univariate Problem20 function
        :align: center

        **Univariate Problem20 function**

    *Global optimum*: :math:`f(x)=-0.0634905` for :math:`x = 1.195137`

    """

    def __init__(self, dimensions=1):
        # 调用 Benchmark 类的构造函数初始化维度
        Benchmark.__init__(self, dimensions)

        # 设置变量的边界约束
        self._bounds = [(-10, 10)]

        # 设置全局最优解和全局最优值
        self.global_optimum = 1.195137
        self.fglob = -0.0634905

    def fun(self, x, *args):
        # 增加函数评估次数的计数器
        self.nfev += 1

        # 提取 x 的值
        x = x[0]

        # 计算并返回函数值
        return -(x - sin(x)) * exp(-x ** 2.0)


class Problem21(Benchmark):
    """
    Univariate Problem21 objective function.

    This class defines the Univariate Problem21 global optimization problem.
    This is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem21}}(x) = x \\sin(x) + x \\cos(2x)

    Bound constraints: :math:`x \\in [0, 10]`

    .. figure:: figures/Problem21.png
        :alt: Univariate Problem21 function
        :align: center

        **Univariate Problem21 function**

    *Global optimum*: :math:`f(x)=-9.50835` for :math:`x = 4.79507`

    """

    def __init__(self, dimensions=1):
        # 调用 Benchmark 类的构造函数初始化维度
        Benchmark.__init__(self, dimensions)

        # 设置变量的边界约束
        self._bounds = [(0, 10)]

        # 设置全局最优解和全局最优值
        self.global_optimum = 4.79507
        self.fglob = -9.50835

    def fun(self, x, *args):
        # 增加函数评估次数的计数器
        self.nfev += 1

        # 提取 x 的值
        x = x[0]

        # 计算并返回函数值
        return x * sin(x) + x * cos(2.0 * x)


class Problem22(Benchmark):
    """
    Univariate Problem22 objective function.

    This class defines the Univariate Problem22 global optimization problem.
    This is a multimodal minimization problem defined as follows:

    .. math::

       f_{\\text{Problem22}}(x) = e^{-3x} - \\sin^3(x)

    Bound constraints: :math:`x \\in [0, 20]`

    .. figure:: figures/Problem22.png
        :alt: Univariate Problem22 function
        :align: center

        **Univariate Problem22 function**

    *Global optimum*: :math:`f(x)=e^{-27\\pi/2} - 1` for :math:`x = 9\\pi/2`

    """

    def __init__(self, dimensions=1):
        # 调用 Benchmark 类的构造函数初始化维度
        Benchmark.__init__(self, dimensions)

        # 设置变量的边界约束
        self._bounds = [(0, 20)]

        # 设置全局最优解和全局最优值
        self.global_optimum = 9.0 * pi / 2.0
        self.fglob = exp(-27.0 * pi / 2.0) - 1.0

    def fun(self, x, *args):
        # 增加函数评估次数的计数器
        self.nfev += 1

        # 提取 x 的值
        x = x[0]

        # 计算并返回函数值
        return exp(-3.0 * x) - (sin(x)) ** 3.0
```