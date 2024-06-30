# `D:\src\scipysrc\scipy\benchmarks\benchmarks\optimize_zeros.py`

```
# 导入数学函数 sqrt, exp, cos, sin
from math import sqrt, exp, cos, sin
# 导入 numpy 库并使用别名 np
import numpy as np

# 从 .common 模块中导入 Benchmark 类和 safe_import 函数
from .common import Benchmark, safe_import

# 安全导入测试参数
with safe_import():
    # 从 scipy.optimize._tstutils 模块中导入 methods, mstrings, functions, fstrings
    from scipy.optimize._tstutils import methods, mstrings, functions, fstrings
# 从 scipy.optimize 模块中导入 newton 函数，newton 函数早于 benchmarks
from scipy.optimize import newton


# 定义 Zeros 类，继承自 Benchmark 类
class Zeros(Benchmark):
    # 参数化测试函数和求解器
    params = [
        fstrings,
        mstrings
    ]
    param_names = ['test function', 'solver']

    # 设置函数，在测试前初始化参数
    def setup(self, func, meth):
        self.a = .5
        self.b = sqrt(3)

        # 根据 func 和 meth 初始化 self.func 和 self.meth
        self.func = functions[fstrings.index(func)]
        self.meth = methods[mstrings.index(meth)]

    # 测试执行时间的方法
    def time_zeros(self, func, meth):
        # 调用 self.meth 方法，传入 self.func, self.a, self.b 作为参数
        self.meth(self.func, self.a, self.b)


# 定义 Newton 类，继承自 Benchmark 类
class Newton(Benchmark):
    # 参数化测试函数和求解器
    params = [
        ['f1', 'f2'],
        ['newton', 'secant', 'halley']
    ]
    param_names = ['test function', 'solver']

    # 设置函数，在测试前初始化参数
    def setup(self, func, meth):
        self.x0 = 3
        self.f_1 = None
        self.f_2 = None
        if func == 'f1':
            # 如果 func 是 'f1'，定义 self.f 和可能的 self.f_1, self.f_2 函数
            self.f = lambda x: x ** 2 - 2 * x - 1
            if meth in ('newton', 'halley'):
                self.f_1 = lambda x: 2 * x - 2
            if meth == 'halley':
                self.f_2 = lambda x: 2.0 + 0 * x
        else:
            # 如果 func 不是 'f1'，定义 self.f 和可能的 self.f_1, self.f_2 函数
            self.f = lambda x: exp(x) - cos(x)
            if meth in ('newton', 'halley'):
                self.f_1 = lambda x: exp(x) + sin(x)
            if meth == 'halley':
                self.f_2 = lambda x: exp(x) + cos(x)

    # 测试执行时间的方法
    def time_newton(self, func, meth):
        # 调用 newton 函数，传入 self.f, self.x0, 空的 args 元组, self.f_1, self.f_2 作为参数
        newton(self.f, self.x0, args=(), fprime=self.f_1, fprime2=self.f_2)


# 定义 NewtonArray 类，继承自 Benchmark 类
class NewtonArray(Benchmark):
    # 参数化向量化方式和求解器
    params = [['loop', 'array'], ['newton', 'secant', 'halley']]
    param_names = ['vectorization', 'solver']

    # 设置函数，在测试前初始化参数
    def setup(self, vec, meth):
        if vec == 'loop':
            if meth == 'newton':
                # 如果 vec 是 'loop' 并且 meth 是 'newton'，定义 self.fvec 函数
                self.fvec = lambda f, x0, args, fprime, fprime2: [
                    newton(f, x, args=(a0, a1) + args[2:], fprime=fprime)
                    for (x, a0, a1) in zip(x0, args[0], args[1])
                ]
            elif meth == 'halley':
                # 如果 vec 是 'loop' 并且 meth 是 'halley'，定义 self.fvec 函数
                self.fvec = lambda f, x0, args, fprime, fprime2: [
                    newton(
                        f, x, args=(a0, a1) + args[2:], fprime=fprime,
                        fprime2=fprime2
                    ) for (x, a0, a1) in zip(x0, args[0], args[1])
                ]
            else:
                # 如果 vec 是 'loop' 并且 meth 是其他求解器，定义 self.fvec 函数
                self.fvec = lambda f, x0, args, fprime, fprime2: [
                    newton(f, x, args=(a0, a1) + args[2:]) for (x, a0, a1)
                    in zip(x0, args[0], args[1])
                ]
        else:
            if meth == 'newton':
                # 如果 vec 是 'array' 并且 meth 是 'newton'，定义 self.fvec 函数
                self.fvec = lambda f, x0, args, fprime, fprime2: newton(
                    f, x0, args=args, fprime=fprime
                )
            elif meth == 'halley':
                # 如果 vec 是 'array' 并且 meth 是 'halley'，定义 self.fvec 函数
                self.fvec = newton
            else:
                # 如果 vec 是 'array' 并且 meth 是其他求解器，定义 self.fvec 函数
                self.fvec = lambda f, x0, args, fprime, fprime2: newton(
                    f, x0, args=args
                )
    # 定义一个方法 `time_array_newton`，接受 `self` (类实例)、`vec` (向量) 和 `meth` (方法) 作为参数
    def time_array_newton(self, vec, meth):
        
        # 定义函数 `f`，接受 `x` 和参数元组 `a` 作为输入，计算并返回函数值
        def f(x, *a):
            # 根据参数 `a` 计算中间变量 `b`
            b = a[0] + x * a[3]
            # 返回函数值，用于牛顿法迭代
            return a[1] - a[2] * (np.exp(b / a[5]) - 1.0) - b / a[4] - x

        # 定义函数 `f_1`，接受 `x` 和参数元组 `a` 作为输入，计算并返回一阶导数值
        def f_1(x, *a):
            # 根据参数 `a` 计算中间变量 `b`
            b = a[3] / a[5]
            # 返回函数 `f` 的一阶导数值，用于牛顿法迭代
            return -a[2] * np.exp(a[0] / a[5] + x * b) * b - a[3] / a[4] - 1

        # 定义函数 `f_2`，接受 `x` 和参数元组 `a` 作为输入，计算并返回二阶导数值
        def f_2(x, *a):
            # 根据参数 `a` 计算中间变量 `b`
            b = a[3] / a[5]
            # 返回函数 `f` 的二阶导数值，用于牛顿法迭代
            return -a[2] * np.exp(a[0] / a[5] + x * b) * b ** 2

        # 定义初始参数数组 `a0`，包含10个预设的浮点数值
        a0 = np.array([
            5.32725221, 5.48673747, 5.49539973,
            5.36387202, 4.80237316, 1.43764452,
            5.23063958, 5.46094772, 5.50512718,
            5.42046290
        ])
        
        # 计算数组 `a1`，其中每个元素是对应位置的正弦值加1乘以7
        a1 = (np.sin(range(10)) + 1.0) * 7.0
        
        # 定义参数元组 `args`，包含用于函数 `f` 的所有必要参数
        args = (a0, a1, 1e-09, 0.004, 10, 0.27456)
        
        # 定义初始迭代点 `x0`，含有10个值为7.0的列表
        x0 = [7.0] * 10
        
        # 调用类实例的 `fvec` 方法，传递函数 `f`、初始迭代点 `x0` 和参数元组 `args`，
        # 同时提供一阶导数函数 `fprime` 和二阶导数函数 `fprime2` 作为可选参数
        self.fvec(f, x0, args=args, fprime=f_1, fprime2=f_2)
```