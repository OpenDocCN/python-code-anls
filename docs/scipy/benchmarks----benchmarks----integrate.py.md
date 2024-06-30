# `D:\src\scipysrc\scipy\benchmarks\benchmarks\integrate.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from .common import Benchmark, safe_import  # 从当前目录下的 common 模块导入 Benchmark 类和 safe_import 函数

from scipy.integrate import quad, cumulative_simpson  # 从 SciPy 的 integrate 模块导入 quad 和 cumulative_simpson 函数

with safe_import():  # 使用 safe_import 函数确保安全导入
    import ctypes  # 导入 ctypes 库，用于调用 C 语言的动态链接库
    import scipy.integrate._test_multivariate as clib_test  # 导入 scipy.integrate._test_multivariate 模块并命名为 clib_test
    from scipy._lib import _ccallback_c  # 从 scipy._lib 模块导入 _ccallback_c 函数

with safe_import() as exc:  # 使用 safe_import 函数并捕获异常
    from scipy import LowLevelCallable  # 从 SciPy 导入 LowLevelCallable 类
    from_cython = LowLevelCallable.from_cython  # 从 LowLevelCallable 类中导入 from_cython 方法
if exc.error:  # 如果发生异常
    def LowLevelCallable(func, data):  # 定义 LowLevelCallable 函数，接受 func 和 data 作为参数
        return (func, data)  # 返回元组 (func, data)

    def from_cython(*a):  # 定义 from_cython 函数，接受任意数量参数
        return a  # 返回参数 a 的元组

with safe_import() as exc:  # 使用 safe_import 函数并捕获异常
    import cffi  # 导入 cffi 库，用于与 C 语言的接口
if exc.error:  # 如果发生异常
    cffi = None  # 设置 cffi 变量为 None，禁用 cffi 功能

with safe_import():  # 使用 safe_import 函数确保安全导入
    from scipy.integrate import solve_bvp  # 从 SciPy 的 integrate 模块导入 solve_bvp 函数

class SolveBVP(Benchmark):  # 定义 SolveBVP 类，继承自 Benchmark 类
    TOL = 1e-5  # 设置公共容差 TOL 为 1e-5

    def fun_flow(self, x, y, p):  # 定义 fun_flow 方法，接受参数 self, x, y, p
        A = p[0]  # 获取参数 p 的第一个元素并赋值给 A
        return np.vstack((
            y[1], y[2], 100 * (y[1] ** 2 - y[0] * y[2] - A),  # 返回一个垂直堆叠的 NumPy 数组
            y[4], -100 * y[0] * y[4] - 1, y[6], -70 * y[0] * y[6]
        ))

    def bc_flow(self, ya, yb, p):  # 定义 bc_flow 方法，接受参数 self, ya, yb, p
        return np.array([  # 返回一个 NumPy 数组
            ya[0], ya[1], yb[0] - 1, yb[1], ya[3], yb[3], ya[5], yb[5] - 1])

    def time_flow(self):  # 定义 time_flow 方法，用于执行流动的时间
        x = np.linspace(0, 1, 10)  # 生成从 0 到 1 等间隔的 10 个数的数组
        y = np.ones((7, x.size))  # 创建元素全为 1 的形状为 (7, 10) 的 NumPy 数组
        solve_bvp(self.fun_flow, self.bc_flow, x, y, p=[1], tol=self.TOL)  # 调用 solve_bvp 函数求解边界值问题

    def fun_peak(self, x, y):  # 定义 fun_peak 方法，接受参数 self, x, y
        eps = 1e-3  # 设置小量 eps 为 0.001
        return np.vstack((
            y[1],
            -(4 * x * y[1] + 2 * y[0]) / (eps + x**2)
        ))

    def bc_peak(self, ya, yb):  # 定义 bc_peak 方法，接受参数 self, ya, yb
        eps = 1e-3  # 设置小量 eps 为 0.001
        v = (1 + eps) ** -1  # 计算 v 的值
        return np.array([ya[0] - v, yb[0] - v])  # 返回一个 NumPy 数组

    def time_peak(self):  # 定义 time_peak 方法，用于执行峰值的时间
        x = np.linspace(-1, 1, 5)  # 生成从 -1 到 1 等间隔的 5 个数的数组
        y = np.zeros((2, x.size))  # 创建元素全为 0 的形状为 (2, 5) 的 NumPy 数组
        solve_bvp(self.fun_peak, self.bc_peak, x, y, tol=self.TOL)  # 调用 solve_bvp 函数求解边界值问题

    def fun_gas(self, x, y):  # 定义 fun_gas 方法，接受参数 self, x, y
        alpha = 0.8  # 设置参数 alpha 为 0.8
        return np.vstack((
            y[1],
            -2 * x * y[1] * (1 - alpha * y[0]) ** -0.5
        ))

    def bc_gas(self, ya, yb):  # 定义 bc_gas 方法，接受参数 self, ya, yb
        return np.array([ya[0] - 1, yb[0]])  # 返回一个 NumPy 数组

    def time_gas(self):  # 定义 time_gas 方法，用于执行气体的时间
        x = np.linspace(0, 3, 5)  # 生成从 0 到 3 等间隔的 5 个数的数组
        y = np.empty((2, x.size))  # 创建形状为 (2, 5) 的空 NumPy 数组
        y[0] = 0.5  # 将 y 的第一行赋值为 0.5
        y[1] = -0.5  # 将 y 的第二行赋值为 -0.5
        solve_bvp(self.fun_gas, self.bc_gas, x, y, tol=self.TOL)  # 调用 solve_bvp 函数求解边界值问题
    # 设置函数，用于初始化测试环境
    def setup(self):
        # 导入数学库中的正弦函数
        from math import sin
    
        # 定义 Python 中的正弦函数
        self.f_python = lambda x: sin(x)
        
        # 从 Cython 中导入通过 C 编写的函数，并将其转换成 Python 可调用的对象
        self.f_cython = from_cython(_ccallback_c, "sine")
    
        # 尝试从 scipy 的测试模块中导入函数
        try:
            from scipy.integrate.tests.test_quadpack import get_clib_test_routine
            # 获取 ctypes 类型的 C 函数指针，用于调用 C 函数 '_multivariate_sin'
            self.f_ctypes = get_clib_test_routine('_multivariate_sin', ctypes.c_double,
                                                  ctypes.c_int, ctypes.c_double)
        except ImportError:
            # 如果导入失败，则使用 ctypes 加载动态链接库，并获取 '_multivariate_sin' 函数指针
            lib = ctypes.CDLL(clib_test.__file__)
            self.f_ctypes = lib._multivariate_sin
            self.f_ctypes.restype = ctypes.c_double
            self.f_ctypes.argtypes = (ctypes.c_int, ctypes.c_double)
    
        # 如果有 cffi 模块，则将 ctypes 函数指针转换为 cffi 的低级可调用对象
        if cffi is not None:
            voidp = ctypes.cast(self.f_ctypes, ctypes.c_void_p)
            address = voidp.value
            ffi = cffi.FFI()
            self.f_cffi = LowLevelCallable(ffi.cast("double (*)(int, double *)",
                                                    address))
    
    # 测试 Python 实现的函数在积分计算中的性能
    def time_quad_python(self):
        quad(self.f_python, 0, np.pi)
    
    # 测试 Cython 实现的函数在积分计算中的性能
    def time_quad_cython(self):
        quad(self.f_cython, 0, np.pi)
    
    # 测试 ctypes 所调用的 C 函数在积分计算中的性能
    def time_quad_ctypes(self):
        quad(self.f_ctypes, 0, np.pi)
    
    # 测试 cffi 所调用的 C 函数在积分计算中的性能
    def time_quad_cffi(self):
        quad(self.f_cffi, 0, np.pi)
# 定义 CumulativeSimpson 类，它是 Benchmark 的子类，用于执行 Simpson 积分的性能测试
class CumulativeSimpson(Benchmark):

    # 设置函数，在测试执行之前初始化数据
    def setup(self) -> None:
        # 生成 0 到 5 之间均匀分布的 1000 个点的数组 x，并返回相邻点之间的步长 dx
        x, self.dx = np.linspace(0, 5, 1000, retstep=True)
        # 计算 x 上的 sin 函数值，并存储在 self.y 中
        self.y = np.sin(2*np.pi*x)
        # 使用 np.tile 方法将 self.y 在两个维度上复制成 100x100 的数组，存储在 self.y2 中
        self.y2 = np.tile(self.y, (100, 100, 1))

    # 测试函数，用于执行一维数据的 Simpson 积分性能测试
    def time_1d(self) -> None:
        # 调用 cumulative_simpson 函数，对 self.y 进行 Simpson 积分，使用步长 self.dx
        cumulative_simpson(self.y, dx=self.dx)

    # 测试函数，用于执行多维数据的 Simpson 积分性能测试
    def time_multid(self) -> None:
        # 调用 cumulative_simpson 函数，对 self.y2 进行 Simpson 积分，使用步长 self.dx
        cumulative_simpson(self.y2, dx=self.dx)
```