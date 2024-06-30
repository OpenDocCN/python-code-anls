# `D:\src\scipysrc\scipy\scipy\optimize\cython_optimize\__init__.py`

```
"""
Cython optimize root finding API
================================
The underlying C functions for the following root finders can be accessed
directly using Cython:

- `~scipy.optimize.bisect`
- `~scipy.optimize.ridder`
- `~scipy.optimize.brenth`
- `~scipy.optimize.brentq`

The Cython API for the root finding functions is similar except there is no
``disp`` argument. Import the root finding functions using ``cimport`` from
`scipy.optimize.cython_optimize`. ::
"""

# 导入 Cython 中的根查找函数
from scipy.optimize.cython_optimize cimport bisect, ridder, brentq, brenth

"""
Callback signature
------------------
The zeros functions in `~scipy.optimize.cython_optimize` expect a callback that
takes a double for the scalar independent variable as the 1st argument and a
user defined ``struct`` with any extra parameters as the 2nd argument. ::
"""

# 定义回调函数签名
# 回调函数需要接受一个双精度浮点数作为独立变量，以及一个用户定义的结构体作为额外参数
callback_type = """
    double (*callback_type)(double, void*) noexcept
"""

"""
Examples
--------
Usage of `~scipy.optimize.cython_optimize` requires Cython to write callbacks
that are compiled into C. For more information on compiling Cython, see the
`Cython Documentation <http://docs.cython.org/en/latest/index.html>`_.

These are the basic steps:

1. Create a Cython ``.pyx`` file, for example: ``myexample.pyx``.
2. Import the desired root finder from `~scipy.optimize.cython_optimize`.
3. Write the callback function, and call the selected root finding function
   passing the callback, any extra arguments, and the other solver
   parameters. ::
"""

# 示例
# 使用 `scipy.optimize.cython_optimize` 需要在 Cython 中编写回调函数，并编译成 C 语言
# 下面是基本步骤：

# 1. 创建一个 Cython 的 `.pyx` 文件，例如：`myexample.pyx`
# 2. 从 `scipy.optimize.cython_optimize` 中导入所需的根查找器
# 3. 编写回调函数，并调用选定的根查找函数，传递回调函数、任何额外参数以及其他求解器参数。

"""
       from scipy.optimize.cython_optimize cimport brentq

       # import math from Cython
       from libc cimport math

       myargs = {'C0': 1.0, 'C1': 0.7}  # a dictionary of extra arguments
       XLO, XHI = 0.5, 1.0  # lower and upper search boundaries
       XTOL, RTOL, MITR = 1e-3, 1e-3, 10  # other solver parameters

       # user-defined struct for extra parameters
       ctypedef struct test_params:
           double C0
           double C1


       # user-defined callback
       cdef double f(double x, void *args) noexcept:
           cdef test_params *myargs = <test_params *> args
           return myargs.C0 - math.exp(-(x - myargs.C1))


       # Cython wrapper function
       cdef double brentq_wrapper_example(dict args, double xa, double xb,
                                          double xtol, double rtol, int mitr):
           # Cython automatically casts dictionary to struct
           cdef test_params myargs = args
           return brentq(
               f, xa, xb, <test_params *> &myargs, xtol, rtol, mitr, NULL)


       # Python function
       def brentq_example(args=myargs, xa=XLO, xb=XHI, xtol=XTOL, rtol=RTOL,
                          mitr=MITR):
           '''Calls Cython wrapper from Python.'''
           return brentq_wrapper_example(args, xa, xb, xtol, rtol, mitr)
"""

# 以上是一个示例，展示了如何使用 Cython 编写回调函数，并在 Python 中调用包装的 Cython 函数来使用 `brentq`。
    # 导入需要的 Cython 类型和函数
    from scipy.optimize.cython_optimize cimport zeros_full_output

    # 定义一个 Cython 函数，用于包装 brentq 函数并返回完整输出
    cdef zeros_full_output brentq_full_output_wrapper_example(
            dict args, double xa, double xb, double xtol, double rtol,
            int mitr):
        # 将传入的参数 args 赋值给 Cython 的 test_params 类型对象 myargs
        cdef test_params myargs = args
        # 声明一个 zeros_full_output 类型的对象 my_full_output
        cdef zeros_full_output my_full_output
        # 调用 brentq 函数，将求解结果存储在 my_full_output 中
        brentq(f, xa, xb, &myargs, xtol, rtol, mitr, &my_full_output)
        # 返回完整输出对象 my_full_output
        return my_full_output

    # Python 函数，调用 Cython 函数并返回完整输出
    def brent_full_output_example(args=myargs, xa=XLO, xb=XHI, xtol=XTOL,
                                  rtol=RTOL, mitr=MITR):
        '''Returns full output'''
        return brentq_full_output_wrapper_example(args, xa, xb, xtol, rtol,
                                                  mitr)

    # 调用 Python 函数获取完整输出，并打印结果
    result = brent_full_output_example()
    # {'error_num': 0,
    #  'funcalls': 6,
    #  'iterations': 5,
    #  'root': 0.6999942848231314}
```