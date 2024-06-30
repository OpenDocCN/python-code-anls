# `D:\src\scipysrc\scipy\scipy\optimize\tnc\_moduleTNC.pyx`

```
# cython: language_level=3, boundscheck=False
# 导入必要的库和模块
from libc.string cimport memcpy  # 导入 C 标准库中的 memcpy 函数
import numpy as np  # 导入 NumPy 库
cimport numpy as np  # 在 Cython 中导入 NumPy 库，以便使用其功能

np.import_array()  # 导入 NumPy 的数组功能

ctypedef np.float64_t float64_t  # 定义 Cython 中的 float64_t 类型别名

# 从外部头文件 "tnc.h" 中导入函数和类型定义
cdef extern from "tnc.h":
    ctypedef void tnc_callback(double[], void*) except *  # 定义 tnc 回调函数类型
    ctypedef int tnc_function(double[], double*, double[], void*) except 1  # 定义 tnc 主函数类型

    # 定义 tnc 返回码枚举
    cdef enum tnc_rc:
        TNC_MINRC
        TNC_ENOMEM
        TNC_EINVAL
        TNC_INFEASIBLE
        TNC_LOCALMINIMUM
        TNC_FCONVERGED
        TNC_XCONVERGED
        TNC_MAXFUN
        TNC_LSFAIL
        TNC_CONSTANT
        TNC_NOPROGRESS
        TNC_USERABORT

    # 定义 tnc 主函数及其参数列表
    cdef int tnc(int n, double x[], double *f, double g[],
            tnc_function *function, void *state,
            double low[], double up[], double scale[], double offset[],
            int messages, int maxCGit, int maxnfeval, double eta, double stepmx,
            double accuracy, double fmin, double ftol, double xtol, double pgtol,
            double rescale, int *nfeval, int *niter, tnc_callback *tnc_callback) except *

# 定义 Cython 结构体 s_pytnc_state，表示 PyTNC 状态
cdef struct s_pytnc_state:
  void *py_function
  void *py_callback
  int n
  int failed
ctypedef s_pytnc_state pytnc_state  # 定义 Cython 类型 pytnc_state

# 定义 Cython 函数 function，实现用户定义的目标函数及其梯度
cdef int function(double x[], double *f, double g[], void *state) except 1:
    # 如果没有错误返回 0，如果有错误返回 1
    # 但是，如果用户函数引发异常，Cython 应当处理它。

    cdef:
        pytnc_state *py_state
        int n
        double *x_data
        double *g_data

    py_state = <pytnc_state *>state
    n = py_state.n

    if py_state.failed:
        # 可能是回调函数代码出现异常？
        # 这会导致 tnc 代码返回 LS_USERABORT
        return 1

    # 确保我们在数据的副本上操作，以防用户函数修改它
    xcopy = np.empty(n, dtype=np.float64)
    x_data = <float64_t *>np.PyArray_DATA(xcopy)
    memcpy(x_data, x, sizeof(double) * n)

    # 调用用户定义的 Python 函数，获取目标函数值和梯度
    fx, gx = (<object>py_state.py_function)(xcopy)

    if not np.isscalar(fx):
        try:
            fx = np.asarray(fx).item()
        except (TypeError, ValueError) as e:
            raise ValueError(
                "The user-provided objective function "
                "must return a scalar value."
            ) from e

    f[0] = <double> fx  # 将目标函数值转换为 double 类型并赋给 f[0]

    gx = np.asarray(gx, dtype=np.float64)
    if gx.size != n:
        raise ValueError("tnc: gradient must have shape (len(x0),)")

    if not gx.flags['C_CONTIGUOUS']:
        gx = np.ascontiguousarray(gx, dtype=np.float64)

    g_data = <float64_t *>np.PyArray_DATA(gx)
    memcpy(g, g_data, n * sizeof(double))  # 将梯度数据复制到 g 数组中

    return 0  # 返回表示成功执行的状态码

# 定义 Cython 回调函数 callback_function，处理用户定义的回调函数
cdef void callback_function(double x[], void *state) except *:
    cdef:
        pytnc_state *py_state
        int n
        double *x_data

    py_state = <pytnc_state *>state
    n = py_state.n

    # 确保我们在数据的副本上操作，以防用户函数修改它
    # 尝试执行以下代码块，捕获任何异常
    try:
        # 创建一个长度为 n 的空 NumPy 数组，数据类型为 float64
        xcopy = np.empty(n, dtype=np.float64)
        # 获取 xcopy 数组的数据指针，并将其转换为 float64_t* 类型
        x_data = <float64_t *>np.PyArray_DATA(xcopy)
        # 使用 memcpy 函数将数组 x 中的数据复制到 x_data 指向的内存位置，复制长度为 sizeof(double) * n

        # TODO 检查回调函数的返回值，决定是否需要终止程序？
        (<object>py_state.py_callback)(xcopy)
    # 捕获所有类型的异常，并将其赋值给变量 exc
    except BaseException as exc:
        # 将 py_state.failed 设为 1，表示执行失败
        py_state.failed = 1
        # 抛出当前捕获到的异常 exc 给上一层调用栈处理
        raise exc
def tnc_minimize(func_and_grad,
                   np.ndarray[np.float64_t] x0,
                   np.ndarray[np.float64_t] low,
                   np.ndarray[np.float64_t] up,
                   np.ndarray[np.float64_t] scale,
                   np.ndarray[np.float64_t] offset,
                   int messages,
                   int maxCGit,
                   int maxfun,
                   double eta,
                   double stepmx,
                   double accuracy,
                   double fmin,
                   double ftol,
                   double xtol,
                   double pgtol,
                   double rescale,
                   callback=None):

    cdef:
        # 定义 Cython 结构体变量
        pytnc_state py_state
        # 变量声明
        int n
        int rc
        double f = np.inf
        int nfeval = 0
        int niter = 0
        double *x_data
        double *g_data
        double *low_data = NULL
        double *up_data = NULL
        double *scale_data = NULL
        double *offset_data = NULL
        tnc_callback *callback_functionP = NULL

    # 初始化结构体变量，使用 memset 进行初始化
    py_state.py_callback = NULL
    py_state.failed = 0

    # 检查传入的 func_and_grad 是否为可调用对象，否则抛出类型错误异常
    if not callable(func_and_grad):
        raise TypeError("tnc: function must be callable")

    # 如果传入了 callback 参数，则检查其是否为可调用对象，否则抛出类型错误异常
    if callback is not None:
        if not callable(callback):
            raise TypeError("tnc: callback must be callable or None.")
        # 将 Python 回调函数转换为 C 指针
        py_state.py_callback = <void*> callback
        callback_functionP = callback_function

    # 获取 x0 的大小，即优化变量的维度
    n = x0.size
    py_state.n = n

    # 获取 scale 数组的大小并转换为 C 的 float64_t 指针
    n3 = np.size(scale)
    if n3:
        scale_data = <float64_t *>np.PyArray_DATA(scale)

    # 获取 low 数组的大小并转换为 C 的 float64_t 指针
    n1 = np.size(low)
    if n1:
        low_data = <float64_t *>np.PyArray_DATA(low)

    # 获取 up 数组的大小并转换为 C 的 float64_t 指针
    n2 = np.size(up)
    if n2:
        up_data = <float64_t *>np.PyArray_DATA(up)

    # 获取 offset 数组的大小并转换为 C 的 float64_t 指针
    n4 = np.size(offset)
    if n4:
        offset_data = <float64_t *>np.PyArray_DATA(offset)

    # 检查向量大小是否一致，如果不一致则抛出值错误异常
    if (n1 != n2 or
        n != n1 or
        (scale_data != NULL and n != n3) or
        (offset_data != NULL and n != n4)):
        raise ValueError("tnc: vector sizes must be equal")

    # 在 C 中按照 C 顺序复制 x0，并获取其数据的 float64_t 指针
    x = np.copy(x0, order="C")
    g = np.zeros_like(x, dtype=np.float64)
    x_data = <float64_t *>np.PyArray_DATA(x)
    g_data = <float64_t *>np.PyArray_DATA(g)

    # 将 func_and_grad 转换为 C 指针
    py_state.py_function = <void*> func_and_grad

    # 调用 C 函数进行优化
    rc = tnc(n, x_data, &f, g_data, function, <void*> &py_state, low_data, up_data,
             scale_data, offset_data, messages, maxCGit, maxfun, eta, stepmx,
             accuracy, fmin, ftol, xtol, pgtol, rescale,
             &nfeval, &niter, callback_functionP);

    # 如果优化失败或者内存不足，返回 None
    if py_state.failed or rc == TNC_ENOMEM:
        return None

    # 返回优化结果和统计信息
    return rc, nfeval, niter, x, f, g
```