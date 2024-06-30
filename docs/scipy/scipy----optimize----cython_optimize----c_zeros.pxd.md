# `D:\src\scipysrc\scipy\scipy\optimize\cython_optimize\c_zeros.pxd`

```
# 从指定的头文件 "../Zeros/zeros.h" 中导入外部定义
cdef extern from "../Zeros/zeros.h":
    # 定义一个回调函数类型 callback_type，接受一个 double 类型参数，返回一个 double 类型值，并且不抛出异常
    ctypedef double (*callback_type)(double, void*) noexcept
    # 定义一个结构体 scipy_zeros_info，包含 funcalls、iterations、error_num 三个整型成员
    ctypedef struct scipy_zeros_info:
        int funcalls
        int iterations
        int error_num

# 从 "../Zeros/bisect.c" 文件中导入外部定义，且不涉及GIL（全局解释器锁）
cdef extern from "../Zeros/bisect.c" nogil:
    # 定义 bisect 函数，接受回调函数 f、起始点 xa 和 xb、容差 xtol 和 rtol、最大迭代次数 iter、函数数据参数 func_data_param 和求解器统计信息 solver_stats
    double bisect(callback_type f, double xa, double xb, double xtol,
                  double rtol, int iter, void *func_data_param,
                  scipy_zeros_info *solver_stats)

# 从 "../Zeros/ridder.c" 文件中导入外部定义，且不涉及GIL
cdef extern from "../Zeros/ridder.c" nogil:
    # 定义 ridder 函数，接受回调函数 f、起始点 xa 和 xb、容差 xtol 和 rtol、最大迭代次数 iter、函数数据参数 func_data_param 和求解器统计信息 solver_stats
    double ridder(callback_type f, double xa, double xb, double xtol,
                  double rtol, int iter, void *func_data_param,
                  scipy_zeros_info *solver_stats)

# 从 "../Zeros/brenth.c" 文件中导入外部定义，且不涉及GIL
cdef extern from "../Zeros/brenth.c" nogil:
    # 定义 brenth 函数，接受回调函数 f、起始点 xa 和 xb、容差 xtol 和 rtol、最大迭代次数 iter、函数数据参数 func_data_param 和求解器统计信息 solver_stats
    double brenth(callback_type f, double xa, double xb, double xtol,
                  double rtol, int iter, void *func_data_param,
                  scipy_zeros_info *solver_stats)

# 从 "../Zeros/brentq.c" 文件中导入外部定义，且不涉及GIL
cdef extern from "../Zeros/brentq.c" nogil:
    # 定义 brentq 函数，接受回调函数 f、起始点 xa 和 xb、容差 xtol 和 rtol、最大迭代次数 iter、函数数据参数 func_data_param 和求解器统计信息 solver_stats
    double brentq(callback_type f, double xa, double xb, double xtol,
                  double rtol, int iter, void *func_data_param,
                  scipy_zeros_info *solver_stats)
```