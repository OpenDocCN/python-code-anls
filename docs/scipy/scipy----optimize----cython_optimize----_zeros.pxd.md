# `D:\src\scipysrc\scipy\scipy\optimize\cython_optimize\_zeros.pxd`

```
# Legacy public Cython API declarations
#
# NOTE: due to the way Cython ABI compatibility works, **no changes
# should be made to this file** --- any API additions/changes should be
# done in `cython_optimize.pxd` (see gh-11793).

# 定义回调函数类型，接受一个双精度浮点数和一个指针参数，且不抛出异常
ctypedef double (*callback_type)(double, void*) noexcept

# 定义结构体 zeros_parameters，包含一个回调函数和一个指针参数
ctypedef struct zeros_parameters:
    callback_type function
    void* args

# 定义结构体 zeros_full_output，包含函数调用次数、迭代次数、错误编号和一个双精度浮点数根
ctypedef struct zeros_full_output:
    int funcalls
    int iterations
    int error_num
    double root

# 声明一个使用二分法求根的函数，接受回调函数、区间端点、指针参数、误差容限、迭代次数、完整输出对象为参数，且不释放全局解锁
cdef double bisect(callback_type f, double xa, double xb, void* args,
                   double xtol, double rtol, int iter,
                   zeros_full_output *full_output) noexcept nogil

# 声明一个使用Ridder方法求根的函数，接受回调函数、区间端点、指针参数、误差容限、迭代次数、完整输出对象为参数，且不释放全局解锁
cdef double ridder(callback_type f, double xa, double xb, void* args,
                   double xtol, double rtol, int iter,
                   zeros_full_output *full_output) noexcept nogil

# 声明一个使用Brenth方法求根的函数，接受回调函数、区间端点、指针参数、误差容限、迭代次数、完整输出对象为参数，且不释放全局解锁
cdef double brenth(callback_type f, double xa, double xb, void* args,
                   double xtol, double rtol, int iter,
                   zeros_full_output *full_output) noexcept nogil

# 声明一个使用Brentq方法求根的函数，接受回调函数、区间端点、指针参数、误差容限、迭代次数、完整输出对象为参数，且不释放全局解锁
cdef double brentq(callback_type f, double xa, double xb, void* args,
                   double xtol, double rtol, int iter,
                   zeros_full_output *full_output) noexcept nogil
```