# `D:\src\scipysrc\scipy\scipy\_lib\_ccallback_c.pxd`

```
#
# Test function exports
#

# 定义使用Cython声明的函数，接受一个double类型参数a，一个指向整数的指针error_flag和一个void指针user_data，无异常处理，无全局解锁（nogil）
cdef double plus1_cython(double a, int *error_flag, void *user_data) except * nogil

# 定义使用Cython声明的函数，接受两个double类型参数a和b，一个指向整数的指针error_flag和一个void指针user_data，无异常处理，无全局解锁（nogil）
cdef double plus1b_cython(double a, double b, int *error_flag, void *user_data) except * nogil

# 定义使用Cython声明的函数，接受三个double类型参数a、b和c，一个指向整数的指针error_flag和一个void指针user_data，无异常处理，无全局解锁（nogil）
cdef double plus1bc_cython(double a, double b, double c, int *error_flag, void *user_data) except * nogil

# 定义使用Cython声明的函数，接受一个double类型参数a和一个void指针user_data，计算正弦值，无异常处理，无全局解锁（nogil）
cdef double sine(double a, void *user_data) except * nogil
```