# `D:\src\scipysrc\numpy\numpy\random\_common.pxd`

```
# 设置 Cython 语言级别为 Python 3
# 从 libc.stdint 中导入特定类型的整数
from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t

# 导入 NumPy 库，并且使用 Cython 语法导入它
import numpy as np
cimport numpy as np

# 定义 C 语言的双精度浮点数变量
cdef double POISSON_LAM_MAX
cdef double LEGACY_POISSON_LAM_MAX
cdef uint64_t MAXSIZE

# 定义枚举类型 ConstraintType，并给出每个类型的值
cdef enum ConstraintType:
    CONS_NONE
    CONS_NON_NEGATIVE
    CONS_POSITIVE
    CONS_POSITIVE_NOT_NAN
    CONS_BOUNDED_0_1
    CONS_BOUNDED_GT_0_1
    CONS_BOUNDED_LT_0_1
    CONS_GT_1
    CONS_GTE_1
    CONS_POISSON
    LEGACY_CONS_POISSON
    LEGACY_CONS_NON_NEGATIVE_INBOUNDS_LONG

# 将 ConstraintType 定义为 constraint_type 类型别名
ctypedef ConstraintType constraint_type

# 声明一些 C 函数的原型，这些函数将在后续的代码中实现
cdef object benchmark(bitgen_t *bitgen, object lock, Py_ssize_t cnt, object method)
cdef object random_raw(bitgen_t *bitgen, object lock, object size, object output)
cdef object prepare_cffi(bitgen_t *bitgen)
cdef object prepare_ctypes(bitgen_t *bitgen)
cdef int check_constraint(double val, object name, constraint_type cons) except -1
cdef int check_array_constraint(np.ndarray val, object name, constraint_type cons) except -1

# 从外部头文件 "include/aligned_malloc.h" 中导入一些 C 函数的原型
cdef extern from "include/aligned_malloc.h":
    cdef void *PyArray_realloc_aligned(void *p, size_t n)
    cdef void *PyArray_malloc_aligned(size_t n)
    cdef void *PyArray_calloc_aligned(size_t n, size_t s)
    cdef void PyArray_free_aligned(void *p)

# 定义几种函数指针类型，这些指针将指向后续定义的函数
ctypedef void (*random_double_fill)(bitgen_t *state, np.npy_intp count, double* out)  noexcept nogil
ctypedef double (*random_double_0)(void *state)  noexcept nogil
ctypedef double (*random_double_1)(void *state, double a)  noexcept nogil
ctypedef double (*random_double_2)(void *state, double a, double b)  noexcept nogil
ctypedef double (*random_double_3)(void *state, double a, double b, double c)  noexcept nogil

ctypedef void (*random_float_fill)(bitgen_t *state, np.npy_intp count, float* out)  noexcept nogil
ctypedef float (*random_float_0)(bitgen_t *state)  noexcept nogil
ctypedef float (*random_float_1)(bitgen_t *state, float a)  noexcept nogil

ctypedef int64_t (*random_uint_0)(void *state)  noexcept nogil
ctypedef int64_t (*random_uint_d)(void *state, double a)  noexcept nogil
ctypedef int64_t (*random_uint_dd)(void *state, double a, double b)  noexcept nogil
ctypedef int64_t (*random_uint_di)(void *state, double a, uint64_t b)  noexcept nogil
ctypedef int64_t (*random_uint_i)(void *state, int64_t a)  noexcept nogil
ctypedef int64_t (*random_uint_iii)(void *state, int64_t a, int64_t b, int64_t c)  noexcept nogil

ctypedef uint32_t (*random_uint_0_32)(bitgen_t *state)  noexcept nogil
ctypedef uint32_t (*random_uint_1_i_32)(bitgen_t *state, uint32_t a)  noexcept nogil

ctypedef int32_t (*random_int_2_i_32)(bitgen_t *state, int32_t a, int32_t b)  noexcept nogil
ctypedef int64_t (*random_int_2_i)(bitgen_t *state, int64_t a, int64_t b)  noexcept nogil

# 声明一个 C 函数的原型，这个函数在后续代码中实现
cdef double kahan_sum(double *darr, np.npy_intp n) noexcept

# 定义一个内联函数，将 uint64_t 类型的整数转换为 double 类型的浮点数
cdef inline double uint64_to_double(uint64_t rnd) noexcept nogil:
    return (rnd >> 11) * (1.0 / 9007199254740992.0)
# 定义 C 语言扩展函数 double_fill，接受指针 func、bitgen_t 类型的 state，以及 size、lock 和 out 作为参数，返回一个 Python 对象
cdef object double_fill(void *func, bitgen_t *state, object size, object lock, object out)

# 定义 C 语言扩展函数 float_fill，接受指针 func、bitgen_t 类型的 state，以及 size、lock 和 out 作为参数，返回一个 Python 对象
cdef object float_fill(void *func, bitgen_t *state, object size, object lock, object out)

# 定义 C 语言扩展函数 float_fill_from_double，接受指针 func、bitgen_t 类型的 state，以及 size、lock 和 out 作为参数，返回一个 Python 对象
cdef object float_fill_from_double(void *func, bitgen_t *state, object size, object lock, object out)

# 定义 C 语言扩展函数 wrap_int，接受 val 和 bits 作为参数，返回一个 Python 对象
cdef object wrap_int(object val, object bits)

# 定义 C 语言扩展函数 int_to_array，接受 value、name、bits 和 uint_size 作为参数，返回一个 NumPy 数组对象
cdef np.ndarray int_to_array(object value, object name, object bits, object uint_size)

# 定义 C 语言扩展函数 validate_output_shape，接受 iter_shape 和 output 作为参数，无返回值
cdef validate_output_shape(iter_shape, np.ndarray output)

# 定义 C 语言扩展函数 cont，接受指针 func、state、size、lock、narg 和多个参数（a、a_name、a_constraint、b、b_name、b_constraint、c、c_name、c_constraint）作为参数，返回一个 Python 对象
cdef object cont(void *func, void *state, object size, object lock, int narg,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint,
                 object out)

# 定义 C 语言扩展函数 disc，接受指针 func、state、size、lock、narg_double、narg_int64 和多个参数（a、a_name、a_constraint、b、b_name、b_constraint、c、c_name、c_constraint）作为参数，返回一个 Python 对象
cdef object disc(void *func, void *state, object size, object lock,
                 int narg_double, int narg_int64,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint)

# 定义 C 语言扩展函数 cont_f，接受指针 func、bitgen_t 类型的 state、size、lock 和多个参数（a、a_name、a_constraint）作为参数，返回一个 Python 对象
cdef object cont_f(void *func, bitgen_t *state, object size, object lock,
                   object a, object a_name, constraint_type a_constraint,
                   object out)

# 定义 C 语言扩展函数 cont_broadcast_3，接受指针 func、state、size、lock、多个 NumPy 数组参数（a_arr、a_name、a_constraint、b_arr、b_name、b_constraint、c_arr、c_name、c_constraint）作为参数，返回一个 Python 对象
cdef object cont_broadcast_3(void *func, void *state, object size, object lock,
                             np.ndarray a_arr, object a_name, constraint_type a_constraint,
                             np.ndarray b_arr, object b_name, constraint_type b_constraint,
                             np.ndarray c_arr, object c_name, constraint_type c_constraint)

# 定义 C 语言扩展函数 discrete_broadcast_iii，接受指针 func、state、size、lock、多个 NumPy 数组参数（a_arr、a_name、a_constraint、b_arr、b_name、b_constraint、c_arr、c_name、c_constraint）作为参数，返回一个 Python 对象
cdef object discrete_broadcast_iii(void *func, void *state, object size, object lock,
                                   np.ndarray a_arr, object a_name, constraint_type a_constraint,
                                   np.ndarray b_arr, object b_name, constraint_type b_constraint,
                                   np.ndarray c_arr, object c_name, constraint_type c_constraint)
```