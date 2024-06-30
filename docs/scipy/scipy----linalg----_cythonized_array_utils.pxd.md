# `D:\src\scipysrc\scipy\scipy\linalg\_cythonized_array_utils.pxd`

```
# 导入必要的 numpy 库，使用 cimport 进行 C 扩展
cimport numpy as cnp

# 定义 lapack_t 类型，包括 float, double, float complex 和 double complex
ctypedef fused lapack_t:
    float
    double
    (float complex)
    (double complex)

# 定义 lapack_cz_t 类型，包括 float complex 和 double complex
ctypedef fused lapack_cz_t:
    (float complex)
    (double complex)

# 定义 lapack_sd_t 类型，包括 float 和 double
ctypedef fused lapack_sd_t:
    float
    double

# 定义 np_numeric_t 类型，包括多种整数和浮点数类型，通过 cnp 引入
ctypedef fused np_numeric_t:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.float32_t
    cnp.float64_t
    cnp.longdouble_t
    cnp.complex64_t
    cnp.complex128_t

# 定义 np_complex_numeric_t 类型，包括 complex64 和 complex128
ctypedef fused np_complex_numeric_t:
    cnp.complex64_t
    cnp.complex128_t

# 声明函数 swap_c_and_f_layout，接受 lapack_t 指针 a 和 b，以及整数 r 和 c，使用 C/C++，不引起 Python 异常，无需 GIL
cdef void swap_c_and_f_layout(lapack_t *a, lapack_t *b, int r, int c) noexcept nogil

# 声明函数 band_check_internal_c，接受 np_numeric_t 二维数组 A，使用 C/C++，不引起 Python 异常，无需 GIL
cdef (int, int) band_check_internal_c(np_numeric_t[:, ::1] A) noexcept nogil

# 声明函数 is_sym_her_real_c_internal，接受 np_numeric_t 二维数组 A，返回布尔值，使用 C/C++，不引起 Python 异常，无需 GIL
cdef bint is_sym_her_real_c_internal(np_numeric_t[:, ::1] A) noexcept nogil

# 声明函数 is_sym_her_complex_c_internal，接受 np_complex_numeric_t 二维数组 A，返回布尔值，使用 C/C++，不引起 Python 异常，无需 GIL
cdef bint is_sym_her_complex_c_internal(np_complex_numeric_t[:, ::1] A) noexcept nogil
```