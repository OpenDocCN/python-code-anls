# `D:\src\scipysrc\scikit-learn\sklearn\utils\_typedefs.pxd`

```
# Commonly used types
# These typedefs define standard C types with specific byte sizes for use in Cython code.
# uint8_t: 8-bit unsigned integer type
ctypedef unsigned char uint8_t
# uint32_t: 32-bit unsigned integer type
ctypedef unsigned int uint32_t
# uint64_t: 64-bit unsigned integer type
ctypedef unsigned long long uint64_t

# Note: In NumPy 2, indexing always happens with npy_intp which is an alias for
# the Py_ssize_t type, see PEP 353.
#
# Note that on most platforms Py_ssize_t is equivalent to C99's intptr_t,
# but they can differ on architecture with segmented memory (none
# supported by scikit-learn at the time of writing).
#
# intp_t: Platform-dependent signed integer type used for array indexing
ctypedef Py_ssize_t intp_t

# float32_t: 32-bit floating point type
ctypedef float float32_t
# float64_t: 64-bit floating point type
ctypedef double float64_t

# Sparse matrices indices and indices' pointers arrays must use int32_t over
# intp_t because intp_t is platform dependent.
# When large sparse matrices are supported, indexing must use int64_t.
# See https://github.com/scikit-learn/scikit-learn/issues/23653 which tracks the
# ongoing work to support large sparse matrices.
#
# int8_t: 8-bit signed integer type
ctypedef signed char int8_t
# int32_t: 32-bit signed integer type
ctypedef signed int int32_t
# int64_t: 64-bit signed integer type
ctypedef signed long long int64_t
```