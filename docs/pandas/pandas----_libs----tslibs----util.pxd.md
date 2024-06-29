# `D:\src\scipysrc\pandas\pandas\_libs\tslibs\util.pxd`

```
# 导入 CPython 的 PyTypeObject 对象
from cpython.object cimport PyTypeObject

# 从 "Python.h" 头文件中导入以下函数声明
# 注意：以 extern 方式导入允许声明为 nogil 函数，而 `from cpython cimport` 不允许
cdef extern from "Python.h":
    bint PyBool_Check(object obj) nogil
    bint PyFloat_Check(object obj) nogil
    bint PyComplex_Check(object obj) nogil
    bint PyObject_TypeCheck(object obj, PyTypeObject* type) nogil

    # 下面的函数可能会引发异常，因此不能声明为 'nogil'
    object PyUnicode_EncodeLocale(object obj, const char *errors) nogil
    object PyUnicode_DecodeLocale(const char *str, const char *errors) nogil


# 使用 cimport 导入 numpy 库，并从中选择特定的对象和类型
cimport numpy as cnp
from numpy cimport (
    PyArray_Check,
    float64_t,
    int64_t,
    is_timedelta64_object,
)

# 从 "numpy/arrayobject.h" 头文件中导入 PyFloatingArrType_Type 对象
cdef extern from "numpy/arrayobject.h":
    PyTypeObject PyFloatingArrType_Type

# 从 "numpy/ndarrayobject.h" 头文件中导入 PyComplexFloatingArrType_Type 和 PyBoolArrType_Type 对象
cdef extern from "numpy/ndarrayobject.h":
    PyTypeObject PyComplexFloatingArrType_Type
    PyTypeObject PyBoolArrType_Type

    # 声明 PyArray_IsIntegerScalar 函数的声明
    bint PyArray_IsIntegerScalar(obj) nogil

# 从 "numpy/npy_common.h" 头文件中导入 NPY_MIN_INT64 常量
cdef extern from "numpy/npy_common.h":
    int64_t NPY_MIN_INT64


# 定义一个内联函数 get_nat()，返回 NPY_MIN_INT64 常量的值
cdef inline int64_t get_nat() noexcept:
    return NPY_MIN_INT64


# --------------------------------------------------------------------
# Type Checking

# 定义一个内联函数 is_integer_object，检查对象是否为整数类型
cdef inline bint is_integer_object(object obj) noexcept:
    """
    Cython equivalent of

    `isinstance(val, (int, np.integer)) and not isinstance(val, (bool, np.timedelta64))`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_integer : bool

    Notes
    -----
    This counts np.timedelta64 objects as integers.
    """
    return (not PyBool_Check(obj) and isinstance(obj, (int, cnp.integer))
            and not is_timedelta64_object(obj))


# 定义一个内联函数 is_float_object，检查对象是否为浮点数类型
cdef inline bint is_float_object(object obj) noexcept nogil:
    """
    Cython equivalent of `isinstance(val, (float, np.floating))`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_float : bool
    """
    return (PyFloat_Check(obj) or
            (PyObject_TypeCheck(obj, &PyFloatingArrType_Type)))


# 定义一个内联函数 is_complex_object，检查对象是否为复数类型
cdef inline bint is_complex_object(object obj) noexcept nogil:
    """
    Cython equivalent of `isinstance(val, (complex, np.complexfloating))`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_complex : bool
    """
    return (PyComplex_Check(obj) or
            PyObject_TypeCheck(obj, &PyComplexFloatingArrType_Type))


# 定义一个内联函数 is_bool_object，检查对象是否为布尔类型
cdef inline bint is_bool_object(object obj) noexcept nogil:
    """
    Cython equivalent of `isinstance(val, (bool, np.bool_))`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_bool : bool
    """
    return (PyBool_Check(obj) or
            PyObject_TypeCheck(obj, &PyBoolArrType_Type))


# 定义一个内联函数 is_real_number_object，检查对象是否为实数类型（包括布尔、整数和浮点数）
cdef inline bint is_real_number_object(object obj) noexcept:
    return is_bool_object(obj) or is_integer_object(obj) or is_float_object(obj)


# 定义一个内联函数 is_array，检查对象是否为 numpy 数组类型
cdef inline bint is_array(object val) noexcept:
    """
    Cython equivalent of `isinstance(val, np.ndarray)`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_array : bool
    """
    # 使用 Cython 实现的功能，判断一个对象是否为 NumPy 数组 `val`。
    
    # 参数：
    # val : object
    #     要检查的对象。
    
    # 返回：
    # is_ndarray : bool
    #     如果 `val` 是 NumPy 数组，则返回 True；否则返回 False。
    """
    Cython equivalent of `isinstance(val, np.ndarray)`
    
    Parameters
    ----------
    val : object
    
    Returns
    -------
    is_ndarray : bool
    """
    return PyArray_Check(val)
cdef inline bint is_nan(object val):
    """
    Check if val is a Not-A-Number float or complex, including
    float('NaN') and np.nan.

    Parameters
    ----------
    val : object
        The input value to check for NaN.

    Returns
    -------
    is_nan : bool
        True if val is NaN, False otherwise.
    """
    cdef float64_t fval  # 声明一个 C 语言风格的双精度浮点数变量 fval
    if is_float_object(val):  # 如果 val 是浮点数对象
        fval = val  # 将 val 赋值给 fval
        return fval != fval  # 返回 fval 是否不等于自身（检测 NaN 的特性）
    return is_complex_object(val) and val != val  # 如果 val 是复数对象且不等于自身，则返回 True


cdef inline bytes string_encode_locale(str py_string):
    """
    Encode a Python string to bytes using the current system locale.

    Parameters
    ----------
    py_string : str
        The Python string to encode.

    Returns
    -------
    encoded_bytes : bytes
        The encoded bytes.
    """
    return PyUnicode_EncodeLocale(py_string, NULL)  # 使用当前系统的地区设置编码 Python 字符串为字节流


cdef inline object char_to_string_locale(const char* data):
    """
    Decode a C-style string to a Python object using the current system locale.

    Parameters
    ----------
    data : const char*
        The C-style string to decode.

    Returns
    -------
    decoded_object : object
        The decoded Python object.
    """
    return PyUnicode_DecodeLocale(data, NULL)  # 使用当前系统的地区设置解码 C 风格的字符串为 Python 对象
```