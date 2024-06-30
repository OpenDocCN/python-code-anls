# `D:\src\scipysrc\scikit-learn\sklearn\utils\_vector_sentinel.pyx`

```
# 导入 deref 函数，用于解引用操作符
from cython.operator cimport dereference as deref
# 导入 Py_INCREF 函数，用于增加 Python 对象的引用计数
from cpython.ref cimport Py_INCREF
# 导入 numpy 的 C API 接口
cimport numpy as cnp

# 调用 numpy 的 import_array 函数，初始化 numpy 数组 API
cnp.import_array()


# 定义一个 Cython 函数 _create_sentinel，返回一个 StdVectorSentinel 对象
cdef StdVectorSentinel _create_sentinel(vector_typed * vect_ptr):
    # 根据 vector_typed 类型判断，选择创建不同类型的 StdVectorSentinel 对象
    if vector_typed is vector[float64_t]:
        return StdVectorSentinelFloat64.create_for(vect_ptr)
    elif vector_typed is vector[int32_t]:
        return StdVectorSentinelInt32.create_for(vect_ptr)
    elif vector_typed is vector[int64_t]:
        return StdVectorSentinelInt64.create_for(vect_ptr)
    else:  # intp_t
        return StdVectorSentinelIntP.create_for(vect_ptr)


# 定义一个 Cython 类 StdVectorSentinel，用于包装引用向量并管理其生命周期
cdef class StdVectorSentinel:
    """Wraps a reference to a vector which will be deallocated with this object.

    When created, the StdVectorSentinel swaps the reference of its internal
    vectors with the provided one (vect_ptr), thus making the StdVectorSentinel
    manage the provided one's lifetime.
    """
    cdef void* get_data(self):
        """Return pointer to data."""
        ...

    cdef int get_typenum(self):
        """Get typenum for PyArray_SimpleNewFromData."""
        ...


# 定义一个 Cython 类 StdVectorSentinelFloat64，继承自 StdVectorSentinel
cdef class StdVectorSentinelFloat64(StdVectorSentinel):
    cdef vector[float64_t] vec

    @staticmethod
    cdef StdVectorSentinel create_for(vector[float64_t] * vect_ptr):
        # 创建 StdVectorSentinelFloat64 对象，直接使用指针 vect_ptr 而不调用 __init__
        # 参考: https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers # noqa
        cdef StdVectorSentinelFloat64 sentinel = StdVectorSentinelFloat64.__new__(StdVectorSentinelFloat64)
        sentinel.vec.swap(deref(vect_ptr))
        return sentinel

    cdef void* get_data(self):
        """Return pointer to data."""
        return self.vec.data()

    cdef int get_typenum(self):
        """Get typenum for PyArray_SimpleNewFromData."""
        return cnp.NPY_FLOAT64


# 定义一个 Cython 类 StdVectorSentinelIntP，继承自 StdVectorSentinel
cdef class StdVectorSentinelIntP(StdVectorSentinel):
    cdef vector[intp_t] vec

    @staticmethod
    cdef StdVectorSentinel create_for(vector[intp_t] * vect_ptr):
        # 创建 StdVectorSentinelIntP 对象，直接使用指针 vect_ptr 而不调用 __init__
        # 参考: https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers # noqa
        cdef StdVectorSentinelIntP sentinel = StdVectorSentinelIntP.__new__(StdVectorSentinelIntP)
        sentinel.vec.swap(deref(vect_ptr))
        return sentinel

    cdef void* get_data(self):
        """Return pointer to data."""
        return self.vec.data()

    cdef int get_typenum(self):
        """Get typenum for PyArray_SimpleNewFromData."""
        return cnp.NPY_INTP


# 定义一个 Cython 类 StdVectorSentinelInt32，继承自 StdVectorSentinel
cdef class StdVectorSentinelInt32(StdVectorSentinel):
    cdef vector[int32_t] vec

    @staticmethod
    cdef StdVectorSentinel create_for(vector[int32_t] * vect_ptr):
        # 创建 StdVectorSentinelInt32 对象，直接使用指针 vect_ptr 而不调用 __init__
        # 参考: https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers # noqa
        cdef StdVectorSentinelInt32 sentinel = StdVectorSentinelInt32.__new__(StdVectorSentinelInt32)
        sentinel.vec.swap(deref(vect_ptr))
        return sentinel
    # 返回向量数据的指针
    cdef void* get_data(self):
        return self.vec.data()

    # 返回数据的类型码，这里指定为32位整数
    cdef int get_typenum(self):
        return cnp.NPY_INT32
cdef class StdVectorSentinelInt64(StdVectorSentinel):
    # 定义一个 Cython 扩展类型 StdVectorSentinelInt64，继承自 StdVectorSentinel
    cdef vector[int64_t] vec  # 声明一个 C++ STL 向量，存储 int64_t 类型的元素

    @staticmethod
    cdef StdVectorSentinel create_for(vector[int64_t] * vect_ptr):
        # 创建一个静态方法 create_for，用于从现有的 C++ 向量指针创建 StdVectorSentinelInt64 对象
        # 这个方法直接初始化对象，而不调用 __init__ 方法
        # 参考：https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers # noqa
        cdef StdVectorSentinelInt64 sentinel = StdVectorSentinelInt64.__new__(StdVectorSentinelInt64)
        sentinel.vec.swap(deref(vect_ptr))  # 交换指针指向的向量内容到 sentinel 的 vec 中
        return sentinel

    cdef void* get_data(self):
        # 返回向量数据的指针
        return self.vec.data()

    cdef int get_typenum(self):
        # 返回 Numpy 中对应 int64 类型的类型编号
        return cnp.NPY_INT64


cdef cnp.ndarray vector_to_nd_array(vector_typed * vect_ptr):
    cdef:
        cnp.npy_intp size = deref(vect_ptr).size()  # 获取向量的大小
        StdVectorSentinel sentinel = _create_sentinel(vect_ptr)  # 创建一个 StdVectorSentinel 对象
        cnp.ndarray arr = cnp.PyArray_SimpleNewFromData(
            1, &size, sentinel.get_typenum(), sentinel.get_data())
        # 从向量数据创建一个一维的 Numpy 数组

    # 让 Numpy 数组管理其缓冲区的生命周期。
    # PyArray_SetBaseObject 调用时会窃取对 StdVectorSentinel 的引用，因此需要增加其引用计数。
    # 参考：https://docs.python.org/3/c-api/intro.html#reference-count-details
    Py_INCREF(sentinel)
    cnp.PyArray_SetBaseObject(arr, sentinel)
    return arr
```