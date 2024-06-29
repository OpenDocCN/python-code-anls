# `D:\src\scipysrc\matplotlib\src\numpy_cpp.h`

```
/* -*- mode: c++; c-basic-offset: 4 -*- */

// 防止多次包含头文件
#ifndef MPL_NUMPY_CPP_H
#define MPL_NUMPY_CPP_H
// 定义宏，清除 Python.h 中可能定义的 PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
/***************************************************************************
 * This file is based on original work by Mark Wiebe, available at:
 *
 *    http://github.com/mwiebe/numpy-cpp
 *
 * However, the needs of matplotlib wrappers, such as treating an
 * empty array as having the correct dimensions, have made this rather
 * matplotlib-specific, so it's no longer compatible with the
 * original.
 */

#ifdef _POSIX_C_SOURCE
// 清除 _POSIX_C_SOURCE 的定义
#    undef _POSIX_C_SOURCE
#endif
#ifndef _AIX
#ifdef _XOPEN_SOURCE
// 清除 _XOPEN_SOURCE 的定义
#    undef _XOPEN_SOURCE
#endif
#endif

// 针对 stdlib.h 和 unistd.h 中 swab 定义的冲突做处理
// 只在 Sun 平台下执行以下操作
#if defined(__sun) || defined(sun)
#if defined(_XPG4)
// 清除 _XPG4 的定义
#undef _XPG4
#endif
#if defined(_XPG3)
// 清除 _XPG3 的定义
#undef _XPG3
#endif
#endif

// 包含 Python.h 和 numpy/ndarrayobject.h 头文件
#include <Python.h>
#include <numpy/ndarrayobject.h>

// 引入自定义的异常处理头文件
#include "py_exceptions.h"

#include <complex>

// 命名空间 numpy 开始
namespace numpy
{

// NumPy 类型的类型特征模板
template <typename T>
struct type_num_of;

/* Be careful with bool arrays as python has sizeof(npy_bool) == 1, but it is
 * not always the case that sizeof(bool) == 1.  Using the array_view_accessors
 * is always fine regardless of sizeof(bool), so do this rather than using
 * array.data() and pointer arithmetic which will not work correctly if
 * sizeof(bool) != 1. */
// 对于 bool 类型的特化处理，确保使用 array_view_accessors 以避免不同 sizeof(bool) 的问题
template <> struct type_num_of<bool>
{
    enum {
        value = NPY_BOOL
    };
};

// 各种 NumPy 类型的特化处理，定义其对应的 value 值
template <>
struct type_num_of<npy_byte>
{
    enum {
        value = NPY_BYTE
    };
};
template <>
struct type_num_of<npy_ubyte>
{
    enum {
        value = NPY_UBYTE
    };
};
template <>
struct type_num_of<npy_short>
{
    enum {
        value = NPY_SHORT
    };
};
template <>
struct type_num_of<npy_ushort>
{
    enum {
        value = NPY_USHORT
    };
};
template <>
struct type_num_of<npy_int>
{
    enum {
        value = NPY_INT
    };
};
template <>
struct type_num_of<npy_uint>
{
    enum {
        value = NPY_UINT
    };
};
template <>
struct type_num_of<npy_long>
{
    enum {
        value = NPY_LONG
    };
};
template <>
struct type_num_of<npy_ulong>
{
    enum {
        value = NPY_ULONG
    };
};
template <>
struct type_num_of<npy_longlong>
{
    enum {
        value = NPY_LONGLONG
    };
};
template <>
struct type_num_of<npy_ulonglong>
{
    enum {
        value = NPY_ULONGLONG
    };
};
template <>
struct type_num_of<npy_float>
{
    enum {
        value = NPY_FLOAT
    };
};
template <>
struct type_num_of<npy_double>
{
    enum {
        value = NPY_DOUBLE
    };
};
#if NPY_LONGDOUBLE != NPY_DOUBLE
template <>
struct type_num_of<npy_longdouble>
{
    enum {
        value = NPY_LONGDOUBLE
    };
};
#endif
template <>
struct type_num_of<npy_cfloat>
{
    enum {
        value = NPY_CFLOAT
    };
};
template <>
struct type_num_of<std::complex<npy_float> >
{
    enum {
        value = NPY_CFLOAT
    };
};
template <>
struct type_num_of<npy_cdouble>
{
    enum {
        value = NPY_CDOUBLE
    };


注释：


# 这行代码表示一个代码块的结束，通常用于结束一个函数、循环、条件语句或类定义。
# 在这个示例中，它可能是一个对象、函数或者其他代码块的结尾。
# 由于缺少上下文，具体代码块的类型和用途无法确定。
};
// 特化模板，针对 std::complex<npy_double> 类型，定义其对应的 NPY_CDOUBLE 值
template <>
struct type_num_of<std::complex<npy_double> >
{
    enum {
        value = NPY_CDOUBLE
    };
};
// 如果 NPY_CLONGDOUBLE 不等于 NPY_CDOUBLE，则特化模板，针对 npy_clongdouble 类型，定义其对应的 NPY_CLONGDOUBLE 值
#if NPY_CLONGDOUBLE != NPY_CDOUBLE
template <>
struct type_num_of<npy_clongdouble>
{
    enum {
        value = NPY_CLONGDOUBLE
    };
};
// 特化模板，针对 std::complex<npy_longdouble> 类型，定义其对应的 NPY_CLONGDOUBLE 值
template <>
struct type_num_of<std::complex<npy_longdouble> >
{
    enum {
        value = NPY_CLONGDOUBLE
    };
};
#endif
// 特化模板，针对 PyObject* 类型，定义其对应的 NPY_OBJECT 值
template <>
struct type_num_of<PyObject *>
{
    enum {
        value = NPY_OBJECT
    };
};
// 模板，针对 T& 类型，定义其是否为 const 的 value 值为 T 类型的 value 值
template <typename T>
struct type_num_of<T &>
{
    enum {
        value = type_num_of<T>::value
    };
};
// 模板，针对 const T 类型，定义其 value 值为 T 类型的 value 值
template <typename T>
struct type_num_of<const T>
{
    enum {
        value = type_num_of<T>::value
    };
};

// 模板，针对任意类型 T，定义其是否为 const 的 value 值为 false
template <typename T>
struct is_const
{
    enum {
        value = false
    };
};
// 模板，针对 const T 类型，定义其是否为 const 的 value 值为 true
template <typename T>
struct is_const<const T>
{
    enum {
        value = true
    };
};

namespace detail
{
// 模板，针对 AV<T, 1> 类型和维度 ND = 1，定义数组访问器类 array_view_accessors
template <template <typename, int> class AV, typename T>
class array_view_accessors<AV, T, 1>
{
  public:
    typedef AV<T, 1> AVC;
    typedef T sub_t;

    // 重载 () 运算符，返回第 i 个元素的引用
    T &operator()(npy_intp i)
    {
        AVC *self = static_cast<AVC *>(this);

        return *reinterpret_cast<T *>(self->m_data + self->m_strides[0] * i);
    }

    // 重载 () 运算符的 const 版本，返回第 i 个元素的常量引用
    const T &operator()(npy_intp i) const
    {
        const AVC *self = static_cast<const AVC *>(this);

        return *reinterpret_cast<const T *>(self->m_data + self->m_strides[0] * i);
    }

    // 重载 [] 运算符，返回第 i 个元素的引用
    T &operator[](npy_intp i)
    {
        AVC *self = static_cast<AVC *>(this);

        return *reinterpret_cast<T *>(self->m_data + self->m_strides[0] * i);
    }

    // 重载 [] 运算符的 const 版本，返回第 i 个元素的常量引用
    const T &operator[](npy_intp i) const
    {
        const AVC *self = static_cast<const AVC *>(this);

        return *reinterpret_cast<const T *>(self->m_data + self->m_strides[0] * i);
    }
};

// 模板，针对 AV<T, 2> 类型和维度 ND = 2，定义数组访问器类 array_view_accessors
template <template <typename, int> class AV, typename T>
class array_view_accessors<AV, T, 2>
{
  public:
    typedef AV<T, 2> AVC;
    typedef AV<T, 1> sub_t;

    // 重载 () 运算符，返回二维数组中 (i, j) 元素的引用
    T &operator()(npy_intp i, npy_intp j)
    {
        AVC *self = static_cast<AVC *>(this);

        return *reinterpret_cast<T *>(self->m_data + self->m_strides[0] * i +
                                      self->m_strides[1] * j);
    }

    // 重载 () 运算符的 const 版本，返回二维数组中 (i, j) 元素的常量引用
    const T &operator()(npy_intp i, npy_intp j) const
    {
        const AVC *self = static_cast<const AVC *>(this);

        return *reinterpret_cast<const T *>(self->m_data + self->m_strides[0] * i +
                                            self->m_strides[1] * j);
    }

    // 返回二维数组中第 i 行的子数组 AV<T, 1>
    sub_t subarray(npy_intp i) const
    {
        const AVC *self = static_cast<const AVC *>(this);

        return sub_t(self->m_arr,
                     self->m_data + self->m_strides[0] * i,
                     self->m_shape + 1,
                     self->m_strides + 1);
    }
};

// 模板，针对 AV<T, 3> 类型和维度 ND = 3，定义数组访问器类 array_view_accessors
template <template <typename, int> class AV, typename T>
class array_view_accessors<AV, T, 3>
{
  public:
    typedef AV<T, 3> AVC;
    typedef AV<T, 2> sub_t;
    # 返回三维数组中指定位置 (i, j, k) 处的元素的引用
    T &operator()(npy_intp i, npy_intp j, npy_intp k)
    {
        # 将当前对象转换为 AVC 类型的指针，以便访问成员变量
        AVC *self = static_cast<AVC *>(this);

        # 根据指定的索引 i, j, k 计算出在内存中的偏移量，并通过 reinterpret_cast 转换为对应类型的指针
        return *reinterpret_cast<T *>(self->m_data + self->m_strides[0] * i +
                                      self->m_strides[1] * j + self->m_strides[2] * k);
    }

    # 返回三维数组中指定位置 (i, j, k) 处的常量引用
    const T &operator()(npy_intp i, npy_intp j, npy_intp k) const
    {
        # 将当前对象转换为 AVC 类型的常量指针，以便访问成员变量
        const AVC *self = static_cast<const AVC *>(this);

        # 根据指定的索引 i, j, k 计算出在内存中的偏移量，并通过 reinterpret_cast 转换为对应类型的常量指针
        return *reinterpret_cast<const T *>(self->m_data + self->m_strides[0] * i +
                                            self->m_strides[1] * j + self->m_strides[2] * k);
    }

    # 返回三维数组中指定第 i 个子数组的子视图
    sub_t subarray(npy_intp i) const
    {
        # 将当前对象转换为 AVC 类型的常量指针，以便访问成员变量
        const AVC *self = static_cast<const AVC *>(this);

        # 创建并返回一个 sub_t 类型的对象，该对象表示原数组的第 i 个子数组的子视图
        return sub_t(self->m_arr,
                     self->m_data + self->m_strides[0] * i,  # 子数组数据在内存中的起始地址
                     self->m_shape + 1,  # 子数组的形状（维度信息），比原数组多一维
                     self->m_strides + 1);  # 子数组的步长信息，比原数组多一维
    }
};

// 结束了类模板 array_view 的定义

// 当添加 array_view_accessors 的实例化时，记得在下面的 zeros[] 中添加条目
// 这里是一个静态成员变量的定义，用于初始化全零的 npy_intp 数组

}

static npy_intp zeros[] = { 0, 0, 0 };

template <typename T, int ND>
class array_view : public detail::array_view_accessors<array_view, T, ND>
{
    friend class detail::array_view_accessors<numpy::array_view, T, ND>;

  private:
    // 数组数据的副本
    PyArrayObject *m_arr;
    // 数组的形状
    npy_intp *m_shape;
    // 数组的步幅
    npy_intp *m_strides;
    // 数组的数据指针
    char *m_data;

  public:
    typedef T value_type;

    enum {
        ndim = ND
    };

    // 默认构造函数，初始化成员变量
    array_view() : m_arr(NULL), m_data(NULL)
    {
        // 将形状和步幅初始化为全零
        m_shape = zeros;
        m_strides = zeros;
    }

    // 带参数的构造函数，初始化成员变量，并根据给定参数设置数组视图
    array_view(PyObject *arr, bool contiguous = false) : m_arr(NULL), m_data(NULL)
    {
        // 如果设置数组视图失败，则抛出异常
        if (!set(arr, contiguous)) {
            throw mpl::exception();
        }
    }

    // 拷贝构造函数，深拷贝另一个 array_view 对象的成员变量
    array_view(const array_view &other) : m_arr(NULL), m_data(NULL)
    {
        m_arr = other.m_arr;
        Py_XINCREF(m_arr);
        m_data = other.m_data;
        m_shape = other.m_shape;
        m_strides = other.m_strides;
    }

    // 构造函数，接受 PyArrayObject 和相关参数，初始化数组视图
    array_view(PyArrayObject *arr, char *data, npy_intp *shape, npy_intp *strides)
    {
        m_arr = arr;
        Py_XINCREF(arr);
        m_data = data;
        m_shape = shape;
        m_strides = strides;
    }

    // 构造函数，接受 PyArrayObject，并从中提取形状、步幅和数据，初始化数组视图
    array_view(PyArrayObject *arr)
    {
        m_arr = arr;
        Py_XINCREF(arr);
        m_shape = PyArray_DIMS(m_arr);
        m_strides = PyArray_STRIDES(m_arr);
        m_data = PyArray_BYTES(m_arr);
    }

    // 构造函数，接受形状数组，创建新的 PyArrayObject，并设置数组视图
    array_view(npy_intp shape[ND]) : m_arr(NULL), m_shape(NULL), m_strides(NULL), m_data(NULL)
    {
        PyObject *arr = PyArray_SimpleNew(ND, shape, type_num_of<T>::value);
        if (arr == NULL) {
            throw mpl::exception();
        }
        if (!set(arr, true)) {
            Py_DECREF(arr);
            throw mpl::exception();
        }
        Py_DECREF(arr);
    }

    // 析构函数，释放 m_arr 的引用计数
    ~array_view()
    {
        Py_XDECREF(m_arr);
    }

    // 赋值运算符重载，深拷贝另一个 array_view 对象的成员变量
    array_view& operator=(const array_view &other)
    {
        if (this != &other)
        {
            Py_XDECREF(m_arr);
            m_arr = other.m_arr;
            Py_XINCREF(m_arr);
            m_data = other.m_data;
            m_shape = other.m_shape;
            m_strides = other.m_strides;
        }
        return *this;
    }

    // 设置数组视图，根据给定的 PyArrayObject 和 contiguous 参数
    {
        PyArrayObject *tmp;  // 定义一个 PyArrayObject 类型的临时指针变量 tmp
    
        if (arr == NULL || arr == Py_None) {  // 如果传入的 arr 是空指针或者 Py_None
            Py_XDECREF(m_arr);  // 释放 m_arr 的引用计数
            m_arr = NULL;  // 将 m_arr 置为 NULL
            m_data = NULL;  // 将 m_data 置为 NULL
            m_shape = zeros;  // 将 m_shape 设为 zeros 数组
            m_strides = zeros;  // 将 m_strides 设为 zeros 数组
        } else {
            if (contiguous) {  // 如果需要连续的数组
                tmp = (PyArrayObject *)PyArray_ContiguousFromAny(arr, type_num_of<T>::value, 0, ND);  // 将 arr 转换为连续的 PyArrayObject
            } else {
                tmp = (PyArrayObject *)PyArray_FromObject(arr, type_num_of<T>::value, 0, ND);  // 将 arr 转换为 PyArrayObject
            }
            if (tmp == NULL) {  // 如果转换失败
                return false;  // 返回 false
            }
    
            if (PyArray_NDIM(tmp) == 0 || PyArray_DIM(tmp, 0) == 0) {  // 如果 tmp 是 0 维或者第一维度大小为 0
                Py_XDECREF(m_arr);  // 释放 m_arr 的引用计数
                m_arr = NULL;  // 将 m_arr 置为 NULL
                m_data = NULL;  // 将 m_data 置为 NULL
                m_shape = zeros;  // 将 m_shape 设为 zeros 数组
                m_strides = zeros;  // 将 m_strides 设为 zeros 数组
                if (PyArray_NDIM(tmp) == 0 && ND == 0) {  // 如果 tmp 是 0 维且 ND 也是 0
                    m_arr = tmp;  // 将 tmp 赋给 m_arr
                    return true;  // 返回 true
                }
            }
            if (PyArray_NDIM(tmp) != ND) {  // 如果 tmp 的维度与 ND 不相等
                PyErr_Format(PyExc_ValueError,
                             "Expected %d-dimensional array, got %d",
                             ND,
                             PyArray_NDIM(tmp));  // 抛出维度不匹配的异常
                Py_DECREF(tmp);  // 释放 tmp 的引用计数
                return false;  // 返回 false
            }
    
            /* 将一些数据复制到视图对象以便更快地访问 */
            Py_XDECREF(m_arr);  // 释放 m_arr 的引用计数
            m_arr = tmp;  // 将 tmp 赋给 m_arr
            m_shape = PyArray_DIMS(m_arr);  // 获取 m_arr 的形状
            m_strides = PyArray_STRIDES(m_arr);  // 获取 m_arr 的步幅
            m_data = PyArray_BYTES(tmp);  // 获取 m_arr 的数据指针
        }
    
        return true;  // 返回 true
    }
    
    npy_intp shape(size_t i) const
    {
        if (i >= ND) {  // 如果 i 大于等于 ND
            return 0;  // 返回 0
        }
        return m_shape[i];  // 返回 m_shape 的第 i 个元素
    }
    
    size_t size() const;  // 声明 size 函数
    
    // 不要用于 array_view<bool, ND>。请参见文件顶部附近的注释。
    const T *data() const
    {
        return (const T *)m_data;  // 返回 m_data 强制转换为 T 类型的指针
    }
    
    // 不要用于 array_view<bool, ND>。请参见文件顶部附近的注释。
    T *data()
    {
        return (T *)m_data;  // 返回 m_data 强制转换为 T 类型的指针
    }
    
    // 返回一个新的引用。
    PyObject *pyobj()
    {
        Py_XINCREF(m_arr);  // 增加 m_arr 的引用计数
        return (PyObject *)m_arr;  // 返回 m_arr 强制转换为 PyObject 指针
    }
    
    // 窃取一个引用。
    PyObject *pyobj_steal()
    {
        return (PyObject *)m_arr;  // 返回 m_arr 强制转换为 PyObject 指针
    }
    
    static int converter(PyObject *obj, void *arrp)
    {
        array_view<T, ND> *arr = (array_view<T, ND> *)arrp;  // 将 arrp 转换为 array_view<T, ND> 指针
    
        if (!arr->set(obj)) {  // 如果无法设置 arr 的值为 obj
            return 0;  // 返回 0
        }
    
        return 1;  // 返回 1
    }
    
    static int converter_contiguous(PyObject *obj, void *arrp)
    {
        array_view<T, ND> *arr = (array_view<T, ND> *)arrp;  // 将 arrp 转换为 array_view<T, ND> 指针
    
        if (!arr->set(obj, true)) {  // 如果无法设置 arr 的值为 obj，并且要求连续
            return 0;  // 返回 0
        }
    
        return 1;  // 返回 1
    }
};

/* 在大多数情况下，代码应该使用 safe_first_shape(obj) 而不是 obj.shape(0)，
   因为当任何维度为0时，safe_first_shape(obj) == 0。 */
template <typename T, int ND>
size_t
safe_first_shape(const array_view<T, ND> &a)
{
    // 检查数组的维度是否为空
    bool empty = (ND == 0);
    // 遍历数组的每个维度
    for (size_t i = 0; i < ND; i++) {
        // 如果某个维度的大小为0，则标记数组为空
        if (a.shape(i) == 0) {
            empty = true;
        }
    }
    // 如果数组为空，则返回0
    if (empty) {
        return 0;
    } else {
        // 否则返回数组的第一个维度的大小
        return (size_t)a.shape(0);
    }
}

template <typename T, int ND>
size_t
array_view<T, ND>::size() const
{
    // 调用 safe_first_shape 函数计算数组视图的大小
    return safe_first_shape<T, ND>(*this);
}

} // namespace numpy

#endif
```