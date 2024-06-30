# `D:\src\scipysrc\scipy\scipy\sparse\sparsetools\util.h`

```
#ifndef __SPTOOLS_UTIL_H__
#define __SPTOOLS_UTIL_H__

/*
 * Same as std::divides, except return x/0 == 0 for integer types, without
 * raising a SIGFPE.
 */
// 定义一个模板结构体 safe_divides，用于实现除法运算，避免对整数类型做除零操作导致的 SIGFPE 信号
template <class T>
struct safe_divides {
    // 重载 () 运算符，实现安全的除法：当除数为0时返回0，否则返回商
    T operator() (const T& x, const T& y) const {
        if (y == 0) {
            return 0;
        }
        else {
            return x/y;
        }
    }

    // 定义类型别名
    typedef T first_argument_type;
    typedef T second_argument_type;
    typedef T result_type;
};

// 宏定义，用于特化 safe_divides 模板结构体的运算符 ()，针对不同的数据类型
#define OVERRIDE_safe_divides(typ) \
    template<> inline typ safe_divides<typ>::operator()(const typ& x, const typ& y) const { return x/y; }

// 特化各种浮点类型的 safe_divides 模板结构体运算符 ()
OVERRIDE_safe_divides(float)
OVERRIDE_safe_divides(double)
OVERRIDE_safe_divides(long double)
OVERRIDE_safe_divides(npy_cfloat_wrapper)
OVERRIDE_safe_divides(npy_cdouble_wrapper)
OVERRIDE_safe_divides(npy_clongdouble_wrapper)

// 取消宏定义 OVERRIDE_safe_divides
#undef OVERRIDE_safe_divides

// 定义模板结构体 maximum，实现比较两个值大小并返回较大值
template <class T>
struct maximum {
    T operator() (const T& x, const T& y) const {
        return std::max(x, y);
    }
};

// 定义模板结构体 minimum，实现比较两个值大小并返回较小值
template <class T>
struct minimum {
    T operator() (const T& x, const T& y) const {
        return std::min(x, y);
    }
};

// 宏定义，用于生成各种数据类型的枚举常量和相应的类型别名
#define SPTOOLS_FOR_EACH_DATA_TYPE_CODE(X)      \
  X(NPY_BOOL, npy_bool_wrapper)                 \
  X(NPY_BYTE, npy_byte)                         \
  X(NPY_UBYTE, npy_ubyte)                       \
  X(NPY_SHORT, npy_short)                       \
  X(NPY_USHORT, npy_ushort)                     \
  X(NPY_INT, npy_int)                           \
  X(NPY_UINT, npy_uint)                         \
  X(NPY_LONG, npy_long)                         \
  X(NPY_ULONG, npy_ulong)                       \
  X(NPY_LONGLONG, npy_longlong)                 \
  X(NPY_ULONGLONG, npy_ulonglong)               \
  X(NPY_FLOAT, npy_float)                       \
  X(NPY_DOUBLE, npy_double)                     \
  X(NPY_LONGDOUBLE, npy_longdouble)             \
  X(NPY_CFLOAT, npy_cfloat_wrapper)             \
  X(NPY_CDOUBLE, npy_cdouble_wrapper)           \
  X(NPY_CLONGDOUBLE, npy_clongdouble_wrapper)

// 如果 npy_longdouble 与 npy_double 不同大小，则执行以下代码段
#if NPY_SIZEOF_LONGDOUBLE != NPY_SIZEOF_DOUBLE
#define SPTOOLS_FOR_EACH_DATA_TYPE(X)           \  // 定义宏 SPTOOLS_FOR_EACH_DATA_TYPE，用于迭代各种数据类型
  X(NPY_BOOL, npy_bool_wrapper)                 \  // 展开宏 X，传入 NPY_BOOL 和 npy_bool_wrapper
  X(NPY_BYTE, npy_byte)                         \  // 展开宏 X，传入 NPY_BYTE 和 npy_byte
  X(NPY_UBYTE, npy_ubyte)                       \  // 展开宏 X，传入 NPY_UBYTE 和 npy_ubyte
  X(NPY_SHORT, npy_short)                       \  // 展开宏 X，传入 NPY_SHORT 和 npy_short
  X(NPY_USHORT, npy_ushort)                     \  // 展开宏 X，传入 NPY_USHORT 和 npy_ushort
  X(NPY_INT, npy_int)                           \  // 展开宏 X，传入 NPY_INT 和 npy_int
  X(NPY_UINT, npy_uint)                         \  // 展开宏 X，传入 NPY_UINT 和 npy_uint
  X(NPY_LONG, npy_long)                         \  // 展开宏 X，传入 NPY_LONG 和 npy_long
  X(NPY_ULONG, npy_ulong)                       \  // 展开宏 X，传入 NPY_ULONG 和 npy_ulong
  X(NPY_LONGLONG, npy_longlong)                 \  // 展开宏 X，传入 NPY_LONGLONG 和 npy_longlong
  X(NPY_ULONGLONG, npy_ulonglong)               \  // 展开宏 X，传入 NPY_ULONGLONG 和 npy_ulonglong
  X(NPY_FLOAT, npy_float)                       \  // 展开宏 X，传入 NPY_FLOAT 和 npy_float
  X(NPY_DOUBLE, npy_double)                     \  // 展开宏 X，传入 NPY_DOUBLE 和 npy_double
  X(NPY_LONGDOUBLE, npy_longdouble)             \  // 展开宏 X，传入 NPY_LONGDOUBLE 和 npy_longdouble
  X(NPY_CFLOAT, npy_cfloat_wrapper)             \  // 展开宏 X，传入 NPY_CFLOAT 和 npy_cfloat_wrapper
  X(NPY_CDOUBLE, npy_cdouble_wrapper)           \  // 展开宏 X，传入 NPY_CDOUBLE 和 npy_cdouble_wrapper
  X(NPY_CLONGDOUBLE, npy_clongdouble_wrapper)   // 展开宏 X，传入 NPY_CLONGDOUBLE 和 npy_clongdouble_wrapper


#define SPTOOLS_FOR_EACH_INDEX_DATA_TYPE_COMBINATION(X) \  // 定义宏 SPTOOLS_FOR_EACH_INDEX_DATA_TYPE_COMBINATION，用于迭代索引数据类型组合
  X(npy_int32, npy_bool_wrapper)                        \  // 展开宏 X，传入 npy_int32 和 npy_bool_wrapper
  X(npy_int32, npy_byte)                                \  // 展开宏 X，传入 npy_int32 和 npy_byte
  X(npy_int32, npy_ubyte)                               \  // 展开宏 X，传入 npy_int32 和 npy_ubyte
  X(npy_int32, npy_short)                               \  // 展开宏 X，传入 npy_int32 和 npy_short
  X(npy_int32, npy_ushort)                              \  // 展开宏 X，传入 npy_int32 和 npy_ushort
  X(npy_int32, npy_int)                                 \  // 展开宏 X，传入 npy_int32 和 npy_int
  X(npy_int32, npy_uint)                                \  // 展开宏 X，传入 npy_int32 和 npy_uint
  X(npy_int32, npy_long)                                \  // 展开宏 X，传入 npy_int32 和 npy_long
  X(npy_int32, npy_ulong)                               \  // 展开宏 X，传入 npy_int32 和 npy_ulong
  X(npy_int32, npy_longlong)                            \  // 展开宏 X，传入 npy_int32 和 npy_longlong
  X(npy_int32, npy_ulonglong)                           \  // 展开宏 X，传入 npy_int32 和 npy_ulonglong
  X(npy_int32, npy_float)                               \  // 展开宏 X，传入 npy_int32 和 npy_float
  X(npy_int32, npy_double)                              \  // 展开宏 X，传入 npy_int32 和 npy_double
  X(npy_int32, npy_longdouble)                          \  // 展开宏 X，传入 npy_int32 和 npy_longdouble
  X(npy_int32, npy_cfloat_wrapper)                      \  // 展开宏 X，传入 npy_int32 和 npy_cfloat_wrapper
  X(npy_int32, npy_cdouble_wrapper)                     \  // 展开宏 X，传入 npy_int32 和 npy_cdouble_wrapper
  X(npy_int32, npy_clongdouble_wrapper)                 \  // 展开宏 X，传入 npy_int32 和 npy_clongdouble_wrapper
  X(npy_int64, npy_bool_wrapper)                        \  // 展开宏 X，传入 npy_int64 和 npy_bool_wrapper
  X(npy_int64, npy_byte)                                \  // 展开宏 X，传入 npy_int64 和 npy_byte
  X(npy_int64, npy_ubyte)                               \  // 展开宏 X，传入 npy_int64 和 npy_ubyte
  X(npy_int64, npy_short)                               \  // 展开宏 X，传入 npy_int64 和 npy_short
  X(npy_int64, npy_ushort)                              \  // 展开宏 X，传入 npy_int64 和 npy_ushort
  X(npy_int64, npy_int)                                 \  // 展开宏 X，传入 npy_int64 和 npy_int
  X(npy_int64, npy_uint)                                \  // 展开宏 X，传入 npy_int64 和 npy_uint
  X(npy_int64, npy_long)                                \  // 展开宏 X，传入 npy_int64 和 npy_long
  X(npy_int64, npy_ulong)                               \  // 展开宏 X，传入 npy_int64 和 npy_ulong
  X(npy_int64, npy_longlong)                            \  // 展开宏 X，传入 npy_int64 和 npy_longlong
  X(npy_int64, npy_ulonglong)                           \  // 展开宏 X，传入 npy_int64 和 npy_ulonglong
  X(npy_int64, npy_float)                               \  // 展开宏 X，传入 npy_int64 和 npy_float
  X(npy_int64, npy_double)                              \  // 展开宏 X，传入 npy_int64 和 npy_double
  X(npy_int64, npy_longdouble)                          \  // 展开宏 X，传入 npy_int64 和 npy_longdouble
  X(npy_int64, npy_cfloat_wrapper)                      \  // 展开宏 X，传入 npy_int64 和 npy_cfloat_wrapper
  X(npy_int64, npy_cdouble_wrapper)                     \  // 展开宏 X，传入 npy_int64 和 npy_cdouble_wrapper
  X(npy_int64, npy_clongdouble_wrapper)                 // 展开宏 X，传入 npy_int64 和 npy_clongdouble_wrapper

#endif  // vvv npy_longdouble is npy_double vvv
#define SPTOOLS_FOR_EACH_DATA_TYPE(X)           \
  X(npy_bool_wrapper)                           \  // 定义宏 SPTOOLS_FOR_EACH_DATA_TYPE，用于展开每种数据类型
  X(npy_byte)                                   \  // 展开 npy_byte 数据类型
  X(npy_ubyte)                                  \  // 展开 npy_ubyte 数据类型
  X(npy_short)                                  \  // 展开 npy_short 数据类型
  X(npy_ushort)                                 \  // 展开 npy_ushort 数据类型
  X(npy_int)                                    \  // 展开 npy_int 数据类型
  X(npy_uint)                                   \  // 展开 npy_uint 数据类型
  X(npy_long)                                   \  // 展开 npy_long 数据类型
  X(npy_ulong)                                  \  // 展开 npy_ulong 数据类型
  X(npy_longlong)                               \  // 展开 npy_longlong 数据类型
  X(npy_ulonglong)                              \  // 展开 npy_ulonglong 数据类型
  X(npy_float)                                  \  // 展开 npy_float 数据类型
  X(npy_double)                                 \  // 展开 npy_double 数据类型
  X(npy_cfloat_wrapper)                         \  // 展开 npy_cfloat_wrapper 数据类型
  X(npy_cdouble_wrapper)                        \  // 展开 npy_cdouble_wrapper 数据类型
  X(npy_clongdouble_wrapper)                    // 展开 npy_clongdouble_wrapper 数据类型

#define SPTOOLS_FOR_EACH_INDEX_DATA_TYPE_COMBINATION(X) \  // 定义宏 SPTOOLS_FOR_EACH_INDEX_DATA_TYPE_COMBINATION，用于展开每种索引和数据类型组合
  X(npy_int32, npy_bool_wrapper)                        \  // 展开 (npy_int32, npy_bool_wrapper) 组合
  X(npy_int32, npy_byte)                                \  // 展开 (npy_int32, npy_byte) 组合
  X(npy_int32, npy_ubyte)                               \  // 展开 (npy_int32, npy_ubyte) 组合
  X(npy_int32, npy_short)                               \  // 展开 (npy_int32, npy_short) 组合
  X(npy_int32, npy_ushort)                              \  // 展开 (npy_int32, npy_ushort) 组合
  X(npy_int32, npy_int)                                 \  // 展开 (npy_int32, npy_int) 组合
  X(npy_int32, npy_uint)                                \  // 展开 (npy_int32, npy_uint) 组合
  X(npy_int32, npy_long)                                \  // 展开 (npy_int32, npy_long) 组合
  X(npy_int32, npy_ulong)                               \  // 展开 (npy_int32, npy_ulong) 组合
  X(npy_int32, npy_longlong)                            \  // 展开 (npy_int32, npy_longlong) 组合
  X(npy_int32, npy_ulonglong)                           \  // 展开 (npy_int32, npy_ulonglong) 组合
  X(npy_int32, npy_float)                               \  // 展开 (npy_int32, npy_float) 组合
  X(npy_int32, npy_double)                              \  // 展开 (npy_int32, npy_double) 组合
  X(npy_int32, npy_cfloat_wrapper)                      \  // 展开 (npy_int32, npy_cfloat_wrapper) 组合
  X(npy_int32, npy_cdouble_wrapper)                     \  // 展开 (npy_int32, npy_cdouble_wrapper) 组合
  X(npy_int32, npy_clongdouble_wrapper)                 \  // 展开 (npy_int32, npy_clongdouble_wrapper) 组合
  X(npy_int64, npy_bool_wrapper)                        \  // 展开 (npy_int64, npy_bool_wrapper) 组合
  X(npy_int64, npy_byte)                                \  // 展开 (npy_int64, npy_byte) 组合
  X(npy_int64, npy_ubyte)                               \  // 展开 (npy_int64, npy_ubyte) 组合
  X(npy_int64, npy_short)                               \  // 展开 (npy_int64, npy_short) 组合
  X(npy_int64, npy_ushort)                              \  // 展开 (npy_int64, npy_ushort) 组合
  X(npy_int64, npy_int)                                 \  // 展开 (npy_int64, npy_int) 组合
  X(npy_int64, npy_uint)                                \  // 展开 (npy_int64, npy_uint) 组合
  X(npy_int64, npy_long)                                \  // 展开 (npy_int64, npy_long) 组合
  X(npy_int64, npy_ulong)                               \  // 展开 (npy_int64, npy_ulong) 组合
  X(npy_int64, npy_longlong)                            \  // 展开 (npy_int64, npy_longlong) 组合
  X(npy_int64, npy_ulonglong)                           \  // 展开 (npy_int64, npy_ulonglong) 组合
  X(npy_int64, npy_float)                               \  // 展开 (npy_int64, npy_float) 组合
  X(npy_int64, npy_double)                              \  // 展开 (npy_int64, npy_double) 组合
  X(npy_int64, npy_cfloat_wrapper)                      \  // 展开 (npy_int64, npy_cfloat_wrapper) 组合
  X(npy_int64, npy_cdouble_wrapper)                     \  // 展开 (npy_int64, npy_cdouble_wrapper) 组合
  X(npy_int64, npy_clongdouble_wrapper)                 // 展开 (npy_int64, npy_clongdouble_wrapper) 组合

#endif  // 结束宏定义 SPTOOLS_FOR_EACH_INDEX_DATA_TYPE_COMBINATION

#endif  // 结束宏定义 SPTOOLS_FOR_EACH_DATA_TYPE
```