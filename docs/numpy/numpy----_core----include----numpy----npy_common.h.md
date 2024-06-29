# `.\numpy\numpy\_core\include\numpy\npy_common.h`

```py
#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_COMMON_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_COMMON_H_

/* 需要 Python.h 以使用 npy_intp、npy_uintp */
#include <Python.h>

/* numpyconfig.h 是自动生成的配置文件 */
#include "numpyconfig.h"
#ifdef HAVE_NPY_CONFIG_H
#include <npy_config.h>
#endif

/*
 * 在定义 npy_math 函数时使用 static inline 修饰符，
 * 允许编译器在可能时进行优化
 */
#ifndef NPY_INLINE_MATH
#if defined(NPY_INTERNAL_BUILD) && NPY_INTERNAL_BUILD
    #define NPY_INLINE_MATH 1
#else
    #define NPY_INLINE_MATH 0
#endif
#endif

/*
 * gcc 即使在 -O3 下也不会展开循环
 * 谨慎使用，现代 CPU 很少因此而加速
 */
#ifdef HAVE_ATTRIBUTE_OPTIMIZE_UNROLL_LOOPS
#define NPY_GCC_UNROLL_LOOPS \
    __attribute__((optimize("unroll-loops")))
#else
#define NPY_GCC_UNROLL_LOOPS
#endif

/* 最高的 gcc 优化级别，启用自动向量化 */
#ifdef HAVE_ATTRIBUTE_OPTIMIZE_OPT_3
#define NPY_GCC_OPT_3 __attribute__((optimize("O3")))
#else
#define NPY_GCC_OPT_3
#endif

/*
 * 标记一个参数（从1开始）不能为空且不会被检查
 * 如果函数检查为空，不要使用！！编译器将移除检查
 */
#ifdef HAVE_ATTRIBUTE_NONNULL
#define NPY_GCC_NONNULL(n) __attribute__((nonnull(n)))
#else
#define NPY_GCC_NONNULL(n)
#endif

/*
 * 给编译器一个提示，哪个分支更可能或更不可能发生，
 * 例如，罕见的错误情况：
 *
 * if (NPY_UNLIKELY(failure == 0))
 *    return NULL;
 *
 * 双重 !! 是为了将表达式（如 NULL）转换为布尔值，符合内置函数的需求
 */
#ifdef HAVE___BUILTIN_EXPECT
#define NPY_LIKELY(x) __builtin_expect(!!(x), 1)
#define NPY_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define NPY_LIKELY(x) (x)
#define NPY_UNLIKELY(x) (x)
#endif

#ifdef HAVE___BUILTIN_PREFETCH
/* 与 _mm_prefetch 不同，也适用于非 x86 平台 */
#define NPY_PREFETCH(x, rw, loc) __builtin_prefetch((x), (rw), (loc))
#else
#ifdef NPY_HAVE_SSE
/* _MM_HINT_ET[01] (rw = 1) 不支持，仅在 gcc >= 4.9 中可用 */
#define NPY_PREFETCH(x, rw, loc) _mm_prefetch((x), loc == 0 ? _MM_HINT_NTA : \
                                             (loc == 1 ? _MM_HINT_T2 : \
                                              (loc == 2 ? _MM_HINT_T1 : \
                                               (loc == 3 ? _MM_HINT_T0 : -1))))
#else
#define NPY_PREFETCH(x, rw,loc)
#endif
#endif

/* `NPY_INLINE` 为向后兼容性保留；应改用 `inline` */
#if defined(_MSC_VER) && !defined(__clang__)
    #define NPY_INLINE __inline
/* 用于处理 Windows 上的 clang-cl */
#elif defined(__GNUC__) || defined(__clang__)
    #if defined(__STRICT_ANSI__)
         #define NPY_INLINE __inline__
    #else
         #define NPY_INLINE inline
    #endif
#else
    #define NPY_INLINE
#endif

#ifdef _MSC_VER
    #define NPY_FINLINE static __forceinline
#elif defined(__GNUC__)
    #define NPY_FINLINE static inline __attribute__((always_inline))
#else
    // 定义一个宏 NPY_FINLINE，它展开为 static
#ifdef _MSC_VER
    // 如果编译器是 Microsoft Visual C++，定义为静态函数，禁止内联优化，使用__declspec(noinline)修饰
    #define NPY_NOINLINE static __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
    // 如果编译器是 GCC 或 Clang，定义为静态函数，禁止内联优化，使用__attribute__((noinline))修饰
    #define NPY_NOINLINE static __attribute__((noinline))
#else
    // 其他情况下，仅定义为静态函数
    #define NPY_NOINLINE static
#endif

#ifdef __cplusplus
    // 如果是 C++ 编译环境，定义为线程局部存储（thread_local）
    #define NPY_TLS thread_local
#elif defined(HAVE_THREAD_LOCAL)
    // 如果支持标准的线程局部存储，也定义为线程局部存储
    #define NPY_TLS thread_local
#elif defined(HAVE__THREAD_LOCAL)
    // 如果使用 _Thread_local 关键字支持线程局部存储
    #define NPY_TLS _Thread_local
#elif defined(HAVE___THREAD)
    // 如果支持 __thread 关键字的线程局部存储
    #define NPY_TLS __thread
#elif defined(HAVE___DECLSPEC_THREAD_)
    // 如果支持 __declspec(thread) 关键字的线程局部存储
    #define NPY_TLS __declspec(thread)
#else
    // 否则不进行任何特殊的线程局部存储定义
    #define NPY_TLS
#endif

#ifdef WITH_CPYCHECKER_RETURNS_BORROWED_REF_ATTRIBUTE
    // 如果定义了特定的返回借用引用的属性，使用 cpychecker_returns_borrowed_ref 修饰
  #define NPY_RETURNS_BORROWED_REF \
    __attribute__((cpychecker_returns_borrowed_ref))
#else
    // 否则不使用任何属性修饰
 #define NPY_RETURNS_BORROWED_REF
#endif

#ifdef WITH_CPYCHECKER_STEALS_REFERENCE_TO_ARG_ATTRIBUTE
    // 如果定义了特定的参数窃取引用的属性，使用 cpychecker_steals_reference_to_arg(n) 修饰
  #define NPY_STEALS_REF_TO_ARG(n) \
   __attribute__((cpychecker_steals_reference_to_arg(n)))
#else
    // 否则不使用任何属性修饰
 #define NPY_STEALS_REF_TO_ARG(n)
#endif

/* 64 bit file position support, also on win-amd64. Issue gh-2256 */
#if defined(_MSC_VER) && defined(_WIN64) && (_MSC_VER > 1400) || \
    defined(__MINGW32__) || defined(__MINGW64__)
    // 如果是在 Microsoft Visual C++ 编译环境下，并且是 64 位 Windows，或者在 MinGW 环境下
    #include <io.h>

    // 定义文件操作为 64 位版本的函数
    #define npy_fseek _fseeki64
    #define npy_ftell _ftelli64
    #define npy_lseek _lseeki64
    // 定义文件偏移量类型为 npy_int64
    #define npy_off_t npy_int64

    // 根据整数类型大小定义 Python 格式化字符串
    #if NPY_SIZEOF_INT == 8
        #define NPY_OFF_T_PYFMT "i"
    #elif NPY_SIZEOF_LONG == 8
        #define NPY_OFF_T_PYFMT "l"
    #elif NPY_SIZEOF_LONGLONG == 8
        #define NPY_OFF_T_PYFMT "L"
    #else
        #error Unsupported size for type off_t
    #endif
#else
#ifdef HAVE_FSEEKO
    // 如果支持 fseeko 函数，定义为 npy_fseek
    #define npy_fseek fseeko
#else
    // 否则定义为标准的 fseek 函数
    #define npy_fseek fseek
#endif
#ifdef HAVE_FTELLO
    // 如果支持 ftello 函数，定义为 npy_ftell
    #define npy_ftell ftello
#else
    // 否则定义为标准的 ftell 函数
    #define npy_ftell ftell
#endif
    // 包含系统类型头文件
    #include <sys/types.h>
    // 非 Windows 环境下包含unistd.h头文件
    #ifndef _WIN32
        #include <unistd.h>
    #endif
    // 定义文件偏移量类型为 off_t
    #define npy_lseek lseek
    #define npy_off_t off_t

    // 根据不同大小的 off_t 定义 Python 格式化字符串
    #if NPY_SIZEOF_OFF_T == NPY_SIZEOF_SHORT
        #define NPY_OFF_T_PYFMT "h"
    #elif NPY_SIZEOF_OFF_T == NPY_SIZEOF_INT
        #define NPY_OFF_T_PYFMT "i"
    #elif NPY_SIZEOF_OFF_T == NPY_SIZEOF_LONG
        #define NPY_OFF_T_PYFMT "l"
    #elif NPY_SIZEOF_OFF_T == NPY_SIZEOF_LONGLONG
        #define NPY_OFF_T_PYFMT "L"
    #else
        #error Unsupported size for type off_t
    #endif
#endif

/* enums for detected endianness */
// 枚举检测到的字节序类型
enum {
        // 未知字节序
        NPY_CPU_UNKNOWN_ENDIAN,
        // 小端字节序
        NPY_CPU_LITTLE,
        // 大端字节序
        NPY_CPU_BIG
};

/*
 * This is to typedef npy_intp to the appropriate size for Py_ssize_t.
 * (Before NumPy 2.0 we used Py_intptr_t and Py_uintptr_t from `pyport.h`.)
 */
// 将 npy_intp 定义为适合 Py_ssize_t 大小的类型
typedef Py_ssize_t npy_intp;
// 定义 npy_uintp 为 size_t 类型
typedef size_t npy_uintp;

/*
 * Define sizes that were not defined in numpyconfig.h.
 */
// 定义在 numpyconfig.h 中未定义的大小
#define NPY_SIZEOF_CHAR 1
#define NPY_SIZEOF_BYTE 1
#define NPY_SIZEOF_DATETIME 8
#define NPY_SIZEOF_TIMEDELTA 8
#define NPY_SIZEOF_HALF 2
#define NPY_SIZEOF_CFLOAT NPY_SIZEOF_COMPLEX_FLOAT
#define NPY_SIZEOF_CDOUBLE NPY_SIZEOF_COMPLEX_DOUBLE
/* 定义 NPY_SIZEOF_CLONGDOUBLE 与 NPY_SIZEOF_COMPLEX_LONGDOUBLE 相等 */
#define NPY_SIZEOF_CLONGDOUBLE NPY_SIZEOF_COMPLEX_LONGDOUBLE

/* 如果 constchar 已定义，则取消定义 */
#ifdef constchar
#undef constchar
#endif

/* 定义 NPY_SSIZE_T_PYFMT 为 "n" */
#define NPY_SSIZE_T_PYFMT "n"

/* 将 constchar 定义为 char */
#define constchar char

/* 
 * NPY_INTP_FMT 注意：
 *      不同于其他的 NPY_*_FMT 宏，这些宏用于 PyOS_snprintf，
 *      NPY_INTP_FMT 用于 PyErr_Format 和 PyUnicode_FromFormat。
 *      这些函数使用不同的格式代码，根据 Python 文档进行了便携式指定。
 *      参见 issue gh-2388。
 */
#if NPY_SIZEOF_INTP == NPY_SIZEOF_LONG
        #define NPY_INTP NPY_LONG
        #define NPY_UINTP NPY_ULONG
        #define PyIntpArrType_Type PyLongArrType_Type
        #define PyUIntpArrType_Type PyULongArrType_Type
        #define NPY_MAX_INTP NPY_MAX_LONG
        #define NPY_MIN_INTP NPY_MIN_LONG
        #define NPY_MAX_UINTP NPY_MAX_ULONG
        #define NPY_INTP_FMT "ld"
#elif NPY_SIZEOF_INTP == NPY_SIZEOF_INT
        #define NPY_INTP NPY_INT
        #define NPY_UINTP NPY_UINT
        #define PyIntpArrType_Type PyIntArrType_Type
        #define PyUIntpArrType_Type PyUIntArrType_Type
        #define NPY_MAX_INTP NPY_MAX_INT
        #define NPY_MIN_INTP NPY_MIN_INT
        #define NPY_MAX_UINTP NPY_MAX_UINT
        #define NPY_INTP_FMT "d"
#elif defined(PY_LONG_LONG) && (NPY_SIZEOF_INTP == NPY_SIZEOF_LONGLONG)
        #define NPY_INTP NPY_LONGLONG
        #define NPY_UINTP NPY_ULONGLONG
        #define PyIntpArrType_Type PyLongLongArrType_Type
        #define PyUIntpArrType_Type PyULongLongArrType_Type
        #define NPY_MAX_INTP NPY_MAX_LONGLONG
        #define NPY_MIN_INTP NPY_MIN_LONGLONG
        #define NPY_MAX_UINTP NPY_MAX_ULONGLONG
        #define NPY_INTP_FMT "lld"
#else
    #error "Failed to correctly define NPY_INTP and NPY_UINTP"
#endif

/*
 * 一些平台未定义 bool、long long 或 long double。
 * 在这里进行处理。
 */
#define NPY_BYTE_FMT "hhd"
#define NPY_UBYTE_FMT "hhu"
#define NPY_SHORT_FMT "hd"
#define NPY_USHORT_FMT "hu"
#define NPY_INT_FMT "d"
#define NPY_UINT_FMT "u"
#define NPY_LONG_FMT "ld"
#define NPY_ULONG_FMT "lu"
#define NPY_HALF_FMT "g"
#define NPY_FLOAT_FMT "g"
#define NPY_DOUBLE_FMT "g"

#ifdef PY_LONG_LONG
typedef PY_LONG_LONG npy_longlong;
typedef unsigned PY_LONG_LONG npy_ulonglong;
#  ifdef _MSC_VER
#    define NPY_LONGLONG_FMT         "I64d"
#    define NPY_ULONGLONG_FMT        "I64u"
#  else
#    define NPY_LONGLONG_FMT         "lld"
#    define NPY_ULONGLONG_FMT        "llu"
#  endif
#  ifdef _MSC_VER
#    define NPY_LONGLONG_SUFFIX(x)   (x##i64)
#    define NPY_ULONGLONG_SUFFIX(x)  (x##Ui64)
#  else
#    define NPY_LONGLONG_SUFFIX(x)   (x##LL)
#    define NPY_ULONGLONG_SUFFIX(x)  (x##ULL)
#  endif
#else
typedef long npy_longlong;
typedef unsigned long npy_ulonglong;
#  define NPY_LONGLONG_SUFFIX(x)  (x##L)
#  define NPY_ULONGLONG_SUFFIX(x) (x##UL)
#endif

/* 定义 npy_bool 为 unsigned char，以及 NPY_FALSE 和 NPY_TRUE 的值 */
typedef unsigned char npy_bool;
#define NPY_FALSE 0
#define NPY_TRUE 1
/*
 * `NPY_SIZEOF_LONGDOUBLE` isn't usually equal to sizeof(long double).
 * In some certain cases, it may forced to be equal to sizeof(double)
 * even against the compiler implementation and the same goes for
 * `complex long double`.
 *
 * Therefore, avoid `long double`, use `npy_longdouble` instead,
 * and when it comes to standard math functions make sure of using
 * the double version when `NPY_SIZEOF_LONGDOUBLE` == `NPY_SIZEOF_DOUBLE`.
 * For example:
 *   npy_longdouble *ptr, x;
 *   #if NPY_SIZEOF_LONGDOUBLE == NPY_SIZEOF_DOUBLE
 *       npy_longdouble r = modf(x, ptr);
 *   #else
 *       npy_longdouble r = modfl(x, ptr);
 *   #endif
 *
 * See https://github.com/numpy/numpy/issues/20348
 */
#if NPY_SIZEOF_LONGDOUBLE == NPY_SIZEOF_DOUBLE
    #define NPY_LONGDOUBLE_FMT "g"
    #define longdouble_t double
    typedef double npy_longdouble;
#else
    #define NPY_LONGDOUBLE_FMT "Lg"
    #define longdouble_t long double
    typedef long double npy_longdouble;
#endif

#ifndef Py_USING_UNICODE
    #error Must use Python with unicode enabled.
#endif


typedef signed char npy_byte;        // 有符号字符
typedef unsigned char npy_ubyte;     // 无符号字符
typedef unsigned short npy_ushort;   // 无符号短整型
typedef unsigned int npy_uint;       // 无符号整型
typedef unsigned long npy_ulong;     // 无符号长整型

/* These are for completeness */
typedef char npy_char;               // 字符
typedef short npy_short;             // 短整型
typedef int npy_int;                 // 整型
typedef long npy_long;               // 长整型
typedef float npy_float;             // 单精度浮点型
typedef double npy_double;           // 双精度浮点型

typedef Py_hash_t npy_hash_t;       // Python 哈希类型
#define NPY_SIZEOF_HASH_T NPY_SIZEOF_INTP  // 哈希类型的大小

#if defined(__cplusplus)

typedef struct
{
    double _Val[2];                 // 双精度复数结构体
} npy_cdouble;

typedef struct
{
    float _Val[2];                  // 单精度复数结构体
} npy_cfloat;

typedef struct
{
    long double _Val[2];            // 长双精度复数结构体
} npy_clongdouble;

#else

#include <complex.h>

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
typedef _Dcomplex npy_cdouble;      // Microsoft Visual C++ 的双精度复数类型
typedef _Fcomplex npy_cfloat;       // Microsoft Visual C++ 的单精度复数类型
typedef _Lcomplex npy_clongdouble;  // Microsoft Visual C++ 的长双精度复数类型
#else /* !defined(_MSC_VER) || defined(__INTEL_COMPILER) */
typedef double _Complex npy_cdouble;        // 标准复数类型的双精度
typedef float _Complex npy_cfloat;          // 标准复数类型的单精度
typedef longdouble_t _Complex npy_clongdouble;   // 标准复数类型的长双精度
#endif

#endif

/*
 * numarray-style bit-width typedefs
 */
#define NPY_MAX_INT8 127                    // 有符号 8 位整型最大值
#define NPY_MIN_INT8 -128                   // 有符号 8 位整型最小值
#define NPY_MAX_UINT8 255                   // 无符号 8 位整型最大值
#define NPY_MAX_INT16 32767                 // 有符号 16 位整型最大值
#define NPY_MIN_INT16 -32768                // 有符号 16 位整型最小值
#define NPY_MAX_UINT16 65535                // 无符号 16 位整型最大值
#define NPY_MAX_INT32 2147483647            // 有符号 32 位整型最大值
#define NPY_MIN_INT32 (-NPY_MAX_INT32 - 1)  // 有符号 32 位整型最小值
#define NPY_MAX_UINT32 4294967295U          // 无符号 32 位整型最大值
#define NPY_MAX_INT64 NPY_LONGLONG_SUFFIX(9223372036854775807)     // 有符号 64 位整型最大值
#define NPY_MIN_INT64 (-NPY_MAX_INT64 - NPY_LONGLONG_SUFFIX(1))    // 有符号 64 位整型最小值
#define NPY_MAX_UINT64 NPY_ULONGLONG_SUFFIX(18446744073709551615)  // 无符号 64 位整型最大值
#define NPY_MAX_INT128 NPY_LONGLONG_SUFFIX(85070591730234615865843651857942052864)     // 有符号 128 位整型最大值
#define NPY_MIN_INT128 (-NPY_MAX_INT128 - NPY_LONGLONG_SUFFIX(1))    // 有符号 128 位整型最小值
#define NPY_MAX_UINT128 NPY_ULONGLONG_SUFFIX(170141183460469231731687303715884105728)  // 无符号 128 位整型最大值
#define NPY_MAX_INT256 NPY_LONGLONG_SUFFIX(57896044618658097711785492504343953926634992332820282019728792003956564819967)     // 有符号 256 位整型最大值
#define NPY_MIN_INT256 (-NPY_MAX_INT256 - NPY_LONGLONG_SUFFIX(1))    // 有符号 256 位整型最小值
/* 定义最大的 256 位无符号整数，即 2^256 - 1 */
#define NPY_MAX_UINT256 NPY_ULONGLONG_SUFFIX(115792089237316195423570985008687907853269984665640564039457584007913129639935)
/* 定义最小日期时间和最大日期时间为 int64_t 类型的最小和最大值 */
#define NPY_MIN_DATETIME NPY_MIN_INT64
#define NPY_MAX_DATETIME NPY_MAX_INT64
/* 定义最小时间差和最大时间差为 int64_t 类型的最小和最大值 */
#define NPY_MIN_TIMEDELTA NPY_MIN_INT64
#define NPY_MAX_TIMEDELTA NPY_MAX_INT64

        /* 需要根据每种类型找到位数，并相应地进行定义。

           C 语言规定 sizeof(char) == 1

           因此，仅使用 sizeof 关键字并不会有帮助。

           看起来 Python 自身也经常使用 sizeof(char)，这应该始终为 1。

           思路：利用 CHAR_BIT，它应该告诉我们每个字符的位数。
        */

        /* 包含平台定义 -- 这些在 C89/90 标准中 */
#include <limits.h>
/* 定义最大有符号字节和最小有符号字节 */
#define NPY_MAX_BYTE SCHAR_MAX
#define NPY_MIN_BYTE SCHAR_MIN
/* 定义最大无符号字节 */
#define NPY_MAX_UBYTE UCHAR_MAX
/* 定义最大有符号短整数和最小有符号短整数 */
#define NPY_MAX_SHORT SHRT_MAX
#define NPY_MIN_SHORT SHRT_MIN
/* 定义最大无符号短整数 */
#define NPY_MAX_USHORT USHRT_MAX
/* 定义最大有符号整数 */
#define NPY_MAX_INT   INT_MAX
#ifndef INT_MIN
#define INT_MIN (-INT_MAX - 1)
#endif
/* 定义最小有符号整数 */
#define NPY_MIN_INT   INT_MIN
/* 定义最大无符号整数 */
#define NPY_MAX_UINT  UINT_MAX
/* 定义最大有符号长整数和最小有符号长整数 */
#define NPY_MAX_LONG  LONG_MAX
#define NPY_MIN_LONG  LONG_MIN
/* 定义最大无符号长整数 */
#define NPY_MAX_ULONG  ULONG_MAX

/* 定义布尔类型的位数 */
#define NPY_BITSOF_BOOL (sizeof(npy_bool) * CHAR_BIT)
/* 定义字符类型的位数 */
#define NPY_BITSOF_CHAR CHAR_BIT
/* 定义字节类型的位数 */
#define NPY_BITSOF_BYTE (NPY_SIZEOF_BYTE * CHAR_BIT)
/* 定义短整数类型的位数 */
#define NPY_BITSOF_SHORT (NPY_SIZEOF_SHORT * CHAR_BIT)
/* 定义整数类型的位数 */
#define NPY_BITSOF_INT (NPY_SIZEOF_INT * CHAR_BIT)
/* 定义长整数类型的位数 */
#define NPY_BITSOF_LONG (NPY_SIZEOF_LONG * CHAR_BIT)
/* 定义长长整数类型的位数 */
#define NPY_BITSOF_LONGLONG (NPY_SIZEOF_LONGLONG * CHAR_BIT)
/* 定义整型指针类型的位数 */
#define NPY_BITSOF_INTP (NPY_SIZEOF_INTP * CHAR_BIT)
/* 定义半精度浮点数类型的位数 */
#define NPY_BITSOF_HALF (NPY_SIZEOF_HALF * CHAR_BIT)
/* 定义单精度浮点数类型的位数 */
#define NPY_BITSOF_FLOAT (NPY_SIZEOF_FLOAT * CHAR_BIT)
/* 定义双精度浮点数类型的位数 */
#define NPY_BITSOF_DOUBLE (NPY_SIZEOF_DOUBLE * CHAR_BIT)
/* 定义长双精度浮点数类型的位数 */
#define NPY_BITSOF_LONGDOUBLE (NPY_SIZEOF_LONGDOUBLE * CHAR_BIT)
/* 定义复数浮点数类型的位数 */
#define NPY_BITSOF_CFLOAT (NPY_SIZEOF_CFLOAT * CHAR_BIT)
/* 定义复数双精度浮点数类型的位数 */
#define NPY_BITSOF_CDOUBLE (NPY_SIZEOF_CDOUBLE * CHAR_BIT)
/* 定义复数长双精度浮点数类型的位数 */
#define NPY_BITSOF_CLONGDOUBLE (NPY_SIZEOF_CLONGDOUBLE * CHAR_BIT)
/* 定义日期时间类型的位数 */
#define NPY_BITSOF_DATETIME (NPY_SIZEOF_DATETIME * CHAR_BIT)
/* 定义时间差类型的位数 */
#define NPY_BITSOF_TIMEDELTA (NPY_SIZEOF_TIMEDELTA * CHAR_BIT)

/* 根据长整数类型的位数选择相应的整数类型 */
#if NPY_BITSOF_LONG == 8
#define NPY_INT8 NPY_LONG
#define NPY_UINT8 NPY_ULONG
        typedef long npy_int8;
        typedef unsigned long npy_uint8;
#define PyInt8ScalarObject PyLongScalarObject
#define PyInt8ArrType_Type PyLongArrType_Type
#define PyUInt8ScalarObject PyULongScalarObject
#define PyUInt8ArrType_Type PyULongArrType_Type
#define NPY_INT8_FMT NPY_LONG_FMT
#define NPY_UINT8_FMT NPY_ULONG_FMT
#elif NPY_BITSOF_LONG == 16
#define NPY_INT16 NPY_LONG
#define NPY_UINT16 NPY_ULONG
        typedef long npy_int16;
        typedef unsigned long npy_uint16;
#define PyInt16ScalarObject PyLongScalarObject
#define PyInt16ArrType_Type PyLongArrType_Type
#define PyUInt16ScalarObject PyULongScalarObject
#define PyUInt16ArrType_Type PyULongArrType_Type
# 如果 NPY_BITSOF_LONG 等于 16，定义整型和无符号整型为长长整型和无符号长长整型
# 否则，如果 NPY_BITSOF_LONG 等于 32，定义整型和无符号整型为长整型和无符号长整型
# 否则，如果 NPY_BITSOF_LONG 等于 64，定义整型和无符号整型为长整型和无符号长整型，并定义特定的 PyLong 相关对象和类型
# 否则，如果 NPY_BITSOF_LONG 等于 128，定义整型和无符号整型为长整型和无符号长整型，并定义特定的 PyLong 相关对象和类型

#ifdef NPY_BITSOF_LONGLONG
# 如果 NPY_BITSOF_LONGLONG 等于 8，定义整型和无符号整型为长长整型和无符号长长整型，并定义特定的 PyLong 相关对象和类型
# 否则，如果 NPY_BITSOF_LONGLONG 等于 16，定义整型和无符号整型为长长整型和无符号长长整型，并定义特定的 PyLong 相关对象和类型
# 否则，如果 NPY_BITSOF_LONGLONG 等于 32，定义整型和无符号整型为长整型和无符号长整型，并定义特定的 PyLong 相关对象和类型
#    define PyInt32ScalarObject PyLongLongScalarObject
#    define PyInt32ArrType_Type PyLongLongArrType_Type
#    define PyUInt32ScalarObject PyULongLongScalarObject
#    define PyUInt32ArrType_Type PyULongLongArrType_Type
# 定义宏，将32位整数类型的标量对象和数组类型重命名为64位整数类型的标量对象和数组类型

#define NPY_INT32_FMT NPY_LONGLONG_FMT
#define NPY_UINT32_FMT NPY_ULONGLONG_FMT
# 定义宏，将32位整数的格式字符串重命名为64位整数的格式字符串

#  endif
#  define NPY_MAX_LONGLONG NPY_MAX_INT32
#  define NPY_MIN_LONGLONG NPY_MIN_INT32
#  define NPY_MAX_ULONGLONG NPY_MAX_UINT32
# 如果定义了64位整数，定义64位整数的最大和最小值以及64位无符号整数的最大值

#elif NPY_BITSOF_LONGLONG == 64
#  ifndef NPY_INT64
#    define NPY_INT64 NPY_LONGLONG
#    define NPY_UINT64 NPY_ULONGLONG
        typedef npy_longlong npy_int64;
        typedef npy_ulonglong npy_uint64;
#    define PyInt64ScalarObject PyLongLongScalarObject
#    define PyInt64ArrType_Type PyLongLongArrType_Type
#    define PyUInt64ScalarObject PyULongLongScalarObject
#    define PyUInt64ArrType_Type PyULongLongArrType_Type
#define NPY_INT64_FMT NPY_LONGLONG_FMT
#define NPY_UINT64_FMT NPY_ULONGLONG_FMT
#    define MyPyLong_FromInt64 PyLong_FromLongLong
#    define MyPyLong_AsInt64 PyLong_AsLongLong
# 定义64位整数类型和无符号整数类型，并重命名相应的标量对象和数组类型
# 定义64位整数类型和无符号整数类型的格式字符串
# 定义将Python长整型转换为64位整数和从64位整数获取Python长整型的宏

#  endif
#  define NPY_MAX_LONGLONG NPY_MAX_INT64
#  define NPY_MIN_LONGLONG NPY_MIN_INT64
#  define NPY_MAX_ULONGLONG NPY_MAX_UINT64
# 如果定义了128位整数，定义128位整数的最大和最小值以及128位无符号整数的最大值

#elif NPY_BITSOF_LONGLONG == 128
#  ifndef NPY_INT128
#    define NPY_INT128 NPY_LONGLONG
#    define NPY_UINT128 NPY_ULONGLONG
        typedef npy_longlong npy_int128;
        typedef npy_ulonglong npy_uint128;
#    define PyInt128ScalarObject PyLongLongScalarObject
#    define PyInt128ArrType_Type PyLongLongArrType_Type
#    define PyUInt128ScalarObject PyULongLongScalarObject
#    define PyUInt128ArrType_Type PyULongLongArrType_Type
#define NPY_INT128_FMT NPY_LONGLONG_FMT
#define NPY_UINT128_FMT NPY_ULONGLONG_FMT
#  endif
#  define NPY_MAX_LONGLONG NPY_MAX_INT128
#  define NPY_MIN_LONGLONG NPY_MIN_INT128
#  define NPY_MAX_ULONGLONG NPY_MAX_UINT128
# 如果定义了256位整数，定义256位整数的最大和最小值以及256位无符号整数的最大值

#elif NPY_BITSOF_LONGLONG == 256
#  define NPY_INT256 NPY_LONGLONG
#  define NPY_UINT256 NPY_ULONGLONG
        typedef npy_longlong npy_int256;
        typedef npy_ulonglong npy_uint256;
#  define PyInt256ScalarObject PyLongLongScalarObject
#  define PyInt256ArrType_Type PyLongLongArrType_Type
#  define PyUInt256ScalarObject PyULongLongScalarObject
#  define PyUInt256ArrType_Type PyULongLongArrType_Type
#define NPY_INT256_FMT NPY_LONGLONG_FMT
#define NPY_UINT256_FMT NPY_ULONGLONG_FMT
#  define NPY_MAX_LONGLONG NPY_MAX_INT256
#  define NPY_MIN_LONGLONG NPY_MIN_INT256
#  define NPY_MAX_ULONGLONG NPY_MAX_UINT256
# 如果定义了8位整数，定义8位整数的最大和最小值以及8位无符号整数的最大值

#endif

#if NPY_BITSOF_INT == 8
#ifndef NPY_INT8
#define NPY_INT8 NPY_INT
#define NPY_UINT8 NPY_UINT
        typedef int npy_int8;
        typedef unsigned int npy_uint8;
#    define PyInt8ScalarObject PyIntScalarObject
#    define PyInt8ArrType_Type PyIntArrType_Type
#    define PyUInt8ScalarObject PyUIntScalarObject
#    define PyUInt8ArrType_Type PyUIntArrType_Type
#define NPY_INT8_FMT NPY_INT_FMT
#define NPY_UINT8_FMT NPY_UINT_FMT
#endif
# 如果定义了16位整数，定义16位整数的最大和最小值以及16位无符号整数的最大值

#elif NPY_BITSOF_INT == 16
#ifndef NPY_INT16
#define NPY_INT16 NPY_INT
#ifdef NPY_BITSOF_SHORT == 8
#ifndef NPY_INT8
// 如果平台的 short 类型占用 8 比特，且未定义 NPY_INT8，则定义 NPY_INT8 和 NPY_UINT8
#define NPY_INT8 NPY_SHORT
#define NPY_UINT8 NPY_USHORT
// 定义 npy_int8 和 npy_uint8 类型别名为 short 和 unsigned short
typedef short npy_int8;
typedef unsigned short npy_uint8;
// 定义 PyInt8ScalarObject 和 PyUInt8ScalarObject 宏别名为 PyShortScalarObject 和 PyUShortScalarObject
#    define PyInt8ScalarObject PyShortScalarObject
#    define PyInt8ArrType_Type PyShortArrType_Type
#    define PyUInt8ScalarObject PyUShortScalarObject
#    define PyUInt8ArrType_Type PyUShortArrType_Type
// 定义 NPY_INT8_FMT 和 NPY_UINT8_FMT 宏别名为 NPY_SHORT_FMT 和 NPY_USHORT_FMT
#define NPY_INT8_FMT NPY_SHORT_FMT
#define NPY_UINT8_FMT NPY_USHORT_FMT
#endif
#elif NPY_BITSOF_SHORT == 16
#ifndef NPY_INT16
// 如果平台的 short 类型占用 16 比特，且未定义 NPY_INT16，则定义 NPY_INT16 和 NPY_UINT16
#define NPY_INT16 NPY_SHORT
#define NPY_UINT16 NPY_USHORT
// 定义 npy_int16 和 npy_uint16 类型别名为 short 和 unsigned short
typedef short npy_int16;
typedef unsigned short npy_uint16;
// 定义 PyInt16ScalarObject 和 PyUInt16ScalarObject 宏别名为 PyShortScalarObject 和 PyUShortScalarObject
#    define PyInt16ScalarObject PyShortScalarObject
#    define PyInt16ArrType_Type PyShortArrType_Type
#    define PyUInt16ScalarObject PyUShortScalarObject
#    define PyUInt16ArrType_Type PyUShortArrType_Type
// 定义 NPY_INT16_FMT 和 NPY_UINT16_FMT 宏别名为 NPY_SHORT_FMT 和 NPY_USHORT_FMT
#define NPY_INT16_FMT NPY_SHORT_FMT
#define NPY_UINT16_FMT NPY_USHORT_FMT
#endif
#elif NPY_BITSOF_SHORT == 32
#ifndef NPY_INT32
// 如果平台的 short 类型占用 32 比特，且未定义 NPY_INT32，则定义 NPY_INT32
#define NPY_INT32 NPY_SHORT


这段代码根据平台中 short 类型的位数（8、16、32 比特）定义了不同的数据类型和相关宏别名，以及格式化字符串宏。
#ifdef NPY_BITSOF_SHORT == 32
// 如果 NPY_BITSOF_SHORT 等于 32，则定义以下内容
#ifndef NPY_INT32
// 如果 NPY_INT32 未定义，则定义 NPY_INT32 和 NPY_UINT32 为 NPY_BYTE
#define NPY_INT32 NPY_BYTE
#define NPY_UINT32 NPY_UBYTE
// 定义 npy_int32 和 npy_uint32 为 signed char 和 unsigned char 类型
typedef signed char npy_int32;
typedef unsigned char npy_uint32;
typedef unsigned char npy_ucs4;
// 将 PyInt32ScalarObject, PyInt32ArrType_Type, PyUInt32ScalarObject, PyUInt32ArrType_Type 分别定义为 PyByteScalarObject 和 PyByteArrType_Type
#define PyInt32ScalarObject PyByteScalarObject
#define PyInt32ArrType_Type PyByteArrType_Type
#define PyUInt32ScalarObject PyUByteScalarObject
#define PyUInt32ArrType_Type PyUByteArrType_Type
// 定义 NPY_INT32_FMT 和 NPY_UINT32_FMT 为 NPY_BYTE_FMT
#define NPY_INT32_FMT NPY_BYTE_FMT
#define NPY_UINT32_FMT NPY_UBYTE_FMT
#endif
#endif
#define NPY_INT64 NPY_BYTE
// 将 NPY_INT64 定义为 NPY_BYTE

#define NPY_UINT64 NPY_UBYTE
// 将 NPY_UINT64 定义为 NPY_UBYTE

typedef signed char npy_int64;
// 定义 npy_int64 为有符号字符类型

typedef unsigned char npy_uint64;
// 定义 npy_uint64 为无符号字符类型

#    define PyInt64ScalarObject PyByteScalarObject
// 将 PyInt64ScalarObject 定义为 PyByteScalarObject

#    define PyInt64ArrType_Type PyByteArrType_Type
// 将 PyInt64ArrType_Type 定义为 PyByteArrType_Type

#    define PyUInt64ScalarObject PyUByteScalarObject
// 将 PyUInt64ScalarObject 定义为 PyUByteScalarObject

#    define PyUInt64ArrType_Type PyUByteArrType_Type
// 将 PyUInt64ArrType_Type 定义为 PyUByteArrType_Type

#define NPY_INT64_FMT NPY_BYTE_FMT
// 将 NPY_INT64_FMT 定义为 NPY_BYTE_FMT

#define NPY_UINT64_FMT NPY_UBYTE_FMT
// 将 NPY_UINT64_FMT 定义为 NPY_UBYTE_FMT

#    define MyPyLong_FromInt64 PyLong_FromLong
// 将 MyPyLong_FromInt64 定义为 PyLong_FromLong

#    define MyPyLong_AsInt64 PyLong_AsLong
// 将 MyPyLong_AsInt64 定义为 PyLong_AsLong

#endif
#elif NPY_BITSOF_CHAR == 128
#ifndef NPY_INT128
#define NPY_INT128 NPY_BYTE
// 如果 NPY_INT128 未定义，则将其定义为 NPY_BYTE

#define NPY_UINT128 NPY_UBYTE
// 如果 NPY_UINT128 未定义，则将其定义为 NPY_UBYTE

typedef signed char npy_int128;
// 定义 npy_int128 为有符号字符类型

typedef unsigned char npy_uint128;
// 定义 npy_uint128 为无符号字符类型

#    define PyInt128ScalarObject PyByteScalarObject
// 将 PyInt128ScalarObject 定义为 PyByteScalarObject

#    define PyInt128ArrType_Type PyByteArrType_Type
// 将 PyInt128ArrType_Type 定义为 PyByteArrType_Type

#    define PyUInt128ScalarObject PyUByteScalarObject
// 将 PyUInt128ScalarObject 定义为 PyUByteScalarObject

#    define PyUInt128ArrType_Type PyUByteArrType_Type
// 将 PyUInt128ArrType_Type 定义为 PyUByteArrType_Type

#define NPY_INT128_FMT NPY_BYTE_FMT
// 将 NPY_INT128_FMT 定义为 NPY_BYTE_FMT

#define NPY_UINT128_FMT NPY_UBYTE_FMT
// 将 NPY_UINT128_FMT 定义为 NPY_UBYTE_FMT

#endif
#endif



#if NPY_BITSOF_DOUBLE == 32
#ifndef NPY_FLOAT32
#define NPY_FLOAT32 NPY_DOUBLE
// 如果 NPY_FLOAT32 未定义，则将其定义为 NPY_DOUBLE

#define NPY_COMPLEX64 NPY_CDOUBLE
// 如果 NPY_COMPLEX64 未定义，则将其定义为 NPY_CDOUBLE

typedef double npy_float32;
// 定义 npy_float32 为双精度浮点类型

typedef npy_cdouble npy_complex64;
// 定义 npy_complex64 为复数类型 npy_cdouble

#    define PyFloat32ScalarObject PyDoubleScalarObject
// 将 PyFloat32ScalarObject 定义为 PyDoubleScalarObject

#    define PyComplex64ScalarObject PyCDoubleScalarObject
// 将 PyComplex64ScalarObject 定义为 PyCDoubleScalarObject

#    define PyFloat32ArrType_Type PyDoubleArrType_Type
// 将 PyFloat32ArrType_Type 定义为 PyDoubleArrType_Type

#    define PyComplex64ArrType_Type PyCDoubleArrType_Type
// 将 PyComplex64ArrType_Type 定义为 PyCDoubleArrType_Type

#define NPY_FLOAT32_FMT NPY_DOUBLE_FMT
// 将 NPY_FLOAT32_FMT 定义为 NPY_DOUBLE_FMT

#define NPY_COMPLEX64_FMT NPY_CDOUBLE_FMT
// 将 NPY_COMPLEX64_FMT 定义为 NPY_CDOUBLE_FMT

#endif
#elif NPY_BITSOF_DOUBLE == 64
#ifndef NPY_FLOAT64
#define NPY_FLOAT64 NPY_DOUBLE
// 如果 NPY_FLOAT64 未定义，则将其定义为 NPY_DOUBLE

#define NPY_COMPLEX128 NPY_CDOUBLE
// 如果 NPY_COMPLEX128 未定义，则将其定义为 NPY_CDOUBLE

typedef double npy_float64;
// 定义 npy_float64 为双精度浮点类型

typedef npy_cdouble npy_complex128;
// 定义 npy_complex128 为复数类型 npy_cdouble

#    define PyFloat64ScalarObject PyDoubleScalarObject
// 将 PyFloat64ScalarObject 定义为 PyDoubleScalarObject

#    define PyComplex128ScalarObject PyCDoubleScalarObject
// 将 PyComplex128ScalarObject 定义为 PyCDoubleScalarObject

#    define PyFloat64ArrType_Type PyDoubleArrType_Type
// 将 PyFloat64ArrType_Type 定义为 PyDoubleArrType_Type

#    define PyComplex128ArrType_Type PyCDoubleArrType_Type
// 将 PyComplex128ArrType_Type 定义为 PyCDoubleArrType_Type

#define NPY_FLOAT64_FMT NPY_DOUBLE_FMT
// 将 NPY_FLOAT64_FMT 定义为 NPY_DOUBLE_FMT

#define NPY_COMPLEX128_FMT NPY_CDOUBLE_FMT
// 将 NPY_COMPLEX128_FMT 定义为 NPY_CDOUBLE_FMT

#endif
#elif NPY_BITSOF_DOUBLE == 80
#ifndef NPY_FLOAT80
#define NPY_FLOAT80 NPY_DOUBLE
// 如果 NPY_FLOAT80 未定义，则将其定义为 NPY_DOUBLE

#define NPY_COMPLEX160 NPY_CDOUBLE
// 如果 NPY_COMPLEX160 未定义，则将其定义为 NPY_CDOUBLE

typedef double npy_float80;
// 定义 npy_float80 为双精度浮点类型

typedef npy_cdouble npy_complex160;
// 定义 npy_complex160 为复数类型 npy_cdouble

#    define PyFloat80ScalarObject PyDoubleScalarObject
// 将 PyFloat80ScalarObject 定义为 PyDoubleScalarObject

#    define PyComplex160ScalarObject PyCDoubleScalarObject
// 将 PyComplex160ScalarObject 定义为 PyCDoubleScalarObject

#    define PyFloat80ArrType_Type PyDoubleArrType_Type
// 将 PyFloat80ArrType_Type 定义为 PyDoubleArrType_Type

#    define PyComplex160ArrType_Type PyCDoubleArrType_Type
// 将 PyComplex160ArrType_Type 定义为 PyCDoubleArrType_Type

#define NPY_FLOAT80_FMT NPY_DOUBLE_FMT
// 将 NPY_FLOAT80_FMT 定义为 NPY_DOUBLE_FMT

#define NPY_COMPLEX160_FMT NPY_CDOUBLE_FMT
// 将 NPY_COMPLEX160_FMT 定义为 NPY_CDOUBLE_FMT

#endif
#elif NPY_BITSOF_DOUBLE == 96
#ifndef NPY_FLOAT96
#define NPY_FLOAT96 NPY_DOUBLE
// 如果 NPY_FLOAT96 未定义，则将其定义为 NPY_DOUBLE

#define NPY_COMPLEX192 NPY_CDOUBLE
// 如果 NPY_COMPLEX192 未定义，则将其定义为 NPY_CDOUBLE

typedef double npy_float96;
// 定义 npy_float96 为双精度浮点类型

typedef npy_cdouble npy_complex192;
// 定义 npy_complex192 为复数类型 npy_cdouble

#    define PyFloat96ScalarObject PyDoubleScalarObject
// 将 PyFloat96ScalarObject 定义为 PyDoubleScalarObject

#    define PyComplex192ScalarObject PyCDoubleScalarObject
// 将 PyComplex192ScalarObject 定义为 PyCDoubleScalarObject

#    define PyFloat96ArrType_Type PyDoubleArrType_Type
// 将 PyFloat96ArrType_Type 定义为 PyDoubleArrType_Type

#    define PyComplex192ArrType_Type PyCDoubleArrType_Type
// 将 PyComplex192ArrType_Type 定义为 PyCDoubleArrType_Type
#if NPY_BITSOF_FLOAT == 32
// 如果浮点数位数为32位
#ifndef NPY_FLOAT32
    // 如果未定义NPY_FLOAT32
#define NPY_FLOAT32 NPY_FLOAT
    // 定义NPY_FLOAT32为NPY_FLOAT
#define NPY_COMPLEX64 NPY_CFLOAT
    // 定义NPY_COMPLEX64为NPY_CFLOAT
    typedef float npy_float32;
    // 定义npy_float32为float类型
    typedef npy_cfloat npy_complex64;
    // 定义npy_complex64为npy_cfloat类型
#    define PyFloat32ScalarObject PyFloatScalarObject
    // 定义PyFloat32ScalarObject为PyFloatScalarObject
#    define PyComplex64ScalarObject PyCFloatScalarObject
    // 定义PyComplex64ScalarObject为PyCFloatScalarObject
#    define PyFloat32ArrType_Type PyFloatArrType_Type
    // 定义PyFloat32ArrType_Type为PyFloatArrType_Type
#    define PyComplex64ArrType_Type PyCFloatArrType_Type
    // 定义PyComplex64ArrType_Type为PyCFloatArrType_Type
#define NPY_FLOAT32_FMT NPY_FLOAT_FMT
    // 定义NPY_FLOAT32_FMT为NPY_FLOAT_FMT
#define NPY_COMPLEX64_FMT NPY_CFLOAT_FMT
    // 定义NPY_COMPLEX64_FMT为NPY_CFLOAT_FMT
#endif
#elif NPY_BITSOF_FLOAT == 64
// 否则如果浮点数位数为64位
#ifndef NPY_FLOAT64
    // 如果未定义NPY_FLOAT64
#define NPY_FLOAT64 NPY_FLOAT
    // 定义NPY_FLOAT64为NPY_FLOAT
#define NPY_COMPLEX128 NPY_CFLOAT
    // 定义NPY_COMPLEX128为NPY_CFLOAT
    typedef float npy_float64;
    // 定义npy_float64为float类型
    typedef npy_cfloat npy_complex128;
    // 定义npy_complex128为npy_cfloat类型
#    define PyFloat64ScalarObject PyFloatScalarObject
    // 定义PyFloat64ScalarObject为PyFloatScalarObject
#    define PyComplex128ScalarObject PyCFloatScalarObject
    // 定义PyComplex128ScalarObject为PyCFloatScalarObject
#    define PyFloat64ArrType_Type PyFloatArrType_Type
    // 定义PyFloat64ArrType_Type为PyFloatArrType_Type
#    define PyComplex128ArrType_Type PyCFloatArrType_Type
    // 定义PyComplex128ArrType_Type为PyCFloatArrType_Type
#define NPY_FLOAT64_FMT NPY_FLOAT_FMT
    // 定义NPY_FLOAT64_FMT为NPY_FLOAT_FMT
#define NPY_COMPLEX128_FMT NPY_CFLOAT_FMT
    // 定义NPY_COMPLEX128_FMT为NPY_CFLOAT_FMT
#endif
#elif NPY_BITSOF_FLOAT == 80
// 否则如果浮点数位数为80位
#ifndef NPY_FLOAT80
    // 如果未定义NPY_FLOAT80
#define NPY_FLOAT80 NPY_FLOAT
    // 定义NPY_FLOAT80为NPY_FLOAT
#define NPY_COMPLEX160 NPY_CFLOAT
    // 定义NPY_COMPLEX160为NPY_CFLOAT
    typedef float npy_float80;
    // 定义npy_float80为float类型
    typedef npy_cfloat npy_complex160;
    // 定义npy_complex160为npy_cfloat类型
#    define PyFloat80ScalarObject PyFloatScalarObject
    // 定义PyFloat80ScalarObject为PyFloatScalarObject
#    define PyComplex160ScalarObject PyCFloatScalarObject
    // 定义PyComplex160ScalarObject为PyCFloatScalarObject
#    define PyFloat80ArrType_Type PyFloatArrType_Type
    // 定义PyFloat80ArrType_Type为PyFloatArrType_Type
#    define PyComplex160ArrType_Type PyCFloatArrType_Type
    // 定义PyComplex160ArrType_Type为PyCFloatArrType_Type
#define NPY_FLOAT80_FMT NPY_FLOAT_FMT
    // 定义NPY_FLOAT80_FMT为NPY_FLOAT_FMT
#define NPY_COMPLEX160_FMT NPY_CFLOAT_FMT
    // 定义NPY_COMPLEX160_FMT为NPY_CFLOAT_FMT
#endif
#elif NPY_BITSOF_FLOAT == 96
// 否则如果浮点数位数为96位
#ifndef NPY_FLOAT96
    // 如果未定义NPY_FLOAT96
#define NPY_FLOAT96 NPY_FLOAT
    // 定义NPY_FLOAT96为NPY_FLOAT
#define NPY_COMPLEX192 NPY_CFLOAT
    // 定义NPY_COMPLEX192为NPY_CFLOAT
    typedef float npy_float96;
    // 定义npy_float96为float类型
    typedef npy_cfloat npy_complex192;
    // 定义npy_complex192为npy_cfloat类型
#    define PyFloat96ScalarObject PyFloatScalarObject
    // 定义PyFloat96ScalarObject为PyFloatScalarObject
#    define PyComplex192ScalarObject PyCFloatScalarObject
    // 定义PyComplex192ScalarObject为PyCFloatScalarObject
#    define PyFloat96ArrType_Type PyFloatArrType_Type
    // 定义PyFloat96ArrType_Type为PyFloatArrType_Type
#    define PyComplex192ArrType_Type PyCFloatArrType_Type
    // 定义PyComplex192ArrType_Type为PyCFloatArrType_Type
#define NPY_FLOAT96_FMT NPY_FLOAT_FMT
    // 定义NPY_FLOAT96_FMT为NPY_FLOAT_FMT
#define NPY_COMPLEX192_FMT NPY_CFLOAT_FMT
    // 定义NPY_COMPLEX192_FMT为NPY_CFLOAT_FMT
#endif
#elif NPY_BITSOF_FLOAT == 128
// 否则如果浮点数位数为128位
#ifndef NPY_FLOAT128
    // 如果未定义NPY_FLOAT128
#define NPY_FLOAT128 NPY_FLOAT
    // 定义NPY_FLOAT128为NPY_FLOAT
#define NPY_COMPLEX256 NPY_CFLOAT
    // 定义NPY_COMPLEX256为NPY_CFLOAT
    typedef float npy_float128;
    // 定义npy_float128为float类型
    typedef npy_cfloat npy_complex256;
    // 定义npy_complex256为npy_cfloat类型
#    define PyFloat128ScalarObject PyFloatScalarObject
    // 定义PyFloat128ScalarObject为PyFloatScalarObject
#    define PyComplex256ScalarObject PyCFloatScalarObject
    // 定义PyComplex256ScalarObject为PyCFloatScalarObject
#    define PyFloat128ArrType_Type PyFloatArrType_Type
    // 定义PyFloat128ArrType_Type为PyFloatArrType_Type
/*    define PyComplex256ArrType_Type PyCFloatArrType_Type
#define NPY_FLOAT128_FMT NPY_FLOAT_FMT
#define NPY_COMPLEX256_FMT NPY_CFLOAT_FMT
#endif
#endif
*/

/* half/float16 isn't a floating-point type in C */
// 定义 NPY_FLOAT16 为 NPY_HALF
#define NPY_FLOAT16 NPY_HALF
// 定义 npy_half 为 npy_uint16
typedef npy_uint16 npy_half;
// 定义 npy_float16 为 npy_half
typedef npy_half npy_float16;

#if NPY_BITSOF_LONGDOUBLE == 32
#ifndef NPY_FLOAT32
// 定义 NPY_FLOAT32 为 NPY_LONGDOUBLE
#define NPY_FLOAT32 NPY_LONGDOUBLE
// 定义 NPY_COMPLEX64 为 NPY_CLONGDOUBLE
#define NPY_COMPLEX64 NPY_CLONGDOUBLE
        // 定义 npy_float32 为 npy_longdouble
        typedef npy_longdouble npy_float32;
        // 定义 npy_complex64 为 npy_clongdouble
        typedef npy_clongdouble npy_complex64;
//    define PyFloat32ScalarObject PyLongDoubleScalarObject
//    define PyComplex64ScalarObject PyCLongDoubleScalarObject
//    define PyFloat32ArrType_Type PyLongDoubleArrType_Type
//    define PyComplex64ArrType_Type PyCLongDoubleArrType_Type
#define NPY_FLOAT32_FMT NPY_LONGDOUBLE_FMT
#define NPY_COMPLEX64_FMT NPY_CLONGDOUBLE_FMT
#endif
#elif NPY_BITSOF_LONGDOUBLE == 64
#ifndef NPY_FLOAT64
// 定义 NPY_FLOAT64 为 NPY_LONGDOUBLE
#define NPY_FLOAT64 NPY_LONGDOUBLE
// 定义 NPY_COMPLEX128 为 NPY_CLONGDOUBLE
#define NPY_COMPLEX128 NPY_CLONGDOUBLE
        // 定义 npy_float64 为 npy_longdouble
        typedef npy_longdouble npy_float64;
        // 定义 npy_complex128 为 npy_clongdouble
        typedef npy_clongdouble npy_complex128;
//    define PyFloat64ScalarObject PyLongDoubleScalarObject
//    define PyComplex128ScalarObject PyCLongDoubleScalarObject
//    define PyFloat64ArrType_Type PyLongDoubleArrType_Type
//    define PyComplex128ArrType_Type PyCLongDoubleArrType_Type
#define NPY_FLOAT64_FMT NPY_LONGDOUBLE_FMT
#define NPY_COMPLEX128_FMT NPY_CLONGDOUBLE_FMT
#endif
#elif NPY_BITSOF_LONGDOUBLE == 80
#ifndef NPY_FLOAT80
// 定义 NPY_FLOAT80 为 NPY_LONGDOUBLE
#define NPY_FLOAT80 NPY_LONGDOUBLE
// 定义 NPY_COMPLEX160 为 NPY_CLONGDOUBLE
#define NPY_COMPLEX160 NPY_CLONGDOUBLE
        // 定义 npy_float80 为 npy_longdouble
        typedef npy_longdouble npy_float80;
        // 定义 npy_complex160 为 npy_clongdouble
        typedef npy_clongdouble npy_complex160;
//    define PyFloat80ScalarObject PyLongDoubleScalarObject
//    define PyComplex160ScalarObject PyCLongDoubleScalarObject
//    define PyFloat80ArrType_Type PyLongDoubleArrType_Type
//    define PyComplex160ArrType_Type PyCLongDoubleArrType_Type
#define NPY_FLOAT80_FMT NPY_LONGDOUBLE_FMT
#define NPY_COMPLEX160_FMT NPY_CLONGDOUBLE_FMT
#endif
#elif NPY_BITSOF_LONGDOUBLE == 96
#ifndef NPY_FLOAT96
// 定义 NPY_FLOAT96 为 NPY_LONGDOUBLE
#define NPY_FLOAT96 NPY_LONGDOUBLE
// 定义 NPY_COMPLEX192 为 NPY_CLONGDOUBLE
#define NPY_COMPLEX192 NPY_CLONGDOUBLE
        // 定义 npy_float96 为 npy_longdouble
        typedef npy_longdouble npy_float96;
        // 定义 npy_complex192 为 npy_clongdouble
        typedef npy_clongdouble npy_complex192;
//    define PyFloat96ScalarObject PyLongDoubleScalarObject
//    define PyComplex192ScalarObject PyCLongDoubleScalarObject
//    define PyFloat96ArrType_Type PyLongDoubleArrType_Type
//    define PyComplex192ArrType_Type PyCLongDoubleArrType_Type
#define NPY_FLOAT96_FMT NPY_LONGDOUBLE_FMT
#define NPY_COMPLEX192_FMT NPY_CLONGDOUBLE_FMT
#endif
#elif NPY_BITSOF_LONGDOUBLE == 128
#ifndef NPY_FLOAT128
// 定义 NPY_FLOAT128 为 NPY_LONGDOUBLE
#define NPY_FLOAT128 NPY_LONGDOUBLE
// 定义 NPY_COMPLEX256 为 NPY_CLONGDOUBLE
#define NPY_COMPLEX256 NPY_CLONGDOUBLE
        // 定义 npy_float128 为 npy_longdouble
        typedef npy_longdouble npy_float128;
        // 定义 npy_complex256 为 npy_clongdouble
        typedef npy_clongdouble npy_complex256;
//    define PyFloat128ScalarObject PyLongDoubleScalarObject
//    define PyComplex256ScalarObject PyCLongDoubleScalarObject
//    define PyFloat128ArrType_Type PyLongDoubleArrType_Type
/* 如果 NPY_BITSOF_LONGDOUBLE 宏定义为 128，则定义一系列与长双精度浮点数相关的宏和类型别名 */
#if NPY_BITSOF_LONGDOUBLE == 128
#define NPY_FLOAT128 NPY_LONGDOUBLE
#define NPY_COMPLEX256 NPY_CLONGDOUBLE
typedef npy_longdouble npy_float128;
typedef npy_clongdouble npy_complex256;
#define PyFloat128ScalarObject PyLongDoubleScalarObject
#define PyComplex256ScalarObject PyCLongDoubleScalarObject
#define PyFloat128ArrType_Type PyLongDoubleArrType_Type
#define PyComplex256ArrType_Type PyCLongDoubleArrType_Type
#define NPY_FLOAT128_FMT NPY_LONGDOUBLE_FMT
#define NPY_COMPLEX256_FMT NPY_CLONGDOUBLE_FMT
#endif

/* 如果 NPY_BITSOF_LONGDOUBLE 宏定义为 256，则定义一系列与长双精度浮点数相关的宏和类型别名 */
#elif NPY_BITSOF_LONGDOUBLE == 256
#define NPY_FLOAT256 NPY_LONGDOUBLE
#define NPY_COMPLEX512 NPY_CLONGDOUBLE
typedef npy_longdouble npy_float256;
typedef npy_clongdouble npy_complex512;
#define PyFloat256ScalarObject PyLongDoubleScalarObject
#define PyComplex512ScalarObject PyCLongDoubleScalarObject
#define PyFloat256ArrType_Type PyLongDoubleArrType_Type
#define PyComplex512ArrType_Type PyCLongDoubleArrType_Type
#define NPY_FLOAT256_FMT NPY_LONGDOUBLE_FMT
#define NPY_COMPLEX512_FMT NPY_CLONGDOUBLE_FMT
#endif

/* datetime 的类型定义 */
typedef npy_int64 npy_timedelta;  // 定义 npy_timedelta 类型为 npy_int64
typedef npy_int64 npy_datetime;   // 定义 npy_datetime 类型为 npy_int64
#define NPY_DATETIME_FMT NPY_INT64_FMT  // 定义日期时间格式为 npy_int64 格式
#define NPY_TIMEDELTA_FMT NPY_INT64_FMT  // 定义时间间隔格式为 npy_int64 格式

/* numarray 风格的位宽命名类型定义结束标志 */
#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NPY_COMMON_H_ */
```