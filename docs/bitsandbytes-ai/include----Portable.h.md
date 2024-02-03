# `bitsandbytes\include\Portable.h`

```py
#pragma once
// 防止头文件被重复包含

#include <limits>
// 包含 limits 头文件，用于获取数值极限值
#include <cmath>
// 包含 cmath 头文件，用于数学计算
#include <stdexcept>
// 包含 stdexcept 头文件，用于异常处理
#include <sstream>
// 包含 sstream 头文件，用于字符串流操作

#ifdef __FMA__
#define USE_FMA
#endif
// 如果定义了 __FMA__，则定义 USE_FMA

#ifdef __AVX2__
#define USE_AVX2
#endif
// 如果定义了 __AVX2__，则定义 USE_AVX2

#ifdef __AVX__
#define USE_AVX
#endif
// 如果定义了 __AVX__，则定义 USE_AVX

#ifdef __SSE4_1__
#define USE_SSE41
#endif
// 如果定义了 __SSE4_1__，则定义 USE_SSE41

#ifdef __SSE4_2__
#define USE_SSE42
#endif
// 如果定义了 __SSE4_2__，则定义 USE_SSE42

#ifndef _MSC_VER
#include <stdint.h>
#endif
// 如果不是 MSC 编译器，则包含 stdint.h 头文件

namespace BinSearch {
// 命名空间 BinSearch

#ifndef _MSC_VER
typedef  int8_t   int8;
typedef uint8_t  uint8;
typedef  int32_t   int32;
typedef uint32_t  uint32;
typedef  int64_t   int64;
typedef uint64_t  uint64;
#else
typedef  __int8   int8;
typedef unsigned __int8  uint8;
typedef  __int32   int32;
typedef unsigned __int32  uint32;
typedef  __int64   int64;
typedef unsigned __int64  uint64;
#endif
// 定义不同平台下的整型别名

namespace Details {
// 命名空间 Details

#define myassert(cond, msg) if (!cond){ std::ostringstream os; os << "\nassertion failed: " << #cond << ", " << msg << "\n"; throw std::invalid_argument(os.str()); }
// 定义宏，用于断言检查条件是否成立，否则抛出异常

#if defined(_MSC_VER)
inline uint32 log2 (uint32 val) {
    if (val == 1) return 0;
    uint32 ret = 0;
    do {
        ret++;
        val >>= 1;
    } while (val > 1);
    return ret;
}
#endif
// 如果是 MSC 编译器，定义 log2 函数

#ifdef _DEBUG
#define DEBUG
#endif
// 如果定义了 _DEBUG，定义 DEBUG

#ifdef _MSC_VER
#   define FORCE_INLINE __forceinline
#   define NO_INLINE __declspec(noinline)
#else
#   define NO_INLINE __attribute__((noinline))
#   ifdef DEBUG
#       define FORCE_INLINE NO_INLINE
#   else
#       define FORCE_INLINE __attribute__((always_inline)) inline
#   endif
#endif
// 根据编译器定义内联函数宏

#ifdef USE_AVX
#define COMISS "vcomiss"
#define COMISD "vcomisd"
#else
#define COMISS "comiss"
#define COMISD "comisd"
#endif
// 根据是否使用 AVX 定义指令

#if defined(_MSC_VER) && (_MSC_VER <= 1500)
#include <float.h>
inline float mynext(float x)
{
    return _nextafterf(x, std::numeric_limits<float>::max());
}

inline double mynext(double x)
{
    return _nextafter(x, std::numeric_limits<double>::max());
}
inline float myprev(float x)
{
    return _nextafterf(x, -std::numeric_limits<float>::max());
}
// 如果是 MSC 编译器且版本小于等于 1500，定义 mynext 和 myprev 函数
// 如果定义了宏，则定义双精度浮点数的前一个值函数
inline double myprev(double x)
{
    // 返回比 x 小的最接近的双精度浮点数
    return _nextafter(x, -std::numeric_limits<double>::max());
}
#else
// 如果未定义宏，则定义单精度浮点数的下一个值函数
inline float mynext(float x)
{
    // 返回比 x 大的最接近的单精度浮点数
    return std::nextafterf(x, std::numeric_limits<float>::max());
}

// 定义双精度浮点数的下一个值函数
inline double mynext(double x)
{
    // 返回比 x 大的最接近的双精度浮点数
    return std::nextafter(x, std::numeric_limits<double>::max());
}
// 定义单精度浮点数的前一个值函数
inline float myprev(float x)
{
    // 返回比 x 小的最接近的单精度浮点数
    return std::nextafterf(x, -std::numeric_limits<float>::max());
}

// 定义双精度浮点数的前一个值函数
inline double myprev(double x)
{
    // 返回比 x 小的最接近的双精度浮点数
    return std::nextafter(x, -std::numeric_limits<double>::max());
}
#endif

// 定义模板函数，返回输入值的下一个值
template <typename T>
inline T next(T x)
{
    // 循环4次，每次调用mynext函数获取下一个值
    for (int i = 0; i < 4; ++i)
        x = mynext(x);
    // 返回最终结果
    return x;
}

// 定义模板函数，返回输入值的前一个值
template <typename T>
inline T prev(T x)
{
    // 循环4次，每次调用myprev函数获取前一个值
    for (int i = 0; i < 4; ++i)
        x = myprev(x);
    // 返回最终结果
    return x;
}

} // namepsace Details
} // namespace BinSearch
```