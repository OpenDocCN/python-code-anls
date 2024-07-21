# `.\pytorch\aten\src\ATen\test\vec_test_all_types.h`

```py
#pragma once
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <c10/util/bit_cast.h>
#include <c10/util/irange.h>
#include <gtest/gtest.h>
#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <complex>
#include <math.h>
#include <float.h>
#include <algorithm>

#if defined(CPU_CAPABILITY_AVX512)
#define CACHE_LINE 64
#else
#define CACHE_LINE 32
#endif

#if defined(__GNUC__)
#define CACHE_ALIGN __attribute__((aligned(CACHE_LINE)))
#define not_inline __attribute__((noinline))
#elif defined(_WIN32)
#define CACHE_ALIGN __declspec(align(CACHE_LINE))
#define not_inline __declspec(noinline)
#else
CACHE_ALIGN #define
#define not_inline
#endif

// 根据 CPU_CAPABILITY_DEFAULT 或 _MSC_VER 定义是否测试默认能力
#if defined(CPU_CAPABILITY_DEFAULT) || defined(_MSC_VER)
#define TEST_AGAINST_DEFAULT 1
// 如果未定义特定的 CPU 能力宏，则也测试默认能力
#elif !defined(CPU_CAPABILITY_AVX512) && !defined(CPU_CAPABILITY_AVX2) && !defined(CPU_CAPABILITY_VSX) && !defined(CPU_CAPABILITY_ZVECTOR)
#define TEST_AGAINST_DEFAULT 1
#else
#undef TEST_AGAINST_DEFAULT
#endif

// 定义宏 NAME_INFO 用于生成带有文件名和行号的字符串
#undef NAME_INFO
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define NAME_INFO(name) TOSTRING(name) " " TOSTRING(__FILE__) ":" TOSTRING(__LINE__)

// 定义 RESOLVE_OVERLOAD 宏用于解析重载函数
#define RESOLVE_OVERLOAD(...)                                  \
  [](auto&&... args) -> decltype(auto) {                       \
    return __VA_ARGS__(std::forward<decltype(args)>(args)...); \
  }

// 根据 CPU 能力和编译器类型定义是否检查低精度的宏
#if defined(CPU_CAPABILITY_ZVECTOR) || defined(CPU_CAPABILITY_VSX) || defined(CPU_CAPABILITY_AVX2) || \
  defined(CPU_CAPABILITY_AVX512) && (defined(__GNUC__) || defined(__GNUG__))
#undef CHECK_DEQUANT_WITH_LOW_PRECISION
#define CHECK_WITH_FMA 1
#elif !defined(CPU_CAPABILITY_VSX) && !defined(CPU_CAPABILITY_AVX2)
#undef CHECK_DEQUANT_WITH_LOW_PRECISION
#undef CHECK_WITH_FMA
#else
#define CHECK_DEQUANT_WITH_LOW_PRECISION 1
#undef CHECK_WITH_FMA
#endif

// 定义 Complex 模板别名用于复数类型
template<typename T>
using Complex = typename c10::complex<T>;

// 定义 VecType 模板别名用于向量化类型
template <typename T>
using VecType = typename at::vec::Vectorized<T>;

// 定义标准向量化类型别名
using vfloat = VecType<float>;
using vdouble = VecType<double>;
using vcomplex = VecType<Complex<float>>;
using vcomplexDbl = VecType<Complex<double>>;
using vlong = VecType<int64_t>;
using vint = VecType<int32_t>;
using vshort = VecType<int16_t>;
using vqint8 = VecType<c10::qint8>;
using vquint8 = VecType<c10::quint8>;
using vqint = VecType<c10::qint32>;
using vBFloat16 = VecType<c10::BFloat16>;
using vHalf = VecType<c10::Half>;

// 定义 ValueType 模板别名用于获取类型的值类型
template <typename T>
using ValueType = typename T::value_type;

// 定义 BitStr 结构模板，根据模板参数大小定义整数类型别名
template <int N>
struct BitStr
{
    using type = uintmax_t;
};

template <>
struct BitStr<8>
{
    using type = uint64_t;
};

template <>
struct BitStr<4>
{
    using type = uint32_t;
};

template <>
struct BitStr<2>
{
    using type = uint16_t;
};

template <>
struct BitStr<1>
{
    using type = uint8_t;
};

// 定义 BitType 模板别名用于根据类型获取整数类型别名
template <typename T>
using BitType = typename BitStr<sizeof(T)>::type;

// 定义 VecTypeHelper 结构模板用于辅助操作 VecType 类型
template<typename T>
struct VecTypeHelper {
    // 定义 holdType 别名用于获取 VecType 的值类型
    using holdType = typename T::value_type;
};
    # 定义一个类型别名，其类型为 T::value_type
    using memStorageType = typename T::value_type;
    # 定义一个静态常量，表示 T 类型的大小或元素个数
    static constexpr int holdCount = T::size();
    # 定义一个静态常量，表示单元存储的数量为 1
    static constexpr int unitStorageCount = 1;
};

// 特化模板 VecTypeHelper<vcomplex>，定义 holdType 和 memStorageType 类型以及单位数目
template<>
struct VecTypeHelper<vcomplex> {
    using holdType = Complex<float>;  // holdType 类型为 Complex<float>
    using memStorageType = float;     // memStorageType 类型为 float
    static constexpr int holdCount = vcomplex::size();  // holdCount 等于 vcomplex 的大小
    static constexpr int unitStorageCount = 2;          // unitStorageCount 等于 2
};

// 特化模板 VecTypeHelper<vcomplexDbl>，定义 holdType 和 memStorageType 类型以及单位数目
template<>
struct VecTypeHelper<vcomplexDbl> {
    using holdType = Complex<double>;  // holdType 类型为 Complex<double>
    using memStorageType = double;     // memStorageType 类型为 double
    static constexpr int holdCount = vcomplexDbl::size();  // holdCount 等于 vcomplexDbl 的大小
    static constexpr int unitStorageCount = 2;             // unitStorageCount 等于 2
};

// 特化模板 VecTypeHelper<vqint8>，定义 holdType 和 memStorageType 类型以及单位数目
template<>
struct VecTypeHelper<vqint8> {
    using holdType = c10::qint8;                   // holdType 类型为 c10::qint8
    using memStorageType = typename c10::qint8::underlying;  // memStorageType 类型为 c10::qint8 的底层类型
    static constexpr int holdCount = vqint8::size();  // holdCount 等于 vqint8 的大小
    static constexpr int unitStorageCount = 1;         // unitStorageCount 等于 1
};

// 特化模板 VecTypeHelper<vquint8>，定义 holdType 和 memStorageType 类型以及单位数目
template<>
struct VecTypeHelper<vquint8> {
    using holdType = c10::quint8;                         // holdType 类型为 c10::quint8
    using memStorageType = typename c10::quint8::underlying;  // memStorageType 类型为 c10::quint8 的底层类型
    static constexpr int holdCount = vquint8::size();    // holdCount 等于 vquint8 的大小
    static constexpr int unitStorageCount = 1;           // unitStorageCount 等于 1
};

// 特化模板 VecTypeHelper<vqint>，定义 holdType 和 memStorageType 类型以及单位数目
template<>
struct VecTypeHelper<vqint> {
    using holdType = c10::qint32;                         // holdType 类型为 c10::qint32
    using memStorageType = typename c10::qint32::underlying;  // memStorageType 类型为 c10::qint32 的底层类型
    static constexpr int holdCount = vqint::size();       // holdCount 等于 vqint 的大小
    static constexpr int unitStorageCount = 1;            // unitStorageCount 等于 1
};

// 特化模板 VecTypeHelper<vBFloat16>，定义 holdType 和 memStorageType 类型以及单位数目
template<>
struct VecTypeHelper<vBFloat16> {
    using holdType = c10::BFloat16;                       // holdType 类型为 c10::BFloat16
    using memStorageType = typename vBFloat16::value_type;  // memStorageType 类型为 vBFloat16 的 value_type 类型
    static constexpr int holdCount = vBFloat16::size();   // holdCount 等于 vBFloat16 的大小
    static constexpr int unitStorageCount = 1;            // unitStorageCount 等于 1
};

// 特化模板 VecTypeHelper<vHalf>，定义 holdType 和 memStorageType 类型以及单位数目
template<>
struct VecTypeHelper<vHalf> {
    using holdType = c10::Half;                           // holdType 类型为 c10::Half
    using memStorageType = typename vHalf::value_type;    // memStorageType 类型为 vHalf 的 value_type 类型
    static constexpr int holdCount = vHalf::size();       // holdCount 等于 vHalf 的大小
    static constexpr int unitStorageCount = 1;            // unitStorageCount 等于 1
};

// 模板别名 UholdType，用于获取 VecTypeHelper<T>::holdType 类型
template <typename T>
using UholdType = typename VecTypeHelper<T>::holdType;

// 模板别名 UvalueType，用于获取 VecTypeHelper<T>::memStorageType 类型
template <typename T>
using UvalueType = typename VecTypeHelper<T>::memStorageType;

// 返回数组的大小
template <class T, size_t N>
constexpr size_t size(T(&)[N]) {
    return N;
}

// 当 Filter 类型为 std::nullptr_t 时，调用接收一个参数的 call_filter 函数模板
template <typename Filter, typename T>
typename std::enable_if_t<std::is_same_v<Filter, std::nullptr_t>, void>
call_filter(Filter filter, T& val) {}

// 当 Filter 类型为 std::nullptr_t 时，调用接收两个参数的 call_filter 函数模板
template <typename Filter, typename T>
typename std::enable_if_t<std::is_same_v<Filter, std::nullptr_t>, void>
call_filter(Filter filter, T& first, T& second) {}

// 当 Filter 类型为 std::nullptr_t 时，调用接收三个参数的 call_filter 函数模板
template <typename Filter, typename T>
typename std::enable_if_t<std::is_same_v<Filter, std::nullptr_t>, void>
call_filter(Filter filter, T& first, T& second, T& third) {}

// 当 Filter 类型不为 std::nullptr_t 时，调用接收一个参数的 call_filter 函数模板
template <typename Filter, typename T>
typename std::enable_if_t<!std::is_same_v<Filter, std::nullptr_t>, void>
call_filter(Filter filter, T& val) {
    return filter(val);
}

// 当 Filter 类型不为 std::nullptr_t 时，调用接收两个参数的 call_filter 函数模板
template <typename Filter, typename T>
typename std::enable_if_t<!std::is_same_v<Filter, std::nullptr_t>, void>
call_filter(Filter filter, T& first, T& second) {
    return filter(first, second);
}

// 当 Filter 类型不为 std::nullptr_t 时，调用接收三个参数的 call_filter 函数模板
template <typename Filter, typename T>
typename std::enable_if_t<!std::is_same_v<Filter, std::nullptr_t>, void>
call_filter(Filter filter, T& first, T& second, T& third) {
    // 未实现内容
}
    # 调用 filter() 函数，对序列进行筛选
    return filter(first, second, third);
}

template <typename T>
struct DomainRange {
    T start;  // 范围的起始点 [
    T end;    // 范围的结束点，不包括此点。可以使用 nextafter 用于包括测试中的结束情况
};

template <typename T>
struct CustomCheck {
    std::vector<UholdType<T>> Args;  // 参数列表
    UholdType<T> expectedResult;     // 预期结果
};

template <typename T>
struct CheckWithinDomains {
    // 每个参数都有一个域范围
    std::vector<DomainRange<T>> ArgsDomain;  // 参数的域范围列表
    // 是否使用误差容忍检查
    bool CheckWithTolerance = false;
    T ToleranceError = (T)0;  // 误差容忍值
};

template <typename T>
std::ostream& operator<<(std::ostream& stream, const CheckWithinDomains<T>& dmn) {
    stream << "Domain: ";
    if (dmn.ArgsDomain.size() > 0) {
        for (const DomainRange<T>& x : dmn.ArgsDomain) {
            if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
                stream << "\n{ " << static_cast<int>(x.start) << ", " << static_cast<int>(x.end) << " }";
            }
            else {
                stream << "\n{ " << x.start << ", " << x.end << " }";
            }
        }
    }
    else {
        stream << "default range";  // 默认范围
    }
    if (dmn.CheckWithTolerance) {
        stream << "\nError tolerance: " << dmn.ToleranceError;  // 显示误差容忍值
    }
    return stream;
}

template <typename T>
bool check_both_nan(T x, T y) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::isnan(x) && std::isnan(y);  // 检查两个浮点数是否都是 NaN
    }
    return false;
}

template <typename T>
bool check_both_inf(T x, T y) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::isinf(x) && std::isinf(y);  // 检查两个浮点数是否都是无穷大
    }
    return false;
}

template<typename T>
std::enable_if_t<!std::is_floating_point_v<T>, bool> check_both_big(T x, T y) {
    return false;  // 对于非浮点数类型，总是返回假
}

template<typename T>
std::enable_if_t<std::is_floating_point_v<T>, bool> check_both_big(T x, T y) {
    T cmax = std::is_same_v<T, float> ? static_cast<T>(1e+30) : static_cast<T>(1e+300);
    T cmin = std::is_same_v<T, float> ? static_cast<T>(-1e+30) : static_cast<T>(-1e+300);
    // 仅在一个数为无穷大时允许
    bool x_inf = std::isinf(x);
    bool y_inf = std::isinf(y);
    bool px = x > 0;
    bool py = y > 0;
    return (px && x_inf && y >= cmax) || (py && y_inf && x >= cmax) ||
        (!px && x_inf && y <= cmin) || (!py && y_inf && x <= cmin);  // 检查两个浮点数是否都大于或小于指定值
}

template<class T> struct is_complex : std::false_type {};  // 默认情况下，不是复数类型

template<class T> struct is_complex<Complex<T>> : std::true_type {};  // 如果是复数类型，则为真

template<typename T>
T safe_fpt_division(T f1, T f2)
{
    // 代码源自 Boost
    // 避免溢出。
    if ((f2 < static_cast<T>(1)) && (f1 > f2 * std::numeric_limits<T>::max())) {
        return std::numeric_limits<T>::max();  // 返回类型的最大值
    }
    // 避免下溢。
    if ((f1 == static_cast<T>(0)) ||
        ((f2 > static_cast<T>(1)) && (f1 < f2 * std::numeric_limits<T>::min()))) {
        return static_cast<T>(0);  // 返回类型的零值
    }
    return f1 / f2;  // 执行安全浮点数除法
}

template<class T>
std::enable_if_t<std::is_floating_point_v<T>, bool>
nearlyEqual(T a, T b, T tolerance) {
    # 如果 a 和 b 都是 NaN，则认为它们相等，返回 true
    if (check_both_nan<T>(a, b)) return true;
    # 如果 a 和 b 都超过了设定的大值范围，认为它们相等，返回 true
    if (check_both_big(a, b)) return true;
    # 计算 a 和 b 的绝对值
    T absA = std::abs(a);
    T absB = std::abs(b);
    # 计算 a 和 b 之间的差的绝对值
    T diff = std::abs(a - b);
    # 如果差的绝对值小于等于容忍度 tolerance，则认为 a 和 b 相等，返回 true
    if (diff <= tolerance) {
        return true;
    }
    # 计算 diff 与 absB 的安全浮点数除法结果
    T d1 = safe_fpt_division<T>(diff, absB);
    # 计算 diff 与 absA 的安全浮点数除法结果
    T d2 = safe_fpt_division<T>(diff, absA);
    # 如果 d1 或 d2 中有任何一个小于等于容忍度 tolerance，则认为 a 和 b 相等，返回 true
    return (d1 <= tolerance) || (d2 <= tolerance);
template<class T>
std::enable_if_t<!std::is_floating_point_v<T>, bool>
nearlyEqual(T a, T b, T tolerance) {
    // 如果模板类型 T 不是浮点类型，则直接比较 a 和 b 是否相等
    return a == b;
}

template <typename T>
T reciprocal(T x) {
    // 返回 x 的倒数
    return 1 / x;
}

template <typename T>
T rsqrt(T x) {
    // 返回 x 的平方根的倒数
    return 1 / std::sqrt(x);
}

template <typename T>
T frac(T x) {
  // 返回 x 的小数部分
  return x - std::trunc(x);
}

template <class T>
T maximum(const T& a, const T& b) {
    // 返回 a 和 b 中的较大值
    return (a > b) ? a : b;
}

template <class T>
T minimum(const T& a, const T& b) {
    // 返回 a 和 b 中的较小值
    return (a < b) ? a : b;
}

template <class T>
T clamp(const T& a, const T& min, const T& max) {
    // 将 a 限制在 [min, max] 的范围内
    return a < min ? min : (a > max ? max : a);
}

template <class T>
T clamp_max(const T& a, const T& max) {
    // 将 a 限制在不超过 max 的范围内
    return a > max ? max : a;
}

template <class T>
T clamp_min(const T& a, const T& min) {
    // 将 a 限制在不低于 min 的范围内
    return a < min ? min : a;
}

template <class VT, size_t N>
void copy_interleave(VT(&vals)[N], VT(&interleaved)[N]) {
    // 将 vals 数组中的值交错复制到 interleaved 数组中
    static_assert(N % 2 == 0, "should be even");
    auto ptr1 = vals;
    auto ptr2 = vals + N / 2;
    for (size_t i = 0; i < N; i += 2) {
        interleaved[i] = *ptr1++;
        interleaved[i + 1] = *ptr2++;
    }
}

template <typename T>
bool is_zero(T val) {
    // 检查值 val 是否为零
    if constexpr (std::is_floating_point_v<T>) {
        return std::fpclassify(val) == FP_ZERO;
    } else {
        return val == 0;
    }
}

template <typename T>
void filter_clamp(T& f, T& s, T& t) {
    // 调整三个值，确保 s <= t
    if (t < s) {
        std::swap(s, t);
    }
}

template <typename T>
std::enable_if_t<std::is_floating_point_v<T>, void> filter_fmod(T& a, T& b) {
    // 在进行 fmod 计算时，避免因除法而导致溢出
    if (std::abs(b) < (T)1) {
      b = b < (T)0 ? (T)-1 : T(1);
    }
}

template <typename T>
std::enable_if_t<std::is_floating_point_v<T>, void> filter_fmadd(T& a, T& b, T& c) {
    // 设置限制以确保 fmadd (a * b + c) 不会溢出
    T max = std::sqrt(std::numeric_limits<T>::max()) / T(2.0);
    T min = ((T)0 - max);

    if (a > max) a = max;
    else if (a < min) a = min;

    if (b > max) b = max;
    else if (b < min) b = min;

    if (c > max) c = max;
    else if (c < min) c = min;
}

template <typename T>
void filter_zero(T& val) {
    // 如果 val 是零，则将其替换为 1，否则保持不变
    val = is_zero(val) ? (T)1 : val;
}

template <typename T>
std::enable_if_t<is_complex<Complex<T>>::value, void> filter_zero(Complex<T>& val) {
    // 如果 Complex 类型的实部或虚部为零，则分别替换为 1，否则保持不变
    T rr = val.real();
    T ii = val.imag();
    rr = is_zero(rr) ? (T)1 : rr;
    ii = is_zero(ii) ? (T)1 : ii;
    val = Complex<T>(rr, ii);
}

template <typename T>
void filter_int_minimum(T& val) {
    // 如果 T 是整数类型且 val 的值为最小值，则将其设置为 0
    if constexpr (!std::is_integral_v<T>) return;
    if (val == std::numeric_limits<T>::min()) {
        val = 0;
    }
}

template <typename T>
std::enable_if_t<is_complex<T>::value, void> filter_add_overflow(T& a, T& b)
{
    // 对于复数类型，未提供 filter_add_overflow 的实现
}

template <typename T>
std::enable_if_t<is_complex<T>::value, void> filter_sub_overflow(T& a, T& b)
{
    // 对于复数类型，未提供 filter_sub_overflow 的实现
}
// 如果 T 不是复数类型，则定义一个函数 filter_add_overflow，用于处理加法溢出情况
template <typename T>
std::enable_if_t<!is_complex<T>::value, void> filter_add_overflow(T& a, T& b) {
    // 如果 T 不是整数类型，直接返回
    if constexpr (std::is_integral_v<T> == false) return;
    // 获取 T 类型的最大和最小值
    T max = std::numeric_limits<T>::max();
    T min = std::numeric_limits<T>::min();
    // 检查加法溢出情况，并进行修正
    // min <= (a + b) <= max;
    // min - b <= a <= max - b
    if (b < 0) {
        if (a < min - b) {
            a = min - b;
        }
    }
    else {
        if (a > max - b) {
            a = max - b;
        }
    }
}

// 如果 T 不是复数类型，则定义一个函数 filter_sub_overflow，用于处理减法溢出情况
template <typename T>
std::enable_if_t<!is_complex<T>::value, void> filter_sub_overflow(T& a, T& b) {
    // 如果 T 不是整数类型，直接返回
    if constexpr (std::is_integral_v<T> == false) return;
    // 获取 T 类型的最大和最小值
    T max = std::numeric_limits<T>::max();
    T min = std::numeric_limits<T>::min();
    // 检查减法溢出情况，并进行修正
    // min <= (a - b) <= max;
    // min + b <= a <= max + b
    if (b < 0) {
        if (a > max + b) {
            a = max + b;
        }
    }
    else {
        if (a < min + b) {
            a = min + b;
        }
    }
}

// 如果 T 是复数类型，则定义一个空函数 filter_mult_overflow，暂不处理乘法溢出情况
template <typename T>
std::enable_if_t<is_complex<T>::value, void> filter_mult_overflow(T& val1, T& val2) {
    // 空函数，暂未实现复数类型的乘法溢出处理
}

// 如果 T 是复数类型，则定义一个空函数 filter_div_ub，暂不处理除法溢出情况
template <typename T>
std::enable_if_t<is_complex<T>::value, void> filter_div_ub(T& val1, T& val2) {
    // 空函数，暂未实现复数类型的除法溢出处理
    // 至少考虑零除法情况
    auto ret = std::abs(val2);
    if (ret == 0) {
        val2 = T(1, 2);
    }
}

// 如果 T 不是复数类型，则定义一个函数 filter_mult_overflow，用于处理乘法溢出情况
template <typename T>
std::enable_if_t<!is_complex<T>::value, void> filter_mult_overflow(T& val1, T& val2) {
    // 如果 T 不是整数类型，直接返回
    if constexpr (std::is_integral_v<T> == false) return;
    // 如果除数不为零，计算出临界值 c
    if (!is_zero(val2)) {
        T c = (std::numeric_limits<T>::max() - 1) / val2;
        // 检查乘法溢出情况，并进行修正
        if (std::abs(val1) >= c) {
            val1 = c;
        }
    }
}

// 如果 T 不是复数类型，则定义一个函数 filter_div_ub，用于处理除法溢出情况
template <typename T>
std::enable_if_t<!is_complex<T>::value, void> filter_div_ub(T& val1, T& val2) {
    // 如果除数为零，设置除数为 1
    if (is_zero(val2)) {
        val2 = 1;
    }
    // 对于特定整数类型 T 的情况，检查除法溢出情况，并进行修正
    else if (std::is_integral_v<T> && val1 == std::numeric_limits<T>::min() && val2 == -1) {
        val2 = 1;
    }
}

// 定义一个结构体 TestSeed，用于生成种子值
struct TestSeed {
    // 默认构造函数，使用当前时间生成种子
    TestSeed() : seed(std::chrono::high_resolution_clock::now().time_since_epoch().count()) {
    }
    // 带参构造函数，使用给定的种子值初始化
    TestSeed(uint64_t seed) : seed(seed) {
    }
    // 返回当前种子值的函数
    uint64_t getSeed() {
        return seed;
    }
    // 将 TestSeed 转换为 uint64_t 类型的隐式转换函数
    operator uint64_t () const {
        return seed;
    }
    // 返回新的 TestSeed 对象，种子值加上给定的索引值
    TestSeed add(uint64_t index) {
        return TestSeed(seed + index);
    }
private:
    uint64_t seed; // 私有成员变量，存储种子值
};

// 定义一个模板结构体 ValueGen，用于生成值的模板类
template <typename T, bool is_floating_point = std::is_floating_point_v<T>, bool is_complex = is_complex<T>::value>
struct ValueGen {
    std::uniform_int_distribution<int64_t> dis; // 均匀分布器，用于生成随机数
    std::mt19937 gen; // 随机数生成引擎
    // 默认构造函数，使用 T 类型的最小值和最大值初始化
    ValueGen() : ValueGen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max()) {
    }
    // 带参构造函数，使用给定的种子值初始化
    ValueGen(uint64_t seed) : ValueGen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed) {
    }
    // 带参构造函数，使用给定的起始值、结束值和种子值初始化
    ValueGen(T start, T stop, uint64_t seed = TestSeed()) {
        gen = std::mt19937(seed);
        dis = std::uniform_int_distribution<int64_t>(start, stop);
    }
    // 生成随机值的函数
    T get() {
        return static_cast<T>(dis(gen));
    }
};


这段代码包含了各种模板函数和结构体的定义，用于处理整数类型的溢出、复数类型的溢出处理未实现、随机数生成等功能。
# 定义一个模板结构体 ValueGen，用于生成类型 T 的值，在需要使用正态分布和舍入的情况下特化

template <typename T>
struct ValueGen<T, true, false>
{
    std::mt19937 gen;  // Mersenne Twister 19937 伪随机数生成器
    std::normal_distribution<T> normal;  // 正态分布对象，生成类型为 T 的随机数
    std::uniform_int_distribution<int> roundChance;  // 均匀整数分布对象，用于决定是否进行舍入
    T _start;  // 起始值
    T _stop;   // 停止值
    bool use_sign_change = false;  // 是否使用符号变更（未使用）
    bool use_round = true;  // 是否进行舍入

    // 默认构造函数，使用 T 类型的最小值和最大值初始化
    ValueGen() : ValueGen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max())
    {
    }

    // 带种子参数的构造函数，使用 T 类型的最小值和最大值初始化，并设置种子
    ValueGen(uint64_t seed) : ValueGen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed)
    {
    }

    // 构造函数，接受起始值、停止值和种子参数
    ValueGen(T start, T stop, uint64_t seed = TestSeed())
    {
        gen = std::mt19937(seed);  // 使用给定的种子初始化 Mersenne Twister 19937
        T mean = start * static_cast<T>(0.5) + stop * static_cast<T>(0.5);  // 计算均值
        T divRange = static_cast<T>(6.0);  // 区间的大小
        T stdev = std::abs(stop / divRange - start / divRange);  // 计算标准差
        normal = std::normal_distribution<T>{ mean, stdev };  // 初始化正态分布对象
        roundChance = std::uniform_int_distribution<int>(0, 5);  // 初始化均匀整数分布对象，范围是 [0, 5]
        _start = start;  // 设置起始值
        _stop = stop;    // 设置停止值
    }

    // 成员函数，返回类型为 T 的随机值
    T get()
    {
        T a = normal(gen);  // 生成一个符合正态分布的随机值
        auto rChoice = roundChance(gen);  // 使用均匀整数分布对象生成一个随机整数，决定是否进行舍入
        if (rChoice == 1)
            a = std::round(a);  // 如果随机整数为 1，则将 a 进行四舍五入
        if (a < _start)
            return nextafter(_start, _stop);  // 如果生成的值小于起始值，则返回靠近起始值的下一个浮点数
        if (a >= _stop)
            return nextafter(_stop, _start);  // 如果生成的值大于等于停止值，则返回靠近停止值的下一个浮点数
        return a;  // 返回生成的随机值
    }
};

# 定义一个模板结构体 ValueGen，用于生成复数类型 Complex<T> 的值，在不需要使用正态分布但需要舍入的情况下特化

template <typename T>
struct ValueGen<Complex<T>, false, true>
{
    std::mt19937 gen;  // Mersenne Twister 19937 伪随机数生成器
    std::normal_distribution<T> normal;  // 正态分布对象，生成类型为 T 的随机数
    std::uniform_int_distribution<int> roundChance;  // 均匀整数分布对象，用于决定是否进行舍入
    T _start;  // 起始值
    T _stop;   // 停止值
    bool use_sign_change = false;  // 是否使用符号变更（未使用）
    bool use_round = true;  // 是否进行舍入

    // 默认构造函数，使用 T 类型的最小值和最大值初始化
    ValueGen() : ValueGen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max())
    {
    }

    // 带种子参数的构造函数，使用 T 类型的最小值和最大值初始化，并设置种子
    ValueGen(uint64_t seed) : ValueGen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed)
    {
    }

    // 构造函数，接受起始值、停止值和种子参数
    ValueGen(T start, T stop, uint64_t seed = TestSeed())
    {
        gen = std::mt19937(seed);  // 使用给定的种子初始化 Mersenne Twister 19937
        T mean = start * static_cast<T>(0.5) + stop * static_cast<T>(0.5);  // 计算均值
        T divRange = static_cast<T>(6.0);  // 区间的大小
        T stdev = std::abs(stop / divRange - start / divRange);  // 计算标准差
        normal = std::normal_distribution<T>{ mean, stdev };  // 初始化正态分布对象
        roundChance = std::uniform_int_distribution<int>(0, 5);  // 初始化均匀整数分布对象，范围是 [0, 5]
        _start = start;  // 设置起始值
        _stop = stop;    // 设置停止值
    }

    // 成员函数，返回类型为 Complex<T> 的随机值
    Complex<T> get()
    {
        // 从生成器中获取正态分布的随机数 a 和 b
        T a = normal(gen);
        T b = normal(gen);
    
        // 获取四舍五入的概率值，并对其进行与 3 的按位与操作
        auto rChoice = roundChance(gen);
        rChoice = rChoice & 3;
    
        // 根据 rChoice 的值进行四舍五入操作
        if (rChoice & 1)
            a = std::round(a);
        if (rChoice & 2)
            b = std::round(b);
    
        // 如果 a 小于 _start，则将其调整为大于等于 _start 的最接近值
        if (a < _start)
            a = nextafter(_start, _stop);
        // 如果 a 大于等于 _stop，则将其调整为小于 _stop 的最接近值
        else if (a >= _stop)
            a = nextafter(_stop, _start);
    
        // 如果 b 小于 _start，则将其调整为大于等于 _start 的最接近值
        if (b < _start)
            b = nextafter(_start, _stop);
        // 如果 b 大于等于 _stop，则将其调整为小于 _stop 的最接近值
        else if (b >= _stop)
            b = nextafter(_stop, _start);
    
        // 返回一个复数对象，其实部为 a，虚部为 b
        return Complex<T>(a, b);
    }
};

// 模板函数：获取试验次数
// test_trials: 测试试验次数
// domains_size: 域的大小
template<class T>
int getTrialCount(int test_trials, int domains_size) {
    int trialCount;
    int trial_default = 1;
    // 如果 T 的大小不超过 2 字节，则设置默认试验次数为 128
    if (sizeof(T) <= 2) {
        trial_default = 128;  // 对于字节的一半覆盖
    }
    else {
        // 否则设置默认试验次数为 2*65536
        trial_default = 2 * std::numeric_limits<uint16_t>::max();
    }
    trialCount = test_trials < 1 ? trial_default : test_trials;  // 如果测试试验次数小于 1，则使用默认值
    if (domains_size > 1) {
        trialCount = trialCount / domains_size;  // 根据域的大小调整试验次数
        trialCount = trialCount < 1 ? 1 : trialCount;  // 保证试验次数至少为 1
    }
    return trialCount;  // 返回计算后的试验次数
}

// 模板类：测试用例构建器
// T: 测试类型
// U: T 对应的值类型
template <typename T, typename U = UvalueType<T>>
class TestCaseBuilder;

// 模板类：测试用例
// T: 测试类型
// U: T 对应的值类型
template <typename T, typename U = UvalueType<T>>
class TestingCase {
public:
    friend class TestCaseBuilder<T, U>;
    
    // 获取测试用例构建器
    static TestCaseBuilder<T, U> getBuilder() { return TestCaseBuilder<T, U>{}; }
    
    // 检查是否包含特殊值（如 NaN、Infinity）
    bool checkSpecialValues() const {
        // 这将用于检查 NaN、Infinity 和其他特殊情况
        return specialCheck;
    }
    
    // 获取试验次数
    size_t getTrialCount() const { return trials; }
    
    // 是否进行位级测试
    bool isBitwise() const { return bitwise; }
    
    // 获取域的检查列表
    const std::vector<CheckWithinDomains<U>>& getDomains() const {
        return domains;
    }
    
    // 获取自定义检查列表
    const std::vector<CustomCheck<T>>& getCustomChecks() const {
        return customCheck;
    }
    
    // 获取测试种子
    TestSeed getTestSeed() const {
        return testSeed;
    }

private:
    // 如果域为空，则测试默认情况
    std::vector<CheckWithinDomains<U>> domains;
    std::vector<CustomCheck<T>> customCheck;
    // 目前未使用
    bool specialCheck = false;
    bool bitwise = false;  // 进行位级测试
    size_t trials = 0;
    TestSeed testSeed;
};

// 模板类：测试用例构建器
// T: 测试类型
// U: T 对应的值类型
template <typename T, typename U >
class TestCaseBuilder {
private:
    TestingCase<T, U> _case;

public:
    // 设置是否进行位级测试和特殊值检查
    TestCaseBuilder<T, U>& set(bool bitwise, bool checkSpecialValues) {
        _case.bitwise = bitwise;
        _case.specialCheck = checkSpecialValues;
        return *this;
    }
    
    // 设置测试种子
    TestCaseBuilder<T, U>& setTestSeed(TestSeed seed) {
        _case.testSeed = seed;
        return *this;
    }
    
    // 设置试验次数
    TestCaseBuilder<T, U>& setTrialCount(size_t trial_count) {
        _case.trials = trial_count;
        return *this;
    }
    
    // 添加域检查
    TestCaseBuilder<T, U>& addDomain(const CheckWithinDomains<U>& domainCheck) {
        _case.domains.emplace_back(domainCheck);
        return *this;
    }
    
    // 添加自定义检查
    TestCaseBuilder<T, U>& addCustom(const CustomCheck<T>& customArgs) {
        _case.customCheck.emplace_back(customArgs);
        return *this;
    }
    
    // 设置检查特殊值
    TestCaseBuilder<T, U>& checkSpecialValues() {
        _case.specialCheck = true;
        return *this;
    }
    
    // 设置进行位级比较
    TestCaseBuilder<T, U>& compareBitwise() {
        _case.bitwise = true;
        return *this;
    }
    
    // 将构建好的测试用例对象移动返回
    operator TestingCase<T, U> && () { return std::move(_case); }
};

// 模板函数：校正 epsilon
// eps: epsilon 值
template <typename T>
typename std::enable_if_t<!is_complex<T>::value&& std::is_unsigned<T>::value, T>
correctEpsilon(const T& eps)
{
    return eps;
}
template <typename T>
# 当模板参数 T 不是复数类型且不是无符号类型时，返回 eps 的绝对值
typename std::enable_if_t<!is_complex<T>::value && !std::is_unsigned<T>::value, T>
correctEpsilon(const T& eps)
{
    return std::abs(eps);
}

# 当模板参数 T 是复数类型时，返回 eps 的绝对值
template <typename T>
typename std::enable_if_t<is_complex<Complex<T>>::value, T>
correctEpsilon(const Complex<T>& eps)
{
    return std::abs(eps);
}

# AssertVectorized 类的构造函数，用于初始化断言对象
class AssertVectorized
{
public:
    # 构造函数，接受详细信息 info、测试种子 seed、期望值 expected、实际值 actual、输入值 input0
    AssertVectorized(const std::string& info, TestSeed seed, const T& expected, const T& actual, const T& input0)
        : additionalInfo(info), testSeed(seed), exp(expected), act(actual), arg0(input0), argSize(1)
    {
    }
    
    # 构造函数，接受详细信息 info、测试种子 seed、期望值 expected、实际值 actual、输入值 input0 和 input1
    AssertVectorized(const std::string& info, TestSeed seed, const T& expected, const T& actual, const T& input0, const T& input1)
        : additionalInfo(info), testSeed(seed), exp(expected), act(actual), arg0(input0), arg1(input1), argSize(2)
    {
    }
    
    # 构造函数，接受详细信息 info、测试种子 seed、期望值 expected、实际值 actual、输入值 input0、input1 和 input2
    AssertVectorized(const std::string& info, TestSeed seed, const T& expected, const T& actual, const T& input0, const T& input1, const T& input2)
        : additionalInfo(info), testSeed(seed), exp(expected), act(actual), arg0(input0), arg1(input1), arg2(input2), argSize(3)
    {
    }
    
    # 构造函数，接受详细信息 info、期望值 expected、实际值 actual，不包含测试种子
    AssertVectorized(const std::string& info, const T& expected, const T& actual)
        : additionalInfo(info), exp(expected), act(actual), hasSeed(false)
    {
    }

    # 返回详细信息的字符串，包括失败详情、附加信息、如果有测试种子则包括测试种子、如果有参数则列出参数、期望值和实际值
    std::string getDetail(int index) const
    {
        using UVT = UvalueType<T>;
        std::stringstream stream;
        stream.precision(std::numeric_limits<UVT>::max_digits10);
        stream << "Failure Details:\n";
        stream << additionalInfo << "\n";
        if (hasSeed)
        {
            stream << "Test Seed to reproduce: " << testSeed << "\n";
        }
        if (argSize > 0)
        {
            stream << "Arguments:\n";
            stream << "#\t " << arg0 << "\n";
            if (argSize == 2)
            {
                stream << "#\t " << arg1 << "\n";
            }
            if (argSize == 3)
            {
                stream << "#\t " << arg2 << "\n";
            }
        }
        stream << "Expected:\n#\t" << exp << "\nActual:\n#\t" << act;
        stream << "\nFirst mismatch Index: " << index;
        return stream.str();
    }

    # 检查断言条件是否满足，支持按位检查和容差检查
    bool check(bool bitwise = false, bool checkWithTolerance = false, ValueType<T> toleranceEps = {}) const
    {
        // 定义类型别名，UVT 为 T 的值类型，BVT 为 UVT 的位类型
        using UVT = UvalueType<T>;
        using BVT = BitType<UVT>;
    
        // 计算绝对误差的阈值
        UVT absErr = correctEpsilon(toleranceEps);
    
        // 计算数组大小并设置存储单元数
        constexpr int sizeX = VecTypeHelper<T>::holdCount * VecTypeHelper<T>::unitStorageCount;
        constexpr int unitStorageCount = VecTypeHelper<T>::unitStorageCount;
    
        // 声明并对齐缓存数组
        CACHE_ALIGN UVT expArr[sizeX];
        CACHE_ALIGN UVT actArr[sizeX];
    
        // 将期望值和实际值存储到对应数组中
        exp.store(expArr);
        act.store(actArr);
    
        // 如果按位比较
        if (bitwise)
        {
            // 遍历数组并按位比较
            for (const auto i : c10::irange(sizeX)) {
                // 将 UVT 类型的数据转换为对应的位类型
                BVT b_exp = c10::bit_cast<BVT>(expArr[i]);
                BVT b_act = c10::bit_cast<BVT>(actArr[i]);
                // 断言两个位类型是否相等，若不相等输出详细信息
                EXPECT_EQ(b_exp, b_act) << getDetail(i / unitStorageCount);
                // 如果测试失败，则返回 true
                if (::testing::Test::HasFailure())
                    return true;
            }
        }
        // 如果使用容差值比较
        else if (checkWithTolerance)
        {
            // 遍历数组并使用容差值进行比较
            for (const auto i : c10::irange(sizeX)) {
                // 检查 UVT 类型的数据是否在给定的绝对误差内接近，若不接近输出详细信息
                EXPECT_EQ(nearlyEqual<UVT>(expArr[i], actArr[i], absErr), true) << expArr[i] << "!=" << actArr[i] << "\n" << getDetail(i / unitStorageCount);
                // 如果测试失败，则返回 true
                if (::testing::Test::HasFailure())
                    return true;
            }
        }
        // 如果不进行特殊比较
        else
        {
            // 遍历数组并根据 UVT 类型选择合适的比较方式
            for (const auto i : c10::irange(sizeX)) {
                // 如果是 float 类型且不是同时为 NaN，则使用浮点数比较
                if constexpr (std::is_same_v<UVT, float>)
                {
                    if (!check_both_nan(expArr[i], actArr[i])) {
                        EXPECT_FLOAT_EQ(expArr[i], actArr[i]) << getDetail(i / unitStorageCount);
                    }
                }
                // 如果是 double 类型且不是同时为 NaN，则使用双精度浮点数比较
                else if constexpr (std::is_same_v<UVT, double>)
                {
                    if (!check_both_nan(expArr[i], actArr[i]))
                    {
                        EXPECT_DOUBLE_EQ(expArr[i], actArr[i]) << getDetail(i / unitStorageCount);
                    }
                }
                // 其他类型直接比较值
                else
                {
                    EXPECT_EQ(expArr[i], actArr[i]) << getDetail(i / unitStorageCount);
                }
                // 如果测试失败，则返回 true
                if (::testing::Test::HasFailure())
                    return true;
            }
        }
        // 所有测试通过，返回 false
        return false;
    }
private:
    std::string additionalInfo;  // 附加信息字段，存储额外的信息
    TestSeed testSeed;  // 测试种子对象，用于生成随机数种子
    T exp;  // 期望值对象
    T act;  // 实际值对象
    T arg0;  // 第一个参数对象
    T arg1;  // 第二个参数对象
    T arg2;  // 第三个参数对象
    int argSize = 0;  // 参数大小，默认为零
    bool hasSeed = true;  // 是否具有种子，默认为真
};

template< typename T, typename Op1, typename Op2, typename Filter = std::nullptr_t>
void test_unary(
    std::string testNameInfo,  // 测试名称信息字符串
    Op1 expectedFunction,  // 期望函数对象
    Op2 actualFunction,  // 实际函数对象
    const TestingCase<T>& testCase,  // 测试用例对象引用
    Filter filter = {}) {  // 过滤器对象，默认为空指针
    using vec_type = T;  // 使用模板参数 T 作为向量类型 vec_type
    using VT = ValueType<T>;  // T 类型的值类型 VT
    using UVT = UvalueType<T>;  // T 类型的无符号值类型 UVT
    constexpr int el_count = vec_type::size();  // 向量类型 vec_type 的大小

    CACHE_ALIGN VT vals[el_count];  // 缓存对齐的值数组 vals
    CACHE_ALIGN VT expected[el_count];  // 缓存对齐的期望值数组 expected
    bool bitwise = testCase.isBitwise();  // 是否按位比较，根据测试用例的位运算标志决定
    UVT default_start = std::is_floating_point_v<UVT> ? std::numeric_limits<UVT>::lowest() : std::numeric_limits<UVT>::min();  // 默认起始值，根据 UVT 类型判断
    UVT default_end = std::numeric_limits<UVT>::max();  // 默认结束值，UVT 类型的最大值
    auto domains = testCase.getDomains();  // 获取测试用例的域
    auto domains_size = domains.size();  // 域的数量
    auto test_trials = testCase.getTrialCount();  // 获取测试用例的试验次数
    int trialCount = getTrialCount<UVT>(test_trials, domains_size);  // 根据域大小和试验次数确定的试验总次数
    TestSeed seed = testCase.getTestSeed();  // 获取测试用例的种子
    uint64_t changeSeedBy = 0;  // 种子增加量初始化为零

    // 对每个域进行循环
    for (const CheckWithinDomains<UVT>& dmn : domains) {
        size_t dmn_argc = dmn.ArgsDomain.size();  // 当前域的参数个数
        UVT start = dmn_argc > 0 ? dmn.ArgsDomain[0].start : default_start;  // 起始值根据域的第一个参数或默认起始值确定
        UVT end = dmn_argc > 0 ? dmn.ArgsDomain[0].end : default_end;  // 结束值根据域的第一个参数或默认结束值确定
        ValueGen<VT> generator(start, end, seed.add(changeSeedBy));  // 值生成器对象，根据起始值、结束值和调整后的种子生成
        for (C10_UNUSED const auto trial : c10::irange(trialCount)) {  // 循环每个试验次数
            for (const auto k : c10::irange(el_count)) {  // 循环每个向量元素
                vals[k] = generator.get();  // 生成值并存储到 vals 数组
                call_filter(filter, vals[k]);  // 调用过滤器处理当前值
                // map 操作符，计算期望值并存储到 expected 数组
                expected[k] = expectedFunction(vals[k]);
            }
            // 测试阶段，加载输入向量并调用实际函数
            auto input = vec_type::loadu(vals);
            auto actual = actualFunction(input);
            auto vec_expected = vec_type::loadu(expected);
            // 断言向量化结果的一致性
            AssertVectorized<vec_type> vecAssert(testNameInfo, seed, vec_expected, actual, input);
            // 检查向量化断言，如果符合要求则返回
            if (vecAssert.check(bitwise, dmn.CheckWithTolerance, dmn.ToleranceError)) return;
        }  // 试验循环结束
        // 增加种子的变化量
        changeSeedBy += 1;
    }  // 域循环结束

    // 对于每个自定义检查，执行以下操作
    for (auto& custom : testCase.getCustomChecks()) {
        auto args = custom.Args;
        if (args.size() > 0) {
            auto input = vec_type{ args[0] };  // 使用自定义参数创建输入向量
            auto actual = actualFunction(input);  // 调用实际函数计算结果
            auto vec_expected = vec_type{ custom.expectedResult };  // 使用自定义的期望结果创建期望向量
            // 断言向量化结果的一致性
            AssertVectorized<vec_type> vecAssert(testNameInfo, seed, vec_expected, actual, input);
            // 检查向量化断言，如果符合要求则返回
            if (vecAssert.check()) return;
        }
    }
}

template< typename T, typename Op1, typename Op2, typename Filter = std::nullptr_t>
void test_binary(
    std::string testNameInfo,  // 测试名称信息字符串
    Op1 expectedFunction,  // 期望函数对象
    Op2 actualFunction,  // 实际函数对象
    const TestingCase<T>& testCase,  // 测试用例对象引用
    Filter filter = {}) {  // 过滤器对象，默认为空指针
    using vec_type = T;  // 使用模板参数 T 作为向量类型 vec_type
    using VT = ValueType<T>;  // T 类型的值类型 VT
    using UVT = UvalueType<T>;  // T 类型的无符号值类型 UVT
    constexpr int el_count = vec_type::size();  // 向量类型 vec_type 的大小
    // 声明三个数组，用于存储测试数据
    CACHE_ALIGN VT vals0[el_count];
    CACHE_ALIGN VT vals1[el_count];
    CACHE_ALIGN VT expected[el_count];
    
    // 判断是否进行位运算
    bool bitwise = testCase.isBitwise();
    
    // 确定默认起始值和结束值
    UVT default_start = std::is_floating_point_v<UVT> ? std::numeric_limits<UVT>::lowest() : std::numeric_limits<UVT>::min();
    UVT default_end = std::numeric_limits<UVT>::max();
    
    // 获取测试用例中的域
    auto domains = testCase.getDomains();
    auto domains_size = domains.size();
    
    // 获取测试用例的试验次数
    auto test_trials = testCase.getTrialCount();
    
    // 计算试验的次数
    int trialCount = getTrialCount<UVT>(test_trials, domains_size);
    
    // 获取测试用例的种子值
    TestSeed seed = testCase.getTestSeed();
    
    // 初始化种子值变更量
    uint64_t changeSeedBy = 0;
    
    // 遍历测试用例中的域
    for (const CheckWithinDomains<UVT>& dmn : testCase.getDomains()) {
        // 获取当前域的参数数量
        size_t dmn_argc = dmn.ArgsDomain.size();
        
        // 确定第一个参数的起始值和结束值
        UVT start0 = dmn_argc > 0 ? dmn.ArgsDomain[0].start : default_start;
        UVT end0 = dmn_argc > 0 ? dmn.ArgsDomain[0].end : default_end;
        
        // 确定第二个参数的起始值和结束值
        UVT start1 = dmn_argc > 1 ? dmn.ArgsDomain[1].start : default_start;
        UVT end1 = dmn_argc > 1 ? dmn.ArgsDomain[1].end : default_end;
        
        // 创建第一个数值生成器
        ValueGen<VT> generator0(start0, end0, seed.add(changeSeedBy));
        
        // 创建第二个数值生成器
        ValueGen<VT> generator1(start1, end1, seed.add(changeSeedBy + 1));
        
        // 对每个试验进行循环
        for (C10_UNUSED const auto trial : c10::irange(trialCount)) {
            // 对每个元素进行循环
            for (const auto k : c10::irange(el_count)) {
                // 生成第一个值并存储
                vals0[k] = generator0.get();
                
                // 生成第二个值并存储
                vals1[k] = generator1.get();
                
                // 调用过滤器函数
                call_filter(filter, vals0[k], vals1[k]);
                
                // 使用期望函数计算预期值
                expected[k] = expectedFunction(vals0[k], vals1[k]);
            }
            
            // 将数组转换为向量类型
            auto input0 = vec_type::loadu(vals0);
            auto input1 = vec_type::loadu(vals1);
            
            // 调用实际函数并存储结果
            auto actual = actualFunction(input0, input1);
            
            // 将期望的向量化结果转换为向量类型
            auto vec_expected = vec_type::loadu(expected);
            
            // 断言向量化结果
            AssertVectorized<vec_type> vecAssert(testNameInfo, seed, vec_expected, actual, input0, input1);
            
            // 检查向量化结果是否符合预期，如果符合则返回
            if (vecAssert.check(bitwise, dmn.CheckWithTolerance, dmn.ToleranceError)) return;
        } // trial
        
        // 更新种子值变更量
        changeSeedBy += 1;
    }
    
    // 针对测试用例中的自定义检查进行循环
    for (auto& custom : testCase.getCustomChecks()) {
        // 获取自定义检查的参数
        auto args = custom.Args;
        
        // 如果参数数量大于零，则创建第一个向量类型
        if (args.size() > 0) {
            auto input0 = vec_type{ args[0] };
            
            // 如果参数数量大于1，则创建第二个向量类型；否则使用第一个参数创建第二个向量类型
            auto input1 = args.size() > 1 ? vec_type{ args[1] } : vec_type{ args[0] };
            
            // 调用实际函数并存储结果
            auto actual = actualFunction(input0, input1);
            
            // 创建期望的向量类型结果
            auto vec_expected = vec_type(custom.expectedResult);
            
            // 断言向量化结果
            AssertVectorized<vec_type> vecAssert(testNameInfo, seed, vec_expected, actual, input0, input1);
            
            // 检查向量化结果是否符合预期，如果符合则返回
            if (vecAssert.check()) return;
        }
    }
}

template< typename T, typename Op1, typename Op2, typename Filter = std::nullptr_t>
void test_ternary(
    std::string testNameInfo,
    Op1 expectedFunction,
    Op2 actualFunction, const TestingCase<T>& testCase, Filter filter = {}) {
    using vec_type = T;  // 定义类型 T 的别名为 vec_type
    using VT = ValueType<T>;  // T 类型的值类型
    using UVT = UvalueType<T>;  // T 类型的无符号值类型
    constexpr int el_count = vec_type::size();  // 获取类型 T 的元素数量
    CACHE_ALIGN VT vals0[el_count];  // 缓存对齐的数组 vals0，用于存储类型 T 的值
    CACHE_ALIGN VT vals1[el_count];  // 缓存对齐的数组 vals1，用于存储类型 T 的值
    CACHE_ALIGN VT vals2[el_count];  // 缓存对齐的数组 vals2，用于存储类型 T 的值
    CACHE_ALIGN VT expected[el_count];  // 缓存对齐的数组 expected，用于存储预期结果的类型 T 的值
    bool bitwise = testCase.isBitwise();  // 检查是否进行位操作
    UVT default_start = std::is_floating_point_v<UVT> ? std::numeric_limits<UVT>::lowest() : std::numeric_limits<UVT>::min();  // 默认起始值根据类型 UVT 确定
    UVT default_end = std::numeric_limits<UVT>::max();  // 默认结束值为类型 UVT 的最大值
    auto domains = testCase.getDomains();  // 获取测试案例的域
    auto domains_size = domains.size();  // 获取域的数量
    auto test_trials = testCase.getTrialCount();  // 获取测试次数
    int trialCount = getTrialCount<UVT>(test_trials, domains_size);  // 计算实际的测试次数
    TestSeed seed = testCase.getTestSeed();  // 获取测试种子
    uint64_t changeSeedBy = 0;  // 初始化种子变化值

    for (const CheckWithinDomains<UVT>& dmn : testCase.getDomains()) {  // 遍历测试案例的域
        size_t dmn_argc = dmn.ArgsDomain.size();  // 获取当前域的参数数量
        UVT start0 = dmn_argc > 0 ? dmn.ArgsDomain[0].start : default_start;  // 确定生成器0的起始值
        UVT end0 = dmn_argc > 0 ? dmn.ArgsDomain[0].end : default_end;  // 确定生成器0的结束值
        UVT start1 = dmn_argc > 1 ? dmn.ArgsDomain[1].start : default_start;  // 确定生成器1的起始值
        UVT end1 = dmn_argc > 1 ? dmn.ArgsDomain[1].end : default_end;  // 确定生成器1的结束值
        UVT start2 = dmn_argc > 2 ? dmn.ArgsDomain[2].start : default_start;  // 确定生成器2的起始值
        UVT end2 = dmn_argc > 2 ? dmn.ArgsDomain[2].end : default_end;  // 确定生成器2的结束值
        ValueGen<VT> generator0(start0, end0, seed.add(changeSeedBy));  // 使用起始值和种子创建生成器0
        ValueGen<VT> generator1(start1, end1, seed.add(changeSeedBy + 1));  // 使用起始值和种子创建生成器1
        ValueGen<VT> generator2(start2, end2, seed.add(changeSeedBy + 2));  // 使用起始值和种子创建生成器2

        for (C10_UNUSED const auto trial : c10::irange(trialCount)) {  // 循环执行测试次数
            for (const auto k : c10::irange(el_count)) {  // 遍历元素数量
                vals0[k] = generator0.get();  // 获取生成器0的下一个值
                vals1[k] = generator1.get();  // 获取生成器1的下一个值
                vals2[k] = generator2.get();  // 获取生成器2的下一个值
                call_filter(filter, vals0[k], vals1[k], vals2[k]);  // 调用过滤器对生成的值进行过滤
                //map operator  // 执行映射操作
                expected[k] = expectedFunction(vals0[k], vals1[k], vals2[k]);  // 计算期望值并存储在数组 expected 中
            }
            // test  // 执行测试
            auto input0 = vec_type::loadu(vals0);  // 加载 vals0 到 SIMD 向量 input0
            auto input1 = vec_type::loadu(vals1);  // 加载 vals1 到 SIMD 向量 input1
            auto input2 = vec_type::loadu(vals2);  // 加载 vals2 到 SIMD 向量 input2
            auto actual = actualFunction(input0, input1, input2);  // 执行实际函数操作
            auto vec_expected = vec_type::loadu(expected);  // 加载 expected 到 SIMD 向量 vec_expected
            AssertVectorized<vec_type> vecAssert(testNameInfo, seed, vec_expected, actual, input0, input1, input2);  // 断言向量化操作结果
            if (vecAssert.check(bitwise, dmn.CheckWithTolerance, dmn.ToleranceError)) return;  // 检查向量化断言是否通过，若通过则返回
        }  // trial
        changeSeedBy += 1;  // 更新种子变化值
    }
}

template <typename T, typename Op>
T func_cmp(Op call, T v0, T v1) {
    using bit_rep = BitType<T>;  // 定义类型 T 的位表示类型为 bit_rep
    constexpr bit_rep mask = std::numeric_limits<bit_rep>::max();  // 计算类型 T 的位表示的最大值
    // 使用位表示形式构造一个值 `ret`，如果 `call(v0, v1)` 返回真，则 `ret` 设为 `mask`，否则设为 0。
    bit_rep ret = call(v0, v1) ? mask : 0;
    // 将 `ret` 强制转换为类型 `T`，这里使用了 C++17 中的 `bit_cast` 来保证位级别的转换。
    return c10::bit_cast<T>(ret);
`
// 结构体 PreventFma，用于禁用 FMA 指令集，实现浮点数的减法和加法操作
struct PreventFma
{
    // 非内联函数，实现浮点数 a 和 b 的减法
    not_inline float sub(float a, float b)
    {
        return a - b;
    }

    // 非内联函数，实现双精度浮点数 a 和 b 的减法
    not_inline double sub(double a, double b)
    {
        return a - b;
    }

    // 非内联函数，实现浮点数 a 和 b 的加法
    not_inline float add(float a, float b)
    {
        return a + b;
    }

    // 非内联函数，实现双精度浮点数 a 和 b 的加法
    not_inline double add(double a, double b)
    {
        return a + b;
    }
};

// 模板函数，如果 T 不是复数类型，则返回 T 类型的参数 x 的对数
template <typename T>
std::enable_if_t<!is_complex<T>::value, T> local_log2(T x) {
    return std::log2(x);
}

// 模板函数，如果 T 是复数类型，则返回 Complex<T> 类型参数 x 的对数
template <typename T>
std::enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_log2(Complex<T> x) {
    T ret = std::log(x); // 计算 x 的自然对数
    T real = ret.real() / std::log(static_cast<T>(2)); // 计算实部的对数
    T imag = ret.imag() / std::log(static_cast<T>(2)); // 计算虚部的对数
    return Complex<T>(real, imag); // 返回以二为底的对数形式的复数
}

// 模板函数，如果 T 不是复数类型，则返回 T 类型参数 x 的绝对值
template <typename T>
std::enable_if_t<!is_complex<T>::value, T> local_abs(T x) {
    return std::abs(x);
}

// 模板函数，如果 T 是复数类型，则返回 Complex<T> 类型参数 x 的绝对值
template <typename T>
std::enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_abs(Complex<T> x) {
#if defined(TEST_AGAINST_DEFAULT)
    return std::abs(x); // 如果定义了 TEST_AGAINST_DEFAULT，则返回 x 的绝对值
#else
    PreventFma noFma; // 创建禁用 FMA 的对象
    T real = x.real(); // 获取复数 x 的实部
    T imag = x.imag(); // 获取复数 x 的虚部
    T rr = real * real; // 实部平方
    T ii = imag * imag; // 虚部平方
    T abs = std::sqrt(noFma.add(rr, ii)); // 计算禁用 FMA 的方式下的绝对值
    return Complex<T>(abs, 0); // 返回绝对值构成的复数对象
#endif
}

// 模板函数，如果 T 不是复数类型，则返回两个 T 类型参数 x 和 y 的乘积
template <typename T>
std::enable_if_t<!is_complex<T>::value, T> local_multiply(T x, T y) {
    return x * y;
}

// 模板函数，如果 T 是复数类型，则返回两个 Complex<T> 类型参数 x 和 y 的乘积
template <typename T>
std::enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_multiply(Complex<T> x, Complex<T> y) {
#if defined(TEST_AGAINST_DEFAULT)
    return x * y; // 如果定义了 TEST_AGAINST_DEFAULT，则返回复数 x 和 y 的乘积
#else
    T x_real = x.real(); // 获取复数 x 的实部
    T x_imag = x.imag(); // 获取复数 x 的虚部
    T y_real = y.real(); // 获取复数 y 的实部
    T y_imag = y.imag(); // 获取复数 y 的虚部
#if defined(CPU_CAPABILITY_VSX) || defined(CPU_CAPABILITY_ZVECTOR)
    // 根据 CPU 的支持情况，选择是否使用特定指令集进行乘法操作
    T rr = x_real * y_real;
    T ii = x_imag * y_real;
    T neg_imag = -y_imag;
    rr = fma(x_imag, neg_imag, rr);
    ii = fma(x_real, y_imag, ii);
#else
    // 使用普通的乘法方式进行计算
    PreventFma noFma;
    T ac = x_real * y_real;
    T bd = x_imag * y_imag;
    T ad = x_real * y_imag;
    T bc = x_imag * (-y_real);
    T rr = noFma.sub(ac, bd);
    T ii = noFma.sub(ad, bc);
#endif
    return Complex<T>(rr, ii); // 返回乘积构成的复数对象
#endif
}

// 模板函数，如果 T 不是复数类型，则返回两个 T 类型参数 x 和 y 的除法结果
template <typename T>
std::enable_if_t<!is_complex<T>::value, T> local_division(T x, T y) {
    return x / y;
}

// 模板函数，如果 T 是复数类型，则返回两个 Complex<T> 类型参数 x 和 y 的除法结果
template <typename T>
std::enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_division(Complex<T> x, Complex<T> y) {
#if defined(TEST_AGAINST_DEFAULT)
    return x / y; // 如果定义了 TEST_AGAINST_DEFAULT，则返回复数 x 和 y 的除法结果
#else /* defined(TEST_AGAINST_DEFAULT) */
    T x_real = x.real(); // 获取复数 x 的实部
    T x_imag = x.imag(); // 获取复数 x 的虚部
    T y_real = y.real(); // 获取复数 y 的实部
    T y_imag = y.imag(); // 获取复数 y 的虚部
    PreventFma noFma; // 创建禁用 FMA 的对象
#if defined(CPU_CAPABILITY_ZVECTOR)
    T abs_c = std::abs(y_real); // 计算 y 的实部的绝对值
    T abs_d = std::abs(y_imag); // 计算 y 的虚部的绝对值
    T scale = 1.0 / std::max(abs_c, abs_d); // 计算比例因子

    T a_sc = x_real * scale; // 实部乘以比例因子
    T b_sc = x_imag * scale; // 虚部乘以比例因子
    T c_sc = y_real * scale; // y 的实部乘以比例因子
#endif
    T d_sc = y_imag * scale; // 计算 y_imag * scale，结果存入 d_sc

    T ac_sc2 = a_sc * c_sc; // 计算 a_sc * c_sc，结果存入 ac_sc2
    T bd_sc2 = b_sc * d_sc; // 计算 b_sc * d_sc，结果存入 bd_sc2

    T neg_d_sc = -1.0 * d_sc; // 计算 -1.0 * d_sc，结果存入 neg_d_sc

    T neg_ad_sc2 = a_sc * neg_d_sc; // 计算 a_sc * neg_d_sc，结果存入 neg_ad_sc2
    T bc_sc2 = b_sc * c_sc; // 计算 b_sc * c_sc，结果存入 bc_sc2

    T ac_bd_sc2 = noFma.add(ac_sc2, bd_sc2); // 计算 ac_sc2 + bd_sc2，结果存入 ac_bd_sc2
    T bc_ad_sc2 = noFma.add(bc_sc2, neg_ad_sc2); // 计算 bc_sc2 + neg_ad_sc2，结果存入 bc_ad_sc2

    T c2_sc2 = c_sc * c_sc; // 计算 c_sc * c_sc，结果存入 c2_sc2
    T d2_sc2 = d_sc * d_sc; // 计算 d_sc * d_sc，结果存入 d2_sc2

    T c2_d2_sc2 = noFma.add(c2_sc2, d2_sc2); // 计算 c2_sc2 + d2_sc2，结果存入 c2_d2_sc2

    T rr = ac_bd_sc2 / c2_d2_sc2; // 计算 ac_bd_sc2 / c2_d2_sc2，结果存入 rr
    T ii = bc_ad_sc2 / c2_d2_sc2; // 计算 bc_ad_sc2 / c2_d2_sc2，结果存入 ii

    return Complex<T>(rr, ii); // 返回由 rr 和 ii 构成的复数对象
#else /* defined(CPU_CAPABILITY_ZVECTOR) */
#if defined(CPU_CAPABILITY_VSX)
    // 对乘法操作进行检查，考虑交换和FMA（fused multiply-add）
    T rr = x_real * y_real;  // 实部的乘积
    T ii = x_imag * y_real;  // 虚部与实部的乘积
    T neg_imag = -y_imag;    // 虚部的负值
    rr = fma(x_imag, y_imag, rr);  // 使用FMA计算实部的最终结果
    ii = fma(x_real, neg_imag, ii);  // 使用FMA计算虚部的最终结果
    // b.abs_2
#else /* defined(CPU_CAPABILITY_VSX) */
    T ac = x_real * y_real;    // 实部的乘积
    T bd = x_imag * y_imag;    // 虚部的乘积
    T ad = x_real * y_imag;    // 实部与虚部的乘积
    T bc = x_imag * y_real;    // 虚部与实部的乘积
    T rr = noFma.add(ac, bd);  // 实部的最终结果
    T ii = noFma.sub(bc, ad);  // 虚部的最终结果
#endif /* defined(CPU_CAPABILITY_VSX) */
    // b.abs_2()
    T abs_rr = y_real * y_real;  // 实部的绝对值平方
    T abs_ii = y_imag * y_imag;  // 虚部的绝对值平方
    T abs_2 = noFma.add(abs_rr, abs_ii);  // 绝对值平方的和
    rr = rr / abs_2;  // 实部除以绝对值平方的和
    ii = ii / abs_2;  // 虚部除以绝对值平方的和
    return Complex<T>(rr, ii);  // 返回复数对象，其实部为rr，虚部为ii
#endif /* defined(CPU_CAPABILITY_ZVECTOR) */
#endif /* defined(TEST_AGAINST_DEFAULT) */
}
    # 提取val0的虚部
    T imag1 = val0.imag();
    # 提取val1的实部
    T real2 = val1.real();
    # 提取val1的虚部
    T imag2 = val1.imag();
    # 使用位级转换将real1和real2进行位或操作，得到复数实部的位表示
    bit_rep real_ret = c10::bit_cast<bit_rep>(real1) | c10::bit_cast<bit_rep>(real2);
    # 使用位级转换将imag1和imag2进行位或操作，得到复数虚部的位表示
    bit_rep imag_ret = c10::bit_cast<bit_rep>(imag1) | c10::bit_cast<bit_rep>(imag2);
    # 根据位表示的复数实部和虚部构造并返回复数对象
    return Complex<T>(c10::bit_cast<T> (real_ret), c10::bit_cast<T>(imag_ret));
// 当 T 不是复数类型时，计算两个值的异或操作并返回结果
template<typename T>
std::enable_if_t<!is_complex<T>::value, T>
local_xor(const T& val0, const T& val1) {
    // 使用 BitType<T> 定义的位表示类型
    using bit_rep = BitType<T>;
    // 将 val0 和 val1 转换为位表示，进行异或操作
    bit_rep ret = c10::bit_cast<bit_rep>(val0) ^ c10::bit_cast<bit_rep>(val1);
    // 将结果转换回类型 T 并返回
    return c10::bit_cast<T> (ret);
}

// 当 T 是复数类型 Complex<T> 时，计算两个复数的每个部分的异或操作并返回结果
template<typename T>
std::enable_if_t<is_complex<Complex<T>>::value, Complex<T>>
local_xor(const Complex<T>& val0, const Complex<T>& val1) {
    // 使用 BitType<T> 定义的位表示类型
    using bit_rep = BitType<T>;
    // 分别提取 val0 和 val1 的实部和虚部
    T real1 = val0.real();
    T imag1 = val0.imag();
    T real2 = val1.real();
    T imag2 = val1.imag();
    // 对实部和虚部的位表示进行异或操作
    bit_rep real_ret = c10::bit_cast<bit_rep>(real1) ^ c10::bit_cast<bit_rep>(real2);
    bit_rep imag_ret = c10::bit_cast<bit_rep>(imag1) ^ c10::bit_cast<bit_rep>(imag2);
    // 构造新的 Complex<T> 对象，返回结果
    return Complex<T>(c10::bit_cast<T> (real_ret), c10::bit_cast<T>(imag_ret));
}

// 将浮点数 value 按照给定的缩放因子和零点量化为类型 T
template <typename T>
T quantize_val(float scale, int64_t zero_point, float value) {
    int64_t qvalue;
    // 获取类型 T 的最小和最大量化值
    constexpr int64_t qmin = std::numeric_limits<T>::min();
    constexpr int64_t qmax = std::numeric_limits<T>::max();
    // 计算缩放因子的倒数
    float inv_scale = 1.0f / scale;
    // 对 value 进行量化并限制在 qmin 和 qmax 之间
    qvalue = static_cast<int64_t>(zero_point + at::native::round_impl<float>(value * inv_scale));
    qvalue = std::max<int64_t>(qvalue, qmin);
    qvalue = std::min<int64_t>(qvalue, qmax);
    // 将结果转换为类型 T 并返回
    return static_cast<T>(qvalue);
}

// 将整数 src 重新量化为类型 T，使用给定的倍数和零点
template <typename T>
#if defined(TEST_AGAINST_DEFAULT)
T requantize_from_int(float multiplier, int32_t zero_point, int32_t src) {
    auto xx = static_cast<float>(src) * multiplier;
    // 对 xx 进行最近整数化
    double xx2 = nearbyint(xx);
    int32_t quantize_down = xx2 + zero_point;
#else
T requantize_from_int(float multiplier, int64_t zero_point, int64_t src) {
    // 根据给定的倍数和零点进行重新量化
    int64_t quantize_down = static_cast<int64_t>(zero_point + std::lrintf(src * multiplier));
#endif
    // 获取类型 T 的最小和最大量化值
    constexpr int64_t min = std::numeric_limits<T>::min();
    constexpr int64_t max = std::numeric_limits<T>::max();
    // 将量化结果限制在 min 和 max 之间，并转换为类型 T 返回
    auto ret = static_cast<T>(std::min<int64_t>(std::max<int64_t>(quantize_down, min), max));
    return ret;
}

// 将类型 T 的量化值反量化为浮点数，使用给定的缩放因子和零点
template <typename T>
float dequantize_val(float scale, int64_t zero_point, T value) {
    // 根据宏定义选择不同的反量化方法
#if defined(CHECK_WITH_FMA)
    // 使用 FMA 指令计算反量化值
    float neg_p = -(zero_point * scale);
    float v = static_cast<float>(value);
    float ret = fma(v, scale, neg_p);
#else
    // 使用标准的反量化计算
    float ret = (static_cast<float>(value) - zero_point) * scale;
#endif
    return ret;
}

// 对输入值 val 进行 ReLU 激活函数处理，如果小于 zero_point 则返回 zero_point
template<typename T>
T relu(const T & val, const T & zero_point) {
    return std::max(val, zero_point);
}

// 对输入值 val 进行 ReLU6 激活函数处理，限制在 [zero_point, q_six] 范围内
template<typename T>
T relu6(T val, T zero_point, T q_six) {
    return std::min<T>(std::max<T>(val, zero_point), q_six);
}

// 宽泛的减法操作，将两个相同类型的值相减并返回结果，结果类型为 int32_t
template<typename T>
int32_t widening_subtract(T val, T b) {
    return static_cast<int32_t>(val) - static_cast<int32_t>(b);
}

// 默认情况下返回类型 T 的默认容差
template<typename T>
T getDefaultTolerance() {
    return static_cast<T>(0.0);
}

// 对 float 类型的特化，返回默认容差 5.e-5f
template<>
float getDefaultTolerance() {
    return 5.e-5f;
}

// 对 double 类型的特化，返回默认容差 1.e-9
template<>
double getDefaultTolerance() {
    return 1.e-9;
}
// 创建一个模板函数，用于生成一个大小为1的向量掩码，其元素类型为T，根据给定的位掩码生成掩码对象
template<typename T>
at::vec::VecMask<T, 1> create_vec_mask(uint64_t bitmask) {
    // 获取类型T的向量化大小N
    constexpr auto N = at::vec::Vectorized<T>::size();
    // 创建一个大小为N的整型数组mask，用于存储位掩码的每一位
    std::array<int, N> mask;
    // 遍历向量化大小N，从位掩码中提取每一位的值并存储到数组mask中
    for (int i = 0; i < N; i++) {
        mask[i] = (bitmask >> i) & 1;
    }
    // 使用数组mask中的数据创建并返回一个大小为1的VecMask对象，类型为T
    return at::vec::VecMask<T, 1>::from(mask.data());
}

// 创建一个模板函数，用于生成一个大小为1的向量掩码，其元素类型为T，基于给定的种子seed生成位掩码
template<typename T>
at::vec::VecMask<T, 1> generate_vec_mask(int seed) {
    // 获取类型T的向量化大小N
    constexpr auto N = at::vec::Vectorized<T>::size();
    // 使用ValueGen类创建一个生成器generator，生成范围在0到(1 << N) - 1之间的随机数位掩码，种子为seed
    ValueGen<uint64_t> generator(0, (1ULL << N) - 1, seed);
    // 获取生成器生成的位掩码
    auto bitmask = generator.get();
    // 调用create_vec_mask函数，使用位掩码生成并返回一个大小为1的VecMask对象，类型为T
    return create_vec_mask<T>(bitmask);
}

// 创建一个模板函数，用于生成默认的一元测试用例，其元素类型为T，可选参数包括种子seed、是否按位进行测试bitwise、是否带有容差检查checkWithTolerance、试验次数trials
template<typename T>
TestingCase<T> createDefaultUnaryTestCase(TestSeed seed = TestSeed(), bool bitwise = false, bool checkWithTolerance = false, size_t trials = 0) {
    // 定义类型UVT为T的无符号值类型
    using UVT = UvalueType<T>;
    // 声明一个空的测试用例testCase
    TestingCase<T> testCase;
    // 如果不进行按位测试且T为浮点类型
    if (!bitwise && std::is_floating_point_v<UVT>) {
        // 设置测试用例为浮点数的默认构建器，添加若干数值范围的域，设置试验次数和种子
        UVT tolerance = getDefaultTolerance<UVT>();
        testCase = TestingCase<T>::getBuilder()
            .set(bitwise, false)
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-10, (UVT)10}}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)10, (UVT)100 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)100, (UVT)1000 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-100, (UVT)-10 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-1000, (UVT)-100 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ {}, checkWithTolerance, tolerance})
            .setTrialCount(trials)
            .setTestSeed(seed);
    }
    else {
        // 否则，设置测试用例为默认构建器，设置试验次数和种子
        testCase = TestingCase<T>::getBuilder()
            .set(bitwise, false)
            .addDomain(CheckWithinDomains<UVT>{})
            .setTrialCount(trials)
            .setTestSeed(seed);
    }
    // 返回生成的测试用例
    return testCase;
}

// 创建一个模板函数，用于生成默认的二元测试用例，其元素类型为T，可选参数包括种子seed、是否按位进行测试bitwise、是否带有容差检查checkWithTolerance、试验次数trials
template<typename T>
TestingCase<T> createDefaultBinaryTestCase(TestSeed seed = TestSeed(), bool bitwise = false, bool checkWithTolerance = false, size_t trials = 0) {
    // 定义类型UVT为T的无符号值类型
    using UVT = UvalueType<T>;
    // 声明一个空的测试用例testCase
    TestingCase<T> testCase;
    if (!bitwise && std::is_floating_point_v<UVT>) {
        // 如果不是位操作且泛型类型 UVT 是浮点型，则添加手动定义的范围
        // 获取默认容差值
        UVT tolerance = getDefaultTolerance<UVT>();
        // 创建测试用例对象，基于 TestingCase<T> 的构建器模式
        testCase = TestingCase<T>::getBuilder()
            // 设置位操作为 false
            .set(bitwise, false)
            // 添加多个数值范围检查域
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-10, (UVT)10}, { (UVT)-10, (UVT)10 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)10, (UVT)100 }, { (UVT)-10, (UVT)100 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)100, (UVT)1000 }, { (UVT)-100, (UVT)1000 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-100, (UVT)-10 }, { (UVT)-100, (UVT)10 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-1000, (UVT)-100 }, { (UVT)-1000, (UVT)100 }}, checkWithTolerance, tolerance})
            // 添加空的检查域，用于其他情况
            .addDomain(CheckWithinDomains<UVT>{ {}, checkWithTolerance, tolerance})
            // 设置试验次数
            .setTrialCount(trials)
            // 设置测试的种子值
            .setTestSeed(seed);
    }
    else {
        // 对于其他情况，创建测试用例对象，基于 TestingCase<T> 的构建器模式
        testCase = TestingCase<T>::getBuilder()
            // 设置位操作为 false
            .set(bitwise, false)
            // 添加空的检查域
            .addDomain(CheckWithinDomains<UVT>{})
            // 设置试验次数
            .setTrialCount(trials)
            // 设置测试的种子值
            .setTestSeed(seed);
    }
    // 返回构建好的测试用例对象
    return testCase;
}

template<typename T>
// 创建默认的三元测试用例，可以指定种子、是否按位检查、是否使用容差检查和试验次数
TestingCase<T> createDefaultTernaryTestCase(TestSeed seed = TestSeed(), bool bitwise = false, bool checkWithTolerance = false, size_t trials = 0) {
    // 使用 TestingCase 类的构建器获取测试用例对象
    TestingCase<T> testCase = TestingCase<T>::getBuilder()
        // 设置按位检查为 false
        .set(bitwise, false)
        // 添加默认域 CheckWithinDomains<UvalueType<T>>
        .addDomain(CheckWithinDomains<UvalueType<T>>{})
        // 设置试验次数
        .setTrialCount(trials)
        // 设置测试种子
        .setTestSeed(seed);
    // 返回构建好的测试用例对象
    return testCase;
}
```