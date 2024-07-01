# `.\numpy\numpy\_core\src\common\half.hpp`

```py
#ifndef NUMPY_CORE_SRC_COMMON_HALF_HPP
#define NUMPY_CORE_SRC_COMMON_HALF_HPP

#include "npstd.hpp"

#include "npy_cpu_dispatch.h" // NPY_HAVE_CPU_FEATURES
#include "half_private.hpp"

// TODO(@seiko2plus):
// - covers half-precision operations that being supported by numpy/halffloat.h
// - add support for arithmetic operations
// - enables __fp16 causes massive FP exceptions on aarch64,
//   needs a deep investigation

namespace np {

/// @addtogroup cpp_core_types
/// @{

/// Provides a type that implements 16-bit floating point (half-precision).
/// This type is ensured to be 16-bit size.
#if 1 // ndef __ARM_FP16_FORMAT_IEEE
class Half final {
  public:
    /// Whether `Half` has a full native HW support.
    static constexpr bool kNative = false;
    /// Whether `Half` has a native HW support for single/double conversion.
    template<typename T>
    static constexpr bool kNativeConversion = (
        (
            std::is_same_v<T, float> &&
        #if defined(NPY_HAVE_FP16) || defined(NPY_HAVE_VSX3)
            true
        #else
            false
        #endif
        ) || (
            std::is_same_v<T, double> &&
        #if defined(NPY_HAVE_AVX512FP16) || (defined(NPY_HAVE_VSX3) && defined(NPY_HAVE_VSX3_HALF_DOUBLE))
            true
        #else
            false
        #endif
        )
    );

    /// Default constructor. initialize nothing.
    Half() = default;

    /// Construct from float
    /// If there are no hardware optimization available, rounding will always
    /// be set to ties to even.
    explicit Half(float f)
    {
    #if defined(NPY_HAVE_FP16)
        // Load float `f` into SSE register
        __m128 mf = _mm_load_ss(&f);
        // Convert float to half-precision and store in bits_
        bits_ = static_cast<uint16_t>(_mm_cvtsi128_si32(_mm_cvtps_ph(mf, _MM_FROUND_TO_NEAREST_INT)));
    #elif defined(NPY_HAVE_VSX3) && defined(NPY_HAVE_VSX_ASM)
        // Load float `f` into vector register
        __vector float vf32 = vec_splats(f);
        __vector unsigned short vf16;
        // Convert float to half-precision using VSX3 instruction set
        __asm__ __volatile__ ("xvcvsphp %x0,%x1" : "=wa" (vf16) : "wa" (vf32));
        #ifdef __BIG_ENDIAN__
        bits_ = vec_extract(vf16, 1); // Extract half-precision value
        #else
        bits_ = vec_extract(vf16, 0); // Extract half-precision value
        #endif
    #else
        // Fallback: convert float `f` to half-precision using software implementation
        bits_ = half_private::FromFloatBits(BitCast<uint32_t>(f));
    #endif
    }

    /// Construct from double.
    /// If there are no hardware optimization available, rounding will always
    /// be set to ties to even.
    explicit Half(double f)
    {
    #if defined(NPY_HAVE_AVX512FP16)
        // Load double `f` into SSE register
        __m128d md = _mm_load_sd(&f);
        // Convert double to half-precision and store in bits_
        bits_ = static_cast<uint16_t>(_mm_cvtsi128_si32(_mm_castph_si128(_mm_cvtpd_ph(md))));
    #elif defined(NPY_HAVE_VSX3) && defined(NPY_HAVE_VSX3_HALF_DOUBLE)
        // Convert double `f` to half-precision using VSX3 instruction set
        __asm__ __volatile__ ("xscvdphp %x0,%x1" : "=wa" (bits_) : "wa" (f));
    #else
        // Fallback: convert double `f` to half-precision using software implementation
        bits_ = half_private::FromDoubleBits(BitCast<uint64_t>(f));
    #endif
    }

    /// Cast to float
    explicit operator float() const
    {
    #if defined(NPY_HAVE_FP16)
        float ret;
        // Convert half-precision to float
        _mm_store_ss(&ret, _mm_cvtph_ps(_mm_cvtsi32_si128(bits_)));
        return ret;
    ```
    #elif defined(NPY_HAVE_VSX3) && defined(vec_extract_fp_from_shorth)
        // 如果定义了 NPY_HAVE_VSX3 和 vec_extract_fp_from_shorth，使用它们来提取特定位数的浮点数
        return vec_extract(vec_extract_fp_from_shorth(vec_splats(bits_)), 0);
    #elif defined(NPY_HAVE_VSX3) && defined(NPY_HAVE_VSX_ASM)
        // 如果定义了 NPY_HAVE_VSX3 和 NPY_HAVE_VSX_ASM，使用内联汇编来执行类型转换操作
        __vector float vf32;
        __asm__ __volatile__("xvcvhpsp %x0,%x1"
                             : "=wa"(vf32)
                             : "wa"(vec_splats(bits_)));
        return vec_extract(vf32, 0);
    #else
        // 否则，使用 BitCast 进行类型转换并返回结果
        return BitCast<float>(half_private::ToFloatBits(bits_));
    #endif
    }

    /// Cast to double
    explicit operator double() const
    {
    #if defined(NPY_HAVE_AVX512FP16)
        // 如果定义了 NPY_HAVE_AVX512FP16，使用 AVX512 指令集进行半精度到双精度的转换
        double ret;
        _mm_store_sd(&ret, _mm_cvtph_pd(_mm_castsi128_ph(_mm_cvtsi32_si128(bits_))));
        return ret;
    #elif defined(NPY_HAVE_VSX3) && defined(NPY_HAVE_VSX3_HALF_DOUBLE)
        // 如果定义了 NPY_HAVE_VSX3 和 NPY_HAVE_VSX3_HALF_DOUBLE，使用 VSX3 指令集进行半精度到双精度的转换
        double f64;
        __asm__ __volatile__("xscvhpdp %x0,%x1"
                             : "=wa"(f64)
                             : "wa"(bits_));
        return f64;
    #else
        // 否则，使用 BitCast 进行类型转换并返回结果
        return BitCast<double>(half_private::ToDoubleBits(bits_));
    #endif
    }

    /// Returns a new Half constructed from the IEEE 754 binary16.
    static constexpr Half FromBits(uint16_t bits)
    {
        // 使用给定的 16 位表示构造一个新的 Half 类型对象
        Half h{};
        h.bits_ = bits;
        return h;
    }
    /// Returns the IEEE 754 binary16 representation.
    constexpr uint16_t Bits() const
    {
        // 返回当前 Half 对象的 16 位表示
        return bits_;
    }

    /// @name Comparison operators (ordered)
    /// @{
    constexpr bool operator==(Half r) const
    {
        // 比较两个 Half 对象是否相等，NaN 时返回 false
        return !(IsNaN() || r.IsNaN()) && Equal(r);
    }
    constexpr bool operator<(Half r) const
    {
        // 比较当前 Half 对象是否小于另一个 Half 对象，NaN 时返回 false
        return !(IsNaN() || r.IsNaN()) && Less(r);
    }
    constexpr bool operator<=(Half r) const
    {
        // 比较当前 Half 对象是否小于等于另一个 Half 对象，NaN 时返回 false
        return !(IsNaN() || r.IsNaN()) && LessEqual(r);
    }
    constexpr bool operator>(Half r) const
    {
        // 比较当前 Half 对象是否大于另一个 Half 对象，NaN 时返回通过反向比较结果
        return r < *this;
    }
    constexpr bool operator>=(Half r) const
    {
        // 比较当前 Half 对象是否大于等于另一个 Half 对象，NaN 时返回通过反向比较结果
        return r <= *this;
    }
    /// @}

    /// @name Comparison operators (unordered)
    /// @{
    constexpr bool operator!=(Half r) const
    {
        // 比较两个 Half 对象是否不相等，NaN 时返回 true
        return !(*this == r);
    }
    /// @} Comparison operators

    /// @name Comparison with no guarantee of NaN behavior
    /// @{
    constexpr bool Less(Half r) const
    {
        // 比较两个 Half 对象的大小，忽略 NaN 的特殊处理
        uint_fast16_t a = static_cast<uint_fast16_t>(bits_),
                      b = static_cast<uint_fast16_t>(r.bits_);
        bool sign_a = (a & 0x8000u) == 0x8000u;
        bool sign_b = (b & 0x8000u) == 0x8000u;
        // 如果两个数符号相同
        //   如果 a 有符号并且 a < b，则返回 true，并确保 a 不等于 b，以避免两者均为 +-0 的情况
        //   如果 a 无符号并且 a < b，则返回 true
        //   否则返回 false
        // 如果两个数符号不同
        //   如果 a 为负数，并且 a 和 b 不全为 0x8000u，则返回 true
        //   否则返回 false
        return (sign_a == sign_b) ? (sign_a ^ (a < b)) && (a != b)
                                  : sign_a && ((a | b) != 0x8000u);
    }
    // 定义一个 constexpr 方法 LessEqual，用于比较当前 Half 对象与参数 r 的大小关系
    constexpr bool LessEqual(Half r) const
    {
        // 将当前对象和参数 r 的 bits_ 转换为 uint_fast16_t 类型
        uint_fast16_t a = static_cast<uint_fast16_t>(bits_),
                      b = static_cast<uint_fast16_t>(r.bits_);
        // 检查当前对象和参数 r 的符号位
        bool sign_a = (a & 0x8000u) == 0x8000u;
        bool sign_b = (b & 0x8000u) == 0x8000u;
        
        // 如果符号相同
        //   测试 `a` > `b` 当 `a` 为正号时
        //       或者 `a` < `b` 当 `a` 不是正号时
        //       或者 a == b （即使上面使用了 <= 也需要测试 +-0 情况）
        // 否则
        //   测试 `a` 是否为正号
        //       或者 `a` 和 `b` 是否都等于 +-0.0
        return (sign_a == sign_b) ? (sign_a ^ (a < b)) || (a == b)
                                  : sign_a || ((a | b) == 0x8000u);
    }
    
    // 定义一个 constexpr 方法 Equal，用于比较当前 Half 对象与参数 r 是否相等
    constexpr bool Equal(Half r) const
    {
        // 不使用 fast16 转换，因为解包操作可能涉及
        uint16_t a = bits_, b = r.bits_;
        // 检查当前对象和参数 r 是否相等
        return a == b || ((a | b) == 0x8000u);
    }
    
    /// @} Comparison
    
    /// @name Properties
    // @{
    
    // 定义一个 constexpr 方法 IsNaN，用于检查当前 Half 对象是否表示 NaN
    constexpr bool IsNaN() const
    {
        // 检查 bits_ 是否符合 NaN 的二进制表示条件
        return ((bits_ & 0x7c00u) == 0x7c00u) &&
               ((bits_ & 0x03ffu) != 0);
    }
    /// @} Properties
    
  private:
    uint16_t bits_;
#else // __ARM_FP16_FORMAT_IEEE
// 如果不是使用 IEEE 格式的 ARM 浮点数，定义一个名为 Half 的类
class Half final {
  public:
    // 表示此类使用本机格式
    static constexpr bool kNative = true;
    // 模板，用于检查 T 是否可以转换为本机格式
    template<typename T>
    static constexpr bool kNativeConversion = (
        std::is_same_v<T, float> || std::is_same_v<T, double>
    );

    // 默认构造函数
    Half() = default;

    // 将 __fp16 转换为 Half 类型的构造函数
    constexpr Half(__fp16 h) : half_(h)
    {}

    // 将 Half 类型转换为 __fp16 类型的隐式转换操作符
    constexpr operator __fp16() const
    { return half_; }

    // 根据给定的位模式创建 Half 类型对象
    static Half FromBits(uint16_t bits)
    {
        Half h;
        h.half_ = BitCast<__fp16>(bits);
        return h;
    }

    // 返回当前 Half 对象的位模式
    uint16_t Bits() const
    { return BitCast<uint16_t>(half_); }

    // 比较当前对象与参数对象的大小关系
    constexpr bool Less(Half r) const
    { return half_ < r.half_; }

    // 比较当前对象与参数对象的大小关系（包含等于）
    constexpr bool LessEqual(Half r) const
    { return half_ <= r.half_; }

    // 检查当前对象是否等于参数对象
    constexpr bool Equal(Half r) const
    { return half_ == r.half_; }

    // 检查当前对象是否表示 NaN（Not a Number）
    constexpr bool IsNaN() const
    { return half_ != half_; }

  private:
    // 内部存储的 __fp16 类型的数据
    __fp16 half_;
};
#endif // __ARM_FP16_FORMAT_IEEE

/// @} cpp_core_types

} // namespace np

#endif // NUMPY_CORE_SRC_COMMON_HALF_HPP
```