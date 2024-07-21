# `.\pytorch\c10\util\TypeCast.h`

```
#pragma once
// 引入必要的头文件，包括C10宏定义和各种数据类型
#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include <type_traits>

// 开始C10命名空间
namespace c10 {

// 定义模板结构体needs_real，用于检测是否需要实数部分
template <typename dest_t, typename src_t>
struct needs_real {
  // 如果源类型是复数且目标类型不是复数，则value为true，否则为false
  constexpr static bool value =
      (is_complex<src_t>::value && !is_complex<dest_t>::value);
};

// 定义模板结构体maybe_real，根据模板参数bool，选择是否返回实部
template <bool, typename src_t>
struct maybe_real {
  // 如果bool为false，直接返回src
  C10_HOST_DEVICE static inline src_t apply(src_t src) {
    return src;
  }
};

// 对于模板参数为true的特化，返回src的实部
template <typename src_t>
struct maybe_real<true, src_t> {
  // 调用src对象的real()方法返回实部
  C10_HOST_DEVICE static inline decltype(auto) apply(src_t src) {
    return src.real();
  }
};

// 定义模板结构体maybe_bool，根据模板参数bool，选择是否返回布尔值
template <bool, typename src_t>
struct maybe_bool {
  // 如果bool为false，直接返回src
  C10_HOST_DEVICE static inline src_t apply(src_t src) {
    return src;
  }
};

// 对于模板参数为true的特化，返回src的实部或虚部是否有非零值
template <typename src_t>
struct maybe_bool<true, src_t> {
  // 如果src对象的实部或虚部有非零值，返回true；否则返回false
  C10_HOST_DEVICE static inline decltype(auto) apply(src_t src) {
    // 不使用bool操作符，以便ComplexHalf类型也能编译通过
    return src.real() || src.imag();
  }
};

// 注意：此处故意忽略未定义行为，与NumPy一致。
// PyTorch的类型转换可能引起多种未定义行为，
// 包括从浮点到整数的溢出，以及从有符号到无符号整数的溢出。
// 下面的代码部分解决了部分未定义行为。

// 定义模板结构体static_cast_with_inter_type，实现静态类型转换
template <typename dest_t, typename src_t>
struct static_cast_with_inter_type {
  // 静态方法apply实现类型转换
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline dest_t apply(
      src_t src) {
    // 检查是否需要处理实数部分
    constexpr bool real = needs_real<dest_t, src_t>::value;
    // 根据real选择是否返回src的实部
    auto r = maybe_real<real, src_t>::apply(src);
    // 将r转换为dest_t类型并返回
    return static_cast<dest_t>(r);
  }
};

// 针对bool类型的部分模板特化
template <typename src_t>
struct static_cast_with_inter_type<bool, src_t> {
  // apply方法实现将src_t类型转换为bool类型
  C10_HOST_DEVICE static inline bool apply(src_t src) {
    // 检查是否需要处理实数部分
    constexpr bool complex = needs_real<bool, src_t>::value;
    // 调用maybe_bool结构体的apply方法返回bool值
    return static_cast<bool>(maybe_bool<complex, src_t>::apply(src));
  }
};

// 针对将src_t类型转换为uint8类型的部分模板实例化
// 注意：在C++中，从负浮点数值转换为无符号整数类型是未定义行为，
// 当前的CPU和GPU编译器表现不一致。将负浮点数值转换为有符号整数类型，
// 然后再转换为无符号整数类型不是未定义行为，因此此转换提高了
// 在各种编译器中将类型转换为uint8的一致性。
template <typename src_t>
struct static_cast_with_inter_type<uint8_t, src_t> {
  // apply方法实现将src_t类型转换为uint8_t类型
  C10_HOST_DEVICE static inline uint8_t apply(src_t src) {
    // 调用maybe_real结构体的apply方法返回src的实部
    auto r = maybe_real<true, src_t>::apply(src);
    // 将r转换为uint8_t类型并返回
    return static_cast<uint8_t>(r);
  }
};

// 此处省略了模板结构体static_cast_with_inter_type的余下部分
// 因为根据要求不允许超出部分

// 结束C10命名空间
} // namespace c10
// 恢复之前的编译器警告设置
C10_CLANG_DIAGNOSTIC_POP()
// 用于处理静态类型转换的模板结构体，转换源类型到 uint8_t 的特化版本
template <typename src_t>
struct static_cast_with_inter_type<uint8_t, src_t> {
    // 在主机和设备端均可使用，并忽略未定义行为检查，执行转换为 uint8_t
    C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline uint8_t apply(
        src_t src) {
        // 确定是否需要实数类型转换
        constexpr bool real = needs_real<uint8_t, src_t>::value;
        // 应用可能的实数类型转换后，再将结果转换为 int64_t，最终转换为 uint8_t
        return static_cast<uint8_t>(
            static_cast<int64_t>(maybe_real<real, src_t>::apply(src)));
    }
};

// 特化模板结构体，将 c10::BFloat16 转换为 c10::complex<c10::Half>
template <>
struct static_cast_with_inter_type<c10::complex<c10::Half>, c10::BFloat16> {
    // 在主机和设备端均可使用，并忽略未定义行为检查，执行转换
    C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
        c10::Half>
    apply(c10::BFloat16 src) {
        // 将 c10::BFloat16 转换为 c10::complex<float>，再转换为 c10::complex<c10::Half>
        return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
    }
};

// 特化模板结构体，将 c10::Float8_e5m2 转换为 c10::complex<c10::Half>
template <>
struct static_cast_with_inter_type<c10::complex<c10::Half>, c10::Float8_e5m2> {
    // 在主机和设备端均可使用，并忽略未定义行为检查，执行转换
    C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
        c10::Half>
    apply(c10::Float8_e5m2 src) {
        // 将 c10::Float8_e5m2 转换为 c10::complex<float>，再转换为 c10::complex<c10::Half>
        return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
    }
};

// 特化模板结构体，将 c10::Float8_e5m2fnuz 转换为 c10::complex<c10::Half>
template <>
struct static_cast_with_inter_type<
    c10::complex<c10::Half>,
    c10::Float8_e5m2fnuz> {
    // 在主机和设备端均可使用，并忽略未定义行为检查，执行转换
    C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
        c10::Half>
    apply(c10::Float8_e5m2fnuz src) {
        // 将 c10::Float8_e5m2fnuz 转换为 c10::complex<float>，再转换为 c10::complex<c10::Half>
        return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
    }
};

// 特化模板结构体，将 c10::Float8_e4m3fn 转换为 c10::complex<c10::Half>
template <>
struct static_cast_with_inter_type<
    c10::complex<c10::Half>,
    c10::Float8_e4m3fn> {
    // 在主机和设备端均可使用，并忽略未定义行为检查，执行转换
    C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
        c10::Half>
    apply(c10::Float8_e4m3fn src) {
        // 将 c10::Float8_e4m3fn 转换为 c10::complex<float>，再转换为 c10::complex<c10::Half>
        return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
    }
};

// 特化模板结构体，将 c10::Float8_e4m3fnuz 转换为 c10::complex<c10::Half>
template <>
struct static_cast_with_inter_type<
    c10::complex<c10::Half>,
    c10::Float8_e4m3fnuz> {
    // 在主机和设备端均可使用，并忽略未定义行为检查，执行转换
    C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
        c10::Half>
    apply(c10::Float8_e4m3fnuz src) {
        // 将 c10::Float8_e4m3fnuz 转换为 c10::complex<float>，再转换为 c10::complex<c10::Half>
        return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
    }
};

// 特化模板结构体，将 c10::Half 转换为 c10::complex<c10::Half>
template <>
struct static_cast_with_inter_type<c10::complex<c10::Half>, c10::Half> {
    // 在主机和设备端均可使用，并忽略未定义行为检查，执行转换
    C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
        c10::Half>
    apply(c10::Half src) {
        // 将 c10::Half 转换为 c10::complex<float>，再转换为 c10::complex<c10::Half>
        return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
    }
};

// 特化模板结构体，将 c10::complex<double> 转换为 c10::complex<c10::Half>
template <>
struct static_cast_with_inter_type<
    c10::complex<c10::Half>,
    c10::complex<double>> {
    // 在主机和设备端均可使用，并忽略未定义行为检查，执行转换
    C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
        c10::Half>
    apply(c10::complex<double> src) {
        // 将 c10::complex<double> 转换为 c10::complex<float>，再转换为 c10::complex<c10::Half>
        return static_cast<c10::complex<c10::Half>>(
            static_cast<c10::complex<float>>(src));
    }
};

// 将 From 类型的参数 f 转换为 To 类型
template <typename To, typename From>
C10_HOST_DEVICE To convert(From f) {
    // 调用 static_cast_with_inter_type 结构体的 apply 函数进行转换，并返回结果
    return static_cast_with_inter_type<To, From>::apply(f);
}

// 报告溢出的函数声明，避免内联以防止代码大小膨胀
[[noreturn]] C10_API void report_overflow(const char* name);

// 模板函数定义，将 From 类型的参数 f 转换为 To 类型
template <typename To, typename From>
To checked_convert(From f, const char* name) {
  // Converting to bool can't overflow so we exclude this case from checking.
  // 如果要转换成 bool，则不可能溢出，因此我们在检查中排除这种情况。
  if (!std::is_same_v<To, bool> && overflows<To, From>(f)) {
    // 如果目标类型不是 bool 并且转换会导致溢出，则报告溢出。
    report_overflow(name);
  }
  // 执行类型转换并返回结果。
  return convert<To, From>(f);
}

} // namespace c10
// 结束命名空间 c10

C10_CLANG_DIAGNOSTIC_POP()
// 恢复之前的 Clang 诊断设置。

// 触发 D25440771 的测试。TODO: 随时可以移除这行。
// 触发针对 D25440771 的测试。TODO: 随时可以移除这行。
```