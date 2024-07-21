# `.\pytorch\c10\util\TypeIndex.h`

```
#pragma once

#include <c10/util/ConstexprCrc.h>  // 包含 ConstexprCrc.h 头文件，提供 constexpr CRC 相关功能
#include <c10/util/IdWrapper.h>     // 包含 IdWrapper.h 头文件，定义 IdWrapper 类模板
#include <c10/util/string_view.h>   // 包含 string_view.h 头文件，提供 string_view 类型支持
#include <cstdint>                  // 包含 stdint.h 头文件，定义整数类型
#include <ostream>                  // 包含 ostream 头文件，定义输出流相关功能
#include <stdexcept>                // 包含 stdexcept 头文件，定义标准异常类
#include <string>                   // 包含 string 头文件，提供字符串相关支持
#include <type_traits>              // 包含 type_traits 头文件，提供类型特性支持

namespace c10::util {

// TODO Make it work for more compilers

// Intel compiler works
#if defined(__INTEL_COMPILER)
#define C10_TYPENAME_SUPPORTS_CONSTEXPR 0     // 如果是 Intel 编译器，则不支持 constexpr
#define C10_TYPENAME_CONSTEXPR                // 空定义

// Clang works
#elif defined(__clang__)

// except for NVCC
#if defined(__CUDACC__)
#define C10_TYPENAME_SUPPORTS_CONSTEXPR 0     // 如果是 Clang 但是在 NVCC 下，则不支持 constexpr
#define C10_TYPENAME_CONSTEXPR
#else
#define C10_TYPENAME_SUPPORTS_CONSTEXPR 1     // 如果是 Clang 并且不在 NVCC 下，则支持 constexpr
#define C10_TYPENAME_CONSTEXPR constexpr
#endif

// Windows works
#elif defined(_MSC_VER)

// except for NVCC
#if defined(__CUDACC__)
#define C10_TYPENAME_SUPPORTS_CONSTEXPR 0     // 如果是 MSVC 但是在 NVCC 下，则不支持 constexpr
#define C10_TYPENAME_CONSTEXPR
#else
#define C10_TYPENAME_SUPPORTS_CONSTEXPR 1     // 如果是 MSVC 并且不在 NVCC 下，则支持 constexpr
#define C10_TYPENAME_CONSTEXPR constexpr
#endif

// GCC works
#elif defined(__GNUC__)

// except when gcc < 9
#if (__GNUC__ < 9) || defined(__CUDACC__)
#define C10_TYPENAME_SUPPORTS_CONSTEXPR 0     // 如果是 GCC 但是版本小于 9 或在 NVCC 下，则不支持 constexpr
#define C10_TYPENAME_CONSTEXPR
#else
#define C10_TYPENAME_SUPPORTS_CONSTEXPR 1     // 如果是 GCC 并且版本 >= 9 且不在 NVCC 下，则支持 constexpr
#define C10_TYPENAME_CONSTEXPR constexpr
#endif

// some other compiler we don't know about
#else
#define C10_TYPENAME_SUPPORTS_CONSTEXPR 1     // 对于其他不认识的编译器，默认支持 constexpr
#define C10_TYPENAME_CONSTEXPR constexpr
#endif

struct type_index final : IdWrapper<type_index, uint64_t> {
  constexpr explicit type_index(uint64_t checksum) : IdWrapper(checksum) {}

  // Allow usage in std::map / std::set
  // TODO Disallow this and rather use std::unordered_map/set everywhere
  // 定义 type_index 类型之间的小于比较运算符，以支持在 std::map / std::set 中使用
  friend constexpr bool operator<(type_index lhs, type_index rhs) noexcept {
    return lhs.underlyingId() < rhs.underlyingId();
  }

  // 定义 type_index 类型的输出流操作符重载，输出底层的 uint64_t 值
  friend std::ostream& operator<<(std::ostream& stream, type_index typeId) {
    return stream << typeId.underlyingId();
  }
};

namespace detail {

#if !defined(__clang__) && !defined(_MSC_VER) && defined(__GNUC__) && \
    __GNUC__ < 5
// Getting __PRETTY_FUNCTION__ at compile time only works with GCC >= 5
#error "You're running a too old version of GCC. We need GCC 5 or later."
#endif

#if defined(__clang__) && __clang_major__ < 4
// Getting __PRETTY_FUNCTION__ at compile time only works with Clang >= 4
#error "You're running a too old version of Clang. We need Clang 4 or later."
#endif

// 定义 constexpr 函数，从给定字符串中提取中间部分，并且做有效性检查
inline constexpr string_view extract(
    string_view prefix,
    string_view suffix,
    string_view str) {
#if !defined(__CUDA_ARCH__) // CUDA doesn't like std::logic_error in device code
  return (!str.starts_with(prefix) || !str.ends_with(suffix))
      ? (throw std::logic_error("Invalid pattern"), string_view())
      : str.substr(prefix.size(), str.size() - prefix.size() - suffix.size());
#else
  return str.substr(prefix.size(), str.size() - prefix.size() - suffix.size());
#endif
}

// 模板函数，根据不同编译器设置不同的类型名全限定名称
template <typename T>
inline C10_TYPENAME_CONSTEXPR c10::string_view fully_qualified_type_name_impl() {
#if defined(_MSC_VER) && !defined(__clang__)
#if defined(__NVCC__)
  // 如果编译器为 NVIDIA CUDA 编译器 (NVCC)，使用 __FUNCSIG__ 提取类型名称信息
  return extract(
      "c10::basic_string_view<char> c10::util::detail::fully_qualified_type_name_impl<",
      ">()",
      __FUNCSIG__);
#else
  // 如果非 NVCC 编译器，使用 __FUNCSIG__ 提取完整的类型名称信息
  return extract(
      "class c10::basic_string_view<char> __cdecl c10::util::detail::fully_qualified_type_name_impl<",
      ">(void)",
      __FUNCSIG__);
#endif
#elif defined(__clang__)
  // 如果编译器为 Clang，使用 __PRETTY_FUNCTION__ 提取类型名称信息
  return extract(
      "c10::string_view c10::util::detail::fully_qualified_type_name_impl() [T = ",
      "]",
      __PRETTY_FUNCTION__);
#elif defined(__GNUC__)
  // 如果编译器为 GCC，根据 C10_TYPENAME_SUPPORTS_CONSTEXPR 定义选择提取类型名称信息方式
  return extract(
#if C10_TYPENAME_SUPPORTS_CONSTEXPR
      "constexpr c10::string_view c10::util::detail::fully_qualified_type_name_impl() [with T = ",
#else
      "c10::string_view c10::util::detail::fully_qualified_type_name_impl() [with T = ",
#endif
      "; c10::string_view = c10::basic_string_view<char>]",
      __PRETTY_FUNCTION__);
#endif
}

#if !defined(__CUDA_ARCH__)
template <typename T>
inline constexpr uint64_t type_index_impl() {
  // 使用编译器特定的宏或函数（__FUNCSIG__ 或 __PRETTY_FUNCTION__）获取函数的完整名称，然后计算其 CRC64 校验和作为类型标识符
#if defined(_MSC_VER) && !defined(__clang__)
  return crc64(__FUNCSIG__, sizeof(__FUNCSIG__)).checksum();
#elif defined(__clang__)
  return crc64(__PRETTY_FUNCTION__, sizeof(__PRETTY_FUNCTION__)).checksum();
#elif defined(__GNUC__)
  return crc64(__PRETTY_FUNCTION__, sizeof(__PRETTY_FUNCTION__)).checksum();
#endif
}
#endif

} // namespace detail

template <typename T>
inline constexpr type_index get_type_index() {
#if !defined(__CUDA_ARCH__)
  // 对于非 CUDA 架构，通过 type_index_impl 获取类型 T 的唯一标识符，并封装在 std::integral_constant 中返回
  return type_index{std::integral_constant<
      uint64_t,
      detail::type_index_impl<std::decay_t<T>>()>::value};
#else
  // 在 CUDA 架构上，直接返回一个无效的类型索引，因为 CUDA 不支持此操作
  return (abort(), type_index(0));
#endif
}

#if !defined(TORCH_PEDANTIC)
// 为 std::string 使用预计算的哈希值作为类型索引
// 这是为了解决类名模糊性问题，特别是在多 ABI C++ 库中的情况下
template <>
inline constexpr type_index get_type_index<std::string>() {
  // 对于 std::string，返回预定义的哈希值作为类型索引
  return type_index{4193213214807308375ULL};
}
#endif

template <typename T>
inline C10_TYPENAME_CONSTEXPR string_view
get_fully_qualified_type_name() noexcept {
#if C10_TYPENAME_SUPPORTS_CONSTEXPR
  constexpr
#else
  static
#endif
  // 如果支持 constexpr，则使用 constexpr 修饰返回类型的字符串视图
  // 否则，使用 static 修饰函数，确保只在编译期间初始化
#endif
      // 终止预处理指令，用于结束条件编译的分支，此处结束未命名的条件编译区段

string_view name = detail::fully_qualified_type_name_impl<T>();
  // 使用模板参数 T 获取其完全限定类型名称，并存储在名为 name 的 string_view 中

return name;
// 返回保存在 name 中的完全限定类型名称

} // namespace c10::util
// 结束命名空间 c10::util 的定义

C10_DEFINE_HASH_FOR_IDWRAPPER(c10::util::type_index);
// 定义宏 C10_DEFINE_HASH_FOR_IDWRAPPER，为 c10::util::type_index 提供哈希定义
```