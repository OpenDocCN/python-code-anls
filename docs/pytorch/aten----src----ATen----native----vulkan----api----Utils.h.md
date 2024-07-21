# `.\pytorch\aten\src\ATen\native\vulkan\api\Utils.h`

```
#pragma once
// 使用预处理命令#pragma once，确保头文件只被包含一次

#include <cmath>
// 包含数学函数库

#include <numeric>
// 包含STL的数值算法库

#include <ATen/native/vulkan/api/vk_api.h>
// 包含Vulkan API的头文件

#include <ATen/native/vulkan/api/Exception.h>
// 包含Vulkan API的异常处理头文件

#ifdef USE_VULKAN_API
// 如果定义了USE_VULKAN_API宏，则编译以下代码块

// Compiler Macros

// 抑制未使用变量的警告，根据编译器不同采用不同的实现方式
#if defined(_MSC_VER) && !defined(__clang__)
#define VK_UNUSED __pragma(warning(suppress : 4100 4101))
#else
#define VK_UNUSED __attribute__((__unused__))
#endif //_MSC_VER

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace utils {

//
// Hashing
//

/**
 * hash_combine is taken from c10/util/hash.h, which in turn is based on
 * implementation from Boost
 */
inline size_t hash_combine(size_t seed, size_t value) {
  // 实现了一个哈希组合算法，结合种子和值进行哈希计算
  return seed ^ (value + 0x9e3779b9 + (seed << 6u) + (seed >> 2u));
}

//
// Alignment
//

template <typename Type>
inline constexpr Type align_down(const Type& number, const Type& multiple) {
  // 对给定的数向下对齐到最近的倍数
  return (number / multiple) * multiple;
}

template <typename Type>
inline constexpr Type align_up(const Type& number, const Type& multiple) {
  // 对给定的数向上对齐到最近的倍数
  return align_down(number + multiple - 1, multiple);
}

template <typename Type>
inline constexpr Type div_up(const Type& numerator, const Type& denominator) {
  // 对给定的分子和分母进行向上整除操作
  return (numerator + denominator - 1) / denominator;
}

//
// Casting Utilities
//

namespace detail {

/*
 * x cannot be less than 0 if x is unsigned
 */
template <typename T>
static inline constexpr bool is_negative(
    const T& /*x*/,
    std::true_type /*is_unsigned*/) {
  // 如果类型T是无符号类型，则不可能为负数，返回false
  return false;
}

/*
 * check if x is less than 0 if x is signed
 */
template <typename T>
static inline constexpr bool is_negative(
    const T& x,
    std::false_type /*is_unsigned*/) {
  // 如果类型T是有符号类型，则判断x是否小于0，返回结果
  return x < T(0);
}

/*
 * Returns true if x < 0
 */
template <typename T>
inline constexpr bool is_negative(const T& x) {
  // 判断给定的数值x是否小于0，根据类型调用不同的实现
  return is_negative(x, std::is_unsigned<T>());
}

/*
 * Returns true if x < lowest(Limit); standard comparison
 */
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(
    const T& x,
    std::false_type /*limit_is_unsigned*/,
    std::false_type /*x_is_unsigned*/) {
  // 如果Limit和x都是有符号类型，则比较x是否小于Limit的最小值
  return x < std::numeric_limits<Limit>::lowest();
}

/*
 * Limit can contained negative values, but x cannot; return false
 */
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(
    const T& /*x*/,
    std::false_type /*limit_is_unsigned*/,
    std::true_type /*x_is_unsigned*/) {
  // 如果Limit是有符号类型，但x是无符号类型，则不可能x小于最小值，返回false
  return false;
}

/*
 * Limit cannot contained negative values, but x can; check if x is negative
 */
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(
    const T& x,
    std::true_type /*limit_is_unsigned*/,
    std::false_type /*x_is_unsigned*/) {
  // 如果Limit是无符号类型，但x是有符号类型，则检查x是否小于0
  return x < T(0);
}

/*
 * Both x and Limit cannot be negative; return false
 */
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(
    const T& /*x*/,
    std::true_type /*limit_is_unsigned*/,
    std::true_type /*x_is_unsigned*/) {
  // 如果Limit和x都是无符号类型，则不可能x小于最小值，返回false
  return false;
}
/*
 * Returns true if x is less than the lowest value of type T
 */
template <typename Limit, typename T>
inline constexpr bool less_than_lowest(const T& x) {
  return less_than_lowest<Limit>(
      x, std::is_unsigned<Limit>(), std::is_unsigned<T>());
}

// Suppress sign compare warning when compiling with GCC
// as later does not account for short-circuit rule before
// raising the warning, see https://godbolt.org/z/Tr3Msnz99
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif

/*
 * Returns true if x is greater than the greatest value of the type Limit
 */
template <typename Limit, typename T>
inline constexpr bool greater_than_max(const T& x) {
  constexpr bool can_overflow =
      std::numeric_limits<T>::digits > std::numeric_limits<Limit>::digits;
  return can_overflow && x > std::numeric_limits<Limit>::max();
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/*
 * Checks if casting from type From to type To would result in overflow
 */
template <typename To, typename From>
std::enable_if_t<std::is_integral_v<From> && !std::is_same_v<From, bool>, bool>
overflows(From f) {
  using limit = std::numeric_limits<To>;
  // Casting from signed to unsigned; allow for negative numbers to wrap using
  // two's complement arithmetic.
  if (!limit::is_signed && std::numeric_limits<From>::is_signed) {
    return greater_than_max<To>(f) ||
        (is_negative(f) && -static_cast<uint64_t>(f) > limit::max());
  }
  // standard case, check if f is outside the range of type To
  else {
    return less_than_lowest<To>(f) || greater_than_max<To>(f);
  }
}

/*
 * Checks if floating point value f would overflow when cast to type To
 */
template <typename To, typename From>
std::enable_if_t<std::is_floating_point_v<From>, bool> overflows(From f) {
  using limit = std::numeric_limits<To>;
  if (limit::has_infinity && std::isinf(static_cast<double>(f))) {
    return false;
  }
  return f < limit::lowest() || f > limit::max();
}

/*
 * Performs a safe downcast from type From to type To, checking for overflow
 */
template <typename To, typename From>
inline constexpr To safe_downcast(const From& v) {
  VK_CHECK_COND(!overflows<To>(v), "Cast failed: out of range!");
  return static_cast<To>(v);
}

/*
 * Checks if From is signed and To is unsigned
 */
template <typename To, typename From>
inline constexpr bool is_signed_to_unsigned() {
  return std::is_signed<From>::value && std::is_unsigned<To>::value;
}

} // namespace detail

/*
 * Performs a safe downcast from signed From to unsigned To, checking for negative values
 */
template <
    typename To,
    typename From,
    std::enable_if_t<detail::is_signed_to_unsigned<To, From>(), bool> = true>
inline constexpr To safe_downcast(const From& v) {
  VK_CHECK_COND(v >= From{}, "Cast failed: negative signed to unsigned!");
  return detail::safe_downcast<To, From>(v);
}

/*
 * Performs a safe downcast from From to To without additional checks for signedness
 */
template <
    typename To,
    typename From,
    std::enable_if_t<!detail::is_signed_to_unsigned<To, From>(), bool> = true>
inline constexpr To safe_downcast(const From& v) {
  return detail::safe_downcast<To, From>(v);
}

//
// Vector Types
//

namespace detail {

/*
 * Generic vector structure with N elements of type Type
 */
template <typename Type, uint32_t N>
struct vec final {
  // NOLINTNEXTLINE
  Type data[N];
};

} // namespace detail

/*
 * Alias for an integer vector with N elements
 */
template <uint32_t N>
using ivec = detail::vec<int32_t, N>;

/*
 * Alias for a 2-element integer vector
 */
using ivec2 = ivec<2u>;

/*
 * Alias for a 3-element integer vector
 */
using ivec3 = ivec<3u>;
using ivec4 = ivec<4u>;

// 模板别名，表示一个包含 N 个 uint32_t 类型元素的向量
template <uint32_t N>
using uvec = detail::vec<uint32_t, N>;

// uvec2 表示一个包含 2 个 uint32_t 类型元素的向量
using uvec2 = uvec<2u>;

// uvec3 表示一个包含 3 个 uint32_t 类型元素的向量
using uvec3 = uvec<3u>;

// uvec4 表示一个包含 4 个 uint32_t 类型元素的向量
using uvec4 = uvec<4u>;

// 模板别名，表示一个包含 N 个 float 类型元素的向量
template <uint32_t N>
using vec = detail::vec<float, N>;

// vec2 表示一个包含 2 个 float 类型元素的向量
using vec2 = vec<2u>;

// vec3 表示一个包含 3 个 float 类型元素的向量
using vec3 = vec<3u>;

// vec4 表示一个包含 4 个 float 类型元素的向量
using vec4 = vec<4u>;

// 对于 uvec3 类型的对象，重载输出操作符，将其格式化输出到流 os 中
inline std::ostream& operator<<(std::ostream& os, const uvec3& v) {
  // 输出 uvec3 对象的数据成员，格式为 (x, y, z)
  os << "(" << v.data[0u] << ", " << v.data[1u] << ", " << v.data[2u] << ")";
  return os;
}

//
// std::vector<T> Handling
//

/*
 * 对 std::vector<T> 进行索引操作的实用函数。支持负索引，例如 -1 表示获取最后一个元素。
 * 如果索引超出边界，则返回 sizes 的长度或 1。
 */
template <typename T>
inline T val_at(const int64_t index, const std::vector<T>& sizes) {
  // 获取 sizes 的长度
  const int64_t ndim = static_cast<int64_t>(sizes.size());
  if (index >= 0) {
    // 如果索引为正数且超出边界，返回 1，否则返回对应索引处的元素
    return index >= ndim ? 1 : sizes[index];
  } else {
    // 如果索引为负数且超出边界，返回 1，否则返回对应索引处的元素
    return ndim + index < 0 ? 1 : sizes[ndim + index];
  }
}

/*
 * 根据给定的 std::vector<int64_t> 和 reverse 标志创建一个 ivec2 对象。
 * reverse 为 true 表示反转顺序。
 */
inline ivec2 make_ivec2(
    const std::vector<int64_t>& ints,
    bool reverse = false) {
  // 检查 ints 的大小是否为 2
  VK_CHECK_COND(ints.size() == 2);
  if (reverse) {
    // 如果 reverse 为 true，反转 ints 中元素顺序后创建 ivec2 对象
    return {safe_downcast<int32_t>(ints[1]), safe_downcast<int32_t>(ints[0])};
  } else {
    // 否则按 ints 中元素顺序创建 ivec2 对象
    return {safe_downcast<int32_t>(ints[0]), safe_downcast<int32_t>(ints[1])};
  }
}

/*
 * 根据给定的 std::vector<int64_t> 和 reverse 标志创建一个 ivec4 对象。
 * reverse 为 true 表示反转顺序。
 */
inline ivec4 make_ivec4(
    const std::vector<int64_t>& ints,
    bool reverse = false) {
  // 检查 ints 的大小是否为 4
  VK_CHECK_COND(ints.size() == 4);
  if (reverse) {
    // 如果 reverse 为 true，反转 ints 中元素顺序后创建 ivec4 对象
    return {
        safe_downcast<int32_t>(ints[3]),
        safe_downcast<int32_t>(ints[2]),
        safe_downcast<int32_t>(ints[1]),
        safe_downcast<int32_t>(ints[0]),
    };
  } else {
    // 否则按 ints 中元素顺序创建 ivec4 对象
    return {
        safe_downcast<int32_t>(ints[0]),
        safe_downcast<int32_t>(ints[1]),
        safe_downcast<int32_t>(ints[2]),
        safe_downcast<int32_t>(ints[3]),
    };
  }
}

/*
 * 根据给定的 std::vector<int64_t> 创建一个 ivec4 对象，前面用 1 填充。
 */
inline ivec4 make_ivec4_prepadded1(const std::vector<int64_t>& ints) {
  // 检查 ints 的大小是否不超过 4
  VK_CHECK_COND(ints.size() <= 4);

  // 创建一个初始为 {1, 1, 1, 1} 的 ivec4 对象
  ivec4 result = {1, 1, 1, 1};
  size_t base = 4 - ints.size();
  for (size_t i = 0; i < ints.size(); ++i) {
    // 将 ints 中的元素安全地转换为 int32_t 类型，填充到 result 中对应位置
    result.data[i + base] = safe_downcast<int32_t>(ints[i]);
  }

  return result;
}

/*
 * 根据给定的 uvec3 对象创建一个 ivec3 对象。
 */
inline ivec3 make_ivec3(uvec3 ints) {
  // 将 uvec3 对象的元素安全地转换为 int32_t 类型，创建一个对应的 ivec3 对象
  return {
      safe_downcast<int32_t>(ints.data[0u]),
      safe_downcast<int32_t>(ints.data[1u]),
      safe_downcast<int32_t>(ints.data[2u])
  };
}

/*
 * 根据给定的 std::vector<int64_t> 构造一个 uvec4 对象，反转数组元素顺序。
 * 数组中的元素表示张量的尺寸，最多可以包含 4 个元素。
 */
inline uvec4 make_whcn_uvec4(const std::vector<int64_t>& arr) {
  // 从 arr 中获取 uint32_t 类型的 w, h, c, n
  uint32_t w = safe_downcast<uint32_t>(val_at(-1, arr));
  uint32_t h = safe_downcast<uint32_t>(val_at(-2, arr));
  uint32_t c = safe_downcast<uint32_t>(val_at(-3, arr));
  uint32_t n = safe_downcast<uint32_t>(val_at(-4, arr));

  // 创建一个 uvec4 对象，依次填充 w, h, c, n
  return {w, h, c, n};
}
/*
 * Given an vector of up to 4 int64_t representing the sizes of a tensor,
 * constructs an ivec4 containing those elements in reverse order.
 */
inline ivec4 make_whcn_ivec4(const std::vector<int64_t>& arr) {
  // Retrieve the last element in the vector `arr` and assign it to `w`
  int32_t w = val_at(-1, arr);
  // Retrieve the second last element in the vector `arr` and assign it to `h`
  int32_t h = val_at(-2, arr);
  // Retrieve the third last element in the vector `arr` and assign it to `c`
  int32_t c = val_at(-3, arr);
  // Retrieve the fourth last element in the vector `arr` and assign it to `n`
  int32_t n = val_at(-4, arr);

  // Return an ivec4 initialized with `w`, `h`, `c`, `n`
  return {w, h, c, n};
}

/*
 * Wrapper around std::accumulate that accumulates values of a container of
 * integral types into int64_t. Taken from `multiply_integers` in
 * <c10/util/accumulate.h>
 */
template <
    typename C,
    std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t multiply_integers(const C& container) {
  // Accumulate the values in `container` starting with an initial value of 1,
  // using multiplication to combine them into an int64_t result
  return std::accumulate(
      container.begin(),
      container.end(),
      static_cast<int64_t>(1),
      std::multiplies<>());
}

} // namespace utils

// Define equality operator for utils::uvec3 to compare each element in data array
inline bool operator==(const utils::uvec3& _1, const utils::uvec3& _2) {
  // Compare each element in the `data` array of `_1` and `_2` for equality
  return (
      _1.data[0u] == _2.data[0u] && 
      _1.data[1u] == _2.data[1u] &&
      _1.data[2u] == _2.data[2u]);
}

// Convert utils::uvec3 to VkOffset3D struct
inline VkOffset3D create_offset3d(const utils::uvec3& offsets) {
  // Initialize a VkOffset3D struct using elements from `offsets.data`
  return VkOffset3D{
      utils::safe_downcast<int32_t>(offsets.data[0u]), // X offset
      static_cast<int32_t>(offsets.data[1u]),           // Y offset
      static_cast<int32_t>(offsets.data[2u])            // Z offset
  };
}

// Convert utils::uvec3 to VkExtent3D struct
inline VkExtent3D create_extent3d(const utils::uvec3& extents) {
  // Initialize a VkExtent3D struct using elements from `extents.data`
  return VkExtent3D{
      extents.data[0u], // Width
      extents.data[1u], // Height
      extents.data[2u]  // Depth
  };
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```