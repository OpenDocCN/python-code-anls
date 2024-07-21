# `.\pytorch\c10\util\accumulate.h`

```py
// 版权声明，指明此代码版权归Facebook所有，保留所有权利
#pragma once

// 包含C++标准库的异常处理头文件
#include <c10/util/Exception.h>

// 包含C++标准库的头文件
#include <cstdint>      // 提供整数类型的定义
#include <functional>   // 提供函数对象相关功能
#include <iterator>     // 提供迭代器相关功能
#include <numeric>      // 提供数值计算相关功能
#include <type_traits>  // 提供类型特性支持
#include <utility>      // 提供一般性实用工具

// C10命名空间
namespace c10 {

/// Sum of a list of integers; accumulates into the int64_t datatype
template <
    typename C,
    std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t sum_integers(const C& container) {
  // std::accumulate根据init参数的类型推断返回类型，如果init类型不足以保存结果，计算可能会溢出。
  // 这里使用int64_t来避免这种情况。
  return std::accumulate(
      container.begin(), container.end(), static_cast<int64_t>(0));
}

/// Sum of integer elements referred to by iterators; accumulates into the
/// int64_t datatype
template <
    typename Iter,
    std::enable_if_t<
        std::is_integral_v<typename std::iterator_traits<Iter>::value_type>,
        int> = 0>
inline int64_t sum_integers(Iter begin, Iter end) {
  // std::accumulate根据init参数的类型推断返回类型，如果init类型不足以保存结果，计算可能会溢出。
  // 这里使用int64_t来避免这种情况。
  return std::accumulate(begin, end, static_cast<int64_t>(0));
}

/// Product of a list of integers; accumulates into the int64_t datatype
template <
    typename C,
    std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t multiply_integers(const C& container) {
  // std::accumulate根据init参数的类型推断返回类型，如果init类型不足以保存结果，计算可能会溢出。
  // 这里使用int64_t来避免这种情况。
  return std::accumulate(
      container.begin(),
      container.end(),
      static_cast<int64_t>(1),
      std::multiplies<>());
}

/// Product of integer elements referred to by iterators; accumulates into the
/// int64_t datatype
template <
    typename Iter,
    std::enable_if_t<
        std::is_integral_v<typename std::iterator_traits<Iter>::value_type>,
        int> = 0>
inline int64_t multiply_integers(Iter begin, Iter end) {
  // std::accumulate根据init参数的类型推断返回类型，如果init类型不足以保存结果，计算可能会溢出。
  // 这里使用int64_t来避免这种情况。
  return std::accumulate(
      begin, end, static_cast<int64_t>(1), std::multiplies<>());
}

/// Return product of all dimensions starting from k
/// Returns 1 if k>=dims.size()
template <
    typename C,
    std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t numelements_from_dim(const int k, const C& dims) {
  // 内部断言，用于调试目的，确保k大于等于0
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(k >= 0);

  // 如果k大于dims的大小，则返回1
  if (k >= static_cast<int>(dims.size())) {
    return 1;
  } else {
    // 定位到从k开始的维度，计算其后所有维度的乘积并返回
    auto cbegin = dims.cbegin();
    std::advance(cbegin, k);
    return multiply_integers(cbegin, dims.cend());
  }
}

} // namespace c10
/// Calculate the number of elements up to the dimension index `k` (not including `dims[k]`).
/// Throws an error if `k` is greater than the size of `dims`.
template <
    typename C,
    std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t numelements_to_dim(const int k, const C& dims) {
  // Ensure `k` is non-negative
  TORCH_INTERNAL_ASSERT(0 <= k);
  // Ensure `k` is within the bounds of `dims`
  TORCH_INTERNAL_ASSERT((unsigned)k <= dims.size());

  auto cend = dims.cbegin();
  // Advance iterator `cend` to `k` positions ahead from the beginning of `dims`
  std::advance(cend, k);
  // Calculate the product of integers from the beginning to `cend` (exclusive)
  return multiply_integers(dims.cbegin(), cend);
}

/// Calculate the number of elements between dimension indices `k` and `l` inclusive of `dims[k]` but exclusive of `dims[l]`.
/// `k` and `l` may be provided in any order.
template <
    typename C,
    std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t numelements_between_dim(int k, int l, const C& dims) {
  // Ensure both `k` and `l` are non-negative
  TORCH_INTERNAL_ASSERT(0 <= k);
  TORCH_INTERNAL_ASSERT(0 <= l);

  // Swap `k` and `l` if `k` is greater than `l`
  if (k > l) {
    std::swap(k, l);
  }

  // Ensure `l` is within the bounds of `dims`
  TORCH_INTERNAL_ASSERT((unsigned)l < dims.size());

  auto cbegin = dims.cbegin();
  auto cend = dims.cbegin();
  // Advance iterator `cbegin` to `k` positions from the beginning of `dims`
  std::advance(cbegin, k);
  // Advance iterator `cend` to `l` positions from the beginning of `dims`
  std::advance(cend, l);
  // Calculate the product of integers from `cbegin` to `cend` (exclusive of `cend`)
  return multiply_integers(cbegin, cend);
}

} // namespace c10
```