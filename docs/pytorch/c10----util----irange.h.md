# `.\pytorch\c10\util\irange.h`

```py
// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <c10/util/Exception.h>
#include <c10/util/TypeSafeSignMath.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace c10 {

namespace detail {

template <
    typename I,
    // 是否是单边迭代器，默认为否
    bool one_sided = false,
    // 如果 I 是整数类型，则启用该模板
    std::enable_if_t<std::is_integral_v<I>, int> = 0>
struct integer_iterator {
  // 迭代器的类型特征
  using iterator_category = std::input_iterator_tag;
  using value_type = I;
  using difference_type = std::ptrdiff_t;
  using pointer = I*;
  using reference = I&;

  // 构造函数，初始化迭代器的当前值
  explicit integer_iterator(I value) : value(value) {}

  // 解引用操作符，返回当前迭代器指向的值
  I operator*() const {
    return value;
  }

  // 箭头操作符，返回指向当前迭代器值的指针
  I const* operator->() const {
    return &value;
  }

  // 前缀递增操作，使迭代器指向下一个值
  integer_iterator& operator++() {
    ++value;
    return *this;
  }

  // 后缀递增操作，返回递增前的迭代器副本
  integer_iterator operator++(int) {
    const auto copy = *this;
    ++*this;
    return copy;
  }

  // 比较操作符，判断两个迭代器是否相等
  bool operator==(const integer_iterator& other) const {
    if constexpr (one_sided) {
      // 对于范围循环，结束条件是 `begin != end` 而不是 `begin < end`。
      // 处理 `c10::irange(n)`，其中 n < 0（应为空），我们使得 `begin != end`
      // 在 `end` 为负数时失败。
      return is_negative(other.value) || value == other.value;
    } else {
      return value == other.value;
    }
    // 抑制 "warning: missing return statement at end of non-void function"
    // 这是 Nvidia 的 Robert Crovella 在此处确认为 NVCC 编译器错误的情况
    // 参考：https://stackoverflow.com/a/64561686/752843（2020-10-27）
    // 这里最好使用 `__builtin_unreachable();`，但并非所有编译器都支持。
    // 因此，我们返回一个任意的值，信任这行代码实际上永远不会被执行。
    return false; // 可怕的 hack
  }

  // 不相等比较操作符，判断两个迭代器是否不相等
  bool operator!=(const integer_iterator& other) const {
    return !(*this == other);
  }

 protected:
  I value;
};

} // namespace detail

// 整数范围类模板，生成半开区间 [begin, end) 的整数范围
// 如果 end <= begin，则范围为空
// 范围的类型与 `end` 整数类型相同；`begin` 整数被转换为该类型
template <
    typename Integer1,
    typename Integer2,
    std::enable_if_t<std::is_integral_v<Integer1>, bool> = true,
    std::enable_if_t<std::is_integral_v<Integer2>, bool> = true>
struct integer_range {
 public:
  // 构造函数，初始化整数范围的起始和结束
  integer_range(Integer1 begin, Integer2 end) : begin_(begin), end_(end) {}
  
  // 定义迭代器类型为 detail 命名空间下的 integer_iterator 模板
  using iterator = detail::integer_iterator<Integer1, one_sided>;
  
  // 返回整数范围的起始迭代器
  iterator begin() const {
    return begin_;
  }
  
  // 返回整数范围的结束迭代器
  iterator end() const {
    return end_;
  }

 private:
  iterator begin_; // 起始迭代器
  iterator end_;   // 结束迭代器
};

/// 创建整数范围，用于半开区间 [begin, end)
/// 如果 end <= begin，则范围为空
/// 范围具有 `end` 整数类型；`begin` 整数被转换为该类型
template <
    typename Integer1,
    typename Integer2,
    std::enable_if_t<std::is_integral_v<Integer1>, bool> = true,
    std::enable_if_t<std::is_integral_v<Integer2>, bool> = true>
/// 定义一个函数模板 integer_range，用于生成半开区间 [begin, end)
/// 如果 end <= begin，则表示区间为空
template <typename Integer1, typename Integer2>
integer_range<Integer2> irange(Integer1 begin, Integer2 end) {
  // 如果 end <= begin，则将 end 和 begin 中较大的一个作为循环终止条件，
  // 以达到空区间的效果
  return {
      static_cast<Integer2>(begin),
      std::max(static_cast<Integer2>(begin), end)};
}

/// 创建一个整数范围，对于半开区间 [0, end)
/// 如果 end <= 0，则表示范围为空
template <
    typename Integer,
    std::enable_if_t<std::is_integral_v<Integer>, bool> = true>
integer_range<Integer, true> irange(Integer end) {
  // 返回从 0 开始到 end 的整数范围
  return {Integer(), end};
}

} // namespace c10


这些注释解释了每行代码的作用，包括函数模板的定义、参数说明以及条件判断的用途。
```