# `.\pytorch\c10\util\ssize.h`

```
#pragma once

#include <c10/util/Exception.h>
#include <c10/util/TypeSafeSignMath.h>

#include <cstddef>
#include <type_traits>

namespace c10 {

// 实现 C++ 20 中的 std::ssize()。
//
// 这对于避免 -Werror=sign-compare 问题特别有用。
//
// 使用时需要通过参数相关的查找，例如：
// 使用 c10::ssize;
// auto size = ssize(container);
//
// 与标准库版本一样，容器可以通过在相同命名空间中定义的自由函数来专门化这个函数。
//
// 更多信息请参见 https://en.cppreference.com/w/cpp/iterator/size
// 以及我们实现的来源。
//
// 我们通过在溢出时添加 assert() 来增强实现。

template <typename C>
constexpr auto ssize(const C& c) -> std::
    common_type_t<std::ptrdiff_t, std::make_signed_t<decltype(c.size())>> {
  using R = std::
      common_type_t<std::ptrdiff_t, std::make_signed_t<decltype(c.size())>>;
  // 我们预期这种情况非常少见，不希望在发布模式下支付性能损失。
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!greater_than_max<R>(c.size()));
  return static_cast<R>(c.size());
}

template <typename T, std::ptrdiff_t N>
// NOLINTNEXTLINE(*-c-arrays)
constexpr auto ssize(const T (&array)[N]) noexcept -> std::ptrdiff_t {
  return N;
}

} // namespace c10
```