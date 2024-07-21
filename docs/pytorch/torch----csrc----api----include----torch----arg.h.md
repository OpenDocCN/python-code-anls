# `.\pytorch\torch\csrc\api\include\torch\arg.h`

```py
#pragma once

#include <utility>

// 宏定义 TORCH_ARG，用于生成属性访问器方法
#define TORCH_ARG(T, name)                                                \
 public:                                                                  \
  // 设置属性值为常量引用的版本
  inline auto name(const T& new_##name) -> decltype(*this) { /* NOLINT */ \
    this->name##_ = new_##name;                                           \
    return *this;                                                         \
  }                                                                       \
  // 设置属性值为右值引用的版本
  inline auto name(T&& new_##name) -> decltype(*this) { /* NOLINT */      \
    this->name##_ = std::move(new_##name);                                \
    return *this;                                                         \
  }                                                                       \
  // 获取属性值的常量引用版本
  inline const T& name() const noexcept { /* NOLINT */                    \
    return this->name##_;                                                 \
  }                                                                       \
  // 获取属性值的非常量引用版本
  inline T& name() noexcept { /* NOLINT */                                \
    return this->name##_;                                                 \
  }                                                                       \
                                                                          \
 private:                                                                 \
  T name##_ /* NOLINT */
```