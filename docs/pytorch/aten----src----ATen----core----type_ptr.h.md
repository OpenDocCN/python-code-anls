# `.\pytorch\aten\src\ATen\core\type_ptr.h`

```
#pragma once

#include <memory>
#include <type_traits>

#include <c10/util/Exception.h>
#include <c10/util/MaybeOwned.h>

namespace c10 {

// Compatibility wrapper around a raw pointer so that existing code
// written to deal with a shared_ptr can keep working.
template <typename T>
class SingletonTypePtr {
 public:
  /* implicit */ SingletonTypePtr(T* p) : repr_(p) {}  // 使用指针构造函数，将原始指针包装到 SingletonTypePtr 中

  // We need this to satisfy Pybind11, but it shouldn't be hit.
  explicit SingletonTypePtr(std::shared_ptr<T>) { TORCH_CHECK(false); }  // 显式构造函数，用于满足 Pybind11 的需求，但实际上不应该被调用

  using element_type = typename std::shared_ptr<T>::element_type;  // 定义 element_type 为 T 的共享指针的元素类型

  template <typename U = T, std::enable_if_t<!std::is_same_v<std::remove_const_t<U>, void>, bool> = true>
  T& operator*() const {  // 解引用操作符重载，返回包装的原始指针的引用
    return *repr_;
  }

  T* get() const {  // 返回包装的原始指针
    return repr_;
  }

  T* operator->() const {  // 重载箭头操作符，返回包装的原始指针
    return repr_;
  }

  operator bool() const {  // 类型转换操作符，检查是否包装了有效的原始指针
    return repr_ != nullptr;
  }

 private:
  T* repr_{nullptr};  // 包装的原始指针，默认为 nullptr
};

template <typename T, typename U>
bool operator==(SingletonTypePtr<T> lhs, SingletonTypePtr<U> rhs) {  // 相等比较操作符重载，比较两个 SingletonTypePtr 是否指向相同的原始指针
  return (void*)lhs.get() == (void*)rhs.get();
}

template <typename T, typename U>
bool operator!=(SingletonTypePtr<T> lhs, SingletonTypePtr<U> rhs) {  // 不等比较操作符重载，通过调用相等比较操作符实现
  return !(lhs == rhs);
}

} // namespace c10
```