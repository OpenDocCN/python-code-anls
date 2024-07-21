# `.\pytorch\torch\csrc\lazy\core\unique.h`

```
/**
 * Unique in this file is adapted from PyTorch/XLA
 * https://github.com/pytorch/xla/blob/master/third_party/xla_client/unique.h
 */

#pragma once

#include <c10/util/Optional.h> // 引入 Optional 类型的头文件

#include <functional> // 引入函数对象的头文件
#include <set> // 引入集合类的头文件

namespace torch {
namespace lazy {

// Helper class to allow tracking zero or more things, which should be forcibly
// be one only thing.
template <typename T, typename C = std::equal_to<T>>
class Unique {
 public:
  // 设置当前值为指定的 value，如果已经有值，则检查是否与当前值相等
  std::pair<bool, const T&> set(const T& value) {
    if (value_) {
      TORCH_CHECK(C()(*value_, value), "'", *value_, "' vs '", value); // 使用 C 检查当前值和新值是否相等
      return std::pair<bool, const T&>(false, *value_); // 如果不相等，返回当前值
    }
    value_ = value; // 设置当前值为新值
    return std::pair<bool, const T&>(true, *value_); // 返回设置成功的标志和当前值
  }

  // 将类转换为 bool 类型，表示当前是否有值
  operator bool() const {
    return value_.has_value();
  }
  
  // 将类转换为 const T& 类型，返回当前的值的引用
  operator const T&() const {
    return *value_;
  }
  
  // 返回当前值的引用
  const T& operator*() const {
    return *value_;
  }
  
  // 返回当前值的指针
  const T* operator->() const {
    return value_.operator->();
  }

  // 将当前值转换为包含单个元素的集合
  std::set<T> AsSet() const {
    std::set<T> vset;
    if (value_.has_value()) {
      vset.insert(*value_);
    }
    return vset;
  }

 private:
  std::optional<T> value_; // 保存当前值的可选类型
};

} // namespace lazy
} // namespace torch
```