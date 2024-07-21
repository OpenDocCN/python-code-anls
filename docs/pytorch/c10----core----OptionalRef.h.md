# `.\pytorch\c10\core\OptionalRef.h`

```py
#pragma once
// 声明命名空间 c10

namespace c10 {

// 模板类 OptionalRef，用于表示可选引用
template <typename T>
class OptionalRef {
 public:
  // 默认构造函数，初始化 data_ 为 nullptr
  OptionalRef() : data_(nullptr) {}

  // 构造函数，接受指向常量 T 类型数据的指针作为参数
  OptionalRef(const T* data) : data_(data) {
    // 在调试模式下断言 data_ 不为空
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(data_);
  }

  // 构造函数，接受常量 T 类型数据的引用作为参数
  OptionalRef(const T& data) : data_(&data) {}

  // 返回是否存在有效值的布尔函数
  bool has_value() const {
    return data_ != nullptr;
  }

  // 返回常量 T 类型数据的引用
  const T& get() const {
    // 在调试模式下断言 data_ 不为空，然后返回 *data_ 的引用
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(data_);
    return *data_;
  }

  // 隐式转换为布尔值，返回是否存在有效值
  operator bool() const {
    return has_value();
  }

 private:
  const T* data_;  // 指向常量 T 类型数据的指针
};

} // namespace c10
```