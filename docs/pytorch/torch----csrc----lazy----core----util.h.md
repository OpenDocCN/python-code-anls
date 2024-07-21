# `.\pytorch\torch\csrc\lazy\core\util.h`

```
/**
 * 大部分工具是从 PyTorch/XLA 适配而来
 * https://github.com/pytorch/xla/blob/master/third_party/xla_client/util.h
 */

#pragma once

#include <exception>                      // 引入异常处理相关头文件
#include <functional>                     // 引入函数对象相关头文件
#include <vector>                         // 引入向量容器相关头文件

#include <c10/util/Optional.h>            // 引入可选值相关头文件
#include <c10/util/OptionalArrayRef.h>    // 引入可选数组引用相关头文件

namespace torch {
namespace lazy {

// 类似于 c10::scope_exit，但带有状态
// TODO(alanwaketan): 与 c10::scope_exit 合并
template <typename T>
class Cleanup {
 public:
  using StatusType = T;

  explicit Cleanup(std::function<void(StatusType&&)>&& func)
      : func_(std::move(func)) {}           // 构造函数，接受一个移动语义的函数对象
  Cleanup(Cleanup&& ref) noexcept
      : func_(std::move(ref.func_)), status_(std::move(ref.status_)) {}  // 移动构造函数
  Cleanup(const Cleanup&) = delete;         // 删除复制构造函数

  ~Cleanup() {                              // 析构函数
    if (func_ != nullptr) {
      func_(std::move(status_));
    }
  }

  Cleanup& operator=(const Cleanup&) = delete;  // 删除赋值运算符

  Cleanup& operator=(Cleanup&& ref) noexcept {  // 移动赋值运算符
    if (this != &ref) {
      func_ = std::move(ref.func_);
      status_ = std::move(ref.status_);
    }
    return *this;
  }

  void Release() {                           // 释放函数，置空 func_
    func_ = nullptr;
  }

  void SetStatus(StatusType&& status) {      // 设置状态的函数
    status_ = std::move(status);
  }

  const StatusType& GetStatus() const {      // 获取状态的函数
    return status_;
  }

 private:
  std::function<void(StatusType&&)> func_;   // 存储函数对象
  StatusType status_;                        // 存储状态
};

using ExceptionCleanup = Cleanup<std::exception_ptr>;  // 使用 Cleanup 类的模板实例化，状态为 std::exception_ptr

// 允许 API 返回 const 引用或值，不需要在签名中强制返回值
// TODO(alanwaketan): 聪明的设计，但是没有 std 或 c10 的支持吗？需要进一步调查
template <typename T>
class MaybeRef {
 public:
  /* implicit */ MaybeRef(const T& ref) : ref_(ref) {}       // 隐式构造函数，接受 const 引用
  /* implicit */ MaybeRef(T&& value)
      : storage_(std::move(value)), ref_(*storage_) {}       // 隐式构造函数，接受右值引用

  const T& Get() const {                    // 获取存储对象的引用
    return ref_;
  }
  const T& operator*() const {              // 解引用操作符重载
    return Get();
  }
  operator const T&() const {               // 隐式类型转换操作符重载
    return Get();
  }

  bool IsStored() const {                   // 判断对象是否已存储
    return storage_.has_value();
  }

 private:
  std::optional<T> storage_;                // 存储对象的可选值容器
  const T& ref_;                            // 引用存储对象
};

template <typename T>
std::vector<T> Iota(size_t size, T init = 0, T incr = 1) {   // 生成指定大小的向量，初始值和增量可选
  std::vector<T> result(size);              // 创建大小为 size 的向量
  T value = init;                           // 初始值
  for (size_t i = 0; i < size; ++i, value += incr) {         // 循环填充向量
    result[i] = value;
  }
  return result;                            // 返回填充好的向量
}

template <typename T, typename S>
std::vector<T> ToVector(const S& input) {    // 将可迭代对象转换为向量
  return std::vector<T>(input.begin(), input.end());   // 使用输入对象的 begin 和 end 迭代器构造向量
}

template <typename T>
std::optional<std::vector<T>> ToOptionalVector(
    c10::OptionalArrayRef<T> arrayRef) {     // 将可选的数组引用转换为可选的向量
  if (arrayRef) {                           // 如果数组引用有值
    return arrayRef->vec();                 // 返回数组引用的向量表示
  }
  return c10::nullopt;                      // 否则返回空的可选值
}

template <typename T>
typename std::underlying_type<T>::type GetEnumValue(T value) {   // 获取枚举类型的底层类型的值
  return static_cast<typename std::underlying_type<T>::type>(value);  // 将枚举值强制转换为其底层类型的值并返回
}

} // namespace lazy
} // namespace torch
```