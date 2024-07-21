# `.\pytorch\torch\csrc\inductor\aoti_runtime\thread_local.h`

```py
#pragma once
// 包含 torch/csrc/inductor/aoti_runtime/arrayref_tensor.h 头文件

#include <torch/csrc/inductor/aoti_runtime/arrayref_tensor.h>

namespace torch {
namespace aot_inductor {

// 模板特化，用于 RAIIAtenTensorHandle 类型
template <typename T>
struct ThreadLocalCachedOutputTensor;

// 模板特化，用于 RAIIAtenTensorHandle 类型
template <>
struct ThreadLocalCachedOutputTensor<RAIIAtenTensorHandle> {
  // 构造函数，接受 RAIIAtenTensorHandle 类型参数
  explicit ThreadLocalCachedOutputTensor(const RAIIAtenTensorHandle&) {}
  // 拷贝数据函数，抛出异常，不可达代码
  void copy_data_from(const RAIIAtenTensorHandle& handle) {
    throw std::runtime_error("can't happen");
  }

  // 返回 AtenTensorHandle 对象，抛出异常，不可达代码
  AtenTensorHandle tensor() const {
    throw std::runtime_error("can't happen");
  }
};

// 模板特化，用于 AtenTensorHandle 类型
template <>
struct ThreadLocalCachedOutputTensor<AtenTensorHandle> {
  // 构造函数，接受 AtenTensorHandle 类型参数
  explicit ThreadLocalCachedOutputTensor(const AtenTensorHandle&) {}
  // 拷贝数据函数，抛出异常，不可达代码
  void copy_data_from(const AtenTensorHandle& handle) {
    throw std::runtime_error("can't happen");
  }

  // 返回 AtenTensorHandle 对象，抛出异常，不可达代码
  AtenTensorHandle tensor() const {
    throw std::runtime_error("can't happen");
  }
};

// 模板特化，用于 ConstantHandle 类型
template <>
struct ThreadLocalCachedOutputTensor<ConstantHandle> {
  // 构造函数，接受 ConstantHandle 类型参数
  explicit ThreadLocalCachedOutputTensor(const ConstantHandle&) {}
  // 拷贝数据函数，抛出异常，不可达代码
  void copy_data_from(const ConstantHandle& handle) {
    throw std::runtime_error("can't happen");
  }

  // 返回 AtenTensorHandle 对象，抛出异常，不可达代码
  AtenTensorHandle tensor() const {
    throw std::runtime_error("can't happen");
  }
};

// 模板特化，用于 ArrayRefTensor<T> 类型
template <typename T>
struct ThreadLocalCachedOutputTensor<ArrayRefTensor<T>> {
  // 构造函数，接受 ArrayRefTensor<T> 类型参数
  explicit ThreadLocalCachedOutputTensor(const ArrayRefTensor<T>& t) {
    realloc(t); // 调用重新分配函数 realloc()
  }

  // 拷贝数据函数，从 ArrayRefTensor<T> 类型参数 t 拷贝数据
  void copy_data_from(const ArrayRefTensor<T>& t) {
    if (t.numel() > capacity_) {
      realloc(t); // 如果数组元素数量大于容量，重新分配存储空间
    }
    // 使用 std::copy() 拷贝数据到 storage_ 中
    std::copy(t.data(), t.data() + t.numel(), storage_.get());
  }

  // 返回 AtenTensorHandle 对象 tensor_
  AtenTensorHandle tensor() const {
    return tensor_.get();
  }

 private:
  // 重新分配存储空间函数，接受 ArrayRefTensor<T> 类型参数 t
  void realloc(const ArrayRefTensor<T>& t) {
    capacity_ = t.numel(); // 更新容量为 t 的元素数量
    storage_ = std::make_unique<T[]>(t.numel()); // 创建大小为 t.numel() 的唯一指针数组
    AtenTensorHandle handle;
    // 调用 aoti_torch_create_tensor_from_blob 函数创建张量 handle
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob(
        storage_.get(), // 存储的数据指针
        t.sizes().size(), // 大小的维度数
        t.sizes().data(), // 大小的数据指针
        t.strides().data(), // 步长的数据指针
        0, // flags
        aoti_torch_dtype<std::remove_const_t<T>>(), // 数据类型
        t.device_type(), // 设备类型
        t.device_idx(), // 设备索引
        &handle)); // 返回的张量 handle
    tensor_ = handle; // 更新成员变量 tensor_ 为 handle
  }

  std::unique_ptr<T[]> storage_; // 唯一指针数组 storage_
  int64_t capacity_ = 0; // 容量
  RAIIAtenTensorHandle tensor_; // RAIIAtenTensorHandle 类型 tensor_
};

// 模板特化，用于 ThreadLocalCachedOutputArray<RAIIAtenTensorHandle> 类型
template <>
struct ThreadLocalCachedOutputArray<RAIIAtenTensorHandle> {
  // 构造函数，接受 RAIIAtenTensorHandle 类型参数，抛出异常，不可达代码
  explicit ThreadLocalCachedOutputArray(const RAIIAtenTensorHandle&) {
    throw std::runtime_error("can't happen");
  }

  // 拷贝数据函数，接受 RAIIAtenTensorHandle 类型参数，抛出异常，不可达代码
  void copy_data_from(const RAIIAtenTensorHandle&) {
    throw std::runtime_error("can't happen");
  }

  // 模板函数，接受 U 类型参数，返回 ArrayRefTensor<U> 对象，抛出异常，不可达代码
  template <typename U>
  ArrayRefTensor<U> arrayref_tensor() const {
    throw std::runtime_error("can't happen");
  }
};

// 模板特化，用于 ThreadLocalCachedOutputArray<RAIIAtenTensorHandle> 类型
template <>
// 定义一个模板结构体 ThreadLocalCachedOutputArray，专门用于存储 ConstantHandle 类型的对象
template <typename T>
struct ThreadLocalCachedOutputArray<ConstantHandle> {
  // 显式构造函数，接受 ConstantHandle 类型的参数，并抛出运行时错误，表示不应该发生这种情况
  explicit ThreadLocalCachedOutputArray(const ConstantHandle&) {
    throw std::runtime_error("can't happen");
  }

  // 这个成员函数用于从 ConstantHandle 类型的对象中复制数据，但是目前不支持，因此抛出运行时错误
  void copy_data_from(const ConstantHandle&) {
    throw std::runtime_error("can't happen");
  }

  // 模板成员函数，返回 ArrayRefTensor<U> 类型的对象，但是目前不支持，因此抛出运行时错误
  template <typename U>
  ArrayRefTensor<U> arrayref_tensor() const {
    throw std::runtime_error("can't happen");
  }
};

// 定义另一个模板结构体 ThreadLocalCachedOutputArray，专门用于存储 ArrayRefTensor<T> 类型的对象
template <typename T>
struct ThreadLocalCachedOutputArray<ArrayRefTensor<T>> {
  // 显式构造函数，接受 ArrayRefTensor<T> 类型的对象作为参数
  explicit ThreadLocalCachedOutputArray(const ArrayRefTensor<T>& t) {}

  // 模板成员函数，返回 ArrayRefTensor<T> 类型的对象
  template <
      typename U,
      std::enable_if_t<
          std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>>,
          bool> = true>
  ArrayRefTensor<T> arrayref_tensor() const {
    return tensor_;  // 返回当前存储的 tensor_ 对象
  }

  // 成员函数，从 ArrayRefTensor<T> 类型的对象中复制数据
  void copy_data_from(const ArrayRefTensor<T>& t) {
    // 如果 t 中元素的数量大于当前容量 capacity_
    if (t.numel() > capacity_) {
      capacity_ = t.numel();  // 更新 capacity_ 的值为 t 的元素数量
      storage_ = std::make_unique<T[]>(capacity_);  // 分配新的存储空间
    }
    // 将 t 中的数据复制到 storage_ 中
    std::copy(t.data(), t.data() + t.numel(), storage_.get());
    tensor_ = t;  // 更新当前存储的 tensor_ 对象为 t
    // 将 tensor_ 设置为使用新的 MiniArrayRef<T> 对象作为其数组引用
    tensor_.set_arrayref(MiniArrayRef<T>(storage_.get(), t.numel()));
  }

 private:
  std::unique_ptr<T[]> storage_;  // 用于存储数据的唯一指针
  uint32_t capacity_ = 0;  // 存储空间的容量
  ArrayRefTensor<T> tensor_;  // 存储的 ArrayRefTensor 对象
};

// 结构体的命名空间，这里是 aot_inductor
namespace aot_inductor {

// 结构体的命名空间，这里是 torch
namespace torch {

// 结构体 ThreadLocalCachedOutputArray 的定义结束
} // namespace torch
} // namespace aot_inductor
```