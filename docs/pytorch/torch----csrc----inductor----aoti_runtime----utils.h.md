# `.\pytorch\torch\csrc\inductor\aoti_runtime\utils.h`

```
// 防止头文件被多次包含，只在第一次包含时有效
#pragma once

// 包含常用的标准输入输出流、智能指针、字符串流、异常处理、字符串、向量等头文件
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// 引入稳定的 Torch C ABI 中定义的头文件，避免引用除此之外的任何 aten/c10 头文件
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

// 根据不同的编译器定义不进行内联的函数特性
#if defined(__GNUC__) || defined(__clang__)
#define AOTI_NOINLINE __attribute__((noinline))
#elif _MSC_VER
#define AOTI_NOINLINE __declspec(noinline)
#else
#define AOTI_NOINLINE
#endif

// 定义静态函数 throw_exception，用于抛出运行时异常，包含调用信息、文件和行号
AOTI_NOINLINE static void throw_exception(
    const char* call,
    const char* file,
    int64_t line) {
  std::stringstream ss;
  ss << call << " API call failed at " << file << ", line " << line;
  throw std::runtime_error(ss.str());
}

// 定义 Torch API 调用的错误码检查宏，如果调用不成功则抛出异常
#define AOTI_TORCH_ERROR_CODE_CHECK(call)       \
  if ((call) != AOTI_TORCH_SUCCESS) {           \
    throw_exception(#call, __FILE__, __LINE__); \
  }

// 定义 AOTI Runtime API 调用的错误码检查宏，如果调用不成功则抛出异常
using AOTIRuntimeError = int32_t;
#define AOTI_RUNTIME_SUCCESS 0
#define AOTI_RUNTIME_FAILURE 1

#define AOTI_RUNTIME_ERROR_CODE_CHECK(call)     \
  if ((call) != AOTI_RUNTIME_SUCCESS) {         \
    throw_exception(#call, __FILE__, __LINE__); \
  }

// 命名空间 torch::aot_inductor 下定义的内容开始

// 定义指向释放函数的指针类型 DeleterFnPtr，用于处理资源释放
using DeleterFnPtr = void (*)(void*);

// 空函数 noop_deleter，用于作为默认的资源释放函数
inline void noop_deleter(void*) {}

// 删除张量对象的函数 delete_tensor_object，调用 aoti_torch 中的函数来删除张量对象
inline void delete_tensor_object(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_delete_tensor_object(reinterpret_cast<AtenTensorHandle>(ptr)));
}

// RAIIAtenTensorHandle 类，管理通过 libtorch C ABI 创建的张量对象的所有权
class RAIIAtenTensorHandle {
 public:
  // 默认构造函数，初始化为空指针，并使用 noop_deleter 作为释放函数
  RAIIAtenTensorHandle() : handle_(nullptr, noop_deleter) {}

  // 删除拷贝构造函数和赋值运算符，只允许移动语义
  RAIIAtenTensorHandle(const RAIIAtenTensorHandle& other) = delete;
  RAIIAtenTensorHandle& operator=(const RAIIAtenTensorHandle& other) = delete;

  // 使用移动语义从另一个 RAIIAtenTensorHandle 中窃取所有权
  RAIIAtenTensorHandle(RAIIAtenTensorHandle&& other) = default;
  RAIIAtenTensorHandle& operator=(RAIIAtenTensorHandle&& other) = default;

  // 从原始的 AtenTensorHandle 窃取所有权的构造函数
  RAIIAtenTensorHandle(AtenTensorHandle handle)
      : handle_(handle, delete_tensor_object) {}

  // 析构函数，释放 handle_ 指向的资源
  ~RAIIAtenTensorHandle() {
    handle_.reset();
  }

  // 将 RAIIAtenTensorHandle 转换为原始的 AtenTensorHandle，不转移所有权
  operator AtenTensorHandle() const {
    return handle_.get();
  }

  // 释放 RAIIAtenTensorHandle 持有的资源，并返回原始的 AtenTensorHandle
  AtenTensorHandle release() {
    return handle_.release();
  }

  // 获取当前 RAIIAtenTensorHandle 持有的原始 AtenTensorHandle
  AtenTensorHandle get() const {
    return handle_.get();
  }

  // 重置 RAIIAtenTensorHandle，释放当前持有的资源
  void reset() {
    handle_.reset();
  }

  // 获取指定维度 d 的张量大小
  int64_t size(int64_t d) {
    int64_t size;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_size(handle_.get(), d, &size));
    return size;
  }

  // 获取指定维度 d 的张量步长
  int64_t stride(int64_t d) {
    int64_t stride;
    // 使用 AOTI_TORCH_ERROR_CODE_CHECK 宏获取张量的步幅（stride），并返回步幅值
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_stride(handle_.get(), d, &stride));
    // 返回获取到的步幅值
    return stride;
  }

  // 返回张量的存储偏移量（storage_offset）
  int64_t storage_offset() {
    // 声明变量 storage_offset 用于保存获取的存储偏移量值
    int64_t storage_offset;
    // 使用 AOTI_TORCH_ERROR_CODE_CHECK 宏获取张量的存储偏移量，并将值保存到 storage_offset 变量中
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_storage_offset(handle_.get(), &storage_offset));
    // 返回获取到的存储偏移量值
    return storage_offset;
  }

 private:
  // 使用 std::unique_ptr 管理的原生指针，指向 AtentTensorOpaque 类型的对象，使用 DeleterFnPtr 进行资源释放
  std::unique_ptr<AtenTensorOpaque, DeleterFnPtr> handle_;
};

// 结束了一个匿名的命名空间或类的定义，这里可能是在某个命名空间或类中定义了一些函数或对象，这里是其结束位置。

// 将从原始的 AtenTensorHandle 转移到 RAIIAtenTensorHandle
inline std::vector<RAIIAtenTensorHandle> steal_from_raw_handles_to_raii_handles(
    AtenTensorHandle* handles,
    size_t size) {
  // 创建一个空的 RAIIAtenTensorHandle 向量，预留足够的空间
  std::vector<RAIIAtenTensorHandle> result;
  result.reserve(size);
  // 遍历原始 AtenTensorHandle 数组
  for (size_t i = 0; i < size; i++) {
    // 将每个 AtenTensorHandle 转移到 RAIIAtenTensorHandle 中，并置空原始指针
    result.emplace_back(handles[i]);
    handles[i] = nullptr;
  }
  return result;  // 返回转移后的 RAIIAtenTensorHandle 向量
}

// 表示一个常量的句柄
class ConstantHandle {
 public:
  // 默认构造函数
  ConstantHandle() = default;

  // 显式构造函数，接受一个 AtenTensorHandle 句柄
  explicit ConstantHandle(AtenTensorHandle handle) : handle_(handle) {
    // 使用 handle 获取数据指针，并进行错误检查
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(handle_, &data_));
  }

  // 类型转换操作符，将 ConstantHandle 转换为 AtenTensorHandle
  operator AtenTensorHandle() const {
    return handle_;
  }

  // 返回句柄的 AtenTensorHandle
  AtenTensorHandle tensor() const {
    return handle_;
  }

  // 返回数据指针
  void* data_ptr() const {
    return data_;
  }

 private:
  AtenTensorHandle handle_;  // 存储的 AtenTensorHandle 句柄
  void* data_ = nullptr;     // 数据指针，默认为空
};

// 包装函数，用于获取 ConstantHandle 对象的数据指针
inline void* get_data_ptr_wrapper(const ConstantHandle& constant) {
  return constant.data_ptr();
}

// 如果需要，解封 RAII 句柄
inline const ConstantHandle& unwrap_raii_handle_if_needed(
    const ConstantHandle& handle) {
  return handle;  // 直接返回传入的 ConstantHandle
}

// 禁用函数声明，禁止调用该函数
inline AtenTensorHandle wrap_with_raii_handle_if_needed(
    const ConstantHandle& handle) = delete;

// 定义宏，用于缓存 Torch 的数据类型
#define CACHE_TORCH_DTYPE(typename) \
  static auto cached_torch_dtype_##typename = aoti_torch_dtype_##typename()

// 定义宏，用于缓存 Torch 的设备类型
#define CACHE_TORCH_DEVICE(device)                \
  static auto cached_torch_device_type_##device = \
      aoti_torch_device_type_##device()

// 定义宏，用于缓存 Torch 的布局类型
#define CACHE_TORCH_LAYOUT(layout) \
  static auto cached_torch_layout_##layout = aoti_torch_layout_##layout()

// 命名空间结束标记
} // namespace torch::aot_inductor
```