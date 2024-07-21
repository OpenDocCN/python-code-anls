# `.\pytorch\torch\csrc\inductor\aoti_torch\utils.h`

```
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <ATen/Generator.h>
// 包含 ATen 库中的 Generator 头文件，用于张量生成器的操作
#include <ATen/Tensor.h>
// 包含 ATen 库中的 Tensor 头文件，用于张量的操作
#include <ATen/core/List.h>
// 包含 ATen 库中的 List 头文件，用于列表操作
#include <c10/core/DeviceType.h>
// 包含 c10 库中的 DeviceType 头文件，用于设备类型的定义
#include <c10/core/SymIntArrayRef.h>
// 包含 c10 库中的 SymIntArrayRef 头文件，用于符号化整数数组引用的操作
#include <c10/util/ArrayRef.h>
// 包含 c10 库中的 ArrayRef 头文件，用于数组引用的操作
#include <c10/util/Logging.h>
// 包含 c10 库中的 Logging 头文件，用于日志记录
#include <c10/util/Optional.h>
// 包含 c10 库中的 Optional 头文件，用于可选值的操作
#include <c10/util/OptionalArrayRef.h>
// 包含 c10 库中的 OptionalArrayRef 头文件，用于可选数组引用的操作
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
// 包含 torch 库中的 shim.h 头文件，提供 aoti_torch 的 C 语言接口的定义

#define AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(...)    \
  try {                                                    \
    __VA_ARGS__                                            \
  } catch (const std::exception& e) {                      \
    LOG(ERROR) << "Exception in aoti_torch: " << e.what(); \
    return AOTI_TORCH_FAILURE;                             \
  } catch (...) {                                          \
    LOG(ERROR) << "Exception in aoti_torch: UNKNOWN";      \
    return AOTI_TORCH_FAILURE;                             \
  }                                                        \
  return AOTI_TORCH_SUCCESS;
// 定义宏，用于执行异常转换为错误码处理逻辑

namespace torch::aot_inductor {
// 命名空间声明，用于包裹 aot_inductor 相关的函数和类型

inline at::Tensor* tensor_handle_to_tensor_pointer(AtenTensorHandle handle) {
  return reinterpret_cast<at::Tensor*>(handle);
}
// 将 AtenTensorHandle 类型的句柄转换为 at::Tensor* 类型指针的函数

inline AtenTensorHandle tensor_pointer_to_tensor_handle(at::Tensor* tensor) {
  return reinterpret_cast<AtenTensorHandle>(tensor);
}
// 将 at::Tensor* 类型指针转换为 AtenTensorHandle 类型句柄的函数

inline at::Generator* generator_handle_to_generator_pointer(
    AtenGeneratorHandle handle) {
  return reinterpret_cast<at::Generator*>(handle);
}
// 将 AtenGeneratorHandle 类型的句柄转换为 at::Generator* 类型指针的函数

inline AtenGeneratorHandle generator_pointer_to_generator_handle(
    at::Generator* generator) {
  return reinterpret_cast<AtenGeneratorHandle>(generator);
}
// 将 at::Generator* 类型指针转换为 AtenGeneratorHandle 类型句柄的函数

inline AtenTensorHandle new_tensor_handle(at::Tensor&& tensor) {
  at::Tensor* new_tensor = new at::Tensor(std::move(tensor));
  return tensor_pointer_to_tensor_handle(new_tensor);
}
// 创建新的张量句柄的函数，接受一个右值引用的 at::Tensor，并返回对应的句柄

inline void assert_inf_and_nan(
    const std::string& tensor_name,
    at::Tensor& check_tensor) {
  auto flattened = check_tensor.view({-1});
  // 将张量展平为一维

  for (int64_t i = 0; i < flattened.numel(); i++) {
    auto value = flattened[i].item<float>();
    // 获取张量中第 i 个元素的值

    if (std::isinf(value)) {
      throw std::runtime_error("At least one INF in " + tensor_name);
      // 如果值为无穷大，则抛出异常
    } else if (std::isnan(value)) {
      throw std::runtime_error("At least one NaN in " + tensor_name);
      // 如果值为 NaN，则抛出异常
    }
  }
}
// 断言函数，用于检查张量中是否包含无穷大或 NaN

// utility functions to convert a pointer to an optional value

template <class T>
inline std::optional<T> pointer_to_optional(T* ptr) {
  return ptr ? c10::make_optional(*ptr) : c10::nullopt;
}
// 将指针转换为可选值的通用函数模板，用于普通类型 T

template <class T, class U, typename = std::enable_if_t<!std::is_same_v<T, U>>>
inline std::optional<T> pointer_to_optional(U* ptr) {
  return ptr ? c10::make_optional<T>(T(*ptr)) : c10::nullopt;
}
// 将指针转换为可选值的函数模板，用于类型转换从 U 到 T，排除 T 和 U 相同的情况

template <>
inline std::optional<at::Tensor> pointer_to_optional(AtenTensorHandle* ptr) {
  return ptr ? c10::make_optional(*tensor_handle_to_tensor_pointer(*ptr))
             : c10::nullopt;
}
// 特化模板函数，将 AtenTensorHandle* 类型指针转换为 at::Tensor 类型的可选值
    // 如果指针 ptr 不为空，则使用 tensor_handle_to_tensor_pointer 函数将 AtenTensorHandle 解引用，并使用 c10::make_optional 将其包装成 optional 类型返回
    // 否则返回一个空的 optional 对象
    const AtenTensorHandle* ptr) {
      return ptr ? c10::make_optional(*tensor_handle_to_tensor_pointer(*ptr))
                 : c10::nullopt;
    }
// 结束当前命名空间 torch::aot_inductor

template <>
// 特化模板函数，将 AtenGeneratorHandle* 指针转换为 std::optional<at::Generator>
inline std::optional<at::Generator> pointer_to_optional(
    AtenGeneratorHandle* ptr) {
  return ptr ? c10::make_optional(*generator_handle_to_generator_pointer(*ptr))
             : c10::nullopt;
}

// 将 int32_t* 和 device_index 转换为 std::optional<c10::Device>
inline std::optional<c10::Device> pointer_to_optional_device(
    int32_t* device_type,
    int32_t device_index) {
  return device_type ? c10::make_optional(c10::Device(
                           static_cast<c10::DeviceType>(*device_type),
                           static_cast<c10::DeviceIndex>(device_index)))
                     : c10::nullopt;
}

// 用于判断类型 T 是否为 std::optional<T> 的模板结构体
template <typename T>
struct is_optional : std::false_type {};

// 特化模板结构体，用于判断 std::optional<T> 是否为 true_type
template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

// 将指针 ptr 和长度 len 转换为 c10::ArrayRef<T>
template <class T>
inline c10::ArrayRef<T> pointer_to_list(T* ptr, int64_t len) {
  return c10::ArrayRef<T>(ptr, len);
}

// 模板函数，将 U* 指针和长度 len 转换为 std::vector<T>
// 并排除 T 和 U 相同，以及 T 为 std::optional 的情况
template <
    class T,
    class U,
    typename = std::enable_if_t<!std::is_same_v<T, U>>,
    typename = std::enable_if_t<!is_optional<T>::value>>
inline std::vector<T> pointer_to_list(U* ptr, int64_t len) {
  // std::vector<T> 在调用点会隐式转换为 c10::ArrayRef<T>
  std::vector<T> result;
  result.reserve(len);
  for (int64_t i = 0; i < len; i++) {
    result.emplace_back(T(ptr[i]));
  }
  return result;
}

// 特化模板函数，将 U** 指针和长度 len 转换为 std::vector<T>
// 这里 U** 表示一个可选参数列表
template <class T, class U, typename = std::enable_if_t<is_optional<T>::value>>
inline std::vector<T> pointer_to_list(U** ptr, int64_t len) {
  // std::vector<T> 在调用点会隐式转换为 c10::ArrayRef<T>
  std::vector<T> result;
  result.reserve(len);
  for (int64_t i = 0; i < len; i++) {
    result.emplace_back(pointer_to_optional<T>(ptr[i]));
  }
  return result;
}

// 特化模板函数，将 const AtenTensorHandle* 指针和长度 len 转换为 std::vector<at::Tensor>
template <>
inline std::vector<at::Tensor> pointer_to_list(
    const AtenTensorHandle* ptr,
    int64_t len) {
  std::vector<at::Tensor> result;
  result.reserve(len);
  for (int64_t i = 0; i < len; i++) {
    result.emplace_back(*tensor_handle_to_tensor_pointer(*ptr));
  }
  return result;
}

// 特化模板函数，将 const AtenTensorHandle** 指针和长度 len 转换为 std::vector<std::optional<at::Tensor>>
template <>
inline std::vector<std::optional<at::Tensor>> pointer_to_list(
    const AtenTensorHandle** ptr,
    int64_t len) {
  std::vector<std::optional<at::Tensor>> result;
  result.reserve(len);
  for (int64_t i = 0; i < len; i++) {
    result.emplace_back(pointer_to_optional<at::Tensor>(ptr[i]));
  }
  return result;
}

// 将 int32_t* 指针转换为长度为 N 的 std::array<bool, N>
template <int N>
inline std::array<bool, N> pointer_to_list(const int32_t* ptr) {
  std::array<bool, N> result;
  std::copy(ptr, ptr + N, result.begin());
  return result;
}

// 将 U** 指针和长度 len 转换为 std::optional<c10::ArrayRef<T>>
// 如果 ptr 为 nullptr，则返回 c10::nullopt
template <class T, class U>
inline std::optional<c10::ArrayRef<T>> pointer_to_optional_list(
    U** ptr,
    int64_t len) {
  return ptr
      ? c10::make_optional<c10::ArrayRef<T>>(pointer_to_list<T>(*ptr, len))
      : c10::nullopt;
}
```