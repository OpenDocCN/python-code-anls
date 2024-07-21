# `.\pytorch\torch\csrc\lazy\core\tensor_util.h`

```
#pragma once
// 预处理指令：确保头文件只包含一次

#include <torch/csrc/lazy/backend/backend_interface.h>
// 引入 Torch 惰性计算模块的后端接口头文件

#include <torch/csrc/lazy/core/shape.h>
// 引入 Torch 惰性计算模块的形状定义头文件

#include <ATen/FunctionalTensorWrapper.h>
// 引入 ATen 的功能性张量包装头文件

#include <string>
// 引入标准字符串库

#include <vector>
// 引入标准向量库

namespace torch {
namespace lazy {

TORCH_API std::vector<int64_t> ComputeArrayStrides(
    c10::ArrayRef<int64_t> sizes);
// 声明 ComputeArrayStrides 函数，计算数组的步幅

TORCH_API std::vector<at::Tensor> DataHandlesToTensors(
    c10::ArrayRef<BackendDataPtr> data_handles,
    at::ScalarType dest_element_type);
// 声明 DataHandlesToTensors 函数，将数据句柄转换为张量集合

// Uploads an ATEN tensor data to the device and fetches the corresponding
// device data handle.
TORCH_API BackendDataPtr
TensorToDataHandle(const at::Tensor& tensor, const BackendDevice& device);
// 声明 TensorToDataHandle 函数，将 ATEN 张量数据上传到设备并获取对应的数据句柄

// Retrieves the device data handles by parallel uploading data onto the
// corresponding devices.
TORCH_API std::vector<BackendDataPtr> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<BackendDevice>& devices);
// 声明 CreateTensorsData 函数，通过并行上传数据到对应设备来检索设备数据句柄集合

// Makes a deep copy of an ATEN tensor.
inline at::Tensor CopyTensor(const at::Tensor& ref) {
  return ref.to(ref.options(), /*non_blocking=*/false, /*copy=*/true);
}
// 定义 inline 函数 CopyTensor，对 ATEN 张量进行深拷贝

// Same as above, with an additional cast.
inline at::Tensor CopyTensor(
    const at::Tensor& ref,
    at::ScalarType dest_type,
    bool copy = true) {
  return ref.to(ref.options().dtype(dest_type), /*non_blocking=*/false, copy);
}
// 定义 inline 函数 CopyTensor，带有额外的类型转换参数

template <typename T, typename S>
T OptionalOr(const std::optional<S>& value, T defval) {
  return value ? static_cast<T>(*value) : defval;
}
// 声明模板函数 OptionalOr，用于返回可选值或默认值

// Unwraps tensor to target dtype if it's a wrapped number.
inline at::Tensor UnwrapNumber(const at::Tensor& tensor, at::ScalarType dtype) {
  return tensor.unsafeGetTensorImpl()->is_wrapped_number() ? tensor.to(dtype)
                                                           : tensor;
}
// 定义 inline 函数 UnwrapNumber，如果张量是包装数，则将其解包到目标数据类型

template <typename T>
at::Scalar MakeIntScalar(T value) {
  return at::Scalar(static_cast<int64_t>(value));
}
// 声明模板函数 MakeIntScalar，用于创建整数标量

// Routing values to device data maximizes the changes for compilation cache
// hits, but it can prevent the compiler to perform optimizations. So tensor
// values which are within a given set, are routed to constant scalars if this
// API returns true.
TORCH_API bool IsSpecialScalar(const at::Scalar& value);
// 声明 IsSpecialScalar 函数，判断是否特殊标量值，用于路由到设备数据

// Note: returns a reference instead of a fresh tensor to avoid refcount bumps.
inline const at::Tensor& maybe_unwrap_functional(const at::Tensor& tensor) {
  if (at::functionalization::impl::isFunctionalTensor(tensor)) {
    return at::functionalization::impl::unsafeGetFunctionalWrapper(tensor)
        ->value();
  } else {
    return tensor;
  }
}
// 定义 inline 函数 maybe_unwrap_functional，可能解包功能性张量，返回引用避免增加引用计数

} // namespace lazy
} // namespace torch
```