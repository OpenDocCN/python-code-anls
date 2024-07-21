# `.\pytorch\aten\src\ATen\DLConvertor.h`

```
#pragma once
// 使用 pragma once 防止头文件被多次包含

#include <ATen/ATen.h>
// 包含 ATen 库的头文件

#include <ATen/Tensor.h>
// 包含 ATen 库的 Tensor 头文件

#include <ATen/dlpack.h>
// 包含 ATen 库的 DLPack 头文件

// ATen 命名空间中的函数和类声明
namespace at {

// 定义在 ATen 中的函数 toScalarType，将 DLPack 中的数据类型转换为 ATen 中的 ScalarType
TORCH_API ScalarType toScalarType(const DLDataType& dtype);

// 定义在 ATen 中的函数 toDLPack，将 ATen 中的 Tensor 包装为 DLPack 中的 DLManagedTensor
TORCH_API DLManagedTensor* toDLPack(const Tensor& src);

// 定义在 ATen 中的函数 fromDLPack，将 DLPack 中的 DLManagedTensor 转换为 ATen 中的 Tensor
TORCH_API Tensor fromDLPack(DLManagedTensor* src);

// 以下为被标记为已弃用的函数，建议迁移到非 const 变体
C10_DEPRECATED_MESSAGE("Please migrate to a non-const variant")
inline Tensor fromDLPack(const DLManagedTensor* src) {
  return fromDLPack(const_cast<DLManagedTensor*>(src));
}

// 定义在 ATen 中的函数 fromDLPack，将 DLPack 中的 DLManagedTensor 转换为 ATen 中的 Tensor，并提供自定义删除器
TORCH_API Tensor fromDLPack(DLManagedTensor* src, std::function<void(void*)> deleter);

// 定义在 ATen 中的函数 getDLDataType，获取 Tensor 对应的 DLPack 数据类型 DLDataType
TORCH_API DLDataType getDLDataType(const Tensor& t);

// 定义在 ATen 中的函数 getDLContext，获取 Tensor 对应的 DLPack 设备信息 DLDevice
TORCH_API DLDevice getDLContext(const Tensor& tensor, const int64_t& device_id);

} // namespace at
// ATen 命名空间结束
```