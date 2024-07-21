# `.\pytorch\aten\src\ATen\native\vulkan\ops\Copy.h`

```
#pragma once
// 使用预处理指令，确保头文件只被编译一次

#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API，则包含以下内容

#include <ATen/native/vulkan/ops/Common.h>
// 包含 Vulkan 操作的通用头文件

namespace at {
namespace native {
namespace vulkan {
namespace ops {

void transfer_cpu_to_vulkan(const Tensor&, vTensor&);
// 将 CPU 上的 Tensor 数据传输到 Vulkan 的 vTensor 中

void transfer_vulkan_to_cpu(vTensor&, Tensor&);
// 将 Vulkan 的 vTensor 数据传输到 CPU 上的 Tensor 中

void pack_cpu_to_vulkan(const Tensor& src, vTensor& dst);
// 将 CPU 上的 Tensor 数据打包到 Vulkan 的 vTensor 中

void pack_vulkan_to_cpu(vTensor& src, Tensor& dst);
// 将 Vulkan 的 vTensor 数据打包到 CPU 上的 Tensor 中

Tensor& copy_(Tensor& dst, const Tensor& src);
// 复制源 Tensor 的数据到目标 Tensor 中，并返回目标 Tensor 的引用

vTensor to_vulkan(
    at::Tensor& src,
    const api::StorageType storage_type = api::StorageType::TEXTURE_3D);
// 将 ATen 中的 Tensor 转换为 Vulkan 的 vTensor，可以指定存储类型

at::Tensor from_vulkan(vTensor& v_src);
// 将 Vulkan 的 vTensor 转换为 ATen 中的 Tensor

//
// Utility functions for memcpy
//

template <typename T>
void memcpy_to_mapping_impl(const Tensor& src, api::MemoryMap& dst_mapping) {
  T* data_ptr = dst_mapping.template data<T>();
  // 获取目标映射的数据指针

  memcpy(
      data_ptr,
      src.const_data_ptr<T>(),
      std::min(src.nbytes(), dst_mapping.nbytes()));
  // 使用 memcpy 将源 Tensor 的数据复制到目标映射中，最多复制最小的字节数

}

template <typename T>
void memcpy_from_mapping_impl(api::MemoryMap& src_mapping, Tensor& dst) {
  T* data_ptr = src_mapping.template data<T>();
  // 获取源映射的数据指针

  memcpy(
      dst.mutable_data_ptr<T>(),
      data_ptr,
      std::min(src_mapping.nbytes(), dst.nbytes()));
  // 使用 memcpy 将源映射的数据复制到目标 Tensor 中，最多复制最小的字节数

}

inline void memcpy_from_mapping_bool(api::MemoryMap& src_mapping, Tensor& dst) {
  uint8_t* src_ptr = src_mapping.template data<uint8_t>();
  // 获取源映射的数据指针，类型为 uint8_t

  bool* dst_ptr = dst.mutable_data_ptr<bool>();
  // 获取目标 Tensor 的数据指针，类型为 bool

  for (int i = 0; (unsigned)i < std::min(src_mapping.nbytes(), dst.nbytes());
       ++i) {
    dst_ptr[i] = static_cast<bool>(src_ptr[i]);
    // 遍历并将源映射的数据以布尔类型复制到目标 Tensor 中
  }
}

inline void memcpy_to_mapping_uint8(
    const Tensor& src,
    api::MemoryMap& dst_mapping) {
  bool* src_ptr = src.mutable_data_ptr<bool>();
  // 获取源 Tensor 的数据指针，类型为 bool

  uint8_t* dst_ptr = dst_mapping.template data<uint8_t>();
  // 获取目标映射的数据指针，类型为 uint8_t

  for (int i = 0; (unsigned)i < std::min(dst_mapping.nbytes(), src.nbytes());
       ++i) {
    dst_ptr[i] = static_cast<uint8_t>(src_ptr[i]);
    // 遍历并将源 Tensor 的数据以 uint8_t 类型复制到目标映射中
  }
}

void memcpy_to_mapping(const Tensor& src, api::MemoryMap& dst_mapping);
// 将源 Tensor 的数据复制到目标映射中

void memcpy_from_mapping(api::MemoryMap& src_mapping, Tensor& dst);
// 将源映射的数据复制到目标 Tensor 中

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
// 结束条件，结束预处理指令区块
```