# `.\pytorch\aten\src\ATen\native\cuda\Resize.h`

```py
#pragma once
// 包含头文件 EmptyTensor.h 和 ResizeCommon.h，用于声明和定义相关函数和结构
#include <ATen/EmptyTensor.h>
#include <ATen/native/ResizeCommon.h>

// 包含 CUDA 相关的头文件，用于 CUDA 内存管理
#include <c10/cuda/CUDAGuard.h>

// 命名空间定义开始，命名空间为 at，嵌套命名空间为 native
namespace at { namespace native {

// 声明一个 CUDA C++ API 函数，用于调整 CUDA 存储空间大小
TORCH_CUDA_CPP_API void resize_bytes_cuda(StorageImpl* storage, size_t size_bytes);

// 定义一个内联函数 maybe_resize_storage_cuda，用于可能调整 CUDA 张量的存储空间大小
static inline void maybe_resize_storage_cuda(TensorImpl* self, size_t new_size_bytes) {
  // 如果张量中元素数为0，则不进行任何操作，直接返回
  if (self->numel() == 0) {
    return;
  }

  // 获取张量的不安全存储引用
  const Storage &storage = self->unsafe_storage();
  // 断言确保存储引用有效，若无效则输出错误信息
  TORCH_CHECK(storage, "Tensor: invalid null storage");
  
  // 如果新的大小超过当前存储空间大小，则调用 resize_bytes_cuda 函数进行存储空间的调整
  if (new_size_bytes > storage.nbytes()) {
    resize_bytes_cuda(storage.unsafeGetStorageImpl(), new_size_bytes);
  }
}

// 定义一个 CUDA 实现的 resize_impl_cuda_ 函数，用于调整张量大小
inline TensorImpl* resize_impl_cuda_(
    TensorImpl* self,
    IntArrayRef size,
    at::OptionalIntArrayRef stride) {
  // 如果当前张量大小和步幅与新的大小和步幅一致，则直接返回当前张量
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  // 计算张量元素的字节大小
  const auto itemsize = self->dtype().itemsize();
  // 获取张量的存储偏移量
  const auto storage_offset = self->storage_offset();
  size_t storage_size = 1;
  
  // 如果提供了步幅，则设置新的大小和步幅，并计算存储空间大小
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    storage_size = at::detail::computeStorageNbytes(
        size, *stride, itemsize, storage_offset);
  } else {
    // 否则设置为连续的大小，并计算存储空间大小
    self->set_sizes_contiguous(size);
    storage_size = at::detail::computeStorageNbytesContiguous(
        size, itemsize, storage_offset);
  }
  
  // 可能调整 CUDA 存储空间大小以适应新的存储空间大小
  maybe_resize_storage_cuda(self, storage_size);

  // 返回调整后的张量实现
  return self;
}

// 命名空间结束
}}
```