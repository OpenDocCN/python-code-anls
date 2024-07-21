# `.\pytorch\aten\src\ATen\native\ResizeCommon.h`

```
#pragma once


// 告诉编译器只包含本头文件一次，防止多重包含
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
// 如果没有定义 AT_PER_OPERATOR_HEADERS，则包含 NativeFunctions.h
#include <ATen/NativeFunctions.h>
#else
// 如果定义了 AT_PER_OPERATOR_HEADERS，则包含 empty.h
#include <ATen/ops/empty.h>
#endif

namespace at::native {

// 定义模板函数 storage_size_for，计算给定 size 和 stride 的存储大小
template <typename T>
inline T storage_size_for(ArrayRef<T> size, ArrayRef<T> stride) {
  // 断言 size 和 stride 的大小相同
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(size.size() == stride.size(),
      "storage_size_for(size, stride) requires that size and stride ",
      "have the same size as a precondition.");
  T storage_size = 1;
  // 计算存储大小
  for (const auto dim : c10::irange(size.size())) {
    if (size[dim] == 0) {
      storage_size = 0;
      break;
    }
    storage_size += (size[dim] - 1) * stride[dim];
  }
  return storage_size;
}

// 定义函数 resize_named_tensor_，用于调整命名张量的大小
inline const Tensor& resize_named_tensor_(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  // 断言输入张量有命名维度
  TORCH_INTERNAL_ASSERT(self.has_names());
  // 检查是否可以调整大小为指定的尺寸
  TORCH_CHECK(
      self.sizes() == size,
      "Cannot resize named tensor with resize_ or resize_as_ (tried to resize "
      "Tensor",
      self.names(),
      " with size ",
      self.sizes(),
      " to ",
      size,
      "). This may be caused by passing a named tensor ",
      "as an `out=` argument; please ensure that the sizes are the same. ");
  // 检查是否支持指定的内存格式
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "Unsupported memory format for named tensor resize ",
      optional_memory_format.value());
  return self;
}

// 函数 fill_resize_deterministic_ 用于在存储大小改变后填充新添加的元素
// 使用 NaN 或 MAX_INT 以确保输出一致性
inline const Tensor& fill_resize_deterministic_(const Tensor& tensor, int64_t old_storage_nbytes) {
  // 获取张量的存储
  const at::Storage& storage = tensor.unsafeGetTensorImpl()->unsafe_storage();
  // 计算改变大小后的存储字节大小
  int64_t new_storage_nbytes = storage.nbytes();
  int64_t old_storage_numel = old_storage_nbytes / tensor.itemsize();
  int64_t new_storage_numel = new_storage_nbytes / tensor.itemsize();
  // 如果新的存储元素数量大于旧的
  if (new_storage_numel > old_storage_numel) {
    // 创建一个与原张量类型和设备相同的空张量视图
    at::Tensor tensor_view = at::empty({}, at::TensorOptions().dtype(tensor.scalar_type()).device(tensor.device()));
    // 在指定偏移和大小下填充视图
    tensor_view.set_(
      storage,
      /*storage_offset=*/old_storage_numel,
      /*size=*/{new_storage_numel - old_storage_numel},
      /*stride=*/{1});
    // 填充空的确定性张量视图
    at::native::fill_empty_deterministic_(tensor_view);
  }
  return tensor;
}

} // namespace at::native
```