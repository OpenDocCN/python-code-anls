# `.\pytorch\aten\src\ATen\native\Resize.h`

```
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/EmptyTensor.h>
#include <ATen/TensorUtils.h>

#include <c10/core/CPUAllocator.h>

#include <utility>

// 命名空间声明：at::native
namespace at::native {

// TODO: make all operations that resize given outputs use this function
//   for consistency and maintainability.
//   Some operations like `cat` might not be able to make the use of
//   resize_output directly. For more details to understand how it works in `cat`,
//   see https://github.com/pytorch/pytorch/pull/62560#discussion_r687363362
// Resizes outputs
// Functions accepting output tensors, like with the "out" kwarg, should
//   call this function to handle resizing their output tensor.
// Issues a warning if the output tensor has one or more elements and
//   needs resizing
// NOTE: In the future the warning will become an error
// Returns a bool saying whether or not the resize actually happened or not
TORCH_API bool resize_output(const Tensor& output, IntArrayRef shape);

// WARNING: Do NOT call this directly. If you are resizing an output and want
// to support dynamic shapes call at::resize__symint and resize_output_check_symint.
// For more details, see: https://github.com/pytorch/pytorch/pull/111530/files#r1365845272
// Resizes outputs with symbolic integer shapes
TORCH_API bool resize_output_symint(const Tensor& output, SymIntArrayRef shape);

// Utility for resize_output
//  Returns a bool saying resize should happen or not and
//  raises a warning if resizing for one or more elements
TORCH_API bool resize_output_check(const Tensor& output, IntArrayRef shape);

// Utility for resize_output with symbolic integer shapes
TORCH_API bool resize_output_check_symint(const Tensor& output, SymIntArrayRef shape);

// Resizes storage for CPU tensors based on byte size
TORCH_API void resize_bytes_cpu(StorageImpl* storage, size_t size_bytes);

// Resizes storage metadata for CPU tensors with symbolic integer sizes
TORCH_API void resize_bytes_meta(StorageImpl* storage, c10::SymInt size_bytes);

// Resizes storage for CPU tensors without CUDA support with symbolic integer sizes
TORCH_API void resize_bytes_nocuda(const Storage& storage, c10::SymInt size_bytes);

// Resizes the underlying storage of a CPU tensor if needed
inline void maybe_resize_storage_cpu(TensorImpl* self, size_t new_size_bytes) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in cuda/Resize.h)
  if (self->numel() == 0) {
    return;
  }

  const Storage& storage = self->unsafe_storage();
  if (!storage) {
    auto new_storage = c10::make_intrusive<StorageImpl>(
        StorageImpl::use_byte_size_t(),
        new_size_bytes,
        c10::GetCPUAllocator(),
        true);
    self->set_storage_keep_dtype(std::move(new_storage));
  } else if (new_size_bytes > storage.nbytes()) {
    resize_bytes_cpu(storage.unsafeGetStorageImpl(), new_size_bytes);
  }
}

// Internal function to resize the implementation of a CPU tensor
TORCH_API TensorImpl* resize_impl_cpu_(
    TensorImpl* self,
    IntArrayRef size,
    at::OptionalIntArrayRef stride,
    bool resize_storage = true);

// Template specialization to prevent conversion of symbolic integers
template <typename T>
T maybe_convert_symint(c10::SymInt) = delete;

// Template specialization to allow conversion of symbolic integers to themselves
template <>
inline c10::SymInt maybe_convert_symint(c10::SymInt x) { return x; }

// End of namespace declaration for at::native
} // namespace at::native
// 将 SymInt 类型 x 转换为 int64_t 类型，并检查其是否在符号整数范围内
inline int64_t maybe_convert_symint(c10::SymInt x) { return x.guard_int(__FILE__, __LINE__); }

/**
 * 检查给定大小、步幅、存储偏移量以及数据类型是否在存储范围内。
 * 如果存储大小为零，则允许任意元素数量的张量。
 */
template <typename T>
inline void checkInBoundsForStorage(
    ArrayRef<T> size,
    ArrayRef<T> stride,
    T storage_offset,
    const caffe2::TypeMeta& data_type,
    const Storage& new_storage) {
  // 计算存储所需的字节数
  T storage_size_bytes =
      at::detail::computeStorageNbytes(size, stride, data_type.itemsize());
  // 计算存储偏移量的字节数
  T storage_offset_bytes = storage_offset * data_type.itemsize();
  if (storage_size_bytes == 0) {
    // 若存储大小为零，则返回
    // 注意：具有任意零维度的张量的存储可以具有任意数量的元素
    return;
  }
  // 计算新存储的字节数，并尝试将 SymInt 类型转换为 T 类型
  T new_storage_size_bytes = maybe_convert_symint<T>(new_storage.sym_nbytes());
  // 检查存储和偏移量是否超出新存储的大小范围
  TORCH_CHECK(
      storage_size_bytes + storage_offset_bytes <= new_storage_size_bytes,
      "setStorage: sizes ",
      size,
      ", strides ",
      stride,
      ","
      " storage offset ",
      storage_offset,
      ", and itemsize ",
      data_type.itemsize(),
      " requiring a storage size of ",
      storage_size_bytes + storage_offset_bytes,
      " are out of bounds for storage of size ",
      new_storage_size_bytes);
}

/**
 * 检查并设置张量的存储。
 * 如果 stride 存在，则检查 size 和 stride 的长度是否相等。
 * 在 DEBUG 模式下，检查 size 的长度是否小于等于 INT_MAX。
 */
template <typename T>
inline void checkSetStorage(Tensor& result, Storage storage, T storage_offset,
                                   ArrayRef<T> size, ArrayRef<T> stride) {
  // FIXME: stride 应该是可选的
  if (stride.data()) {
    // 检查 size 和 stride 的长度是否相等
    TORCH_CHECK(size.size() == stride.size(), "unequal size length (", size.size(),
                                              ") and stride length (", stride.size(), ")");
  }

#ifdef DEBUG
  // 在 DEBUG 模式下，检查 size 的长度是否小于等于 INT_MAX
  TORCH_CHECK(size.size() <= INT_MAX, "size length (", size.size(), ") greater than INT_MAX");
#endif

  // 检查存储是否为别名，如果不是，则设置新的存储
  if (!result.storage().is_alias_of(storage)) {
    // 在 PyTorch 中不允许存储为空
    TORCH_INTERNAL_ASSERT(storage);
    TORCH_INTERNAL_ASSERT(result.storage());

    // 检查设备是否匹配，不匹配则抛出错误信息
    TORCH_CHECK(result.storage().device() == storage.device(),
                "Attempted to set the storage of a tensor on device \"", result.storage().device(),
                "\" to a storage on different device \"", storage.device(),
                "\".  This is no longer allowed; the devices must match.");
    // 设置张量的存储，保持数据类型不变
    result.unsafeGetTensorImpl()->set_storage_keep_dtype(std::move(storage));
  }

  // 检查存储偏移量是否为非负数
  TORCH_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);
}

/**
 * 设置张量的尺寸、步幅和存储偏移量。
 * （size、stride、storage_offset）必须在张量的存储范围内。
 */
template <typename T>
inline void setStrided(
    const Tensor& self,
    ArrayRef<T> size,
    ArrayRef<T> stride,
    T storage_offset) {
}

// 结束 at::native 命名空间
} // namespace at::native
```