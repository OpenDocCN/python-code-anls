# `.\pytorch\aten\src\ATen\native\cuda\Resize.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/Resize.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/PeerToPeerAccess.h>
#include <ATen/native/ResizeCommon.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/resize_native.h>
#endif

namespace at::native {

// 在 CUDA 下重新调整存储器的大小
void resize_bytes_cuda(StorageImpl* storage, size_t size_bytes) {
  // 检查存储器是否可调整大小
  TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");
  // 获取分配器
  auto allocator = storage->allocator();
  // 检查分配器是否存在
  TORCH_CHECK(allocator != nullptr, "Trying to resize storage without an allocator");

  // 获取存储器所在的设备
  c10::Device device = storage->device();

  // 如果大小为0，设置数据指针为空，并返回
  if (size_bytes == 0) {
    storage->set_data_ptr_noswap(at::DataPtr(nullptr, device));
    storage->set_nbytes(0);
    return;
  }

  // 切换到存储器所在的 CUDA 设备
  c10::cuda::CUDAGuard guard(device.index());
  // 分配新的数据指针
  at::DataPtr data = allocator->allocate(size_bytes);
  // 如果存储器有数据，进行异步内存复制
  if (storage->data_ptr()) {
    // 懒初始化 CUDA 上下文
    at::globalContext().lazyInitCUDA();

    // 执行 CUDA 设备到设备的内存复制
    C10_CUDA_CHECK(
        cudaMemcpyAsync(
            data.get(),
            storage->data(),
            std::min(storage->nbytes(), size_bytes),
            cudaMemcpyDeviceToDevice,
            c10::cuda::getCurrentCUDAStream()));
  }

  // 覆盖原始数据指针，完成大小调整
  storage->set_data_ptr_noswap(std::move(data));
  storage->set_nbytes(size_bytes);
}

// CUDA 下的张量尺寸调整操作
const Tensor& resize_cuda_(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  // 如果张量有命名维度，调用命名张量的尺寸调整函数
  if (self.has_names()) {
    return resize_named_tensor_(self, size, optional_memory_format);
  }
  // 获取张量的实现指针
  auto* self_ = self.unsafeGetTensorImpl();
  // 获取旧的存储器字节数
  int64_t old_storage_nbytes = self_->unsafe_storage() ? self_->unsafe_storage().nbytes() : 0;
  // 调用 CUDA 下的尺寸调整实现函数
  resize_impl_cuda_(self_, size, /*strides=*/c10::nullopt);
  // 如果指定了内存格式，重新整理张量的存储
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    // 检查是否支持给定的内存格式
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    // 重置张量的存储方式
    self_->empty_tensor_restride(memory_format);
  }
  // 检查是否启用确定性操作，如果是，则填充未初始化的内存
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_resize_deterministic_(self, old_storage_nbytes);
  }
  // 返回调整后的张量
  return self;
}

} // namespace at::native
```