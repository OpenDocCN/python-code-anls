# `.\pytorch\aten\src\ATen\cuda\PinnedMemoryAllocator.cpp`

```
// 包含必要的头文件，用于 CUDA 异步操作和内存分配
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/Context.h>
#include <ATen/Config.h>
#include <ATen/TensorUtils.h>
#include <c10/core/Storage.h>
#include <ATen/ATen.h>
#include <ATen/CPUFunctions.h>

// 定义 at::native 命名空间，实现与 Tensor 相关的本地操作
namespace at::native {

// 检查给定的 Tensor 是否存储在 CUDA 固定内存中
bool is_pinned_cuda(const Tensor& self, std::optional<Device> device) {
  // 断言：如果指定了设备，则该设备必须是 CUDA 设备
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!device.has_value() || device->is_cuda());
  // TODO: unhook this
  // 调用 CUDA 钩子函数检查是否是固定内存指针
  return detail::getCUDAHooks().isPinnedPtr(self.storage().data());
}

// 将给定的 CUDA Tensor 锁定到固定内存
Tensor _pin_memory_cuda(const Tensor& self, std::optional<Device> device) {
  // 断言：如果指定了设备，则该设备必须是 CUDA 设备
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!device.has_value() || device->is_cuda());
  // 获取 CUDA 固定内存分配器
  auto* allocator = at::cuda::getPinnedMemoryAllocator();
  // 计算存储器的字节大小
  auto storage = Storage(
      Storage::use_byte_size_t(),
      detail::computeStorageNbytes(
          self.sizes(), self.strides(), self.dtype().itemsize()),
      allocator,
      /*resizable=*/false);
  // 使用 CPU 上的空 Tensor 创建一个具有给定存储器的 Tensor
  auto tensor = at::cpu::empty({0}, self.options()).set_(storage, 0, self.sizes(), self.strides());
  // 将输入 Tensor 的数据复制到新的固定内存 Tensor
  tensor.copy_(self);
  // 返回新的固定内存 Tensor
  return tensor;
}

} // namespace at::native
```