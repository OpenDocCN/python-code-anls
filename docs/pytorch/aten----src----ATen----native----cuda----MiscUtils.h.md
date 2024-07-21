# `.\pytorch\aten\src\ATen\native\cuda\MiscUtils.h`

```py
#pragma once
// 引入CUDA异常处理、上下文、配置和固定内存分配器的头文件
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>

// at命名空间内的native命名空间
namespace at {
namespace native {

// 将int64_t类型的value转换为int类型，并进行边界检查
static inline int cuda_int_cast(int64_t value, const char* varname) {
  // 使用static_cast将value转换为int类型
  auto result = static_cast<int>(value);
  // 使用TORCH_CHECK进行断言检查，确保转换后的int值与原始int64_t值相等
  TORCH_CHECK(static_cast<int64_t>(result) == value,
              "cuda_int_cast: The value of ", varname, "(", (long long)value,
              ") is too large to fit into a int (", sizeof(int), " bytes)");
  // 返回转换后的int值
  return result;
}

// 创建一个大小为size个元素、类型为T的数组，使用固定内存分配器分配
// 并将其封装在Storage对象中返回
template<class T>
static inline Storage pin_memory(int64_t size) {
  // 获取CUDA固定内存分配器的指针
  auto* allocator = cuda::getPinnedMemoryAllocator();
  // 计算数组所需的内存空间大小，单位为字节
  int64_t adjusted_size = size * sizeof(T);
  // 使用Storage类创建一个存储对象，使用字节大小作为单位
  // 设置大小为adjusted_size，分配器为allocator，不可调整大小
  return Storage(
      Storage::use_byte_size_t(),
      adjusted_size,
      allocator,
      /*resizable=*/false);
}

} // namespace native
} // namespace at
```