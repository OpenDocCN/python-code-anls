# `.\pytorch\aten\src\ATen\native\utils\Factory.cpp`

```py
// 定义编译选项，仅包含断言方法的运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含必要的头文件：命名张量工具、工厂函数、CPU 分配器、累加工具等
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/utils/Factory.h>
#include <c10/core/CPUAllocator.h>
#include <c10/util/accumulate.h>

// 进入命名空间 at::native::mobile
namespace at {
namespace native {
namespace mobile {

// 函数：根据给定参数创建具有尾部填充的空张量
Tensor empty_with_tail_padding(
    const IntArrayRef size,                     // 张量的维度尺寸
    const caffe2::TypeMeta dtype,               // 张量的数据类型
    const c10::MemoryFormat memory_format,      // 张量的内存格式
    std::optional<DimnameList> maybe_names) {   // 可选的维度名列表

  auto* const allocator_ptr = c10::GetDefaultMobileCPUAllocator();  // 获取默认的移动端 CPU 分配器
  const int64_t nelements = c10::multiply_integers(size);           // 计算张量中元素的总数
  size_t size_bytes = nelements * dtype.itemsize();                 // 计算张量所需的总字节数

  // 使用给定的分配器和数据类型创建张量对象
  Tensor tensor(c10::make_intrusive<c10::TensorImpl>(
      c10::Storage{                                            // 创建存储对象
          c10::Storage::use_byte_size_t(),                     // 使用字节大小
          size_bytes,                                          // 指定大小
          allocator_ptr->allocate(size_bytes),                 // 分配内存
          allocator_ptr,                                       // 分配器
          /*resizable=*/true,                                  // 可调整大小
      },
      DispatchKeySet{DispatchKey::CPU},                         // 设置分发键集
      dtype));                                                 // 指定数据类型

  // 如果可能，传播和保留维度名到张量
  return namedinference::propagate_names_if_present_and_nonempty(
      tensor.resize_(size, memory_format),                      // 调整张量的大小和内存格式
      maybe_names);                                            // 可选的维度名列表
}

// 函数：如果需要，为输入张量分配填充后的连续内存
Tensor allocate_padded_contiguous_if_needed(
    const Tensor& input,                       // 输入张量
    const c10::MemoryFormat memory_format) {   // 所需的内存格式

  const auto* const allocator = input.storage().allocator();       // 获取输入张量的分配器
  const auto* const mobile_allocator = c10::GetDefaultMobileCPUAllocator();  // 获取默认的移动端 CPU 分配器

  // 如果分配器相同并且在请求格式下是连续的，则无需重新分配张量
  if ((allocator == mobile_allocator) && input.is_contiguous(memory_format)) {
    return input;   // 直接返回输入张量
  }

  // 否则，需要重新分配张量，要么是因为分配器不同，要么是因为张量不在请求格式下连续
  // 在这种情况下，分配填充后的空张量并将输入张量拷贝进去
  Tensor padded_input = empty_with_tail_padding(
      input.sizes(),                      // 使用输入张量的尺寸
      input.options().dtype(),            // 使用输入张量的数据类型选项
      memory_format,                      // 使用请求的内存格式
      input.opt_names());                 // 使用输入张量的可选维度名

  return padded_input.copy_(input);       // 将输入张量拷贝到填充后的张量中
}

} // namespace mobile
} // namespace native
} // namespace at


这段代码定义了一些函数和数据结构，用于在移动设备上处理张量的分配和操作。
```