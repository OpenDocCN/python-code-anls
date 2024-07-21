# `.\pytorch\aten\src\ATen\native\quantized\cpu\IntReprQuant.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/int_repr_native.h>
#endif

namespace at {
namespace native {

// 当输入的张量为非稠密时，即分配的内存空间大于所有元素使用的内存时，
// 将其转换为稠密张量；否则，保持输出的内存格式与输入相同
Tensor int_repr_quantized_cpu(const Tensor& self) {
  Tensor dst;
  // 使用 AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES 宏进行类型分发，用于处理量化整数和子字节类型
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(self.scalar_type(), "int_repr", [&]() {
    // 如果量化位宽为 4 或 2
    if (bit_width == 4 || bit_width == 2) {
      // 计算输出大小，以确保能够容纳所有元素
      int64_t out_size = at::ceil_div(self.numel() * bit_width, (int64_t)8);
      // 创建一个空的张量，用于存储转换后的数据
      dst = at::empty(
          {out_size},
          self.options().dtype(UNDERLYING_TYPE),
          self.suggest_memory_format());
      // 将输入张量的数据转换为底层类型的数据，并复制到新创建的张量中
      const underlying_t* qdata = reinterpret_cast<const underlying_t*>(self.const_data_ptr<scalar_t>());
      for (const auto i : c10::irange(dst.numel())) {
        dst[i] = static_cast<underlying_t>(qdata[i]);
      }
    } else {
      // 对于其他位宽的情况，创建一个与输入相同大小的空张量
      dst = at::empty(
          self.sizes(),
          self.options().dtype(UNDERLYING_TYPE),
          self.suggest_memory_format());
      // 配置张量迭代器，用于处理输入和输出张量之间的操作
      auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(dst)
        .add_input(self)
        .build();
      // 调用 CPU 内核函数，处理迭代器配置和指定的操作
      cpu_kernel(iter, [](scalar_t value) -> underlying_t { return value.val_; });
    }
  });
  // 返回处理后的目标张量
  return dst;
}

} // namespace native
} // namespace at
```