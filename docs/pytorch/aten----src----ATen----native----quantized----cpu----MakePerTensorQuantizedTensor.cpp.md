# `.\pytorch\aten\src\ATen\native\quantized\cpu\MakePerTensorQuantizedTensor.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏以仅包含方法操作符

#include <ATen/native/TensorIterator.h>
// 引入张量迭代器相关头文件

#include <ATen/native/cpu/Loops.h>
// 引入CPU环路相关头文件

#include <ATen/core/Tensor.h>
// 引入张量核心头文件

#include <ATen/Dispatch.h>
// 引入分发相关头文件

#ifndef AT_PER_OPERATOR_HEADERS
// 如果未定义 AT_PER_OPERATOR_HEADERS

#include <ATen/Functions.h>
// 引入ATen函数头文件

#else
// 否则，如果定义了 AT_PER_OPERATOR_HEADERS

#include <ATen/ops/_empty_affine_quantized.h>
// 引入空仿射量化头文件

#include <ATen/ops/_make_per_tensor_quantized_tensor_native.h>
// 引入创建每张量量化张量的原生头文件

#endif

namespace at {
namespace native {

Tensor make_per_tensor_quantized_tensor_cpu(
    const Tensor& self,
    double scale,
    int64_t zero_point) {
  // 创建一个空的仿射量化张量，具有给定的尺寸和数据类型
  Tensor dst = at::_empty_affine_quantized(
      self.sizes(),
      self.options().dtype(toQIntType(self.scalar_type())),
      scale,
      zero_point,
      self.suggest_memory_format());

  // 确保输入张量是连续的，并且采用建议的内存格式
  Tensor self_contig = self.contiguous(self.suggest_memory_format());

  // 使用宏来分发处理不同量化整数类型的操作
  AT_DISPATCH_QINT_TYPES(
      dst.scalar_type(), "make_per_tensor_quantized_tensor", [&]() {
        // 获取输入张量和输出张量的底层数据指针
        underlying_t* self_data = self_contig.data_ptr<underlying_t>();
        underlying_t* dst_data =
            reinterpret_cast<underlying_t*>(dst.data_ptr<scalar_t>());

        // 如果输入张量的元素数量大于0，则执行内存复制操作
        if (self.numel() > 0) {
          memcpy(dst_data, self_data, self.nbytes());
        }
      });

  // 返回创建的仿射量化张量
  return dst;
}

} // namespace native
} // namespace at
```