# `.\pytorch\aten\src\ATen\native\AdaptiveMaxPooling2d.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/AdaptivePooling.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/adaptive_max_pool2d_backward_native.h>
#include <ATen/ops/adaptive_max_pool2d_native.h>
#endif

namespace at::meta {

// 定义了 TORCH_META_FUNC 函数，用于自动化元数据处理
TORCH_META_FUNC(adaptive_max_pool2d) (const Tensor& input, IntArrayRef output_size) {
  // 获取输入张量的维度
  int ndim = input.ndimension();
  // 检查张量维度是否为 3 或 4
  TORCH_CHECK(ndim == 3 || ndim == 4,
              "adaptive_max_pool2d(): Expected 3D or 4D tensor, but got: ",
              input.sizes());
  // 对于除了批处理维度以外的每个维度，检查其大小是否大于 0
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(input.size(i) > 0,
        "adaptive_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ", input.sizes(), " with dimension ", i,
        " being empty");
  }

  // 检查输出尺寸是否为 2
  TORCH_CHECK(output_size.size() == 2,
      "adaptive_max_pool2d(): internal error: output_size.size() must be 2");

  // 初始化一些变量
  int dimH = 1;
  int64_t sizeB = 1;
  int64_t sizeD = 0;

  // 如果输入张量是 4 维，调整相关尺寸和维度索引
  if (input.ndimension() == 4) {
    sizeB = input.size(0);
    dimH++;
  }

  sizeD = input.size(dimH - 1);

  int64_t osizeH = output_size[0];
  int64_t osizeW = output_size[1];

  /* resize output */
  // 根据输入张量的维度不同，设置输出张量的尺寸和数据类型
  if (input.ndimension() == 3) {
    set_output_raw_strided(0, {sizeD, osizeH, osizeW}, {}, input.options());
    /* indices will contain i,j locations for each output point */
    set_output_raw_strided(1, {sizeD, osizeH, osizeW}, {}, input.options().dtype(kLong));
  } else {
    set_output_raw_strided(0, {sizeB, sizeD, osizeH, osizeW}, {}, input.options().memory_format(input.suggest_memory_format()));
    /* indices will contain i,j locations for each output point */
    set_output_raw_strided(1, {sizeB, sizeD, osizeH, osizeW}, {}, input.options().memory_format(input.suggest_memory_format()).dtype(kLong));
  }
}

// 定义了 TORCH_META_FUNC 函数，用于自动化元数据处理
TORCH_META_FUNC(adaptive_max_pool2d_backward)
(const Tensor& grad_output, const Tensor& input, const Tensor& indices) {
  // 获取梯度输出张量的维度
  int64_t ndim = grad_output.ndimension();
  // 检查梯度输出张量维度是否为 3 或 4
  TORCH_CHECK(ndim == 3 || ndim == 4,
    "adaptive_max_pooling2d_backward(): Expected 3D or 4D grad_output, but got: ", grad_output.sizes());

  // 检查输入张量和梯度输出张量的数据类型是否一致
  at::native::adaptive_pool_empty_output_check(grad_output, "adaptive_max_pool2d_backward");

  TORCH_CHECK(input.dtype() == grad_output.dtype(),
    "expected dtype ", input.dtype(), " for `grad_output` but got dtype ", grad_output.dtype());

  // 设置输出张量的尺寸和数据类型
  set_output_raw_strided(0, input.sizes(), {}, input.options().memory_format(input.suggest_memory_format()));
}
} // namespace at::meta

namespace at::native {

// 实现了 adaptive_max_pool2d_out_cpu 函数，用于 CPU 下的自适应最大池化操作
TORCH_IMPL_FUNC(adaptive_max_pool2d_out_cpu)
(const Tensor& input, IntArrayRef output_size, const Tensor& output, const Tensor& indices) {
  // 调用底层的自适应最大池化核函数进行计算
  adaptive_max_pool2d_kernel(kCPU, output, indices, input, output_size);
}

// 实现了 adaptive_max_pool2d_backward_out_cpu 函数，用于 CPU 下的自适应最大池化反向传播操作
// 定义一个匿名函数，用于计算自适应最大池化操作的反向传播
(const Tensor& grad_output, const Tensor& input, const Tensor& indices, const Tensor& grad_input) {
  // 将梯度输入张量 grad_input 清零
  grad_input.zero_();
  // 调用自适应最大池化反向传播的核函数，使用 CPU 计算设备，将计算结果写入 grad_input
  adaptive_max_pool2d_backward_kernel(kCPU, grad_input, grad_output, indices);
 }

// 定义自适应最大池化操作的核函数分发器
DEFINE_DISPATCH(adaptive_max_pool2d_kernel);

// 定义自适应最大池化操作反向传播的核函数分发器
DEFINE_DISPATCH(adaptive_max_pool2d_backward_kernel);

// 结束 at::native 命名空间的定义
} // namespace at::native
```