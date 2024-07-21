# `.\pytorch\aten\src\ATen\native\AdaptivePooling.h`

```py
#pragma once
// 防止头文件被多次引用

#include <ATen/core/Tensor.h>
// 包含 ATen 核心张量类的头文件

#include <ATen/native/DispatchStub.h>
// 包含 ATen 原生调度存根的头文件

#include <c10/util/ArrayRef.h>
// 包含 c10 数组引用工具的头文件

#include <c10/util/irange.h>
// 包含 c10 整数范围工具的头文件

#include <cmath>
// 包含数学函数的头文件

namespace at::native {

using adaptive_avg_pooling2d_fn = void(*)(Tensor& output, const Tensor& input, IntArrayRef output_size);
// 定义一个函数指针类型 adaptive_avg_pooling2d_fn，用于自适应平均池化操作

using adaptive_avg_pooling2d_backward_fn = void(*)(Tensor& grad_input, const Tensor& grad_output);
// 定义一个函数指针类型 adaptive_avg_pooling2d_backward_fn，用于自适应平均池化的反向传播操作

DECLARE_DISPATCH(adaptive_avg_pooling2d_fn, adaptive_avg_pool2d_kernel);
// 声明自适应平均池化的调度函数 adaptive_avg_pool2d_kernel

DECLARE_DISPATCH(adaptive_avg_pooling2d_backward_fn, adaptive_avg_pool2d_backward_kernel);
// 声明自适应平均池化反向传播的调度函数 adaptive_avg_pool2d_backward_kernel

using adaptive_max_pooling2d_fn = void(*)(const Tensor& output, const Tensor& indices, const Tensor& input, IntArrayRef output_size);
// 定义一个函数指针类型 adaptive_max_pooling2d_fn，用于自适应最大池化操作

using adaptive_max_pooling2d_backward_fn = void(*)(const Tensor& grad_input, const Tensor& grad_output, const Tensor& indices);
// 定义一个函数指针类型 adaptive_max_pooling2d_backward_fn，用于自适应最大池化的反向传播操作

DECLARE_DISPATCH(adaptive_max_pooling2d_fn, adaptive_max_pool2d_kernel);
// 声明自适应最大池化的调度函数 adaptive_max_pool2d_kernel

DECLARE_DISPATCH(adaptive_max_pooling2d_backward_fn, adaptive_max_pool2d_backward_kernel);
// 声明自适应最大池化反向传播的调度函数 adaptive_max_pool2d_backward_kernel

using adaptive_avg_pooling3d_fn = void(*)(Tensor& output, const Tensor& input, IntArrayRef output_size);
// 定义一个函数指针类型 adaptive_avg_pooling3d_fn，用于自适应三维平均池化操作

using adaptive_avg_pooling3d_backward_fn = void(*)(Tensor& grad_input, const Tensor& grad_output);
// 定义一个函数指针类型 adaptive_avg_pooling3d_backward_fn，用于自适应三维平均池化的反向传播操作

DECLARE_DISPATCH(adaptive_avg_pooling3d_fn, adaptive_avg_pool3d_kernel);
// 声明自适应三维平均池化的调度函数 adaptive_avg_pool3d_kernel

DECLARE_DISPATCH(adaptive_avg_pooling3d_backward_fn, adaptive_avg_pool3d_backward_kernel);
// 声明自适应三维平均池化反向传播的调度函数 adaptive_avg_pool3d_backward_kernel

using adaptive_max_pooling3d_fn = void(*)(const Tensor& output, const Tensor& indices, const Tensor& input, IntArrayRef output_size);
// 定义一个函数指针类型 adaptive_max_pooling3d_fn，用于自适应三维最大池化操作

using adaptive_max_pooling3d_backward_fn = void(*)(const Tensor& grad_input, const Tensor& grad_output, const Tensor& indices);
// 定义一个函数指针类型 adaptive_max_pooling3d_backward_fn，用于自适应三维最大池化的反向传播操作

DECLARE_DISPATCH(adaptive_max_pooling3d_fn, adaptive_max_pool3d_kernel);
// 声明自适应三维最大池化的调度函数 adaptive_max_pool3d_kernel

DECLARE_DISPATCH(adaptive_max_pooling3d_backward_fn, adaptive_max_pool3d_backward_kernel);
// 声明自适应三维最大池化反向传播的调度函数 adaptive_max_pool3d_backward_kernel

inline int64_t start_index(int64_t a, int64_t b, int64_t c) {
  return (a / b) * c + ((a % b) * c) / b;
}
// 定义一个内联函数 start_index，计算起始索引

inline int64_t end_index(int64_t a, int64_t b, int64_t c) {
  return 1 + ((a + 1) * c - 1) / b;
}
// 定义一个内联函数 end_index，计算结束索引

inline void adaptive_pool_empty_output_check(const Tensor& gradOutput_, const char* arg_name) {
  int64_t ndim = gradOutput_.ndimension();
  // 获取梯度输出张量的维度数

  for (const auto i : c10::irange(1, ndim)) {
    // 对于除批次维度之外的每一个维度

    TORCH_CHECK(gradOutput_.size(i) > 0,
      arg_name, "(): Expected grad_output to have non-zero size for non-batch dimensions, "
      "but grad_output has sizes ", gradOutput_.sizes(), " with dimension ", i,
      " being empty");
    // 检查梯度输出张量在非批次维度上的尺寸是否大于零，否则抛出异常
  }
}

} // namespace at::native
// 结束 at::native 命名空间
```