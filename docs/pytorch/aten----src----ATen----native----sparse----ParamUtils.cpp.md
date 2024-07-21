# `.\pytorch\aten\src\ATen\native\sparse\ParamUtils.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/ParamUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <tuple>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

// 对稀疏输入进行 softmax 前处理，返回处理后的输入、输出和维度
std::tuple<Tensor, Tensor, int64_t> softmax_sparse_input_preprocessing(
    const Tensor& input_,          // 输入稀疏张量
    const int64_t dim_,            // 操作维度
    const bool half_to_float,      // 是否进行半精度到单精度的转换
    CheckedFrom function_name) {   // 调用函数名称
  TORCH_INTERNAL_ASSERT(input_.is_sparse());  // 断言输入为稀疏张量
  TORCH_CHECK(
      !half_to_float,                     // 检查是否进行半精度到单精度转换
      std::string(function_name) +        // 错误消息，显示调用函数名称
          ": with half to float conversion is not supported on " +
          input_.device().str());         // 显示不支持的设备信息
  auto input = input_.coalesce();         // 合并输入的稀疏张量
  Tensor output = at::native::empty_like_sparse_coo(input);  // 创建与输入相同形状的空稀疏张量
  int64_t dim = c10::maybe_wrap_dim(dim_, input.dim());      // 处理操作维度的索引
  return std::make_tuple(input, output, dim);                // 返回处理后的输入、输出和维度
}

// 对 softmax 反向传播的稀疏输入进行预处理，返回处理后的梯度、输出、输入和维度
std::tuple<Tensor, Tensor, Tensor, int64_t> softmax_backward_sparse_input_preprocessing(
    const Tensor& grad_,           // 梯度张量
    const Tensor& output_,         // 输出张量
    int64_t dim_,                  // 操作维度
    const Tensor& input_,          // 输入张量
    CheckedFrom function_name) {   // 调用函数名称
  TensorArg grad_arg{grad_, "grad", 1}, output_arg{output_, "output", 2};
  checkSameSize(function_name, grad_arg, output_arg);  // 检查梯度和输出的大小是否一致

  int64_t dim = maybe_wrap_dim(dim_, grad_.dim());     // 处理操作维度的索引

  auto grad = grad_.coalesce();       // 合并梯度张量
  auto output = output_.coalesce();   // 合并输出张量

  Tensor grad_input = at::native::empty_like_sparse_coo(output);  // 创建与输出相同形状的空稀疏张量
  TORCH_CHECK(
      grad.sparse_dim() == output.sparse_dim(),  // 断言梯度和输出的稀疏维度必须相同
      ": grad and output sparse dimensions must be equal");  // 错误消息，显示稀疏维度不匹配
  return std::make_tuple(grad_input, grad, output, dim);  // 返回处理后的梯度、输出、输入和维度
}

} // namespace at::native
```