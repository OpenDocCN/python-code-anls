# `.\pytorch\aten\src\ATen\native\cuda\ScanKernels.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>

#include <ATen/native/cuda/ScanKernels.h>
#include <ATen/native/ReduceOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cummax_helper_native.h>
#include <ATen/ops/_cummin_helper_native.h>
#include <ATen/ops/_logcumsumexp_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

// 命名空间 at::native 下的函数实现
namespace at::native {

// 返回一个可能拥有的张量，确保输出张量是连续的
static c10::MaybeOwned<Tensor> contiguous_out_arg(const Tensor &tensor) {
  // 如果输入张量已经连续，则直接借用它
  if (tensor.is_contiguous()) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
  // 否则创建一个具有相同大小和选项的空张量作为输出
  return c10::MaybeOwned<Tensor>::owned(at::empty(tensor.sizes(), tensor.options()));
}

// 在 CUDA 设备上执行累积最大值操作
void cummax_helper_cuda(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  // 定义张量参数对象
  TensorArg output_arg{ values, "output", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_arg{ self, "input", 3 };
  // 检查所有张量在同一个 GPU 上
  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});

  // 确保输出张量是连续的
  auto values_ = contiguous_out_arg(values);
  auto indices_ = contiguous_out_arg(indices);
  // 调用 CUDA 内核执行累积最大值操作
  launch_cummax_cuda_kernel(self, *values_, *indices_, dim);
  // 如果实际输出张量与返回的张量不同，则复制数据
  if (!values.is_same(*values_)) {
    values.copy_(*values_);
  }
  if (!indices.is_same(*indices_)) {
    indices.copy_(*indices_);
  }
}

// 在 CUDA 设备上执行累积最小值操作
void cummin_helper_cuda(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  // 定义张量参数对象
  TensorArg output_arg{ values, "output", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_arg{ self, "input", 3 };
  // 检查所有张量在同一个 GPU 上
  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});

  // 确保输出张量是连续的
  auto values_ = contiguous_out_arg(values);
  auto indices_ = contiguous_out_arg(indices);
  // 调用 CUDA 内核执行累积最小值操作
  launch_cummin_cuda_kernel(self, *values_, *indices_, dim);
  // 如果实际输出张量与返回的张量不同，则复制数据
  if (!values.is_same(*values_)) {
    values.copy_(*values_);
  }
  if (!indices.is_same(*indices_)) {
    indices.copy_(*indices_);
  }
}

// 在 CUDA 设备上执行对数累加指数操作，并将结果输出到给定的结果张量中
Tensor& _logcumsumexp_out_cuda(const Tensor& self, int64_t dim, Tensor& result) {
  // 将维度包装到有效范围内
  const auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  // 调整输出张量的大小以匹配输入张量
  result.resize_(self.sizes());
  // 如果输入张量是标量，则将结果填充为输入张量的值并返回
  if (self.dim() == 0) {
    result.fill_(self);
    return result;
  }
  // 如果输入张量为空，则将结果张量填充为零并返回
  if (self.numel() == 0) {
    result.zero_();
    return result;
  }

  // 定义张量参数对象
  TensorArg output_arg{ result, "output", 1 };
  TensorArg input_arg{ self, "input", 2 };
  // 检查所有张量在同一个 GPU 上
  checkAllSameGPU(__func__, {output_arg, input_arg});

  // 确保输出张量是连续的
  auto result_ = contiguous_out_arg(result);
  // 调用 CUDA 内核执行对数累加指数操作
  launch_logcumsumexp_cuda_kernel(*result_, self, wrap_dim);
  // 如果实际输出张量与返回的张量不同，则复制数据
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  return result;
}

// 在 CUDA 设备上执行对数累加指数操作，并返回结果张量
Tensor _logcumsumexp_cuda(const Tensor& self, int64_t dim) {
  // 创建一个与输入张量相同大小和格式的空张量
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  // 调用 _logcumsumexp_out_cuda 函数计算结果并返回
  return _logcumsumexp_out_cuda(self, dim, result);
}

// 在 CUDA 设备上执行累积和操作的 CUDA 内核
void cumsum_cuda_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  // 如果输入张量是浮点数或复数类型
  if (self.is_floating_point() || self.is_complex()) {
    // 查看 Note [Writing Nondeterministic Operations] 注释
    // （未提供）

    // 这里可能会有一些特定的实现细节或注意事项，但在提供的代码中未给出具体的注释
    // 报告问题：非确定性行为的问题汇报，参考链接：https://github.com/pytorch/pytorch/issues/75240
    globalContext().alertNotDeterministic("cumsum_cuda_kernel");
  }
  // 获取结果的连续内存视图
  auto result_ = contiguous_out_arg(result);
  // 启动 cumsum_cuda_kernel 内核函数来计算累积和，结果存放在 result_ 中
  launch_cumsum_cuda_kernel(*result_, self, dim);
  // 如果 result 和 result_ 不是同一个张量，则将 result_ 的内容复制到 result 中
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
}

void cumprod_cuda_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  // 获取一个连续的输出张量作为结果
  auto result_ = contiguous_out_arg(result);
  // 调用 CUDA 内核函数执行累积乘积操作
  launch_cumprod_cuda_kernel(*result_, self, dim);
  // 如果输出结果不是原始结果，将结果复制回原始结果张量
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
}

// 注册 CUDA 分发函数，将 cumsum_stub 映射到 cumsum_cuda_kernel 函数
REGISTER_CUDA_DISPATCH(cumsum_stub, &cumsum_cuda_kernel);
// 注册 CUDA 分发函数，将 cumprod_stub 映射到 cumprod_cuda_kernel 函数
REGISTER_CUDA_DISPATCH(cumprod_stub, &cumprod_cuda_kernel);

} // namespace at::native
```