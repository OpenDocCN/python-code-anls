# `.\pytorch\aten\src\ATen\native\cuda\ReduceOps.cpp`

```
// 定义宏，仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 CUDA 下的归约操作头文件
#include <ATen/native/cuda/ReduceOps.h>

// 包含常规的归约操作头文件
#include <ATen/native/ReduceOps.h>
// 包含归约所有操作的头文件
#include <ATen/native/ReduceAllOps.h>
// 包含归约操作的实用工具函数头文件
#include <ATen/native/ReduceOpsUtils.h>
// 包含张量比较操作的头文件
#include <ATen/native/TensorCompare.h>

// 包含 ATen 库的上下文头文件
#include <ATen/Context.h>
// 包含张量操作的实用函数头文件
#include <ATen/TensorUtils.h>
// 包含张量维度包装工具函数头文件
#include <ATen/WrapDimUtils.h>
// 包含命名张量的头文件
#include <ATen/core/NamedTensor.h>
// 包含张量迭代器头文件
#include <ATen/TensorIterator.h>

// 根据是否定义了 AT_PER_OPERATOR_HEADERS 决定包含不同头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/full.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/kthvalue_native.h>
#include <ATen/ops/median_native.h>
#include <ATen/ops/nanmedian_native.h>
#include <ATen/ops/where.h>
#endif

// ATen 库的 native 命名空间
namespace at::native {
// 匿名命名空间，用于定义私有函数

// CUDA 下的 norm_kernel 函数，处理张量迭代器和标量 val
void norm_kernel_cuda(TensorIterator& iter, const Scalar& val) {
  double p;
  // 根据标量的类型，转换为相应的 double 类型 p
  if (val.isIntegral(false)) {
    p = val.to<int64_t>();
  } else if (val.isFloatingPoint()) {
    p = val.to<double>();
  } else {
    // 如果标量类型不是整数或浮点数，抛出错误信息
    TORCH_CHECK(false, "norm_kernel_cuda_impl expects norm to be integer or float");
  }
  // 如果迭代器中的元素个数为 0，将输出张量填充为无穷大或零
  if (iter.numel() == 0) {
    iter.output().fill_((p < 0) ? INFINITY : 0);
    return;
  }

  // 调用 CUDA 下的 norm_launch_kernel 处理迭代器和 p
  norm_launch_kernel(iter, p);

  // 如果输出张量是复数类型，将其虚部置零
  if (isComplexType(iter.output().scalar_type())) {
    at::imag(iter.output()).zero_();
  }
}

// 最小值归约操作的 CUDA 实现，使用元数据构造归约迭代器并调用最小值 CUDA 核函数
void min_kernel_impl(const Tensor& result, const Tensor& indice, const Tensor& self, int64_t dim, bool keepdim) {
  auto iter = meta::make_reduction(self, result, indice, dim, keepdim, self.scalar_type(), kLong);
  min_launch_kernel(iter);
}

// 最大值归约操作的 CUDA 实现，使用元数据构造归约迭代器并调用最大值 CUDA 核函数
void max_kernel_impl(const Tensor& result, const Tensor& indice, const Tensor& self, int64_t dim, bool keepdim) {
  auto iter = meta::make_reduction(self, result, indice, dim, keepdim, self.scalar_type(), kLong);
  max_launch_kernel(iter);
}

// aminmax 归约操作的 CUDA 实现，使用元数据构造归约迭代器并调用 aminmax CUDA 核函数
void aminmax_kernel_impl(
    const Tensor& self, int64_t dim, bool keepdim, Tensor& min_result, Tensor& max_result) {
  at::TensorIterator iter = make_reduction("aminmax_cuda", min_result,
                                           max_result, self, dim, keepdim, self.scalar_type());
  // 如果迭代器中有元素，则调用 aminmax_launch_kernel 处理
  if (iter.numel() != 0) {
    aminmax_launch_kernel(iter);
  }
}

// 全局最小值归约操作的 CUDA 实现，构造全局最小值归约迭代器并调用全局最小值 CUDA 核函数
void min_all_kernel_impl(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = make_reduction("min_all", result, input, IntArrayRef{}, false, dtype);
  min_all_launch_kernel(iter);
}

// 全局最大值归约操作的 CUDA 实现，构造全局最大值归约迭代器并调用全局最大值 CUDA 核函数
void max_all_kernel_impl(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = make_reduction("max_all", result, input, IntArrayRef{}, false, dtype);
  max_all_launch_kernel(iter);
}

// 全局 aminmax 归约操作的 CUDA 实现，构造全局 aminmax 归约迭代器并调用全局 aminmax CUDA 核函数
void aminmax_allreduce_kernel_impl(const Tensor& input, Tensor& min_result, Tensor& max_result) {
  auto dtype = input.scalar_type();
  auto iter = make_reduction("aminmax_cuda", min_result, max_result, input,
                             IntArrayRef{}, false, dtype);
  // 检查迭代器中的元素数量大于 0，否则抛出错误信息
  TORCH_CHECK(iter.numel() > 0, "min_max on a tensor with no elements is not defined.");
  aminmax_allreduce_launch_kernel(iter);
}

}  // namespace (anonymous)
REGISTER_CUDA_DISPATCH(min_stub, &min_kernel_impl);
// 注册 CUDA 分发器，将 min_stub 映射到对应的 CUDA 实现函数 min_kernel_impl

REGISTER_CUDA_DISPATCH(max_stub, &max_kernel_impl);
// 注册 CUDA 分发器，将 max_stub 映射到对应的 CUDA 实现函数 max_kernel_impl

REGISTER_CUDA_DISPATCH(min_all_stub, &min_all_kernel_impl);
// 注册 CUDA 分发器，将 min_all_stub 映射到对应的 CUDA 实现函数 min_all_kernel_impl

REGISTER_CUDA_DISPATCH(max_all_stub, &max_all_kernel_impl);
// 注册 CUDA 分发器，将 max_all_stub 映射到对应的 CUDA 实现函数 max_all_kernel_impl

REGISTER_CUDA_DISPATCH(aminmax_allreduce_stub, &aminmax_allreduce_kernel_impl);
// 注册 CUDA 分发器，将 aminmax_allreduce_stub 映射到对应的 CUDA 实现函数 aminmax_allreduce_kernel_impl

REGISTER_CUDA_DISPATCH(aminmax_stub, &aminmax_kernel_impl);
// 注册 CUDA 分发器，将 aminmax_stub 映射到对应的 CUDA 实现函数 aminmax_kernel_impl

REGISTER_CUDA_DISPATCH(norm_stub, &norm_kernel_cuda);
// 注册 CUDA 分发器，将 norm_stub 映射到对应的 CUDA 实现函数 norm_kernel_cuda

} // namespace at::native
// 结束命名空间 at::native
```