# `.\pytorch\aten\src\ATen\native\cuda\ScanKernels.h`

```
#pragma once
// 预处理指令：确保头文件只被编译一次

#include <cstdint>
// 包含标准整数类型的头文件

namespace at {
class TensorBase;
// 声明名为TensorBase的类，位于at命名空间中

namespace native {
// 声明名为native的命名空间，位于at命名空间中

// 注意：以下函数要求输出张量是连续的
void launch_cummax_cuda_kernel(const TensorBase& self, const TensorBase& values,
                               const TensorBase& indices, int64_t dim);
// 启动cummax的CUDA核心函数，对self张量进行操作，生成values和indices张量，dim为维度

void launch_cummin_cuda_kernel(const TensorBase& self, const TensorBase& values,
                               const TensorBase& indices, int64_t dim);
// 启动cummin的CUDA核心函数，对self张量进行操作，生成values和indices张量，dim为维度

void launch_logcumsumexp_cuda_kernel(const TensorBase& result, const TensorBase& self, int64_t dim);
// 启动logcumsumexp的CUDA核心函数，对self张量进行操作，生成result张量，dim为维度

void launch_cumsum_cuda_kernel(const TensorBase& result, const TensorBase& self, int64_t dim);
// 启动cumsum的CUDA核心函数，对self张量进行操作，生成result张量，dim为维度

void launch_cumprod_cuda_kernel(const TensorBase& result, const TensorBase& self, int64_t dim);
// 启动cumprod的CUDA核心函数，对self张量进行操作，生成result张量，dim为维度

}}  // namespace at::native
// 结束命名空间at::native
```