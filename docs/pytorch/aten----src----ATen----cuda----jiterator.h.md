# `.\pytorch\aten\src\ATen\cuda\jiterator.h`

```py
#pragma once
#include <ATen/jit_macros.h>

// 如果使用了 Jiterator 宏
#if AT_USE_JITERATOR()

#include <c10/macros/Export.h>
#include <c10/util/SmallVector.h>
#include <ATen/core/Tensor.h>

#include <string>
#include <vector>

// 定义了 at::cuda 命名空间
namespace at::cuda {

// 声明了 TORCH_CUDA_CPP_API，返回类型为 SmallVector<at::Tensor>
TORCH_CUDA_CPP_API c10::SmallVector<at::Tensor> CompileAndLaunchKernel(
  const std::string& code_string,       // 编译和启动内核的代码字符串
  const std::string& kernel_name,       // 内核函数的名称
  const int num_outputs,                // 输出张量的数量
  const c10::SmallVector<at::Tensor>& tensors,   // 输入张量列表
  const c10::SmallVector<at::Scalar>& extra_args,  // 额外的标量参数列表
  bool return_by_ref);                  // 是否通过引用返回结果

} // namespace at::cuda

// 如果未使用 Jiterator 宏
#else

namespace at::cuda {

// 定义了 TORCH_CUDA_CPP_API，返回类型为 SmallVector<at::Tensor>
TORCH_CUDA_CPP_API c10::SmallVector<at::Tensor> CompileAndLaunchKernel(
  const std::string& code_string,       // 编译和启动内核的代码字符串
  const std::string& kernel_name,       // 内核函数的名称
  const int num_outputs,                // 输出张量的数量
  const c10::SmallVector<at::Tensor>& tensors,   // 输入张量列表
  const c10::SmallVector<at::Scalar>& extra_args,  // 额外的标量参数列表
  bool return_by_ref) {                 // 是否通过引用返回结果
    TORCH_CHECK(false, "Jiterator is not supported");  // 抛出错误，表示不支持 Jiterator
  }
} // namespace at::cuda

#endif // AT_USE_JITERATOR()
```