# `.\pytorch\tools\autograd\templates\ViewFuncs.cpp`

```py
#include <torch/csrc/autograd/generated/ViewFuncs.h>
// 包含 Torch 的自动生成视图函数的头文件 ViewFuncs.h

// ${generated_comment}
// 插入由代码生成器生成的注释（这里作为占位符）

using at::Tensor;
// 使用 Torch 的 Tensor 类
using at::Scalar;
// 使用 Torch 的 Scalar 类
using at::IntArrayRef;
// 使用 Torch 的 IntArrayRef 类
using at::TensorList;
// 使用 Torch 的 TensorList 类

namespace torch::autograd::generated {
// 进入 torch::autograd::generated 命名空间

${view_func_definitions}
// 插入由代码生成器生成的视图函数定义

} // namespace torch::autograd::generated
// 退出 torch::autograd::generated 命名空间
```