# `.\pytorch\tools\autograd\templates\Functions.cpp`

```py
// 包含自动生成的注释
#include "torch/csrc/autograd/FunctionsManual.h"
#include "torch/csrc/dynamo/compiled_autograd.h"

// ${generated_comment}

// 之前在这里的手动函数定义现在在torch/csrc/autograd/FunctionsManual.cpp中
// 这样可以加快重新编译速度，并且可以共享这些实现，以便用于正向模式自动微分公式

using namespace torch::autograd::generated::details;
using at::Tensor;
using at::Scalar;
using at::IntArrayRef;
using at::TensorList;

namespace torch::autograd::generated {

${autograd_function_definitions}

} // namespace torch::autograd::generated
```