# `.\pytorch\torch\csrc\cuda\Tensor.cpp`

```
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif



// 如果 __STDC_FORMAT_MACROS 宏未定义，则定义它，以启用特定的格式宏，例如 PRIu64 等
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif



// 这里的包含顺序很重要，因此禁用 clang-format 以保持当前的包含顺序
// clang-format off



// 禁用 clang-format 后，包含以下头文件：torch/csrc/python_headers.h 和 structmember.h
#include <torch/csrc/python_headers.h>
#include <structmember.h>



// 接着包含 torch/csrc/cuda/THCP.h 头文件
#include <torch/csrc/cuda/THCP.h>



// 再包含 torch/csrc/utils/tensor_numpy.h 和 torch/csrc/copy_utils.h 头文件
#include <torch/csrc/utils/tensor_numpy.h>
#include <torch/csrc/copy_utils.h>



// 最后包含 torch/csrc/DynamicTypes.h 头文件
#include <torch/csrc/DynamicTypes.h>



// clang-format on
// 恢复 clang-format 的自动格式化功能
```