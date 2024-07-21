# `.\pytorch\torch\script.h`

```py
#pragma once
// 使用预处理指令#pragma once，确保头文件只被编译一次

#include <torch/csrc/api/include/torch/types.h>
// 包含 Torch 的类型定义头文件

#include <torch/csrc/autograd/InferenceMode.h>
// 包含 Torch 的推断模式相关头文件

#include <torch/csrc/autograd/custom_function.h>
// 包含 Torch 自定义函数的头文件

#include <torch/csrc/autograd/generated/variable_factories.h>
// 包含 Torch 自动生成变量工厂的头文件

#include <torch/csrc/autograd/grad_mode.h>
// 包含 Torch 梯度模式相关头文件

#include <torch/csrc/jit/runtime/custom_operator.h>
// 包含 Torch 自定义运算符运行时头文件

#include <torch/csrc/jit/serialization/import.h>
// 包含 Torch 模型导入相关头文件

#include <torch/csrc/jit/serialization/pickle.h>
// 包含 Torch 模型序列化相关头文件

#include <torch/custom_class.h>
// 包含 Torch 自定义类相关头文件

#include <ATen/ATen.h>
// 包含 ATen 库的核心头文件
```