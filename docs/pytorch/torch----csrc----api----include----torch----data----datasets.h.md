# `.\pytorch\torch\csrc\api\include\torch\data\datasets.h`

```
#pragma once
// 使用#pragma once指令，确保头文件只被编译一次，防止多重包含

#include <torch/data/datasets/base.h>
// 包含torch库中的基础数据集模块头文件

#include <torch/data/datasets/chunk.h>
// 包含torch库中的数据集分块处理模块头文件

#include <torch/data/datasets/map.h>
// 包含torch库中的数据集映射处理模块头文件

#include <torch/data/datasets/mnist.h>
// 包含torch库中的MNIST数据集处理模块头文件

#include <torch/data/datasets/shared.h>
// 包含torch库中的数据集共享模块头文件

#include <torch/data/datasets/stateful.h>
// 包含torch库中的有状态数据集处理模块头文件

#include <torch/data/datasets/tensor.h>
// 包含torch库中的张量数据集模块头文件
```