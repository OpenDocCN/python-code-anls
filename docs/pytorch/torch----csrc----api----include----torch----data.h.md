# `.\pytorch\torch\csrc\api\include\torch\data.h`

```py
#pragma once
// 使用预处理器指令 `#pragma once`，确保头文件只被编译一次，避免重复包含

#include <torch/data/dataloader.h>
// 包含 Torch 数据加载器的头文件

#include <torch/data/datasets.h>
// 包含 Torch 数据集的头文件

#include <torch/data/samplers.h>
// 包含 Torch 数据采样器的头文件

#include <torch/data/transforms.h>
// 包含 Torch 数据转换器的头文件

// Some "exports".
// 以下是一些导出定义，将命名空间datasets中的特定类型引入到torch::data命名空间中，以便外部使用

namespace torch {
namespace data {
using datasets::BatchDataset;
// 使用命名空间datasets中的BatchDataset类型，并引入到torch::data命名空间中

using datasets::Dataset;
// 使用命名空间datasets中的Dataset类型，并引入到torch::data命名空间中
} // namespace data
} // namespace torch
```