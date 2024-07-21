# `.\pytorch\torch\csrc\autograd\functions\comm.h`

```py
#pragma once
// 防止头文件被多次包含

#include <torch/csrc/Export.h>
// 导出 Torch 库的符号定义

#include <torch/csrc/autograd/function.h>
// 引入自动微分的函数定义

#include <torch/csrc/autograd/variable.h>
// 引入自动微分的变量定义

#include <ATen/ATen.h>
// 引入 ATen 库的主头文件

#include <c10/cuda/CUDAStream.h>
// 引入 CUDA 流的定义

#include <c10/util/Optional.h>
// 引入 C10 库的可选类型定义

#include <cstddef>
// 引入标准库的大小类型定义

#include <vector>
// 引入标准库的向量容器

namespace torch {
namespace autograd {

struct TORCH_CUDA_CU_API Scatter : public Node {
  // 定义 Scatter 结构体，继承自 Node 类
  explicit Scatter(
      std::vector<at::Device> devices,  // 分散操作的设备列表
      std::optional<std::vector<int64_t>> chunk_sizes = c10::nullopt,  // 可选的分块大小列表
      int64_t dim = 0,  // 操作的维度，默认为0
      std::optional<std::vector<std::optional<at::cuda::CUDAStream>>> streams =
          c10::nullopt,  // 可选的 CUDA 流列表
      bool unsqueeze_scalars = false);  // 是否对标量进行 unsqueeze 操作的标志
  ~Scatter() override;  // 析构函数，用于清理资源

  variable_list apply(variable_list&& inputs) override;
  // 应用 Scatter 操作到输入变量列表，并返回输出变量列表

  std::vector<at::Device> devices_;  // 分散操作的设备列表
  std::optional<std::vector<int64_t>> chunk_sizes_;  // 可选的分块大小列表
  int64_t dim_;  // 操作的维度
  std::optional<std::vector<std::optional<at::cuda::CUDAStream>>> streams_;  // 可选的 CUDA 流列表
  bool unsqueeze_scalars_;  // 是否对标量进行 unsqueeze 操作的标志
};

struct TORCH_CUDA_CU_API Gather : public Node {
  // 定义 Gather 结构体，继承自 Node 类
  explicit Gather(const at::Device& destination_device, int64_t dim = 0);
  // 构造函数，指定聚集操作的目标设备和维度，默认为0
  ~Gather() override;  // 析构函数，用于清理资源

  variable_list apply(variable_list&& inputs) override;
  // 应用 Gather 操作到输入变量列表，并返回输出变量列表

  at::Device destination_device_;  // 聚集操作的目标设备
  int64_t dim_;  // 操作的维度
};

} // namespace autograd
} // namespace torch
```