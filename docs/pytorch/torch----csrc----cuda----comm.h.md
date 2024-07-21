# `.\pytorch\torch\csrc\cuda\comm.h`

```py
#pragma once
// 包含 ATen 库的头文件
#include <ATen/ATen.h>
// 包含 ATen CUDA 相关的通用头文件
#include <ATen/cuda/ATenCUDAGeneral.h>
// 包含 CUDA 上下文管理的头文件
#include <ATen/cuda/CUDAContext.h>
// 包含 C10 库中的 Optional 类型支持
#include <c10/util/Optional.h>
// 包含 Torch 导出相关的头文件
#include <torch/csrc/Export.h>

// 包含标准库中的头文件
#include <cstddef>
#include <vector>

// 定义命名空间 torch::cuda
namespace torch::cuda {

// 定义一个二维张量列表类型
using tensor_list2d = std::vector<std::vector<at::Tensor>>;

// 声明一个函数，用于在 CUDA 设备上广播张量
TORCH_CUDA_CU_API std::vector<at::Tensor>& broadcast_out(
    const at::Tensor& tensor,
    std::vector<at::Tensor>& out_tensors);

// 声明一个函数，用于在多个 CUDA 设备上广播张量
TORCH_CUDA_CU_API std::vector<at::Tensor> broadcast(
    const at::Tensor& tensor,
    at::IntArrayRef devices);

// 声明一个函数，用于在多个 CUDA 设备上协同广播张量
TORCH_CUDA_CU_API tensor_list2d broadcast_coalesced(
    at::TensorList tensors,
    at::IntArrayRef devices,
    size_t buffer_size);

// 声明一个函数，用于在 CUDA 设备上分散张量
TORCH_CUDA_CU_API std::vector<at::Tensor>& scatter_out(
    const at::Tensor& tensor,
    std::vector<at::Tensor>& out_tensors,
    int64_t dim = 0,
    const std::optional<std::vector<std::optional<at::cuda::CUDAStream>>>&
        streams = c10::nullopt);

// 声明一个函数，用于在多个 CUDA 设备上分散张量
TORCH_CUDA_CU_API std::vector<at::Tensor> scatter(
    const at::Tensor& tensor,
    at::IntArrayRef devices,
    const std::optional<std::vector<int64_t>>& chunk_sizes = c10::nullopt,
    int64_t dim = 0,
    const std::optional<std::vector<std::optional<at::cuda::CUDAStream>>>&
        streams = c10::nullopt);

// 声明一个函数，用于在 CUDA 设备上聚集张量结果
TORCH_CUDA_CU_API at::Tensor& gather_out(
    at::TensorList tensors,
    at::Tensor& out_tensor,
    int64_t dim);

// 声明一个函数，用于在多个 CUDA 设备上聚集张量
TORCH_CUDA_CU_API at::Tensor gather(
    at::TensorList tensors,
    int64_t dim,
    std::optional<int32_t> destination_index);

} // namespace torch::cuda
```