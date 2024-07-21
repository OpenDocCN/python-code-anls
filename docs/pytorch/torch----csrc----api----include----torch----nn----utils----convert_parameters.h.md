# `.\pytorch\torch\csrc\api\include\torch\nn\utils\convert_parameters.h`

```py
// 只包含一次的预处理指令，确保本文件只被编译一次
#pragma once

// 引入 Torch 的导出头文件和类型定义
#include <torch/csrc/Export.h>
#include <torch/types.h>

// 定义 Torch 的命名空间
namespace torch {
// 定义神经网络模块的命名空间
namespace nn {
// 定义神经网络工具的命名空间
namespace utils {

// 检查参数是否位于同一设备的辅助函数
// 目前不支持模型参数在不同 GPU 或混合 CPU/GPU 的情况下转换为单一向量形式
inline std::optional<int64_t> _check_param_device(
    const torch::Tensor& param, // 待检查的参数张量
    std::optional<int64_t> old_param_device) { // 上一个参数的设备 ID（可选）
  
  // 如果是第一个参数
  if (old_param_device == c10::nullopt) {
    // 确定参数所在的设备
    old_param_device = param.is_cuda() ? param.get_device() : -1;
  } else {
    bool warn = false;
    // 检查当前参数是否在同一 GPU 上
    if (param.is_cuda()) {
      warn = (param.get_device() != old_param_device.value());
    } else { // 检查当前参数是否在 CPU 上
      warn = (old_param_device.value() != -1);
    }
    // 如果发现参数在不同设备上，则抛出错误
    if (warn) {
      TORCH_CHECK(
          false,
          "Found two parameters on different devices, ",
          "this is currently not supported.");
    }
  }

  return old_param_device; // 返回参数设备 ID
}

// 将参数列表转换为一个向量
inline torch::Tensor parameters_to_vector(
    const std::vector<torch::Tensor>& parameters) { // 参数张量的向量
  
  std::optional<int64_t> param_device; // 参数的设备 ID

  std::vector<torch::Tensor> vec;
  vec.reserve(parameters.size());

  // 遍历参数列表
  for (const torch::Tensor& param : parameters) {
    // 确保参数位于同一设备
    param_device = _check_param_device(param, param_device);

    // 将参数视图展平并加入向量中
    vec.push_back(param.view(-1));
  }

  return torch::cat(vec); // 将向量连接成一个张量并返回
}

// 将一个向量转换回参数列表
inline void vector_to_parameters(
    const torch::Tensor& vec, // 输入的向量
    const std::vector<torch::Tensor>& parameters) { // 目标参数张量的向量

  std::optional<int64_t> param_device; // 参数的设备 ID

  int64_t pointer = 0; // 向量切片的起始点
  // 遍历目标参数列表
  for (const torch::Tensor& param : parameters) {
    // 确保参数位于同一设备
    param_device = _check_param_device(param, param_device);

    // 参数张量的元素数
    auto num_param = param.numel();
    // 切片向量，重新形状化并替换参数的旧数据
    param.set_data(
        vec.slice(0, pointer, pointer + num_param).view_as(param).data());

    // 更新切片的起始点
    pointer += num_param;
  }
}

} // namespace utils
} // namespace nn
} // namespace torch
```