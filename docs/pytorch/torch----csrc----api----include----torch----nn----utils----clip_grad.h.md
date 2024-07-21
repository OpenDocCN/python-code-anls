# `.\pytorch\torch\csrc\api\include\torch\nn\utils\clip_grad.h`

```
#pragma once
// 使用 pragma once 指令确保头文件只被编译一次

#include <torch/csrc/Export.h>
// 引入 PyTorch 的导出头文件

#include <utility>
// 引入 C++ 标准库中的 utility 头文件，用于使用 std::vector 和 std::optional

namespace torch {
namespace nn {
namespace utils {

// 定义命名空间 torch::nn::utils，用于包含本模块的所有函数和类

// Clips gradient norm of a vector of Tensors.
// 对一组 Tensor 的梯度范数进行裁剪。

// See
// https://pytorch.org/docs/stable/nn.html?highlight=clip_grad_norm#torch.nn.utils.clip_grad_norm_
// for more details about this module.
// 参见 PyTorch 文档了解有关此模块的更多详细信息。

// Difference with the python version: unlike the python version, even when
// skipping the finiteness checks (error_if_nonfinite = false), this function
// will introduce a device <=> CPU synchronization (for devices where that makes
// sense!) in order to return a CPU-side `double`. This C++ version therefore
// cannot be run fully asynchronously w.r.t. the device of the gradients.
// 与 Python 版本的不同之处：即使在跳过有限性检查时（error_if_nonfinite = false），此函数
// 也会引入设备 <=> CPU 的同步（对于需要的设备！）以返回 CPU 端的 `double`。因此这个
// C++ 版本无法完全异步运行相对于梯度设备。

inline double clip_grad_norm_(
    const std::vector<Tensor>& parameters,
    double max_norm,
    double norm_type = 2.0,
    bool error_if_nonfinite = false) {
  // 函数定义，裁剪一组 Tensor 的梯度范数

  std::vector<Tensor> params_with_grad;
  // 创建一个存储具有梯度的 Tensor 的向量

  for (const auto& param : parameters) {
    auto& grad = param.grad();
    // 获取当前参数的梯度引用
    if (grad.defined()) {
      params_with_grad.push_back(param);
      // 如果梯度已定义，则将该参数加入 params_with_grad 中
    }
  }

  if (params_with_grad.empty()) {
    return 0.0;
    // 如果没有参数具有梯度，则直接返回 0.0
  }

  Tensor total_norm_tensor;
  // 定义用于存储总范数的 Tensor

  if (norm_type == std::numeric_limits<double>::infinity()) {
    // 如果范数类型是无穷大

    std::vector<Tensor> norms;
    norms.reserve(params_with_grad.size());

    for (const auto& param : params_with_grad) {
      norms.emplace_back(param.grad().data().abs().max());
      // 计算每个参数的梯度数据的绝对值的最大值，形成 norms 向量
    }
    total_norm_tensor =
        (norms.size() == 1) ? norms[0] : torch::max(torch::stack(norms));
    // 计算 norms 向量的最大值，得到总范数的 Tensor
  } else if (norm_type == 0) {
    // 如果范数类型是 0

    total_norm_tensor =
        torch::full({}, static_cast<double>(params_with_grad.size()));
    // 创建一个大小为 params_with_grad.size() 的全 0 Tensor
  } else {
    // 对于其他的范数类型

    std::vector<Tensor> norms;
    norms.reserve(params_with_grad.size());

    for (const auto& param : params_with_grad) {
      norms.emplace_back(param.grad().data().norm(norm_type));
      // 计算每个参数的梯度数据的指定范数，形成 norms 向量
    }
    total_norm_tensor =
        (norms.size() == 1) ? norms[0] : torch::stack(norms).norm(norm_type);
    // 计算 norms 向量的指定范数，得到总范数的 Tensor
  }

  // When possible (ie when skipping the finiteness check), we avoid
  // synchronizing the CPU and the gradients' device until the very end to
  // preserve async execution on the device. When checking for finite-ness, this
  // optional ensures we only sync once.
  // 在可能的情况下（即在跳过有限性检查时），我们避免在最后之前同步 CPU 和梯度设备，
  // 以保持设备上的异步执行。当检查有限性时，此选项确保我们仅同步一次。

  std::optional<double> total_norm = c10::nullopt;
  // 使用 std::optional 存储总范数的值，默认为空

  if (error_if_nonfinite) {
    total_norm = total_norm_tensor.item().toDouble();
    // 将总范数的 Tensor 转换为 double，并存储在 total_norm 中
    TORCH_CHECK(
        std::isfinite(*total_norm),
        "The total norm of order ",
        norm_type,
        " for gradients from `parameters` ",
        "is non-finite, so it cannot be clipped. To disable this error and scale ",
        "the gradients with the non-finite norm anyway, set ",
        "`error_if_nonfinite=false`");
    // 如果 total_norm 不是有限数，则抛出异常
  }

  auto clip_coef = max_norm / (total_norm_tensor + 1e-6);
  // 计算裁剪系数，避免除以 0 的情况，增加一个小的偏置 1e-6

  auto clip_coef_clamped =
      torch::clamp(clip_coef, c10::nullopt /* min */, 1.0 /* max */);
  // 对裁剪系数进行限制，确保其在 [0, 1] 范围内

  for (auto& param : params_with_grad) {
    param.grad().data().mul_(clip_coef_clamped);
    // 对具有梯度的参数的梯度数据乘以裁剪系数
  }

  if (!total_norm.has_value()) {
    // 如果 total_norm 为空
    # 将 total_norm_tensor 的值转换为 Python 中的浮点数类型，并赋值给 total_norm
    total_norm = total_norm_tensor.item().toDouble();
  }
  # 返回 total_norm 的值
  return *total_norm;
// 命名空间 utils 内声明结束

// 命名空间 nn 内声明结束

// 命名空间 torch 内声明结束
```