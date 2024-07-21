# `.\pytorch\torch\csrc\api\include\torch\nn\init.h`

```py
#pragma once

#include <torch/csrc/Export.h>
#include <torch/enum.h>
#include <torch/types.h>

namespace torch {
namespace nn {
namespace init {

// 定义非线性激活函数类型的枚举变体
using NonlinearityType = std::variant<
    enumtype::kLinear,
    enumtype::kConv1D,
    enumtype::kConv2D,
    enumtype::kConv3D,
    enumtype::kConvTranspose1D,
    enumtype::kConvTranspose2D,
    enumtype::kConvTranspose3D,
    enumtype::kSigmoid,
    enumtype::kTanh,
    enumtype::kReLU,
    enumtype::kLeakyReLU>;

// 定义初始化中的扇入/扇出模式类型的枚举变体
using FanModeType = std::variant<enumtype::kFanIn, enumtype::kFanOut>;

} // namespace init
} // namespace nn

namespace nn {
namespace init {

/// 返回给定非线性函数的推荐增益值。
TORCH_API double calculate_gain(
    NonlinearityType nonlinearity,
    double param = 0.01);

/// 用提供的值填充给定的 `tensor`，并返回它。
/// 此操作不会记录梯度。
TORCH_API Tensor constant_(Tensor tensor, Scalar value);

/// 用Dirac delta函数填充给定的 `tensor`，并返回它。
/// 此操作不会记录梯度。
TORCH_API Tensor dirac_(Tensor tensor);

/// 用单位矩阵填充给定的二维 `matrix`。
/// 此操作不会记录梯度。
TORCH_API Tensor eye_(Tensor matrix);

/// 用从正态分布中参数化的 `mean` 和 `std` 填充给定的二维 `tensor`。
/// 此操作不会记录梯度。
TORCH_API Tensor normal_(Tensor tensor, double mean = 0, double std = 1);

/// 用全为1的值填充给定的 `tensor`。
/// 此操作不会记录梯度。
TORCH_API Tensor ones_(Tensor tensor);

/// 用(半)正交矩阵填充输入的 `Tensor`，如"Saxe, A. et al. (2013)"中所述。
/// 输入的 `tensor` 必须至少有2维，对于维度大于2的张量，会将尾部维度展平。
/// 此操作不会记录梯度。
TORCH_API Tensor orthogonal_(Tensor tensor, double gain = 1.0);

/// 用稀疏矩阵形式填充给定的2维 `tensor`，非零元素来自于带有给定标准差 `std` 的中心化正态分布。
/// `sparsity` 是一个实数，控制每列中要设置为零的元素的分数，如"Martens, J. (2010)"中所述。
/// 此操作不会记录梯度。
TORCH_API Tensor sparse_(Tensor tensor, double sparsity, double std = 0.01);

/// 用从均匀分布中参数化的 `low` 和 `high` 填充给定的二维 `tensor`。
/// 此操作不会记录梯度。
TORCH_API Tensor uniform_(Tensor tensor, double low = 0, double high = 1);
/// 填充输入的 `Tensor`，使用 "深入探索修正线性单元：在ImageNet分类任务上超越人类水平性能" 的方法
/// - He, K. 等人 (2015) 中描述的方法，使用正态分布。也称为 He 初始化。
/// 此操作不会记录梯度。
TORCH_API Tensor kaiming_normal_(
    Tensor tensor,                             // 输入的张量
    double a = 0,                              // 参数 a，默认为 0
    FanModeType mode = torch::kFanIn,          // 粉丝模式类型，默认为输入粉丝模式
    NonlinearityType nonlinearity = torch::kLeakyReLU);  // 非线性类型，默认为 LeakyReLU

/// 填充输入的 `Tensor`，使用 "深入探索修正线性单元：在ImageNet分类任务上超越人类水平性能" 的方法
/// - He, K. 等人 (2015) 中描述的方法，使用均匀分布。也称为 He 初始化。
/// 此操作不会记录梯度。
TORCH_API Tensor kaiming_uniform_(
    Tensor tensor,                             // 输入的张量
    double a = 0,                              // 参数 a，默认为 0
    FanModeType mode = torch::kFanIn,          // 粉丝模式类型，默认为输入粉丝模式
    NonlinearityType nonlinearity = torch::kLeakyReLU);  // 非线性类型，默认为 LeakyReLU

/// 填充输入的 `Tensor`，使用 "理解训练深度前馈神经网络的困难性" - Glorot, X. & Bengio, Y. (2010) 中描述的方法
/// 值由 `gain` 参数缩放。此操作不会记录梯度。
TORCH_API Tensor xavier_normal_(
    Tensor tensor,                             // 输入的张量
    double gain = 1.0);                        // 增益参数，默认为 1.0

/// 填充输入的 `Tensor`，使用 "理解训练深度前馈神经网络的困难性" - Glorot, X. & Bengio, Y. (2010) 中描述的方法
/// 使用均匀分布。值由 `gain` 参数缩放。此操作不会记录梯度。
TORCH_API Tensor xavier_uniform_(
    Tensor tensor,                             // 输入的张量
    double gain = 1.0);                        // 增益参数，默认为 1.0

/// 将给定的 `tensor` 填充为零。
/// 此操作不会记录梯度。
TORCH_API Tensor zeros_(
    Tensor tensor);                            // 输入的张量

/// 计算给定张量的 fan_in 和 fan_out。
TORCH_API std::tuple<int64_t, int64_t> _calculate_fan_in_and_fan_out(
    const Tensor& tensor);                     // 输入的张量

} // namespace init
} // namespace nn
} // namespace torch
```