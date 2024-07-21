# `.\pytorch\torch\csrc\api\src\nn\modules\instancenorm.cpp`

```py
# 包含 Torch 实例归一化操作所需的头文件
#include <torch/nn/functional/instancenorm.h>

# 包含 Torch 实例归一化模块所需的头文件
#include <torch/nn/modules/instancenorm.h>

# Torch 命名空间
namespace torch {
# Torch 神经网络命名空间
namespace nn {

# 检查输入张量维度是否为 2 或 3
void InstanceNorm1dImpl::_check_input_dim(const Tensor& input) {
  if (input.dim() != 3 && input.dim() != 2) {
    # 抛出错误，期望 2D 或 3D 输入（实际得到 input.dim() 维输入）
    TORCH_CHECK(
        false, "expected 2D or 3D input (got ", input.dim(), "D input)");
  }
}

# 检查输入张量维度是否为 3 或 4
void InstanceNorm2dImpl::_check_input_dim(const Tensor& input) {
  if (input.dim() != 4 && input.dim() != 3) {
    # 抛出错误，期望 3D 或 4D 输入（实际得到 input.dim() 维输入）
    TORCH_CHECK(
        false, "expected 3D or 4D input (got ", input.dim(), "D input)");
  }
}

# 检查输入张量维度是否为 4 或 5
void InstanceNorm3dImpl::_check_input_dim(const Tensor& input) {
  if (input.dim() != 5 &&
      input.dim() != 4) { // NOLINT(cppcoreguidelines-avoid-magic-numbers)
    # 抛出错误，期望 4D 或 5D 输入（实际得到 input.dim() 维输入）
    TORCH_CHECK(
        false, "expected 4D or 5D input (got ", input.dim(), "D input)");
  }
}

# 实例化 InstanceNormImpl 模板类，参数为维度 1 的 InstanceNorm1dImpl 类
template class InstanceNormImpl<1, InstanceNorm1dImpl>;
# 实例化 InstanceNormImpl 模板类，参数为维度 2 的 InstanceNorm2dImpl 类
template class InstanceNormImpl<2, InstanceNorm2dImpl>;
# 实例化 InstanceNormImpl 模板类，参数为维度 3 的 InstanceNorm3dImpl 类
template class InstanceNormImpl<3, InstanceNorm3dImpl>;

# 结束 Torch 神经网络命名空间
} // namespace nn
# 结束 Torch 命名空间
} // namespace torch
```