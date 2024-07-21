# `.\pytorch\torch\csrc\jit\passes\fold_linear_bn.cpp`

```
#include <torch/csrc/jit/passes/fold_linear_bn.h>

#include <ATen/TensorOperators.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/rsqrt.h>
#endif

namespace torch {
namespace jit {

// 计算更新后的线性层权重和偏置
std::tuple<at::Tensor, at::Tensor> computeUpdatedLinearWeightAndBias(
    const LinearBNParameters& p) {
  // 计算 BatchNorm 的缩放系数
  at::Tensor bn_scale = p.bn_w * at::rsqrt(p.bn_rv + p.bn_eps);
  // 融合权重：线性层权重乘以 BatchNorm 缩放系数，并在最后一个维度上增加维度
  at::Tensor fused_w = p.linear_w * bn_scale.unsqueeze(-1);
  // 融合偏置：线性层偏置先减去 BatchNorm 均值后乘以 BatchNorm 缩放系数，再加上 BatchNorm 偏置
  at::Tensor fused_b = (p.linear_b - p.bn_rm) * bn_scale + p.bn_b;

  // 获取线性层权重和偏置的数据类型
  auto linear_w_dtype = p.linear_w.dtype();
  auto linear_b_dtype = p.linear_b.dtype();

  // 返回更新后的权重和偏置，确保数据类型与原始的线性层权重和偏置一致
  return std::make_tuple(
      fused_w.to(linear_w_dtype), fused_b.to(linear_b_dtype));
}

} // namespace jit
} // namespace torch
```