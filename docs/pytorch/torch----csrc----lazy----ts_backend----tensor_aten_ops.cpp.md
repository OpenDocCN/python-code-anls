# `.\pytorch\torch\csrc\lazy\ts_backend\tensor_aten_ops.cpp`

```
#include <torch/csrc/lazy/ts_backend/tensor_aten_ops.h>

#include <ATen/InferSize.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/ops/arithmetic_ir_ops.h>
#include <torch/csrc/lazy/core/ops/utils.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <torch/csrc/lazy/core/util.h>
#include <torch/csrc/lazy/generated/LazyIr.h>
#include <algorithm>
#include <functional>

namespace torch {
namespace lazy {
namespace {

// to enable operator+-*/ for Value
using namespace torch::lazy;

// 定义匹配输入形状和目标形状的操作，可能进行扩展
torch::lazy::Value MaybeExpand(
    const torch::lazy::Value& input,
    const torch::lazy::Shape& target_shape) {
  // 如果输入的形状与目标形状相同，则直接返回输入
  if (input.shape().sizes() == target_shape.sizes()) {
    return input;
  }
  // 否则创建并返回一个扩展操作
  return torch::lazy::MakeExpand(
      input,
      target_shape.sizes().vec(),
      /*is_scalar_expand=*/false);
}

} // namespace

//////////////////////////////////////////////////////////////////////////////
// 下面是 ATEN 操作符，按字母顺序列出。
//////////////////////////////////////////////////////////////////////////////

// 将 LazyTensorPtr 中的值填充为指定的标量值
void fill_(torch::lazy::LazyTensorPtr& input, const at::Scalar& value) {
  // 获取扩展标量值对应的 IR 值，并设置到输入的 IR 值中
  torch::lazy::Value constant =
      torch::lazy::LazyGraphExecutor::Get()->GetIrValueForExpandedScalar(
          value, input->shape(), input->GetDevice());
  input->SetInPlaceIrValue(std::move(constant));
}

// 将一个 LazyTensorPtr 的值复制到另一个 LazyTensorPtr 中
void copy_(torch::lazy::LazyTensorPtr& input, torch::lazy::LazyTensorPtr& src) {
  // 如果输入和源张量在同一个设备上
  if (input->GetDevice() == src->GetDevice()) {
    torch::lazy::Value copy_value;
    // 如果输入和源张量的数据类型相同，则直接复制源张量的 IR 值
    if (input->dtype() == src->dtype()) {
      copy_value = src->GetIrValue();
    } else {
      // 否则进行类型转换，并获取转换后的 IR 值
      copy_value = torch::lazy::MakeCast(
          src->GetIrValue(), input->dtype(), src->dtype());
    }
    // 将复制的 IR 值可能进行形状扩展后设置到输入的 IR 值中
    input->SetIrValue(MaybeExpand(copy_value, input->shape()));
  } else {
    // 如果输入和源张量不在同一个设备上，则需要更新输入张量的值
    auto input_shape = input->shape();
    // 将源张量转换为 ATen 张量，可能进行分离操作
    at::Tensor src_tensor = src->ToTensor(/*detached=*/true);
    // 如果源张量的形状不等于输入张量的形状，则进行扩展
    if (src_tensor.sizes() != input_shape.Get().sizes()) {
      src_tensor = src_tensor.expand(input_shape.Get().sizes().vec());
    }
    // 更新输入张量的值，并不同步更新
    input->UpdateFromTensor(std::move(src_tensor), /*sync=*/false);
  }
}

} // namespace lazy
} // namespace torch


这段代码是 C++ 代码，实现了一些与懒惰张量（LazyTensor）相关的操作，包括填充操作和复制操作。
```