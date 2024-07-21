# `.\pytorch\aten\src\ATen\native\vulkan\ops\Repeat.cpp`

```
// 包含 Vulkan 操作的通用头文件
#include <ATen/native/vulkan/ops/Common.h>

// 根据是否定义了 AT_PER_OPERATOR_HEADERS 来选择性地包含 ATen 函数头文件或特定操作的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/unsqueeze.h>
#endif

// 包含 Vulkan 操作的实用函数头文件和 Torch 库头文件
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

// 定义 ATen 命名空间下的 Vulkan 相关操作
namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

// 使用 Vulkan 操作相关的 API 工具命名空间
using namespace api::utils;

// 实现 Vulkan 环境下的 repeat 操作，将 self 张量按 repeats 参数进行重复
Tensor repeat(const Tensor& self, const IntArrayRef repeats) {
  // 检查输入张量的维度是否不超过 4，因为 Vulkan 只支持 4 维及以下的张量
  TORCH_CHECK(
      self.dim() <= 4, "Vulkan repeat only supports tensors <= 4 dimensions");
  auto in_ndims = safe_downcast<uint32_t>(self.dim());
  auto out_ndims = safe_downcast<uint32_t>(repeats.size());
  // 检查 repeats 参数的维度数量不小于输入张量的维度数量
  TORCH_CHECK(
      out_ndims >= in_ndims,
      "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor")
  auto add_ndims = out_ndims - in_ndims;

  // 克隆输入张量以备重复操作
  at::Tensor tensor_to_repeat = self.clone();

  // 根据 add_ndims，在需要的维度上进行 unsqueeze 操作，将张量扩展到与 repeats 维度相同
  for (const auto i : c10::irange(add_ndims)) {
    (void)i;
    tensor_to_repeat = at::unsqueeze(tensor_to_repeat, 0);
  }

  // 创建用于存储重复后张量序列的向量
  std::vector<at::Tensor> tensor_seq_to_concat;
  // 根据 repeats 向量，对 tensor_to_repeat 进行重复操作
  for (const auto i : c10::irange(out_ndims)) {
    for (const auto k : c10::irange(repeats[i])) {
      (void)k;
      tensor_seq_to_concat.emplace_back(tensor_to_repeat.clone());
    }
    // 沿着第 i 维度将重复后的张量序列拼接起来
    tensor_to_repeat = at::cat(tensor_seq_to_concat, i);
    tensor_seq_to_concat.clear();
  }
  // 返回重复后的张量
  return tensor_to_repeat;
}

#ifdef USE_VULKAN_API

// 在 Vulkan API 环境下注册 aten::repeat 操作的实现函数
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::repeat"), TORCH_FN(repeat));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
```