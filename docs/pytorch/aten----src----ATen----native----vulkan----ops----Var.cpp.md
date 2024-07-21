# `.\pytorch\aten\src\ATen\native\vulkan\ops\Var.cpp`

```py
// 包含 Vulkan 操作的常用头文件
#include <ATen/native/vulkan/ops/Common.h>
// 包含 Vulkan 操作的工具函数头文件
#include <ATen/native/vulkan/ops/Utils.h>
// 包含 Torch 库的头文件
#include <torch/library.h>

// 定义 at 命名空间
namespace at {
// 定义 native 命名空间
namespace native {
// 定义 Vulkan 命名空间
namespace vulkan {
// 定义 Vulkan 操作命名空间
namespace ops {
// 匿名命名空间，限定作用域
namespace {

// 使用 api::utils 命名空间
using namespace api::utils;

// Vulkan 实现的 var_dim 函数，计算指定维度上的方差
Tensor var_dim_IntList(
    const at::Tensor& self_arg,             // 输入张量
    const OptionalIntArrayRef opt_dim,      // 可选的维度列表
    bool unbiased = true,                   // 是否无偏估计，默认为 true
    bool keepdim = false) {                 // 是否保持维度，默认为 false

  // 检查输入张量的维度是否在支持的范围内
  TORCH_CHECK(
      self_arg.dim() >= 2 && self_arg.dim() <= 4,
      "Vulkan var.dim_IntList only supports 2d, 3d, 4d tensors as input!");

  // 检查是否提供了维度参数
  TORCH_CHECK(
      opt_dim.has_value(), "Vulkan var without a dim arg is not implemented");

  // 如果输入张量不是 Vulkan 张量，则转换为 Vulkan 张量
  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();

  // 用于存储唯一维度的集合
  std::set<int64_t> dims_set;

  if (opt_dim.has_value()) {
    int sample_size = 1;
    auto dims = opt_dim.value();

    // 遍历指定的维度列表
    for (const auto& d : dims) {
      // 检查维度是否在合理范围内
      TORCH_CHECK(d >= -self.dim() || d < self.dim(), "Dimension out of range");

      // 标准化维度索引，确保在合理范围内
      int64_t dim_normalized = utils::normalize(d, self.dim());

      // 检查是否有重复的维度
      if (dims_set.find(dim_normalized) != dims_set.end()) {
        TORCH_CHECK(
            false,
            "dim ",
            dim_normalized,
            " appears multiple times in the list of dims")
      }
      dims_set.insert(dim_normalized);

      // 计算样本大小
      sample_size *= self.sizes().vec()[dim_normalized];
    }

    // 计算沿指定维度上的均值
    at::Tensor self_mean = self.mean(opt_dim, true);
    // 计算原始值减去均值的张量
    at::Tensor self_minus_mean = self.sub(self_mean);
    // 计算平方后的张量，并求沿指定维度的均值
    at::Tensor output =
        self_minus_mean.mul(self_minus_mean).mean(opt_dim, keepdim);

    // 如果 unbiased 参数为 true，则进行无偏估计的修正
    if (unbiased == true) {
      output = output.mul(sample_size * 1.0 / (sample_size - 1));
    }
    return output;
  }

  // 如果未提供维度参数，则直接返回输入张量
  return self;
}

// 当使用 Vulkan API 时注册 Vulkan 实现的 var.dim 函数
#ifdef USE_VULKAN_API
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::var.dim"), TORCH_FN(var_dim_IntList));
}
#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
```