# `.\pytorch\aten\src\ATen\native\vulkan\ops\Batchnorm.h`

```py
#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/VulkanPackedContext.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

// Vulkan 版本的 Batch Normalization 上下文类，继承自 VulkanPackedContext 和 torch::jit::CustomClassHolder
class BatchNormPackedContext final : virtual public VulkanPackedContext,
                                     public torch::jit::CustomClassHolder {
 private:
  // 存储解包后的参数列表
  c10::impl::GenericList unpacked_;

 public:
  // 构造函数，初始化 BatchNormPackedContext 实例
  BatchNormPackedContext(
      const std::optional<Tensor>& weight_opt,
      const std::optional<Tensor>& bias_opt,
      const std::optional<Tensor>& running_mean_opt,
      const std::optional<Tensor>& running_var_opt,
      double eps);

  /*
   * 为打包/解包列表中的每个索引分配一个名称。
   */
  struct ListArgs final {
    static constexpr uint32_t kWeight = 0u;
    static constexpr uint32_t kBias = 1u;
    static constexpr uint32_t kRunningMean = 2u;
    static constexpr uint32_t kRunningVar = 3u;
    static constexpr uint32_t kEps = 4u;

    static constexpr uint32_t kNumArgs = 5u;
  };

  // 静态方法，根据打包列表创建 BatchNormPackedContext 实例
  static BatchNormPackedContext pack(c10::impl::GenericList);

  // 解包函数的实现，返回解包后的参数列表
  const c10::impl::GenericList unpack() const override {
    TORCH_CHECK(unpacked_.size() > 0u, "unpacked_ does not have any elements!");

    return unpacked_;
  }
};

// 创建 BatchNormPackedContext 实例的工厂函数
c10::intrusive_ptr<BatchNormPackedContext> create_batchnorm_context(
    std::optional<Tensor>&& weight_opt,
    std::optional<Tensor>&& bias_opt,
    std::optional<Tensor>&& running_mean_opt,
    std::optional<Tensor>&& running_var_opt,
    bool training,
    double /* momentum */,
    double eps,
    bool /* cudnn_enable, deprecated */);

// 运行 Batch Normalization 上下文的函数，返回处理后的 Tensor
Tensor run_batchnorm_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<BatchNormPackedContext>& context);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```