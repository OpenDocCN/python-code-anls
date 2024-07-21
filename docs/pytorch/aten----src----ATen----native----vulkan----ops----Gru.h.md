# `.\pytorch\aten\src\ATen\native\vulkan\ops\Gru.h`

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

// Vulkan 特定的 GRU 上下文类，继承自 VulkanPackedContext 和 torch 的自定义类持有者
class GruPackedContext final : virtual public VulkanPackedContext,
                               public torch::jit::CustomClassHolder {
 public:
  // 构造函数，初始化 GRU 上下文对象
  GruPackedContext(
      const std::vector<Tensor>& params_cpu, // 权重/偏置（CPU）
      bool has_biases, // 是否包含偏置
      int64_t num_layers, // 层数
      double dropout, // dropout 率
      bool train, // 是否训练模式
      bool bidirectional, // 是否双向
      bool batch_first); // 是否批处理优先

  /*
   * 给解包后的列表中的每个索引分配一个名称。
   */
  struct Unpacked final {
    static constexpr uint32_t Params = 0u; // 参数
    static constexpr uint32_t hasBiases = 1u; // 是否有偏置
    static constexpr uint32_t NumLayers = 2u; // 层数
    static constexpr uint32_t Dropout = 3u; // dropout 率
    static constexpr uint32_t Train = 4u; // 是否训练
    static constexpr uint32_t Bidirectional = 5u; // 是否双向
    static constexpr uint32_t BatchFirst = 6u; // 批处理优先

    static constexpr uint32_t NumArgs = 7u; // 参数个数
  };

  /*
   * 给打包后的列表中的每个索引分配一个名称。
   */
  struct Packed final {
    static constexpr uint32_t LinearContexts = 0u; // 线性上下文
    static constexpr uint32_t hasBiases = 1u; // 是否有偏置
    static constexpr uint32_t NumLayers = 2u; // 层数
    static constexpr uint32_t Dropout = 3u; // dropout 率
    static constexpr uint32_t Train = 4u; // 是否训练
    static constexpr uint32_t Bidirectional = 5u; // 是否双向
    static constexpr uint32_t BatchFirst = 6u; // 批处理优先

    static constexpr uint32_t NumArgs = 7u; // 参数个数
  };

  // 将 c10::impl::GenericList 打包为 GruPackedContext 对象
  static GruPackedContext pack(c10::impl::GenericList);

  // 解包 GruPackedContext 对象，返回 c10::impl::GenericList
  const c10::impl::GenericList unpack() const override;
};

// 创建一个 GRU 上下文对象的工厂函数
c10::intrusive_ptr<GruPackedContext> create_gru_context(
    std::vector<Tensor>&& params_cpu, // 权重/偏置（CPU）
    bool has_biases, // 是否包含偏置
    int64_t num_layers, // 层数
    double dropout, // dropout 率
    bool train, // 是否训练模式
    bool bidirectional, // 是否双向
    bool batch_first); // 是否批处理优先

// 运行 GRU 上下文对象，返回输出张量和隐藏状态张量的元组
std::tuple<Tensor, Tensor> run_gru_context(
    const Tensor& input_vk, // 输入张量
    const Tensor& hx_vk, // 隐藏状态张量
    const c10::intrusive_ptr<GruPackedContext>& vulkan_context); // Vulkan 上下文对象

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```