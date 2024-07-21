# `.\pytorch\aten\src\ATen\native\vulkan\ops\Lstm.h`

```
#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/VulkanPackedContext.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

// 定义 Vulkan 加速 LSTM 上下文类，继承 VulkanPackedContext 和 torch 自定义类
class LstmPackedContext final : virtual public VulkanPackedContext,
                                public torch::jit::CustomClassHolder {
 public:
  // 构造函数，接受权重/偏置参数（CPU）、是否有偏置、层数、dropout 率、是否训练、是否双向、是否批量优先
  LstmPackedContext(
      const std::vector<Tensor>& params_cpu,
      bool has_biases,
      int64_t num_layers,
      double dropout,
      bool train,
      bool bidirectional,
      bool batch_first);

  /*
   * 为解压缩列表中的每个索引分配一个名称。
   */
  struct Unpacked final {
    static constexpr uint32_t Params = 0u;
    static constexpr uint32_t hasBiases = 1u;
    static constexpr uint32_t NumLayers = 2u;
    static constexpr uint32_t Dropout = 3u;
    static constexpr uint32_t Train = 4u;
    static constexpr uint32_t Bidirectional = 5u;
    static constexpr uint32_t BatchFirst = 6u;

    static constexpr uint32_t NumArgs = 7u;
  };

  /*
   * 为打包列表中的每个索引分配一个名称。
   */
  struct Packed final {
    static constexpr uint32_t LinearContexts = 0u;
    static constexpr uint32_t hasBiases = 1u;
    static constexpr uint32_t NumLayers = 2u;
    static constexpr uint32_t Dropout = 3u;
    static constexpr uint32_t Train = 4u;
    static constexpr uint32_t Bidirectional = 5u;
    static constexpr uint32_t BatchFirst = 6u;

    static constexpr uint32_t NumArgs = 7u;
  };

  // 将 C10 泛型列表打包为 LSTM 上下文对象
  static LstmPackedContext pack(c10::impl::GenericList);

  // 解包函数，返回 C10 泛型列表
  const c10::impl::GenericList unpack() const override;
};

// 创建 Vulkan 加速 LSTM 上下文对象的工厂函数
c10::intrusive_ptr<LstmPackedContext> create_lstm_context(
    std::vector<Tensor>&& params_cpu,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first);

// 运行 Vulkan 加速 LSTM 上下文对象的函数，返回隐藏状态、细胞状态和输出张量元组
std::tuple<Tensor, Tensor, Tensor> run_lstm_context(
    const Tensor& input_vk, // 输入序列（Vulkan 加速）
    const Tensor& hx_vk,    // 初始隐藏状态（Vulkan 加速）
    const Tensor& cx_vk,    // 初始细胞状态（Vulkan 加速）
    const c10::intrusive_ptr<LstmPackedContext>& vulkan_context);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```