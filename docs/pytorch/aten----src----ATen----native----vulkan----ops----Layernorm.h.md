# `.\pytorch\aten\src\ATen\native\vulkan\ops\Layernorm.h`

```py
#pragma once

#ifdef USE_VULKAN_API
// 包含 Vulkan 相关的通用头文件
#include <ATen/native/vulkan/ops/Common.h>
// 包含 Vulkan 操作上下文的头文件
#include <ATen/native/vulkan/ops/VulkanPackedContext.h>
// 包含 Torch 库的头文件
#include <torch/library.h>

// 定义 at 命名空间
namespace at {
// 定义 native 命名空间
namespace native {
// 定义 vulkan 命名空间
namespace vulkan {
// 定义 ops 命名空间
namespace ops {

// LayernormPackedContext 类继承自 VulkanPackedContext 并实现 CustomClassHolder 接口
class LayernormPackedContext final : virtual public VulkanPackedContext,
                                     public torch::jit::CustomClassHolder {
 private:
  // 用于存储解压后数据的 GenericList
  c10::impl::GenericList unpacked_;

 public:
  // LayernormPackedContext 构造函数，接收权重、偏置和 epsilon 参数
  LayernormPackedContext(
      const std::optional<Tensor>& weight,
      const std::optional<Tensor>& bias,
      double eps);

  // ListArgs 结构体定义每个元素在 unpacked_ 中的索引
  struct ListArgs final {
    static constexpr uint32_t kWeight = 0u;  // 权重在 unpacked_ 中的索引
    static constexpr uint32_t kBias = 1u;    // 偏置在 unpacked_ 中的索引
    static constexpr uint32_t kEps = 2u;     // epsilon 在 unpacked_ 中的索引

    static constexpr uint32_t kNumArgs = 3u; // unpacked_ 中的元素个数
  };

  // 静态方法，将 GenericList 打包成 LayernormPackedContext 对象
  static LayernormPackedContext pack(const c10::impl::GenericList);

  // 实现虚函数，返回解压后的 GenericList
  const c10::impl::GenericList unpack() const override {
    // 检查 unpacked_ 是否为空，若为空则抛出异常
    TORCH_CHECK(!unpacked_.empty(), "unpacked_ does not have any elements!");

    // 返回解压后的 GenericList
    return unpacked_;
  }
};

// 创建 LayernormPackedContext 对象的工厂函数
c10::intrusive_ptr<LayernormPackedContext> create_layernorm_context(
    std::optional<Tensor>&& weight,
    std::optional<Tensor>&& bias,
    double eps);

// 运行 LayernormPackedContext 对象的函数，对输入进行 layernorm 操作
Tensor run_layernorm_context(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& context);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```