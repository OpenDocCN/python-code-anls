# `.\pytorch\aten\src\ATen\native\vulkan\ops\Mm.h`

```
#pragma once
// 如果定义了 USE_VULKAN_API，则包含以下头文件

#ifdef USE_VULKAN_API
// 包含 Vulkan 相关的头文件
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/ops/VulkanPackedContext.h>
#include <torch/library.h>

// 进入 at 命名空间
namespace at {
// 进入 native 命名空间
namespace native {
// 进入 vulkan 命名空间
namespace vulkan {
// 进入 ops 命名空间
namespace ops {

// 模板函数定义，用于将权重数据打包到 Vulkan 张量中
template <typename T>
void stage_pack_weights(
    api::Context* const context,   // Vulkan 上下文指针
    vTensor& v_weight,             // Vulkan 张量引用
    const Tensor& weight,          // 输入的权重张量
    const int64_t src_kb_sz,       // 输入张量的批量大小
    const int64_t src_kh_sz,       // 输入张量的高度尺寸
    const int64_t src_kw_sz,       // 输入张量的宽度尺寸
    const int64_t dst_kh_sz,       // 目标张量的高度尺寸
    const int64_t dst_kw_sz) {     // 目标张量的宽度尺寸
  const int64_t src_matrix_sz = src_kw_sz * src_kh_sz;   // 源矩阵大小
  const int64_t dst_plane_sz = dst_kw_sz * dst_kh_sz;    // 目标平面大小
  const int64_t dst_matrix_sz = dst_plane_sz * 4;        // 目标矩阵大小
  const T* const src_weight_ptr = weight.const_data_ptr<T>();   // 获取输入权重数据的指针
  api::StorageBuffer staging(context, api::kFloat, v_weight.gpu_numel());   // 创建 Vulkan 存储缓冲区对象

  {
    // 创建内存映射对象，用于写入操作
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

    T* dst_weight_ptr = mapping.template data<T>();   // 获取映射的数据指针

    memset(dst_weight_ptr, 0, v_weight.nbytes());   // 初始化目标数据为零

    // 循环遍历输入张量的每个元素
    for (const auto src_b : c10::irange(src_kb_sz)) {
      for (const auto src_h : c10::irange(src_kh_sz)) {
        for (const auto src_w : c10::irange(src_kw_sz)) {
          // 计算目标平面和索引
          int64_t dst_plane = 2 * (src_h % 2) + (src_w % 2);
          int64_t dst_index = (src_h / 2) * dst_kw_sz + (src_w / 2);
          // 将数据从输入权重复制到目标权重
          memcpy(
              dst_weight_ptr + src_b * dst_matrix_sz +
                  dst_plane * dst_plane_sz + dst_index,
              src_weight_ptr + src_b * src_matrix_sz + src_h * src_kw_sz +
                  src_w,
              sizeof(T));
        }
      }
    }
  }

  // 将暂存的数据打包到 Vulkan 张量中
  utils::pack_staging_to_vtensor(staging.buffer(), v_weight);
}

// LinearPackedContext 类的声明，继承自 VulkanPackedContext 和 torch::jit::CustomClassHolder
class LinearPackedContext final : virtual public VulkanPackedContext,
                                  public torch::jit::CustomClassHolder {
 private:
  c10::impl::GenericList unpacked_;   // 存储解包的张量列表

 public:
  // 构造函数，接受权重张量、偏置张量和一个是否使用批处理的标志位
  LinearPackedContext(
      const Tensor& weight,
      const std::optional<Tensor>& bias,
      const bool use_batch = false);

  // 内部类 Unpacked 的声明，用于为解包列表中的索引分配名称
  struct Unpacked final {
    static constexpr uint32_t Weight = 0u;
    static constexpr uint32_t Bias = 1u;

    static constexpr uint32_t NumArgs = 2u;
  };

  // 内部类 Packed 的声明，用于为打包列表中的索引分配名称
  struct Packed final {
    static constexpr uint32_t Weight = 0u;
    static constexpr uint32_t Bias = 1u;
    static constexpr uint32_t WeightSizes = 2u;
    static constexpr uint32_t BiasDefined = 3u;

    static constexpr uint32_t NumArgs = 4u;
  };

  // 静态方法，用于从泛型列表中打包 LinearPackedContext 对象
  static LinearPackedContext pack(c10::impl::GenericList);

  // 解包方法的实现，返回解包的泛型列表
  const c10::impl::GenericList unpack() const override {
    TORCH_CHECK(unpacked_.size() > 0u, "unpacked_ does not have any elements!");

    return unpacked_;
  }
};

// 创建 LinearPackedContext 对象的工厂函数，接受权重张量和偏置张量的右值引用
c10::intrusive_ptr<LinearPackedContext> create_linear_context(
    Tensor&& weight,
    std::optional<Tensor>&& bias);

// 运行 LinearPackedContext 对象的函数声明，尚未完整
Tensor run_linear_context(
    // 声明一个名为 input 的常量引用，表示输入张量
    const Tensor& input,
    // 声明一个名为 context 的指向 LinearPackedContext 的共享指针，表示线性层打包上下文
    const c10::intrusive_ptr<LinearPackedContext>& context);
    // 声明一个函数，参数为 input 张量和线性层打包上下文的共享指针，无返回值
// 定义一个函数 `run_qlinear_context`，该函数返回一个 `Tensor` 类型的对象，接受以下参数：
// - `input`：输入的张量对象
// - `output_scale`：输出的缩放因子，类型为双精度浮点数
// - `output_zero_point`：输出的零点偏移量，类型为 64 位整数
// - `context`：一个指向 `LinearPackedContext` 类型对象的智能指针

Tensor run_qlinear_context(
    const Tensor& input,
    double output_scale,
    int64_t output_zero_point,
    const c10::intrusive_ptr<LinearPackedContext>& context);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```