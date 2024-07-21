# `.\pytorch\aten\src\ATen\native\quantized\cpu\XnnpackUtils.cpp`

```py
#ifdef USE_XNNPACK
// 如果定义了 USE_XNNPACK 宏，则编译以下代码块

#include <ATen/ATen.h>
// 引入 ATen 库，提供张量操作的功能
#include <ATen/quantized/Quantizer.h>
// 引入 ATen 中的量化器定义
#include <ATen/native/quantized/cpu/XnnpackUtils.h>
// 引入 ATen 中 XNNPACK 工具函数的声明
#include <c10/util/irange.h>
// 引入 c10 库中的范围迭代器工具

namespace at {
namespace native {
namespace xnnp_utils {

std::vector<size_t> get_mem_format_aware_shape(const at::Tensor& in) {
  // 获取张量推荐的内存格式
  const auto mem_format = in.suggest_memory_format();
  // 获取张量的大小
  const auto& sizes = in.sizes();
  // 将张量的大小转换为 size_t 类型的向量
  std::vector<size_t> ret(sizes.begin(), sizes.end());
  // 根据内存格式调整大小顺序
  if (mem_format == c10::MemoryFormat::ChannelsLast) {
    // 如果内存格式为 ChannelsLast，则转换为 NHWC 格式
    // NCHW -> NHWC
    // 0123 -> 0231
    ret[1] = sizes[2]; /* H */
    ret[2] = sizes[3]; /* W */
    ret[3] = sizes[1]; /* C */
  } else if (mem_format == c10::MemoryFormat::ChannelsLast3d) {
    // 如果内存格式为 ChannelsLast3d，则转换为 NDHWC 格式
    // NCDHW -> NDHWC
    // 01234 -> 02341
    ret[1] = sizes[2]; /* D */
    ret[2] = sizes[3]; /* H */
    ret[3] = sizes[4]; /* W */
    ret[4] = sizes[1]; /* C */
  }
  // 返回调整后的大小向量
  return ret;
}

template <typename PT>
void q8_copy_int8_weight_and_add_offset(const at::Tensor& in, at::Tensor& out) {
  // 定义 T 为 PT 类型的基础类型
  using T = typename PT::underlying;
  // 根据 PT 的类型选择偏移量
  static constexpr auto offset = std::is_same<T, uint8_t>::value ? 128 : 0;
  // 检查输入张量的数据类型是否为 kQInt8
  TORCH_CHECK(
      in.scalar_type() == c10::kQInt8,
      "q8_copy_int8_weight_and_add_offset: Expected input weight data type ",
      toString(c10::kQInt8),
      " but got ",
      toString(in.scalar_type()))
  // 获取输入张量的指针并转换为 int8_t 类型
  const int8_t* in_ptr =
      reinterpret_cast<const int8_t*>(in.data_ptr<c10::qint8>());
  // 获取输出张量的指针并转换为 T 类型
  T* out_ptr = reinterpret_cast<T*>(out.data_ptr<PT>());

  // 遍历输入张量的所有元素
  for (const auto i : c10::irange(in.numel())) {
    // 将偏移后的 int8 数据存入输出张量
    out_ptr[i] = static_cast<T>(static_cast<int32_t>(in_ptr[i]) + offset);
  }
}

// 显式实例化 q8_copy_int8_weight_and_add_offset 函数模板
template void q8_copy_int8_weight_and_add_offset<c10::quint8>(
    const at::Tensor& in,
    at::Tensor& out);
template void q8_copy_int8_weight_and_add_offset<c10::qint8>(
    const at::Tensor& in,
    at::Tensor& out);

/*
 * 从 fbgemm_utils::ConvertConvWeightsToChannelLastTensor 偷取以避免依赖于 USE_FBGEMM。
 * 重新排列权重，使其符合 xnnpack 的期望格式。
 * TODO: 添加 3D 变体。
 */
template <>
Tensor convert_conv_weights_to_channel_last_tensor<2>(
    const at::Tensor& src,
    int groups,
    bool transpose) {
  return transpose ?
                   // 2D 卷积转置权重转换
                   // IC OC/G KH KW -> G OC/G KH KW IC/G
      [&]() {
        auto ic_g_oc_g_hw_tensors = src.chunk(groups);
        for (auto& tensor : ic_g_oc_g_hw_tensors) {
          tensor = tensor.unsqueeze(0);
        }
        auto fused_tensor = at::cat(ic_g_oc_g_hw_tensors);
        set_quantizer_(fused_tensor, src.quantizer());
        return fused_tensor.permute({0, 2, 3, 4, 1})
            .contiguous(c10::MemoryFormat::Contiguous);
      }()
                   // 2D 卷积权重转换
                   : src.contiguous(c10::MemoryFormat::ChannelsLast);
}

} // namespace xnnp_utils
} // namespace native
} // namespace at

#endif // USE_XNNPACK
// 结束条件：结束 USE_XNNPACK 宏的条件编译
```