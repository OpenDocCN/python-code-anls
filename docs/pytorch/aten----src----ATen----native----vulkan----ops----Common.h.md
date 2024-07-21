# `.\pytorch\aten\src\ATen\native\vulkan\ops\Common.h`

```py
#pragma once

#ifdef USE_VULKAN_API

#include <c10/util/ArrayRef.h>

#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/vulkan/api/api.h>
#include <ATen/native/vulkan/impl/Common.h>
#include <ATen/native/vulkan/ops/Convert.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

// 定义了不同数据结构的维度索引常量，用于处理四维激活图、卷积滤波器等的维度
struct Layout final {
  // 4D Activation Maps (四维激活图)
  struct Activation4D final {
    static constexpr size_t batch = 0u;   // 批次维度索引
    static constexpr size_t channels = 1u;   // 通道维度索引
    static constexpr size_t height = 2u;   // 高度维度索引
    static constexpr size_t width = 3u;   // 宽度维度索引
  };

  // Convolution Filters (卷积滤波器)
  struct Filter final {
    static constexpr size_t output = 0u;   // 输出维度索引
    static constexpr size_t input = 1u;   // 输入维度索引
    static constexpr size_t height = 2u;   // 高度维度索引
    static constexpr size_t width = 3u;   // 宽度维度索引
  };

  // Transposed Convolution Filters (转置卷积滤波器)
  struct TransposedFilter final {
    static constexpr size_t input = 0u;   // 输入维度索引
    static constexpr size_t output = 1u;   // 输出维度索引
    static constexpr size_t height = 2u;   // 高度维度索引
    static constexpr size_t width = 3u;   // 宽度维度索引
  };

  // Parameters (Pooling Kernels, Dilation, Padding, Stride, etc.) (参数，如池化核、扩展、填充、步长等)
  struct Parameter final {
    static constexpr size_t height = 0u;   // 高度参数索引
    static constexpr size_t width = 1u;   // 宽度参数索引
  };

  // Parameters (Pooling Kernels, Dilation, Padding, Stride, etc.) (参数，如池化核、扩展、填充、步长等)
  struct BatchMatrices final {
    static constexpr size_t batch = 0u;   // 批次维度索引
    static constexpr size_t height = 1u;   // 高度维度索引
    static constexpr size_t width = 2u;   // 宽度维度索引
  };
};

/*
 * The functions below safely return the size of the dimension at the N-th
 * innermost index. If the dimensionality of the size array is not sufficient
 * then 1 will be returned. The structs above are intended to be used with
 * these functions.
 */

// 获取数组大小的安全函数，根据给定的 N 返回 IntArrayRef 的第 N 个内层索引的大小
template <uint32_t N>
uint32_t get_dim(const IntArrayRef sizes) {
  const uint32_t dims = sizes.size();   // 获取尺寸数组的维度数
  return dims < N ? 1 : api::utils::safe_downcast<uint32_t>(sizes[dims - N]);   // 如果维度不足 N，返回 1；否则返回对应索引的大小
}

// 获取张量大小的安全函数，根据给定的 N 返回 Tensor 对象的第 N 个内层索引的大小
template <uint32_t N>
uint32_t get_dim(const Tensor& t_in) {
  return get_dim<N>(t_in.sizes());   // 调用上面的函数，传入张量的尺寸数组
}

// 获取 Vulkan 张量大小的安全函数，根据给定的 N 返回 vTensor 对象的第 N 个内层索引的大小
template <uint32_t N>
uint32_t get_dim(const vTensor& v_in) {
  return get_dim<N>(v_in.sizes());   // 调用上面的函数，传入 Vulkan 张量的尺寸数组
}

// 从泛型列表中获取可选的张量对象
inline std::optional<Tensor> get_optional_tensor(
    const c10::impl::GenericList& gen_list,
    const uint32_t idx) {
  return gen_list.get(idx).isTensor() ? gen_list.get(idx).toTensor()   // 如果泛型列表中指定索引处是张量，则返回该张量
                                      : std::optional<Tensor>();   // 否则返回空的可选张量对象
}

// 从泛型列表中获取可选的标量对象
inline std::optional<Scalar> get_optional_scalar(
    const c10::impl::GenericList& gen_list,
    const uint32_t idx) {
  return gen_list.get(idx).isScalar() ? gen_list.get(idx).toScalar()   // 如果泛型列表中指定索引处是标量，则返回该标量
                                      : std::optional<Scalar>();   // 否则返回空的可选标量对象
}

// 返回最接近的整数值
inline float roundevenf(float v) {
  return (float)nearbyint(v);   // 调用数学库函数，返回最接近 v 的整数值
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```