# `.\pytorch\aten\src\ATen\native\vulkan\impl\Common.h`

```py
#pragma once


// 如果尚未定义 USE_VULKAN_API，则仅在此处包含代码
#ifdef USE_VULKAN_API

// 包含 Vulkan API 的头文件
#include <ATen/native/vulkan/api/api.h>

// 在 at 命名空间内声明
namespace at {
namespace native {
namespace vulkan {

/*
 * 将语义维度名称映射到其在 NCHW 格式的4D张量中的最内层顺序对应的整数。
 * Width 是最内层维度，因此对应于1；height 是其次内层维度，因此对应于2，依此类推。
 */
struct Dim4D {
  static constexpr uint32_t Width = 1u;
  static constexpr uint32_t Height = 2u;
  static constexpr uint32_t Channel = 3u;
  static constexpr uint32_t Batch = 4u;
};

/*
 * 1D张量的语义维度名称
 */
struct Dim1D {
  static constexpr uint32_t Length = 1u;
};

/*
 * 2D卷积核的语义维度名称
 */
struct DimConv2DKernel {
  static constexpr uint32_t Width = 1u;
  static constexpr uint32_t Height = 2u;
  static constexpr uint32_t InChannels = 3u;
  static constexpr uint32_t OutChannels = 4u;
};

/*
 * 与上面相同，但适用于2D转置卷积核。
 */
struct DimTConv2DKernel {
  static constexpr uint32_t Width = 1u;
  static constexpr uint32_t Height = 2u;
  static constexpr uint32_t OutChannels = 3u;
  static constexpr uint32_t InChannels = 4u;
};

/*
 * 下面的函数安全地返回第N个最内层索引处的维度大小。
 * 如果尺寸数组的维度不够，则返回1。
 * 上述结构体旨在与这些函数一起使用。
 */
template <uint32_t N>
uint32_t dim_at(const std::vector<int64_t>& sizes) {
  const uint32_t dims = sizes.size();
  return dims < N ? 1 : api::utils::safe_downcast<uint32_t>(sizes[dims - N]);
}

template <uint32_t N>
uint32_t dim_at(const vTensor& v_in) {
  return dim_at<N>(v_in.sizes());
}

/*
 * 对于大多数全局工作组大小，返回 {4, 4, 4}，但调整为2D全局工作组大小。
 * 总是保持64个调用
 */
api::utils::uvec3 adaptive_work_group_size(
    const api::utils::uvec3& global_work_group);

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```