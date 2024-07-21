# `.\pytorch\aten\src\ATen\native\vulkan\ops\Transpose.cpp`

```py
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

// 定义 Vulkan 下的 4D 转置操作函数，接受输入张量、输入尺寸、输出尺寸、输出维度及 Vulkan 张量作为参数
Tensor transpose_4d(
    const Tensor& input_arg,
    const uvec4& in_size,
    const uvec4& out_size,
    const uvec4& out_dims,
    vTensor& v_output) {
  
  // 获取当前 Vulkan 的上下文
  api::Context* const context = api::context();

  // 如果输入张量是 Vulkan 张量，则直接使用，否则转换成 Vulkan 张量
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_self = convert(input);

  // 提取输出通道数和输入通道数
  uint32_t out_channels = out_size.data[1u];
  uint32_t in_channels = in_size.data[1u];

  // 将输出通道数和输入通道数向上对齐到4的倍数
  uint32_t out_c_aligned = api::utils::align_up(out_channels, 4u);
  uint32_t in_c_aligned = api::utils::align_up(in_channels, 4u);

  // 定义包含数据的结构体 Block
  const struct Block final {
    ivec3 out_extents;
    int32_t fill0;
    ivec3 in_extents;
    int32_t fill1;
    uvec4 out_tensor_size;
    uvec4 in_tensor_size;
    uvec4 out_ndims;
    uvec2 ch_info;
  } block{
      // 使用 Vulkan 张量的形状创建块结构体
      api::utils::make_ivec3(v_output.extents()),
      0,
      api::utils::make_ivec3(v_self.extents()),
      0,
      out_size,
      in_size,
      out_dims,
      {out_c_aligned, in_c_aligned},
  };

  // 创建统一参数缓冲区对象
  api::UniformParamsBuffer params(context, block);
  
  // 创建管线屏障对象
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算任务到 Vulkan 上下文
  context->submit_compute_job(
      // Vulkan shader 核心描述符
      VK_KERNEL(permute_4d),
      // 管线屏障
      pipeline_barrier,
      // 全局工作组大小
      v_output.extents(),
      // 本地工作组大小
      adaptive_work_group_size(v_output.extents()),
      // 栅栏句柄
      VK_NULL_HANDLE,
      // Vulkan 着色器参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());

  // 将 Vulkan 张量转换为普通张量并返回
  return convert(v_output);
}

// Vulkan 下的转置操作函数，接受输入张量和两个索引参数
Tensor transpose(const Tensor& self, int64_t index0, int64_t index1) {
  // 检查输入张量维度是否不超过4
  TORCH_CHECK(
      self.dim() <= 4,
      "Vulkan transpose only supports tensors <= 4 dimensions");

  // 安全地转换张量维度数为 uint32_t 类型
  auto nDims = safe_downcast<uint32_t>(self.dim());
  uvec4 in_size{1u, 1u, 1u, 1u}, out_size{1u, 1u, 1u, 1u};
  uvec4 out_dims{0u, 1u, 2u, 3u};

  // 获取输入张量的维度大小
  auto oldSizes = self.sizes();
  DimVector newSizes(nDims);
  auto new_index0 = safe_downcast<uint32_t>(maybe_wrap_dim(index0, nDims));
  auto new_index1 = safe_downcast<uint32_t>(maybe_wrap_dim(index1, nDims));

  // 如果两个索引相同，返回输入张量的分离副本
  if (new_index0 == new_index1) {
    return self.detach();
  }

  // 将输入张量和输出张量泛化为4维张量
  for (const auto i : c10::irange(nDims)) {
    in_size.data[(4u - nDims) + i] = self.sizes()[i];
    out_size.data[(4u - nDims) + i] = self.sizes()[i];
    // 将新尺寸数组newSizes的第i个元素设置为旧尺寸数组oldSizes的第i个元素的值
    newSizes[i] = oldSizes[i];
  }

  // 通过交换索引0和索引1处的输入尺寸来获取输出的尺寸大小
  // 继续上面的示例，如果index0 = 0，index1 = 2，则输出大小out_size = [1, 4, 3, 2]。
  // 注意：由于输入是广义化为4维，索引偏移为(4u - nDims)。
  out_size.data[(4u - nDims) + new_index0] =
      in_size.data[(4u - nDims) + new_index1];
  out_size.data[(4u - nDims) + new_index1] =
      in_size.data[(4u - nDims) + new_index0];

  // 获取所需的维度顺序，同样需要进行(4u - nDims)的偏移。
  // 使用上面的示例，out_dims = [0, 3, 2, 1]
  auto temp_dim = out_dims.data[(4u - nDims) + new_index0];
  out_dims.data[(4u - nDims) + new_index0] =
      out_dims.data[(4u - nDims) + new_index1];
  out_dims.data[(4u - nDims) + new_index1] = temp_dim;

  // 通过交换输入尺寸来获取输出的尺寸大小。继续上面的示例，newSizes = [1, 4, 3, 2]
  newSizes[new_index0] = oldSizes[new_index1];
  newSizes[new_index1] = oldSizes[new_index0];

  // 创建一个IntArrayRef类型的输出尺寸引用，使用newSizes数组初始化
  IntArrayRef output_size(newSizes);
  // 创建一个vTensor类型的输出张量v_output，指定上下文、尺寸数组、标量类型转换
  vTensor v_output{
      api::context(),
      output_size.vec(),
      convert_dtype(self.scalar_type()),
  };

  // 返回调用transpose_4d函数的结果，传入self、in_size、out_size、out_dims、v_output参数
  return transpose_4d(self, in_size, out_size, out_dims, v_output);
} // 结束 namespace

// 定义函数 t，用于返回输入张量的转置，若输入张量维度超过 2，则报错
Tensor t(const Tensor& self) {
  // 检查输入张量的维度是否不超过 2，否则抛出错误信息
  TORCH_CHECK(self.dim() <= 2, "t() only supports tensors <= 2 dimensions");
  // 返回张量的转置结果，如果维度小于 2，则在第 0 维进行转置，否则在第 0 和 1 维进行转置
  return transpose(self.detach(), 0, self.dim() < 2 ? 0 : 1);
}

#ifdef USE_VULKAN_API

// 在 Vulkan API 使用情况下，实现 aten 命名空间的 TORCH_LIBRARY_IMPL 函数
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  // 将函数 t 的 Vulkan 版本注册到 m 对象中
  m.impl(TORCH_SELECTIVE_NAME("aten::t"), TORCH_FN(t));
  // 将 transpose 函数的 Vulkan 版本注册到 m 对象中
  m.impl(TORCH_SELECTIVE_NAME("aten::transpose.int"), TORCH_FN(transpose));
}

#endif /* USE_VULKAN_API */

} // 结束 namespace at
} // 结束 namespace native
} // 结束 namespace vulkan
} // 结束 namespace ops
} // 结束 namespace
```