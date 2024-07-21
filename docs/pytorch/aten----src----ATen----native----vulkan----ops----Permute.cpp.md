# `.\pytorch\aten\src\ATen\native\vulkan\ops\Permute.cpp`

```py
// 引入 Vulkan 相关头文件和 Torch 库头文件
#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

// 定义命名空间 at::native::vulkan::ops 内部的匿名命名空间
namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

// 使用 api::utils 命名空间
using namespace api::utils;

// 实现一个函数 permute_4d，用于执行 Vulkan 引擎中的 4 维张量排列操作
Tensor permute_4d(
    const Tensor& input_arg,      // 输入张量
    const uvec4& in_size,         // 输入张量尺寸
    const uvec4& out_size,        // 输出张量尺寸
    const uvec4& out_dims,        // 输出张量维度
    vTensor& v_output             // Vulkan 引擎中的张量
) {
  // 获取当前 Vulkan API 的上下文
  api::Context* const context = api::context();

  // 根据输入张量的类型选择是否使用 Vulkan 引擎
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  // 将输入张量转换为 Vulkan 引擎中的张量对象
  const vTensor& v_self = convert(input);

  // 获取输出通道数和输入通道数
  uint32_t out_channels = out_size.data[1u];
  uint32_t in_channels = in_size.data[1u];

  // 将输出通道数和输入通道数向上对齐到 4 的倍数
  uint32_t out_c_aligned = api::utils::align_up(out_channels, 4u);
  uint32_t in_c_aligned = api::utils::align_up(in_channels, 4u);

  // 定义包含数据块，描述输入和输出张量的尺寸和维度信息
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
      api::utils::make_ivec3(v_output.extents()),
      0,
      api::utils::make_ivec3(v_self.extents()),
      0,
      out_size,
      in_size,
      out_dims,
      {out_c_aligned, in_c_aligned},
  };

  // 创建统一参数缓冲区对象，用于传递给 Vulkan Compute Shader
  api::UniformParamsBuffer params(context, block);
  // 创建管线障碍对象
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算任务到 Vulkan 引擎
  context->submit_compute_job(
      // Vulkan 计算着色器描述符
      VK_KERNEL(permute_4d),
      // 管线障碍对象
      pipeline_barrier,
      // 全局工作组大小
      v_output.extents(),
      // 局部工作组大小
      adaptive_work_group_size(v_output.extents()),
      // 栅栏句柄
      VK_NULL_HANDLE,
      // 计算着色器参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());

  // 将 Vulkan 引擎中的张量对象转换为 Torch 张量对象并返回
  return convert(v_output);
}

// 实现 permute 函数，用于对输入张量进行排列操作
Tensor permute(const Tensor& self, IntArrayRef dims) {
  auto nDims = safe_downcast<uint32_t>(self.dim());
  // 检查维度数量是否与指定的排列维度数量一致
  TORCH_CHECK(
      dims.size() == (size_t)nDims, "number of dims don't match in permute");

  // 初始化输入和输出张量的尺寸信息
  uvec4 in_size{1u, 1u, 1u, 1u}, out_size{1u, 1u, 1u, 1u};
  uvec4 out_dims{0u, 1u, 2u, 3u};

  // 获取输入张量的原始尺寸
  auto oldSizes = self.sizes();
  DimVector newSizes(nDims);
  bool sameDims = true;
  std::vector<bool> seen(nDims);
  // 遍历指定的排列维度，并检查是否有重复的维度
  for (const auto i : c10::irange(nDims)) {
    auto dim = safe_downcast<uint32_t>(maybe_wrap_dim(dims[i], nDims));
    TORCH_CHECK(!seen[dim], "repeated dim in permute");
    seen[dim] = true;
    newSizes[i] = oldSizes[dim];
    if (dim != i) {
      sameDims = false;
    }
    // 将张量通用化为 4 维张量
    in_size.data[(4u - nDims) + i] = self.sizes()[i];
    out_size.data[(4u - nDims) + i] = self.sizes()[dim];
    out_dims.data[(4u - nDims) + i] = dim + (4u - nDims);
  }

  // 如果排列后的维度与原始维度相同，则直接返回输入张量
  if (sameDims) {
    return self;
  }



    # 返回自身对象
    return self;



  IntArrayRef output_sizes(newSizes);
  vTensor v_output{
      api::context(),
      output_sizes.vec(),
      convert_dtype(self.scalar_type()),
  };



  # 创建一个存储新尺寸的数组引用
  IntArrayRef output_sizes(newSizes);
  # 创建一个vTensor对象v_output，用于存储输出张量的上下文、尺寸和数据类型转换信息
  vTensor v_output{
      api::context(),
      output_sizes.vec(),
      convert_dtype(self.scalar_type()),
  };



  return permute_4d(self, in_size, out_size, out_dims, v_output);



  # 调用permute_4d函数，并返回其结果
  return permute_4d(self, in_size, out_size, out_dims, v_output);
#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则编译以下代码块

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  // 在 Torch 库中注册 Vulkan 实现的 permute 函数
  m.impl(TORCH_SELECTIVE_NAME("aten::permute"), TORCH_FN(permute));
}

#endif /* USE_VULKAN_API */
// 结束对 USE_VULKAN_API 宏的条件编译

} // namespace
// 结束匿名命名空间

} // namespace ops
// 结束 ops 命名空间

} // namespace vulkan
// 结束 vulkan 命名空间

} // namespace native
// 结束 native 命名空间

} // namespace at
// 结束 at 命名空间
```