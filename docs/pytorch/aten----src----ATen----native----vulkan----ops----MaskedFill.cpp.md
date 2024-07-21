# `.\pytorch\aten\src\ATen\native\vulkan\ops\MaskedFill.cpp`

```
// 引入 Vulkan 相关头文件和 Torch 库
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>
#include <vector>

// 定义 Vulkan 相关操作的命名空间
namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;  // 使用 Vulkan API 中的实用工具命名空间

// 实现在 Vulkan 下进行标量填充操作的函数
Tensor masked_fill_scalar(
    const Tensor& self_arg,   // 输入张量 self
    const Tensor& mask_arg,   // 掩码张量 mask
    const Scalar& value) {    // 要填充的标量值

  utils::is_broadcastable(self_arg, mask_arg);  // 检查 self 和 mask 是否可广播

  // 获取当前 Vulkan 上下文
  api::Context* const context = api::context();

  // 将输入张量 self 转换为 Vulkan 张量
  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();

  // 将掩码张量 mask 转换为 Vulkan 张量，并获取其 Vulkan 格式
  const Tensor mask = mask_arg.is_vulkan() ? mask_arg : mask_arg.vulkan();
  const vTensor& v_mask = convert(mask);

  // 计算输出形状，通过广播 self 和 mask 的形状
  auto in_ndims = safe_downcast<uint32_t>(self_arg.dim());
  auto in_sizes = self_arg.sizes();
  auto mask_sizes = mask_arg.sizes();
  std::vector<int64_t> out_sizes = utils::broadcast_size(self_arg, mask_arg);
  TORCH_INTERNAL_ASSERT(!out_sizes.empty(), "output shape is empty!");

  // 将输出和掩码的形状推广为 4D
  uvec4 generalized_out_sizes{1u, 1u, 1u, 1u},
      generalized_mask_sizes{1u, 1u, 1u, 1u};
  int add_out_ndims = static_cast<int>(4 - out_sizes.size());
  for (int i = 0; (unsigned)i < out_sizes.size(); i++) {
    generalized_out_sizes.data[i + add_out_ndims] = out_sizes[i];
  }
  int add_mask_ndims = static_cast<int>(4 - mask_sizes.size());
  for (int i = 0; (unsigned)i < mask_sizes.size(); i++) {
    generalized_mask_sizes.data[i + add_mask_ndims] = mask_sizes[i];
  }

  auto out_ndims = safe_downcast<uint32_t>(out_sizes.size());

  // 对齐掩码和输出的通道数，使其成为 4 的倍数
  uint32_t mask_c_aligned =
      api::utils::align_up(generalized_mask_sizes.data[1u], 4u);
  uint32_t out_c_aligned =
      api::utils::align_up(generalized_out_sizes.data[1u], 4u);

  // 计算在输出张量中进行重复操作以得到 out_sizes 的重复次数
  auto add_ndims = out_ndims - in_ndims;
  std::vector<int64_t> repeats;
  for (int i = 0; (unsigned)i < out_ndims; i++) {
    if ((unsigned)i < add_ndims || in_sizes[i - add_ndims] == 1) {
      repeats.push_back(out_sizes[i]);
    } else {
      repeats.push_back(1);
    }
  }

  // 对输入张量 self 进行重复操作以生成输出 out_sizes
  at::Tensor out = self.repeat(repeats);
  vTensor& v_out = convert(out);

  // 定义最终的块结构体，包括输出和掩码的维度信息、对齐后的通道信息等
  const struct Block final {
    ivec3 outExtents;
    int32_t fill0;
    ivec3 maskExtents;
    int32_t fill1;
    uvec4 outTensorSize;
    uvec4 maskTensorSize;
    uvec2 alignedChannelInfo;
    // 定义一个浮点类型的变量 value
    float value;
  } block{
      // 使用输出张量 v_out 和掩码张量 v_mask 的维度创建一个 ivec3 类型的结构体
      api::utils::make_ivec3(v_out.extents()),
      // 设置为 0
      0,
      // 使用掩码张量 v_mask 的维度创建一个 ivec3 类型的结构体
      api::utils::make_ivec3(v_mask.extents()),
      // 设置为 0
      0,
      // 传递广义输出大小 generalized_out_sizes
      generalized_out_sizes,
      // 传递广义掩码大小 generalized_mask_sizes
      generalized_mask_sizes,
      // 创建一个包含 out_c_aligned 和 mask_c_aligned 的结构体
      {out_c_aligned, mask_c_aligned},
      // 将 value 转换为 float 类型后传递给结构体
      value.to<float>(),
  };

  // 使用 context 和 block 创建 UniformParamsBuffer 对象 params
  api::UniformParamsBuffer params(context, block);
  // 创建一个空的 PipelineBarrier 对象 pipeline_barrier
  api::PipelineBarrier pipeline_barrier{};

  // masked_fill 的一个可能实现方式是在 mask 上执行重复操作，
  // 生成与输出张量相同形状的广播掩码，然后在 mask 为 True 的位置填充输出的值。
  // 然而，在 mask 上执行重复操作会增加额外的时间和空间开销。
  // 相反，在着色器文件中，我们遍历原始 mask，并在 mask 值为 True 时计算输出张量中相应的广播位置。
  context->submit_compute_job(
      // 使用着色器描述符 VK_KERNEL(masked_fill) 提交计算作业
      VK_KERNEL(masked_fill),
      // 使用空的管线屏障 pipeline_barrier
      pipeline_barrier,
      // 设置全局工作组大小为 v_mask 的尺寸
      v_mask.extents(),
      // 使用自适应工作组大小函数 adaptive_work_group_size 计算本地工作组大小
      adaptive_work_group_size(v_mask.extents()),
      // 使用 VK_NULL_HANDLE 作为 fence handle
      VK_NULL_HANDLE,
      // 设置着色器参数
      v_out.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          // 读写内存访问权限
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_mask.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 使用 params.buffer() 传递参数缓冲区
      params.buffer());

  // 返回转换后的输出张量 v_out
  return convert(v_out);
} // 结束当前命名空间 'at'

} // 结束命名空间 'ops'

} // 结束命名空间 'vulkan'

} // 结束命名空间 'native'

} // 结束命名空间 'at'
```