# `.\pytorch\aten\src\ATen\native\vulkan\ops\Slice.cpp`

```
// 包含 ATen 库中相关头文件
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/vulkan/ops/Common.h>
// 包含 Torch 库中相关头文件
#include <torch/library.h>

// 定义在命名空间 at::native::vulkan::ops 中的匿名命名空间
namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

// 使用 api::utils 命名空间
using namespace api::utils;

// 定义一个函数 slice_4d，用于对四维张量进行切片操作
Tensor slice_4d(
    const Tensor& input_arg,    // 输入张量引用
    const int64_t dim,          // 切片维度
    const int64_t start,        // 切片起始位置
    const int64_t end,          // 切片结束位置
    const int64_t step,         // 切片步长
    const uvec4& in_tsize,      // 输入张量大小
    const uvec4& out_tsize,     // 输出张量大小
    vTensor& v_output           // Vulkan 张量输出
) {
  // 获取当前 Vulkan 上下文
  api::Context* const context = api::context();

  // 如果输入张量是 Vulkan 张量，则使用它；否则将其转换为 Vulkan 张量
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_self = convert(input);

  // 获取输出通道数和输入通道数，并向上对齐到4的倍数
  uint32_t out_channels = out_tsize.data[1u];
  uint32_t in_channels = in_tsize.data[1u];
  uint32_t out_c_aligned = api::utils::align_up(out_channels, 4u);
  uint32_t in_c_aligned = api::utils::align_up(in_channels, 4u);

  // 定义一个包含各种参数的结构体 Block
  const struct Block final {
    ivec3 size;               // 输出纹理大小
    int32_t fill_0;           // 占位符
    ivec3 isize;              // 输入纹理大小
    int32_t fill_1;           // 占位符
    uvec4 tensor_size;        // 输出张量大小
    uvec4 itensor_size;       // 输入张量大小
    uvec4 args;               // 输入参数（维度、起始、结束、步长）
    uvec2 c_info;             // 通道数对齐到4
  } block{
      api::utils::make_ivec3(v_output.extents()),       // 使用 Vulkan 张量的尺寸
      0,
      api::utils::make_ivec3(v_self.extents()),         // 使用输入 Vulkan 张量的尺寸
      0,
      out_tsize,
      in_tsize,
      {safe_downcast<uint32_t>(dim),                    // 安全转换维度为 uint32_t
       safe_downcast<uint32_t>(start),                  // 安全转换起始位置为 uint32_t
       safe_downcast<uint32_t>(end),                    // 安全转换结束位置为 uint32_t
       safe_downcast<uint32_t>(step)},                  // 安全转换步长为 uint32_t
      {out_c_aligned, in_c_aligned},
  };

  // 创建 UniformParamsBuffer 对象，用于传递参数给 Vulkan 着色器
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算作业到 Vulkan 上下文中
  context->submit_compute_job(
      VK_KERNEL(slice_4d),                               // Vulkan 着色器描述符
      pipeline_barrier,                                  // 管道屏障
      v_output.extents(),                                // 全局工作组大小
      adaptive_work_group_size(v_output.extents()),       // 自适应工作组大小
      VK_NULL_HANDLE,                                    // 着色器参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),                 // 输出 Vulkan 图像
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),  // 输入 Vulkan 图像
      params.buffer());                                 // 参数缓冲区

  // 将 Vulkan 张量转换为 Tensor 类型并返回
  return convert(v_output);
}

// 定义一个函数 slice_width，用于对宽度进行切片操作
Tensor slice_width(
    const Tensor& input_arg,    // 输入张量引用
    const int64_t start,        // 切片起始位置
    const int64_t end,          // 切片结束位置
    const int64_t step,         // 切片步长
    vTensor& v_output           // Vulkan 张量输出
) {
  // 获取当前 Vulkan 上下文
  api::Context* const context = api::context();

  // 如果输入张量是 Vulkan 张量，则使用它；否则将其转换为 Vulkan 张量
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_self = convert(input);

  // 定义源偏移和目标偏移的 uvec3 结构
  uvec3 src_offset{};
  uvec3 dst_offset{};

  // 如果步长为1，则设置源偏移的第一个维度为起始位置
  if (step == 1) {
    src_offset.data[0u] = start;

    // 设置复制的范围大小为（end - start）x输入张量的宽度x输入张量的高度
    uvec3 copy_extents{
        safe_downcast<uint32_t>(end - start),
        v_self.extents().data[1u],
        v_self.extents().data[2u]};

    // 创建管道屏障对象
    api::PipelineBarrier pipeline_barrier{};
    context->submit_copy<api::VulkanImage, api::VulkanImage>(
        // pipeline barrier
        pipeline_barrier,
        // 获取当前图像的资源，并设置传输管线阶段
        v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
        // 获取输出图像的资源，并设置传输管线阶段以及写入内存访问类型
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::TRANSFER,
            api::MemoryAccessType::WRITE),
        // 拷贝的详细信息：拷贝的范围
        copy_extents,
        // 源图像的偏移量
        src_offset,
        // 目标图像的偏移量
        dst_offset,
        // 使用空的栅栏句柄
        VK_NULL_HANDLE);
  } else {
    // 设置拷贝的范围为 {1, 当前图像的高度, 当前图像的深度}
    uvec3 copy_extents{
        1u, v_self.extents().data[1u], v_self.extents().data[2u]};

    // 获取当前图像的最大宽度
    const auto x_max = v_self.extents().data[0u];

    // 对于给定的起始和结束索引，以及步长进行迭代
    for (int64_t x = start, x_new = 0; x < end; x += step, ++x_new) {
      // 如果当前索引超出了当前图像的宽度范围，则跳过
      if (x >= x_max) { // out of range
        continue;
      }

      // 设置源图像偏移量的 x 分量
      src_offset.data[0u] = x;
      // 设置目标图像偏移量的 x 分量
      dst_offset.data[0u] = x_new;

      // 创建一个空的管线障碍对象
      api::PipelineBarrier pipeline_barrier{};

      // 提交图像拷贝任务，包括：
      context->submit_copy<api::VulkanImage, api::VulkanImage>(
          // pipeline barrier
          pipeline_barrier,
          // 获取当前图像的资源，并设置传输管线阶段
          v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
          // 获取输出图像的资源，并设置传输管线阶段以及写入内存访问类型
          v_output.image(
              pipeline_barrier,
              api::PipelineStage::TRANSFER,
              api::MemoryAccessType::WRITE),
          // 拷贝的详细信息：拷贝的范围
          copy_extents,
          // 源图像的偏移量
          src_offset,
          // 目标图像的偏移量
          dst_offset,
          // 使用空的栅栏句柄
          VK_NULL_HANDLE);
    }
  }

  // 转换并返回输出图像
  return convert(v_output);
// 定义函数 slice_height，接收输入张量、起始、结束、步长和目标 Vulkan 张量作为参数，返回张量类型
Tensor slice_height(
    const Tensor& input_arg,                // 输入张量的常量引用
    const int64_t start,                    // 切片起始位置
    const int64_t end,                      // 切片结束位置
    const int64_t step,                     // 切片步长
    vTensor& v_output) {                    // 目标 Vulkan 张量的引用作为输出

  api::Context* const context = api::context();   // 获取 API 上下文对象指针

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();   // 根据输入张量是否为 Vulkan 张量选择性地复制到本地变量
  const vTensor& v_self = convert(input);   // 将输入张量转换为 Vulkan 张量对象

  uvec3 src_offset{};   // 定义源偏移量
  uvec3 dst_offset{};   // 定义目标偏移量

  if (step == 1) {
    src_offset.data[1u] = start;   // 设置源偏移量的第二个维度为起始位置

    uvec3 copy_extents{
        v_self.extents().data[0u],                          // 复制的宽度
        safe_downcast<uint32_t>(end - start),               // 复制的高度
        v_self.extents().data[2u]};                         // 复制的深度

    api::PipelineBarrier pipeline_barrier{};   // 创建管线屏障对象

    context->submit_copy<api::VulkanImage, api::VulkanImage>(
        // 提交复制任务到 Vulkan 图像之间，包括管线屏障和内存访问权限
        pipeline_barrier,
        // 源图像
        v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
        // 目标图像
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::TRANSFER,
            api::MemoryAccessType::WRITE),
        // 复制细节
        copy_extents,
        src_offset,
        dst_offset,
        // 回调句柄
        VK_NULL_HANDLE);
  } else {
    uvec3 copy_extents{
        v_self.extents().data[0u], 1u, v_self.extents().data[2u]};   // 复制的宽度、高度、深度

    const auto y_max = v_self.extents().data[1u];   // 获取张量的最大高度
    for (int64_t y = start, y_new = 0; y < end; y += step, ++y_new) {
      if (y >= y_max) { // 超出范围
        continue;
      }
      src_offset.data[1u] = y;   // 设置源偏移量的第二个维度为当前高度
      dst_offset.data[1u] = y_new;   // 设置目标偏移量的第二个维度为新高度

      api::PipelineBarrier pipeline_barrier{};   // 创建管线屏障对象

      context->submit_copy<api::VulkanImage, api::VulkanImage>(
          // 提交复制任务到 Vulkan 图像之间，包括管线屏障和内存访问权限
          pipeline_barrier,
          // 源图像
          v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
          // 目标图像
          v_output.image(
              pipeline_barrier,
              api::PipelineStage::TRANSFER,
              api::MemoryAccessType::WRITE),
          // 复制细节
          copy_extents,
          src_offset,
          dst_offset,
          // 回调句柄
          VK_NULL_HANDLE);
    }
  }

  return convert(v_output);   // 返回转换后的 Vulkan 张量对象
}
  // 获取指定维度上的结束值
  end_val = newSizes[dim];
}

// 计算切片后的长度
auto len = end_val - start_val;
newSizes[dim] = (len + step - 1) / step; // 向上取整

// 将尺寸泛化为4维张量
uvec4 in_tsize{1u, 1u, 1u, 1u}, out_tsize{1u, 1u, 1u, 1u};
for (const auto i : c10::irange(nDims)) {
  // 将输入张量的尺寸填充到泛化后的输入张量尺寸中
  in_tsize.data[(4u - nDims) + i] = self.sizes()[i];
  // 将新计算的尺寸填充到泛化后的输出张量尺寸中
  out_tsize.data[(4u - nDims) + i] = newSizes[i];
}
dim += 4 - nDims;

// 创建输出尺寸的数组引用
IntArrayRef output_sizes(newSizes);
// 创建一个新的 vTensor 对象，表示切片后的输出张量
vTensor v_output{
    api::context(), output_sizes.vec(), convert_dtype(self.scalar_type())};

// 根据维度调用不同的切片函数
if (dim == 3) {
  slice_width(self, start_val, end_val, step, v_output);
} else if (dim == 2) {
  slice_height(self, start_val, end_val, step, v_output);
} else {
  slice_4d(
      self, dim, start_val, end_val, step, in_tsize, out_tsize, v_output);
}

// 将 v_output 转换为相应的结果类型
auto result = convert(v_output);
// 将结果的命名信息从原始张量传播到结果张量
namedinference::propagate_names(result, self);
// 返回切片后的结果
return result;
} // 关闭所有的命名空间定义

#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则执行以下内容

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
    // 在 aten 库中注册 Vulkan 实现的 slice.Tensor 方法，使用 slice 函数
    m.impl(TORCH_SELECTIVE_NAME("aten::slice.Tensor"), TORCH_FN(slice));
}

#endif /* USE_VULKAN_API */

} // 关闭 ops 命名空间
} // 关闭 vulkan 命名空间
} // 关闭 native 命名空间
} // 关闭 at 命名空间
```