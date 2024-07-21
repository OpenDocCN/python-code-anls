# `.\pytorch\aten\src\ATen\native\vulkan\ops\Concat.cpp`

```
// 引入 Vulkan 相关头文件
#include <ATen/native/vulkan/ops/Common.h>
#include <c10/util/irange.h>
#include <torch/library.h>

// 定义命名空间 at::native::vulkan::ops
namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

// 使用 api::utils 命名空间中的内容
using namespace api::utils;

// 匿名命名空间定义，包含辅助函数 normalize_dim
namespace {
// 标准化维度 d，确保在范围 [0, n) 内
inline int64_t normalize_dim(int64_t d, int64_t n) {
  return (d % n + n) % n;
}
} // namespace

// 定义函数 cat_batch，将一组 Vulkan 张量合并为单个张量
Tensor cat_batch(const MaterializedITensorListRef& tensors, vTensor& v_output) {
  // 获取当前 Vulkan 环境上下文
  api::Context* const context = api::context();

  // 源和目标偏移量初始化为零
  uvec3 src_offset{};
  uvec3 dst_offset{};

  // 遍历输入张量列表
  for (const at::Tensor& tensor : tensors) {
    // 将输入张量转换为 Vulkan 张量
    const Tensor self = tensor.is_vulkan() ? tensor : tensor.vulkan();
    const vTensor& v_self = convert(self);

    // 创建 Vulkan 图像拷贝的管线屏障对象
    api::PipelineBarrier pipeline_barrier{};

    // 提交图像拷贝命令，从输入张量拷贝到输出张量
    context->submit_copy<api::VulkanImage, api::VulkanImage>(
        pipeline_barrier,  // 管线屏障
        v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),  // 输入图像
        v_output.image(  // 输出图像
            pipeline_barrier,
            api::PipelineStage::TRANSFER,
            api::MemoryAccessType::WRITE),
        v_self.extents(),  // 拷贝的范围
        src_offset,  // 源偏移量
        dst_offset,  // 目标偏移量
        VK_NULL_HANDLE);  // 信号句柄

    // 更新目标偏移量，增加深度维度的像素数
    dst_offset.data[2u] += v_self.extents().data[2u];
  }

  // 将 Vulkan 张量 v_output 转换为普通张量返回
  return convert(v_output);
}

// 定义函数 cat_feature，合并特征张量
Tensor cat_feature(
    const MaterializedITensorListRef& tensors,
    vTensor& v_output) {
  // 获取当前 Vulkan 环境上下文
  api::Context* const context = api::context();

  // 计算输出张量的通道数总和
  uint32_t ch_total = 0;
  for (const at::Tensor& tensor : tensors) {
    ch_total += get_dim<Dim4D::Channel>(tensor);
  }

  // 当前已经附加的通道数计数器
  uint32_t ch_current = 0;
  for (const at::Tensor& tensor : tensors) {
    // 将输入张量转换为 Vulkan 张量
    const Tensor self = tensor.is_vulkan() ? tensor : tensor.vulkan();
    const vTensor& v_self = convert(self);

    // 计算附加当前输入张量后将修改的通道数像素
    uint32_t start_ch4 = ch_current / 4;
    uint32_t end_ch4 = api::utils::div_up(ch_current + get_dim<Dim4D::Channel>(v_self), 4u);
    uint32_t ch4_range = end_ch4 - start_ch4;
    uint32_t nc4_range = ch4_range * get_dim<Dim4D::Batch>(v_self);

    // 定义结构体 Block，包含用于传递给 UniformParamsBuffer 的参数
    const struct Block final {
      ivec3 outExtents;
      int32_t fill0;
      ivec3 inExtents;
      int32_t fill1;
      uvec2 outChInfo;
      uvec2 inChInfo;
      uvec4 appendedChInfo;
    } block{
        api::utils::make_ivec3(v_output.extents()),
        0,
        api::utils::make_ivec3(v_self.extents()),
        0,
        {
            ch_total,
            api::utils::div_up(ch_total, 4u),
        },
        {
            get_dim<Dim4D::Channel>(v_self),
            api::utils::align_up(get_dim<Dim4D::Channel>(v_self), 4u),
        },
        {
            ch_current,
            start_ch4,
            ch4_range,
            0u,
        },
    };

    // 创建 UniformParamsBuffer 对象，传递 context 和 block 参数
    api::UniformParamsBuffer params(context, block);
    // 创建一个空的管道屏障对象
    api::PipelineBarrier pipeline_barrier{};

    // 使用上下文对象提交计算作业，包括以下参数：
    // - 使用 VK_KERNEL(cat_feature) 激活的着色器描述符
    // - pipeline_barrier 管道屏障对象，用于同步数据访问
    // - 全局工作组大小，从 v_output 中获取宽度和高度，以及 nc4_range
    // - 本地工作组大小，通过 adaptive_work_group_size 函数动态确定
    // - VK_NULL_HANDLE 表示无关联的围栏句柄
    // - 使用 v_output.image() 方法定义的着色器参数，包含读写内存访问权限
    // - 使用 v_self.image() 方法定义的着色器参数，仅包含读内存访问权限
    // - params.buffer() 返回的参数缓冲区
    context->submit_compute_job(
        VK_KERNEL(cat_feature),
        pipeline_barrier,
        {
            get_dim<Dim4D::Width>(v_output),
            get_dim<Dim4D::Height>(v_output),
            nc4_range,
        },
        adaptive_work_group_size(v_self.extents()),
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
        v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());

    // 更新 ch_current 变量，增加 v_self 的通道维度
    ch_current += get_dim<Dim4D::Channel>(v_self);
  }

  // 返回通过 convert(v_output) 转换后的结果
  return convert(v_output);
Tensor cat_feature_mult4ch(
    const MaterializedITensorListRef& tensors,
    vTensor& v_output) {
  // 获取当前运行环境的上下文
  api::Context* const context = api::context();

  // 初始化变量，记录前面所有通道的深度大小和通道间隔
  int64_t depth_size_allprior = 0;
  int64_t ch_interval = 0;
  // 遍历输入的张量列表，累加通道间隔
  for (const at::Tensor& tensor : tensors) {
    ch_interval += get_dim<Dim4D::Channel>(tensor);
  }
  // 计算每个张量所包含的深度间隔，假设每个通道有四个子通道
  const int64_t depth_interval = ch_interval / 4;

  // 源偏移和目标偏移的初始化
  uvec3 src_offset{};
  uvec3 dst_offset{};

  // 遍历张量列表
  for (const at::Tensor& tensor_arg : tensors) {
    // 根据张量的类型选择合适的张量对象
    const Tensor tensor =
        tensor_arg.is_vulkan() ? tensor_arg : tensor_arg.vulkan();
    // 将张量转换为 Vulkan 张量对象
    const vTensor& v_self = convert(tensor);

    // 计算当前张量的深度切片数量
    const uint32_t depth_slice =
        safe_downcast<uint32_t>(get_dim<Dim4D::Channel>(tensor) / 4);

    // 设置复制操作的范围（在宽度和高度上）
    uvec3 copy_extents{
        v_self.extents().data[0u], v_self.extents().data[1u], depth_slice};

    // 对当前张量的每个批次进行处理
    for (const auto b : c10::irange(get_dim<Dim4D::Batch>(tensor))) {
      // 计算源偏移和目标偏移
      src_offset.data[2u] = safe_downcast<uint32_t>(depth_slice * b);
      dst_offset.data[2u] =
          depth_size_allprior + safe_downcast<uint32_t>(depth_interval * b);

      // 创建管线屏障对象
      api::PipelineBarrier pipeline_barrier{};

      // 提交从源 Vulkan 图像到目标 Vulkan 图像的复制操作
      context->submit_copy<api::VulkanImage, api::VulkanImage>(
          // 管线屏障
          pipeline_barrier,
          // 图像对象
          v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
          v_output.image(
              pipeline_barrier,
              api::PipelineStage::TRANSFER,
              api::MemoryAccessType::WRITE),
          // 复制细节
          copy_extents,
          src_offset,
          dst_offset,
          // 等待处理的句柄
          VK_NULL_HANDLE);
    }

    // 更新已处理的深度大小
    depth_size_allprior += depth_slice;
  }

  // 将输出 Vulkan 张量对象转换为 Tensor 类型并返回
  return convert(v_output);
}

Tensor cat_width(const MaterializedITensorListRef& tensors, vTensor& v_output) {
  // 获取当前运行环境的上下文
  api::Context* const context = api::context();

  // 源偏移和目标偏移的初始化
  uvec3 src_offset{};
  uvec3 dst_offset{};

  // 遍历输入的张量列表
  for (const at::Tensor& tensor : tensors) {
    // 根据张量的类型选择合适的张量对象
    const Tensor self = tensor.is_vulkan() ? tensor : tensor.vulkan();
    // 将张量转换为 Vulkan 张量对象
    const vTensor& v_self = convert(self);

    // 创建管线屏障对象
    api::PipelineBarrier pipeline_barrier{};

    // 提交从源 Vulkan 图像到目标 Vulkan 图像的复制操作
    context->submit_copy<api::VulkanImage, api::VulkanImage>(
        // 管线屏障
        pipeline_barrier,
        // 图像对象
        v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::TRANSFER,
            api::MemoryAccessType::WRITE),
        // 复制细节
        v_self.extents(),
        src_offset,
        dst_offset,
        // 等待处理的句柄
        VK_NULL_HANDLE);

    // 按照宽度增加目标偏移
    dst_offset.data[0u] += v_self.extents().data[0u];
  }

  // 将输出 Vulkan 张量对象转换为 Tensor 类型并返回
  return convert(v_output);
}

Tensor cat_height(
    const MaterializedITensorListRef& tensors,
    vTensor& v_output) {
  // 获取当前运行环境的上下文
  api::Context* const context = api::context();

  // 源偏移和目标偏移的初始化
  uvec3 src_offset{};
  uvec3 dst_offset{};

  // 遍历输入的张量列表
  for (const at::Tensor& tensor : tensors) {
    // 检查 tensor 是否已经是 Vulkan 类型，如果不是，则转换为 Vulkan 类型
    const Tensor self = tensor.is_vulkan() ? tensor : tensor.vulkan();
    
    // 将 Vulkan 类型的 Tensor 转换为 vTensor 类型
    const vTensor& v_self = convert(self);

    // 创建一个空的 pipeline_barrier 对象，用于提交管线屏障操作
    api::PipelineBarrier pipeline_barrier{};

    // 在指定的上下文中提交图像拷贝操作，从 v_self 到 v_output
    context->submit_copy<api::VulkanImage, api::VulkanImage>(
        // 提交的管线屏障对象
        pipeline_barrier,
        // 源图像
        v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
        // 目标图像
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::TRANSFER,
            api::MemoryAccessType::WRITE),
        // 拷贝的详细信息
        v_self.extents(),   // 拷贝区域的大小
        src_offset,         // 源图像的偏移量
        dst_offset,         // 目标图像的偏移量
        // 使用 VK_NULL_HANDLE 表示不使用围栏（fence）
        VK_NULL_HANDLE);

    // 将目标图像的偏移量增加高度（在第二维度上）
    dst_offset.data[1u] += v_self.extents().data[1u];
  }

  // 将 v_output 转换为相应类型并返回
  return convert(v_output);
} // 结束 at 命名空间

Tensor cat(const at::ITensorListRef& tensors, const int64_t in_dim) {
  // 检查张量列表是否为空，至少需要一个张量
  TORCH_CHECK(!tensors.empty(), "Vulkan cat expects at least one tensor");
  // 将输入的张量列表实例化为向量
  auto materialized = tensors.materialize();
  // 内部断言，确保实例化后的张量列表不为空
  TORCH_INTERNAL_ASSERT(!materialized.empty(), "Accessing empty array");
  // 获取第一个张量作为参考张量
  const at::Tensor& tensor = materialized[0];
  // 安全地将张量维度转换为 uint32_t
  auto ndim = safe_downcast<uint32_t>(tensor.dim());
  // 根据输入的维度值和张量维度数归一化处理得到实际维度
  const int64_t dim = normalize_dim(in_dim, ndim);
  // 初始化拼接维度的总大小
  int64_t cat_dim_size = 0;
  // 标志位，用于指示是否所有输入张量的通道数均为 4 的倍数
  bool is_mult4ch = true;

  // 遍历实例化后的张量列表
  for (const at::Tensor& t : materialized) {
    // 内部断言，确保张量维度不超过 4
    TORCH_INTERNAL_ASSERT(
        t.dim() <= 4,
        "Vulkan cat expects inputs to have at most 4 dimensions, but got ",
        t.dim(),
        "d");

    // 检查是否当前张量的通道数不是 4 的倍数
    if (ndim < 3 || get_dim<Dim4D::Channel>(t) % 4 != 0) {
      is_mult4ch = false;
    }

    // 检查除了拼接维度外的其他维度是否匹配参考张量的对应维度大小
    for (const auto d : c10::irange(ndim)) {
      if (d == dim) {
        continue;
      }
      // 内部断言，确保张量在除了拼接维度外的其他维度上大小与参考张量相同
      TORCH_INTERNAL_ASSERT(
          t.size(d) == tensor.size(d),
          "Vulkan cat inputs must have matching sizes except concatenated dimension");
    }
    // 累加拼接维度的总大小
    cat_dim_size += t.size(dim);
  }

  // 获取参考张量的大小向量
  auto result_size = tensor.sizes().vec();
  // 内部断言，确保结果大小向量不为空
  TORCH_INTERNAL_ASSERT(!result_size.empty(), "Accessing empty array");
  // 更新结果大小向量中拼接维度的大小
  result_size[dim] = cat_dim_size;

  // 创建 Vulkan 张量 v_output，使用结果大小向量和参考张量的数据类型
  vTensor v_output{
      api::context(), result_size, convert_dtype(tensor.scalar_type())};

  // 根据拼接维度的位置选择不同的拼接方法
  if (dim == ndim - 1) {
    return cat_width(materialized, v_output);
  }
  if (dim == ndim - 2) {
    return cat_height(materialized, v_output);
  } else if (dim == ndim - 3) {
    // 如果所有输入张量的通道数均为 4 的倍数，则选择多通道特征拼接方法
    if (is_mult4ch) {
      return cat_feature_mult4ch(materialized, v_output);
    }
    // 否则选择普通特征拼接方法
    return cat_feature(materialized, v_output);
  }
  // 默认情况下选择批次拼接方法
  return cat_batch(materialized, v_output);
}

#ifdef USE_VULKAN_API

// Vulkan 实现的 aten::cat 方法注册
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::cat"), TORCH_FN(cat));
}

#endif /* USE_VULKAN_API */

} // 结束 ops 命名空间
} // 结束 vulkan 命名空间
} // 结束 native 命名空间
} // 结束 at 命名空间
```