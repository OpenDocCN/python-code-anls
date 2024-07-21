# `.\pytorch\aten\src\ATen\native\vulkan\ops\Mean.cpp`

```py
// 包含 Vulkan 操作的常用头文件
#include <ATen/native/vulkan/ops/Common.h>
// 包含 Vulkan 操作的工具函数头文件
#include <ATen/native/vulkan/ops/Utils.h>
// 包含 Torch 库的头文件
#include <torch/library.h>

// 进入 at 命名空间
namespace at {
// 进入 native 命名空间
namespace native {
// 进入 vulkan 命名空间
namespace vulkan {
// 进入 ops 命名空间
namespace ops {
// 匿名命名空间，用于内部链接性
namespace {

// 使用 api::utils 命名空间中的所有内容
using namespace api::utils;

// 计算指定维度上的均值
Tensor mean_dim(
    const at::Tensor& self,  // 输入张量
    int64_t dim,             // 指定的维度
    bool keepdim,            // 是否保持维度
    const optional<ScalarType> dtype) {  // 可选的输出数据类型
  // 检查输入张量维度是否在支持的范围内
  TORCH_CHECK(
      self.dim() >= 2 && self.dim() <= 4,
      "Vulkan mean_dim supports 2d, 3d, 4d tensors as input!");
  // 检查指定的维度是否在张量维度范围内
  TORCH_CHECK(
      dim >= -self.dim() && dim < self.dim(),
      "Vulkan mean.dim dimension out of range expected to be in range of [",
      -self.dim(),
      ",",
      self.dim() - 1,
      "], but got ",
      dim);

  // 获取全局 Vulkan 上下文
  api::Context* const context = api::context();

  // 将输入张量转换为 vTensor
  const Tensor input = self.is_vulkan() ? self : self.vulkan();
  const vTensor& v_input = convert(input);

  // 将维度标准化到 [0, self.dim()] 范围内
  dim = utils::normalize(dim, self.dim());

  // 创建输出纹理的尺寸
  std::vector<int64_t> output_size = v_input.sizes();
  uint32_t dim_size = output_size[dim];
  if (keepdim) {
    output_size[dim] = 1;
  } else {
    output_size.erase(output_size.begin() + dim);
  }

  // 确定输出张量的数据类型
  ScalarType type = self.scalar_type();
  if (dtype.has_value()) {
    type = dtype.value();
  }

  // 创建 vTensor 对象来保存输出结果
  vTensor v_output{
      context,
      output_size,
      convert_dtype(type),
  };

  // 用于确定如何在命令缓冲区中插入内存屏障
  api::PipelineBarrier pipeline_barrier{};

  // 如果输入张量维度小于4，则将维度转换为4维范围内
  if (self.dim() < 4) {
    dim += (4 - self.dim());
  }

  // 创建参数缓冲区
  const struct Block final {
    uvec2 dim_info;                  // 维度信息
    int32_t channel;                 // 通道数
  } block{
      {static_cast<uint32_t>(dim), dim_size},  // 维度信息
      static_cast<int32_t>(get_dim<Dim4D::Channel>(v_input)),  // 通道数
  };

  // 创建包含参数的 UniformParamsBuffer 对象
  api::UniformParamsBuffer params(context, block);

  // 提交计算任务到 Vulkan 上下文
  context->submit_compute_job(
      // 计算着色器描述符
      keepdim ? VK_KERNEL(mean_dim_keepdim) : VK_KERNEL(mean_dim),
      // 管线屏障
      pipeline_barrier,
      // 全局工作组大小
      v_output.extents(),
      // 局部工作组大小
      adaptive_work_group_size(v_output.extents()),
      // 围栏句柄
      VK_NULL_HANDLE,
      // 着色器参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());

  // 将 vTensor 转换为 Tensor 并返回
  return convert(v_output);
}

// 处理传入的整数列表维度的均值计算
Tensor mean_dim_IntList(
    const at::Tensor& self,             // 输入张量
    const OptionalIntArrayRef opt_dim,  // 可选的整数列表维度
    bool keepdim,                       // 是否保持维度
    const optional<ScalarType> dtype) { // 可选的输出数据类型
  // 如果未提供维度参数，则抛出错误
  TORCH_CHECK(
      opt_dim.has_value(), "Vulkan mean without a dim arg is not implemented");

  // 创建用于存储唯一维度的集合
  std::set<int64_t> dims_set;

  if (opt_dim.has_value()) {
    auto dims = opt_dim.value();
    // 遍历给定的维度列表 dims
    for (const auto& d : dims) {
      // 检查当前维度 d 是否在有效范围内 [-self.dim(), self.dim()-1]
      TORCH_CHECK(
          d >= -self.dim() && d < self.dim(),
          "Vulkan mean.dim_IntList dimension out of range expected to be in range of [",
          -self.dim(),
          ",",
          self.dim() - 1,
          "], but got ",
          d);
      
      // 对当前维度进行归一化处理，确保在合法范围内
      int64_t dim_normalized = utils::normalize(d, self.dim());

      // 检查归一化后的维度是否已经在 dims_set 中存在
      if (dims_set.find(dim_normalized) != dims_set.end()) {
        // 如果存在，则抛出错误，说明该维度在 dims 中出现了多次
        TORCH_CHECK(
            false,
            "dim ",
            dim_normalized,
            " appears multiple times in the list of dims")
      }
      
      // 将归一化后的维度加入 dims_set 集合中，表示已经处理过
      dims_set.insert(dim_normalized);
    }
    
    // 将输出张量初始化为 self
    Tensor output = self;
    
    // 对 dims_set 中的归一化维度从后向前遍历
    for (auto it = dims_set.rbegin(); it != dims_set.rend(); ++it) {
      // 调用 mean_dim 函数对 output 进行维度均值处理
      output = mean_dim(output, *it, keepdim, dtype);
    }
    
    // 返回处理后的输出张量
    return output;
  }
  
  // 如果没有任何处理发生，直接返回原始的 self 张量
  return self;
#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则编译以下代码块

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  // 在 aten 命名空间下注册 Vulkan 实现的 aten::mean.dim 函数
  m.impl(TORCH_SELECTIVE_NAME("aten::mean.dim"), TORCH_FN(mean_dim_IntList));
}

#endif /* USE_VULKAN_API */
// 结束条件编译指令，用于标识 USE_VULKAN_API 宏的结束

} // namespace
// 结束当前命名空间定义

} // namespace ops
// 结束 ops 命名空间定义

} // namespace vulkan
// 结束 vulkan 命名空间定义

} // namespace native
// 结束 native 命名空间定义

} // namespace at
// 结束 at 命名空间定义
```