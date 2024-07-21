# `.\pytorch\aten\src\ATen\native\vulkan\ops\Sum.cpp`

```py
// 包含 Vulkan 操作的常见头文件和工具函数头文件
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
// 包含 Torch 库的头文件
#include <torch/library.h>

// 定义 Vulkan 相关操作的命名空间
namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

// 使用 Vulkan 操作的工具函数命名空间
using namespace api::utils;

// 对指定维度进行求和操作
Tensor sum_dim(
    const at::Tensor& self, // 输入张量
    int64_t dim,            // 指定的维度
    bool keepdim,           // 是否保持维度
    const optional<ScalarType> dtype) {  // 可选的输出数据类型
  TORCH_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan sum.dim_IntList supports 1d, 2d, 3d, 4d tensors as input!");

  // 获取全局 Vulkan 上下文
  api::Context* const context = api::context();

  // 将输入张量转换为 Vulkan 张量（如果尚未是）
  const Tensor input = self.is_vulkan() ? self : self.vulkan();
  const vTensor& v_input = convert(input);

  // 创建输出纹理
  std::vector<int64_t> output_size = v_input.sizes();
  uint32_t dim_size = output_size[dim];
  if (keepdim) {
    output_size[dim] = 1;
  } else {
    output_size.erase(output_size.begin() + dim);
  }

  ScalarType type = self.scalar_type();
  if (dtype.has_value()) {
    type = dtype.value();
  }

  // 创建 Vulkan 张量作为输出
  vTensor v_output{
      context,
      output_size,
      convert_dtype(type),
  };

  // 需要确定在命令缓冲区中如何插入内存屏障
  api::PipelineBarrier pipeline_barrier{};

  // 将维度转换到4维范围内
  if (self.dim() < 4) {
    dim += (4 - self.dim());
  }

  // 创建参数缓冲区
  const struct Block final {
    uvec2 dim_info;
    int32_t channel;
  } block{
      {static_cast<uint32_t>(dim), dim_size},
      static_cast<int32_t>(get_dim<Dim4D::Channel>(v_input)),
  };

  api::UniformParamsBuffer params(context, block);

  // 提交计算作业到 Vulkan 上下文中
  context->submit_compute_job(
      // 着色器描述符
      keepdim ? VK_KERNEL(sum_dim_keepdim) : VK_KERNEL(sum_dim),
      // 管线屏障
      pipeline_barrier,
      // 全局工作组大小
      v_output.extents(),
      // 本地工作组大小
      adaptive_work_group_size(v_output.extents()),
      // 栅栏句柄
      VK_NULL_HANDLE,
      // 着色器参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());
  
  // 将 Vulkan 张量转换为常规张量并返回
  return convert(v_output);
}

// 对指定维度列表进行求和操作
Tensor sum_dim_IntList(
    const at::Tensor& self,          // 输入张量
    const OptionalIntArrayRef opt_dim,  // 可选的维度列表
    bool keepdim,                   // 是否保持维度
    const optional<ScalarType> dtype) {  // 可选的输出数据类型
  TORCH_CHECK(
      opt_dim.has_value(),
      "Vulkan sum.dim_IntList without a dim arg is not implemented");

  std::set<int64_t> dims_set;
  if (opt_dim.has_value()) {
    auto dims = opt_dim.value();
    // 对于每个给定的维度进行遍历
    for (const auto& dim : dims) {
      // 在进行归一化之前检查维度，以便向用户报告特定的错误维度值
      TORCH_CHECK(
          dim >= -self.dim() && dim <= self.dim() - 1,
          "Vulkan sum.dim_IntList dimension out of range expected to be in range of [",
          -self.dim(),
          ",",
          self.dim() - 1,
          "], but got ",
          dim);
      
      // 将维度归一化到范围 [0, self.dim() - 1]
      int64_t dim_normalized = utils::normalize(dim, self.dim());
      
      // 检查归一化后的维度是否已经存在于 dims_set 中
      if (dims_set.find(dim_normalized) != dims_set.end()) {
        TORCH_CHECK(
            false,
            "dim ",
            dim_normalized,
            " appears multiple times in the list of dims")
      }
      // 将归一化后的维度添加到 dims_set 中
      dims_set.insert(dim_normalized);
    }
    
    // 将结果初始化为输入张量 self
    Tensor result = self;
    
    // 逆序遍历 dims_set，以便先减少高维度，这样在 keepdim 为 false 时可以正确减少维度
    for (auto it = dims_set.rbegin(); it != dims_set.rend(); ++it) {
      // 对 result 张量按照当前维度 *it 进行求和操作，结果覆盖原 result
      result = sum_dim(result, *it, keepdim, dtype);
    }
    
    // 返回最终结果张量 result
    return result;
  }
  
  // 如果没有指定任何维度，则直接返回输入张量 self
  return self;
}

// 定义一个名为 sum 的函数，计算给定 Tensor 的总和
Tensor sum(const Tensor& self, const std::optional<ScalarType> dtype) {
  // 存储维度的向量
  std::vector<int64_t> dims;
  // 遍历 Tensor 的各个维度
  for (int64_t d = 0; d < self.dim(); d++) {
    // 如果任何维度的元素数量为零，则返回一个零维的 Tensor
    if (self.size(d) == 0) {
      return self.new_zeros({}, at::device(at::kVulkan).dtype(self.dtype()));
    }

    // 将当前维度加入维度向量
    dims.push_back(d);
  }

  // 调用 sum_dim_IntList 函数计算指定维度的总和，并返回结果
  return sum_dim_IntList(self, dims, false, dtype);
}

#ifdef USE_VULKAN_API

// 在 Vulkan 下实现 aten 模块中 sum.dim_IntList 的函数
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::sum.dim_IntList"), TORCH_FN(sum_dim_IntList));
  // 注册在 Vulkan 下实现的 sum 函数
  m.impl(TORCH_SELECTIVE_NAME("aten::sum"), TORCH_FN(sum));
}

#endif /* USE_VULKAN_API */

// 结束命名空间声明
} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
```