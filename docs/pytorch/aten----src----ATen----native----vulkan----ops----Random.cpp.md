# `.\pytorch\aten\src\ATen\native\vulkan\ops\Random.cpp`

```
Tensor& normal_(
    Tensor& self,
    const double mean,
    const double std,
    const std::optional<at::Generator> /* not implemented */) {
  // 检查是否为 Vulkan 张量，只有 Vulkan 张量支持原地操作
  TORCH_CHECK(
      self.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  // 检查标准差（std）是否为非负数
  TORCH_CHECK(std >= 0, "Vulkan: Standard deviation (std) can be negative.");

  // 获取 Vulkan API 上下文
  api::Context* const context = api::context();

  // 将传入的普通张量转换为 Vulkan 张量
  vTensor& v_self = convert(self);

  // 定义用于传递给 Vulkan shader 的数据块结构
  const struct Block final {
    uvec3 extents;  // 张量的维度
    float mean;     // 正态分布的均值
    float std;      // 正态分布的标准差
  } block{v_self.extents(), static_cast<float>(mean), static_cast<float>(std)};

  // 创建用于存储参数的 UniformParamsBuffer 对象
  api::UniformParamsBuffer params(context, block);

  // 创建管线屏障对象
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算作业到 Vulkan 设备
  context->submit_compute_job(
      VK_KERNEL(normal_),  // Vulkan shader 的描述符
      pipeline_barrier,    // 管线屏障对象
      v_self.extents(),    // 全局工作组大小
      adaptive_work_group_size(v_self.extents()),  // 本地工作组大小
      VK_NULL_HANDLE,      // 信号量句柄
      v_self.image(        // 张量作为 shader 参数，指定写入内存访问类型
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      params.buffer());    // 参数缓冲区对象

  // 返回修改后的自身 Vulkan 张量
  return self;
}
    // 定义一个浮点型变量 std
    float std;
  } block{v_self.extents(), static_cast<float>(mean), static_cast<float>(std)};
  
  // 使用块的尺寸、均值和标准差创建 UniformParamsBuffer 对象
  api::UniformParamsBuffer params(context, block);
  // 创建一个空的管线屏障对象
  api::PipelineBarrier pipeline_barrier{};
  
  // 提交计算作业到图形上下文
  context->submit_compute_job(
      // 指定 Vulkan 内核函数名为 normal_
      VK_KERNEL(normal_),
      // 使用 pipeline_barrier 作为管线屏障
      pipeline_barrier,
      // 全局工作组大小为 v_self 的尺寸
      v_self.extents(),
      // 根据 v_self 的尺寸动态调整本地工作组大小
      adaptive_work_group_size(v_self.extents()),
      // 使用 VK_NULL_HANDLE 作为 fence 句柄
      VK_NULL_HANDLE,
      // 传递图像作为计算着色器参数，指定写入内存访问类型
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      // 将 params 缓冲区作为计算着色器的参数
      params.buffer());

  // 返回当前对象的引用 self
  return self;
}

// 结束 ops 命名空间

Tensor randn_like(
    const at::Tensor& input_arg,
    const std::optional<c10::ScalarType> /* not implemented */,
    const std::optional<c10::Layout> /* not implemented */,
    const std::optional<c10::Device> /* not implemented */,
    const std::optional<bool> /* not implemented */,
    const std::optional<c10::MemoryFormat> /* not implemented */) {
  // 返回一个与输入大小相同的张量，其中填充了来自均值为0、标准差为1的正态分布的随机数。
  return input_arg.clone().detach().normal_(0.0, 1.0);
}

#ifdef USE_VULKAN_API

// 在 Vulkan API 被使用时，注册 aten 库的 Vulkan 实现
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::uniform_"), TORCH_FN(uniform_));
  m.impl(TORCH_SELECTIVE_NAME("aten::rand_like"), TORCH_FN(rand_like));
  m.impl(TORCH_SELECTIVE_NAME("aten::normal_"), TORCH_FN(normal_));
  m.impl(TORCH_SELECTIVE_NAME("aten::randn_like"), TORCH_FN(randn_like));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

// 结束 at 命名空间
```