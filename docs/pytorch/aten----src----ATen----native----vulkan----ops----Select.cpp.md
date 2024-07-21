# `.\pytorch\aten\src\ATen\native\vulkan\ops\Select.cpp`

```
  /*
  Input tensor: (n, c, h, w)
  Output tensor: (c, h, w)
  Input texture coor: (w, h, texels_per_batch * n + c / 4)[c % 4]
    where texels_per_batch = ceil(number_of_channels / 4)
  Output texture coor: (w, h, c / 4)[c % 4]
  */
  // 定义一个名为`block`的结构体，包含一个名为`batch_info`的成员变量，表示批次信息
  const struct Block final {
    ivec2 batch_info;
  } block{
      // 初始化`batch_info`成员，计算每个组的通道数并向上取整，以及传入的索引
      {static_cast<int32_t>(
           std::ceil(static_cast<float>(v_input_sizes[1]) / 4)),
       static_cast<int32_t>(index)}};

  // 使用`block`创建一个参数缓冲区，用于传递给 Vulkan 计算作业
  api::UniformParamsBuffer params(context, block);
  // 创建一个空的管线屏障对象
  api::PipelineBarrier pipeline_barrier{};

  // 提交 Vulkan 计算作业到上下文中
  context->submit_compute_job(
      // 使用名称为`select_batch_4d`的 Vulkan 内核描述符来执行计算作业
      VK_KERNEL(select_batch_4d),
      // 应用管线屏障
      pipeline_barrier,
      // 设置全局工作组大小为输出张量的尺寸
      v_output.extents(),
      // 自适应的局部工作组大小
      adaptive_work_group_size(v_output.extents()),
      // 使用空闲句柄
      VK_NULL_HANDLE,
      // 设置着色器参数，写入到输出图像中
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      // 传入输入图像用于计算
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 传入参数缓冲区
      params.buffer());

  // 将 Vulkan 张量`v_output`转换为普通张量，并返回结果
  return convert(v_output);
}
    ivec4 depth_info;
  } block{
      {static_cast<int32_t>(v_output.extents().data[0u]),
       static_cast<int32_t>(v_output.extents().data[1u]),
       static_cast<int32_t>(v_output.extents().data[2u]),
       static_cast<int32_t>(index)}};

  // 创建一个UniformParamsBuffer对象，使用上下文和block作为参数
  api::UniformParamsBuffer params(context, block);
  
  // 创建一个空的PipelineBarrier对象
  api::PipelineBarrier pipeline_barrier{};

  // 在计算上下文中提交计算作业，包括：
  // - 使用的着色器描述符为VK_KERNEL(select_depth_3d)
  // - 使用的管线障碍为pipeline_barrier
  // - 全局工作组大小为v_output.extents()
  // - 本地工作组大小为根据v_output.extents()自适应确定的大小
  // - 使用的fence句柄为VK_NULL_HANDLE
  // - 着色器参数包括：
  //   - v_output.image()，写入内存的图像数据，使用COMPUTE阶段和WRITE内存访问类型
  //   - v_input.image()，使用COMPUTE阶段的输入图像数据
  //   - params.buffer()，参数缓冲区
  context->submit_compute_job(
      VK_KERNEL(select_depth_3d),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  // 将v_output转换为函数返回类型并返回
  return convert(v_output);
}

Tensor select_depth_4d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  // 根据输入是否为 Vulkan 张量选择合适的输入张量
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  // 将输入张量转换为 Vulkan 张量
  const vTensor& v_input = convert(input);
  // 获取 Vulkan 张量的尺寸信息
  const IntArrayRef v_input_sizes = v_input.sizes();

  // 创建输出 Vulkan 张量，尺寸为 (n, h, w)，与输入张量的第二和第三维度对应
  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[2], v_input_sizes[3]},
      v_input.dtype(),
  };

  /*
  输入张量格式: (n, c, h, w)
  输出张量格式: (n, h, w)
  输入纹理坐标格式: (w, h, texels_per_batch * n + c / 4)[c % 4]
    其中 texels_per_batch = ceil(number_of_channels / 4)
  输出纹理坐标格式: (w, h, n / 4)[n % 4]
  */
  // 定义用于传输到 GPU 的块信息结构体
  const struct Block final {
    ivec4 depth_info;
  } block{
      {static_cast<int32_t>(v_input_sizes[0]),  // n
       static_cast<int32_t>(
           std::ceil(static_cast<float>(v_input_sizes[1]) / 4)),  // texels_per_batch
       static_cast<int32_t>(index),  // index
       0  // unused
      }};
  // 创建参数缓冲区并填充块数据
  api::UniformParamsBuffer params(context, block);
  // 创建管线屏障对象
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算作业到上下文中
  context->submit_compute_job(
      // 着色器描述符
      VK_KERNEL(select_depth_4d),
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

  // 将输出 Vulkan 张量转换为 Tensor 类型并返回
  return convert(v_output);
}

Tensor select_height_3d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  // 根据输入是否为 Vulkan 张量选择合适的输入张量
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  // 将输入张量转换为 Vulkan 张量
  const vTensor& v_input = convert(input);
  // 获取 Vulkan 张量的尺寸信息
  const IntArrayRef v_input_sizes = v_input.sizes();

  // 创建输出 Vulkan 张量，尺寸为 (n, h)，与输入张量的第二维度对应
  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[2]},
      v_input.dtype(),
  };

  // 输入张量格式为 (c, h, w)
  // 输出张量格式为 (c, w)
  // 在着色器中，输入纹理的坐标格式为 (w, h, c)
  // 在着色器中，输出纹理的坐标格式为 (w, c, 1)
  uint32_t w = v_output.extents().data[0u];
  uint32_t c = v_output.extents().data[1u];
  uint32_t z = 1;
  // 定义用于传输到 GPU 的块信息结构体
  const struct Block final {

    ivec4 height_info;
  } block{
      {static_cast<int32_t>(w),  // w
       static_cast<int32_t>(c),  // c
       static_cast<int32_t>(z),  // z
       0  // unused
      }};

  // 创建参数缓冲区并填充块数据
  api::UniformParamsBuffer params(context, block);

  // 提交计算作业到上下文中
  context->submit_compute_job(
      // 着色器描述符
      VK_KERNEL(select_height_3d),
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

  // 将输出 Vulkan 张量转换为 Tensor 类型并返回
  return convert(v_output);
}
    // 定义一个名为 height_info 的 ivec4 向量，用于存储高度信息
    ivec4 height_info;

  } block{
      // 初始化 block 结构体，包含转换后的整数宽度、通道数、深度、索引
      {static_cast<int32_t>(w),
       static_cast<int32_t>(c),
       static_cast<int32_t>(z),
       static_cast<int32_t>(index)}};

  // 将 c 通道的编码打包进纹理单元中，因此只调用 ceil(c/4) 次以最小化调用和读取
  // 对于最后一个维度，是所选的高度。着色器将根据 block.index 进行直接查找。
  // 定义一个 uvec3 全局工作组大小，包含 w, c 除以 4 后的上取整结果, z 维度
  uvec3 global_workgroup_size{w, api::utils::div_up(c, 4u), z};

  // 创建一个 UniformParamsBuffer 对象 params，使用 context 和 block 初始化
  api::UniformParamsBuffer params(context, block);
  
  // 创建一个 PipelineBarrier 对象 pipeline_barrier
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算作业到 context 中
  context->submit_compute_job(
      // 着色器描述符为 VK_KERNEL(select_height_3d)
      VK_KERNEL(select_height_3d),
      // 使用 pipeline_barrier 作为管线屏障
      pipeline_barrier,
      // 设置全局工作组大小为 global_workgroup_size
      global_workgroup_size,
      // 设置本地工作组大小为根据 global_workgroup_size 自适应调整的大小
      adaptive_work_group_size(global_workgroup_size),
      // 使用 VK_NULL_HANDLE 作为 fence 句柄
      VK_NULL_HANDLE,
      // 着色器参数为输出图像 v_output
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      // 着色器参数为输入图像 v_input
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区为 params 的缓冲区
      params.buffer());

  // 返回将 v_output 转换后的结果
  return convert(v_output);
}



Tensor select_height_4d(const Tensor& input_arg, uint32_t index) {
  // 获取当前运行环境的上下文对象
  api::Context* const context = api::context();

  // 根据输入参数确定使用 Vulkan 或者 Vulkan 对象作为输入张量
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  // 将输入张量转换为 Vulkan 张量对象
  const vTensor& v_input = convert(input);
  // 获取 Vulkan 张量的尺寸信息
  const IntArrayRef v_input_sizes = v_input.sizes();

  // 创建输出 Vulkan 张量对象，尺寸为 (n, c, w)，与输入张量第1、2、4维对应
  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[1], v_input_sizes[3]},
      v_input.dtype(),
  };

  /*
  输入张量：(n, c, h, w)
  输出张量：(n, c, w)
  输入纹理坐标：(w, h, texels_per_batch * n + c / 4)[c % 4]
    其中 texels_per_batch = ceil(number_of_channels / 4)
  输出纹理坐标：(w, c, n / 4)[n % 4]
  */
  // 定义包含高度信息的结构体块，这里的 index 作为高度信息的一部分
  const struct Block final {
    ivec4 height_info;
  } block{
      {static_cast<int32_t>(v_input_sizes[0]),
       static_cast<int32_t>(
           std::ceil(static_cast<float>(v_input_sizes[1]) / 4)),
       static_cast<int32_t>(index),
       0}};

  // 创建统一参数缓冲区对象，用于传递给 Vulkan 计算任务
  api::UniformParamsBuffer params(context, block);
  // 创建管线屏障对象
  api::PipelineBarrier pipeline_barrier{};

  // 提交计算任务到 Vulkan 环境中
  context->submit_compute_job(
      // Vulkan shader 描述符
      VK_KERNEL(select_height_4d),
      // Vulkan 管线屏障
      pipeline_barrier,
      // 全局工作组大小
      v_output.extents(),
      // 局部工作组大小，根据输出张量的尺寸自适应选择
      adaptive_work_group_size(v_output.extents()),
      // 使用默认的 fence 句柄
      VK_NULL_HANDLE,
      // Vulkan shader 参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());

  // 将 Vulkan 张量对象转换为普通张量对象并返回
  return convert(v_output);
}

Tensor select_width_3d(const Tensor& input_arg, uint32_t index) {
  // 获取当前运行环境的上下文对象
  api::Context* const context = api::context();

  // 根据输入参数确定使用 Vulkan 或者 Vulkan 对象作为输入张量
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  // 将输入张量转换为 Vulkan 张量对象
  const vTensor& v_input = convert(input);
  // 获取 Vulkan 张量的尺寸信息
  const IntArrayRef v_input_sizes = v_input.sizes();

  // 创建输出 Vulkan 张量对象，尺寸为 (n, c)，与输入张量第1、2维对应
  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[1]},
      v_input.dtype(),
  };

  const struct Block final {
  // 定义一个ivec4类型的变量width_info，用于存储宽度信息
  ivec4 width_info;
} block{
    // 使用static_cast将输出张量的维度转换为int32_t类型，并赋值给width_info的四个成员变量
    {static_cast<int32_t>(v_output.extents().data[0u]),
     static_cast<int32_t>(v_output.extents().data[1u]),
     static_cast<int32_t>(v_output.extents().data[2u]),
     static_cast<int32_t>(index)}};

// 输入张量是(c, h, w)
// 输出张量是(c, h)
// 在着色器中，输入纹理的坐标为(w, h, c)
// 在着色器中，输出纹理的坐标为(h, c, 1)
uint32_t h = v_output.extents().data[0u];  // 提取输出张量的高度信息
uint32_t c = v_output.extents().data[1u];  // 提取输出张量的通道数信息

// 将c通道编码打包到纹素中，因此我们只调用ceil(c/4)次来最小化调用和读取。
// 对于最后一个维度，它是选择的宽度。着色器将根据block.index进行直接查找。
uvec3 global_workgroup_size{h, api::utils::div_up(c, 4u), 1};

// 使用block中的数据创建UniformParamsBuffer对象params
api::UniformParamsBuffer params(context, block);
// 创建PipelineBarrier对象pipeline_barrier
api::PipelineBarrier pipeline_barrier{};

// 提交计算作业到上下文中
context->submit_compute_job(
    // 着色器描述符
    VK_KERNEL(select_width_3d),
    // 管线屏障
    pipeline_barrier,
    // 全局工作组大小
    global_workgroup_size,
    // 自适应工作组大小
    adaptive_work_group_size(global_workgroup_size),
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

// 返回转换后的v_output
return convert(v_output);
} // 结束命名空间 at

Tensor select_width_4d(const Tensor& input_arg, uint32_t index) {
  // 获取当前运行环境的上下文对象指针
  api::Context* const context = api::context();

  // 根据输入参数选择使用 Vulkan 或者转换后的 Vulkan 张量
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  // 将输入张量转换为 Vulkan 张量对象
  const vTensor& v_input = convert(input);
  // 获取转换后 Vulkan 张量的尺寸信息
  const IntArrayRef v_input_sizes = v_input.sizes();

  // 创建 Vulkan 输出张量对象，尺寸为 (n, c, h)
  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[1], v_input_sizes[2]},
      v_input.dtype(),
  };

  /*
  输入张量：(n, c, h, w)
  输出张量：(n, c, h)
  输入纹理坐标：(w, h, texels_per_batch * n + c / 4)[c % 4]
    其中 texels_per_batch = ceil(number_of_channels / 4)
  输出纹理坐标：(h, c, n / 4)[n % 4]
  */
  // 定义 Vulkan 计算任务的参数块
  const struct Block final {
    ivec4 width_info;
  } block{
      static_cast<int32_t>(v_input_sizes[0]),
      static_cast<int32_t>(std::ceil(static_cast<float>(v_input_sizes[1]) / 4)),
      static_cast<int32_t>(index),
      0};

  // 创建 Vulkan 统一参数缓冲区对象
  api::UniformParamsBuffer params(context, block);
  // 创建 Vulkan 管道屏障对象
  api::PipelineBarrier pipeline_barrier{};

  // 提交 Vulkan 计算作业
  context->submit_compute_job(
      // 着色器描述符
      VK_KERNEL(select_width_4d),
      // 管道屏障
      pipeline_barrier,
      // 全局工作组大小
      v_output.extents(),
      // 局部工作组大小
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

  // 返回转换后的 Vulkan 输出张量
  return convert(v_output);
}

// 根据指定维度和索引选择张量的子集
Tensor select(const Tensor& self, int64_t dim, int64_t index) {
  // 检查张量维度是否为 3 或 4
  TORCH_CHECK(
      self.dim() == 3 || self.dim() == 4,
      "Vulkan select only supports 3d and 4d tensors!");

  // 获取指定维度的大小
  const int64_t size = self.size(dim);

  // 检查索引是否超出范围
  if (index < -size || index >= size) {
    TORCH_CHECK_INDEX(
        false,
        "select(): index ",
        index,
        " out of range for tensor of size ",
        self.sizes(),
        " at dimension ",
        dim);
  }
  // 将负索引转换为正索引
  if (index < 0) {
    index += size;
  }

  // 根据张量维度调用相应的选择函数
  if (self.dim() == 3) {
    if (dim == 0) {
      return select_depth_3d(self, index);
    } else if (dim == 1) {
      return select_height_3d(self, index);
    } else {
      return select_width_3d(self, index);
    }
  } else { // self.dim() == 4
    if (dim == 0) {
      return select_batch_4d(self, index);
    } else if (dim == 1) {
      return select_depth_4d(self, index);
    } else if (dim == 2) {
      return select_height_4d(self, index);
    } else {
      return select_width_4d(self, index);
    }
  }
}

#ifdef USE_VULKAN_API

// 实现 Vulkan API 下的 aten::select.int 函数
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::select.int"), TORCH_FN(select));
}

#endif /* USE_VULKAN_API */

} // namespace at
```