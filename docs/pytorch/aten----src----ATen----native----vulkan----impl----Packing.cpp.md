# `.\pytorch\aten\src\ATen\native\vulkan\impl\Packing.cpp`

```
// 获取将 NCHW 格式转换为图像格式的着色器信息
api::ShaderInfo get_nchw_to_image_shader(const vTensor& v_dst) {
    // 如果张量是量化的
    if (v_dst.is_quantized()) {
        // 根据存储类型选择对应的量化类型转换着色器
        switch (v_dst.storage_type()) {
            case api::StorageType::TEXTURE_3D:
                // 根据数据类型选择对应的量化类型转换着色器
                switch (v_dst.dtype()) {
                    case api::ScalarType::QUInt8:
                        return VK_KERNEL(nchw_to_image_uint8);
                    case api::ScalarType::QInt8:
                        return VK_KERNEL(nchw_to_image_int8);
                    case api::ScalarType::QInt32:
                        return VK_KERNEL(nchw_to_image_int32);
                    default:
                        // 抛出异常，当前 Vulkan 不支持的数据类型
                        VK_THROW(
                            "Vulkan quantization currently not supported for dtype ",
                            v_dst.dtype());
                }
            case api::StorageType::TEXTURE_2D:
                // 类似 TEXTURE_3D，但是是二维纹理的情况
                switch (v_dst.dtype()) {
                    case api::ScalarType::QUInt8:
                        return VK_KERNEL(nchw_to_image2d_uint8);
                    case api::ScalarType::QInt8:
                        return VK_KERNEL(nchw_to_image2d_int8);
                    case api::ScalarType::QInt32:
                        return VK_KERNEL(nchw_to_image2d_int32);
                    default:
                        // 抛出异常，当前 Vulkan 不支持的数据类型
                        VK_THROW(
                            "Vulkan quantization currently not supported for dtype ",
                            v_dst.dtype());
                }
            default:
                // 如果未知存储类型，抛出异常
                VK_THROW("No kernel available!");
        }
    }

    // 如果不是量化类型，而是浮点数类型
    if (v_dst.dtype() == api::kFloat) {
        // 根据存储类型选择对应的浮点数类型转换着色器
        switch (v_dst.storage_type()) {
            case api::StorageType::TEXTURE_3D:
                return VK_KERNEL(nchw_to_image);
            case api::StorageType::TEXTURE_2D:
                return VK_KERNEL(nchw_to_image2d);
            default:
                // 抛出异常，当前 Vulkan 不支持的数据类型
                VK_THROW("No kernel available!");
        }
    } else if (v_dst.dtype() == api::kBool) {
        // 如果是布尔类型，只有 TEXTURE_3D 支持
        switch (v_dst.storage_type()) {
            case api::StorageType::TEXTURE_3D:
                return VK_KERNEL(nchw_to_image_bool);
            default:
                // 抛出异常，当前 Vulkan 不支持的数据类型
                VK_THROW("No kernel available!");
        }
    } else {
        // 如果是不支持的数据类型，抛出异常
        VK_THROW("Unsupported dtype!");
    }
}

// 获取将图像格式转换为 NCHW 格式的着色器信息
api::ShaderInfo get_image_to_nchw_shader(const vTensor& v_src) {
    // 如果张量是量化的或者布尔类型
    if (v_src.is_quantized() || v_src.dtype() == api::kBool) {
        // 计算平面大小，高度乘以宽度
        auto plane_size =
            dim_at<Dim4D::Height>(v_src) * dim_at<Dim4D::Width>(v_src);
    # 根据输入的数据源存储类型进行判断和处理
    switch (v_src.storage_type()) {
      # 如果数据源存储类型为 TEXTURE_3D
      case api::StorageType::TEXTURE_3D:
        # 根据数据源的数据类型进行进一步判断
        switch (v_src.dtype()) {
          # 如果数据类型为 QUInt8 或者 QInt8 或者 kBool
          case api::ScalarType::QUInt8:
          case api::ScalarType::QInt8:
          case api::kBool:
            # 如果平面大小能被 4 整除，选择 VK_KERNEL(image_to_nchw_quantized_mul4)，否则选择 VK_KERNEL(image_to_nchw_uint)
            return plane_size % 4 == 0 ? VK_KERNEL(image_to_nchw_quantized_mul4)
                                       : VK_KERNEL(image_to_nchw_uint);
          # 如果数据类型为 QInt32
          case api::ScalarType::QInt32:
            # 返回 VK_KERNEL(image_to_nchw_int32)
            return VK_KERNEL(image_to_nchw_int32);
          # 对于其他未列出的数据类型，抛出异常
          default:
            VK_THROW(
                "Vulkan quantization currently not supported for dtype ",
                v_src.dtype());
        }
      # 如果数据源存储类型不是 TEXTURE_3D，抛出异常
      default:
        VK_THROW("No kernel available!");
      # 如果数据源存储类型为 BUFFER 或者 UNKNOWN，抛出异常
      case api::StorageType::BUFFER:
      case api::StorageType::UNKNOWN:
        VK_THROW("Requested storage type must be a texture type.");
    }
  }

  # 对于数据类型为 kFloat 的情况
  if (v_src.dtype() == api::kFloat) {
    # 根据数据源存储类型进行进一步判断
    switch (v_src.storage_type()) {
      # 如果数据源存储类型为 TEXTURE_3D，返回 VK_KERNEL(image_to_nchw)
      case api::StorageType::TEXTURE_3D:
        return VK_KERNEL(image_to_nchw);
      # 如果数据源存储类型为 TEXTURE_2D，返回 VK_KERNEL(image2d_to_nchw)
      case api::StorageType::TEXTURE_2D:
        return VK_KERNEL(image2d_to_nchw);
      # 对于其他未列出的存储类型，抛出异常
      default:
        VK_THROW("No kernel available!");
    }
  } else {
    # 对于不支持的数据类型，抛出异常
    VK_THROW("Unsupported dtype!");
  }
}

// 结构体定义，描述了从NCHW格式到图像格式转换的参数
struct ToFromTextureParams final {
  api::utils::ivec3 extents; // 三维向量，表示张量的尺寸
  int32_t planeSize; // 整数，表示每个平面的大小
  api::utils::ivec2 channelInfo; // 二维向量，包含通道信息
};

// 将NCHW格式的记录转换为图像格式的操作
void record_nchw_to_image_op(
    api::Context* const context, // API上下文
    api::ShaderInfo& compute_shader, // 计算着色器信息
    api::VulkanBuffer& src_buffer, // 源缓冲区
    vTensor& v_dst, // 目标张量
    api::PipelineBarrier pipeline_barrier, // 管道屏障
    VkFence fence_handle) { // Vulkan屏障句柄

  // 计算全局和局部工作组大小
  api::utils::uvec3 global_size = v_dst.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  // 从张量中获取高度、宽度和通道数
  int32_t height =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Height>(v_dst));
  int32_t width =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Width>(v_dst));
  int32_t channels =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Channel>(v_dst));

  // 计算平面大小和压缩深度
  int32_t plane_size = height * width;
  int32_t c_depth = api::utils::div_up(channels, 4);

  // 创建转换参数块
  ToFromTextureParams block{
      api::utils::make_ivec3(v_dst.extents()),
      plane_size,
      {c_depth, channels},
  };

  // 创建参数缓冲区对象
  api::UniformParamsBuffer params(context, block);
  
  // 提交计算作业到上下文
  context->submit_compute_job(
      // 计算着色器描述符
      compute_shader,
      // 管道屏障
      pipeline_barrier,
      // 全局工作组大小
      global_size,
      // 局部工作组大小
      local_size,
      // 屏障句柄
      fence_handle,
      // 着色器参数
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      src_buffer,
      // 参数缓冲区
      params.buffer());
}

// 将图像格式的记录转换为NCHW格式的操作
bool record_image_to_nchw_op(
    api::Context* const context, // API上下文
    api::ShaderInfo& compute_shader, // 计算着色器信息
    vTensor& v_src, // 源张量
    api::VulkanBuffer& dst_buffer, // 目标缓冲区
    api::PipelineBarrier pipeline_barrier, // 管道屏障
    VkFence fence_handle) { // Vulkan屏障句柄

  // 计算全局和局部工作组大小
  api::utils::uvec3 global_size = v_src.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  // 从张量中获取高度、宽度和通道数
  int32_t height =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Height>(v_src));
  int32_t width =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Width>(v_src));
  int32_t channels =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Channel>(v_src));

  // 计算平面大小和压缩深度
  int32_t plane_size = height * width;
  int32_t c_depth = api::utils::div_up(channels, 4);

  // 创建转换参数块
  ToFromTextureParams block{
      api::utils::make_ivec3(v_src.extents()),
      plane_size,
      {c_depth, channels},
  };

  // 如果源张量是QUInt8、QInt8或者kBool类型
  if (v_src.dtype() == api::ScalarType::QUInt8 ||
      v_src.dtype() == api::ScalarType::QInt8 || v_src.dtype() == api::kBool) {
    // 如果平面大小是4的倍数，使用优化的着色器image_to_nchw_quantized_mul4
    if (plane_size % 4 == 0) {
      global_size.data[0u] = plane_size / 4;
      global_size.data[1u] = 1;
      local_size.data[0u] *= local_size.data[1u];
      local_size.data[1u] = 1;
    }
    // 否则，对于常规的1D缓冲区，设定全局和局部工作组大小
    else {
      uint32_t numel = v_src.numel();
      global_size = {api::utils::div_up(numel, uint32_t(4)), 1u, 1u};
      local_size = {64u, 1u, 1u};
    }
  }
    }
  }

  // 创建一个 UniformParamsBuffer 对象，使用给定的上下文和块参数
  api::UniformParamsBuffer params(context, block);
  // 提交计算任务到上下文中
  return context->submit_compute_job(
      // 计算着色器描述符
      compute_shader,
      // 管线障碍描述
      pipeline_barrier,
      // 全局工作组大小
      global_size,
      // 本地工作组大小
      local_size,
      // 围栏句柄
      fence_handle,
      // 着色器参数
      v_src.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      // 目标缓冲区
      dst_buffer,
      // 参数缓冲区
      params.buffer());
}

void record_nchw_to_buffer_op(
    api::Context* const context,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst,
    api::PipelineBarrier pipeline_barrier,
    VkFence fence_handle) {
  // 计算GPU缓冲区长度
  uint32_t gpu_buf_len = api::utils::safe_downcast<uint32_t>(v_dst.gpu_numel());

  // 设置全局工作组大小和局部工作组大小
  api::utils::uvec3 global_size = {gpu_buf_len, 1u, 1u};
  api::utils::uvec3 local_size = {32u, 1u, 1u};

  // 创建包含CPU缓冲区元数据的Uniform参数缓冲区
  api::UniformParamsBuffer cpu_buffer_metadata(
      context, v_dst.get_cpu_buffer_metadata());

  // 提交计算作业到上下文中
  context->submit_compute_job(
      // 着色器描述符
      VK_KERNEL(buffer_to_buffer),
      // 管道屏障
      pipeline_barrier,
      // 全局工作组大小
      global_size,
      // 局部工作组大小
      local_size,
      // 围栏句柄
      fence_handle,
      // 着色器参数
      v_dst.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_dst.buffer_metadata(),
      src_buffer,
      cpu_buffer_metadata.buffer());
}

bool record_buffer_to_nchw_op(
    api::Context* const context,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier pipeline_barrier,
    VkFence fence_handle) {
  // 计算缓冲区长度
  uint32_t buf_len = api::utils::safe_downcast<uint32_t>(v_src.numel());

  // 设置全局工作组大小和局部工作组大小
  api::utils::uvec3 global_size = {buf_len, 1u, 1u};
  api::utils::uvec3 local_size = {4u, 1u, 1u};

  // 创建包含CPU缓冲区元数据的Uniform参数缓冲区
  api::UniformParamsBuffer cpu_buffer_metadata(
      context, v_src.get_cpu_buffer_metadata());

  // 提交计算作业到上下文中，并返回操作是否成功的布尔值
  return context->submit_compute_job(
      // 着色器描述符
      VK_KERNEL(buffer_to_buffer),
      // 管道屏障
      pipeline_barrier,
      // 全局工作组大小
      global_size,
      // 局部工作组大小
      local_size,
      // 围栏句柄
      fence_handle,
      // 着色器参数
      dst_buffer,
      cpu_buffer_metadata.buffer(),
      v_src.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_src.buffer_metadata());
}

vTensor channel_image_repacking(
    const vTensor& v_input,
    api::GPUMemoryLayout target_layout,
    const api::ShaderInfo& shader_descriptor) {
  // 获取当前上下文
  api::Context* const context = api::context();

  // 创建输出张量，根据目标布局
  vTensor v_output{
      context,
      v_input.sizes(),
      v_input.dtype(),
      v_input.storage_type(),
      target_layout,
  };

  // 创建一个空的管道屏障对象，用于命令缓冲中的内存屏障插入
  api::PipelineBarrier pipeline_barrier{};

  // 着色器假设为4维NCHW格式，用于计算查找坐标
  // 如果输入不是4维，则在前面填充1
  const struct Block final {
    // 定义一个名为 sizes 的 api::utils::ivec4 结构体变量
    api::utils::ivec4 sizes;
  } block{
      // 使用 v_input 的 sizes 创建一个预填充为1的 api::utils::ivec4 结构体
      api::utils::make_ivec4_prepadded1(v_input.sizes()),
  };

  // 使用 block 创建一个 UniformParamsBuffer 对象，使用 context 进行初始化
  api::UniformParamsBuffer params(context, block);

  // 提交计算任务到 context
  context->submit_compute_job(
      // shader 描述符
      // VK_KERNEL(packing_channel_to_height),
      shader_descriptor,
      // 管道屏障
      pipeline_barrier,
      // 全局工作组大小
      v_output.extents(),
      // 局部工作组大小
      adaptive_work_group_size(v_output.extents()),
      // 栅栏句柄
      VK_NULL_HANDLE,
      // shader 参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());

  // 返回计算结果的 v_output
  return v_output;
} // 关闭 at 命名空间

// 将输入张量的通道排列从压缩到高度重新打包
vTensor convert_image_channels_packed_to_height_packed(const vTensor& v_input) {
    // 调用通道图像重新打包函数，将输入张量重新排列为高度打包的格式
    return channel_image_repacking(
        v_input,
        api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
        VK_KERNEL(convert_channels_to_height_packed));
}

// 将输入张量的通道排列从压缩到宽度重新打包
vTensor convert_image_channels_packed_to_width_packed(const vTensor& v_input) {
    // 调用通道图像重新打包函数，将输入张量重新排列为宽度打包的格式
    return channel_image_repacking(
        v_input,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
        VK_KERNEL(convert_channels_to_width_packed));
}

} // 关闭 native 命名空间
} // 关闭 vulkan 命名空间
} // 关闭 packing 命名空间
} // 关闭 namespace 命名空间
```