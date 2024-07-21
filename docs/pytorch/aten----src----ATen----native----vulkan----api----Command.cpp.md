# `.\pytorch\aten\src\ATen\native\vulkan\api\Command.cpp`

```py
// 包含 Vulkan API 头文件
#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Command.h>

// 包含互斥量头文件
#include <mutex>

// ATen 命名空间开始
namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// CommandBuffer 类定义
//

// 构造函数，初始化 CommandBuffer 对象
CommandBuffer::CommandBuffer(
    VkCommandBuffer handle,  // Vulkan 命令缓冲区句柄
    const VkCommandBufferUsageFlags flags)  // 命令缓冲区使用标志
    : handle_(handle),  // 初始化 Vulkan 命令缓冲区句柄
      flags_(flags),    // 初始化命令缓冲区使用标志
      state_(CommandBuffer::State::NEW),  // 初始化命令缓冲区状态为 NEW
      bound_{} {}  // bound_ 对象初始化为空

// 移动构造函数，移动构造 CommandBuffer 对象
CommandBuffer::CommandBuffer(CommandBuffer&& other) noexcept
    : handle_(other.handle_),  // 移动句柄
      flags_(other.flags_),    // 移动使用标志
      state_(CommandBuffer::State::INVALID),  // 设置状态为 INVALID
      bound_(other.bound_) {  // 移动 bound_ 对象
  other.handle_ = VK_NULL_HANDLE;  // 设置原对象句柄为空
  other.bound_.reset();  // 重置原对象的 bound_
}

// 移动赋值运算符，移动赋值 CommandBuffer 对象
CommandBuffer& CommandBuffer::operator=(CommandBuffer&& other) noexcept {
  handle_ = other.handle_;  // 移动句柄
  flags_ = other.flags_;    // 移动使用标志
  state_ = other.state_;    // 移动状态

  bound_ = other.bound_;    // 移动 bound_ 对象

  other.handle_ = VK_NULL_HANDLE;  // 设置原对象句柄为空
  other.bound_.reset();  // 重置原对象的 bound_
  other.state_ = CommandBuffer::State::INVALID;  // 设置原对象状态为 INVALID

  return *this;  // 返回移动后的对象
}

// 开始记录命令到命令缓冲区
void CommandBuffer::begin() {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::NEW,  // 检查命令缓冲区状态是否为 NEW
      "Vulkan CommandBuffer: called begin() on a command buffer whose state "
      "is not NEW.");

  const VkCommandBufferBeginInfo begin_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,  // 结构类型
      nullptr,                                      // 扩展信息
      flags_,                                       // 命令缓冲区使用标志
      nullptr,                                      // 渲染器命令缓冲区信息
  };

  VK_CHECK(vkBeginCommandBuffer(handle_, &begin_info));  // 调用 Vulkan API 开始命令缓冲区记录
  state_ = CommandBuffer::State::RECORDING;  // 更新状态为 RECORDING
}

// 结束记录命令到命令缓冲区
void CommandBuffer::end() {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::RECORDING ||  // 检查命令缓冲区状态是否为 RECORDING 或 SUBMITTED
      state_ == CommandBuffer::State::SUBMITTED,
      "Vulkan CommandBuffer: called end() on a command buffer whose state "
      "is not RECORDING or SUBMITTED.");

  if (state_ == CommandBuffer::State::RECORDING) {
    VK_CHECK(vkEndCommandBuffer(handle_));  // 调用 Vulkan API 结束命令缓冲区记录
  }
  state_ = CommandBuffer::State::READY;  // 更新状态为 READY
}

// 绑定管线到命令缓冲区
void CommandBuffer::bind_pipeline(
    VkPipeline pipeline,                          // Vulkan 管线句柄
    VkPipelineLayout pipeline_layout,              // Vulkan 管线布局句柄
    const utils::uvec3 local_workgroup_size) {     // 本地工作组大小
  VK_CHECK_COND(
      state_ == CommandBuffer::State::RECORDING,  // 检查命令缓冲区状态是否为 RECORDING
      "Vulkan CommandBuffer: called bind_pipeline() on a command buffer whose state "
      "is not RECORDING.");

  if (pipeline != bound_.pipeline) {  // 如果当前绑定的管线与要绑定的管线不同
    vkCmdBindPipeline(handle_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);  // 绑定管线到命令缓冲区

    bound_.pipeline = pipeline;  // 更新绑定的管线
  }

  bound_.pipeline_layout = pipeline_layout;         // 更新绑定的管线布局
  bound_.local_workgroup_size = local_workgroup_size;  // 更新本地工作组大小

  state_ = CommandBuffer::State::PIPELINE_BOUND;  // 更新状态为 PIPELINE_BOUND
}

// 绑定描述符集到命令缓冲区
void CommandBuffer::bind_descriptors(VkDescriptorSet descriptors) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::PIPELINE_BOUND,  // 检查命令缓冲区状态是否为 PIPELINE_BOUND
      "Vulkan CommandBuffer: called bind_descriptors() on a command buffer whose state "
      "is not PIPELINE_BOUND.");

  if (descriptors != bound_.descriptors) {
    vkCmdBindDescriptorSets(
        handle_, // commandBuffer，指定要绑定描述符集的命令缓冲区对象
        VK_PIPELINE_BIND_POINT_COMPUTE, // pipelineBindPoint，指定管线绑定点为计算管线
        bound_.pipeline_layout, // layout，使用的管线布局对象
        0u, // firstSet，描述符集的起始索引
        1u, // descriptorSetCount，要绑定的描述符集数量
        &descriptors, // pDescriptorSets，指向要绑定的描述符集数组的指针
        0u, // dynamicOffsetCount，动态偏移量的数量
        nullptr); // pDynamicOffsets，指向动态偏移量数组的指针（本例中为 nullptr 表示无动态偏移量）
  }

  bound_.descriptors = descriptors; // 将 descriptors 赋值给 bound_ 对象的 descriptors 成员变量

  state_ = CommandBuffer::State::DESCRIPTORS_BOUND; // 更新命令缓冲区的状态为描述符已绑定状态
}

// 向命令缓冲区插入管线障碍
void CommandBuffer::insert_barrier(PipelineBarrier& pipeline_barrier) {
  // 检查命令缓冲区的状态是否为DESCRIPTORS_BOUND或RECORDING，否则抛出错误信息
  VK_CHECK_COND(
      state_ == CommandBuffer::State::DESCRIPTORS_BOUND ||
          state_ == CommandBuffer::State::RECORDING,
      "Vulkan CommandBuffer: called insert_barrier() on a command buffer whose state "
      "is not DESCRIPTORS_BOUND or RECORDING.");

  // 如果存在管线障碍对象
  if (pipeline_barrier) {
    // 清空缓冲器障碍句柄列表（如果不为空）
    if (!pipeline_barrier.buffer_barrier_handles.empty()) {
      pipeline_barrier.buffer_barrier_handles.clear();
    }
    // 将每个缓冲器内存障碍的句柄添加到缓冲器障碍句柄列表中
    for (const api::BufferMemoryBarrier& memory_barrier :
         pipeline_barrier.buffers) {
      pipeline_barrier.buffer_barrier_handles.push_back(memory_barrier.handle);
    }

    // 清空图像障碍句柄列表（如果不为空）
    if (!pipeline_barrier.image_barrier_handles.empty()) {
      pipeline_barrier.image_barrier_handles.clear();
    }
    // 将每个图像内存障碍的句柄添加到图像障碍句柄列表中
    for (const api::ImageMemoryBarrier& memory_barrier :
         pipeline_barrier.images) {
      pipeline_barrier.image_barrier_handles.push_back(memory_barrier.handle);
    }

    // 执行管线障碍命令
    vkCmdPipelineBarrier(
        handle_, // commandBuffer
        pipeline_barrier.stage.src, // srcStageMask
        pipeline_barrier.stage.dst, // dstStageMask
        0u, // dependencyFlags
        0u, // memoryBarrierCount
        nullptr, // pMemoryBarriers
        pipeline_barrier.buffers.size(), // bufferMemoryBarrierCount
        !pipeline_barrier.buffers.empty()
            ? pipeline_barrier.buffer_barrier_handles.data()
            : nullptr, // pMemoryBarriers
        pipeline_barrier.images.size(), // imageMemoryBarrierCount
        !pipeline_barrier.images.empty()
            ? pipeline_barrier.image_barrier_handles.data()
            : nullptr); // pImageMemoryBarriers
  }

  // 设置命令缓冲区的状态为BARRIERS_INSERTED
  state_ = CommandBuffer::State::BARRIERS_INSERTED;
}

// 执行调度命令
void CommandBuffer::dispatch(const utils::uvec3& global_workgroup_size) {
  // 检查命令缓冲区的状态是否为BARRIERS_INSERTED，否则抛出错误信息
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called dispatch() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  // 执行调度命令
  vkCmdDispatch(
      handle_,
      utils::div_up(
          global_workgroup_size.data[0u], bound_.local_workgroup_size.data[0u]),
      utils::div_up(
          global_workgroup_size.data[1u], bound_.local_workgroup_size.data[1u]),
      utils::div_up(
          global_workgroup_size.data[2u],
          bound_.local_workgroup_size.data[2u]));

  // 设置命令缓冲区的状态为RECORDING
  state_ = CommandBuffer::State::RECORDING;
}

// 将缓冲器数据复制到另一个缓冲器
void CommandBuffer::copy_buffer_to_buffer(
    const api::VulkanBuffer& source,
    const api::VulkanBuffer& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    // 检查命令缓冲区的状态是否为BARRIERS_INSERTED，如果不是则输出错误信息
    VK_CHECK_COND(
        state_ == CommandBuffer::State::BARRIERS_INSERTED,
        "Vulkan CommandBuffer: called copy_buffer_to_buffer() on a command buffer whose state "
        "is not BARRIERS_INSERTED.");

    // 定义VkBufferCopy结构体，指定复制的源偏移量、目标偏移量和复制的大小
    const VkBufferCopy copy_details{
        src_offset.data[0u], // srcOffset 源偏移量
        dst_offset.data[0u], // dstOffset 目标偏移量
        copy_range.data[0u], // size 复制的大小
    };

    // 使用Vulkan命令vkCmdCopyBuffer将数据从源缓冲区复制到目标缓冲区
    vkCmdCopyBuffer(
        handle_, source.handle(), destination.handle(), 1u, &copy_details);

    // 将命令缓冲区的状态设置为RECORDING，表示现在正在录制命令
    state_ = CommandBuffer::State::RECORDING;
void CommandBuffer::copy_texture_to_texture(
    const api::VulkanImage& source,                        // 源图像对象
    const api::VulkanImage& destination,                   // 目标图像对象
    const api::utils::uvec3& copy_range,                   // 拷贝范围
    const api::utils::uvec3& src_offset,                   // 源偏移量
    const api::utils::uvec3& dst_offset) {                 // 目标偏移量
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,   // 检查命令缓冲状态是否为BARRIERS_INSERTED
      "Vulkan CommandBuffer: called copy_texture_to_texture() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");                        // 若状态不符合要求则输出错误信息

  const VkImageSubresourceLayers src_subresource_layers{    // 源图像子资源层描述
      VK_IMAGE_ASPECT_COLOR_BIT,                           // 图像方面掩码为颜色位
      0u,                                                   // MIP级别为0
      0u,                                                   // 基本数组层为0
      1u,                                                   // 层数为1
  };

  const VkImageSubresourceLayers dst_subresource_layers{    // 目标图像子资源层描述，与源图像一致
      VK_IMAGE_ASPECT_COLOR_BIT,                           // 图像方面掩码为颜色位
      0u,                                                   // MIP级别为0
      0u,                                                   // 基本数组层为0
      1u,                                                   // 层数为1
  };

  const VkImageCopy copy_details{                          // 图像拷贝详细信息
      src_subresource_layers,                              // 源图像子资源层
      create_offset3d(src_offset),                         // 创建源偏移量的3D结构体
      dst_subresource_layers,                              // 目标图像子资源层
      create_offset3d(dst_offset),                         // 创建目标偏移量的3D结构体
      create_extent3d(copy_range),                         // 创建拷贝范围的3D结构体
  };

  vkCmdCopyImage(                                          // 调用Vulkan命令拷贝图像
      handle_,                                            // 命令缓冲句柄
      source.handle(),                                    // 源图像句柄
      source.layout(),                                    // 源图像布局
      destination.handle(),                               // 目标图像句柄
      destination.layout(),                               // 目标图像布局
      1u,                                                 // 拷贝操作数量为1
      &copy_details);                                     // 拷贝详细信息数组

  state_ = CommandBuffer::State::RECORDING;                // 更新命令缓冲状态为RECORDING
}

void CommandBuffer::copy_texture_to_buffer(
    const api::VulkanImage& source,                        // 源图像对象
    const api::VulkanBuffer& destination,                  // 目标缓冲对象
    const api::utils::uvec3& copy_range,                   // 拷贝范围
    const api::utils::uvec3& src_offset,                   // 源偏移量
    const api::utils::uvec3& dst_offset) {                 // 目标偏移量
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,   // 检查命令缓冲状态是否为BARRIERS_INSERTED
      "Vulkan CommandBuffer: called copy_texture_to_buffer() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");                        // 若状态不符合要求则输出错误信息

  const VkImageSubresourceLayers src_subresource_layers{    // 源图像子资源层描述
      VK_IMAGE_ASPECT_COLOR_BIT,                           // 图像方面掩码为颜色位
      0u,                                                   // MIP级别为0
      0u,                                                   // 基本数组层为0
      1u,                                                   // 层数为1
  };

  const VkBufferImageCopy copy_details{                    // 缓冲到图像拷贝详细信息
      dst_offset.data[0u],                                // 缓冲偏移量
      dst_offset.data[1u],                                // 缓冲行长度
      dst_offset.data[2u],                                // 缓冲图像高度
      src_subresource_layers,                             // 图像子资源层
      create_offset3d(src_offset),                        // 创建源偏移量的3D结构体
      create_extent3d(copy_range),                        // 创建拷贝范围的3D结构体
  };

  vkCmdCopyImageToBuffer(                                 // 调用Vulkan命令拷贝图像到缓冲
      handle_,                                            // 命令缓冲句柄
      source.handle(),                                    // 源图像句柄
      source.layout(),                                    // 源图像布局
      destination.handle(),                               // 目标缓冲句柄
      1u,                                                 // 拷贝操作数量为1
      &copy_details);                                     // 拷贝详细信息数组

  state_ = CommandBuffer::State::RECORDING;                // 更新命令缓冲状态为RECORDING
}
    const api::utils::uvec3& dst_offset) {

// 定义函数的参数：目标图像的偏移量，是一个引用类型的常量引用。

  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called copy_buffer_to_texture() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

// 检查命令缓冲区的状态是否为BARRIERS_INSERTED，如果不是，则抛出异常信息。

  const VkImageSubresourceLayers dst_subresource_layers{
      VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
      0u, // mipLevel
      0u, // baseArrayLayer
      1u, // layerCount
  };

// 定义目标图像子资源的层级信息，包括颜色方面、第0层级、第0数组层、1个层级。

  const VkBufferImageCopy copy_details{
      src_offset.data[0u], // bufferOffset
      src_offset.data[1u], // bufferRowLength
      src_offset.data[2u], // bufferImageHeight
      dst_subresource_layers, // imageSubresource
      create_offset3d(dst_offset), // imageOffset
      create_extent3d(copy_range), // imageExtent
  };

// 定义复制操作的详细信息，包括从缓冲区到图像的偏移量、行长度、图像高度、目标图像子资源层级、目标图像偏移量、复制范围。

  vkCmdCopyBufferToImage(
      handle_,
      source.handle(),
      destination.handle(),
      destination.layout(),
      1u,
      &copy_details);

// 调用Vulkan命令来执行从缓冲区到图像的复制操作，使用之前定义的复制细节。

  state_ = CommandBuffer::State::RECORDING;

// 设置命令缓冲区的状态为RECORDING，表示复制操作完成后进入记录状态。
}

void CommandBuffer::write_timestamp(VkQueryPool querypool, const uint32_t idx) const {
  // 检查命令缓冲区状态是否为RECORDING，如果不是，则输出错误信息
  VK_CHECK_COND(
      state_ == CommandBuffer::State::RECORDING,
      "Vulkan CommandBuffer: called write_timestamp() on a command buffer whose state "
      "is not RECORDING.");

  // 在命令缓冲区中写入时间戳
  vkCmdWriteTimestamp(
      handle_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, querypool, idx);
}

void CommandBuffer::reset_querypool(
    VkQueryPool querypool,
    const uint32_t first_idx,
    const uint32_t count) const {
  // 检查命令缓冲区状态是否为RECORDING，如果不是，则输出错误信息
  VK_CHECK_COND(
      state_ == CommandBuffer::State::RECORDING,
      "Vulkan CommandBuffer: called reset_querypool() on a command buffer whose state "
      "is not RECORDING.");

  // 重置查询池中指定索引范围内的查询
  vkCmdResetQueryPool(handle_, querypool, first_idx, count);
}

VkCommandBuffer CommandBuffer::get_submit_handle(const bool final_use) {
  // 检查命令缓冲区状态是否为READY，如果不是，则输出错误信息
  VK_CHECK_COND(
      state_ == CommandBuffer::State::READY,
      "Vulkan CommandBuffer: called begin() on a command buffer whose state "
      "is not READY.");

  // 获取命令缓冲区的提交句柄
  VkCommandBuffer handle = handle_;

  // 如果命令缓冲区不可重用或者是最终使用，使其无效化
  if (!is_reusable() || final_use) {
    invalidate();
  }
  // 将命令缓冲区状态设置为SUBMITTED
  state_ = CommandBuffer::State::SUBMITTED;

  return handle;
}

//
// CommandPool
//

CommandPool::CommandPool(
    VkDevice device,
    const uint32_t queue_family_idx,
    const CommandPoolConfig& config)
    : device_(device),
      queue_family_idx_(queue_family_idx),
      pool_(VK_NULL_HANDLE),
      config_(config),
      mutex_{},
      buffers_{},
      in_use_(0u) {
  // 创建命令池的配置信息
  const VkCommandPoolCreateInfo create_info{
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      nullptr,
      VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      queue_family_idx_,
  };

  // 创建Vulkan命令池对象
  VK_CHECK(vkCreateCommandPool(device_, &create_info, nullptr, &pool_));

  // 预先分配一些命令缓冲区
  allocate_new_batch(config_.cmdPoolInitialSize);
}

CommandPool::~CommandPool() {
  // 如果命令池句柄为空，直接返回
  if (VK_NULL_HANDLE == pool_) {
    return;
  }
  // 销毁Vulkan命令池对象
  vkDestroyCommandPool(device_, pool_, nullptr);
}

CommandBuffer CommandPool::get_new_cmd(bool reusable) {
  // 使用互斥锁保护，获取新的命令缓冲区
  std::lock_guard<std::mutex> lock(mutex_);

  // 如果有可用的命令缓冲区，无需操作
  allocate_new_batch(config_.cmdPoolBatchSize);

  // 获取当前可用的命令缓冲区句柄
  VkCommandBuffer handle = buffers_[in_use_];

  // 根据是否可重用设置命令缓冲区使用标志
  VkCommandBufferUsageFlags cmd_flags = 0u;
  if (!reusable) {
    cmd_flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  }

  // 增加正在使用的命令缓冲区计数
  in_use_++;
  // 返回新的命令缓冲区对象
  return CommandBuffer(handle, cmd_flags);
}

void CommandPool::flush() {
  // 使用互斥锁保护，重置命令池中所有命令缓冲区
  std::lock_guard<std::mutex> lock(mutex_);
  VK_CHECK(vkResetCommandPool(device_, pool_, 0u));
  // 重置正在使用的命令缓冲区计数
  in_use_ = 0u;
}

void CommandPool::allocate_new_batch(const uint32_t count) {
  // 如果仍有可用的命令缓冲区，无需操作
  if (in_use_ < buffers_.size()) {
    return;
  }
  
  // 调整缓冲区向量的大小，以容纳新增的命令缓冲区
  buffers_.resize(buffers_.size() + count);
  
  // 创建命令缓冲区分配信息结构体
  const VkCommandBufferAllocateInfo allocate_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, // 结构体类型
      nullptr, // 指向扩展信息的指针，这里为空
      pool_, // 命令池对象
      VK_COMMAND_BUFFER_LEVEL_PRIMARY, // 命令缓冲区的级别，主要级别
      count, // 要分配的命令缓冲区数量
  };

  // 分配命令缓冲区到指定的设备上，并将其存储在缓冲区向量中
  VK_CHECK(vkAllocateCommandBuffers(
      device_, &allocate_info, buffers_.data() + in_use_));
}
// 结束 at 命名空间
} // namespace at
// 结束 native 命名空间
} // namespace native
// 结束 vulkan 命名空间
} // namespace vulkan
// 结束 api 命名空间
} // namespace api
```