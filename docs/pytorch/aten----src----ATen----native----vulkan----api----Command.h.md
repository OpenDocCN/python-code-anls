# `.\pytorch\aten\src\ATen\native\vulkan\api\Command.h`

```py
#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/vk_api.h>

#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/Pipeline.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Shader.h>
#include <ATen/native/vulkan/api/Utils.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

// Vulkan命令缓冲区类，用于管理Vulkan命令缓冲区对象
class CommandBuffer final {
 public:
  // 显式构造函数，初始化命令缓冲区对象和使用标志
  explicit CommandBuffer(VkCommandBuffer, const VkCommandBufferUsageFlags);

  // 禁用拷贝构造函数和赋值运算符
  CommandBuffer(const CommandBuffer&) = delete;
  CommandBuffer& operator=(const CommandBuffer&) = delete;

  // 移动构造函数和移动赋值运算符
  CommandBuffer(CommandBuffer&&) noexcept;
  CommandBuffer& operator=(CommandBuffer&&) noexcept;

  // 默认析构函数
  ~CommandBuffer() = default;

  // 命令缓冲区生命周期的状态枚举
  enum State {
    INVALID,            // 表示命令缓冲区已被移动
    NEW,                // 在构造函数中设置
    RECORDING,          // 在调用begin()、dispatch()和copy_*_to_*()期间设置
    PIPELINE_BOUND,     // 在调用bind_pipeline()期间设置
    DESCRIPTORS_BOUND,  // 在调用bind_descriptors()期间设置
    BARRIERS_INSERTED,  // 在调用insert_barrier()期间设置
    READY,              // 在调用end()期间设置
    SUBMITTED           // 在调用get_submit_handle()期间设置
  };

  // 用于描述当前命令缓冲区绑定状态的结构体
  struct Bound {
    VkPipeline pipeline;                // 当前绑定的管线对象
    VkPipelineLayout pipeline_layout;   // 当前绑定的管线布局对象
    utils::uvec3 local_workgroup_size;  // 当前绑定的本地工作组大小
    VkDescriptorSet descriptors;        // 当前绑定的描述符集对象

    // 默认构造函数，初始化成员变量为默认值
    explicit Bound()
        : pipeline{VK_NULL_HANDLE},
          pipeline_layout{VK_NULL_HANDLE},
          local_workgroup_size{0u, 0u, 0u},
          descriptors{VK_NULL_HANDLE} {}

    // 重置方法，将所有成员变量重置为默认值
    inline void reset() {
      pipeline = VK_NULL_HANDLE;
      pipeline_layout = VK_NULL_HANDLE;
      local_workgroup_size = {0u, 0u, 0u};
      descriptors = VK_NULL_HANDLE;
    }
  };

 private:
  VkCommandBuffer handle_;               // Vulkan命令缓冲区句柄
  VkCommandBufferUsageFlags flags_;      // 命令缓冲区使用标志
  State state_;                          // 当前命令缓冲区的状态
  Bound bound_;                          // 当前命令缓冲区的绑定状态

 public:
  // 判断命令缓冲区是否可重用的方法
  inline bool is_reusable() {
    return !(flags_ & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
  }

  // 使命令缓冲区无效的方法
  inline void invalidate() {
    handle_ = VK_NULL_HANDLE;
    // 重置命令缓冲区的状态，清除其当前的绑定状态和内容
    bound_.reset();
  }

  // 开始记录命令到当前命令缓冲区
  void begin();

  // 结束当前命令缓冲区的记录
  void end();

  // 绑定指定的管线、管线布局和视口范围到当前命令缓冲区
  void bind_pipeline(VkPipeline, VkPipelineLayout, const utils::uvec3);

  // 绑定指定的描述符集到当前命令缓冲区
  void bind_descriptors(VkDescriptorSet);

  // 插入指定的流水线屏障到当前命令缓冲区
  void insert_barrier(PipelineBarrier& pipeline_barrier);

  // 分发指定大小的工作组到当前命令缓冲区
  void dispatch(const utils::uvec3&);

  // 在命令缓冲区中执行从一个缓冲区到另一个缓冲区的数据复制操作
  void copy_buffer_to_buffer(
      const api::VulkanBuffer&,
      const api::VulkanBuffer&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      const api::utils::uvec3&);

  // 在命令缓冲区中执行从一个纹理到另一个纹理的数据复制操作
  void copy_texture_to_texture(
      const api::VulkanImage&,
      const api::VulkanImage&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      const api::utils::uvec3&);

  // 在命令缓冲区中执行从一个纹理到一个缓冲区的数据复制操作
  void copy_texture_to_buffer(
      const api::VulkanImage&,
      const api::VulkanBuffer&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      const api::utils::uvec3&);

  // 在命令缓冲区中执行从一个缓冲区到一个纹理的数据复制操作
  void copy_buffer_to_texture(
      const api::VulkanBuffer&,
      const api::VulkanImage&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      const api::utils::uvec3&);

  // 在指定的查询池中写入时间戳数据
  void write_timestamp(VkQueryPool, const uint32_t) const;

  // 重置指定查询池中的查询范围，将其查询编号重置为指定的初始值
  void reset_querypool(VkQueryPool, const uint32_t, const uint32_t) const;

  // 获取用于提交的命令缓冲区句柄，可选择在最终使用后销毁
  VkCommandBuffer get_submit_handle(const bool final_use = false);

  // 将命令缓冲区对象转换为布尔值，表示是否存在有效的 Vulkan 句柄
  inline operator bool() const {
    return VK_NULL_HANDLE != handle_;
  }
};

// 结构体定义：CommandPoolConfig，用于配置命令池的初始大小和批处理大小
struct CommandPoolConfig final {
  uint32_t cmdPoolInitialSize;   // 命令池的初始大小
  uint32_t cmdPoolBatchSize;     // 命令池的批处理大小
};

// 类定义：CommandPool，用于管理Vulkan命令池
class CommandPool final {
 public:
  // 构造函数：初始化CommandPool对象
  explicit CommandPool(VkDevice, const uint32_t, const CommandPoolConfig&);

  // 禁用复制构造函数和赋值运算符
  CommandPool(const CommandPool&) = delete;
  CommandPool& operator=(const CommandPool&) = delete;

  // 禁用移动构造函数和赋值运算符
  CommandPool(CommandPool&&) = delete;
  CommandPool& operator=(CommandPool&&) = delete;

  // 析构函数：释放CommandPool对象
  ~CommandPool();

 private:
  VkDevice device_;               // Vulkan设备对象
  uint32_t queue_family_idx_;     // 命令队列的家族索引
  VkCommandPool pool_;            // Vulkan命令池对象
  CommandPoolConfig config_;      // 命令池的配置参数
  std::mutex mutex_;              // 互斥锁，用于线程安全操作
  std::vector<VkCommandBuffer> buffers_;  // 存储Vulkan命令缓冲区的容器
  size_t in_use_;                 // 当前正在使用的命令缓冲区数量

 public:
  // 获取一个新的命令缓冲区对象
  CommandBuffer get_new_cmd(bool reusable = false);

  // 刷新命令池，重置所有命令缓冲区并释放资源
  void flush();

 private:
  // 分配新的命令缓冲区批次
  void allocate_new_batch(const uint32_t);
};

// 结束命名空间：api
} // namespace api

// 结束命名空间：vulkan
} // namespace vulkan

// 结束命名空间：native
} // namespace native

// 结束命名空间：at
} // namespace at

// 结束条件编译指令：USE_VULKAN_API
#endif /* USE_VULKAN_API */
```