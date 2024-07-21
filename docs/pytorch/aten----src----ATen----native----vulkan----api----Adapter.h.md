# `.\pytorch\aten\src\ATen\native\vulkan\api\Adapter.h`

```
// @pragma once 指示预处理器只包含此头文件一次，用于避免重复包含
#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName 忽略特定的静态代码分析警告

#ifdef USE_VULKAN_API // 如果定义了 USE_VULKAN_API，则编译以下代码块

#include <ATen/native/vulkan/api/vk_api.h> // 包含 Vulkan API 头文件

#include <ATen/native/vulkan/api/Pipeline.h> // 包含 Vulkan 管道相关的头文件
#include <ATen/native/vulkan/api/Shader.h>   // 包含 Vulkan 着色器相关的头文件
#include <ATen/native/vulkan/api/Utils.h>    // 包含 Vulkan 实用工具函数相关的头文件

#include <array>    // 包含标准数组头文件
#include <mutex>    // 包含互斥量头文件
#include <ostream>  // 包含输出流头文件

namespace at {                // 命名空间 at 开始
namespace native {            // 命名空间 native 开始
namespace vulkan {            // 命名空间 vulkan 开始
namespace api {               // 命名空间 api 开始

struct PhysicalDevice final { // 定义 Vulkan 物理设备结构体
  VkPhysicalDevice handle;   // Vulkan 物理设备句柄

  // 从 Vulkan 获取的属性
  VkPhysicalDeviceProperties properties;
  VkPhysicalDeviceMemoryProperties memory_properties;
  std::vector<VkQueueFamilyProperties> queue_families; // 存储队列族属性的向量

  // 元数据
  uint32_t num_compute_queues; // 计算队列数量
  bool has_unified_memory;     // 是否具有统一内存
  bool has_timestamps;         // 是否支持时间戳
  float timestamp_period;      // 时间戳周期

  explicit PhysicalDevice(VkPhysicalDevice); // 物理设备结构体的构造函数声明
};

class DeviceHandle final {  // Vulkan 设备句柄类声明
 public:
  explicit DeviceHandle(VkDevice device); // 显式构造函数声明

  DeviceHandle(const DeviceHandle&) = delete; // 禁用复制构造函数
  DeviceHandle& operator=(const DeviceHandle&) = delete; // 禁用赋值运算符重载

  DeviceHandle(DeviceHandle&&) noexcept; // 移动构造函数声明
  DeviceHandle& operator=(DeviceHandle&&) = delete; // 禁用移动赋值运算符重载

  ~DeviceHandle(); // 析构函数声明

 private:
  VkDevice handle_; // Vulkan 设备句柄私有成员

  friend class Adapter; // 声明 Adapter 类为友元类
};

//
// Vulkan 适配器表示逻辑设备及其所有属性。它管理底层物理设备的所有相关属性、
// 逻辑设备的句柄以及设备可用的计算队列数量。主要负责管理指向 GPU 上逻辑设备
// 对象的 VkDevice 句柄。
//
// 此类主要由 Runtime 类使用，后者对每个 VkInstance 可见的物理设备持有一个
// 适配器实例。在构造时，此类将填充物理设备属性，但直到通过 init_device() 函数
// 明确请求创建逻辑设备时才会创建逻辑设备。
//
// init_device() 将创建逻辑设备并获取其 VkDevice 句柄。它还将创建多个计算队列，
// 直到 Adapter 实例构造时请求的数量为止。
//
// 上下文（代表一个执行线程）将从适配器请求计算队列。适配器将选择一个计算队列
// 分配给上下文，尝试在所有可用队列之间平衡负载。这将允许不同的上下文（通常在
// 单独的线程上执行）并行运行。
//
#define NUM_QUEUE_MUTEXES 4 // 定义计算队列互斥量数量为 4

class Adapter final { // Vulkan 适配器类声明
 public:
  explicit Adapter( // 显式构造函数声明
      VkInstance instance, // VkInstance 实例参数
      PhysicalDevice physical_device, // 物理设备参数
      const uint32_t num_queues);    // 计算队列数量参数

  Adapter(const Adapter&) = delete; // 禁用复制构造函数
  Adapter& operator=(const Adapter&) = delete; // 禁用赋值运算符重载

  Adapter(Adapter&&) = delete; // 禁用移动构造函数
  Adapter& operator=(Adapter&&) = delete; // 禁用移动赋值运算符重载

  ~Adapter() = default; // 默认析构函数声明

  struct Queue { // 队列结构体声明
    uint32_t family_index; // 队列族索引
    uint32_t queue_index;  // 队列索引
    VkQueueFlags capabilities; // 队列能力标志
    // Vulkan队列句柄
    VkQueue handle;
  };

 private:
  // 使用互斥锁来管理队列使用信息，因为可能会被多个线程访问
  std::mutex queue_usage_mutex_;
  // 物理设备信息
  PhysicalDevice physical_device_;
  // 队列管理
  std::vector<Queue> queues_;
  std::vector<uint32_t> queue_usage_;
  // 用于队列的互斥锁数组
  std::array<std::mutex, NUM_QUEUE_MUTEXES> queue_mutexes_;
  // 句柄
  VkInstance instance_;
  DeviceHandle device_;
  // 设备级资源缓存
  ShaderLayoutCache shader_layout_cache_;
  ShaderCache shader_cache_;
  PipelineLayoutCache pipeline_layout_cache_;
  ComputePipelineCache compute_pipeline_cache_;
  // 内存管理
  SamplerCache sampler_cache_;
  MemoryAllocator vma_;

 public:
  // 物理设备元数据

  // 返回物理设备的句柄
  inline VkPhysicalDevice physical_handle() const {
    return physical_device_.handle;
  }

  // 返回设备的句柄
  inline VkDevice device_handle() const {
    return device_.handle_;
  }

  // 返回是否支持统一内存架构
  inline bool has_unified_memory() const {
    return physical_device_.has_unified_memory;
  }

  // 返回计算队列的数量
  inline uint32_t num_compute_queues() const {
    return physical_device_.num_compute_queues;
  }

  // 返回是否支持计算和图形时间戳
  inline bool timestamp_compute_and_graphics() const {
    return physical_device_.has_timestamps;
  }

  // 返回时间戳周期
  inline float timestamp_period() const {
    return physical_device_.timestamp_period;
  }

  // 队列管理

  // 请求一个队列
  Queue request_queue();
  // 归还一个队列
  void return_queue(Queue&);

  // 缓存

  // 返回着色器布局缓存
  inline ShaderLayoutCache& shader_layout_cache() {
    return shader_layout_cache_;
  }

  // 返回着色器缓存
  inline ShaderCache& shader_cache() {
    return shader_cache_;
  }

  // 返回管线布局缓存
  inline PipelineLayoutCache& pipeline_layout_cache() {
    return pipeline_layout_cache_;
  }

  // 返回计算管线缓存
  inline ComputePipelineCache& compute_pipeline_cache() {
    return compute_pipeline_cache_;
  }

  // 内存分配

  // 返回采样器缓存
  inline SamplerCache& sampler_cache() {
    return sampler_cache_;
  }

  // 返回内存分配器
  inline MemoryAllocator& vma() {
    return vma_;
  }

  // 命令缓冲提交

  // 提交单个命令缓冲到指定队列
  void submit_cmd(
      const Queue&,
      VkCommandBuffer,
      VkFence fence = VK_NULL_HANDLE);

  // 提交多个命令缓冲到指定队列
  void submit_cmds(
      const Adapter::Queue&,
      const std::vector<VkCommandBuffer>&,
      VkFence fence = VK_NULL_HANDLE);

  // 其他

  // 返回本地工作组大小
  inline utils::uvec3 local_work_group_size() const {
    return {
        4u,
        4u,
        4u,
    };
  }

  // 返回适配器的字符串表示形式
  std::string stringize() const;
  friend std::ostream& operator<<(std::ostream&, const Adapter&);
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
#endif /* USE_VULKAN_API */
```