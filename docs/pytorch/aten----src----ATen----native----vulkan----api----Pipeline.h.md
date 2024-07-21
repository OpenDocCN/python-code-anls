# `.\pytorch\aten\src\ATen\native\vulkan\api\Pipeline.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName
// 忽略 lint 工具中指定的名为 facebook-hte-BadMemberName 的警告

#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则编译以下内容

#include <ATen/native/vulkan/api/vk_api.h>
// 引入 Vulkan API 头文件

#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Shader.h>
// 引入 Vulkan API 中的资源和着色器头文件

#include <mutex>
// 引入互斥锁相关头文件

#include <unordered_map>
// 引入无序映射相关头文件

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct PipelineBarrier final {
  // 定义 PipelineBarrier 结构体
  struct Stages final {
    VkPipelineStageFlags src;
    VkPipelineStageFlags dst;
    // 定义内部结构 Stages，包含 src 和 dst 两个成员变量
  } stage;

  std::vector<BufferMemoryBarrier> buffers;
  std::vector<ImageMemoryBarrier> images;
  std::vector<VkBufferMemoryBarrier> buffer_barrier_handles;
  std::vector<VkImageMemoryBarrier> image_barrier_handles;
  // 定义多个存储内存和图像屏障的容器

  inline operator bool() const {
    return (0u != stage.src) || (0u != stage.dst) || !buffers.empty() ||
        !images.empty();
    // 转换操作符，用于检查 PipelineBarrier 对象是否有效
  }
};

using PipelineStageFlags = uint8_t;
// 使用 PipelineStageFlags 别名表示 uint8_t 类型的管线阶段标志

enum PipelineStage : PipelineStageFlags {
  NO_STAGE = 0u << 0u,
  COMPUTE = 1u << 0u,
  HOST = 1u << 1u,
  TRANSFER = 1u << 2u,
};
// 定义枚举 PipelineStage，表示不同的管线阶段标志

VkAccessFlags vk_access(const PipelineStageFlags, const MemoryAccessFlags);
// 声明函数 vk_access，根据管线阶段标志和内存访问标志返回 Vulkan 访问标志

VkPipelineStageFlags vk_stage(const PipelineStageFlags);
// 声明函数 vk_stage，根据管线阶段标志返回 Vulkan 管线阶段标志

VkImageLayout vk_layout(const PipelineStageFlags, const MemoryAccessFlags);
// 声明函数 vk_layout，根据管线阶段标志和内存访问标志返回 Vulkan 图像布局

class PipelineLayout final {
  // 定义 PipelineLayout 类
 public:
  explicit PipelineLayout(VkDevice, VkDescriptorSetLayout);
  // 显式构造函数，接受 VkDevice 和 VkDescriptorSetLayout 参数

  PipelineLayout(const PipelineLayout&) = delete;
  PipelineLayout& operator=(const PipelineLayout&) = delete;
  // 删除复制构造函数和复制赋值运算符

  PipelineLayout(PipelineLayout&&) noexcept;
  PipelineLayout& operator=(PipelineLayout&&) = delete;
  // 移动构造函数和移动赋值运算符声明为删除状态

  ~PipelineLayout();
  // 析构函数

 private:
  VkDevice device_;
  VkPipelineLayout handle_;
  // 私有成员变量，包含 Vulkan 设备和管线布局句柄

 public:
  VkPipelineLayout handle() const {
    return handle_;
  }
  // 返回管线布局句柄的公共成员函数

  // 定义自定义交换函数，因为该类不允许移动赋值操作，此交换函数将在哈希映射中使用
  friend void swap(PipelineLayout& lhs, PipelineLayout& rhs) noexcept;
};

class ComputePipeline final {
  // 定义 ComputePipeline 类
 public:
  struct Descriptor final {
    VkPipelineLayout pipeline_layout;
    VkShaderModule shader_module;
    utils::uvec3 local_work_group;
    // 定义 Descriptor 结构体，包含管线布局、着色器模块和本地工作组大小
  };

  explicit ComputePipeline(
      VkDevice device,
      const Descriptor& descriptor,
      VkPipelineCache pipeline_cache);
  // 显式构造函数，接受 Vulkan 设备、描述符和管线缓存参数

  ComputePipeline(const ComputePipeline&) = delete;
  ComputePipeline& operator=(const ComputePipeline&) = delete;
  // 删除复制构造函数和复制赋值运算符

  ComputePipeline(ComputePipeline&&) noexcept;
  ComputePipeline& operator=(ComputePipeline&&) = delete;
  // 移动构造函数和移动赋值运算符声明为删除状态

  ~ComputePipeline();
  // 析构函数

 private:
  VkDevice device_;
  VkPipeline handle_;
  // 私有成员变量，包含 Vulkan 设备和管线句柄

 public:
  inline VkPipeline handle() const {
    return handle_;
  }
  // 返回管线句柄的公共成员函数

  // 定义自定义交换函数，因为该类不允许移动赋值操作，此交换函数将在哈希映射中使用
  friend void swap(ComputePipeline& lhs, ComputePipeline& rhs) noexcept;
};
// Vulkan API 的实现命名空间
namespace api {
namespace vulkan {
namespace native {
namespace at {

// 使用最终类修饰符定义 PipelineLayoutCache 类
class PipelineLayoutCache final {
 public:
  // 显式构造函数，接受 VkDevice 参数
  explicit PipelineLayoutCache(VkDevice device);

  // 删除复制构造函数和复制赋值运算符重载
  PipelineLayoutCache(const PipelineLayoutCache&) = delete;
  PipelineLayoutCache& operator=(const PipelineLayoutCache&) = delete;

  // 移动构造函数声明，不抛出异常
  PipelineLayoutCache(PipelineLayoutCache&&) noexcept;
  // 删除移动赋值运算符重载
  PipelineLayoutCache& operator=(PipelineLayoutCache&&) = delete;

  // 析构函数声明
  ~PipelineLayoutCache();

  // 使用 VkDescriptorSetLayout 作为键
  using Key = VkDescriptorSetLayout;
  // 使用 PipelineLayout 类作为值
  using Value = PipelineLayout;

  // 哈希函数对象，计算 VkDescriptorSetLayout 的哈希值
  struct Hasher {
    inline size_t operator()(VkDescriptorSetLayout descriptor_layout) const {
      return std::hash<VkDescriptorSetLayout>()(descriptor_layout);
    }
  };

 private:
  // 使用互斥量来管理缓存的多线程访问
  std::mutex cache_mutex_;

  // Vulkan 设备对象
  VkDevice device_;
  // 使用自定义哈希器的无序映射作为缓存容器
  std::unordered_map<Key, Value, Hasher> cache_;

 public:
  // 检索指定键对应的 Vulkan 管线布局对象
  VkPipelineLayout retrieve(const Key&);
  // 清空缓存
  void purge();
};

// 使用最终类修饰符定义 ComputePipelineCache 类
class ComputePipelineCache final {
 public:
  // 显式构造函数，接受 VkDevice 参数
  explicit ComputePipelineCache(VkDevice device);

  // 删除复制构造函数和复制赋值运算符重载
  ComputePipelineCache(const ComputePipelineCache&) = delete;
  ComputePipelineCache& operator=(const ComputePipelineCache&) = delete;

  // 移动构造函数声明，不抛出异常
  ComputePipelineCache(ComputePipelineCache&&) noexcept;
  // 删除移动赋值运算符重载
  ComputePipelineCache& operator=(ComputePipelineCache&&) = delete;

  // 析构函数声明
  ~ComputePipelineCache();

  // 使用 ComputePipeline::Descriptor 结构作为键
  using Key = ComputePipeline::Descriptor;
  // 使用 ComputePipeline 类作为值
  using Value = ComputePipeline;

  // 哈希函数对象，计算 ComputePipeline::Descriptor 结构的哈希值
  struct Hasher {
    inline size_t operator()(
        const ComputePipeline::Descriptor& descriptor) const {
      size_t seed = 0;
      // 组合哈希值：pipeline_layout、shader_module、local_work_group 数据
      seed = utils::hash_combine(
          seed, std::hash<VkPipelineLayout>()(descriptor.pipeline_layout));
      seed = utils::hash_combine(
          seed, std::hash<VkShaderModule>()(descriptor.shader_module));
      seed = utils::hash_combine(
          seed, std::hash<uint32_t>()(descriptor.local_work_group.data[0u]));
      seed = utils::hash_combine(
          seed, std::hash<uint32_t>()(descriptor.local_work_group.data[1u]));
      seed = utils::hash_combine(
          seed, std::hash<uint32_t>()(descriptor.local_work_group.data[2u]));

      return seed;
    }
  };

 private:
  // 使用互斥量来管理缓存的多线程访问
  std::mutex cache_mutex_;

  // Vulkan 设备对象
  VkDevice device_;
  // Vulkan 管线缓存对象
  VkPipelineCache pipeline_cache_;
  // 使用自定义哈希器的无序映射作为缓存容器
  std::unordered_map<Key, Value, Hasher> cache_;

 public:
  // 检索指定键对应的 Vulkan 计算管线对象
  VkPipeline retrieve(const Key&);
  // 清空缓存
  void purge();
};

//
// Impl
//

} // namespace at
} // namespace native
} // namespace vulkan
} // namespace api

#endif /* USE_VULKAN_API */
```