# `.\pytorch\aten\src\ATen\native\vulkan\api\Descriptor.h`

```
#pragma once
// 一旦此头文件被包含，忽略 lint 工具 CLANGTIDY 的 facebook-hte-BadMemberName 错误

#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则包含 Vulkan API 相关头文件

#include <ATen/native/vulkan/api/vk_api.h>
// 包含 Vulkan API 的头文件

#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Shader.h>
// 包含 Vulkan API 的资源和着色器相关头文件

#include <unordered_map>
// 包含无序映射（哈希表）的标准库头文件

namespace at {
namespace native {
namespace vulkan {
namespace api {

class DescriptorSet final {
 public:
  explicit DescriptorSet(VkDevice, VkDescriptorSet, ShaderLayout::Signature);
  // 构造函数，初始化 Vulkan 设备、描述符集和着色器布局的签名信息

  DescriptorSet(const DescriptorSet&) = delete;
  // 删除拷贝构造函数，禁止对象的拷贝

  DescriptorSet& operator=(const DescriptorSet&) = delete;
  // 删除赋值运算符重载，禁止对象的赋值

  DescriptorSet(DescriptorSet&&) noexcept;
  // 移动构造函数，支持对象的移动操作

  DescriptorSet& operator=(DescriptorSet&&) noexcept;
  // 移动赋值运算符重载，支持对象的移动赋值操作

  ~DescriptorSet() = default;
  // 默认析构函数，用于销毁对象

  struct ResourceBinding final {
    uint32_t binding_idx;
    VkDescriptorType descriptor_type;
    bool is_image;

    union {
      VkDescriptorBufferInfo buffer_info;
      VkDescriptorImageInfo image_info;
    } resource_info;
    // 描述符资源绑定结构体，包含绑定索引、描述符类型和资源信息联合体
  };

 private:
  VkDevice device_;
  VkDescriptorSet handle_;
  ShaderLayout::Signature shader_layout_signature_;
  std::vector<ResourceBinding> bindings_;
  // Vulkan 设备、描述符集句柄、着色器布局签名和资源绑定向量

 public:
  DescriptorSet& bind(const uint32_t, const VulkanBuffer&);
  // 将 Vulkan 缓冲对象绑定到指定绑定索引的描述符集中

  DescriptorSet& bind(const uint32_t, const VulkanImage&);
  // 将 Vulkan 图像对象绑定到指定绑定索引的描述符集中

  VkDescriptorSet get_bind_handle() const;
  // 获取当前描述符集的描述符集句柄

 private:
  void add_binding(const ResourceBinding& resource);
  // 向描述符集添加指定的资源绑定信息
};

class DescriptorSetPile final {
 public:
  DescriptorSetPile(
      const uint32_t,
      VkDescriptorSetLayout,
      VkDevice,
      VkDescriptorPool);
  // 构造函数，初始化描述符集堆、描述符集布局、Vulkan 设备和描述符池

  DescriptorSetPile(const DescriptorSetPile&) = delete;
  // 删除拷贝构造函数，禁止对象的拷贝

  DescriptorSetPile& operator=(const DescriptorSetPile&) = delete;
  // 删除赋值运算符重载，禁止对象的赋值

  DescriptorSetPile(DescriptorSetPile&&) = default;
  // 移动构造函数，支持对象的移动操作

  DescriptorSetPile& operator=(DescriptorSetPile&&) = default;
  // 移动赋值运算符重载，支持对象的移动赋值操作

  ~DescriptorSetPile() = default;
  // 默认析构函数，用于销毁对象

 private:
  uint32_t pile_size_;
  VkDescriptorSetLayout set_layout_;
  VkDevice device_;
  VkDescriptorPool pool_;
  std::vector<VkDescriptorSet> descriptors_;
  size_t in_use_;
  // 描述符集堆大小、描述符集布局、Vulkan 设备、描述符池、描述符集向量和正在使用的数量

 public:
  VkDescriptorSet get_descriptor_set();
  // 获取描述符集堆中的描述符集

 private:
  void allocate_new_batch();
  // 分配新的描述符集批次
};

struct DescriptorPoolConfig final {
  // 描述符池配置结构体
  // 总体池容量
  uint32_t descriptorPoolMaxSets;
  // 按类型的描述符计数
  uint32_t descriptorUniformBufferCount;
  uint32_t descriptorStorageBufferCount;
  uint32_t descriptorCombinedSamplerCount;
  uint32_t descriptorStorageImageCount;
  // 预分配描述符集的堆大小
  uint32_t descriptorPileSizes;
};

class DescriptorPool final {
 public:
  explicit DescriptorPool(VkDevice, const DescriptorPoolConfig&);
  // 构造函数，初始化 Vulkan 设备和描述符池配置

  DescriptorPool(const DescriptorPool&) = delete;
  // 删除拷贝构造函数，禁止对象的拷贝

  DescriptorPool& operator=(const DescriptorPool&) = delete;
  // 删除赋值运算符重载，禁止对象的赋值

  DescriptorPool(DescriptorPool&&) = delete;
  // 删除移动构造函数，禁止对象的移动

  DescriptorPool& operator=(DescriptorPool&&) = delete;
  // 删除移动赋值运算符重载，禁止对象的移动赋值

  ~DescriptorPool();
  // 析构函数，用于销毁对象

 private:
  VkDevice device_;
  VkDescriptorPool pool_;
  DescriptorPoolConfig config_;
  // Vulkan 设备、描述符池和描述符池配置

  // 新描述符
  std::mutex mutex_;
  std::unordered_map<VkDescriptorSetLayout, DescriptorSetPile> piles_;
  // 互斥锁和描述符集堆的哈希映射
 public:
  operator bool() const {
    // 返回一个布尔值，表示 pool_ 是否不等于 VK_NULL_HANDLE
    return (pool_ != VK_NULL_HANDLE);
  }

  // 初始化函数，接受一个 DescriptorPoolConfig 类型的参数 config
  void init(const DescriptorPoolConfig& config);

  // 获取描述符集合的函数，接受 VkDescriptorSetLayout 类型的 handle 和 ShaderLayout::Signature 类型的 signature 作为参数
  DescriptorSet get_descriptor_set(
      VkDescriptorSetLayout handle,
      const ShaderLayout::Signature& signature);

  // 刷新操作的函数，没有参数和返回值
  void flush();
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */



// 结束Vulkan API的命名空间
};

// 结束vulkan命名空间
} // namespace api

// 结束API命名空间
} // namespace vulkan

// 结束native命名空间
} // namespace native

// 结束at命名空间
} // namespace at

// 结束条件编译指令，结束使用Vulkan API的条件编译块
#endif /* USE_VULKAN_API */
```