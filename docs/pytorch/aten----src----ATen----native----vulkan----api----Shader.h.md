# `.\pytorch\aten\src\ATen\native\vulkan\api\Shader.h`

```py
#pragma once
// 预处理指令，指示编译器只包含本文件一次

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName
// 忽略静态分析工具的指定警告

#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则编译以下内容

#include <ATen/native/vulkan/api/vk_api.h>
// 包含 Vulkan API 头文件

#include <ATen/native/vulkan/api/Types.h>
#include <ATen/native/vulkan/api/Utils.h>
// 包含 Vulkan 相关的类型定义和实用工具函数

#include <mutex>
#include <unordered_map>
// 包含互斥锁和无序映射的标准库头文件

namespace at {
namespace native {
namespace vulkan {
namespace api {

class ShaderLayout final {
// Vulkan 着色器布局类声明

 public:
  using Signature = std::vector<VkDescriptorType>;
  // 使用 Vulkan 描述符类型的向量作为 Signature 别名

  explicit ShaderLayout(VkDevice, const Signature&);
  // 显式构造函数，接受 Vulkan 设备和 Signature 参数

  ShaderLayout(const ShaderLayout&) = delete;
  // 删除复制构造函数

  ShaderLayout& operator=(const ShaderLayout&) = delete;
  // 删除赋值运算符重载

  ShaderLayout(ShaderLayout&&) noexcept;
  // 移动构造函数声明，使用 noexcept

  ShaderLayout& operator=(ShaderLayout&&) = delete;
  // 删除移动赋值运算符

  ~ShaderLayout();
  // 析构函数声明，用于释放资源

 private:
  VkDevice device_;
  // Vulkan 设备句柄

  VkDescriptorSetLayout handle_;
  // Vulkan 描述符集布局句柄

 public:
  VkDescriptorSetLayout handle() const {
    return handle_;
  }
  // 返回 Vulkan 描述符集布局句柄的访问函数

  // 定义自定义交换函数，因为该类不允许移动赋值。该交换函数将在哈希映射中使用。
  friend void swap(ShaderLayout& lhs, ShaderLayout& rhs) noexcept;
};

struct ShaderInfo final {
// Vulkan 着色器信息结构体声明

  struct {
    const uint32_t* bin;
    uint32_t size;
  } src_code;
  // 着色器源代码结构体成员，包含指向代码的指针和代码的大小

  std::string kernel_name{""};
  // 着色器内核名称的字符串，默认为空字符串

  ShaderLayout::Signature kernel_layout{};
  // 着色器布局签名，使用 Vulkan 描述符类型的向量初始化为空

  // 着色器元数据
  utils::uvec3 out_tile_size{1u, 1u, 1u};
  // 输出瓦片大小的向量，默认为 (1, 1, 1)

  std::vector<uint32_t> tile_size;
  // 瓦片大小的无符号整数向量

  StorageType bias_storage_type{StorageType::UNKNOWN};
  // 偏置存储类型，默认为未知

  StorageType weight_storage_type{StorageType::UNKNOWN};
  // 权重存储类型，默认为未知

  explicit ShaderInfo();
  // 显式默认构造函数声明

  explicit ShaderInfo(std::string, const char*);
  // 显式构造函数声明，接受字符串和字符数组参数

  explicit ShaderInfo(
      std::string,
      const uint32_t*,
      const uint32_t,
      std::vector<VkDescriptorType>);
  // 显式构造函数声明，接受字符串、代码指针、代码大小和 Vulkan 描述符类型向量参数

  explicit ShaderInfo(
      std::string,
      const uint32_t*,
      const uint32_t,
      std::vector<VkDescriptorType>,
      const std::vector<uint32_t>& tile_size,
      const StorageType bias_storage_type,
      const StorageType weight_storage_type);
  // 显式构造函数声明，接受字符串、代码指针、代码大小、Vulkan 描述符类型向量、瓦片大小、偏置存储类型和权重存储类型参数
};

bool operator==(const ShaderInfo& _1, const ShaderInfo& _2);
// 着色器信息相等比较运算符重载声明

class ShaderModule final {
// Vulkan 着色器模块类声明

 public:
  explicit ShaderModule(VkDevice device, const ShaderInfo& source);
  // 显式构造函数声明，接受 Vulkan 设备和着色器信息参数

  ShaderModule(const ShaderModule&) = delete;
  // 删除复制构造函数

  ShaderModule& operator=(const ShaderModule&) = delete;
  // 删除赋值运算符重载

  ShaderModule(ShaderModule&&) noexcept;
  // 移动构造函数声明，使用 noexcept

  ShaderModule& operator=(ShaderModule&&) = delete;
  // 删除移动赋值运算符

  ~ShaderModule();
  // 析构函数声明，用于释放资源

 private:
  VkDevice device_;
  // Vulkan 设备句柄

  VkShaderModule handle_;
  // Vulkan 着色器模块句柄

 public:
  inline VkShaderModule handle() const {
    return handle_;
  }
  // 返回 Vulkan 着色器模块句柄的访问函数

  // 定义自定义交换函数，因为该类不允许移动赋值。该交换函数将在哈希映射中使用。
  friend void swap(ShaderModule& lhs, ShaderModule& rhs) noexcept;
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif // USE_VULKAN_API
// 结束条件编译指令，结束 USE_VULKAN_API 宏定义的条件编译内容
// 定义 Vulkan API 的命名空间
namespace api {
namespace vulkan {
namespace native {
namespace at {

// ShaderLayoutCache 类的定义
class ShaderLayoutCache final {
 public:
  // 显式构造函数，接受 VkDevice 参数
  explicit ShaderLayoutCache(VkDevice device);

  // 删除复制构造函数和赋值运算符重载
  ShaderLayoutCache(const ShaderLayoutCache&) = delete;
  ShaderLayoutCache& operator=(const ShaderLayoutCache&) = delete;

  // 移动构造函数声明，标记为 noexcept
  ShaderLayoutCache(ShaderLayoutCache&&) noexcept;
  // 删除移动赋值运算符重载
  ShaderLayoutCache& operator=(ShaderLayoutCache&&) = delete;

  // 析构函数声明
  ~ShaderLayoutCache();

  // 使用 ShaderLayout::Signature 作为键，ShaderLayout 作为值
  using Key = ShaderLayout::Signature;
  using Value = ShaderLayout;

  // Hasher 结构体，用于计算 ShaderLayout::Signature 的哈希值
  struct Hasher {
    // 重载 () 运算符，计算 signature 的哈希值
    inline size_t operator()(const ShaderLayout::Signature& signature) const {
      size_t hashed = 0u;

      // 遍历 signature 中的 VkDescriptorType，使用 utils::hash_combine 计算哈希值
      for (const VkDescriptorType type : signature) {
        hashed =
            utils::hash_combine(hashed, std::hash<VkDescriptorType>()(type));
      }

      return hashed;
    }
  };

 private:
  // 用于管理访问的互斥锁，因为多个线程可能同时向缓存中添加条目
  std::mutex cache_mutex_;

  // Vulkan 设备对象
  VkDevice device_;

  // 使用 Key 和 Value 进行映射的无序哈希映射缓存
  std::unordered_map<Key, Value, Hasher> cache_;

 public:
  // 根据 Key 检索 VkDescriptorSetLayout
  VkDescriptorSetLayout retrieve(const Key&);

  // 清空缓存的方法声明
  void purge();
};

// ShaderCache 类的定义
class ShaderCache final {
 public:
  // 显式构造函数，接受 VkDevice 参数
  explicit ShaderCache(VkDevice device);

  // 删除复制构造函数和赋值运算符重载
  ShaderCache(const ShaderCache&) = delete;
  ShaderCache& operator=(const ShaderCache&) = delete;

  // 移动构造函数声明，标记为 noexcept
  ShaderCache(ShaderCache&&) noexcept;
  // 删除移动赋值运算符重载
  ShaderCache& operator=(ShaderCache&&) = delete;

  // 析构函数声明
  ~ShaderCache();

  // 使用 ShaderInfo 作为键，ShaderModule 作为值
  using Key = ShaderInfo;
  using Value = ShaderModule;

  // Hasher 结构体，用于计算 ShaderInfo 的哈希值
  struct Hasher {
    // 重载 () 运算符，计算 source 的哈希值
    inline size_t operator()(const ShaderInfo& source) const {
      size_t seed = 0;
      seed = utils::hash_combine(
          seed, std::hash<const uint32_t*>()(source.src_code.bin));
      seed = utils::hash_combine(
          seed, std::hash<uint32_t>()(source.src_code.size));

      return seed;
    }
  };

 private:
  // 用于管理访问的互斥锁，因为多个线程可能同时向缓存中添加条目
  std::mutex cache_mutex_;

  // Vulkan 设备对象
  VkDevice device_;

  // 使用 Key 和 Value 进行映射的无序哈希映射缓存
  std::unordered_map<Key, Value, Hasher> cache_;

 public:
  // 根据 Key 检索 VkShaderModule
  VkShaderModule retrieve(const Key&);

  // 清空缓存的方法声明
  void purge();
};

} // namespace at
} // namespace native
} // namespace vulkan
} // namespace api

// 定义全局的 operator==，用于比较 VkDescriptorSetLayoutBinding 的相等性
inline bool operator==(
    const VkDescriptorSetLayoutBinding& _1,
    const VkDescriptorSetLayoutBinding& _2) {
  return (
      _1.binding == _2.binding && _1.descriptorType == _2.descriptorType &&
      _1.descriptorCount == _2.descriptorCount &&
      _1.stageFlags == _2.stageFlags &&
      _1.pImmutableSamplers == _2.pImmutableSamplers);
}

#endif /* USE_VULKAN_API */
```