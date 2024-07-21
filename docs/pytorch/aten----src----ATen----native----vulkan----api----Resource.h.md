# `.\pytorch\aten\src\ATen\native\vulkan\api\Resource.h`

```
#pragma once
// 此指令确保头文件只被包含一次

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName
// 忽略 CLANGTIDY 规则 facebook-hte-BadMemberName 的警告

#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则编译以下代码块

#include <ATen/native/vulkan/api/vk_api.h>
// 包含 Vulkan API 头文件

#include <ATen/native/vulkan/api/Allocator.h>
#include <ATen/native/vulkan/api/Types.h>
#include <ATen/native/vulkan/api/Utils.h>
// 包含 Vulkan API 相关的自定义头文件

#include <mutex>
#include <ostream>
#include <stack>
#include <unordered_map>
// 包含标准库头文件

std::ostream& operator<<(std::ostream& out, VmaTotalStatistics stats);
// 声明输出流操作符重载函数，用于打印 VmaTotalStatistics 统计信息

namespace at {
namespace native {
namespace vulkan {
namespace api {

using MemoryAccessFlags = uint8_t;
// 定义内存访问标志类型为 uint8_t

constexpr VmaAllocationCreateFlags DEFAULT_ALLOCATION_STRATEGY =
    VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT;
// 定义默认的 Vulkan 内存分配策略标志为最小内存策略位

enum MemoryAccessType : MemoryAccessFlags {
  NONE = 0u << 0u,
  READ = 1u << 0u,
  WRITE = 1u << 1u,
};
// 定义内存访问类型枚举，包括 NONE、READ、WRITE

struct MemoryBarrier final {
  VkMemoryBarrier handle;
  // Vulkan 内存屏障结构体

  MemoryBarrier(
      const VkAccessFlags src_access_flags,
      const VkAccessFlags dst_access_flags);
  // 构造函数，用于创建内存屏障对象
};

struct MemoryAllocation final {
  explicit MemoryAllocation();
  // 显式默认构造函数声明

  explicit MemoryAllocation(
      const VmaAllocator,
      const VkMemoryRequirements&,
      const VmaAllocationCreateInfo&);
  // 显式构造函数声明，接受 Vulkan 内存需求和分配信息

  MemoryAllocation(const MemoryAllocation&) = delete;
  MemoryAllocation& operator=(const MemoryAllocation&) = delete;
  // 禁用拷贝构造函数和赋值运算符

  MemoryAllocation(MemoryAllocation&&) noexcept;
  MemoryAllocation& operator=(MemoryAllocation&&) noexcept;
  // 移动构造函数和移动赋值运算符声明

  ~MemoryAllocation();
  // 析构函数声明

  VkMemoryRequirements memory_requirements;
  // Vulkan 内存需求属性
  VmaAllocationCreateInfo create_info;
  // Vulkan 内存分配创建信息
  VmaAllocator allocator;
  // Vulkan 内存分配器对象
  VmaAllocation allocation;
  // Vulkan 内存分配句柄

  operator bool() const {
    return (allocation != VK_NULL_HANDLE);
  }
  // 转换运算符重载，用于检查内存分配句柄是否有效
};

class VulkanBuffer final {
 public:
  struct BufferProperties final {
    VkDeviceSize size;
    // 缓冲区大小
    VkDeviceSize mem_offset;
    // 内存偏移量
    VkDeviceSize mem_range;
    // 内存范围
    VkBufferUsageFlags buffer_usage;
    // 缓冲区使用标志
  };

  explicit VulkanBuffer();
  // 显式默认构造函数声明

  explicit VulkanBuffer(
      const VmaAllocator,
      const VkDeviceSize,
      const VmaAllocationCreateInfo&,
      const VkBufferUsageFlags,
      const bool allocate_memory = true);
  // 显式构造函数声明，用于创建 Vulkan 缓冲区对象

  VulkanBuffer(const VulkanBuffer&) = delete;
  VulkanBuffer& operator=(const VulkanBuffer&) = delete;
  // 禁用拷贝构造函数和赋值运算符

  VulkanBuffer(VulkanBuffer&&) noexcept;
  VulkanBuffer& operator=(VulkanBuffer&&) noexcept;
  // 移动构造函数和移动赋值运算符声明

  ~VulkanBuffer();
  // 析构函数声明

  struct Package final {
    VkBuffer handle;
    // Vulkan 缓冲区句柄
    VkDeviceSize buffer_offset;
    // 缓冲区偏移量
    VkDeviceSize buffer_range;
    // 缓冲区范围
  };

  friend struct BufferMemoryBarrier;
  // 声明友元结构体 BufferMemoryBarrier

 private:
  BufferProperties buffer_properties_;
  // 缓冲区属性对象
  VmaAllocator allocator_;
  // Vulkan 内存分配器对象
  MemoryAllocation memory_;
  // Vulkan 内存分配对象
  bool owns_memory_;
  // 是否拥有内存标志
  VkBuffer handle_;
  // Vulkan 缓冲区句柄

 public:
  inline VkDevice device() const {
    VmaAllocatorInfo allocator_info{};
    vmaGetAllocatorInfo(allocator_, &allocator_info);
    return allocator_info.device;
  }
  // 内联函数，返回 Vulkan 设备句柄

  inline VmaAllocator vma_allocator() const {
  // 返回当前对象的分配器
  return allocator_;
}

inline VmaAllocation allocation() const {
  // 返回当前对象内存的分配信息
  return memory_.allocation;
}

inline VmaAllocationCreateInfo allocation_create_info() const {
  // 返回当前对象内存分配的创建信息
  return VmaAllocationCreateInfo(memory_.create_info);
}

inline VkBuffer handle() const {
  // 返回当前对象的 Vulkan 缓冲区句柄
  return handle_;
}

inline VkDeviceSize mem_offset() const {
  // 返回当前对象内存偏移量
  return buffer_properties_.mem_offset;
}

inline VkDeviceSize mem_range() const {
  // 返回当前对象内存范围
  return buffer_properties_.mem_range;
}

inline VkDeviceSize mem_size() const {
  // 返回当前对象内存大小
  return buffer_properties_.size;
}

inline bool has_memory() const {
  // 检查当前对象是否已经分配了内存
  return (memory_.allocation != VK_NULL_HANDLE);
}

inline bool owns_memory() const {
  // 检查当前对象是否拥有内存
  return owns_memory_;
}

operator bool() const {
  // 检查当前对象是否有效（是否有 Vulkan 缓冲区句柄）
  return (handle_ != VK_NULL_HANDLE);
}

inline void bind_allocation(const MemoryAllocation& memory) {
  // 绑定内存分配到当前对象
  VK_CHECK_COND(!memory_, "Cannot bind an already bound allocation!");
  VK_CHECK(vmaBindBufferMemory(allocator_, memory.allocation, handle_));
  memory_.allocation = memory.allocation;
}

// 获取当前对象的内存需求
VkMemoryRequirements get_memory_requirements() const;
};

// 定义一个名为 MemoryMap 的类，用于映射 Vulkan 缓冲区的内存
class MemoryMap final {
 public:
  // 构造函数，接受 VulkanBuffer 和 MemoryAccessFlags 参数
  explicit MemoryMap(
      const VulkanBuffer& buffer,
      const MemoryAccessFlags access);

  // 禁用复制构造函数和赋值运算符
  MemoryMap(const MemoryMap&) = delete;
  MemoryMap& operator=(const MemoryMap&) = delete;

  // 移动构造函数
  MemoryMap(MemoryMap&&) noexcept;
  // 禁用移动赋值运算符
  MemoryMap& operator=(MemoryMap&&) = delete;

  // 析构函数
  ~MemoryMap();

 private:
  // 访问标志
  uint8_t access_;
  // Vulkan 内存分配器
  VmaAllocator allocator_;
  // Vulkan 内存分配
  VmaAllocation allocation_;
  // 指向映射数据的指针
  void* data_;
  // 映射数据的长度
  VkDeviceSize data_len_;

 public:
  // 返回映射数据的类型转换指针模板
  template <typename T>
  T* data() {
    return reinterpret_cast<T*>(data_);
  }

  // 返回映射数据的字节数
  inline size_t nbytes() {
    return utils::safe_downcast<size_t>(data_len_);
  }

  // 使映射数据失效的方法
  void invalidate();
};

// 定义一个名为 BufferMemoryBarrier 的结构体
struct BufferMemoryBarrier final {
  // Vulkan 缓冲区内存屏障句柄
  VkBufferMemoryBarrier handle;

  // 构造函数，接受源访问标志、目标访问标志和 VulkanBuffer 参数
  BufferMemoryBarrier(
      const VkAccessFlags src_access_flags,
      const VkAccessFlags dst_access_flags,
      const VulkanBuffer& buffer);
};

// 定义一个名为 ImageSampler 的类
class ImageSampler final {
 public:
  // 内部属性结构体
  struct Properties final {
    VkFilter filter;
    VkSamplerMipmapMode mipmap_mode;
    VkSamplerAddressMode address_mode;
    VkBorderColor border_color;
  };

  // 构造函数，接受 VkDevice 和 Properties 参数
  explicit ImageSampler(VkDevice, const Properties&);

  // 禁用复制构造函数和赋值运算符
  ImageSampler(const ImageSampler&) = delete;
  ImageSampler& operator=(const ImageSampler&) = delete;

  // 移动构造函数
  ImageSampler(ImageSampler&&) noexcept;
  // 禁用移动赋值运算符
  ImageSampler& operator=(ImageSampler&&) = delete;

  // 析构函数
  ~ImageSampler();

 private:
  // Vulkan 设备
  VkDevice device_;
  // Vulkan 采样器句柄
  VkSampler handle_;

 public:
  // 返回采样器句柄的方法
  VkSampler handle() const {
    return handle_;
  }

  // 用于自定义哈希的结构体
  struct Hasher {
    size_t operator()(const Properties&) const;
  };

  // 自定义的交换函数，用于哈希映射中使用
  friend void swap(ImageSampler& lhs, ImageSampler& rhs) noexcept;
};

// 定义一个名为 VulkanImage 的类
class VulkanImage final {
 public:
  // 图像属性结构体
  struct ImageProperties final {
    VkImageType image_type;
    VkFormat image_format;
    VkExtent3D image_extents;
    VkImageUsageFlags image_usage;
  };

  // 视图属性结构体
  struct ViewProperties final {
    VkImageViewType view_type;
    VkFormat view_format;
  };

  // 使用 ImageSampler 的属性
  using SamplerProperties = ImageSampler::Properties;

  // 句柄结构体
  struct Handles final {
    VkImage image;
    VkImageView image_view;
    VkSampler sampler;
  };

  // 默认构造函数
  explicit VulkanImage();

  // 构造函数，接受 VmaAllocator、VmaAllocationCreateInfo、ImageProperties、ViewProperties、SamplerProperties、VkImageLayout 和 VkSampler 参数
  explicit VulkanImage(
      const VmaAllocator,
      const VmaAllocationCreateInfo&,
      const ImageProperties&,
      const ViewProperties&,
      const SamplerProperties&,
      const VkImageLayout layout,
      VkSampler,
      const bool allocate_memory = true);

  // 禁用复制构造函数和赋值运算符
  VulkanImage(const VulkanImage&) = delete;
  VulkanImage& operator=(const VulkanImage&) = delete;

  // 移动构造函数
  VulkanImage(VulkanImage&&) noexcept;
  // 移动赋值运算符
  VulkanImage& operator=(VulkanImage&&) noexcept;

  // 析构函数
  ~VulkanImage();

  // 包结构体
  struct Package final {
    VkImage handle;
    VkImageLayout image_layout;
    VkImageView image_view;
  // 图像采样器对象
  VkSampler image_sampler;
};

// 图像内存屏障结构体
friend struct ImageMemoryBarrier;

private:
// 图像属性
ImageProperties image_properties_;
// 视图属性
ViewProperties view_properties_;
// 采样器属性
SamplerProperties sampler_properties_;
// 分配器对象，用于内存管理
VmaAllocator allocator_;
// 分配的内存句柄
MemoryAllocation memory_;
// 标识底层内存是否由资源对象管理
bool owns_memory_;
// 句柄集合
Handles handles_;
// 图像布局
VkImageLayout layout_;

public:
// 创建图像视图
void create_image_view();

// 返回设备对象
inline VkDevice device() const {
  VmaAllocatorInfo allocator_info{};
  vmaGetAllocatorInfo(allocator_, &allocator_info);
  return allocator_info.device;
}

// 返回VmaAllocator分配器对象
inline VmaAllocator vma_allocator() const {
  return allocator_;
}

// 返回分配句柄
inline VmaAllocation allocation() const {
  return memory_.allocation;
}

// 返回分配信息
inline VmaAllocationCreateInfo allocation_create_info() const {
  return VmaAllocationCreateInfo(memory_.create_info);
}

// 返回图像格式
inline VkFormat format() const {
  return image_properties_.image_format;
}

// 返回图像尺寸
inline VkExtent3D extents() const {
  return image_properties_.image_extents;
}

// 返回图像句柄
inline VkImage handle() const {
  return handles_.image;
}

// 返回图像视图句柄
inline VkImageView image_view() const {
  return handles_.image_view;
}

// 返回图像采样器句柄
inline VkSampler sampler() const {
  return handles_.sampler;
}

// 返回图像包装对象
Package package() const {
  return {
      handles_.image,
      layout_,
      handles_.image_view,
      handles_.sampler,
  };
}

// 返回当前图像布局
inline VkImageLayout layout() const {
  return layout_;
}

// 设置图像布局
inline void set_layout(const VkImageLayout layout) {
  layout_ = layout;
}

// 检查是否有分配内存
inline bool has_memory() const {
  return (memory_.allocation != VK_NULL_HANDLE);
}

// 检查是否资源对象管理内存
inline bool owns_memory() const {
  return owns_memory_;
}

// 隐式布尔转换，判断图像是否有效
inline operator bool() const {
  return (handles_.image != VK_NULL_HANDLE);
}

// 绑定分配的内存
inline void bind_allocation(const MemoryAllocation& memory) {
  VK_CHECK_COND(!memory_, "Cannot bind an already bound allocation!");
  VK_CHECK(vmaBindImageMemory(allocator_, memory.allocation, handles_.image));
  memory_.allocation = memory.allocation;

  // 只有当图像绑定了内存后才创建图像视图
  create_image_view();
}

// 获取图像内存需求
VkMemoryRequirements get_memory_requirements() const;
};

/*
 * Vulkan图像内存屏障结构体，包装VkImageMemoryBarrier对象
 */
struct ImageMemoryBarrier final {
  VkImageMemoryBarrier handle;

  /*
   * 构造函数，初始化内存屏障对象
   * @param src_access_flags 源访问标志
   * @param dst_access_flags 目标访问标志
   * @param src_layout_flags 源图像布局标志
   * @param dst_layout_flags 目标图像布局标志
   * @param image Vulkan图像对象的引用
   */
  ImageMemoryBarrier(
      const VkAccessFlags src_access_flags,
      const VkAccessFlags dst_access_flags,
      const VkImageLayout src_layout_flags,
      const VkImageLayout dst_layout_flags,
      const VulkanImage& image);
};

/*
 * 纹理采样器缓存类，管理Vulkan纹理采样器对象的缓存
 */
class SamplerCache final {
 public:
  explicit SamplerCache(VkDevice device);

  SamplerCache(const SamplerCache&) = delete;
  SamplerCache& operator=(const SamplerCache&) = delete;

  SamplerCache(SamplerCache&&) noexcept;
  SamplerCache& operator=(SamplerCache&&) = delete;

  ~SamplerCache();

  using Key = ImageSampler::Properties;
  using Value = ImageSampler;
  using Hasher = ImageSampler::Hasher;

 private:
  // 多个线程可能同时向缓存中添加条目，因此使用互斥锁管理访问
  std::mutex cache_mutex_;

  VkDevice device_;
  std::unordered_map<Key, Value, Hasher> cache_;

 public:
  /*
   * 根据给定键值检索纹理采样器对象
   * @param key 纹理采样器属性结构体
   * @return VkSampler对象
   */
  VkSampler retrieve(const Key&);

  /*
   * 清空缓存中的所有条目
   */
  void purge();
};

/*
 * 内存分配器类，管理Vulkan设备内存的分配与释放
 */
class MemoryAllocator final {
 public:
  explicit MemoryAllocator(
      VkInstance instance,
      VkPhysicalDevice physical_device,
      VkDevice device);

  MemoryAllocator(const MemoryAllocator&) = delete;
  MemoryAllocator& operator=(const MemoryAllocator&) = delete;

  MemoryAllocator(MemoryAllocator&&) noexcept;
  MemoryAllocator& operator=(MemoryAllocator&&) = delete;

  ~MemoryAllocator();

 private:
  VkInstance instance_;
  VkPhysicalDevice physical_device_;
  VkDevice device_;
  VmaAllocator allocator_;

 public:
  /*
   * 根据内存需求和分配信息创建内存分配对象
   * @param memory_requirements Vulkan内存需求对象
   * @param create_info 内存分配创建信息
   * @return MemoryAllocation对象
   */
  MemoryAllocation create_allocation(
      const VkMemoryRequirements& memory_requirements,
      const VmaAllocationCreateInfo& create_info);

  /*
   * 创建Vulkan图像对象
   * @param extent 图像的尺寸
   * @param format 图像的格式
   * @param image_type 图像类型
   * @param view_type 图像视图类型
   * @param sampler_props 纹理采样器属性
   * @param sampler 纹理采样器对象
   * @param allow_transfer 是否允许传输操作，默认为false
   * @param allocate_memory 是否分配内存，默认为true
   * @return VulkanImage对象
   */
  VulkanImage create_image(
      const VkExtent3D& extent,
      const VkFormat format,
      const VkImageType image_type,
      const VkImageViewType view_type,
      const VulkanImage::SamplerProperties& sampler_props,
      VkSampler sampler,
      const bool allow_transfer = false,
      const bool allocate_memory = true);

  /*
   * 创建Vulkan存储缓冲区对象
   * @param size 缓冲区大小
   * @param gpu_only 是否仅限GPU访问，默认为true
   * @param allocate_memory 是否分配内存，默认为true
   * @return VulkanBuffer对象
   */
  VulkanBuffer create_storage_buffer(
      const VkDeviceSize size,
      const bool gpu_only = true,
      const bool allocate_memory = true);

  /*
   * 创建用于传输操作的临时缓冲区对象
   * @param size 缓冲区大小
   * @return VulkanBuffer对象
   */
  VulkanBuffer create_staging_buffer(const VkDeviceSize size);

  /*
   * 创建指定大小的均匀缓冲区对象
   * @param size 缓冲区大小
   * @return VulkanBuffer对象
   */
  VulkanBuffer create_uniform_buffer(const VkDeviceSize size);

  /*
   * 创建包含任意结构体数据的均匀缓冲区对象
   * @tparam Block 数据块类型
   * @param block 数据块对象
   * @return VulkanBuffer对象
   */
  template <typename Block>
  VulkanBuffer create_params_buffer(const Block& block);

  /*
   * 获取内存统计信息
   * @return VmaTotalStatistics对象，包含分配器的内存使用统计信息
   */
  VmaTotalStatistics get_memory_statistics() const {
    VmaTotalStatistics stats = {};
    vmaCalculateStatistics(allocator_, &stats);
    return stats;
  }
};
// VulkanFence 类的声明和实现，用于管理 Vulkan 中的 Fence 对象
class VulkanFence final {
 public:
  // TODO: This is required for the lazy allocation pattern in api/Tensor.
  //       It will be disabled pending future refactors.
  // 显式构造函数，用于创建 VulkanFence 对象
  explicit VulkanFence();

  // 显式构造函数，用指定的 VkDevice 创建 VulkanFence 对象
  explicit VulkanFence(VkDevice);

  // 删除拷贝构造函数和拷贝赋值运算符，禁止对象的拷贝
  VulkanFence(const VulkanFence&) = delete;
  VulkanFence& operator=(const VulkanFence&) = delete;

  // 移动构造函数和移动赋值运算符，支持对象的移动语义
  VulkanFence(VulkanFence&&) noexcept;
  VulkanFence& operator=(VulkanFence&&) noexcept;

  // 析构函数，清理对象资源
  ~VulkanFence();

 private:
  VkDevice device_;  // Vulkan 设备句柄
  VkFence handle_;   // Vulkan Fence 句柄
  bool waiting_;     // 标志，指示是否正在等待 Fence 信号

 public:
  // 获取用于提交队列的 Fence 句柄
  VkFence get_submit_handle() {
    if (handle_ != VK_NULL_HANDLE) {
      // 标记当前 Fence 正在等待信号
      waiting_ = true;
    }
    return handle_;
  }

  // 返回 Fence 句柄
  VkFence handle() {
    return handle_;
  }

  // 触发同步等待 Fence 信号
  void wait();

  // 返回当前 Fence 是否正在等待信号
  bool waiting() const {
    return waiting_;
  }

  // 将对象转换为布尔值，判断是否有效（handle_ != VK_NULL_HANDLE）
  operator bool() const {
    return (VK_NULL_HANDLE != handle_);
  }
};

// FencePool 结构体，用于管理 VulkanFence 对象的池，实现对象的重用
// 只能由单个线程修改
struct FencePool final {
  VkDevice device_;           // Vulkan 设备句柄

  std::stack<VulkanFence> pool_;  // 存储 VulkanFence 对象的堆栈

  // 显式构造函数，初始化 FencePool 对象
  explicit FencePool(VkDevice device) : device_(device), pool_{} {}

  // 返回一个右值引用，以便可以移动对象
  inline VulkanFence get_fence() {
    if (pool_.empty()) {
      // 如果池为空，创建一个新的 VulkanFence 对象
      VulkanFence new_fence = VulkanFence(device_);
      return new_fence;
    }

    // 从池顶部获取 Fence 对象，并移除该对象
    VulkanFence top_fence = std::move(pool_.top());
    pool_.pop();

    return top_fence;
  }

  // 标记 Fence 对象为可用状态，放回池中
  inline void return_fence(VulkanFence& fence) {
    pool_.push(std::move(fence));
  }
};

//
// Impl
//

// MemoryAllocator 类的成员函数模板，用于创建 VulkanBuffer 对象，并填充数据
template <typename Block>
inline VulkanBuffer MemoryAllocator::create_params_buffer(const Block& block) {
  // 创建指定大小的 uniform buffer
  VulkanBuffer uniform_buffer = create_uniform_buffer(sizeof(Block));

  // 将 block 中的数据填充到 uniform buffer 中
  {
    MemoryMap mapping(uniform_buffer, MemoryAccessType::WRITE);
    Block* data_ptr = mapping.template data<Block>();

    *data_ptr = block;
  }

  return uniform_buffer;  // 返回填充后的 uniform buffer
}

// namespace 结束标记，用于 Vulkan API 的命名空间
} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```