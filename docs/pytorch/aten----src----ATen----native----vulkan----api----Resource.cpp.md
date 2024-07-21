# `.\pytorch\aten\src\ATen\native\vulkan\api\Resource.cpp`

```py
      VK_CHECK(vmaCreateBuffer(
          allocator_, // allocator
          &buffer_create_info, // pBufferCreateInfo
          &memory_.create_info, // pAllocationCreateInfo
          &handle_, // pBuffer
          &memory_.allocation, // pAllocation
          nullptr)); // pAllocationInfo



    }
}

VulkanBuffer::~VulkanBuffer() {
  if (VK_NULL_HANDLE != handle_) {
    vkDestroyBuffer(vmaGetDeviceAllocator(allocator_), handle_, nullptr);
  }
  if (owns_memory_ && VK_NULL_HANDLE != memory_.allocation) {
    vmaFreeMemory(allocator_, memory_.allocation);
  }
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at



} // namespace api
    # 如果使用了 Vulkan 内存分配器 (VMA)，则使用其创建缓冲区
    VK_CHECK(vmaCreateBuffer(
        allocator_,
        &buffer_create_info,
        &allocation_create_info,
        &handle_,
        &(memory_.allocation),
        nullptr));
  } else {
    # 如果未使用 Vulkan 内存分配器 (VMA)，获取分配器信息
    VmaAllocatorInfo allocator_info{};
    vmaGetAllocatorInfo(allocator_, &allocator_info);
    # 使用 Vulkan 标准 API 创建缓冲区
    VK_CHECK(vkCreateBuffer(
        allocator_info.device, &buffer_create_info, nullptr, &handle_));
  }
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept
    : buffer_properties_(other.buffer_properties_),  // 初始化成员变量 buffer_properties_
      allocator_(other.allocator_),  // 初始化成员变量 allocator_
      memory_(std::move(other.memory_)),  // 移动构造函数调用，初始化成员变量 memory_
      owns_memory_(other.owns_memory_),  // 初始化成员变量 owns_memory_
      handle_(other.handle_) {  // 初始化成员变量 handle_
  other.handle_ = VK_NULL_HANDLE;  // 将 other 的 handle_ 设置为 VK_NULL_HANDLE
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept {
  VkBuffer tmp_buffer = handle_;  // 保存当前对象的 handle_ 到 tmp_buffer
  bool tmp_owns_memory = owns_memory_;  // 保存当前对象的 owns_memory_ 到 tmp_owns_memory

  buffer_properties_ = other.buffer_properties_;  // 赋值操作，更新 buffer_properties_
  allocator_ = other.allocator_;  // 赋值操作，更新 allocator_
  memory_ = std::move(other.memory_);  // 移动赋值操作，更新 memory_
  owns_memory_ = other.owns_memory_;  // 赋值操作，更新 owns_memory_
  handle_ = other.handle_;  // 赋值操作，更新 handle_

  other.handle_ = tmp_buffer;  // 将保存的 tmp_buffer 赋值给 other 的 handle_
  other.owns_memory_ = tmp_owns_memory;  // 将保存的 tmp_owns_memory 赋值给 other 的 owns_memory_

  return *this;  // 返回当前对象的引用
}

VulkanBuffer::~VulkanBuffer() {
  if (VK_NULL_HANDLE != handle_) {  // 检查 handle_ 是否为 VK_NULL_HANDLE
    if (owns_memory_) {  // 如果 owns_memory_ 为 true
      vmaDestroyBuffer(allocator_, handle_, memory_.allocation);  // 销毁由 Vulkan Memory Allocator 分配的 buffer
    } else {
      vkDestroyBuffer(this->device(), handle_, nullptr);  // 使用 Vulkan API 销毁 buffer
    }
    // 防止底层内存分配被释放；它可能已经被 vmaDestroyBuffer 释放，或者这个资源没有拥有底层内存
    memory_.allocation = VK_NULL_HANDLE;  // 将 memory_.allocation 设置为 VK_NULL_HANDLE
  }
}

VkMemoryRequirements VulkanBuffer::get_memory_requirements() const {
  VkMemoryRequirements memory_requirements;
  vkGetBufferMemoryRequirements(this->device(), handle_, &memory_requirements);  // 获取 buffer 的内存需求
  return memory_requirements;  // 返回内存需求结构体
}

//
// MemoryMap
//

MemoryMap::MemoryMap(const VulkanBuffer& buffer, const uint8_t access)
    : access_(access),  // 初始化成员变量 access_
      allocator_(buffer.vma_allocator()),  // 调用 Vulkan Memory Allocator 获取 allocator
      allocation_(buffer.allocation()),  // 获取 buffer 的内存分配
      data_(nullptr),  // 初始化 data_ 为 nullptr
      data_len_{buffer.mem_size()} {  // 初始化 data_len_ 为 buffer 的内存大小
  if (allocation_) {  // 如果 allocation_ 不为空
    VK_CHECK(vmaMapMemory(allocator_, allocation_, &data_));  // 使用 Vulkan Memory Allocator 映射内存
  }
}

MemoryMap::MemoryMap(MemoryMap&& other) noexcept
    : access_(other.access_),  // 初始化成员变量 access_
      allocator_(other.allocator_),  // 初始化成员变量 allocator_
      allocation_(other.allocation_),  // 初始化成员变量 allocation_
      data_(other.data_),  // 初始化成员变量 data_
      data_len_{other.data_len_} {  // 初始化成员变量 data_len_
  other.allocation_ = VK_NULL_HANDLE;  // 将 other 的 allocation_ 设置为 VK_NULL_HANDLE
  other.data_ = nullptr;  // 将 other 的 data_ 设置为 nullptr
}

MemoryMap::~MemoryMap() {
  if (!data_) {  // 如果 data_ 为 nullptr，则直接返回
    return;
  }

  if (allocation_) {  // 如果 allocation_ 不为空
    if (access_ & MemoryAccessType::WRITE) {  // 如果访问类型包含写权限
      // 如果该内存类型不是 HOST_VISIBLE 或者是 HOST_COHERENT，则实现会忽略此调用，这正是我们期望的行为。
      // 在析构函数中不检查结果，因为析构函数无法抛出异常。
      vmaFlushAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE);  // 刷新内存映射
    }

    vmaUnmapMemory(allocator_, allocation_);  // 取消内存映射
  }
}

void MemoryMap::invalidate() {
  if (access_ & MemoryAccessType::READ && allocation_) {  // 如果访问类型包含读权限且 allocation_ 不为空
    // 如果该内存类型不是 HOST_VISIBLE 或者是 HOST_COHERENT，则实现会忽略此调用，这正是我们期望的行为。
    VK_CHECK(
        vmaInvalidateAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE));  // 使内存映射无效
  }
}

//
// BufferMemoryBarrier
//

BufferMemoryBarrier::BufferMemoryBarrier(
    // VulkanBufferMemoryBarrier 类的构造函数，用于创建 Vulkan 缓冲区内存屏障对象
    const VkAccessFlags src_access_flags,  // 源访问标志，指定内存屏障之前的访问方式
    const VkAccessFlags dst_access_flags,  // 目标访问标志，指定内存屏障之后的访问方式
    const VulkanBuffer& buffer)            // VulkanBuffer 对象的引用，表示涉及的缓冲区
    : handle{
          VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,   // Vulkan 结构体类型，表明这是一个缓冲区内存屏障对象
          nullptr,                                   // 指向下一个结构体的指针，在此处未使用
          src_access_flags,                          // 指定内存屏障前的访问掩码
          dst_access_flags,                          // 指定内存屏障后的访问掩码
          VK_QUEUE_FAMILY_IGNORED,                   // 源队列家族索引，此处表示忽略队列家族的影响
          VK_QUEUE_FAMILY_IGNORED,                   // 目标队列家族索引，同样表示忽略队列家族的影响
          buffer.handle_,                            // 关联的 Vulkan 缓冲区句柄
          buffer.buffer_properties_.mem_offset,      // 缓冲区内存偏移量，标识内存区域的起始位置
          buffer.buffer_properties_.mem_range,       // 缓冲区内存范围，表示内存区域的大小
      } {}
//
// ImageSampler
//

// 定义 ImageSampler 类的相等运算符重载，用于比较两个 ImageSampler::Properties 结构体是否相等
bool operator==(
    const ImageSampler::Properties& _1,
    const ImageSampler::Properties& _2) {
  return (
      _1.filter == _2.filter && _1.mipmap_mode == _2.mipmap_mode &&
      _1.address_mode == _2.address_mode && _1.border_color == _2.border_color);
}

// ImageSampler 类的构造函数，初始化 Vulkan 采样器对象
ImageSampler::ImageSampler(
    VkDevice device,
    const ImageSampler::Properties& props)
    : device_(device), handle_(VK_NULL_HANDLE) {
  // 创建 VkSamplerCreateInfo 结构体，用于描述 Vulkan 采样器的创建信息
  const VkSamplerCreateInfo sampler_create_info{
      VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      props.filter, // magFilter
      props.filter, // minFilter
      props.mipmap_mode, // mipmapMode
      props.address_mode, // addressModeU
      props.address_mode, // addressModeV
      props.address_mode, // addressModeW
      0.0f, // mipLodBias
      VK_FALSE, // anisotropyEnable
      1.0f, // maxAnisotropy,
      VK_FALSE, // compareEnable
      VK_COMPARE_OP_NEVER, // compareOp
      0.0f, // minLod
      VK_LOD_CLAMP_NONE, // maxLod
      props.border_color, // borderColor
      VK_FALSE, // unnormalizedCoordinates
  };

  // 创建 Vulkan 采样器对象，并将其句柄存储到 handle_ 成员变量中
  VK_CHECK(vkCreateSampler(device_, &sampler_create_info, nullptr, &handle_));
}

// ImageSampler 类的移动构造函数，用于有效地移动另一个 ImageSampler 对象的资源
ImageSampler::ImageSampler(ImageSampler&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE; // 将原对象的句柄置为空，避免重复销毁
}

// ImageSampler 类的析构函数，用于释放 Vulkan 采样器对象的资源
ImageSampler::~ImageSampler() {
  if (VK_NULL_HANDLE == handle_) {
    return; // 如果句柄为空，则无需执行销毁操作
  }
  // 销毁 Vulkan 采样器对象
  vkDestroySampler(device_, handle_, nullptr);
}

// 定义 ImageSampler::Properties 结构体的哈希计算函数，用于作为哈希表的键值
size_t ImageSampler::Hasher::operator()(
    const ImageSampler::Properties& props) const {
  size_t seed = 0;
  // 使用 utils::hash_combine 函数依次计算结构体成员的哈希值，并合并得到最终的哈希值
  seed = utils::hash_combine(seed, std::hash<VkFilter>()(props.filter));
  seed = utils::hash_combine(
      seed, std::hash<VkSamplerMipmapMode>()(props.mipmap_mode));
  seed = utils::hash_combine(
      seed, std::hash<VkSamplerAddressMode>()(props.address_mode));
  seed =
      utils::hash_combine(seed, std::hash<VkBorderColor>()(props.border_color));
  return seed;
}

// 定义 swap 函数，用于交换两个 ImageSampler 对象的成员变量
void swap(ImageSampler& lhs, ImageSampler& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkSampler tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// VulkanImage
//

// VulkanImage 类的默认构造函数，初始化成员变量
VulkanImage::VulkanImage()
    : image_properties_{},
      view_properties_{},
      sampler_properties_{},
      allocator_(VK_NULL_HANDLE),
      memory_{},
      owns_memory_(false),
      handles_{
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
      },
      layout_{} {}

// VulkanImage 类的构造函数，用于创建 Vulkan 图像对象
VulkanImage::VulkanImage(
    VmaAllocator vma_allocator,
    const VmaAllocationCreateInfo& allocation_create_info,
    const ImageProperties& image_props,
    const ViewProperties& view_props,
    const SamplerProperties& sampler_props,
    const VkImageLayout layout,
    VkSampler sampler,
    const bool allocate_memory)
    // 使用成员初始化列表初始化类的成员变量
    : image_properties_(image_props),
      view_properties_(view_props),
      sampler_properties_(sampler_props),
      allocator_(vma_allocator),
      memory_{},  // 初始化 memory_ 为默认构造的值
      owns_memory_{allocate_memory},  // 初始化 owns_memory_ 为 allocate_memory 的值
      handles_{  // 初始化 handles_ 的三个成员，最后一个成员 sampler 为给定的值
          VK_NULL_HANDLE,  // 第一个成员初始化为 VK_NULL_HANDLE
          VK_NULL_HANDLE,  // 第二个成员初始化为 VK_NULL_HANDLE
          sampler,         // 第三个成员初始化为 sampler 的值
      },
      layout_(layout)  // 初始化 layout_ 为给定的 layout 的值
    {
      // 获取分配器的信息
      VmaAllocatorInfo allocator_info{};
      vmaGetAllocatorInfo(allocator_, &allocator_info);
    
      // 如果图像的任何维度为零，则不会为图像分配内存
      if (image_props.image_extents.width == 0 ||
          image_props.image_extents.height == 0 ||
          image_props.image_extents.depth == 0) {
        return;  // 如果有任何一个维度为零，直接返回，不继续执行后续代码
      }
    
      // 创建图像的 Vulkan 图像创建信息结构体
      const VkImageCreateInfo image_create_info{
          VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,  // sType
          nullptr,                              // pNext
          0u,                                   // flags
          image_properties_.image_type,          // imageType
          image_properties_.image_format,        // format
          image_properties_.image_extents,       // extents
          1u,                                   // mipLevels
          1u,                                   // arrayLayers
          VK_SAMPLE_COUNT_1_BIT,                 // samples
          VK_IMAGE_TILING_OPTIMAL,               // tiling
          image_properties_.image_usage,         // usage
          VK_SHARING_MODE_EXCLUSIVE,             // sharingMode
          0u,                                   // queueFamilyIndexCount
          nullptr,                              // pQueueFamilyIndices
          layout_,                              // initialLayout
      };
    
      // 将分配信息设置给 memory_ 的 create_info 成员
      memory_.create_info = allocation_create_info;
    
      if (allocate_memory) {
        // 使用 VMA 分配图像内存和创建图像的图像视图
        VK_CHECK(vmaCreateImage(
            allocator_,
            &image_create_info,
            &allocation_create_info,
            &(handles_.image),
            &(memory_.allocation),
            nullptr));
        // 只有当图像已经绑定到内存时才创建图像视图
        create_image_view();
      } else {
        // 直接使用 Vulkan API 创建图像（不使用 VMA）
        VK_CHECK(vkCreateImage(
            allocator_info.device, &image_create_info, nullptr, &(handles_.image)));
      }
}

VulkanImage::VulkanImage(VulkanImage&& other) noexcept
    : image_properties_(other.image_properties_),  // 初始化图像属性
      view_properties_(other.view_properties_),    // 初始化视图属性
      sampler_properties_(other.sampler_properties_),  // 初始化采样器属性
      allocator_(other.allocator_),                // 初始化分配器
      memory_(std::move(other.memory_)),           // 移动内存资源
      owns_memory_(other.owns_memory_),            // 拥有内存标志
      handles_(other.handles_),                    // 处理句柄
      layout_(other.layout_) {                     // 初始化布局
  other.handles_.image = VK_NULL_HANDLE;           // 将原对象的图像句柄置为空
  other.handles_.image_view = VK_NULL_HANDLE;      // 将原对象的图像视图句柄置为空
  other.handles_.sampler = VK_NULL_HANDLE;         // 将原对象的采样器句柄置为空
  other.owns_memory_ = false;                      // 将原对象的内存所有权标志设为假
}

VulkanImage& VulkanImage::operator=(VulkanImage&& other) noexcept {
  VkImage tmp_image = handles_.image;              // 保存当前对象的图像句柄
  VkImageView tmp_image_view = handles_.image_view;  // 保存当前对象的图像视图句柄
  bool tmp_owns_memory = owns_memory_;             // 保存当前对象的内存所有权标志

  image_properties_ = other.image_properties_;     // 赋值图像属性
  view_properties_ = other.view_properties_;       // 赋值视图属性
  sampler_properties_ = other.sampler_properties_; // 赋值采样器属性
  allocator_ = other.allocator_;                   // 赋值分配器
  memory_ = std::move(other.memory_);              // 移动内存资源
  owns_memory_ = other.owns_memory_;               // 赋值内存所有权标志
  handles_ = other.handles_;                       // 赋值处理句柄
  layout_ = other.layout_;                         // 赋值布局

  other.handles_.image = tmp_image;                // 将原对象的图像句柄恢复为当前对象的图像句柄
  other.handles_.image_view = tmp_image_view;      // 将原对象的图像视图句柄恢复为当前对象的图像视图句柄
  other.owns_memory_ = tmp_owns_memory;            // 将原对象的内存所有权标志恢复为当前对象的内存所有权标志

  return *this;                                    // 返回当前对象的引用
}

VulkanImage::~VulkanImage() {
  if (VK_NULL_HANDLE != handles_.image_view) {     // 如果图像视图句柄不为空
    vkDestroyImageView(this->device(), handles_.image_view, nullptr);  // 销毁图像视图
  }

  if (VK_NULL_HANDLE != handles_.image) {          // 如果图像句柄不为空
    if (owns_memory_) {                            // 如果当前对象拥有内存
      vmaDestroyImage(allocator_, handles_.image, memory_.allocation);  // 使用VMA销毁图像
    } else {
      vkDestroyImage(this->device(), handles_.image, nullptr);  // 否则使用Vulkan API销毁图像
    }
    // 防止底层内存分配被释放；这可能是通过vmaDestroyImage释放的，或者这个资源不拥有底层内存
    memory_.allocation = VK_NULL_HANDLE;           // 将内存分配句柄置为空
  }
}

void VulkanImage::create_image_view() {
  VmaAllocatorInfo allocator_info{};               // 分配器信息结构体
  vmaGetAllocatorInfo(allocator_, &allocator_info);  // 获取分配器信息

  const VkComponentMapping component_mapping{
      VK_COMPONENT_SWIZZLE_IDENTITY,               // 红色通道映射
      VK_COMPONENT_SWIZZLE_IDENTITY,               // 绿色通道映射
      VK_COMPONENT_SWIZZLE_IDENTITY,               // 蓝色通道映射
      VK_COMPONENT_SWIZZLE_IDENTITY,               // Alpha通道映射
  };

  const VkImageSubresourceRange subresource_range{
      VK_IMAGE_ASPECT_COLOR_BIT,                   // 图像资源方面掩码
      0u,                                           // 基本Mip层级
      VK_REMAINING_MIP_LEVELS,                     // 层级数
      0u,                                           // 基本数组层
      VK_REMAINING_ARRAY_LAYERS,                   // 数组层级
  };

  const VkImageViewCreateInfo image_view_create_info{
      VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,    // 结构类型
      nullptr,                                      // pNext
      0u,                                           // 标志
      handles_.image,                              // 图像
      view_properties_.view_type,                  // 视图类型
      view_properties_.view_format,                // 格式
      component_mapping,                           // 组件映射
      subresource_range,                           // 子资源范围
  };

  VK_CHECK(vkCreateImageView(
      allocator_info.device,
      &(image_view_create_info),
      nullptr,
      &(handles_.image_view)));                    // 创建图像视图
}
// 获取 Vulkan 图像对象的内存需求信息
VkMemoryRequirements VulkanImage::get_memory_requirements() const {
    VkMemoryRequirements memory_requirements;
    // 调用 Vulkan API 获取图像对象的内存需求
    vkGetImageMemoryRequirements(this->device(), handles_.image, &memory_requirements);
    return memory_requirements; // 返回内存需求结构体
}

//
// ImageMemoryBarrier
//

// ImageMemoryBarrier 类的构造函数
ImageMemoryBarrier::ImageMemoryBarrier(
    const VkAccessFlags src_access_flags,
    const VkAccessFlags dst_access_flags,
    const VkImageLayout src_layout_flags,
    const VkImageLayout dst_layout_flags,
    const VulkanImage& image)
    : handle{
          VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, // sType，指定结构体类型
          nullptr, // pNext，指向扩展结构体的指针，这里为nullptr
          src_access_flags, // srcAccessMask，源访问标记
          dst_access_flags, // dstAccessMask，目标访问标记
          src_layout_flags, // oldLayout，原始布局
          dst_layout_flags, // newLayout，目标布局
          VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex，源队列族索引
          VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex，目标队列族索引
          image.handles_.image, // image，关联的 Vulkan 图像句柄
          {
              // subresourceRange 子资源范围
              VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask，图像方面掩码
              0u, // baseMipLevel，基础 mip 层级
              VK_REMAINING_MIP_LEVELS, // levelCount，mip 层级数量
              0u, // baseArrayLayer，基础数组层
              VK_REMAINING_ARRAY_LAYERS, // layerCount，数组层数量
          },
      } {}

//
// SamplerCache
//

// SamplerCache 类的构造函数，使用给定的 VkDevice
SamplerCache::SamplerCache(VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

// 移动构造函数，使用给定的 VkDevice 和其他实例的缓存
SamplerCache::SamplerCache(SamplerCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_), cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
}

// 析构函数，清理缓存
SamplerCache::~SamplerCache() {
  purge(); // 清理缓存
}

// 检索缓存中的 VkSampler 对象
VkSampler SamplerCache::retrieve(const SamplerCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if (cache_.cend() == it) {
    // 如果缓存中不存在该键，则插入新的 SamplerCache::Value 实例
    it = cache_.insert({key, SamplerCache::Value(device_, key)}).first;
  }

  return it->second.handle(); // 返回缓存中的 VkSampler 句柄
}

// 清空缓存
void SamplerCache::purge() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_.clear(); // 清空缓存
}

//
// MemoryAllocator
//

// MemoryAllocator 类的构造函数，初始化 Vulkan 内存分配器
MemoryAllocator::MemoryAllocator(
    VkInstance instance,
    VkPhysicalDevice physical_device,
    VkDevice device)
    : instance_{},
      physical_device_(physical_device),
      device_(device),
      allocator_{VK_NULL_HANDLE} {
  VmaVulkanFunctions vk_functions{};
  vk_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr; // Vulkan 实例获取函数
  vk_functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr; // Vulkan 设备获取函数

  // 初始化 Vulkan 内存分配器的创建信息结构体
  const VmaAllocatorCreateInfo allocator_create_info{
      0u, // flags，标志位
      physical_device_, // physicalDevice，物理设备句柄
      device_, // device，逻辑设备句柄
      0u, // preferredLargeHeapBlockSize，优选的大堆块大小
      nullptr, // pAllocationCallbacks，分配回调
      nullptr, // pDeviceMemoryCallbacks，设备内存回调
      nullptr, // pHeapSizeLimit，堆大小限制
      &vk_functions, // pVulkanFunctions，Vulkan 函数指针
      instance, // instance，Vulkan 实例句柄
      VK_API_VERSION_1_0, // vulkanApiVersion，Vulkan API 版本
      nullptr, // pTypeExternalMemoryHandleTypes，外部内存类型句柄
  };

  VK_CHECK(vmaCreateAllocator(&allocator_create_info, &allocator_)); // 创建 Vulkan 内存分配器
}
// 实现移动构造函数，使用 noexcept 以确保该操作不会抛出异常
MemoryAllocator::MemoryAllocator(MemoryAllocator&& other) noexcept
    : instance_(other.instance_),           // 移动其他实例的成员到当前实例
      physical_device_(other.physical_device_), // 移动物理设备句柄
      device_(other.device_),               // 移动设备句柄
      allocator_(other.allocator_) {        // 移动分配器句柄
  other.allocator_ = VK_NULL_HANDLE;        // 将移动源的分配器句柄置为无效
  other.device_ = VK_NULL_HANDLE;           // 将移动源的设备句柄置为无效
  other.physical_device_ = VK_NULL_HANDLE;  // 将移动源的物理设备句柄置为无效
  other.instance_ = VK_NULL_HANDLE;         // 将移动源的实例句柄置为无效
}

// 析构函数，用于销毁分配器对象
MemoryAllocator::~MemoryAllocator() {
  if (VK_NULL_HANDLE == allocator_) {       // 检查分配器句柄是否有效
    return;                                 // 若无效则直接返回
  }
  vmaDestroyAllocator(allocator_);          // 销毁 Vulkan Memory Allocator 对象
}

// 创建内存分配的方法
MemoryAllocation MemoryAllocator::create_allocation(
    const VkMemoryRequirements& memory_requirements,
    const VmaAllocationCreateInfo& create_info) {
  VmaAllocationCreateInfo alloc_create_info = create_info;  // 复制传入的创建信息
  // 针对使用 VMA_MEMORY_USAGE_AUTO_* 标志直接分配内存的情况进行保护
  switch (create_info.usage) {
    // 阻止直接分配时使用以下复杂的使用选项
    case VMA_MEMORY_USAGE_AUTO:
    case VMA_MEMORY_USAGE_AUTO_PREFER_HOST:
      VK_THROW(
          "Only the VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE usage flag is compatible with create_allocation()");
      break;
    // 大多数情况下，VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE 将设置 DEVICE_LOCAL_BIT 作为首选内存标志
    case VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE:
      alloc_create_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      alloc_create_info.usage = VMA_MEMORY_USAGE_UNKNOWN;
      break;
    default:
      break;
  }

  return MemoryAllocation(allocator_, memory_requirements, alloc_create_info);  // 返回内存分配对象
}

// 创建图像对象的方法
VulkanImage MemoryAllocator::create_image(
    const VkExtent3D& extents,
    const VkFormat image_format,
    const VkImageType image_type,
    const VkImageViewType image_view_type,
    const VulkanImage::SamplerProperties& sampler_props,
    VkSampler sampler,
    const bool allow_transfer,
    const bool allocate_memory) {
  VkImageUsageFlags usage =
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;  // 设置图像使用标志

  if (allow_transfer) {
    usage |=
        (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);  // 若允许传输，则添加传输标志
  }

  VmaAllocationCreateInfo alloc_create_info = {};  // 初始化 Vulkan Memory Allocator 创建信息结构体
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY;  // 设置默认分配策略标志
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;  // 设置内存使用标志为优先设备本地内存

  const VulkanImage::ImageProperties image_props{  // 初始化图像属性结构体
      image_type,
      image_format,
      extents,
      usage,
  };

  const VulkanImage::ViewProperties view_props{  // 初始化图像视图属性结构体
      image_view_type,
      image_format,
  };

  const VkImageLayout initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;  // 设置初始布局为未定义状态

  return VulkanImage(  // 返回创建的 Vulkan 图像对象
      allocator_,
      alloc_create_info,
      image_props,
      view_props,
      sampler_props,
      initial_layout,
      sampler,
      allocate_memory);
}
// 创建存储缓冲区函数，根据给定的大小、GPU专用标志和内存分配标志创建 VulkanBuffer 对象
VulkanBuffer MemoryAllocator::create_storage_buffer(
    const VkDeviceSize size,
    const bool gpu_only,
    const bool allocate_memory) {
  // 设置缓冲区使用标志为存储缓冲区位
  const VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  // 初始化内存分配创建信息
  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  // 如果不是仅 GPU 使用，则设置适当的标志以表明主机设备将从此缓冲区访问数据
  if (!gpu_only) {
    // 如果需要分配内存，则检查是否仅 GPU 使用的缓冲区
    VK_CHECK_COND(
        allocate_memory,
        "Only GPU-only buffers should use deferred memory allocation");

    // 设置分配标志为允许主机随机访问
    alloc_create_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    // 设置内存使用模式为自动优先主机
    alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    // 设置所需的内存属性标志为主机可见
    alloc_create_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    // 设置首选的内存属性标志为主机一致性和主机缓存
    alloc_create_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }

  // 创建并返回 VulkanBuffer 对象，用于存储缓冲区
  return VulkanBuffer(
      allocator_, size, alloc_create_info, buffer_usage, allocate_memory);
}

// 创建分段缓冲区函数，根据给定的大小创建 VulkanBuffer 对象
VulkanBuffer MemoryAllocator::create_staging_buffer(const VkDeviceSize size) {
  // 初始化内存分配创建信息
  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY;
  // 设置内存使用模式为自动优先主机
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

  // 设置缓冲区使用标志为传输源和传输目的地位
  VkBufferUsageFlags buffer_usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  // 创建并返回 VulkanBuffer 对象，用于分段缓冲区
  return VulkanBuffer(allocator_, size, alloc_create_info, buffer_usage);
}

// 创建统一缓冲区函数，根据给定的大小创建 VulkanBuffer 对象
VulkanBuffer MemoryAllocator::create_uniform_buffer(const VkDeviceSize size) {
  // 初始化内存分配创建信息
  VmaAllocationCreateInfo alloc_create_info = {};
  // 设置分配策略标志和主机顺序写访问标志
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY |
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  // 设置内存使用模式为自动
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;

  // 设置缓冲区使用标志为统一缓冲区位
  VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

  // 创建 VulkanBuffer 对象并返回，用于统一缓冲区
  VulkanBuffer uniform_buffer(
      allocator_, size, alloc_create_info, buffer_usage);
  return uniform_buffer;
}

//
// VulkanFence
//

// VulkanFence 类的默认构造函数，初始化设备句柄、 Vulkan 标记和等待标记
VulkanFence::VulkanFence()
    : device_(VK_NULL_HANDLE), handle_(VK_NULL_HANDLE), waiting_(false) {}

// VulkanFence 类的构造函数，接收 Vulkan 设备句柄并创建 VulkanFence 对象
VulkanFence::VulkanFence(VkDevice device)
    : device_(device), handle_(VK_NULL_HANDLE), waiting_(VK_NULL_HANDLE) {
  // 初始化 VulkanFence 的创建信息
  const VkFenceCreateInfo fence_create_info{
      VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
  };

  // 调用 Vulkan API 创建 VulkanFence 对象，并将句柄存储在 handle_ 成员变量中
  VK_CHECK(vkCreateFence(device_, &fence_create_info, nullptr, &handle_));
}

// VulkanFence 类的移动构造函数，使用右值引用，确保资源移动正确
VulkanFence::VulkanFence(VulkanFence&& other) noexcept
    : device_(other.device_), handle_(other.handle_), waiting_(other.waiting_) {
  // 将其他对象的句柄置为空，等待标志置为 false
  other.handle_ = VK_NULL_HANDLE;
  other.waiting_ = false;
}
// VulkanFence 类的移动赋值运算符重载函数的实现，使用了移动语义和 noexcept 声明
VulkanFence& VulkanFence::operator=(VulkanFence&& other) noexcept {
  // 将其他对象的成员变量移动到当前对象
  device_ = other.device_;
  handle_ = other.handle_;
  waiting_ = other.waiting_;

  // 清空其他对象的成员变量，确保移动语义的正确性
  other.device_ = VK_NULL_HANDLE;
  other.handle_ = VK_NULL_HANDLE;
  other.waiting_ = false;

  return *this;  // 返回当前对象的引用
}

// VulkanFence 类的析构函数实现
VulkanFence::~VulkanFence() {
  // 如果 Vulkan 句柄为空，则直接返回，无需进行销毁操作
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  // 销毁 Vulkan Fence 对象
  vkDestroyFence(device_, handle_, nullptr);
}

// VulkanFence 类的 wait 方法实现
void VulkanFence::wait() {
  // 如果 waiting_ 标志为 true，则执行等待操作
  if (waiting_) {
    VkResult fence_status = VK_NOT_READY;
    // 使用循环来保持 CPU 活跃性，因为单次调用 vkWaitForFences 可能会导致线程被调度出去
    do {
      // 执行等待操作，超时时间为 100000 纳秒
      fence_status = vkWaitForFences(device_, 1u, &handle_, VK_TRUE, 100000);

      // 检查等待操作中是否出现设备丢失错误
      VK_CHECK_COND(
          fence_status != VK_ERROR_DEVICE_LOST,
          "Vulkan Fence: Device lost while waiting for fence!");
    } while (fence_status != VK_SUCCESS);

    // 等待操作完成后重置 Fence 对象
    VK_CHECK(vkResetFences(device_, 1u, &handle_));

    // 设置 waiting_ 标志为 false，表示等待操作已完成
    waiting_ = false;
  }
}
```