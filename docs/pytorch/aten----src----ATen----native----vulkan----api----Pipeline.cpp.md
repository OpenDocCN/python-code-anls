# `.\pytorch\aten\src\ATen\native\vulkan\api\Pipeline.cpp`

```py
// 包含 Vulkan API 的 Pipeline 头文件

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// 实用函数
//

// 将内存访问阶段和内存访问类型映射到 Vulkan 访问标志
VkAccessFlags vk_access(
    const PipelineStageFlags stage,         // 管道阶段标志
    const MemoryAccessFlags access) {       // 内存访问类型标志
  VkAccessFlags vk_access = 0u;

  // 如果访问类型包含读取标志
  if (access & MemoryAccessType::READ) {
    // 如果管道阶段包含计算阶段
    if (stage & PipelineStage::COMPUTE) {
      vk_access |= VK_ACCESS_SHADER_READ_BIT;  // 添加 Vulkan 的着色器读取标志
    }

    // 如果管道阶段包含主机阶段
    if (stage & PipelineStage::HOST) {
      vk_access |= VK_ACCESS_HOST_READ_BIT;    // 添加 Vulkan 的主机读取标志
    }

    // 如果管道阶段包含传输阶段
    if (stage & PipelineStage::TRANSFER) {
      vk_access |= VK_ACCESS_TRANSFER_READ_BIT;  // 添加 Vulkan 的传输读取标志
    }
  }

  // 如果访问类型包含写入标志
  if (access & MemoryAccessType::WRITE) {
    // 如果管道阶段包含计算阶段
    if (stage & PipelineStage::COMPUTE) {
      vk_access |= VK_ACCESS_SHADER_WRITE_BIT;  // 添加 Vulkan 的着色器写入标志
    }

    // 如果管道阶段包含主机阶段
    if (stage & PipelineStage::HOST) {
      vk_access |= VK_ACCESS_HOST_WRITE_BIT;    // 添加 Vulkan 的主机写入标志
    }

    // 如果管道阶段包含传输阶段
    if (stage & PipelineStage::TRANSFER) {
      vk_access |= VK_ACCESS_TRANSFER_WRITE_BIT;  // 添加 Vulkan 的传输写入标志
    }
  }

  return vk_access;  // 返回组合后的 Vulkan 访问标志
}

// 将管道阶段映射到 Vulkan 管道阶段标志
VkPipelineStageFlags vk_stage(const PipelineStageFlags stage) {  // 管道阶段标志
  VkPipelineStageFlags vk_stage = 0u;

  // 如果管道阶段包含计算阶段
  if (stage & PipelineStage::COMPUTE) {
    vk_stage |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;  // 添加 Vulkan 的计算着色器阶段标志
  }

  // 如果管道阶段包含主机阶段
  if (stage & PipelineStage::HOST) {
    vk_stage |= VK_PIPELINE_STAGE_HOST_BIT;  // 添加 Vulkan 的主机阶段标志
  }

  // 如果管道阶段包含传输阶段
  if (stage & PipelineStage::TRANSFER) {
    vk_stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;  // 添加 Vulkan 的传输阶段标志
  }

  return vk_stage;  // 返回组合后的 Vulkan 管道阶段标志
}

// 将管道阶段和内存访问类型映射到适当的 Vulkan 图像布局
VkImageLayout vk_layout(
    const PipelineStageFlags stage,         // 管道阶段标志
    const MemoryAccessFlags access) {       // 内存访问类型标志
  switch (stage) {
    case PipelineStage::COMPUTE:
      switch (access) {
        case MemoryAccessType::READ:
          return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;  // 返回 Vulkan 的着色器只读最优布局
        default:
          return VK_IMAGE_LAYOUT_GENERAL;  // 返回 Vulkan 的一般布局
      }
      break;
    case PipelineStage::TRANSFER:
      switch (access) {
        case MemoryAccessType::READ:
          return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;  // 返回 Vulkan 的传输源最优布局
        case MemoryAccessType::WRITE:
          return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;  // 返回 Vulkan 的传输目标最优布局
        default:
          VK_THROW("Invalid memory access type for transfer stage!");  // 抛出异常，传输阶段使用无效的内存访问类型
      }
      break;
    default:
      VK_THROW("Cannot determine appropriate image layout");  // 抛出异常，无法确定适当的图像布局
  }

  return VK_IMAGE_LAYOUT_UNDEFINED;  // 默认返回 Vulkan 的未定义布局
}

//
// PipelineLayout 类
//

// Vulkan 管道布局的构造函数
PipelineLayout::PipelineLayout(
    VkDevice device,                       // Vulkan 设备句柄
    VkDescriptorSetLayout descriptor_layout)  // Vulkan 描述符集布局句柄
    : device_(device), handle_{VK_NULL_HANDLE} {
  // TODO: 启用推送常量
  // 定义 Vulkan 管道布局创建信息结构体
  const VkPipelineLayoutCreateInfo pipeline_layout_create_info{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,  // sType
      nullptr,                                        // pNext
      0u,                                             // flags
      1u,                                             // setLayoutCount
      &descriptor_layout,                             // pSetLayouts
      0u,                                             // pushConstantRangeCount
      nullptr,                                        // pPushConstantRanges
  };

  // 创建 Vulkan 管道布局对象
  VK_CHECK(vkCreatePipelineLayout(
      device_, &pipeline_layout_create_info, nullptr, &handle_));
}

// 移动构造函数，使用 noexcept 说明不抛出异常
PipelineLayout::PipelineLayout(PipelineLayout&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;  // 设置其他对象的句柄为 Vulkan 的空句柄
}
// 析构函数：销毁管线布局对象，如果对象句柄为 VK_NULL_HANDLE 则直接返回
PipelineLayout::~PipelineLayout() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  // 调用 Vulkan API 销毁管线布局对象
  vkDestroyPipelineLayout(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

// 交换函数：交换两个 PipelineLayout 对象的设备和句柄
void swap(PipelineLayout& lhs, PipelineLayout& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkPipelineLayout tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// ComputePipeline
//

// 构造函数：创建计算管线对象，接收设备、描述符和管线缓存作为参数
ComputePipeline::ComputePipeline(
    VkDevice device,
    const ComputePipeline::Descriptor& descriptor,
    VkPipelineCache pipeline_cache)
    : device_(device), handle_{VK_NULL_HANDLE} {
  // NOLINTNEXTLINE
  // 定义用于特化着色器的特化常量映射条目数组
  constexpr VkSpecializationMapEntry specialization_map_entries[3]{
      // X
      {
          0u,
          offsetof(utils::uvec3, data[0u]),
          sizeof(utils::uvec3::data[0u]),
      },
      // Y
      {
          1u,
          offsetof(utils::uvec3, data[1u]),
          sizeof(utils::uvec3::data[1u]),
      },
      // Z
      {
          2u,
          offsetof(utils::uvec3, data[2u]),
          sizeof(utils::uvec3::data[2u]),
      },
  };

  // 定义特化信息结构体，包含特化映射条目和本地工作组数据
  const VkSpecializationInfo specialization_info{
      3u, // mapEntryCount
      specialization_map_entries, // pMapEntries
      sizeof(descriptor.local_work_group), // dataSize
      &descriptor.local_work_group, // pData
  };

  // 定义着色器阶段创建信息结构体，描述计算着色器阶段的相关信息
  const VkPipelineShaderStageCreateInfo shader_stage_create_info{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      VK_SHADER_STAGE_COMPUTE_BIT, // stage
      descriptor.shader_module, // module
      "main", // pName
      &specialization_info, // pSpecializationInfo
  };

  // 定义计算管线创建信息结构体，描述计算管线的相关信息
  const VkComputePipelineCreateInfo compute_pipeline_create_info{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      shader_stage_create_info, // stage
      descriptor.pipeline_layout, // layout
      VK_NULL_HANDLE, // basePipelineHandle
      0u, // basePipelineIndex
  };

  // 调用 Vulkan API 创建计算管线对象，存储句柄至 handle_
  VK_CHECK(vkCreateComputePipelines(
      device_,
      pipeline_cache,
      1u,
      &compute_pipeline_create_info,
      nullptr,
      &handle_));
}

// 移动构造函数：移动构造函数，将另一个 ComputePipeline 对象的设备和句柄移动至当前对象
ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

// 析构函数：销毁计算管线对象，如果对象句柄为 VK_NULL_HANDLE 则直接返回
ComputePipeline::~ComputePipeline() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  // 调用 Vulkan API 销毁计算管线对象
  vkDestroyPipeline(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

// 交换函数：交换两个 ComputePipeline 对象的设备和句柄
void swap(ComputePipeline& lhs, ComputePipeline& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkPipeline tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

// 比较运算符重载：用于比较两个 ComputePipeline::Descriptor 对象是否相等
bool operator==(const ComputePipeline::Descriptor& _1,
    // 定义函数并声明参数为 _1 和 _2，返回一个布尔值
    const ComputePipeline::Descriptor& _2) {
  // 返回一个布尔表达式，检查 _1 的 pipeline_layout、shader_module 和 local_work_group 是否与 _2 相等
  return (
      _1.pipeline_layout == _2.pipeline_layout &&
      _1.shader_module == _2.shader_module &&
      _1.local_work_group == _2.local_work_group);
} // 关闭 ComputePipelineCache 类的命名空间

//
// PipelineLayoutCache
//

PipelineLayoutCache::PipelineLayoutCache(VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}  // PipelineLayoutCache 构造函数初始化设备和缓存

PipelineLayoutCache::PipelineLayoutCache(PipelineLayoutCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_), cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
}  // PipelineLayoutCache 移动构造函数，使用互斥锁锁住其他缓存

PipelineLayoutCache::~PipelineLayoutCache() {
  purge();  // 调用 purge() 方法清空缓存
}  // PipelineLayoutCache 析构函数，清理资源

VkPipelineLayout PipelineLayoutCache::retrieve(
    const PipelineLayoutCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);  // 使用互斥锁锁住缓存

  auto it = cache_.find(key);  // 在缓存中查找给定的键值

  if (cache_.cend() == it) {  // 如果未找到对应键值的缓存条目
    it = cache_.insert({key, PipelineLayoutCache::Value(device_, key)}).first;  // 插入新的缓存条目
  }

  return it->second.handle();  // 返回找到或新创建的缓存条目的句柄
}

void PipelineLayoutCache::purge() {
  std::lock_guard<std::mutex> lock(cache_mutex_);  // 使用互斥锁锁住缓存
  cache_.clear();  // 清空缓存
}

//
// ComputePipelineCache
//

ComputePipelineCache::ComputePipelineCache(VkDevice device)
    : cache_mutex_{},
      device_(device),
      pipeline_cache_{VK_NULL_HANDLE},
      cache_{} {
  const VkPipelineCacheCreateInfo pipeline_cache_create_info{
      VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      0u, // initialDataSize
      nullptr, // pInitialData
  };

  VK_CHECK(vkCreatePipelineCache(
      device, &pipeline_cache_create_info, nullptr, &pipeline_cache_));
}  // ComputePipelineCache 构造函数，创建管线缓存对象并初始化

ComputePipelineCache::ComputePipelineCache(
    ComputePipelineCache&& other) noexcept
    : cache_mutex_{},
      device_(other.device_),
      pipeline_cache_(other.pipeline_cache_),
      cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);

  other.pipeline_cache_ = VK_NULL_HANDLE;
}  // ComputePipelineCache 移动构造函数，使用互斥锁锁住其他缓存

ComputePipelineCache::~ComputePipelineCache() {
  purge();  // 调用 purge() 方法清空缓存

  if (VK_NULL_HANDLE == pipeline_cache_) {
    return;
  }
  vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);  // 销毁管线缓存对象
  pipeline_cache_ = VK_NULL_HANDLE;
}  // ComputePipelineCache 析构函数，清理资源

VkPipeline ComputePipelineCache::retrieve(
    const ComputePipelineCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);  // 使用互斥锁锁住缓存

  auto it = cache_.find(key);  // 在缓存中查找给定的键值

  if (cache_.cend() == it) {  // 如果未找到对应键值的缓存条目
    it = cache_
             .insert(
                 {key,
                  ComputePipelineCache::Value(device_, key, pipeline_cache_)})
             .first;  // 插入新的缓存条目
  }

  return it->second.handle();  // 返回找到或新创建的缓存条目的句柄
}

void ComputePipelineCache::purge() {
  cache_.clear();  // 清空缓存
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
```