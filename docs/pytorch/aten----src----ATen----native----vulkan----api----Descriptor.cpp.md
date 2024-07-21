# `.\pytorch\aten\src\ATen\native\vulkan\api\Descriptor.cpp`

```py
// 包含Vulkan的描述符相关头文件
#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/Utils.h>

#include <algorithm>
#include <utility>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// DescriptorSet
//

// 构造函数：初始化描述符集对象
DescriptorSet::DescriptorSet(
    VkDevice device,
    VkDescriptorSet handle,
    ShaderLayout::Signature shader_layout_signature)
    : device_(device),
      handle_(handle),
      shader_layout_signature_(std::move(shader_layout_signature)),
      bindings_{} {}

// 移动构造函数：从另一个描述符集对象移动构造
DescriptorSet::DescriptorSet(DescriptorSet&& other) noexcept
    : device_(other.device_),
      handle_(other.handle_),
      shader_layout_signature_(std::move(other.shader_layout_signature_)),
      bindings_(std::move(other.bindings_)) {
  other.handle_ = VK_NULL_HANDLE;
}

// 移动赋值运算符：从另一个描述符集对象移动赋值
DescriptorSet& DescriptorSet::operator=(DescriptorSet&& other) noexcept {
  device_ = other.device_;
  handle_ = other.handle_;
  shader_layout_signature_ = std::move(other.shader_layout_signature_);
  bindings_ = std::move(other.bindings_);

  other.handle_ = VK_NULL_HANDLE;

  return *this;
}

// 绑定缓冲资源到描述符集中的指定索引
DescriptorSet& DescriptorSet::bind(
    const uint32_t idx,
    const VulkanBuffer& buffer) {
  // 检查缓冲是否已经绑定到内存，否则无法使用
  VK_CHECK_COND(
      buffer.has_memory(),
      "Buffer must be bound to memory for it to be usable");

  // 创建资源绑定对象
  DescriptorSet::ResourceBinding binder{};
  binder.binding_idx = idx; // 绑定的索引
  binder.descriptor_type = shader_layout_signature_[idx]; // 描述符类型
  binder.is_image = false; // 标识为缓冲类型
  binder.resource_info.buffer_info.buffer = buffer.handle(); // 缓冲句柄
  binder.resource_info.buffer_info.offset = buffer.mem_offset(); // 缓冲偏移量
  binder.resource_info.buffer_info.range = buffer.mem_range(); // 缓冲数据范围
  add_binding(binder);

  return *this;
}

// 绑定图像资源到描述符集中的指定索引
DescriptorSet& DescriptorSet::bind(
    const uint32_t idx,
    const VulkanImage& image) {
  // 检查图像是否已经绑定到内存，否则无法使用
  VK_CHECK_COND(
      image.has_memory(), "Image must be bound to memory for it to be usable");

  // 确定绑定时的图像布局
  VkImageLayout binding_layout = image.layout();
  if (shader_layout_signature_[idx] == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
    binding_layout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // 创建资源绑定对象
  DescriptorSet::ResourceBinding binder{};
  binder.binding_idx = idx; // 绑定的索引
  binder.descriptor_type = shader_layout_signature_[idx]; // 描述符类型
  binder.is_image = true; // 标识为图像类型
  binder.resource_info.image_info.sampler = image.sampler(); // 采样器
  binder.resource_info.image_info.imageView = image.image_view(); // 图像视图
  binder.resource_info.image_info.imageLayout = binding_layout; // 图像布局
  add_binding(binder);

  return *this;
}

// 获取描述符集的句柄
VkDescriptorSet DescriptorSet::get_bind_handle() const {
  std::vector<VkWriteDescriptorSet> write_descriptor_sets;

  // 遍历所有绑定的资源，并生成写入描述符集的对象
  for (const ResourceBinding& binding : bindings_) {
    # 创建一个 VkWriteDescriptorSet 结构体对象，并初始化其各个字段
    VkWriteDescriptorSet write{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // sType，指定结构体类型
        nullptr, // pNext，扩展信息（当前为nullptr）
        handle_, // dstSet，目标描述符集合句柄
        binding.binding_idx, // dstBinding，目标绑定索引
        0u, // dstArrayElement，目标数组元素索引
        1u, // descriptorCount，描述符数量
        binding.descriptor_type, // descriptorType，描述符类型
        nullptr, // pImageInfo，图像信息（初始为nullptr）
        nullptr, // pBufferInfo，缓冲信息（初始为nullptr）
        nullptr, // pTexelBufferView，缓冲视图（初始为nullptr）
    };
    
    # 如果当前绑定的资源是图像，则将 write 结构体中的 pImageInfo 指向对应的图像信息
    if (binding.is_image) {
        write.pImageInfo = &binding.resource_info.image_info;
    } else {
        # 如果当前绑定的资源不是图像，则将 write 结构体中的 pBufferInfo 指向对应的缓冲信息
        write.pBufferInfo = &binding.resource_info.buffer_info;
    }
    
    # 将构建好的 write 结构体添加到 write_descriptor_sets 向量中
    write_descriptor_sets.emplace_back(write);
    }
    
    # 调用 Vulkan API 更新描述符集合
    vkUpdateDescriptorSets(
        device_, // Vulkan 设备句柄
        write_descriptor_sets.size(), // 要更新的描述符集合数量
        write_descriptor_sets.data(), // 指向描述符集合数组的指针
        0u, // 要更新的起始索引
        nullptr // pNext，扩展信息（当前为nullptr）
    );
    
    # 将描述符集合句柄作为返回值返回
    VkDescriptorSet ret = handle_;
    return ret;
}

// 向描述符集合中添加绑定
void DescriptorSet::add_binding(const ResourceBinding& binding) {
  // 在绑定列表中查找是否已存在与新绑定相同索引的绑定
  const auto bindings_itr = std::find_if(
      bindings_.begin(),
      bindings_.end(),
      [binding_idx = binding.binding_idx](const ResourceBinding& other) {
        return other.binding_idx == binding_idx;
      });

  // 如果未找到相同索引的绑定，则添加新的绑定
  if (bindings_.end() == bindings_itr) {
    bindings_.emplace_back(binding);
  } else {
    // 如果找到相同索引的绑定，则替换它
    *bindings_itr = binding;
  }
}

//
// DescriptorSetPile
//

// 描述符集合堆的构造函数
DescriptorSetPile::DescriptorSetPile(
    const uint32_t pile_size,
    VkDescriptorSetLayout descriptor_set_layout,
    VkDevice device,
    VkDescriptorPool descriptor_pool)
    : pile_size_{pile_size},
      set_layout_{descriptor_set_layout},
      device_{device},
      pool_{descriptor_pool},
      descriptors_{},
      in_use_(0u) {
  // 调整描述符集合的大小为堆大小
  descriptors_.resize(pile_size_);
  // 分配新的批次描述符集合
  allocate_new_batch();
}

// 获取描述符集合的函数
VkDescriptorSet DescriptorSetPile::get_descriptor_set() {
  // 如果仍有可用的描述符集合，则不执行任何操作
  allocate_new_batch();

  // 获取当前正在使用的描述符集合的句柄
  VkDescriptorSet handle = descriptors_[in_use_];
  // 将当前正在使用的描述符集合句柄置为 VK_NULL_HANDLE
  descriptors_[in_use_] = VK_NULL_HANDLE;

  // 增加正在使用的描述符集合索引
  in_use_++;
  return handle;
}

// 分配新批次描述符集合的函数
void DescriptorSetPile::allocate_new_batch() {
  // 如果仍有可用的描述符集合，并且当前描述符集合不为 VK_NULL_HANDLE，则不执行任何操作
  if (in_use_ < descriptors_.size() &&
      descriptors_[in_use_] != VK_NULL_HANDLE) {
    return;
  }

  // 创建与堆大小相等的描述符集合布局列表
  std::vector<VkDescriptorSetLayout> layouts(descriptors_.size());
  std::fill(layouts.begin(), layouts.end(), set_layout_);

  // 配置描述符集合分配信息
  const VkDescriptorSetAllocateInfo allocate_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, // sType
      nullptr, // pNext
      pool_, // descriptorPool
      utils::safe_downcast<uint32_t>(layouts.size()), // descriptorSetCount
      layouts.data(), // pSetLayouts
  };

  // 分配描述符集合
  VK_CHECK(
      vkAllocateDescriptorSets(device_, &allocate_info, descriptors_.data()));

  // 将正在使用的描述符集合索引重置为 0
  in_use_ = 0u;
}

//
// DescriptorPool
//

// 描述符池的构造函数
DescriptorPool::DescriptorPool(
    VkDevice device,
    const DescriptorPoolConfig& config)
    : device_(device),
      pool_(VK_NULL_HANDLE),
      config_(config),
      mutex_{},
      piles_{} {
  // 如果配置中的描述符池最大集合数大于 0，则进行初始化
  if (config.descriptorPoolMaxSets > 0) {
    init(config);
  }
}

// 描述符池的析构函数
DescriptorPool::~DescriptorPool() {
  // 如果描述符池句柄为 VK_NULL_HANDLE，则直接返回
  if (VK_NULL_HANDLE == pool_) {
    return;
  }
  // 销毁描述符池
  vkDestroyDescriptorPool(device_, pool_, nullptr);
}
void DescriptorPool::init(const DescriptorPoolConfig& config) {
  // 检查描述符池是否已经创建，如果已创建则输出错误信息
  VK_CHECK_COND(
      pool_ == VK_NULL_HANDLE,
      "Trying to init a DescriptorPool that has already been created!");

  // 将传入的配置保存到成员变量中
  config_ = config;

  // 准备描述符类型和数量的数组
  std::vector<VkDescriptorPoolSize> type_sizes{
      {
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          config_.descriptorUniformBufferCount,
      },
      {
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          config_.descriptorStorageBufferCount,
      },
      {
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          config_.descriptorCombinedSamplerCount,
      },
      {
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          config_.descriptorStorageBufferCount,
      },
  };

  // 创建描述符池的配置信息
  const VkDescriptorPoolCreateInfo create_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      config_.descriptorPoolMaxSets, // maxSets
      static_cast<uint32_t>(type_sizes.size()), // poolSizeCounts
      type_sizes.data(), // pPoolSizes
  };

  // 使用 Vulkan API 创建描述符池对象
  VK_CHECK(vkCreateDescriptorPool(device_, &create_info, nullptr, &pool_));
}

DescriptorSet DescriptorPool::get_descriptor_set(
    VkDescriptorSetLayout set_layout,
    const ShaderLayout::Signature& signature) {
  // 检查描述符池是否已经初始化，如果未初始化则输出错误信息
  VK_CHECK_COND(
      pool_ != VK_NULL_HANDLE, "DescriptorPool has not yet been initialized!");

  // 查找指定布局的描述符集合是否已存在
  auto it = piles_.find(set_layout);
  if (piles_.cend() == it) {
    // 如果不存在，则创建新的描述符集合并加入到集合中
    it = piles_
             .insert({
                 set_layout,
                 DescriptorSetPile(
                     config_.descriptorPileSizes, set_layout, device_, pool_),
             })
             .first;
  }

  // 获取或创建的描述符集合的句柄
  VkDescriptorSet handle = it->second.get_descriptor_set();

  // 返回描述符集合对象
  return DescriptorSet(device_, handle, signature);
}

void DescriptorPool::flush() {
  // 如果描述符池已经创建
  if (pool_ != VK_NULL_HANDLE) {
    // 使用 Vulkan API 重置描述符池，并清空描述符集合列表
    VK_CHECK(vkResetDescriptorPool(device_, pool_, 0u));
    piles_.clear();
  }
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
```