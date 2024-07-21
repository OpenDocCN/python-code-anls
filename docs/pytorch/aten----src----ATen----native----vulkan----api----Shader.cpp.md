# `.\pytorch\aten\src\ATen\native\vulkan\api\Shader.cpp`

```
//
// ShaderModule
//

ShaderModule::ShaderModule(VkDevice device, const ShaderInfo& source)
    : device_(device), handle_{VK_NULL_HANDLE} {
  // 初始化着色器模块的信息
  VkShaderModuleCreateInfo create_info{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, // sType
      nullptr, // pNext
      0, // flags
      source.src_code.size * sizeof(uint32_t), // codeSize
      source.src_code.bin // pCode
  };

  // 创建 Vulkan 着色器模块对象
  VK_CHECK(vkCreateShaderModule(device_, &create_info, nullptr, &handle_));
}

ShaderModule::ShaderModule(ShaderModule&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  // 将另一个着色器模块对象的句柄移动到当前对象，同时置空原对象的句柄
  other.handle_ = VK_NULL_HANDLE;
}

ShaderModule::~ShaderModule() {
  // 如果句柄为空，直接返回，不执行销毁操作
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  // 销毁 Vulkan 着色器模块对象
  vkDestroyShaderModule(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE; // 置空句柄以避免重复销毁
}

void swap(ShaderModule& lhs, ShaderModule& rhs) noexcept {
  // 交换两个着色器模块对象的设备和句柄
  VkDevice tmp_device = lhs.device_;
  VkShaderModule tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}
  : device_(device), handle_{VK_NULL_HANDLE} {

初始化成员变量 `device_` 和 `handle_`，其中 `device_` 是传入的设备参数，`handle_` 初始化为 `VK_NULL_HANDLE`。


  const uint32_t* code = source.src_code.bin;

从 `source` 中获取着色器代码的指针，并赋值给 `code`，假设其类型为 `const uint32_t*`。


  uint32_t size = source.src_code.size;

从 `source` 中获取着色器代码的大小，并将其赋值给 `size`，假设其类型为 `uint32_t`。


  const VkShaderModuleCreateInfo shader_module_create_info{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      size, // codeSize
      code, // pCode
  };

创建 `VkShaderModuleCreateInfo` 结构体，并初始化其中的字段：
- `sType`: 结构体类型，指定为 `VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO`。
- `pNext`: 指向扩展信息的指针，这里设为 `nullptr`，表示无扩展信息。
- `flags`: 标志位，设为 `0u`，表示无特殊标志。
- `codeSize`: 着色器代码的大小，从之前定义的 `size` 变量获取。
- `pCode`: 指向着色器代码的指针，从之前定义的 `code` 变量获取。


  VK_CHECK(vkCreateShaderModule(
      device_, &shader_module_create_info, nullptr, &handle_));

调用 `vkCreateShaderModule` 函数，用于在指定的设备上创建着色器模块对象，并将创建的对象句柄存储在 `handle_` 中。函数的参数包括：
- `device_`: 指定的 Vulkan 设备。
- `&shader_module_create_info`: 指向着色器模块创建信息结构体的指针。
- `nullptr`: 指向自定义内存分配器的指针，这里使用默认分配器。
- `&handle_`: 存储着色器模块对象句柄的指针。

以上注释详细解释了给定代码块中每一行的作用和含义。
}

// ShaderModule 类的移动构造函数，使用了移动语义并将资源所有权转移
ShaderModule::ShaderModule(ShaderModule&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE; // 将原对象的句柄置为无效
}

// ShaderModule 类的析构函数，释放 Vulkan 着色器模块资源
ShaderModule::~ShaderModule() {
  if (VK_NULL_HANDLE == handle_) {
    return; // 如果句柄无效，直接返回
  }
  vkDestroyShaderModule(device_, handle_, nullptr); // 销毁 Vulkan 着色器模块资源
  handle_ = VK_NULL_HANDLE; // 将句柄置为无效
}

// ShaderModule 类的交换函数，交换两个 ShaderModule 对象的成员变量
void swap(ShaderModule& lhs, ShaderModule& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkShaderModule tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// ShaderLayoutCache 类
//

// ShaderLayoutCache 类的构造函数，初始化成员变量和缓存
ShaderLayoutCache::ShaderLayoutCache(VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

// ShaderLayoutCache 类的移动构造函数，使用了移动语义并将资源所有权转移
ShaderLayoutCache::ShaderLayoutCache(ShaderLayoutCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_), cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_); // 锁定移动源的互斥量
}

// ShaderLayoutCache 类的析构函数，清空缓存
ShaderLayoutCache::~ShaderLayoutCache() {
  purge(); // 清空缓存
}

// 根据键检索描述符集布局
VkDescriptorSetLayout ShaderLayoutCache::retrieve(const ShaderLayoutCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_); // 锁定互斥量

  auto it = cache_.find(key); // 在缓存中查找键对应的值
  if (cache_.cend() == it) { // 如果未找到
    // 插入新的键值对到缓存中
    it = cache_.insert({key, ShaderLayoutCache::Value(device_, key)}).first;
  }

  return it->second.handle(); // 返回缓存中对应键的值的句柄
}

// 清空缓存
void ShaderLayoutCache::purge() {
  std::lock_guard<std::mutex> lock(cache_mutex_); // 锁定互斥量
  cache_.clear(); // 清空缓存
}

//
// ShaderCache 类
//

// ShaderCache 类的构造函数，初始化成员变量和缓存
ShaderCache::ShaderCache(VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

// ShaderCache 类的移动构造函数，使用了移动语义并将资源所有权转移
ShaderCache::ShaderCache(ShaderCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_), cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_); // 锁定移动源的互斥量
}

// ShaderCache 类的析构函数，清空缓存
ShaderCache::~ShaderCache() {
  purge(); // 清空缓存
}

// 根据键检索着色器模块
VkShaderModule ShaderCache::retrieve(const ShaderCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_); // 锁定互斥量

  auto it = cache_.find(key); // 在缓存中查找键对应的值
  if (cache_.cend() == it) { // 如果未找到
    // 插入新的键值对到缓存中
    it = cache_.insert({key, ShaderCache::Value(device_, key)}).first;
  }

  return it->second.handle(); // 返回缓存中对应键的值的句柄
}

// 清空缓存
void ShaderCache::purge() {
  cache_.clear(); // 清空缓存
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
```