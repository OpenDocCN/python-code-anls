# `.\pytorch\aten\src\ATen\native\vulkan\api\Adapter.cpp`

```py
// 引入 Vulkan 头文件中 Adapter 类的实现
#include <ATen/native/vulkan/api/Adapter.h>

// 引入 C++ 标准库头文件
#include <bitset>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <utility>

// 定义 at 命名空间
namespace at {
// 定义 native 命名空间
namespace native {
// 定义 Vulkan API 的命名空间
namespace vulkan {
// 定义 Vulkan API 内部的命名空间 api
namespace api {

// Vulkan 物理设备类的构造函数，初始化各成员变量
PhysicalDevice::PhysicalDevice(VkPhysicalDevice physical_device_handle)
    : handle(physical_device_handle), // 初始化 Vulkan 物理设备句柄
      properties{},                 // 初始化 Vulkan 物理设备属性
      memory_properties{},          // 初始化 Vulkan 内存属性
      queue_families{},             // 初始化队列族属性列表
      num_compute_queues(0),        // 初始化计算队列数量为 0
      has_unified_memory(false),    // 初始化标志位，表示未检测到统一内存类型
      has_timestamps(properties.limits.timestampComputeAndGraphics), // 根据属性初始化时间戳支持
      timestamp_period(properties.limits.timestampPeriod) {         // 根据属性初始化时间戳周期

  // 获取物理设备的属性信息
  vkGetPhysicalDeviceProperties(handle, &properties);
  // 获取物理设备的内存属性信息
  vkGetPhysicalDeviceMemoryProperties(handle, &memory_properties);

  // 检查是否存在同时具有 HOST_VISIBLE 和 DEVICE_LOCAL 属性标志的内存类型
  const VkMemoryPropertyFlags unified_memory_flags =
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  for (size_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
    if (memory_properties.memoryTypes[i].propertyFlags | unified_memory_flags) {
      has_unified_memory = true;
      break;
    }
  }

  // 获取设备支持的队列族属性信息
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
      handle, &queue_family_count, nullptr);

  queue_families.resize(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      handle, &queue_family_count, queue_families.data());

  // 计算支持的计算队列总数
  for (const VkQueueFamilyProperties& p : queue_families) {
    // 检查该队列族是否支持计算功能
    if (p.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      num_compute_queues += p.queueCount;
    }
  }
}

// 私有函数：查找请求的设备扩展
void find_requested_device_extensions(
    VkPhysicalDevice physical_device,
    std::vector<const char*>& enabled_extensions,
    const std::vector<const char*>& requested_extensions) {

  uint32_t device_extension_properties_count = 0;
  // 查询设备支持的扩展属性数量
  VK_CHECK(vkEnumerateDeviceExtensionProperties(
      physical_device, nullptr, &device_extension_properties_count, nullptr));
  // 获取设备支持的扩展属性列表
  std::vector<VkExtensionProperties> device_extension_properties(
      device_extension_properties_count);
  VK_CHECK(vkEnumerateDeviceExtensionProperties(
      physical_device,
      nullptr,
      &device_extension_properties_count,
      device_extension_properties.data()));

  // 遍历请求的扩展列表，将支持的扩展添加到已启用的扩展列表中
  for (const auto& requested_extension : requested_extensions) {
    for (const auto& extension : device_extension_properties) {
      if (strcmp(requested_extension, extension.extensionName) == 0) {
        enabled_extensions.push_back(requested_extension);
        break;
      }
    }
  }
}

// 创建逻辑设备的函数
VkDevice create_logical_device(
    const PhysicalDevice& physical_device,
    const uint32_t num_queues_to_create,
    std::vector<Adapter::Queue>& queues,
  std::vector<uint32_t>& queue_usage) {
  // 初始化队列创建信息的容器，预留足够的空间以容纳所需创建的队列数目
  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  queue_create_infos.reserve(num_queues_to_create);

  // 用于存储即将获取的队列的队列族索引和队列索引的容器，预留足够的空间以容纳所需创建的队列数目
  std::vector<std::pair<uint32_t, uint32_t>> queues_to_get;
  queues_to_get.reserve(num_queues_to_create);

  // 初始化剩余队列数为所需创建的队列总数
  uint32_t remaining_queues = num_queues_to_create;
  
  // 遍历物理设备支持的所有队列族
  for (uint32_t family_i = 0; family_i < physical_device.queue_families.size();
       ++family_i) {
    // 获取当前队列族的属性信息
    const VkQueueFamilyProperties& queue_properties =
        physical_device.queue_families.at(family_i);
    
    // 检查当前队列族是否支持计算队列
    if (queue_properties.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      // 计算当前队列族实际初始化的队列数，取剩余需要的队列数和当前队列族支持的队列总数的最小值
      const uint32_t queues_to_init =
          std::min(remaining_queues, queue_properties.queueCount);

      // 创建队列优先级数组，所有队列的优先级都设为1.0
      const std::vector<float> queue_priorities(queues_to_init, 1.0f);
      
      // 将队列创建信息添加到队列创建信息容器中
      queue_create_infos.push_back({
          VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, // sType
          nullptr, // pNext
          0u, // flags
          family_i, // queueFamilyIndex
          queues_to_init, // queueCount
          queue_priorities.data(), // pQueuePriorities
      });

      // 将每个初始化的队列的队列族索引和队列索引添加到待获取队列的容器中
      for (size_t queue_i = 0; queue_i < queues_to_init; ++queue_i) {
        queues_to_get.emplace_back(family_i, queue_i);
      }
      
      // 更新剩余需要创建的队列数
      remaining_queues -= queues_to_init;
    }
    
    // 如果剩余需要创建的队列数已经为0，则退出循环
    if (remaining_queues == 0) {
      break;
    }
  }

  // 预留足够的空间以容纳即将获取的队列的句柄
  queues.reserve(queues_to_get.size());
  // 预留足够的空间以容纳即将获取的队列的使用情况
  queue_usage.reserve(queues_to_get.size());

  // 创建 VkDevice

  // 请求的设备扩展名称数组
  std::vector<const char*> requested_device_extensions{
#ifdef VK_KHR_portability_subset
      VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
#endif /* VK_KHR_portability_subset */


#ifdef VK_KHR_portability_subset
      VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
#endif /* VK_KHR_portability_subset */


  };

  // 存储启用的设备扩展名称的向量
  std::vector<const char*> enabled_device_extensions;
  // 查找请求的设备扩展并存储到 enabled_device_extensions 中
  find_requested_device_extensions(
      physical_device.handle,
      enabled_device_extensions,
      requested_device_extensions);

  // Vulkan 设备创建信息结构体
  const VkDeviceCreateInfo device_create_info{
      VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      static_cast<uint32_t>(queue_create_infos.size()), // queueCreateInfoCount
      queue_create_infos.data(), // pQueueCreateInfos
      0u, // enabledLayerCount
      nullptr, // ppEnabledLayerNames
      static_cast<uint32_t>(
          enabled_device_extensions.size()), // enabledExtensionCount
      enabled_device_extensions.data(), // ppEnabledExtensionNames
      nullptr, // pEnabledFeatures
  };

  // Vulkan 设备句柄
  VkDevice handle = nullptr;
  // 创建 Vulkan 设备对象
  VK_CHECK(vkCreateDevice(
      physical_device.handle, &device_create_info, nullptr, &handle));

#ifdef USE_VULKAN_VOLK
  // 加载 Vulkan 设备函数指针
  volkLoadDevice(handle);
#endif /* USE_VULKAN_VOLK */

  // 获取创建的队列句柄，并初始化队列使用的启发式信息

  for (const std::pair<uint32_t, uint32_t>& queue_idx : queues_to_get) {
    VkQueue queue_handle = VK_NULL_HANDLE;
    // 获取设备队列句柄
    VkQueueFlags flags =
        physical_device.queue_families.at(queue_idx.first).queueFlags;
    vkGetDeviceQueue(handle, queue_idx.first, queue_idx.second, &queue_handle);
    // 将队列信息加入到 queues 向量中
    queues.push_back({queue_idx.first, queue_idx.second, flags, queue_handle});
    // 初始化队列使用值
    queue_usage.push_back(0);
  }

  // 返回 Vulkan 设备句柄
  return handle;
}

// 打印工具函数

// 获取设备类型的字符串表示
std::string get_device_type_str(const VkPhysicalDeviceType type) {
  switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return "INTEGRATED_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return "DISCRETE_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return "VIRTUAL_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return "CPU";
    default:
      return "UNKNOWN";
  }
}

// 获取内存属性标志的字符串表示
std::string get_memory_properties_str(const VkMemoryPropertyFlags flags) {
  std::bitset<10> values(flags);
  std::stringstream ss("|");
  if (values[0]) {
    ss << " DEVICE_LOCAL |";
  }
  if (values[1]) {
    ss << " HOST_VISIBLE |";
  }
  if (values[2]) {
    ss << " HOST_COHERENT |";
  }
  if (values[3]) {
    ss << " HOST_CACHED |";
  }
  if (values[4]) {
    ss << " LAZILY_ALLOCATED |";
  }

  return ss.str();
}

// 获取队列族属性标志的字符串表示
std::string get_queue_family_properties_str(const VkQueueFlags flags) {
  std::bitset<10> values(flags);
  std::stringstream ss("|");
  if (values[0]) {
    ss << " GRAPHICS |";
  }
  if (values[1]) {
    ss << " COMPUTE |";
  }
  if (values[2]) {
    ss << " TRANSFER |";
  }

  return ss.str();
}

} // namespace

//
// DeviceHandle
//

// Vulkan 设备句柄类的构造函数
DeviceHandle::DeviceHandle(VkDevice device) : handle_(device) {}

// Vulkan 设备句柄类的移动构造函数
DeviceHandle::DeviceHandle(DeviceHandle&& other) noexcept
    : handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}
// 实现析构函数，用于释放设备句柄所关联的资源
DeviceHandle::~DeviceHandle() {
  // 如果句柄为空，则直接返回，无需释放资源
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  // 销毁 Vulkan 设备对象
  vkDestroyDevice(handle_, nullptr);
}

//
// 适配器
//

// 构造函数，初始化适配器对象
Adapter::Adapter(
    VkInstance instance,                    // Vulkan 实例
    PhysicalDevice physical_device,         // 物理设备对象
    const uint32_t num_queues)              // 需要创建的队列数量
    : queue_usage_mutex_{},                 // 队列使用量互斥锁
      physical_device_(std::move(physical_device)), // 移动构造物理设备对象
      queues_{},                            // 队列数组
      queue_usage_{},                       // 队列使用量数组
      queue_mutexes_{},                     // 队列互斥锁数组
      instance_(instance),                  // Vulkan 实例对象
      device_(create_logical_device(        // 创建逻辑设备对象
          physical_device_,                 // 物理设备对象
          num_queues,                       // 队列数量
          queues_,                          // 队列数组
          queue_usage_)),                   // 队列使用量数组
      shader_layout_cache_(device_.handle_), // 着色器布局缓存
      shader_cache_(device_.handle_),        // 着色器缓存
      pipeline_layout_cache_(device_.handle_), // 管道布局缓存
      compute_pipeline_cache_(device_.handle_), // 计算管道缓存
      sampler_cache_(device_.handle_),       // 采样器缓存
      vma_(instance_, physical_device_.handle, device_.handle_) {} // Vulkan 内存分配器初始化

// 请求队列的方法，确保线程安全
Adapter::Queue Adapter::request_queue() {
  // 锁定队列使用量互斥锁，因为多个线程可能同时请求队列
  std::lock_guard<std::mutex> lock(queue_usage_mutex_);

  uint32_t min_usage = UINT32_MAX;
  uint32_t min_used_i = 0;
  // 找到使用量最小的队列索引
  for (size_t i = 0; i < queues_.size(); ++i) {
    if (queue_usage_[i] < min_usage) {
      min_used_i = i;
      min_usage = queue_usage_[i];
    }
  }
  // 增加选定队列的使用量
  queue_usage_[min_used_i] += 1;

  // 返回请求的队列对象
  return queues_[min_used_i];
}

// 归还队列的方法
void Adapter::return_queue(Adapter::Queue& compute_queue) {
  for (size_t i = 0; i < queues_.size(); ++i) {
    // 根据队列索引匹配队列对象，确保返回正确的队列
    if ((queues_[i].family_index == compute_queue.family_index) &&
        (queues_[i].queue_index == compute_queue.queue_index)) {
      // 锁定队列使用量互斥锁
      std::lock_guard<std::mutex> lock(queue_usage_mutex_);
      // 减少队列使用量
      queue_usage_[i] -= 1;
      break;
    }
  }
}

// 提交单个命令缓冲区到队列的方法
void Adapter::submit_cmd(
    const Adapter::Queue& device_queue,     // 设备队列对象
    VkCommandBuffer cmd,                    // Vulkan 命令缓冲区
    VkFence fence) {                        // Vulkan 栅栏对象
  // 创建提交信息结构体
  const VkSubmitInfo submit_info{
      VK_STRUCTURE_TYPE_SUBMIT_INFO,        // 结构体类型
      nullptr,                              // 扩展信息，目前为空
      0u,                                   // 等待信号量数量，此处为0
      nullptr,                              // 等待信号量数组，此处为空
      nullptr,                              // 等待阶段掩码，此处为空
      1u,                                   // 命令缓冲区数量，固定为1
      &cmd,                                 // 命令缓冲区数组
      0u,                                   // 信号量数量，此处为0
      nullptr,                              // 信号量数组，此处为空
  };

  // 锁定指定队列的互斥锁
  std::lock_guard<std::mutex> queue_lock(
      queue_mutexes_[device_queue.queue_index % NUM_QUEUE_MUTEXES]);

  // 提交命令缓冲区到 Vulkan 队列
  VK_CHECK(vkQueueSubmit(device_queue.handle, 1u, &submit_info, fence));
}

// 提交多个命令缓冲区到队列的方法
void Adapter::submit_cmds(
    const Adapter::Queue& device_queue,               // 设备队列对象
    const std::vector<VkCommandBuffer>& cmds,         // Vulkan 命令缓冲区数组
    VkFence fence) {                                  // Vulkan 栅栏对象
  // 创建提交信息结构体
  const VkSubmitInfo submit_info{
      VK_STRUCTURE_TYPE_SUBMIT_INFO,                  // 结构体类型
      nullptr,                                        // 扩展信息，目前为空
      0u,                                             // 等待信号量数量，此处为0
      nullptr,                                        // 等待信号量数组，此处为空
      nullptr,                                        // 等待阶段掩码，此处为空
      utils::safe_downcast<uint32_t>(cmds.size()),    // 命令缓冲区数量
      cmds.data(),                                    // 命令缓冲区数组
      0u,                                             // 信号量数量，此处为0
      nullptr,                                        // 信号量数组，此处为空
  };

  // 提交多个命令缓冲区到 Vulkan 队列
  VK_CHECK(vkQueueSubmit(device_queue.handle, 1u, &submit_info, fence));
}
std::string Adapter::stringize() const {
  // 创建一个字符串流对象
  std::stringstream ss;

  // 获取物理设备的属性信息
  VkPhysicalDeviceProperties properties = physical_device_.properties;
  // 从API版本中提取主版本号和次版本号
  uint32_t v_major = VK_VERSION_MAJOR(properties.apiVersion);
  uint32_t v_minor = VK_VERSION_MINOR(properties.apiVersion);
  // 获取设备类型的字符串表示
  std::string device_type = get_device_type_str(properties.deviceType);
  // 获取设备的限制属性信息
  VkPhysicalDeviceLimits limits = properties.limits;

  // 将物理设备信息写入字符串流
  ss << "{" << std::endl;
  ss << "  Physical Device Info {" << std::endl;
  ss << "    apiVersion:    " << v_major << "." << v_minor << std::endl;
  ss << "    driverversion: " << properties.driverVersion << std::endl;
  ss << "    deviceType:    " << device_type << std::endl;
  ss << "    deviceName:    " << properties.deviceName << std::endl;

  // 定义和使用宏来打印设备限制的属性
#define PRINT_LIMIT_PROP(name)                                         \
  ss << "      " << std::left << std::setw(36) << #name << limits.name \
     << std::endl;

#define PRINT_LIMIT_PROP_VEC3(name)                                       \
  ss << "      " << std::left << std::setw(36) << #name << limits.name[0] \
     << "," << limits.name[1] << "," << limits.name[2] << std::endl;

  // 打印物理设备的限制属性到字符串流
  ss << "    Physical Device Limits {" << std::endl;
  PRINT_LIMIT_PROP(maxImageDimension1D);
  PRINT_LIMIT_PROP(maxImageDimension2D);
  PRINT_LIMIT_PROP(maxImageDimension3D);
  PRINT_LIMIT_PROP(maxTexelBufferElements);
  PRINT_LIMIT_PROP(maxPushConstantsSize);
  PRINT_LIMIT_PROP(maxMemoryAllocationCount);
  PRINT_LIMIT_PROP(maxSamplerAllocationCount);
  PRINT_LIMIT_PROP(maxComputeSharedMemorySize);
  PRINT_LIMIT_PROP_VEC3(maxComputeWorkGroupCount);
  PRINT_LIMIT_PROP(maxComputeWorkGroupInvocations);
  PRINT_LIMIT_PROP_VEC3(maxComputeWorkGroupSize);
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ;

  // 获取物理设备的内存属性信息
  const VkPhysicalDeviceMemoryProperties& mem_props =
      physical_device_.memory_properties;

  // 打印内存类型信息到字符串流
  ss << "  Memory Info {" << std::endl;
  ss << "    Memory Types [" << std::endl;
  // 遍历内存类型并打印信息
  for (size_t i = 0; i < mem_props.memoryTypeCount; ++i) {
    ss << "      "
       << " [Heap " << mem_props.memoryTypes[i].heapIndex << "] "
       << get_memory_properties_str(mem_props.memoryTypes[i].propertyFlags)
       << std::endl;
  }
  ss << "    ]" << std::endl;
  ss << "    Memory Heaps [" << std::endl;
  // 打印内存堆信息到字符串流
  for (size_t i = 0; i < mem_props.memoryHeapCount; ++i) {
    ss << "      " << mem_props.memoryHeaps[i].size << std::endl;
  }
  ss << "    ]" << std::endl;
  ss << "  }" << std::endl;

  // 打印队列家族信息到字符串流
  ss << "  Queue Families {" << std::endl;
  for (const VkQueueFamilyProperties& queue_family_props :
       physical_device_.queue_families) {
    ss << "    (" << queue_family_props.queueCount << " Queues) "
       << get_queue_family_properties_str(queue_family_props.queueFlags)
       << std::endl;
  }
  ss << "  }" << std::endl;
  ss << "  VkDevice: " << device_.handle_ << std::endl;
  ss << "  Compute Queues [" << std::endl;

  // 继续打印计算队列信息到字符串流
  for (const Adapter::Queue& compute_queue : queues_) {
    ss << "    Family " << compute_queue.family_index << ", Queue "
       << compute_queue.queue_index << ": " << compute_queue.handle
       << std::endl;
    // 将 compute_queue 的 family_index、queue_index 和 handle 写入到字符串流 ss 中，格式为 "Family <family_index>, Queue <queue_index>: <handle>"
    // std::endl 用于在字符串流中添加换行符

    ;
  }
  ss << "  ]" << std::endl;
  // 将字符串流 ss 中的内容结束标记为一个数组结尾，添加换行符

  ss << "}";
  // 向字符串流 ss 中添加对象结尾的标记

  return ss.str();
  // 返回字符串流 ss 的内容作为字符串
}

// 重载流插入运算符 << ，将 Adapter 对象转换为字符串并输出到流 os 中，然后输出换行符
std::ostream& operator<<(std::ostream& os, const Adapter& adapter) {
    // 调用 Adapter 类的 stringize 方法，将其返回的字符串输出到流 os 中
    os << adapter.stringize() << std::endl;
    // 返回输出流 os
    return os;
}

// 结束命名空间 at
} // namespace at
// 结束命名空间 native
} // namespace native
// 结束命名空间 vulkan
} // namespace vulkan
// 结束命名空间 api
} // namespace api
```