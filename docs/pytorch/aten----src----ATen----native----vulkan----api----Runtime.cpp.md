# `.\pytorch\aten\src\ATen\native\vulkan\api\Runtime.cpp`

```
    // 查找请求的 Vulkan 层和扩展并启用它们
    void find_requested_layers_and_extensions(
        std::vector<const char*>& enabled_layers,                  // 用于存储启用的层的容器
        std::vector<const char*>& enabled_extensions,              // 用于存储启用的扩展的容器
        const std::vector<const char*>& requested_layers,          // 请求启用的层列表
        const std::vector<const char*>& requested_extensions) {    // 请求启用的扩展列表

      // 获取支持的实例层数量
      uint32_t layer_count = 0;
      VK_CHECK(vkEnumerateInstanceLayerProperties(&layer_count, nullptr));

      // 获取支持的实例层属性
      std::vector<VkLayerProperties> layer_properties(layer_count);
      VK_CHECK(vkEnumerateInstanceLayerProperties(
          &layer_count, layer_properties.data()));

      // 搜索请求的层
      for (const auto& requested_layer : requested_layers) {
        for (const auto& layer : layer_properties) {
          if (strcmp(requested_layer, layer.layerName) == 0) {
            enabled_layers.push_back(requested_layer);   // 将找到的层添加到启用的层列表中
            break;
          }
        }
      }

      // 获取支持的实例扩展数量
      uint32_t extension_count = 0;
      VK_CHECK(vkEnumerateInstanceExtensionProperties(
          nullptr, &extension_count, nullptr));

      // 获取支持的实例扩展属性
      std::vector<VkExtensionProperties> extension_properties(extension_count);
      VK_CHECK(vkEnumerateInstanceExtensionProperties(
          nullptr, &extension_count, extension_properties.data()));

      // 搜索请求的扩展
      for (const auto& requested_extension : requested_extensions) {
        for (const auto& extension : extension_properties) {
          if (strcmp(requested_extension, extension.extensionName) == 0) {
            enabled_extensions.push_back(requested_extension); // 将找到的扩展添加到启用的扩展列表中
            break;
          }
        }
      }
    }

    // 创建 Vulkan 实例
    VkInstance create_instance(const RuntimeConfiguration& config) {
      // 应用程序信息结构体
      const VkApplicationInfo application_info{
          VK_STRUCTURE_TYPE_APPLICATION_INFO,   // 结构类型
          nullptr,                              // 下一个结构的指针
          "PyTorch Vulkan Backend",             // 应用程序名称
          0,                                     // 应用程序版本号
          nullptr,                              // 引擎名称
          0,                                     // 引擎版本号
          VK_API_VERSION_1_0,                   // Vulkan API 版本
      };

      // 启用的层和扩展列表
      std::vector<const char*> enabled_layers;
      std::vector<const char*> enabled_extensions;

      // 如果启用验证消息，则添加请求的层和扩展
      if (config.enableValidationMessages) {
        std::vector<const char*> requested_layers{
            // "VK_LAYER_LUNARG_api_dump",    // 请求的层：API 转储层（已注释掉）
            "VK_LAYER_KHRONOS_validation",   // 请求的层：Khronos 验证层
        };
        std::vector<const char*> requested_extensions{
    #ifdef VK_EXT_debug_report
            VK_EXT_DEBUG_REPORT_EXTENSION_NAME, // 请求的扩展：调试报告扩展
    #endif /* VK_EXT_debug_report */
        };
    // 调用函数查找请求的 Vulkan 层和扩展，传入已启用的层和扩展列表以及请求的层和扩展列表
    find_requested_layers_and_extensions(
        enabled_layers,
        enabled_extensions,
        requested_layers,
        requested_extensions);
  }

  // 创建 Vulkan 实例的配置信息结构体
  const VkInstanceCreateInfo instance_create_info{
      VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, // 指定结构体类型
      nullptr, // 暂未使用的扩展指针
      0u, // 标志位，这里设置为0
      &application_info, // Vulkan 应用程序信息结构体指针
      static_cast<uint32_t>(enabled_layers.size()), // 启用的层数量
      enabled_layers.data(), // 启用的层名称数组指针
      static_cast<uint32_t>(enabled_extensions.size()), // 启用的扩展数量
      enabled_extensions.data(), // 启用的扩展名称数组指针
  };

  VkInstance instance{}; // 定义 Vulkan 实例对象
  VK_CHECK(vkCreateInstance(&instance_create_info, nullptr, &instance)); // 创建 Vulkan 实例，并检查返回结果
  VK_CHECK_COND(instance, "Invalid Vulkan instance!"); // 检查 Vulkan 实例是否有效
#ifdef USE_VULKAN_VOLK
  // 如果定义了 USE_VULKAN_VOLK 宏，则调用 volkLoadInstance 加载 Vulkan 实例
  volkLoadInstance(instance);
#endif /* USE_VULKAN_VOLK */

  // 返回 Vulkan 实例对象
  return instance;
}

// 创建并返回物理设备列表
std::vector<Runtime::DeviceMapping> create_physical_devices(
    VkInstance instance) {
  // 如果 Vulkan 实例为空，则返回空的设备映射列表
  if (VK_NULL_HANDLE == instance) {
    return std::vector<Runtime::DeviceMapping>();
  }

  // 获取物理设备数量
  uint32_t device_count = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &device_count, nullptr));

  // 分配物理设备数组并获取物理设备列表
  std::vector<VkPhysicalDevice> devices(device_count);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()));

  // 准备设备映射列表，预留足够空间
  std::vector<Runtime::DeviceMapping> device_mappings;
  device_mappings.reserve(device_count);
  // 将每个物理设备映射为 DeviceMapping 对象，并加入到映射列表中
  for (VkPhysicalDevice physical_device : devices) {
    device_mappings.emplace_back(PhysicalDevice(physical_device), -1);
  }

  // 返回设备映射列表
  return device_mappings;
}

// Vulkan 调试报告回调函数
VKAPI_ATTR VkBool32 VKAPI_CALL debug_report_callback_fn(
    const VkDebugReportFlagsEXT flags,
    const VkDebugReportObjectTypeEXT /* object_type */,
    const uint64_t /* object */,
    const size_t /* location */,
    const int32_t message_code,
    const char* const layer_prefix,
    const char* const message,
    void* const /* user_data */) {
  (void)flags;

  // 将调试信息格式化为字符串并输出到标准输出流
  std::stringstream stream;
  stream << layer_prefix << " " << message_code << " " << message << std::endl;
  const std::string log = stream.str();

  std::cout << log;

  // 返回 VK_FALSE 表示调试报告处理完成
  return VK_FALSE;
}

// 创建并返回 Vulkan 调试报告回调对象
VkDebugReportCallbackEXT create_debug_report_callback(
    VkInstance instance,
    const RuntimeConfiguration config) {
  // 如果 Vulkan 实例为空或未启用验证消息，则返回空的调试报告回调对象
  if (VK_NULL_HANDLE == instance || !config.enableValidationMessages) {
    return VkDebugReportCallbackEXT{};
  }

  // 配置调试报告回调创建信息
  const VkDebugReportCallbackCreateInfoEXT debugReportCallbackCreateInfo{
      VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT, // sType
      nullptr, // pNext
      VK_DEBUG_REPORT_INFORMATION_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT |
          VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT |
          VK_DEBUG_REPORT_ERROR_BIT_EXT |
          VK_DEBUG_REPORT_DEBUG_BIT_EXT, // flags
      debug_report_callback_fn, // pfnCallback
      nullptr, // pUserData
  };

  // 获取 vkCreateDebugReportCallbackEXT 函数指针
  const auto vkCreateDebugReportCallbackEXT =
      (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(
          instance, "vkCreateDebugReportCallbackEXT");

  // 检查函数指针是否有效
  VK_CHECK_COND(
      vkCreateDebugReportCallbackEXT,
      "Could not load vkCreateDebugReportCallbackEXT");

  // 创建调试报告回调对象
  VkDebugReportCallbackEXT debug_report_callback{};
  VK_CHECK(vkCreateDebugReportCallbackEXT(
      instance,
      &debugReportCallbackCreateInfo,
      nullptr,
      &debug_report_callback));

  // 检查调试报告回调对象是否有效
  VK_CHECK_COND(debug_report_callback, "Invalid Vulkan debug report callback!");

  // 返回调试报告回调对象
  return debug_report_callback;
}

//
// 适配器选择方法
//

// 选择第一个符合条件的设备索引
uint32_t select_first(const std::vector<Runtime::DeviceMapping>& devices) {
  // 如果设备列表为空，则返回设备数量加一，表示无效索引
  if (devices.empty()) {
    return devices.size() + 1; // return out of range to signal invalidity
  }

  // 选择第一个具有计算能力的适配器
  for (size_t i = 0; i < devices.size(); ++i) {
    # 检查设备列表中索引为 i 的设备是否具有至少一个计算队列
    if (devices[i].first.num_compute_queues > 0) {
      # 如果是，则返回该设备的索引 i
      return i;
    }
  }

  # 如果未找到符合条件的设备，则返回设备列表的大小加一
  return devices.size() + 1;
}

//
// Global runtime initialization
//

// 初始化全局 Vulkan 运行时环境
std::unique_ptr<Runtime> init_global_vulkan_runtime() {
  // 加载 Vulkan 驱动程序
#if defined(USE_VULKAN_VOLK)
  // 使用 Volk 初始化 Vulkan
  if (VK_SUCCESS != volkInitialize()) {
    return std::unique_ptr<Runtime>(nullptr);
  }
#elif defined(USE_VULKAN_WRAPPER)
  // 使用 Vulkan Wrapper 初始化 Vulkan
  if (!InitVulkan()) {
    return std::unique_ptr<Runtime>(nullptr);
  }
#endif /* USE_VULKAN_VOLK, USE_VULKAN_WRAPPER */

  // 是否启用验证消息
  const bool enableValidationMessages =
#if defined(VULKAN_DEBUG)
      true;
#else
      false;
#endif /* VULKAN_DEBUG */
  // 是否初始化默认设备
  const bool initDefaultDevice = true;
  // 请求的队列数量
  const uint32_t numRequestedQueues = 1; // TODO: 提高此值

  // 默认配置
  const RuntimeConfiguration default_config{
      enableValidationMessages,
      initDefaultDevice,
      AdapterSelector::First,
      numRequestedQueues,
  };

  try {
    // 创建 Runtime 对象并返回
    return std::make_unique<Runtime>(Runtime(default_config));
  } catch (...) {
  }

  // 返回空的 Runtime 智能指针，表示初始化失败
  return std::unique_ptr<Runtime>(nullptr);
}

} // namespace

// Runtime 类的构造函数实现
Runtime::Runtime(const RuntimeConfiguration config)
    : config_(config),
      instance_(create_instance(config_)),
      device_mappings_(create_physical_devices(instance_)),
      adapters_{},
      default_adapter_i_(UINT32_MAX),
      debug_report_callback_(create_debug_report_callback(instance_, config_)) {
  // adapters_ 列表的预留空间不超过物理设备数量
  adapters_.reserve(device_mappings_.size());

  // 如果配置要求初始化默认设备
  if (config.initDefaultDevice) {
    try {
      // 根据选择器创建默认适配器
      switch (config.defaultSelector) {
        case AdapterSelector::First:
          default_adapter_i_ = create_adapter(select_first);
      }
    } catch (...) {
    }
  }
}

// Runtime 类的析构函数实现
Runtime::~Runtime() {
  // 如果 instance_ 是空 handle，则直接返回
  if (VK_NULL_HANDLE == instance_) {
    return;
  }

  // 清空 adapters_ 列表，以触发在销毁 VkInstance 之前销毁设备
  adapters_.clear();

  // 必须最后销毁 instance_，因为它用于销毁 debug report 回调
  if (debug_report_callback_) {
    // 获取 vkDestroyDebugReportCallbackEXT 函数指针
    const auto vkDestroyDebugReportCallbackEXT =
        (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(
            instance_, "vkDestroyDebugReportCallbackEXT");

    // 调用回调销毁函数
    if (vkDestroyDebugReportCallbackEXT) {
      vkDestroyDebugReportCallbackEXT(
          instance_, debug_report_callback_, nullptr);
    }

    debug_report_callback_ = {};
  }

  // 销毁 Vulkan 实例
  vkDestroyInstance(instance_, nullptr);
  instance_ = VK_NULL_HANDLE;
}

// Runtime 类的移动构造函数实现
Runtime::Runtime(Runtime&& other) noexcept
    : config_(other.config_),
      instance_(other.instance_),
      adapters_(std::move(other.adapters_)),
      default_adapter_i_(other.default_adapter_i_),
      debug_report_callback_(other.debug_report_callback_) {
  // 移动构造后，将原对象的 instance_ 置为空 handle 和 debug_report_callback_ 置空
  other.instance_ = VK_NULL_HANDLE;
  other.debug_report_callback_ = {};
}
// 创建适配器的函数，根据选择器确定使用哪个物理设备来创建适配器
uint32_t Runtime::create_adapter(const Selector& selector) {
  // 检查设备映射是否为空，如果是，则输出错误信息并返回
  VK_CHECK_COND(
      !device_mappings_.empty(),
      "Pytorch Vulkan Runtime: Could not initialize adapter because no "
      "devices were found by the Vulkan instance.");

  // 使用选择器函数确定要使用的物理设备索引
  uint32_t physical_device_i = selector(device_mappings_);
  // 检查选择的物理设备索引是否有效
  VK_CHECK_COND(
      physical_device_i < device_mappings_.size(),
      "Pytorch Vulkan Runtime: no suitable device adapter was selected! "
      "Device could not be initialized");

  // 获取选定物理设备对应的设备映射
  Runtime::DeviceMapping& device_mapping = device_mappings_[physical_device_i];

  // 如果已经创建了适配器，则直接返回其索引
  int32_t adapter_i = device_mapping.second;
  if (adapter_i >= 0) {
    return adapter_i;
  }

  // 否则，为选定的物理设备创建一个新的适配器
  adapter_i = utils::safe_downcast<int32_t>(adapters_.size());
  adapters_.emplace_back(
      new Adapter(instance_, device_mapping.first, config_.numRequestedQueues));
  // 将新创建的适配器索引与设备映射关联起来
  device_mapping.second = adapter_i;

  return adapter_i;
}

// 获取全局 Vulkan 运行时的函数
Runtime* runtime() {
  // 使用静态局部变量声明全局 Vulkan 运行时，确保具有外部链接。
  // 如果它是全局静态变量，每个包含 Runtime.h 的翻译单元会有一个副本，因为它将具有内部链接。
  static const std::unique_ptr<Runtime> p_runtime =
      init_global_vulkan_runtime();

  // 检查全局运行时是否成功初始化，如果未能初始化，则输出错误信息
  VK_CHECK_COND(
      p_runtime,
      "Pytorch Vulkan Runtime: The global runtime could not be retrieved "
      "because it failed to initialize.");

  return p_runtime.get();  // 返回全局运行时的指针
}
```