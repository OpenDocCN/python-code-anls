# `.\pytorch\aten\src\ATen\native\vulkan\api\Runtime.h`

```
#pragma once
// 一旦这个头文件被包含，禁止重复包含

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName
// 忽略 CLANGTIDY 工具的 facebook-hte-BadMemberName 错误

#include <functional>
// 包含 C++ 标准库中的 functional 头文件，用于支持函数对象和函数指针的封装和调用
#include <memory>
// 包含 C++ 标准库中的 memory 头文件，用于动态内存管理，包括智能指针

#ifdef USE_VULKAN_API
// 如果定义了 USE_VULKAN_API 宏，则编译以下代码块

#include <ATen/native/vulkan/api/vk_api.h>
// 包含 Vulkan API 头文件，用于 Vulkan 图形和计算 API 的功能支持

#include <ATen/native/vulkan/api/Adapter.h>
// 包含 Adapter 类的头文件，Adapter 是用于管理 Vulkan 物理设备的类

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// A Vulkan Runtime initializes a Vulkan instance and decouples the concept of
// Vulkan instance initialization from initialization of, and subsequent
// interactions with,  Vulkan [physical and logical] devices as a precursor to
// multi-GPU support.  The Vulkan Runtime can be queried for available Adapters
// (i.e. physical devices) in the system which in turn can be used for creation
// of a Vulkan Context (i.e. logical devices).  All Vulkan tensors in PyTorch
// are associated with a Context to make tensor <-> device affinity explicit.
//
// Vulkan Runtime 初始化 Vulkan 实例，并将 Vulkan 实例初始化的概念与 Vulkan 物理和逻辑设备的初始化及后续交互分离，
// 作为支持多 GPU 的先决条件。Vulkan Runtime 可以查询系统中可用的 Adapter（即物理设备），
// 这些 Adapter 可用于创建 Vulkan 上下文（即逻辑设备）。PyTorch 中的所有 Vulkan 张量都与上下文关联，
// 以明确张量与设备的关系。

enum AdapterSelector {
  First,
};
// 定义 AdapterSelector 枚举类型，用于选择 Adapter 的策略

struct RuntimeConfiguration final {
  bool enableValidationMessages;
  // 是否启用验证消息
  bool initDefaultDevice;
  // 是否初始化默认设备
  AdapterSelector defaultSelector;
  // 默认选择器类型
  uint32_t numRequestedQueues;
  // 请求的队列数量
};

class Runtime final {
 public:
  explicit Runtime(const RuntimeConfiguration);
  // 显式构造函数，根据 RuntimeConfiguration 初始化 Runtime 对象

  // Do not allow copying. There should be only one global instance of this
  // class.
  // 禁止拷贝构造和赋值运算符重载，确保该类只能有一个全局实例

  Runtime(const Runtime&) = delete;
  // 删除拷贝构造函数

  Runtime& operator=(const Runtime&) = delete;
  // 删除拷贝赋值运算符

  Runtime(Runtime&&) noexcept;
  // 移动构造函数

  Runtime& operator=(Runtime&&) = delete;
  // 删除移动赋值运算符

  ~Runtime();
  // 析构函数，释放资源

  using DeviceMapping = std::pair<PhysicalDevice, int32_t>;
  // 定义 DeviceMapping 类型，表示物理设备和整数的对

  using AdapterPtr = std::unique_ptr<Adapter>;
  // 定义 AdapterPtr 类型，表示 Adapter 的智能指针

 private:
  RuntimeConfiguration config_;
  // Runtime 配置对象

  VkInstance instance_;
  // Vulkan 实例对象

  std::vector<DeviceMapping> device_mappings_;
  // 存储 DeviceMapping 的向量

  std::vector<AdapterPtr> adapters_;
  // 存储 AdapterPtr 的向量

  uint32_t default_adapter_i_;
  // 默认 Adapter 的索引

  VkDebugReportCallbackEXT debug_report_callback_;
  // Vulkan 调试报告回调对象

 public:
  inline VkInstance instance() const {
    return instance_;
  }
  // 返回 Vulkan 实例对象的内联函数

  inline Adapter* get_adapter_p() {
    VK_CHECK_COND(
        default_adapter_i_ >= 0 && default_adapter_i_ < adapters_.size(),
        "Pytorch Vulkan Runtime: Default device adapter is not set correctly!");
    return adapters_[default_adapter_i_].get();
  }
  // 返回默认 Adapter 指针的内联函数，检查索引范围

  inline Adapter* get_adapter_p(uint32_t i) {
    VK_CHECK_COND(
        i >= 0 && i < adapters_.size(),
        "Pytorch Vulkan Runtime: Adapter at index ",
        i,
        " is not available!");
    return adapters_[i].get();
  }
  // 返回指定索引处的 Adapter 指针的内联函数，检查索引范围

  inline uint32_t default_adapter_i() const {
    return default_adapter_i_;
  }
  // 返回默认 Adapter 索引的内联函数

  using Selector =
      std::function<uint32_t(const std::vector<Runtime::DeviceMapping>&)>;
  // 定义 Selector 类型，是一个函数对象类型，接受 DeviceMapping 向量并返回 uint32_t

  uint32_t create_adapter(const Selector&);
  // 创建 Adapter 的函数，接受 Selector 函数对象作为参数

};

// The global runtime is retrieved using this function, where it is declared as
// a static local variable.
// 使用此函数获取全局 Runtime 实例，在此函数中声明为静态局部变量。
Runtime* runtime();

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
// 结束条件编译指令，结束 ifdef USE_VULKAN_API 块
```