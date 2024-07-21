# `.\pytorch\aten\src\ATen\vulkan\Context.cpp`

```py
#include <atomic>  // 引入原子操作库，用于多线程安全的操作

#include <ATen/Tensor.h>  // 引入 PyTorch 的 Tensor 类定义
#include <ATen/vulkan/Context.h>  // 引入 PyTorch Vulkan 的上下文定义

#ifdef USE_VULKAN_API
#include <ATen/native/vulkan/api/Context.h>  // 如果定义了 USE_VULKAN_API，引入 Vulkan API 的上下文定义
#endif /* USE_VULKAN_API */

namespace at {
namespace vulkan {

std::atomic<const VulkanImplInterface*> g_vulkan_impl_registry;  // 定义全局原子指针，用于注册 Vulkan 实现接口

VulkanImplRegistrar::VulkanImplRegistrar(VulkanImplInterface* impl) {
  g_vulkan_impl_registry.store(impl);  // 将传入的 Vulkan 实现接口指针存储到全局原子指针中
}

at::Tensor& vulkan_copy_(at::Tensor& self, const at::Tensor& src) {
  auto p = at::vulkan::g_vulkan_impl_registry.load();  // 从全局原子指针中加载 Vulkan 实现接口指针
  if (p) {
    return p->vulkan_copy_(self, src);  // 如果接口指针非空，则调用 Vulkan 实现的复制函数
  }
  AT_ERROR("Vulkan backend was not linked to the build");  // 如果 Vulkan 后端未链接到构建系统，则报错
}

} // namespace vulkan

namespace native {
bool is_vulkan_available() {
#ifdef USE_VULKAN_API
  return native::vulkan::api::available();  // 如果定义了 USE_VULKAN_API，调用 Vulkan API 的可用性检查函数
#else
  auto p = at::vulkan::g_vulkan_impl_registry.load();  // 否则从全局原子指针中加载 Vulkan 实现接口指针
  return p ? p->is_vulkan_available() : false;  // 如果接口指针非空，则调用接口的 Vulkan 可用性检查函数，否则返回 false
#endif
}
} // namespace native

} // namespace at
```