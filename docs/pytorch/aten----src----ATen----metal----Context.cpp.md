# `.\pytorch\aten\src\ATen\metal\Context.cpp`

```
#include <atomic>  // 包含 C++ 标准库中的原子操作支持

#include <ATen/Tensor.h>  // 包含 PyTorch 的 Tensor 类定义
#include <ATen/metal/Context.h>  // 包含 PyTorch Metal 后端的上下文定义

namespace at::metal {

std::atomic<const MetalInterface*> g_metal_impl_registry;  // 定义一个原子指针，用于存储 Metal 接口的注册表

MetalImplRegistrar::MetalImplRegistrar(MetalInterface* impl) {
  g_metal_impl_registry.store(impl);  // 在 Metal 接口注册表中存储给定的 Metal 接口实现对象
}

at::Tensor& metal_copy_(at::Tensor& self, const at::Tensor& src) {
  auto p = at::metal::g_metal_impl_registry.load();  // 加载当前注册的 Metal 接口实现对象
  if (p) {
    return p->metal_copy_(self, src);  // 如果 Metal 接口对象存在，调用其实现的 metal_copy_ 方法进行张量的复制操作
  }
  AT_ERROR("Metal backend was not linked to the build");  // 如果 Metal 后端未链接到构建系统，则抛出错误
}
} // namespace at::metal

namespace at::native {
bool is_metal_available() {
  auto p = at::metal::g_metal_impl_registry.load();  // 加载当前注册的 Metal 接口实现对象
  return p ? p->is_metal_available() : false;  // 返回 Metal 接口对象是否可用的布尔值，如果对象存在则调用其 is_metal_available 方法，否则返回 false
}

} // namespace at::native
```