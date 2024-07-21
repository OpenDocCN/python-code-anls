# `.\pytorch\aten\src\ATen\metal\Context.h`

```py
#ifndef MetalContext_h
#define MetalContext_h

#include <atomic>  // 引入原子操作库，用于多线程安全的操作

#include <ATen/Tensor.h>  // 引入 ATen 库中的 Tensor 类

namespace at::metal {

struct MetalInterface {
  virtual ~MetalInterface() = default;  // 虚析构函数，基类的默认析构函数
  virtual bool is_metal_available() const = 0;  // 纯虚函数，判断 Metal 是否可用
  virtual at::Tensor& metal_copy_(at::Tensor& self, const at::Tensor& src) const = 0;  // 纯虚函数，Metal 实现的 Tensor 拷贝
};

extern std::atomic<const MetalInterface*> g_metal_impl_registry;  // 原子指针，注册 Metal 接口的全局变量

class MetalImplRegistrar {
 public:
  explicit MetalImplRegistrar(MetalInterface*);  // Metal 接口注册类构造函数声明
};

at::Tensor& metal_copy_(at::Tensor& self, const at::Tensor& src);  // Metal 版本的 Tensor 拷贝函数声明

} // namespace at::metal

namespace at::native {
bool is_metal_available();  // 判断 Metal 是否可用的函数声明
} // namespace at::native

#endif /* MetalContext_h */
```