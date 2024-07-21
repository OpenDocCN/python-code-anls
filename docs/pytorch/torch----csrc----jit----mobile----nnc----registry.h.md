# `.\pytorch\torch\csrc\jit\mobile\nnc\registry.h`

```py
#pragma once

#include <c10/util/Exception.h>  // 引入异常处理工具
#include <c10/util/Registry.h>   // 引入注册表工具

namespace torch {
namespace jit {
namespace mobile {
namespace nnc {

using nnc_kernel_function_type = int(void**);  // 定义函数指针类型 nnc_kernel_function_type

struct TORCH_API NNCKernel {
  virtual ~NNCKernel() = default;          // 虚析构函数，用于多态销毁对象
  virtual int execute(void** /* args */) = 0;  // 纯虚函数，需要在派生类中实现，执行核函数
};

TORCH_DECLARE_REGISTRY(NNCKernelRegistry, NNCKernel);  // 声明 NNCKernelRegistry 注册表

#define REGISTER_NNC_KERNEL(id, kernel, ...)     \
  extern "C" {                                   \
  nnc_kernel_function_type kernel;               \
  }                                              \
  struct NNCKernel_##kernel : public NNCKernel { \
    int execute(void** args) override {          \
      return kernel(args);                       \
    }                                            \
  };                                             \
  C10_REGISTER_TYPED_CLASS(NNCKernelRegistry, id, NNCKernel_##kernel);  // 注册核函数的类型和 ID

namespace registry {

inline bool has_nnc_kernel(const std::string& id) {
  return NNCKernelRegistry()->Has(id);  // 检查是否存在指定 ID 的核函数
}

inline std::unique_ptr<NNCKernel> get_nnc_kernel(const std::string& id) {
  return NNCKernelRegistry()->Create(id);  // 创建指定 ID 的核函数对象
}

} // namespace registry

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
```