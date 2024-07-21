# `.\pytorch\torch\csrc\jit\backends\backend_interface.h`

```
#pragma once
// 预处理指令，确保头文件只包含一次

#include <torch/custom_class.h>
// 包含 Torch 自定义类相关的头文件

namespace torch {
namespace jit {

// Torch JIT 后端接口的定义
class TORCH_API PyTorchBackendInterface : public torch::CustomClassHolder {
 public:
  // 默认构造函数，不抛出异常
  PyTorchBackendInterface() noexcept;
  // 虚析构函数，确保正确释放资源
  ~PyTorchBackendInterface() override;

  // 判断后端是否可用于处理委托调用
  virtual bool is_available() = 0;

  // 编译给定 \p processed 中的模块，使用 \p method_compile_spec 中提供的细节
  // 每个应由后端编译的模块方法。 \p method_compile_spec 应为类型为 Dict<string, Any> 的字典。
  // \returns 类型为 Dict<string, Any> 的字典，包含每个可以在后端运行的方法的后端句柄（即 \p method_compile_spec 中的每个键）。
  virtual c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) = 0;

  // 使用 \p handle 指定的方法和 \p inputs 执行方法。 \returns 输出作为元组。
  virtual c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) = 0;
};
} // namespace jit
} // namespace torch
```