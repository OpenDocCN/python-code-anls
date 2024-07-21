# `.\pytorch\torch\csrc\jit\backends\backend_preprocess.h`

```
#pragma once
// 包含 Torch 的 JIT 后端细节头文件
#include <torch/csrc/jit/backends/backend_detail.h>

// Torch 命名空间
namespace torch {
// JIT 命名空间
namespace jit {

// 定义了一个后端预处理注册类
class backend_preprocess_register {
  // 后端名称字符串
  std::string backend_name_;

 public:
  // 构造函数，接受后端名称和预处理函数，并注册
  backend_preprocess_register(
      const std::string& name,
      const detail::BackendPreprocessFunction& preprocess)
      : backend_name_(name) {
    // 调用细节命名空间中的函数，注册后端预处理函数
    detail::registerBackendPreprocessFunction(name, preprocess);
  }
};

} // namespace jit
} // namespace torch
```