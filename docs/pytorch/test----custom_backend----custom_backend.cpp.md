# `.\pytorch\test\custom_backend\custom_backend.cpp`

```
// 包含自定义后端头文件
#include "custom_backend.h"
// 包含 Torch 后端预处理的头文件
#include <torch/csrc/jit/backends/backend_preprocess.h>

// Torch 命名空间
namespace torch {
  // 自定义后端命名空间
  namespace custom_backend {
    // 匿名命名空间，用于限制符号的作用域
    namespace {
      // 后端名称的常量字符串
      constexpr auto kBackendName = "custom_backend";
      // 使用自定义后端类注册 Torch 后端
      static auto cls = torch::jit::backend<CustomBackend>(kBackendName);
      // 注册后端预处理函数到自定义后端
      static auto pre_reg = torch::jit::backend_preprocess_register(kBackendName, preprocess);
    }

    // 获取后端名称的函数
    std::string getBackendName() {
      return std::string(kBackendName);
    }
  } // custom_backend
} // torch
```