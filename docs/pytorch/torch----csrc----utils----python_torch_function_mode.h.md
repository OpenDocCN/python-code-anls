# `.\pytorch\torch\csrc\utils\python_torch_function_mode.h`

```
#pragma once
// 包含头文件 PythonTorchFunctionTLS.h，该头文件应属于 ATen 库
#include <ATen/PythonTorchFunctionTLS.h>

// 声明命名空间 torch::overrides
namespace torch::overrides {

// 定义结构体 StashTorchFunctionModeGuard
struct StashTorchFunctionModeGuard {
  // 构造函数，初始化时从栈中弹出当前的 Torch 函数模式
  StashTorchFunctionModeGuard() {
    cur_mode_ = at::impl::PythonTorchFunctionTLS::pop_stack();
  }
  // 析构函数，对象销毁时将之前弹出的 Torch 函数模式重新压入栈中
  ~StashTorchFunctionModeGuard() {
    at::impl::PythonTorchFunctionTLS::push_onto_stack(cur_mode_);
  }

  // 返回当前模式的安全 Python 对象的共享指针
  const std::shared_ptr<c10::SafePyObject>& get_cur_mode() {
    return cur_mode_;
  }

 private:
  // 成员变量，用于存储当前 Torch 函数模式的安全 Python 对象的共享指针
  std::shared_ptr<c10::SafePyObject> cur_mode_;
};

} // namespace torch::overrides
```