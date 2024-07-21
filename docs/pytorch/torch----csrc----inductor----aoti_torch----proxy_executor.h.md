# `.\pytorch\torch\csrc\inductor\aoti_torch\proxy_executor.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/core/ivalue.h>
// 包含 ATen 库的 IValue 头文件

#include <c10/macros/Export.h>
// 包含 c10 库的导出宏定义头文件

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
// 包含 Torch 库的某个 C API 头文件

namespace torch {
namespace aot_inductor {

class ProxyExecutor {
 public:
  ProxyExecutor() {}
  // 默认构造函数，用于创建 ProxyExecutor 对象

  virtual ~ProxyExecutor() {}
  // 虚析构函数，确保子类正确释放资源

  virtual void call_function(
      int extern_node_index,
      int num_ints,
      int64_t* flatten_int_args,
      int num_tensors,
      AtenTensorHandle* flatten_tensor_args) = 0;
  // 纯虚函数声明，用于子类实现具体的函数调用逻辑
};

} // namespace aot_inductor
} // namespace torch
```