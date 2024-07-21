# `.\pytorch\torch\csrc\jit\backends\backend_debug_info.cpp`

```py
// 包含头文件<c10/macros/Macros.h>，定义了一些常用的宏
#include <c10/macros/Macros.h>
// 包含头文件<torch/csrc/jit/backends/backend_debug_info.h>，提供了后端调试信息的接口
#include <torch/csrc/jit/backends/backend_debug_info.h>

// torch命名空间
namespace torch {
// JIT（Just-In-Time）编译器命名空间
namespace jit {
// 后端命名空间
namespace backend {
// 匿名命名空间，用于定义内部链接的静态变量和函数
namespace {

// 根据宏BUILD_LITE_INTERPRETER的定义，选择性地定义PyTorchBackendDebugInfoDummy或PyTorchBackendDebugInfo的类
#ifdef BUILD_LITE_INTERPRETER
// 静态变量cls，使用torch::class_模板定义了PyTorchBackendDebugInfoDummy类，初始化时绑定了相关的命名空间和类名
static auto cls = torch::class_<PyTorchBackendDebugInfoDummy>(
                      kBackendUtilsNamespace,  // 使用宏kBackendUtilsNamespace定义的命名空间
                      kBackendDebugInfoClass)  // 使用宏kBackendDebugInfoClass定义的类名
                      .def(torch::init<>());  // 定义构造函数初始化函数
#else
// 静态变量cls，使用torch::class_模板定义了PyTorchBackendDebugInfo类，初始化时绑定了相关的命名空间和类名
static auto cls = torch::class_<PyTorchBackendDebugInfo>(
                      kBackendUtilsNamespace,  // 使用宏kBackendUtilsNamespace定义的命名空间
                      kBackendDebugInfoClass)  // 使用宏kBackendDebugInfoClass定义的类名
                      .def(torch::init<>());  // 定义构造函数初始化函数
#endif

} // namespace
} // namespace backend
} // namespace jit
} // namespace torch
```