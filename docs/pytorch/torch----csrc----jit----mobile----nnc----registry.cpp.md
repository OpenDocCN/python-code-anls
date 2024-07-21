# `.\pytorch\torch\csrc\jit\mobile\nnc\registry.cpp`

```
#include <torch/csrc/jit/mobile/nnc/registry.h>
// 包含 Torch 库中 NN 模块的注册头文件

namespace torch {
namespace jit {
namespace mobile {
namespace nnc {

// 定义一个名为 NNCKernelRegistry 的注册表，注册类型为 NNCKernel
C10_DEFINE_REGISTRY(NNCKernelRegistry, NNCKernel);

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
// 在 Torch 的命名空间中定义了一个名为 NNCKernelRegistry 的全局注册表，
// 用于注册 NNCKernel 类型的对象和函数。
```