# `.\pytorch\aten\src\ATen\detail\CUDAHooksInterface.cpp`

```py
// 包含 ATen 库中 CUDA hooks 接口的头文件
#include <ATen/detail/CUDAHooksInterface.h>

// 包含 C10 库中的 CallOnce 工具函数
#include <c10/util/CallOnce.h>

// 包含标准库中的内存管理功能
#include <memory>

// ATen 命名空间
namespace at {
namespace detail {

// 注意事项提到故意泄漏 CUDA hooks 对象，因为在某些情况下，我们可能需要在运行构造器之前引用 CUDA hooks。
// getCUDAHooks 函数用于获取 CUDA hooks 接口对象。
// 在 JIT 中的融合内核缓存（kernel cache）是触发此更改的示例，该缓存是一个全局变量，缓存 CPU 和 CUDA 内核。
// CUDA 内核在程序销毁时需要与 CUDA hooks 交互，因此允许泄漏 CUDA hooks 对象。
static CUDAHooksInterface* cuda_hooks = nullptr;

// 获取 CUDA hooks 接口对象的函数
const CUDAHooksInterface& getCUDAHooks() {
  // 如果不是 C10_MOBILE 宏定义的情况下，使用 call_once 确保只初始化一次 CUDA hooks
  static c10::once_flag once;
  c10::call_once(once, [] {
    // 尝试从 CUDAHooksRegistry 创建 CUDA hooks 对象，如果失败则分配新的 CUDAHooksInterface 对象
    cuda_hooks =
        CUDAHooksRegistry()->Create("CUDAHooks", CUDAHooksArgs{}).release();
    if (!cuda_hooks) {
      cuda_hooks = new CUDAHooksInterface();
    }
  });
  
  // 如果是 C10_MOBILE，直接分配一个新的 CUDAHooksInterface 对象给 cuda_hooks
  if (cuda_hooks == nullptr) {
    cuda_hooks = new CUDAHooksInterface();
  }
  
  // 返回 CUDA hooks 接口对象引用
  return *cuda_hooks;
}

} // namespace detail

// 定义 CUDAHooksRegistry，注册 CUDA hooks 接口
C10_DEFINE_REGISTRY(CUDAHooksRegistry, CUDAHooksInterface, CUDAHooksArgs)

} // namespace at
```