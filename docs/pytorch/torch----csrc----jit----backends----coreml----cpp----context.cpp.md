# `.\pytorch\torch\csrc\jit\backends\coreml\cpp\context.cpp`

```
# 包含 Torch 的 CoreML 后端上下文头文件
#include <torch/csrc/jit/backends/coreml/cpp/context.h>

# 包含原子操作相关的头文件
#include <atomic>

# 定义命名空间 torch::jit::mobile::coreml
namespace torch {
namespace jit {
namespace mobile {
namespace coreml {

# 定义全局原子指针 g_coreml_ctx_registry，用于存储 CoreML 上下文接口对象
std::atomic<ContextInterface*> g_coreml_ctx_registry;

# BackendRegistrar 类的构造函数实现
BackendRegistrar::BackendRegistrar(ContextInterface* ctx) {
  # 将传入的上下文指针存储到全局原子指针 g_coreml_ctx_registry 中
  g_coreml_ctx_registry.store(ctx);
}

# 设置模型缓存目录的函数实现
void setModelCacheDirectory(std::string path) {
  # 加载全局原子指针中的上下文接口对象指针
  auto p = g_coreml_ctx_registry.load();
  # 如果指针非空，则调用其方法设置模型缓存目录
  if (p) {
    p->setModelCacheDirectory(path);
  }
}

} // namespace coreml
} // namespace mobile
} // namespace jit
} // namespace torch
```