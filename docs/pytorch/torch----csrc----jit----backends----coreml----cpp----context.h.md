# `.\pytorch\torch\csrc\jit\backends\coreml\cpp\context.h`

```py
#ifndef PTM_COREML_Context_h
#define PTM_COREML_Context_h

#include <atomic>       // 引入原子操作库，用于多线程安全操作
#include <string>       // 引入字符串库，用于处理字符串操作

namespace torch {       // 命名空间 torch
namespace jit {         // 命名空间 jit
namespace mobile {      // 命名空间 mobile
namespace coreml {      // 命名空间 coreml

struct ContextInterface {   // 定义接口 ContextInterface
  virtual ~ContextInterface() = default;   // 虚析构函数，默认实现
  virtual void setModelCacheDirectory(std::string path) = 0;   // 纯虚函数，设置模型缓存目录的接口方法
};

class BackendRegistrar {   // 定义类 BackendRegistrar
 public:
  explicit BackendRegistrar(ContextInterface* ctx);   // 显式构造函数，接受 ContextInterface 指针参数
};

void setModelCacheDirectory(std::string path);   // 设置模型缓存目录的全局函数声明

} // namespace coreml
} // namespace mobile
} // namespace jit
} // namespace torch

#endif   // 结束条件编译指令
```