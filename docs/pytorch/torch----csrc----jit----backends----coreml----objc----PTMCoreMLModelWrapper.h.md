# `.\pytorch\torch\csrc\jit\backends\coreml\objc\PTMCoreMLModelWrapper.h`

```
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/backends/coreml/objc/PTMCoreMLExecutor.h>
#include <torch/csrc/jit/backends/coreml/objc/PTMCoreMLTensorSpec.h>

namespace torch {
namespace jit {
namespace mobile {
namespace coreml {

// 定义一个名为 MLModelWrapper 的类，继承自 CustomClassHolder
class MLModelWrapper : public CustomClassHolder {
 public:
  // 指向 PTMCoreMLExecutor 对象的指针
  PTMCoreMLExecutor* executor;
  // 存储 TensorSpec 对象的向量
  std::vector<TensorSpec> outputs;

  // 禁止默认构造函数
  MLModelWrapper() = delete;

  // 构造函数，接受一个 PTMCoreMLExecutor 对象指针作为参数
  MLModelWrapper(PTMCoreMLExecutor* executor) : executor(executor) {
    // 增加 executor 的引用计数
    [executor retain];
  }

  // 拷贝构造函数，复制另一个 MLModelWrapper 对象的 executor 和 outputs
  MLModelWrapper(const MLModelWrapper& oldObject) {
    executor = oldObject.executor;
    outputs = oldObject.outputs;
    // 增加 executor 的引用计数
    [executor retain];
  }

  // 移动构造函数，获取另一个 MLModelWrapper 对象的 executor 和 outputs
  MLModelWrapper(MLModelWrapper&& oldObject) {
    executor = oldObject.executor;
    outputs = oldObject.outputs;
    // 增加 executor 的引用计数
    [executor retain];
  }

  // 析构函数，释放 executor 的引用计数
  ~MLModelWrapper() {
    [executor release];
  }
};

} // namespace coreml
} // namespace mobile
} // namespace jit
} // namespace torch
```