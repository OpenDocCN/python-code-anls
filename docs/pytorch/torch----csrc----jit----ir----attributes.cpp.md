# `.\pytorch\torch\csrc\jit\ir\attributes.cpp`

```
// 引入头文件：包含了C++标准库和Torch库中定义的必要组件
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/ir.h>

// Torch的jit命名空间，包含了与JIT相关的类和函数
namespace torch::jit {

// GraphAttr类的clone方法的实现
AttributeValue::Ptr GraphAttr::clone() const {
  // 返回一个新的AttributeValue::Ptr，包含当前GraphAttr对象的名称和值的副本
  return Ptr(new GraphAttr(name, value_->copy()));
}

// GraphsAttr类的clone方法的实现
std::unique_ptr<AttributeValue> GraphsAttr::clone() const {
  // 创建一个新的std::vector，用于存储Graph对象的共享指针的副本
  std::vector<std::shared_ptr<Graph>> copy(value_.size());
  // 使用范围循环遍历value_中的每个元素
  for (const auto i : c10::irange(value_.size())) {
    // 将每个Graph对象的副本存储在copy中的相应位置
    copy[i] = value_.at(i)->copy();
  }
  // 返回一个新的GraphsAttr对象，包含当前GraphsAttr对象的名称和copy中的Graph对象副本
  return Ptr(new GraphsAttr(name, std::move(copy)));
}

} // namespace torch::jit
```