# `.\pytorch\torch\csrc\jit\passes\shape_analysis.h`

```
#pragma once

#include <torch/csrc/Export.h>  // 包含 Torch 的导出声明
#include <torch/csrc/jit/ir/ir.h>  // 包含 Torch JIT 的 IR 相关头文件
#include <memory>  // 包含内存管理相关的标准库头文件

namespace torch {
namespace jit {

struct Graph;  // 声明 Graph 结构体

struct propagation_error : std::exception {};  // 定义异常结构 propagation_error

class PropertyPropBase {
  // 用于形状传播和 Dtype/Device 传播的基类
 public:
  explicit PropertyPropBase(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}  // 构造函数，初始化 graph_ 成员变量

  virtual ~PropertyPropBase() = default;  // 虚析构函数

  void propagateBlock(Block* block, bool insert_expands = true);
  // 对块进行传播，insert_expands 参数用于形状推断

  void processIf(Node* node);
  // 处理 If 节点的方法

  void processLoop(Node* node);
  // 处理 Loop 节点的方法

 protected:
  virtual void propagateNode(Node* node, bool insert_expands = true) = 0;
  // 纯虚函数，派生类需要实现的节点传播方法

  void setUnshapedType(Value* o);
  // 设置未形状化的类型

  void setUnshapedType(Node* node);
  // 设置未形状化的类型，针对节点

  std::shared_ptr<Graph> graph_;  // 图对象的智能指针
};

TORCH_API void EraseShapeInformation(const std::shared_ptr<Graph>& graph);
// 删除图中的形状信息的函数声明

TORCH_API void PropagateInputShapes(const std::shared_ptr<Graph>& graph);
// 推广输入形状的函数声明

TORCH_API bool mergeTypes(
    ArrayRef<Value*> lhs,
    ArrayRef<Value*> rhs,
    ArrayRef<Value*> outputs);
// 合并类型的函数声明，接受左右值和输出值的数组引用

} // namespace jit
} // namespace torch
```