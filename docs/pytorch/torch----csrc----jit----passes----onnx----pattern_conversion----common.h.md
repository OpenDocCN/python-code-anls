# `.\pytorch\torch\csrc\jit\passes\onnx\pattern_conversion\common.h`

```
#pragma once


// 声明预处理命令，确保本头文件只被编译一次



#include <torch/csrc/jit/ir/ir.h>


// 包含 Torch 库中的 IR 相关头文件，用于访问和操作中间表示（Intermediate Representation, IR）



namespace torch {
namespace jit {


// 声明命名空间 torch::jit，用于包裹所有 JIT（即时编译）相关的代码



struct IndexingPatternFinder {
 public:
  static std::vector<Node*> FetchSliceAndSelect(const Node* node);

 private:
  static bool IsSameSource(const Node* n, const Node* m);
};


// 定义结构体 IndexingPatternFinder，用于查找索引模式
// 包含两个公共静态成员函数：
// - FetchSliceAndSelect：用于从给定节点中提取切片和选择操作的节点列表
// - 私有静态成员函数 IsSameSource：用于比较两个节点是否来自同一源



} // namespace jit
} // namespace torch


// 结束 torch::jit 命名空间
```