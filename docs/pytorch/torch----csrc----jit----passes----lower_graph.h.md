# `.\pytorch\torch\csrc\jit\passes\lower_graph.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 IR 模块头文件

namespace torch {
namespace jit {

using ModulePtr = c10::intrusive_ptr<c10::ivalue::Object>;
// 定义 ModulePtr 别名，指向 c10::ivalue::Object 类型的智能指针

// 给定一个方法的图表，其中第一个参数是 %self，在新图中降低它，将所有属性访问替换为图的显式输入
// （而不是在 %self 上执行 prim::GetAttr 后的结果）。
//
// 返回一个元组 (graph, parameters)，其中图的最后 module.parameters.size() 个输入是此方法中使用的可训练参数。
// 其余输入是函数的真正输入。
TORCH_API std::pair<std::shared_ptr<Graph>, std::vector<IValue>> LowerGraph(
    Graph& graph,
    const ModulePtr& self);
// LowerGraph 函数声明，接受一个图对象和 ModulePtr 智能指针作为参数，返回一个包含图和参数的 pair 对象

} // namespace jit
} // namespace torch
// 命名空间闭合
```