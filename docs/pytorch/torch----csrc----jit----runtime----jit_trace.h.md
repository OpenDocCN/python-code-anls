# `.\pytorch\torch\csrc\jit\runtime\jit_trace.h`

```py
#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 JIT IR 相关头文件

#include <memory>
// 包含 C++ 标准库中的内存管理相关头文件

namespace torch::jit {
// 进入 torch::jit 命名空间

TORCH_API std::shared_ptr<Graph> TraceGraph(
    std::shared_ptr<Graph> graph,
    Stack& stack);
// 声明 TraceGraph 函数，该函数返回一个 std::shared_ptr 智能指针，
// 指向 Graph 类型的对象，接受一个图对象的智能指针和一个栈引用作为参数

} // namespace torch::jit
// 结束 torch::jit 命名空间
```