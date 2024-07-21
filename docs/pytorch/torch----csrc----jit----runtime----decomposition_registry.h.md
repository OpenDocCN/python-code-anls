# `.\pytorch\torch\csrc\jit\runtime\decomposition_registry.h`

```
#pragma once
// 这个文件是临时的，直到native_functions.yaml和derivatives.yaml被合并。
// 理想情况下，所有内容都应该放入native_functions.yaml

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

// 获取给定函数模式的分解图（Graph）的可选共享指针
TORCH_API std::optional<std::shared_ptr<Graph>> GetDecomposition(
    const FunctionSchema& schema);

// 注册给定函数模式的分解图（Graph）
TORCH_API void RegisterDecomposition(
    const FunctionSchema& schema,
    std::shared_ptr<Graph> g);

// 运行给定图（Graph）的分解操作
TORCH_API void RunDecompositions(std::shared_ptr<Graph> g);

// 获取给定函数模式的分解函数的可选指针
TORCH_API std::optional<GraphFunction*> GetDecompositionFunction(
    const FunctionSchema& schema);

// 在C++中调用时，建议将其分配给静态局部变量
// 获取给定函数模式的分解执行器（Executor）的函数指针
TORCH_API Function* GetDecompositionExecutor(const char* schema_literal);

// 获取给定函数模式的分解执行器（Executor）的函数指针
TORCH_API Function* GetDecompositionExecutor(const FunctionSchema& schema);

// 运行JIT分解，操作符句柄（OperatorHandle）和堆栈（Stack）的调用
TORCH_API void run_jit_decomposition(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack);

// 检查给定函数模式是否有JIT分解
TORCH_API bool has_jit_decomposition(const FunctionSchema& schema);

} // namespace torch::jit
```