# `.\pytorch\torch\csrc\jit\mobile\nnc\aot_compiler.h`

```py
// 预处理指令，指示编译器在编译本文件时只包含一次
#pragma once

// 包含 Torch 的导出头文件
#include <torch/csrc/Export.h>
// 包含 Torch JIT 的中间表示(IR)相关头文件
#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 移动端神经网络编译上下文相关头文件
#include <torch/csrc/jit/mobile/nnc/context.h>

// Torch 命名空间
namespace torch {
// Torch JIT 命名空间
namespace jit {
// Torch 移动端 JIT 命名空间
namespace mobile {
// Torch 移动端神经网络编译命名空间
namespace nnc {

// 执行给定模型方法的Ahead Of Time(提前编译)编译，返回编译后的函数和 LLVM 汇编代码
TORCH_API std::pair<std::unique_ptr<Function>, const std::string> aotCompile(
    // 方法名
    const std::string& method_name,
    // 共享指针，指向表示子图的图(Graph)对象
    std::shared_ptr<Graph>& subgraph,
    // 多维度数组的大小
    const std::vector<std::vector<int64_t>>& sizes,
    // 数据类型向量
    const std::vector<at::ScalarType>& types,
    // 核心函数名，默认为"func"
    const std::string& kernel_func_name = "func");

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
```