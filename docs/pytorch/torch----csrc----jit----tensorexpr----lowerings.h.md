# `.\pytorch\torch\csrc\jit\tensorexpr\lowerings.h`

```py
// This file defines classes for registering standard lowerings from JIT to TE
// IR.
#pragma once

#include <torch/csrc/jit/ir/ir.h>                     // 引入 JIT 的 IR 定义
#include <torch/csrc/jit/runtime/interpreter.h>      // 引入 JIT 的运行时解释器定义
#include <torch/csrc/jit/tensorexpr/analysis.h>      // 引入 TensorExpr 的分析功能
#include <torch/csrc/jit/tensorexpr/codegen.h>       // 引入 TensorExpr 的代码生成功能
#include <torch/csrc/jit/tensorexpr/tensor.h>        // 引入 TensorExpr 的张量定义

namespace torch {
namespace jit {
namespace tensorexpr {

using ArgNone = std::monostate;                      // 定义表示无参数的类型
using BufList = std::vector<tensorexpr::BufHandle>;  // 定义缓冲列表类型，存储 BufHandle
using DoubleList = std::vector<double>;              // 定义双精度浮点数列表类型
using IntList = std::vector<int64_t>;                // 定义整数列表类型
using ArgValue = std::variant<                       // 定义参数值类型，支持多种可能的值
    tensorexpr::BufHandle,                          // - 缓冲句柄
    tensorexpr::VarHandle,                          // - 变量句柄
    double,                                         // - 双精度浮点数
    int64_t,                                        // - 64 位整数
    bool,                                           // - 布尔值
    BufList,                                        // - 缓冲列表
    DoubleList,                                     // - 双精度浮点数列表
    IntList,                                        // - 整数列表
    std::string,                                    // - 字符串
    ArgNone>;                                       // - 无参数

using NNCLoweringFunction = std::function<Tensor(    // 定义 NNCLoweringFunction 类型，表示从 ArgValue 到 Tensor 的映射函数
    const std::vector<ArgValue>&,                   // - 参数列表
    const std::vector<ExprHandle>&,                 // - 表达式句柄列表
    const std::vector<ExprHandle>&,                 // - 可选的表达式句柄列表
    const std::optional<ScalarType>&,               // - 可选的标量类型
    at::Device)>;                                   // - 设备类型

TORCH_API FunctionSchemaMap<NNCLoweringFunction>& getNNCLoweringRegistry();  // 获取 NNCLoweringFunction 注册表的引用
TORCH_API NNCLoweringFunction getStandardLoweringFor(const std::string& op);  // 获取指定操作的标准降低函数

struct RegisterNNCLoweringsFunction {
  RegisterNNCLoweringsFunction(
      const std::vector<std::string>& schemas,      // 构造函数，接受操作模式字符串列表
      NNCLoweringFunction fn);                      // - NNCLoweringFunction 类型的降低函数
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```