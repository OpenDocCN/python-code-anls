# `.\pytorch\benchmarks\static_runtime\test_utils.h`

```py
// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>
#include <vector>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace c10 {
struct IValue;
}

namespace torch {
namespace jit {

// 定义节点结构体
struct Node;
// 前向声明静态模块类
class StaticModule;

namespace test {

// 给定 JIT 脚本或 IR 脚本中的模型/函数，使用 JIT 解释器和静态运行时运行模型/函数，并比较结果
void testStaticRuntime(
    const std::string& source,              // JIT 或 IR 脚本源代码
    const std::vector<c10::IValue>& args,   // 输入参数
    const std::vector<c10::IValue>& args2 = {},  // 可选的第二组输入参数，默认为空
    const bool use_allclose = false,         // 是否使用 allclose 进行结果比较，默认为否
    const bool use_equalnan = false,         // 是否比较 NaN 时相等，默认为否
    const bool check_resize = true);         // 是否检查大小变化，默认为是

// 从 JIT 脚本中获取图形对象的共享指针
std::shared_ptr<Graph> getGraphFromScript(const std::string& jit_script);

// 从 IR 字符串中获取图形对象的共享指针
std::shared_ptr<Graph> getGraphFromIR(const std::string& ir);

// 检查 StaticModule 中是否存在已处理的具有特定名称的节点
bool hasProcessedNodeWithName(
    torch::jit::StaticModule& smodule,      // 静态模块对象的引用
    const char* name);                      // 节点名称字符串

// 从 at::IValue 对象中获取 at::Tensor 对象
at::Tensor getTensor(const at::IValue& ival);

// 查找 StaticModule 中具有特定类型的节点，并返回该节点的指针
Node* getNodeWithKind(const StaticModule& smodule, const std::string& kind);
// 查找图形对象中具有特定类型的节点，并返回该节点的指针
Node* getNodeWithKind(std::shared_ptr<Graph>& graph, const std::string& kind);

// 检查 StaticModule 中是否存在具有特定类型的节点
bool hasNodeWithKind(const StaticModule& smodule, const std::string& kind);
// 检查图形对象中是否存在具有特定类型的节点
bool hasNodeWithKind(std::shared_ptr<Graph>& graph, const std::string& kind);

// 使用 JIT 运行时与 JIT 生成的结果进行比较
void compareResultsWithJIT(
    StaticRuntime& runtime,                 // 静态运行时对象的引用
    const std::shared_ptr<Graph>& graph,    // JIT 图形对象的共享指针
    const std::vector<c10::IValue>& args,   // 输入参数
    const bool use_allclose = false,        // 是否使用 allclose 进行结果比较，默认为否
    const bool use_equalnan = false);       // 是否比较 NaN 时相等，默认为否

// 比较期望的 IValue 对象和实际的 IValue 对象，可选使用 allclose 和 equalnan 进行比较
void compareResults(
    const IValue& expect,                   // 期望的 IValue 对象
    const IValue& actual,                   // 实际的 IValue 对象
    const bool use_allclose = false,        // 是否使用 allclose 进行结果比较，默认为否
    const bool use_equalnan = false);       // 是否比较 NaN 时相等，默认为否

} // namespace test
} // namespace jit
} // namespace torch
```