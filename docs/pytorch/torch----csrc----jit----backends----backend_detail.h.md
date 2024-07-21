# `.\pytorch\torch\csrc\jit\backends\backend_detail.h`

```py
#pragma once
// 防止头文件被多次包含的预处理指令

#include <torch/csrc/jit/api/module.h>
// 引入 Torch 的模块 API 头文件

#include <ATen/core/jit_type.h>
// 引入 ATen 核心的 JIT 类型头文件

#include <functional>
// 引入 C++ 标准库中的 functional 头文件，用于支持函数对象

namespace torch {
namespace jit {

using DebugHandleType = int64_t;
// 定义 DebugHandleType 作为 int64_t 类型的别名，用于表示调试句柄

using NodeToDebugHandle = std::unordered_map<Node*, DebugHandleType>;
// 定义 NodeToDebugHandle 类型，是一个从 Node* 到 DebugHandleType 的无序映射，用于存储节点到调试句柄的映射关系

using BackendDebugHandleGenerator =
    std::function<NodeToDebugHandle(const std::shared_ptr<Graph>&)>;
// 定义 BackendDebugHandleGenerator 类型，是一个函数对象类型，接受 std::shared_ptr<Graph> 参数并返回 NodeToDebugHandle 对象，
// 用于生成后端调试句柄的函数对象类型

namespace detail {

using BackendPreprocessFunction = std::function<c10::IValue(
    const Module&,
    const c10::Dict<IValue, IValue>&,
    const BackendDebugHandleGenerator& generate_debug_handles)>;
// 定义 BackendPreprocessFunction 类型，是一个函数对象类型，接受 Module、c10::Dict<IValue, IValue> 和 BackendDebugHandleGenerator 参数，
// 返回 c10::IValue 对象，用于表示后端预处理函数对象类型

TORCH_API void registerBackendPreprocessFunction(
    const std::string& name,
    const BackendPreprocessFunction& preprocess);
// 声明一个函数 registerBackendPreprocessFunction，用于注册后端预处理函数，
// 接受名称和 BackendPreprocessFunction 函数对象作为参数

bool hasBackendPreprocessFunction(const std::string& name);
// 声明一个函数 hasBackendPreprocessFunction，用于检查是否存在指定名称的后端预处理函数

BackendPreprocessFunction getBackendPreprocessFunction(const std::string& name);
// 声明一个函数 getBackendPreprocessFunction，用于获取指定名称的后端预处理函数对象

TORCH_API Module codegen_backend_module(
    const std::string& backend_name,
    const Module& orig_module,
    const c10::Dict<IValue, IValue>& method_compile_spec,
    const c10::DictTypePtr& any_dict_ty);
// 声明一个函数 codegen_backend_module，用于生成指定后端名称的模块，
// 接受后端名称、原始模块、方法编译规范和字典类型参数，并返回生成的模块对象

} // namespace detail
} // namespace jit
} // namespace torch
// 命名空间尾部
```