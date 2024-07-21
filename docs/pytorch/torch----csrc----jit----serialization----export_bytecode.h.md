# `.\pytorch\torch\csrc\jit\serialization\export_bytecode.h`

```
#pragma once
// 只允许头文件被包含一次的预处理指令

#include <tuple>
// 引入元组的头文件

#include <unordered_map>
// 引入无序映射的头文件

#include <ATen/core/function_schema.h>
// 引入函数模式的头文件

#include <ATen/core/ivalue.h>
// 引入IValue的头文件

#include <ATen/core/jit_type.h>
// 引入JIT类型的头文件

#include <ATen/core/qualified_name.h>
// 引入限定名称的头文件

#include <torch/csrc/jit/backends/backend_debug_handler.h>
// 引入后端调试处理的头文件

#include <torch/csrc/jit/mobile/function.h>
// 引入移动功能的头文件

#include <torch/csrc/jit/mobile/module.h>
// 引入移动模块的头文件

#include <torch/csrc/jit/runtime/interpreter.h>
// 引入解释器的头文件

#include <torch/csrc/jit/serialization/type_name_uniquer.h>
// 引入类型名称唯一化的头文件

namespace torch::jit {
// 进入torch::jit命名空间

struct TORCH_API CompilationOptions {
  bool incl_interface_call = false;
  // 是否包含接口调用，默认为false

  bool enable_default_value_for_unspecified_arg = false;
  // 是否为未指定的参数启用默认值，默认为false

  bool enable_default_args_before_out_args = true;
  // 是否在输出参数之前启用默认参数，默认为true

  bool enable_emit_promoted_ops = true;
  // 是否启用推广操作的发出，默认为true

  int model_version = caffe2::serialize::kProducedBytecodeVersion;
  // 模型版本号，默认为caffe2::serialize::kProducedBytecodeVersion
};

TORCH_API mobile::Module jitModuleToMobile(
    const Module& module,
    const CompilationOptions& options);
// 将JIT模块转换为移动模块的函数声明

mobile::Code compileGraphToMobileCode(
    const std::string& name,
    const std::shared_ptr<Graph>& graph,
    const CompilationOptions& compilation_options,
    BackendDebugInfoRecorder& debug_info_recorder);
// 将图编译为移动代码的函数声明

TORCH_API std::unique_ptr<mobile::Function> convertJitFunctionToMobileFunction(
    const GraphFunction& function,
    const CompilationOptions& options);
// 将JIT函数转换为移动函数的函数声明

TORCH_API IValue convertMobileFunctionToCodeTable(
    const mobile::Function& func,
    const CompilationOptions& compilation_options);
// 将移动函数转换为代码表的函数声明

} // namespace torch::jit
// 结束torch::jit命名空间
```