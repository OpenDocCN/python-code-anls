# `.\pytorch\torch\csrc\jit\python\python_tracer.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/csrc/jit/frontend/source_range.h>
// 引入 Torch 的源代码范围头文件

#include <torch/csrc/jit/frontend/tracer.h>
// 引入 Torch 的追踪器头文件

#include <torch/csrc/python_headers.h>
// 引入 Torch 的 Python 头文件

#include <torch/csrc/utils/pybind.h>
// 引入 Torch 的 PyBind11 实用工具头文件

#include <memory>
// 引入 C++ 标准库的内存管理头文件

#include <string>
// 引入 C++ 标准库的字符串处理头文件

namespace torch::jit {

struct Module;
// Torch JIT 模块结构声明

namespace tracer {
// Torch 追踪器命名空间

void initPythonTracerBindings(PyObject* module);
// 初始化 Python 追踪器绑定函数声明，接受 Python 模块对象作为参数

SourceRange getPythonInterpreterSourceRange();
// 获取 Python 解释器源代码范围的函数声明

Node* preRecordPythonTrace(
    THPObjectPtr pyobj,
    const std::string& arg_types,
    at::ArrayRef<autograd::Variable> inputs,
    std::vector<THPObjectPtr> scalar_args);
// 预录制 Python 追踪的函数声明，接受 Python 对象、参数类型字符串、输入变量数组引用和标量参数数组作为参数

std::pair<std::shared_ptr<Graph>, Stack> createGraphByTracingWithDict(
    const py::function& func,
    const py::dict& inputs_dict,
    Stack inputs,
    const py::function& var_name_lookup_fn,
    bool strict,
    bool force_outplace,
    Module* self = nullptr,
    const std::vector<std::string>& argument_names = {});
// 使用字典创建追踪图的函数声明，接受 Python 函数对象、输入字典、输入堆栈、变量名称查找函数、严格模式、强制独立输出、模块自身和参数名数组作为参数

std::pair<std::shared_ptr<Graph>, Stack> createGraphByTracing(
    const py::function& func,
    Stack inputs,
    const py::function& var_name_lookup_fn,
    bool strict,
    bool force_outplace,
    Module* self = nullptr,
    const std::vector<std::string>& argument_names = {});
// 追踪创建图的函数声明，接受 Python 函数对象、输入堆栈、变量名称查找函数、严格模式、强制独立输出、模块自身和参数名数组作为参数

} // namespace tracer
} // namespace torch::jit
```