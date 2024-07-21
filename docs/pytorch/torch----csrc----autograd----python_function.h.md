# `.\pytorch\torch\csrc\autograd\python_function.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <torch/csrc/python_headers.h>
// 包含 Torch 的 Python 头文件

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/object_ptr.h>
// 包含 Torch 的自动微分相关头文件

#include <c10/core/DeviceGuard.h>
#include <c10/util/Optional.h>
// 包含 C10 库的相关头文件

#include <memory>
#include <optional>
#include <vector>
// 包含 C++ 标准库的相关头文件

namespace torch::jit {
struct Graph;
}
// Torch JIT 模块的命名空间及 Graph 结构体声明

namespace torch::autograd {

// 一个由 Python 对象实现的函数（即 THPFunction）。
// 对 'apply' 的调用被转发到 Python 方法的实现。
struct PyNode : public Node {
  // 构造函数，接受一个 THPObjectPtr 对象并释放其所有权
  PyNode(THPObjectPtr obj) : obj(obj.release()) {}

  // 将变量列表转换为 Python 参数元组
  PyObject* to_py_args(
      const variable_list& inputs,
      at::OptionalDeviceGuard* device_guard);

  // 将 Python 对象转换为变量列表
  variable_list to_variable_list(
      const PyObject* r,
      const std::vector<bool>& is_variable_input);

  // 应用函数，接受变量列表并返回变量列表
  variable_list apply(variable_list&& inputs) override;

  // 延迟到 Dynamo 的函数，接受变量列表和可选的编译器对象，并返回变量列表
  variable_list defer_to_dynamo(
      variable_list&& inputs,
      std::optional<PyObject*> compiler);

  // 释放变量函数
  void release_variables() override;

  // 返回节点名称的函数
  std::string name() const override;

  // 检查是否可追踪的函数
  bool is_traceable() override;

  // 编译参数的函数，接受编译节点参数的引用
  void compiled_args(CompiledNodeArgs& args) override;

  // 应用保存的变量的函数，接受变量列表和保存的变量交换对象，并返回变量列表
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  // 检查是否应提升编译自动微分的函数
  bool compiled_autograd_should_lift() const;

  // 包裹的 THPFunction 对象，拥有其所有权
  PyObject* obj;

  // 与该节点的后向传播对应的 AutogradCompilerCall::hooks 索引
  std::optional<int> _backward_idx;

  // 与该节点的 backward_state 对应的 AutogradCompilerCall::hooks 索引
  std::optional<int> _backward_state_idx;

  // 析构函数，释放 Python 对象的资源
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~PyNode() override {
    // 无法在此类中使用 THPObjectPtr 作为字段；析构函数不会释放 GIL！
    // 如果忘记手动释放，Python 将会崩溃。
    // 如果 Python 已经终止，则泄漏包装的 Python 对象。
    if (Py_IsInitialized()) {
      pybind11::gil_scoped_acquire gil; // 获取 GIL，确保安全访问 Python
      Py_DECREF(obj); // 释放 Python 对象的引用计数
    }
  }
};

/**
 * 如果对象不是元组，则将其转换为元组。如果原始对象不是元组，则返回 true。
 */
inline bool ensure_tuple(THPObjectPtr& obj) {
  // 检查对象是否为元组
  if (PyTuple_Check(obj.get()))
    return false;

  // 如果不是元组，则创建一个包含原始对象的元组
  PyObject* tuple = PyTuple_New(1);
  if (!tuple)
    throw python_error(); // 抛出 Python 错误异常
  PyTuple_SET_ITEM(tuple, 0, obj.release()); // 设置元组的第一个元素
  obj = tuple; // 将 obj 指向新创建的元组对象
  return true;
}

} // namespace torch::autograd
// 结束 torch::autograd 命名空间的声明
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// 定义一个结构体 THPFunction，表示一个 PyTorch 自定义函数的对象
struct THPFunction {
    PyObject_HEAD  // Python 对象的头部信息

    PyObject* needs_input_grad;  // 指示是否需要输入梯度的 Python 对象指针

    // 要保存的变量的 Python 元组，由 Python 的 'save_for_backward' 设置。
    // 如果为 nullptr，则没有保存任何张量。
    PyObject* to_save;

    // 不可微分张量的 Python 元组，由 Python 的 'mark_non_differentiable' 设置。
    // 如果为 nullptr，则没有非可微分张量。
    PyObject* non_differentiable;

    // 在前向传播中进行原位更新的张量的 Python 元组。
    // 由 Python 的 'mark_dirty' 设置。如果为 nullptr，则没有原位修改的张量。
    PyObject* dirty_tensors;

    // 布尔值，指示是否将未定义输出梯度张量实现为全零张量。
    // 由 Python 的 'set_materialize_grads' 设置。默认为 true。
    bool materialize_grads;

    // 布尔值，指示是否实现与不可微分输出对应的输出梯度张量。
    // 通常情况下，通过关闭 materialize_grads 就可以实现这一行为，
    // 但有些情况下不可行，详见链接说明。
    bool materialize_non_diff_grads;

    // 编译的自动求导跟踪是否启用的布尔值，用作向 AotAutograd 发送信号，
    // 指示应调用原始的 FX 图而不是编译。
    bool compiled_autograd_tracing;

    PyObject* compiled_autograd_backward_state;  // 编译的自动求导后向状态的 Python 对象指针

    std::vector<c10::SymInt> compiled_autograd_symints;  // 编译的自动求导符号整数的向量

    std::vector<torch::autograd::VariableInfo> output_info;  // 输出信息的 VariableInfo 结构向量

    std::vector<torch::autograd::VariableInfo> input_info;  // 输入信息的 VariableInfo 结构向量

    std::vector<torch::autograd::SavedVariable> saved_variables;  // 保存的变量的 SavedVariable 结构向量

    // 对于每个输入，如果输入是 THPVariable，则为 true
    std::vector<bool> is_variable_input;

    char has_freed_buffers;  // 表示是否释放了缓冲区的字符变量

    PyObject* saved_for_forward;  // 保存前向计算的 Python 对象指针

    // 实际的 PyNode（在自动求导图中），这些数据是为其保存的。
    // 该字段可能为 NULL（因为用户可以直接从 Python 构造 THPFunction），
    // 但当该字段非空时，可以保证 cdata.lock()->obj == this。
    std::weak_ptr<torch::autograd::PyNode> cdata;
};

// 初始化 THPFunction 模块的函数声明
bool THPFunction_initModule(PyObject* module);

// THPFunction 类型的 PyTypeObject 对象声明
extern PyTypeObject THPFunctionType;

// THPFunction 类的 Python 类对象声明
extern PyObject* THPFunctionClass;

// THPGradientEdge 类的 Python 类对象声明
extern PyObject* THPGradientEdgeClass;

// 检查一个对象是否为 THPFunction 类型的辅助函数
inline bool THPFunction_Check(PyObject* obj) {
    return PyObject_IsInstance(obj, (PyObject*)&THPFunctionType);
}
```