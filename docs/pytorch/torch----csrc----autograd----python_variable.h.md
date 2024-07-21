# `.\pytorch\torch\csrc\autograd\python_variable.h`

```py
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pythoncapi_compat.h>

#include <ATen/core/function_schema.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

// Python object that backs torch.autograd.Variable
struct THPVariable {
  PyObject_HEAD;
  // Payload
  c10::MaybeOwned<at::Tensor> cdata;
  // Hooks to be run on backwards pass (corresponds to Python attr
  // '_backwards_hooks', set by 'register_hook')
  PyObject* backward_hooks = nullptr;
  // Hooks to be run in the backwards pass after accumulate grad,
  // i.e., after the .grad has been set (corresponds to Python attr
  // '_post_accumulate_grad_hooks', set by 'register_post_accumulate_grad_hook')
  PyObject* post_accumulate_grad_hooks = nullptr;
};

// Register Python tensor class with a specified device
TORCH_PYTHON_API void registerPythonTensorClass(
    const std::string& device,
    PyObject* python_tensor_class);

// Activate GPU trace for debugging
TORCH_PYTHON_API void activateGPUTrace();

// External declarations for Python classes
TORCH_PYTHON_API extern PyObject* THPVariableClass;
TORCH_PYTHON_API extern PyObject* ParameterClass;

// Initialize THPVariable module with Python object
bool THPVariable_initModule(PyObject* module);

// Wrap an ATen tensor into a Python THPVariable
TORCH_PYTHON_API PyObject* THPVariable_Wrap(at::TensorBase var);

// Check if a Python type object matches exactly THPVariable or ParameterClass
inline bool THPVariable_CheckTypeExact(PyTypeObject* tp) {
  // Check that a python object is a `Tensor`, but not a `Tensor` subclass.
  // (A subclass could have different semantics.) The one exception is
  // Parameter, which is used for Python bookkeeping but is equivalent to
  // Tensor as far as C++ is concerned.
  return (
      tp == (PyTypeObject*)THPVariableClass ||
      tp == (PyTypeObject*)ParameterClass);
}

// Check if a Python object is an exact THPVariable type
inline bool THPVariable_CheckExact(PyObject* obj) {
  return THPVariable_CheckTypeExact(Py_TYPE(obj));
}

// Check if a Python object is an instance of THPVariable class
inline bool THPVariable_Check(PyObject* obj) {
  if (!THPVariableClass)
    return false;

  // Fast path
  if (THPVariable_CheckExact(obj)) {
    return true;
  }

  // Slow path: check instance using PyObject_IsInstance
  const auto result = PyObject_IsInstance(obj, THPVariableClass);
  if (result == -1)
    throw python_error();
  return result;
}

// Unpack THPVariable structure to obtain the underlying ATen Tensor
inline const at::Tensor& THPVariable_Unpack(THPVariable* var) {
  return *var->cdata;
}

// Unpack a Python object assumed to be a THPVariable to obtain the underlying ATen Tensor
inline const at::Tensor& THPVariable_Unpack(PyObject* obj) {
  return THPVariable_Unpack(reinterpret_cast<THPVariable*>(obj));
}

// Convert OperatorHandle arguments to Python arguments and keyword arguments
std::pair<py::object, py::dict> parseIValuesToPyArgsKwargs(
    const c10::OperatorHandle& op,
    const std::vector<c10::IValue>& arguments);

// Push Python object output to Torch's C++ stack
void pushPyOutToStack(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    py::object out,
    const char* msg);

// Wrap a list of Torch autograd variables into a Python list of THPVariable
inline PyObject* THPVariable_WrapList(
    const torch::autograd::variable_list& inputs) {
  PyObject* pyinput = PyList_New(static_cast<Py_ssize_t>(inputs.size()));
  for (const auto i : c10::irange(inputs.size())) {
    PyList_SET_ITEM(pyinput, i, THPVariable_Wrap(inputs[i]));
  }
  return pyinput;
}
// 检查传入的 pyresult 是否确实是一个 Python 列表对象
inline torch::autograd::variable_list THPVariable_UnpackList(
    PyObject* pyresult) {
  // 使用 TORCH_CHECK 确保 pyresult 是一个 PyList 对象
  TORCH_CHECK(PyList_CheckExact(pyresult));
  // 获取列表的长度
  auto result_len = PyList_GET_SIZE(pyresult);
  // 创建一个变量列表 result 来存储解包后的变量
  torch::autograd::variable_list result;
  // 预先分配 result 的容量以提高效率
  result.reserve(result_len);
  // 遍历列表中的每个元素
  for (const auto i : c10::irange(result_len)) {
    // 获取列表中的第 i 个元素
    PyObject* item = PyList_GET_ITEM(pyresult, i);
    // 如果元素不是 None
    if (!Py_IsNone(item)) {
      // 使用 TORCH_INTERNAL_ASSERT_DEBUG_ONLY 确保 item 是一个 THPVariable 对象
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(THPVariable_Check(item));
      // 将 THPVariable 对象解包并添加到 result 中
      result.emplace_back(THPVariable_Unpack(item));
    } else {
      // 如果元素是 None，则添加一个空变量到 result 中
      result.emplace_back();
    }
  }
  // 返回包含解包后变量的列表 result
  return result;
}
```