# `.\pytorch\torch\csrc\autograd\python_variable_indexing.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <c10/core/SymInt.h>
// 包含 C10 库的 SymInt 头文件

#include <torch/csrc/autograd/python_variable.h>
// 包含 Torch 自动求导模块的 Python 变量头文件

#include <torch/csrc/python_headers.h>
// 包含 Torch 的 Python 头文件

#include <torch/csrc/utils/pybind.h>
// 包含 Torch 的 pybind11 工具头文件

#include <torch/csrc/utils/python_symnode.h>
// 包含 Torch 的 Python 符号节点头文件

namespace torch::autograd {
// 进入 torch::autograd 命名空间

struct UnpackedSlice {
  c10::SymInt start;
  // 起始索引，使用 SymInt 类型表示
  c10::SymInt stop;
  // 终止索引，使用 SymInt 类型表示
  c10::SymInt step;
  // 步长，使用 SymInt 类型表示
};

// This mirrors Cpython's PySlice_Unpack method
// 模仿 CPython 的 PySlice_Unpack 方法
inline UnpackedSlice __PySlice_Unpack(PyObject* _r) {
  PySliceObject* r = (PySliceObject*)_r;
  // 将 PyObject 转换为 PySliceObject 类型

  /* this is harder to get right than you might think */
  // 这比你想象的更难以正确实现

  c10::SymInt start_sym, stop_sym, step_sym;
  // 定义 SymInt 类型的起始、终止、步长变量

  auto clip_val = [](Py_ssize_t val) {
    // 定义一个 lambda 函数 clip_val，用于裁剪值
    if (val < c10::SymInt::min_representable_int()) {
      auto r = PyErr_WarnEx(
          PyExc_UserWarning,
          "Truncating the start/stop/step "
          "of slice. This is likely because of "
          "saved old models when the start/stop/step were larger.",
          1);
      if (r != 0) {
        throw python_error();
      }
      return (Py_ssize_t)(c10::SymInt::min_representable_int());
    }
    return val;
  };

  if (r->step == Py_None) {
    step_sym = c10::SymInt(1);
    // 如果步长为 None，则设置为 SymInt 类型的 1
  } else {
    if (torch::is_symint(r->step)) {
      step_sym = py::handle(r->step).cast<c10::SymInt>();
      // 如果步长是符号整数类型，则转换为 SymInt 类型
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      Py_ssize_t step;
      if (!_PyEval_SliceIndex(r->step, &step)) {
        throw python_error();
      }
      if (step == 0) {
        PyErr_SetString(PyExc_ValueError, "slice step cannot be zero");
        // 如果步长为 0，则抛出 ValueError 异常
      }

      step = clip_val(step);
      // 裁剪步长值
      step_sym = c10::SymInt(step);
      // 将裁剪后的步长值转换为 SymInt 类型
    }
  }

  if (torch::is_symint(r->start)) {
    start_sym = py::handle(r->start).cast<c10::SymInt>();
    // 如果起始索引是符号整数类型，则转换为 SymInt 类型
  } else if (r->start == Py_None) {
    start_sym = c10::SymInt(step_sym < 0 ? PY_SSIZE_T_MAX : 0);
    // 如果起始索引为 None，则根据步长设置 SymInt 类型的起始索引
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Py_ssize_t start;
    if (!_PyEval_SliceIndex(r->start, &start)) {
      throw python_error();
    }
    start = clip_val(start);
    // 裁剪起始索引值
    start_sym = c10::SymInt(start);
    // 将裁剪后的起始索引值转换为 SymInt 类型
  }

  if (torch::is_symint(r->stop)) {
    stop_sym = py::handle(r->stop).cast<c10::SymInt>();
    // 如果终止索引是符号整数类型，则转换为 SymInt 类型
  } else if (r->stop == Py_None) {
    stop_sym = c10::SymInt(
        step_sym < 0 ? c10::SymInt::min_representable_int() : PY_SSIZE_T_MAX);
    // 如果终止索引为 None，则根据步长设置 SymInt 类型的终止索引
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Py_ssize_t stop;
    if (!_PyEval_SliceIndex(r->stop, &stop)) {
      throw python_error();
    }
    stop = clip_val(stop);
    // 裁剪终止索引值
    stop_sym = c10::SymInt(stop);
    // 将裁剪后的终止索引值转换为 SymInt 类型
  }

  return UnpackedSlice{
      std::move(start_sym), std::move(stop_sym), std::move(step_sym)};
  // 返回解析后的 UnpackedSlice 结构体对象
}

Py_ssize_t THPVariable_length(PyObject* self);
// 声明函数 THPVariable_length，用于获取对象的长度

PyObject* THPVariable_getitem(PyObject* self, PyObject* index);
// 声明函数 THPVariable_getitem，用于获取对象的指定索引的元素

int THPVariable_setitem(PyObject* self, PyObject* index, PyObject* value);
// 声明函数 THPVariable_setitem，用于设置对象的指定索引的元素

Variable valueToTensor(
    c10::TensorOptions options,
    PyObject* value,
    const at::Device& device);
// 声明函数 valueToTensor，将 Python 对象转换为 Tensor 类型的变量

} // namespace torch::autograd
// 结束 torch::autograd 命名空间
```