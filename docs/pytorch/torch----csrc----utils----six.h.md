# `.\pytorch\torch\csrc\utils\six.h`

```
#pragma`
#pragma once

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/structseq.h>

namespace six {

// 检查输入的 handle 是否为 structseq 类型
inline bool isStructSeq(pybind11::handle input) {
  // 获取类型的模块名称，如果模块名称为 "torch.return_types"，则判断为 structseq 类型
  return pybind11::cast<std::string>(input.get_type().attr("__module__")) ==
      "torch.return_types";
}

// 使用 PyObject* 检查是否为 structseq 类型的对象
inline bool isStructSeq(PyObject* obj) {
  return isStructSeq(pybind11::handle(obj));
}

// 检查输入的 handle 是否为 tuple 类型
inline bool isTuple(pybind11::handle input) {
  // 如果是 tuple 类型，则返回 true
  if (PyTuple_Check(input.ptr())) {
    return true;
  }
  // 否则返回 false
  return false;
}

// 使用 PyObject* 检查是否为 tuple 类型的对象
inline bool isTuple(PyObject* obj) {
  return isTuple(pybind11::handle(obj));
}

// 如果输入是 structseq 类型的对象，则尝试转换为 tuple
//
// 在 Python 3 中，structseq 是 tuple 的子类，因此可以直接使用这些 API。
// 但在 Python 2 中，structseq 不是 tuple 的子类，所以我们需要手动从 structseq 创建一个新的 tuple 对象。
inline THPObjectPtr maybeAsTuple(PyStructSequence* obj) {
  // 增加对象的引用计数，防止释放
  Py_INCREF(obj);
  // 返回一个 THPObjectPtr 对象，包含传入的 PyStructSequence 对象
  return THPObjectPtr((PyObject*)obj);
}

// 尝试将 PyObject* 对象转换为 tuple
inline THPObjectPtr maybeAsTuple(PyObject* obj) {
  // 如果输入对象是 structseq 类型，则调用 maybeAsTuple(PyStructSequence* obj) 函数进行转换
  if (isStructSeq(obj))
    return maybeAsTuple((PyStructSequence*)obj);
  // 增加对象的引用计数，防止释放
  Py_INCREF(obj);
  // 返回一个 THPObjectPtr 对象，包含传入的 PyObject* 对象
  return THPObjectPtr(obj);
}

} // namespace six
```