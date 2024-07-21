# `.\pytorch\torch\csrc\utils\python_strings.h`

```
#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <stdexcept>
#include <string>

// Utilities for handling Python strings. Note that PyString, when defined, is
// the same as PyBytes.

// Returns true if obj is a bytes/str or unicode object
// As of Python 3.6, this does not require the GIL
inline bool THPUtils_checkString(PyObject* obj) {
  return PyBytes_Check(obj) || PyUnicode_Check(obj);
}

// Unpacks PyBytes (PyString) or PyUnicode as std::string
// PyBytes are unpacked as-is. PyUnicode is unpacked as UTF-8.
// NOTE: this method requires the GIL
inline std::string THPUtils_unpackString(PyObject* obj) {
  if (PyBytes_Check(obj)) { // 检查是否为字节对象
    size_t size = PyBytes_GET_SIZE(obj); // 获取字节对象的大小
    return std::string(PyBytes_AS_STRING(obj), size); // 返回以字节对象内容构造的 std::string
  }
  if (PyUnicode_Check(obj)) { // 检查是否为 Unicode 对象
    Py_ssize_t size = 0;
    const char* data = PyUnicode_AsUTF8AndSize(obj, &size); // 获取 UTF-8 编码的字符串和大小
    if (!data) {
      throw std::runtime_error("error unpacking string as utf-8"); // 抛出异常，表示解包 UTF-8 字符串时出错
    }
    return std::string(data, (size_t)size); // 返回以 UTF-8 编码字符串构造的 std::string
  }
  throw std::runtime_error("unpackString: expected bytes or unicode object"); // 抛出异常，表示预期的对象类型错误
}

// Unpacks PyBytes (PyString) or PyUnicode as c10::string_view
// PyBytes are unpacked as-is. PyUnicode is unpacked as UTF-8.
// NOTE: If `obj` is destroyed, then the non-owning c10::string_view will
//   become invalid. If the string needs to be accessed at any point after
//   `obj` is destroyed, then the c10::string_view should be copied into
//   a std::string, or another owning object, and kept alive. For an example,
//   look at how IValue and autograd nodes handle c10::string_view arguments.
// NOTE: this method requires the GIL
inline c10::string_view THPUtils_unpackStringView(PyObject* obj) {
  if (PyBytes_Check(obj)) { // 检查是否为字节对象
    size_t size = PyBytes_GET_SIZE(obj); // 获取字节对象的大小
    return c10::string_view(PyBytes_AS_STRING(obj), size); // 返回以字节对象内容构造的 c10::string_view
  }
  if (PyUnicode_Check(obj)) { // 检查是否为 Unicode 对象
    Py_ssize_t size = 0;
    const char* data = PyUnicode_AsUTF8AndSize(obj, &size); // 获取 UTF-8 编码的字符串和大小
    if (!data) {
      throw std::runtime_error("error unpacking string as utf-8"); // 抛出异常，表示解包 UTF-8 字符串时出错
    }
    return c10::string_view(data, (size_t)size); // 返回以 UTF-8 编码字符串构造的 c10::string_view
  }
  throw std::runtime_error("unpackString: expected bytes or unicode object"); // 抛出异常，表示预期的对象类型错误
}

// Packs a C string into a Python Unicode object
inline PyObject* THPUtils_packString(const char* str) {
  return PyUnicode_FromString(str); // 将 C 字符串打包为 Python Unicode 对象
}

// Packs a std::string into a Python Unicode object
inline PyObject* THPUtils_packString(const std::string& str) {
  return PyUnicode_FromStringAndSize(str.c_str(), str.size()); // 将 std::string 打包为 Python Unicode 对象
}

// Interns a std::string into a Python Unicode object
inline PyObject* THPUtils_internString(const std::string& str) {
  return PyUnicode_InternFromString(str.c_str()); // 将 std::string 做字符串驻留处理，并返回 Python Unicode 对象
}

// Precondition: THPUtils_checkString(obj) must be true
// Checks if a Python Unicode object is interned
inline bool THPUtils_isInterned(PyObject* obj) {
  return PyUnicode_CHECK_INTERNED(obj); // 检查 Python Unicode 对象是否已经做了驻留处理
}

// Precondition: THPUtils_checkString(obj) must be true
// Interns a Python Unicode object in place
inline void THPUtils_internStringInPlace(PyObject** obj) {
  PyUnicode_InternInPlace(obj); // 将 Python Unicode 对象做原地驻留处理
}
/*
 * 引用:
 * https://github.com/numpy/numpy/blob/f4c497c768e0646df740b647782df463825bfd27/numpy/core/src/common/get_attr_string.h#L42
 *
 * PyObject_GetAttrString 的简化版本，
 * 避免对 None、tuple 和 list 对象进行查找，
 * 并且不创建 PyErr，因为此代码会忽略错误。
 *
 * 在不需要异常处理的情况下，这比 PyObject_GetAttrString 更快。
 *
 * 'obj' 是要搜索属性的对象。
 *
 * 'name' 是要搜索的属性名称。
 *
 * 返回一个包装了返回值的 py::object。如果属性查找失败，返回值将为 NULL。
 *
 */

inline py::object PyObject_FastGetAttrString(PyObject* obj, const char* name) {
  // 获取对象的类型
  PyTypeObject* tp = Py_TYPE(obj);
  // 结果对象初始化为 nullptr
  PyObject* res = (PyObject*)nullptr;

  /* 使用 (char *)name 引用的属性 */
  if (tp->tp_getattr != nullptr) {
    // 根据 tp_getattr 函数指针调用对象的属性查找方法
    res = (*tp->tp_getattr)(obj, const_cast<char*>(name));
    // 如果结果为 nullptr，则清除异常状态
    if (res == nullptr) {
      PyErr_Clear();
    }
  }
  /* 使用 (PyObject *)name 引用的属性 */
  else if (tp->tp_getattro != nullptr) {
    // 使用 THPUtils_internString 将 name 转换为 PyObject 对象
    auto w = py::reinterpret_steal<py::object>(THPUtils_internString(name));
    // 如果转换失败，返回空对象
    if (w.ptr() == nullptr) {
      return py::object();
    }
    // 根据 tp_getattro 函数指针调用对象的属性查找方法
    res = (*tp->tp_getattro)(obj, w.ptr());
    // 如果结果为 nullptr，则清除异常状态
    if (res == nullptr) {
      PyErr_Clear();
    }
  }
  // 将 PyObject 指针 res 转换为 py::object，并返回
  return py::reinterpret_steal<py::object>(res);
}
```