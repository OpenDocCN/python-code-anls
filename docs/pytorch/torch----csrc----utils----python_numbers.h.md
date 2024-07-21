# `.\pytorch\torch\csrc\utils\python_numbers.h`

```
#pragma once

#include <c10/core/Device.h>  // 引入 c10::Device 相关头文件
#include <torch/csrc/Exceptions.h>  // 引入 Torch 异常处理头文件
#include <torch/csrc/jit/frontend/tracer.h>  // 引入 Torch 的追踪器前端头文件
#include <torch/csrc/python_headers.h>  // 引入 Torch Python 头文件
#include <torch/csrc/utils/object_ptr.h>  // 引入 Torch 对象指针工具头文件
#include <torch/csrc/utils/tensor_numpy.h>  // 引入 Torch 张量与 NumPy 转换头文件
#include <cstdint>  // 引入 C++ 标准整数类型头文件
#include <limits>  // 引入数值上限头文件
#include <stdexcept>  // 引入标准异常处理头文件

// 表示 double 类型可以连续表示的最大整数
const int64_t DOUBLE_INT_MAX = 9007199254740992;

// 将 c10::DeviceIndex 类型的值打包成 Python 的 PyObject* 类型
inline PyObject* THPUtils_packDeviceIndex(c10::DeviceIndex value) {
  return PyLong_FromLong(value);
}

// 将 int32_t 类型的值打包成 Python 的 PyObject* 类型
inline PyObject* THPUtils_packInt32(int32_t value) {
  return PyLong_FromLong(value);
}

// 将 int64_t 类型的值打包成 Python 的 PyObject* 类型
inline PyObject* THPUtils_packInt64(int64_t value) {
  return PyLong_FromLongLong(value);
}

// 将 uint32_t 类型的值打包成 Python 的 PyObject* 类型
inline PyObject* THPUtils_packUInt32(uint32_t value) {
  return PyLong_FromUnsignedLong(value);
}

// 将 uint64_t 类型的值打包成 Python 的 PyObject* 类型
inline PyObject* THPUtils_packUInt64(uint64_t value) {
  return PyLong_FromUnsignedLongLong(value);
}

// 将 double 类型的值打包成 Python 的 PyObject* 类型
inline PyObject* THPUtils_packDoubleAsInt(double value) {
  return PyLong_FromDouble(value);
}

// 检查 PyObject* 是否精确是长整型，而非布尔类型
inline bool THPUtils_checkLongExact(PyObject* obj) {
  return PyLong_CheckExact(obj) && !PyBool_Check(obj);
}

// 检查 PyObject* 是否是长整型，支持快速路径和 NumPy 检查
inline bool THPUtils_checkLong(PyObject* obj) {
  // 快速路径
  if (THPUtils_checkLongExact(obj)) {
    return true;
  }

#ifdef USE_NUMPY
  // 使用 NumPy 检查是否是 NumPy 整数类型
  if (torch::utils::is_numpy_int(obj)) {
    return true;
  }
#endif

  // 检查是否是长整型，且不是布尔类型
  return PyLong_Check(obj) && !PyBool_Check(obj);
}

// 解包 PyObject* 中的整数值为 int32_t 类型
inline int32_t THPUtils_unpackInt(PyObject* obj) {
  int overflow = 0;
  long value = PyLong_AsLongAndOverflow(obj, &overflow);
  if (value == -1 && PyErr_Occurred()) {
    throw python_error();
  }
  if (overflow != 0) {
    throw std::runtime_error("Overflow when unpacking long");
  }
  if (value > std::numeric_limits<int32_t>::max() ||
      value < std::numeric_limits<int32_t>::min()) {
    throw std::runtime_error("Overflow when unpacking long");
  }
  return (int32_t)value;
}

// 解包 PyObject* 中的整数值为 int64_t 类型
inline int64_t THPUtils_unpackLong(PyObject* obj) {
  int overflow = 0;
  long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
  if (value == -1 && PyErr_Occurred()) {
    throw python_error();
  }
  if (overflow != 0) {
    throw std::runtime_error("Overflow when unpacking long");
  }
  return (int64_t)value;
}

// 解包 PyObject* 中的整数值为 uint32_t 类型
inline uint32_t THPUtils_unpackUInt32(PyObject* obj) {
  unsigned long value = PyLong_AsUnsignedLong(obj);
  if (PyErr_Occurred()) {
    throw python_error();
  }
  if (value > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Overflow when unpacking unsigned long");
  }
  return (uint32_t)value;
}

// 解包 PyObject* 中的整数值为 uint64_t 类型
inline uint64_t THPUtils_unpackUInt64(PyObject* obj) {
  unsigned long long value = PyLong_AsUnsignedLongLong(obj);
  if (PyErr_Occurred()) {
    throw python_error();
  }
  return (uint64_t)value;
}

// 检查 PyObject* 是否是有效的索引类型
bool THPUtils_checkIndex(PyObject* obj);

// 解包 PyObject* 中的整数值为 int64_t 类型的索引
inline int64_t THPUtils_unpackIndex(PyObject* obj) {
  if (!THPUtils_checkLong(obj)) {
    auto index = THPObjectPtr(PyNumber_Index(obj));
    if (index == nullptr) {
      throw python_error();
    }
    // 如果 `index` 是一个有效对象，则调用 `THPUtils_unpackLong()` 函数解包并返回其值
    // 这是因为在 `index` 超出作用域并且其底层对象的引用计数减少之前，需要先调用此函数
    return THPUtils_unpackLong(index.get());
  }
  // 如果 `index` 不是有效对象，则直接调用 `THPUtils_unpackLong()` 函数解包并返回 `obj` 的值
  return THPUtils_unpackLong(obj);
inline bool THPUtils_unpackBool(PyObject* obj) {
    // 检查是否为 Python 中的 True 对象，返回 true
    if (obj == Py_True) {
        return true;
    }
    // 检查是否为 Python 中的 False 对象，返回 false
    else if (obj == Py_False) {
        return false;
    }
    // 如果既不是 True 也不是 False，则抛出异常
    else {
        throw std::runtime_error("couldn't convert python object to boolean");
    }
}

inline bool THPUtils_checkBool(PyObject* obj) {
    // 如果定义了 USE_NUMPY 宏，并且对象是 NumPy 布尔类型，则返回 true
#ifdef USE_NUMPY
    if (torch::utils::is_numpy_bool(obj)) {
        return true;
    }
#endif
    // 否则，检查是否为 Python 内置的布尔类型对象
    return PyBool_Check(obj);
}

inline bool THPUtils_checkDouble(PyObject* obj) {
    // 如果定义了 USE_NUMPY 宏，并且对象是 NumPy 标量，则返回 true
#ifdef USE_NUMPY
    if (torch::utils::is_numpy_scalar(obj)) {
        return true;
    }
#endif
    // 否则，检查对象是否为 Python 浮点数或长整型
    return PyFloat_Check(obj) || PyLong_Check(obj);
}

inline double THPUtils_unpackDouble(PyObject* obj) {
    // 如果对象是 Python 浮点数，直接获取其值并返回
    if (PyFloat_Check(obj)) {
        return PyFloat_AS_DOUBLE(obj);
    }
    // 否则，尝试将对象转换为双精度浮点数
    double value = PyFloat_AsDouble(obj);
    // 如果转换失败并且出现了异常，则抛出 python_error 异常
    if (value == -1 && PyErr_Occurred()) {
        throw python_error();
    }
    return value;
}

inline c10::complex<double> THPUtils_unpackComplexDouble(PyObject* obj) {
    // 将 Python 复数对象转换为 C++ 复数对象
    Py_complex value = PyComplex_AsCComplex(obj);
    // 如果转换失败并且出现了异常，则抛出 python_error 异常
    if (value.real == -1.0 && PyErr_Occurred()) {
        throw python_error();
    }
    // 返回 C++ 复数对象
    return c10::complex<double>(value.real, value.imag);
}

inline bool THPUtils_unpackNumberAsBool(PyObject* obj) {
    // 如果对象是 Python 浮点数，则将其转换为双精度浮点数并返回其布尔值
    if (PyFloat_Check(obj)) {
        return (bool)PyFloat_AS_DOUBLE(obj);
    }

    // 如果对象是 Python 复数，则获取其实部和虚部的值，并判断是否为零
    if (PyComplex_Check(obj)) {
        double real_val = PyComplex_RealAsDouble(obj);
        double imag_val = PyComplex_ImagAsDouble(obj);
        return !(real_val == 0 && imag_val == 0);
    }

    // 否则，尝试将对象转换为长长整型，并返回其布尔值
    int overflow;
    long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
    // 如果转换失败并且出现了异常，则抛出 python_error 异常
    if (value == -1 && PyErr_Occurred()) {
        throw python_error();
    }
    // 不需要检查溢出，因为当溢出发生时，应该返回 true，以保持与 numpy 相同的行为
    return (bool)value;
}

inline c10::DeviceIndex THPUtils_unpackDeviceIndex(PyObject* obj) {
    // 尝试将 Python 对象转换为长整型
    int overflow = 0;
    long value = PyLong_AsLongAndOverflow(obj, &overflow);
    // 如果转换失败并且出现了异常，则抛出 python_error 异常
    if (value == -1 && PyErr_Occurred()) {
        throw python_error();
    }
    // 如果转换过程中发生溢出，则抛出运行时异常
    if (overflow != 0) {
        throw std::runtime_error("Overflow when unpacking DeviceIndex");
    }
    // 如果转换后的值超出了 c10::DeviceIndex 的范围，则抛出运行时异常
    if (value > std::numeric_limits<c10::DeviceIndex>::max() ||
        value < std::numeric_limits<c10::DeviceIndex>::min()) {
        throw std::runtime_error("Overflow when unpacking DeviceIndex");
    }
    // 返回 DeviceIndex 类型的值
    return (c10::DeviceIndex)value;
}
```