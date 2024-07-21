# `.\pytorch\torch\csrc\utils\python_scalars.h`

```
#pragma once
// 预处理指令：确保头文件只包含一次

#include <ATen/ATen.h>
// 包含 ATen 库的头文件

#include <c10/util/TypeCast.h>
// 包含 c10 库中的类型转换工具头文件

#include <torch/csrc/python_headers.h>
// 包含 Torch 中用于 Python 的头文件

#include <torch/csrc/Exceptions.h>
// 包含 Torch 中异常处理的头文件

#include <torch/csrc/utils/python_numbers.h>
// 包含 Torch 中用于 Python 数字处理的实用工具头文件

namespace torch::utils {
// 定义 torch::utils 命名空间

template <typename T>
inline T unpackIntegral(PyObject* obj, const char* type) {
// 模板函数：从 Python 对象中解包整数类型 T
#if PY_VERSION_HEX >= 0x030a00f0
  // 如果 Python 版本大于等于 3.10
  // Python-3.10 不再允许浮点数隐式转换为整数
  // 保持向后兼容行为
  if (PyFloat_Check(obj)) {
    return c10::checked_convert<T>(THPUtils_unpackDouble(obj), type);
    // 如果对象是浮点数，则解包为双精度浮点数后进行转换为类型 T
  }
  return c10::checked_convert<T>(THPUtils_unpackLong(obj), type);
  // 否则，解包为长整型后进行转换为类型 T
#else
  return static_cast<T>(THPUtils_unpackLong(obj));
  // 对于 Python 版本低于 3.10，直接解包为长整型并转换为类型 T
#endif
}

inline void store_scalar(void* data, at::ScalarType scalarType, PyObject* obj) {
// 存储标量函数：根据标量类型将 Python 对象的值存储到数据指针中
  switch (scalarType) {
    case at::kByte:
      *(uint8_t*)data = unpackIntegral<uint8_t>(obj, "uint8");
      // 存储类型为 uint8 的标量值到数据指针中
      break;
    case at::kUInt16:
      *(uint16_t*)data = unpackIntegral<uint16_t>(obj, "uint16");
      // 存储类型为 uint16 的标量值到数据指针中
      break;
    case at::kUInt32:
      *(uint32_t*)data = unpackIntegral<uint32_t>(obj, "uint32");
      // 存储类型为 uint32 的标量值到数据指针中
      break;
    case at::kUInt64:
      // 注意：这不允许浮点数隐式转换为整数
      *(uint64_t*)data = THPUtils_unpackUInt64(obj);
      // 存储类型为 uint64 的标量值到数据指针中
      break;
    case at::kChar:
      *(int8_t*)data = unpackIntegral<int8_t>(obj, "int8");
      // 存储类型为 int8 的标量值到数据指针中
      break;
    case at::kShort:
      *(int16_t*)data = unpackIntegral<int16_t>(obj, "int16");
      // 存储类型为 int16 的标量值到数据指针中
      break;
    case at::kInt:
      *(int32_t*)data = unpackIntegral<int32_t>(obj, "int32");
      // 存储类型为 int32 的标量值到数据指针中
      break;
    case at::kLong:
      *(int64_t*)data = unpackIntegral<int64_t>(obj, "int64");
      // 存储类型为 int64 的标量值到数据指针中
      break;
    case at::kHalf:
      *(at::Half*)data =
          at::convert<at::Half, double>(THPUtils_unpackDouble(obj));
      // 存储类型为 Half 的标量值到数据指针中
      break;
    case at::kFloat:
      *(float*)data = (float)THPUtils_unpackDouble(obj);
      // 存储类型为 float 的标量值到数据指针中
      break;
    case at::kDouble:
      *(double*)data = THPUtils_unpackDouble(obj);
      // 存储类型为 double 的标量值到数据指针中
      break;
    case at::kComplexHalf:
      *(c10::complex<at::Half>*)data =
          (c10::complex<at::Half>)static_cast<c10::complex<float>>(
              THPUtils_unpackComplexDouble(obj));
      // 存储类型为 complex<Half> 的标量值到数据指针中
      break;
    case at::kComplexFloat:
      *(c10::complex<float>*)data =
          (c10::complex<float>)THPUtils_unpackComplexDouble(obj);
      // 存储类型为 complex<float> 的标量值到数据指针中
      break;
    case at::kComplexDouble:
      *(c10::complex<double>*)data = THPUtils_unpackComplexDouble(obj);
      // 存储类型为 complex<double> 的标量值到数据指针中
      break;
    case at::kBool:
      *(bool*)data = THPUtils_unpackNumberAsBool(obj);
      // 存储类型为 bool 的标量值到数据指针中
      break;
    case at::kBFloat16:
      *(at::BFloat16*)data =
          at::convert<at::BFloat16, double>(THPUtils_unpackDouble(obj));
      // 存储类型为 BFloat16 的标量值到数据指针中
      break;
    case at::kFloat8_e5m2:
      *(at::Float8_e5m2*)data =
          at::convert<at::Float8_e5m2, double>(THPUtils_unpackDouble(obj));
      // 存储类型为 Float8_e5m2 的标量值到数据指针中
      break;
    case at::kFloat8_e5m2fnuz:
      *(at::Float8_e5m2fnuz*)data =
          at::convert<at::Float8_e5m2fnuz, double>(THPUtils_unpackDouble(obj));
      // 存储类型为 Float8_e5m2fnuz 的标量值到数据指针中
      break;
    // 如果类型为 kFloat8_e4m3fn
    case at::kFloat8_e4m3fn:
      // 将 obj 解包成 double 类型，并转换为 Float8_e4m3fn 类型，存储在 data 中
      *(at::Float8_e4m3fn*)data = at::convert<at::Float8_e4m3fn, double>(THPUtils_unpackDouble(obj));
      break;
    // 如果类型为 kFloat8_e4m3fnuz
    case at::kFloat8_e4m3fnuz:
      // 将 obj 解包成 double 类型，并转换为 Float8_e4m3fnuz 类型，存储在 data 中
      *(at::Float8_e4m3fnuz*)data = at::convert<at::Float8_e4m3fnuz, double>(THPUtils_unpackDouble(obj));
      break;
    // 默认情况下抛出运行时错误
    default:
      throw std::runtime_error("invalid type");
  }
}

// 结束命名空间 torch::utils 的声明

inline PyObject* load_scalar(const void* data, at::ScalarType scalarType) {
  // 根据标量类型加载相应的 Python 对象
  switch (scalarType) {
    case at::kByte:
      return THPUtils_packInt64(*(uint8_t*)data);  // 将 uint8_t 类型数据打包成 Python 整数对象
    case at::kUInt16:
      return THPUtils_packInt64(*(uint16_t*)data);  // 将 uint16_t 类型数据打包成 Python 整数对象
    case at::kUInt32:
      return THPUtils_packUInt32(*(uint32_t*)data);  // 将 uint32_t 类型数据打包成 Python 整数对象
    case at::kUInt64:
      return THPUtils_packUInt64(*(uint64_t*)data);  // 将 uint64_t 类型数据打包成 Python 整数对象
    case at::kChar:
      return THPUtils_packInt64(*(int8_t*)data);  // 将 int8_t 类型数据打包成 Python 整数对象
    case at::kShort:
      return THPUtils_packInt64(*(int16_t*)data);  // 将 int16_t 类型数据打包成 Python 整数对象
    case at::kInt:
      return THPUtils_packInt64(*(int32_t*)data);  // 将 int32_t 类型数据打包成 Python 整数对象
    case at::kLong:
      return THPUtils_packInt64(*(int64_t*)data);  // 将 int64_t 类型数据打包成 Python 整数对象
    case at::kHalf:
      return PyFloat_FromDouble(
          at::convert<double, at::Half>(*(at::Half*)data));  // 将 at::Half 类型数据转换为 Python 浮点数对象
    case at::kFloat:
      return PyFloat_FromDouble(*(float*)data);  // 将 float 类型数据转换为 Python 浮点数对象
    case at::kDouble:
      return PyFloat_FromDouble(*(double*)data);  // 将 double 类型数据转换为 Python 浮点数对象
    case at::kComplexHalf: {
      auto data_ = reinterpret_cast<const c10::complex<at::Half>*>(data);
      return PyComplex_FromDoubles(data_->real(), data_->imag());  // 将 c10::complex<at::Half> 类型数据转换为 Python 复数对象
    }
    case at::kComplexFloat: {
      auto data_ = reinterpret_cast<const c10::complex<float>*>(data);
      return PyComplex_FromDoubles(data_->real(), data_->imag());  // 将 c10::complex<float> 类型数据转换为 Python 复数对象
    }
    case at::kComplexDouble:
      return PyComplex_FromCComplex(
          *reinterpret_cast<Py_complex*>((c10::complex<double>*)data));  // 将 c10::complex<double> 类型数据转换为 Python 复数对象
    case at::kBool:
      return PyBool_FromLong(*(bool*)data);  // 将 bool 类型数据转换为 Python 布尔对象
    case at::kBFloat16:
      return PyFloat_FromDouble(
          at::convert<double, at::BFloat16>(*(at::BFloat16*)data));  // 将 at::BFloat16 类型数据转换为 Python 浮点数对象
    case at::kFloat8_e5m2:
      return PyFloat_FromDouble(
          at::convert<double, at::Float8_e5m2>(*(at::Float8_e5m2*)data));  // 将 at::Float8_e5m2 类型数据转换为 Python 浮点数对象
    case at::kFloat8_e4m3fn:
      return PyFloat_FromDouble(
          at::convert<double, at::Float8_e4m3fn>(*(at::Float8_e4m3fn*)data));  // 将 at::Float8_e4m3fn 类型数据转换为 Python 浮点数对象
    case at::kFloat8_e5m2fnuz:
      return PyFloat_FromDouble(at::convert<double, at::Float8_e5m2fnuz>(
          *(at::Float8_e5m2fnuz*)data));  // 将 at::Float8_e5m2fnuz 类型数据转换为 Python 浮点数对象
    case at::kFloat8_e4m3fnuz:
      return PyFloat_FromDouble(at::convert<double, at::Float8_e4m3fnuz>(
          *(at::Float8_e4m3fnuz*)data));  // 将 at::Float8_e4m3fnuz 类型数据转换为 Python 浮点数对象
    default:
      throw std::runtime_error("invalid type");  // 抛出运行时错误，表示无效的数据类型
  }
}

} // namespace torch::utils
```