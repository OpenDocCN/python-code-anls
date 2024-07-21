# `.\pytorch\torch\csrc\utils\pybind.cpp`

```py
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_symnode.h>

namespace pybind11::detail {

// 用于加载 c10::SymInt 类型的转换器
bool type_caster<c10::SymInt>::load(py::handle src, bool) {
  // 检查是否为符号整数
  if (torch::is_symint(src)) {
    // 获取节点对象
    auto node = src.attr("node");
    // 如果节点是 c10::SymNodeImpl 类型
    if (py::isinstance<c10::SymNodeImpl>(node)) {
      // 将节点转换为 c10::SymInt 类型并赋值给 value
      value = c10::SymInt(py::cast<c10::SymNode>(node));
      return true;
    }

    // 否则，将 Python 对象包装为 torch::impl::PythonSymNodeImpl，并转换为 c10::SymInt 类型
    value = c10::SymInt(static_cast<c10::SymNode>(
        c10::make_intrusive<torch::impl::PythonSymNodeImpl>(node)));
    return true;
  }

  // 如果是 THPVariable 类型的对象
  auto raw_obj = src.ptr();
  if (THPVariable_Check(raw_obj)) {
    // 解包变量并检查其是否为标量整数类型
    auto& var = THPVariable_Unpack(raw_obj);
    if (var.numel() == 1 &&
        at::isIntegralType(var.dtype().toScalarType(), /*include_bool*/ true)) {
      // 获取标量值并转换为 c10::SymInt 类型
      auto scalar = var.item();
      TORCH_INTERNAL_ASSERT(scalar.isIntegral(/*include bool*/ false));
      value = scalar.toSymInt();
      return true;
    }
  }

  // 如果是整数类型，直接转换为 c10::SymInt 类型
  if (THPUtils_checkIndex(raw_obj)) {
    value = c10::SymInt{THPUtils_unpackIndex(raw_obj)};
    return true;
  }
  return false;
}

// 将 c10::SymInt 类型转换为 Python 对象的转换器
py::handle type_caster<c10::SymInt>::cast(
    const c10::SymInt& si,
    return_value_policy /* policy */,
    handle /* parent */) {
  // 如果是符号整数类型
  if (si.is_symbolic()) {
    // 尝试获取 PythonSymNodeImpl 类型的指针
    auto* py_node = dynamic_cast<torch::impl::PythonSymNodeImpl*>(
        si.toSymNodeImplUnowned());
    if (py_node) {
      // 直接返回 Python 对象
      return torch::get_symint_class()(py_node->getPyObj()).release();
    } else {
      // 将 C++ 对象封装为 Python 对象
      auto inner = py::cast(si.toSymNode());
      if (!inner) {
        throw python_error();
      }
      return torch::get_symint_class()(inner).release();
    }
  } else {
    // 否则，将值转换为 Python 对象
    auto m = si.maybe_as_int();
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return py::cast(*m).release();
  }
}

// 用于加载 c10::SymFloat 类型的转换器
bool type_caster<c10::SymFloat>::load(py::handle src, bool) {
  // 检查是否为符号浮点数
  if (torch::is_symfloat(src)) {
    // 将 Python 对象包装为 torch::impl::PythonSymNodeImpl，并转换为 c10::SymFloat 类型
    value = c10::SymFloat(static_cast<c10::SymNode>(
        c10::make_intrusive<torch::impl::PythonSymNodeImpl>(src.attr("node"))));
    return true;
  }

  // 如果是 double 类型的 Python 对象，将其转换为 c10::SymFloat 类型
  auto raw_obj = src.ptr();
  if (THPUtils_checkDouble(raw_obj)) {
    value = c10::SymFloat{THPUtils_unpackDouble(raw_obj)};
    return true;
  }
  return false;
}

// 将 c10::SymFloat 类型转换为 Python 对象的转换器
py::handle type_caster<c10::SymFloat>::cast(
    const c10::SymFloat& si,
    return_value_policy /* policy */,
    handle /* parent */) {
  // 如果是符号浮点数类型
  if (si.is_symbolic()) {
    // 尝试获取 PythonSymNodeImpl 类型的指针
    auto* py_node =
        dynamic_cast<torch::impl::PythonSymNodeImpl*>(si.toSymNodeImpl().get());
    TORCH_INTERNAL_ASSERT(py_node);
    // 直接返回 Python 对象
    return torch::get_symfloat_class()(py_node->getPyObj()).release();
  } else {
    // 否则，将值转换为 Python 对象
    return py::cast(si.as_float_unchecked()).release();
  }
}

// 用于加载 c10::SymBool 类型的转换器
bool type_caster<c10::SymBool>::load(py::handle src, bool) {
  // 检查是否为符号布尔类型
  if (torch::is_symbool(src)) {
    // 将 Python 对象包装为 torch::impl::PythonSymNodeImpl，并转换为 c10::SymBool 类型
    value = c10::SymBool(static_cast<c10::SymNode>(
        c10::make_intrusive<torch::impl::PythonSymNodeImpl>(src.attr("node"))));
    return true;
  }
  return false;
}
    return true;
  }

  auto raw_obj = src.ptr();
  // 将源对象的指针提取到 raw_obj 变量中
  if (THPUtils_checkBool(raw_obj)) {
    // 检查 raw_obj 对象是否表示一个有效的布尔值
    value = c10::SymBool{THPUtils_unpackBool(raw_obj)};
    // 如果是布尔值，则解包并存储到 value 变量中
    return true;
  }
  // 如果不是布尔值，返回 false
  return false;
}

// 实现对 c10::SymBool 类型的转换函数 cast
py::handle type_caster<c10::SymBool>::cast(
    const c10::SymBool& si,
    return_value_policy /* policy */,
    handle /* parent */) {
  // 检查 SymBool 是否可以转换为 bool
  if (auto m = si.maybe_as_bool()) {
    // 如果可以转换为 bool，则返回其对应的 Python 对象
    return py::cast(*m).release();
  } else {
    // TODO: generalize this to work with C++ backed class
    // 尝试获取 SymBool 对应的 Python 实现对象
    auto* py_node =
        dynamic_cast<torch::impl::PythonSymNodeImpl*>(si.toSymNodeImpl().get());
    TORCH_INTERNAL_ASSERT(py_node);
    // 使用 Python 实现对象创建 SymBool 类对象，并返回其 Python 对象
    return torch::get_symbool_class()(py_node->getPyObj()).release();
  }
}

// 加载函数，用于 c10::Scalar 类型，但未实现
bool type_caster<c10::Scalar>::load(py::handle src, bool) {
  // 抛出错误，暂未实现加载函数
  TORCH_INTERNAL_ASSERT(
      0, "pybind11 loading for c10::Scalar NYI (file a bug if you need it)");
}

// 实现对 c10::Scalar 类型的转换函数 cast
py::handle type_caster<c10::Scalar>::cast(
    const c10::Scalar& scalar,
    return_value_policy /* policy */,
    handle /* parent */) {
  // 如果是整数类型
  if (scalar.isIntegral(/*includeBool*/ false)) {
    // 如果是符号化的整数，返回其 SymInt 表示
    if (scalar.isSymbolic()) {
      return py::cast(scalar.toSymInt()).release();
    } else {
      // 根据具体的整数类型进行转换
      if (scalar.type() == at::ScalarType::UInt64) {
        return py::cast(scalar.toUInt64()).release();
      } else {
        return py::cast(scalar.toLong()).release();
      }
    }
  } else if (scalar.isFloatingPoint()) {  // 如果是浮点数类型
    // 如果是符号化的浮点数，返回其 SymFloat 表示
    if (scalar.isSymbolic()) {
      return py::cast(scalar.toSymFloat()).release();
    } else {
      return py::cast(scalar.toDouble()).release();
    }
  } else if (scalar.isBoolean()) {  // 如果是布尔类型
    // 如果是符号化的布尔值，返回其 SymBool 表示
    if (scalar.isSymbolic()) {
      return py::cast(scalar.toSymBool()).release();
    }
    // 否则，返回普通的布尔值
    return py::cast(scalar.toBool()).release();
  } else if (scalar.isComplex()) {  // 如果是复数类型
    // 返回其复数双精度浮点数表示
    return py::cast(scalar.toComplexDouble()).release();
  } else {
    // 如果识别不出具体的标量类型，则抛出错误
    TORCH_INTERNAL_ASSERT(0, "unrecognized scalar type ", scalar.type());
  }
}

// 命名空间结束注释
} // namespace pybind11::detail
```