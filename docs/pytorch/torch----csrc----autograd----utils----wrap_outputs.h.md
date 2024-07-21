# `.\pytorch\torch\csrc\autograd\utils\wrap_outputs.h`

```
#pragma once
// 一次性包装张量操作的输出为 PyObject*

#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>
#include <torch/csrc/python_headers.h>
#include <initializer_list>
#include <tuple>

#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/tensor_qschemes.h>

namespace torch {
namespace autograd {
namespace utils {

// 将布尔值包装为 PyObject*
inline PyObject* wrap(bool value) {
  if (value) {
    // 如果值为真，返回 Python 中的 True
    Py_RETURN_TRUE;
  } else {
    // 如果值为假，返回 Python 中的 False
    Py_RETURN_FALSE;
  }
}

// 将设备索引包装为 PyObject*
inline PyObject* wrap(c10::DeviceIndex value) {
  return THPUtils_packDeviceIndex(value);
}

// 将 int64_t 类型的值包装为 PyObject*
inline PyObject* wrap(int64_t value) {
  return THPUtils_packInt64(value);
}

// 将 double 类型的值包装为 PyObject*
inline PyObject* wrap(double value) {
  return PyFloat_FromDouble(value);
}

// 将复数类型 c10::complex<double> 包装为 PyObject*
inline PyObject* wrap(c10::complex<double> value) {
  // 通过实部和虚部创建 Python 的复数对象
  return PyComplex_FromDoubles(value.real(), value.imag());
}

// 将 void* 类型的指针值包装为 PyObject*
inline PyObject* wrap(void* value) {
  return THPUtils_packInt64(reinterpret_cast<intptr_t>(value));
}

// 将 THPDtype* 类型的对象包装为 PyObject*
inline PyObject* wrap(THPDtype* dtype) {
  return Py_NewRef(dtype);
}

// 将 at::ScalarType 类型的对象包装为 PyObject*
inline PyObject* wrap(at::ScalarType scalarType) {
  return Py_NewRef(getTHPDtype(scalarType));
}

// 将 THPLayout* 类型的对象包装为 PyObject*
inline PyObject* wrap(THPLayout* layout) {
  return Py_NewRef(layout);
}

// 将 at::Layout 类型的对象包装为 PyObject*
inline PyObject* wrap(at::Layout layout) {
  return Py_NewRef(getTHPLayout(layout));
}

// 将 at::Tensor 类型的对象包装为 PyObject*
inline PyObject* wrap(at::Tensor tensor) {
  // 将 at::Tensor 包装为 Variable，然后再转换为 PyObject*
  return THPVariable_Wrap(Variable(std::move(tensor)));
}

// 将 at::Scalar 类型的对象包装为 PyObject*
inline PyObject* wrap(const at::Scalar& scalar) {
  // 将 at::Scalar 转换为对应的 at::Tensor，然后包装为 PyObject*
  return wrap(scalar_to_tensor(scalar));
}

// 将 at::QScheme 类型的对象包装为 PyObject*
inline PyObject* wrap(at::QScheme qscheme) {
  // 获取对应的 THPQScheme* 并增加其引用计数后返回
  auto* thp_qscheme = torch::utils::getTHPQScheme(qscheme);
  Py_INCREF(thp_qscheme);
  return thp_qscheme;
}

// 将 at::TensorList 类型的对象包装为 PyObject*
inline PyObject* wrap(at::TensorList tl) {
  auto r = THPObjectPtr{PyTuple_New(tl.size())};
  if (!r)
    throw python_error();
  for (const auto i : c10::irange(tl.size())) {
    // 逐个将 at::Tensor 包装为 PyObject* 并设置到 tuple 中
    PyTuple_SET_ITEM(r.get(), i, wrap(tl[i]));
  }
  return r.release();
}

// 将 at::IntArrayRef 类型的对象包装为 PyObject*
inline PyObject* wrap(at::IntArrayRef list) {
  auto r = THPObjectPtr{PyTuple_New(list.size())};
  if (!r)
    throw python_error();
  for (const auto i : c10::irange(list.size())) {
    // 逐个将 int64_t 类型的值包装为 PyObject* 并设置到 tuple 中
    PyTuple_SET_ITEM(r.get(), i, wrap(list[i]));
  }
  return r.release();
}

// 将 at::Stream 类型的对象包装为 PyObject*
inline PyObject* wrap(at::Stream stream) {
  return THPStream_Wrap(stream);
}

namespace detail {
// 对 tuple(t1, t2, t3, ...) 调用 f(t1, 0), f(t2, 1), f(t3, 2), ... 的实现
template <typename F, typename Tuple, size_t... Is>
void apply_with_idx_impl(
    const F& f,
    Tuple& t,
    std::index_sequence<Is...> /*indices*/) {
  // 调用每个元素的包装函数，并传入其索引
  (void)std::initializer_list<int>{(f(std::get<Is>(t), Is), 0)...};
}

// 对 tuple(a, b, c, ...) 调用 f(a, 0), f(b, 1), f(c, 2), ... 的接口
template <typename F, typename... Ts>
void apply_with_idx(const F& f, std::tuple<Ts...>& t) {
  apply_with_idx_impl(f, t, std::index_sequence_for<Ts...>{});
}
} // namespace detail

在这里结束了 `detail` 命名空间的定义。


template <typename... Ts>
PyObject* wrap(std::tuple<Ts...> values) {

定义了一个模板函数 `wrap`，接受一个包含类型为 `Ts...` 的参数的元组，并返回一个 `PyObject*` 类型的指针。


  auto r = THPObjectPtr{PyTuple_New(sizeof...(Ts))};

创建了一个 `THPObjectPtr` 类型的智能指针 `r`，用来持有调用 `PyTuple_New` 创建的元组对象。


  if (!r)
    throw python_error();

如果 `r` 指针为空（即指向的元组对象创建失败），则抛出 `python_error` 异常。


  detail::apply_with_idx(
      [&](auto& value, size_t idx) {
        PyTuple_SET_ITEM(r.get(), idx, wrap(std::move(value)));
      },
      values);

调用 `detail::apply_with_idx` 函数，该函数接受一个 Lambda 表达式和 `values` 元组。Lambda 表达式用于处理 `values` 中的每个元素 `value` 和其索引 `idx`，将 `wrap(std::move(value))` 的结果设置为 `r` 元组的第 `idx` 项。


  return r.release();

释放 `r` 指针的所有权，并返回其中包含的元组对象的指针。


template <typename... Ts>
PyObject* wrap(PyTypeObject* type, std::tuple<Ts...> values) {

定义了一个模板函数 `wrap`，接受一个 `PyTypeObject*` 类型的指针 `type` 和一个包含类型为 `Ts...` 的参数的元组，并返回一个 `PyObject*` 类型的指针。


  auto r = THPObjectPtr{PyStructSequence_New(type)};

创建了一个 `THPObjectPtr` 类型的智能指针 `r`，用来持有调用 `PyStructSequence_New` 函数创建的结构序列对象。


  if (!r)
    throw python_error();

如果 `r` 指针为空（即指向的结构序列对象创建失败），则抛出 `python_error` 异常。


  detail::apply_with_idx(
      [&](auto& value, size_t idx) {
        PyStructSequence_SET_ITEM(r.get(), idx, wrap(std::move(value)));
      },
      values);

调用 `detail::apply_with_idx` 函数，该函数接受一个 Lambda 表达式和 `values` 元组。Lambda 表达式用于处理 `values` 中的每个元素 `value` 和其索引 `idx`，将 `wrap(std::move(value))` 的结果设置为 `r` 结构序列对象的第 `idx` 项。


  return r.release();

释放 `r` 指针的所有权，并返回其中包含的结构序列对象的指针。


} // namespace utils
} // namespace autograd
} // namespace torch

在这里结束了 `utils`, `autograd` 和 `torch` 命名空间的定义。
```