# `.\pytorch\torch\csrc\utils\pybind.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <torch/csrc/python_headers.h>
// 包含 PyTorch C++ 前端需要的 Python 头文件

#include <torch/csrc/utils/pythoncapi_compat.h>
// 包含 PyTorch C++ 前端的 Python 兼容性工具函数

#include <ATen/core/Tensor.h>
// 包含 ATen 核心的 Tensor 类定义

#include <ATen/core/jit_type_base.h>
// 包含 ATen 核心的 jit_type_base 定义

#include <c10/util/irange.h>
// 包含 c10 库的 irange 工具函数

#include <pybind11/pybind11.h>
// 包含 pybind11 库的 pybind11 头文件

#include <pybind11/stl.h>
// 包含 pybind11 库的与 STL 容器交互的功能

#include <torch/csrc/Device.h>
// 包含 PyTorch C++ 前端的 Device 类定义

#include <torch/csrc/Dtype.h>
// 包含 PyTorch C++ 前端的 Dtype 类定义

#include <torch/csrc/DynamicTypes.h>
// 包含 PyTorch C++ 前端的动态类型定义

#include <torch/csrc/Generator.h>
// 包含 PyTorch C++ 前端的 Generator 类定义

#include <torch/csrc/MemoryFormat.h>
// 包含 PyTorch C++ 前端的 MemoryFormat 类定义

#include <torch/csrc/Stream.h>
// 包含 PyTorch C++ 前端的 Stream 类定义

#include <torch/csrc/utils/tensor_memoryformats.h>
// 包含 PyTorch C++ 前端的与张量内存格式相关的工具函数

namespace py = pybind11;
// 定义 pybind11 命名空间别名为 py

// 声明 c10::intrusive_ptr<T> 为 pybind11 的自定义智能指针持有类型
PYBIND11_DECLARE_HOLDER_TYPE(T, c10::intrusive_ptr<T>, true);

// 声明 c10::SingletonOrSharedTypePtr<T> 为 pybind11 的自定义智能指针持有类型
PYBIND11_DECLARE_HOLDER_TYPE(T, c10::SingletonOrSharedTypePtr<T>);

// 声明 c10::SingletonTypePtr<T> 为 pybind11 的自定义智能指针持有类型
PYBIND11_DECLARE_HOLDER_TYPE(T, c10::SingletonTypePtr<T>, true);

namespace pybind11::detail {

// 定义 torch.Tensor <-> at::Tensor 的转换器，无需解包
template <>
struct TORCH_PYTHON_API type_caster<at::Tensor> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::Tensor, _("torch.Tensor"));

  // 加载函数，从 Python 对象加载 at::Tensor
  bool load(handle src, bool);

  // 将 at::Tensor 转换为 Python 对象的静态函数
  static handle cast(
      const at::Tensor& src,
      return_value_policy /* policy */,
      handle /* parent */);
};

// 定义 torch._StorageBase <-> at::Storage 的转换器
template <>
struct type_caster<at::Storage> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::Storage, _("torch.StorageBase"));

  // 加载函数，从 Python 对象加载 at::Storage
  bool load(handle src, bool);

  // 将 at::Storage 转换为 Python 对象的静态函数
  static handle cast(
      const at::Storage& src,
      return_value_policy /* policy */,
      handle /* parent */);
};

// 定义 torch.Generator <-> at::Generator 的转换器
template <>
struct type_caster<at::Generator> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::Generator, _("torch.Generator"));

  // 加载函数，从 Python 对象加载 at::Generator
  bool load(handle src, bool);

  // 将 at::Generator 转换为 Python 对象的静态函数
  static handle cast(
      const at::Generator& src,
      return_value_policy /* policy */,
      handle /* parent */);
};

// 定义 torch.IntArrayRef <-> at::IntArrayRef 的转换器
template <>
struct TORCH_PYTHON_API type_caster<at::IntArrayRef> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::IntArrayRef, _("Tuple[int, ...]"));

  // 加载函数，从 Python 对象加载 at::IntArrayRef
  bool load(handle src, bool);

  // 将 at::IntArrayRef 转换为 Python 对象的静态函数
  static handle cast(
      at::IntArrayRef src,
      return_value_policy /* policy */,
      handle /* parent */);

 private:
  std::vector<int64_t> v_value;  // 存储转换的 int 数组
};

template <>
// 继续 type_caster 模板的特化
struct
// 定义一个模板特化，用于将 `at::SymIntArrayRef` 类型转换为 Python 中的列表类型 `List[int]`
struct TORCH_PYTHON_API type_caster<at::SymIntArrayRef> {
 public:
  // 使用 PYBIND11_TYPE_CASTER 宏定义转换器，将 `at::SymIntArrayRef` 映射到 _("List[int]")，这是其在 Python 中的类型名称
  PYBIND11_TYPE_CASTER(at::SymIntArrayRef, _("List[int]"));

  // 加载函数，从 Python 对象 `src` 中加载数据到 `at::SymIntArrayRef`
  bool load(handle src, bool);

  // 静态 cast 函数，将 `at::SymIntArrayRef` 转换为 Python 对象
  static handle cast(
      at::SymIntArrayRef src,
      return_value_policy /* policy */,
      handle /* parent */);

 private:
  // 内部存储的数据类型为 `std::vector<c10::SymInt>`
  std::vector<c10::SymInt> v_value;
};

// 定义另一个模板特化，用于将 `at::ArrayRef<c10::SymNode>` 转换为 Python 中的列表类型 `List[SymNode]`
template <>
struct TORCH_PYTHON_API type_caster<at::ArrayRef<c10::SymNode>> {
 public:
  // 使用 PYBIND11_TYPE_CASTER 宏定义转换器，将 `at::ArrayRef<c10::SymNode>` 映射到 _("List[SymNode]")，这是其在 Python 中的类型名称
  PYBIND11_TYPE_CASTER(at::ArrayRef<c10::SymNode>, _("List[SymNode]"));

  // 加载函数，从 Python 对象 `src` 中加载数据到 `at::ArrayRef<c10::SymNode>`
  bool load(handle src, bool);

  // 静态 cast 函数，将 `at::ArrayRef<c10::SymNode>` 转换为 Python 对象
  static handle cast(
      at::ArrayRef<c10::SymNode> src,
      return_value_policy /* policy */,
      handle /* parent */);

 private:
  // 内部存储的数据类型为 `std::vector<c10::SymNode>`
  std::vector<c10::SymNode> v_value;
};

// 特化模板，用于将 `at::MemoryFormat` 转换为 Python 中的 `torch.memory_format` 类型
template <>
struct type_caster<at::MemoryFormat> {
 public:
  // 使用 PYBIND11_TYPE_CASTER 宏定义转换器，将 `at::MemoryFormat` 映射到 _("torch.memory_format")，这是其在 Python 中的类型名称
  PYBIND11_TYPE_CASTER(at::MemoryFormat, _("torch.memory_format"));

  // 加载函数，从 Python 对象 `src` 中加载数据到 `at::MemoryFormat`
  bool load(handle src, bool);

  // 静态 cast 函数，将 `at::MemoryFormat` 转换为 Python 对象
  static handle cast(
      at::MemoryFormat src,
      return_value_policy /* policy */,
      handle /* parent */);

 private:
  // 内部存储的数据类型为 `at::MemoryFormat`
  at::MemoryFormat value;
};

// 特化模板，用于将 `at::Device` 转换为 Python 中的 `torch.device` 类型
template <>
struct type_caster<at::Device> {
 public:
  // 使用 PYBIND11_TYPE_CASTER 宏定义转换器，将 `at::Device` 映射到 _("torch.device")，这是其在 Python 中的类型名称
  PYBIND11_TYPE_CASTER(at::Device, _("torch.device"));

  // 默认构造函数，初始化 `value` 字段为 `c10::kCPU`
  type_caster() : value(c10::kCPU) {}

  // 加载函数，从 Python 对象 `src` 中加载数据到 `at::Device`
  bool load(handle src, bool);

  // 静态 cast 函数，将 `at::Device` 转换为 Python 对象
  static handle cast(
      const at::Device& src,
      return_value_policy /* policy */,
      handle /* parent */);

 private:
  // 内部存储的数据类型为 `at::Device`
  at::Device value;
};

// 特化模板，用于将 `at::ScalarType` 转换为 Python 中的 `torch.dtype` 类型
template <>
struct type_caster<at::ScalarType> {
 public:
  // 使用 PYBIND11_TYPE_CASTER 宏定义转换器，将 `at::ScalarType` 映射到 _("torch.dtype")，这是其在 Python 中的类型名称
  PYBIND11_TYPE_CASTER(at::ScalarType, _("torch.dtype"));

  // 默认构造函数，初始化 `value` 字段为 `at::kFloat`
  type_caster() : value(at::kFloat) {}

  // 加载函数，从 Python 对象 `src` 中加载数据到 `at::ScalarType`
  bool load(handle src, bool);

  // 静态 cast 函数，将 `at::ScalarType` 转换为 Python 对象
  static handle cast(
      at::ScalarType src,
      return_value_policy /* policy */,
      handle /* parent */);

 private:
  // 内部存储的数据类型为 `at::ScalarType`
  at::ScalarType value;
};
    // 如果 obj 是 THPDtype 类型的对象
    if (THPDtype_Check(obj)) {
      // 获取 obj 对象中的 scalar_type 字段的值，并赋给变量 value
      value = reinterpret_cast<THPDtype*>(obj)->scalar_type;
      // 返回 true 表示成功转换
      return true;
    }
    // 如果 obj 不是 THPDtype 类型的对象，则返回 false 表示转换失败
    return false;
  }

  // 静态方法 cast，用于将 at::ScalarType 转换为 Python 对象
  static handle cast(
      const at::ScalarType& src,
      return_value_policy /* policy */, // 返回值策略，此处未使用
      handle /* parent */) {  // 父对象句柄，此处未使用
    // 调用 torch::getTHPDtype 获取 src 对应的 THPDtype 对象，并返回其 Python 句柄
    return Py_NewRef(torch::getTHPDtype(src));
  }
};

// 特化模板，用于将 c10::Stream 类型转换为 Python 对象
template <>
struct type_caster<c10::Stream> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(c10::Stream, _("torch.Stream"));

  // 构造函数，初始化 value 字段为 c10::Stream::DEFAULT 和 c10::Device(c10::kCPU, 0)
  // 这里的具体数值不重要，因为会在 load 函数成功调用后被覆盖
  type_caster() : value(c10::Stream::DEFAULT, c10::Device(c10::kCPU, 0)) {}

  // 加载函数，从 Python 对象 src 装载数据到 value
  bool load(handle src, bool) {
    PyObject* obj = src.ptr();
    // 检查 src 是否为 THPStream 类型
    if (THPStream_Check(obj)) {
      // 如果是 THPStream 类型，则从该对象中解包 stream_id、device_index 和 device_type
      value = c10::Stream::unpack3(
          ((THPStream*)obj)->stream_id,
          static_cast<c10::DeviceIndex>(((THPStream*)obj)->device_index),
          static_cast<c10::DeviceType>(((THPStream*)obj)->device_type));
      return true;
    }
    return false;
  }

  // 转换函数，将 c10::Stream 类型对象 src 转换为 Python 对象
  static handle cast(
      const c10::Stream& src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return handle(THPStream_Wrap(src));
  }
};

// 特化模板，用于将 c10::DispatchKey 类型转换为 Python 对象
template <>
struct type_caster<c10::DispatchKey>
    : public type_caster_base<c10::DispatchKey> {
  using base = type_caster_base<c10::DispatchKey>;
  c10::DispatchKey tmp{};

 public:
  // 加载函数，从 Python 对象 src 装载数据到 value
  bool load(handle src, bool convert) {
    // 调用基类的加载函数
    if (base::load(src, convert)) {
      return true;
    } else if (py::isinstance(
                   src, py::module_::import("builtins").attr("str"))) {
      // 如果 src 是字符串类型，则解析为 c10::DispatchKey 并赋值给 value
      tmp = c10::parseDispatchKey(py::cast<std::string>(src));
      value = &tmp;
      return true;
    }
    return false;
  }

  // 转换函数，将 c10::DispatchKey 类型对象 src 转换为 Python 对象
  static handle cast(
      c10::DispatchKey src,
      return_value_policy policy,
      handle parent) {
    return base::cast(src, policy, parent);
  }
};

// 特化模板，用于将 c10::Scalar 类型转换为 Python 对象
template <>
struct TORCH_PYTHON_API type_caster<c10::Scalar> {
 public:
  PYBIND11_TYPE_CASTER(
      c10::Scalar,
      _("Union[Number, torch.SymInt, torch.SymFloat, torch.SymBool]"));
  
  // 加载函数声明
  bool load(py::handle src, bool);

  // 转换函数，将 c10::Scalar 类型对象 si 转换为 Python 对象
  static py::handle cast(
      const c10::Scalar& si,
      return_value_policy /* policy */,
      handle /* parent */);
};

// 特化模板，用于将 c10::SymInt 类型转换为 Python 对象
template <>
struct TORCH_PYTHON_API type_caster<c10::SymInt> {
 public:
  PYBIND11_TYPE_CASTER(c10::SymInt, _("Union[int, torch.SymInt]"));
  
  // 加载函数声明
  bool load(py::handle src, bool);

  // 转换函数，将 c10::SymInt 类型对象 si 转换为 Python 对象
  static py::handle cast(
      const c10::SymInt& si,
      return_value_policy /* policy */,
      handle /* parent */);
};

// 特化模板，用于将 c10::SymFloat 类型转换为 Python 对象
template <>
struct TORCH_PYTHON_API type_caster<c10::SymFloat> {
 public:
  PYBIND11_TYPE_CASTER(c10::SymFloat, _("float"));
  
  // 加载函数声明
  bool load(py::handle src, bool);

  // 转换函数，将 c10::SymFloat 类型对象 si 转换为 Python 对象
  static py::handle cast(
      const c10::SymFloat& si,
      return_value_policy /* policy */,
      handle /* parent */);
};
// 定义一个类型转换器，用于将 c10::SymBool 类型转换为 Python 对象
struct TORCH_PYTHON_API type_caster<c10::SymBool> {
 public:
  // 声明类型转换器，指定返回类型信息字符串
  PYBIND11_TYPE_CASTER(c10::SymBool, _("Union[bool, torch.SymBool]"));
  
  // 加载函数，从 Python 对象中加载数据到 c10::SymBool 类型
  bool load(py::handle src, bool);
  
  // 静态转换函数，将 c10::SymBool 转换为 Python 对象
  static py::handle cast(
      const c10::SymBool& si,
      return_value_policy /* policy */,
      handle /* parent */);
};

// 定义一个类型转换器模板，用于将 c10::complex<T> 类型转换为 Python 对象
template <typename T>
struct type_caster<c10::complex<T>> {
 public:
  // 声明类型转换器，指定返回类型信息字符串
  PYBIND11_TYPE_CASTER(c10::complex<T>, _("complex"));

  // 加载函数，从 Python 对象中加载数据到 c10::complex<T> 类型
  bool load(handle src, bool) {
    // 获取 Python 对象指针
    PyObject* obj = src.ptr();

    // 调用 Python C API 函数 PyComplex_AsCComplex 解析复数对象
    Py_complex py_complex = PyComplex_AsCComplex(obj);
    // 检查解析结果是否有效
    if (py_complex.real == -1.0 && PyErr_Occurred()) {
      return false;
    }

    // 将 Python 中的复数对象转换为 c10::complex<double> 类型
    value = c10::complex<double>(py_complex.real, py_complex.imag);
    return true;
  }

  // 静态转换函数，将 c10::complex<T> 转换为 Python 复数对象
  static handle cast(
      const c10::complex<T>& complex,
      return_value_policy /* policy */,
      handle /* parent */) {
    // 创建 Python 复数对象，并返回其句柄
    return handle(PyComplex_FromDoubles(complex.real(), complex.imag()));
  }
};

} // namespace pybind11::detail

namespace torch::impl {

// 用于释放 Python GIL 的函数，适用于同时在 C++ 和 Python 上下文中使用的对象
//
// 这个函数可以作为 shared_ptr 的析构函数使用，用于方便地分配一个 shared_ptr
// 指向的对象，在 Python 上下文中无需 GIL 就能销毁它。
//
// 将 GIL 释放逻辑附加到 holder 指针而不是 T 的实际析构函数，当 T 是与 Python 无关且
// 不应引用 Python API 时，这样做很有帮助。
//
// 注意，使用这种方法存在一些正确性的限制。特别是，如果一个 shared_ptr 是从 C++
// 代码构造而来，没有使用这个析构函数，然后传递给 pybind11，pybind11 将乐意接管
// 这个 shared_ptr（并愿意从持有 GIL 的上下文中销毁它）。带有类型品牌化删除器的
// unique_ptr 则不太容易出现这个问题，因为标准删除器的 unique_ptr 与其不可转换。
// 我计划通过在真实的 C++ 析构函数中增加 DEBUG-only 断言来减轻这个问题，以确保 GIL
// 没有被持有（使用虚拟调用来进入 Python 解释器）；或者，我们可以使用虚拟调用
// 简单地确保我们在 C++ 析构函数中释放 GIL，但这是一种分层违反（为什么显然与
// Python 无关的代码调用到 GIL 中）。
//
// 改编自 https://github.com/pybind/pybind11/issues/1446#issuecomment-406341510
template <typename T>
// 释放指向的对象，但不需要全局解释器锁（GIL）
inline void destroy_without_gil(T* ptr) {
  // 由于 shared_ptr 的所有权是模糊的，无法确切预测对象的最后引用是否来自 Python 还是 C++。
  // 这意味着在这里的析构函数中，我们不能确定是否真正拥有 GIL；事实上，我们甚至不知道 Python 解释器是否仍然存在！
  // 因此，在释放 GIL 之前，我们必须对其进行测试。

  // PyGILState_Check 函数用于检查当前线程是否拥有 GIL。
  // Py_IsInitialized 函数用于检查 Python 解释器是否已初始化。
  // _PyIsFinalizing 在 Python 结束过程中设置，用于指示 Python 解释器是否正在最终化。
  // 因此，我们需要确保在 Python 解释器已经初始化后才释放 GIL，这时候才能确认不再需要 GIL。

  if (Py_IsInitialized() && PyGILState_Check()) {
    // 如果 Python 解释器已初始化且当前线程持有 GIL，则释放 GIL。
    pybind11::gil_scoped_release nogil;
    // 删除指针所指向的对象。
    delete ptr;
  } else {
    // 如果无法确定是否需要 GIL 或者 Python 解释器不存在，则直接删除指针所指向的对象。
    delete ptr;
  }
}

} // namespace torch::impl
```