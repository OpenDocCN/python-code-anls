# `.\pytorch\torch\csrc\jit\python\pybind.h`

```py
#pragma once
// 使用 #pragma once 来确保头文件只被编译一次，避免重复包含

#include <torch/csrc/python_headers.h>
// 包含 Torch 的 Python 头文件，用于与 Python 解释器交互

#include <ATen/core/ivalue.h>
#include <ATen/core/symbol.h>
#include <c10/util/irange.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>
// 包含一系列 Torch 和 ATen 的头文件，用于定义和实现各种数据结构和函数

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// 包含 pybind11 头文件，用于将 C++ 函数包装成 Python 可调用对象

namespace py = pybind11;

namespace torch::jit {

// This is a variant of shared_ptr that "sees through" a wrapper.
// We use it to convert Value, Node, Block and node to "wrapped" Python
// values. When we destruct the C++ object, the wrapper's pointer will
// be set to 0 and any future dereferencing will throw. We need this
// because the Python objects may hang around after the C++ object
// has already been destroyed.
// This also needs the magic type_caster below, which is from the
// workaround offered in https://github.com/pybind/pybind11/issues/2751
// 定义了一个 unwrapping_shared_ptr 模板类，用于“透视”包装器中的 shared_ptr，
// 用于将 Value、Node、Block 和 node 转换为“包装” Python 值。

template <typename T>
class unwrapping_shared_ptr {
  static_assert(
      std::is_same<T, torch::jit::Value>::value ||
          std::is_same<T, torch::jit::Node>::value ||
          std::is_same<T, torch::jit::Block>::value,
      "unwrapping type only defined for Graph object types");
  // 静态断言，确保 T 只能是 torch::jit::Value、torch::jit::Node 或 torch::jit::Block 之一，
  // 因为这些类型才支持透视操作。

 private:
  std::shared_ptr<torch::jit::Wrap<T>> impl;
  // 实际存储的是指向 Wrap<T> 包装对象的 shared_ptr

 public:
  unwrapping_shared_ptr() : impl({}) {}
  // 默认构造函数，初始化为空的 shared_ptr

  explicit unwrapping_shared_ptr(T* p) : impl(p->wrap()) {
    impl->clear_cb = &clear_registered_instances;
    // 显式构造函数，接受 T* 指针并封装成 wrapped_ptr，设置清理回调函数
  }

  T* get() const {
    if (!impl->elem) {
      throw std::logic_error("has been invalidated");
    }
    return impl->elem;
    // 获取指向 T 对象的指针，如果为空则抛出逻辑错误异常
  }

  // we need to disable the overloaded & for PyBind11 < 2.3 due.
  // see https://github.com/pybind/pybind11/pull/1435
  // 针对 PyBind11 版本小于 2.3 的版本，需要禁用 & 操作符的重载

#if (PYBIND11_VERSION_MAJOR > 2) || \
    ((PYBIND11_VERSION_MAJOR == 2) && (PYBIND11_VERSION_MINOR >= 3))
  T** operator&() {
    if (!impl->elem) {
      throw std::logic_error("has been invalidated");
    }
    return &(impl->elem);
    // 重载 & 操作符，返回 T 对象的指针的指针，如果为空则抛出逻辑错误异常
  }
#endif
};

} // namespace torch::jit

PYBIND11_DECLARE_HOLDER_TYPE(T, torch::jit::unwrapping_shared_ptr<T>, true);
// 声明 pybind11 的类型持有者，指定为 torch::jit::unwrapping_shared_ptr<T>，支持自动转换为 Python 对象

namespace pybind11::detail {

#define CREATE_UNWRAPPING_CASTER(Class)                                                   \
  template <>                                                                             \
  struct type_caster<Class> : public type_caster_base<Class> {                            \
   public:                                                                                \
    using type = Class;                                                                   \
    using holder_type = torch::jit::unwrapping_shared_ptr<Class>;                         \
                                                                                          \
// 定义一个 CREATE_UNWRAPPING_CASTER 宏，用于生成 type_caster<Class> 的特化版本，
// 类型持有者为 torch::jit::unwrapping_shared_ptr<Class>
    # 定义一个名为 `load` 的成员函数，用于加载数据，接受两个参数：src 和 convert
    bool load(handle src, bool convert) {                                                 \
      return load_impl<type_caster<Class>>(src, convert);                                 \
    }                                                                                     \
                                                                                          \
    # 定义一个类型转换操作符，将该类转换为指向 type 类型的指针
    explicit operator type*() {                                                           \
      return static_cast<type*>(value);                                                   \
    }                                                                                     \
    # 定义一个类型转换操作符，将该类转换为 type 类型的引用
    explicit operator type&() {                                                           \
      return *static_cast<type*>(value);                                                  \
    }                                                                                     \
                                                                                          \
   protected:                                                                             \
    # 声明 type_caster_generic 类为友元类
    friend class type_caster_generic;                                                     \
                                                                                          \
    # 加载给定的值和持有器，并将值加载到类的 `value` 成员变量中
    bool load_value(value_and_holder&& v_h) {                                             \
      # 如果持有器已构建，则将其值存储到 `value` 成员变量，并返回 true
      if (v_h.holder_constructed()) {                                                     \
        value = v_h.template holder<holder_type>().get();                                 \
        return true;                                                                      \
      } else {                                                                            \
        # 如果持有器未构建，则抛出类型转换错误
        throw cast_error(                                                                 \
            "Unable to cast from non-held to held instance (#Class& to Holder<#Class>)"); \
      }                                                                                   \
    }                                                                                     \
  }
CREATE_UNWRAPPING_CASTER(torch::jit::Node);
// 创建一个用于将 torch::jit::Node 解包的类型转换器

CREATE_UNWRAPPING_CASTER(torch::jit::Value);
// 创建一个用于将 torch::jit::Value 解包的类型转换器

CREATE_UNWRAPPING_CASTER(torch::jit::Block);
// 创建一个用于将 torch::jit::Block 解包的类型转换器

#undef CREATE_UNWRAPPING_CASTER
// 取消之前定义的 CREATE_UNWRAPPING_CASTER 宏，确保不会在后续代码中重复定义

template <>
struct type_caster<torch::jit::IValue> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(torch::jit::IValue, _("IValue"));
  // 定义了一个用于 torch::jit::IValue 的类型转换器，其名称为 "IValue"

  bool load(handle src, bool) {
    try {
      // 尝试将 Python 对象转换为 torch::jit::IValue
      value = torch::jit::toTypeInferredIValue(src);
      return true;
    } catch (std::exception& e) {
      return false;
    }
  }

  static handle cast(
      torch::jit::IValue src,
      return_value_policy /* policy */,
      handle /* parent */) {
    // 将 torch::jit::IValue 转换为 Python 对象
    return torch::jit::toPyObject(std::move(src)).release();
  }
};

template <>
struct type_caster<torch::jit::Symbol> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(torch::jit::Symbol, _("Symbol"));
  // 定义了一个用于 torch::jit::Symbol 的类型转换器，其名称为 "Symbol"

  bool load(handle src, bool) {
    // 尝试从 Python 对象中加载数据
    std::string src_str;
    try {
      src_str = py::cast<std::string>(src);
    } catch (std::exception& e) {
      return false;
    }
    // 将字符串转换为 torch::jit::Symbol 对象
    value = torch::jit::Symbol::fromQualString(src_str);
    return true;
  }

  static handle cast(
      torch::jit::Symbol src,
      return_value_policy /* policy */,
      handle /* parent */) {
    // 将 torch::jit::Symbol 转换为 Python 对象
    return py::cast(std::string(src.toQualString()), return_value_policy::copy)
        .release();
  }
};

template <>
struct type_caster<torch::jit::AttributeKind> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(torch::jit::AttributeKind, _("AttributeKind"));
  // 定义了一个用于 torch::jit::AttributeKind 的类型转换器，其名称为 "AttributeKind"

  bool load(handle src, bool) {
    // 加载失败，返回 false
    return false;
  }

  static handle cast(
      torch::jit::AttributeKind src,
      return_value_policy /* policy */,
      handle /* parent */) {
    // 将 torch::jit::AttributeKind 转换为 Python 对象
    return py::cast(
               std::string(torch::jit::toString(src)),
               return_value_policy::copy)
        .release();
  }
};

// See https://github.com/pybind/pybind11/issues/637
using ListCasterBase = pybind11::detail::
    list_caster<std::vector<torch::jit::Node*>, torch::jit::Node*>;
template <>
struct type_caster<std::vector<torch::jit::Node*>> : ListCasterBase {
  static handle cast(
      const std::vector<torch::jit::Node*>& src,
      return_value_policy,
      handle parent) {
    // 将 std::vector<torch::jit::Node*> 转换为 Python 列表对象
    return ListCasterBase::cast(src, return_value_policy::reference, parent);
  }
  static handle cast(
      const std::vector<torch::jit::Node*>* src,
      return_value_policy pol,
      handle parent) {
    // 将 std::vector<torch::jit::Node*> 指针转换为 Python 列表对象
    return cast(*src, pol, parent);
  }
};

} // namespace pybind11::detail

namespace torch::jit {

static inline py::tuple tuple_tail(const py::tuple& tup) {
  // 创建一个新的 Python 元组，其中包含传入元组除第一个元素外的所有元素
  py::tuple r(tup.size() - 1);
  for (const auto i : c10::irange(1, tup.size())) {
    r[i - 1] = tup[i];
  }
  return r;
}

} // namespace torch::jit
```