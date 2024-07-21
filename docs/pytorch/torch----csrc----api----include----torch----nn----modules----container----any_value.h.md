# `.\pytorch\torch\csrc\api\include\torch\nn\modules\container\any_value.h`

```
#pragma once

#include <torch/detail/static.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/variadic.h>

#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AnyValue ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// An implementation of `std::any` which stores
/// a type erased object, whose concrete value can be retrieved at runtime by
/// checking if the `typeid()` of a requested type matches the `typeid()` of
/// the object stored.
class AnyValue {
 public:
  /// Move construction and assignment is allowed, and follows the default
  /// behavior of move for `std::unique_ptr`.
  AnyValue(AnyValue&&) = default;
  AnyValue& operator=(AnyValue&&) = default;

  /// Copy construction and assignment is allowed.
  AnyValue(const AnyValue& other) : content_(other.content_->clone()) {}
  AnyValue& operator=(const AnyValue& other) {
    content_ = other.content_->clone();
    return *this;
  }

  /// Constructs the `AnyValue` from value type.
  template <typename T>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  explicit AnyValue(T&& value)
      : content_(
            std::make_unique<Holder<std::decay_t<T>>>(std::forward<T>(value))) {
  }

  /// Returns a pointer to the value contained in the `AnyValue` if the type
  /// passed as template parameter matches the type of the value stored, and
  /// returns a null pointer otherwise.
  template <typename T>
  T* try_get() {
    // Ensure T is not a reference type
    static_assert(
        !std::is_reference<T>::value,
        "AnyValue stores decayed types, you cannot cast it to a reference type");
    // Ensure T is not an array type
    static_assert(
        !std::is_array<T>::value,
        "AnyValue stores decayed types, you must cast it to T* instead of T[]");
    
    // Check if the type of T matches the stored type and return the pointer to the value
    if (typeid(T).hash_code() == type_info().hash_code()) {
      return &static_cast<Holder<T>&>(*content_).value;
    }
    return nullptr;
  }

  /// Returns the value contained in the `AnyValue` if the type passed as
  /// template parameter matches the type of the value stored, and throws an
  /// exception otherwise.
  template <typename T>
  T get() {
    // Try to get the value of type T
    if (auto* maybe_value = try_get<T>()) {
      return *maybe_value;
    }
    
    // If type does not match, throw an exception detailing the actual type stored
    AT_ERROR(
        "Attempted to cast AnyValue to ",
        c10::demangle(typeid(T).name()),
        ", but its actual type is ",
        c10::demangle(type_info().name()));
  }

  /// Returns the `type_info` object of the contained value.
  const std::type_info& type_info() const noexcept {
    return content_->type_info;
  }

 private:
  friend struct AnyModulePlaceholder;
  friend struct TestAnyValue;

  /// \internal
  /// The static type of the object we store in the `AnyValue`, which erases the
  /// actual object's type, allowing us only to check the `type_info` of the
  /// type stored in the dynamic type.
  struct Placeholder {
    virtual ~Placeholder() = default;
    virtual std::unique_ptr<Placeholder> clone() const = 0;
    const std::type_info& type_info;
  };

  /// \internal
  /// Template class to hold the actual value and implement cloning.
  template <typename ValueType>
  struct Holder : public Placeholder {
    explicit Holder(ValueType&& value)
        : value(std::forward<ValueType>(value)), type_info(typeid(ValueType)) {}
    std::unique_ptr<Placeholder> clone() const override {
      return std::make_unique<Holder>(value);
    }
    ValueType value;
  };

  std::unique_ptr<Placeholder> content_;
};
    /// \internal
    /// 表示我们在 `AnyValue` 中存储的对象的动态类型，它隐藏了我们在这个 `AnyValue` 中擦除的实际对象。
    struct Placeholder {
        /// 用给定的类型信息构造函数，不抛出异常
        explicit Placeholder(const std::type_info& type_info_) noexcept
            : type_info(type_info_) {}
        /// 使用默认拷贝构造函数
        Placeholder(const Placeholder&) = default;
        /// 使用默认移动构造函数
        Placeholder(Placeholder&&) = default;
        /// 使用默认析构函数
        virtual ~Placeholder() = default;
        /// 克隆函数，返回一个指向新对象的唯一指针
        virtual std::unique_ptr<Placeholder> clone() const {
          // 断言：不应该在 `AnyValue::Holder` 上调用 `clone()`
          TORCH_CHECK(false, "clone() should only be called on `AnyValue::Holder`");
        }
        /// 存储对象的类型信息的常引用
        const std::type_info& type_info;
      };
    
    /// \internal
    /// 存储在 `AnyValue` 中的对象的动态类型，它隐藏了我们在这个 `AnyValue` 中擦除的实际对象。
    template <typename T>
    struct Holder : public Placeholder {
        /// 由于 T&& 在这里不是通用引用，所以使用模板
        /// 使用给定值的类型 U 构造函数，不抛出异常
        explicit Holder(U&& value_) noexcept
            : Placeholder(typeid(T)), value(std::forward<U>(value_)) {}
        /// 克隆函数，返回指向新对象的唯一指针
        std::unique_ptr<Placeholder> clone() const override {
          // 返回一个新的指向 Holder<T> 的唯一指针，持有当前 value 的副本
          return std::make_unique<Holder<T>>(value);
        }
        /// 存储的值的实例
        T value;
      };
    
    /// 存储类型擦除后的对象。
    std::unique_ptr<Placeholder> content_;
};

// 结束 nn 命名空间
} // namespace nn
// 结束 torch 命名空间
} // namespace torch
```