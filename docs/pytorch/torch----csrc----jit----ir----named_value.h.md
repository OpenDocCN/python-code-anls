# `.\pytorch\torch\csrc\jit\ir\named_value.h`

```py
#pragma once
// 引入ATen库中的ivalue.h头文件
#include <ATen/core/ivalue.h>
// 引入torch库中的source_range.h头文件
#include <torch/csrc/jit/frontend/source_range.h>
// 引入torch库中的constants.h头文件
#include <torch/csrc/jit/ir/constants.h>
// 引入torch库中的variadic.h头文件
#include <torch/csrc/utils/variadic.h>

// torch命名空间
namespace torch {
// JIT命名空间
namespace jit {

// 值结构体
struct Value;

/**
 * NamedValue结构体表示一个值，带有可选的额外名称和位置信息。
 * 在模式匹配中用于提供额外的错误信息和解析关键字参数。
 */
struct NamedValue {
  // 带位置、名称和值的构造函数
  NamedValue(const SourceRange& loc, const std::string& name, Value* value)
      : loc_(loc), name_(name), value_(value) {}
  // 带位置和值的构造函数
  NamedValue(const SourceRange& loc, Value* value) : loc_(loc), value_(value) {}

  /* implicit */ NamedValue(Value* value) : value_(value) {}
  // 带名称和值的构造函数
  NamedValue(const std::string& name, Value* value)
      : name_(name), value_(value) {}

  /* implicit */ NamedValue(IValue value)
      : value_(nullptr), ivalue_(std::move(value)) {}

  // 带名称和IValue的构造函数
  NamedValue(const std::string& name, IValue value)
      : name_(name), ivalue_(std::move(value)) {}

  // 模板构造函数，接受任何T类型参数，并将其转换为IValue
  template <
      typename T,
      typename = std::enable_if_t<
          (!std::is_same_v<std::decay_t<T>, NamedValue> &&
           !std::is_same_v<std::decay_t<T>, Value*> &&
           !std::is_same_v<std::decay_t<T>, IValue>)>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  NamedValue(T&& t) : NamedValue(IValue(std::forward<T>(t))) {}

  // 带名称和任何T类型参数的构造函数，转换为IValue
  template <
      typename T,
      typename = std::enable_if_t<
          (!std::is_same_v<std::decay_t<T>, Value*> &&
           !std::is_same_v<std::decay_t<T>, IValue>)>>
  NamedValue(const std::string& name, T&& t)
      : NamedValue(name, IValue(std::forward<T>(t))) {}

  // 返回位置信息或备用位置信息
  SourceRange locOr(const SourceRange& backup_location) const {
    if (!loc_)
      return backup_location;
    return loc();
  }

  // 返回值的指针，如果值为空，则在当前插入点插入一个常量节点
  Value* value(Graph& g) const {
    if (!value_)
      return insertConstant(
          g, ivalue_); // 使用insertConstant来避免在此处包含ir.h的需要
    return value_;
  }

  // 返回名称的引用
  const std::string& name() const {
    AT_ASSERT(name_);
    return *name_;
  }

  // 返回位置的引用
  const SourceRange& loc() const {
    AT_ASSERT(loc_);
    return *loc_;
  }

  // 返回类型指针
  at::TypePtr type() const;

 private:
  // 可选的位置信息
  std::optional<SourceRange> loc_;
  // 可选的名称
  std::optional<std::string> name_;
  // 值的指针，如果为nullptr，则ivalue_有效
  Value* value_{nullptr};
  // 当value_ == nullptr时有效的IValue
  IValue ivalue_;
};

} // namespace jit
} // namespace torch
```