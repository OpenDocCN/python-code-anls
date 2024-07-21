# `.\pytorch\c10\core\ConstantSymNodeImpl.h`

```
#pragma once

#include <c10/core/SymNodeImpl.h>  // 包含 SymNodeImpl 类的头文件
#include <c10/macros/Export.h>     // 包含导出宏的头文件
#include <c10/util/Exception.h>    // 包含异常处理工具的头文件
#include <c10/util/Optional.h>     // 包含可选值工具的头文件
#include <cstdint>                 // 包含整数类型的头文件
#include <string>                  // 包含字符串处理的头文件
#include <variant>                 // 包含变体类型的头文件

namespace c10 {

// Unlike other SymNodeImpl, this cannot be "dispatched" conventionally,
// as it typically needs to defer to another SymNodeImpl
//
// Can either represent a bool, int (don't support float yet) this is useful
// for representing otherwise unrepresentable large negative integer constant.
// 定义模板类 ConstantSymNodeImpl，继承自 SymNodeImpl 类，用于表示常量符号节点
template <typename T>
class C10_API ConstantSymNodeImpl : public SymNodeImpl {
  static_assert(
      ::std::is_same_v<T, int64_t> || ::std::is_same_v<T, bool>,
      "ConstantSymNodeImpl can only accept int64_t or bool types");

 public:
  // 构造函数，接受一个值作为参数，初始化 value_ 成员
  ConstantSymNodeImpl(T val) : value_(val) {}

  // 检查是否为整数类型
  bool is_int() override {
    return is_int_();
  }

  // 检查是否为布尔类型
  bool is_bool() override {
    return is_bool_();
  }

  // 检查是否为浮点数类型（始终返回 false，因为不支持浮点数）
  bool is_float() override {
    return false;
  }

  // 返回整数值，若不是整数则抛出异常
  int64_t guard_int(const char* file, int64_t line) override {
    TORCH_CHECK(is_int(), "not an int");
    return int_();
  }

  // 返回布尔值，若不是布尔类型则抛出异常
  bool guard_bool(const char* file, int64_t line) override {
    TORCH_CHECK(is_bool(), "not a bool");
    return bool_();
  }

  // 返回浮点数值，若不是浮点数则抛出异常（始终抛出异常，因为不支持浮点数）
  double guard_float(const char* file, int64_t line) override {
    TORCH_CHECK(false, "not a float");
  }

  // 返回整数值，若不是整数则抛出异常
  int64_t int_() override {
    TORCH_CHECK(is_int(), "not an int");
    return ::std::get<int64_t>(value_);
  }

  // 返回布尔值，若不是布尔类型则抛出异常
  bool bool_() override {
    TORCH_CHECK(is_bool(), "not a bool");
    return ::std::get<bool>(value_);
  }

  // 返回是否有提示（始终返回 true）
  bool has_hint() override {
    return true;
  }

  // 比较操作符的重载，返回对应的 SymNode 对象
  c10::SymNode eq(const c10::SymNode& other) override;
  c10::SymNode ne(const c10::SymNode& other) override;
  c10::SymNode ge(const c10::SymNode& other) override;
  c10::SymNode le(const c10::SymNode& other) override;
  c10::SymNode lt(const c10::SymNode& other) override;
  c10::SymNode gt(const c10::SymNode& other) override;
  c10::SymNode mul(const c10::SymNode& other) override;

  // 返回描述该常量节点的字符串表示
  ::std::string str() override {
    if constexpr (is_int_()) {
      return ::std::to_string(::std::get<int64_t>(value_));
    } else {
      return ::std::get<bool>(value_) ? "true" : "false";
    }
  }

  // 返回常量整数值的可选项（若不是整数则返回空）
  std::optional<int64_t> constant_int() override {
    if constexpr (is_int_()) {
      return ::std::get<int64_t>(value_);
    } else {
      return c10::nullopt;
    }
  }

  // 返回常量布尔值的可选项（若不是布尔类型则返回空）
  std::optional<bool> constant_bool() override {
    if constexpr (is_bool_()) {
      return ::std::get<bool>(value_);
    } else {
      return c10::nullopt;
    }
  }

  // 返回是否为常量（始终返回 true）
  bool is_constant() override {
    return true;
  }

  // 返回是否为符号（始终返回 false）
  bool is_symbolic() override {
    return false;
  }

 private:
  // 值的变体类型，可以是 int64_t 或 bool
  ::std::variant<int64_t, bool> value_;

  // 检查是否为整数类型的静态成员函数
  static constexpr bool is_int_() {
    return ::std::is_same_v<T, int64_t>;
  }

  // 检查是否为布尔类型的静态成员函数
  static constexpr bool is_bool_() {
    return ::std::is_same_v<T, bool>;
  }
};

} // namespace c10
```