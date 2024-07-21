# `.\pytorch\torch\csrc\jit\ir\attributes.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/core/Tensor.h>
// 包含 ATen 库的 Tensor 头文件

#include <string>
#include <vector>
// 包含标准库的 string 和 vector 头文件

#include <ATen/core/jit_type_base.h>
#include <ATen/core/symbol.h>
// 包含 ATen 库的 jit_type_base 和 symbol 头文件

#include <torch/csrc/Export.h>
// 包含 Torch 库的 Export 头文件

namespace torch {
namespace jit {
// 命名空间 torch::jit，用于 Torch JIT 模块的代码

using ::c10::Symbol;
// 使用 c10 命名空间中的 Symbol 类

constexpr int max_tensor_display_size = 10;
// 定义最大张量显示尺寸为 10

enum class AttributeKind {
  f,   // 单个浮点数属性
  fs,  // 浮点数向量属性
  c,   // 单个复数属性
  cs,  // 复数向量属性
  i,   // 单个整数属性
  is,  // 整数向量属性
  s,   // 单个字符串属性
  ss,  // 字符串向量属性
  t,   // 单个张量属性
  ts,  // 张量向量属性
  g,   // TODO: 待确认
  gs,  // TODO: 待确认
  ty,  // TODO: 待确认
  tys, // TODO: 待确认
  ival // 整数值属性
};

static inline const char* toString(AttributeKind kind) {
  // 返回 AttributeKind 对应的字符串表示
  static const char* names[] = {
      "f",
      "c",
      "cs",
      "fs",
      "i",
      "is",
      "s",
      "ss",
      "t",
      "ts",
      "g",
      "gs",
      "ty",
      "tys",
      "ival"};
  // 断言确保 kind 在有效范围内
  AT_ASSERT(size_t(kind) < sizeof(names) / sizeof(*names));
  return names[int(kind)];
}

struct AttributeValue {
  AttributeValue(Symbol name) : name(name) {}
  // 属性值结构，使用给定的符号名称初始化

  using Ptr = std::unique_ptr<AttributeValue>;
  // 定义 Ptr 为指向 AttributeValue 的 unique_ptr

  Symbol name;  // 属性的符号名称

  virtual AttributeKind kind() const = 0;
  // 纯虚函数，返回属性值的种类

  virtual Ptr clone() const = 0;
  // 纯虚函数，克隆当前属性值对象

  virtual ~AttributeValue() = default;
  // 虚析构函数，默认实现
};

template <typename T, AttributeKind Kind>
struct ScalarAttributeValue : public AttributeValue {
  // 标量属性值模板，继承自 AttributeValue

  using ConstructorType = T;
  using ValueType = T;

  ScalarAttributeValue(Symbol name, ConstructorType value_)
      : AttributeValue(name), value_(std::move(value_)) {}
  // 构造函数，使用给定的符号名称和值初始化

  ValueType& value() {
    return value_;
  }
  // 返回当前属性的值引用

  Ptr clone() const override {
    return Ptr(new ScalarAttributeValue(name, value_));
  }
  // 克隆当前标量属性对象

  AttributeKind kind() const override {
    return Kind;
  }
  // 返回属性值的种类
 private:
  ValueType value_;
  // 存储属性的值
};

template <typename T, AttributeKind Kind>
struct VectorAttributeValue : public AttributeValue {
  // 向量属性值模板，继承自 AttributeValue

  using ConstructorType = std::vector<T>;
  using ValueType = std::vector<T>;

  VectorAttributeValue(Symbol name, ConstructorType value_)
      : AttributeValue(name), value_(std::move(value_)) {}
  // 构造函数，使用给定的符号名称和值初始化

  ValueType& value() {
    return value_;
  }
  // 返回当前属性的值引用

  AttributeKind kind() const override {
    return Kind;
  }
  // 返回属性值的种类

  std::unique_ptr<AttributeValue> clone() const override {
    auto copy = value_;
    return Ptr(new VectorAttributeValue(name, std::move(copy)));
  }
  // 克隆当前向量属性对象

 private:
  ValueType value_;
  // 存储属性的值
};

using ComplexAttr =
    ScalarAttributeValue<c10::complex<double>, AttributeKind::c>;
using ComplexValsAttr =
    VectorAttributeValue<c10::complex<double>, AttributeKind::cs>;
using FloatAttr = ScalarAttributeValue<double, AttributeKind::f>;
using FloatsAttr = VectorAttributeValue<double, AttributeKind::fs>;
using IntAttr = ScalarAttributeValue<int64_t, AttributeKind::i>;
using IntsAttr = VectorAttributeValue<int64_t, AttributeKind::is>;
using StringAttr = ScalarAttributeValue<std::string, AttributeKind::s>;
using StringsAttr = VectorAttributeValue<std::string, AttributeKind::ss>;
using TensorAttr = ScalarAttributeValue<at::Tensor, AttributeKind::t>;
using TensorsAttr = VectorAttributeValue<at::Tensor, AttributeKind::ts>;
// 定义不同类型的属性及其模板化的版本
// 使用 TypeAttr 别名表示 ScalarAttributeValue 类模板实例化结果，其模板参数为 c10::TypePtr 和 AttributeKind::ty
using TypeAttr = ScalarAttributeValue<c10::TypePtr, AttributeKind::ty>;

// 使用 TypesAttr 别名表示 VectorAttributeValue 类模板实例化结果，其模板参数为 c10::TypePtr 和 AttributeKind::tys
using TypesAttr = VectorAttributeValue<c10::TypePtr, AttributeKind::tys>;

// 使用 IValueAttr 别名表示 ScalarAttributeValue 类模板实例化结果，其模板参数为 at::IValue 和 AttributeKind::ival
using IValueAttr = ScalarAttributeValue<at::IValue, AttributeKind::ival>;

// 定义 Graph 结构体，暂时未给出具体定义
struct Graph;

// 由于需要确保在克隆操作时调用 Graph::copy()，因此对 Graph 类型的属性进行特殊处理
// 继承自 AttributeValue 类
struct TORCH_API GraphAttr : public AttributeValue {
  using ConstructorType = std::shared_ptr<Graph>;
  using ValueType = std::shared_ptr<Graph>;
  
  // 构造函数，初始化 GraphAttr 对象
  GraphAttr(Symbol name, ConstructorType value_)
      : AttributeValue(name), value_(std::move(value_)) {}
  
  // 返回 GraphAttr 对象的值引用
  ValueType& value() {
    return value_;
  }
  
  // 返回属性类型为 AttributeKind::g
  AttributeKind kind() const override {
    return AttributeKind::g;
  }

  // 克隆操作，返回当前对象的副本指针
  Ptr clone() const override;
  
 private:
  std::shared_ptr<Graph> value_;
};

// 继承自 AttributeValue 类，用于表示包含多个 Graph 对象的属性
struct TORCH_API GraphsAttr : public AttributeValue {
  using ConstructorType = std::vector<std::shared_ptr<Graph>>;
  using ValueType = std::vector<std::shared_ptr<Graph>>;
  
  // 构造函数，初始化 GraphsAttr 对象
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  GraphsAttr(Symbol name, ConstructorType value_)
      : AttributeValue(name), value_(std::move(value_)) {}
  
  // 返回 GraphsAttr 对象的值引用
  ValueType& value() {
    return value_;
  }
  
  // 返回属性类型为 AttributeKind::gs
  AttributeKind kind() const override {
    return AttributeKind::gs;
  }

  // 克隆操作，返回当前对象的唯一指针
  std::unique_ptr<AttributeValue> clone() const override;
  
 private:
  ValueType value_;
};

// 表示 IR 属性错误的异常类，继承自 std::exception
struct IRAttributeError : public std::exception {
  // 构造函数，根据属性名称和定义情况生成错误消息
  IRAttributeError(Symbol name, bool defined) {
    std::stringstream ss;
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (!defined) {
      ss << "required keyword attribute '" << name.toUnqualString()
         << "' is undefined";
    } else {
      ss << "required keyword attribute '" << name.toUnqualString()
         << "' has the wrong type";
    }
    msg = ss.str();
  }
  
  // 返回异常消息字符串
  const char* what() const noexcept override {
    return msg.c_str();
  }

 private:
  std::string msg;
};

// 命名空间 jit 下的 torch 子命名空间结束
} // namespace jit
} // namespace torch
```