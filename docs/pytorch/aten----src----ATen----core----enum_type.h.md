# `.\pytorch\aten\src\ATen\core\enum_type.h`

```
#pragma once

#include <ATen/core/ivalue.h>  // 包含 ATen 库中的 IValue 头文件

#include <utility>  // 包含标准库中的 utility 头文件，用于使用 std::pair 和 std::move

namespace c10 {

struct EnumType;  // 前向声明 EnumType 结构体
using EnumTypePtr = std::shared_ptr<EnumType>;  // 使用智能指针管理 EnumType
using EnumNameValue = std::pair<std::string, IValue>;  // 定义 EnumNameValue 类型为 std::pair<std::string, IValue>

// EnumType 结构体，继承自 NamedType
struct TORCH_API EnumType : public NamedType {
  friend struct Type;  // 声明 Type 结构体为友元

  static const TypeKind Kind = TypeKind::EnumType;  // 定义静态成员变量 Kind 为 TypeKind::EnumType

  // 静态方法，创建 EnumType 对象
  static EnumTypePtr create(
      const c10::QualifiedName& qualified_class_name,  // 枚举类名的限定名称
      TypePtr value,  // 类型指针，表示枚举值的类型
      std::vector<EnumNameValue> enum_names_values,  // 包含枚举名和对应值的向量
      std::weak_ptr<::torch::jit::CompilationUnit> cu) {  // torch JIT 编译单元的弱引用
    switch (value->kind()) {  // 根据 value 的类型分支
      case TypeKind::IntType:  // 整数类型
      case TypeKind::FloatType:  // 浮点数类型
      case TypeKind::StringType:  // 字符串类型
        return EnumTypePtr(new EnumType(  // 返回新创建的 EnumTypePtr 对象
            qualified_class_name,
            std::move(value),
            std::move(enum_names_values),
            std::move(cu)));
      default:
        AT_ERROR(  // 抛出错误，表示不支持该类型的枚举值
            "Cannot create Enum with value type '",
            value->str(),
            "', only int, float and string are supported");
    }
  }

  std::string str() const override {  // 返回枚举类型的字符串表示
    return "Enum<" + annotation_str() + ">";
  }

  std::string repr_str() const override {  // 返回枚举类型的字符串表示
    return str();
  }

  const TypePtr& getValueType() const {  // 返回枚举值的类型指针
    return value_type_;
  }

  bool equals(const Type& rhs) const override {  // 比较枚举类型是否相等
    if (auto* enum_rhs = rhs.castRaw<EnumType>()) {  // 尝试将 rhs 转换为 EnumType 类型
      return name().value() == enum_rhs->name().value() &&  // 比较枚举类型名称是否相同
          *getValueType() == *(enum_rhs->getValueType()) &&  // 比较枚举值类型是否相同
          this->compilation_unit() == enum_rhs->compilation_unit();  // 比较编译单元是否相同
    }
    return false;
  }

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;  // 检查枚举类型是否是 rhs 的子类型

  std::shared_ptr<const ::torch::jit::CompilationUnit> compilation_unit()  // 返回 JIT 编译单元的共享指针
      const {
    auto cu = cu_.lock();  // 获取编译单元的强引用
    return cu;
  }

  const QualifiedName& qualifiedClassName() const {  // 返回枚举类的限定名称
    return name().value();
  }

  at::ArrayRef<TypePtr> containedTypes() const override {  // 返回包含的类型指针数组的引用
    return value_type_;
  }

  const at::ArrayRef<EnumNameValue> enumNamesValues() const {  // 返回枚举名称和对应值的引用
    return enum_names_values_;
  }

 private:
  EnumType(
      c10::QualifiedName qualified_class_name,  // 枚举类名的限定名称
      TypePtr value_type,  // 枚举值的类型指针
      std::vector<EnumNameValue> enum_names_values,  // 包含枚举名和对应值的向量
      std::weak_ptr<torch::jit::CompilationUnit> cu)  // torch JIT 编译单元的弱引用
      : NamedType(TypeKind::EnumType, std::move(qualified_class_name)),  // 调用基类 NamedType 的构造函数
        value_type_(std::move(value_type)),  // 初始化成员变量 value_type_
        enum_names_values_(std::move(enum_names_values)),  // 初始化成员变量 enum_names_values_
        cu_(std::move(cu)) {}  // 初始化成员变量 cu_

  std::string annotation_str_impl(  // 返回枚举类型的注释字符串表示
      C10_UNUSED const TypePrinter& printer = nullptr) const override {
    const auto& n = name().value();  // 获取枚举类型的名称
    return n.qualifiedName();  // 返回限定名称
  }

  TypePtr value_type_;  // 枚举值的类型指针
  std::vector<EnumNameValue> enum_names_values_;  // 包含枚举名和对应值的向量
  std::weak_ptr<::torch::jit::CompilationUnit> cu_;  // torch JIT 编译单元的弱引用
};

}  // namespace c10
```