# `.\pytorch\aten\src\ATen\core\Dimname.h`

```
#pragma once
// 预处理指令，确保此头文件只被编译一次

#include <ATen/core/symbol.h>
// 包含 ATen 库中的 symbol.h 头文件

#include <c10/util/ArrayRef.h>
// 包含 c10 库中的 ArrayRef.h 头文件

#include <c10/util/Optional.h>
// 包含 c10 库中的 Optional.h 头文件

#include <ostream>
// 包含输出流的头文件，用于支持输出流操作符重载

namespace at {
// 命名空间 at，定义了 ATen 库的命名空间

enum class NameType: uint8_t { BASIC, WILDCARD };
// 枚举类型 NameType，表示维度名的类型，可以是基本类型或通配符类型

struct TORCH_API Dimname {
  // 维度名结构体 Dimname 的声明

  static Dimname fromSymbol(Symbol name);
  // 静态方法，通过符号创建 Dimname 对象

  static Dimname wildcard();
  // 静态方法，返回一个通配符类型的 Dimname 对象

  static bool isValidName(const std::string& name);
  // 静态方法，检查给定名称是否是有效的维度名

  NameType type() const { return type_; }
  // 成员方法，返回当前维度名的类型

  Symbol symbol() const { return name_; }
  // 成员方法，返回当前维度名的符号对象

  bool isBasic() const { return type_ == NameType::BASIC; }
  // 成员方法，检查当前维度名是否为基本类型

  bool isWildcard() const { return type_ == NameType::WILDCARD; }
  // 成员方法，检查当前维度名是否为通配符类型

  bool matches(Dimname other) const;
  // 成员方法，检查当前维度名是否与另一个维度名匹配

  std::optional<Dimname> unify(Dimname other) const;
  // 成员方法，尝试将当前维度名与另一个维度名统一

 private:
  Dimname(Symbol name)
    : name_(name), type_(NameType::BASIC) {}
  // 私有构造函数，通过符号创建基本类型的 Dimname 对象

  Dimname(Symbol name, NameType type)
    : name_(name), type_(type) {}
  // 私有构造函数，通过符号和类型创建 Dimname 对象

  Symbol name_;
  // 维度名的符号对象

  NameType type_;
  // 维度名的类型
};

using DimnameList = c10::ArrayRef<Dimname>;
// 使用 ArrayRef 定义维度名列表类型 DimnameList

TORCH_API std::ostream& operator<<(std::ostream& out, const Dimname& dimname);
// 在全局命名空间下重载输出流操作符 <<，用于打印维度名对象到输出流

inline bool operator==(const Dimname& lhs, const Dimname& rhs) {
  return lhs.symbol() == rhs.symbol();
}
// 在全局命名空间下重载相等比较操作符 ==，用于比较两个维度名对象是否相等

inline bool operator!=(const Dimname& lhs, const Dimname& rhs) {
  return !(lhs == rhs);
}
// 在全局命名空间下重载不等比较操作符 !=，用于比较两个维度名对象是否不相等

} // namespace at
// 结束命名空间 at
```