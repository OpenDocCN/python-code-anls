# `.\pytorch\aten\src\ATen\core\operator_name.h`

```py
#pragma once

#include <c10/macros/Macros.h>  // 引入 c10 宏定义
#include <c10/util/Exception.h>  // 引入 c10 异常处理工具
#include <c10/util/Optional.h>   // 引入 c10 可选值工具
#include <c10/util/string_view.h>  // 引入 c10 字符串视图工具
#include <string>  // 引入标准字符串库
#include <utility>  // 引入实用工具（如 std::move）
#include <ostream>  // 引入输出流

namespace c10 {

// TODO: consider storing namespace separately too
// OperatorName 结构体，用于表示操作符名称
struct OperatorName final {
  std::string name;  // 操作符名称
  std::string overload_name;  // 重载名称

  // OperatorName 构造函数，初始化 name 和 overload_name
  OperatorName(std::string name, std::string overload_name)
      : name(std::move(name)), overload_name(std::move(overload_name)) {}

  // TODO: These two functions below are slow!  Fix internal data structures so
  // I don't have to manually reconstruct the namespaces!

  // 获取 OperatorName 的命名空间，返回一个可选的 string_view
  // 如果不存在命名空间，则返回 nullopt
  std::optional<c10::string_view> getNamespace() const {
    auto pos = name.find("::");
    if (pos == std::string::npos) {
      return c10::nullopt;
    } else {
      return c10::make_optional(c10::string_view(name.data(), pos));
    }
  }

  // 如果命名空间不存在，则设置命名空间为 ns，并返回 true
  // 否则返回 false
  bool setNamespaceIfNotSet(const char* ns) {
    if (!getNamespace().has_value()) {
      const auto ns_len = strlen(ns);
      const auto old_name_size = name.size();
      name.resize(ns_len + 2 + old_name_size);
      // 将当前 name 的值移到新空间的末尾
      name.replace(name.size() - old_name_size, old_name_size, name, 0, old_name_size);
      // 替换前缀为新的命名空间 ns
      name.replace(0, ns_len, ns, ns_len);
      name[ns_len] = ':';
      name[ns_len + 1] = ':';
      return true;
    } else {
      return false;
    }
  }
};

// OperatorName 的非拥有视图类 OperatorNameView
// 大多数函数都是 constexpr，可以用于编译时计算
struct OperatorNameView final {
  c10::string_view name;  // 操作符名称视图
  c10::string_view overload_name;  // 重载名称视图

  // constexpr 构造函数，初始化 name 和 overload_name 视图
  constexpr OperatorNameView(c10::string_view name, c10::string_view overload_name)
    : name(name), overload_name(overload_name) {}

  // 解析 full_name，返回 OperatorNameView 对象
  // full_name 的格式可能是 "foo.overload" 或 "foo"
  constexpr static OperatorNameView parse(c10::string_view full_name) {
    auto i = full_name.find('.');
    if (i == c10::string_view::npos) {
      return OperatorNameView(full_name, c10::string_view());
    } else {
      return OperatorNameView(full_name.substr(0, i), full_name.substr(i + 1));
    }
  }
};

// OperatorName 的相等比较运算符重载
inline bool operator==(const OperatorName& lhs, const OperatorName& rhs) {
  return lhs.name == rhs.name && lhs.overload_name == rhs.overload_name;
}

// OperatorName 的不等比较运算符重载
inline bool operator!=(const OperatorName& lhs, const OperatorName& rhs) {
  return !operator==(lhs, rhs);
}

// OperatorName 的 toString 函数声明
TORCH_API std::string toString(const OperatorName& opName);

// OperatorName 的输出流运算符重载声明
TORCH_API std::ostream& operator<<(std::ostream&, const OperatorName&);

} // namespace c10

namespace std {
  // std 命名空间中的哈希模板特化，用于 OperatorName 类型
  template <>
  struct hash<::c10::OperatorName> {
    size_t operator()(const ::c10::OperatorName& x) const {
      // 定义哈希函数的操作符重载，接收一个类型为 ::c10::OperatorName 的参数 x
      // 计算 x.name 的哈希值，并与 x.overload_name 的哈希值进行按位异或操作
      return std::hash<std::string>()(x.name) ^ (~ std::hash<std::string>()(x.overload_name));
    }
  };
}
```