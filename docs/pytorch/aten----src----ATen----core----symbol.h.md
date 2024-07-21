# `.\pytorch\aten\src\ATen\core\symbol.h`

```
#pragma once
// 指令，确保头文件只被包含一次

#include <c10/macros/Export.h>
// 包含 c10 库的导出宏头文件

#include <cstdint>
// 包含 C++ 标准头文件，定义了固定宽度的整数类型

#include <functional>  // For std::hash
// 包含 C++ 标准头文件，用于 std::hash 函数对象

#include <string>
// 包含 C++ 标准头文件，定义了字符串类型和相关操作

namespace c10 {

// 命名空间 c10 的开始

// 'prim' symbols are synthetic operators that occur only in the IR
// and don't have corresponding implementations in ATen.
// 'prim' 符号是仅出现在 IR 中的合成运算符，没有对应的 ATen 实现

// 'onnx' symbols correspond to ONNX operators.  Their semantics
// are defined in https://github.com/onnx/onnx/blob/master/docs/Operators.md
// The particular version we are targeting is specified by '_onnx_opset_version'
// in torch.onnx.symbolic_helper
// 'onnx' 符号对应于 ONNX 运算符。它们的语义在 https://github.com/onnx/onnx/blob/master/docs/Operators.md 中定义。
// 我们要针对的特定版本由 torch.onnx.symbolic_helper 中的 '_onnx_opset_version' 指定。

// In general, most ONNX operators won't get an entry here, because they
// are handled from the Python end.  However, you may occasionally need
// to intern an ONNX symbol here so that you can conveniently write an
// optimization on ONNX operations.
// 一般情况下，大多数 ONNX 运算符不会在这里列出，因为它们从 Python 端处理。
// 但是，偶尔需要在这里创建一个 ONNX 符号，以便可以方便地对 ONNX 操作进行优化。

// 'attr' symbols are attribute keys.  They are shared between both ONNX and ATen
// operators (you disambiguate their meaning by looking at the operator itself).
// In general, you only need to define attribute keys that are used by
// onnx or prim; ATen attributes are automatically generated in FORALL_ATTR_BASE_SYMBOLS.
// 'attr' 符号是属性键。它们在 ONNX 和 ATen 运算符之间共享（通过查看运算符本身来消除它们的含义歧义）。
// 通常情况下，只需要定义被 onnx 或 prim 使用的属性键；ATen 属性在 FORALL_ATTR_BASE_SYMBOLS 中自动生成。

// Note [Symbol allocation]
// ~~~~~~~~~~~~~~~~~~~~~~~~
// 符号分配的注意事项说明

//  1. Symbol namespace is split up into namespaces.
//     符号命名空间分为多个命名空间。

//  2. The intended access pattern for built-in symbols is onnx::MatMul
//     in the c10 namespace (this is a Symbol).
//     内置符号的预期访问模式是在 c10 命名空间中的 onnx::MatMul（这是一个符号）。

// Built-in constant definition strategy:
// 内置常量定义策略：

// - Enum is the most convenient way to generate a contiguous sequence
//   of numbers for an identifier.
//   枚举是生成连续数值标识符的最便捷方式。

// - However, an enum gives you a fresh type.  We want onnx::MatMul to
//   be type Symbol, not some random enum type!
//   但是，枚举会给你一个新类型。我们希望 onnx::MatMul 是 Symbol 类型，而不是某个随机的枚举类型！

// - Therefore, after using enums to generate the sequence of integers,
//   we then declare constexpr Symbols to get everything the actual Symbol
//   type we want.  Symbols must be constexpr to be valid to be "case"ed on.
//   因此，在使用枚举生成整数序列后，我们声明 constexpr 符号来获取我们想要的实际 Symbol 类型。
//   符号必须是 constexpr 类型，才能有效地用于 "case"。

using unique_t = uint32_t;
// 定义别名 unique_t 为 uint32_t 类型，用于表示唯一标识符

const std::string& domain_prefix();
// 声明函数 domain_prefix()，返回一个常量引用 std::string 类型

// A Symbol is like an interned string, but with a little extra
// structure; it is namespaced via SymbolNamespace and the resulting
// intern pointers support efficient namespace testing.
// 符号类似于一个被国际化的字符串，但具有一些额外的结构；
// 它通过 SymbolNamespace 进行命名空间化，并且生成的国际化指针支持高效的命名空间测试。

// 命名空间 c10 的结束
}
// 定义一个名为 Symbol 的结构体
struct TORCH_API Symbol {
  // 显式默认构造函数，初始化 value 为 0
  explicit constexpr Symbol() : value(0) {};
  // 构造函数，接受一个 unique_t 类型参数 uniq，用于初始化 value
  explicit constexpr Symbol(unique_t uniq)
  : value(uniq) {}

  // 根据限定字符串 s 创建一个 Symbol 对象
  static Symbol fromQualString(const std::string & s);

  // 根据域名 d 和非限定字符串 s 创建一个 Symbol 对象
  static Symbol fromDomainAndUnqualString(const std::string & d, const std::string & s);

  // 构造各种命名空间字符串的构造函数声明，构造并尝试进行内部化
  // 注意：不要将此函数用于字符串字面值；对于 attr::foo 这种情况，应该在上面的内建列表中
  static Symbol attr(const std::string & s);
  static Symbol aten(const std::string & s);
  static Symbol cuda(const std::string & s);
  static Symbol onnx(const std::string & s);
  static Symbol prim(const std::string & s);
  static Symbol user(const std::string & s);
  static Symbol caffe2(const std::string & s);
  static Symbol dimname(const std::string & s);
  // TODO: eliminate me
  static Symbol scope(const std::string & s);

  // 判断当前 Symbol 是否属于 attr 命名空间
  bool is_attr() const;
  // 判断当前 Symbol 是否属于 aten 命名空间
  bool is_aten() const;
  // 判断当前 Symbol 是否属于 cuda 命名空间
  bool is_cuda() const;
  // 判断当前 Symbol 是否属于 prim 命名空间
  bool is_prim() const;
  // 判断当前 Symbol 是否属于 prims 命名空间
  bool is_prims() const;
  // 判断当前 Symbol 是否属于 nvprims 命名空间
  bool is_nvprims() const;
  // 判断当前 Symbol 是否属于 onnx 命名空间
  bool is_onnx() const;
  // 判断当前 Symbol 是否属于 user 命名空间
  bool is_user() const;
  // 判断当前 Symbol 是否属于 caffe2 命名空间
  bool is_caffe2() const;
  // 判断当前 Symbol 是否属于 dimname 命名空间
  bool is_dimname() const;

  // 将当前 Symbol 转换为 unique_t 类型，以便进行比较
  constexpr operator unique_t() const {
    return value;
  }

  // 返回当前 Symbol 的命名空间
  Symbol ns() const;

  // 获取当前 Symbol 的非限定字符串表示形式
  const char * toUnqualString() const;

  // 获取当前 Symbol 的限定字符串表示形式，如 "aten::mm"
  const char * toQualString() const;

  // 获取当前 Symbol 的显示字符串表示形式，与 toQualString 相同
  // 返回 const char* 是因为很多 printf 风格的宏使用它
  const char * toDisplayString() const;

  // 获取当前 Symbol 的域名字符串表示形式，如 "org.pytorch.aten"
  std::string domainString() const;

private:
  // 构造函数，用于构造指定命名空间 ns 和字符串 s 的 Symbol 对象
  explicit Symbol(Symbol ns, const std::string & s);
  // 存储 Symbol 的唯一标识符
  unique_t value;
};

// 定义 Symbol 类的相等运算符重载
static inline bool operator==(Symbol lhs, Symbol rhs) {
  return static_cast<unique_t>(lhs) == static_cast<unique_t>(rhs);
}

// 定义 Symbol 类的 attr 命名空间构造函数的内联实现
inline Symbol Symbol::attr(const std::string & s) { return Symbol::fromQualString("attr::" + s); }
// 定义 Symbol 类的 aten 命名空间构造函数的内联实现
inline Symbol Symbol::aten(const std::string & s)  { return Symbol::fromQualString("aten::" + s); }
// 定义 Symbol 类的 cuda 命名空间构造函数的内联实现
inline Symbol Symbol::cuda(const std::string & s)  { return Symbol::fromQualString("cuda::" + s); }
// 将字符串转换为以 "onnx::" 开头的符号对象并返回
inline Symbol Symbol::onnx(const std::string & s)  { return Symbol::fromQualString("onnx::" + s); }
// 将字符串转换为以 "prim::" 开头的符号对象并返回
inline Symbol Symbol::prim(const std::string & s)  { return Symbol::fromQualString("prim::" + s); }
// 将字符串转换为以 "scope::" 开头的符号对象并返回
inline Symbol Symbol::scope(const std::string & s) { return Symbol::fromQualString("scope::" + s); }
// 将字符串转换为以 "user::" 开头的符号对象并返回
inline Symbol Symbol::user(const std::string & s) { return Symbol::fromQualString("user::" + s); }
// 将字符串转换为以 "_caffe2::" 开头的符号对象并返回
inline Symbol Symbol::caffe2(const std::string & s) { return Symbol::fromQualString("_caffe2::" + s); }
// 将字符串转换为以 "dimname::" 开头的符号对象并返回
inline Symbol Symbol::dimname(const std::string & s) { return Symbol::fromQualString("dimname::" + s); }

} // namespace c10

// 使符号对象在哈希表中表现得像整数一样
namespace std {
// 重载哈希结构体，计算给定符号的哈希值
template <>
struct hash<c10::Symbol> {
  size_t operator()(c10::Symbol s) const {
    // 调用标准库的哈希函数，将符号转换为其底层整数类型再哈希
    return std::hash<uint32_t>()(static_cast<uint32_t>(s));
  }
};
}
```