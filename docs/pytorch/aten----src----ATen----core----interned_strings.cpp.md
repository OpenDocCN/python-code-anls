# `.\pytorch\aten\src\ATen\core\interned_strings.cpp`

```py
// aten_interned_strings.h 包含所有操作符的名称
#undef TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/interned_strings.h>  // 引入 ATen 内部字符串相关头文件
#include <cstdint>  // 引入 C++ 标准整数类型
#include <cstring>  // 引入 C 字符串操作相关函数
#include <mutex>  // 引入互斥锁相关功能
#include <sstream>  // 引入字符串流功能
#include <string>  // 引入 C++ 标准字符串类型
#include <c10/util/Exception.h>  // 引入 C10 异常处理相关头文件
#include <ATen/core/interned_strings_class.h>  // 引入 ATen 内部字符串类相关头文件

namespace c10 {

// 返回静态字符串 "org.pytorch."
const std::string& domain_prefix() {
  static const std::string _domain_prefix = "org.pytorch.";
  return _domain_prefix;
}

// 返回字符串对应的符号(Symbol)，使用互斥锁保护
Symbol InternedStrings::symbol(const std::string& s) {
  std::lock_guard<std::mutex> guard(mutex_);
  return _symbol(s);  // 调用内部符号获取函数
}

// 根据符号获取字符串对，特定情况下绕过锁，直接获取字符串
std::pair<const char*, const char*> InternedStrings::string(Symbol sym) {
  // 在 C10_MOBILE 宏定义情况下，使用自定义字符串获取函数
#if defined C10_MOBILE
  return customString(sym);
#else
  // 根据符号枚举返回相应的命名空间和字符串名
  switch (sym) {
#define DEFINE_CASE(ns, s) \
  case static_cast<unique_t>(ns::s): \
    return {#ns "::" #s, #s};  // 返回命名空间::字符串名
    FORALL_NS_SYMBOLS(DEFINE_CASE)  // 对所有命名空间符号执行宏定义的操作
#undef DEFINE_CASE
    default:
      return customString(sym);  // 返回自定义字符串
  }
#endif
}

// 根据符号获取其命名空间，特定情况下使用锁保护
Symbol InternedStrings::ns(Symbol sym) {
#if defined C10_MOBILE
  std::lock_guard<std::mutex> guard(mutex_);
  return sym_to_info_.at(sym).ns;  // 返回符号对应的命名空间
#else
  switch (sym) {
#define DEFINE_CASE(ns, s) \
  case static_cast<unique_t>(ns::s): \
    return namespaces::ns;  // 返回命名空间 ns
    // NOLINTNEXTLINE(bugprone-branch-clone)
    FORALL_NS_SYMBOLS(DEFINE_CASE)  // 对所有命名空间符号执行宏定义的操作
#undef DEFINE_CASE
    default: {
      std::lock_guard<std::mutex> guard(mutex_);
      return sym_to_info_.at(sym).ns;  // 返回符号对应的命名空间
    }
  }
#endif
}

// 根据字符串获取符号(Symbol)，使用锁保护
Symbol InternedStrings::_symbol(const std::string& s) {
  auto it = string_to_sym_.find(s);  // 查找字符串对应的符号
  if (it != string_to_sym_.end())
    return it->second;  // 如果找到直接返回

  auto pos = s.find("::");  // 查找命名空间分隔符 "::"
  if (pos == std::string::npos) {
    std::stringstream ss;
    ss << "all symbols must have a namespace, <namespace>::<string>, but found: " << s;
    throw std::runtime_error(ss.str());  // 抛出运行时错误，要求所有符号必须有命名空间
  }
  // 递归获取命名空间对应的符号
  Symbol ns = _symbol("namespaces::" + s.substr(0, pos));

  Symbol sym(sym_to_info_.size());  // 创建新符号
  string_to_sym_[s] = sym;  // 记录字符串到符号的映射
  sym_to_info_.push_back({ns, s, s.substr(pos + strlen("::"))});  // 记录符号信息
  return sym;  // 返回新创建的符号
}

// 根据符号获取自定义字符串对，使用锁保护
std::pair<const char*, const char*> InternedStrings::customString(Symbol sym) {
  std::lock_guard<std::mutex> guard(mutex_);
  SymbolInfo& s = sym_to_info_.at(sym);  // 获取符号对应的信息
  return {s.qual_name.c_str(), s.unqual_name.c_str()};  // 返回符号的限定名和非限定名
}

// 全局唯一的 InternedStrings 对象获取函数
static InternedStrings & globalStrings() {
  static InternedStrings s;
  return s;
}

// 根据限定字符串创建符号(Symbol)
Symbol Symbol::fromQualString(const std::string & s) {
  return globalStrings().symbol(s);  // 返回全局唯一 InternedStrings 对象中的符号
}

// 获取符号的非限定字符串
const char * Symbol::toUnqualString() const {
  return globalStrings().string(*this).second;  // 返回全局唯一 InternedStrings 对象中符号的非限定字符串
}

// 获取符号的限定字符串
const char * Symbol::toQualString() const {
  return globalStrings().string(*this).first;  // 返回全局唯一 InternedStrings 对象中符号的限定字符串
}

}  // 命名空间 c10
const char * Symbol::toDisplayString() const {
    // TODO: 实际上应该返回一些"用户友好"的内容。
    // 问题在于，为了在 printf 风格的断言语句中使用，这必须返回一个 const char*（其生命周期是全局的），
    // 因此我们无法实时组装一个字符串。
    // 目前该函数仅返回 toQualString() 的结果。
    return toQualString();
}

Symbol Symbol::ns() const {
    // 调用 globalStrings() 的 ns() 方法，返回一个 Symbol 对象
    return globalStrings().ns(*this);
}

std::string Symbol::domainString() const {
    // 返回 domain_prefix() 和 ns().toUnqualString() 组合而成的字符串
    return domain_prefix() + ns().toUnqualString();
}

Symbol Symbol::fromDomainAndUnqualString(const std::string & d, const std::string & s) {
    // 检查给定的 domain 字符串是否以 domain_prefix() 开头，如果不是则抛出异常
    if (d.compare(0, domain_prefix().size(), domain_prefix()) != 0) {
        std::ostringstream ss;
        ss << "Symbol: domain string is expected to be prefixed with '"
           << domain_prefix() << "', e.g. 'org.pytorch.aten'";
        throw std::runtime_error(ss.str());
    }
    // 将 d 去除 domain_prefix() 后与 s 组合成一个限定符字符串，然后调用 fromQualString() 返回对应的 Symbol 对象
    std::string qualString = d.substr(domain_prefix().size()) + "::" + s;
    return fromQualString(qualString);
}

bool Symbol::is_attr() const { return ns() == namespaces::attr; }
bool Symbol::is_aten() const { return ns() == namespaces::aten; }
bool Symbol::is_cuda() const { return ns() == namespaces::cuda; }
bool Symbol::is_prim() const { return ns() == namespaces::prim; }
bool Symbol::is_prims() const { return ns() == namespaces::prims; }
bool Symbol::is_nvprims() const { return ns() == namespaces::nvprims; }
bool Symbol::is_onnx() const { return ns() == namespaces::onnx; }
bool Symbol::is_user() const { return ns() == namespaces::user; }
bool Symbol::is_caffe2() const { return ns() == namespaces::_caffe2; }
bool Symbol::is_dimname() const { return ns() == namespaces::dimname; }
```