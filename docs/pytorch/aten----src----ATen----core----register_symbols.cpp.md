# `.\pytorch\aten\src\ATen\core\register_symbols.cpp`

```
// aten_interned_strings.h 包含所有运算符的名称
#undef TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/interned_strings.h>
#include <ATen/core/interned_strings_class.h>

namespace c10 {

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-const-or-ref-data-members)
// 定义一个结构体 Entry，包含命名空间、未限定名、符号、命名空间符号
struct Entry {
  const char* const namespace_;  // 命名空间的常量字符指针
  const char* const unqual_name; // 未限定名的常量字符指针
  const Symbol sym;              // 符号
  const Symbol ns_sym;           // 命名空间符号
};
// NOLINTEND(cppcoreguidelines-avoid-const-or-ref-data-members)

// 计算一个 Entry 对象的限定名
std::string qual_name_for_entry(const Entry& entry) {
  const char* const sep = "::";  // 命名空间和未限定名之间的分隔符
  const auto namespace_len = strlen(entry.namespace_);  // 命名空间的长度
  const auto sep_len = strlen(sep);                    // 分隔符的长度
  const auto unqual_name_len = strlen(entry.unqual_name);  // 未限定名的长度
  std::string s;
  s.reserve(namespace_len + sep_len + unqual_name_len);  // 预留字符串长度
  s.append(entry.namespace_, namespace_len);             // 添加命名空间部分
  s.append(sep, sep_len);                               // 添加分隔符
  s.append(entry.unqual_name, unqual_name_len);         // 添加未限定名部分
  return s;                                             // 返回完整的限定名字符串
}

// 注意：通过以下方式，我们可以进一步节省空间：
// constexpr char namespaces[] = "namespaces\0prim\0aten\0...";
// constexpr char unqual_names[] = "prim\0aten\0cuda\0...";
// 然后在 Entry 中存储两个 uint16_t（或需要时 uint32_t）的偏移量，
// 指向原始字符串表中的位置，而不是使用 8 字节的指针。
// 我没有实现这种方式，因为在 C++14 中如何在编译时对命名空间数组进行去重不太清楚，
// 特别是在没有代码生成的情况下，但如果我们转向代码生成，这将是直接的。
// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
// 声明一个 constexpr 的 Entry 数组，包含所有命名空间的符号定义
constexpr Entry entries[] = {
#define SYMBOL_ENTRY(n, s) {#n, #s, n::s, namespaces::n},
    FORALL_NS_SYMBOLS(SYMBOL_ENTRY)  // 展开宏 FORALL_NS_SYMBOLS，生成符号条目
#undef SYMBOL_ENTRY
};

} // namespace

// InternedStrings 类的构造函数
InternedStrings::InternedStrings()
    : sym_to_info_(static_cast<size_t>(_keys::num_symbols)) {
  // 替代循环的方式是，将赋值直接扩展到 FORALL_NS_SYMBOLS 中，
  // 但这会创建一个庞大的函数（由于所有的 std::string 构造函数和操作符[]），
  // 编译优化需要几分钟时间。使用静态的 constexpr 可构造结构体的 C 数组，
  // 则编译时间几乎为零。
  for (const auto& entry : entries) {
    auto qual_name = qual_name_for_entry(entry);  // 获取条目的限定名
    string_to_sym_[qual_name] = entry.sym;       // 将限定名映射到符号
    sym_to_info_[entry.sym] = {                  // 将符号映射到信息结构
        entry.ns_sym, std::move(qual_name), entry.unqual_name};
  }
}

} // namespace c10
```