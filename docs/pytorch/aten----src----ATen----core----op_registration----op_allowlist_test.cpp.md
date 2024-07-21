# `.\pytorch\aten\src\ATen\core\op_registration\op_allowlist_test.cpp`

```
namespace allowlist_contains_test {
  // 静态断言，检查 allowlist_contains 函数对空字符串的处理
  static_assert(c10::impl::allowlist_contains("", ""), "");

  // 静态断言，检查 allowlist_contains 函数对包含非空字符串的情况的处理
  static_assert(!c10::impl::allowlist_contains("", "a"), "");

  // 静态断言，检查 allowlist_contains 函数对包含非空字符串的情况的处理
  static_assert(!c10::impl::allowlist_contains("a", ""), "");

  // 静态断言，检查 allowlist_contains 函数对空字符串的处理
  static_assert(!c10::impl::allowlist_contains("a;bc", ""), "");

  // 静态断言，检查 allowlist_contains 函数能否正确识别包含的字符串
  static_assert(c10::impl::allowlist_contains("a;bc;d", "a"), "");

  // 静态断言，检查 allowlist_contains 函数能否正确识别包含的字符串
  static_assert(c10::impl::allowlist_contains("a;bc;d", "bc"), "");

  // 静态断言，检查 allowlist_contains 函数能否正确识别包含的字符串
  static_assert(c10::impl::allowlist_contains("a;bc;d", "d"), "");

  // 静态断言，检查 allowlist_contains 函数对未包含的字符串的处理
  static_assert(!c10::impl::allowlist_contains("a;bc;d", "e"), "");

  // 静态断言，检查 allowlist_contains 函数对空字符串的处理
  static_assert(!c10::impl::allowlist_contains("a;bc;d", ""), "");

  // 静态断言，检查 allowlist_contains 函数对只包含分隔符的处理
  static_assert(c10::impl::allowlist_contains(";", ""), "");

  // 静态断言，检查 allowlist_contains 函数能否正确识别包含的字符串
  static_assert(c10::impl::allowlist_contains("a;", ""), "");

  // 静态断言，检查 allowlist_contains 函数能否正确识别包含的字符串
  static_assert(c10::impl::allowlist_contains("a;", "a"), "");
}
```