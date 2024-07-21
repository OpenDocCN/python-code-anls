# `.\pytorch\aten\src\ATen\test\Dimname_test.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <ATen/Dimname.h>  // 引入 ATen 库中的 Dimname 头文件
#include <c10/util/Exception.h>  // 引入 c10 库中的 Exception 头文件
#include <c10/util/Optional.h>  // 引入 c10 库中的 Optional 头文件

using at::NameType;  // 使用 ATen 库中的 NameType 命名空间
using at::Symbol;  // 使用 ATen 库中的 Symbol 命名空间
using at::Dimname;  // 使用 ATen 库中的 Dimname 命名空间

TEST(DimnameTest, isValidIdentifier) {
  ASSERT_TRUE(Dimname::isValidName("a"));  // 断言检查 "a" 是否是有效的 Dimname
  ASSERT_TRUE(Dimname::isValidName("batch"));  // 断言检查 "batch" 是否是有效的 Dimname
  ASSERT_TRUE(Dimname::isValidName("N"));  // 断言检查 "N" 是否是有效的 Dimname
  ASSERT_TRUE(Dimname::isValidName("CHANNELS"));  // 断言检查 "CHANNELS" 是否是有效的 Dimname
  ASSERT_TRUE(Dimname::isValidName("foo_bar_baz"));  // 断言检查 "foo_bar_baz" 是否是有效的 Dimname
  ASSERT_TRUE(Dimname::isValidName("batch1"));  // 断言检查 "batch1" 是否是有效的 Dimname
  ASSERT_TRUE(Dimname::isValidName("batch_9"));  // 断言检查 "batch_9" 是否是有效的 Dimname
  ASSERT_TRUE(Dimname::isValidName("_"));  // 断言检查 "_" 是否是有效的 Dimname
  ASSERT_TRUE(Dimname::isValidName("_1"));  // 断言检查 "_1" 是否是有效的 Dimname

  ASSERT_FALSE(Dimname::isValidName(""));  // 断言检查空字符串是否是有效的 Dimname（预期为无效）
  ASSERT_FALSE(Dimname::isValidName(" "));  // 断言检查空格是否是有效的 Dimname（预期为无效）
  ASSERT_FALSE(Dimname::isValidName(" a "));  // 断言检查包含空格的字符串是否是有效的 Dimname（预期为无效）
  ASSERT_FALSE(Dimname::isValidName("1batch"));  // 断言检查以数字开头的字符串是否是有效的 Dimname（预期为无效）
  ASSERT_FALSE(Dimname::isValidName("?"));  // 断言检查包含特殊字符的字符串是否是有效的 Dimname（预期为无效）
  ASSERT_FALSE(Dimname::isValidName("-"));  // 断言检查包含破折号的字符串是否是有效的 Dimname（预期为无效）
  ASSERT_FALSE(Dimname::isValidName("1"));  // 断言检查单独的数字是否是有效的 Dimname（预期为无效）
  ASSERT_FALSE(Dimname::isValidName("01"));  // 断言检查以零开头的数字字符串是否是有效的 Dimname（预期为无效）
}

TEST(DimnameTest, wildcardName) {
  Dimname wildcard = Dimname::wildcard();  // 创建一个通配符 Dimname
  ASSERT_EQ(wildcard.type(), NameType::WILDCARD);  // 断言检查通配符 Dimname 的类型是否为 WILDCARD
  ASSERT_EQ(wildcard.symbol(), Symbol::dimname("*"));  // 断言检查通配符 Dimname 的符号是否为 "*"
}

TEST(DimnameTest, createNormalName) {
  auto foo = Symbol::dimname("foo");  // 创建一个名为 "foo" 的 Symbol
  auto dimname = Dimname::fromSymbol(foo);  // 从 Symbol 创建 Dimname 对象
  ASSERT_EQ(dimname.type(), NameType::BASIC);  // 断言检查 Dimname 的类型是否为 BASIC
  ASSERT_EQ(dimname.symbol(), foo);  // 断言检查 Dimname 的符号是否与创建的 Symbol 相同

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(Dimname::fromSymbol(Symbol::dimname("inva.lid")), c10::Error);  // 断言检查传入无效符号会抛出异常
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(Dimname::fromSymbol(Symbol::dimname("1invalid")), c10::Error);  // 断言检查传入无效符号会抛出异常
}

static void check_unify_and_match(
    const std::string& dimname,
    const std::string& other,
    at::optional<const std::string> expected) {
  auto dimname1 = Dimname::fromSymbol(Symbol::dimname(dimname));  // 根据字符串创建 Dimname 对象
  auto dimname2 = Dimname::fromSymbol(Symbol::dimname(other));  // 根据字符串创建另一个 Dimname 对象
  auto result = dimname1.unify(dimname2);  // 尝试统一两个 Dimname 对象

  if (expected) {
    auto expected_result = Dimname::fromSymbol(Symbol::dimname(*expected));  // 根据预期字符串创建 Dimname 对象
    ASSERT_EQ(result->symbol(), expected_result.symbol());  // 断言检查结果的符号是否与预期符号相同
    ASSERT_EQ(result->type(), expected_result.type());  // 断言检查结果的类型是否与预期类型相同
    ASSERT_TRUE(dimname1.matches(dimname2));  // 断言检查两个 Dimname 对象是否匹配
  } else {
    ASSERT_FALSE(result);  // 断言检查结果应为空指针（无法统一的情况）
    ASSERT_FALSE(dimname1.matches(dimname2));  // 断言检查两个 Dimname 对象是否不匹配
  }
}

TEST(DimnameTest, unifyAndMatch) {
  check_unify_and_match("a", "a", "a");  // 测试两个相同的 Dimname 字符串是否能统一
  check_unify_and_match("a", "*", "a");  // 测试一个具体 Dimname 和通配符是否能统一
  check_unify_and_match("*", "a", "a");  // 测试一个通配符和一个具体 Dimname 是否能统一
  check_unify_and_match("*", "*", "*");  // 测试两个通配符是否能统一
  check_unify_and_match("a", "b", c10::nullopt);  // 测试两个不同的具体 Dimname 是否不能统一
}
```