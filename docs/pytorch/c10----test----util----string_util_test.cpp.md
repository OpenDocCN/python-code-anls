# `.\pytorch\c10\test\util\string_util_test.cpp`

```py
// 包含 C10 库中的 StringUtil.h 头文件
#include <c10/util/StringUtil.h>

// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 匿名命名空间，用于定义测试用例
namespace {

// 嵌套命名空间 test_str_narrow_single，测试 StringUtil::str 方法对窄字符的处理
namespace test_str_narrow_single {
// 定义测试用例 StringUtilTest.testStrNarrowSingle
TEST(StringUtilTest, testStrNarrowSingle) {
  // 初始化字符串 s
  std::string s = "narrow test string";
  // 断言 StringUtil::str(s) 返回的结果与 s 相等
  EXPECT_EQ(s, c10::str(s));

  // 获取 s 的 C 风格字符串指针
  const char* c_str = s.c_str();
  // 断言 StringUtil::str(c_str) 返回的结果与 s 相等
  EXPECT_EQ(s, c10::str(c_str));

  // 初始化字符 c
  char c = 'a';
  // 断言 StringUtil::str(c) 返回的结果与 std::string(1, c) 相等
  EXPECT_EQ(std::string(1, c), c10::str(c));
}
} // namespace test_str_narrow_single

// 嵌套命名空间 test_str_wide_single，测试 StringUtil::str 方法对宽字符的处理
namespace test_str_wide_single {
// 定义测试用例 StringUtilTest.testStrWideSingle
TEST(StringUtilTest, testStrWideSingle) {
  // 初始化宽字符串 s
  std::wstring s = L"wide test string";
  // 初始化窄字符串 narrow
  std::string narrow = "wide test string";
  // 断言 StringUtil::str(s) 返回的结果与 narrow 相等
  EXPECT_EQ(narrow, c10::str(s));

  // 获取 s 的 C 风格宽字符串指针
  const wchar_t* c_str = s.c_str();
  // 断言 StringUtil::str(c_str) 返回的结果与 narrow 相等
  EXPECT_EQ(narrow, c10::str(c_str));

  // 初始化宽字符 c
  wchar_t c = L'a';
  // 初始化窄字符 narrowC
  std::string narrowC = "a";
  // 断言 StringUtil::str(c) 返回的结果与 narrowC 相等
  EXPECT_EQ(narrowC, c10::str(c));
}
} // namespace test_str_wide_single

// 嵌套命名空间 test_str_wide_single_multibyte，测试 StringUtil::str 方法对多字节宽字符的处理
namespace test_str_wide_single_multibyte {
// 定义测试用例 StringUtilTest.testStrWideSingleMultibyte
TEST(StringUtilTest, testStrWideSingleMultibyte) {
  // 初始化宽字符串 s，包含多字节字符
  std::wstring s = L"\u00EC blah";
  // 初始化窄字符串 narrow，对应多字节字符的 UTF-8 表示
  std::string narrow = "\xC3\xAC blah";
  // 断言 StringUtil::str(s) 返回的结果与 narrow 相等
  EXPECT_EQ(narrow, c10::str(s));

  // 获取 s 的 C 风格宽字符串指针
  const wchar_t* c_str = s.c_str();
  // 断言 StringUtil::str(c_str) 返回的结果与 narrow 相等
  EXPECT_EQ(narrow, c10::str(c_str));

  // 初始化宽字符 c，为多字节字符的宽字符表示
  wchar_t c = L'\u00EC';
  // 初始化窄字符 narrowC，为多字节字符的 UTF-8 表示
  std::string narrowC = "\xC3\xAC";
  // 断言 StringUtil::str(c) 返回的结果与 narrowC 相等
  EXPECT_EQ(narrowC, c10::str(c));
}
} // namespace test_str_wide_single_multibyte

// 嵌套命名空间 test_str_wide_empty，测试 StringUtil::str 方法对空宽字符串的处理
namespace test_str_wide_empty {
// 定义测试用例 StringUtilTest.testStrWideEmpty
TEST(StringUtilTest, testStrWideEmpty) {
  // 初始化空宽字符串 s
  std::wstring s = L"";
  // 初始化空窄字符串 narrow
  std::string narrow = "";
  // 断言 StringUtil::str(s) 返回的结果与 narrow 相等
  EXPECT_EQ(narrow, c10::str(s));

  // 获取 s 的 C 风格宽字符串指针
  const wchar_t* c_str = s.c_str();
  // 断言 StringUtil::str(c_str) 返回的结果与 narrow 相等
  EXPECT_EQ(narrow, c10::str(c_str));

  // 初始化宽字符 c，为 null 字符
  wchar_t c = L'\0';
  // 初始化窄字符 narrowC，为单个 null 字符的字符串表示
  std::string narrowC(1, '\0');
  // 断言 StringUtil::str(c) 返回的结果与 narrowC 相等
  EXPECT_EQ(narrowC, c10::str(c));
}
} // namespace test_str_wide_empty

// 嵌套命名空间 test_str_multi，测试 StringUtil::str 方法对多参数组合的处理
namespace test_str_multi {
// 定义测试用例 StringUtilTest.testStrMulti
TEST(StringUtilTest, testStrMulti) {
  // 调用 StringUtil::str 方法，传入多种参数类型
  std::string result = c10::str(
      "c_str ",              // C 风格字符串
      'c',                   // 字符
      std::string(" std::string "),  // 标准字符串
      42,                    // 整数
      L" wide c_str ",       // 宽字符 C 风格字符串
      L'w',                  // 宽字符
      std::wstring(L" std::wstring "));  // 宽字符标准字符串
  // 预期的结果字符串
  std::string expected = "c_str c std::string 42 wide c_str w std::wstring ";
  // 断言 StringUtil::str 的结果与预期结果相等
  EXPECT_EQ(expected, result);
}
} // namespace test_str_multi

} // namespace
```