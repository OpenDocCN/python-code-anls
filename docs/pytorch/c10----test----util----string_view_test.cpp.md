# `.\pytorch\c10\test\util\string_view_test.cpp`

```
// 包含头文件 c10/util/string_view.h，该头文件定义了 string_view 类型
#include <c10/util/string_view.h>

// 包含 Google Mock 框架的头文件
#include <gmock/gmock.h>

// NOLINTBEGIN(modernize*, readability*, bugprone-string-constructor)
// 使用 NOLINTBEGIN 指令禁用特定的代码检查规则，以免产生对现代化、可读性和字符串构造的建议

// 使用 c10 命名空间中的 string_view 类
using c10::string_view;

// 声明测试工具的命名空间
namespace {
namespace testutils {

// 递归函数，比较两个字符串是否相等
constexpr bool string_equal(const char* lhs, const char* rhs, size_t size) {
  return (size == 0)   ? true
      : (*lhs != *rhs) ? false
                       : string_equal(lhs + 1, rhs + 1, size - 1);
}

// 静态断言，验证字符串相等性比较函数的正确性
static_assert(string_equal("hi", "hi", 2), "");
static_assert(string_equal("", "", 0), "");
static_assert(string_equal("hi", "hi2", 2), "");
static_assert(string_equal("hi2", "hi", 2), "");
static_assert(!string_equal("hi", "hi2", 3), "");
static_assert(!string_equal("hi2", "hi", 3), "");
static_assert(!string_equal("hi", "ha", 2), "");

// 模板函数，用于验证 Functor 是否会抛出 Exception 异常，且异常消息包含期望的字符串
template <class Exception, class Functor>
inline void expectThrows(Functor&& functor, const char* expectMessageContains) {
  try {
    std::forward<Functor>(functor)();
  } catch (const Exception& e) {
    // 使用 Google Mock 断言验证异常消息是否包含期望的子串
    EXPECT_THAT(e.what(), testing::HasSubstr(expectMessageContains));
    return;
  }
  // 如果未抛出异常，则测试失败
  ADD_FAILURE() << "Expected to throw exception containing \""
                << expectMessageContains << "\" but didn't throw";
}
} // namespace testutils

// 使用 testutils 命名空间中的函数和类
using testutils::expectThrows;
using testutils::string_equal;

// 测试 typedefs 命名空间，验证 string_view 的类型定义是否正确
namespace test_typedefs {
static_assert(std::is_same<char, string_view::value_type>::value, "");
static_assert(std::is_same<char*, string_view::pointer>::value, "");
static_assert(std::is_same<const char*, string_view::const_pointer>::value, "");
static_assert(std::is_same<char&, string_view::reference>::value, "");
static_assert(
    std::is_same<const char&, string_view::const_reference>::value,
    "");
static_assert(std::is_same<std::size_t, string_view::size_type>::value, "");
static_assert(
    std::is_same<std::ptrdiff_t, string_view::difference_type>::value,
    "");
} // namespace test_typedefs

// 测试默认构造函数命名空间，验证 string_view 的默认构造行为
namespace test_default_constructor {
static_assert(string_view().empty());
static_assert(string_view().data() == nullptr, "");
static_assert(string_view() == string_view(""));
} // namespace test_default_constructor

// 测试 const char* 构造函数命名空间，验证 string_view 对 const char* 的构造行为
namespace test_constchar_constructor {
static_assert(string_view("").size() == 0, "");
constexpr string_view hello = "hello";
static_assert(5 == hello.size(), "");
static_assert(string_equal("hello", hello.data(), hello.size()), "");
} // namespace test_constchar_constructor

// 测试带长度的构造函数命名空间，验证 string_view 对指定长度的构造行为
namespace test_sized_constructor {
static_assert(string_view("", 0).size() == 0, "");
constexpr string_view hell("hello", 4);
static_assert(4 == hell.size(), "");
static_assert(string_equal("hell", hell.data(), hell.size()), "");
} // namespace test_sized_constructor

// 测试 string 转换构造函数命名空间，验证 string_view 对 std::string 的隐式转换行为
namespace test_string_constructor {
void test_conversion_is_implicit(string_view a) {}
namespace test_reverse_iteration {
// 创建一个 constexpr 的 string_view，内容为 "hello"
constexpr string_view hello = "hello";

// 静态断言，验证字符串长度为 5
static_assert('h' == *(hello.begin() + 0), "");  // 验证第一个字符是 'h'
static_assert('e' == *(hello.begin() + 1), "");  // 验证第二个字符是 'e'
static_assert('l' == *(hello.begin() + 2), "");  // 验证第三个字符是 'l'
static_assert('l' == *(hello.begin() + 3), "");  // 验证第四个字符是 'l'
static_assert('o' == *(hello.begin() + 4), "");  // 验证第五个字符是 'o'
static_assert(hello.end() == hello.begin() + 5, "");  // 验证末尾迭代器位置正确
}
// 静态断言，检查反向迭代的结果是否符合预期：最后一个字符为'o'
static_assert('o' == *(hello.rbegin() + 0), "");

// 静态断言，检查反向迭代的结果是否符合预期：倒数第二个字符为'l'
static_assert('l' == *(hello.rbegin() + 1), "");

// 静态断言，检查反向迭代的结果是否符合预期：倒数第三个字符为'l'
static_assert('l' == *(hello.rbegin() + 2), "");

// 静态断言，检查反向迭代的结果是否符合预期：倒数第四个字符为'e'
static_assert('e' == *(hello.rbegin() + 3), "");

// 静态断言，检查反向迭代的结果是否符合预期：倒数第五个字符为'h'
static_assert('h' == *(hello.rbegin() + 4), "");

// 静态断言，验证反向迭代的结束迭代器与起始迭代器相加后是否为相同位置，即确保反向迭代的完整性
static_assert(hello.rend() == hello.rbegin() + 5, "");
} // namespace test_reverse_iteration

namespace test_random_access {
// constexpr 字符串视图，检查索引访问操作是否正确
constexpr string_view hello = "hello";
static_assert('h' == hello[0], "");
static_assert('e' == hello[1], "");
static_assert('l' == hello[2], "");
static_assert('l' == hello[3], "");
static_assert('o' == hello[4], "");

// 静态断言，使用 at() 方法检查索引访问是否正确
static_assert('h' == hello.at(0), "");
static_assert('e' == hello.at(1), "");
static_assert('l' == hello.at(2), "");
static_assert('l' == hello.at(3), "");
static_assert('o' == hello.at(4), "");

// 测试异常情况下的 at() 方法，确保超出范围时抛出 std::out_of_range 异常
TEST(StringViewTest, whenCallingAccessOperatorOutOfRange_thenThrows) {
  expectThrows<std::out_of_range>(
      [] { string_view("").at(1); },
      "string_view::operator[] or string_view::at() out of range. Index: 1, size: 0");

  expectThrows<std::out_of_range>(
      [] { string_view("hello").at(5); },
      "string_view::operator[] or string_view::at() out of range. Index: 5, size: 5");

  expectThrows<std::out_of_range>(
      [] { string_view("hello").at(100); },
      "string_view::operator[] or string_view::at() out of range. Index: 100, size: 5");

  expectThrows<std::out_of_range>(
      [] { string_view("hello").at(string_view::npos); },
      "string_view::operator[] or string_view::at() out of range. Index: 18446744073709551615, size: 5");
}
} // namespace test_random_access

namespace test_front_back {
// 静态断言，检查 front() 方法返回的第一个字符是否为 'h'
static_assert('h' == string_view("hello").front(), "");

// 静态断言，检查 back() 方法返回的最后一个字符是否为 'o'
static_assert('o' == string_view("hello").back(), "");
} // namespace test_front_back

namespace test_data {
// 静态断言，验证 data() 方法返回的数据与原字符串是否相等
static_assert(string_equal("hello", string_view("hello").data(), 5), "");
} // namespace test_data

namespace test_size_length {
// 静态断言，验证 size() 方法返回空字符串的大小是否为 0
static_assert(0 == string_view("").size(), "");

// 静态断言，验证 size() 方法返回 "hello" 字符串的大小是否为 5
static_assert(5 == string_view("hello").size(), "");

// 静态断言，验证 length() 方法返回空字符串的长度是否为 0
static_assert(0 == string_view("").length(), "");

// 静态断言，验证 length() 方法返回 "hello" 字符串的长度是否为 5
static_assert(5 == string_view("hello").length(), "");
} // namespace test_size_length

namespace test_empty {
// 静态断言，验证空字符串视图是否为空
static_assert(string_view().empty(), "");

// 静态断言，验证空字符串的字符串视图是否为空
static_assert(string_view("").empty(), "");

// 静态断言，验证 "hello" 字符串视图不为空
static_assert(!string_view("hello").empty(), "");
} // namespace test_empty

namespace test_remove_prefix {
// constexpr 函数，移除字符串视图的前缀，返回修改后的字符串视图
constexpr string_view remove_prefix(string_view input, size_t len) {
  input.remove_prefix(len);
  return input;
}
TEST(StringViewTest, whenRemovingValidPrefix_thenWorks) {
  // 静态断言：移除长度为0的前缀，结果应与原字符串相同
  static_assert(
      remove_prefix(string_view("hello"), 0) == string_view("hello"), "");
  // 静态断言：移除长度为1的前缀，结果应为"ello"
  static_assert(
      remove_prefix(string_view("hello"), 1) == string_view("ello"), "");
  // 静态断言：移除长度为5的前缀，结果应为空字符串
  static_assert(remove_prefix(string_view("hello"), 5) == string_view(""), "");

  // 期望断言：移除长度为0的前缀，结果应与原字符串相同
  EXPECT_EQ(remove_prefix(string_view("hello"), 0), string_view("hello"));
  // 期望断言：移除长度为1的前缀，结果应为"ello"
  EXPECT_EQ(remove_prefix(string_view("hello"), 1), string_view("ello"));
  // 期望断言：移除长度为5的前缀，结果应为空字符串
  EXPECT_EQ(remove_prefix(string_view("hello"), 5), string_view(""));
}

TEST(StringViewTest, whenRemovingTooLargePrefix_thenThrows) {
  // 期望抛出异常，尝试移除超出字符串长度的前缀
  expectThrows<std::out_of_range>(
      [] { remove_prefix(string_view("hello"), 6); },
      "basic_string_view::remove_prefix: out of range. PrefixLength: 6, size: 5");
}
} // namespace test_remove_prefix

namespace test_remove_suffix {
constexpr string_view remove_suffix(string_view input, size_t len) {
  // 移除输入字符串末尾指定长度的后缀
  input.remove_suffix(len);
  return input;
}

TEST(StringViewTest, whenRemovingValidSuffix_thenWorks) {
  // 静态断言：移除长度为0的后缀，结果应与原字符串相同
  static_assert(
      remove_suffix(string_view("hello"), 0) == string_view("hello"), "");
  // 静态断言：移除长度为1的后缀，结果应为"hell"
  static_assert(
      remove_suffix(string_view("hello"), 1) == string_view("hell"), "");
  // 静态断言：移除长度为5的后缀，结果应为空字符串
  static_assert(remove_suffix(string_view("hello"), 5) == string_view(""), "");

  // 期望断言：移除长度为0的后缀，结果应与原字符串相同
  EXPECT_EQ(remove_suffix(string_view("hello"), 0), string_view("hello"));
  // 期望断言：移除长度为1的后缀，结果应为"hell"
  EXPECT_EQ(remove_suffix(string_view("hello"), 1), string_view("hell"));
  // 期望断言：移除长度为5的后缀，结果应为空字符串
  EXPECT_EQ(remove_suffix(string_view("hello"), 5), string_view(""));
}

TEST(StringViewTest, whenRemovingTooLargeSuffix_thenThrows) {
  // 期望抛出异常，尝试移除超出字符串长度的后缀
  expectThrows<std::out_of_range>(
      [] { remove_suffix(string_view("hello"), 6); },
      "basic_string_view::remove_suffix: out of range. SuffixLength: 6, size: 5");
}
} // namespace test_remove_suffix

namespace test_swap_function {
// 返回一对交换后的字符串视图
constexpr std::pair<string_view, string_view> get() {
  string_view first = "first";
  string_view second = "second";
  // 使用 swap 函数交换两个字符串视图
  swap(first, second);
  return std::make_pair(first, second);
}
TEST(StringViewTest, testSwapFunction) {
  // 静态断言：交换后第一个视图应为 "second"
  static_assert(string_view("second") == get().first, "");
  // 静态断言：交换后第二个视图应为 "first"
  static_assert(string_view("first") == get().second, "");

  // 期望断言：交换后第一个视图应为 "second"
  EXPECT_EQ(string_view("second"), get().first);
  // 期望断言：交换后第二个视图应为 "first"
  EXPECT_EQ(string_view("first"), get().second);
}
} // namespace test_swap_function

namespace test_swap_method {
// 返回一对交换后的字符串视图
constexpr std::pair<string_view, string_view> get() {
  string_view first = "first";
  string_view second = "second";
  // 使用 swap 方法交换两个字符串视图
  first.swap(second);
  return std::make_pair(first, second);
}
TEST(StringViewTest, testSwapMethod) {
  // 静态断言：交换后第一个视图应为 "second"
  static_assert(string_view("second") == get().first, "");
  // 静态断言：交换后第二个视图应为 "first"
  static_assert(string_view("first") == get().second, "");

  // 期望断言：交换后第一个视图应为 "second"
  EXPECT_EQ(string_view("second"), get().first);
  // 期望断言：交换后第二个视图应为 "first"
  EXPECT_EQ(string_view("first"), get().second);
}
} // namespace test_swap_method

namespace test_copy {
TEST(StringViewTest, whenCopyingFullStringView_thenDestinationHasCorrectData) {
  // 设置测试用的字符串视图为 "hello"
  string_view data = "hello";
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  // 定义一个长度为 5 的字符数组 result
  char result[5];
  // 将 data 中的内容复制到 result 中，最多复制 5 个字符
  size_t num_copied = data.copy(result, 5);
  // 验证复制的字符数是否为 5
  EXPECT_EQ(5, num_copied);
  // 验证 result 是否与 "hello" 相等，比较长度为 5
  EXPECT_TRUE(string_equal("hello", result, 5));
}

TEST(StringViewTest, whenCopyingSubstr_thenDestinationHasCorrectData) {
  // 设置测试用的字符串视图为 "hello"
  string_view data = "hello";
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  // 定义一个长度为 2 的字符数组 result
  char result[2];
  // 将 data 中从索引 2 开始的内容复制到 result 中，最多复制 2 个字符
  size_t num_copied = data.copy(result, 2, 2);
  // 验证复制的字符数是否为 2
  EXPECT_EQ(2, num_copied);
  // 验证 result 是否与 "ll" 相等，比较长度为 2
  EXPECT_TRUE(string_equal("ll", result, 2));
}

TEST(StringViewTest, whenCopyingTooMuch_thenJustCopiesLess) {
  // 设置测试用的字符串视图为 "hello"
  string_view data = "hello";
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  // 定义一个长度为 100 的字符数组 result
  char result[100];
  // 将 data 中从索引 2 开始的内容复制到 result 中，最多复制 100 个字符
  size_t num_copied = data.copy(result, 100, 2);
  // 验证实际复制的字符数是否为 3
  EXPECT_EQ(3, num_copied);
  // 验证 result 是否与 "llo" 相等，比较长度为 3
  EXPECT_TRUE(string_equal("llo", result, 3));
}

TEST(StringViewTest, whenCopyingJustAtRange_thenDoesntCrash) {
  // 设置测试用的字符串视图为 "hello"
  string_view data = "hello";
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  // 定义一个长度为 1 的字符数组 result
  char result[1];
  // 尝试将 data 中从索引 5 开始的内容复制到 result 中，最多复制 2 个字符
  size_t num_copied = data.copy(result, 2, 5);
  // 验证实际复制的字符数是否为 0
  EXPECT_EQ(0, num_copied);
}

TEST(StringViewTest, whenCopyingOutOfRange_thenThrows) {
  // 设置测试用的字符串视图为 "hello"
  string_view data = "hello";
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  // 定义一个长度为 2 的字符数组 result
  char result[2];
  // 预期会抛出 std::out_of_range 异常，因为尝试从索引 6 处开始复制，而字符串长度为 5
  expectThrows<std::out_of_range>(
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
      [&] { data.copy(result, 2, 6); },
      "basic_string_view::copy: out of range. Index: 6, size: 5");
}
} // namespace test_copy

namespace test_substr {
static_assert(string_view("").substr() == string_view(""), "");
static_assert(string_view("").substr(0) == string_view(""), "");
static_assert(string_view("").substr(0, 0) == string_view(""), "");

static_assert(string_view("hello").substr() == string_view("hello"), "");
static_assert(string_view("hello").substr(0) == string_view("hello"), "");
static_assert(string_view("hello").substr(1) == string_view("ello"), "");
static_assert(string_view("hello").substr(5) == string_view(""), "");

static_assert(string_view("hello").substr(0, 0) == string_view(""), "");
static_assert(string_view("hello").substr(0, 2) == string_view("he"), "");
static_assert(string_view("hello").substr(1, 2) == string_view("el"), "");
static_assert(string_view("hello").substr(4, 1) == string_view("o"), "");

static_assert(string_view("hello").substr(0, 100) == string_view("hello"), "");
static_assert(string_view("hello").substr(1, 100) == string_view("ello"), "");
static_assert(string_view("hello").substr(5, 100) == string_view(""), "");
namespace test_substr {
// 定义测试用例：当调用 substr 时，若位置超出范围则抛出异常
TEST(StringViewTest, whenCallingSubstrWithPosOutOfRange_thenThrows) {
  // 期望抛出 std::out_of_range 异常，因为索引 6 超出字符串 "hello" 的长度 5
  expectThrows<std::out_of_range>(
      [] { string_view("hello").substr(6); },
      "basic_string_view::substr parameter out of bounds. Index: 6, size: 5");

  // 期望抛出 std::out_of_range 异常，因为索引 6 超出字符串 "hello" 的长度 5
  expectThrows<std::out_of_range>(
      [] { string_view("hello").substr(6, 0); },
      "basic_string_view::substr parameter out of bounds. Index: 6, size: 5");
}
} // namespace test_substr

namespace test_compare_overload1 {
// 静态断言测试不同情况下 string_view 的 compare 函数返回值是否符合预期
static_assert(0 == string_view("").compare(string_view("")), "");
static_assert(0 == string_view("a").compare(string_view("a")), "");
static_assert(0 == string_view("hello").compare(string_view("hello")), "");
static_assert(0 < string_view("hello").compare(string_view("")), "");
static_assert(0 < string_view("hello").compare(string_view("aello")), "");
static_assert(0 < string_view("hello").compare(string_view("a")), "");
static_assert(
    0 < string_view("hello").compare(string_view("abcdefghijklmno")),
    "");
static_assert(0 < string_view("hello").compare(string_view("hela")), "");
static_assert(0 < string_view("hello").compare(string_view("helao")), "");
static_assert(
    0 < string_view("hello").compare(string_view("helaobcdefgh")),
    "");
static_assert(0 < string_view("hello").compare(string_view("hell")), "");
static_assert(0 > string_view("").compare(string_view("hello")), "");
static_assert(0 > string_view("hello").compare(string_view("zello")), "");
static_assert(0 > string_view("hello").compare(string_view("z")), "");
static_assert(
    0 > string_view("hello").compare(string_view("zabcdefghijklmno")),
    "");
static_assert(0 > string_view("hello").compare(string_view("helz")), "");
static_assert(0 > string_view("hello").compare(string_view("helzo")), "");
static_assert(
    0 > string_view("hello").compare(string_view("helzobcdefgh")),
    "");
static_assert(0 > string_view("hello").compare(string_view("helloa")), "");
} // namespace test_compare_overload1

namespace test_compare_overload2 {
// 静态断言测试 string_view 的 compare 函数重载形式二的不同情况
static_assert(0 == string_view("").compare(0, 0, string_view("")), "");
static_assert(0 == string_view("hello").compare(2, 2, string_view("ll")), "");
static_assert(0 < string_view("hello").compare(2, 2, string_view("l")), "");
static_assert(0 > string_view("hello").compare(2, 2, string_view("lll")), "");
static_assert(0 < string_view("hello").compare(2, 2, string_view("la")), "");
static_assert(0 > string_view("hello").compare(2, 2, string_view("lz")), "");
} // namespace test_compare_overload2

namespace test_compare_overload3 {
// 静态断言测试 string_view 的 compare 函数重载形式三的不同情况
static_assert(0 == string_view("").compare(0, 0, string_view(""), 0, 0), "");
static_assert(
    0 == string_view("hello").compare(2, 2, string_view("hello"), 2, 2),
    "");
static_assert(
    0 < string_view("hello").compare(2, 2, string_view("hello"), 2, 1),
    "");
static_assert(
    0 > string_view("hello").compare(2, 2, string_view("hello"), 2, 3),
    "");
static_assert(
    0 < string_view("hello").compare(2, 2, string_view("hellola"), 5, 2),
    "");
namespace test_compare_overload3 {
// 断言比较：用字符串视图比较 "hello" 的子串 "lz" 和 "hellolz" 的子串 "lz"
static_assert(
    0 > string_view("hello").compare(2, 2, string_view("hellolz"), 5, 2),
    "");
} // namespace test_compare_overload3

namespace test_compare_overload4 {
// 不同情况下的字符串视图比较断言
static_assert(0 == string_view("").compare(""), "");  // 空字符串与空字符串相等
static_assert(0 == string_view("a").compare("a"), "");  // "a" 与 "a" 相等
static_assert(0 == string_view("hello").compare("hello"), "");  // "hello" 与 "hello" 相等
static_assert(0 < string_view("hello").compare(""), "");  // "hello" 大于空字符串
static_assert(0 < string_view("hello").compare("aello"), "");  // "hello" 大于 "aello"
static_assert(0 < string_view("hello").compare("a"), "");  // "hello" 大于 "a"
static_assert(0 < string_view("hello").compare("abcdefghijklmno"), "");  // "hello" 大于 "abcdefghijklmno"
static_assert(0 < string_view("hello").compare("hela"), "");  // "hello" 大于 "hela"
static_assert(0 < string_view("hello").compare("helao"), "");  // "hello" 大于 "helao"
static_assert(0 < string_view("hello").compare("helaobcdefgh"), "");  // "hello" 大于 "helaobcdefgh"
static_assert(0 < string_view("hello").compare("hell"), "");  // "hello" 大于 "hell"
static_assert(0 > string_view("").compare("hello"), "");  // 空字符串小于 "hello"
static_assert(0 > string_view("hello").compare("zello"), "");  // "hello" 小于 "zello"
static_assert(0 > string_view("hello").compare("z"), "");  // "hello" 小于 "z"
static_assert(0 > string_view("hello").compare("zabcdefghijklmno"), "");  // "hello" 小于 "zabcdefghijklmno"
static_assert(0 > string_view("hello").compare("helz"), "");  // "hello" 小于 "helz"
static_assert(0 > string_view("hello").compare("helzo"), "");  // "hello" 小于 "helzo"
static_assert(0 > string_view("hello").compare("helzobcdefgh"), "");  // "hello" 小于 "helzobcdefgh"
static_assert(0 > string_view("hello").compare("helloa"), "");  // "hello" 小于 "helloa"
} // namespace test_compare_overload4

namespace test_compare_overload5 {
// 字符串视图的不同位置和长度比较断言
static_assert(0 == string_view("").compare(0, 0, ""), "");  // 空字符串的空子串与空字符串的空子串相等
static_assert(0 == string_view("hello").compare(2, 2, "ll"), "");  // "hello" 的子串 "ll" 与 "ll" 相等
static_assert(0 < string_view("hello").compare(2, 2, "l"), "");  // "hello" 的子串 "ll" 大于 "l"
static_assert(0 > string_view("hello").compare(2, 2, "lll"), "");  // "hello" 的子串 "ll" 小于 "lll"
static_assert(0 < string_view("hello").compare(2, 2, "la"), "");  // "hello" 的子串 "ll" 大于 "la"
static_assert(0 > string_view("hello").compare(2, 2, "lz"), "");  // "hello" 的子串 "ll" 小于 "lz"
} // namespace test_compare_overload5

namespace test_compare_overload6 {
// 字符串视图的不同位置和长度比较断言，使用另一个字符串的子串作为比较对象
static_assert(0 == string_view("").compare(0, 0, "", 0, 0), "");  // 空字符串的空子串与空字符串的空子串相等
static_assert(0 == string_view("hello").compare(2, 2, "hello", 2, 2), "");  // "hello" 的子串 "ll" 与 "hello" 的子串 "ll" 相等
static_assert(0 < string_view("hello").compare(2, 2, "hello", 2, 1), "");  // "hello" 的子串 "ll" 大于 "hello" 的子串 "l"
static_assert(0 > string_view("hello").compare(2, 2, "hello", 2, 3), "");  // "hello" 的子串 "ll" 小于 "hello" 的子串 "lll"
static_assert(0 < string_view("hello").compare(2, 2, "hellola", 5, 2), "");  // "hello" 的子串 "ll" 大于 "hellola" 的子串 "la"
static_assert(0 > string_view("hello").compare(2, 2, "hellolz", 5, 2), "");  // "hello" 的子串 "ll" 小于 "hellolz" 的子串 "lz"
} // namespace test_compare_overload6

namespace test_equality_comparison {
// 字符串视图的相等性比较断言
static_assert(string_view("hi") == string_view("hi"), "");  // "hi" 等于 "hi"
static_assert(!(string_view("hi") != string_view("hi")), "");  // "hi" 不不等于 "hi"

static_assert(string_view("") == string_view(""), "");  // 空字符串等于空字符串
static_assert(!(string_view("") != string_view("")), "");  // 空字符串不不等于空字符串

static_assert(string_view("hi") != string_view("hi2"), "");  // "hi" 不等于 "hi2"
static_assert(!(string_view("hi") == string_view("hi2")), "");  // "hi" 不等于 "hi2"

static_assert(string_view("hi2") != string_view("hi"), "");  // "hi2" 不等于 "hi"
static_assert(!(string_view("hi2") == string_view("hi")), "");  // "hi2" 不等于 "hi"

static_assert(string_view("hi") != string_view("ha"), "");  // "hi" 不等于 "ha"
} // namespace test_equality_comparison
// 确保字符串视图 "hi" 不等于 "ha"，如果等于则产生编译时错误信息为空字符串
static_assert(!(string_view("hi") == string_view("ha")), "");

// 确保字符串视图 "ha" 不等于 "hi"，如果等于则产生编译时错误信息为空字符串
static_assert(string_view("ha") != string_view("hi"), "");

// 确保字符串视图 "ha" 不等于 "hi"，如果等于则产生编译时错误信息为空字符串
static_assert(!(string_view("ha") == string_view("hi")), "");
} // namespace test_equality_comparison

namespace test_less_than {
// 确保空字符串视图不小于空字符串视图，如果小于则产生编译时错误信息为空字符串
static_assert(!(string_view("") < string_view("")), "");

// 确保相同的单字符视图不小于自身，如果小于则产生编译时错误信息为空字符串
static_assert(!(string_view("a") < string_view("a")), "");

// 确保相同的长字符串视图不小于自身，如果小于则产生编译时错误信息为空字符串
static_assert(!(string_view("hello") < string_view("hello")), "");

// 确保非空字符串视图不小于空字符串视图，如果小于则产生编译时错误信息为空字符串
static_assert(!(string_view("hello") < string_view("")), "");

// 确保字符串视图按字典序正确比较，如果错误则产生编译时错误信息为空字符串
static_assert(!(string_view("hello") < string_view("aello")), "");
static_assert(!(string_view("hello") < string_view("a")), "");
static_assert(!(string_view("hello") < string_view("abcdefghijklmno")), "");
static_assert(!(string_view("hello") < string_view("hela")), "");
static_assert(!(string_view("hello") < string_view("helao")), "");
static_assert(!(string_view("hello") < string_view("helaobcdefgh")), "");
static_assert(!(string_view("hello") < string_view("hell")), "");

// 确保空字符串视图小于非空字符串视图，如果不小于则产生编译时错误信息为空字符串
static_assert(string_view("") < string_view("hello"), "");
static_assert(string_view("hello") < string_view("zello"), "");
static_assert(string_view("hello") < string_view("z"), "");
static_assert(string_view("hello") < string_view("zabcdefghijklmno"), "");
static_assert(string_view("hello") < string_view("helz"), "");
static_assert(string_view("hello") < string_view("helzo"), "");
static_assert(string_view("hello") < string_view("helzobcdefgh"), "");
static_assert(string_view("hello") < string_view("helloa"), "");
} // namespace test_less_than

namespace test_less_or_equal_than {
// 确保空字符串视图小于或等于空字符串视图，如果不小于或等于则产生编译时错误信息为空字符串
static_assert(string_view("") <= string_view(""), "");

// 确保相同的单字符视图小于或等于自身，如果不小于或等于则产生编译时错误信息为空字符串
static_assert(string_view("a") <= string_view("a"), "");

// 确保相同的长字符串视图小于或等于自身，如果不小于或等于则产生编译时错误信息为空字符串
static_assert(string_view("hello") <= string_view("hello"), "");

// 确保非空字符串视图不小于空字符串视图，如果小于则产生编译时错误信息为空字符串
static_assert(!(string_view("hello") <= string_view("")), "");

// 确保字符串视图按字典序正确比较，如果错误则产生编译时错误信息为空字符串
static_assert(!(string_view("hello") <= string_view("aello")), "");
static_assert(!(string_view("hello") <= string_view("a")), "");
static_assert(!(string_view("hello") <= string_view("abcdefghijklmno")), "");
static_assert(!(string_view("hello") <= string_view("hela")), "");
static_assert(!(string_view("hello") <= string_view("helao")), "");
static_assert(!(string_view("hello") <= string_view("helaobcdefgh")), "");
static_assert(!(string_view("hello") <= string_view("hell")), "");

// 确保空字符串视图小于或等于非空字符串视图，如果不小于或等于则产生编译时错误信息为空字符串
static_assert(string_view("") <= string_view("hello"), "");
static_assert(string_view("hello") <= string_view("zello"), "");
static_assert(string_view("hello") <= string_view("z"), "");
static_assert(string_view("hello") <= string_view("zabcdefghijklmno"), "");
static_assert(string_view("hello") <= string_view("helz"), "");
static_assert(string_view("hello") <= string_view("helzo"), "");
static_assert(string_view("hello") <= string_view("helzobcdefgh"), "");
static_assert(string_view("hello") <= string_view("helloa"), "");
} // namespace test_less_or_equal_than

namespace test_greater_than {
// 确保空字符串视图不大于空字符串视图，如果大于则产生编译时错误信息为空字符串
static_assert(!(string_view("") > string_view("")), "");
namespace test_greater_than {
    // 断言：空字符串视图不大于空字符串视图
    static_assert(!(string_view("a") > string_view("a")), "");

    // 断言：相同的字符串视图不大于相同的字符串视图
    static_assert(!(string_view("hello") > string_view("hello")), "");

    // 断言：非空字符串视图大于空字符串视图
    static_assert(string_view("hello") > string_view(""), "");

    // 断言："hello" 大于 "aello"
    static_assert(string_view("hello") > string_view("aello"), "");

    // 断言："hello" 大于 "a"
    static_assert(string_view("hello") > string_view("a"), "");

    // 断言："hello" 大于 "abcdefghijklmno"
    static_assert(string_view("hello") > string_view("abcdefghijklmno"), "");

    // 断言："hello" 大于 "hela"
    static_assert(string_view("hello") > string_view("hela"), "");

    // 断言："hello" 大于 "helao"
    static_assert(string_view("hello") > string_view("helao"), "");

    // 断言："hello" 大于 "helaobcdefgh"
    static_assert(string_view("hello") > string_view("helaobcdefgh"), "");

    // 断言："hello" 大于 "hell"
    static_assert(string_view("hello") > string_view("hell"), "");

    // 断言：空字符串视图不大于 "hello"
    static_assert(!(string_view("") > string_view("hello")), "");

    // 断言："hello" 不大于 "zello"
    static_assert(!(string_view("hello") > string_view("zello")), "");

    // 断言："hello" 不大于 "z"
    static_assert(!(string_view("hello") > string_view("z")), "");

    // 断言："hello" 不大于 "zabcdefghijklmno"
    static_assert(!(string_view("hello") > string_view("zabcdefghijklmno")), "");

    // 断言："hello" 不大于 "helz"
    static_assert(!(string_view("hello") > string_view("helz")), "");

    // 断言："hello" 不大于 "helzo"
    static_assert(!(string_view("hello") > string_view("helzo")), "");

    // 断言："hello" 不大于 "helzobcdefgh"
    static_assert(!(string_view("hello") > string_view("helzobcdefgh")), "");

    // 断言："hello" 不大于 "helloa"
    static_assert(!(string_view("hello") > string_view("helloa")), "");
} // namespace test_greater_than

namespace test_greater_or_equals_than {
    // 断言：空字符串视图大于等于空字符串视图
    static_assert(string_view("") >= string_view(""), "");

    // 断言："a" 大于等于 "a"
    static_assert(string_view("a") >= string_view("a"), "");

    // 断言："hello" 大于等于 "hello"
    static_assert(string_view("hello") >= string_view("hello"), "");

    // 断言："hello" 大于等于空字符串视图
    static_assert(string_view("hello") >= string_view(""), "");

    // 断言："hello" 大于等于 "aello"
    static_assert(string_view("hello") >= string_view("aello"), "");

    // 断言："hello" 大于等于 "a"
    static_assert(string_view("hello") >= string_view("a"), "");

    // 断言："hello" 大于等于 "abcdefghijklmno"
    static_assert(string_view("hello") >= string_view("abcdefghijklmno"), "");

    // 断言："hello" 大于等于 "hela"
    static_assert(string_view("hello") >= string_view("hela"), "");

    // 断言："hello" 大于等于 "helao"
    static_assert(string_view("hello") >= string_view("helao"), "");

    // 断言："hello" 大于等于 "helaobcdefgh"
    static_assert(string_view("hello") >= string_view("helaobcdefgh"), "");

    // 断言："hello" 大于等于 "hell"
    static_assert(string_view("hello") >= string_view("hell"), "");

    // 断言：空字符串视图不大于等于 "hello"
    static_assert(!(string_view("") >= string_view("hello")), "");

    // 断言："hello" 不大于等于 "zello"
    static_assert(!(string_view("hello") >= string_view("zello")), "");

    // 断言："hello" 不大于等于 "z"
    static_assert(!(string_view("hello") >= string_view("z")), "");

    // 断言："hello" 不大于等于 "zabcdefghijklmno"
    static_assert(!(string_view("hello") >= string_view("zabcdefghijklmno")), "");

    // 断言："hello" 不大于等于 "helz"
    static_assert(!(string_view("hello") >= string_view("helz")), "");

    // 断言："hello" 不大于等于 "helzo"
    static_assert(!(string_view("hello") >= string_view("helzo")), "");

    // 断言："hello" 不大于等于 "helzobcdefgh"
    static_assert(!(string_view("hello") >= string_view("helzobcdefgh")), "");

    // 断言："hello" 不大于等于 "helloa"
    static_assert(!(string_view("hello") >= string_view("helloa")), "");
} // namespace test_greater_or_equals_than

namespace test_starts_with {
    // 断言："hi" 以 "hi" 开头
    static_assert(string_view("hi").starts_with(string_view("hi")), "");

    // 断言：空字符串视图以空字符串视图开头
    static_assert(string_view("").starts_with(string_view("")), "");

    // 断言："hi2" 以空字符串视图开头
    static_assert(string_view("hi2").starts_with(string_view("")), "");

    // 断言：空字符串视图不以 "hi" 开头
    static_assert(!string_view("").starts_with(string_view("hi")), "");

    // 断言："hi2" 以 "hi" 开头
    static_assert(string_view("hi2").starts_with(string_view("hi")), "");
} // namespace test_starts_with
// 使用 static_assert 断言检查：string_view("hi") 不以 string_view("hi2") 开头，如果不成立则输出空字符串
static_assert(!string_view("hi").starts_with(string_view("hi2")), "");
// 使用 static_assert 断言检查：string_view("hi") 不以 string_view("ha") 开头，如果不成立则输出空字符串
static_assert(!string_view("hi").starts_with(string_view("ha")), "");

// 使用 static_assert 断言检查：string_view("hi") 以 "hi" 开头，如果不成立则输出空字符串
static_assert(string_view("hi").starts_with("hi"), "");
// 使用 static_assert 断言检查：空 string_view 以空 string_view 开头，如果不成立则输出空字符串
static_assert(string_view("").starts_with(""), "");
// 使用 static_assert 断言检查："hi2" 以空 string_view 开头，如果不成立则输出空字符串
static_assert(string_view("hi2").starts_with(""), "");
// 使用 static_assert 断言检查：空 string_view 不以 "hi" 开头，如果不成立则输出空字符串
static_assert(!string_view("").starts_with("hi"), "");
// 使用 static_assert 断言检查："hi2" 以 "hi" 开头，如果不成立则输出空字符串
static_assert(string_view("hi2").starts_with("hi"), "");
// 使用 static_assert 断言检查："hi" 不以 "hi2" 开头，如果不成立则输出空字符串
static_assert(!string_view("hi").starts_with("hi2"), "");
// 使用 static_assert 断言检查："hi" 不以 "ha" 开头，如果不成立则输出空字符串
static_assert(!string_view("hi").starts_with("ha"), "");

// 使用 static_assert 断言检查：空 string_view 不以字符 'a' 开头，如果不成立则输出空字符串
static_assert(!string_view("").starts_with('a'), "");
// 使用 static_assert 断言检查：空 string_view 不以字符 '\0' 开头，如果不成立则输出空字符串
static_assert(!string_view("").starts_with('\0'), "");
// 使用 static_assert 断言检查："hello" 不以字符 'a' 开头，如果不成立则输出空字符串
static_assert(!string_view("hello").starts_with('a'), "");
// 使用 static_assert 断言检查："hello" 以字符 'h' 开头，如果不成立则输出空字符串
static_assert(string_view("hello").starts_with('h'), "");

} // namespace test_starts_with

namespace test_ends_with {
// 使用 static_assert 断言检查：string_view("hi") 以 string_view("hi") 结尾，如果不成立则输出空字符串
static_assert(string_view("hi").ends_with(string_view("hi")), "");
// 使用 static_assert 断言检查：空 string_view 以空 string_view 结尾，如果不成立则输出空字符串
static_assert(string_view("").ends_with(string_view("")), "");
// 使用 static_assert 断言检查："hi2" 以空 string_view 结尾，如果不成立则输出空字符串
static_assert(string_view("hi2").ends_with(string_view("")), "");
// 使用 static_assert 断言检查：空 string_view 不以 string_view("hi") 结尾，如果不成立则输出空字符串
static_assert(!string_view("").ends_with(string_view("hi")), "");
// 使用 static_assert 断言检查："hi2" 以 string_view("i2") 结尾，如果不成立则输出空字符串
static_assert(string_view("hi2").ends_with(string_view("i2")), "");
// 使用 static_assert 断言检查："i2" 不以 string_view("hi2") 结尾，如果不成立则输出空字符串
static_assert(!string_view("i2").ends_with(string_view("hi2")), "");
// 使用 static_assert 断言检查："hi" 不以 string_view("ha") 结尾，如果不成立则输出空字符串
static_assert(!string_view("hi").ends_with(string_view("ha")), "");

// 使用 static_assert 断言检查：string_view("hi") 以 "hi" 结尾，如果不成立则输出空字符串
static_assert(string_view("hi").ends_with("hi"), "");
// 使用 static_assert 断言检查：空 string_view 以空字符串结尾，如果不成立则输出空字符串
static_assert(string_view("").ends_with(""), "");
// 使用 static_assert 断言检查："hi2" 以空字符串结尾，如果不成立则输出空字符串
static_assert(string_view("hi2").ends_with(""), "");
// 使用 static_assert 断言检查：空 string_view 不以 "hi" 结尾，如果不成立则输出空字符串
static_assert(!string_view("").ends_with("hi"), "");
// 使用 static_assert 断言检查："hi2" 以 "i2" 结尾，如果不成立则输出空字符串
static_assert(string_view("hi2").ends_with("i2"), "");
// 使用 static_assert 断言检查："i2" 不以 "hi2" 结尾，如果不成立则输出空字符串
static_assert(!string_view("i2").ends_with("hi2"), "");
// 使用 static_assert 断言检查："hi" 不以 "ha" 结尾，如果不成立则输出空字符串
static_assert(!string_view("hi").ends_with("ha"), "");

// 使用 static_assert 断言检查：空 string_view 不以字符 'a' 结尾，如果不成立则输出空字符串
static_assert(!string_view("").ends_with('a'), "");
// 使用 static_assert 断言检查：空 string_view 不以字符 '\0' 结尾，如果不成立则输出空字符串
static_assert(!string_view("").ends_with('\0'), "");
// 使用 static_assert 断言检查："hello" 不以字符 'a' 结尾，如果不成立则输出空字符串
static_assert(!string_view("hello").ends_with('a'), "");
// 使用 static_assert 断言检查："hello" 以字符 'o' 结尾，如果不成立则输出空字符串
static_assert(string_view("hello").ends_with('o'), "");

} // namespace test_ends_with

namespace test_find_overload1 {
// 使用 static_assert 断言检查：空 string_view 中查找空 string_view 的结果为 0，如果不成立则输出空字符串
static_assert(0 == string_view("").find(string_view("")), "");
// 使用 static_assert 断言检查：空 string_view 中查找 string_view("a") 的结果为 npos，如果不成立则输出空字符串
static_assert(string_view::npos == string_view("").find(string_view("a")), "");
// 使用 static_assert 断言检查：从空 string_view 的第 1 个位置开始查找空 string_view 的结果为 npos，如果不成立则输出空字符串
static_assert(
    string_view::npos == string_view("").find(string_view(""), 1),
    "");
// 使用 static_assert 断言检查："abc" 中查找空 string_view 的结果为 0，如果不成立则输出空字符串
static_assert(0 == string_view("abc").find(string_view("")), "");
// 使用 static_assert 断言检查："abc" 中从第 2 个位置开始查找空 string_view 的结果为 2，如果不成立则输出空字符串
static_assert(2 == string_view("abc").find(string_view(""), 2), "");
// 使用 static_assert 断言检查："abc" 中查找 string_view("a") 的结果为 0，如果不成立则输出空字符串
static_assert(0 == string_view("abc").find(string_view("a")), "");
// 使用 static_assert 断言检查："abc" 中查找 string_view("ab") 的结果为 0，如果不成立则输出空字符串
static_assert(0 == string_view("abc").find(string_view("ab")), "");
// 使用 static_assert 断言检查："abc" 中查找 string_view("abc") 的结果为 0，如果不成立则输出空字符串
static_assert(0 == string_view("abc").find(string_view("abc")), "");
// 使用 static_assert 断言检查："abc" 中查找 string_view("bc") 的结果为 1，如果不成立则输出空字符串
static_assert(1 == string_view("abc").find(string_view("bc")), "");
// 使用 static_assert 断言检查："abc" 中查找 string_view("b") 的结果为 1，如果不成立则输出空字符串
static_assert(1 == string_view("abc").find(string_view("b")), "");
// 使用 static_assert 断言检查："abc" 中查找 string_view("c") 的结果为 2，如果不成立则输出空字符串
static_assert(2 == string_view("abc").find(string_view("c")), "");
// 使用 static_assert 断言检查："ababa" 中查找 string_view("ba") 的结果为 1，如果不成立则输出空字符串
// 静态断言，检查从字符串视图 "ababa" 中的偏移量 2 开始，是否可以找到子字符串 "ba" 的位置为 3
static_assert(3 == string_view("ababa").find(string_view("ba"), 2), "");

// 静态断言，检查从字符串视图 "ababa" 中的偏移量 3 开始，是否可以找到子字符串 "ba" 的位置为 3
static_assert(3 == string_view("ababa").find(string_view("ba"), 3), "");

// 静态断言，检查从字符串视图 "ababa" 中的偏移量 4 开始，是否找不到子字符串 "ba"，返回 string_view::npos
static_assert(
    string_view::npos == string_view("ababa").find(string_view("ba"), 4),
    "");

// 静态断言，检查字符串视图 "abc" 是否找不到子字符串 "abcd"，返回 string_view::npos
static_assert(
    string_view::npos == string_view("abc").find(string_view("abcd")),
    "");
} // namespace test_find_overload1

namespace test_find_overload2 {
// 静态断言，检查空字符串视图 "" 是否找不到字符 'a'，返回 string_view::npos
static_assert(string_view::npos == string_view("").find('a'), "");

// 静态断言，检查字符串视图 "a" 中是否可以找到字符 'a' 的位置为 0
static_assert(0 == string_view("a").find('a'), "");

// 静态断言，检查字符串视图 "abc" 中是否可以找到字符 'a' 的位置为 0
static_assert(0 == string_view("abc").find('a'), "");

// 静态断言，检查字符串视图 "a" 中从偏移量 1 开始是否找不到字符 'a'，返回 string_view::npos
static_assert(string_view::npos == string_view("a").find('a', 1), "");

// 静态断言，检查字符串视图 "abc" 中是否可以找到字符 'b' 的位置为 1
static_assert(1 == string_view("abc").find('b'), "");

// 静态断言，检查字符串视图 "abc" 中从偏移量 1 开始是否可以找到字符 'b' 的位置为 1
static_assert(1 == string_view("abc").find('b', 1), "");

// 静态断言，检查字符串视图 "abc" 中从偏移量 2 开始是否找不到字符 'b'，返回 string_view::npos
static_assert(string_view::npos == string_view("abc").find('b', 2), "");

// 静态断言，检查字符串视图 "abc" 中是否可以找到字符 'c' 的位置为 2
static_assert(2 == string_view("abc").find('c'), "");

// 静态断言，检查字符串视图 "abc" 中从偏移量 1 开始是否可以找到字符 'c' 的位置为 2
static_assert(2 == string_view("abc").find('c', 1), "");

// 静态断言，检查字符串视图 "abc" 中从偏移量 2 开始是否可以找到字符 'c' 的位置为 2
static_assert(2 == string_view("abc").find('c', 2), "");

// 静态断言，检查字符串视图 "abc" 中从偏移量 3 开始是否找不到字符 'c'，返回 string_view::npos
static_assert(string_view::npos == string_view("abc").find('c', 3), "");

// 静态断言，检查字符串视图 "abc" 中从偏移量 100 开始是否找不到字符 'a'，返回 string_view::npos
static_assert(string_view::npos == string_view("abc").find('a', 100), "");

// 静态断言，检查字符串视图 "abc" 中是否找不到字符 'z'，返回 string_view::npos
static_assert(string_view::npos == string_view("abc").find('z'), "");

// 静态断言，检查字符串视图 "ababa" 中是否可以找到字符 'a' 的位置为 0
static_assert(0 == string_view("ababa").find('a'), "");

// 静态断言，检查字符串视图 "ababa" 中从偏移量 0 开始是否可以找到字符 'a' 的位置为 0
static_assert(0 == string_view("ababa").find('a', 0), "");

// 静态断言，检查字符串视图 "ababa" 中从偏移量 1 开始是否可以找到字符 'a' 的位置为 2
static_assert(2 == string_view("ababa").find('a', 1), "");

// 静态断言，检查字符串视图 "ababa" 中从偏移量 2 开始是否可以找到字符 'a' 的位置为 2
static_assert(2 == string_view("ababa").find('a', 2), "");

// 静态断言，检查字符串视图 "ababa" 中从偏移量 3 开始是否可以找到字符 'a' 的位置为 4
static_assert(4 == string_view("ababa").find('a', 3), "");

// 静态断言，检查字符串视图 "ababa" 中从偏移量 4 开始是否找不到字符 'a'，返回 string_view::npos
static_assert(4 == string_view("ababa").find('a', 4), "");

// 静态断言，检查字符串视图 "ababa" 中从偏移量 5 开始是否找不到字符 'a'，返回 string_view::npos
static_assert(string_view::npos == string_view("ababa").find('a', 5), "");
} // namespace test_find_overload2

namespace test_find_overload3 {
// 静态断言，检查空字符串视图 "" 中从偏移量 0 开始是否可以找到空字符串的位置为 0
static_assert(0 == string_view("").find("", 0, 0), "");

// 静态断言，检查空字符串视图 "" 中从偏移量 0 开始是否找不到子字符串 "a"，返回 string_view::npos
static_assert(string_view::npos == string_view("").find("a", 0, 1), "");

// 静态断言，检查空字符串视图 "" 中从偏移量 1 开始是否找不到空字符串，返回 string_view::npos
static_assert(string_view::npos == string_view("").find("", 1, 0), "");

// 静态断言，检查字符串视图 "abc" 中从偏移量 0 开始是否可以找到空字符串的位置为 0
static_assert(0 == string_view("abc").find("", 0, 0), "");

// 静态断言，检查字符串视图 "abc" 中从偏移量 2 开始是否可以找到空字符串的位置为 2
static_assert(2 == string_view("abc").find("", 2, 0), "");

// 静态断言，检查字符串视图 "abc" 中从偏移量 0 开始是否可以找到子字符串 "a" 的位置为 0
static_assert(0 == string_view("abc").find("a", 0, 1), "");

// 静态断言，检查字符串视图 "abc" 中从偏移量 0 开始是否可以找到子字符串 "ab" 的位置为 0
static_assert(0 == string_view("abc").find("ab", 0, 2), "");

// 静态断言，检查字符串视图 "abc" 中从偏移量 0 开始是否可以找到子字符串 "abc" 的位置为 0
static_assert(0 == string_view("abc").find("abc", 0, 3), "");

// 静态断言，检查字符串视图 "abc" 中从偏移量 0 开始是否可以找到子字符串 "bc" 的位置为 1
static_assert(1 == string_view("abc").find("bc", 0, 2), "");

// 静态断言，检查字符串视图 "abc" 中从偏移量 0 开始是否可以找到字符 'b' 的位置为 1
static_assert(1 == string_view("abc").find("b", 0, 1), "");

// 静态断言，检查字符串视图 "abc" 中从偏移量 0 开始是否可以找到字符 'c' 的位置为 2
static_assert(2 == string_view("abc").find("c
// 使用静态断言检查空字符串在空字符串中查找的结果为0
static_assert(0 == string_view("").find(""), "");
// 使用静态断言检查空字符串在空字符串中查找字符'a'的结果为string_view::npos
static_assert(string_view::npos == string_view("").find("a"), "");
// 使用静态断言检查空字符串在索引1处查找空字符串的结果为string_view::npos
static_assert(string_view::npos == string_view("").find("", 1), "");
// 使用静态断言检查字符串"abc"在自身中查找空字符串的结果为0
static_assert(0 == string_view("abc").find(""), "");
// 使用静态断言检查字符串"abc"在索引2处查找空字符串的结果为2
static_assert(2 == string_view("abc").find("", 2), "");
// 使用静态断言检查字符串"abc"在自身中查找字符'a'的结果为0
static_assert(0 == string_view("abc").find("a"), "");
// 使用静态断言检查字符串"abc"在自身中查找字符串"ab"的结果为0
static_assert(0 == string_view("abc").find("ab"), "");
// 使用静态断言检查字符串"abc"在自身中查找字符串"abc"的结果为0
static_assert(0 == string_view("abc").find("abc"), "");
// 使用静态断言检查字符串"abc"在自身中查找字符串"bc"的结果为1
static_assert(1 == string_view("abc").find("bc"), "");
// 使用静态断言检查字符串"abc"在自身中查找字符'b'的结果为1
static_assert(1 == string_view("abc").find("b"), "");
// 使用静态断言检查字符串"abc"在自身中查找字符'c'的结果为2
static_assert(2 == string_view("abc").find("c"), "");
// 使用静态断言检查字符串"abc"在自身中查找字符'a'的结果为0
static_assert(0 == string_view("abc").find("a"), "");
// 使用静态断言检查字符串"abc"在自身中查找字符串"ab"的结果为0
static_assert(0 == string_view("abc").find("ab"), "");
// 使用静态断言检查字符串"abc"在自身中查找字符串"abc"的结果为0
static_assert(0 == string_view("abc").find("abc"), "");
// 使用静态断言检查字符串"ababa"在自身中查找字符串"ba"的结果为1
static_assert(1 == string_view("ababa").find("ba"), "");
// 使用静态断言检查字符串"ababa"在索引2处开始查找字符串"ba"的结果为3
static_assert(3 == string_view("ababa").find("ba", 2), "");
// 使用静态断言检查字符串"ababa"在索引3处开始查找字符串"ba"的结果为3
static_assert(3 == string_view("ababa").find("ba", 3), "");
// 使用静态断言检查字符串"ababa"在索引4处开始查找字符串"ba"的结果为string_view::npos
static_assert(string_view::npos == string_view("ababa").find("ba", 4), "");
// 使用静态断言检查字符串"abc"在自身中查找字符串"abcd"的结果为string_view::npos
static_assert(string_view::npos == string_view("abc").find("abcd"), "");
} // namespace test_find_overload4

namespace test_rfind_overload1 {
// 使用静态断言检查空字符串在空字符串中反向查找空字符串的结果为0
static_assert(0 == string_view("").rfind(string_view("")), "");
// 使用静态断言检查空字符串在空字符串中反向查找字符'a'的结果为string_view::npos
static_assert(string_view::npos == string_view("").rfind(string_view("a")), "");
// 使用静态断言检查空字符串在索引1处反向查找空字符串的结果为0
static_assert(0 == string_view("").rfind(string_view(""), 1), "");
// 使用静态断言检查字符串"abc"在自身中反向查找空字符串的结果为3
static_assert(3 == string_view("abc").rfind(string_view("")), "");
// 使用静态断言检查字符串"abc"在索引0处反向查找空字符串的结果为0
static_assert(0 == string_view("abc").rfind(string_view(""), 0), "");
// 使用静态断言检查字符串"abc"在自身中反向查找字符'a'的结果为0
static_assert(0 == string_view("abc").rfind(string_view("a")), "");
// 使用静态断言检查字符串"abc"在自身中反向查找字符串"ab"的结果为0
static_assert(0 == string_view("abc").rfind(string_view("ab")), "");
// 使用静态断言检查字符串"abc"在自身中反向查找字符串"abc"的结果为0
static_assert(0 == string_view("abc").rfind(string_view("abc")), "");
// 使用静态断言检查字符串"abc"在自身中反向查找字符串"bc"的结果为1
static_assert(1 == string_view("abc").rfind(string_view("bc")), "");
// 使用静态断言检查字符串"abc"在自身中反向查找字符'b'的结果为1
static_assert(1 == string_view("abc").rfind(string_view("b")), "");
// 使用静态断言检查字符串"abc"在自身中反向查找字符'c'的结果为2
static_assert(2 == string_view("abc").rfind(string_view("c")), "");
// 使用静态断言检查字符串"abc"在自身中反向查找字符'a'的结果为0
static_assert(0 == string_view("abc").rfind(string_view("a")), "");
// 使用静态断言检查字符串"abc"在自身中反向查找字符串"ab"的结果为0
static_assert(0 == string_view("abc").rfind(string_view("ab")), "");
// 使用静态断言检查字符串"abc"在自身中反向查找字符串"abc"的结果为0
static_assert(0 == string_view("abc").rfind(string_view("abc")), "");
// 使用静态断言检查字符串"ababa"在自身中反向查找字符串"ba"的结果为3
static_assert(3 == string_view("ababa").rfind(string_view("ba")), "");
// 使用静态断言检查字符串"ababa"在索引2处开始反向查找字符串"ba"的结果为1
static_assert(1 == string_view("ababa").rfind(string_view("ba"), 2), "");
// 使用静态断言检查字符串"ababa"在索引1处开始反向查找字符串"ba"的结果为1
static_assert(1 == string_view("ababa").rfind(string_view("ba"), 1), "");
// 使用静态断言检查字符串"ababa"在索引0处开始反向查找字符串"ba"的结果为string_view::npos
static_assert(
    string_view::npos == string_view("ababa").rfind(string_view("ba"), 0),
    "");
// 使用静态断言检查字符串"abc"在自身中反向查找字符串"abcd"的结果为string_view::npos
static_assert(
    string_view::npos == string_view("abc").rfind(string_view("abcd")),
    "");
} // namespace test_rfind_overload1

namespace test_rfind_overload2 {
// 使用静态断言检查空字符串在空字符串中反向查找字符'a'的结果为string_view::npos
static_assert(string_view::npos == string_view("").rfind('a'), "");
// 使用静态断言检查字符串"a"在自身中反向查找字符'a'的结果为0
static_assert(0 == string_view("a").rfind('a'), "");
// 使用静态断言检查字符串"abc"在自身中反向查找字符'a'的结果为0
static_assert(0 == string_view("abc").rfind('a'), "");
// 使用静态断言检查字符串"a"在索引0处开始反向查找字符'a'的结果为0
static_assert(0 == string_view("a").rfind('a', 0), "");
// 使用静态断言检查字符串"abc"在自身中反向查找字符'b'的结果为1
static_assert(1 == string_view("abc").rfind('b'), "");
// 使用静态断言检查字符串"abc
// 静态断言，验证从字符串视图 "abc" 中逆向查找字符 'b' 在索引 1 处，应为 1
static_assert(1 == string_view("abc").rfind('b', 1), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找字符 'c'，应为索引 2
static_assert(2 == string_view("abc").rfind('c'), "");

// 静态断言，验证从空字符串视图中逆向查找字符 'c' 在索引 0 处，结果应为 string_view::npos
static_assert(string_view::npos == string_view("abc").rfind('c', 0), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找字符 'c' 在索引 1 处，结果应为 string_view::npos
static_assert(string_view::npos == string_view("abc").rfind('c', 1), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找字符 'c' 在索引 2 处，结果应为 2
static_assert(2 == string_view("abc").rfind('c', 2), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找字符 'c' 在索引 3 处，结果应为 2
static_assert(2 == string_view("abc").rfind('c', 3), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找字符 'a' 在超出字符串长度的索引 100 处，结果应为 0
static_assert(0 == string_view("abc").rfind('a', 100), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找字符 'z'，结果应为 string_view::npos
static_assert(string_view::npos == string_view("abc").rfind('z'), "");

// 静态断言，验证从字符串视图 "ababa" 中逆向查找字符 'a'，结果应为索引 4
static_assert(4 == string_view("ababa").rfind('a'), "");

// 静态断言，验证从字符串视图 "ababa" 中逆向查找字符 'a' 在索引 0 处，结果应为 0
static_assert(0 == string_view("ababa").rfind('a', 0), "");

// 静态断言，验证从字符串视图 "ababa" 中逆向查找字符 'a' 在索引 1 处，结果应为 0
static_assert(0 == string_view("ababa").rfind('a', 1), "");

// 静态断言，验证从字符串视图 "ababa" 中逆向查找字符 'a' 在索引 2 处，结果应为 2
static_assert(2 == string_view("ababa").rfind('a', 2), "");

// 静态断言，验证从字符串视图 "ababa" 中逆向查找字符 'a' 在索引 3 处，结果应为 2
static_assert(2 == string_view("ababa").rfind('a', 3), "");

// 静态断言，验证从字符串视图 "ababa" 中逆向查找字符 'a' 在索引 4 处，结果应为 4
static_assert(4 == string_view("ababa").rfind('a', 4), "");

// 静态断言，验证从字符串视图 "ababa" 中逆向查找字符 'a' 在索引 5 处，结果应为 4
static_assert(4 == string_view("ababa").rfind('a', 5), "");
} // namespace test_rfind_overload2

namespace test_rfind_overload3 {
// 静态断言，验证从空字符串视图中逆向查找空字符串在任意位置，结果应为 0
static_assert(0 == string_view("").rfind("", string_view::npos, 0), "");

// 静态断言，验证从空字符串视图中逆向查找字符 'a'，结果应为 string_view::npos
static_assert(string_view::npos == string_view("").rfind("a", string_view::npos, 1), "");

// 静态断言，验证从空字符串视图中逆向查找空字符串在索引 1 处，结果应为 0
static_assert(0 == string_view("").rfind("", 1, 0), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找空字符串在任意位置，结果应为索引 3
static_assert(3 == string_view("abc").rfind("", string_view::npos, 0), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找空字符串在索引 0 处，结果应为 0
static_assert(0 == string_view("abc").rfind("", 0, 0), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找字符串 "a"，结果应为 0
static_assert(0 == string_view("abc").rfind("a", string_view::npos, 1), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找字符串 "ab"，结果应为 0
static_assert(0 == string_view("abc").rfind("ab", string_view::npos, 2), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找字符串 "abc"，结果应为 0
static_assert(0 == string_view("abc").rfind("abc", string_view::npos, 3), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找字符串 "bc"，结果应为索引 1
static_assert(1 == string_view("abc").rfind("bc", string_view::npos, 2), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找字符串 "b"，结果应为索引 1
static_assert(1 == string_view("abc").rfind("b", string_view::npos, 1), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找字符串 "c"，结果应为索引 2
static_assert(2 == string_view("abc").rfind("c", string_view::npos, 1), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找字符串 "a"，结果应为索引 0
static_assert(0 == string_view("abc").rfind("a", string_view::npos, 1), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找字符串 "ab"，结果应为索引 0
static_assert(0 == string_view("abc").rfind("ab", string_view::npos, 2), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找字符串 "abc"，结果应为索引 0
static_assert(0 == string_view("abc").rfind("abc", string_view::npos, 3), "");

// 静态断言，验证从字符串视图 "ababa" 中逆向查找字符串 "ba"，结果应为索引 3
static_assert(3 == string_view("ababa").rfind("ba", string_view::npos, 2), "");

// 静态断言，验证从字符串视图 "ababa" 中逆向查找字符串 "ba" 在索引 2 处，范围为 2，结果应为索引 1
static_assert(1 == string_view("ababa").rfind("ba", 2, 2), "");

// 静态断言，验证从字符串视图 "ababa" 中逆向查找字符串 "ba" 在索引 1 处，范围为 2，结果应为索引 1
static_assert(1 == string_view("ababa").rfind("ba", 1, 2), "");

// 静态断言，验证从字符串视图 "ababa" 中逆向查找字符串 "ba" 在索引 0 处，范围为 2，结果应为 string_view::npos
static_assert(string_view::npos == string_view("ababa").rfind("ba", 0, 2), "");

// 静态断言，验证从字符串视图 "abc" 中逆向查找字符串 "abcd"，结果应为 string_view::npos
static_assert(string_view::npos == string_view("abc").rfind
// 静态断言，验证在字符串视图 "abc" 中查找子串 "bc" 的最后出现位置为1
static_assert(1 == string_view("abc").rfind("bc"), "");
// 静态断言，验证在字符串视图 "abc" 中查找字符 'b' 的最后出现位置为1
static_assert(1 == string_view("abc").rfind("b"), "");
// 静态断言，验证在字符串视图 "abc" 中查找字符 'c' 的最后出现位置为2
static_assert(2 == string_view("abc").rfind("c"), "");
// 静态断言，验证在字符串视图 "abc" 中查找字符 'a' 的最后出现位置为0
static_assert(0 == string_view("abc").rfind("a"), "");
// 静态断言，验证在字符串视图 "abc" 中查找子串 "ab" 的最后出现位置为0
static_assert(0 == string_view("abc").rfind("ab"), "");
// 静态断言，验证在字符串视图 "abc" 中查找子串 "abc" 的最后出现位置为0
static_assert(0 == string_view("abc").rfind("abc"), "");
// 静态断言，验证在字符串视图 "ababa" 中查找子串 "ba" 的最后出现位置为3
static_assert(3 == string_view("ababa").rfind("ba"), "");
// 静态断言，验证在字符串视图 "ababa" 中从索引2开始查找子串 "ba" 的最后出现位置为1
static_assert(1 == string_view("ababa").rfind("ba", 2), "");
// 静态断言，验证在字符串视图 "ababa" 中从索引1开始查找子串 "ba" 的最后出现位置为1
static_assert(1 == string_view("ababa").rfind("ba", 1), "");
// 静态断言，验证在字符串视图 "ababa" 中从索引0开始查找子串 "ba" 的最后出现位置为未找到
static_assert(string_view::npos == string_view("ababa").rfind("ba", 0), "");
// 静态断言，验证在字符串视图 "abc" 中查找子串 "abcd" 的最后出现位置为未找到
static_assert(string_view::npos == string_view("abc").rfind("abcd"), "");
} // namespace test_rfind_overload4

namespace test_find_first_of_overload1 {
// 静态断言，验证在空字符串视图中查找任何子串的首次出现位置为未找到
static_assert(
    string_view::npos == string_view("").find_first_of(string_view("")),
    "");
// 静态断言，验证在空字符串视图中查找任何子串的首次出现位置为未找到
static_assert(
    string_view::npos == string_view("").find_first_of(string_view("a")),
    "");
// 静态断言，验证在空字符串视图中查找任何子串的首次出现位置为未找到
static_assert(
    string_view::npos == string_view("").find_first_of(string_view("abc")),
    "");
// 静态断言，验证在字符串视图 "abc" 中查找空子串的首次出现位置为未找到
static_assert(
    string_view::npos == string_view("abc").find_first_of(string_view("")),
    "");
// 静态断言，验证在字符串视图 "abc" 中查找字符 'd' 的首次出现位置为未找到
static_assert(
    string_view::npos == string_view("abc").find_first_of(string_view("d")),
    "");
// 静态断言，验证在字符串视图 "abc" 中查找字符集合 "def" 的首次出现位置为未找到
static_assert(
    string_view::npos == string_view("abc").find_first_of(string_view("def")),
    "");

// 静态断言，验证在字符串视图 "abcabc" 中查找字符 'a' 的首次出现位置为0
static_assert(0 == string_view("abcabc").find_first_of(string_view("a")), "");
// 静态断言，验证在字符串视图 "abcabc" 中查找字符 'b' 的首次出现位置为1
static_assert(1 == string_view("abcabc").find_first_of(string_view("b")), "");
// 静态断言，验证在字符串视图 "abcabc" 中查找字符 'c' 的首次出现位置为2
static_assert(2 == string_view("abcabc").find_first_of(string_view("c")), "");
// 静态断言，验证在字符串视图 "abcabc" 中查找字符集合 "bc" 的首次出现位置为1
static_assert(1 == string_view("abcabc").find_first_of(string_view("bc")), "");
// 静态断言，验证在字符串视图 "abcabc" 中查找字符集合 "cbd" 的首次出现位置为1
static_assert(1 == string_view("abcabc").find_first_of(string_view("cbd")), "");

// 静态断言，验证在空字符串视图中从索引1开始查找任何子串的首次出现位置为未找到
static_assert(
    string_view::npos == string_view("").find_first_of(string_view(""), 1),
    "");
// 静态断言，验证在空字符串视图中从索引1开始查找子串 "a" 的首次出现位置为未找到
static_assert(
    string_view::npos == string_view("").find_first_of(string_view("a"), 1),
    "");
// 静态断言，验证在空字符串视图中从索引100开始查找字符集合 "abc" 的首次出现位置为未找到
static_assert(
    string_view::npos == string_view("").find_first_of(string_view("abc"), 100),
    "");
// 静态断言，验证在字符串视图 "abc" 中从索引1开始查找任何子串的首次出现位置为未找到
static_assert(
    string_view::npos == string_view("abc").find_first_of(string_view(""), 1),
    "");
// 静态断言，验证在字符串视图 "abc" 中从索引3开始查找字符 'd' 的首次出现位置为未找到
static_assert(
    string_view::npos == string_view("abc").find_first_of(string_view("d"), 3),
    "");
// 静态断言，验证在字符串视图 "abc" 中从索引2开始查找字符集合 "def" 的首次出现位置为未找到
static_assert(
    string_view::npos ==
        string_view("abc").find_first_of(string_view("def"), 2),
    "");

// 静态断言，验证在字符串视图 "abcabc" 中从索引1开始查找字符 'a' 的首次出现位置为3
static_assert(
    3 == string_view("abcabc").find_first_of(string_view("a"), 1),
    "");
// 静态断言，验证在字符串视图 "abcabc" 中从索引3开始查找字符 'b' 的首次出现位置为4
static_assert(
    4 == string_view("abcabc").find_first_of(string_view("b"), 3),
    "");
// 静态断言，验证在字符串视图 "abcabc" 中从索引5开始查找字符 'c' 的首次出现位置为5
static_assert(
    5 == string_view("abcabc").find_first_of(string_view("c"), 5),
    "");
// 静态断言，验证在字符串视图 "abcabc" 中从索引3开始查找字符集合 "bc" 的首次出现位置为4
static_assert(
    4 == string_view("abcabc").find_first_of(string_view("bc"), 3),
    "");
// 静态断言，验证在字符串视图 "abcabc" 中从索引4开始查找字符集合 "cbd" 的首次出现位置为4
static_assert(
    4 == string_view("abcabc").find_first_of(string_view("cbd"), 4),
    "");
} // namespace test_find_first_of_overload1

namespace test_find_first_of_overload2 {
// 静态断言，验证在空字符串视图中查找字符 'a' 的首次出现位置为未找到
static_assert(string_view::npos == string_view("").find_first_of('a'), "");
// 静态断言，确保字符串视图中第一个出现的字符索引为0
static_assert(0 == string_view("a").find_first_of('a'), "");

// 静态断言，确保字符串视图中第一个出现的字符索引为0
static_assert(0 == string_view("abc").find_first_of('a'), "");

// 静态断言，确保从索引1开始查找字符'a'，返回结果为string_view::npos
static_assert(string_view::npos == string_view("a").find_first_of('a', 1), "");

// 静态断言，确保字符串视图中第一个出现的字符'b'索引为1
static_assert(1 == string_view("abc").find_first_of('b'), "");

// 静态断言，确保从索引1开始查找字符'b'，返回结果为1
static_assert(1 == string_view("abc").find_first_of('b', 1), "");

// 静态断言，确保从索引2开始查找字符'b'，返回结果为string_view::npos
static_assert(string_view::npos == string_view("abc").find_first_of('b', 2), "");

// 静态断言，确保字符串视图中第一个出现的字符'c'索引为2
static_assert(2 == string_view("abc").find_first_of('c'), "");

// 静态断言，确保从索引1开始查找字符'c'，返回结果为2
static_assert(2 == string_view("abc").find_first_of('c', 1), "");

// 静态断言，确保从索引2开始查找字符'c'，返回结果为2
static_assert(2 == string_view("abc").find_first_of('c', 2), "");

// 静态断言，确保从索引3开始查找字符'c'，返回结果为string_view::npos
static_assert(string_view::npos == string_view("abc").find_first_of('c', 3), "");

// 静态断言，确保从索引100开始查找字符'a'，返回结果为string_view::npos
static_assert(string_view::npos == string_view("abc").find_first_of('a', 100), "");

// 静态断言，确保字符串视图中不存在字符'z'，返回结果为string_view::npos
static_assert(string_view::npos == string_view("abc").find_first_of('z'), "");

// 静态断言，确保字符串视图中第一个出现的字符'a'索引为0
static_assert(0 == string_view("ababa").find_first_of('a'), "");

// 静态断言，确保从索引0开始查找字符'a'，返回结果为0
static_assert(0 == string_view("ababa").find_first_of('a', 0), "");

// 静态断言，确保从索引1开始查找字符'a'，返回结果为2
static_assert(2 == string_view("ababa").find_first_of('a', 1), "");

// 静态断言，确保从索引2开始查找字符'a'，返回结果为2
static_assert(2 == string_view("ababa").find_first_of('a', 2), "");

// 静态断言，确保从索引3开始查找字符'a'，返回结果为4
static_assert(4 == string_view("ababa").find_first_of('a', 3), "");

// 静态断言，确保从索引4开始查找字符'a'，返回结果为4
static_assert(4 == string_view("ababa").find_first_of('a', 4), "");

// 静态断言，确保从索引5开始查找字符'a'，返回结果为string_view::npos
static_assert(string_view::npos == string_view("ababa").find_first_of('a', 5), "");

// 静态断言，确保空字符串中不包含字符组合"ab"，返回结果为string_view::npos
static_assert(string_view::npos == string_view("").find_first_of("ab", 0, 0), "");

// 静态断言，确保空字符串中不包含字符组合"abc"，返回结果为string_view::npos
static_assert(string_view::npos == string_view("").find_first_of("abc", 0, 1), "");

// 静态断言，确保空字符串中不包含字符组合"abcdef"，返回结果为string_view::npos
static_assert(string_view::npos == string_view("").find_first_of("abcdef", 0, 3), "");

// 静态断言，确保字符串"abc"中不包含字符组合"abcdef"，返回结果为string_view::npos
static_assert(string_view::npos == string_view("abc").find_first_of("abcdef", 0, 0), "");

// 静态断言，确保字符串"abc"中不包含字符组合"defa"，返回结果为string_view::npos
static_assert(string_view::npos == string_view("abc").find_first_of("defa", 0, 1), "");

// 静态断言，确保字符串"abc"中不包含字符组合"defabc"，返回结果为string_view::npos
static_assert(string_view::npos == string_view("abc").find_first_of("defabc", 0, 3), "");

// 静态断言，确保字符串"abcabc"中第一个出现字符组合"abc"的索引为0
static_assert(0 == string_view("abcabc").find_first_of("abc", 0, 1), "");

// 静态断言，确保字符串"abcabc"中第一个出现字符组合"bac"的索引为1
static_assert(1 == string_view("abcabc").find_first_of("bac", 0, 1), "");

// 静态断言，确保字符串"abcabc"中第一个出现字符组合"cab"的索引为2
static_assert(2 == string_view("abcabc").find_first_of("cab", 0, 1), "");

// 静态断言，确保字符串"abcabc"中第一个出现字符组合"bccda"的索引为1
static_assert(1 == string_view("abcabc").find_first_of("bccda", 0, 2), "");

// 静态断言，确保字符串"abcabc"中第一个出现字符组合"cbdab"的索引为1
static_assert(1 == string_view("abcabc").find_first_of("cbdab", 0, 3), "");

// 静态断言，确保从索引1开始，空字符串中不包含字符组合"ab"，返回结果为string_view::npos
static_assert(string_view::npos == string_view("").find_first_of("ab", 1, 0), "");

// 静态断言，确保从索引1开始，空字符串中不包含字符组合"abc"，返回结果为string_view::npos
static_assert(string_view::npos == string_view("").find_first_of("abc", 1, 1), "");

// 静态断言，确保从索引100开始，空字符串中不包含字符组合"abcdef"，返回结果为string_view::npos
static_assert(string_view::npos == string_view("").find_first_of("abcdef", 100, 3), "");

// 静态断言，确保从索引1开始，字符串"abc"中不包含字符组合"abcdef"，返回结果为string_view::npos
static_assert(string_view::npos == string_view("abc").find_first_of("abcdef", 1, 0), "");

// 静态断言，确保从索引3开始，字符串"abc"中不包含字符组合"defa"，返回结果为string_view::npos
static_assert(string_view::npos == string_view("abc").find_first_of("defa", 3, 1), "");

// 静态断言，确保从索引2开始，字符串"abc"中不包含字符组合"defabc"，返回结果为string_view::npos
static_assert(string_view::npos == string_view("abc").find_first_of("defabc", 2, 3), "");
namespace test_find_first_of_overload3 {
// 静态断言：在字符串视图 "abcabc" 中从索引 1 开始查找 "abc"，应返回索引 3
static_assert(3 == string_view("abcabc").find_first_of("abc", 1, 1), "");

// 静态断言：在字符串视图 "abcabc" 中从索引 3 开始查找 "bac"，应返回索引 4
static_assert(4 == string_view("abcabc").find_first_of("bac", 3, 1), "");

// 静态断言：在字符串视图 "abcabc" 中从索引 5 开始查找 "cab"，应返回索引 5
static_assert(5 == string_view("abcabc").find_first_of("cab", 5, 1), "");

// 静态断言：在字符串视图 "abcabc" 中从索引 3 开始查找 "bccda"，应返回索引 4
static_assert(4 == string_view("abcabc").find_first_of("bccda", 3, 2), "");

// 静态断言：在字符串视图 "abcabc" 中从索引 4 开始查找 "cbdab"，应返回索引 4
static_assert(4 == string_view("abcabc").find_first_of("cbdab", 4, 3), "");
} // namespace test_find_first_of_overload3

namespace test_find_first_of_overload4 {
// 静态断言：空字符串查找空字符串，应返回 string_view::npos
static_assert(string_view::npos == string_view("").find_first_of(""), "");

// 静态断言：空字符串查找字符 'a'，应返回 string_view::npos
static_assert(string_view::npos == string_view("").find_first_of("a"), "");

// 静态断言：空字符串查找字符串 "abc"，应返回 string_view::npos
static_assert(string_view::npos == string_view("").find_first_of("abc"), "");

// 静态断言：字符串 "abc" 中查找空字符串，应返回 string_view::npos
static_assert(string_view::npos == string_view("abc").find_first_of(""), "");

// 静态断言：字符串 "abc" 中查找字符 'd'，应返回 string_view::npos
static_assert(string_view::npos == string_view("abc").find_first_of("d"), "");

// 静态断言：字符串 "abc" 中查找字符串 "def"，应返回 string_view::npos
static_assert(string_view::npos == string_view("abc").find_first_of("def"), "");

// 静态断言：在字符串视图 "abcabc" 中查找字符 'a'，应返回索引 0
static_assert(0 == string_view("abcabc").find_first_of("a"), "");

// 静态断言：在字符串视图 "abcabc" 中查找字符 'b'，应返回索引 1
static_assert(1 == string_view("abcabc").find_first_of("b"), "");

// 静态断言：在字符串视图 "abcabc" 中查找字符 'c'，应返回索引 2
static_assert(2 == string_view("abcabc").find_first_of("c"), "");

// 静态断言：在字符串视图 "abcabc" 中查找字符串 "bc"，应返回索引 1
static_assert(1 == string_view("abcabc").find_first_of("bc"), "");

// 静态断言：在字符串视图 "abcabc" 中查找字符串 "cbd"，应返回索引 1
static_assert(1 == string_view("abcabc").find_first_of("cbd"), "");

// 静态断言：空字符串从索引 1 开始查找空字符串，应返回 string_view::npos
static_assert(string_view::npos == string_view("").find_first_of("", 1), "");

// 静态断言：空字符串从索引 1 开始查找字符 'a'，应返回 string_view::npos
static_assert(string_view::npos == string_view("").find_first_of("a", 1), "");

// 静态断言：空字符串从索引 100 开始查找字符串 "abc"，应返回 string_view::npos
static_assert(
    string_view::npos == string_view("").find_first_of("abc", 100),
    "");

// 静态断言：字符串 "abc" 从索引 1 开始查找空字符串，应返回 string_view::npos
static_assert(string_view::npos == string_view("abc").find_first_of("", 1), "");

// 静态断言：字符串 "abc" 从索引 3 开始查找字符 'd'，应返回 string_view::npos
static_assert(
    string_view::npos == string_view("abc").find_first_of("d", 3),
    "");

// 静态断言：字符串 "abc" 从索引 2 开始查找字符串 "def"，应返回 string_view::npos
static_assert(
    string_view::npos == string_view("abc").find_first_of("def", 2),
    "");

// 静态断言：在字符串视图 "abcabc" 中从索引 1 开始查找字符 'a'，应返回索引 3
static_assert(3 == string_view("abcabc").find_first_of("a", 1), "");

// 静态断言：在字符串视图 "abcabc" 中从索引 3 开始查找字符 'b'，应返回索引 4
static_assert(4 == string_view("abcabc").find_first_of("b", 3), "");

// 静态断言：在字符串视图 "abcabc" 中从索引 5 开始查找字符 'c'，应返回索引 5
static_assert(5 == string_view("abcabc").find_first_of("c", 5), "");

// 静态断言：在字符串视图 "abcabc" 中从索引 3 开始查找字符串 "bc"，应返回索引 4
static_assert(4 == string_view("abcabc").find_first_of("bc", 3), "");

// 静态断言：在字符串视图 "abcabc" 中从索引 4 开始查找字符串 "cbd"，应返回索引 4
static_assert(4 == string_view("abcabc").find_first_of("cbd", 4), "");
} // namespace test_find_first_of_overload4

namespace test_find_last_of_overload1 {
// 静态断言：空字符串中查找空字符串，应返回 string_view::npos
static_assert(
    string_view::npos == string_view("").find_last_of(string_view("")),
    "");

// 静态断言：空字符串中查找字符 'a'，应返回 string_view::npos
static_assert(
    string_view::npos == string_view("").find_last_of(string_view("a")),
    "");

// 静态断言：空字符串中查找字符串 "abc"，应返回 string_view::npos
static_assert(
    string_view::npos == string_view("").find_last_of(string_view("abc")),
    "");

// 静态断言：字符串 "abc" 中查找空字符串，应返回 string_view::npos
static_assert(
    string_view::npos == string_view("abc").find_last_of(string_view("")),
    "");

// 静态断言：字符串 "abc" 中查找字符 'd'，应返回 string_view::npos
static_assert(
    string_view::npos == string_view("abc").find_last_of(string_view("d")),
    "");

// 静态断言：字符串 "abc" 中查找字符串 "def"，应返回 string_view::npos
static_assert(
    string_view::npos == string_view("abc").find_last_of(string_view("def")),
    "");

// 静态断言：在字符串视图 "abcabc" 中从末尾查找字符 'a'，应返回索引 3
static_assert(3 == string_view("abcabc").find_last_of(string_view("a")), "");

// 静态断言：在字符串视图 "abcabc" 中从末尾查找字符 'b'，应返回索引 4
static_assert(4 == string_view("abcabc").find_last_of(string_view("b")), "");
} // namespace test_find_last_of_overload1
// 在字符串 "abcabc" 中查找最后一个匹配子串 "c" 的位置，期望结果是索引 5
static_assert(5 == string_view("abcabc").find_last_of(string_view("c")), "");

// 在字符串 "abcabc" 中查找最后一个匹配子串 "bc" 的位置，期望结果是索引 5
static_assert(5 == string_view("abcabc").find_last_of(string_view("bc")), "");

// 在字符串 "abcabc" 中查找最后一个匹配子串 "cbd" 的位置，期望结果是索引 5
static_assert(5 == string_view("abcabc").find_last_of(string_view("cbd")), "");

// 对空字符串查找最后一个匹配子串 "" 的位置，期望结果是 string_view::npos
static_assert(
    string_view::npos == string_view("").find_last_of(string_view(""), 1),
    "");

// 对空字符串查找最后一个匹配子串 "a" 的位置，限定搜索范围为 0，期望结果是 string_view::npos
static_assert(
    string_view::npos == string_view("").find_last_of(string_view("a"), 0),
    "");

// 对空字符串查找最后一个匹配子串 "abc" 的位置，限定搜索范围为 100，期望结果是 string_view::npos
static_assert(
    string_view::npos == string_view("").find_last_of(string_view("abc"), 100),
    "");

// 在字符串 "abc" 中查找最后一个匹配子串 "" 的位置，限定搜索范围为 1，期望结果是 string_view::npos
static_assert(
    string_view::npos == string_view("abc").find_last_of(string_view(""), 1),
    "");

// 在字符串 "abc" 中查找最后一个匹配子串 "d" 的位置，限定搜索范围为 3，期望结果是 string_view::npos
static_assert(
    string_view::npos == string_view("abc").find_last_of(string_view("d"), 3),
    "");

// 在字符串 "abc" 中查找最后一个匹配子串 "def" 的位置，限定搜索范围为 2，期望结果是 string_view::npos
static_assert(
    string_view::npos == string_view("abc").find_last_of(string_view("def"), 2),
    "");

// 在字符串 "abcabc" 中查找最后一个匹配字符 'a' 的位置，限定搜索范围为 2，期望结果是索引 0
static_assert(0 == string_view("abcabc").find_last_of(string_view("a"), 2), "");

// 在字符串 "abcabc" 中查找最后一个匹配字符 'b' 的位置，限定搜索范围为 3，期望结果是索引 1
static_assert(1 == string_view("abcabc").find_last_of(string_view("b"), 3), "");

// 在字符串 "abcabc" 中查找最后一个匹配字符 'c' 的位置，限定搜索范围为 2，期望结果是索引 2
static_assert(2 == string_view("abcabc").find_last_of(string_view("c"), 2), "");

// 在字符串 "abcabc" 中查找最后一个匹配子串 "bc" 的位置，限定搜索范围为 3，期望结果是索引 2
static_assert(
    2 == string_view("abcabc").find_last_of(string_view("bc"), 3),
    "");

// 在字符串 "abcabc" 中查找最后一个匹配子串 "cbd" 的位置，限定搜索范围为 2，期望结果是索引 2
static_assert(
    2 == string_view("abcabc").find_last_of(string_view("cbd"), 2),
    "");

} // namespace test_find_last_of_overload1

namespace test_find_last_of_overload2 {

// 对空字符串查找最后一个匹配字符 'a' 的位置，期望结果是 string_view::npos
static_assert(string_view::npos == string_view("").find_last_of('a'), "");

// 在字符串 "a" 中查找最后一个匹配字符 'a' 的位置，期望结果是索引 0
static_assert(0 == string_view("a").find_last_of('a'), "");

// 在字符串 "abc" 中查找最后一个匹配字符 'a' 的位置，期望结果是索引 0
static_assert(0 == string_view("abc").find_last_of('a'), "");

// 在字符串 "a" 中查找最后一个匹配字符 'a' 的位置，限定搜索范围为 0，期望结果是索引 0
static_assert(0 == string_view("a").find_last_of('a', 0), "");

// 在字符串 "abc" 中查找最后一个匹配字符 'b' 的位置，期望结果是索引 1
static_assert(1 == string_view("abc").find_last_of('b'), "");

// 在字符串 "abc" 中查找最后一个匹配字符 'b' 的位置，限定搜索范围为 0，期望结果是 string_view::npos
static_assert(string_view::npos == string_view("abc").find_last_of('b', 0), "");

// 在字符串 "abc" 中查找最后一个匹配字符 'b' 的位置，限定搜索范围为 1，期望结果是索引 1
static_assert(1 == string_view("abc").find_last_of('b', 1), "");

// 在字符串 "abc" 中查找最后一个匹配字符 'c' 的位置，期望结果是索引 2
static_assert(2 == string_view("abc").find_last_of('c'), "");

// 在字符串 "abc" 中查找最后一个匹配字符 'c' 的位置，限定搜索范围为 0，期望结果是 string_view::npos
static_assert(string_view::npos == string_view("abc").find_last_of('c', 0), "");

// 在字符串 "abc" 中查找最后一个匹配字符 'c' 的位置，限定搜索范围为 1，期望结果是 string_view::npos
static_assert(string_view::npos == string_view("abc").find_last_of('c', 1), "");

// 在字符串 "abc" 中查找最后一个匹配字符 'c' 的位置，限定搜索范围为 2，期望结果是索引 2
static_assert(2 == string_view("abc").find_last_of('c', 2), "");

// 在字符串 "abc" 中查找最后一个匹配字符 'c' 的位置，限定搜索范围为 3，期望结果是索引 2
static_assert(2 == string_view("abc").find_last_of('c', 3), "");

// 在字符串 "abc" 中查找最后一个匹配字符 'a' 的位置，限定搜索范围为 100，期望结果是索引 0
static_assert(0 == string_view("abc").find_last_of('a', 100), "");

// 在字符串 "abc" 中查找最后一个匹配字符 'z' 的位置，期望结果是 string_view::npos
static_assert(string_view::npos == string_view("abc").find_last_of('z'), "");

// 在字符串 "ababa" 中查找最后一个匹配字符 'a' 的位置，期望结果是索引 4
static_assert(4 == string_view("ababa").find_last_of('a'), "");

// 在字符串 "ababa" 中查找最后一个匹配字符 'a' 的位置，限定搜索范围为 0，期望结果是索引 0
static_assert(0 == string_view("ababa").find_last_of('a', 0), "");

// 在字符串 "ababa" 中查找最后一个匹配字符 'a' 的位置，限定搜索范围为 1，期望结果是索引 0
static_assert(0 == string_view("ababa").find_last_of('a', 1), "");

// 在字符串 "ababa" 中查找最后一个匹配字符 'a' 的位置，限定搜索范围为 2，期望结果是索引 2
static_assert(2 == string_view("ababa").find_last_of('a', 2), "");

// 在字符串 "ababa" 中查找最后一个匹配字符 'a' 的位置，限定搜索范围为 3，期望结果是索引 2
static_assert(2 == string_view("ababa").find_last_of('a', 3), "");

// 在字符串 "ababa" 中查找最后一个
static_assert(
    string_view::npos ==
        string_view("").find_last_of("abc", string_view::npos, 1),
    "");
// 确保空字符串中找不到 "abc" 的最后一个字符，返回 string_view::npos

static_assert(
    string_view::npos ==
        string_view("").find_last_of("abcdef", string_view::npos, 3),
    "");
// 确保空字符串中找不到 "abcdef" 的最后一个字符，返回 string_view::npos

static_assert(
    string_view::npos ==
        string_view("abc").find_last_of("abcdef", string_view::npos, 0),
    "");
// 确保在字符串 "abc" 中从索引 0 开始找不到 "abcdef" 的最后一个字符，返回 string_view::npos

static_assert(
    string_view::npos ==
        string_view("abc").find_last_of("defa", string_view::npos, 1),
    "");
// 确保在字符串 "abc" 中从索引 1 开始找不到 "defa" 的最后一个字符，返回 string_view::npos

static_assert(
    string_view::npos ==
        string_view("abc").find_last_of("defcba", string_view::npos, 3),
    "");
// 确保在字符串 "abc" 中从索引 3 开始找不到 "defcba" 的最后一个字符，返回 string_view::npos

static_assert(
    3 == string_view("abcabc").find_last_of("abc", string_view::npos, 1),
    "");
// 确保在字符串 "abcabc" 中从索引 1 开始能找到 "abc" 的最后一个字符，返回索引 3

static_assert(
    4 == string_view("abcabc").find_last_of("bca", string_view::npos, 1),
    "");
// 确保在字符串 "abcabc" 中从索引 1 开始能找到 "bca" 的最后一个字符，返回索引 4

static_assert(
    5 == string_view("abcabc").find_last_of("cab", string_view::npos, 1),
    "");
// 确保在字符串 "abcabc" 中从索引 1 开始能找到 "cab" 的最后一个字符，返回索引 5

static_assert(
    5 == string_view("abcabc").find_last_of("bcab", string_view::npos, 2),
    "");
// 确保在字符串 "abcabc" 中从索引 2 开始能找到 "bcab" 的最后一个字符，返回索引 5

static_assert(
    5 == string_view("abcabc").find_last_of("cbdac", string_view::npos, 3),
    "");
// 确保在字符串 "abcabc" 中从索引 3 开始能找到 "cbdac" 的最后一个字符，返回索引 5

static_assert(
    string_view::npos == string_view("").find_last_of("ab", 1, 0),
    "");
// 确保使用错误的参数 (1, 0) 时，无法在空字符串中找到 "ab" 的最后一个字符，返回 string_view::npos

static_assert(
    string_view::npos == string_view("").find_last_of("abc", 0, 1),
    "");
// 确保使用错误的参数 (0, 1) 时，无法在空字符串中找到 "abc" 的最后一个字符，返回 string_view::npos

static_assert(
    string_view::npos == string_view("").find_last_of("abcdef", 100, 3),
    "");
// 确保使用错误的参数 (100, 3) 时，无法在空字符串中找到 "abcdef" 的最后一个字符，返回 string_view::npos

static_assert(
    string_view::npos == string_view("abc").find_last_of("abcdef", 1, 0),
    "");
// 确保使用错误的参数 (1, 0) 时，无法在字符串 "abc" 中找到 "abcdef" 的最后一个字符，返回 string_view::npos

static_assert(
    string_view::npos == string_view("abc").find_last_of("defa", 3, 1),
    "");
// 确保使用错误的参数 (3, 1) 时，无法在字符串 "abc" 中找到 "defa" 的最后一个字符，返回 string_view::npos

static_assert(
    string_view::npos == string_view("abc").find_last_of("defcba", 2, 3),
    "");
// 确保使用错误的参数 (2, 3) 时，无法在字符串 "abc" 中找到 "defcba" 的最后一个字符，返回 string_view::npos

static_assert(
    0 == string_view("abcabc").find_last_of("abc", 2, 1), "");
// 在字符串 "abcabc" 中从索引 2 开始找到 "abc" 的最后一个字符，返回索引 0

static_assert(
    1 == string_view("abcabc").find_last_of("bca", 3, 1), "");
// 在字符串 "abcabc" 中从索引 3 开始找到 "bca" 的最后一个字符，返回索引 1

static_assert(
    2 == string_view("abcabc").find_last_of("cab", 2, 1), "");
// 在字符串 "abcabc" 中从索引 2 开始找到 "cab" 的最后一个字符，返回索引 2

static_assert(
    2 == string_view("abcabc").find_last_of("bcab", 3, 2), "");
// 在字符串 "abcabc" 中从索引 3 开始找到 "bcab" 的最后一个字符，返回索引 2

static_assert(
    2 == string_view("abcabc").find_last_of("cbdac", 2, 2), "");
// 在字符串 "abcabc" 中从索引 2 开始找到 "cbdac" 的最后一个字符，返回索引 2
// 断言：在空字符串中查找最后一个 "a" 的位置，应该返回 npos
static_assert(string_view::npos == string_view("").find_last_of("a", 0), "");

// 断言：在空字符串中从索引 100 开始查找最后一个 "abc" 的位置，应该返回 npos
static_assert(
    string_view::npos == string_view("").find_last_of("abc", 100),
    "");

// 断言：在字符串 "abc" 中查找最后一个空字符串的位置，应该返回 npos
static_assert(string_view::npos == string_view("abc").find_last_of("", 1), "");

// 断言：在字符串 "abc" 中查找最后一个 "d" 的位置，应该返回 npos
static_assert(string_view::npos == string_view("abc").find_last_of("d", 3), "");

// 断言：在字符串 "abc" 中从索引 2 开始查找最后一个 "def" 的位置，应该返回 npos
static_assert(
    string_view::npos == string_view("abc").find_last_of("def", 2),
    "");

// 断言：在字符串 "abcabc" 中从索引 2 开始查找最后一个 "a" 的位置，应该返回 0
static_assert(0 == string_view("abcabc").find_last_of("a", 2), "");

// 断言：在字符串 "abcabc" 中从索引 3 开始查找最后一个 "b" 的位置，应该返回 1
static_assert(1 == string_view("abcabc").find_last_of("b", 3), "");

// 断言：在字符串 "abcabc" 中从索引 2 开始查找最后一个 "c" 的位置，应该返回 2
static_assert(2 == string_view("abcabc").find_last_of("c", 2), "");

// 断言：在字符串 "abcabc" 中从索引 3 开始查找最后一个 "bc" 的位置，应该返回 2
static_assert(2 == string_view("abcabc").find_last_of("bc", 3), "");

// 断言：在字符串 "abcabc" 中从索引 2 开始查找最后一个 "cbd" 的位置，应该返回 2
static_assert(2 == string_view("abcabc").find_last_of("cbd", 2), "");

// 闭合 test_find_last_of_overload4 命名空间
} // namespace test_find_last_of_overload4

// 开始 test_find_first_not_of_overload1 命名空间
namespace test_find_first_not_of_overload1 {

// 断言：在空字符串中查找第一个不在空视图中的字符，应该返回 npos
static_assert(
    string_view::npos == string_view("").find_first_not_of(string_view("")),
    "");

// 断言：在空字符串中查找第一个不在 "a" 视图中的字符，应该返回 npos
static_assert(
    string_view::npos == string_view("").find_first_not_of(string_view("a")),
    "");

// 断言：在空字符串中查找第一个不在 "abc" 视图中的字符，应该返回 npos
static_assert(
    string_view::npos == string_view("").find_first_not_of(string_view("abc")),
    "");

// 断言：在字符串 "abc" 中查找第一个不在 "abc" 视图中的字符，应该返回 npos
static_assert(
    string_view::npos ==
        string_view("abc").find_first_not_of(string_view("abc")),
    "");

// 断言：在字符串 "abc" 中查找第一个不在 "acdb" 视图中的字符，应该返回 npos
static_assert(
    string_view::npos ==
        string_view("abc").find_first_not_of(string_view("acdb")),
    "");

// 断言：在字符串 "abc" 中查找第一个不在 "defabc" 视图中的字符，应该返回 npos
static_assert(
    string_view::npos ==
        string_view("abc").find_first_not_of(string_view("defabc")),
    "");

// 断言：在字符串 "abcabc" 中查找第一个不在空视图中的字符，应该返回 0
static_assert(
    0 == string_view("abcabc").find_first_not_of(string_view("")),
    "");

// 断言：在字符串 "abcabc" 中查找第一个不在 "bc" 视图中的字符，应该返回 0
static_assert(
    0 == string_view("abcabc").find_first_not_of(string_view("bc")),
    "");

// 断言：在字符串 "abcabc" 中查找第一个不在 "ac" 视图中的字符，应该返回 1
static_assert(
    1 == string_view("abcabc").find_first_not_of(string_view("ac")),
    "");

// 断言：在字符串 "abcabc" 中查找第一个不在 "ab" 视图中的字符，应该返回 2
static_assert(
    2 == string_view("abcabc").find_first_not_of(string_view("ab")),
    "");

// 断言：在字符串 "abcabc" 中查找第一个不在 "a" 视图中的字符，应该返回 1
static_assert(
    1 == string_view("abcabc").find_first_not_of(string_view("a")),
    "");

// 断言：在字符串 "abcabc" 中查找第一个不在 "da" 视图中的字符，应该返回 1
static_assert(
    1 == string_view("abcabc").find_first_not_of(string_view("da")),
    "");

// 断言：在空字符串中从索引 1 开始查找第一个不在空视图中的字符，应该返回 npos
static_assert(
    string_view::npos == string_view("").find_first_not_of(string_view(""), 1),
    "");

// 断言：在空字符串中从索引 1 开始查找第一个不在 "a" 视图中的字符，应该返回 npos
static_assert(
    string_view::npos == string_view("").find_first_not_of(string_view("a"), 1),
    "");

// 断言：在空字符串中从索引 100 开始查找第一个不在 "abc" 视图中的字符，应该返回 npos
static_assert(
    string_view::npos ==
        string_view("").find_first_not_of(string_view("abc"), 100),
    "");

// 断言：在字符串 "abc" 中从索引 1 开始查找第一个不在 "abc" 视图中的字符，应该返回 npos
static_assert(
    string_view::npos ==
        string_view("abc").find_first_not_of(string_view("abc"), 1),
    "");

// 断言：在字符串 "abc" 中从索引 3 开始查找第一个不在 "acdb" 视图中的字符，应该返回 npos
static_assert(
    string_view::npos ==
        string_view("abc").find_first_not_of(string_view("acdb"), 3),
    "");

// 断言：在字符串 "abc" 中从索引 2 开始查找第一个不在 "defabc" 视图中的字符，应该返回 npos
static_assert(
    string_view::npos ==
        string_view("abc").find_first_not_of(string_view("defabc"), 2),
    "");

// 断言：在字符串 "abcabc" 中从索引 1 开始查找第一个不在空视图中的字符，应该返回 1
static_assert(
    1 == string_view("abcabc").find_first_not_of(string_view(""), 1),
    "");

// 断言：在字符串 "abcabc" 中从索引 1 开始查找第一个不在 "bc" 视图中的字符，应该返回 3
static_assert(
    3 == string_view("abcabc").find_first_not_of(string_view("bc"), 1),
    "");

// 闭合 test_find_first_not_of_overload1 命名空间
} // namespace test_find_first_not_of_overload1
    4 == string_view("abcabc").find_first_not_of(string_view("ac"), 4),
    "");



// 检查在字符串视图 "abcabc" 中，从索引位置 4 开始，第一个不匹配字符串视图 "ac" 中任何字符的位置
4 == string_view("abcabc").find_first_not_of(string_view("ac"), 4),
// 空字符串，用作断言的消息，通常用于标识预期失败的情况
"");


这段代码使用了C++中的字符串视图和查找函数来执行特定的字符串操作。
static_assert(
    // 使用 string_view 对象查找在指定位置之后首个不属于字符串 "ab" 的字符索引，预期结果是 5
    5 == string_view("abcabc").find_first_not_of(string_view("ab"), 5),
    "");

static_assert(
    // 使用 string_view 对象查找在指定位置之后首个不属于字符 'a' 的字符索引，预期结果是 4
    4 == string_view("abcabc").find_first_not_of(string_view("a"), 3),
    "");

static_assert(
    // 使用 string_view 对象查找在指定位置之后首个不属于字符串 "da" 的字符索引，预期结果是 4
    4 == string_view("abcabc").find_first_not_of(string_view("da"), 4),
    "");
} // namespace test_find_first_not_of_overload1

namespace test_find_first_not_of_overload2 {
static_assert(string_view::npos == string_view("").find_first_not_of('a'), "");
static_assert(string_view::npos == string_view("a").find_first_not_of('a'), "");

static_assert(
    // 使用 string_view 对象查找首个不属于字符 'a' 的字符索引，预期结果是 1
    1 == string_view("abc").find_first_not_of('a'), "");

static_assert(
    string_view::npos == string_view("a").find_first_not_of('a', 1),
    "");

static_assert(
    // 使用 string_view 对象查找首个不属于字符 'b' 的字符索引，预期结果是 0
    0 == string_view("abc").find_first_not_of('b'), "");

static_assert(
    // 使用 string_view 对象查找在指定位置之后首个不属于字符 'b' 的字符索引，预期结果是 2
    2 == string_view("abc").find_first_not_of('b', 1), "");

static_assert(
    // 使用 string_view 对象查找在指定位置之后首个不属于字符 'b' 的字符索引，预期结果是 2
    2 == string_view("abc").find_first_not_of('b', 2), "");

static_assert(
    string_view::npos == string_view("abc").find_first_not_of('b', 3),
    "");

static_assert(
    // 使用 string_view 对象查找首个不属于字符 'c' 的字符索引，预期结果是 0
    0 == string_view("abc").find_first_not_of('c'), "");

static_assert(
    // 使用 string_view 对象查找在指定位置之后首个不属于字符 'c' 的字符索引，预期结果是 1
    1 == string_view("abc").find_first_not_of('c', 1), "");

static_assert(
    string_view::npos == string_view("abc").find_first_not_of('c', 2),
    "");

static_assert(
    string_view::npos == string_view("abc").find_first_not_of('c', 3),
    "");

static_assert(
    string_view::npos == string_view("abc").find_first_not_of('a', 100),
    "");

static_assert(
    // 使用 string_view 对象查找首个不属于字符 'a' 的字符索引，预期结果是 1
    1 == string_view("ababa").find_first_not_of('a'), "");

static_assert(
    // 使用 string_view 对象查找在指定位置之后首个不属于字符 'a' 的字符索引，预期结果是 1
    1 == string_view("ababa").find_first_not_of('a', 0), "");

static_assert(
    // 使用 string_view 对象查找在指定位置之后首个不属于字符 'a' 的字符索引，预期结果是 1
    1 == string_view("ababa").find_first_not_of('a', 1), "");

static_assert(
    // 使用 string_view 对象查找在指定位置之后首个不属于字符 'a' 的字符索引，预期结果是 3
    3 == string_view("ababa").find_first_not_of('a', 2), "");

static_assert(
    string_view::npos == string_view("ababa").find_first_not_of('a', 4),
    "");

static_assert(
    string_view::npos == string_view("ababa").find_first_not_of('a', 5),
    "");
} // namespace test_find_first_not_of_overload2

namespace test_find_first_not_of_overload3 {
static_assert(
    // 使用 string_view 对象查找首个不属于字符串 "ab" 的字符索引，预期结果是 string_view::npos
    string_view::npos == string_view("").find_first_not_of("ab", 0, 0),
    "");

static_assert(
    // 使用 string_view 对象查找首个不属于字符串 "abc" 的字符索引，预期结果是 string_view::npos
    string_view::npos == string_view("").find_first_not_of("abc", 0, 1),
    "");

static_assert(
    // 使用 string_view 对象查找首个不属于字符串 "abcdef" 的字符索引，预期结果是 string_view::npos
    string_view::npos == string_view("").find_first_not_of("abcdef", 0, 3),
    "");

static_assert(
    // 使用 string_view 对象查找首个不属于字符串 "abcdef" 的字符索引，预期结果是 string_view::npos
    string_view::npos == string_view("abc").find_first_not_of("abcdef", 0, 3),
    "");

static_assert(
    // 使用 string_view 对象查找首个不属于字符串 "acdbef" 的字符索引，预期结果是 string_view::npos
    string_view::npos == string_view("abc").find_first_not_of("acdbef", 0, 4),
    "");

static_assert(
    // 使用 string_view 对象查找首个不属于字符串 "defabcas" 的字符索引，预期结果是 string_view::npos
    string_view::npos == string_view("abc").find_first_not_of("defabcas", 0, 6),
    "");

static_assert(
    // 使用 string_view 对象查找在指定位置之后首个不属于字符串 "abc" 的字符索引，预期结果是 0
    0 == string_view("abcabc").find_first_not_of("abc", 0, 0),
    "");

static_assert(
    // 使用 string_view 对象查找在指定位置之后首个不属于字符串 "bca" 的字符索引，预期结果是 0
    0 == string_view("abcabc").find_first_not_of("bca", 0, 2),
    "");

static_assert(
    // 使用 string_view 对象查找在指定位置之后首个不属于字符串 "acb" 的字符索引，预期结果是 1
    1 == string_view("abcabc").find_first_not_of("acb", 0, 2),
    "");

static_assert(
    // 使用 string_view 对象查找在指定位置之后首个不属于字符串 "abc" 的字符索引，预期结果是 2
    2 == string_view("abcabc").find_first_not_of("abc", 0, 2),
    "");
static_assert(1 == string_view("abcabc").find_first_not_of("abac", 0, 1), "");
# 在字符串 "abcabc" 中，从索引 0 开始查找，找到第一个不在 "abac" 中的字符，应为索引 1 处的字符 'b'

static_assert(1 == string_view("abcabc").find_first_not_of("dadab", 0, 2), "");
# 在字符串 "abcabc" 中，从索引 0 开始查找，找到第一个不在 "dadab" 中的字符，应为索引 1 处的字符 'b'

static_assert(
    string_view::npos == string_view("").find_first_not_of("ab", 1, 0),
    "");
# 在空字符串中，从索引 1 开始查找，未找到任何不在 "ab" 中的字符，返回 string_view::npos

static_assert(
    string_view::npos == string_view("").find_first_not_of("abc", 1, 1),
    "");
# 在空字符串中，从索引 1 开始查找，未找到任何不在 "abc" 中的字符，返回 string_view::npos

static_assert(
    string_view::npos == string_view("").find_first_not_of("abcdef", 100, 3),
    "");
# 在空字符串中，从索引 100 开始查找，未找到任何不在 "abcdef" 中的字符，返回 string_view::npos

static_assert(
    string_view::npos == string_view("abc").find_first_not_of("abcdef", 1, 3),
    "");
# 在字符串 "abc" 中，从索引 1 开始查找，未找到任何不在 "abcdef" 中的字符，返回 string_view::npos

static_assert(
    string_view::npos == string_view("abc").find_first_not_of("acdbef", 3, 4),
    "");
# 在字符串 "abc" 中，从索引 3 开始查找，未找到任何不在 "acdbef" 中的字符，返回 string_view::npos

static_assert(
    string_view::npos == string_view("abc").find_first_not_of("defabcas", 2, 6),
    "");
# 在字符串 "abc" 中，从索引 2 开始查找，未找到任何不在 "defabcas" 中的字符，返回 string_view::npos

static_assert(1 == string_view("abcabc").find_first_not_of("bca", 1, 0), "");
# 在字符串 "abcabc" 中，从索引 1 开始查找，找到第一个不在 "bca" 中的字符，应为索引 1 处的字符 'b'

static_assert(3 == string_view("abcabc").find_first_not_of("bca", 1, 2), "");
# 在字符串 "abcabc" 中，从索引 1 开始查找，找到第一个不在 "bca" 中的字符，应为索引 3 处的字符 'a'

static_assert(4 == string_view("abcabc").find_first_not_of("acb", 4, 2), "");
# 在字符串 "abcabc" 中，从索引 4 开始查找，找到第一个不在 "acb" 中的字符，应为索引 4 处的字符 'c'

static_assert(5 == string_view("abcabc").find_first_not_of("abc", 5, 2), "");
# 在字符串 "abcabc" 中，从索引 5 开始查找，找到第一个不在 "abc" 中的字符，应为索引 5 处的结束符

static_assert(4 == string_view("abcabc").find_first_not_of("abac", 3, 1), "");
# 在字符串 "abcabc" 中，从索引 3 开始查找，找到第一个不在 "abac" 中的字符，应为索引 4 处的字符 'c'

static_assert(4 == string_view("abcabc").find_first_not_of("dadab", 4, 2), "");
# 在字符串 "abcabc" 中，从索引 4 开始查找，找到第一个不在 "dadab" 中的字符，应为索引 4 处的字符 'c'
namespace test_find_first_not_of_overload4 {

// 第一个断言：在字符串 "abcabc" 中从索引 1 开始查找不是 'bc' 中任何字符的位置，预期结果是 3
static_assert(3 == string_view("abcabc").find_first_not_of("bc", 1), "");

// 第二个断言：在字符串 "abcabc" 中从索引 4 开始查找不是 'ac' 中任何字符的位置，预期结果是 4
static_assert(4 == string_view("abcabc").find_first_not_of("ac", 4), "");

// 第三个断言：在字符串 "abcabc" 中从索引 5 开始查找不是 'ab' 中任何字符的位置，预期结果是 5
static_assert(5 == string_view("abcabc").find_first_not_of("ab", 5), "");

// 第四个断言：在字符串 "abcabc" 中从索引 3 开始查找不是 'a' 中任何字符的位置，预期结果是 4
static_assert(4 == string_view("abcabc").find_first_not_of("a", 3), "");

// 第五个断言：在字符串 "abcabc" 中从索引 4 开始查找不是 'da' 中任何字符的位置，预期结果是 4
static_assert(4 == string_view("abcabc").find_first_not_of("da", 4), "");

} // namespace test_find_first_not_of_overload4

namespace test_find_last_not_of_overload1 {

// 第一个断言：在空字符串中查找最后一个不是空字符串的字符，预期结果是 string_view::npos
static_assert(
    string_view::npos == string_view("").find_last_not_of(string_view("")),
    "");

// 第二个断言：在空字符串中查找最后一个不是 'a' 的字符，预期结果是 string_view::npos
static_assert(
    string_view::npos == string_view("").find_last_not_of(string_view("a")),
    "");

// 第三个断言：在空字符串中查找最后一个不是 'abc' 的字符，预期结果是 string_view::npos
static_assert(
    string_view::npos == string_view("").find_last_not_of(string_view("abc")),
    "");

// 第四个断言：在字符串 "abc" 中查找最后一个不是 'abc' 的字符，预期结果是 string_view::npos
static_assert(
    string_view::npos == string_view("abc").find_last_not_of(string_view("abc")),
    "");

// 第五个断言：在字符串 "abc" 中查找最后一个不是 'acdb' 的字符，预期结果是 string_view::npos
static_assert(
    string_view::npos == string_view("abc").find_last_not_of(string_view("acdb")),
    "");

// 第六个断言：在字符串 "abc" 中查找最后一个不是 'defabc' 的字符，预期结果是 string_view::npos
static_assert(
    string_view::npos == string_view("abc").find_last_not_of(string_view("defabc")),
    "");

// 第七个断言：在字符串 "abcabc" 中查找最后一个不是空字符串的字符，预期结果是 5
static_assert(5 == string_view("abcabc").find_last_not_of(string_view("")), "");

// 第八个断言：在字符串 "abcabc" 中查找最后一个不是 'bc' 的字符，预期结果是 3
static_assert(
    3 == string_view("abcabc").find_last_not_of(string_view("bc")),
    "");

// 第九个断言：在字符串 "abcabc" 中查找最后一个不是 'ac' 的字符，预期结果是 4
static_assert(
    4 == string_view("abcabc").find_last_not_of(string_view("ac")),
    "");

// 第十个断言：在字符串 "abcabc" 中查找最后一个不是 'ab' 的字符，预期结果是 5
static_assert(
    5 == string_view("abcabc").find_last_not_of(string_view("ab")),
    "");

// 第十一个断言：在字符串 "abcabc" 中查找最后一个不是 'c' 的字符，预期结果是 4
static_assert(
    4 == string_view("abcabc").find_last_not_of(string_view("c")),
    "");

// 第十二个断言：在字符串 "abcabc" 中查找最后一个不是 'ca' 的字符，预期结果是 4
static_assert(
    4 == string_view("abcabc").find_last_not_of(string_view("ca")),
    "");

// 第十三个断言：在空字符串中从索引 1 开始查找最后一个不是空字符串的字符，预期结果是 string_view::npos
static_assert(
    string_view::npos == string_view("").find_last_not_of(string_view(""), 1),
    "");

// 第十四个断言：在空字符串中从索引 0 开始查找最后一个不是 'a' 的字符，预期结果是 string_view::npos
static_assert(
    string_view::npos == string_view("").find_last_not_of(string_view("a"), 0),
    "");

// 第十五个断言：在空字符串中从索引 100 开始查找最后一个不是 'abc' 的字符，预期结果是 string_view::npos
static_assert(
    string_view::npos ==
        string_view("").find_last_not_of(string_view("abc"), 100),
    "");

// 第十六个断言：在字符串 "abc" 中从索引 1 开始查找最后一个不是 'abc' 的字符，预期结果是 string_view::npos
static_assert(
    string_view::npos ==
        string_view("abc").find_last_not_of(string_view("abc"), 1),
    "");

// 第十七个断言：在字符串 "abc" 中从索引 3 开始查找最后一个不是 'acdb' 的字符，预期结果是 string_view::npos
static_assert(
    string_view::npos ==
        string_view("abc").find_last_not_of(string_view("acdb"), 3),
    "");

// 第十八个断言：在字符串 "abc" 中从索引 2 开始查找最后一个不是 'defabc' 的字符，预期结果是 string_view::npos
static_assert(
    string_view::npos ==
        string_view("abc").find_last_not_of(string_view("defabc"), 2),
    "");

// 第十九个断言：在字符串 "abcabc" 中从索引 4 开始查找最后一个不是空字符串的字符，预期结果是 4
static_assert(
    4 == string_view("abcabc").find_last_not_of(string_view(""), 4),
    "");

// 第二十个断言：在字符串 "abcabc" 中从索引 2 开始查找最后一个不是 'bc' 的字符，预期结果是 0
static_assert(
    0 == string_view("abcabc").find_last_not_of(string_view("bc"), 2),
    "");

// 第二十一个断言：在字符串 "abcabc" 中从索引 2 开始查找最后一个不是 'ac' 的字符，预期结果是 1
static_assert(
    1 == string_view("abcabc").find_last_not_of(string_view("ac"), 2),
    "");

// 第二十二个断言：在字符串 "abcabc" 中从索引 2 开始查找最后一个不是 'ab' 的字符，预期结果是 2
static_assert(
    2 == string_view("abcabc").find_last_not_of(string_view("ab"), 2),
    "");

// 第二十三个断言：在字符串 "abcabc" 中从索引 4 开始查找最后一个不是 'c' 的字符，预期结果是 4
static_assert(
    4 == string_view("abcabc").find_last_not_of(string_view("c"), 4),
    "");

// 第二十四个断言：在字符串 "abcabc" 中从索引 2 开始查找最后一个不是 'ca' 的字符，预期结果是 1
static_assert(
    1 == string_view("abcabc").find_last_not_of(string_view("ca"), 2),
    "");

} // namespace test_find_last_not_of_overload1
// 确保空的 string_view 中查找最后一个不等于 'a' 的位置是 npos
static_assert(string_view::npos == string_view("").find_last_not_of('a'), "");

// 确保只包含一个 'a' 的 string_view 中查找最后一个不等于 'a' 的位置是 npos
static_assert(string_view::npos == string_view("a").find_last_not_of('a'), "");

// 确保 "abc" 中查找最后一个不等于 'a' 的位置是 2
static_assert(2 == string_view("abc").find_last_not_of('a'), "");

// 确保 "abc" 中查找最后一个不等于 'c' 的位置是 1
static_assert(1 == string_view("abc").find_last_not_of('c'), "");

// 确保 "a" 中在第 0 位置开始查找最后一个不等于 'a' 的位置是 npos
static_assert(string_view::npos == string_view("a").find_last_not_of('a', 0), "");

// 确保 "abc" 中查找最后一个不等于 'b' 的位置是 2
static_assert(2 == string_view("abc").find_last_not_of('b'), "");

// 确保 "abc" 中在第 0 位置开始查找最后一个不等于 'a' 的位置是 npos
static_assert(string_view::npos == string_view("abc").find_last_not_of('a', 0), "");

// 确保 "abc" 中在第 1 位置之前查找最后一个不等于 'b' 的位置是 0
static_assert(0 == string_view("abc").find_last_not_of('b', 1), "");

// 确保 "abc" 中在第 0 位置之前查找最后一个不等于 'c' 的位置是 0
static_assert(0 == string_view("abc").find_last_not_of('c', 0), "");

// 确保 "abc" 中在第 1 位置之前查找最后一个不等于 'c' 的位置是 1
static_assert(1 == string_view("abc").find_last_not_of('c', 1), "");

// 确保 "abc" 中在第 2 位置之前查找最后一个不等于 'c' 的位置是 1
static_assert(1 == string_view("abc").find_last_not_of('c', 2), "");

// 确保 "abc" 中在第 3 位置之前查找最后一个不等于 'c' 的位置是 1
static_assert(1 == string_view("abc").find_last_not_of('c', 3), "");

// 确保 "abc" 中在第 100 位置之前查找最后一个不等于 'a' 的位置是 2
static_assert(2 == string_view("abc").find_last_not_of('a', 100), "");

// 确保 "ababa" 中查找最后一个不等于 'a' 的位置是 3
static_assert(3 == string_view("ababa").find_last_not_of('a'), "");

// 确保 "ababa" 中在第 0 位置之前查找最后一个不等于 'a' 的位置是 npos
static_assert(string_view::npos == string_view("ababa").find_last_not_of('a', 0), "");

// 确保 "ababa" 中在第 1 位置之前查找最后一个不等于 'a' 的位置是 1
static_assert(1 == string_view("ababa").find_last_not_of('a', 1), "");

// 确保 "ababa" 中在第 2 位置之前查找最后一个不等于 'a' 的位置是 1
static_assert(1 == string_view("ababa").find_last_not_of('a', 2), "");

// 确保 "ababa" 中在第 3 位置之前查找最后一个不等于 'a' 的位置是 3
static_assert(3 == string_view("ababa").find_last_not_of('a', 3), "");

// 确保 "ababa" 中在第 4 位置之前查找最后一个不等于 'a' 的位置是 3
static_assert(3 == string_view("ababa").find_last_not_of('a', 4), "");

// 确保 "ababa" 中在第 5 位置之前查找最后一个不等于 'a' 的位置是 3
static_assert(3 == string_view("ababa").find_last_not_of('a', 5), "");

// namespace test_find_last_not_of_overload2 的结束标记
} // namespace test_find_last_not_of_overload2

namespace test_find_last_not_of_overload3 {

// 确保空的 string_view 中查找最后一个不包含在 "ab" 中的字符的位置是 npos
static_assert(
    string_view::npos ==
        string_view("").find_last_not_of("ab", string_view::npos, 0),
    "");

// 确保空的 string_view 中查找最后一个不包含在 "abc" 中的字符的位置是 npos
static_assert(
    string_view::npos ==
        string_view("").find_last_not_of("abc", string_view::npos, 1),
    "");

// 确保空的 string_view 中查找最后一个不包含在 "abcdef" 中的字符的位置是 npos
static_assert(
    string_view::npos ==
        string_view("").find_last_not_of("abcdef", string_view::npos, 3),
    "");

// 确保 "abc" 中查找最后一个不包含在 "abcdef" 中的字符的位置是 npos
static_assert(
    string_view::npos ==
        string_view("abc").find_last_not_of("abcdef", string_view::npos, 3),
    "");

// 确保 "abc" 中查找最后一个不包含在 "acdbef" 中的字符的位置是 npos
static_assert(
    string_view::npos ==
        string_view("abc").find_last_not_of("acdbef", string_view::npos, 4),
    "");

// 确保 "abc" 中查找最后一个不包含在 "defabcas" 中的字符的位置是 npos
static_assert(
    string_view::npos ==
        string_view("abc").find_last_not_of("defabcas", string_view::npos, 6),
    "");

// 确保 "abcabc" 中查找最后一个不包含在 "cab" 中的字符的位置是 5
static_assert(
    5 == string_view("abcabc").find_last_not_of("cab", string_view::npos, 0),
    "");

// 确保 "abcabc" 中查找最后一个不包含在 "bca" 中的字符的位置是 3
static_assert(
    3 == string_view("abcabc").find_last_not_of("bca", string_view::npos, 2),
    "");

// 确保 "abcabc" 中查找最后一个不包含在 "acb" 中的字符的位置是 4
static_assert(
    4 == string_view("abcabc").find_last_not_of("acb", string_view::npos, 2),
    "");

// 确保 "abcabc" 中查找最后一个不包含在 "abc" 中的字符的位置是 5
static_assert(
    5 == string_view("abcabc").find_last_not_of("abc", string_view::npos, 2),
    "");

// 确保 "abcabc" 中查找最后一个不包含在 "caba" 中的字符的位置是 4
static_assert(
    4 == string_view("abcabc").find_last_not_of("caba", string_view::npos, 1),
    "");

// 确保 "abcabc" 中查找最后一个不包含在 "cacab" 中的字符的位置是 4
static_assert(
    4 == string_view("abcabc").find_last_not_of("cacab", string_view::npos, 2),
    "");

// namespace test_find_last_not_of_overload3 的结束标记
static_assert(
    # 使用 string_view 类的静态成员变量 npos，表示未找到时的返回值
    # 在空的 string_view 中查找最后一个不属于字符集 "ab" 的字符，从索引 1 开始向前查找，最大字符数为 0
    string_view::npos == string_view("").find_last_not_of("ab", 1, 0),
    # 断言空的 string_view 的结果为 true，表示未找到不属于字符集 "ab" 的字符
    "");
static_assert(
    string_view::npos == string_view("").find_last_not_of("abc", 0, 1),
    "");
# 空字符串查找最后一个不是"abc"字符集合的位置，预期结果是npos

static_assert(
    string_view::npos == string_view("").find_last_not_of("abcdef", 100, 3),
    "");
# 空字符串查找最后一个不是"abcdef"字符集合的位置，预期结果是npos

static_assert(
    string_view::npos == string_view("abc").find_last_not_of("abcdef", 1, 3),
    "");
# "abc"字符串在索引1开始查找最后一个不是"abcdef"字符集合的位置，预期结果是npos

static_assert(
    string_view::npos == string_view("abc").find_last_not_of("acdbef", 3, 4),
    "");
# "abc"字符串在索引3开始查找最后一个不是"acdbef"字符集合的位置，预期结果是npos

static_assert(
    string_view::npos == string_view("abc").find_last_not_of("defabcas", 2, 6),
    "");
# "abc"字符串在索引2开始查找最后一个不是"defabcas"字符集合的位置，预期结果是npos

static_assert(4 == string_view("abcabc").find_last_not_of("bca", 4, 0), "");
# "abcabc"字符串在索引4开始向左查找最后一个不是"bca"字符集合的位置，预期结果是索引4

static_assert(0 == string_view("abcabc").find_last_not_of("bca", 2, 2), "");
# "abcabc"字符串在索引2开始向左查找最后一个不是"bca"字符集合的位置，预期结果是索引0

static_assert(1 == string_view("abcabc").find_last_not_of("acb", 2, 2), "");
# "abcabc"字符串在索引2开始向左查找最后一个不是"acb"字符集合的位置，预期结果是索引1

static_assert(2 == string_view("abcabc").find_last_not_of("abc", 2, 2), "");
# "abcabc"字符串在索引2开始向左查找最后一个不是"abc"字符集合的位置，预期结果是索引2

static_assert(4 == string_view("abcabc").find_last_not_of("caba", 4, 1), "");
# "abcabc"字符串在索引4开始向左查找最后一个不是"caba"字符集合的位置，预期结果是索引4

static_assert(1 == string_view("abcabc").find_last_not_of("cacab", 2, 2), "");
# "abcabc"字符串在索引2开始向左查找最后一个不是"cacab"字符集合的位置，预期结果是索引1
namespace test_find_last_not_of_overload4

这段代码结束了之前的命名空间 `test_find_last_not_of_overload4`。


namespace test_output_operator {

定义了一个新的命名空间 `test_output_operator`。


void testOutputIterator(const std::string& str) {

定义了一个函数 `testOutputIterator`，接受一个 `std::string` 引用参数 `str`。


std::ostringstream stream;

创建了一个 `std::ostringstream` 类型的对象 `stream`，用于字符串流操作。


stream << string_view(str);

将 `str` 转换为 `string_view` 类型并输出到流 `stream` 中。


std::string actual = stream.str();

将流 `stream` 中的内容转换为 `std::string` 类型并赋给 `actual`。


EXPECT_EQ(str, actual);

使用测试框架的 `EXPECT_EQ` 断言比较 `str` 和 `actual` 是否相等。


} // namespace test_output_operator

结束了命名空间 `test_output_operator`。


namespace test_hash {

定义了一个新的命名空间 `test_hash`。


TEST(StringViewTest, testHash) {

开始了一个测试用例，名称为 `StringViewTest.testHash`。


EXPECT_EQ(
    std::hash<string_view>()(string_view()), std::hash<string_view>()(""));

使用测试框架的 `EXPECT_EQ` 断言，比较空的 `string_view` 的哈希值是否相等。


EXPECT_EQ(
    std::hash<string_view>()(string_view("hello")),
    std::hash<string_view>()("hello"));

使用测试框架的 `EXPECT_EQ` 断言，比较包含 "hello" 的 `string_view` 的哈希值是否相等。


EXPECT_NE(
    std::hash<string_view>()(string_view("hello")),
    std::hash<string_view>()(""));

使用测试框架的 `EXPECT_NE` 断言，比较包含 "hello" 的 `string_view` 的哈希值是否不等于空 `string_view` 的哈希值。


} // namespace test_hash

结束了命名空间 `test_hash`。


} // namespace

结束了最外层的命名空间。


// NOLINTEND(modernize*, readability*, bugprone-string-constructor)

这是一个注释，用于指示代码静态分析工具禁止对代码进行一些特定的规则检查，如现代化改进、可读性和潜在的字符串构造函数使用问题。
```