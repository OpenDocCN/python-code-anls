# `.\pytorch\c10\test\util\optional_test.cpp`

```py
#include <c10/util/Optional.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>

#include <c10/util/ArrayRef.h>

// 匿名命名空间，用于定义本文件内部的私有类型和函数
namespace {

using testing::Eq;
using testing::Ge;
using testing::Gt;
using testing::Le;
using testing::Lt;
using testing::Ne;
using testing::Not;

// 泛型类模板 OptionalTest，继承自 testing::Test 用于测试 std::optional 的不同类型
template <typename T>
class OptionalTest : public ::testing::Test {
 public:
  using optional = std::optional<T>;  // 使用类型别名 optional 代表 std::optional<T>
};

// 模板函数 getSampleValue 的特化定义，返回不同类型的示例值
template <>
bool getSampleValue() {
  return true;  // 返回布尔类型的示例值 true
}

template <>
uint64_t getSampleValue() {
  return 42;  // 返回无符号64位整数类型的示例值 42
}

template <>
c10::IntArrayRef getSampleValue() {
  return {};  // 返回 c10::IntArrayRef 类型的空对象
}

template <>
std::string getSampleValue() {
  return "hello";  // 返回字符串类型的示例值 "hello"
}

// 使用 using 关键字，定义 OptionalTypes 作为测试类型列表
using OptionalTypes = ::testing::Types<
    // 32位标量优化。
    bool,
    // 可平凡析构但不是32位标量。
    uint64_t,
    // ArrayRef 优化。
    c10::IntArrayRef,
    // 非平凡析构函数。
    std::string>;

// 使用 TYPED_TEST_SUITE 定义测试套件 OptionalTest，测试不同类型的 std::optional
TYPED_TEST_SUITE(OptionalTest, OptionalTypes);

// 测试用例 TYPED_TEST(OptionalTest, Empty) 对空的 std::optional 进行测试
TYPED_TEST(OptionalTest, Empty) {
  typename TestFixture::optional empty;  // 定义空的 std::optional 对象

  EXPECT_FALSE((bool)empty);  // 断言空对象的 bool 转换结果为 false
  EXPECT_FALSE(empty.has_value());  // 断言空对象没有值

  // NOLINTNEXTLINE(bugprone-unchecked-optional-access,hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(empty.value(), c10::bad_optional_access);  // 使用空对象的 value() 方法抛出异常 c10::bad_optional_access
}

// 测试用例 TYPED_TEST(OptionalTest, Initialized) 对已初始化的 std::optional 进行测试
TYPED_TEST(OptionalTest, Initialized) {
  using optional = typename TestFixture::optional;

  const auto val = getSampleValue<TypeParam>();  // 获取当前类型的示例值
  optional opt((val));  // 用示例值初始化 std::optional 对象
  auto copy(opt), moveFrom1(opt), moveFrom2(opt);  // 使用示例值进行拷贝初始化
  optional move(std::move(moveFrom1));  // 使用移动语义初始化另一个 std::optional
  optional copyAssign;  // 定义空的 std::optional 对象
  copyAssign = opt;  // 将 opt 赋值给 copyAssign
  optional moveAssign;  // 定义空的 std::optional 对象
  moveAssign = std::move(moveFrom2);  // 使用移动语义赋值给 moveAssign

  std::array<typename TestFixture::optional*, 5> opts = {
      &opt, &copy, &copyAssign, &move, &moveAssign};  // 定义 std::optional 指针数组

  for (auto* popt : opts) {  // 遍历 std::optional 指针数组
    auto& opt = *popt;
    EXPECT_TRUE((bool)opt);  // 断言对象不为空
    EXPECT_TRUE(opt.has_value());  // 断言对象有值

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(opt.value(), val);  // 断言对象的值等于示例值
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(*opt, val);  // 断言对象解引用后的值等于示例值
  }
}

// 自定义测试类 SelfCompareTest，用于比较 std::optional 自身
class SelfCompareTest : public testing::TestWithParam<std::optional<int>> {};

// 测试 SelfCompareTest 类的实例
TEST_P(SelfCompareTest, SelfCompare) {
  std::optional<int> x = GetParam();  // 获取测试参数
  EXPECT_THAT(x, Eq(x));  // 断言对象等于自身
  EXPECT_THAT(x, Le(x));  // 断言对象小于等于自身
  EXPECT_THAT(x, Ge(x));  // 断言对象大于等于自身
  EXPECT_THAT(x, Not(Ne(x)));  // 断言对象不不等于自身
  EXPECT_THAT(x, Not(Lt(x)));  // 断言对象不小于自身
  EXPECT_THAT(x, Not(Gt(x)));  // 断言对象不大于自身
}

// 实例化 SelfCompareTest 类，使用 c10::nullopt 作为参数
INSTANTIATE_TEST_SUITE_P(
    nullopt,
    SelfCompareTest,
    testing::Values(c10::nullopt));

// 实例化 SelfCompareTest 类，使用 c10::make_optional(2) 作为参数
INSTANTIATE_TEST_SUITE_P(
    int,
    SelfCompareTest,
    testing::Values(c10::make_optional(2)));

}  // namespace
TEST(OptionalTest, Nullopt) {
  // 创建一个包含整数值 2 的 std::optional 对象 x
  std::optional<int> x = 2;

  // 断言 c10::nullopt 不等于 x
  EXPECT_THAT(c10::nullopt, Not(Eq(x)));
  // 断言 x 不等于 c10::nullopt
  EXPECT_THAT(x, Not(Eq(c10::nullopt)));

  // 断言 x 不等于 c10::nullopt
  EXPECT_THAT(x, Ne(c10::nullopt));
  // 断言 c10::nullopt 不等于 x
  EXPECT_THAT(c10::nullopt, Ne(x));

  // 断言 x 不小于 c10::nullopt
  EXPECT_THAT(x, Not(Lt(c10::nullopt)));
  // 断言 c10::nullopt 小于 x
  EXPECT_THAT(c10::nullopt, Lt(x));

  // 断言 x 不小于等于 c10::nullopt
  EXPECT_THAT(x, Not(Le(c10::nullopt)));
  // 断言 c10::nullopt 小于等于 x
  EXPECT_THAT(c10::nullopt, Le(x));

  // 断言 x 大于 c10::nullopt
  EXPECT_THAT(x, Gt(c10::nullopt));
  // 断言 c10::nullopt 不大于 x
  EXPECT_THAT(c10::nullopt, Not(Gt(x)));

  // 断言 x 大于等于 c10::nullopt
  EXPECT_THAT(x, Ge(c10::nullopt));
  // 断言 c10::nullopt 不大于等于 x
  EXPECT_THAT(c10::nullopt, Not(Ge(x)));
}

// Ensure comparisons work...
using CmpTestTypes = testing::Types<
    // between two optionals
    std::pair<std::optional<int>, std::optional<int>>,

    // between an optional and a value
    std::pair<std::optional<int>, int>,
    // between a value and an optional
    std::pair<int, std::optional<int>>,

    // between an optional and a differently typed value
    std::pair<std::optional<int>, long>,
    // between a differently typed value and an optional
    std::pair<long, std::optional<int>>>;
template <typename T>
class CmpTest : public testing::Test {};
TYPED_TEST_SUITE(CmpTest, CmpTestTypes);

TYPED_TEST(CmpTest, Cmp) {
  // 从 TypeParam 中获取一对值 {2, 3}
  TypeParam pair = {2, 3};
  auto x = pair.first;
  auto y = pair.second;

  // 断言 x 不等于 y
  EXPECT_THAT(x, Not(Eq(y)));

  // 断言 x 不等于 y
  EXPECT_THAT(x, Ne(y));

  // 断言 x 小于 y
  EXPECT_THAT(x, Lt(y));
  // 断言 y 不小于 x
  EXPECT_THAT(y, Not(Lt(x)));

  // 断言 x 小于等于 y
  EXPECT_THAT(x, Le(y));
  // 断言 y 不小于等于 x
  EXPECT_THAT(y, Not(Le(x)));

  // 断言 x 不大于 y
  EXPECT_THAT(x, Not(Gt(y)));
  // 断言 y 大于 x
  EXPECT_THAT(y, Gt(x));

  // 断言 x 不大于等于 y
  EXPECT_THAT(x, Not(Ge(y)));
  // 断言 y 大于等于 x
  EXPECT_THAT(y, Ge(x));
}
```