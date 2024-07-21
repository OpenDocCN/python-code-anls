# `.\pytorch\c10\test\util\logging_test.cpp`

```
#include <algorithm>  // 引入算法库，用于一些算法操作
#include <optional>   // 引入可选项库，用于处理可能为空的对象

#include <c10/util/ArrayRef.h>  // 引入C10库中的ArrayRef头文件，用于处理数组引用
#include <c10/util/Logging.h>   // 引入C10库中的Logging头文件，用于日志记录
#include <gtest/gtest.h>        // 引入Google测试框架的头文件

namespace c10_test {

using std::set;      // 使用标准库中的set集合容器
using std::string;   // 使用标准库中的字符串类型
using std::vector;   // 使用标准库中的向量容器

TEST(LoggingTest, TestEnforceTrue) {
  // 这应该正常工作。
  CAFFE_ENFORCE(true, "Isn't it?");   // 使用CAFFE_ENFORCE宏来确保条件为真
}

TEST(LoggingTest, TestEnforceFalse) {
  bool kFalse = false;
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kFalse);   // 交换标志以防止致命错误抛出
  try {
    CAFFE_ENFORCE(false, "This throws.");   // 使用CAFFE_ENFORCE宏来确保条件为假
    // 这应该永远不会触发。
    ADD_FAILURE();   // 在测试中标记为失败
  } catch (const ::c10::Error&) {
  }
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kFalse);   // 恢复标志状态
}

TEST(LoggingTest, TestEnforceEquals) {
  int x = 4;
  int y = 5;
  int z = 0;
  try {
    CAFFE_ENFORCE_THAT(std::equal_to<void>(), ==, ++x, ++y, "Message: ", z++);
    // 这应该永远不会触发。
    ADD_FAILURE();   // 在测试中标记为失败
  } catch (const ::c10::Error& err) {
    auto errStr = std::string(err.what());
    EXPECT_NE(errStr.find("5 vs 6"), string::npos);   // 检查错误消息中是否包含特定字符串
    EXPECT_NE(errStr.find("Message: 0"), string::npos);
  }

  // 参数只被展开一次
  CAFFE_ENFORCE_THAT(std::equal_to<void>(), ==, ++x, y);   // 使用CAFFE_ENFORCE_THAT宏来确保条件满足
  EXPECT_EQ(x, 6);   // 检查x是否为6
  EXPECT_EQ(y, 6);   // 检查y是否为6
  EXPECT_EQ(z, 1);   // 检查z是否为1
}

namespace {
struct EnforceEqWithCaller {
  void test(const char* x) {
    CAFFE_ENFORCE_EQ_WITH_CALLER(1, 1, "variable: ", x, " is a variable");   // 使用CAFFE_ENFORCE_EQ_WITH_CALLER宏来确保两个值相等
  }
};
} // namespace

TEST(LoggingTest, TestEnforceMessageVariables) {
  const char* const x = "hello";
  CAFFE_ENFORCE_EQ(1, 1, "variable: ", x, " is a variable");   // 使用CAFFE_ENFORCE_EQ宏来确保两个值相等

  EnforceEqWithCaller e;
  e.test(x);   // 调用结构体中的测试函数
}

TEST(
    LoggingTest,
    EnforceEqualsObjectWithReferenceToTemporaryWithoutUseOutOfScope) {
  std::vector<int> x = {1, 2, 3, 4};
  // 这种情况有些棘手。我们有一个临时的std::initializer_list，
  // 我们的临时ArrayRef引用到它。通过将const引用绑定到ArrayRef来延长
  // std::initializer_list的生命周期，并不仅仅是ArrayRef，所以我们得到一个
  // 悬空的ArrayRef。这个测试迫使实现处理正确。
  CAFFE_ENFORCE_EQ(x, (at::ArrayRef<int>{1, 2, 3, 4}));   // 使用CAFFE_ENFORCE_EQ宏来确保两个值相等
}

namespace {
struct Noncopyable {
  int x;

  explicit Noncopyable(int a) : x(a) {}

  Noncopyable(const Noncopyable&) = delete;
  Noncopyable(Noncopyable&&) = delete;
  Noncopyable& operator=(const Noncopyable&) = delete;
  Noncopyable& operator=(Noncopyable&&) = delete;

  bool operator==(const Noncopyable& rhs) const {
    return x == rhs.x;
  }
};

std::ostream& operator<<(std::ostream& out, const Noncopyable& nc) {
  out << "Noncopyable(" << nc.x << ")";
  return out;
}
} // namespace

TEST(LoggingTest, DoesntCopyComparedObjects) {
  CAFFE_ENFORCE_EQ(Noncopyable(123), Noncopyable(123));   // 使用CAFFE_ENFORCE_EQ宏来确保两个对象相等
}

TEST(LoggingTest, EnforceShowcase) {
  // 实际上这不是一个测试，而是一个方便的东西，你可以运行和查看所有消息
  int one = 1;
  int two = 2;
  int three = 3;
}
#define WRAP_AND_PRINT(exp)                    \
  try {                                        \
    // 尝试执行传入的表达式
    exp;                                       \
  } catch (const ::c10::Error&) {              \
    /* ::c10::Error 已经处理了 LOG(ERROR) */  \
    // 捕获 ::c10::Error 异常，已经执行了 LOG(ERROR)
  }

WRAP_AND_PRINT(CAFFE_ENFORCE_EQ(one, two));
// 使用 CAFFE_ENFORCE_EQ 检查 one 是否等于 two

WRAP_AND_PRINT(CAFFE_ENFORCE_NE(one * 2, two));
// 使用 CAFFE_ENFORCE_NE 检查 one * 2 是否不等于 two

WRAP_AND_PRINT(CAFFE_ENFORCE_GT(one, two));
// 使用 CAFFE_ENFORCE_GT 检查 one 是否大于 two

WRAP_AND_PRINT(CAFFE_ENFORCE_GE(one, two));
// 使用 CAFFE_ENFORCE_GE 检查 one 是否大于或等于 two

WRAP_AND_PRINT(CAFFE_ENFORCE_LT(three, two));
// 使用 CAFFE_ENFORCE_LT 检查 three 是否小于 two

WRAP_AND_PRINT(CAFFE_ENFORCE_LE(three, two));
// 使用 CAFFE_ENFORCE_LE 检查 three 是否小于或等于 two

WRAP_AND_PRINT(CAFFE_ENFORCE_EQ(
    one * two + three, three * two, "It's a pretty complicated expression"));
// 使用 CAFFE_ENFORCE_EQ 检查复杂表达式 one * two + three 是否等于 three * two

WRAP_AND_PRINT(CAFFE_ENFORCE_THAT(
    std::equal_to<void>(), ==, one * two + three, three * two));
// 使用 CAFFE_ENFORCE_THAT 使用 std::equal_to<void>() 检查是否 one * two + three 等于 three * two

TEST(LoggingTest, Join) {
  auto s = c10::Join(", ", vector<int>({1, 2, 3}));
  // 使用 c10::Join 将 vector<int> {1, 2, 3} 中的元素用 ", " 连接成字符串
  EXPECT_EQ(s, "1, 2, 3");
  s = c10::Join(":", vector<string>());
  // 使用 c10::Join 将空的 vector<string> 用 ":" 连接成字符串
  EXPECT_EQ(s, "");
  s = c10::Join(", ", set<int>({3, 1, 2}));
  // 使用 c10::Join 将 set<int> {3, 1, 2} 中的元素用 ", " 连接成字符串
  EXPECT_EQ(s, "1, 2, 3");
}

TEST(LoggingTest, TestDanglingElse) {
  if (true)
    TORCH_DCHECK_EQ(1, 1);
  else
    GTEST_FAIL();
  // 如果条件为真，则调用 TORCH_DCHECK_EQ(1, 1)，否则调用 GTEST_FAIL()
}

#if GTEST_HAS_DEATH_TEST
TEST(LoggingDeathTest, TestEnforceUsingFatal) {
  bool kTrue = true;
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kTrue);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  // 交换 FLAGS_caffe2_use_fatal_for_enforce 和 kTrue 的值
  EXPECT_DEATH(CAFFE_ENFORCE(false, "This goes fatal."), "");
  // 期望 CAFFE_ENFORCE(false, "This goes fatal.") 导致程序退出
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kTrue);
  // 恢复 FLAGS_caffe2_use_fatal_for_enforce 的值
}
#endif

C10_NOINLINE void f1() {
  CAFFE_THROW("message");
  // 抛出异常 "message"
}

C10_NOINLINE void f2() {
  f1();
  // 调用 f1() 函数
}

C10_NOINLINE void f3() {
  f2();
  // 调用 f2() 函数
}

#ifdef FBCODE_CAFFE2
TEST(LoggingTest, ExceptionWhat) {
  std::optional<::c10::Error> error;
  try {
    f3();
    // 调用 f3() 函数
  } catch (const ::c10::Error& e) {
    error = e;
    // 捕获 ::c10::Error 异常并存储到 error 可选类型中
  }

  ASSERT_TRUE(error);
  // 断言 error 不为空

  std::string what = error->what();
  // 获取异常对象的描述信息

  EXPECT_TRUE(what.find("c10_test::f1()") != std::string::npos) << what;
  // 断言描述信息包含 "c10_test::f1()"

  EXPECT_TRUE(what.find("c10_test::f2()") != std::string::npos) << what;
  // 断言描述信息包含 "c10_test::f2()"

  EXPECT_TRUE(what.find("c10_test::f3()") != std::string::npos) << what;
  // 断言描述信息包含 "c10_test::f3()"

  // 添加新的上下文信息到异常对象
  error->add_context("NewContext");
  what = error->what();
  EXPECT_TRUE(what.find("c10_test::f1()") != std::string::npos) << what;
  // 断言描述信息包含 "c10_test::f1()"

  EXPECT_TRUE(what.find("c10_test::f2()") != std::string::npos) << what;
  // 断言描述信息包含 "c10_test::f2()"

  EXPECT_TRUE(what.find("c10_test::f3()") != std::string::npos) << what;
  // 断言描述信息包含 "c10_test::f3()"

  EXPECT_TRUE(what.find("NewContext") != std::string::npos) << what;
  // 断言描述信息包含 "NewContext"
}
#endif

TEST(LoggingTest, LazyBacktrace) {
  struct CountingLazyString : ::c10::OptimisticLazyValue<std::string> {
    mutable size_t invocations{0};

    std::string compute() const override {
      ++invocations;
      return "A string";
      // 每次计算时增加调用次数并返回字符串 "A string"
    }
  };
}
  }
};

// 创建一个共享指针，指向一个 CountingLazyString 对象，表示回溯信息
auto backtrace = std::make_shared<CountingLazyString>();
// 创建一个 c10::Error 对象，使用空字符串和上面创建的回溯信息
::c10::Error ex("", backtrace);

// 断言：在构造时未计算回溯信息，因此调用次数为零
EXPECT_EQ(backtrace->invocations, 0);

// 获取错误消息的首次调用，期望回溯信息被计算一次
const char* w1 = ex.what();
EXPECT_EQ(backtrace->invocations, 1);

// 再次获取错误消息，期望回溯信息不被重新计算，因此调用次数仍为一次
const char* w2 = ex.what();
EXPECT_EQ(backtrace->invocations, 1);

// 断言：两次获取的错误消息应该是相同的对象，即地址相同
EXPECT_EQ(w1, w2);

// 向错误对象添加上下文信息
ex.add_context("");
// 获取错误消息，预期回溯信息不会被重新计算，因此调用次数保持为一次
ex.what();
EXPECT_EQ(backtrace->invocations, 1);
}

} // namespace c10_test
```