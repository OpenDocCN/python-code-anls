# `.\pytorch\test\cpp\api\any.cpp`

```py
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

#include <algorithm>
#include <string>

using namespace torch::nn;

// 定义一个测试夹具，继承自 SeedingFixture，用于测试 AnyModule 类
struct AnyModuleTest : torch::test::SeedingFixture {};

// 测试用例：测试简单的返回类型
TEST_F(AnyModuleTest, SimpleReturnType) {
  // 定义一个简单的 Module 结构体 M，继承自 torch::nn::Module
  struct M : torch::nn::Module {
    // forward 函数返回整数值 123
    int forward() {
      return 123;
    }
  };
  // 创建 AnyModule 对象 any，并传入 M 类的实例
  AnyModule any(M{});
  // 断言调用 any 对象的 forward 函数返回值为 123
  ASSERT_EQ(any.forward<int>(), 123);
}

// 测试用例：测试返回类型为 int，并接受单个整数参数
TEST_F(AnyModuleTest, SimpleReturnTypeAndSingleArgument) {
  // 定义一个 Module 结构体 M，继承自 torch::nn::Module
  struct M : torch::nn::Module {
    // forward 函数接受一个整数参数 x，并返回该参数值
    int forward(int x) {
      return x;
    }
  };
  // 创建 AnyModule 对象 any，并传入 M 类的实例
  AnyModule any(M{});
  // 断言调用 any 对象的 forward 函数传入参数 5 后返回值为 5
  ASSERT_EQ(any.forward<int>(5), 5);
}

// 测试用例：测试返回类型为 const char*，并接受 const char* 参数
TEST_F(AnyModuleTest, StringLiteralReturnTypeAndArgument) {
  // 定义一个 Module 结构体 M，继承自 torch::nn::Module
  struct M : torch::nn::Module {
    // forward 函数接受一个 const char* 参数 x，并返回该参数值
    const char* forward(const char* x) {
      return x;
    }
  };
  // 创建 AnyModule 对象 any，并传入 M 类的实例
  AnyModule any(M{});
  // 断言调用 any 对象的 forward 函数传入字符串 "hello" 后返回 std::string 类型的 "hello"
  ASSERT_EQ(any.forward<const char*>("hello"), std::string("hello"));
}

// 测试用例：测试返回类型为 std::string，并接受 int 和 const double 参数
TEST_F(AnyModuleTest, StringReturnTypeWithConstArgument) {
  // 定义一个 Module 结构体 M，继承自 torch::nn::Module
  struct M : torch::nn::Module {
    // forward 函数接受一个整数参数 x 和一个 const double 参数 f，并返回它们的和的字符串形式
    std::string forward(int x, const double f) {
      return std::to_string(static_cast<int>(x + f));
    }
  };
  // 创建 AnyModule 对象 any，并传入 M 类的实例
  AnyModule any(M{});
  int x = 4;
  // 断言调用 any 对象的 forward 函数传入参数 x = 4, f = 3.14 后返回 std::string 类型的 "7"
  ASSERT_EQ(any.forward<std::string>(x, 3.14), std::string("7"));
}

// 测试用例：测试返回类型为 torch::Tensor，并接受多种字符串参数
TEST_F(
    AnyModuleTest,
    TensorReturnTypeAndStringArgumentsWithFunkyQualifications) {
  // 定义一个 Module 结构体 M，继承自 torch::nn::Module
  struct M : torch::nn::Module {
    // forward 函数接受三个字符串参数 a, b, c，计算它们的长度并返回一个长度为和的全为1的 Tensor
    torch::Tensor forward(
        std::string a,
        const std::string& b,
        std::string&& c) {
      const auto s = a + b + c;
      return torch::ones({static_cast<int64_t>(s.size())});
    }
  };
  // 创建 AnyModule 对象 any，并传入 M 类的实例
  AnyModule any(M{});
  // 断言调用 any 对象的 forward 函数传入参数 "a", "ab", "abc" 后 Tensor 求和为 6
  ASSERT_TRUE(
      any.forward(std::string("a"), std::string("ab"), std::string("abc"))
          .sum()
          .item<int32_t>() == 6);
}

// 测试用例：测试传入错误类型的参数
TEST_F(AnyModuleTest, WrongArgumentType) {
  // 定义一个 Module 结构体 M，继承自 torch::nn::Module
  struct M : torch::nn::Module {
    // forward 函数接受一个浮点数参数 x，并返回该参数值
    int forward(float x) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      return x;
    }
  };
  // 创建 AnyModule 对象 any，并传入 M 类的实例
  AnyModule any(M{});
  // 断言调用 any 对象的 forward 函数传入参数 5.0 时，抛出类型不匹配的异常
  ASSERT_THROWS_WITH(
      any.forward(5.0),
      "Expected argument #0 to be of type float, "
      "but received value of type double");
}

// 定义一个 Module 结构体 M_test_wrong_number_of_arguments，继承自 torch::nn::Module
struct M_test_wrong_number_of_arguments : torch::nn::Module {
  // forward 函数接受两个整数参数 a 和 b，并返回它们的和
  int forward(int a, int b) {
    return a + b;
  }
};

// 测试用例：测试传入错误数量的参数
TEST_F(AnyModuleTest, WrongNumberOfArguments) {
  // 创建 AnyModule 对象 any，并传入 M_test_wrong_number_of_arguments 类的实例
  AnyModule any(M_test_wrong_number_of_arguments{});
#if defined(_MSC_VER)
  std::string module_name = "struct M_test_wrong_number_of_arguments";
#else
  std::string module_name = "M_test_wrong_number_of_arguments";
#endif
// 断言：调用 any 对象的 forward() 方法，期望抛出异常并检查错误信息是否包含模块名称和相关错误信息
ASSERT_THROWS_WITH(
    any.forward(),
    module_name +
        "'s forward() method expects 2 argument(s), but received 0. "
        "If " +
        module_name +
        "'s forward() method has default arguments, "
        "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.");

// 断言：调用 any 对象的 forward() 方法，传入一个参数，期望抛出异常并检查错误信息是否包含模块名称和相关错误信息
ASSERT_THROWS_WITH(
    any.forward(5),
    module_name +
        "'s forward() method expects 2 argument(s), but received 1. "
        "If " +
        module_name +
        "'s forward() method has default arguments, "
        "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.");

// 断言：调用 any 对象的 forward() 方法，传入三个参数，期望抛出异常并检查错误信息是否包含模块名称和相关错误信息
ASSERT_THROWS_WITH(
    any.forward(1, 2, 3),
    module_name +
        "'s forward() method expects 2 argument(s), but received 3.");
}

// 定义带有默认参数和宏的结构体 M_default_arg_with_macro，继承自 torch::nn::Module
struct M_default_arg_with_macro : torch::nn::Module {
  // 定义 forward() 方法，接受一个必选参数 a 和两个可选参数 b 和 c，并返回它们的和
  double forward(int a, int b = 2, double c = 3.0) {
    return a + b + c;
  }

 protected:
  // 定义宏 FORWARD_HAS_DEFAULT_ARGS，指定了 b 和 c 的默认值
  FORWARD_HAS_DEFAULT_ARGS(
      {1, torch::nn::AnyValue(2)},
      {2, torch::nn::AnyValue(3.0)})
};

// 定义不带宏的结构体 M_default_arg_without_macro，继承自 torch::nn::Module
struct M_default_arg_without_macro : torch::nn::Module {
  // 定义 forward() 方法，接受一个必选参数 a 和两个可选参数 b 和 c，并返回它们的和
  double forward(int a, int b = 2, double c = 3.0) {
    return a + b + c;
  }
};

// 测试用例：验证传递参数给带有默认参数的 forward() 方法
TEST_F(
    AnyModuleTest,
    PassingArgumentsToModuleWithDefaultArgumentsInForwardMethod) {
  {
    // 创建 AnyModule 对象，并传入 M_default_arg_with_macro 实例
    AnyModule any(M_default_arg_with_macro{});

    // 断言：调用 any 对象的 forward() 方法，传入一个参数，返回值为 6.0
    ASSERT_EQ(any.forward<double>(1), 6.0);
    // 断言：调用 any 对象的 forward() 方法，传入两个参数，返回值为 7.0
    ASSERT_EQ(any.forward<double>(1, 3), 7.0);
    // 断言：调用 any 对象的 forward() 方法，传入三个参数，返回值为 9.0
    ASSERT_EQ(any.forward<double>(1, 3, 5.0), 9.0);

    // 断言：调用 any 对象的 forward() 方法，不传入任何参数，期望抛出异常并检查错误信息是否包含相应信息
    ASSERT_THROWS_WITH(
        any.forward(),
        "M_default_arg_with_macro's forward() method expects at least 1 argument(s) and at most 3 argument(s), but received 0.");
    // 断言：调用 any 对象的 forward() 方法，传入四个参数，期望抛出异常并检查错误信息是否包含相应信息
    ASSERT_THROWS_WITH(
        any.forward(1, 2, 3.0, 4),
        "M_default_arg_with_macro's forward() method expects at least 1 argument(s) and at most 3 argument(s), but received 4.");
  }
  {
    // 创建 AnyModule 对象，并传入 M_default_arg_without_macro 实例
    AnyModule any(M_default_arg_without_macro{});

    // 断言：调用 any 对象的 forward() 方法，传入三个参数，返回值为 9.0
    ASSERT_EQ(any.forward<double>(1, 3, 5.0), 9.0);

#if defined(_MSC_VER)
    std::string module_name = "struct M_default_arg_without_macro";
#else
    std::string module_name = "M_default_arg_without_macro";
#endif

    // 断言：调用 any 对象的 forward() 方法，不传入任何参数，期望抛出异常并检查错误信息是否包含模块名称和相关信息
    ASSERT_THROWS_WITH(
        any.forward(),
        module_name +
            "'s forward() method expects 3 argument(s), but received 0. "
            "If " +
            module_name +
            "'s forward() method has default arguments, "
            "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.");
    // 断言：调用 any 对象的 forward() 方法，传入一个参数，期望抛出异常并检查错误信息是否包含模块名称和相关信息
    ASSERT_THROWS_WITH(
        any.forward<double>(1),
        module_name +
            "'s forward() method expects 3 argument(s), but received 1. "
            "If " +
            module_name +
            "'s forward() method has default arguments, "
            "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.");
    // 使用 ASSERT_THROWS_WITH 宏测试函数调用，验证是否抛出异常
    ASSERT_THROWS_WITH(
        // 调用 any 对象的 forward<double> 方法，传入参数 1 和 3
        any.forward<double>(1, 3),
        // 构建异常消息，指示模块名以及 forward() 方法期望的参数个数
        module_name +
            "'s forward() method expects 3 argument(s), but received 2. "
            "If " +
            module_name +
            // 如果 forward() 方法有默认参数，提醒使用对应的宏 FORWARD_HAS_DEFAULT_ARGS 声明
            "'s forward() method has default arguments, "
            "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.");
    // 使用 ASSERT_THROWS_WITH 宏再次测试函数调用，验证是否抛出异常
    ASSERT_THROWS_WITH(
        // 调用 any 对象的 forward 方法，传入参数 1, 2, 3.0 和 4
        any.forward(1, 2, 3.0, 4),
        // 构建异常消息，指示模块名以及 forward() 方法期望的参数个数
        module_name +
            "'s forward() method expects 3 argument(s), but received 4.");
    }
}

// 结构体定义：M，继承自torch::nn::Module
struct M : torch::nn::Module {
  // 构造函数，初始化 Module 名称和成员变量 value
  explicit M(int value_) : torch::nn::Module("M"), value(value_) {}
  int value; // 整数成员变量 value
  // 前向传播函数，参数为 float 类型，返回值为 int 类型
  int forward(float x) {
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    return x; // 返回参数 x 的值，进行类型转换（可能会导致信息损失）
  }
};

// 测试用例：GetWithCorrectTypeSucceeds
TEST_F(AnyModuleTest, GetWithCorrectTypeSucceeds) {
  // 创建 AnyModule 对象 any，包含 M 类型的实例，值为 5
  AnyModule any(M{5});
  ASSERT_EQ(any.get<M>().value, 5); // 断言获取 M 类型实例的 value 成员变量值为 5
}

// 测试用例：GetWithIncorrectTypeThrows
TEST_F(AnyModuleTest, GetWithIncorrectTypeThrows) {
  // 内部结构体定义：N，继承自 torch::nn::Module，定义了 forward 函数
  struct N : torch::nn::Module {
    torch::Tensor forward(torch::Tensor input) {
      return input; // 返回输入的张量 input
    }
  };
  AnyModule any(M{5}); // 创建 AnyModule 对象 any，包含 M 类型的实例，值为 5
  ASSERT_THROWS_WITH(any.get<N>(), "Attempted to cast module"); // 断言尝试获取 N 类型实例会抛出异常
}

// 测试用例：PtrWithBaseClassSucceeds
TEST_F(AnyModuleTest, PtrWithBaseClassSucceeds) {
  // 创建 AnyModule 对象 any，包含 M 类型的实例，值为 5
  AnyModule any(M{5});
  auto ptr = any.ptr(); // 获取 AnyModule 的指针
  ASSERT_NE(ptr, nullptr); // 断言指针非空
  ASSERT_EQ(ptr->name(), "M"); // 断言指针的名称为 "M"
}

// 测试用例：PtrWithGoodDowncastSuccceeds
TEST_F(AnyModuleTest, PtrWithGoodDowncastSuccceeds) {
  // 创建 AnyModule 对象 any，包含 M 类型的实例，值为 5
  AnyModule any(M{5});
  auto ptr = any.ptr<M>(); // 尝试向下转型获取 M 类型的指针
  ASSERT_NE(ptr, nullptr); // 断言指针非空
  ASSERT_EQ(ptr->value, 5); // 断言指针的 value 成员变量值为 5
}

// 测试用例：PtrWithBadDowncastThrows
TEST_F(AnyModuleTest, PtrWithBadDowncastThrows) {
  // 内部结构体定义：N，继承自 torch::nn::Module，定义了 forward 函数
  struct N : torch::nn::Module {
    torch::Tensor forward(torch::Tensor input) {
      return input; // 返回输入的张量 input
    }
  };
  AnyModule any(M{5}); // 创建 AnyModule 对象 any，包含 M 类型的实例，值为 5
  ASSERT_THROWS_WITH(any.ptr<N>(), "Attempted to cast module"); // 断言尝试获取 N 类型实例会抛出异常
}

// 测试用例：DefaultStateIsEmpty
TEST_F(AnyModuleTest, DefaultStateIsEmpty) {
  // 内部结构体定义：M，继承自 torch::nn::Module，定义了 forward 函数
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {} // 构造函数，初始化成员变量 value
    int value; // 整数成员变量 value
    int forward(float x) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      return x; // 返回参数 x 的值，进行类型转换（可能会导致信息损失）
    }
  };
  AnyModule any; // 创建空的 AnyModule 对象
  ASSERT_TRUE(any.is_empty()); // 断言 AnyModule 对象为空
  any = std::make_shared<M>(5); // 将 M 类型的实例添加到 AnyModule 中
  ASSERT_FALSE(any.is_empty()); // 断言 AnyModule 对象非空
  ASSERT_EQ(any.get<M>().value, 5); // 断言从 AnyModule 中获取 M 类型实例的 value 成员变量值为 5
}

// 测试用例：AllMethodsThrowForEmptyAnyModule
TEST_F(AnyModuleTest, AllMethodsThrowForEmptyAnyModule) {
  // 内部结构体定义：M，继承自 torch::nn::Module，定义了 forward 函数
  struct M : torch::nn::Module {
    int forward(int x) {
      return x; // 返回输入的整数 x
    }
  };
  AnyModule any; // 创建空的 AnyModule 对象
  ASSERT_TRUE(any.is_empty()); // 断言 AnyModule 对象为空
  // 断言调用各种方法时会抛出相应的异常消息
  ASSERT_THROWS_WITH(any.get<M>(), "Cannot call get() on an empty AnyModule");
  ASSERT_THROWS_WITH(any.ptr<M>(), "Cannot call ptr() on an empty AnyModule");
  ASSERT_THROWS_WITH(any.ptr(), "Cannot call ptr() on an empty AnyModule");
  ASSERT_THROWS_WITH(
      any.type_info(), "Cannot call type_info() on an empty AnyModule");
  ASSERT_THROWS_WITH(
      any.forward<int>(5), "Cannot call forward() on an empty AnyModule");
}

// 测试用例：CanMoveAssignDifferentModules
TEST_F(AnyModuleTest, CanMoveAssignDifferentModules) {
  // 内部结构体定义：M，继承自 torch::nn::Module，定义了 forward 函数
  struct M : torch::nn::Module {
    std::string forward(int x) {
      return std::to_string(x); // 返回整数 x 的字符串表示
    }
  };
  // 内部结构体定义：N，继承自 torch::nn::Module，定义了 forward 函数
  struct N : torch::nn::Module {
    int forward(float x) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      return 3 + x; // 返回整数 3 加上浮点数 x 的值
    }
  };
  AnyModule any; // 创建空的 AnyModule 对象
  ASSERT_TRUE(any.is_empty()); // 断言 AnyModule 对象为空
  any = std::make_shared<M>(); // 将 M 类型的实例添加到 AnyModule 中
  ASSERT_FALSE(any.is_empty()); // 断言 AnyModule 对象非空
  ASSERT_EQ(any.forward<std::string>(5), "5"); // 断言从 AnyModule 中调用 forward 函数返回 "5"
  any = std::make_shared<N>(); // 将 N 类型的实例添加到 AnyModule 中
  ASSERT_FALSE(any.is_empty()); // 断言 AnyModule 对象非空
  ASSERT_EQ(any.forward<int>(5.0f), 8); // 断言从 AnyModule 中调用 forward 函数返回整数 8
}
    // 显式构造函数，初始化 MImpl 结构体，继承 torch::nn::Module
    explicit MImpl(int value_) : torch::nn::Module("M"), value(value_) {}

    // 整数类型的成员变量 value
    int value;

    // 前向传播函数，接收一个浮点数 x 作为参数
    int forward(float x) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      // 返回浮点数 x，存在潜在的窄化转换警告
      return x;
    }
  };

  // 结构体 M，作为 MImpl 的 torch::nn::Module 包装
  struct M : torch::nn::ModuleHolder<MImpl> {
    using torch::nn::ModuleHolder<MImpl>::ModuleHolder;
    using torch::nn::ModuleHolder<MImpl>::get;
  };

  // 创建一个 AnyModule 对象 any，包含 MImpl 类型的对象，初始化 value 为 5
  AnyModule any(M{5});
  // 使用断言验证 any 中包含的 MImpl 对象的 value 是否为 5
  ASSERT_EQ(any.get<MImpl>().value, 5);
  // 使用断言验证 any 中包含的 M 对象的 value 是否为 5
  ASSERT_EQ(any.get<M>()->value, 5);

  // 创建一个 AnyModule 对象 module，包含 Linear(3, 4) 对象
  AnyModule module(Linear(3, 4));
  // 获取 module 中 Linear 对象的指针，并存储在 std::shared_ptr<Module> ptr 中
  std::shared_ptr<Module> ptr = module.ptr();
  // 创建一个 Linear 对象 linear，使用 module 中的 Linear 对象进行初始化
  Linear linear(module.get<Linear>());
}

// 定义测试用例 AnyModuleTest 中的 ConvertsVariableToTensorCorrectly 测试方法
TEST_F(AnyModuleTest, ConvertsVariableToTensorCorrectly) {
  // 定义结构 M，继承自 torch::nn::Module
  struct M : torch::nn::Module {
    // 实现 forward 方法，接受一个 torch::Tensor 输入并返回它
    torch::Tensor forward(torch::Tensor input) {
      return input;
    }
  };

  // 创建 AnyModule 对象 any，并传入 M 结构体的实例
  AnyModule any(M{});
  // 断言：对于 torch::autograd::Variable 类型的输入，转换为 torch::Tensor 后求和为 5
  ASSERT_TRUE(
      any.forward(torch::autograd::Variable(torch::ones(5)))
          .sum()
          .item<float>() == 5);
  // 断言：对于 at::Tensor 类型的输入，转换为 torch::Tensor 后求和为 5
  ASSERT_EQ(any.forward(at::ones(5)).sum().item<float>(), 5);
}

// 命名空间 torch::nn 内部的 TestAnyValue 结构体
namespace torch {
namespace nn {
// 定义 TestAnyValue 结构体
struct TestAnyValue {
  // 构造函数模板，接受任意类型 T 的值作为参数
  template <typename T>
  explicit TestAnyValue(T&& value) : value_(std::forward<T>(value)) {}
  // 括号运算符重载，返回 value_ 的副本作为 AnyValue 类型
  AnyValue operator()() {
    return std::move(value_);
  }
  // 成员变量，存储 AnyValue 类型的值
  AnyValue value_;
};
// make_value 函数模板，接受任意类型 T 的值作为参数，并返回 TestAnyValue 对象的调用结果
template <typename T>
AnyValue make_value(T&& value) {
  return TestAnyValue(std::forward<T>(value))();
}
} // namespace nn
} // namespace torch

// 定义测试用例 AnyValueTest 中的 CorrectlyAccessesIntWhenCorrectType 测试方法
struct AnyValueTest : torch::test::SeedingFixture {};

TEST_F(AnyValueTest, CorrectlyAccessesIntWhenCorrectType) {
  // 调用 make_value<int>，创建一个包含整数 5 的 AnyValue 对象 value
  auto value = make_value<int>(5);
  // 断言：尝试获取 int 类型的值不为空指针
  ASSERT_NE(value.try_get<int>(), nullptr);
  // 断言：获取 int 类型的值为 5
  ASSERT_EQ(value.get<int>(), 5);
}

// 定义测试用例 AnyValueTest 中的 CorrectlyAccessesStringLiteralWhenCorrectType 测试方法
TEST_F(AnyValueTest, CorrectlyAccessesStringLiteralWhenCorrectType) {
  // 调用 make_value，创建一个包含字符串字面量 "hello" 的 AnyValue 对象 value
  auto value = make_value("hello");
  // 断言：尝试获取 const char* 类型的值不为空指针
  ASSERT_NE(value.try_get<const char*>(), nullptr);
  // 断言：获取 const char* 类型的值为 std::string("hello")
  ASSERT_EQ(value.get<const char*>(), std::string("hello"));
}

// 定义测试用例 AnyValueTest 中的 CorrectlyAccessesStringWhenCorrectType 测试方法
TEST_F(AnyValueTest, CorrectlyAccessesStringWhenCorrectType) {
  // 调用 make_value，创建一个包含字符串 "hello" 的 AnyValue 对象 value
  auto value = make_value(std::string("hello"));
  // 断言：尝试获取 std::string 类型的值不为空指针
  ASSERT_NE(value.try_get<std::string>(), nullptr);
  // 断言：获取 std::string 类型的值为 "hello"
  ASSERT_EQ(value.get<std::string>(), "hello");
}

// 定义测试用例 AnyValueTest 中的 CorrectlyAccessesPointersWhenCorrectType 测试方法
TEST_F(AnyValueTest, CorrectlyAccessesPointersWhenCorrectType) {
  std::string s("hello");
  std::string* p = &s;
  // 调用 make_value，创建一个包含指向字符串 "hello" 的指针的 AnyValue 对象 value
  auto value = make_value(p);
  // 断言：尝试获取 std::string* 类型的值不为空指针
  ASSERT_NE(value.try_get<std::string*>(), nullptr);
  // 断言：获取 std::string* 类型的值解引用后为 "hello"
  ASSERT_EQ(*value.get<std::string*>(), "hello");
}

// 定义测试用例 AnyValueTest 中的 CorrectlyAccessesReferencesWhenCorrectType 测试方法
TEST_F(AnyValueTest, CorrectlyAccessesReferencesWhenCorrectType) {
  std::string s("hello");
  const std::string& t = s;
  // 调用 make_value，创建一个包含字符串 "hello" 的引用的 AnyValue 对象 value
  auto value = make_value(t);
  // 断言：尝试获取 std::string 类型的值不为空指针
  ASSERT_NE(value.try_get<std::string>(), nullptr);
  // 断言：获取 std::string 类型的值为 "hello"
  ASSERT_EQ(value.get<std::string>(), "hello");
}
# 测试用例：TryGetReturnsNullptrForTheWrongType
TEST_F(AnyValueTest, TryGetReturnsNullptrForTheWrongType) {
  # 创建一个包含整数值的 AnyValue 对象
  auto value = make_value(5);
  # 断言尝试获取整数值时返回非空指针
  ASSERT_NE(value.try_get<int>(), nullptr);
  # 断言尝试获取浮点数值时返回空指针
  ASSERT_EQ(value.try_get<float>(), nullptr);
  # 断言尝试获取长整型值时返回空指针
  ASSERT_EQ(value.try_get<long>(), nullptr);
  # 断言尝试获取字符串值时返回空指针
  ASSERT_EQ(value.try_get<std::string>(), nullptr);
}

# 测试用例：GetThrowsForTheWrongType
TEST_F(AnyValueTest, GetThrowsForTheWrongType) {
  # 创建一个包含整数值的 AnyValue 对象
  auto value = make_value(5);
  # 断言尝试获取整数值时返回非空指针
  ASSERT_NE(value.try_get<int>(), nullptr);
  # 断言调用 get 方法获取浮点数值时抛出异常并带有正确的错误消息
  ASSERT_THROWS_WITH(
      value.get<float>(),
      "Attempted to cast AnyValue to float, "
      "but its actual type is int");
  # 断言调用 get 方法获取长整型值时抛出异常并带有正确的错误消息
  ASSERT_THROWS_WITH(
      value.get<long>(),
      "Attempted to cast AnyValue to long, "
      "but its actual type is int");
}

# 测试用例：MoveConstructionIsAllowed
TEST_F(AnyValueTest, MoveConstructionIsAllowed) {
  # 创建一个包含整数值的 AnyValue 对象
  auto value = make_value(5);
  # 使用移动构造函数创建一个新的 AnyValue 对象
  auto copy = make_value(std::move(value));
  # 断言新对象尝试获取整数值时返回非空指针
  ASSERT_NE(copy.try_get<int>(), nullptr);
  # 断言新对象通过 get 方法获取整数值为 5
  ASSERT_EQ(copy.get<int>(), 5);
}

# 测试用例：MoveAssignmentIsAllowed
TEST_F(AnyValueTest, MoveAssignmentIsAllowed) {
  # 创建一个包含整数值的 AnyValue 对象
  auto value = make_value(5);
  # 创建一个包含整数值 10 的 AnyValue 对象
  auto copy = make_value(10);
  # 使用移动赋值运算符将原对象的值移动到新对象
  copy = std::move(value);
  # 断言新对象尝试获取整数值时返回非空指针
  ASSERT_NE(copy.try_get<int>(), nullptr);
  # 断言新对象通过 get 方法获取整数值为 5
  ASSERT_EQ(copy.get<int>(), 5);
}

# 测试用例：TypeInfoIsCorrectForInt
TEST_F(AnyValueTest, TypeInfoIsCorrectForInt) {
  # 创建一个包含整数值的 AnyValue 对象
  auto value = make_value(5);
  # 断言 AnyValue 对象的类型信息哈希码与整数类型的类型信息哈希码相等
  ASSERT_EQ(value.type_info().hash_code(), typeid(int).hash_code());
}

# 测试用例：TypeInfoIsCorrectForStringLiteral
TEST_F(AnyValueTest, TypeInfoIsCorrectForStringLiteral) {
  # 创建一个包含字符串常量 "hello" 的 AnyValue 对象
  auto value = make_value("hello");
  # 断言 AnyValue 对象的类型信息哈希码与字符串常量类型的类型信息哈希码相等
  ASSERT_EQ(value.type_info().hash_code(), typeid(const char*).hash_code());
}

# 测试用例：TypeInfoIsCorrectForString
TEST_F(AnyValueTest, TypeInfoIsCorrectForString) {
  # 创建一个包含字符串对象 "hello" 的 AnyValue 对象
  auto value = make_value(std::string("hello"));
  # 断言 AnyValue 对象的类型信息哈希码与字符串类型的类型信息哈希码相等
  ASSERT_EQ(value.type_info().hash_code(), typeid(std::string).hash_code());
}
```