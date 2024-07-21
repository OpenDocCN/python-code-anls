# `.\pytorch\test\cpp\api\static.cpp`

```py
#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/csrc/utils/variadic.h>
#include <torch/detail/static.h>
#include <torch/torch.h>

#include <string>
#include <type_traits>
#include <vector>

// 模板函数 f，接受类型 T，当 T 不是 torch 模块时返回 false
template <
    typename T,
    typename = std::enable_if_t<!torch::detail::is_module<T>::value>>
bool f(T&& m) {
  return false;
}

// 模板函数 f 的特化版本，当 T 是 torch 模块时返回 true
template <typename T>
torch::detail::enable_if_module_t<T, bool> f(T&& m) {
  return true;
}

// 测试用例 TestStatic.EnableIfModule，测试 f 函数对各种类型的行为
TEST(TestStatic, EnableIfModule) {
  ASSERT_TRUE(f(torch::nn::LinearImpl(1, 2)));  // 期望返回 true，因为 LinearImpl 是 torch 模块
  ASSERT_FALSE(f(5));  // 期望返回 false，因为 5 不是 torch 模块
  ASSERT_TRUE(torch::detail::check_not_lvalue_references<int>());  // 检查 int 类型是否非左值引用
  ASSERT_TRUE((torch::detail::check_not_lvalue_references<float, int, char>()));  // 检查多个类型是否非左值引用
  ASSERT_FALSE(
      (torch::detail::check_not_lvalue_references<float, int&, char>()));  // 检查包含左值引用的类型是否非左值引用
  ASSERT_TRUE(torch::detail::check_not_lvalue_references<std::string>());  // 检查 std::string 类型是否非左值引用
  ASSERT_FALSE(torch::detail::check_not_lvalue_references<std::string&>());  // 检查 std::string 的引用是否非左值引用
}

namespace {

// 匿名命名空间中定义模块 A，包含一个 forward 函数返回整数 5
struct A : torch::nn::Module {
  int forward() {
    return 5;
  }
};

// 匿名命名空间中定义模块 B，包含一个带有 torch::Tensor 参数的 forward 函数，返回空字符串
struct B : torch::nn::Module {
  std::string forward(torch::Tensor tensor) {
    return "";
  }
};

// 匿名命名空间中定义模块 C，包含一个带有 torch::Tensor& 参数的 forward 函数，返回浮点数 5.0
struct C : torch::nn::Module {
  float forward(torch::Tensor& tensor) {
    return 5.0;
  }
};

// 匿名命名空间中定义模块 D，包含一个带有 torch::Tensor&& 参数的 forward 函数，返回字符 'x'
struct D : torch::nn::Module {
  char forward(torch::Tensor&& tensor) {
    return 'x';
  }
};

// 匿名命名空间中定义模块 E，未包含 forward 函数
struct E : torch::nn::Module {};

} // anonymous namespace

// assert_has_expected_type 函数模板，测试模块的 forward 函数的返回类型
template <typename Module, typename ExpectedType, typename... Args>
void assert_has_expected_type() {
  // 获取模块的 forward 函数的返回类型
  using ReturnType =
      typename torch::detail::return_type_of_forward<Module, Args...>::type;
  // 检查返回类型是否与期望类型相同
  constexpr bool is_expected_type =
      std::is_same<ReturnType, ExpectedType>::value;
  ASSERT_TRUE(is_expected_type) << Module().name();  // 断言返回类型与期望类型相同，打印模块名称作为错误消息
}

// 测试用例 TestStatic.ReturnTypeOfForward，测试各模块的 forward 函数的返回类型
TEST(TestStatic, ReturnTypeOfForward) {
  assert_has_expected_type<A, int>();  // 检查模块 A 的 forward 返回类型是否为 int
  assert_has_expected_type<B, std::string, torch::Tensor>();  // 检查模块 B 的 forward 返回类型是否为 std::string
  assert_has_expected_type<C, float, torch::Tensor&>();  // 检查模块 C 的 forward 返回类型是否为 float
  assert_has_expected_type<D, char, torch::Tensor&&>();  // 检查模块 D 的 forward 返回类型是否为 char
  assert_has_expected_type<E, void>();  // 检查模块 E 的 forward 返回类型是否为 void
}

// 测试用例 TestStatic.Apply，测试 torch::apply 函数的行为
TEST(TestStatic, Apply) {
  std::vector<int> v;
  // 使用 torch::apply 将函数 lambda 应用到参数列表 1, 2, 3, 4, 5 上，将结果存入 v
  torch::apply([&v](int x) { v.push_back(x); }, 1, 2, 3, 4, 5);
  ASSERT_EQ(v.size(), 5);  // 确保 v 中有 5 个元素
  for (const auto i : c10::irange(v.size())) {
    ASSERT_EQ(v.at(i), i + 1);  // 确保 v 中的元素依次为 1, 2, 3, 4, 5
  }
}
```