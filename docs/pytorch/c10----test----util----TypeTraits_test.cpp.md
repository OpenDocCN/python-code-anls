# `.\pytorch\c10\test\util\TypeTraits_test.cpp`

```py
// 包含头文件 TypeTraits.h 和 gtest.h
#include <c10/util/TypeTraits.h>
#include <gtest/gtest.h>

// 使用 c10::guts 命名空间
using namespace c10::guts;

// NOLINTBEGIN(modernize-unary-static-assert)
// 匿名命名空间，用于测试等价比较性
namespace {

// 命名空间 test_is_equality_comparable
namespace test_is_equality_comparable {

// 不具备等价比较性的类
class NotEqualityComparable {};

// 具备等价比较性的类
class EqualityComparable {};

// 重载操作符==，用于等价比较
inline bool operator==(const EqualityComparable&, const EqualityComparable&) {
  return false;
}

// 静态断言：检查 NotEqualityComparable 是否不具备等价比较性
static_assert(!is_equality_comparable<NotEqualityComparable>::value, "");

// 静态断言：检查 EqualityComparable 是否具备等价比较性
static_assert(is_equality_comparable<EqualityComparable>::value, "");

// 静态断言：检查 int 是否具备等价比较性
static_assert(is_equality_comparable<int>::value, "");

// v_ 变量用于消除编译器关于操作符==不需要的警告
const bool v_ = EqualityComparable() == EqualityComparable();

} // namespace test_is_equality_comparable

// 命名空间 test_is_hashable
namespace test_is_hashable {

// 不可哈希的类
class NotHashable {};

// 可哈希的类
class Hashable {};
} // namespace test_is_hashable
} // namespace

// 在 std 命名空间下特化 std::hash 模板，用于 Hashable 类
namespace std {
template <>
struct hash<test_is_hashable::Hashable> final {
  size_t operator()(const test_is_hashable::Hashable&) {
    return 0;
  }
};
} // namespace std

// 匿名命名空间，用于测试函数类型
namespace {

// 命名空间 test_is_hashable 继续
namespace test_is_hashable {

// 静态断言：检查 int 是否可哈希
static_assert(is_hashable<int>::value, "");

// 静态断言：检查 Hashable 类是否可哈希
static_assert(is_hashable<Hashable>::value, "");

// 静态断言：检查 NotHashable 类是否不可哈希
static_assert(!is_hashable<NotHashable>::value, "");

} // namespace test_is_hashable

// 命名空间 test_is_function_type，用于测试函数类型
namespace test_is_function_type {

// 示例类 MyClass
class MyClass {};

// 示例仿函数 Functor
struct Functor {
  void operator()() {}
};

// Lambda 表达式
auto lambda = []() {};

// func() 和 func__ 变量用于消除编译器关于 lambda 未使用的警告
bool func() {
  lambda();
  return true;
}
bool func__ = func();

// 各种函数类型的静态断言，检查是否为函数类型
static_assert(is_function_type<void()>::value, "");
static_assert(is_function_type<int()>::value, "");
static_assert(is_function_type<MyClass()>::value, "");
static_assert(is_function_type<void(MyClass)>::value, "");
static_assert(is_function_type<void(int)>::value, "");
static_assert(is_function_type<void(void*)>::value, "");
static_assert(is_function_type<int()>::value, "");
static_assert(is_function_type<int(MyClass)>::value, "");
static_assert(is_function_type<int(const MyClass&)>::value, "");
static_assert(is_function_type<int(MyClass&&)>::value, "");
static_assert(is_function_type<MyClass && ()>::value, "");
static_assert(is_function_type<MyClass && (MyClass&&)>::value, "");
static_assert(is_function_type<const MyClass&(int, float, MyClass)>::value, "");

// 静态断言：检查非函数类型
static_assert(!is_function_type<void>::value, "");
static_assert(!is_function_type<int>::value, "");
static_assert(!is_function_type<MyClass>::value, "");
static_assert(!is_function_type<void*>::value, "");
static_assert(!is_function_type<const MyClass&>::value, "");
static_assert(!is_function_type<MyClass&&>::value, "");

// 静态断言：检查函数指针和仿函数不是普通函数类型
static_assert(
    !is_function_type<void (*)()>::value,
    "function pointers aren't plain functions");
static_assert(
    !is_function_type<Functor>::value,
    "Functors aren't plain functions");
static_assert(
    !is_function_type<decltype(lambda)>::value,
    "Lambdas aren't plain functions");
```cpp`
// 结束 test_is_function_type 命名空间

namespace test_is_instantiation_of {
// 定义 MyClass 类
class MyClass {};
// 定义模板 Single，接受一个类型参数 T
template <class T>
class Single {};
// 定义模板 Double，接受两个类型参数 T1 和 T2
template <class T1, class T2>
class Double {};
// 定义模板 Multiple，接受可变数量的类型参数 T
template <class... T>
class Multiple {};

// 静态断言：检查 Single<void> 是否为 Single 的实例化
static_assert(is_instantiation_of<Single, Single<void>>::value, "");
// 静态断言：检查 Single<MyClass> 是否为 Single 的实例化
static_assert(is_instantiation_of<Single, Single<MyClass>>::value, "");
// 静态断言：检查 Single<int> 是否为 Single 的实例化
static_assert(is_instantiation_of<Single, Single<int>>::value, "");
// 静态断言：检查 Single<void*> 是否为 Single 的实例化
static_assert(is_instantiation_of<Single, Single<void*>>::value, "");
// 静态断言：检查 Single<int*> 是否为 Single 的实例化
static_assert(is_instantiation_of<Single, Single<int*>>::value, "");
// 静态断言：检查 Single<const MyClass&> 是否为 Single 的实例化
static_assert(is_instantiation_of<Single, Single<const MyClass&>>::value, "");
// 静态断言：检查 Single<MyClass&&> 是否为 Single 的实例化
static_assert(is_instantiation_of<Single, Single<MyClass&&>>::value, "");
// 静态断言：检查 Double<int, void> 是否为 Double 的实例化
static_assert(is_instantiation_of<Double, Double<int, void>>::value, "");
// 静态断言：检查 Double<const int&, MyClass*> 是否为 Double 的实例化
static_assert(
    is_instantiation_of<Double, Double<const int&, MyClass*>>::value,
    "");
// 静态断言：检查 Multiple<> 是否为 Multiple 的实例化
static_assert(is_instantiation_of<Multiple, Multiple<>>::value, "");
// 静态断言：检查 Multiple<int> 是否为 Multiple 的实例化
static_assert(is_instantiation_of<Multiple, Multiple<int>>::value, "");
// 静态断言：检查 Multiple<MyClass&, int> 是否为 Multiple 的实例化
static_assert(
    is_instantiation_of<Multiple, Multiple<MyClass&, int>>::value,
    "");
// 静态断言：检查 Multiple<MyClass&, int, MyClass> 是否为 Multiple 的实例化
static_assert(
    is_instantiation_of<Multiple, Multiple<MyClass&, int, MyClass>>::value,
    "");
// 静态断言：检查 Multiple<MyClass&, int, MyClass, void*> 是否为 Multiple 的实例化
static_assert(
    is_instantiation_of<Multiple, Multiple<MyClass&, int, MyClass, void*>>::
        value,
    "");

// 静态断言：检查 Double<int, int> 是否为 Single 的实例化
static_assert(!is_instantiation_of<Single, Double<int, int>>::value, "");
// 静态断言：检查 Double<int, void> 是否为 Single 的实例化
static_assert(!is_instantiation_of<Single, Double<int, void>>::value, "");
// 静态断言：检查 Multiple<int> 是否为 Single 的实例化
static_assert(!is_instantiation_of<Single, Multiple<int>>::value, "");
// 静态断言：检查 Single<int> 是否为 Double 的实例化
static_assert(!is_instantiation_of<Double, Single<int>>::value, "");
// 静态断言：检查 Multiple<int, int> 是否为 Double 的实例化
static_assert(!is_instantiation_of<Double, Multiple<int, int>>::value, "");
// 静态断言：检查 Multiple<> 是否为 Double 的实例化
static_assert(!is_instantiation_of<Double, Multiple<>>::value, "");
// 静态断言：检查 Double<int, int> 是否为 Multiple 的实例化
static_assert(!is_instantiation_of<Multiple, Double<int, int>>::value, "");
// 静态断言：检查 Single<int> 是否为 Multiple 的实例化
static_assert(!is_instantiation_of<Multiple, Single<int>>::value, "");
} // 结束 test_is_instantiation_of 命名空间

namespace test_is_type_condition {
// 定义模板 NotATypeCondition，接受一个类型参数
template <class>
class NotATypeCondition {};
// 静态断言：检查 std::is_reference 是否为类型条件
static_assert(is_type_condition<std::is_reference>::value, "");
// 静态断言：检查 NotATypeCondition 是否为类型条件
static_assert(!is_type_condition<NotATypeCondition>::value, "");
} // 结束 test_is_type_condition 命名空间
} // 结束命名空间

namespace test_lambda_is_stateless {
// 定义 MyStatelessFunctor 结构模板，接受一个结果类型和多个参数类型
template <class Result, class... Args>
struct MyStatelessFunctor final {
  // 函数调用运算符重载：返回类型为 Result，参数为 Args...
  Result operator()(Args...) {}
};

// 定义 MyStatelessConstFunctor 结构模板，接受一个结果类型和多个参数类型
template <class Result, class... Args>
struct MyStatelessConstFunctor final {
  // 常量函数调用运算符重载：返回类型为 Result，参数为 Args...，修饰为常量成员函数
  Result operator()(Args...) const {}
}
void func() {
  // 定义一个无状态的 Lambda 表达式，接受一个整数参数并返回该参数
  auto stateless_lambda = [](int a) { return a; };
  // 使用 static_assert 检查 stateless_lambda 是否是无状态的 Lambda
  static_assert(is_stateless_lambda<decltype(stateless_lambda)>::value, "");

  int b = 4;
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  // 定义一个捕获了外部变量 b 的有状态 Lambda 表达式，接受一个整数参数并返回参数与 b 的和
  auto stateful_lambda_1 = [&](int a) { return a + b; };
  // 使用 static_assert 检查 stateful_lambda_1 是否是有状态的 Lambda
  static_assert(!is_stateless_lambda<decltype(stateful_lambda_1)>::value, "");

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  // 定义一个捕获了所有外部变量的有状态 Lambda 表达式，接受一个整数参数并返回参数与 b 的和
  auto stateful_lambda_2 = [=](int a) { return a + b; };
  // 使用 static_assert 检查 stateful_lambda_2 是否是有状态的 Lambda
  static_assert(!is_stateless_lambda<decltype(stateful_lambda_2)>::value, "");

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  // 定义一个捕获了变量 b 的有状态 Lambda 表达式，接受一个整数参数并返回参数与 b 的和
  auto stateful_lambda_3 = [b](int a) { return a + b; };
  // 使用 static_assert 检查 stateful_lambda_3 是否是有状态的 Lambda
  static_assert(!is_stateless_lambda<decltype(stateful_lambda_3)>::value, "");

  // 使用 static_assert 检查 MyStatelessFunctor 和 MyStatelessConstFunctor 是否是无状态 Lambda
  static_assert(
      !is_stateless_lambda<MyStatelessFunctor<int, int>>::value,
      "即使是无状态的，一个仿函数不是 Lambda，所以返回 false");
  static_assert(
      !is_stateless_lambda<MyStatelessFunctor<void, int>>::value,
      "即使是无状态的，一个仿函数不是 Lambda，所以返回 false");
  static_assert(
      !is_stateless_lambda<MyStatelessConstFunctor<int, int>>::value,
      "即使是无状态的，一个仿函数不是 Lambda，所以返回 false");
  static_assert(
      !is_stateless_lambda<MyStatelessConstFunctor<void, int>>::value,
      "即使是无状态的，一个仿函数不是 Lambda，所以返回 false");

  // 定义一个 Dummy 类并使用 static_assert 检查它是否是 Lambda
  class Dummy final {};
  static_assert(
      !is_stateless_lambda<Dummy>::value,
      "非仿函数类型也不是 Lambda");

  // 使用 static_assert 检查一个整数是否是 Lambda
  static_assert(!is_stateless_lambda<int>::value, "整数不是 Lambda");

  // 定义一个函数类型 Func，并使用 static_assert 检查它是否是 Lambda
  using Func = int(int);
  static_assert(
      !is_stateless_lambda<Func>::value, "函数类型不是 Lambda");
  static_assert(
      !is_stateless_lambda<Func*>::value, "函数指针不是 Lambda");
}
// namespace test_lambda_is_stateless
// NOLINTEND(modernize-unary-static-assert)
```