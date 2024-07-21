# `.\pytorch\c10\util\TypeTraits.h`

```py
#pragma once
// 指示编译器只包含此头文件一次

#include <functional>
// 包含功能库中的 <functional> 头文件

#include <type_traits>
// 包含功能库中的 <type_traits> 头文件

namespace c10::guts {

/**
 * is_equality_comparable<T> is true_type iff the equality operator is defined
 * for T.
 */
template <class T, class Enable = void>
struct is_equality_comparable : std::false_type {};
// 默认情况下，假设类型 T 不具有相等性比较运算符，因此将 is_equality_comparable 定义为 false_type

template <class T>
struct is_equality_comparable<
    T,
    std::void_t<decltype(std::declval<T&>() == std::declval<T&>())>>
    : std::true_type {};
// 如果类型 T 支持相等性比较运算符（即 T == T 可以编译通过），则将 is_equality_comparable 设置为 true_type

template <class T>
using is_equality_comparable_t = typename is_equality_comparable<T>::type;
// 使用类型别名，简化对 is_equality_comparable 的调用

/**
 * is_hashable<T> is true_type iff std::hash is defined for T
 */
template <class T, class Enable = void>
struct is_hashable : std::false_type {};
// 默认情况下，假设类型 T 不可哈希，因此将 is_hashable 定义为 false_type

template <class T>
struct is_hashable<T, std::void_t<decltype(std::hash<T>()(std::declval<T&>()))>>
    : std::true_type {};
// 如果类型 T 支持 std::hash，即 std::hash<T>()(T 对象) 可以编译通过，则将 is_hashable 设置为 true_type

template <class T>
using is_hashable_t = typename is_hashable<T>::type;
// 使用类型别名，简化对 is_hashable 的调用

/**
 * is_function_type<T> is true_type iff T is a plain function type (i.e.
 * "Result(Args...)")
 */
template <class T>
struct is_function_type : std::false_type {};
// 默认情况下，假设类型 T 不是函数类型，因此将 is_function_type 定义为 false_type

template <class Result, class... Args>
struct is_function_type<Result(Args...)> : std::true_type {};
// 如果类型 T 是普通函数类型（例如 Result(Args...)），则将 is_function_type 设置为 true_type

template <class T>
using is_function_type_t = typename is_function_type<T>::type;
// 使用类型别名，简化对 is_function_type 的调用

/**
 * is_instantiation_of<T, I> is true_type iff I is a template instantiation of T
 * (e.g. vector<int> is an instantiation of vector) Example:
 *    is_instantiation_of_t<vector, vector<int>> // true
 *    is_instantiation_of_t<pair, pair<int, string>> // true
 *    is_instantiation_of_t<vector, pair<int, string>> // false
 */
template <template <class...> class Template, class T>
struct is_instantiation_of : std::false_type {};
// 默认情况下，假设类型 T 不是模板实例化类型，因此将 is_instantiation_of 定义为 false_type

template <template <class...> class Template, class... Args>
struct is_instantiation_of<Template, Template<Args...>> : std::true_type {};
// 如果类型 T 是模板 Template 的实例化类型（例如 Template<Args...>），则将 is_instantiation_of 设置为 true_type

template <template <class...> class Template, class T>
using is_instantiation_of_t = typename is_instantiation_of<Template, T>::type;
// 使用类型别名，简化对 is_instantiation_of 的调用

namespace detail {
/**
 * strip_class: helper to remove the class type from pointers to `operator()`.
 */

template <typename T>
struct strip_class {};
// 默认情况下，没有 strip_class 的具体实现，用于特化

template <typename Class, typename Result, typename... Args>
struct strip_class<Result (Class::*)(Args...)> {
  using type = Result(Args...);
};
// 如果 T 是成员函数指针，从中提取出类成员函数的类型

template <typename Class, typename Result, typename... Args>
struct strip_class<Result (Class::*)(Args...) const> {
  using type = Result(Args...);
};
// 如果 T 是 const 成员函数指针，从中提取出类 const 成员函数的类型

template <typename T>
using strip_class_t = typename strip_class<T>::type;
// 使用类型别名，简化对 strip_class 的调用
} // namespace detail

/**
 * Evaluates to true_type, iff the given class is a Functor
 * (i.e. has a call operator with some set of arguments)
 */

template <class Functor, class Enable = void>
struct is_functor : std::false_type {};
// 默认情况下，假设类型 Functor 不是函数对象（Functor），因此将 is_functor 定义为 false_type

template <class Functor>
struct is_functor<
    Functor,
    std::enable_if_t<is_function_type<
        detail::strip_class_t<decltype(&Functor::operator())>>::value>>
    : std::true_type {};
// 如果类型 Functor 的 operator() 是一个普通函数类型（即不是成员函数），则将 is_functor 设置为 true_type
/**
 * lambda_is_stateless<T> is true iff the lambda type T is stateless
 * (i.e. does not have a closure).
 * Example:
 *  auto stateless_lambda = [] (int a) {return a;};
 *  lambda_is_stateless<decltype(stateless_lambda)> // true
 *  auto stateful_lambda = [&] (int a) {return a;};
 *  lambda_is_stateless<decltype(stateful_lambda)> // false
 */
// 命名空间 detail 开始
namespace detail {
// LambdaType 为模板参数，FuncType 为函数类型
template <class LambdaType, class FuncType>
struct is_stateless_lambda__ final {
  // 断言，不应该触发基础情况
  static_assert(
      !std::is_same_v<LambdaType, LambdaType>,
      "Base case shouldn't be hit");
};

// 根据 C++ 标准，无状态的 lambda 可以转换为函数指针
template <class LambdaType, class C, class Result, class... Args>
struct is_stateless_lambda__<LambdaType, Result (C::*)(Args...) const>
    : std::is_convertible<LambdaType, Result (*)(Args...)> {};

// 非 const 成员函数指针的情况
template <class LambdaType, class C, class Result, class... Args>
struct is_stateless_lambda__<LambdaType, Result (C::*)(Args...)>
    : std::is_convertible<LambdaType, Result (*)(Args...)> {};

// LambdaType 不是 functor 的情况
template <class LambdaType, class Enable = void>
struct is_stateless_lambda_ final : std::false_type {};

// LambdaType 是 functor 的情况
template <class LambdaType>
struct is_stateless_lambda_<
    LambdaType,
    std::enable_if_t<is_functor<LambdaType>::value>>
    : is_stateless_lambda__<LambdaType, decltype(&LambdaType::operator())> {};
} // 命名空间 detail 结束

/**
 * is_type_condition<C> is true_type iff C<...> is a type trait representing a
 * condition (i.e. has a constexpr static bool ::value member) Example:
 *   is_type_condition<std::is_reference>  // true
 */
// 模板参数 C 为模板类型，Enable 为 void 类型
template <template <class> class C, class Enable = void>
struct is_type_condition : std::false_type {};

// 特化模板，C<int>::value 的结果为 bool 类型
template <template <class> class C>
struct is_type_condition<
    C,
    std::enable_if_t<
        std::is_same_v<bool, std::remove_cv_t<decltype(C<int>::value)>>>>
    : std::true_type {};

/**
 * is_fundamental<T> is true_type iff the lambda type T is a fundamental type
 * (that is, arithmetic type, void, or nullptr_t). Example: is_fundamental<int>
 * // true We define it here to resolve a MSVC bug. See
 * https://github.com/pytorch/pytorch/issues/30932 for details.
 */
// 模板参数 T 为类型
template <class T>
struct is_fundamental : std::is_fundamental<T> {};
} // 命名空间 c10::guts 结束
```