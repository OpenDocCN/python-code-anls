# `.\pytorch\aten\src\ATen\detail\FunctionTraits.h`

```py
#pragma once

#include <cstddef>
#include <tuple>

// Modified from https://stackoverflow.com/questions/7943525/is-it-possible-to-figure-out-the-parameter-type-and-return-type-of-a-lambda

// Fallback, anything with an operator()
template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {
    // 使用模板元编程，获取任何具有 operator() 的类型的函数特征
};

// Pointers to class members that are themselves functors.
// For example, in the following code:
// template <typename func_t>
// struct S {
//     func_t f;
// };
// template <typename func_t>
// S<func_t> make_s(func_t f) {
//     return S<func_t> { .f = f };
// }
//
// auto s = make_s([] (int, float) -> double { /* ... */ });
//
// function_traits<decltype(&s::f)> traits;
template <typename ClassType, typename T>
struct function_traits<T ClassType::*> : public function_traits<T> {
    // 处理成员函数指针，继承自 T 类型的函数特征
};

// Const class member functions
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const> : public function_traits<ReturnType(Args...)> {
    // 处理类的 const 成员函数，继承自 ReturnType(Args...) 的函数特征
};

// Reference types
template <typename T>
struct function_traits<T&> : public function_traits<T> {};
template <typename T>
struct function_traits<T*> : public function_traits<T> {};

// Free functions
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType(Args...)> {
    // 函数特征模板，用于捕获返回类型和参数类型
    enum { arity = sizeof...(Args) }; // 参数个数

    using ArgsTuple = std::tuple<Args...>; // 参数类型元组
    using result_type = ReturnType; // 返回类型

    template <size_t i>
    struct arg
    {
        using type = typename std::tuple_element<i, std::tuple<Args...>>::type;
        // 第 i 个参数的类型，等同于参数类型元组的第 i 个元素类型
    };
};

template <typename T>
struct nullary_function_traits {
    using traits = function_traits<T>;
    using result_type = typename traits::result_type;
    // 零元函数特征，获取返回类型
};

template <typename T>
struct unary_function_traits {
    using traits = function_traits<T>;
    using result_type = typename traits::result_type;
    using arg1_t = typename traits::template arg<0>::type;
    // 一元函数特征，获取返回类型和第一个参数类型
};

template <typename T>
struct binary_function_traits {
    using traits = function_traits<T>;
    using result_type = typename traits::result_type;
    using arg1_t = typename traits::template arg<0>::type;
    using arg2_t = typename traits::template arg<1>::type;
    // 二元函数特征，获取返回类型和前两个参数类型
};


// Traits for calling with c10::guts::invoke, where member_functions have a first argument of ClassType
template <typename T>
struct invoke_traits : public function_traits<T>{
    // 用于 c10::guts::invoke 调用的函数特征，继承自 T 类型的函数特征
};

template <typename T>
struct invoke_traits<T&> : public invoke_traits<T>{
    // 引用类型的 invoke 特征，继承自 T 的 invoke 特征
};

template <typename T>
struct invoke_traits<T&&> : public invoke_traits<T>{
    // 右值引用类型的 invoke 特征，继承自 T 的 invoke 特征
};

template <typename ClassType, typename ReturnType, typename... Args>
struct invoke_traits<ReturnType(ClassType::*)(Args...)> :
  public function_traits<ReturnType(ClassType&, Args...)> {
    // 类成员函数指针的 invoke 特征，继承自 ReturnType(ClassType&, Args...) 的函数特征
};

template <typename ClassType, typename ReturnType, typename... Args>
# 定义模板结构体 `invoke_traits`，用于提取成员函数指针类型的常量成员函数的特征
struct invoke_traits<ReturnType(ClassType::*)(Args...) const> :
  # 继承自 `function_traits`，用于提取常量成员函数的特征
  public function_traits<ReturnType(const ClassType&, Args...)> {
};
```