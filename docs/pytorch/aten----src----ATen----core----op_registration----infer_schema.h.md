# `.\pytorch\aten\src\ATen\core\op_registration\infer_schema.h`

```
#pragma once

/**
 * This file contains functionality to take a C++ function and infer its
 * c10::FunctionSchema.
 */

#include <ATen/core/function_schema.h>
#include <c10/util/Metaprogramming.h>

namespace c10 {
namespace detail {

namespace infer_schema {

/// The templated inference code creates `ArgumentDef` instead of `Argument`,
/// because that can be constructed at compile time and has a much smaller
/// binary size than having calls to `Argument` constructors in the template.
/// Creating `Argument` objects from `ArgumentDef` can then be done at
/// runtime in a non-templated way.
struct ArgumentDef final {
  using GetTypeFn = TypePtr();  // Define GetTypeFn as a function pointer type returning TypePtr
  GetTypeFn* getTypeFn;         // Pointer to a function returning TypePtr for getting type
  GetTypeFn* getFakeTypeFn;     // Pointer to a function returning TypePtr for getting fake type
  constexpr ArgumentDef(): getTypeFn(nullptr), getFakeTypeFn(nullptr) {}  // Default constructor initializing function pointers to nullptr
  explicit constexpr ArgumentDef(GetTypeFn *getTypeFn, GetTypeFn *getFakeTypeFn): getTypeFn(getTypeFn), getFakeTypeFn(getFakeTypeFn) {}  // Constructor initializing function pointers with provided functions
};

template<bool V>
struct bool_t {};  // Template struct for boolean traits

template<> struct bool_t<true> : std::true_type {};   // Specialization for true type
template<> struct bool_t<false> : std::false_type {}; // Specialization for false type

/// Checks the static C++ types `Types` for correctness to catch common error cases.
template <class... Types>
constexpr int checkStaticTypes() {
 // Give nice error messages for some of the common error cases.
 // Use a LOUD ERROR MESSAGE SO USERS SEE THE STATIC_ASSERT
 static_assert(std::conjunction<
     bool_t<!std::is_integral<Types>::value || std::is_same<Types, int8_t>::value || std::is_same<Types, int64_t>::value || std::is_same<Types, bool>::value>...
   >::value, "INVALID TYPE: Only int8_t, int64_t and bool are supported as an integral argument type");
 static_assert(std::conjunction<
     bool_t<!std::is_same<Types, float>::value>...
   >::value, "INVALID TYPE: float is not supported as an argument type, use double instead");
 return 0;  // Return 0 indicating success
}

template <typename... Ts, size_t... Is>
constexpr std::array<ArgumentDef, sizeof...(Ts)> createArgumentVectorFromTypes(std::index_sequence<Is...>) {
  return (
    // Check types for common errors
    checkStaticTypes<Ts...>(),

    // Create the return value
    std::array<ArgumentDef, sizeof...(Ts)>{
      ArgumentDef(&getTypePtrCopy<std::decay_t<Ts>>, &getFakeTypePtrCopy<std::decay_t<Ts>>)...}
  );
}

/// Creates a vector of `ArgumentDef` from a list of C++ types that are specified
/// as template arguments.
template<class ParameterTypes> struct createArguments final {};
template<class... ParameterTypes>
struct createArguments<guts::typelist::typelist<ParameterTypes...>> final {
  static constexpr std::array<ArgumentDef, sizeof...(ParameterTypes)> call() {
    return createArgumentVectorFromTypes<ParameterTypes...>(
        std::make_index_sequence<sizeof...(ParameterTypes)>()
    );
  }
};

/// Creates a vector of `ArgumentDef` from a list of C++ types that are specified
/// as a tuple (i.e. in the way c10 kernels return values).
/// It can be a tuple<A, B, C> if there's three output arguments with types A, B, C.
/// 根据模板参数创建返回类型的定义
template<class ReturnTypeTuple, class Enable = void> struct createReturns final {};

/// 当返回类型为 std::tuple<ReturnTypes...> 时的特化
template<class... ReturnTypes>
struct createReturns<std::tuple<ReturnTypes...>, void> final {
  /// 生成返回类型的参数定义数组，从参数类型生成 ArgumentDef 对象
  static constexpr std::array<ArgumentDef, sizeof...(ReturnTypes)> call() {
    return createArgumentVectorFromTypes<ReturnTypes...>(
        std::make_index_sequence<sizeof...(ReturnTypes)>()
    );
  }
};

/// 当返回类型为单一类型 ReturnType 且不是 void 或 std::tuple 时的特化
template<class ReturnType>
struct createReturns<ReturnType, std::enable_if_t<!std::is_same<void, ReturnType>::value && !guts::is_instantiation_of<std::tuple, ReturnType>::value>> final {
  /// 生成单一返回类型的参数定义数组
  static constexpr std::array<ArgumentDef, 1> call() {
    return createReturns<std::tuple<ReturnType>>::call();
  }
};

/// 当返回类型为 void 时的特化
template<>
struct createReturns<void, void> final {
  /// 生成空参数定义数组
  static constexpr std::array<ArgumentDef, 0> call() {
    return createReturns<std::tuple<>>::call();
  }
};

/// 根据单一返回类型创建返回类型的定义
template <typename ReturnType>
struct createSingleReturn {
  /// 生成单一返回类型的参数定义数组
  static constexpr std::array<ArgumentDef, 1> call() {
    return createArgumentVectorFromTypes<ReturnType>(std::make_index_sequence<1>());
  }
};

/// 从 FunctionTraits 类型创建 FunctionSchema 对象，对 std::tuple 返回类型进行扁平化处理
template <typename FunctionTraits>
FunctionSchema createFunctionSchemaFromTraitsFlattenedReturns() {
  using ReturnType = typename FunctionTraits::return_type;
  using ParameterTypes = typename FunctionTraits::parameter_types;

  /// 在编译时计算参数和返回值的 std::array，并嵌入到二进制文件中
  constexpr auto arguments = createArguments<ParameterTypes>::call();
  constexpr auto returns = createReturns<ReturnType>::call();

  /// 创建 FunctionSchema 对象
  return make_function_schema(arguments, returns);
}

/// 从 FunctionTraits 类型创建 FunctionSchema 对象，保留 std::tuple 返回类型作为元组返回类型
template <typename FunctionTraits>
// 使用函数特征创建函数模式，该函数适用于单返回值的情况
FunctionSchema createFunctionSchemaFromTraitsSingleReturn(std::string&& name, std::string&& overload_name) {
    // 定义返回类型为函数特征的返回类型
    using ReturnType = typename FunctionTraits::return_type;
    // 定义参数类型为函数特征的参数类型
    using ParameterTypes = typename FunctionTraits::parameter_types;

    // 在编译时计算参数和返回值，并将它们嵌入二进制文件中的 std::array 中。
    // 这里在运行时唯一执行的代码是创建参数/返回值的 std::vector
    constexpr auto arguments = createArguments<ParameterTypes>::call();
    constexpr auto returns = createSingleReturn<ReturnType>::call();

    // 使用移动语义创建函数模式，并返回
    return make_function_schema(std::move(name), std::move(overload_name), arguments, returns);
}

// 推断函数模式，用于函数具有展平返回值的情况
template<class FuncType>
FunctionSchema inferFunctionSchemaFlattenedReturns() {
    // 推断函数类型的函数特征，创建函数模式，并返回
    return detail::infer_schema::createFunctionSchemaFromTraitsFlattenedReturns<guts::infer_function_traits_t<FuncType>>();
}

// 推断函数模式，用于函数具有单返回值的情况
template<class FuncType>
FunctionSchema inferFunctionSchemaSingleReturn(std::string&& name, std::string&& overload_name) {
    // 推断函数类型的函数特征，创建函数模式，并返回
    return detail::infer_schema::createFunctionSchemaFromTraitsSingleReturn<guts::infer_function_traits_t<FuncType>>(std::move(name), std::move(overload_name));
}

// 在 Torch API 中寻找推断的函数模式与指定的函数模式之间的差异，返回一个可选的字符串
TORCH_API std::optional<std::string> findSchemaDifferences(const FunctionSchema& inferred, const FunctionSchema& specified);
```