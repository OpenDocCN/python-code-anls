# `.\pytorch\aten\src\ATen\core\boxing\impl\boxing.h`

```
#pragma once

// This file contains boxing (not unboxing) logic,
// i.e. how to make a vector<IValue> from a set of concrete arguments.

#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <c10/core/TensorOptions.h>

#include <ATen/core/boxing/BoxedKernel.h>

#include <c10/util/Metaprogramming.h>
#include <type_traits>

namespace c10 {
namespace impl {

//
// utils
//

// is_mutable_tensor_ref checks if the type T is a reference to at::Tensor
template <class T> struct is_mutable_tensor_ref : std::false_type {};
template <> struct is_mutable_tensor_ref<at::Tensor&> : std::true_type {};

// is_tuple_of_mutable_tensor_refs checks if the type T is a tuple of mutable tensor references
template <class T, class Enable = void>
struct is_tuple_of_mutable_tensor_refs : std::false_type {};

template <class T>
struct is_tuple_of_mutable_tensor_refs<T, std::enable_if_t<guts::is_instantiation_of<std::tuple, T>::value, void>>
: guts::typelist::all<is_mutable_tensor_ref, guts::typelist::from_tuple_t<T>>
{};

// has_ivalue_to<T> tests the presence/absence of instance method IValue::to<T>()
//
template <class T, class Enable = void>
struct has_ivalue_to : std::false_type {};

template <class T>
struct ivalue_to_helper
{
    using type = decltype(std::declval<IValue>().template to<T>());
};
template <class T>
using ivalue_to_helper_t = typename ivalue_to_helper<T>::type;

template <class T>
struct has_ivalue_to<T, std::void_t<ivalue_to_helper_t<T>>>
: std::true_type
{};

//
// boxing predicates
//

// can_box<T> determines if type T can be boxed into an IValue
template <typename T>
using can_box =
  std::disjunction<
    std::is_constructible<IValue, std::decay_t<T>>,
    // TensorOptions are not directly constructible into IValue,
    // but torch::jit::push knows how to handle them
    std::is_same<TensorOptions, std::decay_t<T>>
  >;

// can_box_all<Ts...> checks if all types Ts can be boxed into IValue
template <typename... Ts>
using can_box_all = std::conjunction<can_box<Ts>...>;

// can_unbox<T> determines if type T can be unboxed from an IValue
template <typename T>
using can_unbox =
   std::conjunction<
    std::disjunction<
      has_ivalue_to<T>,
      // void returns are ok
      std::is_same<void, T>
    >,
    std::negation<std::is_lvalue_reference<T>>
  >;

//
// boxArgs - utility for pushing unboxed args onto IValue stack
//
template <class... Args>
torch::jit::Stack boxArgs(Args... args) {
  // TODO Reuse stack vector instead of allocating?
  // Create an empty stack to hold IValues
  torch::jit::Stack stack;
  // Reserve space in the stack for the arguments
  stack.reserve(sizeof...(Args));
  // Push the arguments onto the stack as IValues
  torch::jit::push(stack, std::forward<Args>(args)...);
  // Return the filled stack
  return stack;
}

// boxed_size_one<T>() returns the number of IValues required to box a single T
template <class T>
static inline constexpr size_t boxed_size_one() {
  // Compile-time assertion to ensure TensorOptions are not passed by reference
  static_assert(!std::is_same<std::decay_t<T>, c10::TensorOptions>::value, "need to patch this path to support TensorOptions passed by reference");
  return 1;
}

// Specialization for TensorOptions to return the number of IValues needed
// to box TensorOptions (which is 4)
template <>
inline constexpr size_t boxed_size_one<c10::TensorOptions>() {
  return 4;
}

// NOTE: this could probably be simplified with C++17 fold expressions.
template <typename...>
// 定义一个结构体模板 BoxedSize，继承自 std::integral_constant<size_t, 0>
struct BoxedSize : std::integral_constant<size_t, 0> {};

// BoxedSize 的偏特化模板，计算模板参数包中所有类型的 boxed_size_one<T>() 的总和
template <class T, class... Args>
struct BoxedSize<T, Args...> : std::integral_constant<size_t, boxed_size_one<T>() + BoxedSize<Args...>::value> {};

// 返回一个 constexpr 函数，计算模板参数包 Args 的大小
template <class... Args>
static inline constexpr size_t boxed_size() {
  return BoxedSize<Args...>::value;
}

// 使用 std::aligned_storage_t 创建 IValueAlignedStorage 类型的别名，用于存储 IValue 对象
using IValueAlignedStorage = std::aligned_storage_t<sizeof(IValue), alignof(IValue)>;

// 当不在移动设备上时，将对象 arg 打包到 dest 数组的指定位置，并更新 lastIdx
template <typename T>
C10_ALWAYS_INLINE_UNLESS_MOBILE void boxToStack(IValueAlignedStorage* dest, T& arg, int& lastIdx) {
  new (&dest[lastIdx]) IValue(arg);
  lastIdx++;
}

// 当不在移动设备上时，将 TensorOptions 对象 options 的各属性依次打包到 dest 数组的指定位置，并更新 lastIdx
C10_ALWAYS_INLINE_UNLESS_MOBILE void boxToStack(IValueAlignedStorage* dest, c10::TensorOptions options, int& lastIdx) {
  new (&dest[lastIdx++]) IValue(c10::typeMetaToScalarType(options.dtype()));
  new (&dest[lastIdx++]) IValue(options.layout());
  new (&dest[lastIdx++]) IValue(options.device());
  new (&dest[lastIdx++]) IValue(options.pinned_memory());
}

// 递归终止条件，不做任何操作
inline void boxArgsToStack(IValueAlignedStorage*, int&) {}

// 当不在移动设备上时，将参数 args 及其后续参数打包到 dest 数组的指定位置，并更新 lastIdx
template<typename T, typename... Args>
C10_ALWAYS_INLINE_UNLESS_MOBILE void boxArgsToStack(IValueAlignedStorage* dest, int& lastIdx, T& arg, Args &... args) {
  boxToStack(dest, arg, lastIdx);
  boxArgsToStack(dest, lastIdx, args...);
}

//
// PopResult 是一个辅助类，其特化处理单返回值和多返回值情况。
//

// 对于单一返回值的情况，从栈上弹出一个值并转换为 Result 类型
template <class Result>
struct PopResult final {
  static Result call(Stack& stack) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == 1,
      "Boxed kernel was expected to return one value on the stack, ",
      "but instead pushed ", stack.size(), " values."
    );
    return std::move(stack[0]).to<Result>();
  }
};

// 对于多返回值（tuple）的情况，从栈上弹出多个值并转换为 std::tuple<Types...> 类型
template <class... Types>
struct PopResult<std::tuple<Types...>> final {
  using Result = std::tuple<Types...>;

  static Result call(Stack& stack) {
    // 检查栈上值的数量是否与返回值的类型数量相匹配
    constexpr int RetCount = sizeof...(Types);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == RetCount,
      "Boxed kernel was expected to return ", RetCount, " values on the stack, ",
      "but instead pushed ", stack.size(), " values."
    );
    // 调用内部实现函数将栈上的值转换为 tuple
    return pop_to_tuple_impl(stack, std::make_index_sequence<RetCount>());
  }

private:
  // 将栈上的值转换为 tuple 的内部实现函数
  // 使用 std::index_sequence 来展开参数包 indices
  template <size_t... indices>
  static Result pop_to_tuple_impl(Stack& stack, std::index_sequence<indices...>) {
    return std::make_tuple((std::move(stack[indices]).to<Types>())...);
  }
};

//
// BoxedKernelWrapper
//
// 对于给定的函数类型 FT，BoxedKernelWrapper<FT> 实现了一个 `call` 方法，
// 该方法：
// - 接受一个 boxed kernel 和由 FT 指定的未打包的参数
// - 调用 `boxArgs` 打包参数
// - 调用 boxed kernel
// - 解包并返回结果
//
// The partial specializations below handle various cases: in
// particular, not all types appearing in op signatures are supported,
// and ops returning references have nonstandard wrapper implementations.
//

// 1. The base specialization of BoxedKernelWrapper should never be instantiated.
// A "no call method defined on BoxedKernelWrapper" compile error means that
// an op signature has failed to trigger any of the partial specializations
// that follow this one.
//
template <class FuncType, class Enable = void>
struct BoxedKernelWrapper {
  // The reason we're not just doing straight up static_assert(false, ...) here:
  // Basically, the way to make sure a static_assert only fires if a template
  // is actually instantiated (rather than every time the file is parsed) is to use
  // template parameters in the expression, e.g. FuncType here. However, since
  // `sizeof(FuncType) != sizeof(FuncType)` is always false, this has the same
  // effect.
  static_assert(sizeof(FuncType) != sizeof(FuncType),
     "Function signature contains one or more unsupported parameter and/or return types. "
     "Look for a nearby error like "
     "\"'call' is not a member of 'c10::impl::BoxedKernelWrapper<(your function type), void>'\" "
     "- (your function type) is the unsupported signature.");
};

//
// 2. Supported signatures, other than those involving non-const Tensor refs -
// i.e., "functional" ops.
//

template <class Result, class... Args>
struct BoxedKernelWrapper<
  Result(Args...),
  std::enable_if_t<
    can_box_all<Args...>::value && can_unbox<Result>::value && !is_tuple_of_mutable_tensor_refs<Result>::value,
    void
  >
> {
  // Implementation of the call method for supported signatures.
  static Result call(
    const BoxedKernel& boxed_kernel_func,
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    Args... args
  ) {
    // Boxing arguments into torch::jit::Stack.
    torch::jit::Stack stack = boxArgs<Args...>(std::forward<Args>(args)...);
    // Calling the boxed kernel function with boxed arguments.
    boxed_kernel_func.callBoxed(opHandle, dispatchKeySet, &stack);

    if constexpr (!std::is_same_v<void, Result>) {
        // If the result type is not void, retrieve the result from the stack.
        return PopResult<Result>::call(stack);
    } else {
      // If the result type is void, ensure the stack is empty (no values returned).
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          stack.empty(),
          "Boxed kernel was expected to return no values on the stack, ",
          "but instead returned ", stack.size(), " values."
      );
    }
  }
};

//
// 3. in-place ops take a single non-const Tensor reference
// as their first argument, and return it.
//
// Note: all signatures matching this pattern are assumed to be for such ops.
// Because of this, the generated BoxedKernelWrapper specializations simply
// return the in-place argument.
//

template <class... OtherArgs>
struct BoxedKernelWrapper<
  at::Tensor&(at::Tensor&, OtherArgs...),
  std::enable_if_t<can_box_all<OtherArgs...>::value, void>
> {
  // Implementation of the call method for in-place ops.
  static at::Tensor& call(
    const BoxedKernel& boxed_kernel_func,
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    at::Tensor& tensor,
    OtherArgs... otherArgs
  ) {
    // Calling the boxed kernel function with the tensor argument and other boxed arguments.
    boxed_kernel_func.callBoxed(opHandle, dispatchKeySet, &tensor, boxArgs(otherArgs)...);
    // Returning the modified tensor (in-place operation).
    return tensor;
  }
};
    // 定义函数模板，接受多个参数，包括 DispatchKeySet 和一个引用类型的 Tensor 对象及其它参数
    DispatchKeySet dispatchKeySet,
    // outArg 是一个传入的 Tensor 对象的引用，用于接收函数返回的计算结果
    at::Tensor& outArg, OtherArgs... otherArgs
  ) {
    // 将所有参数（包括 outArg 和其他参数）打包到 torch::jit::Stack 对象中
    torch::jit::Stack stack = boxArgs<at::Tensor&, OtherArgs...>(outArg, std::forward<OtherArgs>(otherArgs)...);
    // 调用封装的 kernel 函数，使用给定的操作句柄和调度键集合，传递封装后的栈对象
    boxed_kernel_func.callBoxed(opHandle, dispatchKeySet, &stack);
    // 使用断言检查封装后的栈中返回值的数量是否为 1，用于调试目的
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == 1,
      "Boxed kernel was expected to return a single value on the stack, ",
      "but instead returned ", stack.size(), " values."
    );

    // 返回传入的 Tensor 对象引用 outArg
    return outArg;
  }
//
// 3.5. In-process migration to make in-place ops take and return
// const references instead.
template <class... OtherArgs>
struct BoxedKernelWrapper<
  const at::Tensor&(const at::Tensor&, OtherArgs...),
  std::enable_if_t<can_box_all<OtherArgs...>::value, void>
> {
  // Define a static function 'call' that wraps boxed kernel functions operating on const references.
  static const at::Tensor& call(
    const BoxedKernel& boxed_kernel_func,
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    const at::Tensor& outArg, OtherArgs... otherArgs
  ) {
    // Box arguments (outArg and otherArgs) into a torch::jit::Stack.
    torch::jit::Stack stack = boxArgs(outArg, otherArgs...);
    // Call the boxed kernel function with the given operator handle, dispatch key set, and stack.
    boxed_kernel_func.callBoxed(opHandle, dispatchKeySet, &stack);
    // Assert that the stack size is 1, indicating the boxed kernel returned a single value.
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == 1,
      "Boxed kernel was expected to return a single value on the stack, ",
      "but instead returned ", stack.size(), " values."
    );

    // Return the original 'outArg' as it is a const reference.
    return outArg;
  }
};

//
// 4. out of place ops that take a single non-const Tensor reference as their
// final argument, and also return it.
//
// Note: all signatures matching this pattern are assumed to be for such ops.
// This assumption permits the generated BoxedKernelWrapper specializations to simply
// return out arguments.
//
template <class FirstArg, class... RestArgs>
struct BoxedKernelWrapper<
  at::Tensor&(FirstArg, RestArgs...),
  std::enable_if_t<
    can_box_all<FirstArg, RestArgs...>::value
    // this skips over in-place kernels with a non-const Tensor
    // arg at the front, so those can unambiguously trigger the preceding specialization.
    && !is_mutable_tensor_ref<FirstArg>::value,
    void
  >
> {
  // Define a static function 'call' that wraps boxed kernel functions returning a non-const Tensor reference.
  static at::Tensor& call(
    const BoxedKernel& boxed_kernel_func,
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    FirstArg firstArg, RestArgs... restArgs
  ) {
    // Box arguments (firstArg and restArgs) into a torch::jit::Stack.
    torch::jit::Stack stack = boxArgs<FirstArg, RestArgs...>(std::forward<FirstArg>(firstArg), std::forward<RestArgs>(restArgs)...);
    // Call the boxed kernel function with the given operator handle, dispatch key set, and stack.
    boxed_kernel_func.callBoxed(opHandle, dispatchKeySet, &stack);
    // Assert that the stack size is 1, indicating the boxed kernel returned a single value.
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == 1,
      "Boxed kernel was expected to return a single value on the stack, ",
      "but instead returned ", stack.size(), " values."
    );

    // Return the last argument in 'restArgs', assuming it is a non-const Tensor reference.
    return std::get<sizeof...(RestArgs) - 1>(std::tuple<RestArgs...>{restArgs...});
  }
};

//
// 5. out of place ops that take multiple non-const Tensor references as their
// final arguments, and return them in a std::tuple.
//
// Note: all signatures matching this pattern are assumed to be for such ops.
// This assumption permits the generated BoxedKernelWrapper specializations to simply
// return the out arguments.
//
template <class Result, class... Args>
struct BoxedKernelWrapper<
  Result(Args...),
  std::enable_if_t<
    can_box_all<Args...>::value && is_tuple_of_mutable_tensor_refs<Result>::value,
    void
  >
> {
  // Define a static function 'call' that wraps boxed kernel functions returning multiple non-const Tensor references.
  static Result call(
    const BoxedKernel& boxed_kernel_func,
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    Args... args
  ) {
    // Box arguments (args) into a torch::jit::Stack.
    torch::jit::Stack stack = boxArgs(args...);
    // Call the boxed kernel function with the given operator handle, dispatch key set, and stack.
    boxed_kernel_func.callBoxed(opHandle, dispatchKeySet, &stack);
    // Assert that the stack size is equal to the number of tensors in 'Result'.
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == std::tuple_size<Result>::value,
      "Boxed kernel was expected to return ", std::tuple_size<Result>::value,
      " values on the stack, but instead returned ", stack.size(), " values."
    );

    // Return a std::tuple constructed from the stack values, each assumed to be a non-const Tensor reference.
    return stackToTuple<Result>(stack);
  }
};
    // 定义一个模板函数，接受操作句柄、调度键集合和可变数量的参数包
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    Args... args
  ) {
    // 定义一个类型别名，表示参数包 Args... 的元组类型
    using ArgTuple = std::tuple<Args...>;
    // constexpr 常量，表示返回结果的元素数量，通过 Result 类型的元组大小确定
    constexpr int RetCount = std::tuple_size<Result>();

    // 将参数 args 转换为 torch::jit::Stack 对象
    torch::jit::Stack stack = boxArgs<Args...>(std::forward<Args>(args)...);
    // 调用 boxed_kernel_func 对象的 callBoxed 方法，传递操作句柄、调度键集合和参数栈的地址
    boxed_kernel_func.callBoxed(opHandle, dispatchKeySet, &stack);
    // 使用 TORCH_INTERNAL_ASSERT_DEBUG_ONLY 宏断言，检查栈的大小是否与 RetCount 相等
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stack.size() == RetCount,
      "Boxed kernel was expected to return ", RetCount, " values on the stack, ",
      "but instead returned ", stack.size(), " values."
    );

    // 重新使用 args 参数包，因为我们知道最后的 RetCount 个元素是 Tensor& 类型
    auto result = guts::tuple_take<ArgTuple, -RetCount>(ArgTuple{std::forward<Args>(args)...});
    // 使用 static_assert 静态断言，验证 result 的类型与 Result 类型相同
    static_assert(
        std::is_same<Result, decltype(result)>::value,
        "The parameter list of an op returning a tuple of Tensor references "
            "must end with an equal number of Tensor reference parameters."
    );
    // 返回 result 结果
    return result;
  }
};

} // impl
} // c10
```