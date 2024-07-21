# `.\pytorch\aten\src\ATen\core\boxing\KernelFunction.h`

```py
#pragma once
// 声明本头文件只能包含一次

#include <ATen/core/ATen_fwd.h>
// 引入 ATen 库的前置声明头文件

#include <ATen/core/boxing/BoxedKernel.h>
// 引入 ATen 库中的 BoxedKernel 头文件

#include <ATen/core/stack.h>
// 引入 ATen 库中的 stack 头文件

#include <c10/core/DispatchKeySet.h>
// 引入 c10 库中的 DispatchKeySet 头文件

#include <c10/util/intrusive_ptr.h>
// 引入 c10 库中的 intrusive_ptr 头文件

#include <c10/util/TypeList.h>
// 引入 c10 库中的 TypeList 头文件

#include <type_traits>
// 引入标准类型特性头文件

namespace c10 {

using Stack = torch::jit::Stack;
// 使用 torch::jit::Stack 类型别名为 Stack，并置于 c10 命名空间中

class OperatorHandle;
// 声明 OperatorHandle 类

struct OperatorKernel;
// 声明 OperatorKernel 结构体

class KernelFunction;
// 声明 KernelFunction 类

template <typename T>
using has_symint =
  std::disjunction<
    std::is_same<c10::SymInt, T>,
    std::is_same<c10::SymIntArrayRef, T>,
    std::is_same<at::OptionalSymIntArrayRef, T>,
    std::is_same<std::optional<c10::SymInt>, T>
  >;
// 定义模板 has_symint，用于检查类型 T 是否为 SymInt 相关类型之一

template <typename T>
struct remove_symint {
  using type = T;
};
// 定义模板结构 remove_symint，如果 T 不是 SymInt 相关类型，则类型保持不变

template <>
struct remove_symint<c10::SymInt> {
  using type = int64_t;
};
// 对于类型为 c10::SymInt 的特化，将其替换为 int64_t 类型

template <>
struct remove_symint<at::OptionalSymIntArrayRef> {
  using type = OptionalIntArrayRef;
};
// 对于类型为 at::OptionalSymIntArrayRef 的特化，将其替换为 OptionalIntArrayRef 类型

template <>
struct remove_symint<c10::SymIntArrayRef> {
  using type = c10::IntArrayRef;
};
// 对于类型为 c10::SymIntArrayRef 的特化，将其替换为 c10::IntArrayRef 类型

template <>
struct remove_symint<std::optional<c10::SymInt>> {
  using type = std::optional<int64_t>;
};
// 对于类型为 std::optional<c10::SymInt> 的特化，将其替换为 std::optional<int64_t> 类型

template <bool symint, typename T>
struct maybe_keep_symint final {};
// 定义模板结构 maybe_keep_symint，根据 symint 的值选择是否保持 SymInt 类型

template <typename T>
struct maybe_keep_symint<true, T> { using type = T; };
// 如果 symint 为 true，则保持 T 类型不变

template <typename T>
struct maybe_keep_symint<false, T> { using type = typename remove_symint<T>::type; };
// 如果 symint 为 false，则使用 remove_symint 结构中的类型替换 T 类型

template <typename T>
using fn_has_symint = typename guts::typelist::true_for_any_type<
  has_symint,
  typename guts::infer_function_traits<T>::type::parameter_types
>;
// 定义模板 fn_has_symint，检查函数类型 T 的参数中是否有 SymInt 相关类型之一

template <typename T>
struct fn_remove_symint;

template <typename Ret, typename... Args>
struct fn_remove_symint<Ret(Args...)> {
  using type = Ret(typename remove_symint<Args>::type...);
};
// 定义模板 fn_remove_symint，移除函数类型中参数的 SymInt 相关类型

/**
 * KernelFunction 类类似于 std::function，但用于存储内核函数。
 * 可以从箱式或非箱式的函数/函数对象/lambda 创建 KernelFunction，
 * 并以箱式或非箱式方式调用它。如果创建方式与调用方式不匹配，
 * 将根据需要进行箱式或非箱式处理。
 */
class TORCH_API KernelFunction final {
private:

  explicit KernelFunction(
      std::unique_ptr<OperatorKernel> functor,
      InternalBoxedKernelFunction* boxed_kernel_func,
      void* unboxed_kernel_func,
      void* sym_unboxed_kernel_func);
  // 私有构造函数，用于初始化 KernelFunction，接受不同类型的内核函数作为参数

  explicit KernelFunction(
      BoxedKernel boxed_fn,
      void* unboxed_kernel_func,
      void* sym_unboxed_kernel_func);
  // 私有构造函数，用于初始化 KernelFunction，接受箱式函数作为参数

  BoxedKernel boxed_kernel_func_;
  void* unboxed_kernel_func_;
  void* sym_unboxed_kernel_func_;
  // 成员变量，分别存储箱式内核函数、非箱式内核函数和符号非箱式内核函数的指针
};

}

#include <ATen/core/boxing/KernelFunction_impl.h>
// 引入 KernelFunction 的实现头文件
```