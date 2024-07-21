# `.\pytorch\c10\core\CompileTimeFunctionPointer.h`

```py
#pragma once

#include <c10/util/TypeTraits.h>  // 引入类型特性工具
#include <type_traits>            // 引入类型特性

namespace c10 {

/**
 * Represent a function pointer as a C++ type.
 * This allows using the function pointer as a type
 * in a template and calling it from inside the template
 * allows the compiler to inline the call because it
 * knows the function pointer at compile time.
 *
 * Example 1:
 *  int add(int a, int b) {return a + b;}
 *  using Add = TORCH_FN_TYPE(add);
 *  template<class Func> struct Executor {
 *    int execute(int a, int b) {
 *      return Func::func_ptr()(a, b);
 *    }
 *  };
 *  Executor<Add> executor;
 *  EXPECT_EQ(3, executor.execute(1, 2));
 *
 * Example 2:
 *  int add(int a, int b) {return a + b;}
 *  template<class Func> int execute(Func, int a, int b) {
 *    return Func::func_ptr()(a, b);
 *  }
 *  EXPECT_EQ(3, execute(TORCH_FN(add), 1, 2));
 */

template <class FuncType_, FuncType_* func_ptr_>
struct CompileTimeFunctionPointer final {
  static_assert(
      guts::is_function_type<FuncType_>::value,
      "TORCH_FN can only wrap function types.");  // 静态断言确保 FuncType_ 是函数类型
  using FuncType = FuncType_;  // 定义 FuncType 为模板参数 FuncType_

  static constexpr FuncType* func_ptr() {
    return func_ptr_;  // 返回编译时函数指针
  }
};

template <class T>
struct is_compile_time_function_pointer : std::false_type {};  // 默认为假

template <class FuncType, FuncType* func_ptr>
struct is_compile_time_function_pointer<
    CompileTimeFunctionPointer<FuncType, func_ptr>> : std::true_type {};  // 当为编译时函数指针时设为真

} // namespace c10

#define TORCH_FN_TYPE(func)                                           \
  ::c10::CompileTimeFunctionPointer<                                  \
      std::remove_pointer_t<std::remove_reference_t<decltype(func)>>, \
      func>  // 定义宏 TORCH_FN_TYPE，用于创建 CompileTimeFunctionPointer

#define TORCH_FN(func) TORCH_FN_TYPE(func)()  // 定义宏 TORCH_FN，用于调用 TORCH_FN_TYPE 创建的对象
```