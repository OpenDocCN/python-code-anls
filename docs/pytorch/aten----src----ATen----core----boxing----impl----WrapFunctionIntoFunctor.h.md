# `.\pytorch\aten\src\ATen\core\boxing\impl\WrapFunctionIntoFunctor.h`

```py
#pragma once
// 预处理指令：确保本头文件只被编译一次

#include <c10/core/CompileTimeFunctionPointer.h>
// 包含头文件：引入CompileTimeFunctionPointer.h，其中可能定义了函数指针相关的功能

namespace c10 {
namespace impl {
  namespace detail {
    template<class FuncPtr, class ReturnType, class ParameterList> class WrapFunctionIntoFunctor_ {};
    // 声明模板类WrapFunctionIntoFunctor_，用于将函数指针包装成仿函数，具体实现留待后续定义
    template<class FuncPtr, class ReturnType, class... Parameters>
    class WrapFunctionIntoFunctor_<FuncPtr, ReturnType, guts::typelist::typelist<Parameters...>> final : public c10::OperatorKernel {
    public:
      // 类模板特化：继承OperatorKernel，重载操作符()，用于调用函数指针并返回结果
      C10_ALWAYS_INLINE decltype(auto) operator()(Parameters... args) {
        return (*FuncPtr::func_ptr())(std::forward<Parameters>(args)...);
      }
    };
  }

  // WrapFunctionIntoFunctor: Wraps a compile time function pointer into a kernel functor.
  // Since it is a compile time function pointer, many compilers can inline it
  // into the wrapper and you don't get any performance overhead for wrapping.
  // 模板结构体WrapFunctionIntoFunctor：将编译时函数指针包装成内核仿函数
  // 由于它是编译时函数指针，许多编译器可以将其内联到包装器中，因此包装过程不会增加性能开销。
  template<class FuncPtr>
  struct WrapFunctionIntoFunctor final {
    static_assert(c10::is_compile_time_function_pointer<FuncPtr>::value, "WrapFunctionIntoFunctor can only wrap functions created with TORCH_FN.");
    // 静态断言：确保FuncPtr是通过TORCH_FN创建的编译时函数指针

    using type = detail::WrapFunctionIntoFunctor_<
        FuncPtr,
        typename guts::function_traits<typename FuncPtr::FuncType>::return_type,
        typename guts::function_traits<typename FuncPtr::FuncType>::parameter_types
    >;
    // type类型别名：使用detail::WrapFunctionIntoFunctor_实例化模板，包含FuncPtr的返回类型和参数类型
  };
}

}
```