# `.\pytorch\aten\src\ATen\core\boxing\impl\WrapFunctionIntoRuntimeFunctor.h`

```py
#pragma once

# 预处理指令，确保当前头文件只被编译一次，避免重复包含


#include <c10/util/TypeTraits.h>

# 包含头文件 `<c10/util/TypeTraits.h>`，用于类型特性的工具函数和类


namespace c10 {

# 进入命名空间 `c10`


namespace impl {
  namespace detail {

# 嵌套命名空间 `impl` 和 `detail`，用于实现细节


template<class FuncType, class ReturnType, class ParameterList> class WrapFunctionIntoRuntimeFunctor_ {};

# 定义模板类 `WrapFunctionIntoRuntimeFunctor_`，用于将运行时函数包装为运行时函数对象


template<class FuncType, class ReturnType, class... Parameters>
class WrapFunctionIntoRuntimeFunctor_<FuncType, ReturnType, guts::typelist::typelist<Parameters...>> final : public c10::OperatorKernel {
public:
  template<class FuncType_>
  explicit WrapFunctionIntoRuntimeFunctor_(FuncType_&& kernel_func)
  : kernel_func_(std::forward<FuncType_>(kernel_func)) {}

  decltype(auto) operator()(Parameters... args) {
    return kernel_func_(std::forward<Parameters>(args)...);
  }

private:
  FuncType kernel_func_;
};

# 定义模板特化类 `WrapFunctionIntoRuntimeFunctor_`，继承自 `c10::OperatorKernel`，用于将函数包装成运行时函数对象


// WrapFunctionIntoRuntimeFunctor: Wraps any runtime functor into a functor that
// inherits from c10::OperatorKernel, so it can be used as a c10 kernel.
// This can, for example, be used for lambdas, functors or even function pointers.
// In the case of function pointers, since it is a runtime function pointer,
// there is an overhead for calling it whenever the kernel is invoked.

# 注释说明 `WrapFunctionIntoRuntimeFunctor` 类，将任意运行时函数包装成继承自 `c10::OperatorKernel` 的函数对象，可用作 c10 内核。适用于 lambda 表达式、函数对象或函数指针。


template<class FuncType>
using WrapFunctionIntoRuntimeFunctor = detail::WrapFunctionIntoRuntimeFunctor_<
    FuncType,
    typename guts::infer_function_traits_t<FuncType>::return_type,
    typename guts::infer_function_traits_t<FuncType>::parameter_types
>;

# 定义模板别名 `WrapFunctionIntoRuntimeFunctor`，简化使用 `WrapFunctionIntoRuntimeFunctor_` 类的方式，根据 `FuncType` 推导其返回类型和参数类型。


}
}

# 结束命名空间 `detail` 和 `impl`


}

# 结束命名空间 `c10`
```