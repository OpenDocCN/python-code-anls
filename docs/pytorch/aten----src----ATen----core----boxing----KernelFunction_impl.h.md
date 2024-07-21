# `.\pytorch\aten\src\ATen\core\boxing\KernelFunction_impl.h`

```py
// 包含 ATen 库中有关 boxing 的头文件
#include <ATen/core/boxing/impl/boxing.h>
#include <ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h>
#include <ATen/core/boxing/impl/WrapFunctionIntoFunctor.h>
#include <ATen/core/boxing/impl/WrapFunctionIntoRuntimeFunctor.h>

// 包含 C++17 标准库的一些实用功能
#include <c10/util/C++17.h>
#include <type_traits>

// 定义 c10 命名空间
namespace c10 {

// KernelFunction 类的默认构造函数实现
inline KernelFunction::KernelFunction()
    : boxed_kernel_func_()                        // 初始化 boxed_kernel_func_ 为空
    , unboxed_kernel_func_(nullptr)               // 初始化 unboxed_kernel_func_ 为空指针
    , sym_unboxed_kernel_func_(nullptr)           // 初始化 sym_unboxed_kernel_func_ 为空指针
{}

// KernelFunction 类的构造函数，接受一个 std::unique_ptr<OperatorKernel> 和两个指针作为参数
inline KernelFunction::KernelFunction(std::unique_ptr<OperatorKernel> functor, InternalBoxedKernelFunction* boxed_kernel_func, void* unboxed_kernel_func, void* sym_unboxed_kernel_func /* = nullptr */)
  : boxed_kernel_func_(std::move(functor), boxed_kernel_func)   // 初始化 boxed_kernel_func_ 使用移动语义和 boxed_kernel_func 参数
  , unboxed_kernel_func_(unboxed_kernel_func)                   // 初始化 unboxed_kernel_func_ 使用 unboxed_kernel_func 参数
  , sym_unboxed_kernel_func_(sym_unboxed_kernel_func)           // 初始化 sym_unboxed_kernel_func_ 使用 sym_unboxed_kernel_func 参数
{}

// KernelFunction 类的构造函数，接受一个 BoxedKernel 和两个指针作为参数
inline KernelFunction::KernelFunction(BoxedKernel boxed_fn, void* unboxed_kernel_func, void* sym_unboxed_kernel_func /* = nullptr */)
  : boxed_kernel_func_(std::move(boxed_fn))      // 初始化 boxed_kernel_func_ 使用移动语义和 boxed_fn 参数
  , unboxed_kernel_func_(unboxed_kernel_func)    // 初始化 unboxed_kernel_func_ 使用 unboxed_kernel_func 参数
  , sym_unboxed_kernel_func_(sym_unboxed_kernel_func)  // 初始化 sym_unboxed_kernel_func_ 使用 sym_unboxed_kernel_func 参数
{}

// 检查 unboxed_kernel_func_ 是否有效的成员函数
inline bool KernelFunction::isValidUnboxed() const {
  return unboxed_kernel_func_ != nullptr;
}

// 检查 sym_unboxed_kernel_func_ 是否有效的成员函数
inline bool KernelFunction::isValidSymUnboxed() const {
  return sym_unboxed_kernel_func_ != nullptr;
}

// 检查 boxed_kernel_func_ 是否有效的成员函数
inline bool KernelFunction::isValid() const {
  return boxed_kernel_func_.isValid();
}

// 检查 boxed_kernel_func_ 是否为 fallthrough 的成员函数
inline bool KernelFunction::isFallthrough() const {
  return boxed_kernel_func_.isFallthrough();
}

// 调用 boxed_kernel_func_ 的 callBoxed 成员函数，执行包装的内核函数
inline void KernelFunction::callBoxed(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Stack* stack) const {
  boxed_kernel_func_.callBoxed(opHandle, dispatchKeySet, stack);
}

// 调用未包装的内核函数，使用给定的 unboxed_kernel_func 指针和 functor 参数，执行内核函数
template<class Return, class... Args>
inline Return callUnboxedKernelFunction(void* unboxed_kernel_func, OperatorKernel* functor, DispatchKeySet dispatchKeySet, Args&&... args) {
    using ActualSignature = Return (OperatorKernel*, DispatchKeySet, Args...);
    ActualSignature* func = reinterpret_cast<ActualSignature*>(unboxed_kernel_func);
    return (*func)(functor, dispatchKeySet, std::forward<Args>(args)...);
}

// 模板函数，用于取消包装 SymInt 类型的参数，返回其原始类型
// 如果模板类型 T 不是 SymInt，则返回参数本身
template <typename T>
inline typename remove_symint<T>::type unpackSymInt(T x) { return x; }

// 特化模板函数，用于取消包装 SymInt 类型的参数，返回其原始类型
template <>
inline typename remove_symint<c10::SymInt>::type unpackSymInt(c10::SymInt x) {
  return x.guard_int(__FILE__, __LINE__);  // 调用 guard_int 函数，对 SymInt 进行解包处理
}

// 特化模板函数，用于取消包装 SymIntArrayRef 类型的参数，返回其原始类型
template <>
inline typename remove_symint<c10::SymIntArrayRef>::type unpackSymInt(c10::SymIntArrayRef x) {
  return C10_AS_INTARRAYREF_SLOW(x);  // 调用 C10_AS_INTARRAYREF_SLOW 宏，对 SymIntArrayRef 进行解包处理
}

// 特化模板函数，用于取消包装 std::optional<c10::SymInt> 类型的参数，返回其原始类型
template <>
inline typename remove_symint<std::optional<c10::SymInt>>::type unpackSymInt(std::optional<c10::SymInt> x) {
  return x.has_value() ? c10::make_optional(x->guard_int(__FILE__, __LINE__)) : c10::nullopt;  // 如果有值，则调用 guard_int 函数处理 SymInt，否则返回 nullopt
}

// 结束模板声明
// 注意：不要省略模板声明的结束部分
// 解包包含可选符号整数数组的值，返回解包后的结果
inline typename remove_symint<at::OptionalSymIntArrayRef>::type unpackSymInt(at::OptionalSymIntArrayRef x) {
  return x.has_value() ? c10::make_optional(C10_AS_INTARRAYREF_SLOW(*x)) : c10::nullopt;
}

// 调用内核函数的模板方法，根据传入的操作符句柄和分发键集，调用相应的内核函数
template<class Return, class... Args>
C10_ALWAYS_INLINE Return KernelFunction::call(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Args... args) const {
    // 注意：上面的 Args 故意不是 Args&&。我们不想进行完美转发，因为这要求 Args 被推断，而是希望调用者显式指定 Args。

    // 如果参数包含符号整数类型，则进行如下处理
    if constexpr (std::disjunction_v<has_symint<Args>...>) {
      // 如果存在符号整数未装箱的内核函数指针，则调用未装箱的内核函数
      if (sym_unboxed_kernel_func_ != nullptr) {
        auto *functor = boxed_kernel_func_.getFunctor();
        return callUnboxedKernelFunction<Return, Args...>(
            sym_unboxed_kernel_func_, functor, dispatchKeySet, std::forward<Args>(args)...);
      }

      // 如果不存在符号整数未装箱的内核函数指针，则调用装箱的内核函数
      if (unboxed_kernel_func_ != nullptr) {
        auto *functor = boxed_kernel_func_.getFunctor();
        return callUnboxedKernelFunction<Return, typename remove_symint<Args>::type...>(
            unboxed_kernel_func_, functor, dispatchKeySet, unpackSymInt<Args>(args)...);
      }
    } else {
      // 如果不包含符号整数类型，则直接调用未装箱的内核函数
      if (C10_LIKELY(unboxed_kernel_func_ != nullptr)) {
        auto *functor = boxed_kernel_func_.getFunctor();
        return callUnboxedKernelFunction<Return, Args...>(
            unboxed_kernel_func_, functor, dispatchKeySet, std::forward<Args>(args)...);
      }
    }

    // 如果以上条件都不满足，则调用包装的内核函数
    return impl::BoxedKernelWrapper<Return(Args...)>::call(
        boxed_kernel_func_,
        opHandle,
        dispatchKeySet,
        std::forward<Args>(args)...
    );
}

// 根据装箱的内核函数生成内核函数对象
inline KernelFunction KernelFunction::makeFromBoxedKernel(BoxedKernel boxed_fn) {
  return KernelFunction(std::move(boxed_fn), nullptr);  // 没有未装箱的函数指针
}

// 根据装箱的函数指针生成内核函数对象的模板特化方法
template<KernelFunction::BoxedKernelFunction* func>
inline KernelFunction KernelFunction::makeFromBoxedFunction() {
  return KernelFunction::makeFromBoxedKernel(
      BoxedKernel::makeFromFunction<func>());
}

// 根据带有分发键的装箱函数指针生成内核函数对象的模板特化方法
template<KernelFunction::BoxedKernelFunction_withDispatchKeys* func>
inline KernelFunction KernelFunction::makeFromBoxedFunction() {
  return KernelFunction::makeFromBoxedKernel(
      BoxedKernel::makeFromFunction<func>());
}

// 生成一个落空的内核函数对象
inline KernelFunction KernelFunction::makeFallthrough() {
  return KernelFunction::makeFromBoxedKernel(
      BoxedKernel::makeFallthrough());
}

// 生成一个不明确的自动微分其他类型的内核函数对象
inline KernelFunction KernelFunction::makeAmbiguousAutogradOther() {
  return KernelFunction::makeFromBoxedKernel(
      BoxedKernel::makeAmbiguousAutogradOther());
}

// 生成一个不支持命名的内核函数对象
inline KernelFunction KernelFunction::makeNamedNotSupported() {
  return KernelFunction::makeFromBoxedKernel(
      BoxedKernel::makeNamedNotSupported());
}

// 根据未装箱的内核函数对象生成内核函数对象
template<bool AllowLegacyTypes, class KernelFunctor>
inline KernelFunction KernelFunction::makeFromUnboxedFunctor(std::unique_ptr<OperatorKernel> kernelFunctor) {
#ifndef NDEBUG
  // 这个断言在调试时打开，对构建时间有一定的开销。

// 这个断言在调试时打开，对构建时间有一定的开销。
#ifndef NDEBUG
    # 使用静态断言检查是否类型KernelFunctor是一个函数对象(functor)，如果不是则输出错误信息
    static_assert(guts::is_functor<KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor> but the argument is not a functor.");
#endif
    // 预处理指令，结束条件编译块，匹配与#ifdef相对应的条件编译指令

    // 使用静态断言确保KernelFunctor派生自OperatorKernel，否则抛出静态断言错误信息
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    // 获取未包装函数指针的地址，并转换为void指针
    auto* unboxed_fn = &impl::wrap_kernel_functor_unboxed<KernelFunctor>::call;
    void* void_unboxed_fn = reinterpret_cast<void*>(unboxed_fn);

    // 检查函数是否具有符号整数类型的特化，用于后续选择函数指针的传递方式
    bool is_symint = fn_has_symint<decltype(unboxed_fn)>::value;

    // 返回一个KernelFunction对象，传递kernelFunctor的移动语义，
    // 包装函数的包装函数指针，以及根据是否具有符号整数类型的条件传递不同的函数指针
    return KernelFunction(
        std::move(kernelFunctor),
        &impl::make_boxed_from_unboxed_functor<KernelFunctor, AllowLegacyTypes>::call,
        is_symint ? nullptr : void_unboxed_fn,
        is_symint ? void_unboxed_fn : nullptr
    );
}

// 使用已包装的函数对象创建KernelFunction对象的静态成员函数
template<class KernelFunctor>
inline KernelFunction KernelFunction::makeFromBoxedFunctor(std::unique_ptr<KernelFunctor> kernelFunctor) {
  return KernelFunction::makeFromBoxedKernel(
      BoxedKernel::makeFromFunctor(std::move(kernelFunctor)));
}

// 使用未包装的函数指针创建KernelFunction对象的静态成员函数
template<class FuncPtr, bool AllowLegacyTypes>
inline KernelFunction KernelFunction::makeFromUnboxedFunction(FuncPtr func_ptr) {
    // 静态断言，确保FuncPtr是编译时函数指针，而非其他类型的指针
    static_assert(is_compile_time_function_pointer<FuncPtr>::value, "Tried to call KernelFunction::makeFromUnboxedFunction with an invalid parameter. It must be a function pointer created with TORCH_FN.");
    // 静态断言，确保FuncPtr不是BoxedKernelFunction的别名
    static_assert(!std::is_same<typename FuncPtr::FuncType, BoxedKernelFunction>::value, "Tried to call KernelFunction::makeFromUnboxedFunction with a boxed function pointer. Please use KernelFunction::makeFromBoxedFunction instead.");
    // 静态断言，确保函数指针不为空
    static_assert(FuncPtr::func_ptr() != nullptr, "Kernel function cannot be nullptr");

    // 在非移动设备上，根据是否允许旧类型创建KernelFunction对象
#if !defined(C10_MOBILE)
    (void)func_ptr; // 抑制未使用变量的警告
    // 调用makeFromUnboxedFunctor生成KernelFunction对象并返回
    return makeFromUnboxedFunctor<AllowLegacyTypes, typename impl::WrapFunctionIntoFunctor<FuncPtr>::type>(
        guts::make_unique_base<OperatorKernel, typename impl::WrapFunctionIntoFunctor<FuncPtr>::type>()
    );
#else
    // 在移动设备上，优化二进制大小而不是性能，使用makeFromUnboxedRuntimeFunction创建KernelFunction对象并返回
    // 该函数避免内联kernel到包装函数中，而是直接使用运行时函数指针
    return makeFromUnboxedRuntimeFunction(func_ptr.func_ptr());
#endif
}

// 使用未包装的运行时函数指针创建KernelFunction对象的静态成员函数
template<bool AllowLegacyTypes, class FuncType>
inline KernelFunction KernelFunction::makeFromUnboxedRuntimeFunction(FuncType* func) {
    // 静态断言，确保FuncType是函数类型
    static_assert(guts::is_function_type<FuncType>::value, "Tried to call KernelFunction::makeFromUnboxedRuntimeFunction with a non-function type.");
    // 静态断言，确保FuncType不是BoxedKernelFunction的别名
    static_assert(!std::is_same<FuncType, BoxedKernelFunction>::value, "Tried to call KernelFunction::makeFromUnboxedRuntimeFunction with a boxed function pointer. Please use KernelFunction::makeFromBoxedFunction instead.");
    // 断言，函数指针不为空
    TORCH_INTERNAL_ASSERT(func != nullptr, "Kernel function cannot be nullptr");
    // 使用 makeFromUnboxedFunctor 函数创建一个对象，并返回该对象
    return makeFromUnboxedFunctor<AllowLegacyTypes, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>(
        // 创建一个 std::decay_t<FuncType> 类型的 WrapFunctionIntoRuntimeFunctor 对象
        guts::make_unique_base<OperatorKernel, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>(func)
    );
}

// 对于模板函数 KernelFunction::makeFromUnboxedLambda，接受一个 Lambda 表达式，当 Lambda 是无状态时使用
template<bool AllowLegacyTypes, class Lambda>
inline std::enable_if_t<guts::is_stateless_lambda<std::decay_t<Lambda>>::value, KernelFunction> KernelFunction::makeFromUnboxedLambda(Lambda&& lambda) {
    // 静态断言，检查 Lambda 类型是否是可调用对象（functor）
    static_assert(guts::is_functor<std::decay_t<Lambda>>::value, "Tried to call KernelFunction::makeFromUnboxedLambda with a non-lambda type.");

    // 如果不是移动端编译
#if !defined(C10_MOBILE)
    // 使用 Lambda 封装成运行时 functor，并调用 makeFromUnboxedFunctor 生成 KernelFunction
    return makeFromUnboxedFunctor<AllowLegacyTypes, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>(
        guts::make_unique_base<OperatorKernel, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>(std::forward<Lambda>(lambda))
    );
#else
    // 在移动端，优化二进制大小而非性能，不内联 kernel 到 wrapper 中，而是使用 makeFromUnboxedRuntimeFunction
    using FuncType = typename guts::infer_function_traits_t<std::decay_t<Lambda>>::func_type;
    return makeFromUnboxedRuntimeFunction<AllowLegacyTypes, FuncType>(lambda);
#endif
}

// 对于模板函数 KernelFunction::makeFromUnboxedLambda，接受一个 Lambda 表达式，当 Lambda 有状态时使用
template<bool AllowLegacyTypes, class Lambda>
inline std::enable_if_t<!guts::is_stateless_lambda<std::decay_t<Lambda>>::value, KernelFunction> KernelFunction::makeFromUnboxedLambda(Lambda&& lambda) {
    // 静态断言，检查 Lambda 类型是否是可调用对象（functor）
    static_assert(guts::is_functor<std::decay_t<Lambda>>::value, "Tried to call KernelFunction::makeFromUnboxedLambda with a non-lambda type.");

    // 使用 Lambda 封装成运行时 functor，并调用 makeFromUnboxedFunctor 生成 KernelFunction
    return makeFromUnboxedFunctor<AllowLegacyTypes, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>(
        guts::make_unique_base<OperatorKernel, impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>(std::forward<Lambda>(lambda))
    );
}

}


这段代码是关于 C++ 中模板函数 `KernelFunction::makeFromUnboxedLambda` 的实现。它根据 Lambda 表达式的状态（有状态或无状态）选择不同的路径来生成 `KernelFunction` 对象。
```