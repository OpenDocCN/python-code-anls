# `.\pytorch\aten\src\ATen\core\boxing\BoxedKernel_impl.h`

```
#pragma once

namespace c10 {

// 默认构造函数，初始化 functor_ 为空，boxed_kernel_func_ 为空指针
inline BoxedKernel::BoxedKernel()
    : functor_()
    , boxed_kernel_func_(nullptr)
{}

// 构造函数，接受一个 unique_ptr 的 OperatorKernel 对象和一个 InternalBoxedKernelFunction 指针
inline BoxedKernel::BoxedKernel(std::unique_ptr<OperatorKernel> functor, InternalBoxedKernelFunction* boxed_kernel_func)
    : functor_(std::move(functor))
    , boxed_kernel_func_(boxed_kernel_func)
{}

// 模板函数，接受一个 BoxedKernelFunction 指针 func，用于生成 boxed function
template<BoxedKernel::BoxedKernelFunction* func>
inline void BoxedKernel::make_boxed_function(OperatorKernel*, const OperatorHandle& opHandle, DispatchKeySet, Stack* stack) {
    // 注意：我们省略了 DispatchKeySet 参数。
    // 详见 Note [Plumbing Keys Through The Dispatcher 2] 以获取更多详情。
    func(opHandle, stack);
}

// 模板函数，接受一个 BoxedKernelFunction_withDispatchKeys 指针 func，用于生成 boxed function
template<BoxedKernel::BoxedKernelFunction_withDispatchKeys* func>
inline void BoxedKernel::make_boxed_function(OperatorKernel*, const OperatorHandle& opHandle, DispatchKeySet ks, Stack* stack) {
    // 详见 Note [Plumbing Keys Through The Dispatcher 2] 以获取更多详情。
    func(opHandle, ks, stack);
}

// 检查 boxed_kernel_func_ 是否为非空，用于验证 BoxedKernel 的有效性
inline bool BoxedKernel::isValid() const {
    return boxed_kernel_func_ != nullptr;
}

// 检查 boxed_kernel_func_ 是否为 fallthrough_kernel，用于检测是否为 fallthrough 函数
inline bool BoxedKernel::isFallthrough() const {
    return boxed_kernel_func_ == &fallthrough_kernel;
}

// 调用 boxed_kernel_func_ 执行具体的 boxed 函数
inline void BoxedKernel::callBoxed(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Stack* stack) const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        boxed_kernel_func_ != nullptr,
        "Tried to call BoxedKernel::callBoxed() on an uninitialized BoxedKernel."
    );
    (*boxed_kernel_func_)(functor_.get(), opHandle, dispatchKeySet, stack);
}

// 通过给定的 func 模板参数创建 BoxedKernel 对象
template<BoxedKernel::BoxedKernelFunction* func>
inline BoxedKernel BoxedKernel::makeFromFunction() {
    return BoxedKernel(
        nullptr,  // 没有 functor_ 对象
        &make_boxed_function<func>
    );
}

// 通过给定的 func 模板参数创建带 DispatchKey 的 BoxedKernel 对象
template<BoxedKernel::BoxedKernelFunction_withDispatchKeys* func>
inline BoxedKernel BoxedKernel::makeFromFunction() {
    return BoxedKernel(
        nullptr,  // 没有 functor_ 对象
        &make_boxed_function<func>
    );
}

// 创建一个 fallthrough 类型的 BoxedKernel 对象
inline BoxedKernel BoxedKernel::makeFallthrough() {
    return BoxedKernel(
        nullptr,  // 没有 functor_ 对象
        &fallthrough_kernel
    );
}

// 创建一个 ambiguous_autogradother_kernel 类型的 BoxedKernel 对象
inline BoxedKernel BoxedKernel::makeAmbiguousAutogradOther() {
    return BoxedKernel(
        nullptr,  // 没有 functor_ 对象
        &ambiguous_autogradother_kernel
    );
}

// 创建一个 named_not_supported_kernel 类型的 BoxedKernel 对象
inline BoxedKernel BoxedKernel::makeNamedNotSupported() {
    return BoxedKernel(
        nullptr,  // 没有 functor_ 对象
        &named_not_supported_kernel
    );
}

// 通过给定的 KernelFunctor 对象创建 BoxedKernel 对象，确保 KernelFunctor 继承自 OperatorKernel
template<class KernelFunctor>
inline BoxedKernel BoxedKernel::makeFromFunctor(std::unique_ptr<KernelFunctor> kernelFunctor) {
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to call BoxedKernel::makeFromFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");
    // 返回一个 BoxedKernel 对象，functor_ 为空，boxed_kernel_func_ 指向由 kernelFunctor 生成的 boxed 函数
    return BoxedKernel(
        std::move(kernelFunctor),
        nullptr
    );
}
    # 返回一个 BoxedKernel 对象，使用给定的 kernelFunctor 进行初始化
    return BoxedKernel(
        std::move(kernelFunctor),
        # 创建一个 lambda 表达式作为 BoxedKernel 的第二个参数，用于执行具体的操作
        [](OperatorKernel* kernel, const OperatorHandle& op, DispatchKeySet ks, Stack* stack) {
          # 通过转换类型为 KernelFunctor*，调用 kernel 函数对象来执行操作
          (*static_cast<KernelFunctor*>(kernel))(op, ks, stack);
        }
    );
}



# 结束了 c10 命名空间的定义



inline OperatorKernel* BoxedKernel::getFunctor() const {
  return functor_.get();
}



# 返回封装在 BoxedKernel 对象中的 functor 指针



inline BoxedKernel::InternalBoxedKernelFunction* BoxedKernel::getFnPtr() const {
  return boxed_kernel_func_;
}



# 返回封装在 BoxedKernel 对象中的 boxed_kernel_func_ 指针



}  // namespace c10



# 结束了 c10 命名空间的定义
```