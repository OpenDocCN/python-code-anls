# `.\pytorch\aten\src\ATen\core\op_registration\op_registration.h`

```py
/**
 * Directive to include this header file only once in a compilation unit.
 */
#pragma once

/**
 * Include this file if you want to register operators. It includes all
 * functionality needed to do so for you.
 */
#include <c10/core/DispatchKey.h> // Include definition of DispatchKey
#include <c10/core/DispatchKeySet.h> // Include definition of DispatchKeySet
#include <c10/core/CompileTimeFunctionPointer.h> // Include definition of CompileTimeFunctionPointer
#include <ATen/core/boxing/KernelFunction.h> // Include definition of KernelFunction
#include <ATen/core/dispatch/CppSignature.h> // Include C++ signature utilities
#include <ATen/core/dispatch/RegistrationHandleRAII.h> // Include handle for operator registration
#include <ATen/core/op_registration/infer_schema.h> // Include schema inference utilities
#if defined(EXPOSE_C2_OPS) || !defined(CAFFE2_IS_XPLAT_BUILD)
#include <torch/csrc/jit/frontend/function_schema_parser.h> // Include function schema parser for JIT
#endif
#include <ATen/core/ATenOpList.h> // Include ATen operator list

namespace c10 {

namespace detail {
// The first argument of the schema might be of type DispatchKeySet, in which case we remove it.
// We do this because every argument in a function schema is expected to be convertable
// to an ivalue, but DispatchKeySet is not a type we want the jit to be aware of.
// See Note [Plumbing Keys Through The Dispatcher]
template<class KernelFunctor>
std::unique_ptr<FunctionSchema> inferFunctionSchemaFromFunctor() {
  using func_type = typename c10::remove_DispatchKeySet_arg_from_func<KernelFunctor>::func_type;
  return std::make_unique<FunctionSchema>(inferFunctionSchemaFlattenedReturns<func_type>());
}
}

/**
 * An instance of this class handles the registration for one or more operators.
 * Make sure you keep the RegisterOperators instance around since it will
 * deregister the operator it's responsible for in its destructor.
 *
 * Example:
 *
 * > namespace {
 * >   class my_kernel_cpu final : public c10::OperatorKernel {
 * >   public:
 * >     Tensor operator()(Tensor a, Tensor b) {...}
 * >   };
 * > }
 * >
 * > static auto registry = c10::RegisterOperators()
 * >     .op(c10::RegisterOperators::options()
 * >         .schema("my_op")
 * >         .kernel<my_kernel_cpu>(DispatchKey::CPU));
 */
class TORCH_API RegisterOperators final {
public:
  RegisterOperators() = default;
  ~RegisterOperators() = default;

  RegisterOperators(const RegisterOperators&) = delete; // Disable copy constructor
  RegisterOperators& operator=(const RegisterOperators&) = delete; // Disable copy assignment operator
  RegisterOperators(RegisterOperators&&) noexcept = default; // Enable move constructor
  RegisterOperators& operator=(RegisterOperators&&) noexcept = default; // Enable move assignment operator

  /**
   * Options class for configuring operator registration.
   */
  class TORCH_API Options final {
  public:
    Options(const Options&) = delete; // Disable copy constructor
    Options(Options&&) noexcept = delete; // Disable move constructor
    Options& operator=(const Options&) = delete; // Disable copy assignment operator
    Options& operator=(Options&&) noexcept = delete; // Disable move assignment operator

    /**
     * Internal method for registering stack-based kernels.
     */
    template<KernelFunction::BoxedKernelFunction* kernel_func>
    Options&& kernel(DispatchKey dispatch_key) && {
      return std::move(*this).kernel(dispatch_key, KernelFunction::makeFromBoxedFunction<kernel_func>(), nullopt, nullptr);
    }

    /**
     * Internal method for registering stack-based catch-all kernels.
     */
    template<KernelFunction::BoxedKernelFunction* kernel_func>
    // 当对象为右值引用时调用，用于捕获所有的 kernel
    Options&& catchAllKernel() && {
      // 返回移动语义后的 kernel 调用结果
      return std::move(*this).kernel(c10::nullopt, KernelFunction::makeFromBoxedFunction<kernel_func>(), nullopt, nullptr);
    }

    // 仅供内部使用，用于注册 Caffe2 操作符
    Options&& schema(FunctionSchema&& schema) {
        // 检查是否已经指定过 schema，确保只能注册一次
        TORCH_CHECK(!schemaOrName_.has_value(), "You can only specify the schema once per operator registration.");
        // 将传入的 schema 移动赋值给 schemaOrName_
        schemaOrName_ = FunctionSchema(std::move(schema));
        // 返回移动语义后的当前对象
        return std::move(*this);
    }

    /**
     * 使用此方法为操作符指定 schema。也可以只指定操作符名称，
     * 此时函数签名部分会从 kernel 函数中推断得到。
     *
     * 示例:
     *
     * > // 从 my_kernel_cpu 推断函数签名
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .kernel<my_kernel_cpu>(DispatchKey::CPU));
     * >
     * >
     * > // 明确指定完整的 schema
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op(Tensor a) -> Tensor")
     * >         .kernel<my_kernel_cpu>(DispatchKey::CPU));
     */
    Options&& schema(const std::string& schemaOrName) {
      // 检查是否已经指定过 schema，确保只能注册一次
      TORCH_CHECK(!schemaOrName_.has_value(), "Tried to register operator ", schemaOrName," but specified schema multiple times. You can only specify the schema once per operator registration.");

      // 在移动构造时，根据条件解析 schemaOrName 字符串，生成对应的 FunctionSchema
      #if !defined(EXPOSE_C2_OPS) && defined(CAFFE2_IS_XPLAT_BUILD)
        throw std::logic_error("Tried to register operator " + schemaOrName + ". We don't support registering c10 ops on mobile yet because the function schema parser isn't present in the mobile build.");
      #else
        schemaOrName_ = torch::jit::parseSchemaOrName(schemaOrName);
      #endif

      // 返回移动语义后的当前对象
      return std::move(*this);
    }
    /**
     * Register a kernel implemented as a functor for a specific dispatch key.
     * Only callable when KernelFunctor is a functor type.
     * Throws compile-time errors for invalid kernel functor or constructor arguments.
     *
     * Example usage:
     *
     * > namespace {
     * >   class my_kernel_cpu final : public c10::OperatorKernel {
     * >   public:
     * >     Tensor operator()(Tensor a, Tensor b) {...}
     * >   };
     * > }
     * >
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .kernel<my_kernel_cpu>(DispatchKey::CPU));
     *
     * The functor's constructor may accept parameters configured during registration.
     * These parameters are defined during the kernel registration.
     * Example:
     *
     * > namespace {
     * >   class my_kernel_cpu final : public c10::OperatorKernel {
     * >   public:
     * >     explicit my_kernel_cpu(std::string some_configuration, int a, bool b)
     * >         : ... {...}
     * >
     * >     Tensor operator()(Tensor a, Tensor b) {...}
     * >   };
     * > }
     * >
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .kernel<my_kernel_cpu>(DispatchKey::CPU, "some_configuration", 3, true));
     */
    template<class KernelFunctor, class... ConstructorParameters>
    // enable_if: enable only if KernelFunctor is a functor
    std::enable_if_t<guts::is_functor<KernelFunctor>::value, Options&&> kernel(DispatchKey dispatch_key, ConstructorParameters&&... constructorParameters) && {
      // Ensure KernelFunctor inherits from OperatorKernel
      static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");
      // Ensure ConstructorParameters match a valid constructor of KernelFunctor
      static_assert(std::is_constructible<KernelFunctor, ConstructorParameters...>::value, "Wrong argument list for constructor of kernel functor. The arguments to kernel<Functor>(arguments...) must match one of the constructors of Functor.");
    
      // Register the kernel with the specified dispatch key and constructor parameters
      return std::move(*this).kernel(
        dispatch_key,
        // Create a boxed functor from KernelFunctor with provided constructor parameters
        KernelFunction::makeFromUnboxedFunctor<false, KernelFunctor>(std::make_unique<KernelFunctor>(std::forward<ConstructorParameters>(constructorParameters)...)),
        // Generate C++ function signature for KernelFunctor
        impl::CppSignature::make<KernelFunctor>(),
        // Infer function schema from KernelFunctor
        detail::inferFunctionSchemaFromFunctor<KernelFunctor>()
      );
    }
    /**
     * Register a functor-based operator kernel without input dispatch.
     * This kernel is invoked independent of input types.
     * 
     * Example usage:
     * > namespace {
     * >   class my_kernel_cpu final : public c10::OperatorKernel {
     * >   public:
     * >     Tensor operator()(Tensor a, Tensor b) {...}
     * >   };
     * > }
     * >
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .catchAllKernel<my_kernel_cpu>());
     *
     * The functor can be configured with constructor parameters during registration.
     * These parameters are specified in the kernel registration.
     * 
     * Example:
     * > namespace {
     * >   class my_kernel_cpu final : public c10::OperatorKernel {
     * >   public:
     * >     explicit my_kernel_cpu(std::string some_configuration, int a, bool b)
     * >         : ... {...}
     * >
     * >     Tensor operator()(Tensor a, Tensor b) {...}
     * >   };
     * > }
     * >
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .catchAllKernel<my_kernel_cpu>("some_configuration", 3, true));
     */
    template<class KernelFunctor, class... ConstructorParameters>
    // enable_if: Enable only if KernelFunctor is indeed a functor
    std::enable_if_t<guts::is_functor<KernelFunctor>::value, Options&&> catchAllKernel(ConstructorParameters&&... constructorParameters) && {
      // Ensure KernelFunctor inherits from OperatorKernel
      static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");
      // Ensure ConstructorParameters match the constructor of KernelFunctor
      static_assert(std::is_constructible<KernelFunctor, ConstructorParameters...>::value, "Wrong argument list for constructor of kernel functor. The arguments to kernel<Functor>(arguments...) must match one of the constructors of Functor.");
    
      // Register the kernel functor with associated parameters
      return std::move(*this).kernel(
        c10::nullopt,
        KernelFunction::makeFromUnboxedFunctor<false, KernelFunctor>(std::make_unique<KernelFunctor>(std::forward<ConstructorParameters>(constructorParameters)...)),
        impl::CppSignature::make<KernelFunctor>(),
        detail::inferFunctionSchemaFromFunctor<KernelFunctor>()
      );
    }
    /**
     * 用于注册一个由函数实现的操作符的核函数。
     * 核函数只会对与给定调度键匹配的输入调用。
     * 可以为不同的调度键注册多个核函数。
     *
     * 示例:
     *
     * > namespace { Tensor my_kernel_cpu(Tensor a, Tensor b) {...} }
     * >
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .kernel<decltype(my_kernel_cpu), &my_kernel_cpu>(DispatchKey::CPU));
     */
    template<class FuncType, FuncType* kernel_func>
    // 如果 FuncType 实际上是一个函数类型，则启用此模板
    std::enable_if_t<guts::is_function_type<FuncType>::value, Options&&> kernel(DispatchKey dispatch_key) && {
      static_assert(!std::is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, "试图使用公共 kernel<...>() API 注册基于堆栈的（即内部的）核函数。请使用内部的 kernel(...) API 或按照公共 API 定义的方式实现核函数。");
      static_assert(kernel_func != nullptr, "核函数不能为空指针");
    
      return std::move(*this).kernel(
        dispatch_key,
        KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernel_func)),
        impl::CppSignature::make<FuncType>(),
        // TODO 在不依赖于 WrapFunctionIntoFunctor 的情况下进行模式推断
        detail::inferFunctionSchemaFromFunctor<typename impl::WrapFunctionIntoFunctor<CompileTimeFunctionPointer<FuncType, kernel_func>>::type>()
      );
    }
    
    
    
    /**
     * 用于注册一个由函数实现的操作符的核函数。
     * 这是一个全捕获核函数，意味着它独立于输入被调用。对于这个操作符，禁用了调度。
     *
     * 示例:
     *
     * > namespace { Tensor my_kernel_cpu(Tensor a, Tensor b) {...} }
     * >
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .catchAllKernel<decltype(my_kernel_cpu), &my_kernel_cpu>());
     */
    template<class FuncType, FuncType* kernel_func>
    // 如果 FuncType 实际上是一个函数类型，则启用此模板
    // 使用 SFINAE（Substitution Failure Is Not An Error）技术，仅在 FuncType 为函数类型时启用该函数模板，并且是右值引用
    std::enable_if_t<guts::is_function_type<FuncType>::value, Options&&> catchAllKernel() && {
      // 断言：确保 FuncType 不是 KernelFunction::BoxedKernelFunction 类型，否则静态断言失败
      static_assert(!std::is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, "Tried to register a stackbased (i.e. internal) kernel function using the public kernel<...>() API. Please either use the internal kernel(...) API or also implement the kernel function as defined by the public API.");
      // 断言：确保 kernel_func 不为 nullptr，否则静态断言失败
      static_assert(kernel_func != nullptr, "Kernel function cannot be nullptr");

      // 调用当前对象的 kernel 方法，传入空的 dispatch_key，使用 KernelFunction::makeFromUnboxedFunction 包装 kernel_func，
      // 使用 impl::CppSignature::make<FuncType>() 创建函数签名信息，
      // 使用 detail::inferFunctionSchemaFromFunctor<...>() 推断函数的 schema 信息
      return std::move(*this).kernel(
        c10::nullopt,
        KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernel_func)),
        impl::CppSignature::make<FuncType>(),
        // TODO 在不依赖 WrapFunctionIntoFunctor 的情况下进行 schema 推断
        detail::inferFunctionSchemaFromFunctor<typename impl::WrapFunctionIntoFunctor<CompileTimeFunctionPointer<FuncType, kernel_func>>::type>()
      );
    }

    template<class FuncType>
    // enable_if: 仅在 FuncType 为函数类型时启用该函数模板，并且是右值引用
    std::enable_if_t<guts::is_function_type<FuncType>::value, Options&&> kernel(DispatchKey dispatch_key, FuncType* kernel_func) && {
      // 断言：确保 FuncType 不是 KernelFunction::BoxedKernelFunction 类型，否则静态断言失败
      static_assert(!std::is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, "Tried to register a stackbased (i.e. internal) kernel function using the public kernel<...>() API. Please either use the internal kernel(...) API or also implement the kernel function as defined by the public API.");
      // 断言：确保 kernel_func 不为 nullptr，否则静态断言失败
      TORCH_INTERNAL_ASSERT(kernel_func != nullptr, "Kernel function cannot be nullptr");

      // 调用当前对象的 kernel 方法，传入 dispatch_key，使用 KernelFunction::makeFromUnboxedRuntimeFunction 包装 kernel_func，
      // 使用 impl::CppSignature::make<FuncType>() 创建函数签名信息，
      // 使用 detail::inferFunctionSchemaFromFunctor<...>() 推断函数的 schema 信息
      return std::move(*this).kernel(
        dispatch_key,
        KernelFunction::makeFromUnboxedRuntimeFunction(kernel_func),
        impl::CppSignature::make<FuncType>(),
        // TODO 在不依赖 WrapFunctionIntoFunctor 的情况下进行 schema 推断
        detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>()
      );
    }

    template<class FuncType>
    // enable_if: 仅在 FuncType 为函数类型时启用该函数模板，并且是右值引用
    std::enable_if_t<guts::is_function_type<FuncType>::value, Options&&> catchAllKernel(FuncType* kernel_func) && {
      // 断言：确保 FuncType 不是 KernelFunction::BoxedKernelFunction 类型，否则静态断言失败
      static_assert(!std::is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, "Tried to register a stackbased (i.e. internal) kernel function using the public kernel<...>() API. Please either use the internal kernel(...) API or also implement the kernel function as defined by the public API.");
      // 断言：确保 kernel_func 不为 nullptr，否则静态断言失败
      TORCH_INTERNAL_ASSERT(kernel_func != nullptr, "Kernel function cannot be nullptr");

      // 调用当前对象的 kernel 方法，传入空的 dispatch_key，使用 KernelFunction::makeFromUnboxedRuntimeFunction 包装 kernel_func，
      // 使用 impl::CppSignature::make<FuncType>() 创建函数签名信息，
      // 使用 detail::inferFunctionSchemaFromFunctor<...>() 推断函数的 schema 信息
      return std::move(*this).kernel(
        c10::nullopt,
        KernelFunction::makeFromUnboxedRuntimeFunction(kernel_func),
        impl::CppSignature::make<FuncType>(),
        // TODO 在不依赖 WrapFunctionIntoFunctor 的情况下进行 schema 推断
        detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>()
      );
    }
    /**
     * Use this to register an operator whose kernel is implemented as a lambda.
     * The kernel is only called for inputs matching the given dispatch key.
     * You can register multiple kernels for different dispatch keys.
     *
     * The lambda must be stateless, i.e. not have a capture. If your kernel
     * needs to store some configuration parameters, write the kernel as a
     * functor instead.
     *
     * Example:
     *
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .kernel(DispatchKey::CPU, [] (Tensor a) -> Tensor {...}));
     */
    template<class Lambda>
    // enable_if: only enable it if Lambda is a functor (note: lambdas are functors)
    std::enable_if_t<
        guts::is_functor<std::decay_t<Lambda>>::value
        && !std::is_same<typename guts::infer_function_traits_t<std::decay_t<Lambda>>::func_type, KernelFunction::BoxedKernelFunction>::value,
        Options&&> kernel(DispatchKey dispatch_key, Lambda&& functor) && {
      static_assert(!std::is_base_of<OperatorKernel, std::decay_t<Lambda>>::value, "The kernel(x) API for registering a kernel is only meant to be used with lambdas. Your kernel is a functor. Please use the kernel<Functor>() API instead.");
    
      // We don't support stateful lambdas (i.e. lambdas with a capture), because their
      // behavior would be nonobvious. A functor kernel with cache gets a new instance of
      // its cache each time the kernel is looked up from the dispatch table.
      // A lambda with a capture would be global and share its capture between all kernel lookups.
      // So, instead of making users having to think about it (including the thread-safety
      // issues this causes), let's just forbid stateful lambdas altogether.
      static_assert(guts::is_stateless_lambda<std::decay_t<Lambda>>::value, "The kernel(x) API for registering a kernel only works for stateless lambdas (i.e. lambdas without captures). If you need a cache, please use the functor based API kernel<Functor>() instead.");
    
      // Call the kernel registration method with the lambda functor, dispatch key, and schema inference
      return std::move(*this).kernel(
        dispatch_key,
        KernelFunction::makeFromUnboxedLambda(std::forward<Lambda>(functor)),  // Convert lambda to unboxed kernel function
        impl::CppSignature::make<Lambda>(),                                    // Deduce C++ signature of the lambda
        // TODO Do schema inference without relying on WrapFunctionIntoRuntimeFunctor
        detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>()
      );
    }
    /**
     * Register an operator whose kernel is implemented as a lambda function.
     * This lambda serves as a catch-all kernel that operates independently of input.
     * Dispatch for this operator is disabled.
     *
     * The lambda function must be stateless, i.e., it should not capture any variables.
     * If the kernel requires configuration parameters, use a functor instead of a lambda.
     *
     * Example:
     *
     * > static auto registry = c10::RegisterOperators()
     * >     .op(c10::RegisterOperators::options()
     * >         .schema("my_op")
     * >         .catchAllKernel([] (Tensor a) -> Tensor {...}));
     */
    template<class Lambda>
    // enable_if: enable this function template only if Lambda is a functor and not a specific type of KernelFunction
    std::enable_if_t<
        guts::is_functor<std::decay_t<Lambda>>::value
        && !std::is_same<typename guts::infer_function_traits_t<std::decay_t<Lambda>>::func_type, KernelFunction::BoxedKernelFunction>::value,
        Options&&> catchAllKernel(Lambda&& lambda) && {
      static_assert(!std::is_base_of<OperatorKernel, std::decay_t<Lambda>>::value, "The kernel(x) API for registering a kernel is only meant to be used with lambdas. Your kernel is a functor. Please use the kernel<Functor>() API instead.");

      // Ensure that the lambda used does not capture any variables (is stateless)
      static_assert(guts::is_stateless_lambda<std::decay_t<Lambda>>::value, "The kernel(x) API for registering a kernel only works for stateless lambdas (i.e., lambdas without captures). If you need a cache, please use the functor based API kernel<Functor>() instead.");

      // Register the lambda kernel with the operator registration
      return std::move(*this).kernel(
        c10::nullopt,
        KernelFunction::makeFromUnboxedLambda(std::forward<Lambda>(lambda)),  // Create a kernel function from the lambda
        impl::CppSignature::make<Lambda>(),  // Generate the C++ signature for the lambda
        // TODO Do schema inference without relying on WrapFunctionIntoRuntimeFunctor
        detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>()  // Infer function schema from the lambda
      );
    }

    /**
     * Specify alias analysis kind for the operator registration.
     * This restricts how TorchScript compiler may optimize memory aliasing.
     */
    Options&& aliasAnalysis(AliasAnalysisKind aliasAnalysisKind) && {
      // Ensure aliasAnalysis() is called only once per operator registration
      TORCH_CHECK(!aliasAnalysisKind_.has_value(), "You can only call aliasAnalysis() once per operator registration.");
      aliasAnalysisKind_ = aliasAnalysisKind;  // Set the alias analysis kind for the operator
      return std::move(*this);  // Return the modified Options object
    }

  private:
    // 移动语义右值引用版本的 kernel 函数，用于注册操作的内核函数及相关配置
    Options&& kernel(std::optional<DispatchKey> dispatch_key, KernelFunction&& func, std::optional<impl::CppSignature> cpp_signature, std::unique_ptr<FunctionSchema>&& inferred_function_schema) && {
      // 创建 KernelRegistrationConfig 对象
      KernelRegistrationConfig config;
      // 设置分发键
      config.dispatch_key = dispatch_key;
      // 移动内核函数对象
      config.func = std::move(func);
      // 设置 C++ 签名（如果提供的话）
      config.cpp_signature = cpp_signature;
      // 移动推断的函数模式对象
      config.inferred_function_schema = std::move(inferred_function_schema);
      // 将配置对象添加到 kernels 向量中
      kernels.push_back(std::move(config));
      // 返回移动后的当前对象实例
      return std::move(*this);
    }

    // Options 类的默认构造函数，初始化成员变量
    Options()
    : schemaOrName_(c10::nullopt)  // 初始化 schemaOrName_ 为 nullptr
    , kernels()                     // 初始化 kernels 为空向量
    , aliasAnalysisKind_(c10::nullopt)  // 初始化 aliasAnalysisKind_ 为 nullptr
    {}

    // 内核注册配置类，累积 RegisterOperators::op() 调用中传递的所有参数到一个对象中
    struct KernelRegistrationConfig final {
      KernelRegistrationConfig()
        : dispatch_key(c10::nullopt)  // 初始化 dispatch_key 为 nullptr
        , func()                       // 初始化 func
        , cpp_signature(c10::nullopt)  // 初始化 cpp_signature 为 nullptr
        , inferred_function_schema(nullptr)  // 初始化 inferred_function_schema 为 nullptr
      {}

      std::optional<DispatchKey> dispatch_key;  // 可选的分发键
      KernelFunction func;                     // 内核函数对象
      std::optional<impl::CppSignature> cpp_signature;  // 可选的 C++ 签名
      std::unique_ptr<FunctionSchema> inferred_function_schema;  // 唯一指针，推断的函数模式
    };

    std::optional<std::variant<OperatorName, FunctionSchema>> schemaOrName_;  // 可选的枚举类型，操作符名称或函数模式

    std::vector<KernelRegistrationConfig> kernels;  // 内核注册配置对象的向量
    optional<AliasAnalysisKind> aliasAnalysisKind_;  // 可选的别名分析类型
    friend class RegisterOperators;  // 声明 RegisterOperators 类为友元类
    friend class Library;  // 声明 Library 类为友元类
  };

  /**
   * 获取注册选项的实例，可传递给 RegisterOperators::op() 调用以指定操作符注册的选项。
   * 详见类文档注释中的示例。
   */
  static Options options() {
    return {};
  }

  /**
   * 注册操作符的函数。详见类文档注释中的示例。
   */
  RegisterOperators&& op(Options&& options) && {
    // 检查操作模式并注册操作符
    checkSchemaAndRegisterOp_(std::move(options));
    // 返回移动后的当前对象实例
    return std::move(*this);
  }

  // 上述 && 版本的常规 mutator 版本
  RegisterOperators& op(Options&& options) & {
    // 检查操作模式并注册操作符
    checkSchemaAndRegisterOp_(std::move(options));
    // 返回当前对象实例的引用
    return *this;
  }

  /**
   * 这是 RegisterOperators::op(Options) 的简写形式，允许在 options 参数之外指定操作符模式或名称。
   * 详见类文档注释中的示例。
   */
  RegisterOperators&& op(const std::string& schemaOrName, Options&& options = RegisterOperators::options()) && {
    // 移动当前对象并调用 op 方法，传递指定的 schema
    return std::move(*this).op(std::move(options).schema(schemaOrName));
  }

  // 仅用于注册 caffe2 操作符的内部函数
  RegisterOperators&& op(FunctionSchema schema, Options&& options) && {
    // 移动当前对象并调用 op 方法，传递指定的 schema
    return std::move(*this).op(std::move(options).schema(std::move(schema)));
  }

  template<class FuncType>
  explicit RegisterOperators(const std::string& schemaOrName, FuncType&& func, Options&& options = RegisterOperators::options())
  : RegisterOperators() {
  std::move(*this).op(schemaOrName, std::forward<FuncType>(func), std::move(options));
}

/**
 * This API registers an operator based on a kernel function pointer.
 *
 * Given a kernel
 *
 * > namespace { Tensor my_kernel_cpu(Tensor a, Tensor b) {...} }
 *
 * This API looks like:
 *
 * > static auto registry = c10::RegisterOperators()
 * >     .op("my_op", &my_kernel_cpu);
 *
 * If your kernel is small and the overhead of calling it matters,
 * then this API might be the wrong choice since the following API
 * has a slightly lower overhead for calling into the kernel:
 *
 * > static auto registry = c10::RegisterOperators()
 * >     .op("my_op", c10::RegisterOperators::options()
 * >         .kernel<decltype(my_kernel_cpu), &my_kernel_cpu>());
 *
 * Or, alternatively, write your kernel as a functor:
 *
 * > namespace {
 * >   class my_kernel_cpu final : public c10::OperatorKernel {
 * >   public:
 * >     Tensor operator()(Tensor a, Tensor b) {...}
 * >   };
 * > }
 * >
 * > static auto registry = c10::RegisterOperators()
 * >     .op("my_op", c10::RegisterOperators::options()
 * >         .kernel<my_kernel_cpu>());
 */
template<class FuncType>
// enable_if: only enable it if FuncType is actually a function, but not a stack based BoxedKernelFunction.
std::enable_if_t<guts::is_function_type<FuncType>::value && !std::is_same<FuncType, KernelFunction::BoxedKernelFunction>::value, RegisterOperators&&>
op(const std::string& schemaOrName, FuncType* func, Options&& options = RegisterOperators::options()) && {
  constexpr bool AllowLegacyTypes = true;
  return std::move(*this).op(std::move(options).schema(schemaOrName).kernel(
    c10::nullopt,
    KernelFunction::makeFromUnboxedRuntimeFunction<AllowLegacyTypes>(func),
    impl::CppSignature::make<FuncType>(),
    // TODO Do schema inference without relying on WrapFunctionIntoRuntimeFunctor
    detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>>>()
  ));
}

/**
 * This API registers an operator based on a kernel lambda.
 *
 * This API looks like:
 *
 * > static auto registry = c10::RegisterOperators()
 * >     .op("my_op", [] (Tensor a, Tensor b) {...});
 *
 * This is equivalent to:
 *
 * > static auto registry = c10::RegisterOperators()
 * >     .op("my_op", c10::RegisterOperators::options()
 * >         .catchAllKernel([] (Tensor a, Tensor b) {...}));
 *
 */
template<class Lambda>
// enable_if: only enable it if Lambda is actually a stateless lambda
std::enable_if_t<guts::is_functor<Lambda>::value && guts::is_stateless_lambda<std::decay_t<Lambda>>::value, RegisterOperators&&>
    op(const std::string& schemaOrName, Lambda&& lambda, Options&& options = RegisterOperators::options()) && {
      // 使用右值引用绑定的方法，注册一个操作符，并且该方法只能在右值对象上调用
      static_assert(!std::is_base_of<OperatorKernel, Lambda>::value, "c10::OperatorKernel is part of the new kernel registration API and shouldn't be used together with the deprecated registration API. Please use the new RegisterOperators::options().kernel() based API instead.");
      
      // 确定允许使用旧类型的标志位为真
      constexpr bool AllowLegacyTypes = true;
      // 返回当前对象的右值引用，调用op方法，将lambda函数注册到指定的schemaOrName
      return std::move(*this).op(std::move(options).schema(schemaOrName).kernel(
        // 不使用操作符内核
        c10::nullopt,
        // 从无盒lambda函数生成内核函数
        KernelFunction::makeFromUnboxedLambda<AllowLegacyTypes>(std::forward<Lambda>(lambda)),
        // 生成lambda函数的C++签名
        impl::CppSignature::make<Lambda>(),
        // TODO 不依赖WrapFunctionIntoRuntimeFunctor进行模式推断
        detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>()
      ));
    }

    template<class Lambda>
    // 使用模板类Lambda，显示消息已弃用
    C10_DEPRECATED_MESSAGE("Registering operator kernels with stateful lambdas (i.e. lambdas with a capture) has non-obvious behavior. This is deprecated. Please use a lambda without a capture or a functor class instead.")
    // 如果Lambda是一个函数对象但不是无状态lambda，则启用它
    std::enable_if_t<guts::is_functor<Lambda>::value && !guts::is_stateless_lambda<std::decay_t<Lambda>>::value, RegisterOperators&&>
    // 使用右值引用绑定的方法，注册一个操作符，并且该方法只能在右值对象上调用
    op(const std::string& schemaOrName, Lambda&& lambda, Options&& options = RegisterOperators::options()) && {
      // 确保不使用操作符内核
      static_assert(!std::is_base_of<OperatorKernel, Lambda>::value, "c10::OperatorKernel is part of the new kernel registration API and shouldn't be used together with the deprecated registration API. Please use the new RegisterOperators::options().kernel() based API instead.");
      
      // 确定允许使用旧类型的标志位为真
      constexpr bool AllowLegacyTypes = true;
      // 返回当前对象的右值引用，调用op方法，将lambda函数注册到指定的schemaOrName
      return std::move(*this).op(std::move(options).schema(schemaOrName).kernel(
        // 不使用操作符内核
        c10::nullopt,
        // 从无盒lambda函数生成内核函数
        KernelFunction::makeFromUnboxedLambda<AllowLegacyTypes>(std::forward<Lambda>(lambda)),
        // 生成lambda函数的C++签名
        impl::CppSignature::make<Lambda>(),
        // TODO 不依赖WrapFunctionIntoRuntimeFunctor进行模式推断
        detail::inferFunctionSchemaFromFunctor<impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>()
      ));
    }
private:
  // 声明一个私有成员函数，用于检查模式并注册操作，采用移动语义传递配置参数
  void checkSchemaAndRegisterOp_(Options&& config);

  // 声明一个静态成员函数，根据操作名称和选项推断函数模式的函数架构
  static c10::FunctionSchema inferSchemaFromKernels_(const OperatorName& opNameStr, const Options& options);

  // 声明一个私有成员函数，用于检查选项中没有重复的内核
  void checkNoDuplicateKernels_(const Options& options);

  // 声明一个私有成员函数，用于注册操作，采用移动语义传递选项
  void registerOp_(Options&& options);

  // 声明一个私有成员变量，存储RegistrationHandleRAII对象的向量
  std::vector<RegistrationHandleRAII> registrars_;
};

} // namespace c10

namespace torch {
  // 使用c10命名空间中的RegisterOperators作为torch命名空间的别名，提供向后兼容性
  // 旧式API
  using RegisterOperators = c10::RegisterOperators;
}
```