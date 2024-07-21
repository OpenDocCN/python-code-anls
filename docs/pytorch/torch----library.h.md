# `.\pytorch\torch\library.h`

```py
#pragma once
/// \file
///
/// This header provides an API for extending PyTorch's core library
/// of operators with user defined operators and data types.  This
/// API can be used in a few ways:
///
/// * You can define new custom operators and classes with TORCH_LIBRARY(),
///   making them available for use in both eager Python as well as in
///   TorchScript. This API is modeled off of pybind11's `PYBIND11_MODULE`
///   macro, as the provided functionality is similar (pybind11 lets you bind
///   C++ to Python only; `torch/library.h` lets you bind C++ simultaneously to
///   Python and TorchScript).
///
/// * You can override existing operators with TORCH_LIBRARY_IMPL(),
///   providing a new implementation for these operators for a custom
///   backend (e.g., XLA).  When you pass operators with tensors of your custom
///   backend, your overridden implementations will be called instead
///   of the standard implementations.
///
/// * You can use both capabilities at the same time, allowing you
///   to write custom operators that register CPU/CUDA/Autograd
///   implementations without having to write the boilerplate
///   conditionals yourself.
///
/// For a tutorial style introduction to the library API, check
/// out the [Extending TorchScript with Custom C++
/// Operators](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)
/// tutorial.
///
/// ```
/// // Define a library whose operators live in the namespace 'myops'.
/// // You must define all of the operators for this library in
/// // this namespace.
/// TORCH_LIBRARY(myops, m) {
///   // Define a operator with exactly one implementation for all backends.
///   m.def("add(Tensor self, Tensor other) -> Tensor", &add_impl);
///
///   // Define a schema for an operator, but provide no implementation
///   // (use this syntax if you want to use the dispatcher)
///   m.def("mul(Tensor self, Tensor other) -> Tensor");
///
///   // Provide an implementation for a defined operator (you can
///   // provide multiple; one per backend).  The dispatcher takes care of
///   // calling the correct implementation depending on if we get a CPU
///   // tensor or a CUDA tensor
///   m.impl("mul", torch::kCPU, &mul_cpu_impl);
///   m.impl("mul", torch::kCUDA, &mul_cuda_impl);
/// }
///
/// // Define implementations for operators for a non-standard backend,
/// // e.g., XLA (valid values are entries of DispatchKey).  This can
/// // be used to define operators in a different file than the initial
/// // TORCH_LIBRARY definition (e.g., if it is in an external library)
/// TORCH_LIBRARY_IMPL(myops, XLA, m) {
///   m.impl("mul", &mul_xla_impl);
/// }
/// ```py

#include <ATen/core/op_registration/infer_schema.h>
/// Include for inferFunctionSchemaFromFunctor, which infers a function schema from a functor.
#include <ATen/core/op_registration/op_allowlist.h>
/// Include for the operation allowlist, which maintains the set of registered operators.
#include <ATen/core/dispatch/Dispatcher.h>
/// Include for the dispatcher, which dispatches operations to the appropriate implementation.
#include <c10/core/DispatchKey.h>
/// Include for the DispatchKey, which represents a dispatch key for dispatching operations.
#include <torch/csrc/jit/frontend/function_schema_parser.h>
/// Include for the function schema parser, which parses function schemas.

// Just for inferFunctionSchemaFromFunctor
#include <ATen/core/enum_tag.h>
/// Include for enum_tag, which provides tags for enum types.
``
// CppFunction 类的定义，用于封装不同类型函数的注册
class TORCH_API CppFunction final {

  // TODO: This is morally the same thing as KernelRegistrationConfig, but it's
  // opaque to the user.

 public:
  /// 此重载接受函数指针，例如 `CppFunction(&add_impl)`
  template <typename Func>
  explicit CppFunction(
      Func* f,
      std::enable_if_t<
          c10::guts::is_function_type<Func>::value,
          std::nullptr_t> = nullptr)
      : func_(c10::KernelFunction::makeFromUnboxedRuntimeFunction(f)), // 使用运行时函数指针创建 KernelFunction 对象
        cpp_signature_(c10::impl::CppSignature::make<Func>()), // 根据函数类型 Func 创建 C++ 签名
        schema_(
            c10::detail::inferFunctionSchemaFromFunctor<std::decay_t<Func>>()), // 推断函数的 schema
        debug_() {}

  /// 此重载接受编译时函数指针，例如 `CppFunction(TORCH_FN(add_impl))`
  template <typename FuncPtr>
  explicit CppFunction(
      FuncPtr f,
      std::enable_if_t<
          c10::is_compile_time_function_pointer<FuncPtr>::value,
          std::nullptr_t> = nullptr)
      : func_(c10::KernelFunction::makeFromUnboxedFunction(f)), // 使用编译时函数指针创建 KernelFunction 对象
        cpp_signature_(
            c10::impl::CppSignature::make<typename FuncPtr::FuncType>()), // 根据函数指针的 FuncType 创建 C++ 签名
        schema_(c10::detail::inferFunctionSchemaFromFunctor<
                typename FuncPtr::FuncType>()), // 推断函数的 schema
        debug_() {}

  /// 此重载接受 lambda 表达式，例如 `CppFunction([](const Tensor& self) { ... })`
  template <typename Lambda>
  explicit CppFunction(
      Lambda&& f,
      std::enable_if_t<
          c10::guts::is_functor<std::decay_t<Lambda>>::value,
          std::nullptr_t> = nullptr)
      : func_(c10::KernelFunction::makeFromUnboxedLambda(
            std::forward<Lambda>(f))), // 使用 lambda 表达式创建 KernelFunction 对象
        cpp_signature_(c10::impl::CppSignature::make<Lambda>()), // 根据 lambda 表达式的类型创建 C++ 签名
        schema_(c10::detail::inferFunctionSchemaFromFunctor<
                std::decay_t<Lambda>>()), // 推断函数的 schema
        debug_() {}
#if defined C10_MOBILE
/// 这个重载接受函数指针，例如 `CppFunction(&add_impl, NoInferSchemaTag())`
template <typename Func>
explicit CppFunction(
    Func* f,
    NoInferSchemaTag,
    std::enable_if_t<
        c10::guts::is_function_type<Func>::value,
        std::nullptr_t> = nullptr)
    : func_(c10::KernelFunction::makeFromUnboxedRuntimeFunction(f)),
      cpp_signature_(c10::impl::CppSignature::make<Func>())
      // TODO: 不要通过 WrapRuntimeKernelFunctor 进行处理
      ,
      schema_(nullptr),
      debug_() {}

/// 这个重载接受编译时函数指针，例如 `CppFunction(TORCH_FN(add_impl), NoInferSchemaTag())`
template <typename FuncPtr>
explicit CppFunction(
    FuncPtr f,
    NoInferSchemaTag,
    std::enable_if_t<
        c10::is_compile_time_function_pointer<FuncPtr>::value,
        std::nullptr_t> = nullptr)
    : func_(c10::KernelFunction::makeFromUnboxedFunction(f)),
      cpp_signature_(
          c10::impl::CppSignature::make<typename FuncPtr::FuncType>())
      // TODO: 不要通过 WrapRuntimeKernelFunctor 进行处理
      ,
      schema_(nullptr),
      debug_() {}

/// 这个重载接受 lambda 表达式，例如 `CppFunction([](const Tensor& self) { ... }, NoInferSchemaTag())`
template <typename Lambda>
explicit CppFunction(
    Lambda&& f,
    NoInferSchemaTag,
    std::enable_if_t<
        c10::guts::is_functor<std::decay_t<Lambda>>::value,
        std::nullptr_t> = nullptr)
    : func_(c10::KernelFunction::makeFromUnboxedLambda(
          std::forward<Lambda>(f))),
      cpp_signature_(c10::impl::CppSignature::make<Lambda>())
      // TODO: 不要通过 WrapRuntimeKernelFunctor 进行处理
      ,
      schema_(nullptr),
      debug_() {}
#endif

/// 析构函数
~CppFunction();

/// 移动构造函数
CppFunction(CppFunction&&) noexcept = default;

/// 移动赋值运算符
CppFunction& operator=(CppFunction&&) = default;

/// \private
/// 从类型擦除的 boxed kernel 创建函数
static CppFunction makeFromBoxedKernel(c10::BoxedKernel kernel) {
  return CppFunction(
      c10::KernelFunction::makeFromBoxedKernel(std::move(kernel)),
      /* cpp_signature */ c10::nullopt, // 对于 boxed 函数未知
      /* schema */ nullptr);
}

/// 创建一个 fallthrough 函数。Fallthrough 函数会立即重新分派到下一个可用的 dispatch key，
/// 但比以相同方式手动编写的函数更高效实现。
static CppFunction makeFallthrough() {
  return makeFromBoxedKernel(c10::BoxedKernel::makeFallthrough());
}

/// \private
///
/// 创建一个函数，当调用时抛出错误，指示不支持命名张量。
static CppFunction makeNamedNotSupported() {
  // 使用 c10::BoxedKernel 的静态方法创建不支持命名的内核函数，并将其封装成 CppFunction 返回
  return makeFromBoxedKernel(c10::BoxedKernel::makeNamedNotSupported());
}

/// 根据具有签名 `void(const OperatorHandle&, Stack*)` 的箱式内核函数创建函数；
/// 这些函数在箱式调用约定中接收参数堆栈，而不是本地的 C++ 调用约定。
/// 箱式函数通常仅用于通过 torch::Library::fallback() 注册后端回退。
template <c10::BoxedKernel::BoxedKernelFunction* func>
static CppFunction makeFromBoxedFunction() {
  // 使用 c10::BoxedKernel 的静态方法创建具有指定模板参数 func 的函数，并将其封装成 CppFunction 返回
  return makeFromBoxedKernel(c10::BoxedKernel::makeFromFunction<func>());
}

// 接受具有 plumbed DispatchKeySet 的箱式内核函数的变体。详情参见“注意 [Plumbing Keys Through The Dispatcher]”。
template <c10::BoxedKernel::BoxedKernelFunction_withDispatchKeys* func>
static CppFunction makeFromBoxedFunction() {
  // 使用 c10::BoxedKernel 的静态方法创建具有指定模板参数 func 的函数，并将其封装成 CppFunction 返回
  return makeFromBoxedKernel(c10::BoxedKernel::makeFromFunction<func>());
}

/// 根据定义了 `operator()(const OperatorHandle&, DispatchKeySet, Stack*)` 的箱式内核仿函数创建函数；
/// 这些函数从箱式调用约定中接收参数，并继承自 `c10::OperatorKernel`。
/// 与 makeFromBoxedFunction 不同，以这种方式注册的函数还可以携带仿函数管理的额外状态；
/// 这在编写适配器时很有用，例如与某些其他实现（如 Python 可调用对象）动态关联的内核。
template <class KernelFunctor>
static CppFunction makeFromBoxedFunctor(
    std::unique_ptr<KernelFunctor> kernelFunctor) {
  // 使用 c10::BoxedKernel 的静态方法创建从给定 kernelFunctor 构造的函数，并将其封装成 CppFunction 返回
  return makeFromBoxedKernel(
      c10::BoxedKernel::makeFromFunctor(std::move(kernelFunctor)));
}

/// 根据非箱式内核函数创建函数。
/// 这通常用于注册常见操作符。
template <
    typename FuncPtr,
    std::enable_if_t<
        c10::guts::is_function_type<FuncPtr>::value,
        std::nullptr_t> = nullptr>
static CppFunction makeFromUnboxedFunction(FuncPtr* f) {
  // 直接使用给定的函数指针创建 CppFunction 并返回
  return CppFunction(f);
}

/// 根据编译时非箱式内核函数指针创建函数。
/// 这通常用于注册常见操作符。
/// 编译时函数指针可用于允许编译器优化（例如内联）对其的调用。
template <
    typename FuncPtr,
    std::enable_if_t<
        c10::is_compile_time_function_pointer<FuncPtr>::value,
        std::nullptr_t> = nullptr>
static CppFunction makeFromUnboxedFunction(FuncPtr f) {
  // 直接使用给定的函数指针创建 CppFunction 并返回
  return CppFunction(f);
}

// 将调试信息设置为移动语义的参数 d，返回右值引用以支持链式调用
CppFunction&& debug(std::string d) && {
  // 将参数 d 移动到 debug_ 成员变量中，并返回当前对象的右值引用
  debug_ = std::move(d);
    // 返回当前对象的右值引用，即将对象作为右值传递出去
    return std::move(*this);
  }

 private:
  // 可选的分发键值，用于选择合适的内核函数
  std::optional<c10::DispatchKey> dispatch_key_;
  // 内核函数指针
  c10::KernelFunction func_;
  // 可选的 C++ 签名，描述函数的 C++ 接口
  std::optional<c10::impl::CppSignature> cpp_signature_;
  // 函数模式的唯一指针，描述函数的参数和返回值
  std::unique_ptr<c10::FunctionSchema> schema_;
  // 调试信息字符串
  std::string debug_;

  // 用于设置 dispatch_key_ 的 "setter" 函数模板
  template <typename Func>
  friend CppFunction dispatch(c10::DispatchKey, Func&&);

  // 唯一从 CppFunction 中提取值的类（以破坏性方式进行），作者觉得写访问器麻烦而且不希望用户使用
  friend class Library;

  // CppFunction 构造函数，接受内核函数、可选的 C++ 签名和函数模式指针作为参数
  CppFunction(
      c10::KernelFunction func,
      std::optional<c10::impl::CppSignature> cpp_signature,
      std::unique_ptr<c10::FunctionSchema> schema);
/// \defgroup torch-dispatch-overloads torch::dispatch overloads
/// 定义了 torch::dispatch 的重载函数群组

/// Create a torch::CppFunction which is associated with a specific
/// dispatch key.  torch::CppFunctions that are tagged with a
/// c10::DispatchKey don't get invoked unless the dispatcher determines
/// that this particular c10::DispatchKey is the one that should be
/// dispatched to.
///
/// This function is generally not used directly, instead, prefer using
/// TORCH_LIBRARY_IMPL(), which will implicitly set the c10::DispatchKey
/// for all registration calls inside of its body.
///
/// 创建一个与特定 dispatch key 关联的 torch::CppFunction。
/// 被标记为 c10::DispatchKey 的 torch::CppFunction 只有在调度程序确定
/// 应该调度到这个特定 c10::DispatchKey 时才会被调用。
///
/// 通常不直接使用这个函数，而是使用 TORCH_LIBRARY_IMPL()，
/// 它会隐式地为其体内的所有注册调用设置 c10::DispatchKey。
///
/// \ingroup torch-dispatch-overloads
template <typename Func>
inline CppFunction dispatch(c10::DispatchKey k, Func&& raw_f) {
  CppFunction f(std::forward<Func>(raw_f));
  if (k == c10::DispatchKey::CatchAll) {
    f.dispatch_key_ = c10::nullopt;
  } else {
    f.dispatch_key_ = k;
  }
  return f;
}

/// Convenience overload of dispatch() which accepts c10::DeviceType
///
/// \ingroup torch-dispatch-overloads
template <typename Func>
inline CppFunction dispatch(c10::DeviceType type, Func&& raw_f) {
  auto deviceTypeToDispatchKey = [](c10::DeviceType t) {
    switch (t) {
      // This list is synchronized with the k-constants in c10/core/DeviceType.h
      case c10::DeviceType::CPU:
        return c10::DispatchKey::CPU;
      case c10::DeviceType::CUDA:
        return c10::DispatchKey::CUDA;
      case c10::DeviceType::IPU:
        return c10::DispatchKey::IPU;
      case c10::DeviceType::XLA:
        return c10::DispatchKey::XLA;
      case c10::DeviceType::Lazy:
        return c10::DispatchKey::Lazy;
      case c10::DeviceType::XPU:
        return c10::DispatchKey::XPU;
      case c10::DeviceType::MPS:
        return c10::DispatchKey::MPS;
      case c10::DeviceType::Meta:
        return c10::DispatchKey::Meta;
      case c10::DeviceType::HIP:
        return c10::DispatchKey::HIP;
      case c10::DeviceType::MAIA:
        return c10::DispatchKey::MAIA;
      case c10::DeviceType::HPU:
        return c10::DispatchKey::HPU;
      case c10::DeviceType::MTIA:
        return c10::DispatchKey::MTIA;
      case c10::DeviceType::PrivateUse1:
        return c10::DispatchKey::PrivateUse1;
      default:
        TORCH_CHECK(
            false,
            "Device type ",
            t,
            " cannot be overloaded at dispatch time, "
            "please file a bug report explaining what you were trying to do.");
    }
  };
  return dispatch(deviceTypeToDispatchKey(type), std::forward<Func>(raw_f));
}

/// \defgroup torch-schema-overloads torch::schema overloads
/// 定义了 torch::schema 的重载函数群组

/// Construct a c10::FunctionSchema from a string, with an explicitly
/// specified c10::AliasAnalysisKind.  Ordinarily, schemas are simply
/// passed in as strings, but if you need to specify a custom alias
/// analysis, you can replace the string with a call to this function.
///
/// ```
/// // Default alias analysis (FROM_SCHEMA)
/// m.def("def3(Tensor self) -> Tensor");
/// // Pure function alias analysis
///
/// 从字符串构造一个 c10::FunctionSchema，并显式指定 c10::AliasAnalysisKind。
/// 通常，模式被简单地作为字符串传递，但如果需要指定自定义的别名分析，可以
/// 用此函数调用替换字符串。
///
/// ```py
/// // 默认别名分析（FROM_SCHEMA）
/// m.def("def3(Tensor self) -> Tensor");
/// // 纯函数别名分析
/// 基于给定的字符串和别名分析类型，解析函数的模式，并返回其函数模式对象
/// \param str 要解析的函数模式字符串
/// \param k 别名分析类型
/// \param allow_typevars 是否允许类型变量，默认为 false
inline c10::FunctionSchema schema(const char* str, c10::AliasAnalysisKind k, bool allow_typevars=false) {
    // 调用 torch::jit::parseSchema 解析给定的字符串 str，并设置是否允许类型变量
    c10::FunctionSchema s = torch::jit::parseSchema(str, /*allow_typevars*/allow_typevars);
    // 设置函数模式对象的别名分析类型
    s.setAliasAnalysis(k);
    // 返回解析后的函数模式对象
    return s;
}

/// 函数模式可以直接从字符串字面量构造
///
/// \ingroup torch-schema-overloads
inline c10::FunctionSchema schema(const char* s, bool allow_typevars=false) {
    // 调用前一个 schema 函数，设置别名分析类型为 FROM_SCHEMA，并返回函数模式对象
    return schema(s, c10::AliasAnalysisKind::FROM_SCHEMA, allow_typevars);
}

/// \private
///
/// 已构造的函数模式如果是右值引用则被接受
///
/// \ingroup torch-schema-overloads
inline c10::FunctionSchema&& schema(c10::FunctionSchema&& s) {
    // 返回移动后的函数模式对象
    return std::move(s);
}

namespace detail {

/// 从移动的函数模式对象构造函数模式或操作符名称的 variant
///
/// \param s 移动的函数模式对象
/// \return 变体，包含构造的操作符名称或函数模式对象
inline std::variant<c10::OperatorName, c10::FunctionSchema> constructSchemaOrName(
    c10::FunctionSchema&& s) {
    // 直接返回移动后的函数模式对象
    return std::move(s);
}

/// 从移动的操作符名称构造函数模式或操作符名称的 variant
///
/// \param n 移动的操作符名称
/// \return 变体，包含构造的操作符名称或函数模式对象
inline std::variant<c10::OperatorName, c10::FunctionSchema> constructSchemaOrName(
    c10::OperatorName&& n) {
    // 直接返回移动后的操作符名称
    return std::move(n);
}

/// 从字符串字面量解析函数模式或名称，并根据需要设置别名分析类型
///
/// \param str 要解析的函数模式字符串
/// \return 变体，包含解析的操作符名称或函数模式对象
inline std::variant<c10::OperatorName, c10::FunctionSchema>
constructSchemaOrName(const char* str) {
    // 调用 torch::jit::parseSchemaOrName 解析给定的字符串 str
    auto s = torch::jit::parseSchemaOrName(str);
    // 如果解析结果是函数模式对象，则设置别名分析类型为 FROM_SCHEMA
    if (std::holds_alternative<c10::FunctionSchema>(s)) {
        std::get<c10::FunctionSchema>(s).setAliasAnalysis(
            c10::AliasAnalysisKind::FROM_SCHEMA);
    }
    // 返回包含解析结果的 variant
    return s;
}

} // namespace detail

// Note [Selective build]
// ~~~~~~~~~~~~~~~~~~~~~~
// 在某些设置中，特别是移动平台，避免编译任何实际不会使用的函数引用是很重要的，
// 这样可以通过链接器进行消除。我们称之为 "selective build" 能力。
//
// 实现选择性构建的一种简单方法是，在每个注册调用周围添加 ifdef，
// 但这意味着需要在每个注册站点编写大量额外的代码行，并且还需要定义一些映射方案，
// 将运算符映射到宏。
//
// 我们采用了不同的机制，集中在 SelectiveStr 的概念上。选择性名称类似于 const char* 字符串，
// 但它在编译时还携带一个布尔值，表示是否应该实际进行注册。我们进行了 constexpr 测试，
// 查看操作符是否应该启用或禁用；这目前在 ATen/core/op_registration/op_allowlist.h 中实现。

namespace detail {

// 用于未选择的自定义 torchbind 类的虚拟类
class ClassNotSelected {
 public:
  // 定义不做任何事情的 dummy 函数
  ClassNotSelected& def_pickle(...) {
    return *this;
  }
  // 定义不做任何事情的 dummy 函数
  ClassNotSelected& def(...) {
    return *this;
  }
};

// SelectiveStr 类似于 const char*，但在编译时还包含一个布尔值来指示是否应该实际进行注册
// 声明一个模板类 SelectiveStr，根据模板参数 enabled 决定是否启用注册功能
// 当字符串被禁用时，在编译时不会生成注册调用。这个类不直接调用，而是使用下面的
// TORCH_SELECTIVE_NAME 或 TORCH_SELECTIVE_SCHEMA 宏来创建它。
template <bool enabled>
class SelectiveStr {
 public:
  // 构造函数，接受一个 const char* 参数，用于初始化 name_
  constexpr explicit SelectiveStr(const char* name) : name_(name) {}
  // 类型转换运算符，返回 name_ 的 const char* 指针
  constexpr operator const char*() {
    return name_;
  }

 private:
  const char* name_;  // 存储传入的字符串指针
};

// 宏定义，用于创建 SelectiveStr 实例，根据 custom_class_allowlist_check 函数的返回值决定是否启用
#define TORCH_SELECTIVE_CLASS(n) \
  torch::detail::SelectiveStr<c10::impl::custom_class_allowlist_check(n)>(n)
// 宏定义，用于创建 SelectiveStr 实例，根据 op_allowlist_check 函数的返回值决定是否启用
#define TORCH_SELECTIVE_NAME(n) \
  torch::detail::SelectiveStr<c10::impl::op_allowlist_check(n)>(n)
// 宏定义，用于创建 SelectiveStr 实例，根据 schema_allowlist_check 函数的返回值决定是否启用
#define TORCH_SELECTIVE_SCHEMA(n) \
  torch::detail::SelectiveStr<c10::impl::schema_allowlist_check(n)>(n)

} // namespace detail

/// 提供定义运算符和在分派键上提供实现的 API 对象。通常，torch::Library
/// 不会直接分配；而是通过 TORCH_LIBRARY() 或 TORCH_LIBRARY_IMPL() 宏创建。
///
/// 大多数 torch::Library 的方法返回对自身的引用，支持方法链式调用。
///
/// ```
/// // 示例:
///
/// TORCH_LIBRARY(torchvision, m) {
///    // m 是一个 torch::Library 的实例
///    m.def("roi_align", ...);
///    ...
/// }
///
/// TORCH_LIBRARY_IMPL(aten, XLA, m) {
///    // m 是一个 torch::Library 的实例
///    m.impl("add", ...);
///    ...
/// }
/// ```py
///
class TORCH_API Library final {
 public:
  /// \private
  ///
  /// 表示这个 Library 是由哪种类型的宏产生的
  enum Kind {
    DEF, // 来自 TORCH_LIBRARY (无修饰符)
    IMPL,
  /// \private
  ///
  /// Use TORCH_LIBRARY() or TORCH_LIBRARY_IMPL() instead of using these
  /// constructors directly
  Library(
      Kind kind,                               // 构造函数，接受 Library 的种类参数
      std::string ns,                          // 命名空间名称
      std::optional<c10::DispatchKey> k,       // 可选的调度键
      const char* file,                        // 文件名
      uint32_t line);                          // 行号信息

  Library(const Library&) = delete;            // 复制构造函数已删除
  Library& operator=(const Library&) = delete; // 复制赋值运算符已删除
  Library(Library&&) = default;                // 移动构造函数默认生成
  Library& operator=(Library&&) = default;     // 移动赋值运算符默认生成

  // Some notes about the API design here.  We had the following constraints:
  //
  //  - We need to support multiple "types" of arguments for schema and
  //    functions (e.g., unnamed lambda types, regular functions, const char*,
  //    fully instantiated schemas)
  //  - We don't want to write exponentially many overloads
  //  - We don't want to rely on implicit conversion to a common type,
  //    because the C++ compiler will only be willing to do a single
  //    implicit conversion (reducing the set of valid types which you
  //    can invoke with); also error messages are worse when an implicit
  //    conversion is not selected (as the compiler will not explain
  //    why it didn't select an implicit conversion; this is different
  //    from overloads where it will explain each candidate overload and
  //    why it didn't apply)
  //
  // To solve all of these constraints at the same time, we use a trick taken
  // from the pybind11 library: template over the argument in the user visible
  // API, and inside of the templated function explicitly call an overloaded
  // function to resolve the argument to a real type.  You get the good error
  // messages from overloads, but at the same time you only need to write the
  // overload for any given argument type once.

  /// Declare an operator with a schema, but don't provide any implementations
  /// for it.  You're expected to then provide implementations using the
  /// impl() method.  All template arguments are inferred.
  ///
  /// \param raw_schema The schema of the operator to be defined.
  ///     Typically, this is a `const char*` string literal, but any type
  ///     accepted by torch::schema() is accepted here.
  ///
  /// ```
  /// // Example:
  /// TORCH_LIBRARY(myops, m) {
  ///   m.def("add(Tensor self, Tensor other) -> Tensor");
  /// }
  /// ```py

  template <typename Schema>
  Library& def(
      Schema&& raw_schema,                     // 定义一个带有特定模式的运算符，但不提供实现
      const std::vector<at::Tag>& tags = {},   // 标签向量，默认为空
      _RegisterOrVerify rv = _RegisterOrVerify::REGISTER) & {
    c10::FunctionSchema s = schema(std::forward<Schema>(raw_schema));  // 生成函数模式 schema
  // 将 s 通过移动语义传递给 _def 函数，并使用 nullptr 和 tags 作为其它参数，返回结果 rv
  return _def(std::move(s), nullptr, tags, rv);
}

/// 声明对所有后续定义的运算符，它们的虚拟实现可能位于给定的 Python 模块 (pymodule) 中。
/// 如果找不到虚拟实现，注册一些帮助文本用于错误消息。
///
/// Args:
/// - pymodule: Python 模块的名称
/// - context: 可选参数，可以包含在错误消息中
Library& set_python_module(const char* pymodule, const char* context = "") {
  // 设置 python_module_ 成员变量为包含 pymodule 和 context 的元组
  python_module_ = {pymodule, context};
  return *this;
}

/// 已弃用；使用 set_python_module 替代
Library& impl_abstract_pystub(const char* pymodule, const char* context = "") {
  // 调用 set_python_module 函数进行实现
  return set_python_module(pymodule, context);
}

/// 定义一个 schema 的操作符，并注册其实现。如果不使用调度程序来结构化操作符实现，
/// 这通常是你会使用的方式。大致相当于调用 def() 和 impl()，
/// 如果省略操作符的 schema，则从 C++ 函数的类型推断出来。所有模板参数都会被推断。
///
/// \param raw_name_or_schema 操作符的 schema，或者如果要从 raw_f 推断 schema，则只是操作符的名称。
///   通常是一个 `const char*` 字面量。
/// \param raw_f 实现此操作符的 C++ 函数。可以接受 torch::CppFunction 的任何有效构造函数；
///   通常提供函数指针或 lambda。
///
/// ```
/// // 示例:
/// TORCH_LIBRARY(myops, m) {
///   m.def("add", add_fn);
/// }
/// ```py
template <typename NameOrSchema, typename Func>
Library& def(NameOrSchema&& raw_name_or_schema, Func&& raw_f,
    const std::vector<at::Tag>& tags = {}) & {
  // 使用 raw_name_or_schema 和 raw_f 创建一个 CppFunction 对象 f
  CppFunction f(std::forward<Func>(raw_f));
    return _def(
        detail::constructSchemaOrName(
            ::std::forward<NameOrSchema>(raw_name_or_schema)),
        ::std::move(f), tags);
  }



  /// Register an implementation for an operator.  You may register multiple
  /// implementations for a single operator at different dispatch keys
  /// (see torch::dispatch()).  Implementations must have a corresponding
  /// declaration (from def()), otherwise they are invalid.  If you plan
  /// to register multiple implementations, DO NOT provide a function
  /// implementation when you def() the operator.
  ///
  /// \param name The name of the operator to implement.  Do NOT provide
  ///   schema here.
  /// \param raw_f The C++ function that implements this operator.  Any
  ///   valid constructor of torch::CppFunction is accepted here;
  ///   typically you provide a function pointer or lambda.
  ///
  /// ```
  /// // Example:
  /// TORCH_LIBRARY_IMPL(myops, CUDA, m) {
  ///   m.impl("add", add_cuda);
  /// }
  /// ```py
  template <typename Name, typename Func>
  Library& impl(
      Name name,
      Func&& raw_f,
      _RegisterOrVerify rv = _RegisterOrVerify::REGISTER) & {
    // TODO: need to raise an error when you impl a function that has a
    // catch all def


注释：

    // 调用 _def 函数来注册或验证操作符的实现
    return _def(
        // 根据传入的原始名称或模式构造出名称或模式对象
        detail::constructSchemaOrName(
            ::std::forward<NameOrSchema>(raw_name_or_schema)),
        // 移动传入的函数对象到成员变量 f
        ::std::move(f), tags);
  }


在这段代码中，主要是一个 C++ 的模板函数 `impl`，用于注册操作符的实现。下面是详细的注释解释了每一行代码的作用。
#if defined C10_MOBILE
// 如果定义了 C10_MOBILE 宏，则使用无推断模式标记构造 CppFunction 对象 f
CppFunction f(std::forward<Func>(raw_f), NoInferSchemaTag());
#else
// 否则，直接使用原始的 raw_f 构造 CppFunction 对象 f
CppFunction f(std::forward<Func>(raw_f));
#endif
// 调用 _impl 函数，传入名称 name、移动构造的 CppFunction 对象 f 和 rv 参数，并返回结果
return _impl(name, std::move(f), rv);
}

#if defined C10_MOBILE
// 注意: 此重载仅在 C10_MOBILE 环境下需要，因为CppFunction的自动生成拷贝构造函数不包含额外的 NoInferSchemaTag 参数。
// 我们定义了此重载函数以接受 CppFunction&& 参数。已构造的 CppFunction 对象可能有或者没有推断模式的标记，但这对我们的目的没有影响，
// 因为如果它已经有了推断模式的标记，那么我们可以直接通过传递它来处理。
//
template <typename Name>
Library& impl(Name name, CppFunction&& raw_f) & {
// TODO: 当实现一个具有 catch all def 的函数时需要引发错误
CppFunction f(std::forward<CppFunction>(raw_f));
// 调用 _impl 函数，传入名称 name 和移动构造的 CppFunction 对象 f，并返回结果
return _impl(name, std::move(f));
}
#endif

// 获取一个 const char* 类型的 OperatorName 的辅助函数。你可能不需要这个函数。
c10::OperatorName _resolve(const char* name) const;

/// \private
///
/// 方便重载，直接指定调度键在 impl() 中的重载。你可能不需要这个；更好的方法是在 TORCH_LIBRARY_IMPL() 中指定整个块的调度键。
template <typename Name, typename Dispatch, typename Func>
Library& impl(Name name, Dispatch&& key, Func&& raw_f) & {
return impl(
name, dispatch(std::forward<Dispatch>(key), std::forward<Func>(raw_f)));
}

// 当一个 SelectiveStr（参见笔记 [Selective build]）在编译时被禁用时，这些重载函数将处理这种情况。在这种情况下，不生成任何引用传入函数的代码。
Library& def(detail::SelectiveStr<false>, const std::vector<at::Tag>& tags = {}) & {
return *this;
}
Library& def(detail::SelectiveStr<true> raw_schema, const std::vector<at::Tag>& tags = {}) & {
return def(raw_schema.operator const char*(), tags);
}
template <typename Func>
Library& def(detail::SelectiveStr<false>, Func&& /*raw_f*/, const std::vector<at::Tag>& tags = {}) & {
return *this;
}
template <typename Func>
Library& def(detail::SelectiveStr<true> raw_name_or_schema, Func&& raw_f, const std::vector<at::Tag>& tags = {}) & {
return def(
raw_name_or_schema.operator const char*(), std::forward<Func>(raw_f), tags);
}

template <typename Func>
Library& impl(detail::SelectiveStr<false>, Func&& /*raw_f*/) & {
    // 返回当前对象的引用，用于支持链式调用
    return *this;
  }

  template <typename Dispatch, typename Func>
  // 用于实现指定条件下的函数，但在此处并未使用 key 和 raw_f 参数
  Library& impl(
      detail::SelectiveStr<false>,
      Dispatch&& /*key*/,
      Func&& /*raw_f*/) & {
    // 返回当前对象的引用，用于支持链式调用
    return *this;
  }

  template <typename Func>
  // 实现不带包装的函数的接口，但已经弃用，并建议使用 .impl(...) 替代
  Library& impl_UNBOXED(
      detail::SelectiveStr<false> /*name*/,
      Func* /*raw_f*/) & {
    // 静态断言，提示 .impl_UNBOXED(...) 已移除，请使用 .impl(...) 替代
    static_assert(
        c10::guts::false_t<Func>(),
        ".impl_UNBOXED(...) was removed. Please use .impl(...) instead.");
    // 返回当前对象的引用，用于支持链式调用
    return *this;
  }

  template <typename Func>
  // 实现带有包装的函数接口
  Library& impl(detail::SelectiveStr<true> name, Func&& raw_f) & {
    // 调用 .impl(...) 实现具体逻辑
    return impl(name.operator const char*(), std::forward<Func>(raw_f));
  }

  template <typename Dispatch, typename Func>
  // 实现带有包装和条件的函数接口
  Library& impl(
      detail::SelectiveStr<true> name,
      Dispatch&& key,
      Func&& raw_f) & {
    // 调用 .impl(...) 实现具体逻辑
    return impl(
        name.operator const char*(),
        std::forward<Dispatch>(key),
        std::forward<Func>(raw_f));
  }

  template <typename Func>
  // 弃用的接口，提醒使用者不再使用 .impl_UNBOXED(...)，而是使用 .impl(...)
  Library& impl_UNBOXED(
      detail::SelectiveStr<true> /*name*/,
      Func* /*raw_f*/) & {
    static_assert(
        c10::guts::false_t<Func>(),
        ".impl_UNBOXED(...) was removed. Please use .impl(...) instead.");
    // 返回当前对象的引用，用于支持链式调用
    return *this;
  }

  /// 注册所有操作符的回退实现，如果没有特定操作符的实现可用，则使用该回退实现。
  /// 必须与一个 DispatchKey 关联；例如，只能从 TORCH_LIBRARY_IMPL() 的命名空间 `_` 中调用。
  ///
  /// \param raw_f 实现回退的函数。通常，未包装的函数不适合作为回退函数，
  ///   因为回退函数必须适用于每个操作符（即使它们具有不同的类型签名）。
  ///   典型的参数是 CppFunction::makeFallthrough() 或 CppFunction::makeFromBoxedFunction()
  ///
  /// ```
  /// // 示例:
  /// TORCH_LIBRARY_IMPL(_, AutogradXLA, m) {
  ///   // 如果没有为 AutogradXLA 显式注册内核，则回退到下一个可用的内核
  ///   m.fallback(torch::CppFunction::makeFallthrough());
  /// }
  ///
  /// // 详见 aten/src/ATen/core/dispatch/backend_fallback_test.cpp
  /// // 获取关于包装回退的完整示例
  /// ```py
  template <typename Func>
  // 注册回退实现的函数模板
  Library& fallback(Func&& raw_f) & {
    // 创建 CppFunction 对象，将传入的函数对象 raw_f 转发给它
    CppFunction f((std::forward<Func>(raw_f)));
    // 返回当前对象的引用，用于支持链式调用
    return *this;
  }
    // 返回带有 std::move 操作的 _fallback 函数的结果
    return _fallback(std::move(f));
    
    
    
    // 为给定类名创建一个 torch::class_ 对象的模板函数声明
    template <class CurClass>
    inline torch::class_<CurClass> class_(const std::string& className);
    
    // 这些重载函数允许在库中注册的类上使用选择性构建。
    // API 与之前相同，但有一个小变化。
    // 现在使用 TORCH_SELECTIVE_CLASS 宏来注册类，例如：m.class_<foo>(TORCH_SELECTIVE_CLASS("foo"))
    template <class CurClass>
    inline torch::class_<CurClass> class_(detail::SelectiveStr<true> className);
    
    // 当选择性构建未激活时，返回 detail::ClassNotSelected 对象
    template <class CurClass>
    inline detail::ClassNotSelected class_(detail::SelectiveStr<false> className);
    
    // 取消注册该库中创建的所有注册项
    void reset();
    
    private:
    // 类型和命名空间相关信息
    Kind kind_;
    std::optional<std::string> ns_;
    // 分发键相关信息
    std::optional<c10::DispatchKey> dispatch_key_;
    // Python 模块相关信息
    std::optional<std::pair<const char*, const char*>> python_module_;
    // 文件路径
    const char* file_;
    // 行号
    uint32_t line_;
    
    // 用于管理注册的对象句柄列表
    std::vector<c10::RegistrationHandleRAII> registrars_;
    
    friend class detail::TorchLibraryInit;
    
    // 实际函数的非用户可见实现。这些函数不是公开的，因为我们只实现了 & 限定符，而不是 && 限定符。
    // 下面是几个 _def 函数的声明
    Library& _def(
        c10::FunctionSchema&& schema,
        c10::OperatorName* out_name = nullptr,
        const std::vector<at::Tag>& tags = {},
        _RegisterOrVerify rv = _RegisterOrVerify::REGISTER) &;
    
    Library& _def(
        std::variant<c10::OperatorName, c10::FunctionSchema>&&,
        CppFunction&& f,
        const std::vector<at::Tag>& tags = {}) &;
    
    // 为给定函数名注册具体实现的函数
    Library& _impl(
        const char* name,
        CppFunction&& f,
        _RegisterOrVerify rv = _RegisterOrVerify::REGISTER) &;
    
    // 返回带有回退操作的函数
    Library& _fallback(CppFunction&& f) &;
    
    // 解析给定名称字符串，返回对应的 OperatorName 对象
    at::OperatorName _parseNameForLib(const char* name_str) const;
/// 结束了全局命名空间 "torch" 的定义
};

/// 包含了 "torch" 命名空间内部的细节实现
namespace detail {

/// TorchLibraryInit 类用于在构造时初始化一个特定的库对象
class TorchLibraryInit final {
 private:
  using InitFn = void(Library&); ///< 定义了一个函数指针类型 InitFn，用于初始化 Library 对象
  Library lib_; ///< 私有成员变量，表示要初始化的 Library 对象

 public:
  /// TorchLibraryInit 构造函数，接受多个参数用于初始化
  /// \param kind 表示 Library 的种类
  /// \param fn 指向初始化函数的函数指针
  /// \param ns 表示命名空间的名称
  /// \param k 可选的 DispatchKey
  /// \param file 包含初始化函数的文件名
  /// \param line 包含初始化函数的行号
  TorchLibraryInit(
      Library::Kind kind,
      InitFn* fn,
      const char* ns,
      std::optional<c10::DispatchKey> k,
      const char* file,
      uint32_t line)
      : lib_(kind, ns, k, file, line) { ///< 初始化 lib_ 成员变量
    fn(lib_); ///< 调用传入的初始化函数指针，对 lib_ 进行初始化
  }
};

} // namespace detail

} // namespace torch

// NB: The EXACT NAMING of the initializer functions (e.g.,
// TORCH_LIBRARY_init_aten) matters for the code analyzer;
// see the regexes at tools/code_analyzer/run_analyzer.sh

/// 宏定义，用于定义在静态初始化时运行的函数，以在命名空间 `ns` 中定义操作库
/// \param ns 命名空间的名称，必须是有效的 C++ 标识符，无引号
/// \param m torch::Library 的引用，用于注册操作符
/// \note 只能为给定的命名空间定义一个 TORCH_LIBRARY()
#define TORCH_LIBRARY(ns, m)                                                   \
  static void TORCH_LIBRARY_init_##ns(torch::Library&);                        \
  static const torch::detail::TorchLibraryInit TORCH_LIBRARY_static_init_##ns( \
      torch::Library::DEF,                                                     \
      &TORCH_LIBRARY_init_##ns,                                                \
      #ns,                                                                     \
      c10::nullopt,                                                            \
      __FILE__,                                                                \
      __LINE__);                                                               \
  void TORCH_LIBRARY_init_##ns(torch::Library& m)

/// \private
///
/// 这是 TORCH_LIBRARY() 的一个版本，不强制要求只能有一个库（是一个“片段”）。
/// 在 PerOpRegistration.cpp 文件中使用，以及在不容易将所有操作注册到同一宏块中的地方
#define TORCH_LIBRARY_FRAGMENT(ns, m) _TORCH_LIBRARY_FRAGMENT(ns, m, C10_UID)

/// \private
///
/// 上述宏需要一个额外的唯一标识符（uid）来防止变量名冲突。如果在同一翻译单元中多次调用
/// TORCH_LIBRARY_FRAGMENT 且命名空间相同，则可能会发生此问题。注意，TORCH_LIBRARY
/// 变体不会遇到此问题，因为它强制要求对于给定命名空间只能调用一次。
/// 定义一个宏，用于在静态初始化时定义操作符重载，针对分派键 `k`（必须是 c10::DispatchKey 的未限定枚举成员）在命名空间 `ns` 中。
#define _TORCH_LIBRARY_FRAGMENT(ns, m, uid)                       \
  // 声明静态初始化函数，以初始化命名空间 `ns` 的库片段
  static void C10_CONCATENATE(                                    \
      TORCH_LIBRARY_FRAGMENT_init_##ns##_, uid)(torch::Library&); \
  // 声明静态初始化对象，负责注册命名空间 `ns` 的库片段
  static const torch::detail::TorchLibraryInit C10_CONCATENATE(   \
      TORCH_LIBRARY_FRAGMENT_static_init_##ns##_, uid)(           \
      torch::Library::FRAGMENT,                                   \
      &C10_CONCATENATE(TORCH_LIBRARY_FRAGMENT_init_##ns##_, uid), \
      #ns,                                                        \
      c10::nullopt,                                               \
      __FILE__,                                                   \
      __LINE__);                                                  \
  // 定义初始化函数，用于注册命名空间 `ns` 中的函数
  void C10_CONCATENATE(                                           \
      TORCH_LIBRARY_FRAGMENT_init_##ns##_, uid)(torch::Library & m)

/// 宏用于定义一个函数，该函数在静态初始化时运行，以为分派键 `k` 中的操作符提供重载实现。
/// 命名空间 `ns` 必须是有效的 C++ 标识符（不带引号）。
#define TORCH_LIBRARY_IMPL(ns, k, m) _TORCH_LIBRARY_IMPL(ns, k, m, C10_UID)

/// \private
///
/// 上述宏需要额外的唯一标识符（uid），以防止在同一翻译单元中多次使用相同命名空间和分派键时出现变量名冲突。
// 定义 TORCH_LIBRARY_IMPL 宏，用于注册 Torch 库的实现
#define _TORCH_LIBRARY_IMPL(ns, k, m, uid)                                \
  // 定义静态初始化函数，用于初始化 Torch 库
  static void C10_CONCATENATE(                                            \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library&);       \
  // 定义静态初始化对象，包括库的实现类型、初始化函数指针、命名空间、DispatchKey、文件名和行号
  static const torch::detail::TorchLibraryInit C10_CONCATENATE(           \
      TORCH_LIBRARY_IMPL_static_init_##ns##_##k##_, uid)(                 \
      torch::Library::IMPL,                                               \
      (c10::impl::dispatch_key_allowlist_check(c10::DispatchKey::k)       \
           // 检查 DispatchKey 是否允许注册，选择性地设置初始化函数指针
           ? &C10_CONCATENATE(TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid) \
           : [](torch::Library&) -> void {}),                             \
      #ns,                                                                \
      c10::make_optional(c10::DispatchKey::k),                            \
      __FILE__,                                                           \
      __LINE__);                                                          \
  // 定义初始化函数，用于实际初始化 Torch 库
  void C10_CONCATENATE(                                                   \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library & m)

// 以下是用于测试的宏变体，不设置静态初始化器，允许手动控制库的可见性
//
// 在生产代码中不要使用这些宏，因为代码分析器无法正确分析它们。

/// \private
// 创建 Torch 库对象，用于公共库
#define MAKE_TORCH_LIBRARY(ns) \
  torch::Library(torch::Library::DEF, #ns, c10::nullopt, __FILE__, __LINE__)
/// \private
// 创建 Torch 库对象，用于实现库
#define MAKE_TORCH_LIBRARY_IMPL(ns, k)         \
  torch::Library(                              \
      torch::Library::IMPL,                    \
      #ns,                                     \
      c10::make_optional(c10::DispatchKey::k), \
      __FILE__,                                \
      __LINE__)

// 使自定义类 API 可见，以便从 torch::Library 中访问

#include <torch/custom_class.h>
```