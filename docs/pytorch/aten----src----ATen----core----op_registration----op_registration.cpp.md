# `.\pytorch\aten\src\ATen\core\op_registration\op_registration.cpp`

```py
// 包含C10和Torch的头文件，用于宏和操作注册
#include <c10/macros/Macros.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/op_allowlist.h>
#include <ATen/core/op_registration/op_registration.h>

// 如果不是跨平台构建，包含用于函数模式解析的头文件
#if !defined(CAFFE2_IS_XPLAT_BUILD)
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#endif

// C10命名空间内定义实现相关功能
namespace c10 {
namespace impl {

// 如果需要的功能不可用，抛出异常
void build_feature_required_feature_not_available(const char* feature) {
  TORCH_CHECK(
      false,
      "Required feature '" + std::string(feature) + "' is not available");
}
} // namespace impl

// 静态断言，验证可选的RegistrationHandleRAII是否具有无异常移动构造和赋值
static_assert(std::is_nothrow_move_constructible<
              std::optional<RegistrationHandleRAII>>::value);
static_assert(std::is_nothrow_move_assignable<
              std::optional<RegistrationHandleRAII>>::value);

// 注册运算符的类实现
void RegisterOperators::checkSchemaAndRegisterOp_(Options&& options) {
  // 检查是否提供了模式或运算符名称
  TORCH_CHECK(
      options.schemaOrName_.has_value(),
      "In operator registration: Tried to register an operator without specifying a schema or operator name.");
  
  // 如果索引为1，表示模式已明确指定
  if (options.schemaOrName_->index() == 1) {
    // 明确指定了模式，检查是否存在重复的内核，并注册运算符
    checkNoDuplicateKernels_(options);
    registerOp_(std::move(options));
  } else {
    // 模式未明确指定，从内核推断模式并注册运算符
    OperatorName name =
        std::get<OperatorName>(std::move(*options.schemaOrName_));
    FunctionSchema inferred_schema = inferSchemaFromKernels_(name, options);

    // 使用推断出的模式填充选项中的模式或名称
    options.schemaOrName_ = FunctionSchema(
        std::move(name.name),
        std::move(name.overload_name),
        inferred_schema.arguments(),
        inferred_schema.returns(),
        inferred_schema.is_vararg(),
        inferred_schema.is_varret());

    // 检查是否存在重复的内核，并注册运算符
    checkNoDuplicateKernels_(options);

    // 如果使用FROM_SCHEMA进行别名分析，会导致不符合预期的行为，因为推断出的模式不包含别名注解
    TORCH_CHECK(
        options.aliasAnalysisKind_ != AliasAnalysisKind::FROM_SCHEMA,
        "In operator registration: Tried to register operator ",
        std::get<FunctionSchema>(options.schemaOrName_.value()),
        " with AliasAnalysisKind::FROM_SCHEMA, but the schema is inferred.");

    // 使用推断出的模式注册所有内核
    registerOp_(std::move(options));
  }
}

// 根据内核推断运算符的函数模式
c10::FunctionSchema RegisterOperators::inferSchemaFromKernels_(
    const OperatorName& opName,
    const RegisterOperators::Options& options) {
  // 检查是否指定了内核，否则无法推断运算符的模式
  TORCH_CHECK(
      !options.kernels.empty(),
      "Cannot infer operator schema in registration of operator ",
      opName,
      " because there is no kernel specified.");

  std::optional<FunctionSchema> inferred_schema = c10::nullopt;
  for (const auto& kernel : options.kernels) {
    // 如果推断函数模式不为空，则使用第一个推断的函数模式
    if (nullptr != kernel.inferred_function_schema.get()) {
      if (!inferred_schema.has_value()) {
        inferred_schema = *kernel.inferred_function_schema;
        break;
      }
      // 如果有多个内核具有推断的函数模式，则忽略后续的内核
    }
  }
  }
  // 关闭 inner if 语句块

  TORCH_CHECK(
      inferred_schema.has_value(),
      // 使用 TORCH_CHECK 来检查 inferred_schema 是否有值，若无值则抛出错误信息
      "Cannot infer operator schema for this kind of kernel in registration of operator ",
      opName,
      ". Please explicitly specify the operator schema or specify at least one kernel for which we can infer the schema.");
      // 错误信息说明无法推断操作符的模式，需要显式指定操作符模式或者至少指定一个能够推断模式的内核。

  return *inferred_schema;
  // 返回推断出的操作符模式
}

// 检查操作符注册选项中没有重复的内核函数
void RegisterOperators::checkNoDuplicateKernels_(const Options& options) {
  // 使用无序集合存储分派键，确保每个分派键只注册一次
  std::unordered_set<DispatchKey> dispatch_keys;
  // 标记是否已经存在通用内核函数
  bool has_catchall_kernel = false;

  // 遍历所有内核函数选项
  for (const auto& kernel : options.kernels) {
    // 如果内核函数有指定的分派键
    if (kernel.dispatch_key.has_value()) {
      // 检查是否已经注册过相同的分派键
      TORCH_CHECK(
          0 == dispatch_keys.count(*kernel.dispatch_key),
          "In operator registration: Tried to register multiple kernels with same dispatch key ",
          *kernel.dispatch_key,
          " for operator schema ",
          toString(std::get<FunctionSchema>(options.schemaOrName_.value())));
      // 将分派键添加到集合中
      dispatch_keys.insert(*kernel.dispatch_key);
    } else {
      // 如果没有指定分派键，检查是否已经存在通用内核函数
      TORCH_CHECK(
          !has_catchall_kernel,
          "In operator registration: Tried to register multiple catch-all kernels for operator schema ",
          toString(std::get<FunctionSchema>(options.schemaOrName_.value())));
      // 标记已经存在通用内核函数
      has_catchall_kernel = true;
    }
  }
}

// 注册操作符的内部实现
void RegisterOperators::registerOp_(Options&& options) {
  // 获取函数模式架构
  FunctionSchema schema =
      std::get<FunctionSchema>(std::move(options.schemaOrName_.value()));

  // HACK: 直接从旧API中传入别名分析类型到模式架构中
  if (options.aliasAnalysisKind_.has_value()) {
    // 设置模式架构的别名分析类型
    schema.setAliasAnalysis(*options.aliasAnalysisKind_);
  }

  // 获取操作符名称
  OperatorName op_name = schema.operator_name();

  // 将操作符架构注册到全局调度器中
  registrars_.emplace_back(Dispatcher::singleton().registerDef(
      std::move(schema), "registered by RegisterOperators"));

  // 遍历所有内核函数选项
  for (auto& kernel : options.kernels) {
    // 注册操作符的具体实现到全局调度器中
    registrars_.emplace_back(Dispatcher::singleton().registerImpl(
        op_name,
        kernel.dispatch_key,
        std::move(kernel.func),
        kernel.cpp_signature,
        std::move(kernel.inferred_function_schema),
        "registered by RegisterOperators"));
  }
}

} // namespace c10
```