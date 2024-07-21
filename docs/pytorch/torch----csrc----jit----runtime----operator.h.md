# `.\pytorch\torch\csrc\jit\runtime\operator.h`

```py
// 在内存中描述所有类似于 Caffe2 schema 的 ATen 操作
// 一旦 C10 存在，这段代码可以被移除或者替换，但现在我们需要它来实现脚本的正确语义检查
#pragma once

#include <ATen/core/dispatch/Dispatcher.h>                 // 引入 ATen 的调度器头文件
#include <ATen/core/dispatch/OperatorOptions.h>           // 引入 ATen 的操作选项头文件
#include <ATen/core/op_registration/op_allowlist.h>       // 引入 ATen 的操作允许列表头文件
#include <ATen/core/stack.h>                              // 引入 ATen 的堆栈头文件
#include <c10/util/Exception.h>                           // 引入 C10 的异常处理头文件
#include <c10/util/overloaded.h>                          // 引入 C10 的 overloaded 实用程序
#include <torch/csrc/jit/frontend/function_schema_parser.h> // 引入 Torch 的函数模式解析器头文件
#include <torch/csrc/jit/runtime/operator_options.h>       // 引入 Torch 的运算符选项头文件
#include <torch/library.h>                                // 引入 Torch 库

#include <ATen/core/function_schema.h>                    // 引入 ATen 的函数模式头文件
#include <ATen/core/symbol.h>                             // 引入 ATen 的符号头文件

#include <functional>                                     // 引入函数头文件
#include <initializer_list>                               // 引入初始化列表头文件
#include <memory>                                         // 引入内存头文件
#include <string>                                         // 引入字符串头文件
#include <unordered_map>                                  // 引入无序映射头文件
#include <utility>                                        // 引入实用程序头文件
#include <variant>                                        // 引入变体头文件
#include <vector>                                         // 引入向量头文件

namespace torch::jit {

struct Node;                                              // 声明 Node 结构体
using ::c10::Argument;                                    // 使用 C10 的 Argument
using ::c10::FunctionSchema;                              // 使用 C10 的 FunctionSchema
using ::c10::Symbol;                                      // 使用 C10 的 Symbol

using OperationCreator = Operation (*)(const Node*);       // 定义 OperationCreator 类型别名，是一个指向操作的函数指针

namespace {
const std::array<at::Tag, 1> kJitOnlyOperatorTags = {      // 定义只在 JIT 中使用的操作标签数组
    at::Tag::pt2_compliant_tag};                          // 标记为 pt2_compliant_tag
}

/*
 * 注意：JIT 依赖于操作符实例具有静态生存期，因为它在 Node 类中存储了一个非拥有的 FunctionSchema* 指针，
 * 指向存储在 Operator 实例中的函数模式。
 * 此外，jit::Operator 旨在存储更多与操作符相关的信息，如符号导数，这也要求它们具有静态生存期，
 * 以便记住符号导数的更改。
 *
 * 目前，JIT 操作符库包含一个 jit::Operator 实例，其中每个 c10 操作符都有一个包装器。
 * c10 操作符库使用 register_c10_ops.cpp 中的监听器注册这些包装器。
 * TODO：与其以这种方式做，我们应该只在 JIT 库中有纯 JIT 操作符，但是 JIT 操作符查找也应该查看 c10 库。
 */

// Operator 是一个围绕纯 JIT 操作符（如 prim 操作）或 c10 操作符的薄包装器，
// 允许一些常见操作并抽象出具体操作符的性质。
struct TORCH_API Operator {
 private:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  struct C10Operator final {
    c10::OperatorHandle handle_;                          // c10 操作符句柄
    Operation op_;                                        // 操作对象
  };
  struct UnparsedFunctionSchema final {
    std::string schema_string_;                           // 未解析的函数模式字符串
    mutable std::optional<c10::AliasAnalysisKind> alias_analysis_; // 可变的别名分析种类的可选项
  };
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  struct JitOnlyOperator final {
    // schema_ 的唯一有效转换是从右向左，即当模式被解析时。
    mutable std::variant<FunctionSchema, UnparsedFunctionSchema> schema_; // 可变的模式变量，可以是解析后的函数模式或未解析的函数模式
    std::variant<Operation, OperationCreator> op_;

# 定义一个 std::variant 对象 op_，它可以存储两种类型的数据：Operation 或 OperationCreator。

  };

 public:
  Operator(c10::OperatorHandle opHandle, Operation operation)
      : op_(C10Operator{std::move(opHandle), std::move(operation)}) {}

# Operator 类的构造函数，接受一个操作句柄 opHandle 和一个操作对象 operation，将它们封装到 C10Operator 中，并存储到 op_ 中。

  Operator(
      std::string schema,
      Operation op,
      c10::AliasAnalysisKind alias_analysis)
      : op_(JitOnlyOperator{
            UnparsedFunctionSchema{std::move(schema), alias_analysis},
            Operation(std::move(op))}) {}

# Operator 类的构造函数，接受一个字符串 schema、一个操作对象 op 和一个别名分析类型 alias_analysis。它将这些信息封装到 JitOnlyOperator 中，并存储到 op_ 中。

  Operator(
      std::string name,
      std::string overload_name,
      std::vector<Argument> arguments,
      std::vector<Argument> returns,
      Operation op,
      c10::AliasAnalysisKind alias_analysis)
      : op_(JitOnlyOperator{
            FunctionSchema(varArgSchemaWithName(
                std::move(name),
                std::move(overload_name),
                std::move(arguments),
                std::move(returns),
                alias_analysis)),
            std::move(op)}) {}

# Operator 类的构造函数，接受操作的名称 name、重载名称 overload_name、参数列表 arguments、返回值列表 returns、操作对象 op 和别名分析类型 alias_analysis。它创建一个带有函数模式的 JitOnlyOperator 并存储到 op_ 中。

  Operator(
      std::string schema,
      OperationCreator op_creator,
      c10::AliasAnalysisKind alias_analysis)
      : op_(JitOnlyOperator{
            UnparsedFunctionSchema{std::move(schema), alias_analysis},
            op_creator}) {}

# Operator 类的构造函数，接受一个字符串 schema、一个操作创建者对象 op_creator 和一个别名分析类型 alias_analysis。它将这些信息封装到 JitOnlyOperator 中，并存储到 op_ 中。

  // Helper constructor to register `op` to run
  // run for _every_ IR Node where n.kind() == name, regardless of arguments.
  // This is accomplished by marking the schema varargs and having no required
  // arguments.
  Operator(
      Symbol name,
      OperationCreator op_creator,
      c10::AliasAnalysisKind alias_analysis)
      : op_(JitOnlyOperator{
            FunctionSchema(varArgSchemaWithName(name, alias_analysis)),
            op_creator}) {}

# Operator 类的构造函数，接受一个符号名称 name、一个操作创建者对象 op_creator 和一个别名分析类型 alias_analysis。它创建一个带有变长参数模式的 JitOnlyOperator 并存储到 op_ 中。

  Operation getOperation(const Node* node = nullptr) const {
    return std::visit(
        c10::overloaded(
            [](const C10Operator& op) { return op.op_; },
            [node](const JitOnlyOperator& op) {
              return std::visit(
                  c10::overloaded(
                      [](const Operation& op) { return op; },
                      [node](const OperationCreator& op_creator) {
                        return op_creator(node);
                      }),
                  op.op_);
            }),
        op_);
  }

# 返回当前 Operator 对象中存储的操作对象。如果是 C10Operator 类型，则直接返回其 op_ 数据成员；如果是 JitOnlyOperator 类型，则根据情况返回其中的 Operation 或通过 OperationCreator 在给定节点上创建操作。

  Operation getOperationForDispatchKey(c10::DispatchKey dk) const {
    // TODO: some sort of caching mechanism?
    return std::visit(
        c10::overloaded(
            [dk](const C10Operator& op) {
              return Operation([op, dk](Stack& stack) {
                op.handle_.callBoxedForDispatchKey(dk, stack);
              });
            },
            [](const JitOnlyOperator& op) {
              TORCH_CHECK(
                  false,
                  "calling a JIT operator for dispatch key is not supported");
              return Operation(nullptr);
            }),
        op_);
  }

# 根据给定的 DispatchKey 返回操作对象。如果是 C10Operator 类型，则返回一个操作对象，它会调用 C10Operator 对象中的 callBoxedForDispatchKey 方法；如果是 JitOnlyOperator 类型，则抛出错误信息，因为不支持直接调用 JIT 操作的 Dispatch Key。

  const FunctionSchema& schema() const {

# 返回当前 Operator 对象中存储的函数模式对象的引用。
    // 使用 std::visit 访问包含不同类型操作符的 variant（op_），并执行对应的操作
    return std::visit(
        c10::overloaded(
            // 如果是 C10Operator 类型，则返回其关联的 FunctionSchema 对象
            [](const C10Operator& op) -> const FunctionSchema& {
              return op.handle_.schema();
            },
            // 如果是 JitOnlyOperator 类型，则处理其关联的 FunctionSchema 对象
            [](const JitOnlyOperator& op) -> const FunctionSchema& {
              // 延迟解析由字符串初始化的 schema，以减少静态操作符注册时的工作量
              if (op.schema_.index() == 1) {
                auto& unmaterializedSchema =
                    std::get<UnparsedFunctionSchema>(op.schema_);
                // 解析字符串表示的 schema
                FunctionSchema schema =
                    parseSchema(unmaterializedSchema.schema_string_);
                // 如果有别名分析信息，设置到解析后的 schema 中
                if (unmaterializedSchema.alias_analysis_.has_value()) {
                  schema.setAliasAnalysis(
                      *unmaterializedSchema.alias_analysis_);
                }
                // 将解析后的 schema 移动到 op 对象的 schema_ 成员中
                op.schema_ = std::move(schema);
              }
              // 返回 JitOnlyOperator 对象的解析后的 FunctionSchema
              return std::get<FunctionSchema>(op.schema_);
            }),
        op_);
  }

  // 获取操作符的标签数组
  c10::ArrayRef<at::Tag> getTags() const {
    return std::visit(
        c10::overloaded(
            // 如果是 C10Operator 类型，返回其关联的标签数组
            [](const C10Operator& op) { return op.handle_.getTags(); },
            // 如果是 JitOnlyOperator 类型，返回预定义的标签数组 kJitOnlyOperatorTags
            [](const JitOnlyOperator& op) {
              // JitOnlyOperator 没有 OperatorHandle 或指定标签的方法，弃用时默认使用 pt2_compliant_tag
              return c10::ArrayRef<at::Tag>(kJitOnlyOperatorTags);
            }),
        op_);
  }

  // 判断操作符是否是 C10Operator 类型
  bool isC10Op() const {
    // 判断 op_ 的 variant index 是否为 0，即判断是否是 C10Operator 类型
    return op_.index() == 0;
  }

  // 获取操作符的别名分析类型
  c10::AliasAnalysisKind aliasAnalysisKind() const {
    // 获取操作符的 FunctionSchema
    const FunctionSchema& schemaRef = schema();
    // 获取 schema 的别名分析类型
    c10::AliasAnalysisKind alias_analysis = schemaRef.aliasAnalysis();

    // 检查别名分析类型是否有效
    TORCH_CHECK(
        alias_analysis == AliasAnalysisKind::FROM_SCHEMA ||
            !schemaRef.hasAnyAliasInfo(),
        "In operator registration: Tried to register operator ",
        schemaRef,
        " with aliasing information in the schema but without AliasAnalysisKind::FROM_SCHEMA.");
    // 返回别名分析类型
    return alias_analysis;
  }

  // 判断操作符是否定义了操作
  bool hasOperation() const {
    // 使用 std::visit 访问 op_ 的 variant，根据类型判断是否有操作定义
    return std::visit(
        c10::overloaded(
            // 如果是 C10Operator 类型，返回 true
            [](const C10Operator&) { return true; },
            // 如果是 JitOnlyOperator 类型，判断其 op_ 的 index 是否为 0
            [](const JitOnlyOperator& op) { return op.op_.index() == 0; }),
        op_);
  }

 private:
  // 创建带有可变参数的 FunctionSchema 对象，并设置别名分析类型
  static FunctionSchema varArgSchemaWithName(
      Symbol name,
      AliasAnalysisKind alias_analysis) {
    auto result = FunctionSchema(
        name,
        "",
        {},
        {},
        /*is_vararg*/ true,
        /*is_varret*/ true);
    result.setAliasAnalysis(alias_analysis);
    return result;
  }

  // 创建带有可变参数的 FunctionSchema 对象，并设置别名分析类型
  static FunctionSchema varArgSchemaWithName(
      std::string name,
      std::string overload_name,
      std::vector<Argument> arguments,
      std::vector<Argument> returns,
      AliasAnalysisKind alias_analysis) {
    // 调用 FunctionSchema 构造函数，传入移动语义的参数 name, overload_name, arguments, returns
    // 设置 is_vararg 为 false，表示函数不接受可变参数
    // 设置 is_varret 为 false，表示函数不返回可变数量的结果
    auto result = FunctionSchema(
        std::move(name),
        std::move(overload_name),
        std::move(arguments),
        std::move(returns),
        /*is_vararg*/ false,
        /*is_varret*/ false);
    
    // 设置 FunctionSchema 对象的别名分析策略
    result.setAliasAnalysis(alias_analysis);
    
    // 返回构建好的 FunctionSchema 对象
    return result;
  }

  // 声明一个 std::variant 类型的成员变量 op_
  std::variant<C10Operator, JitOnlyOperator> op_;
// 下面是命名空间 torch::jit 的结尾

}; // 结束命名空间 torch::jit

// 返回给定函数模式的规范化字符串表示
TORCH_API std::string canonicalSchemaString(const FunctionSchema& schema);

// 获取所有操作符的列表
TORCH_API const std::vector<std::shared_ptr<Operator>> getAllOperators();

// 获取特定名称下的所有操作符的引用
TORCH_API const std::vector<std::shared_ptr<Operator>>& getAllOperatorsFor(
    Symbol name);

// 按 OpOverloadPacket 解析顺序返回特定名称下的所有操作符
TORCH_API std::vector<std::shared_ptr<Operator>> getAllSortedOperatorsFor(
    Symbol name);

// 给定一个带有重载名称的操作符，找到相关的操作符，如果不存在则返回 nullptr
TORCH_API std::shared_ptr<Operator> findOperatorFor(
    const c10::OperatorName& full_name);

// 查找与输入操作符类似的符号
TORCH_API std::vector<Symbol> findSimilarOperators(Symbol input_op);

// 注册一个操作符
TORCH_API void registerOperator(Operator&& op);

// 注销特定函数模式的操作符
TORCH_API void deregisterOperator(const FunctionSchema& schema);

// XXX: 此函数仅用于字符串字面量！
TORCH_API std::shared_ptr<Operator> getOperatorForLiteral(
    const char* signature);

// 确保注册 c10 操作符的函数被定义
// 如果未定义，注册表将不包含 c10 操作符。在静态初始化期间查询已注册的操作符时可能会遇到这种情况。
// 此函数在 register_c10_ops.cpp 中定义
TORCH_API void ensure_c10_registerer_defined();

// 用于断言未编制模式的操作符是否具有特殊情况分析方法
TORCH_API bool aliasAnalysisHasSpecialCaseFor(c10::Symbol sym);

// 一个工厂函数，生成一个可选的操作符。它有两种实例化方式，取决于模板布尔参数值。该参数可以是基于模式字符串的选择性操作注册的编译时函数。
template <typename Func>
std::optional<Operator> OperatorGenerator(
    const char* schema_str,
    Func&& op,
    AliasAnalysisKind alias_analysis) {
  return std::optional<Operator>(Operator(
      std::string(schema_str), std::forward<Func>(op), alias_analysis));
}

// 根据选择的布尔参数值生成可选操作符的工厂函数实例化方式之一
template <typename Func>
std::optional<Operator> OperatorGenerator(
    torch::detail::SelectiveStr<true> schema_str,
    Func&& op,
    AliasAnalysisKind alias_analysis) {
  return OperatorGenerator(
      static_cast<const char*>(schema_str),
      std::forward<Func>(op),
      alias_analysis);
}

// 根据选择的布尔参数值生成可选操作符的工厂函数实例化方式之一
template <typename Func>
std::optional<Operator> OperatorGenerator(
    torch::detail::SelectiveStr<false> schema_str,
    Func&& op,
    AliasAnalysisKind alias_analysis) {
  return c10::nullopt;
}

// 根据名称、重载名称、参数、返回值生成可选操作符的工厂函数实例化方式之一
template <typename Func>
std::optional<Operator> OperatorGenerator(
    const std::string name,
    const std::string overload_name,
    const std::vector<c10::Argument> arguments,
    const std::vector<c10::Argument> returns,
    Func&& op,
    AliasAnalysisKind alias_analysis) {
  return std::optional<Operator>(Operator(
      name,
      overload_name,
      arguments,
      returns,
      std::forward<Func>(op),
      alias_analysis));
}

} // namespace torch::jit
```