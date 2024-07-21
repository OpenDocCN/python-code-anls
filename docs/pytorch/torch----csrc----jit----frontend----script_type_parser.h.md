# `.\pytorch\torch\csrc\jit\frontend\script_type_parser.h`

```py
#pragma once
#include <ATen/core/jit_type.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/frontend/tree_views.h>

namespace torch {
namespace jit {

/**
 * class ScriptTypeParser
 *
 * Parses expressions in our typed AST format (TreeView) into types and
 * typenames.
 */
class TORCH_API ScriptTypeParser {
 public:
  explicit ScriptTypeParser() = default;  // 默认构造函数

  // 使用给定的解析器构造函数
  explicit ScriptTypeParser(ResolverPtr resolver)
      : resolver_(std::move(resolver)) {}

  // 从表达式中解析类型
  c10::TypePtr parseTypeFromExpr(const Expr& expr) const;

  // 解析广播列表
  std::optional<std::pair<c10::TypePtr, int32_t>> parseBroadcastList(
      const Expr& expr) const;

  // 解析字符串形式的类型
  c10::TypePtr parseType(const std::string& str);

  // 从函数定义解析函数模式
  FunctionSchema parseSchemaFromDef(const Def& def, bool skip_self);

  // 解析类常量
  c10::IValue parseClassConstant(const Assign& assign);

 private:
  // 实现从表达式解析类型
  c10::TypePtr parseTypeFromExprImpl(const Expr& expr) const;

  // 解析基本类型名称
  std::optional<std::string> parseBaseTypeName(const Expr& expr) const;

  // 将下标转换为类型
  at::TypePtr subscriptToType(
      const std::string& typeName,
      const Subscript& subscript) const;

  // 评估默认值
  std::vector<IValue> evaluateDefaults(
      const SourceRange& r,
      const std::vector<Expr>& default_types,
      const std::vector<Expr>& default_exprs);

  // 从声明中解析参数
  std::vector<Argument> parseArgsFromDecl(const Decl& decl, bool skip_self);

  // 从声明中解析返回值
  std::vector<Argument> parseReturnFromDecl(const Decl& decl);

  ResolverPtr resolver_ = nullptr;  // 解析器指针

  // 在序列化中需要使用 `evaluateDefaults`
  friend struct ConstantTableValue;
  friend struct SourceImporterImpl;
};

} // namespace jit
} // namespace torch
```