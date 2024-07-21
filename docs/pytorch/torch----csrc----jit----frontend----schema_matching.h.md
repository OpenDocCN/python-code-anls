# `.\pytorch\torch\csrc\jit\frontend\schema_matching.h`

```
// 预处理指令，用于确保头文件只被包含一次
#pragma once

// 包含导出符号定义的 Torch 头文件
#include <torch/csrc/Export.h>

// 包含 JIT IR 的头文件
#include <torch/csrc/jit/ir/ir.h>

// 包含命名值定义的头文件
#include <torch/csrc/jit/ir/named_value.h>

// 包含函数模式定义的头文件
#include <ATen/core/function_schema.h>

// Torch 名称空间
namespace torch {

// JIT 名称空间
namespace jit {

// 匹配函数模式的结果结构体
struct MatchedSchema {
  std::vector<Value*> inputs;       // 匹配后的位置输入列表
  std::vector<TypePtr> return_types;    // 返回类型列表
  c10::OptNameList return_field_names; // 返回字段名的可选列表
  std::string schema_name;          // 函数模式的名称
};

// 检查函数模式是否在阻止列表中
TORCH_API bool isBlockListedSchema(const FunctionSchema& schema);

// 尝试将输入与关键字参数匹配到函数模式上
TORCH_API MatchedSchema matchSchema(
    const ::c10::FunctionSchema& schema, // 函数模式
    const SourceRange& loc,             // 源代码范围
    Graph& graph,                       // 图形对象
    at::ArrayRef<NamedValue> args,      // 位置参数列表
    at::ArrayRef<NamedValue> kwargs,    // 关键字参数列表
    const std::optional<NamedValue>& self = c10::nullopt); // 可选的 self 参数

// 尝试将输入与多个函数模式匹配，并返回最佳匹配的索引和结果
TORCH_API std::pair<size_t, MatchedSchema> matchSchemas(
    const std::vector<const ::c10::FunctionSchema*>& schemas, // 多个函数模式的列表
    const SourceRange& loc,             // 源代码范围
    Graph& graph,                       // 图形对象
    at::ArrayRef<NamedValue> args,      // 位置参数列表
    at::ArrayRef<NamedValue> kwargs,    // 关键字参数列表
    const std::optional<NamedValue>& self = c10::nullopt, // 可选的 self 参数
    bool render_errors = false);        // 是否渲染错误信息

// 检查类型是否可转换为列表类型
TORCH_API bool convertibleToList(
    const TypePtr& type,        // 类型指针
    const TypePtr& list_type_); // 列表类型指针

// 获取函数模式的完整名称
TORCH_API std::string getFullSchemaName(const ::c10::FunctionSchema& schema);

// 发出内置调用
TORCH_API Value* emitBuiltinCall(
    const SourceRange& loc,             // 源代码范围
    Graph& graph,                       // 图形对象
    Symbol name,                        // 符号名称
    at::ArrayRef<NamedValue> args,      // 位置参数列表
    at::ArrayRef<NamedValue> kwargs,    // 关键字参数列表
    const std::optional<NamedValue>& self = c10::nullopt); // 可选的 self 参数

// 查找具有特定名称的输入参数在关键字参数列表中的位置
TORCH_API std::optional<size_t> findInputWithName(
    const std::string& name,    // 参数名称
    at::ArrayRef<NamedValue> kwargs, // 关键字参数列表
    bool is_aten = false);      // 是否为 ATen 函数

// 尝试将值隐式转换为指定类型
TORCH_API Value* tryConvertToType(
    const SourceRange& loc,             // 源代码范围
    Graph& graph,                       // 图形对象
    const TypePtr& concrete_type,       // 具体的目标类型
    Value* value,                       // 要转换的值
    bool allow_conversions);            // 是否允许类型转换
} // namespace jit
} // namespace torch
```