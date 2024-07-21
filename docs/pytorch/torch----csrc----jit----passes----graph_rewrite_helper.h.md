# `.\pytorch\torch\csrc\jit\passes\graph_rewrite_helper.h`

```py
#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {
namespace graph_rewrite_helper {

// 定义一个函数，用于从给定的值获取其函数名
std::string getFuncName(Value* func_value);

// 根据名称从映射中获取值，映射中保存了匹配的值对应关系
Value* getValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap);

// 根据名称从映射中获取值，并尝试将其转换为 IValue 类型
std::optional<IValue> getIValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap);

// 替换图中的卷积操作为 Aten 格式的卷积
TORCH_API void replaceConvolutionWithAtenConv(std::shared_ptr<Graph>& graph);

// 判断是否可以合并 Clamp 操作
bool isClampFusable(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

// 此结构体包含一个编译后的 IR 模式，用于在 findPatternMatches 函数中使用
// 该结构体封装了从 parseIR 中提取的通用信息，用于模式匹配功能
// 可以存储一个 const 实例以缓存编译后的 IR 模式，降低运行时成本
struct PatternInfo {
  std::string pattern_string;  // IR 模式的字符串表示
  std::unique_ptr<Graph> pattern_graph;  // 解析后的图表示 IR 模式
  std::unordered_map<std::string, Value*> vmap;  // 值的映射，用于变量名到值的查找
  std::vector<MatchFilter> filters;  // 匹配过滤器

  // 从字符串解析 IR 模式，并返回 PatternInfo 结构体
  static PatternInfo parse_from_str(
      std::string pattern_string,
      const std::vector<MatchFilter>& filters = {}) {
    PatternInfo rv{
        std::move(pattern_string),
        std::make_unique<Graph>(),
        decltype(vmap){},
        filters};
    // 使用 parseIR 函数解析字符串为图形式的 IR 模式，并更新 vmap 映射
    parseIR(rv.pattern_string, rv.pattern_graph.get(), rv.vmap);
    return rv;
  }
};

} // namespace graph_rewrite_helper
} // namespace jit
} // namespace torch
```