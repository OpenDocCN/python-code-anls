# `.\pytorch\torch\csrc\jit\passes\value_refinement_utils.h`

```
#pragma once

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/peephole_list_idioms.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

namespace torch {
namespace jit {

// 表示从值类型为列表的值到长度的精炼信息
// 如果在块中存在列表值 * 到长度的精炼映射，则可以保证列表的长度
// TODO: 使用 vector 可能更快
using ListRefinement = std::unordered_map<Value*, int64_t>;

// 对两个列表精炼信息进行交集操作
TORCH_API ListRefinement
intersectRefinements(const ListRefinement& ref1, const ListRefinement& ref2);

// 对两个列表精炼信息进行并集操作
TORCH_API ListRefinement
unionRefinements(const ListRefinement& ref1, const ListRefinement& ref2);

// 表示可以在布尔值上携带的精炼信息
struct BooleanRefinementMapping {
  BooleanRefinementMapping(
      ListRefinement true_refine,
      ListRefinement false_refine)
      : true_refine_(std::move(true_refine)),
        false_refine_(std::move(false_refine)){};
  BooleanRefinementMapping() = default; // 空的默认构造函数

  // 返回一个包含 false 精炼信息的布尔值精炼映射
  static BooleanRefinementMapping FalseRefinements(
      ListRefinement false_refine) {
    return BooleanRefinementMapping({}, std::move(false_refine));
  }

  // 返回一个包含 true 精炼信息的布尔值精炼映射
  static BooleanRefinementMapping TrueRefinements(ListRefinement true_refine) {
    return BooleanRefinementMapping(std::move(true_refine), {});
  }

  // 对两个布尔值精炼映射进行交集操作
  BooleanRefinementMapping intersectBooleanRefinementMapping(
      BooleanRefinementMapping& other) {
    return BooleanRefinementMapping(
        intersectRefinements(true_refine_, other.true_refine()),
        intersectRefinements(false_refine_, other.false_refine()));
  }

  // 返回 true 精炼信息的引用
  ListRefinement& true_refine() {
    return true_refine_;
  }

  // 返回 false 精炼信息的引用
  ListRefinement& false_refine() {
    return false_refine_;
  }

 private:
  ListRefinement true_refine_;    // 保存 true 精炼信息的成员变量
  ListRefinement false_refine_;   // 保存 false 精炼信息的成员变量
};

// 处理将块添加到抛出块并通过布尔比较传播精炼信息的操作
TORCH_API void joinIfRefinements(
    Node* if_node,
    std::unordered_set<Block*>& throwing_blocks,
    ListRefinement& curr_block_refinements,
    ListRefinement& true_block_refinements,
    ListRefinement& false_block_refinements,
    std::unordered_map<Value*, BooleanRefinementMapping>& info);

// 处理常见的精炼操作符，如布尔比较，将块添加到抛出块，并通过布尔比较传播精炼信息
TORCH_API bool handleCommonRefinentOperators(
    Node* n,
    std::unordered_set<Block*>& throwing_blocks,
    std::unordered_map<Value*, BooleanRefinementMapping>& info);

} // namespace jit
} // namespace torch
```