# `.\pytorch\torch\csrc\jit\passes\subgraph_rewrite.h`

```py
/** This file defines API for pattern-based subgraph rewrites.
 *
 * The API can be used for finding concrete patterns in the model and replacing
 * the corresponding subgraphs with another subgraph. A special case of such
 * rewrites is fusion, where the new subgraph consists of just a single node.
 *
 * There is a default set of the most common patterns that everyone could use.
 * Alternatively, an arbitrary pattern can be registered.
 */
#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

#include <functional>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {

// Forward declarations.
struct RewritePatternDescr;
struct Match;

using MatchFilter = std::function<
    bool(const Match&, const std::unordered_map<std::string, Value*>&)>;

/** Run pattern-based subgraph rewrites on all methods in the module.
 *
 * This pass will go through all methods in the module and try to replace all
 * recognized patterns (see SubgraphRewriter::RegisterDefaultPatterns for the
 * list of these patterns).
 */
TORCH_API Module PatternBasedRewrite(const Module& module);

/** A class implementing API for pattern-based subgraph rewrites.
 *
 * To perform pattern-based subgraph rewrites on a module using this API, one
 * needs to create an object of such class, register rewrite patterns and run
 * the transformation pass (`runOnModule`).
 *
 * To use standard patterns, one could use `RegisterDefaultPatterns`.
 *
 * To enable rewrites of custom patterns, the custom patterns must be registered
 * with `RegisterRewritePattern`.
 */
class TORCH_API SubgraphRewriter {
 public:
  // Run pattern-based subgraph rewrite pass on the module.
  Module runOnModule(const Module& module);

  // Run pattern-based subgraph rewrite pass on the graph (used in testing).
  // `filter` is a function that does extra filtering on the match. If it
  // returns false for a given Match, we'll skip the Match. The filter
  // function's arguments consist of a Match and a value map from parsing the
  // pattern graph. Both the Match and the value map are necessary because we
  // need to 1) do extra filtering on the matched result as well as 2) refer to
  // the values in the matched result through the values in the pattern graph.
  void runOnGraph(
      std::shared_ptr<Graph>& graph,
      const std::vector<MatchFilter>& filters);

  // Overloaded runOnGraph method with a default filter function.
  void runOnGraph(
      std::shared_ptr<Graph>& graph,
      const MatchFilter& filter =
          [](const Match&, const std::unordered_map<std::string, Value*>&) {
            return true;
          }) {
  // 在给定图形上运行匹配过滤器为 `std::vector<MatchFilter>` 的图匹配操作。
  runOnGraph(graph, std::vector<MatchFilter>({filter}));
}

// 注册标准重写模式。
void RegisterDefaultPatterns();

/** 注册自定义重写模式。
 *
 * 该方法接受三个参数，分别指定模式和替换子图：
 * \p PATTERN - 表示模式子图的IR字符串。
 * \p REPLACEMENT - 表示替换子图的IR字符串。
 * \p value name map - 用于将替换图中的值与模式图中的值进行映射的向量对。这在保留图重写过程中的源范围信息时很有用。
 *
 * 参见 `RegisterDefaultPatterns` 中的模式注册示例。
 */
void RegisterRewritePattern(
    const std::string& pattern,
    const std::string& replacement,
    const std::vector<std::pair<std::string, std::string>>& value_name_pair =
        {});

private:
std::vector<RewritePatternDescr> patterns_;  // 存储重写模式描述符的向量
std::unordered_set<Node*> nodes_to_delete_;  // 存储待删除节点的无序集合

// 在图形上应用单个重写模式。
void rewriteSinglePatternOnGraph(
    std::shared_ptr<Graph>& graph,
    const RewritePatternDescr& pattern,
    const std::vector<MatchFilter>& filters);

// 检查当前匹配是否与先前的匹配重叠。
bool overlapsWithPreviousMatches(const Match* match);
};

/** Rewrite pattern descriptor.
 *
 * This structure is used in the implementation of `SubgraphRewriter` and
 * is not supposed to be used externally.
 */
// 定义重写模式描述符结构体
struct RewritePatternDescr {
  // 模式字符串，表示待匹配的模式
  std::string pattern;
  // 替换字符串，表示匹配到模式后应该替换成的内容
  std::string replacement;
  // 值名称映射，用于将模式中的特定名称映射到其他名称
  std::unordered_map<std::string, std::string> value_name_map;
};

} // namespace jit
} // namespace torch
```