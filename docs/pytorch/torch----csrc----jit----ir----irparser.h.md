# `.\pytorch\torch\csrc\jit\ir\irparser.h`

```
#pragma once

#include <torch/csrc/Export.h>  // 包含 Torch 库的导出定义
#include <string>               // 包含标准字符串库
#include <unordered_map>        // 包含无序映射容器库

#include <c10/util/Optional.h>  // 包含 C10 库的 Optional 实用工具
#include <torch/csrc/Export.h>  // 再次包含 Torch 库的导出定义

namespace torch {
namespace jit {

struct Graph;
struct Value;

// \brief 从字符串 \p str 解析 IR，构建对应的图形在 \p graph 中。
// 如果 parse_tensor_constants 为 true，则为张量常量构建空张量，
// 其内容为随机或未初始化内容；否则将抛出异常。
TORCH_API void parseIR(
    const std::string& str,
    torch::jit::Graph* graph,
    bool parse_tensor_constants = false);

/** \brief 从字符串 \p str 解析 IR，构建对应的图形在 \p graph 中。
 *
 * \p vmap 被填充为从字符串到值的映射，允许通过原始 IR 字符串中的名称索引
 * 新创建图形中的值。
 * 如果 parse_tensor_constants 为 true，则为张量常量构建空张量，
 * 其内容为随机或未初始化内容；否则将抛出异常。
 */
TORCH_API void parseIR(
    const std::string& str,
    torch::jit::Graph* graph,
    std::unordered_map<std::string, Value*>& vmap,
    bool parse_tensor_constants = false);

} // namespace jit
} // namespace torch
```