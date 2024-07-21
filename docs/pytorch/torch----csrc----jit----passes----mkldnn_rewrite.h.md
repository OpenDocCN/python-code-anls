# `.\pytorch\torch\csrc\jit\passes\mkldnn_rewrite.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/Config.h>
// 包含 ATen 库的配置头文件

#include <torch/csrc/jit/api/module.h>
// 包含 Torch JIT API 中的模块定义头文件

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch JIT 中的 IR（Intermediate Representation，中间表示）定义头文件

#include <torch/csrc/jit/passes/subgraph_rewrite.h>
// 包含 Torch JIT 中的子图重写（subgraph rewrite）相关头文件

#if AT_MKLDNN_ENABLED()
// 如果 MKLDNN 加速被启用，则编译以下代码块

#include <ideep/tensor.hpp>
// 包含 MKLDNN 的张量操作定义头文件

#endif // AT_MKLDNN_ENABLED()
// 结束 MKLDNN 加速代码块的条件编译

namespace torch {
namespace jit {

#if AT_MKLDNN_ENABLED()
// 如果 MKLDNN 加速被启用，则编译以下代码块

namespace mkldnn {
// 命名空间 mkldnn，用于组织 MKLDNN 相关内容

const static std::map<std::string, std::vector<torch::jit::MatchFilter>>
    fusion_rewrite_map = {
        {"none", {}},   // 空的匹配过滤器列表，用于指定无融合重写
        {"relu", {}},   // 空的匹配过滤器列表，用于指定 ReLU 融合重写
};
// 静态常量映射，将字符串与 Torch JIT 中的匹配过滤器向量关联起来，用于指定不同类型的融合重写策略

} // namespace mkldnn
// 结束 mkldnn 命名空间

#endif // AT_MKLDNN_ENABLED()
// 结束 MKLDNN 加速代码块的条件编译

void FuseConvWithEltwise(std::shared_ptr<Graph>& graph);
// 声明函数 FuseConvWithEltwise，用于将卷积操作与逐元素操作融合，参数是图的共享指针

} // namespace jit
// 结束 jit 命名空间

} // namespace torch
// 结束 torch 命名空间
```