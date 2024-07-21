# `.\pytorch\torch\csrc\jit\passes\frozen_concat_linear.cpp`

```py
// 包含头文件：C10 库中的 irange.h
// 用于 Torch 的 JIT 框架中的别名分析
// 用于 Torch 的 JIT 框架中的 IR 表示
// 用于 Torch 的 JIT 框架中的 IR 视图
// 用于 Torch 的 JIT 框架中的日志记录
// 用于 Torch 的 JIT 框架中的冻结 CONCAT 线性层的优化
// 用于 Torch 的 JIT 框架中的冻结卷积折叠优化
// 用于 Torch 的 JIT 框架中的冻结图优化
// 用于 Torch 的 JIT 框架中的移除 dropout 层的优化
// 用于 Torch 的 JIT 框架中的优化工具函数
// 用于 Torch 的 JIT 框架中的图执行器
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/frozen_concat_linear.h>
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/utils/optimization_utils.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含 ATen 库中的 Functions 头文件；否则包含 ops/cat 头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/cat.h>
#endif

// 包含标准库头文件
#include <unordered_set>
#include <utility>
#include <vector>

// Torch JIT 命名空间
namespace torch {
namespace jit {
namespace {

// 使用别名定义 Tensor 类型
using Tensor = at::Tensor;

// 类 ConcatLinearLayers 的定义，用于合并线性层
class ConcatLinearLayers {
 public:
  // 构造函数，接收一个共享指针指向 Graph 对象
  explicit ConcatLinearLayers(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  // 执行合并线性层的主要方法
  bool run() {
    // 处理主块及其子块中的线性层
    handleBlockAndSubblocks(graph_->block());
    // 返回图是否被修改的标志
    return graph_modified;
  }

  // 获取别名分析器的方法
  AliasDb* getAliasDb() {
    // 如果别名分析器为空，则创建一个新的并返回
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }

  // 收集常量线性层的方法
  void collectConstantLinearLayers(
      Block* b,
      std::unordered_map<Value*, std::vector<Node*>>& grouped_linear_layers,
      std::vector<Value*>& ordered_tensor_inputs) {
    // 使用有序列表，使得我们只需检查向前移动是否有效，不需向后移动时需要重建 aliasDb。

    // 遍历块中的每个节点
    for (Node* n : b->nodes()) {
      // 如果节点不是线性层，继续下一个节点
      if (n->kind() != aten::linear) {
        continue;
      }

      // 获取权重和偏置
      auto weight = n->namedInput("weight");
      auto bias = n->namedInput("bias");
      // 如果权重或偏置为空类型，则继续下一个节点
      if (weight->type() == NoneType::get() ||
          bias->type() == NoneType::get()) {
        continue;
      }

      // 如果参数非常量，则继续下一个节点
      if (nonConstantParameters(n)) {
        continue;
      }
      // 获取权重张量
      auto weight_tensor = constant_as<Tensor>(weight).value();
      // 如果权重张量不在 CUDA 设备上，则继续下一个节点
      if (!weight_tensor.device().is_cuda()) {
        continue;
      }

      // 获取线性层的输入值
      Value* linear_input = n->inputs().at(0);
      // 如果该输入值在映射中不存在，则创建新条目并记录在有序输入张量列表中
      if (grouped_linear_layers.find(linear_input) ==
          grouped_linear_layers.cend()) {
        grouped_linear_layers.insert({linear_input, std::vector<Node*>()});
        ordered_tensor_inputs.push_back(linear_input);
      }
      // 将当前节点加入到该输入值对应的线性层组中
      grouped_linear_layers.find(linear_input)->second.push_back(n);
    }
  }

  // 合并线性层的方法
  void mergeLinearLayers(std::vector<Node*>& compatible_layers) {
    // 设置图已修改标志为真
    graph_modified = true;
    // 断言兼容层列表不为空
    assert(!compatible_layers.empty());
    // 获取基准节点
    Node* base_node = compatible_layers[0];

    // 需要作用域以确保在删除 base_node 前释放 WithInsertPoint 保护并重置插入点
    Node* linear_node = nullptr;
    {
      // 在 base_node 处设置插入点，确保操作发生在 base_node 之后
      WithInsertPoint guard(base_node);
    
      // 从 compatible_layers 中提取权重张量，并将其拼接成一个新的张量
      auto weight_list = c10::fmap(compatible_layers, [](Node* n) {
        return constant_as<Tensor>(n->namedInput("weight")).value();
      });
      Tensor cat_weight = at::cat(weight_list, /*dim=*/0);
      // 将拼接后的权重张量作为常量插入图中，并获取其对应的 Value
      Value* cat_weight_value = graph_->insertConstant(std::move(cat_weight));
    
      // 从 compatible_layers 中提取偏置张量，并将其拼接成一个新的张量
      auto bias_list = c10::fmap(compatible_layers, [](Node* n) {
        return constant_as<Tensor>(n->namedInput("bias")).value();
      });
      Tensor cat_bias = at::cat(bias_list, /*dim=*/0);
      // 将拼接后的偏置张量作为常量插入图中，并获取其对应的 Value
      Value* cat_bias_value = graph_->insertConstant(std::move(cat_bias));
    
      // 获取 base_node 的输入张量作为线性层的输入，构建输入列表
      auto tensor_input = base_node->inputs().at(0);
      std::vector<Value*> linear_in = {
          tensor_input, cat_weight_value, cat_bias_value};
      // 在图中创建一个线性层节点，并将其插入到 base_node 之前
      linear_node = graph_->create(aten::linear, linear_in);
      linear_node->insertBefore(base_node);
    }
    
    // 更新节点的输出
    WithInsertPoint guard2(linear_node);
    // 插入常量节点表示 -1 和 1
    Value* neg1 = graph_->insertConstant(-1);
    Value* one = graph_->insertConstant(1);
    
    // 初始化切片起始位置
    int64_t slice_start = 0;
    Value* slice_start_val = graph_->insertConstant(0);
    
    for (Node* orig_node : compatible_layers) {
      // 对 compatible_layers 中的每个节点，计算切片结束位置并创建切片节点
      Tensor weight_tensor =
          constant_as<Tensor>(orig_node->namedInput("weight")).value();
      int64_t slice_end = slice_start + weight_tensor.size(0);
      Value* slice_end_val = graph_->insertConstant(slice_end);
    
      // 在图中创建切片节点，并替换原节点的所有使用，然后销毁原节点
      Node* slice = graph_->create(
          aten::slice,
          {linear_node->output(), neg1, slice_start_val, slice_end_val, one});
      slice->insertAfter(linear_node);
      orig_node->replaceAllUsesWith(slice);
      orig_node->destroy();
    
      // 更新切片起始位置为当前切片的结束位置
      slice_start = slice_end;
      slice_start_val = slice_end_val;
    }
    }
    
    // 检查两个张量的非零维度是否相等
    bool isNonZeroDimEqual(Tensor& tensor_a, Tensor& tensor_b) {
      if (tensor_a.dim() != tensor_b.dim()) {
        return false;
      }
      for (int64_t i = 1; i < tensor_a.dim(); i++) {
        if (tensor_a.size(i) != tensor_b.size(i)) {
          return false;
        }
      }
      return true;
    }
    
    // 收集和合并线性层组中的节点
    void collectAndMergeLinearLayers(std::vector<Node*>& linear_layer_group) {
      std::unordered_set<Node*> checked_nodes;
      // 函数体未提供，在此处无需注释
    }
    // 遍历线性层组中的每个节点
    for (size_t i = 0; i < linear_layer_group.size(); i++) {
      // 获取当前节点
      Node* base_node = linear_layer_group[i];
      // 如果当前节点已经被检查过，则跳过处理
      if (checked_nodes.count(base_node) != 0) {
        continue;
      }

      // 初始化与当前节点兼容的层列表，将当前节点加入其中
      std::vector<Node*> compatible_layers;
      compatible_layers.push_back(base_node);

      // 获取当前节点的权重和偏置
      auto base_weight =
          constant_as<Tensor>(base_node->namedInput("weight")).value();
      auto base_bias =
          constant_as<Tensor>(base_node->namedInput("bias")).value();

      // 现在遍历剩余的线性层组中的节点，看是否有可以与当前节点合并的
      for (size_t j = i + 1; j < linear_layer_group.size(); j++) {
        auto node = linear_layer_group[j];
        // 如果节点已经被检查过，则跳过处理
        if (checked_nodes.count(node) != 0) {
          continue;
        }
        // 获取当前节点的权重和偏置
        auto weight = constant_as<Tensor>(node->namedInput("weight")).value();
        auto bias = constant_as<Tensor>(node->namedInput("bias")).value();

        // 简单要求匹配类型来判断是否可以合并
        if (base_weight.dtype() != weight.dtype() ||
            base_weight.device() != weight.device() ||
            base_bias.dtype() != bias.dtype() ||
            base_bias.device() != bias.device()) {
          continue;
        }

        // 进一步检查权重和偏置的维度是否一致，以确定是否可以合并
        if (!isNonZeroDimEqual(base_weight, weight) ||
            !isNonZeroDimEqual(base_bias, bias)) {
          continue;
        }

        // 检查当前节点是否可以在拓扑上移动到兼容层列表中的所有节点之前
        bool can_move_before_all = true;
        for (auto n : compatible_layers) {
          can_move_before_all &=
              getAliasDb()->couldMoveBeforeTopologically(node, n);
        }
        if (!can_move_before_all) {
          continue;
        }

        // 找到一个可以合并的节点，将其加入兼容层列表，并标记为已检查过
        compatible_layers.push_back(node);
        checked_nodes.insert(node);
      }

      // 如果兼容层列表中只有当前节点自身，则跳过合并操作
      if (compatible_layers.size() == 1) {
        continue; // 没有其他层可合并
      }

      // 调用合并线性层函数，将兼容层列表作为参数
      mergeLinearLayers(compatible_layers);
    }
  }

  // 处理块及其子块的函数
  void handleBlockAndSubblocks(Block* block) {
    // 遍历块中的每个节点
    for (auto node : block->nodes()) {
      // 递归处理节点中的子块
      for (Block* subblock : node->blocks()) {
        handleBlockAndSubblocks(subblock);
      }
    }

    // 对当前块进行处理
    // 收集该块中的常量线性层，并按顺序处理
    std::unordered_map<Value*, std::vector<Node*>> grouped_linear_layers;
    std::vector<Value*> ordered_tensor_inputs;
    collectConstantLinearLayers(
        block, grouped_linear_layers, ordered_tensor_inputs);

    // 使用逆拓扑顺序处理线性层，以避免需要更新别名数据库
    for (auto tensor_it = ordered_tensor_inputs.rbegin();
         tensor_it != ordered_tensor_inputs.rend();
         ++tensor_it) {
      // 收集并合并当前张量输入相关的线性层
      collectAndMergeLinearLayers(grouped_linear_layers.at(*tensor_it));
    }
  }

 private:
  // 成员变量：图、图是否修改标志和别名数据库
  std::shared_ptr<Graph> graph_;
  bool graph_modified = false;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
};
} // namespace



// 结束了一个匿名命名空间的定义，匿名命名空间中的内容在当前文件中可见，但对其他文件是不可见的
};

namespace jit {
namespace torch {

TORCH_API bool FrozenConcatLinear(std::shared_ptr<Graph>& graph) {
    // 创建 ConcatLinearLayers 对象，用于操作给定的图 graph
    ConcatLinearLayers concatLayers(graph);
    // 在执行 FrozenConcatLinear 函数之前，将图的结构输出到日志中，便于调试和分析
    GRAPH_DUMP("Before FrozenConcatLinear", graph);
    // 运行 concatLayers 的操作，返回一个布尔值表示是否修改了图的结构
    bool changed = concatLayers.run();
    // 如果修改了图的结构，则再次将修改后的图结构输出到日志中
    if (changed) {
        GRAPH_DUMP("After FrozenConcatLinear", graph);
    }
    // 返回是否修改了图的结构的布尔值
    return changed;
}

} // namespace jit
} // namespace torch



} // namespace torch



} // namespace jit



TORCH_API



bool FrozenConcatLinear(std::shared_ptr<Graph>& graph) {



{



ConcatLinearLayers concatLayers(graph);



GRAPH_DUMP("Before FrozenConcatLinear", graph);



bool changed = concatLayers.run();



if (changed) {



GRAPH_DUMP("After FrozenConcatLinear", graph);



return changed;
```