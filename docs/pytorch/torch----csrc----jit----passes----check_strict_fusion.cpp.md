# `.\pytorch\torch\csrc\jit\passes\check_strict_fusion.cpp`

```py
#include <torch/csrc/jit/passes/check_strict_fusion.h>  // 导入严格融合检查的头文件

#include <c10/util/Exception.h>  // 导入C10库的异常处理
#include <torch/csrc/jit/frontend/error_report.h>  // 导入前端错误报告相关头文件
#include <torch/csrc/jit/ir/ir.h>  // 导入IR（Intermediate Representation）相关头文件
#include <torch/csrc/jit/jit_log.h>  // 导入JIT日志相关头文件
#include <torch/csrc/jit/passes/quantization/helper.h>  // 导入量化辅助函数的头文件
#include <torch/csrc/jit/runtime/graph_iterator.h>  // 导入图遍历相关头文件
#include <unordered_map>  // 导入无序映射相关头文件

namespace torch {
namespace jit {

namespace {

bool isStrictFusion(Value* value) {
  const auto class_name = getModuleName(value);  // 获取值对应的模块名
  return class_name.has_value() &&
      (*class_name == "__torch__.torch.jit.strict_fusion");  // 判断是否为严格融合模块
}

} // namespace

static bool fusionGuardCheck(Symbol k) {
  // 检查是否为融合保护节点
  return k == Symbol::prim("TensorExprDynamicGuard") || k == prim::TypeCheck ||
      k == prim::CudaFusionGuard || k == prim::RequiresGradCheck;
}

static std::unordered_set<Node*> collectValuesUsedInGuard(
    Node* guarding_if,
    Node* enter_node) {
  // 收集在融合保护中使用的值的节点集合（深度优先搜索）
  std::unordered_set<Node*> visited_nodes;
  std::vector<Node*> queue = {guarding_if};

  while (!queue.empty()) {
    Node* curr = queue[queue.size() - 1];
    queue.pop_back();
    visited_nodes.insert(curr);
    
    // 检查当前节点是否是融合保护节点，如果是则跳过
    if (fusionGuardCheck(curr->kind())) {
      continue;
    }
    
    // 遍历当前节点的输入节点，将符合条件的节点加入队列
    for (Value* v : curr->inputs()) {
      Node* inp_node = v->node();
      if (inp_node->isBefore(enter_node) ||
          inp_node->owningBlock() != enter_node->owningBlock()) {
        continue;
      }
      if (visited_nodes.count(inp_node)) {
        continue;
      }
      queue.push_back(inp_node);
    }
  }
  return visited_nodes;  // 返回所有访问过的节点集合
}

static void checkForUnfusedOps(Node* enter_node) {
  std::vector<Node*> unsupported_nodes;
  std::vector<Node*> guarding_ifs; // 如果存在多个融合保护节点，将抛出异常

  // 遍历进入节点之后的所有节点，查找不支持的节点和融合保护节点
  for (Node* node = enter_node->next(); node->kind() != prim::Exit;
       node = node->next()) {
    if (node->kind() == prim::If &&
        fusionGuardCheck(node->input()->node()->kind())) {
      guarding_ifs.push_back(node);  // 将符合条件的融合保护节点加入列表
      continue;
    }
    unsupported_nodes.push_back(node);  // 将不支持的节点加入列表
  }

  if (guarding_ifs.size() > 1) {
    // 如果存在多个融合保护节点，抛出异常报告
    std::stringstream ss;
    ss << "Found multiple fusions: \n";
    for (Node* n : guarding_ifs) {
      ss << *n << "\n";
    }
    throw ErrorReport(enter_node->input()->node()->sourceRange()) << ss.str();
  }

  // 如果只有一个融合保护节点，收集其保护的所有节点
  std::unordered_set<Node*> guarding_check_nodes;
  if (guarding_ifs.size() == 1) {
    guarding_check_nodes =
        collectValuesUsedInGuard(guarding_ifs[0], enter_node);
  }

  // 查找未被融合保护节点使用的未融合操作节点
  std::vector<Node*> unfused_nodes_not_used_in_guard;
  for (Node* unfused : unsupported_nodes) {
    if (!guarding_check_nodes.count(unfused)) {
      unfused_nodes_not_used_in_guard.push_back(unfused);
      // 这里可以继续处理未融合的操作节点
    }
  }
}
  }
}
// 检查未使用的未融合节点是否为空
if (!unfused_nodes_not_used_in_guard.empty()) {
  // 创建一个字符串流对象
  std::stringstream ss;
  // 构建错误信息字符串
  ss << "Found unfused operators: \n";
  // 遍历未使用的未融合节点列表
  for (Node* unfused : unfused_nodes_not_used_in_guard) {
    // 添加缩进
    ss << "\t";
    // 如果未融合节点可能具有模式信息，则添加模式信息到字符串流中
    if (unfused->maybeSchema()) {
      ss << unfused->schema();
    } else {
      // 否则将未融合节点的类型转换为显示字符串并添加到字符串流中
      unfused->kind().toDisplayString();
    }
    // 添加换行符
    ss << "\n";
  }
  // 获取进入节点输入的节点的源范围
  auto range = enter_node->input()->node()->sourceRange();
  // 抛出错误报告，包含源范围和构建好的错误信息字符串
  throw ErrorReport(enter_node->input()->node()->sourceRange()) << ss.str();
}
} // 关闭 namespace jit
} // 关闭 namespace torch
```