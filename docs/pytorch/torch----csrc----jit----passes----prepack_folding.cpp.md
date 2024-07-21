# `.\pytorch\torch\csrc\jit\passes\prepack_folding.cpp`

```py
// 包含必要的头文件
#include <stack>  // 使用堆栈数据结构

// 引入 Torch 库的相关模块和函数
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/prepack_folding.h>

// 定义命名空间
namespace torch {
namespace jit {

// 在常量折叠之后运行此优化
void PrePackingOpsFolder(
    script::Module& m,                              // 输入参数：Torch 模块的引用
    const PrePackingOpsFilterFn& is_foldable_op,    // 输入参数：用于判断是否可折叠操作的函数对象
    const std::string& attr_prefix) {               // 输入参数：属性名称前缀
  for (auto& method : m.get_methods()) {            // 遍历模块中的每个方法
    int64_t uid = 0;                               // 唯一标识符，结合方法名称生成唯一值
    auto graph = method.graph();                    // 获取当前方法的计算图
    std::stack<Block*> blocks_to_visit;             // 创建一个堆栈，用于存储待访问的基本块
    std::unordered_set<Node*> nodes_to_delete;      // 创建一个集合，用于存储待删除的节点
    blocks_to_visit.push(graph->block());           // 将方法的基本块入栈
    std::string attr_name_base =                    // 构建属性名称的基础部分
        attr_prefix + "_" + method.name() + "._jit_pass_packed_weight_";
    while (!blocks_to_visit.empty()) {              // 遍历待访问的基本块直到堆栈为空
      Block* b = blocks_to_visit.top();             // 获取堆栈顶部的基本块
      blocks_to_visit.pop();                        // 弹出堆栈顶部的基本块
      for (Node* n : b->nodes()) {                  // 遍历基本块中的每个节点
        if (is_foldable_op(n)) {                    // 判断节点是否可折叠
          auto optional_outputs = runNodeIfInputsAreConstant(n);  // 如果输入为常量，则运行节点
          if (optional_outputs) {                   // 如果节点执行成功
            auto outputs = optional_outputs.value(); // 获取节点的输出
            TORCH_CHECK(outputs.size() == 1, "Prepack ops have single output");  // 检查节点的输出数量是否为1
            auto attr_name = attr_name_base + std::to_string(uid++);  // 构建唯一的属性名称
            TORCH_CHECK(                           // 检查属性名称是否已存在
                !(m.type()->findAttributeSlot(attr_name)),
                "Attribute name ",
                attr_name,
                " already exist in",
                " module of type:",
                m.type()->name()->qualifiedName(),
                ". Please make sure that",
                " FoldPrePackingOps is run at the top level module only.");
            m.register_attribute(attr_name, n->output(0)->type(), outputs[0]);  // 在模块中注册新属性
            Value* prepack_op_value = n->output(0);  // 获取节点的输出值
            WithInsertPoint ins(prepack_op_value->node());  // 设置插入点
            Value* packed_weight_attr =              // 插入获取属性值的操作
                graph->insertGetAttr(graph->inputs()[0], attr_name)
                    ->setType(n->output(0)->type());
            prepack_op_value->replaceAllUsesWith(packed_weight_attr);  // 替换节点的所有使用
            nodes_to_delete.insert(n);               // 将节点添加到待删除集合中
          }
        }
        for (Block* subblock : n->blocks()) {       // 遍历节点的子块
          blocks_to_visit.push(subblock);           // 将子块压入堆栈
        }
      }
    }
    for (auto n : nodes_to_delete) {                // 删除待删除集合中的每个节点的输入
      n->removeAllInputs();
    }
    for (auto n : nodes_to_delete) {                // 销毁待删除集合中的每个节点
      n->destroy();
    }
  }
}

} // namespace jit
} // namespace torch
```