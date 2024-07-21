# `.\pytorch\torch\csrc\jit\passes\erase_number_types.cpp`

```
// 引入 Torch 库中的相关头文件，用于擦除数值类型
#include <torch/csrc/jit/passes/erase_number_types.h>

// 引入 Torch 库中的常量定义和日志记录相关头文件
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

// 引入 ATen 库中的标量操作相关头文件
#include <ATen/ScalarOps.h>

// Torch 命名空间
namespace torch {
// JIT 子命名空间
namespace jit {

// 将数值类型设置为张量类型的静态函数
static void SetNumTypeToTensorType(Value* v) {
  // 如果值的类型是数值类型的子类型
  if (v->type()->isSubtypeOf(*NumberType::get())) {
    // 将值的类型设置为对应的张量类型
    v->setType(TensorType::fromNumberType(*v->type()));
  } else if (v->type()->isSubtypeOf(*BoolType::get())) {
    // 如果值的类型是布尔类型的子类型，将其类型设置为布尔张量类型
    v->setType(TensorType::fromBoolType());
  }
}

// 递归地在基本块上执行数值类型擦除
void EraseNumberTypesOnBlock(Block* block) {
  // 遍历基本块中的每个节点
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    // 处理当前节点的输入
    for (auto inp : it->inputs()) {
      SetNumTypeToTensorType(inp);
    }
    // 递归处理当前节点的子块
    for (auto sub : it->blocks()) {
      EraseNumberTypesOnBlock(sub);
    }
    // 根据节点的种类进行不同的处理
    switch (it->kind()) {
      case prim::Constant: {
        // 删除原始常量，替换为对应的张量等效形式
        // ONNX 不支持非张量常量
        if (it->output()->type()->isSubtypeOf(*NumberType::get()) ||
            it->output()->type()->isSubtypeOf(*BoolType::get())) {
          // 获取常量的值并转换为张量
          at::Scalar s;
          if (it->output()->type()->isSubtypeOf(*BoolType::get())) {
            s = *constant_as<bool>(it->output());
          } else {
            s = *constant_as<at::Scalar>(it->output());
          }

          // 在当前节点的插入点创建新的常量节点
          WithInsertPoint guard(*it);
          Value* r = block->owningGraph()->insertConstant(
              scalar_to_tensor(s), c10::nullopt, it->scope());
          r->copyMetadata(it->output());
          // 替换所有使用原始常量的地方为新的张量常量
          it->output()->replaceAllUsesWith(r);
          // 删除当前节点
          it.destroyCurrent();
        }
      } break;
      // 处理不同类型的标量操作节点，将其输出替换为输入并删除节点
      case aten::Bool:
      case aten::Float:
      case aten::Int:
      case aten::FloatImplicit:
      case aten::IntImplicit:
      case aten::ScalarImplicit:
      case prim::NumToTensor: {
        it->output()->replaceAllUsesWith(it->inputs()[0]);
        it.destroyCurrent();
      } break;
      // 对于其他类型的节点，将输出设置为张量类型
      default: {
        for (auto o : it->outputs()) {
          SetNumTypeToTensorType(o);
        }
      } break;
    }
  }
}

// 在整个图中执行数值类型擦除
void EraseNumberTypes(const std::shared_ptr<Graph>& graph) {
  // 处理图的输入，将数值类型设置为张量类型
  for (auto inp : graph->inputs()) {
    SetNumTypeToTensorType(inp);
  }
  // 调用递归函数在整个基本块上执行数值类型擦除
  EraseNumberTypesOnBlock(graph->block());
  // 输出执行擦除数值类型后的图的状态信息
  GRAPH_DUMP("After EraseNumberTypes: ", graph);
}

} // namespace jit
} // namespace torch
```