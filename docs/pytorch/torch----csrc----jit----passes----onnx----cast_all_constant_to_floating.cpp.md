# `.\pytorch\torch\csrc\jit\passes\onnx\cast_all_constant_to_floating.cpp`

```
// 包含头文件，导入 Torch 的 ONNX 相关库
#include <torch/csrc/jit/passes/onnx/cast_all_constant_to_floating.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

// Torch 命名空间
namespace torch {
  // JIT 模块命名空间
  namespace jit {
    // ONNX 操作命名空间，引入 C10 和 ONNX 命名空间
    namespace onnx {
      using namespace ::c10::onnx;
    }

    // 将所有常量转换为浮点数的函数
    void CastAllConstantToFloating(Block* block) {
      // 获取当前图形
      auto graph = block->owningGraph();
      // 遍历当前块中的每个节点
      auto it = block->nodes().begin();
      while (it != block->nodes().end()) {
        auto node = *it;
        ++it;
        // 递归处理每个节点的子块
        for (auto block : node->blocks()) {
          CastAllConstantToFloating(block);
        }

        // 如果当前节点是常量节点
        if (node->kind() == onnx::Constant) {
          // 获取节点的值
          auto val = node->t(attr::value);
          // 获取值的数据类型
          at::ScalarType dtype = val.scalar_type();
          auto val_type = TensorType::create(val);
          // 如果数据类型不是浮点数或半精度浮点数，需要转换类型
          if (dtype != at::ScalarType::Double && dtype != at::ScalarType::Float &&
              dtype != at::ScalarType::Half) {
            // 定义目标类型的整数表示
            int to_type;
            switch (val.scalar_type()) {
              // 根据当前值的数据类型选择目标类型
              case at::ScalarType::Byte:
              case at::ScalarType::Char:
              case at::ScalarType::Int:
              case at::ScalarType::Short:
              case at::ScalarType::Bool:
                to_type = ATenTypeToOnnxType(val.scalar_type());
                val = val.to(at::ScalarType::Float);
                break;

              case at::ScalarType::Long:
                to_type = ATenTypeToOnnxType(val.scalar_type());
                val = val.to(at::ScalarType::Double);
                break;

              default:
                // 如果是不支持的类型，抛出运行时错误
                throw std::runtime_error("Unsupported types: complex, string");
            }
            // 移除原始节点的值属性，并设置新的值属性
            node->removeAttribute(attr::value);
            node->t_(attr::value, val);
            // 创建一个类型转换节点
            Node* cast_node = graph->create(onnx::Cast, 1);
            cast_node->i_(attr::to, to_type);
            cast_node->output()->setType(val_type);
            cast_node->insertAfter(node);
            // 替换原始节点的输出使用为类型转换节点的输出
            node->outputs().at(0)->replaceAllUsesWith(cast_node->outputs().at(0));
            // 将原始节点的输出作为类型转换节点的输入
            cast_node->addInput(node->outputs().at(0));
            // 复制节点的元数据
            cast_node->copyMetadata(node);
          }
        }
      }
    }

    // 对外接口函数，将所有常量转换为浮点数
    void CastAllConstantToFloating(const std::shared_ptr<Graph>& graph) {
      CastAllConstantToFloating(graph->block());
    }

  } // namespace jit
} // namespace torch
```