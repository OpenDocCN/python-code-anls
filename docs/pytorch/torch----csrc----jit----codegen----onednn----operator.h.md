# `.\pytorch\torch\csrc\jit\codegen\onednn\operator.h`

```
#pragma once
// 使用 pragma once 确保头文件只被编译一次

#include <oneapi/dnnl/dnnl_graph.hpp>
// 引入 DNNL 图形库的头文件

#include <torch/csrc/jit/codegen/onednn/LlgaTensorImpl.h>
// 引入 Torch 的 LLGA 张量实现的头文件

#include <torch/csrc/jit/ir/ir.h>
// 引入 Torch 的 IR 头文件

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

class Operator {
 public:
  Operator(const Node* node, dnnl::graph::op::kind kind)
      : n(node), o(getId(node), kind, node->kind().toQualString()), k(kind) {}
  // Operator 类的构造函数，接受节点指针和操作种类作为参数

  // 如果值是图输出，则返回输出索引；否则返回 -1
  int32_t graphOutputIdx(Value* v) {
    int32_t i = 0;
    for (const Value* output : v->owningGraph()->outputs()) {
      if (v == output) {
        return i;
      }
      i++;
    }
    return -1;
  }

  // 设置输入值
  Operator& setInputValue(Value* v) {
    if (v->mustNotBeNone()) {
      if (v->type()->kind() == c10::TensorType::Kind) {
        o.add_input(createLogicalTensor(v));
      }
    }
    return *this;
  }

  // 设置输入，根据偏移量
  Operator& setInput(size_t offset) {
    return setInputValue(n->input(offset));
  }

  // 设置多个输入参数
  template <typename... Ts>
  Operator& setInput(size_t offset, Ts... other) {
    setInput(offset);
    return setInput(other...);
  }

  // 设置输出值
  Operator& setOutputValue(Value* v) {
    if (v->mustNotBeNone()) {
      o.add_output(createLogicalTensor(v));
    }
    return *this;
  }

  // 设置输出值，并连接到图的 End 节点
  Operator& setOutputValue(Value* v, std::unique_ptr<dnnl::graph::graph>& g) {
    if (v->mustNotBeNone()) {
      auto output_tensor = createLogicalTensor(v);
      o.add_output(output_tensor);
      if (g) {
        int32_t outputIndex = graphOutputIdx(v);
        if (outputIndex != -1) {
          dnnl::graph::op newEndNode(
              LONG_MAX - outputIndex,
              dnnl::graph::op::kind::End,
              "EndNodeForGraphOutput");
          newEndNode.add_input(output_tensor);
          g->add_op(newEndNode);
        }
      }
    }
    return *this;
  }

  // 设置输出，根据偏移量
  Operator& setOutput(std::unique_ptr<dnnl::graph::graph>& g, size_t offset) {
    return setOutputValue(n->output(offset), g);
  }

  // 设置输出，根据偏移量
  Operator& setOutput(size_t offset) {
    return setOutputValue(n->output(offset));
  }

  // 设置多个输出参数
  template <typename... Ts>
  Operator& setOutput(
      std::unique_ptr<dnnl::graph::graph>& g,
      size_t offset,
      Ts... other) {
    setOutput(g, offset);
    return setOutput(g, other...);
  }

  // 设置属性，使用给定的名称和值
  template <typename Attr>
  Operator& setAttr(dnnl::graph::op::attr name, Attr&& attr) {
    o.set_attr(name, std::forward<Attr>(attr));
    return *this;
  }

  // 设置属性，使用给定的名称和函数
  template <typename F>
  Operator& setAttr(dnnl::graph::op::attr name, const F& fn, size_t offset) {
    return setAttr(name, fn(n, offset));
  }

  // 将节点的标量转换为浮点数
  static float ScalarToFloat(const Node* node, size_t offset) {
    // 省略部分函数实现，未提供完整代码
    return toIValue(node->input(offset))->toScalar().to<float>();


    // 获取节点的输入在指定偏移量处的值，并将其转换为标量，再转换为 float 类型返回
    // node: 节点对象指针
    // offset: 偏移量，表示输入列表中的位置
    return toIValue(node->input(offset))->toScalar().to<float>();
  }

  static std::vector<int64_t> Ints(const Node* node, size_t offset) {
    // 获取节点的输入在指定偏移量处的值，并将其转换为 int64_t 类型的向量返回
    // node: 节点对象指针
    // offset: 偏移量，表示输入列表中的位置
    return toIValue(node->input(offset))->toIntVector();
  }

  static int64_t Int(const Node* node, size_t offset) {
    // 获取节点的输入在指定偏移量处的值，并将其转换为 int64_t 类型返回
    // node: 节点对象指针
    // offset: 偏移量，表示输入列表中的位置
    return toIValue(node->input(offset))->toInt();
  }

  static float Float(const Node* node, size_t offset) {
    // 获取节点的输入在指定偏移量处的值，并将其转换为 double 类型，再转换为 float 类型返回
    // node: 节点对象指针
    // offset: 偏移量，表示输入列表中的位置
    return static_cast<float>(toIValue(node->input(offset))->toDouble());
  }

  static bool Bool(const Node* node, size_t offset) {
    // 获取节点的输入在指定偏移量处的值，并将其转换为 bool 类型返回
    // node: 节点对象指针
    // offset: 偏移量，表示输入列表中的位置
    return toIValue(node->input(offset))->toBool();
  }

  static uint64_t getId(const Node* node) {
    // 将节点的地址重新解释为 uint64_t 类型，作为操作的唯一标识符返回
    // node: 节点对象指针
    return reinterpret_cast<uint64_t>(node); // cast node address as op id
  }

  dnnl::graph::op::kind kind() const {
    // 返回操作的类型 (kind)
    return k;
  }

  dnnl::graph::op llgaOp() const {
    // 返回操作对象 (llgaOp)
    return o;
  }

 private:
  dnnl::graph::logical_tensor createLogicalTensor(Value* value) const {
    // 根据给定的值创建逻辑张量对象
    // value: 值对象指针
    return LlgaTensorDesc(value).logical_tensor();
  }

  const Node* n;           // 节点指针成员变量
  dnnl::graph::op o;       // 操作对象成员变量
  dnnl::graph::op::kind k; // 操作类型成员变量
};

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch


注释：


// 关闭命名空间 torch
};

// 结束命名空间 jit
} // namespace jit

// 结束命名空间 fuser
} // namespace fuser

// 结束命名空间 onednn
} // namespace onednn


这段代码是用来结束嵌套的命名空间声明。在C++或类似的编程语言中，命名空间用于组织代码以避免命名冲突，而这里的代码是在声明了多个嵌套的命名空间之后，用来关闭每个命名空间的。
```